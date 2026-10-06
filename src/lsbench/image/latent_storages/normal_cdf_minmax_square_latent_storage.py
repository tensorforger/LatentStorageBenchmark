import math

import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    dequantize_minmax,
    from_square_groups,
    pack_tensors,
    quantize_minmax,
    to_square_groups,
    unpack_tensors,
)

CDF_EPS = 1e-6  # keeps ndtri finite; clips latents beyond ~4.75 sigma


class NormalCdfMinMaxSquareLatentStorage(ImageLatentStorage):
    """
    Min/max block quantization on block x block patches of one channel with `bits`-bit codes.
    Latents are first mapped to uniform with the normal CDF using the per-sample, per-channel
    mean/std (stored in the bytes), then quantized uniformly, so no per-sample table is fitted.
    `std_scale` widens the CDF (std_scale = 1 is plain equal-probability levels; sqrt(3) is the
    MSE-optimal high-resolution compander for a normal source).
    """

    NAME = "normal_cdf_minmax_square"

    def __init__(
        self,
        bits: int = 4,
        block: int = 4,
        range_bits: int = 6,
        std_scale: float = 1.0,
    ):
        super().__init__()
        if not 2 <= bits <= 8 or not 2 <= range_bits <= 8:
            raise ValueError("bits and range_bits must be in [2, 8]")
        self.bits = bits
        self.block = block
        self.range_bits = range_bits
        self.std_scale = std_scale

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b, c = x.shape[:2]
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), correction=0).clamp(min=1e-6) * self.std_scale
        groups = to_square_groups(x, self.block).reshape(b, c, -1, self.block**2)
        z = (groups - mean[:, :, None, None]) / std[:, :, None, None]
        u = 0.5 * torch.erfc(-z / math.sqrt(2.0))
        u = u.clamp(CDF_EPS, 1 - CDF_EPS).reshape(b, -1, self.block**2)
        t = quantize_minmax(u, c, self.bits, self.range_bits)
        t["mean"] = mean
        t["std"] = std
        return pack_tensors(t, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        b, c, h, w = shape
        n_groups = c * math.ceil(h / self.block) * math.ceil(w / self.block)
        mean, std = t.pop("mean"), t.pop("std")
        u = dequantize_minmax(t, self.block**2, self.bits, self.range_bits, n_groups)
        z = torch.special.ndtri(u.clamp(CDF_EPS, 1 - CDF_EPS))
        groups = z.reshape(b, c, -1, self.block**2) * std[:, :, None, None]
        groups = (groups + mean[:, :, None, None]).reshape(b, n_groups, -1)
        return from_square_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        name = f"{self.NAME}_c{self.bits}_b{self.block}_r{self.range_bits}"
        return name if self.std_scale == 1.0 else f"{name}_s{self.std_scale:g}"
