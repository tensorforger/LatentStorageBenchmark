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


class LloydMinMaxSquareLatentStorage(ImageLatentStorage):
    """
    Min/max block quantization on block x block patches of one channel with `bits`-bit codes.
    Min/max are stored as `range_bits`-bit indices into the per-channel range, and the levels
    inside [min, max] are a per-sample Lloyd-Max table (stored in the bytes), not uniform.
    """

    NAME = "lloyd_minmax_square"

    def __init__(self, bits: int = 4, block: int = 4, range_bits: int = 6):
        super().__init__()
        if not 2 <= bits <= 8 or not 2 <= range_bits <= 8:
            raise ValueError("bits and range_bits must be in [2, 8]")
        self.bits = bits
        self.block = block
        self.range_bits = range_bits

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        groups = to_square_groups(x, self.block)
        return pack_tensors(
            quantize_minmax(groups, x.shape[1], self.bits, self.range_bits, lloyd=True),
            x.shape,
        )

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
        groups = dequantize_minmax(
            t, self.block**2, self.bits, self.range_bits, n_groups
        )
        return from_square_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_c{self.bits}_b{self.block}_r{self.range_bits}"
