import math

import torch

from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage
from lsbench.utils.quantization import (
    dequantize_minmax,
    from_square_groups,
    pack_tensors,
    quantize_minmax,
    to_square_groups,
    unpack_tensors,
)


class LloydMinMaxSquareLatentStorage(VideoLatentStorage):
    """
    Min/max block quantization on block x block spatial patches of one channel and frame with
    `bits`-bit codes. Min/max are stored as `range_bits`-bit indices into the per-channel range,
    and the levels inside [min, max] are a per-sample Lloyd-Max table (stored in the bytes), not
    uniform. With `per_channel` every latent channel gets its own table (more metadata).
    """

    NAME = "lloyd_minmax_square"

    def __init__(
        self,
        bits: int = 4,
        block: int = 4,
        range_bits: int = 6,
        per_channel: bool = False,
    ):
        super().__init__()
        if not 2 <= bits <= 8 or not 2 <= range_bits <= 8:
            raise ValueError("bits and range_bits must be in [2, 8]")
        self.bits = bits
        self.block = block
        self.range_bits = range_bits
        self.per_channel = per_channel

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b, c, t, h, w = x.shape
        groups = to_square_groups(x.reshape(b, c * t, h, w), self.block)
        return pack_tensors(
            quantize_minmax(
                groups,
                c,
                self.bits,
                self.range_bits,
                lloyd=True,
                lloyd_per_channel=self.per_channel,
            ),
            x.shape,
        )

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        b, c, nt, h, w = shape
        n_groups = c * nt * math.ceil(h / self.block) * math.ceil(w / self.block)
        groups = dequantize_minmax(
            t, self.block**2, self.bits, self.range_bits, n_groups
        )
        x = from_square_groups(groups, (b, c * nt, h, w), self.block)
        return x.reshape(shape)

    def get_storage_name(self) -> str:
        suffix = "_pc" if self.per_channel else ""
        return f"{self.NAME}_c{self.bits}_b{self.block}_r{self.range_bits}{suffix}"
