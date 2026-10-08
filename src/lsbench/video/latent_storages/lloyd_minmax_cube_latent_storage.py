import math

import torch

from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage
from lsbench.utils.quantization import (
    dequantize_minmax,
    from_cube_groups,
    pack_tensors,
    quantize_minmax,
    to_cube_groups,
    unpack_tensors,
)


class LloydMinMaxCubeLatentStorage(VideoLatentStorage):
    """
    Like lloyd_minmax_square, but the groups are block x block x block spatiotemporal cubes of one
    channel (a single min/max per cube) instead of per-frame block x block patches.
    """

    NAME = "lloyd_minmax_cube"

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
        groups = to_cube_groups(x, self.block)
        return pack_tensors(
            quantize_minmax(
                groups,
                x.shape[1],
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
        _, c, nt, h, w = shape
        n_groups = (
            c
            * math.ceil(nt / self.block)
            * math.ceil(h / self.block)
            * math.ceil(w / self.block)
        )
        groups = dequantize_minmax(
            t, self.block**3, self.bits, self.range_bits, n_groups
        )
        return from_cube_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        suffix = "_pc" if self.per_channel else ""
        return f"{self.NAME}_c{self.bits}_b{self.block}_r{self.range_bits}{suffix}"
