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


class INT4MinMaxSquareLatentStorage(VideoLatentStorage):
    """Asymmetric INT4 on the true min/max of each block x block spatial patch of one channel and frame; min/max stored as 8-bit indices into the per-channel range."""

    NAME = "int4_minmax_square"

    def __init__(self, block: int = 4):
        super().__init__()
        self.block = block

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
        return pack_tensors(quantize_minmax(groups, c, 4), x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        b, c, nt, h, w = shape
        groups = dequantize_minmax(t, self.block**2, 4)
        x = from_square_groups(groups, (b, c * nt, h, w), self.block)
        return x.reshape(shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_b{self.block}"
