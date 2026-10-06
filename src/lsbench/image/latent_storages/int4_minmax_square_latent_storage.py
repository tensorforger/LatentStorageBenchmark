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


class INT4MinMaxSquareLatentStorage(ImageLatentStorage):
    """Asymmetric INT4 on the true min/max of each block x block patch of one channel; min/max stored as 8-bit indices into the per-channel range."""

    NAME = "int4_minmax_square"

    def __init__(self, block: int = 4):
        super().__init__()
        self.block = block

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        groups = to_square_groups(x, self.block)
        return pack_tensors(quantize_minmax(groups, x.shape[1], 4), x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        groups = dequantize_minmax(t, self.block**2, 4)
        return from_square_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_b{self.block}"
