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


class INT8MinMaxSquareLatentStorage(ImageLatentStorage):
    """8-bit variant of int4_minmax_square: true min/max per block x block patch, 256 levels."""

    NAME = "int8_minmax_square"
    BITS = 8

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
        return pack_tensors(quantize_minmax(groups, x.shape[1], self.BITS), x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        groups = dequantize_minmax(t, self.block**2, self.BITS)
        return from_square_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_b{self.block}"
