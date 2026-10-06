import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    dequantize_int4_asym,
    from_square_groups,
    pack_tensors,
    quantize_int4_asym,
    to_square_groups,
    unpack_tensors,
)


class INT4AsymSquareLatentStorage(ImageLatentStorage):
    """Asymmetric INT4 (fp16 scale, 4-bit zero-point) per square block x block patch of a single channel."""

    NAME = "int4_asym_square"

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
        return pack_tensors(quantize_int4_asym(groups), x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        groups = dequantize_int4_asym(t, self.block**2)
        return from_square_groups(groups, shape, self.block)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_b{self.block}"
