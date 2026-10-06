import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    dequantize_minmax,
    from_groups,
    pack_tensors,
    quantize_minmax,
    to_groups,
    unpack_tensors,
)


class INT4MinMaxGroupLatentStorage(ImageLatentStorage):
    """Asymmetric INT4 on the true min/max of each group of `group_size` consecutive values; min/max stored as 8-bit indices into the per-channel range."""

    NAME = "int4_minmax_group"

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.group_size = group_size

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        c, h, w = x.shape[1:]
        # per-channel reference ranges only when groups do not cross channels
        refs = c if (h * w) % self.group_size == 0 else 1
        groups = to_groups(x, self.group_size)
        return pack_tensors(quantize_minmax(groups, refs, 4), x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        return from_groups(dequantize_minmax(t, self.group_size, 4), shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
