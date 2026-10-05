import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    pack_tensors,
    to_groups,
    unpack_tensors,
)


class INT8GroupLatentStorage(ImageLatentStorage):
    """Symmetric INT8 with an fp16 absmax scale per group of `group_size` consecutive values (block-wise quantization)."""

    NAME = "int8_group"

    def __init__(self, group_size: int = 64):
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
        groups = to_groups(x, self.group_size)
        scale16 = fp16_scale(groups.abs().amax(-1, keepdim=True), 127)
        q = torch.round(groups / scale16.float()).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        return from_groups(t["q"].float() * t["scale"].float(), shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
