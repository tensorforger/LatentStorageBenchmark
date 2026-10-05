import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    from_groups,
    hadamard_matrix,
    pack_tensors,
    to_groups,
    unpack_tensors,
)


class INT8HadamardLatentStorage(ImageLatentStorage):
    """QuaRot-style incoherence processing: Walsh-Hadamard rotation of each group of `group_size` values, then INT8 with one absmax scale per sample."""

    NAME = "int8_hadamard"

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
        b = x.shape[0]
        h = hadamard_matrix(self.group_size).to(x.device)
        rot = to_groups(x, self.group_size) @ h
        scale = (rot.abs().reshape(b, -1).amax(1).clamp(min=1e-12) / 127).view(b, 1, 1)
        q = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "scale": scale.reshape(b)}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        h = hadamard_matrix(self.group_size).to(device)
        rot = t["q"].float() * t["scale"].view(-1, 1, 1)
        return from_groups(rot @ h, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
