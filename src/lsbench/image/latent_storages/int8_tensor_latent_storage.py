import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


class INT8TensorLatentStorage(ImageLatentStorage):
    """Symmetric INT8 with one absmax scale per sample (per-tensor quantization)."""

    NAME = "int8_tensor"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b = x.shape[0]
        amax = x.abs().reshape(b, -1).amax(1).clamp(min=1e-12)
        scale = (amax / 127).view(b, 1, 1, 1)
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "scale": scale.reshape(b)}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        return t["q"].float() * t["scale"].view(-1, 1, 1, 1)

    def get_storage_name(self) -> str:
        return self.NAME
