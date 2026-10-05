import torch
from safetensors.torch import save, load

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import FP8_E5M2_MAX


class FP8E5M2LatentStorage(ImageLatentStorage):
    """Plain cast to FP8 E5M2 without any scaling."""

    NAME = "fp8_e5m2"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        t = latents.detach().float().clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX)
        return save({"latents": t.to(torch.float8_e5m2).cpu().contiguous()})

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        return load(data)["latents"].to(device=device, dtype=torch.float32)

    def get_storage_name(self) -> str:
        return self.NAME
