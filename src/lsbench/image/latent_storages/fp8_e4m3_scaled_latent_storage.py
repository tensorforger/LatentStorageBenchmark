import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import FP8_E4M3_MAX, pack_tensors, unpack_tensors


class FP8E4M3ScaledLatentStorage(ImageLatentStorage):
    """FP8 E4M3 with a per-sample amax scale (Transformer Engine style per-tensor scaling)."""

    NAME = "fp8_e4m3_scaled"

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
        scale = (amax / FP8_E4M3_MAX).view(b, 1, 1, 1)
        q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
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
