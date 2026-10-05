import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import mse_clip_scale, pack_tensors, unpack_tensors


class INT8MSEClipLatentStorage(ImageLatentStorage):
    """Symmetric INT8 per (sample, channel) with the clipping range searched to minimize MSE (calibration-style clipping)."""

    NAME = "int8_mse_clip"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b, c = x.shape[:2]
        rows = x.reshape(b, c, -1)
        scale = mse_clip_scale(rows, 127)
        q = torch.round(rows / scale).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "scale": scale}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        return (t["q"].float() * t["scale"]).reshape(shape)

    def get_storage_name(self) -> str:
        return self.NAME
