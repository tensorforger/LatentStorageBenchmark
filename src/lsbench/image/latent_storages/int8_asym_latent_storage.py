import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


class INT8AsymLatentStorage(ImageLatentStorage):
    """Asymmetric (affine min/max, zero-point) UINT8 per (sample, channel)."""

    NAME = "int8_asym"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        lo = x.amin(dim=(2, 3), keepdim=True)
        hi = x.amax(dim=(2, 3), keepdim=True)
        scale = ((hi - lo) / 255).clamp(min=1e-12)
        q = torch.round((x - lo) / scale).clamp(0, 255).to(torch.uint8)
        return pack_tensors({"q": q, "scale": scale, "min": lo}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        return t["q"].float() * t["scale"] + t["min"]

    def get_storage_name(self) -> str:
        return self.NAME
