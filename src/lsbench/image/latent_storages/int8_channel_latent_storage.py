import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


class INT8ChannelLatentStorage(ImageLatentStorage):
    """Symmetric INT8 with one absmax scale per (sample, channel)."""

    NAME = "int8_channel"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        amax = x.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12)
        scale = amax / 127
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "scale": scale}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        return t["q"].float() * t["scale"]

    def get_storage_name(self) -> str:
        return self.NAME
