import torch
from safetensors.torch import save, load
from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage


class BF16LatentStorage(ImageLatentStorage):
    NAME = "bf16"

    def __init__(
        self,
    ):
        super().__init__()

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        t = latents.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        return save({"latents": t})

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t = load(data)["latents"]
        return t.to(device=device, dtype=torch.bfloat16)

    def get_storage_name(self) -> str:
        return self.NAME
