import torch
from safetensors.torch import save, load
from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage


class FP16LatentStorage(VideoLatentStorage):
    NAME = "fp16"

    def __init__(
        self,
    ):
        super().__init__()

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            bytes
        """
        t = latents.detach().to(device="cpu", dtype=torch.float16).contiguous()
        return save({"latents": t})

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        t = load(data)["latents"]
        return t.to(device=device, dtype=torch.float16)

    def get_storage_name(self) -> str:
        return self.NAME
