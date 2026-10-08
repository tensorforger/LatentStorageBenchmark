import torch
from safetensors.torch import save, load

from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage
from lsbench.utils.quantization import FP8_E4M3_MAX


class FP8E4M3LatentStorage(VideoLatentStorage):
    """Plain cast to FP8 E4M3 without any scaling."""

    NAME = "fp8_e4m3"

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            bytes
        """
        t = latents.detach().float().clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        return save({"latents": t.to(torch.float8_e4m3fn).cpu().contiguous()})

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        return load(data)["latents"].to(device=device, dtype=torch.float32)

    def get_storage_name(self) -> str:
        return self.NAME
