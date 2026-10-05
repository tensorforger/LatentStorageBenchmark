import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


class INT8OutlierLatentStorage(ImageLatentStorage):
    """LLM.int8()-style mixed decomposition: values with |x| > threshold stay in fp16 (sparse), the rest is INT8 per (sample, channel)."""

    NAME = "int8_outlier"

    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        mask = x.abs() > self.threshold
        idx = mask.reshape(-1).nonzero().squeeze(1)
        vals = x.reshape(-1)[idx].to(torch.float16)
        dense = x.masked_fill(mask, 0)
        scale = dense.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12) / 127
        q = torch.round(dense / scale).clamp(-127, 127).to(torch.int8)
        return pack_tensors(
            {"q": q, "scale": scale, "idx": idx.to(torch.int32), "vals": vals},
            x.shape,
        )

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        x = t["q"].float() * t["scale"]
        x.view(-1)[t["idx"].long()] = t["vals"].float()
        return x

    def get_storage_name(self) -> str:
        return f"{self.NAME}_t{self.threshold:g}"
