import math

import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


class INT8MuLawLatentStorage(ImageLatentStorage):
    """Non-uniform INT8: mu-law companding of the absmax-normalized values, then uniform 8-bit quantization (finer steps near zero)."""

    NAME = "int8_mulaw"

    def __init__(self, mu: float = 255.0):
        super().__init__()
        self.mu = mu

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b = x.shape[0]
        amax = x.abs().reshape(b, -1).amax(1).clamp(min=1e-12).view(b, 1, 1, 1)
        y = x / amax
        c = torch.sign(y) * torch.log1p(self.mu * y.abs()) / math.log1p(self.mu)
        q = torch.round(c * 127).clamp(-127, 127).to(torch.int8)
        return pack_tensors({"q": q, "amax": amax.reshape(b)}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        c = t["q"].float() / 127
        y = torch.sign(c) * torch.expm1(c.abs() * math.log1p(self.mu)) / self.mu
        return y * t["amax"].view(-1, 1, 1, 1)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_mu{self.mu:g}"
