import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    FP16_MIN_SCALE,
    from_groups,
    pack_nibbles,
    pack_tensors,
    to_groups,
    unpack_nibbles,
    unpack_tensors,
)


class INT4AsymGroupLatentStorage(ImageLatentStorage):
    """Asymmetric INT4 (AWQ/GPTQ zero-point format): fp16 scale and 4-bit integer zero-point per group."""

    NAME = "int4_asym_group"

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.group_size = group_size

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        groups = to_groups(x, self.group_size)
        lo = groups.amin(-1, keepdim=True).clamp(max=0)
        hi = groups.amax(-1, keepdim=True).clamp(min=0)
        scale16 = ((hi - lo) / 15).clamp(min=FP16_MIN_SCALE).to(torch.float16)
        scale = scale16.float()
        zero = torch.round(-lo / scale).clamp(0, 15)
        q = (torch.round(groups / scale) + zero).clamp(0, 15).to(torch.uint8)
        return pack_tensors(
            {
                "codes": pack_nibbles(q),
                "scale": scale16,
                "zero": pack_nibbles(zero.to(torch.uint8)),
            },
            x.shape,
        )

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        b, n_groups = t["scale"].shape[:2]
        q = unpack_nibbles(t["codes"], n_groups * self.group_size).reshape(
            b, n_groups, self.group_size
        )
        zero = unpack_nibbles(t["zero"], n_groups).reshape(b, n_groups, 1)
        groups = (q.float() - zero.float()) * t["scale"].float()
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
