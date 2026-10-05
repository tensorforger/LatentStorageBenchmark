import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    FP16_MIN_SCALE,
    from_groups,
    mse_clip_scale,
    pack_int4_symmetric,
    pack_tensors,
    to_groups,
    unpack_int4_symmetric,
    unpack_tensors,
)


class INT4MSEClipLatentStorage(ImageLatentStorage):
    """Symmetric INT4 per group with the clipping range searched to minimize MSE (AWQ/OmniQuant-style clip search)."""

    NAME = "int4_mse_clip"

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
        scale16 = mse_clip_scale(groups, 7).clamp(min=FP16_MIN_SCALE).to(torch.float16)
        codes = pack_int4_symmetric(groups, scale16.float())
        return pack_tensors({"codes": codes, "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        groups = unpack_int4_symmetric(t["codes"], t["scale"].float(), self.group_size)
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
