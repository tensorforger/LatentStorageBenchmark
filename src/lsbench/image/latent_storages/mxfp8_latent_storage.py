import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    FP8_E4M3_MAX,
    from_groups,
    pack_tensors,
    to_groups,
    unpack_tensors,
)


class MXFP8LatentStorage(ImageLatentStorage):
    """OCP MXFP8: blocks of 32 values share a power-of-two (E8M0) scale, elements are FP8 E4M3."""

    NAME = "mxfp8"
    BLOCK = 32

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        groups = to_groups(x, self.BLOCK)
        amax = groups.abs().amax(-1, keepdim=True)
        # ceil so the block maximum never clips
        exp = torch.ceil(torch.log2(amax / FP8_E4M3_MAX)).clamp(-127, 127)
        q = (groups / torch.exp2(exp)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        return pack_tensors(
            {"q": q.to(torch.float8_e4m3fn), "exp": (exp + 127).to(torch.uint8)},
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
        groups = t["q"].float() * torch.exp2(t["exp"].float() - 127)
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return self.NAME
