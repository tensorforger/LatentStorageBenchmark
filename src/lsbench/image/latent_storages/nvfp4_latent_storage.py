import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    E2M1_MAX,
    FP8_E4M3_MAX,
    e2m1_codebook,
    from_groups,
    lookup,
    nearest_code,
    pack_nibbles,
    pack_tensors,
    to_groups,
    unpack_nibbles,
    unpack_tensors,
)


class NVFP4LatentStorage(ImageLatentStorage):
    """NVIDIA NVFP4: FP4 E2M1 elements, FP8 E4M3 scale per block of 16, plus an fp32 per-sample scale."""

    NAME = "nvfp4"
    BLOCK = 16

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b = x.shape[0]
        groups = to_groups(x, self.BLOCK)
        global_scale = (
            groups.abs().reshape(b, -1).amax(1) / (E2M1_MAX * FP8_E4M3_MAX)
        ).clamp(min=1e-12)
        g = global_scale.view(b, 1, 1)
        block_scale = (groups.abs().amax(-1, keepdim=True) / E2M1_MAX / g).clamp(
            max=FP8_E4M3_MAX
        )
        block_scale8 = block_scale.to(torch.float8_e4m3fn)
        eff = block_scale8.float() * g
        eff = torch.where(eff == 0, torch.ones_like(eff), eff)
        codes = nearest_code(groups / eff, e2m1_codebook().to(x.device))
        return pack_tensors(
            {
                "codes": pack_nibbles(codes),
                "block_scale": block_scale8,
                "global_scale": global_scale,
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
        b, n_groups = t["block_scale"].shape[:2]
        codes = unpack_nibbles(t["codes"], n_groups * self.BLOCK).reshape(
            b, n_groups, self.BLOCK
        )
        eff = t["block_scale"].float() * t["global_scale"].view(b, 1, 1)
        groups = lookup(e2m1_codebook().to(device), codes) * eff
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return self.NAME
