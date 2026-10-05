import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    E2M1_MAX,
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


class MXFP4LatentStorage(ImageLatentStorage):
    """OCP MXFP4: FP4 E2M1 elements with a power-of-two (E8M0) scale per block of 32."""

    NAME = "mxfp4"
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
        exp = torch.ceil(torch.log2(amax / E2M1_MAX)).clamp(-127, 127)
        codes = nearest_code(groups / torch.exp2(exp), e2m1_codebook().to(x.device))
        return pack_tensors(
            {"codes": pack_nibbles(codes), "exp": (exp + 127).to(torch.uint8)},
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
        b, n_groups = t["exp"].shape[:2]
        codes = unpack_nibbles(t["codes"], n_groups * self.BLOCK).reshape(
            b, n_groups, self.BLOCK
        )
        groups = lookup(e2m1_codebook().to(device), codes) * torch.exp2(
            t["exp"].float() - 127
        )
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return self.NAME
