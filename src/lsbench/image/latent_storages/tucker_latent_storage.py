import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import pack_tensors, unpack_tensors


def _top_eigvecs(gram: torch.Tensor, rank: int) -> torch.Tensor:
    """[B, N, N] symmetric -> [B, N, rank] leading eigenvectors (left singular vectors of the unfolding)."""
    _, vecs = torch.linalg.eigh(gram)
    return vecs[..., -rank:].flip(-1)


class TuckerLatentStorage(ImageLatentStorage):
    """Truncated HOSVD (Tucker) per sample: orthogonal factors for channel/height/width modes
    plus a small core, all stored as fp16."""

    NAME = "tucker"

    def __init__(self, rank_c: int = 16, rank_hw: int = 32):
        super().__init__()
        self.rank_c = rank_c
        self.rank_hw = rank_hw

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        _, c, h, w = x.shape
        rc, rh, rw = min(self.rank_c, c), min(self.rank_hw, h), min(self.rank_hw, w)
        uc = _top_eigvecs(torch.einsum("bchw,bdhw->bcd", x, x), rc)
        uh = _top_eigvecs(torch.einsum("bchw,bcgw->bhg", x, x), rh)
        uw = _top_eigvecs(torch.einsum("bchw,bchv->bwv", x, x), rw)
        core = torch.einsum("bchw,bcr,bhs,bwt->brst", x, uc, uh, uw)
        tensors = {
            "core": core.half(),
            "uc": uc.half(),
            "uh": uh.half(),
            "uw": uw.half(),
        }
        return pack_tensors(tensors, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, _ = unpack_tensors(data, device)
        return torch.einsum(
            "brst,bcr,bhs,bwt->bchw",
            t["core"].float(),
            t["uc"].float(),
            t["uh"].float(),
            t["uw"].float(),
        )

    def get_storage_name(self) -> str:
        return f"{self.NAME}_c{self.rank_c}_s{self.rank_hw}"
