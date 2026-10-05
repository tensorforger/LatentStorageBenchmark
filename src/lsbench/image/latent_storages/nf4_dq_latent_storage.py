import math

import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    FP16_MIN_SCALE,
    fp16_scale,
    from_groups,
    lookup,
    nearest_code,
    normal_float_codebook,
    pack_nibbles,
    pack_tensors,
    to_groups,
    unpack_nibbles,
    unpack_tensors,
)

SCALE_GROUP = 256


def _dequant_scales(
    q: torch.Tensor, scale2: torch.Tensor, mean: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Second-level INT8 scales -> first-level absmax [B, G, 1] (shared by serialize and deserialize)."""
    absmax = q.float() * scale2.float() + mean.view(-1, 1, 1)
    absmax = absmax.reshape(q.shape[0], -1)[:, :n_groups]
    return absmax.clamp(min=FP16_MIN_SCALE).unsqueeze(-1)


class NF4DoubleQuantLatentStorage(ImageLatentStorage):
    """NF4 with QLoRA double quantization: the per-group absmax scales are themselves stored as INT8 (mean-centered, groups of 256)."""

    NAME = "nf4_dq"

    def __init__(self, group_size: int = 64):
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
        b = x.shape[0]
        groups = to_groups(x, self.group_size)
        n_groups = groups.shape[1]
        absmax = groups.abs().amax(-1)
        mean = absmax.mean(1)
        second = to_groups(absmax - mean.view(b, 1), SCALE_GROUP)
        scale2 = fp16_scale(second.abs().amax(-1, keepdim=True), 127)
        q = torch.round(second / scale2.float()).clamp(-127, 127).to(torch.int8)
        scale = _dequant_scales(q, scale2, mean, n_groups)
        codes = nearest_code(groups / scale, normal_float_codebook(4).to(x.device))
        return pack_tensors(
            {"codes": pack_nibbles(codes), "q": q, "scale2": scale2, "mean": mean},
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
        b = shape[0]
        n_groups = -(-math.prod(shape[1:]) // self.group_size)
        scale = _dequant_scales(t["q"], t["scale2"], t["mean"], n_groups)
        codes = unpack_nibbles(t["codes"], n_groups * self.group_size).reshape(
            b, n_groups, self.group_size
        )
        book = normal_float_codebook(4).to(device)
        return from_groups(lookup(book, codes) * scale, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
