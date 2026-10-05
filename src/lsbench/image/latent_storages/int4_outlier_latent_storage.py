import math

import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    pack_int4_symmetric,
    pack_tensors,
    to_groups,
    unpack_int4_symmetric,
    unpack_tensors,
)


class INT4OutlierLatentStorage(ImageLatentStorage):
    """SqueezeLLM/SpQR-style dense-and-sparse: the largest `outlier_fraction` of values per sample stay in fp16, the rest is group-wise INT4."""

    NAME = "int4_outlier"

    def __init__(self, group_size: int = 32, outlier_fraction: float = 0.005):
        super().__init__()
        self.group_size = group_size
        self.outlier_fraction = outlier_fraction

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        b = x.shape[0]
        flat = x.reshape(b, -1)
        k = max(1, math.ceil(self.outlier_fraction * flat.shape[1]))
        idx = flat.abs().topk(k, dim=1).indices
        vals = flat.gather(1, idx).to(torch.float16)
        dense = flat.scatter(1, idx, 0.0).reshape(x.shape)
        groups = to_groups(dense, self.group_size)
        scale16 = fp16_scale(groups.abs().amax(-1, keepdim=True), 7)
        codes = pack_int4_symmetric(groups, scale16.float())
        return pack_tensors(
            {
                "codes": codes,
                "scale": scale16,
                "idx": idx.to(torch.int32),
                "vals": vals,
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
        groups = unpack_int4_symmetric(t["codes"], t["scale"].float(), self.group_size)
        flat = from_groups(groups, shape).reshape(shape[0], -1)
        flat.scatter_(1, t["idx"].long(), t["vals"].float())
        return flat.reshape(shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}_p{self.outlier_fraction * 100:g}"
