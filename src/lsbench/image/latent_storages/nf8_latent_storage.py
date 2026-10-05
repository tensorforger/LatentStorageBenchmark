import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    lookup,
    nearest_code,
    normal_float_codebook,
    pack_tensors,
    to_groups,
    unpack_tensors,
)


class NF8LatentStorage(ImageLatentStorage):
    """8-bit NormalFloat (QLoRA codebook generalized to 256 levels) with an fp16 absmax scale per group."""

    NAME = "nf8"

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
        groups = to_groups(x, self.group_size)
        scale16 = fp16_scale(groups.abs().amax(-1, keepdim=True), 1.0)
        book = normal_float_codebook(8).to(x.device)
        codes = nearest_code(groups / scale16.float(), book)
        return pack_tensors({"codes": codes, "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        book = normal_float_codebook(8).to(device)
        groups = lookup(book, t["codes"]) * t["scale"].float()
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
