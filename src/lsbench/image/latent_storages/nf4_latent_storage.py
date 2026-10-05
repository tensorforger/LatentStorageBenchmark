import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
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


class NF4LatentStorage(ImageLatentStorage):
    """QLoRA NormalFloat4 with an fp16 absmax scale per group."""

    NAME = "nf4"

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
        book = normal_float_codebook(4).to(x.device)
        codes = nearest_code(groups / scale16.float(), book)
        return pack_tensors({"codes": pack_nibbles(codes), "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        b, n_groups = t["scale"].shape[:2]
        codes = unpack_nibbles(t["codes"], n_groups * self.group_size).reshape(
            b, n_groups, self.group_size
        )
        book = normal_float_codebook(4).to(device)
        return from_groups(lookup(book, codes) * t["scale"].float(), shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
