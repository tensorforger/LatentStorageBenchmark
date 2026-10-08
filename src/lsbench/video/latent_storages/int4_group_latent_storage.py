import torch

from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    pack_int4_symmetric,
    pack_tensors,
    to_groups,
    unpack_int4_symmetric,
    unpack_tensors,
)


class INT4GroupLatentStorage(VideoLatentStorage):
    """Round-to-nearest symmetric INT4 (GPTQ/AWQ-style group quantization) with an fp16 absmax scale per group."""

    NAME = "int4_group"

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.group_size = group_size

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float()
        groups = to_groups(x, self.group_size)
        scale16 = fp16_scale(groups.abs().amax(-1, keepdim=True), 7)
        codes = pack_int4_symmetric(groups, scale16.float())
        return pack_tensors({"codes": codes, "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        groups = unpack_int4_symmetric(t["codes"], t["scale"].float(), self.group_size)
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
