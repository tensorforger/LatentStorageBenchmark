import torch

from lsbench.video.latent_storages.video_latent_storage import VideoLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    hadamard_matrix,
    pack_int4_symmetric,
    pack_tensors,
    to_groups,
    unpack_int4_symmetric,
    unpack_tensors,
)


class INT4HadamardLatentStorage(VideoLatentStorage):
    """QuaRot/QuIP-style incoherence processing: Walsh-Hadamard rotation of each group, then symmetric INT4 with an fp16 absmax scale per group."""

    NAME = "int4_hadamard"

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
        h = hadamard_matrix(self.group_size).to(x.device)
        rot = to_groups(x, self.group_size) @ h
        scale16 = fp16_scale(rot.abs().amax(-1, keepdim=True), 7)
        codes = pack_int4_symmetric(rot, scale16.float())
        return pack_tensors({"codes": codes, "scale": scale16}, x.shape)

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        t, shape = unpack_tensors(data, device)
        h = hadamard_matrix(self.group_size).to(device)
        rot = unpack_int4_symmetric(t["codes"], t["scale"].float(), self.group_size)
        return from_groups(rot @ h, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
