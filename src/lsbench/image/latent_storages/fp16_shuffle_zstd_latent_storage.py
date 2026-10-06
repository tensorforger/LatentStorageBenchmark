import struct

import numpy as np
import torch
import zstandard

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage


class FP16ShuffleZstdLatentStorage(ImageLatentStorage):
    """Lossless fp16 + Blosc-style byte shuffle (exponent/mantissa byte planes) + zstd."""

    NAME = "fp16_shuffle_zstd"

    def __init__(self, level: int = 19):
        super().__init__()
        self.level = level

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        a = latents.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
        planes = np.ascontiguousarray(a.view(np.uint8).reshape(-1, 2).T)
        header = struct.pack("<4i", *a.shape)
        return header + zstandard.ZstdCompressor(level=self.level).compress(
            planes.tobytes()
        )

    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        shape = struct.unpack("<4i", data[:16])
        raw = np.frombuffer(
            zstandard.ZstdDecompressor().decompress(data[16:]), dtype=np.uint8
        )
        a = np.ascontiguousarray(raw.reshape(2, -1).T).view(np.float16).reshape(shape)
        return torch.from_numpy(a).to(device)

    def get_storage_name(self) -> str:
        return self.NAME
