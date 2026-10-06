import math
import struct

import numpy as np
import torch
import zstandard

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import from_square_groups, to_square_groups


def _dct2_matrix(block: int) -> torch.Tensor:
    """Orthonormal 2D DCT-II as [n, n] matrix acting on row-major flattened block x block patches."""
    k = torch.arange(block, dtype=torch.float64)
    d = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * block))
    d *= math.sqrt(2.0 / block)
    d[0] /= math.sqrt(2.0)
    return torch.kron(d, d).float()


class DCTZstdLatentStorage(ImageLatentStorage):
    """Transform coding: blockwise 2D DCT per channel, uniform quantization (one step for all
    coefficients; orthonormal transform => MSE ~ step^2/12), frequency-major int16 byte planes, zstd."""

    NAME = "dct_zstd"

    def __init__(self, step: float = 0.1, block: int = 8, level: int = 19):
        super().__init__()
        self.step = step
        self.block = block
        self.level = level

    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            bytes
        """
        x = latents.detach().float().cpu()
        groups = to_square_groups(x, self.block)  # [B, G, n]
        coef = groups @ _dct2_matrix(self.block).T
        q = torch.round(coef / self.step).clamp(-32768, 32767).to(torch.int16)
        q = (
            q.permute(2, 0, 1).contiguous().numpy()
        )  # frequency-major: similar statistics adjacent
        planes = np.ascontiguousarray(q.view(np.uint8).reshape(-1, 2).T)
        header = struct.pack("<4ifi", *x.shape, self.step, self.block)
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
        *shape, step, block = struct.unpack("<4ifi", data[:24])
        b, c, h, w = shape
        n = block * block
        g = c * (-(-h // block)) * (-(-w // block))
        raw = np.frombuffer(
            zstandard.ZstdDecompressor().decompress(data[24:]), dtype=np.uint8
        )
        q = np.ascontiguousarray(raw.reshape(2, -1).T).view(np.int16).reshape(n, b, g)
        coef = torch.from_numpy(q.copy()).float().permute(1, 2, 0) * step
        groups = coef @ _dct2_matrix(block)
        return from_square_groups(groups, shape, block).contiguous().to(device)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_s{self.step:g}_b{self.block}"
