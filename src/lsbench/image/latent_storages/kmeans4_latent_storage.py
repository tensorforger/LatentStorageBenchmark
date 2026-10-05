import torch

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.utils.quantization import (
    fp16_scale,
    from_groups,
    lookup,
    nearest_code,
    pack_nibbles,
    pack_tensors,
    to_groups,
    unpack_nibbles,
    unpack_tensors,
)

MAX_FIT_SAMPLES = 262144


def _fit_codebook(values: torch.Tensor, levels: int, iters: int) -> torch.Tensor:
    """1D k-means (Lloyd-Max) codebook, sorted, fp16-representable."""
    stride = max(1, values.numel() // MAX_FIT_SAMPLES)
    v = values.reshape(-1)[::stride]
    q = (torch.arange(levels, device=v.device, dtype=v.dtype) + 0.5) / levels
    book = torch.quantile(v, q)
    for _ in range(iters):
        assign = nearest_code(v, book).long()
        sums = torch.zeros_like(book).index_add_(0, assign, v)
        counts = torch.bincount(assign, minlength=levels).to(v.dtype)
        book = torch.where(counts > 0, sums / counts.clamp(min=1), book)
    return book.to(torch.float16).float().sort().values


class KMeans4LatentStorage(ImageLatentStorage):
    """SqueezeLLM-style non-uniform 4-bit: per-group absmax normalization, then a 16-entry Lloyd-Max (k-means) codebook fitted per sample and stored in the bytes."""

    NAME = "kmeans4"
    LEVELS = 16
    ITERS = 15

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
        y = groups / scale16.float()
        books, codes = [], []
        for sample in y:
            book = _fit_codebook(sample, self.LEVELS, self.ITERS)
            books.append(book)
            codes.append(nearest_code(sample, book))
        return pack_tensors(
            {
                "codes": pack_nibbles(torch.stack(codes)),
                "scale": scale16,
                "book": torch.stack(books).to(torch.float16),
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
        b, n_groups = t["scale"].shape[:2]
        codes = unpack_nibbles(t["codes"], n_groups * self.group_size).reshape(
            b, n_groups, self.group_size
        )
        groups = lookup(t["book"].float(), codes) * t["scale"].float()
        return from_groups(groups, shape)

    def get_storage_name(self) -> str:
        return f"{self.NAME}_g{self.group_size}"
