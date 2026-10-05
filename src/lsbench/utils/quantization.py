"""Shared building blocks for the quantizing latent storages (grouping, nibble packing, codebooks, containers)."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from safetensors.torch import load, save

FP16_MIN_SCALE = (
    2.0**-14
)  # smallest normal fp16, keeps stored fp16 scales exact and non-zero
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0


def pack_tensors(tensors: dict[str, torch.Tensor], shape) -> bytes:
    out = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    out["shape"] = torch.tensor(list(shape), dtype=torch.int32)
    return save(out)


def unpack_tensors(data: bytes, device) -> tuple[dict[str, torch.Tensor], list[int]]:
    tensors = load(data)
    shape = tensors.pop("shape").tolist()
    return {k: v.to(device) for k, v in tensors.items()}, shape


def to_groups(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """[B, ...] -> [B, G, group_size]; groups never cross samples (zero padded per sample)."""
    flat = x.reshape(x.shape[0], -1)
    pad = (-flat.shape[1]) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat.reshape(x.shape[0], -1, group_size)


def from_groups(groups: torch.Tensor, shape) -> torch.Tensor:
    n = math.prod(shape[1:])
    return groups.reshape(shape[0], -1)[:, :n].reshape(shape)


def fp16_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    return (amax / qmax).clamp(min=FP16_MIN_SCALE).to(torch.float16)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes in [0, 15], [B, ...] -> uint8 [B, ceil(n / 2)]."""
    flat = codes.reshape(codes.shape[0], -1)
    if flat.shape[1] % 2:
        flat = F.pad(flat, (0, 1))
    return flat[:, 0::2] | (flat[:, 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """uint8 [B, ceil(n / 2)] -> uint8 [B, n]."""
    both = torch.stack((packed & 0xF, packed >> 4), dim=-1)
    return both.reshape(packed.shape[0], -1)[:, :n]


def nearest_code(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Index of the nearest entry of a sorted 1D codebook, uint8 (<= 256 entries)."""
    mids = (codebook[1:] + codebook[:-1]) / 2
    return torch.bucketize(x.contiguous(), mids).to(torch.uint8)


def lookup(codebook: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """codebook [K] or [B, K], codes [B, ...] -> values with the shape of codes."""
    book = codebook.reshape(-1, codebook.shape[-1]).expand(codes.shape[0], -1)
    return book.gather(1, codes.reshape(codes.shape[0], -1).long()).reshape(codes.shape)


def pack_int4_symmetric(groups: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """groups [B, G, g] and float scale [B, G, 1] -> packed nibbles, levels -7..7."""
    codes = (torch.round(groups / scale).clamp(-7, 7) + 8).to(torch.uint8)
    return pack_nibbles(codes)


def unpack_int4_symmetric(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    b, g = scale.shape[:2]
    codes = unpack_nibbles(packed, g * group_size).reshape(b, g, group_size)
    return (codes.float() - 8) * scale


def e2m1_codebook() -> torch.Tensor:
    """16 sorted FP4 E2M1 values (code 7 is -0, code 8 is +0)."""
    pos = torch.tensor(E2M1_POSITIVE)
    return torch.cat([-pos.flip(0), pos])


@lru_cache(maxsize=None)
def normal_float_codebook(bits: int) -> torch.Tensor:
    """QLoRA NormalFloat codebook generalized to `bits`: normal quantiles in [-1, 1] with an exact zero."""
    n = 2**bits
    offset = 1 - 0.5 * (1 / (2 * n) + 1 / (2 * (n - 1)))
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    pos = normal.icdf(torch.linspace(offset, 0.5, n // 2 + 1, dtype=torch.float64)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, n // 2, dtype=torch.float64)[:-1])
    zero = torch.zeros(1, dtype=torch.float64)
    book = torch.cat([neg, zero, pos]).sort().values
    return (book / book.abs().max()).float()


@lru_cache(maxsize=None)
def hadamard_matrix(n: int) -> torch.Tensor:
    """Orthonormal Sylvester Walsh-Hadamard matrix (symmetric, so it is its own inverse)."""
    if n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def mse_clip_scale(
    groups: torch.Tensor, qmax: int, steps: int = 20, min_ratio: float = 0.5
) -> torch.Tensor:
    """Symmetric scale [..., 1] per last-dim group, searched over clip ratios to minimize the squared error."""
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=FP16_MIN_SCALE * qmax)
    best_scale, best_err = None, None
    for ratio in torch.linspace(min_ratio, 1.0, steps).tolist():
        scale = amax * ratio / qmax
        deq = torch.round(groups / scale).clamp(-qmax, qmax) * scale
        err = (deq - groups).pow(2).sum(-1, keepdim=True)
        if best_scale is None:
            best_scale, best_err = scale, err
        else:
            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_err = torch.where(better, err, best_err)
    return best_scale
