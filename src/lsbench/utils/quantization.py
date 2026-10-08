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


def to_square_groups(x: torch.Tensor, block: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, G, block * block]; each group is a block x block patch of one channel (zero padded)."""
    b, c, h, w = x.shape
    x = F.pad(x, (0, (-w) % block, 0, (-h) % block))
    hb, wb = x.shape[2] // block, x.shape[3] // block
    x = x.reshape(b, c, hb, block, wb, block).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(b, c * hb * wb, block * block)


def from_square_groups(groups: torch.Tensor, shape, block: int) -> torch.Tensor:
    b, c, h, w = shape
    hb, wb = -(-h // block), -(-w // block)
    x = groups.reshape(b, c, hb, wb, block, block).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(b, c, hb * block, wb * block)[:, :, :h, :w]


def to_cube_groups(x: torch.Tensor, block: int) -> torch.Tensor:
    """[B, C, T, H, W] -> [B, G, block^3]; each group is a block^3 spatiotemporal cube of one channel (zero padded)."""
    b, c, t, h, w = x.shape
    x = F.pad(x, (0, (-w) % block, 0, (-h) % block, 0, (-t) % block))
    tb, hb, wb = (s // block for s in x.shape[2:])
    x = x.reshape(b, c, tb, block, hb, block, wb, block).permute(0, 1, 2, 4, 6, 3, 5, 7)
    return x.reshape(b, c * tb * hb * wb, block**3)


def from_cube_groups(groups: torch.Tensor, shape, block: int) -> torch.Tensor:
    b, c, t, h, w = shape
    tb, hb, wb = -(-t // block), -(-h // block), -(-w // block)
    x = groups.reshape(b, c, tb, hb, wb, block, block, block).permute(
        0, 1, 2, 5, 3, 6, 4, 7
    )
    return x.reshape(b, c, tb * block, hb * block, wb * block)[:, :, :t, :h, :w]


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


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes in [0, 2^bits), [B, ...] -> uint8 [B, ceil(n * bits / 8)], LSB first."""
    flat = codes.reshape(codes.shape[0], -1)
    if bits == 8:
        return flat.contiguous()
    shifts = torch.arange(bits, device=flat.device, dtype=torch.uint8)
    bit_rows = ((flat.unsqueeze(-1) >> shifts) & 1).reshape(flat.shape[0], -1)
    bit_rows = F.pad(bit_rows, (0, (-bit_rows.shape[1]) % 8))
    weights = 1 << torch.arange(8, device=flat.device, dtype=torch.uint8)
    return (bit_rows.reshape(flat.shape[0], -1, 8) * weights).sum(-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    """uint8 [B, ceil(n * bits / 8)] -> uint8 [B, n]."""
    if bits == 8:
        return packed[:, :n]
    b = packed.shape[0]
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bit_rows = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(b, -1)[:, : n * bits]
    weights = 1 << torch.arange(bits, device=packed.device, dtype=torch.uint8)
    return (bit_rows.reshape(b, n, bits) * weights).sum(-1).to(torch.uint8)


def nearest_code(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Index of the nearest entry of a sorted 1D codebook, uint8 (<= 256 entries)."""
    mids = (codebook[1:] + codebook[:-1]) / 2
    return torch.bucketize(x.contiguous(), mids).to(torch.uint8)


def lookup(codebook: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """codebook [K] or [B, K], codes [B, ...] -> values with the shape of codes."""
    book = codebook.reshape(-1, codebook.shape[-1]).expand(codes.shape[0], -1)
    return book.gather(1, codes.reshape(codes.shape[0], -1).long()).reshape(codes.shape)


def quantize_int4_asym(groups: torch.Tensor) -> dict[str, torch.Tensor]:
    """groups [B, G, g] -> packed 4-bit codes, fp16 scale and packed 4-bit zero-point per group."""
    lo = groups.amin(-1, keepdim=True).clamp(max=0)
    hi = groups.amax(-1, keepdim=True).clamp(min=0)
    scale16 = ((hi - lo) / 15).clamp(min=FP16_MIN_SCALE).to(torch.float16)
    scale = scale16.float()
    zero = torch.round(-lo / scale).clamp(0, 15)
    q = (torch.round(groups / scale) + zero).clamp(0, 15).to(torch.uint8)
    return {
        "codes": pack_nibbles(q),
        "scale": scale16,
        "zero": pack_nibbles(zero.to(torch.uint8)),
    }


def dequantize_int4_asym(t: dict[str, torch.Tensor], group_size: int) -> torch.Tensor:
    b, n_groups = t["scale"].shape[:2]
    q = unpack_nibbles(t["codes"], n_groups * group_size).reshape(
        b, n_groups, group_size
    )
    zero = unpack_nibbles(t["zero"], n_groups).reshape(b, n_groups, 1)
    return (q.float() - zero.float()) * t["scale"].float()


def _minmax_grid(
    lo: torch.Tensor,
    hi: torch.Tensor,
    cmin: torch.Tensor,
    cmax: torch.Tensor,
    range_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group start and span from the range indices; shared by quantize and dequantize."""
    d = ((cmax - cmin) / (2**range_bits - 1)).clamp(min=1e-8)
    return cmin + lo * d, (hi - lo) * d


def fit_lloyd_table(
    u: torch.Tensor,
    w: torch.Tensor,
    levels: int,
    iters: int = 20,
    max_points: int = 2**17,
) -> torch.Tensor:
    """Weighted 1D Lloyd-Max levels for values u in [0, 1], sorted and fp16-representable."""
    stride = max(1, u.numel() // max_points)
    u, w = u[::stride], w[::stride]
    w = w / w.mean().clamp(min=1e-30)
    table = torch.linspace(0, 1, levels, device=u.device)
    for _ in range(iters):
        assign = nearest_code(u, table).long()
        num = torch.zeros_like(table).index_add_(0, assign, u * w)
        den = torch.zeros_like(table).index_add_(0, assign, w)
        table = torch.where(den > 0, num / den.clamp(min=1e-30), table)
    return table.to(torch.float16).float().sort().values


def quantize_minmax(
    groups: torch.Tensor,
    refs: int,
    bits: int,
    range_bits: int = 8,
    lloyd: bool = False,
    lloyd_per_channel: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Asymmetric `bits`-bit quantization on the true [min, max] of each group (no forced zero,
    all codes usable). The group min/max are stored as `range_bits`-bit indices into the range
    of a reference block (`refs` equal blocks per sample, e.g. channels).
    With `lloyd` the levels inside [min, max] are a per-sample Lloyd-Max table (stored in the
    result) instead of uniform; with `lloyd_per_channel` one table per reference block (channel).
    groups [B, G, g] with G % refs == 0, blocks contiguous along G.
    """
    b, n_groups, n = groups.shape
    x = groups.reshape(b, refs, -1, n)
    levels = 2**range_bits - 1
    cmin = x.amin(dim=(2, 3), keepdim=True)
    cmax = x.amax(dim=(2, 3), keepdim=True)
    d = ((cmax - cmin) / levels).clamp(min=1e-8)
    # floor/ceil so the quantized range always covers the group
    lo = torch.floor((x.amin(-1, keepdim=True) - cmin) / d).clamp(0, levels - 1)
    hi = torch.ceil((x.amax(-1, keepdim=True) - cmin) / d).clamp(max=levels)
    hi = torch.maximum(hi, lo + 1)
    start, span = _minmax_grid(lo, hi, cmin, cmax, range_bits)
    out = {}
    if lloyd:
        rows = (b, refs) if lloyd_per_channel else (b,)  # one table per row
        u = ((x - start) / span).reshape(-1, x.numel() // math.prod(rows))
        w = (span.expand_as(x) ** 2).reshape_as(u)  # squared error scales with span^2
        tables = torch.stack([fit_lloyd_table(ui, wi, 2**bits) for ui, wi in zip(u, w)])
        q = torch.stack([nearest_code(ui, ti) for ui, ti in zip(u, tables)]).reshape(
            b, -1
        )
        out["table"] = tables.reshape(*rows, -1).to(torch.float16)
    else:
        step = span / (2**bits - 1)
        q = torch.round((x - start) / step).clamp(0, 2**bits - 1).to(torch.uint8)
    out |= {
        "codes": pack_bits(q, bits),
        "lo": pack_bits(lo.reshape(b, n_groups).to(torch.uint8), range_bits),
        "hi": pack_bits(hi.reshape(b, n_groups).to(torch.uint8), range_bits),
        "cmin": cmin.reshape(b, refs),
        "cmax": cmax.reshape(b, refs),
    }
    return out


def dequantize_minmax(
    t: dict[str, torch.Tensor],
    group_size: int,
    bits: int,
    range_bits: int = 8,
    n_groups: int | None = None,
) -> torch.Tensor:
    """n_groups can be omitted only when range_bits == 8 (the index tensors are then unpacked)."""
    b = t["lo"].shape[0]
    if n_groups is None:
        n_groups = t["lo"].shape[1]
    refs = t["cmin"].shape[1]
    q = unpack_bits(t["codes"], n_groups * group_size, bits).reshape(
        b, refs, -1, group_size
    )
    start, span = _minmax_grid(
        unpack_bits(t["lo"], n_groups, range_bits).float().reshape(b, refs, -1, 1),
        unpack_bits(t["hi"], n_groups, range_bits).float().reshape(b, refs, -1, 1),
        t["cmin"].reshape(b, refs, 1, 1),
        t["cmax"].reshape(b, refs, 1, 1),
        range_bits,
    )
    if "table" in t:
        table = t["table"].float()
        if table.dim() == 3:  # per-channel tables [B, refs, K]
            vals = table.gather(2, q.reshape(b, refs, -1).long()).reshape(q.shape)
        else:
            vals = lookup(table, q)
        x = vals * span + start
    else:
        x = q.float() * (span / (2**bits - 1)) + start
    return x.reshape(b, n_groups, group_size)


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
