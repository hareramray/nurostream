"""GGML quantization formats -> float tensors.

Written against torch rather than numpy on purpose: the raw quantized bytes get
uploaded to the GPU *compressed* and expanded there, so PCIe carries 4.5 bits
per weight instead of 32. On this machine that is a ~7x reduction in transfer.

Every kernel here is block-vectorised. Blocks are independent, and GGUF stores
2D tensors row-major with each row spanning a whole number of blocks, so any
contiguous row range can be dequantized on its own. That is what makes
neuron-block streaming possible.
"""
from __future__ import annotations

import torch

from ..format.gguf import TYPE_LAYOUT, GGMLType


def _f16_field(buf: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret a 2-byte little-endian field as float32, shape (nb, 1)."""
    return buf[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def _k_scales_mins(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the 6-bit scale/min pairs shared by Q4_K and Q5_K.

    Mirrors ggml's get_scale_min_k4 for all 8 sub-blocks at once.
    """
    q = scales.to(torch.int32)  # (nb, 12)
    # sub-blocks 0..3: plain 6-bit fields
    d_lo = q[:, 0:4] & 63
    m_lo = q[:, 4:8] & 63
    # sub-blocks 4..7: 4 low bits from bytes 8..11, 2 high bits borrowed
    d_hi = (q[:, 8:12] & 0x0F) | ((q[:, 0:4] >> 6) << 4)
    m_hi = (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)
    sc = torch.cat([d_lo, d_hi], dim=1).to(torch.float32)  # (nb, 8)
    mn = torch.cat([m_lo, m_hi], dim=1).to(torch.float32)  # (nb, 8)
    return sc, mn


def dequant_q4_k(buf: torch.Tensor) -> torch.Tensor:
    """Q4_K: 256 elements / 144 bytes. d|dmin|scales[12]|qs[128]."""
    nb = buf.shape[0]
    d = _f16_field(buf, 0, 2)
    dmin = _f16_field(buf, 2, 4)
    sc, mn = _k_scales_mins(buf[:, 4:16])

    qs = buf[:, 16:144].reshape(nb, 4, 32).to(torch.int32)
    low = qs & 0x0F
    high = qs >> 4
    # interleave: sub-block order is low(g), high(g) for each 64-element group
    q = torch.stack([low, high], dim=2).reshape(nb, 8, 32).to(torch.float32)

    scale = (d * sc).unsqueeze(-1)  # (nb, 8, 1)
    offset = (dmin * mn).unsqueeze(-1)
    return (scale * q - offset).reshape(nb, 256)


def dequant_q5_k(buf: torch.Tensor) -> torch.Tensor:
    """Q5_K: 256 elements / 176 bytes. d|dmin|scales[12]|qh[32]|qs[128]."""
    nb = buf.shape[0]
    d = _f16_field(buf, 0, 2)
    dmin = _f16_field(buf, 2, 4)
    sc, mn = _k_scales_mins(buf[:, 4:16])

    qh = buf[:, 16:48].to(torch.int32)  # (nb, 32), one bit per sub-block
    qs = buf[:, 48:176].reshape(nb, 4, 32).to(torch.int32)

    low = qs & 0x0F
    high = qs >> 4
    # the 5th bit lives in qh; group g consumes bit pair (2g, 2g+1)
    g = torch.arange(4, device=buf.device).view(1, 4, 1)
    hbit_lo = ((qh.unsqueeze(1) >> (2 * g)) & 1) << 4
    hbit_hi = ((qh.unsqueeze(1) >> (2 * g + 1)) & 1) << 4

    q = torch.stack([low + hbit_lo, high + hbit_hi], dim=2)
    q = q.reshape(nb, 8, 32).to(torch.float32)

    scale = (d * sc).unsqueeze(-1)
    offset = (dmin * mn).unsqueeze(-1)
    return (scale * q - offset).reshape(nb, 256)


def dequant_q6_k(buf: torch.Tensor) -> torch.Tensor:
    """Q6_K: 256 elements / 210 bytes. ql[128]|qh[64]|scales[16]|d."""
    nb = buf.shape[0]
    ql = buf[:, 0:128].reshape(nb, 2, 64).to(torch.int32)
    qh = buf[:, 128:192].reshape(nb, 2, 32).to(torch.int32)
    sc = buf[:, 192:208].view(torch.int8).reshape(nb, 2, 8).to(torch.float32)
    d = _f16_field(buf, 208, 210).unsqueeze(-1)  # (nb, 1, 1)

    l = torch.arange(32, device=buf.device)
    is_ = (l // 16).view(1, 1, 32).expand(nb, 2, 32)

    # four 32-wide lanes per half, each taking a different 2-bit slice of qh
    q1 = (ql[:, :, 0:32] & 0x0F) | (((qh >> 0) & 3) << 4)
    q2 = (ql[:, :, 32:64] & 0x0F) | (((qh >> 2) & 3) << 4)
    q3 = (ql[:, :, 0:32] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[:, :, 32:64] >> 4) | (((qh >> 6) & 3) << 4)

    out = buf.new_empty((nb, 2, 128), dtype=torch.float32)
    for lane, (qv, base) in enumerate(
        ((q1, 0), (q2, 2), (q3, 4), (q4, 6))
    ):
        s = torch.gather(sc, 2, is_ + base)
        out[:, :, lane * 32 : (lane + 1) * 32] = d * s * (qv - 32).to(torch.float32)
    return out.reshape(nb, 256)


def dequant_q8_0(buf: torch.Tensor) -> torch.Tensor:
    """Q8_0: 32 elements / 34 bytes. d|qs[32]."""
    nb = buf.shape[0]
    d = _f16_field(buf, 0, 2)
    q = buf[:, 2:34].view(torch.int8).to(torch.float32)
    return (d * q).reshape(nb, 32)


def dequant_q4_0(buf: torch.Tensor) -> torch.Tensor:
    """Q4_0: 32 elements / 18 bytes. d|qs[16], nibbles split 0..15 / 16..31."""
    nb = buf.shape[0]
    d = _f16_field(buf, 0, 2)
    qs = buf[:, 2:18].to(torch.int32)
    lo = (qs & 0x0F) - 8
    hi = (qs >> 4) - 8
    q = torch.cat([lo, hi], dim=1).to(torch.float32)
    return (d * q).reshape(nb, 32)


def dequant_mxfp4(buf: torch.Tensor) -> torch.Tensor:
    """MXFP4: one E8M0 exponent and 16 packed E2M1 bytes per 32 values."""
    exponent = buf[:, :1].to(torch.int32)
    bits = torch.where(exponent < 2, 0x00200000 << exponent,
                       (exponent - 1) << 23)
    scale = bits.contiguous().view(torch.float32)
    packed = buf[:, 1:]
    codes = torch.cat((packed & 15, packed >> 4), dim=1).long()
    values = torch.tensor((0, 1, 2, 3, 4, 6, 8, 12,
                           0, -1, -2, -3, -4, -6, -8, -12),
                          device=buf.device, dtype=torch.float32)
    return scale * values[codes]


_KERNELS = {
    GGMLType.MXFP4: dequant_mxfp4,
    GGMLType.Q4_K: dequant_q4_k,
    GGMLType.Q5_K: dequant_q5_k,
    GGMLType.Q6_K: dequant_q6_k,
    GGMLType.Q8_0: dequant_q8_0,
    GGMLType.Q4_0: dequant_q4_0,
}


def dequantize(
    raw: torch.Tensor,
    dtype: GGMLType,
    n_elements: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand `raw` bytes (uint8, 1-D) into a flat tensor of `n_elements`.

    Runs on whatever device `raw` already lives on.
    """
    if raw.dtype != torch.uint8:
        raise TypeError(f"expected uint8 byte buffer, got {raw.dtype}")

    if dtype == GGMLType.F32:
        return raw.contiguous().view(torch.float32)[:n_elements].to(out_dtype)
    if dtype == GGMLType.F16:
        return raw.contiguous().view(torch.float16)[:n_elements].to(out_dtype)
    if dtype == GGMLType.BF16:
        return raw.contiguous().view(torch.bfloat16)[:n_elements].to(out_dtype)

    kernel = _KERNELS.get(dtype)
    if kernel is None:
        raise NotImplementedError(
            f"no dequant kernel for {dtype.name}; supported: "
            f"F32 F16 BF16 " + " ".join(k.name for k in _KERNELS)
        )

    block_elems, block_bytes = TYPE_LAYOUT[dtype]
    nb = raw.numel() // block_bytes
    blocks = raw[: nb * block_bytes].reshape(nb, block_bytes)
    flat = kernel(blocks).reshape(-1)
    return flat[:n_elements].to(out_dtype)
