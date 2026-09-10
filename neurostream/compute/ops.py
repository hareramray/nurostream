"""Transformer primitives.

Plain torch. These are not where the performance lives — the streaming engine
is — so they stay readable and device-agnostic.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, computed in fp32 for stability."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x * weight.to(torch.float32)).to(orig_dtype)


def rope_tables(
    head_dim: int,
    positions: torch.Tensor,
    base: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX-style RoPE cos/sin tables, shape (T, head_dim).

    Qwen3 uses the half-split convention: the table is inv_freq duplicated,
    not interleaved.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    freqs = positions.to(device, torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def mrope_tables(
    head_dim: int,
    pos_ids: torch.Tensor,
    base: float,
    sections,
    device,
    dtype: torch.dtype = torch.float32,
):
    """Multimodal RoPE: different frequency bands index different axes.

    pos_ids is (3, T) holding (temporal, height, width) positions. Qwen3-VL
    splits head_dim/2 into sections - [24, 20, 20] for head_dim 128 - and
    each section reads one axis.

    For pure text every axis carries the same position, so this reduces
    exactly to 1-D RoPE. That is why the text path was already correct
    without it.
    """
    half = head_dim // 2
    inv = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    # Map each of the `half` frequency dims to an axis via the section sizes.
    axis = torch.zeros(half, dtype=torch.long, device=device)
    cut = 0
    for a, n in enumerate(s for s in sections if s > 0):
        axis[cut : cut + n] = a
        cut += n
    if cut < half:
        axis[cut:] = 0

    pos = pos_ids.to(device, torch.float32)          # (3, T)
    sel = pos[axis]                                   # (half, T)
    freqs = (sel * inv.unsqueeze(1)).transpose(0, 1)  # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: (H, T, D); cos/sin: (T, D)."""
    return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(H_kv, T, D) -> (H_kv * n_rep, T, D) for grouped-query attention."""
    if n_rep == 1:
        return x
    h_kv, t, d = x.shape
    return x.unsqueeze(1).expand(h_kv, n_rep, t, d).reshape(h_kv * n_rep, t, d)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Scaled dot-product attention. q:(H,Tq,D) k,v:(H,Tk,D) -> (H,Tq,D).

    `causal` is only meaningful when Tq == Tk (prefill). During decode Tq == 1
    and every cached position is visible, so no mask is applied.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    if causal and q.shape[1] > 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up
