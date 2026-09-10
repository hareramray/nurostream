"""Mixture-of-Experts: route first, then read only what was chosen.

This is the module that makes the 235B target reachable. A dense 235B model
would move 146 GB across the bus per token and take a minute per token on this
drive. Qwen3-VL-235B-A22B activates 8 of 128 experts per layer, so the same
file yields ~13 GB of reads per token instead — an order of magnitude, bought
by asking the router before touching the disk.

Expert e of a stacked (n_expert, out, in) tensor owns rows
[e * out, (e+1) * out). That is a contiguous range, so a selective fetch is
one sequential read per expert and needs no special support anywhere else in
the I/O path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..compute import triton_kernels as tk
from ..io.arena import BudgetExceeded


@dataclass
class RouterStats:
    tokens_routed: int = 0
    expert_reads: int = 0
    hits: Counter = field(default_factory=Counter)

    def top_experts(self, layer: int, k: int = 8) -> list[tuple[int, int]]:
        return [
            (e, c)
            for (l, e), c in self.hits.most_common()
            if l == layer
        ][:k]

    def skew(self) -> float:
        """Fraction of all routings taken by the busiest 20% of experts.

        Uniform routing gives 0.2; anything much above that is cacheable
        structure the residency planner can exploit.
        """
        if not self.hits:
            return 0.0
        counts = sorted(self.hits.values(), reverse=True)
        cut = max(1, len(counts) // 5)
        return sum(counts[:cut]) / sum(counts)

    def summary(self) -> str:
        return (
            f"routed {self.tokens_routed} tokens | "
            f"{self.expert_reads} expert reads | "
            f"top-20% skew {self.skew():.1%}"
        )


def route(
    model, h: torch.Tensor, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (expert_ids, weights), each (T, n_expert_used)."""
    cfg = model.cfg
    logits = h @ model.w(f"blk.{layer}.ffn_gate_inp.weight").T
    probs = F.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, cfg.n_expert_used, dim=-1)
    # Qwen3 MoE renormalises the top-k probabilities.
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights.to(h.dtype)


def _expert_slabs(cfg):
    """(tensor suffix, rows per expert) for the three expert matrices."""
    return (
        ("ffn_gate_exps.weight", cfg.n_ffn_expert),
        ("ffn_up_exps.weight", cfg.n_ffn_expert),
        ("ffn_down_exps.weight", cfg.n_embd),
    )


def submit_expert_reads(model, layer: int, experts) -> dict:
    """Queue every non-resident slab for the chosen experts at once.

    This is the difference between queue depth 1 and queue depth ~24. Reading
    each expert synchronously when it is needed leaves the NVMe idle between
    requests and yields about 1 GB/s; issuing all of a layer's expert reads
    together lets the drive reach its ~2.3 GB/s sequential rate. Since the
    router has already named every expert this layer will use, there is no
    reason to discover them one at a time.
    """
    src = getattr(model.cache, "source", None)
    if src is None or not hasattr(src, "submit_rows"):
        return {}
    cache = model.cache
    pending: dict = {}
    for e in experts:
        for nm, rpe in _expert_slabs(model.cfg):
            name = f"blk.{layer}.{nm}"
            if name not in model.gguf.tensors:
                continue
            lo, hi = e * rpe, (e + 1) * rpe
            if cache.is_resident(name) or cache.has_slab(name, lo, hi):
                continue
            try:
                # submit_rows already returns (future, nbytes)
                pending[(name, lo)] = src.submit_rows(name, lo, hi, timeout=0)
            except (BudgetExceeded, TimeoutError):
                return pending  # budget full; the rest resolve synchronously
    return pending


def _expert_matmul(model, name: str, expert: int, rows_per_expert: int, x,
                   pending: dict | None = None):
    """x @ Wexpert.T for one expert, fused when possible.

    Goes through the cache, not the source, so a hot expert promoted into
    VRAM costs no I/O at all. Reading experts straight off the source would
    make the residency planner pointless for exactly the tensors it matters
    most for.
    """
    lo = expert * rows_per_expert
    hi = lo + rows_per_expert

    # Resolve an in-flight read from submit_expert_reads if there is one.
    entry = pending.pop((name, lo), None) if pending else None
    if entry is not None:
        fut, nbytes = entry
        src = model.cache.source
        reader = (
            src._reader_for(name)
            if hasattr(src, "_reader_for") else src.reader
        )
        raw = reader.wait(fut)
        src.arena.release(nbytes)
        raw = raw.to(model.device, non_blocking=True)
        dt = model.gguf.tensors[name].dtype
        if tk.can_fuse(dt, x):
            row_elems = model.gguf.tensors[name].torch_shape[-1]
            return tk.fused_gemv(
                raw, dt, x, rows_per_expert, row_elems, out_dtype=model.dtype
            )
        from ..compute.quant import dequantize

        row_elems = model.gguf.tensors[name].torch_shape[-1]
        w = dequantize(
            raw, dt, (hi - lo) * row_elems, out_dtype=model.dtype
        ).reshape(hi - lo, row_elems)
        return x @ w.T

    hit = model.cache.raw_rows(name, lo, hi)
    if hit is not None:
        row_elems = model.gguf.tensors[name].torch_shape[-1]
        if tk.can_fuse(hit[1], x):
            return tk.fused_gemv(
                hit[0], hit[1], x, rows_per_expert, row_elems,
                out_dtype=model.dtype,
            )
        # Batched prefill cannot use GEMV, but can still reuse the cached
        # bytes. fetch_rows here used to discard this hit and reread disk.
        from ..compute.quant import dequantize

        w = dequantize(
            hit[0], hit[1], rows_per_expert * row_elems,
            out_dtype=model.dtype,
        ).reshape(rows_per_expert, row_elems)
        return x @ w.T
    w = model.cache.fetch_rows(
        name, lo, hi, device=model.device, dtype=model.dtype
    )
    return x @ w.T


def moe_ffn(model, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Sparse SwiGLU FFN. Only the selected experts are ever read."""
    cfg = model.cfg
    p = f"blk.{layer}."
    stats: RouterStats = getattr(model, "router_stats", None) or RouterStats()
    model.router_stats = stats

    ids, weights = route(model, h, layer)
    stats.tokens_routed += h.shape[0]

    chosen = torch.unique(ids).tolist()
    pending = submit_expert_reads(model, layer, chosen)

    out = torch.zeros_like(h)
    n_ffn_e = cfg.n_ffn_expert
    n_embd = cfg.n_embd

    # Group tokens by expert so each expert is read at most once per forward.
    for e in chosen:
        mask = ids == e
        rows = mask.any(dim=-1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        w = (weights * mask).sum(dim=-1)[rows].unsqueeze(-1)

        stats.hits[(layer, e)] += int(rows.numel())
        stats.expert_reads += 1

        x = h[rows]
        gate = _expert_matmul(
            model, p + "ffn_gate_exps.weight", e, n_ffn_e, x, pending
        )
        up = _expert_matmul(
            model, p + "ffn_up_exps.weight", e, n_ffn_e, x, pending
        )
        act = F.silu(gate) * up
        out[rows] += w * _expert_matmul(
            model, p + "ffn_down_exps.weight", e, n_embd, act, pending
        )

    # Qwen3 MoE has no shared expert; other Qwen MoE generations do.
    if model.has(p + "ffn_gate_shexp.weight"):
        sg = model._linear(p + "ffn_gate_shexp.weight", h)
        su = model._linear(p + "ffn_up_shexp.weight", h)
        out = out + model._linear(p + "ffn_down_shexp.weight", F.silu(sg) * su)

    return out


def speculative_prefetch(model, layer: int, top_k: int = 4) -> int:
    """Queue the experts this layer has historically used most.

    Cheap insurance: if the router agrees, the read is already in flight; if
    it does not, the bytes are dropped and the only cost was idle queue depth
    the compute thread was not using anyway.
    """
    src = getattr(model.cache, "source", None)
    stats = getattr(model, "router_stats", None)
    if src is None or stats is None or not hasattr(src, "submit_rows"):
        return 0
    cfg = model.cfg
    queued = 0
    for e, _count in stats.top_experts(layer, top_k):
        for nm, rpe in (
            ("ffn_gate_exps.weight", cfg.n_ffn_expert),
            ("ffn_up_exps.weight", cfg.n_ffn_expert),
        ):
            name = f"blk.{layer}.{nm}"
            if name not in src.gguf.tensors:
                continue
            try:
                src.submit_rows(name, e * rpe, (e + 1) * rpe)
                queued += 1
            except Exception:
                return queued
    return queued
