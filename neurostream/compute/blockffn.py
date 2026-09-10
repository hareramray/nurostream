"""Neuron-granular streaming.

`stream_linear` computes x @ W.T without ever holding all of W. It walks W in
row-blocks: read rows [i, i+B), multiply, accumulate, retire. Peak weight
memory is O(B * in_features) regardless of how large W is.

Rows are the right axis for two reasons. They are output neurons, so a block
is a contiguous run of neurons — the "one neuron at a time" semantics, batched
up to a size the SSD is willing to serve. And GGUF stores rows contiguously
with each row a whole number of quant blocks, so a row range is one sequential
read and one independent dequant.

The FFN needs care. ffn_down is stored (n_embd, n_ffn), so slicing it *by
neuron* would mean strided column reads — exactly the access pattern this
drive punishes. Instead the FFN runs in two row-contiguous phases:

    phase 1   walk gate/up by neuron   -> activations a, shape (T, n_ffn)
    phase 2   walk down by output dim  -> out, shape (T, n_embd)

Both phases read contiguous rows. `a` is tiny (T x n_ffn floats), so nothing
is lost by materialising it between the phases.

A mixture-of-experts layer is the same walk over a window. Expert e of a
stacked (n_expert, out, in) tensor owns rows [e * out, (e+1) * out), so
"stream expert e a few neurons at a time" is `stream_linear` bounded to that
window -- see `stream_expert_ffn`. Routing picks the window; when the router
chooses differently on the next token the loop walks a different one and
nothing else about the path changes.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import torch
import torch.nn.functional as F

from ..io.source import row_geometry

NeuronCallback = Callable[[str, int, int, torch.Tensor], None]

DEFAULT_BLOCK_ROWS = 1024
# Blocks kept in flight. One block of lookahead leaves the drive idle for the
# duration of each block's arithmetic, and a neuron block is several times
# smaller than a whole expert slab, so this DRAM-less SSD needs depth to stay
# at its sequential rate rather than its random-read rate.
DEFAULT_LOOKAHEAD = 2


def rows_for_min_read(provider, name: str, min_bytes: int) -> int:
    """Smallest row count whose read clears the drive's efficient-read floor."""
    info = provider.gguf.tensors[name]
    _, _, row_bytes = row_geometry(info)
    return max(1, -(-min_bytes // max(1, row_bytes)))


def stream_linear(
    provider,
    name: str,
    x: torch.Tensor,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    callback: NeuronCallback | None = None,
    out: torch.Tensor | None = None,
    row_start: int = 0,
    row_end: int | None = None,
    lookahead: int = DEFAULT_LOOKAHEAD,
    prefetcher: "BlockPrefetcher | None" = None,
) -> torch.Tensor:
    """x @ W[row_start:row_end].T, streaming those rows a block at a time.

    `provider` is a TieredCache or a WeightSource. When the tensor is already
    resident, rows are sliced out of the resident bytes and no I/O happens;
    when it is on disk, blocks are read with `lookahead` blocks in flight.
    Same loop either way, which is why residency stays invisible to the model.

    `row_start`/`row_end` bound the walk to a window of W. That is what makes
    one MoE expert streamable by neuron: the expert's rows are a contiguous
    window inside the stacked tensor, so each block is still one sequential
    read and one independent dequant.

    `lookahead` is queue depth. Every in-flight block holds its arena lease
    until it is consumed, so `lookahead * block_rows` is clamped to the
    budget instead of being allowed to deadlock against it.

    x: (T, in_features) -> (T, row_end - row_start)
    """
    info = provider.gguf.tensors[name]
    n_rows, row_elems, row_bytes = row_geometry(info)
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    if x.shape[-1] != row_elems:
        raise ValueError(
            f"{name}: x has {x.shape[-1]} features, W rows span {row_elems}"
        )
    row_start = max(0, row_start)
    row_end = n_rows if row_end is None else min(n_rows, row_end)
    window = max(0, row_end - row_start)

    if out is None:
        out = torch.empty((x.shape[0], window), device=device, dtype=dtype)
    if window == 0:
        return out

    # Async lookahead only pays off for tensors actually on disk.
    src = getattr(provider, "source", provider)
    resident = getattr(provider, "is_resident", lambda _n: False)(name)
    if not resident:
        # A pinned expert slab covers this window even though the stacked
        # tensor as a whole never is. Without this the async path would
        # queue disk reads for rows already sitting in VRAM, which is
        # exactly what pinning was measured to avoid.
        has_slab = getattr(provider, "has_slab", None)
        resident = bool(has_slab and has_slab(name, row_start, row_end))
    can_async = (
        not resident
        and hasattr(src, "submit_rows")
        and hasattr(src, "reader")
    )
    depth = max(1, lookahead)
    if can_async:
        # The pipeline holds depth + 1 blocks: `depth` queued, plus the one
        # being multiplied, whose lease is only released after its block_out
        # exists. Size for depth + 1 buffers here rather than deadlocking on
        # the refill that follows the first wait.
        rows_in_budget = src.arena.budget // row_bytes
        if rows_in_budget < 2:
            can_async = False
            block_rows = 1
        else:
            block_rows = min(block_rows, max(1, rows_in_budget // (depth + 1)))
            while depth > 1 and (depth + 1) * block_rows > rows_in_budget:
                depth -= 1

    from ..compute.quant import dequantize
    from ..compute import triton_kernels as tk

    fuse = tk.can_fuse(info.dtype, x)
    shared = prefetcher if (prefetcher is not None and prefetcher.enabled) else None
    if shared is not None:
        # The shared queue owns block sizing; its bounds are what it queued.
        can_async = False
        bounds = shared.bounds(row_start, row_end)
    else:
        bounds = [
            (lo, min(lo + block_rows, row_end))
            for lo in range(row_start, row_end, block_rows)
        ]
    pending: deque = deque()
    queued = 0

    try:
        if can_async:
            for lo, hi in bounds[:depth]:
                pending.append(src.submit_rows(name, lo, hi))
                queued += 1

        for start, end in bounds:
            got = shared.take(name, start) if shared is not None else None
            if got is not None:
                raw, nbytes = got
                try:
                    raw = raw.to(device, non_blocking=True)
                    if fuse:
                        block_out = tk.fused_gemv(
                            raw, info.dtype, x, end - start, row_elems,
                            out_dtype=dtype,
                        )
                        w = None
                    else:
                        w = dequantize(
                            raw, info.dtype, (end - start) * row_elems,
                            out_dtype=dtype,
                        ).reshape(end - start, row_elems)
                        block_out = x @ w.T
                finally:
                    shared.src.arena.release(nbytes)
            elif can_async:
                fut, nbytes = pending.popleft()
                reader = (
                    src._reader_for(name)
                    if hasattr(src, '_reader_for') else src.reader
                )
                raw = reader.wait(fut)
                # Refill before computing, not after: the drive should have
                # `depth` reads outstanding while this block multiplies,
                # which is the whole difference between depth 1 and depth N.
                if queued < len(bounds):
                    nlo, nhi = bounds[queued]
                    pending.append(src.submit_rows(name, nlo, nhi))
                    queued += 1
                try:
                    raw = raw.to(device, non_blocking=True)
                    if fuse:
                        # Never materialise this block's weights.
                        block_out = tk.fused_gemv(
                            raw, info.dtype, x, end - start, row_elems,
                            out_dtype=dtype,
                        )
                        w = None
                    else:
                        w = dequantize(
                            raw, info.dtype, (end - start) * row_elems,
                            out_dtype=dtype,
                        ).reshape(end - start, row_elems)
                        block_out = x @ w.T
                finally:
                    src.arena.release(nbytes)
            elif fuse and hasattr(provider, "raw_rows"):
                # Resident tensor: slice this block's quantized bytes and fuse.
                # Keeps neuron granularity without paying a full dequant.
                hit = provider.raw_rows(name, start, end)
                if hit is not None:
                    block_out = tk.fused_gemv(
                        hit[0], hit[1], x, end - start, row_elems,
                        out_dtype=dtype,
                    )
                    w = None
                else:
                    w = provider.fetch_rows(
                        name, start, end, device=device, dtype=dtype
                    )
                    block_out = x @ w.T
            else:
                w = provider.fetch_rows(
                    name, start, end, device=device, dtype=dtype
                )
                block_out = x @ w.T

            out[:, start - row_start:end - row_start] = block_out
            if callback is not None:
                callback(name, start, end, block_out)
            del w
    finally:
        # A block queued but never consumed must not strand its lease.
        for fut, nbytes in pending:
            fut.cancel()
            src.arena.release(nbytes)

    return out


class BlockStreamingFFN:
    """Dense SwiGLU FFN evaluated one neuron-block at a time."""

    def __init__(
        self,
        provider,
        block_rows: int = DEFAULT_BLOCK_ROWS,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        callback: NeuronCallback | None = None,
    ) -> None:
        self.provider = provider
        self.block_rows = block_rows
        self.device = device
        self.dtype = dtype
        self.callback = callback

    def __call__(self, h: torch.Tensor, prefix: str) -> torch.Tensor:
        gate = stream_linear(
            self.provider, prefix + "ffn_gate.weight", h, self.block_rows,
            self.device, self.dtype, self.callback,
        )
        up = stream_linear(
            self.provider, prefix + "ffn_up.weight", h, self.block_rows,
            self.device, self.dtype, self.callback,
        )
        a = F.silu(gate) * up
        del gate, up
        return stream_linear(
            self.provider, prefix + "ffn_down.weight", a, self.block_rows,
            self.device, self.dtype, self.callback,
        )


class BlockPrefetcher:
    """Keeps `depth` block reads in flight across a whole plan of windows.

    `stream_linear`'s own pipeline lives and dies inside one call, so it
    drains at every projection boundary -- 36 layers x 4 experts x 3
    matrices = 432 times per token. Each drain costs an unhidden disk
    latency, and with a 2880-row window at 1024 rows there are only three
    blocks to hide it behind. Measured: queue depth 3, 66% prefetch hit,
    76 s of stall, and 0.58 GB/s from a drive that does 2.0.

    Shrinking blocks reaches depth but multiplies per-block cost (456k reads
    at 128 rows dropped throughput to 0.29 GB/s). This decouples the two: the
    plan spans every projection and expert of a layer, so a block retiring in
    `ffn_gate` immediately pulls in one belonging to `ffn_down` or to the
    next expert, and the queue only empties once per layer.

    Peak bytes are unchanged -- `depth + 1` blocks are charged to the arena,
    exactly as the per-call pipeline held. Only the ordering changes.
    """

    def __init__(self, provider, plan, block_rows, depth=DEFAULT_LOOKAHEAD):
        self.provider = provider
        self.src = getattr(provider, "source", provider)
        self.block_rows = max(1, block_rows)
        self.depth = max(1, depth)
        self._queue: deque = deque()
        self._inflight: dict = {}
        self._plan: list = []

        self.enabled = hasattr(self.src, "submit_rows") and hasattr(
            self.src, "reader"
        )
        if not self.enabled or not plan:
            return

        widest = max(
            row_geometry(provider.gguf.tensors[n])[2] for n, _, _ in plan
        )
        rows_in_budget = self.src.arena.budget // widest
        if rows_in_budget < 2:
            self.enabled = False
            return
        # depth + 1: the block being multiplied still owns its lease.
        self.block_rows = min(
            self.block_rows, max(1, rows_in_budget // (self.depth + 1))
        )
        while self.depth > 1 and (self.depth + 1) * self.block_rows > rows_in_budget:
            self.depth -= 1

        for name, lo, hi in plan:
            # Rows the cache already holds must never be queued for disk.
            if self._cached(name, lo, hi):
                continue
            for start in range(lo, hi, self.block_rows):
                self._plan.append((name, start, min(start + self.block_rows, hi)))
        self._fill()

    def _cached(self, name, lo, hi):
        if getattr(self.provider, "is_resident", lambda _n: False)(name):
            return True
        has_slab = getattr(self.provider, "has_slab", None)
        return bool(has_slab and has_slab(name, lo, hi))

    def _fill(self):
        while len(self._inflight) < self.depth and self._plan:
            name, start, end = self._plan.pop(0)
            try:
                self._inflight[(name, start)] = self.src.submit_rows(
                    name, start, end
                )
            except Exception:
                # Budget momentarily full; the block resolves synchronously.
                self._plan.insert(0, (name, start, end))
                return
            self._queue.append((name, start))

    def bounds(self, lo, hi):
        """Block boundaries for one window, matching what was queued."""
        return [
            (start, min(start + self.block_rows, hi))
            for start in range(lo, hi, self.block_rows)
        ]

    def take(self, name, start):
        """(raw, nbytes) for a queued block, or None if it was not queued."""
        entry = self._inflight.pop((name, start), None)
        if entry is None:
            return None
        try:
            self._queue.remove((name, start))
        except ValueError:
            pass
        fut, nbytes = entry
        reader = (
            self.src._reader_for(name)
            if hasattr(self.src, "_reader_for") else self.src.reader
        )
        raw = reader.wait(fut)
        # Refill before the caller computes, so the drive stays busy.
        self._fill()
        return raw, nbytes

    def close(self):
        """Release anything queued but never consumed."""
        for _key, (fut, nbytes) in self._inflight.items():
            fut.cancel()
            self.src.arena.release(nbytes)
        self._inflight.clear()
        self._queue.clear()
        self._plan.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stream_expert_ffn(
    provider,
    prefix: str,
    expert: int,
    x: torch.Tensor,
    rows_per_expert: int,
    n_embd: int,
    activation,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    bias: bool = False,
    lookahead: int = DEFAULT_LOOKAHEAD,
    callback: NeuronCallback | None = None,
    prefetcher: "BlockPrefetcher | None" = None,
) -> torch.Tensor:
    """One routed expert's FFN, evaluated a few neurons at a time.

    The expert is never held whole. Phase 1 walks its gate/up rows -- its
    neurons -- in blocks: read a block, multiply, accumulate, retire it, take
    the next block. Phase 2 walks ffn_down by output dimension, which is the
    same row-contiguous access for the one matrix stored transposed. Peak
    weight memory is O(lookahead * block_rows * n_embd) whatever the expert
    is, instead of the whole expert at once.

    Routing decides *which* neurons. Expert e owns a window of the stacked
    tensor, so when the router picks a different expert for the next token
    the same loop walks a different window; nothing carries over except what
    the residency planner pinned, which `TieredCache._slab_for` still serves
    even though these reads are sub-ranges of a pinned slab.

    `activation` receives the whole (T, rows_per_expert) gate and up
    activations. Those are floats per token, not weights -- kilobytes, not
    gigabytes -- so materialising them between the phases costs nothing.
    """
    def project(kind, value, lo, span):
        name = prefix + f"ffn_{kind}_exps"
        y = stream_linear(
            provider, name + ".weight", value, block_rows, device, dtype,
            callback, row_start=lo, row_end=lo + span, lookahead=lookahead,
            prefetcher=prefetcher,
        )
        if bias:
            # One row per expert; cheap enough to fetch whole.
            y = y + provider.fetch_rows(
                name + ".bias", expert, expert + 1, device=device, dtype=dtype
            )
        return y

    gate = project("gate", x, expert * rows_per_expert, rows_per_expert)
    up = project("up", x, expert * rows_per_expert, rows_per_expert)
    a = activation(gate, up)
    del gate, up
    return project("down", a, expert * n_embd, n_embd)


def stream_logits(
    provider,
    name: str,
    x: torch.Tensor,
    block_rows: int = 8192,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vocabulary projection in row-blocks.

    The LM head is often the single largest tensor in the file (297 MB even in
    a 0.6B model), so streaming it is what actually lowers the budget floor.
    """
    return stream_linear(
        provider, name, x, block_rows, device=device, dtype=dtype
    )


def gather_embeddings(
    provider,
    name: str,
    tokens: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Read only the embedding rows this sequence actually uses.

    A 151936-row table is 297 MB; a 13-token prompt needs 26 KB of it.
    """
    ids = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
    uniq = sorted(set(ids))
    rows = {
        t: provider.fetch_rows(name, t, t + 1, device=device, dtype=dtype)[0]
        for t in uniq
    }
    return torch.stack([rows[t] for t in ids])
