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
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..io.source import row_geometry

NeuronCallback = Callable[[str, int, int, torch.Tensor], None]

DEFAULT_BLOCK_ROWS = 1024


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
) -> torch.Tensor:
    """x @ W.T, streaming W by row-blocks with one block of lookahead.

    `provider` is a TieredCache or a WeightSource. When the tensor is already
    resident, rows are sliced out of the resident bytes and no I/O happens;
    when it is on disk, blocks are read with one block of lookahead. Same
    loop either way, which is why residency stays invisible to the model.

    x: (T, in_features) -> (T, out_features)
    """
    info = provider.gguf.tensors[name]
    n_rows, row_elems, _ = row_geometry(info)
    if x.shape[-1] != row_elems:
        raise ValueError(
            f"{name}: x has {x.shape[-1]} features, W rows span {row_elems}"
        )

    if out is None:
        out = torch.empty((x.shape[0], n_rows), device=device, dtype=dtype)

    # Async lookahead only pays off for tensors actually on disk.
    src = getattr(provider, "source", provider)
    resident = getattr(provider, "is_resident", lambda _n: False)(name)
    can_async = (
        not resident
        and hasattr(src, "submit_rows")
        and hasattr(src, "reader")
    )
    pending = None
    if can_async:
        pending = src.submit_rows(name, 0, min(block_rows, n_rows))

    from ..compute.quant import dequantize
    from ..compute import triton_kernels as tk

    fuse = tk.can_fuse(info.dtype, x)

    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)

        if can_async:
            fut, nbytes = pending
            reader = (
                src._reader_for(name)
                if hasattr(src, '_reader_for') else src.reader
            )
            raw = reader.wait(fut)
            # Queue the next block before computing this one. Both leases are
            # held simultaneously and the arena is charged for both, so the
            # budget a caller sets is the budget actually observed.
            nxt = end
            pending = (
                src.submit_rows(name, nxt, min(nxt + block_rows, n_rows))
                if nxt < n_rows
                else None
            )
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
                    hit[0], hit[1], x, end - start, row_elems, out_dtype=dtype
                )
                w = None
            else:
                w = provider.fetch_rows(
                    name, start, end, device=device, dtype=dtype
                )
                block_out = x @ w.T
        else:
            w = provider.fetch_rows(name, start, end, device=device, dtype=dtype)
            block_out = x @ w.T

        out[:, start:end] = block_out
        if callback is not None:
            callback(name, start, end, block_out)
        del w

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
