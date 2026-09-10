"""Streaming weight source with layer-ahead prefetch.

The compute cursor sits on layer N while the prefetch cursor issues reads for
layer N+1 (and optionally N+2). With 2.35 GB/s of disk against ~18 GB/s of
compute, a layer's arithmetic covers a useful fraction of the next layer's
read, and what the overlap does not hide shows up honestly as `stall_seconds`.
"""
from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import torch

from ..compute.quant import dequantize
from ..format.gguf import GGUFFile
from ..format.sharded import ShardedGGUF, discover_shards
from .arena import Arena
from .reader import AsyncReader
from .source import row_geometry

# Tensor suffixes that make up one transformer block, dense and MoE.
DENSE_SUFFIXES = (
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)
MOE_SUFFIXES = (
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_norm.weight", "ffn_gate_inp.weight",
)


class StreamingSource:
    """WeightSource backed by AsyncReader, bounded by an Arena."""

    def __init__(
        self,
        path: str | Path,
        arena: Arena,
        n_workers: int = 16,
        pin_memory: bool = False,
    ) -> None:
        shards = discover_shards(path)
        self.sharded = len(shards) > 1
        self.gguf = ShardedGGUF(path) if self.sharded else GGUFFile(path)
        self.arena = arena
        # One reader per shard: each keeps its own pool of file handles, so a
        # 3-shard model still gets full queue depth against every file.
        self.readers = [
            AsyncReader(sp, n_workers=n_workers) for sp in shards
        ]
        self.reader = self.readers[0]
        self.pin_memory = pin_memory
        self._inflight: dict[str, tuple[Future, int]] = {}

    def _reader_for(self, name: str) -> AsyncReader:
        if not self.sharded:
            return self.readers[0]
        return self.readers[self.gguf.shard_index(name)]

    @property
    def stats(self):
        """Aggregate I/O counters across every shard reader."""
        if len(self.readers) == 1:
            return self.readers[0].stats
        from .reader import IOStats

        total = IOStats()
        for r in self.readers:
            st = r.stats
            total.requests += st.requests
            total.bytes_read += st.bytes_read
            total.read_seconds += st.read_seconds
            total.stall_seconds += st.stall_seconds
            total.prefetch_hits += st.prefetch_hits
            total.prefetch_misses += st.prefetch_misses
            if st.first_read:
                total.first_read = (
                    st.first_read if total.first_read == 0.0
                    else min(total.first_read, st.first_read)
                )
                total.last_read = max(total.last_read, st.last_read)
        return total

    # -- prefetch ---------------------------------------------------------

    def prefetch(self, names) -> int:
        """Queue reads for `names`. Returns bytes queued.

        Silently skips anything already in flight or absent from the file, so
        callers can pass a superset of suffixes without branching.
        """
        queued = 0
        for name in names:
            if name in self._inflight:
                continue
            info = self.gguf.tensors.get(name)
            if info is None:
                continue
            # Never let speculative reads monopolise the budget; the
            # compute path must always be able to make progress.
            if queued + info.nbytes > self.arena.budget // 2:
                break
            try:
                self.arena.acquire(info.nbytes, timeout=5.0)
            except Exception:
                break  # budget full; the rest will be read on demand
            fut = self._reader_for(name).submit(
                self.gguf.file_offset(info), info.nbytes
            )
            self._inflight[name] = (fut, info.nbytes)
            queued += info.nbytes
        return queued

    def prefetch_layer(self, layer: int, moe: bool = False) -> int:
        suffixes = MOE_SUFFIXES if moe else DENSE_SUFFIXES
        return self.prefetch(f"blk.{layer}.{s}" for s in suffixes)

    def drop_inflight(self) -> None:
        """Cancel and release everything still queued (e.g. on a router miss)."""
        for name, (fut, nbytes) in list(self._inflight.items()):
            fut.cancel()
            self.arena.release(nbytes)
        self._inflight.clear()

    # -- access -----------------------------------------------------------

    def raw_bytes(self, name: str) -> torch.Tensor:
        """Quantized bytes, resolving a prefetch if one is pending."""
        pending = self._inflight.pop(name, None)
        if pending is not None:
            fut, nbytes = pending
            raw = self._reader_for(name).wait(fut)
            self.arena.release(nbytes)
            return raw
        info = self.gguf.tensors[name]
        with_lease = info.nbytes
        self.arena.acquire(with_lease)
        try:
            return self._reader_for(name).read(
                self.gguf.file_offset(info), info.nbytes
            )
        finally:
            self.arena.release(with_lease)

    def fetch(
        self, name: str, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        info = self.gguf.tensors[name]
        raw = self.raw_bytes(name)
        if self.pin_memory and str(device) != "cpu":
            raw = raw.pin_memory()
        raw = raw.to(device, non_blocking=True)
        flat = dequantize(raw, info.dtype, info.n_elements, out_dtype=dtype)
        return flat.reshape(info.torch_shape)

    def fetch_rows(
        self, name: str, start: int, end: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Row range = neuron range. The unit of P2 block streaming."""
        info = self.gguf.tensors[name]
        n_rows, row_elems, row_bytes = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        if start >= end:
            return torch.empty((0, row_elems), dtype=dtype, device=device)

        nbytes = (end - start) * row_bytes
        self.arena.acquire(nbytes)
        try:
            raw = self._reader_for(name).read(
                self.gguf.file_offset(info) + start * row_bytes, nbytes
            )
            raw = raw.to(device, non_blocking=True)
            flat = dequantize(
                raw, info.dtype, (end - start) * row_elems, out_dtype=dtype
            )
            return flat.reshape(end - start, row_elems)
        finally:
            self.arena.release(nbytes)

    def raw_rows(self, name: str, start: int, end: int) -> torch.Tensor:
        """Quantized bytes for a row range, read under the arena budget."""
        info = self.gguf.tensors[name]
        n_rows, _, row_bytes = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        nbytes = (end - start) * row_bytes
        self.arena.acquire(nbytes)
        try:
            return self._reader_for(name).read(
                self.gguf.file_offset(info) + start * row_bytes, nbytes
            )
        finally:
            self.arena.release(nbytes)

    def submit_rows(
        self, name: str, start: int, end: int, *, timeout: float | None = None,
    ) -> tuple[Future, int]:
        """Async variant of fetch_rows; caller must release the arena bytes."""
        info = self.gguf.tensors[name]
        n_rows, _, row_bytes = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        nbytes = (end - start) * row_bytes
        self.arena.acquire(nbytes, timeout=timeout)
        try:
            fut = self._reader_for(name).submit(
                self.gguf.file_offset(info) + start * row_bytes, nbytes
            )
        except Exception:
            self.arena.release(nbytes)
            raise
        return fut, nbytes

    def close(self) -> None:
        self.drop_inflight()
        for r in self.readers:
            r.close()

    def __enter__(self) -> "StreamingSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
