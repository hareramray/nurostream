"""Weight source for HuggingFace safetensors checkpoints.

Reads rows of a `SafetensorsFile` recipe: gather the source rows, permute
within the row if the recipe says to, apply the elementwise fixup, done.

The gather is what needs care. A recipe's row permutation is head-wise, so a
block of output rows maps to a handful of long contiguous runs rather than to
scattered single rows - the reader walks those runs and reads each one, so a
block still costs block-sized memory rather than whole-tensor memory. That is
the property the arena depends on, and it survives the translation.
"""
from __future__ import annotations

import mmap
import time
from pathlib import Path

import numpy as np
import torch

from ..format.safetensors import DTYPES, SafetensorsFile
from .reader import IOStats
from .source import row_geometry

# Reading one row at a time would be correct but slow, so short gaps between
# runs are read through rather than split. Bounded so a stray permutation
# cannot quietly turn a block read into a whole-tensor read.
GAP_TOLERANCE = 64


class SafetensorsSource:
    """Mmap-backed source over a checkpoint's shards.

    Like `MmapSource` this leans on the page cache rather than an async
    reader: the translation layer has to touch every byte it returns anyway,
    so there is nothing for a prefetch thread to overlap with.
    """

    def __init__(self, path: str | Path, part: str = "text", arena=None) -> None:
        self.gguf = SafetensorsFile(path, part)
        # Only consulted for chunk sizing when a resident tensor is larger
        # than the I/O budget; reads here are page-cache backed, not queued.
        self.arena = arena
        self._files: list = []
        self._maps: list = []
        for shard in self.gguf.paths:
            fh = open(shard, "rb")
            self._files.append(fh)
            self._maps.append(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
        # Reported by `--verbose`; prefetch stays 0% because there is none.
        self.stats = IOStats()

    # -- reading ----------------------------------------------------------

    def _source_rows(self, name: str, rows: torch.Tensor | None,
                     start: int, end: int) -> torch.Tensor:
        """Source rows [start, end) of an output tensor, as stored on disk."""
        recipe = self.gguf.recipes[name]
        entry = self.gguf.entries[recipe.source]
        torch_dtype = DTYPES[entry.dtype][1]
        base = self.gguf.data_offsets[entry.shard] + entry.begin
        mm = self._maps[entry.shard]

        # Source geometry: the recipe never changes a row's element count.
        elements = 1
        for d in entry.shape:
            elements *= d
        src_row_elems = entry.shape[-1] if len(entry.shape) > 1 else elements
        itemsize = torch.empty(0, dtype=torch_dtype).element_size()
        src_row_bytes = src_row_elems * itemsize

        def read(first: int, count: int) -> torch.Tensor:
            offset = base + first * src_row_bytes
            nbytes = count * src_row_bytes
            if offset + nbytes > self.gguf.data_offsets[entry.shard] + entry.end:
                raise ValueError(f"{name}: read runs past the end of {recipe.source}")
            started = time.perf_counter()
            raw = np.frombuffer(mm, dtype=np.uint8, count=nbytes, offset=offset)
            block = (torch.from_numpy(raw.copy())
                     .view(torch_dtype).reshape(count, src_row_elems))
            now = time.perf_counter()
            self.stats.requests += 1
            self.stats.bytes_read += nbytes
            self.stats.read_seconds += now - started
            self.stats.first_read = self.stats.first_read or started
            self.stats.last_read = now
            return block

        if rows is None:
            return read(start, end - start)

        wanted = rows[start:end]
        # Walk contiguous (or near-contiguous) runs so a permuted block still
        # reads in a handful of sequential chunks.
        chunks: list[torch.Tensor] = []
        pieces: list[torch.Tensor] = []
        run_start = int(wanted[0])
        previous = run_start
        for value in wanted[1:].tolist():
            if 0 <= value - previous <= GAP_TOLERANCE:
                previous = value
                continue
            chunks.append(read(run_start, previous - run_start + 1))
            pieces.append(torch.arange(run_start, previous + 1))
            run_start = previous = value
        chunks.append(read(run_start, previous - run_start + 1))
        pieces.append(torch.arange(run_start, previous + 1))

        block = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        available = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        if torch.equal(available, wanted):
            return block
        # A run may have been read through a gap, and runs need not ascend,
        # so index back into what was actually read.
        lookup = torch.full((int(available.max()) + 1,), -1, dtype=torch.long)
        lookup[available] = torch.arange(available.numel())
        return block[lookup[wanted]]

    def _rows(self, name: str, start: int, end: int,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Output rows [start, end) as floats, recipe applied."""
        recipe = self.gguf.recipes[name]
        info = self.gguf.tensors[name]
        n_rows, row_elems, _ = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        if start >= end:
            return torch.empty((0, row_elems), dtype=dtype)

        block = self._source_rows(name, recipe.rows, start, end)
        if recipe.cols is not None:
            block = block.index_select(1, recipe.cols)
        if recipe.op == "copy":
            return block.to(dtype).reshape(end - start, row_elems)
        # The fixups are defined on real numbers, so they run in fp32 and are
        # rounded once, on the way out.
        out = block.float()
        if recipe.op == "shift":
            out = out + 1.0
        elif recipe.op == "neg_exp":
            out = -torch.exp(out)
        else:
            raise NotImplementedError(f"unknown recipe op {recipe.op!r}")
        return out.to(dtype).reshape(end - start, row_elems)

    # -- WeightSource ------------------------------------------------------

    def fetch(self, name: str, device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
        info = self.gguf.tensors[name]
        n_rows, _, _ = row_geometry(info)
        out = self._rows(name, 0, n_rows, dtype)
        return out.reshape(info.torch_shape).to(device)

    def fetch_rows(self, name: str, start: int, end: int,
                   device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self._rows(name, start, end, dtype).to(device)

    def raw_bytes(self, name: str) -> torch.Tensor:
        """The tensor as the engine's own dtype would store it, on the host."""
        info = self.gguf.tensors[name]
        n_rows, _, _ = row_geometry(info)
        return self.raw_rows(name, 0, n_rows)

    def raw_rows(self, name: str, start: int, end: int) -> torch.Tensor:
        info = self.gguf.tensors[name]
        _, dtype = _ggml_to_torch(info.dtype)
        return self._rows(name, start, end, dtype).contiguous().view(torch.uint8).reshape(-1)

    @property
    def bytes_read(self) -> int:
        return self.stats.bytes_read

    def close(self) -> None:
        for mm in self._maps:
            mm.close()
        for fh in self._files:
            fh.close()
        self._maps.clear()
        self._files.clear()

    def __enter__(self) -> "SafetensorsSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ggml_to_torch(dtype):
    for name, (ggml, torch_dtype) in DTYPES.items():
        if ggml == dtype:
            return name, torch_dtype
    raise NotImplementedError(f"no torch dtype for {dtype!r}")
