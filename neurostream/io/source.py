"""Weight sources.

A WeightSource answers one question: "give me tensor T, or rows [a,b) of it,
as a float tensor on device D". Everything above this line — the model code —
never learns whether those bytes came from VRAM, page cache, or a 146 GB file
on the SSD. Everything below it is the streaming engine.

P0 ships MmapSource. P1 adds StreamingSource behind the same interface.
"""
from __future__ import annotations

import mmap
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from ..compute.quant import dequantize
from ..format.gguf import GGUFFile, TensorInfo, nbytes_for


class WeightSource(Protocol):
    gguf: GGUFFile

    def fetch(
        self, name: str, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: ...

    def fetch_rows(
        self, name: str, start: int, end: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: ...


def row_geometry(info: TensorInfo) -> tuple[int, int, int]:
    """(n_rows, row_elements, row_bytes), flattening all leading dimensions.

    Rows are the unit of neuron-granular streaming. GGUF guarantees each row
    spans a whole number of quantization blocks, so a row range is always
    independently decodable.

    Expert tensors are 3-D — (n_expert, out, in) — and flattening the leading
    dims is what makes expert-selective fetch a plain row range: expert e owns
    rows [e * out, (e+1) * out). No special case, no strided reads.
    """
    if info.geom is not None:
        return info.geom

    shape = info.torch_shape
    if len(shape) == 1:
        out = (1, shape[0], info.nbytes)
    else:
        row_elems = shape[-1]
        n_rows = 1
        for d in shape[:-1]:
            n_rows *= d
        out = (n_rows, row_elems, nbytes_for(info.dtype, row_elems))
    info.geom = out
    return out


class MmapSource:
    """Baseline source: mmap the file, let the OS page cache do the work.

    This is already a streaming implementation — pages fault in on demand and
    the kernel evicts under pressure — but it has no prefetch, no queue depth,
    and no memory ceiling. It is the correctness reference that P1 must match.
    """

    def __init__(self, path: str | Path) -> None:
        self.gguf = GGUFFile(path)
        self._fh = open(self.gguf.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.bytes_read = 0

    def _raw(self, offset: int, nbytes: int) -> torch.Tensor:
        arr = np.frombuffer(self._mm, dtype=np.uint8, count=nbytes, offset=offset)
        self.bytes_read += nbytes
        return torch.from_numpy(arr.copy())

    def raw_bytes(self, name: str) -> torch.Tensor:
        """Still-quantized bytes for a whole tensor, on the host."""
        info = self.gguf.tensors[name]
        return self._raw(self.gguf.file_offset(info), info.nbytes)

    def fetch(
        self, name: str, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        info = self.gguf.tensors[name]
        raw = self._raw(self.gguf.file_offset(info), info.nbytes).to(device)
        flat = dequantize(raw, info.dtype, info.n_elements, out_dtype=dtype)
        return flat.reshape(info.torch_shape)

    def raw_rows(self, name: str, start: int, end: int) -> torch.Tensor:
        """Quantized bytes for a row range."""
        info = self.gguf.tensors[name]
        n_rows, _, row_bytes = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        return self._raw(
            self.gguf.file_offset(info) + start * row_bytes,
            (end - start) * row_bytes,
        )

    def fetch_rows(
        self, name: str, start: int, end: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        info = self.gguf.tensors[name]
        n_rows, row_elems, row_bytes = row_geometry(info)
        start = max(0, start)
        end = min(n_rows, end)
        if start >= end:
            return torch.empty((0, row_elems), dtype=dtype, device=device)

        offset = self.gguf.file_offset(info) + start * row_bytes
        raw = self._raw(offset, (end - start) * row_bytes).to(device)
        flat = dequantize(raw, info.dtype, (end - start) * row_elems, out_dtype=dtype)
        return flat.reshape(end - start, row_elems)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "MmapSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
