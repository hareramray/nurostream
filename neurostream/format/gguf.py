"""GGUF container parsing.

Reads the header, metadata and tensor index WITHOUT touching tensor data.
This is the property the whole library depends on: opening a 146 GB model must
cost the same as opening a 400 MB one.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30
    MXFP4 = 39


# (elements per block, bytes per block)
TYPE_LAYOUT: dict[GGMLType, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.MXFP4: (32, 17),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
}


def nbytes_for(dtype: GGMLType, n_elements: int) -> int:
    """Byte size of `n_elements` stored as `dtype`."""
    if dtype not in TYPE_LAYOUT:
        raise NotImplementedError(f"unsupported ggml type: {dtype!r}")
    block_elems, block_bytes = TYPE_LAYOUT[dtype]
    if n_elements % block_elems:
        raise ValueError(
            f"{n_elements} elements is not a multiple of {dtype.name} "
            f"block size {block_elems}"
        )
    return (n_elements // block_elems) * block_bytes


class _ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FMT = {
    _ValueType.UINT8: "<B",
    _ValueType.INT8: "<b",
    _ValueType.UINT16: "<H",
    _ValueType.INT16: "<h",
    _ValueType.UINT32: "<I",
    _ValueType.INT32: "<i",
    _ValueType.FLOAT32: "<f",
    _ValueType.BOOL: "<?",
    _ValueType.UINT64: "<Q",
    _ValueType.INT64: "<q",
    _ValueType.FLOAT64: "<d",
}


@dataclass(slots=True)
class TensorInfo:
    """Where a tensor lives on disk. Note: no data, just coordinates."""

    name: str
    shape: tuple[int, ...]  # ggml order: fastest-varying dimension first
    dtype: GGMLType
    offset: int  # relative to the start of the data section
    nbytes: int
    # Row geometry memo, filled in by io.source.row_geometry. Cached on the
    # instance rather than in a dict keyed by id(): CPython reuses addresses
    # after GC, so an id-keyed memo hands one model's strides to the next
    # model loaded in the same process, and every weight is then read from
    # the wrong offset.
    geom: tuple[int, int, int] | None = None

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def torch_shape(self) -> tuple[int, ...]:
        """Row-major shape, i.e. ggml dims reversed."""
        return tuple(reversed(self.shape))


class _Cursor:
    """Minimal buffered reader for the metadata region."""

    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"truncated GGUF: wanted {n} bytes, got {len(b)}")
        return b

    def scalar(self, vt: _ValueType) -> Any:
        fmt = _SCALAR_FMT[vt]
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        (n,) = struct.unpack("<Q", self.read(8))
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vt: _ValueType) -> Any:
        if vt == _ValueType.STRING:
            return self.string()
        if vt == _ValueType.ARRAY:
            (elem_t,) = struct.unpack("<I", self.read(4))
            (count,) = struct.unpack("<Q", self.read(8))
            elem_t = _ValueType(elem_t)
            return [self.value(elem_t) for _ in range(count)]
        return self.scalar(vt)


class GGUFFile:
    """Parsed GGUF index. Holds no tensor data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.alignment = DEFAULT_ALIGNMENT
        self.data_offset = 0
        self._parse()

    def _parse(self) -> None:
        with open(self.path, "rb") as fh:
            cur = _Cursor(fh)
            magic = cur.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"not a GGUF file (magic={magic!r}): {self.path}")
            version, n_tensors, n_kv = struct.unpack("<IQQ", cur.read(20))
            if version != 3:
                raise ValueError(f"unsupported GGUF version {version} (need 3)")
            self.version = version
            self.n_kv = n_kv

            # Remember where the key/value block sits so a rewriter can copy
            # it verbatim. Re-encoding it would mean tracking every value's
            # original width, which nothing else here needs.
            self.kv_start = fh.tell()
            for _ in range(n_kv):
                key = cur.string()
                (vt,) = struct.unpack("<I", cur.read(4))
                self.metadata[key] = cur.value(_ValueType(vt))
            self.kv_end = fh.tell()

            self.alignment = int(
                self.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
            )

            infos: list[tuple[str, tuple[int, ...], GGMLType, int]] = []
            for _ in range(n_tensors):
                name = cur.string()
                (n_dims,) = struct.unpack("<I", cur.read(4))
                dims = struct.unpack(f"<{n_dims}Q", cur.read(8 * n_dims))
                (raw_t,) = struct.unpack("<I", cur.read(4))
                (offset,) = struct.unpack("<Q", cur.read(8))
                infos.append((name, dims, GGMLType(raw_t), offset))

            pos = fh.tell()
            pad = (self.alignment - (pos % self.alignment)) % self.alignment
            self.data_offset = pos + pad

        for name, dims, dtype, offset in infos:
            n = 1
            for d in dims:
                n *= d
            self.tensors[name] = TensorInfo(
                name=name,
                shape=dims,
                dtype=dtype,
                offset=offset,
                nbytes=nbytes_for(dtype, n),
            )

    # -- convenience ------------------------------------------------------

    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def cfg(self, suffix: str, default: Any = None) -> Any:
        """Read an architecture-scoped key, e.g. cfg('block_count')."""
        return self.metadata.get(f"{self.arch()}.{suffix}", default)

    def file_offset(self, info: TensorInfo) -> int:
        """Absolute byte offset of a tensor within the file."""
        return self.data_offset + info.offset

    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def __repr__(self) -> str:
        return (
            f"<GGUFFile {self.path.name} arch={self.arch()} "
            f"tensors={len(self.tensors)} "
            f"data={self.total_tensor_bytes() / 1e9:.2f}GB>"
        )
