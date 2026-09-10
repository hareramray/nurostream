"""Multi-shard GGUF.

Large models ship split across files (`...-00001-of-00003.gguf`). Each shard
is a complete GGUF with its own header and its own data section, so tensor
offsets are shard-relative and a reader has to know which file a tensor lives
in before it can seek.

This presents the set as one logical file: a merged tensor index where every
entry carries its shard, and metadata taken from shard 1 (which holds the
architecture keys and the tokenizer).
"""
from __future__ import annotations

import re
from pathlib import Path

from .gguf import GGUFFile, TensorInfo

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def discover_shards(path: str | Path) -> list[Path]:
    """All shards of the set `path` belongs to, in order.

    Accepts any shard (or a plain unsharded file) and returns the full set.
    """
    path = Path(path)
    m = SHARD_RE.match(path.name)
    if not m:
        return [path]
    stem, total = m.group("stem"), int(m.group("total"))
    shards = [
        path.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf"
        for i in range(1, total + 1)
    ]
    missing = [s.name for s in shards if not s.exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete shard set, missing: {', '.join(missing)}"
        )
    return shards


class ShardedGGUF:
    """A set of GGUF shards presented as one index.

    Quacks like GGUFFile: `.metadata`, `.tensors`, `.arch()`, `.cfg()`. The
    difference is `file_offset` alone is not enough to locate a tensor, so
    callers use `locate()` to get (path, absolute offset).
    """

    def __init__(self, path: str | Path) -> None:
        self.paths = discover_shards(path)
        self.shards: list[GGUFFile] = [GGUFFile(p) for p in self.paths]
        self.path = self.paths[0]

        base = self.shards[0]
        self.metadata = base.metadata
        self.alignment = base.alignment
        self.version = base.version

        self.tensors: dict[str, TensorInfo] = {}
        self._shard_of: dict[str, int] = {}
        for i, sh in enumerate(self.shards):
            for name, info in sh.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"tensor {name!r} appears in two shards")
                self.tensors[name] = info
                self._shard_of[name] = i

        declared = int(self.metadata.get("split.count", len(self.shards)) or 0)
        if declared and declared != len(self.shards):
            raise ValueError(
                f"metadata says {declared} shards, found {len(self.shards)}"
            )

    # -- GGUFFile-compatible surface --------------------------------------

    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def cfg(self, suffix: str, default=None):
        return self.metadata.get(f"{self.arch()}.{suffix}", default)

    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    # -- sharded addressing -----------------------------------------------

    def shard_index(self, name: str) -> int:
        return self._shard_of[name]

    def locate(self, name: str) -> tuple[Path, int]:
        """(file, absolute byte offset) for a tensor."""
        i = self._shard_of[name]
        sh = self.shards[i]
        return self.paths[i], sh.file_offset(sh.tensors[name])

    def file_offset(self, info: TensorInfo) -> int:
        sh = self.shards[self._shard_of[info.name]]
        return sh.file_offset(info)

    def __repr__(self) -> str:
        return (
            f"<ShardedGGUF {len(self.shards)} shards arch={self.arch()} "
            f"tensors={len(self.tensors)} "
            f"data={self.total_tensor_bytes() / 1e9:.1f}GB>"
        )
