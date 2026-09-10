"""Multi-shard GGUF, validated on a split we build ourselves.

The 235B target ships as 3 shards. Waiting for a 146 GB download to discover
an off-by-one in shard addressing would be a poor trade, so this splits a
small model into real shards and checks the sharded reader reproduces the
single-file logits exactly.

The splitter copies the metadata block byte-for-byte rather than re-encoding
it, so shard 1 keeps the exact architecture keys and tokenizer of the source.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.format.gguf import GGUFFile
from neurostream.format.sharded import ShardedGGUF, discover_shards
from neurostream.io.arena import Arena
from neurostream.io.source import MmapSource
from neurostream.io.stream import StreamingSource
from neurostream.model.qwen3 import KVCache, Qwen3Model
from neurostream.residency.cache import TieredCache

SRC = ROOT / "models" / "Qwen3-0.6B-BF16.gguf"
TOKENS = [785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 6323, 374]


def _write_string(out, s: str) -> None:
    b = s.encode("utf-8")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def split_gguf(src: Path, out_dir: Path, n_shards: int = 3) -> list[Path]:
    """Split one GGUF into `n_shards` valid GGUF files."""
    g = GGUFFile(src)
    raw = src.read_bytes()
    kv_blob = raw[g.kv_start : g.kv_end]

    names = list(g.tensors)
    groups: list[list[str]] = [[] for _ in range(n_shards)]
    for i, n in enumerate(names):
        groups[i % n_shards].append(n)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    paths = []
    for si, group in enumerate(groups, start=1):
        p = out_dir / f"{stem}-{si:05d}-of-{n_shards:05d}.gguf"
        with open(p, "wb") as out:
            out.write(b"GGUF")
            out.write(struct.pack("<IQQ", 3, len(group), g.n_kv))
            out.write(kv_blob)

            # tensor index, with offsets relative to this shard's data section
            off = 0
            offsets = []
            for name in group:
                t = g.tensors[name]
                _write_string(out, name)
                dims = t.shape
                out.write(struct.pack("<I", len(dims)))
                out.write(struct.pack(f"<{len(dims)}Q", *dims))
                out.write(struct.pack("<I", int(t.dtype)))
                out.write(struct.pack("<Q", off))
                offsets.append(off)
                pad = (-t.nbytes) % g.alignment
                off += t.nbytes + pad

            pos = out.tell()
            out.write(b"\0" * ((-pos) % g.alignment))

            for name in group:
                t = g.tensors[name]
                start = g.file_offset(t)
                out.write(raw[start : start + t.nbytes])
                out.write(b"\0" * ((-t.nbytes) % g.alignment))
        paths.append(p)
    return paths


def logits_single() -> np.ndarray:
    src = MmapSource(SRC)
    model = Qwen3Model(TieredCache(src), device="cpu")
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    src.close()
    return out.float().numpy()


def logits_sharded(first_shard: Path, budget: int) -> np.ndarray:
    arena = Arena(budget)
    src = StreamingSource(first_shard, arena)
    model = Qwen3Model(
        TieredCache(src), device="cpu", dtype=torch.float32, block_rows=1024
    )
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    peak = arena.stats.peak_bytes
    reqs = src.stats.requests
    src.close()
    return out.float().numpy(), peak, reqs


def main() -> int:
    out_dir = ROOT / "models" / "_shard_test"
    print("splitting into 3 shards ...")
    paths = split_gguf(SRC, out_dir, n_shards=3)
    for p in paths:
        print(f"  {p.name}  {p.stat().st_size / 1e6:.0f} MB")

    found = discover_shards(paths[1])  # discovery from a middle shard
    assert len(found) == 3, found
    print(f"  discovery from shard 2 found all {len(found)}")

    g = ShardedGGUF(paths[0])
    single = GGUFFile(SRC)
    print(f"  {g}")
    assert set(g.tensors) == set(single.tensors), "tensor set differs"
    spread = {g.shard_index(n) for n in g.tensors}
    print(f"  tensors spread across shards: {sorted(spread)}")

    print("running both ...")
    ref = logits_single()
    got, peak, reqs = logits_sharded(paths[0], 8 << 20)

    same = np.allclose(got, ref, atol=1e-4)
    print("\n--- P6 (shard mechanics) RESULT ---")
    print(f"  max abs diff : {np.abs(got - ref).max():.3e}")
    print(f"  top-1 match  : {got[-1].argmax()} vs {ref[-1].argmax()}")
    print(f"  arena peak   : {peak / 1e6:.1f} MB across {reqs} reads")
    print(f"  identical    : {same}")

    for p in paths:
        p.unlink()
    out_dir.rmdir()
    print(f"  cleaned up {len(paths)} shard files")
    print(f"\n  {'PASS' if same else 'FAIL'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
