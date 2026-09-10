"""P1 exit criterion: async streaming under a hard memory ceiling.

Two things are proven here:
  1. Output is bit-identical to the mmap baseline at every budget.
  2. The arena's peak never crosses the budget it was given.

That second property is the one the whole library is for, so it is asserted,
not merely reported.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.io.arena import Arena
from neurostream.io.source import MmapSource
from neurostream.io.stream import StreamingSource
from neurostream.model.qwen3 import KVCache, Qwen3Model
from neurostream.residency.cache import TieredCache

GGUF = ROOT / "models" / "Qwen3-0.6B-BF16.gguf"
TOKENS = [785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 6323, 374]

MB = 1 << 20


def run_mmap() -> np.ndarray:
    src = MmapSource(GGUF)
    cache = TieredCache(src, device="cpu", compute_dtype=torch.float32)
    model = Qwen3Model(cache, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    src.close()
    return out.float().numpy()


def run_stream(budget: int, depth: int, workers: int = 16):
    arena = Arena(budget)
    src = StreamingSource(GGUF, arena, n_workers=workers)
    cache = TieredCache(src, device="cpu", compute_dtype=torch.float32)
    model = Qwen3Model(
        cache, device="cpu", dtype=torch.float32, prefetch_depth=depth
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    dt = time.perf_counter() - t0
    logits = out.float().numpy()
    stats = (arena, src.stats, dt)
    src.close()
    return logits, stats


def main() -> int:
    print("baseline (mmap) ...")
    ref = run_mmap()
    print(f"  logits {ref.shape}, top-1 {ref[-1].argmax()}")

    largest = max(
        t.nbytes for t in MmapSource(GGUF).gguf.tensors.values()
    )
    print(f"  largest single tensor: {largest / MB:.0f}MB "
          f"(the P1 budget floor; P2 row-streaming removes it)\n")

    ok = True
    print(f"{'budget':>9} {'depth':>6} {'peak':>9} {'time':>7} "
          f"{'GB/s':>6} {'stall':>7} {'pf-hit':>7}  match")
    print("-" * 68)
    for budget_mb, depth in [
        (512, 0), (512, 1), (512, 2), (1024, 2), (2048, 2),
    ]:
        logits, (arena, io, dt) = run_stream(budget_mb * MB, depth)
        same = np.array_equal(logits, ref)
        within = arena.stats.peak_bytes <= arena.budget
        ok &= same and within
        print(
            f"{budget_mb:>7}MB {depth:>6} "
            f"{arena.stats.peak_bytes / MB:>7.0f}MB {dt:>6.2f}s "
            f"{io.throughput_gbps:>6.2f} {io.stall_seconds:>6.2f}s "
            f"{io.prefetch_rate:>6.1%}  {'yes' if same else 'NO'}"
            f"{'' if within else '  BUDGET EXCEEDED'}"
        )

    print("\n--- P1 RESULT ---")
    print(f"  identical output at every budget : {ok}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
