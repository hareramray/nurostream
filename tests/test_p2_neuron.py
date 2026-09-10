"""P2 exit criterion: peak memory is O(block), not O(layer).

The proof is a budget far below the largest single tensor. The LM head in this
model is 297 MB; P1 could not go under that. Here the same model runs in a
budget of a few megabytes and produces the same logits.
"""
from __future__ import annotations

import sys
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
Q4 = ROOT / "models" / "Qwen3-0.6B-Q4_K_M.gguf"
TOKENS = [785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 6323, 374]
MB = 1 << 20


def reference() -> np.ndarray:
    src = MmapSource(GGUF)
    model = Qwen3Model(TieredCache(src), device="cpu")
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    src.close()
    return out.float().numpy()


def run(path, budget: int, block_rows: int, count_neurons=False):
    arena = Arena(budget)
    src = StreamingSource(path, arena, n_workers=16)
    seen = {"n": 0, "blocks": 0}

    def cb(name, start, end, out):
        seen["n"] += end - start
        seen["blocks"] += 1

    model = Qwen3Model(
        TieredCache(src), device="cpu", dtype=torch.float32,
        block_rows=block_rows, prefetch_depth=0,
        neuron_callback=cb if count_neurons else None,
    )
    with torch.no_grad():
        out = model.forward(torch.tensor(TOKENS), KVCache(model.cfg.n_layer))
    logits = out.float().numpy()
    peak = arena.stats.peak_bytes
    src.close()
    return logits, peak, seen


def main() -> int:
    ref = reference()
    largest = max(t.nbytes for t in MmapSource(GGUF).gguf.tensors.values())
    print(f"largest tensor in file: {largest / MB:.0f}MB")
    print(f"reference top-1: {ref[-1].argmax()}\n")

    ok = True
    print(f"{'block_rows':>11} {'budget':>8} {'peak':>8} {'ratio':>7}  match")
    print("-" * 48)
    for block_rows, budget_mb in [(256, 4), (1024, 8), (4096, 32)]:
        logits, peak, _ = run(GGUF, budget_mb * MB, block_rows)
        same = np.allclose(logits, ref, atol=1e-4)
        ok &= same and peak <= budget_mb * MB
        print(
            f"{block_rows:>11} {budget_mb:>6}MB {peak / MB:>6.1f}MB "
            f"{largest / peak:>6.0f}x  {'yes' if same else 'NO'}"
        )

    print("\nneuron callback (block_rows=1024, budget 8MB):")
    _, _, seen = run(GGUF, 8 * MB, 1024, count_neurons=True)
    print(f"  {seen['n']:,} neurons computed across {seen['blocks']:,} blocks")

    print("\nQ4_K_M through the same path (budget 8MB):")
    q4_logits, q4_peak, _ = run(Q4, 8 * MB, 1024)
    ref_p = np.exp(ref[-1] - ref[-1].max())
    ref_p /= ref_p.sum()
    q4_p = np.exp(q4_logits[-1] - q4_logits[-1].max())
    q4_p /= q4_p.sum()
    kl = float((ref_p * np.log((ref_p + 1e-12) / (q4_p + 1e-12))).sum())
    print(f"  peak {q4_peak / MB:.1f}MB | top-1 {q4_logits[-1].argmax()} "
          f"(bf16 ref {ref[-1].argmax()}) | KL(bf16||q4) {kl:.5f} nats")

    print("\n--- P2 RESULT ---")
    print(f"  O(block) peak, identical output : {ok}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
