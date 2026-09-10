"""P6: Qwen3-VL-235B-A22B Q4_K_M (146 GB, 3 shards) on an 8 GB GPU.

Exit criterion: >= 0.1 tok/s.

The whole library exists for this case. 146 GB of weights, 8 GB of VRAM and
16 GB of RAM. It works because only 22B of 235B parameters are active per
token, and because nothing is ever held that is not being multiplied right
now.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.api import NeuroStream
from neurostream.format.sharded import ShardedGGUF, discover_shards

DIR = ROOT / "models" / "Qwen3-VL-235B-A22B-Instruct-GGUF"
SHARD1 = DIR / "Q4_K_M" / "Qwen3-VL-235B-A22B-Instruct-Q4_K_M-00001-of-00003.gguf"
MMPROJ = DIR / "mmproj-F16.gguf"

PROMPT = "The capital of France is"
TARGET = 0.1


def main() -> int:
    if not SHARD1.exists():
        print(f"shard 1 not found at {SHARD1}")
        return 2

    shards = discover_shards(SHARD1)
    total = sum(s.stat().st_size for s in shards)
    print(f"shards: {len(shards)}, {total / 1e9:.1f} GB on disk")
    for s in shards:
        print(f"    {s.name}  {s.stat().st_size / 1e9:.1f} GB")

    t0 = time.perf_counter()
    g = ShardedGGUF(SHARD1)
    print(f"\nindex parsed in {time.perf_counter() - t0:.2f}s "
          f"(no tensor data touched)")
    print(f"  {g}")

    expert_bytes = sum(t.nbytes for n, t in g.tensors.items() if "_exps" in n)
    other = g.total_tensor_bytes() - expert_bytes
    n_exp = int(g.cfg("expert_count", 0) or 0)
    n_used = int(g.cfg("expert_used_count", 0) or 0)
    per_tok = other + expert_bytes * (n_used / max(n_exp, 1))
    print(f"  {n_exp} experts, {n_used} active per token")
    print(f"  predicted read/token: {per_tok / 1e9:.1f} GB "
          f"({per_tok / g.total_tensor_bytes():.1%} of the model)")

    print("\nloading ...", flush=True)
    t0 = time.perf_counter()
    ns = NeuroStream.load(
        str(SHARD1), mem_budget="1GB", vram_budget="7GB", ram_budget="2GB",
        slab_budget="3.5GB", reserve_vram="1.5GB", block_rows=4096,
        n_workers=8,
    )
    print(f"  ready in {time.perf_counter() - t0:.1f}s")
    print(f"  placement: {ns.placement.summary()}")
    print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nwarmup, then pin the experts this model actually routes to ...",
          flush=True)
    t0 = time.perf_counter()
    ns.warmup(prompt=PROMPT, tokens=3)
    ns.pin_hot_experts(verbose=True)
    print(f"  warmup+pin in {time.perf_counter() - t0:.0f}s")
    print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    n_tokens = 8
    print(f"\ngenerating {n_tokens} tokens ...", flush=True)
    out = "".join(ns.generate(PROMPT, max_tokens=n_tokens, temperature=0.0))

    tps = ns.stats.tokens_per_second
    read_per_tok = (
        ns.source.stats.bytes_read / 1e9 / max(1, ns.stats.generated_tokens)
    )

    print(f"\n  prompt: {PROMPT!r}")
    print(f"  output: {out!r}")
    print("\n" + ns.report())

    print("\n--- P6 RESULT ---")
    print(f"  model on disk    : {total / 1e9:.1f} GB across {len(shards)} shards")
    print(f"  VRAM available   : 8.0 GB   RAM: 16 GB")
    print(f"  read per token   : {read_per_tok:.2f} GB "
          f"({read_per_tok / (total / 1e9):.2%} of the model)")
    print(f"  throughput       : {tps:.3f} tok/s "
          f"({1 / max(tps, 1e-9):.1f} s/token)")
    print(f"  arena peak       : {ns.arena.stats.peak_bytes / 1e6:.0f} MB "
          f"of {ns.arena.budget / 1e6:.0f} MB budget")

    ok = tps >= TARGET and len(out.strip()) > 0
    within = ns.arena.stats.peak_bytes <= ns.arena.budget
    print(f"  budget respected : {within}")
    print(f"\n  {'PASS' if ok and within else 'FAIL'} "
          f"(target >= {TARGET} tok/s)")
    ns.close()
    return 0 if (ok and within) else 1


if __name__ == "__main__":
    raise SystemExit(main())
