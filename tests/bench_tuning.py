"""Repeatable 235B block/cache benchmark; writes measured decode-only results.

python tests/bench_tuning.py --output models/_tuning/current.json
python tests/bench_tuning.py --rows 1024,4096,8192 --output models/_tuning/tuned.json

Uses identical teacher-forced tokens for every configuration. Load, warmup,
pinning, and prompt prefill are excluded from decode timing and read counts.
The OS file cache is not flushed; compare more than one run before claiming
a throughput improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.api import NeuroStream
from neurostream.model.qwen3 import KVCache
from run_235b import find_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model')
    ap.add_argument('--rows', default='4096')
    ap.add_argument('--slab-budget', default='3GB')
    ap.add_argument('--tokens', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    rows = [int(v) for v in a.rows.split(',')]
    if min(rows) <= 0 or a.tokens <= 0 or a.repeats <= 0:
        ap.error('rows, tokens, and repeats must be positive')
    path, _ = find_model(a.model)
    results = []
    with NeuroStream.load(
        path, mem_budget='1GB', vram_budget='7GB', ram_budget='2GB',
        slab_budget=a.slab_budget, reserve_vram='1.5GB', block_rows=4096,
        n_workers=8,
    ) as ns:
        prompt = ns.tokenizer.encode('The capital of France is')
        continuation = ns.tokenizer.encode(' Paris, which is also the most populous city in France.')
        if a.tokens > len(continuation):
            ap.error(f'--tokens must be <= {len(continuation)}')
        print(ns.placement.summary(), flush=True)
        print('Warmup: 3 tokens, then pin hot experts ...', flush=True)
        ns.warmup(prompt='The capital of France is', tokens=3)
        ns.pin_hot_experts(verbose=True)
        for repeat in range(a.repeats):
            for block_rows in (rows if repeat % 2 == 0 else rows[::-1]):
                print(f'Prefill: repeat {repeat + 1}, block_rows={block_rows}', flush=True)
                ns.model.block_rows = block_rows
                kv = KVCache(ns.cfg.n_layer)
                with torch.no_grad():
                    torch.cuda.synchronize()
                    prefill_start = time.perf_counter()
                    logits = ns.model.forward(torch.tensor(prompt), kv)
                    predictions = [int(logits[-1].argmax())]
                    torch.cuda.synchronize()
                    prefill_seconds = time.perf_counter() - prefill_start
                    print(f'Decode: {a.tokens} fixed tokens ...', flush=True)
                    before = ns.source.stats.bytes_read
                    t0 = time.perf_counter()
                    for i, token in enumerate(continuation[:a.tokens]):
                        logits = ns.model.forward(torch.tensor([token]), kv,
                                                  start_pos=len(prompt) + i)
                        predictions.append(int(logits[-1].argmax()))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                result = dict(
                    repeat=repeat, block_rows=block_rows, slab_budget=a.slab_budget,
                    prefill_seconds=prefill_seconds,
                    seconds=elapsed, tokens_per_second=a.tokens / elapsed,
                    bytes_per_token=(ns.source.stats.bytes_read - before) / a.tokens,
                    predictions=predictions, arena_peak=ns.arena.stats.peak_bytes,
                    vram_weights=ns.cache.stats.vram_bytes, ram_weights=ns.cache.stats.ram_bytes,
                    expert_cache_bytes=ns.cache.slab_bytes,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                out = Path(a.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
