"""Qwen3.5-4B benchmarks: what residency, format, block size and context cost.

    python tests/bench_qwen35.py --suite vram --output models/_bench/vram.json
    python tests/bench_qwen35.py --suite all  --output models/_bench/all.json

Suites
  vram     VRAM budget vs latency. The point of the engine: a 9 GB model on an
           8 GB card. Decode reads every non-resident weight once per token, so
           latency is set by how much of the model did not fit.
  ram      Host RAM as the middle tier, with VRAM held fixed.
  format   The same weights as a safetensors checkpoint and as a BF16 GGUF.
           Measures what the checkpoint translation costs, if anything.
  blocks   Streaming block size vs latency, at fixed residency.
  context  KV and recurrent state growth vs context length. Three quarters of
           Qwen3.5's layers are recurrent and keep fixed-size state, so this is
           where the hybrid design either pays off or does not.
  image    Vision prefill: what one image costs against the same prompt without.

Every configuration decodes the same teacher-forced tokens, so the work is
identical and only the plumbing differs. Load, prefill and decode are timed
separately; the OS file cache is not flushed, so run more than once before
believing a small difference.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.api import NeuroStream
from neurostream.model.qwen3 import KVCache

CHECKPOINT = ROOT / 'models' / 'Qwen3.5-4B'
GGUF = ROOT / 'models' / 'qwen35' / 'Qwen3.5-4B-BF16.gguf'
PROMPT = 'The capital of France is'


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def measure(ns, prompt_ids, tokens, device):
    """Prefill once, then decode `tokens` greedily. Returns timings and I/O."""
    kv = KVCache(ns.cfg.n_layer)
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = ns.model.forward(torch.tensor(prompt_ids), kv)
    sync(device)
    prefill = time.perf_counter() - t0

    before = ns.source.stats.bytes_read
    steps, predicted = [], []
    with torch.no_grad():
        for i in range(tokens):
            nxt = int(logits[-1].argmax())
            predicted.append(nxt)
            sync(device)
            t0 = time.perf_counter()
            logits = ns.model.forward(torch.tensor([nxt], device=ns.device), kv,
                                      start_pos=len(prompt_ids) + i)
            sync(device)
            steps.append(time.perf_counter() - t0)
    read = ns.source.stats.bytes_read - before
    return dict(
        prefill_seconds=round(prefill, 3),
        decode_seconds=round(sum(steps), 3),
        tokens_per_second=round(tokens / sum(steps), 4) if steps else 0.0,
        ms_per_token=round(1e3 * statistics.median(steps), 1) if steps else 0.0,
        gb_per_token=round(read / tokens / 1e9, 3) if tokens else 0.0,
        kv_mb=round(kv.nbytes() / 1e6, 2),
        # Identical across configurations, or the plumbing changed the answer.
        predicted=predicted,
    )


def residency(ns):
    """Read *after* measuring: hit rate is zero until something is fetched.

    `arena_peak_mb` only means something on the GGUF path. A safetensors
    checkpoint is mmap-backed and takes no arena reservations, so it reports
    zero however much it streams.
    """
    return dict(
        vram_weights_gb=round(ns.cache.stats.vram_bytes / 1e9, 3),
        ram_weights_gb=round(ns.cache.stats.ram_bytes / 1e9, 3),
        cache_hit_rate=round(ns.cache.stats.hit_rate, 4),
        arena_peak_mb=round(ns.arena.stats.peak_bytes / 1e6, 1),
        source=type(ns.source).__name__,
    )


def run_one(path, tokens, device, record, **load_kw):
    """Load, measure, close. Returns the record or an 'error' entry."""
    started = time.perf_counter()
    try:
        ns = NeuroStream.load(path, device=device, **load_kw)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        return {**record, 'error': f'{type(exc).__name__}: {exc}'.split('\n')[0][:160]}
    try:
        load_seconds = round(time.perf_counter() - started, 2)
        ids = ns.tokenizer.encode(PROMPT)
        timings = measure(ns, ids, tokens, device)
        out = {**record, 'load_seconds': load_seconds, 'prompt_tokens': len(ids),
               **timings, **residency(ns)}
    finally:
        ns.close()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return out


def suite_vram(a, emit):
    """The headline: how much latency each GB of residency buys."""
    budgets = [b.strip() for b in a.vram.split(',')]
    for repeat in range(a.repeats):
        # Alternate the order so a warming page cache cannot masquerade as a
        # trend across the sweep.
        for budget in (budgets if repeat % 2 == 0 else budgets[::-1]):
            emit(run_one(a.model, a.tokens, a.device,
                         {'suite': 'vram', 'vram_budget': budget, 'repeat': repeat},
                         mem_budget=a.mem_budget, vram_budget=budget,
                         ram_budget='0', reserve_vram=a.reserve_vram,
                         block_rows=a.block_rows, n_workers=a.workers))


def suite_ram(a, emit):
    """The middle tier. VRAM is held fixed and host RAM takes the overflow.

    On a card too small for the model this is the only way to cover the rest
    without going back to the drive, so what matters is whether RAM residency
    buys more per GB than the disk read it replaces.
    """
    budgets = [b.strip() for b in a.ram.split(',')]
    for repeat in range(a.repeats):
        for budget in (budgets if repeat % 2 == 0 else budgets[::-1]):
            emit(run_one(a.model, a.tokens, a.device,
                         {'suite': 'ram', 'vram_budget': a.format_vram,
                          'ram_budget': budget, 'repeat': repeat},
                         mem_budget=a.mem_budget, vram_budget=a.format_vram,
                         ram_budget=budget, reserve_vram=a.reserve_vram,
                         block_rows=a.block_rows, n_workers=a.workers))


def suite_format(a, emit):
    """Same weights, two containers: safetensors checkpoint against BF16 GGUF."""
    pairs = [('safetensors', CHECKPOINT), ('gguf-bf16', GGUF)]
    for repeat in range(a.repeats):
        for label, path in (pairs if repeat % 2 == 0 else pairs[::-1]):
            if not Path(path).exists():
                emit({'suite': 'format', 'format': label, 'error': f'missing: {path}'})
                continue
            emit(run_one(path, a.tokens, a.device,
                         {'suite': 'format', 'format': label, 'repeat': repeat},
                         mem_budget=a.mem_budget, vram_budget=a.format_vram,
                         ram_budget='0', reserve_vram=a.reserve_vram,
                         block_rows=a.block_rows, n_workers=a.workers))


def suite_blocks(a, emit):
    """Block size vs latency, without reloading: residency is held fixed."""
    ns = NeuroStream.load(a.model, device=a.device, mem_budget=a.mem_budget,
                          vram_budget=a.format_vram, ram_budget='0',
                          reserve_vram=a.reserve_vram, block_rows=a.block_rows,
                          n_workers=a.workers)
    try:
        ids = ns.tokenizer.encode(PROMPT)
        for rows in (int(v) for v in a.blocks.split(',')):
            ns.model.block_rows = rows
            timings = measure(ns, ids, a.tokens, a.device)
            emit({'suite': 'blocks', 'block_rows': rows, **timings, **residency(ns)})
    finally:
        ns.close()


def suite_context(a, emit):
    """KV and recurrent state against context length.

    Only the full-attention layers grow a KV history; the recurrent layers
    hold a fixed-size state no matter how long the context gets. The dense
    figure is what the same depth would cost if every layer kept a history.
    """
    ns = NeuroStream.load(a.model, device=a.device, mem_budget=a.mem_budget,
                          vram_budget=a.format_vram, ram_budget='0',
                          reserve_vram=a.reserve_vram, block_rows=a.block_rows,
                          n_workers=a.workers)
    try:
        cfg = ns.cfg
        full = sum(1 for r in cfg.recurrent_layers if not r)
        for length in (int(v) for v in a.context.split(',')):
            ids = (ns.tokenizer.encode(PROMPT) * length)[:length]
            kv = KVCache(cfg.n_layer)
            sync(a.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = ns.model.forward(torch.tensor(ids), kv)
            sync(a.device)
            prefill = time.perf_counter() - t0
            sync(a.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                ns.model.forward(torch.tensor([int(logits[-1].argmax())]), kv,
                                 start_pos=length)
            sync(a.device)
            step = time.perf_counter() - t0
            dense = (cfg.n_layer * 2 * cfg.n_head_kv * cfg.head_dim
                     * length * ns.dtype.itemsize)
            emit({'suite': 'context', 'context_tokens': length,
                  'prefill_seconds': round(prefill, 3),
                  'prefill_tokens_per_second': round(length / prefill, 1),
                  'decode_ms': round(step * 1e3, 1),
                  'kv_mb': round(kv.nbytes() / 1e6, 2),
                  'dense_equivalent_kv_mb': round(dense / 1e6, 2),
                  'full_attention_layers': f'{full}/{cfg.n_layer}'})
    finally:
        ns.close()


def suite_image(a, emit):
    """What one image costs: tower encode, extra prompt tokens, prefill."""
    from PIL import Image
    import numpy as np

    path = Path(a.image) if a.image else None
    if path is None or not path.exists():
        path = Path(a.output).parent / 'bench-image.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (448, 448, 3), dtype='uint8')).save(path)
    ns = NeuroStream.load(a.model, device=a.device, mem_budget=a.mem_budget,
                          vram_budget=a.format_vram, ram_budget='0',
                          reserve_vram=a.reserve_vram, block_rows=a.block_rows,
                          n_workers=a.workers)
    try:
        started = time.perf_counter()
        ns.attach_vision()
        tower_seconds = time.perf_counter() - started
        for max_patches in (int(v) for v in a.patches.split(',')):
            sync(a.device)
            t0 = time.perf_counter()
            embeds, _, gh, gw = ns.tower.encode(path, max_patches=max_patches)
            sync(a.device)
            encode = time.perf_counter() - t0
            started = time.perf_counter()
            pieces = list(ns.generate_vl(path, 'Describe this image.', max_tokens=a.tokens,
                                         max_patches=max_patches, enable_thinking=False))
            emit({'suite': 'image', 'max_patches': max_patches,
                  'tower_load_seconds': round(tower_seconds, 2),
                  'image_tokens': int(embeds.shape[0]), 'merged_grid': f'{gh}x{gw}',
                  'encode_seconds': round(encode, 3),
                  'prompt_tokens': ns.stats.prompt_tokens,
                  'prefill_seconds': round(ns.stats.prefill_seconds, 3),
                  'decode_tokens_per_second': round(ns.stats.tokens_per_second, 3),
                  'total_seconds': round(time.perf_counter() - started, 2),
                  'reply_chars': len(''.join(pieces))})
    finally:
        ns.close()


SUITES = {'vram': suite_vram, 'ram': suite_ram, 'format': suite_format,
          'blocks': suite_blocks, 'context': suite_context, 'image': suite_image}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=str(CHECKPOINT),
                    help='safetensors checkpoint directory or GGUF file')
    ap.add_argument('--suite', default='vram',
                    help=f"comma separated, or 'all': {', '.join(SUITES)}")
    ap.add_argument('--output', default='models/_bench/qwen35.json')
    ap.add_argument('--tokens', type=int, default=3, help='decode tokens per configuration')
    ap.add_argument('--repeats', type=int, default=1,
                    help='passes over the vram/ram sweeps; report the median')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--vram', default='0,1GB,2GB,3GB,4GB,5GB,6GB')
    ap.add_argument('--ram', default='0,1GB,2GB,4GB,6GB')
    ap.add_argument('--blocks', default='256,1024,4096')
    ap.add_argument('--context', default='128,512,1024,2048')
    ap.add_argument('--patches', default='256,1024')
    ap.add_argument('--image', default=None)
    ap.add_argument('--mem-budget', default='512MB')
    ap.add_argument('--reserve-vram', default='1.5GB')
    ap.add_argument('--format-vram', default='4GB',
                    help='residency held fixed by the non-vram suites')
    ap.add_argument('--block-rows', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.tokens <= 0 or a.repeats <= 0:
        ap.error('--tokens and --repeats must be positive')
    names = list(SUITES) if a.suite == 'all' else [s.strip() for s in a.suite.split(',')]
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        ap.error(f"unknown suite(s): {', '.join(unknown)}")
    if not Path(a.model).exists():
        ap.error(f'model not found: {a.model}')

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def emit(record):
        results.append(record)
        print(json.dumps(record), flush=True)
        out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')

    for name in names:
        print(f'\n=== {name} ===', flush=True)
        SUITES[name](a, emit)
    summarize(results)
    print(f'\nwrote {out}', flush=True)


# The columns worth reading back; everything else stays in the JSON.
REPORT = ('gb_per_token', 'ms_per_token', 'tokens_per_second', 'prefill_seconds',
          'cache_hit_rate', 'kv_mb')
IDENT = ('vram_budget', 'ram_budget', 'format', 'block_rows', 'context_tokens',
         'max_patches')


def summarize(results):
    """Median per configuration, so repeats collapse into one row."""
    groups: dict = {}
    for r in results:
        if 'error' in r:
            continue
        groups.setdefault((r.get('suite'), *(r.get(k) for k in IDENT)), []).append(r)
    if not groups:
        return
    print('\n=== medians ===', flush=True)
    for ident, rows in groups.items():
        label = ' '.join(f'{k}={v}' for k, v in zip(('suite',) + IDENT, ident)
                         if v is not None)
        cells = []
        for key in REPORT:
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if values:
                cells.append(f'{key}={statistics.median(values):g}')
        print(f'  {label:<40} n={len(rows):<3} ' + ' '.join(cells), flush=True)
    # A configuration that changed the answer is a bug, not a speed result.
    outputs = {tuple(r['predicted']) for r in results if 'predicted' in r}
    if outputs:
        print(f'  distinct decoded outputs across all runs: {len(outputs)}', flush=True)


if __name__ == '__main__':
    main()
