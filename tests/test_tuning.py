"""Small CPU regressions for block sizing and cache reuse; no model download."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurostream.compute.blockffn import stream_linear
from neurostream.api import NeuroStream
from neurostream.compute.quant import dequantize
from neurostream.format.gguf import GGMLType, TensorInfo, nbytes_for
from neurostream.io.arena import Arena
from neurostream.io.source import row_geometry
from neurostream.io.stream import StreamingSource
from neurostream.moe.router import _expert_matmul, submit_expert_reads
from neurostream.residency.cache import TieredCache
from neurostream.residency.planner import plan


def tensor(name, shape, dtype=GGMLType.F32):
    count = 1
    for size in shape:
        count *= size
    return TensorInfo(name, tuple(reversed(shape)), dtype, 0, nbytes_for(dtype, count))


class MemorySource:
    """In-memory bytes with real arena accounting and immediate futures."""

    def __init__(self, infos, raw, budget=4096):
        self.gguf = SimpleNamespace(tensors={t.name: t for t in infos})
        self.raw = raw
        self.arena = Arena(budget)
        self.reader = SimpleNamespace(wait=lambda fut: fut.result())
        self.reads = 0

    def raw_bytes(self, name):
        self.reads += 1
        return self.raw[name].clone()

    def raw_rows(self, name, start, end):
        self.reads += 1
        rb = row_geometry(self.gguf.tensors[name])[2]
        return self.raw[name][start * rb:end * rb].clone()

    def fetch_rows(self, name, start, end, device='cpu', dtype=torch.float32):
        info = self.gguf.tensors[name]
        _, elems, rb = row_geometry(info)
        self.arena.acquire((end - start) * rb, timeout=0)
        try:
            return dequantize(self.raw_rows(name, start, end), info.dtype,
                              (end - start) * elems, out_dtype=dtype).reshape(end - start, elems).to(device)
        finally:
            self.arena.release((end - start) * rb)

    def fetch(self, name, device='cpu', dtype=torch.float32):
        info = self.gguf.tensors[name]
        return dequantize(self.raw_bytes(name), info.dtype, info.n_elements,
                          out_dtype=dtype).reshape(info.torch_shape).to(device)

    def submit_rows(self, name, start, end, *, timeout=None):
        size = (end - start) * row_geometry(self.gguf.tensors[name])[2]
        # Fail immediately instead of hanging the regression test.
        self.arena.acquire(size, timeout=0)
        fut = Future()
        fut.set_result(self.raw_rows(name, start, end))
        return fut, size


class TuningTests(unittest.TestCase):
    def test_oversized_blocks_fit_two_buffer_budget(self):
        w = torch.arange(17 * 8, dtype=torch.float32).reshape(17, 8) / 100
        x = torch.ones(2, 8)
        info = tensor('w', w.shape)
        for budget in (32, 96, 256):
            with self.subTest(budget=budget):
                src = MemorySource([info], {'w': w.flatten().view(torch.uint8)}, budget)
                actual = stream_linear(src, 'w', x, block_rows=4096)
                torch.testing.assert_close(actual, x @ w.T)
                self.assertEqual(src.arena.used, 0)
                self.assertLessEqual(src.arena.stats.peak_bytes, budget)

    def test_cache_admission_is_idempotent_and_respects_pins(self):
        infos = [tensor(n, (8,)) for n in ('a', 'b')]
        src = MemorySource(infos, {n: torch.ones(8).view(torch.uint8) for n in ('a', 'b')})
        for tier in ('vram', 'ram'):
            cache = TieredCache(src, vram_budget=32, ram_budget=32)
            admit = getattr(cache, f'admit_{tier}')
            self.assertTrue(admit('a'))
            reads = src.reads
            self.assertTrue(admit('a'))
            self.assertEqual(src.reads, reads)
            cache.pin(['a'])
            self.assertFalse(admit('b'))
            self.assertEqual(getattr(cache.stats, f'{tier}_bytes'), 32)

    def test_pinned_slab_serves_neuron_subranges(self):
        """A pinned expert must still hit when read a neuron block at a time.

        Neuron-block streaming asks for sub-ranges of the slab the planner
        pinned. Matching only exact row ranges would miss every one of them
        and go back to disk, which would make pinning worthless precisely
        where it was measured to help.
        """
        info = tensor('expert', (1, 4, 32), GGMLType.Q8_0)
        row = torch.cat((torch.tensor([1.], dtype=torch.float16).view(torch.uint8),
                         torch.arange(1, 33, dtype=torch.uint8)))
        src = MemorySource([info], {'expert': row.repeat(4)})
        cache = TieredCache(src)
        cache.slab_budget = info.nbytes
        self.assertTrue(cache.admit_slab('expert', 0, 4))
        reads = src.reads

        row_bytes = row_geometry(info)[2]
        self.assertTrue(cache.has_slab('expert', 1, 3))
        for start, end in ((0, 1), (1, 3), (2, 4), (0, 4)):
            with self.subTest(rows=(start, end)):
                torch.testing.assert_close(
                    cache.fetch_rows('expert', start, end),
                    torch.arange(1, 33).float().repeat(end - start, 1),
                )
                raw, dtype = cache.raw_rows('expert', start, end)
                self.assertEqual(dtype, GGMLType.Q8_0)
                self.assertEqual(raw.numel(), (end - start) * row_bytes)
        self.assertEqual(src.reads, reads)
        self.assertEqual(cache.stats.slab_misses, 0)

    def test_neuron_blocks_read_the_pinned_slab_not_the_disk(self):
        """Block streaming over a pinned expert must not touch the source."""
        info = tensor('expert', (1, 4, 32), GGMLType.Q8_0)
        row = torch.cat((torch.tensor([1.], dtype=torch.float16).view(torch.uint8),
                         torch.arange(1, 33, dtype=torch.uint8)))
        src = MemorySource([info], {'expert': row.repeat(4)}, budget=64)
        cache = TieredCache(src)
        cache.slab_budget = info.nbytes
        self.assertTrue(cache.admit_slab('expert', 0, 4))
        reads = src.reads

        x = torch.ones(2, 32)
        # One row per block: the finest possible neuron granularity, and a
        # budget far too small to have streamed this from disk.
        out = stream_linear(cache, 'expert', x, block_rows=1,
                            row_start=0, row_end=4)
        torch.testing.assert_close(out, torch.full((2, 4), 528.))
        self.assertEqual(src.reads, reads)
        self.assertEqual(src.arena.stats.total_acquired, 0)

    def test_untied_embedding_loses_to_output_and_norms(self):
        emb = tensor('token_embd.weight', (2048, 256), GGMLType.Q8_0)
        head = tensor('output.weight', (2048, 256), GGMLType.Q8_0)
        norm = tensor('output_norm.weight', (256,))
        gguf = SimpleNamespace(tensors={i.name: i for i in (emb, head, norm)})
        placed = plan(gguf, vram_budget=head.nbytes + 512)
        self.assertIn(head.name, placed.vram)
        self.assertIn(norm.name, placed.vram)
        self.assertIn(emb.name, placed.disk)
        del gguf.tensors[head.name]
        tied = plan(gguf, vram_budget=emb.nbytes + 512)
        self.assertIn(emb.name, tied.vram)

    def test_float_residency_uses_actual_compute_size(self):
        info = tensor('w', (256,))
        gguf = SimpleNamespace(tensors={'w': info})
        placed = plan(gguf, vram_budget=512, float_itemsize=4)
        self.assertEqual(placed.vram, [])

    def test_cached_expert_prefill_does_not_read_again(self):
        info = tensor('expert', (1, 2, 32), GGMLType.Q8_0)
        # Two Q8_0 rows: fp16 scale 1 followed by signed values 1..32.
        row = torch.cat((torch.tensor([1.], dtype=torch.float16).view(torch.uint8),
                         torch.arange(1, 33, dtype=torch.uint8)))
        raw = row.repeat(2)
        src = MemorySource([info], {'expert': raw})
        cache = TieredCache(src)
        cache.slab_budget = info.nbytes
        self.assertTrue(cache.admit_slab('expert', 0, 2))
        reads = src.reads
        model = SimpleNamespace(cache=cache, gguf=src.gguf, device='cpu', dtype=torch.float32)
        x = torch.ones(3, 32)
        actual = _expert_matmul(model, 'expert', 0, 2, x)
        torch.testing.assert_close(actual, torch.full((3, 2), 528.))
        torch.testing.assert_close(cache.fetch_rows('expert', 0, 2),
                                   torch.arange(1, 33).float().repeat(2, 1))
        self.assertEqual(src.reads, reads)

    def test_expert_prefetch_stops_at_budget(self):
        cfg = SimpleNamespace(n_ffn_expert=2, n_embd=2)
        infos = [tensor(f'blk.0.ffn_{kind}_exps.weight', (2, 2, 32), GGMLType.Q8_0)
                 for kind in ('gate', 'up', 'down')]
        src = MemorySource(infos, {i.name: torch.zeros(i.nbytes, dtype=torch.uint8) for i in infos}, 68)
        model = SimpleNamespace(cache=TieredCache(src), gguf=src.gguf, cfg=cfg)
        pending = submit_expert_reads(model, 0, [0, 1])
        self.assertEqual(len(pending), 1)
        for _, size in pending.values():
            src.arena.release(size)
        self.assertEqual(src.arena.used, 0)

    def test_expert_pinning_does_not_leave_partial_groups(self):
        infos = [tensor(f'blk.0.ffn_{kind}_exps.weight', (2, 2, 32), GGMLType.Q8_0)
                 for kind in ('gate', 'up', 'down')]
        src = MemorySource(infos, {i.name: torch.zeros(i.nbytes, dtype=torch.uint8) for i in infos})
        ns = NeuroStream.__new__(NeuroStream)
        ns.cache = TieredCache(src)
        ns.gguf = src.gguf
        ns.cfg = SimpleNamespace(n_ffn_expert=2, n_embd=2)
        ns.model = SimpleNamespace(router_stats=SimpleNamespace(hits=Counter({(0, 0): 3})))
        ns.cache.slab_budget = 136
        self.assertEqual(ns.pin_hot_experts(), 0)
        self.assertEqual(src.reads, 0)
        self.assertEqual(ns.cache.slab_bytes, 0)
        ns.cache.slab_budget = 204
        self.assertEqual(ns.pin_hot_experts(), 1)
        self.assertEqual(ns.cache.slab_bytes, 204)
        self.assertEqual(ns.pin_hot_experts(), 0)

    def test_submit_failure_releases_reservation(self):
        info = tensor('w', (8,))
        src = StreamingSource.__new__(StreamingSource)
        src.gguf = SimpleNamespace(tensors={'w': info}, file_offset=lambda _: 0)
        src.arena = Arena(128)
        def fail(*args):
            raise RuntimeError('closed reader')
        src._reader_for = lambda _: SimpleNamespace(submit=fail)
        with self.assertRaisesRegex(RuntimeError, 'closed reader'):
            src.submit_rows('w', 0, 1)
        self.assertEqual(src.arena.used, 0)


if __name__ == '__main__':
    unittest.main()
