# neurostream

Run transformers larger than memory. Model size is bounded by **disk
capacity**, not RAM or VRAM. Weights stream from NVMe on demand at neuron
granularity, under a hard memory ceiling that holds regardless of model size.

```python
from neurostream import NeuroStream

ns = NeuroStream.load("model.gguf", mem_budget="512MB", vram_budget="6GB")
for piece in ns.generate("The capital of France is", max_tokens=64):
    print(piece, end="", flush=True)
```

```bash
python -m neurostream info  model.gguf
python -m neurostream run   model.gguf -p "Hello" --mem-budget 1GB --vram-budget 6GB
python -m neurostream run   model.gguf --image cat.png -p "What is this?"
python -m neurostream bench model.gguf --budgets 64MB,256MB,1GB
```

## The idea

Single-batch inference reads every active weight exactly once per token, so

```
tokens/sec  =  effective_bandwidth  /  active_bytes_per_token
```

Literal one-parameter-at-a-time I/O is impossible — an SSD's minimum transfer
is a 4 KB page at ~80 µs. The library separates the two granularities:

- **Execution granularity: one neuron.** Every projection is computed in
  row-blocks; peak working memory is O(block), never O(layer).
- **I/O granularity: one block.** Rows are fetched in contiguous 256 KB–4 MB
  runs, prefetched ahead of the compute cursor.

The public API is neuron-at-a-time. The bytes leaving the disk are large and
sequential. Both are true at once, and that is the whole trick.

## Measured hardware (the design targets these numbers)

| Tier | Capacity | Bandwidth |
|---|---|---|
| VRAM (RTX 5050 Laptop, Blackwell SM 12.0) | 8.0 GB | ~350 GB/s |
| RAM (1× 16 GB DDR4-3200, **single channel**) | 16 GB | **18.2 GB/s** measured |
| NVMe (WD SN5000S, DRAM-less) | 292 GB free | **2.35 GB/s** seq (fresh 12 GB file) · **1.13 GB/s** seq / **2.00 GB/s** random-parallel on a cold 50 GB file |
| CPU | i5-13420H, 8C/12T | AVX2 |

## Results

| Phase | Exit criterion | Result |
|---|---|---|
| **P0** GGUF + fp32 CPU forward | logits match HF < 1e-3 | **PASS** — max abs diff **3.2e-5**, argmax matches all positions |
| **P1** async I/O + hard ceiling | identical output at every budget | **PASS** — identical at 512 MB–2 GB; prefetch drives stall to **0.00 s** |
| **P2** neuron-block streaming | peak O(block) not O(layer) | **PASS** — **3.0 MB peak vs 297 MB largest tensor (99×)**, identical logits |
| **P3** CUDA + residency planner | 8B Q4 ≥ 30 tok/s | **CORRECT, TARGET MISSED** — coherent output, **6.9–9.5 tok/s** (see below) |
| **P4** MoE selective fetch | 30B-A3B ≥ 8 tok/s | **CORRECT, TARGET MISSED** — 18.6 GB model in 8 GB VRAM, reads **6.9–7.2% of the model per token**, 0.7 → **1.9–2.3 tok/s** pinned |
| **P5** vision + image prefill | correct captions, prefill < 15 s | **PASS** — correct shapes, colours, spatial relations, OCR; **2.96 s** prefill (354 tok/s) |
| **P6** 235B multi-shard | ≥ 0.1 tok/s | **PASS** — **142 GB across 3 shards on an 8 GB GPU at 0.105 tok/s**, coherent output, arena peak 404 MB of 1074 MB |

### Where the throughput targets were missed, and why

P3 asked for 30 tok/s and reached ~9. The gap is the substitution made at the
start: no MSVC, CMake, or CUDA toolkit is installed on this machine, so the
engine is PyTorch + Triton rather than C++/ggml. The decode step is ~110 ms,
of which the fused kernels are ~50 ms; the rest is Python-level per-call
overhead across ~250 projections per token. Closing it needs CUDA graphs or a
C++ core — not more tuning. Kernel launch overhead was measured and ruled out
(252 fused calls = 11.8 ms).

P4 asked for 8 tok/s and reached ~2.3, but the *architectural* claim it was
testing holds cleanly: a 235B-class MoE reads only its active experts.
Measured at **1.33 GB read per token from an 18.6 GB model (7.2%)** at
**2.0 GB/s**, which the follow-up benchmark below shows is this drive's real
parallel-read ceiling. At ~1.3 GB/token against 2.0 GB/s, ~2 tok/s *is* the
hardware limit; beating it requires the experts resident, which needs memory
this machine does not have.

### P6: 235B on 8 GB of VRAM

```
model on disk    : 142.2 GB across 3 shards
VRAM available   : 8.0 GB    RAM: 16 GB
prompt           : 'The capital of France is'
output           : ' Paris, which is also the most populous'
read per token   : ~13.2 GB steady state (9.3% of the model)
throughput       : 0.105 tok/s (9.5 s/token)
arena peak       : 404 MB of a 1074 MB budget
```

Two changes took this from 0.045 to 0.105 tok/s, and neither was about
compute:

1. **Batch the expert reads.** The router names all 8 experts before any of
   them is needed, so all 24 slab reads for a layer are issued at once
   instead of one at a time. Queue depth 1 → ~8. Prefetch hit rate went
   0% → 92%.
2. **Stop over-threading.** See below.

## What the numbers taught us

**Quantized weights must stay quantized in VRAM.** The obvious design caches
dequantized fp16, but an 8B Q4 model is 5 GB on disk and 16 GB as fp16 — it
does not fit in 8 GB; quantized, it does. See `residency/cache.py`.

**Fused kernels matter more than I/O here.** Expanding the LM head to fp16 and
calling `torch.matmul` cost **6028 ms per token**. The Triton fused
dequant-GEMV does the same work in **7.6 ms** — a **794× speedup** — because
the weights never materialise. Total engine speedup: 0.63 → 9.5 tok/s (15×).

**MoE expert locality is real but prompt-specific.** Top-20% of experts take
~57% of routings (uniform would be 20%). Pinning hot expert slabs from a
warmup gives **85.7%** slab hit-rate on the *same* prompt but only **37.5%**
on a different one. Warmup-based pinning helps; it does not generalise as
much as the aggregate skew suggests.

**More I/O threads is not more throughput.** Measured with 5.85 MB random
reads (one expert slab) on the 50 GB shard: 1 worker 0.62 GB/s, **8 workers
2.00 GB/s**, 24 workers 1.57, 48 workers 1.37. A DRAM-less controller thrashes
its mapping cache when too many streams compete. Running P6 at 24 workers
instead of 8 cost about a third of the throughput.

**Benchmark on cold, realistically-sized files.** The initial 2.35 GB/s
sequential figure came from a freshly-written 12 GB file still in the drive's
SLC cache. The same drive reads a cold 50 GB file at 1.13 GB/s sequential.
Every projection built on the first number was ~2x optimistic.

**Image prefill is nearly free.** 1048 image tokens prefill in 2.96 s
(354 tok/s) because all patch tokens share one pass over the weights — the
same tokens generated one at a time would take ~110 s.

## Layout

```
neurostream/
  format/    gguf.py  sharded.py  tokenizer.py     container, multi-shard, BPE
  io/        reader.py  arena.py  stream.py  source.py   async I/O, budget
  residency/ cache.py  planner.py                   VRAM/RAM/disk placement
  compute/   quant.py  triton_kernels.py  blockffn.py  ops.py
  moe/       router.py                               selective expert fetch
  vision/    tower.py  prompt.py                     ViT, mRoPE, DeepStack
  model/     qwen3.py                                Qwen3 / Qwen3-VL / MoE
  api.py  cli.py
```

Supported: Qwen3, Qwen3-VL, Qwen3-VL-MoE. Quantization: Q4_K, Q5_K, Q6_K,
Q8_0, Q4_0, F32, F16, BF16 — Q4_K and Q6_K bit-exact against the reference
`gguf` implementation, and fused in Triton for decode.

## Tests

```bash
python tests/test_p0_logits.py      # vs HuggingFace
python tests/test_p1_streaming.py   # budget independence
python tests/test_p2_neuron.py      # O(block) peak memory
python tests/test_p6_sharded.py     # splits a model into 3 real shards
python tests/test_p3_p4_p5.py       # CUDA, MoE, vision (skips absent models)
```

The tokenizer is validated against HuggingFace on 314 cases including 300
fuzzed strings: zero encode mismatches, zero round-trip failures.

## Two bugs worth knowing about

Both were found by testing, and both are the kind that produce plausible-looking
garbage rather than an exception:

- **`id()`-keyed memoization.** A row-geometry cache keyed by `id(tensor_info)`
  silently hands one model's strides to the next model loaded in the same
  process, because CPython reuses addresses after GC. Every weight then reads
  from the wrong offset. Now memoized on the instance.
- **Prefetch vs block-streaming deadlock.** Whole-tensor prefetch reserves
  arena bytes released only by `raw_bytes()`, but the block path consumes via
  `fetch_rows()` and never calls it, so the arena starves. Whole-tensor
  prefetch is now disabled whenever block streaming is active.
