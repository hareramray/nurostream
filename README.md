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

- **Streamed projections use blocks of output neurons.** A block is a
  contiguous run of rows — rows *are* output neurons, and GGUF stores each row
  as a whole number of quantization blocks, so a row range is one sequential
  read and one independent dequant. Blocks shrink when necessary to fit the
  pipeline — `lookahead + 1` raw I/O buffers — inside the arena.
- **Resident projections can use a whole-matrix fused kernel.** Frequently
  used weights stay cached in RAM or VRAM instead of being read again.
- **MoE routing picks which neurons to read.** The router names the active
  experts, and each one is then walked in neuron blocks like any other
  projection: expert `e` owns rows `[e*out, (e+1)*out)`, so `--block-rows`
  splits it the same way. Peak weight memory follows the block, not the
  expert: one GPT-OSS 120B expert matrix is 2,880 MXFP4 rows = 4.41 MB, and
  streaming it 128 rows at a time measured a 0.39 MB arena peak — the same
  output, bit for bit, in 11× less memory. Setting `--block-rows 0` restores
  the older whole-expert reads.

This is block streaming, not literal one-neuron-at-a-time loading. The
streaming arena bounds its raw I/O reservations; resident caches, expanded
weights, activations, and the KV cache also require memory.

**Smaller blocks are not automatically better.** The 0.39 MB peak above is a
memory result from a single-matrix microbenchmark, not a speed
recommendation — in the full model 128-row blocks were the *slowest* setting
tested, because per-block cost (a future, a host→device copy, a kernel
launch, ~456k times a run) outgrew the I/O it saved. Block size trades I/O
stall against per-block overhead, and the measured optimum here is 512 rows.
See [Tuning blocks and caching](#tuning-blocks-and-caching).

## How it works

One decode step, end to end, with the code that does each part:

**1. Placement — `residency/planner.py`, `residency/cache.py`.**
Before the first token, `plan()` assigns every tensor to VRAM, RAM, or disk.
`TieredCache` runs three policies rather than the obvious two: *VRAM-Q* keeps
raw quantized bytes on the GPU and expands them per access (the default, since
a Q4 model is 4.5 bits/weight on disk but 16 as fp16), *VRAM-F* keeps tensors
that are already float, and *RAM* holds quantized bytes on the host. For
GPT-OSS-120B this puts all 2.3 GB of non-expert weights resident with a 100%
hit rate; the 61 GB of experts cannot fit anywhere and stream.

**2. Routing — `moe/router.py`, `model/gpt_oss.py`.**
The router runs first and names which experts this token needs. Only those are
read. GPT-OSS activates 4 of 128, turning a 63.4 GB model into ~1.9 GB of
expert reads per token.

**3. Prefetch planning — `compute/blockffn.py:BlockPrefetcher`.**
Because routing already named every expert the layer will touch, one plan
covers the whole layer: all active experts × gate/up/down, in the order they
will be consumed, chopped into blocks. The prefetcher keeps `depth` blocks in
flight and refills as each retires, so a block finishing in `ffn_gate`
immediately pulls in one belonging to `ffn_down` or the next expert. Windows
the slab cache already holds are skipped rather than re-read.

**4. Streaming — `compute/blockffn.py:stream_linear`.**
Walks a row window of a weight matrix: read a block, multiply, accumulate,
retire it, take the next. `row_start`/`row_end` bound the walk, which is what
makes one expert streamable — its rows are a contiguous window inside the
stacked `(n_expert, out, in)` tensor. The same loop serves disk, RAM and VRAM,
which is why residency stays invisible to the model code.

**5. Compute — `compute/triton_kernels.py`.**
On the decode path a fused dequant-GEMV multiplies straight from the quantized
bytes, so a block's weights are never materialised as floats.

**6. The ceiling — `io/arena.py`.**
Every in-flight block holds a byte reservation. The arena blocks allocation
until one retires, which back-pressures the prefetcher instead of letting it
run away. This is what makes the memory bound hold regardless of model size.

The FFN needs one extra step. `ffn_down` is stored `(n_embd, n_ffn)`, so
slicing it *by neuron* would mean strided column reads — the access pattern
this drive punishes. Instead the FFN runs in two row-contiguous phases: walk
gate/up by neuron to get activations, then walk down by output dimension.
Activations are floats per token — kilobytes, not gigabytes — so materialising
them between phases costs nothing.

## Tuning blocks and caching

The cache planner prioritizes small frequently used tensors and full output
projections over untied input embedding tables, whose decode access only
needs one row. Quantized expert slabs are reused during both prompt prefill
and decode. Warmup in `run_235b.py` uses the same chat formatting as generation
to make expert selection more representative of the actual request.

Use the repeatable benchmark to compare block sizes and expert cache budgets:

```bash
python tests/bench_tuning.py --rows 1024,4096,8192 --repeats 2 --output models/_tuning/blocks.json
python tests/bench_tuning.py --slab-budget 3.5GB --output models/_tuning/cache-3.5GB.json
python tests/test_tuning.py
python tests/bench_qwen35.py --suite all --output models/_bench/qwen35.json
```

Run one GPU workload at a time. The benchmark replays identical tokens and
records decode time, read bytes, and predictions, excluding model loading,
warmup, pinning, and prefill. It does not clear the OS file cache. Short-run
timings and hot-expert cache benefits depend on the prompt; repeat runs before
treating small differences as improvements. A larger expert cache also leaves
less VRAM for ordinary projections, so increasing it can cause extra reads.

Two habits this project learned the hard way, both from measurements that were
individually correct and jointly misleading:

- **Tune against the whole model, not one matrix.** A single-matrix
  microbenchmark ranked 128-row blocks fastest; in the full model they were
  slowest. The microbenchmark measured I/O in isolation and never saw
  per-block overhead accumulate over ~2,600 blocks a token.
- **A short decode overstates sustained throughput.** Every figure below and
  in the GPT-OSS table is a 16-token run immediately after warmup, when the
  pinned experts still match what warmup routed to. A ~450-token generation
  sustained about 63% of the short-run rate. Prefill is far more
  reproducible — under 2% spread across runs, against ~20% for a 16-token
  decode — so prefer it when comparing configurations.

The block and prefetch numbers for GPT-OSS-120B are in
[the GPT-OSS section](#gpt-oss-120b); the table below is the 235B model.

On the RTX 5050 Laptop GPU, the September 11 tuning run measured the following
with the 235B model, a 1 GB I/O budget, 7 GB VRAM budget (including 1.5 GB
reserved for working memory), 2 GB RAM cache, and 8 I/O workers:

| Code | Block rows | Expert cache | Decode tok/s | GB read/token |
|---|---:|---:|---:|---:|
| Original (`0c7f17a`) | 4096 | 3.5 GB | 0.101 | 6.150 |
| Revised | 1024 | 3.5 GB | 0.120 | 5.702 |
| Revised | 4096 | 3.5 GB | 0.114 | 5.705 |
| Revised | 8192 | 3.5 GB | 0.119 | 5.705 |
| Revised, runner defaults | 4096 | 3 GB | 0.124 | 5.899 |

These are single trials of three fixed decode tokens after warmup on
"The capital of France is", not sustained chat benchmarks. All predicted
tokens matched. The selected defaults were about 23% faster than the original
in this test; block-size differences alone were inconclusive. Cache budget
strings use binary GB (GiB); read-volume figures above use decimal GB.
Raw measurements and settings are in [tests/tuning_results.json](https://github.com/hareramray/nurostream/blob/main/tests/tuning_results.json).

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

**Queue depth is a scheduling property, not a parameter.** Raising
`--expert-lookahead` from 4 to 8 changed nothing, because a 2,880-row window
at 1,024 rows per block is only *three* blocks — the pipeline had nothing more
to queue. What mattered was where the pipeline ended: one per projection meant
432 drains per token. A prefetch plan spanning the whole layer took stall from
76.3 s to 0.8 s and prefetch hit rate from 66% to 99.5%. Depth stops helping
above 8.

**Then the bottleneck moves.** With stall gone, per-block cost dominates:
128-row blocks had the *best* stall (17.2 s → 0.6 s) and the *worst* decode,
because 456k reads a run cost more in futures, host→device copies and kernel
launches than they saved in waiting. Optimum is where the two curves cross —
512 rows here. A microbenchmark on one isolated matrix picked 128 and was
wrong about the full model; it measured I/O in isolation and never saw the
per-block overhead accumulate.

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
  compute/   quant.py  triton_kernels.py  ops.py    dequant, fused GEMV
             blockffn.py                            stream_linear, BlockPrefetcher
  moe/       router.py                               selective expert fetch
  vision/    tower.py  prompt.py                     ViT, mRoPE, DeepStack
  model/     qwen3.py  gpt_oss.py                    Qwen3 / Qwen3-VL / MoE
  api.py  cli.py
```

Supported: Qwen3, Qwen3-VL, Qwen3-VL-MoE, GPT-OSS, Gemma 4 E4B (text/chat),
Qwen3.5-4B (text/chat/images, GGUF or safetensors). Quantization: Q4_K, Q5_K,
Q6_K, Q8_0, Q4_0, MXFP4, F32, F16, BF16 — Q4_K, Q6_K and MXFP4 bit-exact
against the reference `gguf` implementation, and fused in Triton for decode.
Weights come from a GGUF file or, for Qwen3.5, straight from a HuggingFace
safetensors checkpoint.

## Qwen3.5-4B

[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) is a hybrid-attention
model: three Gated DeltaNet layers for every one gated full-attention layer,
32 layers over a 2560-wide residual stream. The recurrent layers hold a
fixed-size state instead of a growing KV history, so memory per token stops
growing on three quarters of the layers.

Three weight formats load: the **original safetensors checkpoint** from the
Hugging Face repo, an unquantized **BF16 GGUF**, and quantized GGUF. The format
is detected from the path; no flag is needed.

### From the Hugging Face checkpoint

Point at the repo directory and the original BF16 safetensors (about 9.33 GB,
vision tower included) are read directly — no GGUF conversion step:

```bash
python -m pip install -e ".[chat]" huggingface_hub
python -m huggingface_hub.cli.hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
python -m neurostream info models/Qwen3.5-4B
python run_qwen35.py --model models/Qwen3.5-4B -p "Explain gravity simply."
python run_qwen35.py --model models/Qwen3.5-4B --image photo.jpg -p "What is in this picture?"
```

The checkpoint holds the text model, the vision tower and the tokenizer in one
directory, so `attach_vision()` takes no argument there and no `mmproj` file is
needed. Sharded checkpoints load through `model.safetensors.index.json`; a
single `.safetensors` file or the index itself is also accepted.

Nothing is read at open time beyond one header per shard, so opening the
checkpoint costs the same as opening a GGUF, and weights still stream by neuron
block under `--mem-budget`. The engine speaks llama.cpp's tensor dialect
internally, so the checkpoint is translated at the index: tensors are renamed,
RMSNorm weights gain the `+1` that HF applies at runtime, `A_log` becomes
`-exp(A_log)`, and linear-attention value heads are retiled. Every one of those
transforms is row-local, which is what lets block streaming survive the
translation. The MTP block in the repo is skipped, since it is a speculative
decoding head rather than model depth.

Only float checkpoints load this way (F32/F16/BF16); a quantized checkpoint
should be loaded as GGUF instead.

### Unquantized BF16 GGUF

The BF16 conversion of the text model is about **8.42 GB**:

```bash
python -m pip install -e ".[chat]" huggingface_hub
python -m huggingface_hub.cli.hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-BF16.gguf --local-dir models/qwen35
python -m neurostream run models/qwen35/Qwen3.5-4B-BF16.gguf --chat --no-thinking -p "Explain gravity simply." --mem-budget 512MB --vram-budget 6GB
```

The download command invokes the Hugging Face CLI through Python, so it also
works on Windows when `hf` is not on PATH. Add `--dry-run` to check the download
without fetching the weights.

BF16 weights remain floating point. Weights that do not fit the cache stream
from disk, so the entire 8.42 GB file does not need to fit in VRAM. CPU
inference expands BF16 blocks to float32 for computation; CUDA defaults to
bfloat16. The recurrent state and the delta-rule update stay in float32 at both
precisions, because a decayed running state compounds rounding error in a way
attention does not.

### Quantized

Any quantized GGUF from the same repository loads the same way, for example
`Qwen3.5-4B-Q4_K_M.gguf` (about 2.74 GB):

```bash
python -m huggingface_hub.cli.hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models/qwen35
python -m neurostream info models/qwen35/Qwen3.5-4B-Q4_K_M.gguf
python -m neurostream run models/qwen35/Qwen3.5-4B-Q4_K_M.gguf --chat --no-thinking -p "Explain gravity simply." --mem-budget 512MB --vram-budget 6GB
```

### Python API and options

Use the dedicated launcher to run the downloaded BF16 model with the default
memory settings, or pass `--model` to select a quantized file:

```bash
python run_qwen35.py -p "Explain gravity simply."
python run_qwen35.py --chat
python run_qwen35.py --model models/qwen35/Qwen3.5-4B-Q4_K_M.gguf -p "Hello"
python run_qwen35.py --device cpu --ram-budget 2GB -p "Hello"
```

The launcher applies chat formatting automatically and disables thinking by
default. `--chat` keeps conversation history; use `/reset` to clear it and
`/quit` to exit. Add `--thinking -n 1024` to enable thinking output. Run
`python run_qwen35.py --help` for memory and sampling options.

Use `--device cpu` with `--vram-budget 0` to stream on CPU. Both files use the
same Python API; for example, with unquantized weights:

```python
from neurostream import NeuroStream

with NeuroStream.load("models/qwen35/Qwen3.5-4B-BF16.gguf",
                      mem_budget="512MB", vram_budget="6GB") as ns:
    for piece in ns.chat("Explain gravity simply.", enable_thinking=False, max_tokens=128):
        print(piece, end="", flush=True)
```

`--chat` applies the GGUF's chat template; without it, `run` completes a raw
prompt. `--thinking` enables the template's thinking mode; allow more output
tokens for that mode. Generation stops at `<|im_end|>` and buffers split UTF-8
bytes correctly.

The native text path implements the gated delta rule with its causal depthwise
convolution, L2-normalized queries and keys, per-head decay, gated attention
with a per-head output gate, partial (quarter-width) RoPE, and the interval-4
recurrent/full layer pattern. Weights are read with llama.cpp's conventions:
RMSNorm weights arrive pre-shifted by one, `A_log` arrives already negated and
exponentiated, and linear-attention value heads arrive tiled by key head rather
than grouped. A Qwen3.5 GGUF also reports one more block than the model has
layers, because the converter counts the multi-token-prediction block; the
extra block is metadata, not depth, and is excluded.

The sparse MoE variants of the family are not supported; loading one reports
an unsupported architecture rather than producing wrong output.

### Images

The vision tower ships as a separate `mmproj` file (about **0.68 GB** in BF16).
Download it alongside the text model and pass an image:

```bash
python -m huggingface_hub.cli.hf download unsloth/Qwen3.5-4B-GGUF mmproj-BF16.gguf --local-dir models/qwen35
python run_qwen35.py --image photo.jpg -p "What is in this picture?"
python -m neurostream run models/qwen35/Qwen3.5-4B-BF16.gguf --image photo.jpg --mmproj models/qwen35/mmproj-BF16.gguf -p "What is in this picture?" --no-thinking
```

Both entry points fall back to an `mmproj-*.gguf` sitting beside the model, so
`--mmproj` is only needed when it lives elsewhere. `--max-patches` caps how many
tokens an image may become, so a large photo cannot fill the context on its own.

```python
with NeuroStream.load("models/qwen35/Qwen3.5-4B-BF16.gguf", vram_budget="6GB") as ns:
    ns.attach_vision("models/qwen35/mmproj-BF16.gguf")
    for piece in ns.generate_vl("photo.jpg", "What is in this picture?",
                                enable_thinking=False, max_tokens=128):
        print(piece, end="", flush=True)
```

The whole image is encoded in one batched pass, so a 400-token image costs
about as much weight traffic as a single text token — the asymmetry that makes
vision affordable on a streaming engine.

Three things have to agree for an image to mean anything, and each is checked
against Transformers rather than assumed. The tower resamples its learned 48x48
position grid **bilinearly with align_corners=True**, where Qwen3-VL uses
bicubic — corner alignment moves every patch's sample point. Its patch merger
uses exact GELU while the encoder MLP uses the tanh approximation. And the
merged patches enter the text model with real **interleaved M-RoPE** positions:
each image token carries its own (frame, row, column), and Qwen3.5 scatters the
three axes across neighbouring frequency dims instead of giving each axis one
contiguous band, so text after an image resumes past the wider of the two image
axes. Only the full-attention layers read those positions; the recurrent layers
have no rotary embedding at all.

llama.cpp converts the Qwen3-VL and Qwen3.5 towers with the same code, so the
`mmproj` tensor layout is shared, including the Conv3d patch embedding split
into two Conv2d kernels along the temporal axis. A still image is fed as two
identical frames, which is what that split expects.

Video input is not supported: Qwen3.5 separates frames with timestamp tokens
and gives each frame its own temporal position, and neither is implemented.
The 4B ships an empty `deepstack_visual_indexes`, so there are no DeepStack
taps to inject; a tower that produced them is rejected rather than ignored.

Validation uses tiny models against Transformers for prefill, incremental and
chunked decode, recurrent state and convolution history carried across chunks,
tied/untied output weights, Q8_0/Q4_0 projections, BF16 storage, RAM/VRAM cache
loading, MTP block counting, and CPU/CUDA execution. The image tests cover the
tower against the reference vision model, the interleaved M-RoPE tables, the
position ids against `get_rope_index`, and multimodal prefill and decode. Full
4B weights are not needed by any of them:

```bash
python -m pip install gguf "transformers>=5.17" jinja2 pillow safetensors
python tests/test_qwen35.py
python tests/test_qwen35_vision.py
python tests/test_qwen35_safetensors.py
```

The safetensors tests write their checkpoint with the reference `safetensors`
writer, so the index parser is checked against an independent producer, and
they assert the three translation fixups directly — a checkpoint that skipped
any of them would load happily and answer wrongly.

### Measured: Qwen3.5-4B on an 8 GB card

```bash
python tests/bench_qwen35.py --suite vram --repeats 3 --output models/_bench/vram.json
python tests/bench_qwen35.py --suite all --output models/_bench/all.json
```

Measured on an RTX 5050 Laptop (8.5 GB) with 16.9 GB host RAM, against the
9.33 GB safetensors checkpoint in BF16 — a model that does not fit the card.
Every configuration decodes the same teacher-forced tokens; figures are
medians over the stated repeats, with the sweep order reversed on alternate
passes so a warming page cache cannot masquerade as a trend.

**VRAM budget vs latency** (n=3, `reserve-vram 1.5GB`, `block-rows 1024`):

| `--vram-budget` | resident | read/token | ms/token | tok/s | cache hits |
|---|---|---|---|---|---|
| 0 | 0.00 GB | 8.42 GB | 7126 | 0.137 | 0% |
| 1 GB | 0.00 GB | 8.42 GB | 7411 | 0.133 | 0% |
| 2 GB | 0.54 GB | 7.88 GB | 6484 | 0.154 | 16.3% |
| 3 GB | 1.61 GB | 6.81 GB | 4340 | 0.207 | 30.1% |
| 4 GB | 2.68 GB | 5.73 GB | 4361 | 0.229 | 39.9% |
| 5 GB | 3.76 GB | 4.66 GB | 3354 | 0.286 | 51.2% |
| 6 GB | 4.83 GB | 3.58 GB | 2548 | 0.379 | 60.6% |

Decode reads every non-resident weight exactly once per token: `read/token`
tracks `8.42 GB − resident` to within 0.01 GB at every budget. Latency follows
it, because at ~1 GB/s effective this workload is entirely I/O. The budget is
what the planner may *spend*, not what it places — `--reserve-vram` is taken
off the top, which is why 1 GB buys nothing. **All 21 runs produced identical
output**, which is the memory-ceiling claim stated as a measurement.

**Host RAM as the second tier** (n=2, VRAM held at 4 GB / 2.68 GB resident):

| `--ram-budget` | RAM resident | read/token | ms/token | tok/s | cache hits |
|---|---|---|---|---|---|
| 0 | 0.00 GB | 5.73 GB | 4189 | 0.241 | 39.9% |
| 1 GB | 1.07 GB | 4.66 GB | 3703 | 0.262 | 51.2% |
| 2 GB | 2.15 GB | 3.58 GB | 3289 | 0.299 | 61.0% |
| 4 GB | 4.29 GB | 1.44 GB | 2319 | 0.419 | 82.7% |
| 6 GB | 5.73 GB | **0.00 GB** | 1292 | **0.775** | 100% |

This is the more useful knob on this machine. 2.68 GB of VRAM plus 5.73 GB of
RAM covers all 8.4 GB of weights, the drive drops out of the loop entirely,
and throughput reaches **5.7× the no-residency baseline** — roughly twice what
the best VRAM-only configuration managed, simply because there is more RAM
available than spare VRAM.

**Context scaling.** Only 8 of 32 layers keep a KV history; the other 24 are
recurrent and hold fixed-size state. `dense` is what the same depth would cost
if every layer kept a history:

| context | prefill | prefill tok/s | decode ms | KV + state | dense equivalent |
|---|---|---|---|---|---|
| 128 | 7.3 s | 17.6 | 4685 | 55.7 MB | 16.8 MB |
| 512 | 6.5 s | 79.4 | 4757 | 68.3 MB | 67.1 MB |
| 1024 | 7.7 s | 133.8 | 3664 | 85.1 MB | 134.2 MB |
| 2048 | 11.1 s | 185.1 | 3586 | 118.7 MB | 268.4 MB |
| 4096 | 18.6 s | 220.1 | 3479 | 185.8 MB | 536.9 MB |

The hybrid design costs memory before it saves any. The recurrent state is a
**fixed ~51.7 MB** regardless of context, so at 128 tokens Qwen3.5 uses 3.3×
what a dense model of the same shape would; it breaks even near **512 tokens**
and is at 0.35× by 4096. Growth past the constant is **0.033 MB/token against
0.131 dense — exactly the 8/32 ratio**. Decode latency is flat across a 32×
context range, because it is set by weight streaming rather than by attention.

Prefill tok/s rises with length for the same reason: prefill reads the weights
once for the whole chunk, so a longer prompt amortizes the same I/O.

**Image prefill** carries that to its conclusion — a 1024-token image prefills
*faster* than a 256-token one, because the cost is one weight pass, not the
token count:

| `--max-patches` | image tokens | grid | tower encode | prompt | prefill |
|---|---|---|---|---|---|
| 256 | 256 | 16×16 | 0.57 s | 278 tok | 9.45 s |
| 1024 | 1024 | 32×32 | 1.40 s | 1046 tok | 7.62 s |

**Format and block size.** safetensors and BF16 GGUF are equivalent: identical
5.73 GB/token, and 3712 vs 4043 ms/token (n=3) — a gap inside run-to-run
variance, so the checkpoint translation costs nothing measurable. Block size
has a shallow optimum at **1024 rows** (4150 / 3644 / 4244 ms/token for
256 / 1024 / 4096), matching the finding elsewhere in this README that smaller
blocks are not automatically better.

Caveats: the OS file cache is not flushed between runs, throughput at these
rates is dominated by a single drive, and the absolute tok/s carries the same
PyTorch-level per-call overhead described under
[the throughput targets](#where-the-throughput-targets-were-missed-and-why).
Ratios and read-per-token figures are far more reproducible than wall clock.

## Gemma 4 E4B

[Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) is Google's
effective-4B model. Its per-layer embedding tables add parameters beyond the
effective count. The engine reads only the requested rows of those tables.

Both unquantized **BF16 GGUF** and quantized GGUF weights are supported.
The storage format is detected from the file; no precision flag is needed.
GGUF is a container and can hold floating-point weights without quantization.

### Unquantized BF16

The [llama.cpp BF16 conversion](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
of Gemma 4 E4B IT is about **15.1 GB**. Download only the text model file:

```bash
python -m pip install -e ".[chat]" huggingface_hub
python -m huggingface_hub.cli.hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-BF16.gguf --local-dir models/gemma4
python -m neurostream run models/gemma4/gemma-4-E4B-it-BF16.gguf --chat --no-thinking -p "Explain gravity simply." --mem-budget 512MB --vram-budget 6GB
```

The download command invokes the Hugging Face CLI through Python, so it also
works on Windows when `hf` is not on PATH. Add `--dry-run` to check the download
without fetching the weights.

BF16 weights remain floating point. Weights that do not fit the cache stream
from disk, so the entire 15.1 GB file does not need to fit in VRAM. Cache loading
also reads large tensors in bounded chunks: the 1.34 GB token embedding can be
loaded into VRAM with a 512 MB I/O budget. CPU inference expands BF16 blocks to
float32 for computation; CUDA defaults to bfloat16.

### Quantized Q4_0

Download Google's [Q4_0 GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
(about 5.15 GB) and run a chat prompt:

```bash
python -m pip install -e ".[chat]" huggingface_hub
python -m huggingface_hub.cli.hf download google/gemma-4-E4B-it-qat-q4_0-gguf gemma-4-E4B_q4_0-it.gguf --local-dir models/gemma4
python -m neurostream info models/gemma4/gemma-4-E4B_q4_0-it.gguf
python -m neurostream run models/gemma4/gemma-4-E4B_q4_0-it.gguf --chat --no-thinking -p "Explain gravity simply." --mem-budget 512MB --vram-budget 6GB
```

### Python API and options

Use the dedicated launcher to run the downloaded BF16 model with the default
memory settings, or pass `--model` to select a quantized file:

```bash
python run_gemmamodel.py -p "Explain gravity simply."
python run_gemmamodel.py --chat
python run_gemmamodel.py --model models/gemma4/gemma-4-E4B_q4_0-it.gguf -p "Hello"
python run_gemmamodel.py --device cpu --ram-budget 2GB -p "Hello"
```

The launcher applies chat formatting automatically and disables thinking by
default. `--chat` keeps conversation history; use `/reset` to clear it and
`/quit` to exit. Add `--thinking -n 1024` to enable thinking output. Run
`python run_gemmamodel.py --help` for memory and sampling options.

Use `--device cpu` with `--vram-budget 0` to stream on CPU. Both files use the
same Python API; for example, with unquantized weights:

```python
from neurostream import NeuroStream

with NeuroStream.load("models/gemma4/gemma-4-E4B-it-BF16.gguf",
                      mem_budget="512MB", vram_budget="6GB") as ns:
    for piece in ns.chat("Explain gravity simply.", enable_thinking=False, max_tokens=128):
        print(piece, end="", flush=True)
```

`--chat` applies the GGUF's chat template; without it, `run` completes a raw
prompt. `--thinking` enables the template's thinking mode; allow more output
tokens for that mode. Generation stops at Gemma's end-of-turn token and buffers
split UTF-8 bytes correctly. CUDA defaults to bfloat16; CPU uses float32.

The native text path implements local/global attention with different head
dimensions, proportional RoPE, shared KV layers, per-layer embeddings, GELU
gating, and logit softcapping. Sliding KV history is cropped after shared layers
consume it. The I/O budget bounds streaming buffers; activations, tokenizer,
resident weights, and global KV history need additional memory. The advertised
128K context is a model limit, not a guarantee it fits the available memory.

Gemma image/audio input and the 26B/31B architectures are not supported by this
implementation. The Qwen vision projector cannot be attached to Gemma.

Validation uses tiny models against Transformers for prefill, incremental and
chunked decode, shared caches, tied/untied output weights, Q8_0/Q4_0 projections,
BF16 storage with tensors larger than the I/O budget, RAM/VRAM cache loading,
and CPU/CUDA execution. Full E4B weights are not needed by these tests:

```bash
python -m pip install gguf "transformers>=5.5" jinja2
python tests/test_gemma4.py
```

For full-vocabulary tokenizer parity, set `NEUROSTREAM_GEMMA4_TOKENIZER_DIR` to a
directory with Google's `tokenizer.json` and `chat_template.jinja` before running
the tests. Full-model generation and throughput have not been benchmarked.

## GPT-OSS-120B

GPT-OSS adds biased projections and experts, YaRN, attention sinks,
alternating sliding/full attention, and a clamped SwiGLU. GPT-OSS-120B is
36 layers of 128 experts with 4 active, 63.4 GB on disk of which 96.4% is
expert weights, so a token reads about 4.2 GB — 6.6% of the file, of which
~1.9 GB is experts and the rest is served from VRAM:

```bash
python download_gpt_oss.py                  # 63 GB, resumable, SHA-256 checked
python run_gpt_oss.py -p "Hello"
python run_gpt_oss.py -p "Hello" --reasoning low --final-only
python run_gpt_oss.py --chat --slab-budget 4GB
```

GPT-OSS replies in **harmony** format: an `analysis` chain of thought first,
then the `final` answer. Two consequences for the runner:

- `--max-tokens` has to cover both. The default is 512; a low cap returns
  reasoning and no answer, and the runner says so explicitly rather than
  printing a truncated thought and stopping. `--reasoning low` shortens the
  thinking, which is usually the cheaper way to reach the answer.
- `--final-only` hides the chain of thought. `HarmonyStream` in
  `run_gpt_oss.py` splits the channels out of the token stream, including
  when a channel marker is split across two tokens.

Measured on the RTX 5050 Laptop (8.5 GB VRAM, 15.7 GB RAM, DRAM-less NVMe),
warmed and pinned at `--slab-budget 4GB`:

| Setting | I/O stall | Prefetch hit | Decode |
|---|---:|---:|---:|
| 1024 rows, per-projection pipeline | 76.3 s | 66.3% | 0.44 tok/s |
| 256 rows, shared prefetch | 0.6 s | 99.8% | 0.51 tok/s |
| **512 rows, shared prefetch** | **0.8 s** | **99.5%** | **0.73 tok/s** |

The shared layer-wide prefetch plan is what removed the stall: a
per-projection pipeline drains 12 times per layer — 432 times per token — and
each drain costs an unhidden disk latency, which held effective queue depth at
3 and throughput at the drive's queue-depth-1 rate.

**These decode figures are a warm best case.** They come from 16-token runs
taken immediately after warmup, while the pinned experts still match the
routing warmup exercised. A ~450-token generation on the same prompt sustained
about **0.46 tok/s**, roughly 63% of the short-run number, as routing drifts
off the pinned set. Treat the rankings as sound and the absolute rates as
optimistic.

## Tests

```bash
python tests/test_p0_logits.py      # vs HuggingFace
python tests/test_p1_streaming.py   # budget independence
python tests/test_p2_neuron.py      # O(block) peak memory
python tests/test_p6_sharded.py     # splits a model into 3 real shards
python tests/test_p3_p4_p5.py       # CUDA, MoE, vision (skips absent models)
python tests/test_gpt_oss.py        # vs Transformers; neuron-block experts
python tests/test_gemma4.py         # vs Transformers; Gemma text/chat and shared KV
python tests/test_qwen35.py         # vs Transformers; Qwen3.5 hybrid recurrent/full layers
python tests/test_qwen35_vision.py  # vs Transformers; Qwen3.5 tower, M-RoPE, image splice
python tests/test_qwen35_safetensors.py  # vs Transformers; HF checkpoint read without GGUF
python tests/test_tuning.py         # block sizing, slab reuse, pinning
```

The tokenizer is validated against HuggingFace on 314 cases including 300
fuzzed strings: zero encode mismatches, zero round-trip failures.

## Bugs worth knowing about

Found by testing, and mostly the kind that produce plausible-looking garbage
or plausible-looking *numbers* rather than an exception:

- **`id()`-keyed memoization.** A row-geometry cache keyed by `id(tensor_info)`
  silently hands one model's strides to the next model loaded in the same
  process, because CPython reuses addresses after GC. Every weight then reads
  from the wrong offset. Now memoized on the instance.
- **Prefetch vs block-streaming deadlock.** Whole-tensor prefetch reserves
  arena bytes released only by `raw_bytes()`, but the block path consumes via
  `fetch_rows()` and never calls it, so the arena starves. Whole-tensor
  prefetch is now disabled whenever block streaming is active.
- **The pipeline holds `depth + 1` blocks, not `depth`.** The block currently
  being multiplied still owns its arena lease — it is only released after its
  output exists. Sizing the budget for `depth` blocks deadlocked on the refill
  that follows the first wait. Caught by `test_tuning.py`, which runs
  `stream_linear` against deliberately tiny budgets.
- **A pinned expert is not a resident tensor.** Slabs are cached per row
  range, so `is_resident(name)` is false for the stacked expert tensor even
  when the planner pinned the exact rows being asked for. The async path
  checked only residency and would have re-read pinned experts from disk —
  silently correct, and silently pointless. `TieredCache._slab_for` now
  resolves sub-ranges against enclosing slabs.
- **Counters that were never incremented.** `RouterStats.expert_hits` was
  defined and printed as `(N cached)` but nothing ever set it, so it read zero
  regardless of caching and briefly sent tuning in the wrong direction.
  `MIN_READ` and `rows_for_min_read` are likewise documented as widening small
  reads but are called from nowhere — so there is currently **no floor on read
  size**, which is part of why 128-row blocks were free to fall off the
  drive's efficiency cliff. A dead counter is worse than a missing one.
