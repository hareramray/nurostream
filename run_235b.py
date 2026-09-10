#!/usr/bin/env python
"""Run Qwen3-VL-235B-A22B on this laptop.

142 GB of weights, 8 GB of VRAM. Expect roughly 8-10 seconds per token, so
tokens are streamed as they arrive rather than made you wait for the whole
reply.

    python run_235b.py -p "Explain how an SSD wears out."
    python run_235b.py --image photo.jpg -p "What is in this picture?"
    python run_235b.py --chat
    python run_235b.py -p "Hi" --no-warmup      # skip the 90s pin step

The defaults use the local block/cache tuning results. Two settings matter and
are easy to get wrong:

  --workers 8     Queue depth. Measured on this drive with 5.85 MB random
                  reads: 1 worker 0.62 GB/s, 8 workers 2.00, 24 workers 1.57,
                  48 workers 1.37. More threads is not more throughput.
  --slab-budget   VRAM held for hot MoE expert slabs. Warmup measures which
                  experts this prompt actually routes to and pins those;
                  without it, the reserved expert VRAM sits idle. The latest
                  short fixed-token comparison selected 3 GB (0.124 tok/s)
                  over 3.5 GB (0.114 tok/s), both with 4096-row blocks.
                  See tests/tuning_results.json. Note the pinning is tuned to the
                  warmup prompt; a very different follow-up prompt keeps less
                  of the benefit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "models" / "gpt-oss-120b-GGUF"


def find_model(explicit: str | None) -> tuple[Path, Path | None]:
    """Locate shard 1 and the vision projector."""
    if explicit:
        model = Path(explicit)
        if not model.exists():
            raise SystemExit(f"model not found: {model}")
    else:
        matches = sorted(DEFAULT_DIR.rglob("*-00001-of-*.gguf"))
        if not matches:
            raise SystemExit(
                f"no shard 1 found under {DEFAULT_DIR}\n"
                f"pass the path explicitly with --model"
            )
        model = matches[0]

    mmproj = None
    for cand in ("mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"):
        for base in (model.parent, model.parent.parent):
            if (base / cand).exists():
                mmproj = base / cand
                break
        if mmproj:
            break
    return model, mmproj


def human_eta(tokens: int, tps: float) -> str:
    if tps <= 0:
        return "unknown"
    secs = tokens / tps
    return f"{secs / 60:.1f} min" if secs > 90 else f"{secs:.0f}s"


def stream(gen, label: str = "") -> str:
    """Print tokens as they arrive, with a live rate in the margin."""
    out = []
    t0 = time.perf_counter()
    if label:
        print(label, end="", flush=True)
    for piece in gen:
        out.append(piece)
        sys.stdout.write(piece)
        sys.stdout.flush()
    dt = time.perf_counter() - t0
    n = len(out)
    # Wall clock from the call, so this includes prefill. The decode-only
    # rate is in the report printed at the end.
    print(f"\n\n  [{n} tokens in {dt:.0f}s end-to-end "
          f"= {n / max(dt, 1e-9):.3f} tok/s incl. prefill]")
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Qwen3-VL-235B-A22B (142 GB) on an 8 GB GPU."
    )
    ap.add_argument("--model", default=None,
                    help="path to shard 1; auto-detected by default")
    ap.add_argument("--mmproj", default=None, help="vision projector gguf")
    ap.add_argument("-p", "--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-tokens", type=int, default=32)
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("--image", default=None, help="image to ask about")
    ap.add_argument("--max-patches", type=int, default=1024)
    ap.add_argument("--chat", action="store_true",
                    help="interactive loop; keeps the model resident")
    ap.add_argument("--raw", action="store_true",
                    help="feed the prompt verbatim, no chat template")

    ap.add_argument("--mem-budget", default="1GB")
    ap.add_argument("--vram-budget", default="7GB")
    ap.add_argument("--ram-budget", default="2GB")
    ap.add_argument("--slab-budget", default="3GB",
                    help="VRAM for hot expert weights (default 3GB)")
    ap.add_argument("--reserve-vram", default="1.5GB")
    ap.add_argument("--block-rows", type=int, default=4096,
                    help="maximum rows per streamed projection block; "
                         "shrinks to fit two I/O buffers")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip warmup+pin for faster startup; uncached "
                         "experts require more disk reads")
    ap.add_argument("--warmup-tokens", type=int, default=3)
    a = ap.parse_args(argv)

    import torch

    from neurostream.api import NeuroStream
    from neurostream.format.sharded import discover_shards

    model, mmproj = find_model(a.model)
    if a.mmproj:
        mmproj = Path(a.mmproj)

    shards = discover_shards(model)
    total = sum(s.stat().st_size for s in shards)
    print(f"model   : {model.name}")
    print(f"          {len(shards)} shard(s), {total / 1e9:.1f} GB on disk")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu     : {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print("gpu     : none (CPU only - this will be extremely slow)")

    t0 = time.perf_counter()
    ns = NeuroStream.load(
        str(model),
        mem_budget=a.mem_budget, vram_budget=a.vram_budget,
        ram_budget=a.ram_budget, slab_budget=a.slab_budget,
        reserve_vram=a.reserve_vram, block_rows=a.block_rows,
        n_workers=a.workers,
    )
    print(f"loaded  : {time.perf_counter() - t0:.1f}s   "
          f"{ns.placement.summary()}")

    if ns.cfg.is_moe:
        exp_bytes = sum(
            t.nbytes for n, t in ns.gguf.tensors.items() if "_exps" in n
        )
        other = ns.gguf.total_tensor_bytes() - exp_bytes
        per_tok = other + exp_bytes * (ns.cfg.n_expert_used / ns.cfg.n_expert)
        print(f"moe     : {ns.cfg.n_expert} experts, "
              f"{ns.cfg.n_expert_used} active -> "
              f"~{per_tok / 1e9:.1f} GB read per token "
              f"({per_tok / ns.gguf.total_tensor_bytes():.1%} of the model)")

    if not a.no_warmup and ns.cfg.is_moe:
        print(f"warmup  : {a.warmup_tokens} tokens to learn expert routing "
              f"(~90s) ...", flush=True)
        t0 = time.perf_counter()
        warmup_prompt = a.prompt if a.raw else ns.tokenizer.apply_chat_template(
            [{"role": "user", "content": a.prompt}]
        )
        ns.warmup(prompt=warmup_prompt, tokens=a.warmup_tokens)
        n = ns.pin_hot_experts()
        print(f"          pinned {n} experts "
              f"({ns.cache.slab_bytes / 1e9:.2f} GB) in "
              f"{time.perf_counter() - t0:.0f}s")

    if a.image:
        if mmproj is None:
            raise SystemExit("no mmproj found; pass --mmproj")
        print(f"vision  : {Path(a.image).name} via {mmproj.name}", flush=True)
        ns.attach_vision(str(mmproj))

    try:
        if a.chat:
            print("\ninteractive - blank line or Ctrl-C to quit\n")
            while True:
                try:
                    q = input("you > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    break
                stream(
                    ns.generate(
                        ns.tokenizer.apply_chat_template(
                            [{"role": "user", "content": q}]
                        ),
                        max_tokens=a.max_tokens, temperature=a.temperature,
                    ),
                    label="\n235b > ",
                )
        else:
            rate = 0.048 if a.no_warmup else 0.12
            est = human_eta(a.max_tokens, rate)
            print(f"\ngenerating up to {a.max_tokens} tokens "
                  f"(~{est} at {rate} tok/s, plus ~40-80s prefill)\n")
            if a.image:
                gen = ns.generate_vl(
                    a.image, a.prompt, max_tokens=a.max_tokens,
                    temperature=a.temperature, max_patches=a.max_patches,
                )
                print(f"> [{Path(a.image).name}] {a.prompt}\n")
            else:
                text = a.prompt if a.raw else ns.tokenizer.apply_chat_template(
                    [{"role": "user", "content": a.prompt}]
                )
                gen = ns.generate(text, max_tokens=a.max_tokens,
                                  temperature=a.temperature)
                print(f"> {a.prompt}\n")
            stream(gen)
            print(ns.report())
    except KeyboardInterrupt:
        print("\n\ninterrupted")
    finally:
        ns.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
