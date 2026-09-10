"""Command line entry point.

    python -m neurostream run model.gguf -p "Hello" --mem-budget 2GB
    python -m neurostream run model.gguf --image cat.png -p "What is this?"
    python -m neurostream info model.gguf
    python -m neurostream bench model.gguf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="path to a .gguf (any shard of a split set)")
    p.add_argument("--mem-budget", default="1GB",
                   help="hard ceiling on streaming buffers (default 1GB)")
    p.add_argument("--vram-budget", default="0",
                   help="weights to keep resident in VRAM, e.g. 6GB")
    p.add_argument("--ram-budget", default="0",
                   help="weights to keep resident in host RAM")
    p.add_argument("--slab-budget", default="0",
                   help="VRAM reserved for hot MoE expert slabs, e.g. 4GB")
    p.add_argument("--reserve-vram", default="1.5GB",
                   help="VRAM left free for activations and KV cache")
    p.add_argument("--block-rows", type=int, default=4096,
                   help="maximum rows per streamed block; shrinks to fit the I/O budget")
    p.add_argument("--workers", type=int, default=16,
                   help="NVMe queue depth")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])


def cmd_info(a) -> int:
    from .format.gguf import GGUFFile
    from .format.sharded import ShardedGGUF, discover_shards
    from .model.qwen3 import Qwen3Config

    shards = discover_shards(a.model)
    g = ShardedGGUF(a.model) if len(shards) > 1 else GGUFFile(a.model)
    print(g)
    if len(shards) > 1:
        for s in shards:
            print(f"  shard {s.name}  {s.stat().st_size / 1e9:.1f}GB")
    if g.arch() == 'gpt-oss':
        from .model.gpt_oss import GptOssConfig
        cfg = GptOssConfig.from_gguf(g)
    else:
        cfg = Qwen3Config.from_gguf(g)
    print(f"  {cfg}")

    from collections import Counter

    print("  dtypes:", dict(Counter(t.dtype.name for t in g.tensors.values())))
    if cfg.is_moe:
        active = cfg.n_expert_used / cfg.n_expert
        expert_bytes = sum(
            t.nbytes for n, t in g.tensors.items() if "_exps" in n
        )
        other = g.total_tensor_bytes() - expert_bytes
        per_token = other + expert_bytes * active
        print(f"  MoE: {cfg.n_expert} experts, {cfg.n_expert_used} active")
        print(f"  bytes touched per token: {per_token / 1e9:.2f}GB "
              f"of {g.total_tensor_bytes() / 1e9:.1f}GB "
              f"({per_token / g.total_tensor_bytes():.1%})")
    return 0


def _load(a, verbose=True):
    from .api import NeuroStream

    return NeuroStream.load(
        a.model, mem_budget=a.mem_budget, vram_budget=a.vram_budget,
        ram_budget=a.ram_budget, slab_budget=a.slab_budget,
        reserve_vram=a.reserve_vram, block_rows=a.block_rows,
        n_workers=a.workers, device=a.device, verbose=verbose,
    )


def cmd_run(a) -> int:
    t0 = time.perf_counter()
    ns = _load(a)
    print(f"  ready in {time.perf_counter() - t0:.1f}s\n", flush=True)

    if a.warmup and ns.cfg.is_moe:
        print("  warming up to learn expert routing ...", flush=True)
        ns.warmup(prompt=a.prompt, tokens=a.warmup)
        ns.pin_hot_experts(verbose=True)
        print(flush=True)

    if a.image:
        ns.attach_vision(a.mmproj or _guess_mmproj(a.model))
        stream = ns.generate_vl(
            a.image, a.prompt, max_tokens=a.max_tokens,
            temperature=a.temperature, max_patches=a.max_patches,
        )
    else:
        stream = ns.generate(
            a.prompt, max_tokens=a.max_tokens, temperature=a.temperature
        )

    for piece in stream:
        sys.stdout.write(piece)
        sys.stdout.flush()
    print("\n")
    print(ns.report())
    ns.close()
    return 0


def _guess_mmproj(model_path) -> str:
    d = Path(model_path).parent
    for cand in ("mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"):
        if (d / cand).exists():
            return str(d / cand)
        if (d.parent / cand).exists():
            return str(d.parent / cand)
    raise SystemExit("no mmproj found next to the model; pass --mmproj")


def cmd_bench(a) -> int:
    """Prove the memory ceiling: same prompt, several budgets, same output."""
    from .api import NeuroStream

    print(f"{'budget':>10} {'peak':>9} {'tok/s':>8} {'GB read':>9}  output")
    print("-" * 62)
    first = None
    ok = True
    for budget in a.budgets.split(","):
        ns = NeuroStream.load(
            a.model, mem_budget=budget.strip(), vram_budget=a.vram_budget,
            ram_budget=a.ram_budget, block_rows=a.block_rows,
            n_workers=a.workers, device=a.device,
        )
        out = "".join(ns.generate(a.prompt, max_tokens=a.max_tokens,
                                  temperature=0.0))
        first = out if first is None else first
        same = out == first
        ok &= same
        print(f"{budget.strip():>10} "
              f"{ns.arena.stats.peak_bytes / 1e6:>7.0f}MB "
              f"{ns.stats.tokens_per_second:>8.2f} "
              f"{ns.source.stats.bytes_read / 1e9:>8.2f} "
              f" {'same' if same else 'DIFFERENT'}")
        ns.close()
    print(f"\n  identical output at every budget: {ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="neurostream",
        description="Run transformers larger than memory by streaming "
                    "weights from disk one neuron-block at a time.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="inspect a model without loading weights")
    p.add_argument("model")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("run", help="generate text, optionally about an image")
    _add_common(p)
    p.add_argument("-p", "--prompt", default="Hello, ")
    p.add_argument("-n", "--max-tokens", type=int, default=128)
    p.add_argument("-t", "--temperature", type=float, default=0.0)
    p.add_argument("--image", default=None)
    p.add_argument("--mmproj", default=None)
    p.add_argument("--max-patches", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=0,
                   help="tokens of warmup before pinning hot MoE experts")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("bench", help="verify output is budget-independent")
    _add_common(p)
    p.add_argument("-p", "--prompt", default="The capital of France is")
    p.add_argument("-n", "--max-tokens", type=int, default=16)
    p.add_argument("--budgets", default="64MB,256MB,1GB")
    p.set_defaults(fn=cmd_bench)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
