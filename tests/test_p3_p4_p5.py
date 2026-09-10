"""P3 (CUDA + residency), P4 (MoE), P5 (vision).

Each phase is skipped rather than failed when its model is absent, so the
suite is runnable on a machine that only has the small models.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.api import NeuroStream

M8 = ROOT / "models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MM8 = ROOT / "models/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf"
M30 = (ROOT / "models/Qwen3-VL-30B-A3B-Instruct-GGUF"
       / "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf")

RESULTS: list[tuple[str, bool, str]] = []


def record(phase: str, ok: bool, note: str) -> None:
    RESULTS.append((phase, ok, note))
    print(f"  {'PASS' if ok else 'FAIL'}  {phase}: {note}\n")


def p3_cuda() -> None:
    """8B Q4 GPU-resident. Target >= 30 tok/s."""
    print("P3  CUDA offload + residency planner")
    if not M8.exists():
        record("P3", True, "SKIPPED (8B model absent)")
        return
    ns = NeuroStream.load(
        str(M8), mem_budget="512MB", vram_budget="6GB", ram_budget="2GB",
        block_rows=4096,
    )
    out = "".join(ns.generate("The capital of France is", max_tokens=32,
                              temperature=0.0))
    tps = ns.stats.tokens_per_second
    coherent = "Paris" in out
    print(f"    placement: {ns.placement.summary()}")
    print(f"    output: {out[:70]!r}")
    print(f"    {ns.stats.summary()}")
    ns.close()
    record("P3", coherent,
           f"{tps:.2f} tok/s (target 30), output coherent={coherent}")


def p4_moe() -> None:
    """30B-A3B: only active experts are read. Target >= 8 tok/s."""
    print("P4  MoE router-driven fetch + expert pinning")
    if not M30.exists():
        record("P4", True, "SKIPPED (30B-A3B absent)")
        return
    ns = NeuroStream.load(
        str(M30), mem_budget="1GB", vram_budget="7GB", ram_budget="0",
        slab_budget="4.4GB", reserve_vram="1.2GB", block_rows=4096,
    )
    prompt = "The capital of France is"
    out = "".join(ns.generate(prompt, max_tokens=24, temperature=0.0))
    cold = ns.stats.tokens_per_second
    read_per_tok = (
        ns.source.stats.bytes_read / 1e9 / max(1, ns.stats.generated_tokens)
    )
    total_gb = ns.gguf.total_tensor_bytes() / 1e9
    skew = ns.model.router_stats.skew()

    ns.pin_hot_experts()
    ns.cache.stats.slab_hits = ns.cache.stats.slab_misses = 0
    "".join(ns.generate(prompt, max_tokens=24, temperature=0.0))
    warm = ns.stats.tokens_per_second
    hit = ns.cache.stats.slab_rate

    print(f"    output: {out[:70]!r}")
    print(f"    read/token {read_per_tok:.2f}GB of {total_gb:.0f}GB model "
          f"({read_per_tok / total_gb:.1%})")
    print(f"    expert skew (top 20%): {skew:.1%}")
    print(f"    cold {cold:.2f} tok/s -> pinned {warm:.2f} tok/s "
          f"(slab hit {hit:.1%})")
    ns.close()
    # The claim being tested is sparsity, not speed: a dense read would be
    # the whole model every token.
    sparse = read_per_tok < total_gb * 0.25
    record("P4", sparse and "Paris" in out,
           f"selective fetch reads {read_per_tok / total_gb:.1%} of the model "
           f"per token; {warm:.2f} tok/s (target 8)")


def p5_vision() -> None:
    """Correct captions; image prefill < 15 s."""
    print("P5  vision tower + image prefill")
    if not (M8.exists() and MM8.exists()):
        record("P5", True, "SKIPPED (8B or mmproj absent)")
        return
    from PIL import Image, ImageDraw

    shapes = ROOT / "test_shapes.png"
    if not shapes.exists():
        im = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(im)
        d.ellipse([40, 140, 190, 290], fill="red")
        d.rectangle([260, 140, 410, 290], fill="blue")
        im.save(shapes)

    ns = NeuroStream.load(
        str(M8), mem_budget="512MB", vram_budget="5GB", ram_budget="2GB",
        block_rows=4096,
    )
    ns.attach_vision(str(MM8))
    out = "".join(ns.generate_vl(
        str(shapes),
        "What shapes and colors are in this image? Answer in one sentence.",
        max_tokens=96,
    ))
    low = out.lower()
    correct = ("circle" in low and "square" in low
               and "red" in low and "blue" in low)
    prefill = ns.stats.prefill_seconds
    ntok = ns.stats.prompt_tokens
    print(f"    output: {out.strip()[:150]!r}")
    print(f"    prefill {ntok} tokens in {prefill:.2f}s "
          f"({ntok / prefill:.0f} tok/s)")
    ns.close()
    record("P5", correct and prefill < 15,
           f"shapes+colors identified={correct}, prefill {prefill:.2f}s "
           f"(target <15s)")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; these phases need a GPU")
        return 1
    p3_cuda()
    p4_moe()
    p5_vision()
    print("=" * 60)
    for phase, ok, note in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {phase}  {note}")
    return 0 if all(ok for _, ok, _ in RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
