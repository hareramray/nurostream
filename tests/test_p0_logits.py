"""P0 exit criterion: logits match HuggingFace transformers to < 1e-3.

The two models are run sequentially and the reference is freed in between,
because holding both in fp32 needs ~5 GB and this machine does not have it.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_DIR = ROOT / "models" / "Qwen3-0.6B-hf"
GGUF = ROOT / "models" / "Qwen3-0.6B-BF16.gguf"
PROMPT = "The capital of France is Paris, and the capital of Japan is"


def reference_logits() -> tuple[np.ndarray, list[int]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(ids).logits[0].float().numpy()
    token_ids = ids[0].tolist()
    del model, tok
    gc.collect()
    return out, token_ids


def ours(token_ids: list[int]) -> np.ndarray:
    from neurostream.io.source import MmapSource
    from neurostream.model.qwen3 import KVCache, Qwen3Model
    from neurostream.residency.cache import TieredCache

    src = MmapSource(GGUF)
    cache = TieredCache(
        src, device="cpu", compute_dtype=torch.float32,
        vram_budget=0, ram_budget=8 << 30,
    )
    model = Qwen3Model(cache, device="cpu", dtype=torch.float32)
    print(f"  config: {model.cfg}")
    with torch.no_grad():
        logits = model.forward(
            torch.tensor(token_ids), KVCache(model.cfg.n_layer), last_only=False
        )
    print(f"  cache: {cache.stats.summary()}")
    src.close()
    return logits.float().numpy()


def main() -> int:
    print("[1/2] HuggingFace reference ...")
    ref, ids = reference_logits()
    print(f"  tokens={ids}")
    print(f"  ref logits {ref.shape}")

    print("[2/2] neurostream ...")
    mine = ours(ids)
    print(f"  our logits {mine.shape}")

    assert mine.shape == ref.shape, f"shape {mine.shape} != {ref.shape}"
    diff = np.abs(mine - ref)
    scale = np.abs(ref).max()
    print("\n--- P0 RESULT ---")
    print(f"  max abs diff : {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  logit scale  : {scale:.3f}")
    print(f"  rel error    : {diff.max() / scale:.3e}")
    print(f"  argmax match : {(mine.argmax(-1) == ref.argmax(-1)).all()}")
    print(f"  top-1 last   : ours={mine[-1].argmax()} ref={ref[-1].argmax()}")

    ok = diff.max() < 1e-3 and (mine.argmax(-1) == ref.argmax(-1)).all()
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
