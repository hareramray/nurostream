"""Public surface.

    ns = NeuroStream.load("model.gguf", mem_budget="2GB")
    for piece in ns.generate("Hello", max_tokens=64):
        print(piece, end="", flush=True)

Everything else in the package is an implementation detail of the one promise
this class makes: the model runs inside `mem_budget` no matter how big it is.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .compute.blockffn import DEFAULT_BLOCK_ROWS
from .format.tokenizer import GGUFTokenizer
from .io.arena import Arena
from .io.stream import StreamingSource
from .model.qwen3 import KVCache, Qwen3Config, Qwen3Model
from .residency.cache import TieredCache
from .residency.planner import apply_plan, plan

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([KMGT]?)B?\s*$", re.I)
_MULT = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(v: int | str) -> int:
    if isinstance(v, int):
        return v
    m = _SIZE_RE.match(v)
    if not m:
        raise ValueError(f"cannot parse size: {v!r}")
    return int(float(m.group(1)) * _MULT[m.group(2).upper()])


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.decode_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.decode_seconds

    def summary(self) -> str:
        return (
            f"prefill {self.prompt_tokens} tok in {self.prefill_seconds:.2f}s "
            f"({self.prompt_tokens / max(self.prefill_seconds, 1e-9):.1f} tok/s) | "
            f"decode {self.generated_tokens} tok in {self.decode_seconds:.2f}s "
            f"({self.tokens_per_second:.2f} tok/s)"
        )


class NeuroStream:
    def __init__(
        self,
        path: str | Path,
        mem_budget: int | str = "2GB",
        vram_budget: int | str = 0,
        ram_budget: int | str = 0,
        device: str | None = None,
        compute_dtype: torch.dtype | None = None,
        block_rows: int = DEFAULT_BLOCK_ROWS,
        n_workers: int = 16,
        prefetch_depth: int = 1,
        reserve_vram: int | str = "1.5GB",
        slab_budget: int | str = 0,
        strict_neuron: bool = False,
        verbose: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_dtype is None:
            compute_dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.dtype = compute_dtype
        self.verbose = verbose

        self.arena = Arena(parse_size(mem_budget))
        self.source = StreamingSource(path, self.arena, n_workers=n_workers)
        self.gguf = self.source.gguf
        self.cfg = Qwen3Config.from_gguf(self.gguf)
        self.tokenizer = GGUFTokenizer(self.gguf)

        vram_budget = parse_size(vram_budget)
        ram_budget = parse_size(ram_budget)
        self.cache = TieredCache(
            self.source, device=device, compute_dtype=compute_dtype,
            vram_budget=vram_budget, ram_budget=ram_budget,
        )
        self.cache.slab_budget = parse_size(slab_budget)
        if vram_budget or ram_budget:
            t0 = time.perf_counter()
            self.placement = plan(
                self.gguf, self.cfg, vram_budget, ram_budget,
                reserve_vram=parse_size(reserve_vram),
                slab_budget=self.cache.slab_budget,
            )
            if verbose:
                print(f"  placement: {self.placement.summary()}", flush=True)
            apply_plan(self.cache, self.placement, verbose=verbose)
            if verbose:
                print(f"  resident in {time.perf_counter() - t0:.1f}s",
                      flush=True)
        else:
            self.placement = None

        self.model = Qwen3Model(
            self.cache, self.cfg, device=device, dtype=compute_dtype,
            block_rows=block_rows, prefetch_depth=prefetch_depth,
            strict_neuron=strict_neuron,
        )
        self.stats = GenerationStats()

    @classmethod
    def load(cls, path: str | Path, **kw) -> "NeuroStream":
        return cls(path, **kw)

    # -- generation -------------------------------------------------------

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float
    ) -> int:
        if temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if 0 < top_p < 1:
            srt, idx = torch.sort(probs, descending=True)
            keep = (torch.cumsum(srt, 0) - srt) < top_p
            srt = srt * keep
            srt = srt / srt.sum()
            return int(idx[torch.multinomial(srt, 1)])
        return int(torch.multinomial(probs, 1))

    def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        stop_on_eos: bool = True,
    ) -> Iterator[str]:
        ids = (
            self.tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        kv = KVCache(self.cfg.n_layer)
        self.stats = GenerationStats(prompt_tokens=len(ids))

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model.forward(torch.tensor(ids), kv, start_pos=0)
        self.stats.prefill_seconds = time.perf_counter() - t0

        eos = {self.tokenizer.eos_id} if self.tokenizer.eos_id else set()
        eos |= {
            self.tokenizer.vocab[t]
            for t in ("<|im_end|>", "<|endoftext|>")
            if t in self.tokenizer.vocab
        }

        pos = len(ids)
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            nxt = self._sample(logits[-1], temperature, top_p)
            if stop_on_eos and nxt in eos:
                break
            self.stats.generated_tokens += 1
            yield self.tokenizer.decode([nxt])
            with torch.no_grad():
                logits = self.model.forward(
                    torch.tensor([nxt]), kv, start_pos=pos
                )
            pos += 1
        self.stats.decode_seconds = time.perf_counter() - t0

    # -- vision -----------------------------------------------------------

    def attach_vision(self, mmproj_path) -> None:
        from .vision.tower import VisionTower

        self.tower = VisionTower(
            mmproj_path, device=self.device, dtype=self.dtype
        )

    def generate_vl(
        self,
        image,
        question: str = "Describe this image.",
        max_tokens: int = 64,
        temperature: float = 0.0,
        max_patches: int = 1024,
    ) -> Iterator[str]:
        """Caption or answer about an image.

        Prefill is one batched pass over the whole image, so the weight
        traffic for a 400-token image is about the same as for one text
        token. That asymmetry is what makes vision affordable on a streaming
        engine.
        """
        from .vision.prompt import IMAGE_PAD, build_prompt, next_position, position_ids

        tower = getattr(self, "tower", None)
        if tower is None:
            raise RuntimeError("call attach_vision(mmproj_path) first")

        embeds, taps, gh, gw = tower.encode(image, max_patches=max_patches)
        n_img = embeds.shape[0]

        text = build_prompt(self.tokenizer, n_img, question)
        ids = self.tokenizer.encode(text)
        img_id = self.tokenizer.vocab[IMAGE_PAD]
        if sum(1 for i in ids if i == img_id) != n_img:
            raise RuntimeError("image placeholder count does not match tower output")

        tokens = torch.tensor(ids, device=self.device)
        x = self.model.embed(tokens)
        mask = tokens == img_id
        x = x.clone()
        x[mask] = embeds.to(x.dtype)

        pos = position_ids(ids, img_id, gh, gw, self.device)
        kv = KVCache(self.cfg.n_layer)
        self.stats = GenerationStats(prompt_tokens=len(ids))

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model.forward(
                inputs_embeds=x, kv=kv, start_pos=0, pos_ids=pos,
                deepstack=taps, vision_mask=mask,
            )
        self.stats.prefill_seconds = time.perf_counter() - t0

        eos = {self.tokenizer.eos_id} if self.tokenizer.eos_id else set()
        eos |= {
            self.tokenizer.vocab[t]
            for t in ("<|im_end|>", "<|endoftext|>")
            if t in self.tokenizer.vocab
        }

        t0 = time.perf_counter()
        for _ in range(max_tokens):
            nxt = self._sample(logits[-1], temperature, 0.95)
            if nxt in eos:
                break
            self.stats.generated_tokens += 1
            yield self.tokenizer.decode([nxt])
            pos = next_position(pos)
            with torch.no_grad():
                logits = self.model.forward(
                    torch.tensor([nxt]), kv, pos_ids=pos
                )
        self.stats.decode_seconds = time.perf_counter() - t0

    # -- expert residency -------------------------------------------------

    def warmup(self, prompt: str = "Explain in detail how a computer works.",
               tokens: int = 24) -> None:
        """Generate a little, purely to observe which experts actually fire."""
        for _ in self.generate(prompt, max_tokens=tokens, temperature=0.0):
            pass

    def pin_hot_experts(self, verbose: bool = False) -> int:
        """Promote the most-used expert slabs into VRAM.

        Uses measured routing, not the uniform 8/128 assumption the static
        planner has to make. Real routing is heavily skewed, so a slab budget
        far smaller than the expert weights still covers a large share of
        actual reads.
        """
        rs = getattr(self.model, "router_stats", None)
        if rs is None or not self.cache.slab_budget:
            return 0
        cfg = self.cfg
        pinned = 0
        for (layer, e), _count in rs.hits.most_common():
            ok = True
            for nm, rpe in (
                ("ffn_gate_exps.weight", cfg.n_ffn_expert),
                ("ffn_up_exps.weight", cfg.n_ffn_expert),
                ("ffn_down_exps.weight", cfg.n_embd),
            ):
                name = f"blk.{layer}.{nm}"
                if name not in self.gguf.tensors:
                    continue
                ok &= self.cache.admit_slab(name, e * rpe, (e + 1) * rpe)
            if not ok:
                break  # slab budget exhausted
            pinned += 1
        if verbose:
            print(f"  pinned {pinned} expert slabs "
                  f"({self.cache.slab_bytes / 1e9:.2f}GB)", flush=True)
        return pinned

    def chat(self, message: str, **kw) -> Iterator[str]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}]
        )
        return self.generate(text, **kw)

    # -- introspection ----------------------------------------------------

    def report(self) -> str:
        lines = [
            f"model    : {self.gguf.path.name}",
            f"arch     : {self.gguf.arch()}  "
            f"({self.gguf.total_tensor_bytes() / 1e9:.1f}GB on disk)",
            f"device   : {self.device} / {self.dtype}",
            f"budget   : {self.arena.summary()}",
            f"io       : {self.source.stats.summary()}",
            f"cache    : {self.cache.stats.summary()}",
            f"generate : {self.stats.summary()}",
        ]
        rs = getattr(self.model, "router_stats", None)
        if rs is not None:
            lines.append(f"router   : {rs.summary()}")
        if self.cache.slab_bytes:
            lines.append(
                f"slabs    : {len(self.cache._slab)} experts, "
                f"{self.cache.slab_bytes / 1e9:.2f}GB"
            )
        return "\n".join(lines)

    def close(self) -> None:
        tower = getattr(self, "tower", None)
        if tower is not None:
            tower.cache.clear()
            tower.close()
            self.tower = None
        self.cache.clear()
        self.source.close()

    def __enter__(self) -> "NeuroStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
