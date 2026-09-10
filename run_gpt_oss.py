#!/usr/bin/env python
"""Run GPT-OSS-120B on this laptop.

63.4 GB of weights, 8 GB of VRAM. The model is 96.4% expert weights and
routes 4 of its 128 experts per token, so a token reads about 4.2 GB rather
than the whole file -- 6.6% of the model.

    python run_gpt_oss.py -p "Explain how an SSD wears out."
    python run_gpt_oss.py --chat
    python run_gpt_oss.py -p "Hi" --final-only     # hide the analysis channel
    python run_gpt_oss.py -p "Hi" --reasoning low  # shorter chain of thought
    python run_gpt_oss.py -p "Hi" --no-warmup      # skip the pin step

GPT-OSS always writes an `analysis` chain of thought before the answer, so
--max-tokens has to cover both. A low cap returns reasoning and no reply;
the runner says so explicitly when that happens. --reasoning low shortens
the thinking, which is the cheaper way to reach the answer sooner.

Weights are never held whole. Each routed expert is walked a block of
neurons at a time -- read a block of rows, multiply, accumulate, retire it,
take the next block -- so peak memory follows --block-rows and not the size
of an expert. When the router picks different experts for the next token,
the same loop simply walks the rows those experts own.

  --block-rows 512        Neurons per read. One expert matrix is 2880 rows
                          of MXFP4 at 1530 B/row = 4.41 MB; 512 rows is a
                          0.78 MB read. Measured on this machine, decode:
                          256 rows 0.51 tok/s, 512 rows 0.73 (mean of 3),
                          1024 rows slower. Below ~512 the per-block cost
                          outgrows the I/O it saves.
  --expert-lookahead 8    Neuron blocks kept in flight, shared across a whole
                          layer's experts and projections rather than
                          restarted per matrix. This is what took I/O stall
                          from 76.3 s to under 1 s and prefetch hit rate from
                          66% to 99.5%. Depth stops mattering above 8.
  --workers 8             I/O queue depth. Measured on this drive: 1 worker
                          0.62 GB/s, 8 workers 2.00, 24 workers 1.57, 48
                          workers 1.37. More threads is not more throughput.

The budget defaults are inherited from the 235B tuning in
tests/tuning_results.json and have NOT been re-measured for this model.
Unlike the 235B, GPT-OSS has only 2.3 GB of non-expert weights, so most of
them can stay resident and the slab budget is doing most of the work.
--slab-budget 4GB measured 26% faster than 3GB and is worth trying.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "models" / "gpt-oss-120b-GGUF"
FILENAME = "gpt-oss-120b-MXFP4.gguf"


def find_model(explicit: str | None) -> Path:
    """Locate the GGUF, reporting download progress if it is not ready."""
    if explicit:
        model = Path(explicit)
        if not model.exists():
            raise SystemExit(f"model not found: {model}")
        return model

    model = DEFAULT_DIR / FILENAME
    if model.exists():
        return model

    # The directory also holds header.gguf, a 32 MB prefix of the file kept
    # so the metadata can be read while the rest is still downloading. Any
    # real 120B GGUF is tens of GB, so size tells the two apart.
    matches = sorted(
        p for p in DEFAULT_DIR.glob("*.gguf")
        if p.stat().st_size > (1 << 30)
    )
    if matches:
        return matches[0]

    part = DEFAULT_DIR / (FILENAME + ".part")
    progress = DEFAULT_DIR / (FILENAME + ".progress.json")
    if part.exists() and progress.exists():
        done = len(json.loads(progress.read_text()).get("chunks", []))
        total = (part.stat().st_size + (64 << 20) - 1) // (64 << 20)
        raise SystemExit(
            f"download is still running: {done}/{total} chunks "
            f"({done / max(total, 1):.0%}).\n"
            f"finish it with  python download_gpt_oss.py  "
            f"(it resumes where it left off)"
        )
    raise SystemExit(
        f"no model under {DEFAULT_DIR}\n"
        f"fetch it with  python download_gpt_oss.py"
    )


class HarmonyStream:
    """Separates GPT-OSS's channels out of a raw token stream.

    GPT-OSS replies in harmony format: an `analysis` chain of thought first,
    then the `final` answer. Both arrive as ordinary tokens, so splitting
    them is a matter of watching for the channel marker as it streams past --
    including when a token boundary lands in the middle of it.
    """

    FINAL = "<|channel|>final<|message|>"
    # Control tokens that belong to the transport, not to the reply.
    NOISE = (
        "<|channel|>analysis<|message|>", "<|start|>assistant", "<|start|>",
        "<|channel|>", "<|message|>", "<|end|>", "<|return|>",
    )
    _KEEP = max(len(m) for m in (FINAL,) + NOISE) - 1

    def __init__(self, final_only: bool = False) -> None:
        self.final_only = final_only
        self.reached_final = False
        self.buf = ""
        self.shown = 0  # chars of cleaned analysis already printed

    def _clean(self, text: str) -> str:
        for marker in self.NOISE:
            text = text.replace(marker, "")
        return text

    def feed(self, piece: str) -> str:
        self.buf += piece
        if self.reached_final:
            out, self.buf = self.buf, ""
            return out

        cut = self.buf.find(self.FINAL)
        if cut >= 0:
            head, rest = self.buf[:cut], self.buf[cut + len(self.FINAL):]
            self.buf = ""
            self.reached_final = True
            if self.final_only:
                return rest
            return self._clean(head)[self.shown:] + rest

        if self.final_only:
            return ""
        # Clean the whole analysis so far and emit only what is new. Cleaning
        # each fragment separately would miss any marker that straddles the
        # boundary, which is how the control tokens leaked into the output.
        full = self._clean(self.buf)
        upto = max(0, len(full) - self._KEEP)
        out = full[self.shown:upto]
        self.shown = max(self.shown, upto)
        return out

    def drain(self) -> str:
        """Whatever is still buffered when generation stops."""
        if self.reached_final:
            out, self.buf = self.buf, ""
            return out
        if self.final_only:
            self.buf = ""
            return ""
        out = self._clean(self.buf)[self.shown:]
        self.buf, self.shown = "", 0
        return out


def human_eta(tokens: int, tps: float) -> str:
    if tps <= 0:
        return "unknown"
    secs = tokens / tps
    return f"{secs / 60:.1f} min" if secs > 90 else f"{secs:.0f}s"


def stream(gen, label: str = "", final_only: bool = False,
           max_tokens: int = 0) -> str:
    """Print tokens as they arrive, with a rate line at the end."""
    channel = HarmonyStream(final_only)
    out = []
    n = 0
    t0 = time.perf_counter()
    if label:
        print(label, end="", flush=True)
    if not final_only:
        print("[thinking] ", end="", flush=True)
    for piece in gen:
        n += 1
        shown = channel.feed(piece)
        if shown:
            out.append(shown)
            sys.stdout.write(shown)
            sys.stdout.flush()
    tail = channel.drain()
    if tail:
        out.append(tail)
        sys.stdout.write(tail)
    dt = time.perf_counter() - t0
    # Wall clock from the call, so this includes prefill. The decode-only
    # rate is in the report printed at the end.
    print(f"\n\n  [{n} tokens in {dt:.0f}s end-to-end "
          f"= {n / max(dt, 1e-9):.3f} tok/s incl. prefill]")
    if not channel.reached_final:
        # The reply lives on the `final` channel, which this run never
        # reached: every token went into the chain of thought.
        print(f"  [!] stopped after {n} tokens while still reasoning - the "
              f"answer had not started.\n"
              f"      raise --max-tokens (currently {max_tokens or n}) or use "
              f"--reasoning low for a shorter chain of thought.")
    return "".join(out)


def main(argv=None) -> int:
    # The model writes Unicode punctuation - non-breaking hyphens, dashes,
    # smart quotes - which the Windows console's cp1252 cannot encode. Left
    # alone that raises mid-stream and throws away a run that already cost
    # minutes of warmup and decode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description="Run GPT-OSS-120B (63 GB) on an 8 GB GPU.",
    )
    ap.add_argument("--model", default=None,
                    help="path to the gguf; auto-detected by default")
    ap.add_argument("-p", "--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-tokens", type=int, default=512,
                    help="GPT-OSS writes a chain of thought before the "
                         "answer, so a low cap returns only reasoning "
                         "(default 512)")
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("--chat", action="store_true",
                    help="interactive loop; keeps the model resident")
    ap.add_argument("--raw", action="store_true",
                    help="feed the prompt verbatim, no harmony template")
    ap.add_argument("--reasoning", default="medium",
                    choices=("low", "medium", "high"),
                    help="harmony reasoning effort (default medium)")
    ap.add_argument("--final-only", action="store_true",
                    help="hide the analysis channel, print only the answer")

    ap.add_argument("--mem-budget", default="1GB")
    ap.add_argument("--vram-budget", default="7GB")
    ap.add_argument("--ram-budget", default="2GB")
    ap.add_argument("--slab-budget", default="3GB",
                    help="VRAM held for hot expert slabs (default 3GB)")
    ap.add_argument("--reserve-vram", default="1.5GB")
    ap.add_argument("--block-rows", type=int, default=512,
                    help="neurons per streamed block (default 512, measured)")
    ap.add_argument("--expert-lookahead", type=int, default=8,
                    help="neuron blocks kept in flight (default 8)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip warmup+pin for faster startup; uncached "
                         "experts require more disk reads")
    ap.add_argument("--warmup-tokens", type=int, default=3)
    a = ap.parse_args(argv)
    if a.block_rows < 1:
        ap.error("--block-rows must be positive")
    if a.expert_lookahead < 1:
        ap.error("--expert-lookahead must be positive")

    import torch

    from neurostream.api import NeuroStream

    model = find_model(a.model)
    print(f"model   : {model.name}")
    print(f"          {model.stat().st_size / 1e9:.1f} GB on disk")
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
        expert_lookahead=a.expert_lookahead, n_workers=a.workers,
    )
    print(f"loaded  : {time.perf_counter() - t0:.1f}s   "
          f"{ns.placement.summary() if ns.placement else 'all streamed'}")

    cfg = ns.cfg
    exp_bytes = sum(
        t.nbytes for n, t in ns.gguf.tensors.items() if "_exps" in n
    )
    total = ns.gguf.total_tensor_bytes()
    per_tok = (total - exp_bytes) + exp_bytes * (cfg.n_expert_used / cfg.n_expert)
    print(f"moe     : {cfg.n_expert} experts, {cfg.n_expert_used} active -> "
          f"~{per_tok / 1e9:.1f} GB read per token "
          f"({per_tok / total:.1%} of the model)")
    print(f"neurons : {cfg.n_ffn_expert} per expert, walked "
          f"{a.block_rows} at a time, {a.expert_lookahead} blocks in flight")

    def render(text: str) -> str:
        if a.raw:
            return text
        return ns.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            reasoning_effort=a.reasoning,
        )

    if not a.no_warmup:
        print(f"warmup  : {a.warmup_tokens} tokens to learn expert routing "
              f"...", flush=True)
        t0 = time.perf_counter()
        ns.warmup(prompt=render(a.prompt), tokens=a.warmup_tokens)
        n = ns.pin_hot_experts()
        print(f"          pinned {n} experts "
              f"({ns.cache.slab_bytes / 1e9:.2f} GB) in "
              f"{time.perf_counter() - t0:.0f}s")

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
                    ns.generate(render(q), max_tokens=a.max_tokens,
                                temperature=a.temperature),
                    label="\ngpt-oss > ",
                    final_only=a.final_only, max_tokens=a.max_tokens,
                )
        else:
            est = human_eta(a.max_tokens, 0.73)
            print(f"\ngenerating up to {a.max_tokens} tokens "
                  f"(~{est} at 0.73 tok/s, plus prefill)\n")
            print(f"> {a.prompt}\n")
            stream(
                ns.generate(render(a.prompt), max_tokens=a.max_tokens,
                            temperature=a.temperature),
                final_only=a.final_only, max_tokens=a.max_tokens,
            )
            print(ns.report())
    except KeyboardInterrupt:
        print("\n\ninterrupted")
    finally:
        ns.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
