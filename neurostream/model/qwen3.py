"""Qwen3 / Qwen3-VL forward pass.

Weights are never held by this module. Every weight access goes through the
cache, which goes through the source, which may or may not touch the disk.
The model does not know and must not care — that indifference is what lets a
146 GB model run in 8 GB of RAM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compute.ops import (
    apply_rope,
    mrope_tables,
    attention,
    repeat_kv,
    rms_norm,
    rope_tables,
    swiglu,
)
from ..format.gguf import GGUFFile


@dataclass
class Qwen3Config:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_ffn: int
    vocab_size: int
    rms_eps: float
    rope_base: float
    context_length: int
    # MoE (0 for dense models)
    n_expert: int = 0
    n_expert_used: int = 0
    n_ffn_expert: int = 0
    # Qwen3-VL multimodal extras
    rope_sections: tuple = ()
    n_deepstack: int = 0

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_head_kv

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "Qwen3Config":
        n_embd = int(g.cfg("embedding_length"))
        n_head = int(g.cfg("attention.head_count"))
        head_dim = int(g.cfg("attention.key_length", n_embd // n_head))
        vocab = g.tensors["token_embd.weight"].torch_shape[0]
        return cls(
            n_layer=int(g.cfg("block_count")),
            n_embd=n_embd,
            n_head=n_head,
            n_head_kv=int(g.cfg("attention.head_count_kv", n_head)),
            head_dim=head_dim,
            n_ffn=int(g.cfg("feed_forward_length", 0)),
            vocab_size=vocab,
            rms_eps=float(g.cfg("attention.layer_norm_rms_epsilon", 1e-6)),
            rope_base=float(g.cfg("rope.freq_base", 10000.0)),
            rope_sections=tuple(g.cfg("rope.dimension_sections", ()) or ()),
            n_deepstack=int(g.cfg("n_deepstack_layers", 0) or 0),
            context_length=int(g.cfg("context_length", 4096)),
            n_expert=int(g.cfg("expert_count", 0) or 0),
            n_expert_used=int(g.cfg("expert_used_count", 0) or 0),
            n_ffn_expert=int(g.cfg("expert_feed_forward_length", 0) or 0),
        )


class KVCache:
    """Per-layer key/value history, shape (H_kv, T, D) each."""

    def __init__(self, n_layer: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer
        # Hybrid Qwen3.5 layers keep convolution history and a fixed-size
        # recurrent matrix instead of a growing key/value history.
        self.conv: list[torch.Tensor | None] = [None] * n_layer
        self.recurrent: list[torch.Tensor | None] = [None] * n_layer
        self.length = 0

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=1)
            self.v[layer] = torch.cat([self.v[layer], v], dim=1)
        return self.k[layer], self.v[layer]

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (*self.k, *self.v, *self.conv, *self.recurrent)
            if t is not None
        )


class Qwen3Model:
    def __init__(
        self,
        cache,
        cfg: Qwen3Config | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        ffn_runner=None,
        prefetch_depth: int = 1,
        block_rows: int = 0,
        neuron_callback=None,
        strict_neuron: bool = False,
    ) -> None:
        self.cache = cache
        self.gguf = cache.gguf
        self.cfg = cfg or Qwen3Config.from_gguf(self.gguf)
        self.device = torch.device(device)
        self.dtype = dtype
        # P2 swaps in the neuron-block streaming FFN here.
        self.ffn_runner = ffn_runner
        self.prefetch_depth = prefetch_depth
        # block_rows > 0 puts every projection on the neuron-block path,
        # which is what makes peak memory O(block) instead of O(layer).
        self.block_rows = block_rows
        self.neuron_callback = neuron_callback
        # strict_neuron forbids the whole-tensor shortcut, so every
        # projection really is walked one neuron-block at a time even
        # when the weights are already resident in VRAM.
        self.strict_neuron = strict_neuron
        self._tied_output = "output.weight" not in self.gguf.tensors
        src = getattr(cache, "source", None)
        self._prefetcher = src if hasattr(src, "prefetch_layer") else None

    def _prefetch_ahead(self, layer: int) -> None:
        """Queue reads for the next layers while this one computes.

        Only for the whole-tensor path. Block streaming does its own
        lookahead inside stream_linear, and mixing the two deadlocks: a
        whole-tensor prefetch reserves arena bytes that only raw_bytes()
        releases, but the block path consumes through fetch_rows() and never
        calls it, so the reservation is never returned and the arena starves.
        """
        if self._prefetcher is None or self.block_rows:
            return
        from ..io.stream import DENSE_SUFFIXES, MOE_SUFFIXES

        suffixes = MOE_SUFFIXES if self.cfg.is_moe else DENSE_SUFFIXES
        for d in range(1, self.prefetch_depth + 1):
            nxt = layer + d
            if nxt >= self.cfg.n_layer:
                break
            self._prefetcher.prefetch(
                f"blk.{nxt}.{s}"
                for s in suffixes
                if not self.cache.is_resident(f"blk.{nxt}.{s}")
            )

    def w(self, name: str) -> torch.Tensor:
        return self.cache.get(name)

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """x @ W.T, streamed by neuron-block unless W is already resident."""
        # Fastest path: weights resident in VRAM, still quantized, decode
        # step. The fused kernel reads 4.5-bit blocks and never expands them.
        if not self.strict_neuron and hasattr(self.cache, 'raw_quantized'):
            from ..compute import triton_kernels as tk
            from ..io.source import row_geometry

            hit = (
                self.cache.raw_quantized(name)
                if tk.can_fuse(self.gguf.tensors[name].dtype, x) else None
            )
            if hit is not None:
                n_rows, row_elems, _ = row_geometry(self.gguf.tensors[name])
                return tk.fused_gemv(
                    hit[0], hit[1], x, n_rows, row_elems, out_dtype=self.dtype
                )

        if self.block_rows and hasattr(self.cache, 'fetch_rows'):
            from ..compute.blockffn import stream_linear

            return stream_linear(
                self.cache, name, x, self.block_rows, self.device,
                self.dtype, self.neuron_callback,
            )
        return x @ self.w(name).T

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    # -- blocks -----------------------------------------------------------

    def _attention_block(
        self,
        x: torch.Tensor,
        layer: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: KVCache,
        causal: bool,
    ) -> torch.Tensor:
        c = self.cfg
        p = f"blk.{layer}."
        t = x.shape[0]

        h = rms_norm(x, self.w(p + "attn_norm.weight"), c.rms_eps)

        q = self._linear(p + "attn_q.weight", h).view(t, c.n_head, c.head_dim)
        k = self._linear(p + "attn_k.weight", h).view(t, c.n_head_kv, c.head_dim)
        v = self._linear(p + "attn_v.weight", h).view(t, c.n_head_kv, c.head_dim)

        # Qwen3 normalises each head's vector before RoPE.
        if self.has(p + "attn_q_norm.weight"):
            q = rms_norm(q, self.w(p + "attn_q_norm.weight"), c.rms_eps)
        if self.has(p + "attn_k_norm.weight"):
            k = rms_norm(k, self.w(p + "attn_k_norm.weight"), c.rms_eps)

        q = apply_rope(q.transpose(0, 1), cos, sin)
        k = apply_rope(k.transpose(0, 1), cos, sin)
        v = v.transpose(0, 1)

        k_all, v_all = kv.append(layer, k, v)
        out = attention(
            q, repeat_kv(k_all, c.n_rep), repeat_kv(v_all, c.n_rep), causal
        )
        out = out.transpose(0, 1).reshape(t, c.n_head * c.head_dim)
        return x + self._linear(p + "attn_output.weight", out)

    def _dense_ffn(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"blk.{layer}."
        if self.ffn_runner is not None:
            return self.ffn_runner(h, p)
        gate = self._linear(p + "ffn_gate.weight", h)
        up = self._linear(p + "ffn_up.weight", h)
        return self._linear(p + "ffn_down.weight", swiglu(gate, up))

    def _ffn_block(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        c = self.cfg
        h = rms_norm(x, self.w(f"blk.{layer}.ffn_norm.weight"), c.rms_eps)
        if c.is_moe:
            from ..moe.router import moe_ffn

            return x + moe_ffn(self, h, layer)
        return x + self._dense_ffn(h, layer)

    # -- forward ----------------------------------------------------------

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.block_rows and hasattr(self.cache, 'fetch_rows'):
            from ..compute.blockffn import gather_embeddings

            return gather_embeddings(
                self.cache, 'token_embd.weight', tokens, self.device,
                self.dtype,
            )
        return self.w('token_embd.weight')[tokens].to(self.dtype)

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        kv: KVCache | None = None,
        start_pos: int = 0,
        inputs_embeds: torch.Tensor | None = None,
        last_only: bool = True,
        pos_ids: torch.Tensor | None = None,
        deepstack: list | None = None,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits: (T, vocab) or (1, vocab) when `last_only`."""
        c = self.cfg
        if inputs_embeds is None:
            if tokens is None:
                raise ValueError("need tokens or inputs_embeds")
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens, dtype=torch.long)
            x = self.embed(tokens.to(self.device))
        else:
            x = inputs_embeds.to(self.device, self.dtype)

        t = x.shape[0]
        kv = kv if kv is not None else KVCache(c.n_layer)
        if pos_ids is not None and c.rope_sections:
            cos, sin = mrope_tables(
                c.head_dim, pos_ids, c.rope_base, c.rope_sections,
                self.device, self.dtype,
            )
        else:
            positions = torch.arange(
                start_pos, start_pos + t, device=self.device
            )
            cos, sin = rope_tables(
                c.head_dim, positions, c.rope_base, self.device, self.dtype
            )

        self._prefetch_ahead(-1)  # prime layer 0 before the loop starts
        for layer in range(c.n_layer):
            self._prefetch_ahead(layer)
            x = self._attention_block(x, layer, cos, sin, kv, causal=True)
            x = self._ffn_block(x, layer)
            # DeepStack: multi-level ViT features are injected into the first
            # few LLM layers at image positions, not only at the embedding.
            if deepstack and layer < len(deepstack) and vision_mask is not None:
                x = x.clone()
                x[vision_mask] = x[vision_mask] + deepstack[layer].to(x.dtype)

        kv.length = start_pos + t

        if last_only:
            x = x[-1:]
        x = rms_norm(x, self.w("output_norm.weight"), c.rms_eps)
        head_name = (
            "token_embd.weight" if self._tied_output else "output.weight"
        )
        return self._linear(head_name, x)
