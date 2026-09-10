"""GPT-OSS GGUF inference using the same bounded weight streaming/cache.

Implements biased projections and experts, YaRN, attention sinks, alternating
sliding/full attention, and the GPT-OSS clamped SwiGLU. GGUF conversion has
already separated the interleaved gate/up expert weights into two tensors.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .qwen3 import KVCache, Qwen3Config, Qwen3Model
from ..compute.blockffn import (
    DEFAULT_LOOKAHEAD, BlockPrefetcher, stream_expert_ffn,
)
from ..compute.ops import apply_rope, repeat_kv, rms_norm
from ..moe.router import RouterStats, _expert_matmul, submit_expert_reads


@dataclass
class GptOssConfig(Qwen3Config):
    sliding_window: int = 128
    rope_factor: float = 32.0
    original_context: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    @classmethod
    def from_gguf(cls, g):
        values = asdict(Qwen3Config.from_gguf(g))
        values['n_ffn_expert'] = values['n_ffn_expert'] or values['n_ffn']
        return cls(**values,
                   sliding_window=int(g.cfg('attention.sliding_window', 128)),
                   rope_factor=float(g.cfg('rope.scaling.factor', 32.0)),
                   original_context=int(g.cfg('rope.scaling.original_context_length', 4096)),
                   beta_fast=float(g.cfg('rope.scaling.yarn_beta_fast', 32.0)),
                   beta_slow=float(g.cfg('rope.scaling.yarn_beta_slow', 1.0)))


def yarn_tables(cfg, positions, device, dtype):
    half = cfg.head_dim // 2
    dims = torch.arange(half, dtype=torch.float32, device=device)
    inv = 1.0 / cfg.rope_base ** (dims / half)
    factor = cfg.rope_factor
    if factor != 1:
        def correction(rotations):
            return cfg.head_dim * math.log(cfg.original_context / (rotations * 2 * math.pi)) / (2 * math.log(cfg.rope_base))
        # GPT-OSS uses the untruncated (non-integer) YaRN correction range.
        low = max(correction(cfg.beta_fast), 0)
        high = min(correction(cfg.beta_slow), cfg.head_dim - 1)
        ramp = ((dims - low) / (high - low if high != low else 0.001)).clamp(0, 1)
        inv = inv * (1 - ramp) + inv / factor * ramp
    freqs = positions.float()[:, None] * inv[None, :]
    freqs = torch.cat((freqs, freqs), -1)
    amplitude = 1 + 0.1 * math.log(factor) if factor > 1 else 1.0
    return (freqs.cos() * amplitude).to(dtype), (freqs.sin() * amplitude).to(dtype)


def clamped_swiglu(gate, up):
    """GPT-OSS activation: clamp both halves, then a shifted SwiGLU.

    Elementwise per neuron, so it composes with any block decomposition of
    the gate/up walk -- the result does not depend on how the neurons were
    grouped into reads.
    """
    gate = gate.clamp(max=7)
    up = up.clamp(-7, 7)
    return (up + 1) * (gate * torch.sigmoid(gate * 1.702))


def sink_attention(q, k, v, sinks, sliding_window=0):
    """Causal attention with one learned softmax sink per query head."""
    tq, tk = q.shape[1], k.shape[1]
    qp = torch.arange(tk - tq, tk, device=q.device)[:, None]
    kp = torch.arange(tk, device=q.device)[None, :]
    visible = kp <= qp
    if sliding_window:
        visible &= kp > qp - sliding_window
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~visible[None], -torch.inf)
    sink = sinks.float()[:, None, None].expand(-1, tq, 1)
    probs = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
    return (probs @ v.float()).to(q.dtype)


class GptOssModel(Qwen3Model):
    def __init__(self, cache, cfg=None, *, expert_lookahead=DEFAULT_LOOKAHEAD,
                 **kwargs):
        super().__init__(cache, cfg or GptOssConfig.from_gguf(cache.gguf), **kwargs)
        # Queue depth for the routed-expert walk. A neuron block is a
        # fraction of an expert slab, so depth is what keeps the drive at its
        # sequential rate now that reads are smaller than a whole expert.
        self.expert_lookahead = expert_lookahead

    def _biased_linear(self, name, x):
        out = self._linear(name + '.weight', x)
        return out + self.w(name + '.bias')

    def _attention_block(self, x, layer, cos, sin, kv, causal=True):
        c = self.cfg
        p = f'blk.{layer}.'
        t = x.shape[0]
        h = rms_norm(x, self.w(p + 'attn_norm.weight'), c.rms_eps)
        q = self._biased_linear(p + 'attn_q', h).view(t, c.n_head, c.head_dim).transpose(0, 1)
        k = self._biased_linear(p + 'attn_k', h).view(t, c.n_head_kv, c.head_dim).transpose(0, 1)
        v = self._biased_linear(p + 'attn_v', h).view(t, c.n_head_kv, c.head_dim).transpose(0, 1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = kv.append(layer, k, v)
        out = sink_attention(q, repeat_kv(k, c.n_rep), repeat_kv(v, c.n_rep),
                             self.w(p + 'attn_sinks.weight'), c.sliding_window if layer % 2 == 0 else 0)
        out = out.transpose(0, 1).reshape(t, c.n_head * c.head_dim)
        return x + self._biased_linear(p + 'attn_output', out)

    def _route(self, h, layer):
        """Top-k over the biased router. Returns (chosen, ids, weights)."""
        logits = self._biased_linear(f'blk.{layer}.ffn_gate_inp', h)
        top, ids = torch.topk(logits, self.cfg.n_expert_used, dim=-1)
        weights = torch.softmax(top.float(), dim=-1).to(h.dtype)
        return torch.unique(ids).tolist(), ids, weights

    def _ffn_block(self, x, layer):
        c = self.cfg
        p = f'blk.{layer}.'
        h = rms_norm(x, self.w(p + 'post_attention_norm.weight'), c.rms_eps)
        chosen, ids, weights = self._route(h, layer)
        stats = getattr(self, 'router_stats', None) or RouterStats()
        self.router_stats = stats
        stats.tokens_routed += h.shape[0]
        out = torch.zeros_like(h)

        if self.block_rows:
            # Neuron-granular path. The router names the active experts, and
            # each one is then walked a block of neurons at a time rather
            # than read whole, so peak weight memory is set by --block-rows
            # and not by the expert size. 120B in MXFP4 is ~4.4 MB per expert
            # matrix; at 1024 rows a block is ~1.5 MB of that.
            active = []
            for e in chosen:
                mask = ids == e
                rows = mask.any(-1).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                active.append((e, rows, (weights * mask).sum(-1)[rows, None]))

            # One prefetch plan for the whole layer, in the order the experts
            # will actually be consumed. A per-projection pipeline drains 12
            # times a layer and the drive goes idle at each boundary; this
            # keeps the same number of blocks in flight but lets a retiring
            # gate block pull in a down block, or the next expert's.
            plan = [
                (p + f'ffn_{kind}_exps.weight', e * span, (e + 1) * span)
                for e, _rows, _w in active
                for kind, span in (('gate', c.n_ffn_expert),
                                   ('up', c.n_ffn_expert),
                                   ('down', c.n_embd))
            ]
            prefetcher = BlockPrefetcher(
                self.cache, plan, self.block_rows, self.expert_lookahead,
            )
            try:
                for e, rows, weight in active:
                    stats.hits[(layer, e)] += rows.numel()
                    stats.expert_reads += 1
                    out[rows] += weight * stream_expert_ffn(
                        self.cache, p, e, h[rows], c.n_ffn_expert, c.n_embd,
                        clamped_swiglu, self.block_rows, self.device,
                        self.dtype, bias=True,
                        lookahead=self.expert_lookahead,
                        callback=self.neuron_callback,
                        prefetcher=prefetcher,
                    )
            finally:
                # A failed forward must not strand reservations in the arena.
                prefetcher.close()
            return x + out

        # Whole-expert path: one read per expert matrix, queued together for
        # queue depth. Lower overhead per expert, but peak memory scales with
        # the expert rather than with the block.
        pending = submit_expert_reads(self, layer, chosen)
        try:
            for e in chosen:
                mask = ids == e
                rows = mask.any(-1).nonzero(as_tuple=True)[0]
                weight = (weights * mask).sum(-1)[rows, None]
                stats.hits[(layer, e)] += rows.numel()
                stats.expert_reads += 1
                def project(kind, value, rpe):
                    name = p + f'ffn_{kind}_exps'
                    y = _expert_matmul(self, name + '.weight', e, rpe, value, pending)
                    bias = self.cache.fetch_rows(name + '.bias', e, e + 1,
                                                 device=self.device, dtype=self.dtype)
                    return y + bias
                gate = project('gate', h[rows], c.n_ffn_expert)
                up = project('up', h[rows], c.n_ffn_expert)
                out[rows] += weight * project(
                    'down', clamped_swiglu(gate, up), c.n_embd)
        finally:
            # A failed forward must not strand reservations in the arena.
            for future, nbytes in pending.values():
                future.cancel()
                self.cache.source.arena.release(nbytes)
        return x + out

    def forward(self, tokens=None, kv=None, start_pos=0, inputs_embeds=None,
                last_only=True, **kwargs):
        if inputs_embeds is None:
            if tokens is None:
                raise ValueError('need tokens or inputs_embeds')
            x = self.embed(torch.as_tensor(tokens, dtype=torch.long, device=self.device))
        else:
            x = inputs_embeds.to(self.device, self.dtype)
        t = x.shape[0]
        if start_pos + t > self.cfg.context_length:
            raise ValueError('prompt and continuation exceed model context length')
        kv = kv if kv is not None else KVCache(self.cfg.n_layer)
        positions = torch.arange(start_pos, start_pos + t, device=self.device)
        cos, sin = yarn_tables(self.cfg, positions, self.device, self.dtype)
        for layer in range(self.cfg.n_layer):
            x = self._attention_block(x, layer, cos, sin, kv)
            x = self._ffn_block(x, layer)
        kv.length = start_pos + t
        if last_only:
            x = x[-1:]
        x = rms_norm(x, self.w('output_norm.weight'), self.cfg.rms_eps)
        return self._linear('output.weight', x)
