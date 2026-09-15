"""Gemma 4 E4B text inference using the existing GGUF streaming projections.

Tensor names and RoPE frequency factors follow llama.cpp's Gemma4 converter.
The auxiliary token embedding table is gathered by row, never loaded whole.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..compute.blockffn import gather_embeddings
from ..compute.ops import apply_rope, repeat_kv, rms_norm
from .qwen3 import KVCache, Qwen3Model


@dataclass
class Gemma4Config:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: tuple[int, ...]
    head_dim: int
    head_dim_swa: int
    n_ffn: tuple[int, ...]
    vocab_size: int
    rms_eps: float
    rope_base: float
    rope_base_swa: float
    context_length: int
    sliding_window: int
    sliding_pattern: tuple[bool, ...]
    n_kv_shared: int
    n_embd_per_layer: int
    logit_softcap: float
    n_expert: int = 0

    @property
    def is_moe(self) -> bool:
        return False

    @classmethod
    def from_gguf(cls, g) -> "Gemma4Config":
        if g.cfg('expert_count', 0) or not g.cfg('embedding_length_per_layer_input', 0):
            raise ValueError('Gemma 4 support requires a dense E4B-style model with per-layer embeddings')
        n = int(g.cfg('block_count'))

        def per_layer(key):
            value = g.cfg(key)
            values = tuple(value) if isinstance(value, (list, tuple)) else (value,) * n
            if len(values) != n or any(int(v) <= 0 for v in values):
                raise ValueError(f'gemma4.{key} must contain positive values for {n} layers')
            return tuple(int(v) for v in values)

        pattern = g.cfg('attention.sliding_window_pattern')
        if not isinstance(pattern, (list, tuple)) or len(pattern) != n:
            raise ValueError('Gemma 4 requires an explicit sliding-window pattern for every layer')
        c = cls(
            n_layer=n, n_embd=int(g.cfg('embedding_length')),
            n_head=int(g.cfg('attention.head_count')),
            n_head_kv=per_layer('attention.head_count_kv'),
            head_dim=int(g.cfg('attention.key_length')),
            head_dim_swa=int(g.cfg('attention.key_length_swa')),
            n_ffn=per_layer('feed_forward_length'),
            vocab_size=g.tensors['token_embd.weight'].torch_shape[0],
            rms_eps=float(g.cfg('attention.layer_norm_rms_epsilon', 1e-6)),
            rope_base=float(g.cfg('rope.freq_base', 1e6)),
            rope_base_swa=float(g.cfg('rope.freq_base_swa', 1e4)),
            context_length=int(g.cfg('context_length', 131072)),
            sliding_window=int(g.cfg('attention.sliding_window', 512)),
            sliding_pattern=tuple(bool(v) for v in pattern),
            n_kv_shared=int(g.cfg('attention.shared_kv_layers', 0)),
            n_embd_per_layer=int(g.cfg('embedding_length_per_layer_input')),
            logit_softcap=float(g.cfg('final_logit_softcapping', 30.0)),
        )
        if not 0 <= c.n_kv_shared < n or c.sliding_window <= 0:
            raise ValueError('invalid Gemma 4 shared KV count or sliding window')
        if any(c.n_head % h for h in c.n_head_kv):
            raise ValueError('Gemma 4 query heads must be divisible by KV heads')
        if c.head_dim % 2 or c.head_dim_swa % 2:
            raise ValueError('Gemma 4 RoPE requires even head dimensions')
        for suffix, dim in (('', c.head_dim), ('_swa', c.head_dim_swa)):
            if int(g.cfg('attention.value_length' + suffix, dim)) != dim:
                raise ValueError('Gemma 4 key and value dimensions must match')
            if int(g.cfg('rope.dimension_count' + suffix, dim)) != dim:
                raise ValueError('Gemma 4 E4B requires full-width RoPE with frequency factors')
        first_shared = n - c.n_kv_shared
        for i in range(first_shared, n):
            donors = [j for j in range(first_shared) if pattern[j] == pattern[i]]
            if not donors or c.n_head_kv[donors[-1]] != c.n_head_kv[i]:
                raise ValueError(f'Gemma 4 layer {i} has no compatible shared KV source')
        return c


class Gemma4Model(Qwen3Model):
    def __init__(self, cache, cfg=None, **kwargs):
        super().__init__(cache, cfg or Gemma4Config.from_gguf(cache.gguf), **kwargs)
        c = self.cfg
        self._kv_donors = {}
        for i in range(c.n_layer - c.n_kv_shared):
            self._kv_donors[c.sliding_pattern[i]] = i
        # Converted GGUFs store one shared full-attention frequency tensor.
        if not self.has('rope_freqs.weight'):
            raise ValueError('Gemma 4 GGUF is missing rope_freqs.weight; reconvert with llama.cpp')

    def _gather(self, name, tokens):
        if hasattr(self.cache, 'fetch_rows'):
            return gather_embeddings(self.cache, name, tokens, self.device, self.dtype)
        return self.w(name)[tokens].to(self.dtype)

    def embed(self, tokens):
        scale = torch.tensor(math.sqrt(self.cfg.n_embd), device=self.device, dtype=self.dtype)
        return self._gather('token_embd.weight', tokens) * scale

    def _rope(self, positions, sliding):
        c = self.cfg
        dim = c.head_dim_swa if sliding else c.head_dim
        base = c.rope_base_swa if sliding else c.rope_base
        inv = base ** (-torch.arange(0, dim, 2, device=self.device).float() / dim)
        if not sliding:
            factors = self.w('rope_freqs.weight').float()
            if factors.shape != inv.shape:
                raise ValueError('Gemma 4 rope_freqs.weight has an invalid shape')
            inv = inv / factors
        angles = positions.float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(self.dtype), angles.sin().to(self.dtype)

    def forward(self, tokens=None, kv=None, start_pos=0, last_only=True):
        c = self.cfg
        if tokens is None:
            raise ValueError('Gemma 4 text inference requires token IDs')
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError('Gemma 4 requires a nonempty one-dimensional token sequence')
        t = len(tokens)
        kv = kv if kv is not None else KVCache(c.n_layer)
        if start_pos != kv.length or start_pos + t > c.context_length:
            raise ValueError('Gemma 4 position must follow the KV cache and fit the context length')
        x = self.embed(tokens)
        ple = self._gather('per_layer_token_embd.weight', tokens)
        scale = torch.tensor(math.sqrt(c.n_embd_per_layer), device=self.device, dtype=self.dtype)
        ple = ple.reshape(t, c.n_layer, c.n_embd_per_layer) * scale
        projection = self._linear('per_layer_model_proj.weight', x) / math.sqrt(c.n_embd)
        projection = projection.reshape_as(ple)
        ple = (ple + rms_norm(projection, self.w('per_layer_proj_norm.weight'), c.rms_eps)) / math.sqrt(2)
        positions = torch.arange(start_pos, start_pos + t, device=self.device)
        ropes = {s: self._rope(positions, s) for s in set(c.sliding_pattern)}
        # Keep current-chunk KV views until all sharing layers have consumed them.
        # Persistent sliding caches are cropped only after the layer loop.
        shared = {}
        for i, sliding in enumerate(c.sliding_pattern):
            p = f'blk.{i}.'
            dim = c.head_dim_swa if sliding else c.head_dim
            cos, sin = ropes[sliding]
            h = rms_norm(x, self.w(p + 'attn_norm.weight'), c.rms_eps)
            q = self._linear(p + 'attn_q.weight', h).view(t, c.n_head, dim)
            q = rms_norm(q, self.w(p + 'attn_q_norm.weight'), c.rms_eps)
            q = apply_rope(q.transpose(0, 1), cos, sin)
            if i < c.n_layer - c.n_kv_shared:
                k = self._linear(p + 'attn_k.weight', h).view(t, c.n_head_kv[i], dim)
                v = self._linear(p + 'attn_v.weight', h).view_as(k) if self.has(p + 'attn_v.weight') else k
                v = rms_norm(v, torch.ones((), device=self.device), c.rms_eps).transpose(0, 1)
                k = rms_norm(k, self.w(p + 'attn_k_norm.weight'), c.rms_eps)
                k = apply_rope(k.transpose(0, 1), cos, sin)
                k, v = kv.append(i, k, v)
                if i == self._kv_donors[sliding]:
                    shared[sliding] = (k, v)
            else:
                k, v = shared[sliding]
            key_pos = torch.arange(start_pos + t - k.shape[1], start_pos + t, device=self.device)
            mask = key_pos[None, :] <= positions[:, None]
            if sliding:
                mask &= key_pos[None, :] > positions[:, None] - c.sliding_window
            n_rep = c.n_head // c.n_head_kv[i]
            out = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                                  attn_mask=mask, scale=1.0)
            out = out.transpose(0, 1).reshape(t, c.n_head * dim)
            out = self._linear(p + 'attn_output.weight', out)
            x = x + rms_norm(out, self.w(p + 'post_attention_norm.weight'), c.rms_eps)
            h = rms_norm(x, self.w(p + 'ffn_norm.weight'), c.rms_eps)
            gate = F.gelu(self._linear(p + 'ffn_gate.weight', h), approximate='tanh')
            out = self._linear(p + 'ffn_down.weight', gate * self._linear(p + 'ffn_up.weight', h))
            x = x + rms_norm(out, self.w(p + 'post_ffw_norm.weight'), c.rms_eps)
            gate = F.gelu(self._linear(p + 'inp_gate.weight', x), approximate='tanh')
            out = self._linear(p + 'proj.weight', gate * ple[:, i])
            x = x + rms_norm(out, self.w(p + 'post_norm.weight'), c.rms_eps)
            if self.has(p + 'layer_output_scale.weight'):
                x = x * self.w(p + 'layer_output_scale.weight')
        for i in range(c.n_layer - c.n_kv_shared):
            if c.sliding_pattern[i]:
                # clone prevents a small view from retaining a full prefill allocation.
                kv.k[i] = kv.k[i][:, -c.sliding_window:].clone()
                kv.v[i] = kv.v[i][:, -c.sliding_window:].clone()
        kv.length = start_pos + t
        if last_only:
            x = x[-1:]
        x = rms_norm(x, self.w('output_norm.weight'), c.rms_eps)
        logits = self._linear('token_embd.weight' if self._tied_output else 'output.weight', x).float()
        if c.logit_softcap:
            logits = torch.tanh(logits / c.logit_softcap) * c.logit_softcap
        return logits
