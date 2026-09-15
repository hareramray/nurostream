"""Qwen3.5 dense text inference: Gated DeltaNet plus gated full attention.

Uses llama.cpp GGUF tensor conventions, including pre-shifted RMSNorm weights,
negative exp(A_log), and tiled value heads. Projections stream through the
same row-block engine as Qwen3. Recurrent state stays in float32.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..compute.ops import apply_rope, mrope_tables, repeat_kv, rms_norm, rope_tables
from .qwen3 import KVCache, Qwen3Config, Qwen3Model


@dataclass
class Qwen35Config(Qwen3Config):
    recurrent_layers: tuple[bool, ...] = ()
    conv_kernel: int = 4
    linear_key_dim: int = 128
    linear_value_dim: int = 128
    linear_key_heads: int = 16
    linear_value_heads: int = 32
    rope_dim: int = 64

    @classmethod
    def from_gguf(cls, g):
        if g.cfg('expert_count', 0):
            raise ValueError('Qwen3.5 support currently requires a dense model such as Qwen3.5-4B')
        base = Qwen3Config.from_gguf(g)
        n_total = base.n_layer
        n_mtp = int(g.cfg('nextn_predict_layers', 0))
        if not 0 <= n_mtp < n_total:
            raise ValueError('invalid Qwen3.5 MTP layer count')
        base.n_layer -= n_mtp
        pattern = g.cfg('attention.recurrent_layers')
        if pattern is None:
            interval = int(g.cfg('full_attention_interval', 4))
            if interval <= 0:
                raise ValueError('Qwen3.5 full_attention_interval must be positive')
            pattern = [(i + 1) % interval != 0 for i in range(n_total)]
        if not isinstance(pattern, (list, tuple)) or len(pattern) != n_total:
            raise ValueError('Qwen3.5 recurrent layer metadata must match block_count')
        nv = int(g.cfg('ssm.time_step_rank'))
        inner = int(g.cfg('ssm.inner_size'))
        if nv <= 0 or inner <= 0 or inner % nv:
            raise ValueError('invalid Qwen3.5 linear attention dimensions')
        c = cls(**vars(base),
                recurrent_layers=tuple(bool(v) for v in pattern[:base.n_layer]),
                conv_kernel=int(g.cfg('ssm.conv_kernel')),
                linear_key_dim=int(g.cfg('ssm.state_size')),
                linear_value_dim=inner // nv,
                linear_key_heads=int(g.cfg('ssm.group_count')),
                linear_value_heads=nv,
                rope_dim=int(g.cfg('rope.dimension_count', base.head_dim // 4)))
        if c.conv_kernel <= 0 or c.linear_key_dim <= 0 or c.linear_key_heads <= 0:
            raise ValueError('Qwen3.5 convolution and head dimensions must be positive')
        if nv % c.linear_key_heads or c.n_head_kv <= 0 or c.n_head % c.n_head_kv:
            raise ValueError('Qwen3.5 query/value heads must be divisible by key heads')
        if c.linear_key_dim != c.linear_value_dim:
            raise ValueError('Qwen3.5 GGUF requires matching linear key/value head dimensions')
        if not 0 < c.rope_dim <= c.head_dim or c.rope_dim % 2:
            raise ValueError('invalid Qwen3.5 rotary dimension')
        if int(g.cfg('attention.value_length', c.head_dim)) != c.head_dim:
            raise ValueError('Qwen3.5 requires matching full-attention key/value dimensions')
        if g.cfg('rope.scaling.type', 'none') not in ('none', 'default'):
            raise ValueError('Qwen3.5 currently supports native-context RoPE without scaling')
        if 'blk.0.attn_norm.weight' not in g.tensors:
            raise ValueError('Qwen3.5 requires the main text model, not an MTP-only file')
        return c


class Qwen35Model(Qwen3Model):
    def __init__(self, cache, cfg=None, **kwargs):
        super().__init__(cache, cfg or Qwen35Config.from_gguf(cache.gguf), **kwargs)

    def _full_attention(self, h, layer, kv, cos, sin, start_pos):
        c = self.cfg
        p = f'blk.{layer}.'
        t = h.shape[0]
        if self.has(p + 'attn_q.weight'):
            qg = self._linear(p + 'attn_q.weight', h)
            k = self._linear(p + 'attn_k.weight', h)
            v = self._linear(p + 'attn_v.weight', h)
        else:
            qg, k, v = self._linear(p + 'attn_qkv.weight', h).split(
                [2 * c.n_head * c.head_dim, c.n_head_kv * c.head_dim, c.n_head_kv * c.head_dim], -1)
        # Each head stores its query immediately followed by its output gate.
        q, gate = qg.reshape(t, c.n_head, 2 * c.head_dim).chunk(2, -1)
        q = rms_norm(q, self.w(p + 'attn_q_norm.weight'), c.rms_eps).transpose(0, 1)
        k = rms_norm(k.reshape(t, c.n_head_kv, c.head_dim),
                     self.w(p + 'attn_k_norm.weight'), c.rms_eps).transpose(0, 1)
        q = torch.cat((apply_rope(q[..., :c.rope_dim], cos, sin), q[..., c.rope_dim:]), -1)
        k = torch.cat((apply_rope(k[..., :c.rope_dim], cos, sin), k[..., c.rope_dim:]), -1)
        v = v.reshape(t, c.n_head_kv, c.head_dim).transpose(0, 1)
        k, v = kv.append(layer, k, v)
        positions = torch.arange(start_pos, start_pos + t, device=self.device)
        mask = torch.arange(k.shape[1], device=self.device)[None, :] <= positions[:, None]
        out = F.scaled_dot_product_attention(q, repeat_kv(k, c.n_rep), repeat_kv(v, c.n_rep),
                                              attn_mask=mask, scale=1 / math.sqrt(c.head_dim))
        out = out.transpose(0, 1) * gate.sigmoid()
        return self._linear(p + 'attn_output.weight', out.reshape(t, -1))

    def _linear_attention(self, h, layer, kv):
        c = self.cfg
        p = f'blk.{layer}.'
        t = h.shape[0]
        nk, nv = c.linear_key_heads, c.linear_value_heads
        dk, dv = c.linear_key_dim, c.linear_value_dim
        key_size, value_size = nk * dk, nv * dv
        mixed = self._linear(p + 'attn_qkv.weight', h).T
        z = self._linear(p + 'attn_gate.weight', h).reshape(t, nv, dv)
        beta = self._linear(p + 'ssm_beta.weight', h).sigmoid().float()
        alpha = self._linear(p + 'ssm_alpha.weight', h).float()
        # The converter already applied -exp to A_log.
        decay = (self.w(p + 'ssm_a').float() *
                 F.softplus(alpha + self.w(p + 'ssm_dt.bias').float())).exp()

        history = kv.conv[layer]
        if history is None:
            history = mixed.new_zeros(mixed.shape[0], c.conv_kernel - 1)
        conv_input = torch.cat((history, mixed), -1)
        kv.conv[layer] = conv_input[:, -(c.conv_kernel - 1):].clone() if c.conv_kernel > 1 else conv_input[:, :0].clone()
        weight = self.w(p + 'ssm_conv1d.weight').reshape(mixed.shape[0], 1, c.conv_kernel)
        convolved = F.conv1d(conv_input[None], weight, groups=mixed.shape[0])[0].T
        q, k, v = F.silu(convolved).split((key_size, key_size, value_size), -1)
        q, k = (x.reshape(t, nk, dk).float() for x in (q, k))
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        # GGUF value heads are tiled by key head, unlike HF's grouped ordering.
        q, k = (x.repeat(1, nv // nk, 1) for x in (q, k))
        v = v.reshape(t, nv, dv).float()
        state = kv.recurrent[layer]
        if state is None:
            state = torch.zeros(nv, dk, dv, dtype=torch.float32, device=self.device)
        out = torch.empty_like(v)
        # We stream each projection once for the entire input chunk. Only the
        # small state update is sequential in the token dimension.
        for j in range(t):
            state = state * decay[j, :, None, None]
            prediction = (state * k[j, :, :, None]).sum(-2)
            delta = (v[j] - prediction) * beta[j, :, None]
            state = state + k[j, :, :, None] * delta[:, None, :]
            out[j] = (state * q[j, :, :, None]).sum(-2)
        kv.recurrent[layer] = state
        out = out.to(self.dtype)
        # Gated RMSNorm rounds the normalized values before applying its weight.
        normalized = (out.float() * torch.rsqrt(out.float().square().mean(-1, keepdim=True) + c.rms_eps)).to(self.dtype)
        normalized = normalized * self.w(p + 'ssm_norm.weight')
        out = (normalized.float() * F.silu(z.float())).to(self.dtype)
        return self._linear(p + 'ssm_out.weight', out.reshape(t, value_size))

    def forward(self, tokens=None, kv=None, start_pos=0, inputs_embeds=None,
                last_only=True, pos_ids=None, deepstack=None, vision_mask=None):
        c = self.cfg
        if inputs_embeds is None:
            if tokens is None:
                raise ValueError('Qwen3.5 text inference requires token IDs')
            tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
            if tokens.ndim != 1 or tokens.numel() == 0:
                raise ValueError('Qwen3.5 requires a nonempty one-dimensional token sequence')
            x = self.embed(tokens)
        else:
            x = inputs_embeds.to(self.device, self.dtype)
            if x.ndim != 2 or x.shape[0] == 0:
                raise ValueError('Qwen3.5 requires a nonempty (tokens, embedding) input')
        t = x.shape[0]
        if deepstack:
            # The 4B ships an empty deepstack_visual_indexes; a tower that
            # produced taps would belong to a model this path cannot run.
            raise ValueError('Qwen3.5 vision features are injected at the embedding layer, not by DeepStack')
        kv = kv if kv is not None else KVCache(c.n_layer)
        if start_pos != kv.length or start_pos + t > c.context_length:
            raise ValueError('Qwen3.5 position must follow the cache and fit the context length')
        if pos_ids is None:
            positions = torch.arange(start_pos, start_pos + t, device=self.device)
            cos, sin = rope_tables(c.rope_dim, positions, c.rope_base, self.device, self.dtype)
        else:
            # Image tokens carry their own (t, h, w) position, so the rotary
            # tables have to be built per axis rather than from a counter.
            if not c.rope_sections:
                raise ValueError('Qwen3.5 GGUF is missing rope.dimension_sections needed for image positions')
            pos_ids = torch.as_tensor(pos_ids, dtype=torch.long, device=self.device)
            if pos_ids.shape != (3, t):
                raise ValueError('Qwen3.5 position ids must have shape (3, tokens)')
            cos, sin = mrope_tables(c.rope_dim, pos_ids, c.rope_base, c.rope_sections,
                                    self.device, self.dtype, interleaved=True)
        for layer, recurrent in enumerate(c.recurrent_layers):
            p = f'blk.{layer}.'
            h = rms_norm(x, self.w(p + 'attn_norm.weight'), c.rms_eps)
            if recurrent:
                out = self._linear_attention(h, layer, kv)
            else:
                out = self._full_attention(h, layer, kv, cos, sin, start_pos)
            x = x + out
            h = rms_norm(x, self.w(p + 'post_attention_norm.weight'), c.rms_eps)
            x = x + self._dense_ffn(h, layer)
        kv.length = start_pos + t
        if last_only:
            x = x[-1:]
        x = rms_norm(x, self.w('output_norm.weight'), c.rms_eps)
        return self._linear('token_embd.weight' if self._tied_output else 'output.weight', x).float()
