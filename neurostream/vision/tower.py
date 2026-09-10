"""Qwen3-VL vision tower.

A SigLIP-style ViT stored in a separate `mmproj` GGUF, plus the two pieces
that make Qwen3-VL specific:

  spatial merge  2x2 neighbouring patches are concatenated (1152 -> 4608)
                 before projection, so an image costs a quarter as many LLM
                 tokens as it has patches.
  DeepStack      hidden states are tapped at blocks 8/16/24, projected, and
                 added into the *first three LLM layers* at the image token
                 positions - not just handed in at the embedding layer.

Prefill is where this pays off for a streaming engine. Every patch token is
processed in one batched pass, so a 1000-token image costs roughly one
token's worth of weight traffic rather than a thousand. The engine must not
re-stream weights per token here, and it does not: the ViT runs as a single
batched forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..format.gguf import GGUFFile
from ..io.source import MmapSource
from ..residency.cache import TieredCache


@dataclass
class VisionConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_ffn: int
    patch_size: int
    image_size: int
    merge_size: int
    eps: float
    proj_dim: int
    deepstack_layers: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "VisionConfig":
        md = g.metadata
        flags = md.get("clip.vision.is_deepstack_layers", [])
        return cls(
            n_layer=int(md["clip.vision.block_count"]),
            n_embd=int(md["clip.vision.embedding_length"]),
            n_head=int(md["clip.vision.attention.head_count"]),
            n_ffn=int(md["clip.vision.feed_forward_length"]),
            patch_size=int(md["clip.vision.patch_size"]),
            image_size=int(md["clip.vision.image_size"]),
            merge_size=int(md.get("clip.vision.spatial_merge_size", 2)),
            eps=float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
            proj_dim=int(md["clip.vision.projection_dim"]),
            deepstack_layers=tuple(i for i, f in enumerate(flags) if f),
        )


def layer_norm(x, w, b, eps):
    """LayerNorm computed in fp32. The ViT runs in fp16 and 27 stacked
    normalisations accumulate visible error otherwise."""
    out = F.layer_norm(
        x.float(), (x.shape[-1],), w.float(), b.float(), eps
    )
    return out.to(x.dtype)


def rope_2d(h: int, w: int, dim: int, device, dtype):
    """2-D RoPE over the patch grid; half the dims index rows, half columns."""
    half = dim // 2
    quarter = half // 2
    inv = 1.0 / (
        10000.0
        ** (torch.arange(0, quarter, device=device, dtype=torch.float32) / quarter)
    )
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    fy = torch.outer(ys, inv).repeat_interleave(w, dim=0)      # (h*w, quarter)
    fx = torch.outer(xs, inv).repeat(h, 1)                      # (h*w, quarter)
    f = torch.cat([fy, fx], dim=-1)                             # (h*w, half)
    emb = torch.cat([f, f], dim=-1)                             # (h*w, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class VisionTower:
    def __init__(
        self,
        mmproj_path,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_rope: bool = True,
    ) -> None:
        self.source = MmapSource(mmproj_path)
        self.gguf = self.source.gguf
        self.cfg = VisionConfig.from_gguf(self.gguf)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_rope = use_rope
        self.cache = TieredCache(
            self.source, device=device, compute_dtype=dtype,
            vram_budget=4 << 30, ram_budget=0,
        )
        # The tower is ~1.2 GB and every patch touches all of it, so it is
        # loaded once rather than streamed.
        for name in self.gguf.tensors:
            self.cache.admit_vram(name)

    def w(self, name: str) -> torch.Tensor:
        return self.cache.get(name)

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    # -- preprocessing ----------------------------------------------------

    def preprocess(self, image, max_patches: int = 1280):
        """PIL image -> (pixel tensor, grid_h, grid_w).

        Sizes to a multiple of patch*merge and caps the total patch count so
        a large photo cannot silently produce a 10k-token prefill.
        """
        from PIL import Image

        if isinstance(image, (str,)):
            image = Image.open(image)
        image = image.convert("RGB")

        p = self.cfg.patch_size * self.cfg.merge_size
        w, h = image.size
        scale = math.sqrt(max_patches * p * p / (w * h))
        scale = min(scale, 1.0) if w * h > max_patches * p * p else scale
        nw = max(p, int(round(w * scale / p)) * p)
        nh = max(p, int(round(h * scale / p)) * p)
        image = image.resize((nw, nh), Image.BICUBIC)

        import numpy as np

        arr = torch.from_numpy(np.asarray(image, dtype="float32") / 255.0)
        arr = (arr - 0.5) / 0.5                     # mean/std both 0.5
        arr = arr.permute(2, 0, 1).unsqueeze(0)     # (1,3,H,W)
        return (
            arr.to(self.device, self.dtype),
            nh // self.cfg.patch_size,
            nw // self.cfg.patch_size,
        )

    # -- forward ----------------------------------------------------------

    def patch_embed(self, px: torch.Tensor) -> torch.Tensor:
        """Conv patch embedding. Two temporal slices, identical for a still."""
        c = self.cfg
        w0 = self.w("v.patch_embd.weight")
        b = self.w("v.patch_embd.bias")
        x = F.conv2d(px, w0, b, stride=c.patch_size)
        if self.has("v.patch_embd.weight.1"):
            # A still image is fed as two identical frames, so the second
            # temporal kernel sees the same pixels.
            x = x + F.conv2d(px, self.w("v.patch_embd.weight.1"), None,
                             stride=c.patch_size)
        return x.flatten(2).transpose(1, 2)[0]      # (n_patch, n_embd)

    def position_embed(self, x, gh: int, gw: int) -> torch.Tensor:
        pe = self.w("v.position_embd.weight")       # (S*S, n_embd)
        s = int(math.sqrt(pe.shape[0]))
        if (gh, gw) != (s, s):
            pe = (
                pe.reshape(1, s, s, -1)
                .permute(0, 3, 1, 2)
                .float()
            )
            pe = F.interpolate(pe, size=(gh, gw), mode="bicubic",
                               align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(gh * gw, -1).to(x.dtype)
        return x + pe

    def block(self, x, i: int, cos, sin):
        c = self.cfg
        p = f"v.blk.{i}."
        n = x.shape[0]

        h = layer_norm(x, self.w(p + "ln1.weight"), self.w(p + "ln1.bias"), c.eps)
        qkv = h @ self.w(p + "attn_qkv.weight").T + self.w(p + "attn_qkv.bias")
        q, k, v = qkv.reshape(n, 3, c.n_head, c.head_dim).permute(1, 2, 0, 3)

        if cos is not None:
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        # Bidirectional attention: every patch sees every other patch.
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(0, 1).reshape(n, c.n_embd)
        x = x + o @ self.w(p + "attn_out.weight").T + self.w(p + "attn_out.bias")

        h = layer_norm(x, self.w(p + "ln2.weight"), self.w(p + "ln2.bias"), c.eps)
        h = h @ self.w(p + "ffn_up.weight").T + self.w(p + "ffn_up.bias")
        h = F.gelu(h, approximate="tanh")
        h = h @ self.w(p + "ffn_down.weight").T + self.w(p + "ffn_down.bias")
        return x + h

    def spatial_merge(self, x, gh: int, gw: int) -> torch.Tensor:
        """Fold 2x2 patch neighbourhoods into single tokens."""
        m = self.cfg.merge_size
        d = x.shape[-1]
        x = x.reshape(gh // m, m, gw // m, m, d)
        x = x.permute(0, 2, 1, 3, 4).reshape((gh // m) * (gw // m), m * m * d)
        return x

    def merger(self, x) -> torch.Tensor:
        x = x @ self.w("mm.0.weight").T + self.w("mm.0.bias")
        x = F.gelu(x, approximate="tanh")
        return x @ self.w("mm.2.weight").T + self.w("mm.2.bias")

    def deepstack(self, x, idx: int) -> torch.Tensor:
        p = f"v.deepstack.{idx}."
        x = layer_norm(x, self.w(p + "norm.weight"), self.w(p + "norm.bias"),
                       self.cfg.eps)
        x = x @ self.w(p + "fc1.weight").T + self.w(p + "fc1.bias")
        x = F.gelu(x, approximate="tanh")
        return x @ self.w(p + "fc2.weight").T + self.w(p + "fc2.bias")

    @torch.no_grad()
    def encode(self, image, max_patches: int = 1280):
        """image -> (embeds, deepstack_features, grid_h, grid_w).

        embeds: (n_tokens, proj_dim) to splice into the LLM input.
        deepstack_features: one (n_tokens, proj_dim) tensor per tap.
        """
        c = self.cfg
        px, gh, gw = self.preprocess(image, max_patches)
        x = self.patch_embed(px)
        x = self.position_embed(x, gh, gw)

        cos = sin = None
        if self.use_rope:
            cos, sin = rope_2d(gh, gw, c.head_dim, self.device, self.dtype)
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

        taps: list[torch.Tensor] = []
        for i in range(c.n_layer):
            x = self.block(x, i, cos, sin)
            if i in c.deepstack_layers:
                taps.append(self.deepstack(self.spatial_merge(x, gh, gw), i))

        x = layer_norm(x, self.w("v.post_ln.weight"), self.w("v.post_ln.bias"),
                       c.eps)
        embeds = self.merger(self.spatial_merge(x, gh, gw))
        return embeds, taps, gh // c.merge_size, gw // c.merge_size

    def close(self) -> None:
        self.source.close()
