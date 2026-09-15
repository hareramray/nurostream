"""HuggingFace safetensors presented as a GGUF-shaped index.

The engine speaks one dialect of weights: llama.cpp's. Tensors are named
`blk.N.attn_q.weight`, RMSNorm weights arrive pre-shifted by one, Qwen3.5's
`A_log` arrives already negated and exponentiated, and its linear-attention
value heads arrive tiled by key head rather than grouped. A `.safetensors`
checkpoint speaks HuggingFace's dialect instead.

Rather than teach every model a second dialect, the translation happens here,
at the index. Each GGUF-shaped tensor is a *recipe*: one source tensor, an
optional row permutation, an optional within-row permutation, and an
elementwise fixup. No weight data is read - opening a 9 GB checkpoint costs
one header parse per shard, which is the same promise `GGUFFile` makes.

Every transform llama.cpp's converter applies turns out to be row-local: row r
of the result depends on exactly one row of one source tensor. That is what
keeps neuron-block streaming intact across the translation.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .gguf import GGMLType, TensorInfo, nbytes_for

# safetensors spells its dtypes out; only float ones make sense here.
DTYPES = {
    "F32": (GGMLType.F32, torch.float32),
    "F16": (GGMLType.F16, torch.float16),
    "BF16": (GGMLType.BF16, torch.bfloat16),
}

SHARD_RE = re.compile(r"\.safetensors$")


@dataclass(slots=True)
class Recipe:
    """How to build one GGUF-shaped tensor from the checkpoint.

    `rows` maps an output row index to a source row index and `cols` maps an
    output column to a source column; either may be None for identity. `op`
    is the elementwise fixup applied last.
    """

    source: str
    rows: torch.Tensor | None = None
    cols: torch.Tensor | None = None
    op: str = "copy"  # copy | shift | neg_exp


@dataclass(slots=True)
class Entry:
    """Where a source tensor sits inside a shard."""

    shard: int
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


def discover_safetensors(path: str | Path) -> tuple[Path, list[Path]]:
    """(model directory, shards in index order).

    Accepts the directory, any one shard, or the index JSON. Shard order
    follows the index when there is one, so a tensor's shard number means the
    same thing it means to Transformers.
    """
    path = Path(path)
    root = path if path.is_dir() else path.parent
    index = next(
        (p for p in (path, root / "model.safetensors.index.json")
         if p.is_file() and p.suffix == ".json"),
        None,
    )
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names: list[str] = []
        for shard in weight_map.values():
            if shard not in names:
                names.append(shard)
        shards = [root / n for n in sorted(names)]
    elif path.is_file() and path.suffix == ".safetensors":
        shards = [path]
    else:
        shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files found in {root}")
    missing = [s.name for s in shards if not s.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shard(s): {', '.join(missing)}")
    return root, shards


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """(header dict, byte offset of the data section). Reads only the header."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path.name} is too short to be safetensors")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > (1 << 30):
            raise ValueError(f"{path.name} has an implausible header length {length}")
        header = json.loads(fh.read(length).decode("utf-8"))
    return header, 8 + length


def tiled_v_perm(groups: int, per_group: int, head_dim: int) -> torch.Tensor:
    """Grouped-by-key-head order -> the tiled order ggml broadcasts against.

    Returns output index -> source index, so gathering with it reproduces
    llama.cpp's `_reorder_v_heads`.
    """
    idx = torch.arange(groups * per_group * head_dim, dtype=torch.long)
    return idx.reshape(groups, per_group, head_dim).permute(1, 0, 2).reshape(-1)


class SafetensorsFile:
    """A checkpoint directory presented the way `GGUFFile` is.

    `part` picks which half of a multimodal checkpoint to expose: the text
    model, or the vision tower that a GGUF build would ship as a separate
    `mmproj` file. Keeping them apart mirrors that split, so the planner does
    not spend the text model's cache budget on vision weights.
    """

    def __init__(self, path: str | Path, part: str = "text") -> None:
        if part not in ("text", "vision"):
            raise ValueError("part must be 'text' or 'vision'")
        self.part = part
        self.root, self.paths = discover_safetensors(path)
        self.path = self.paths[0]

        self.entries: dict[str, Entry] = {}
        self.data_offsets: list[int] = []
        for i, shard in enumerate(self.paths):
            header, data_offset = read_header(shard)
            self.data_offsets.append(data_offset)
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if name in self.entries:
                    raise ValueError(f"tensor {name!r} appears in two shards")
                begin, end = spec["data_offsets"]
                self.entries[name] = Entry(
                    i, spec["dtype"], tuple(spec["shape"]), begin, end
                )

        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"no config.json beside the weights in {self.root}")
        self.config: dict[str, Any] = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.recipes: dict[str, Recipe] = {}
        self.alignment = 1
        self.version = 0
        self._build()

    # -- GGUFFile-compatible surface --------------------------------------

    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def cfg(self, suffix: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.arch()}.{suffix}", default)

    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def locate(self, name: str) -> tuple[Path, int]:
        """(file, absolute offset) of the *source* tensor a recipe reads."""
        entry = self.entries[self.recipes[name].source]
        return self.paths[entry.shard], self.data_offsets[entry.shard] + entry.begin

    def file_offset(self, info: TensorInfo) -> int:
        path, offset = self.locate(info.name)
        del path
        return offset

    def source_entry(self, name: str) -> Entry:
        return self.entries[self.recipes[name].source]

    def __repr__(self) -> str:
        return (
            f"<SafetensorsFile {len(self.paths)} shard(s) arch={self.arch()} "
            f"part={self.part} tensors={len(self.tensors)} "
            f"data={self.total_tensor_bytes() / 1e9:.1f}GB>"
        )

    # -- index construction -----------------------------------------------

    def _declare(self, name, source, shape, rows=None, cols=None, op="copy"):
        """Register one output tensor with the recipe that produces it."""
        entry = self.entries.get(source)
        if entry is None:
            raise KeyError(f"checkpoint has no tensor {source!r}")
        if entry.dtype not in DTYPES:
            raise NotImplementedError(
                f"{source} is stored as {entry.dtype}; neurostream reads "
                f"safetensors in {', '.join(DTYPES)} "
                f"(quantized checkpoints should be loaded as GGUF)"
            )
        ggml_type = DTYPES[entry.dtype][0]
        n_elements = 1
        for d in shape:
            n_elements *= d
        self.tensors[name] = TensorInfo(
            name=name,
            shape=tuple(reversed(shape)),  # ggml order
            dtype=ggml_type,
            offset=0,
            nbytes=nbytes_for(ggml_type, n_elements),
        )
        self.recipes[name] = Recipe(source, rows, cols, op)

    def _build(self) -> None:
        architectures = self.config.get("architectures") or []
        model_type = self.config.get("model_type", "")
        if model_type == "qwen3_5" or any(
            a.startswith("Qwen3_5For") and "Moe" not in a for a in architectures
        ):
            _build_qwen35(self)
        else:
            raise ValueError(
                f"unsupported safetensors checkpoint: model_type={model_type!r} "
                f"architectures={architectures}; convert it to GGUF instead"
            )
        if not self.tensors:
            raise ValueError(
                f"checkpoint has no {self.part} weights"
                + (" (is this a text-only model?)" if self.part == "vision" else "")
            )


# -- Qwen3.5 ---------------------------------------------------------------


def _prefix(index: SafetensorsFile, *candidates: str) -> str:
    """First tensor-name prefix actually present in the checkpoint.

    A Qwen3.5 checkpoint saved as a conditional-generation model nests the
    language model under `model.language_model.`; one saved as a plain causal
    LM puts it at `model.`.
    """
    for candidate in candidates:
        if any(k.startswith(candidate) for k in index.entries):
            return candidate
    raise KeyError(f"checkpoint has none of the expected prefixes: {candidates}")


def _build_qwen35(index: SafetensorsFile) -> None:
    text = {**index.config, **index.config.get("text_config", {})}
    if text.get("num_experts") or text.get("n_routed_experts"):
        raise ValueError("Qwen3.5 support currently requires a dense model such as Qwen3.5-4B")
    if index.part == "vision":
        _build_qwen35_vision(index, text)
        return

    index.metadata["general.architecture"] = "qwen35"
    n_layer = int(text["num_hidden_layers"])
    nk = int(text["linear_num_key_heads"])
    nv = int(text["linear_num_value_heads"])
    hk = int(text["linear_key_head_dim"])
    hv = int(text["linear_value_head_dim"])
    rep = nv // nk
    if nk <= 0 or nv <= 0 or nv % nk:
        raise ValueError("Qwen3.5 value heads must be a multiple of key heads")
    hidden = int(text["hidden_size"])
    head_dim = int(text.get("head_dim") or hidden // int(text["num_attention_heads"]))
    rope = dict(text.get("rope_parameters") or {})
    layer_types = text.get("layer_types")
    interval = int(text.get("full_attention_interval", 4))
    if not layer_types:
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n_layer)
        ]
    if len(layer_types) != n_layer:
        raise ValueError("Qwen3.5 layer_types must match num_hidden_layers")

    key_dim, value_dim = nk * hk, nv * hv
    rope_dim = int(head_dim * float(rope.get("partial_rotary_factor", 0.25)))
    sections = list(rope.get("mrope_section") or [11, 11, 10])

    index.metadata.update({
        # The MTP block is skipped outright, so the block count is the real
        # depth and no nextn metadata is needed.
        "qwen35.block_count": n_layer,
        "qwen35.embedding_length": hidden,
        "qwen35.feed_forward_length": int(text["intermediate_size"]),
        "qwen35.context_length": int(text.get("max_position_embeddings", 262144)),
        "qwen35.attention.head_count": int(text["num_attention_heads"]),
        "qwen35.attention.head_count_kv": int(text["num_key_value_heads"]),
        "qwen35.attention.key_length": head_dim,
        "qwen35.attention.value_length": head_dim,
        "qwen35.attention.layer_norm_rms_epsilon": float(text.get("rms_norm_eps", 1e-6)),
        "qwen35.rope.freq_base": float(rope.get("rope_theta", 10000.0)),
        "qwen35.rope.dimension_count": rope_dim,
        "qwen35.rope.dimension_sections": sections + [0],
        "qwen35.ssm.conv_kernel": int(text["linear_conv_kernel_dim"]),
        "qwen35.ssm.state_size": hk,
        "qwen35.ssm.group_count": nk,
        "qwen35.ssm.time_step_rank": nv,
        "qwen35.ssm.inner_size": value_dim,
        "qwen35.full_attention_interval": interval,
        "qwen35.attention.recurrent_layers": [
            t == "linear_attention" for t in layer_types
        ],
    })
    if str(rope.get("rope_type", "default")) not in ("default", "none"):
        index.metadata["qwen35.rope.scaling.type"] = str(rope["rope_type"])

    lm = _prefix(index, "model.language_model.", "model.")
    index._declare("token_embd.weight", lm + "embed_tokens.weight",
                   index.entries[lm + "embed_tokens.weight"].shape)
    # HF applies (1 + w) in its RMSNorm; ggml multiplies by w directly.
    index._declare("output_norm.weight", lm + "norm.weight", (hidden,), op="shift")
    if "lm_head.weight" in index.entries:
        index._declare("output.weight", "lm_head.weight",
                       index.entries["lm_head.weight"].shape)

    # Value heads are reordered from grouped to tiled everywhere they appear.
    v_rows = tiled_v_perm(nk, rep, hv)
    v_heads = tiled_v_perm(nk, rep, 1)
    qkv_rows = torch.cat([torch.arange(2 * key_dim), 2 * key_dim + v_rows])

    for i, kind in enumerate(layer_types):
        p, src = f"blk.{i}.", f"{lm}layers.{i}."
        index._declare(p + "attn_norm.weight", src + "input_layernorm.weight",
                       (hidden,), op="shift")
        index._declare(p + "post_attention_norm.weight",
                       src + "post_attention_layernorm.weight", (hidden,), op="shift")
        for target, source in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"),
                               ("ffn_down", "down_proj")):
            name = f"{src}mlp.{source}.weight"
            index._declare(p + target + ".weight", name, index.entries[name].shape)
        if kind == "full_attention":
            for target, source in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                                   ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                name = f"{src}self_attn.{source}.weight"
                index._declare(p + target + ".weight", name, index.entries[name].shape)
            for target, source in (("attn_q_norm", "q_norm"), ("attn_k_norm", "k_norm")):
                index._declare(p + target + ".weight",
                               f"{src}self_attn.{source}.weight", (head_dim,), op="shift")
            continue
        la = src + "linear_attn."
        index._declare(p + "attn_qkv.weight", la + "in_proj_qkv.weight",
                       (2 * key_dim + value_dim, hidden), rows=qkv_rows)
        index._declare(p + "attn_gate.weight", la + "in_proj_z.weight",
                       (value_dim, hidden), rows=v_rows)
        index._declare(p + "ssm_beta.weight", la + "in_proj_b.weight",
                       (nv, hidden), rows=v_heads)
        index._declare(p + "ssm_alpha.weight", la + "in_proj_a.weight",
                       (nv, hidden), rows=v_heads)
        # out_proj is reordered along its input dimension, i.e. within a row.
        index._declare(p + "ssm_out.weight", la + "out_proj.weight",
                       (hidden, value_dim), cols=v_rows)
        conv = index.entries[la + "conv1d.weight"]
        index._declare(p + "ssm_conv1d.weight", la + "conv1d.weight",
                       (conv.shape[0], conv.shape[-1]), rows=qkv_rows)
        # A 1-D parameter is a single row, so its heads permute within it.
        index._declare(p + "ssm_a", la + "A_log", (nv,), cols=v_heads, op="neg_exp")
        index._declare(p + "ssm_dt.bias", la + "dt_bias", (nv,), cols=v_heads)
        index._declare(p + "ssm_norm.weight", la + "norm.weight", (hv,))

    _read_tokenizer(index, text)


def _build_qwen35_vision(index: SafetensorsFile, text: dict) -> None:
    vision = index.config.get("vision_config")
    if not vision:
        raise ValueError("checkpoint has no vision_config; it is a text-only model")
    visual = _prefix(index, "model.visual.", "visual.")
    hidden = int(vision["hidden_size"])
    depth = int(vision["depth"])
    patch = int(vision["patch_size"])
    temporal = int(vision.get("temporal_patch_size", 2))
    n_pos = int(vision.get("num_position_embeddings", 2304))
    merged = hidden * int(vision.get("spatial_merge_size", 2)) ** 2

    index.metadata.update({
        "general.architecture": "clip",
        "clip.vision.block_count": depth,
        "clip.vision.embedding_length": hidden,
        "clip.vision.attention.head_count": int(vision["num_heads"]),
        "clip.vision.feed_forward_length": int(vision["intermediate_size"]),
        "clip.vision.patch_size": patch,
        "clip.vision.image_size": int(n_pos ** 0.5) * patch,
        "clip.vision.spatial_merge_size": int(vision.get("spatial_merge_size", 2)),
        "clip.vision.projection_dim": int(vision["out_hidden_size"]),
        # llama.cpp reuses the text model's epsilon for the vision norms.
        "clip.vision.attention.layer_norm_epsilon": float(text.get("rms_norm_eps", 1e-6)),
        "clip.vision.is_deepstack_layers": [
            i in (vision.get("deepstack_visual_indexes") or []) for i in range(depth)
        ],
    })
    if vision.get("deepstack_visual_indexes"):
        raise ValueError(
            "this Qwen3.5 checkpoint uses DeepStack visual layers, which are not implemented"
        )

    # A Conv3d over `temporal` frames becomes one Conv2d kernel per frame.
    proj = visual + "patch_embed.proj.weight"
    channels = index.entries[proj].shape[1]
    rows = torch.arange(hidden * channels * patch, dtype=torch.long)
    for t in range(temporal):
        frame = ((rows // patch) * temporal + t) * patch + rows % patch
        suffix = ".weight" if t == 0 else f".weight.{t}"
        index._declare("v.patch_embd" + suffix, proj,
                       (hidden, channels, patch, patch), rows=frame)
    index._declare("v.patch_embd.bias", visual + "patch_embed.proj.bias", (hidden,))
    index._declare("v.position_embd.weight", visual + "pos_embed.weight",
                   (n_pos, hidden))
    for i in range(depth):
        p, src = f"v.blk.{i}.", f"{visual}blocks.{i}."
        for target, source in (("ln1", "norm1"), ("ln2", "norm2"),
                               ("attn_qkv", "attn.qkv"), ("attn_out", "attn.proj"),
                               ("ffn_up", "mlp.linear_fc1"),
                               ("ffn_down", "mlp.linear_fc2")):
            for suffix in (".weight", ".bias"):
                name = src + source + suffix
                index._declare(p + target + suffix, name, index.entries[name].shape)
    # The merger's own LayerNorm is the tower's post-norm in GGUF naming.
    for suffix in (".weight", ".bias"):
        index._declare("v.post_ln" + suffix, visual + "merger.norm" + suffix,
                       index.entries[visual + "merger.norm" + suffix].shape)
        for idx, source in ((0, "linear_fc1"), (2, "linear_fc2")):
            name = f"{visual}merger.{source}{suffix}"
            index._declare(f"mm.{idx}{suffix}", name, index.entries[name].shape)
    del merged


# -- tokenizer -------------------------------------------------------------


def _read_tokenizer(index: SafetensorsFile, text: dict) -> None:
    """Fill in the `tokenizer.ggml.*` keys `GGUFTokenizer` reads.

    The vocabulary, merges and chat template all live beside the weights, so
    no HuggingFace tokenizer library is needed - the same reason the GGUF path
    does not need one either.
    """
    path = index.root / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"no tokenizer.json beside the weights in {index.root}")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model") or {}
    vocab = model.get("vocab") or {}
    if not vocab:
        raise ValueError("tokenizer.json has no BPE vocabulary")

    size = max(max(vocab.values()), *(t["id"] for t in data.get("added_tokens") or [0])) + 1 \
        if data.get("added_tokens") else max(vocab.values()) + 1
    tokens = [""] * size
    types = [1] * size
    for token, i in vocab.items():
        tokens[i] = token
    for added in data.get("added_tokens") or []:
        tokens[added["id"]] = added["content"]
        types[added["id"]] = 3 if added.get("special") else 4

    merges = []
    for pair in model.get("merges") or []:
        merges.append(" ".join(pair) if isinstance(pair, (list, tuple)) else str(pair))

    config_path = index.root / "tokenizer_config.json"
    tok_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file() else {}
    )

    def token_id(value):
        if isinstance(value, dict):
            value = value.get("content")
        return vocab.get(value, next(
            (t["id"] for t in data.get("added_tokens") or []
             if t["content"] == value), None))

    index.metadata.update({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
    })
    eos = text.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0]
    if eos is None:
        eos = token_id(tok_config.get("eos_token"))
    if eos is not None:
        index.metadata["tokenizer.ggml.eos_token_id"] = int(eos)
    bos = text.get("bos_token_id", tok_config.get("bos_token"))
    bos = token_id(bos) if not isinstance(bos, int) else bos
    if bos is not None:
        index.metadata["tokenizer.ggml.bos_token_id"] = int(bos)

    template_path = index.root / "chat_template.jinja"
    template = (
        template_path.read_text(encoding="utf-8") if template_path.is_file()
        else tok_config.get("chat_template")
    )
    if isinstance(template, list):  # some repos ship a list of named templates
        template = next((t.get("template") for t in template
                         if t.get("name") == "default"), template[0].get("template"))
    if template:
        index.metadata["tokenizer.chat_template"] = template


def looks_like_safetensors(path: str | Path) -> bool:
    """True when `path` names a checkpoint directory, shard, or index."""
    path = Path(path)
    if path.is_dir():
        return any(path.glob("*.safetensors"))
    if path.suffix == ".safetensors":
        return True
    return path.name.endswith("index.json") and path.is_file()
