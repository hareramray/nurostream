"""Qwen3.5 image input versus Transformers; tiny GGUFs require no model download.

Run: python tests/test_qwen35_vision.py

Covers the three places vision can silently go wrong: the tower's own maths,
the interleaved M-RoPE tables that give image tokens a (t, h, w) position, and
the splice of merged patches into the prompt. The mmproj is written the way
llama.cpp's conversion/qwen3vl.py writes one, including the Conv3d patch
embedding split into two Conv2d kernels along the temporal axis.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import gguf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurostream.api import NeuroStream
from neurostream.compute.ops import mrope_tables
from neurostream.model.qwen3 import KVCache
from neurostream.vision.prompt import IMAGE_PAD, VISION_END, VISION_START, position_ids
from neurostream.vision.tower import Qwen35VisionTower

from test_qwen35 import write_reference

# Grid chosen so the learned 6x6 position table is resampled on one axis only
# in height and stretched in width: a square-only path would still pass.
GRID_H, GRID_W = 6, 8
PATCH, MERGE, POS_SIDE = 4, 2, 6
IMAGE_TOKEN, VISION_START_ID, VISION_END_ID = 60, 58, 59


def tiny_multimodal():
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
    )

    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256,
        layer_types=['linear_attention'] * 3 + ['full_attention'],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={'rope_type': 'default', 'rope_theta': 1e6,
                         'partial_rotary_factor': 0.25, 'mrope_section': [2, 1, 1],
                         'mrope_interleaved': True},
        tie_word_embeddings=True,
    )
    vision = Qwen3_5VisionConfig(
        depth=2, hidden_size=32, num_heads=2, intermediate_size=64,
        patch_size=PATCH, spatial_merge_size=MERGE, temporal_patch_size=2,
        out_hidden_size=text.hidden_size, num_position_embeddings=POS_SIDE ** 2,
        in_channels=3,
    )
    config = Qwen3_5Config(
        text_config=text, vision_config=vision, image_token_id=IMAGE_TOKEN,
        video_token_id=61, vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=text.tie_word_embeddings,
    )
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    model = Qwen3_5ForCausalLMShim(Qwen3_5ForConditionalGeneration(config).eval())
    with torch.no_grad():
        for name, param in model.full.named_parameters():
            if name.endswith('norm.weight') and 'visual' not in name:
                param.uniform_(-0.3, 0.3)
            elif name.endswith('.dt_bias'):
                param.uniform_(-1.0, 1.0)
            elif 'visual' in name and param.dim() == 1 and 'norm' in name:
                # LayerNorm weights near 1, biases small but non-zero.
                param.uniform_(0.7, 1.3) if name.endswith('weight') else param.uniform_(-0.2, 0.2)
    return model, text, vision


class Qwen3_5ForCausalLMShim:
    """Presents the language model where `write_reference` expects to find it."""

    def __init__(self, full):
        self.full = full
        self.model = full.model.language_model
        self.lm_head = full.lm_head


def vocabulary(size=64):
    tokens = [f'x{i}' for i in range(size)]
    specials = {0: '<|endoftext|>', 1: '<|im_start|>', 2: '<|im_end|>',
                VISION_START_ID: VISION_START, VISION_END_ID: VISION_END,
                IMAGE_TOKEN: IMAGE_PAD}
    types = [1] * size
    for i, token in specials.items():
        tokens[i], types[i] = token, 3
    return tokens, types


def write_mmproj(path, visual, v, eps=1e-6):
    """Write the vision tower as llama.cpp's Qwen3VLVisionModel converter does."""
    writer = gguf.GGUFWriter(path, 'clip')
    writer.add_uint32('clip.vision.block_count', v.depth)
    writer.add_uint32('clip.vision.embedding_length', v.hidden_size)
    writer.add_uint32('clip.vision.attention.head_count', v.num_heads)
    writer.add_uint32('clip.vision.feed_forward_length', v.intermediate_size)
    writer.add_uint32('clip.vision.patch_size', v.patch_size)
    # Derived the way the converter derives it, from the position table.
    writer.add_uint32('clip.vision.image_size',
                      int(v.num_position_embeddings ** 0.5) * v.patch_size)
    writer.add_uint32('clip.vision.spatial_merge_size', v.spatial_merge_size)
    writer.add_uint32('clip.vision.projection_dim', v.out_hidden_size)
    writer.add_float32('clip.vision.attention.layer_norm_epsilon', eps)
    writer.add_array('clip.vision.is_deepstack_layers', [False] * v.depth)

    def add(name, tensor):
        writer.add_tensor(name, tensor.detach().float().cpu().numpy().copy())

    # A Conv3d over two temporal slices becomes two Conv2d kernels.
    proj = visual.patch_embed.proj
    add('v.patch_embd.weight', proj.weight[:, :, 0])
    add('v.patch_embd.weight.1', proj.weight[:, :, 1])
    add('v.patch_embd.bias', proj.bias)
    add('v.position_embd.weight', visual.pos_embed.weight)
    for i, blk in enumerate(visual.blocks):
        p = f'v.blk.{i}.'
        for target, module in {'ln1': blk.norm1, 'ln2': blk.norm2,
                               'attn_qkv': blk.attn.qkv, 'attn_out': blk.attn.proj,
                               'ffn_up': blk.mlp.linear_fc1,
                               'ffn_down': blk.mlp.linear_fc2}.items():
            add(p + target + '.weight', module.weight)
            add(p + target + '.bias', module.bias)
    # The merger's own LayerNorm is stored as the tower's post-norm.
    add('v.post_ln.weight', visual.merger.norm.weight)
    add('v.post_ln.bias', visual.merger.norm.bias)
    for idx, module in ((0, visual.merger.linear_fc1), (2, visual.merger.linear_fc2)):
        add(f'mm.{idx}.weight', module.weight)
        add(f'mm.{idx}.bias', module.bias)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def patchify(pixels, patch=PATCH, merge=MERGE, temporal=2):
    """Pixels -> the flat per-patch rows Transformers' vision model consumes.

    Patches come out in spatial-merge-block order, and a still image is
    repeated across the temporal axis.
    """
    _, channels, height, width = pixels.shape
    gh, gw = height // patch, width // patch
    x = pixels.reshape(1, channels, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flat = (x.unsqueeze(6)
             .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
             .reshape(gh * gw, channels * temporal * patch * patch))
    return flat, gh, gw


def image_and_tokens(seed=7):
    torch.manual_seed(seed)
    pixels = torch.randn(1, 3, GRID_H * PATCH, GRID_W * PATCH)
    flat, gh, gw = patchify(pixels)
    n_image = (gh // MERGE) * (gw // MERGE)
    ids = [1, 5, VISION_START_ID] + [IMAGE_TOKEN] * n_image + [VISION_END_ID, 6, 7, 2]
    return pixels, flat, torch.tensor([[1, gh, gw]]), ids


class Qwen35VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.text_cfg, cls.vision_cfg = tiny_multimodal()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.text_path, cls.mmproj_path = root / 'text.gguf', root / 'mmproj.gguf'
        write_reference(cls.text_path, cls.model, cls.text_cfg,
                        vocab=vocabulary(cls.text_cfg.vocab_size))
        write_mmproj(cls.mmproj_path, cls.model.full.model.visual, cls.vision_cfg,
                     eps=cls.text_cfg.rms_norm_eps)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, **kw):
        ns = NeuroStream.load(self.text_path, device='cpu', mem_budget=65536,
                              n_workers=2, **kw)
        ns.attach_vision(self.mmproj_path)
        return ns

    def test_tower_matches_transformers(self):
        pixels, flat, grid, _ = image_and_tokens()
        with self.load() as ns, torch.no_grad():
            self.assertIsInstance(ns.tower, Qwen35VisionTower)
            # Feed the tower the exact pixels Transformers sees.
            ns.tower.preprocess = lambda image, max_patches=0: (
                image.to(ns.tower.device, ns.tower.dtype), GRID_H, GRID_W)
            embeds, taps, mh, mw = ns.tower.encode(pixels)
            expected = self.model.full.model.visual(
                hidden_states=flat, grid_thw=grid).pooler_output
            self.assertEqual(taps, [])
            self.assertEqual((mh, mw), (GRID_H // MERGE, GRID_W // MERGE))
            self.assertEqual(embeds.shape, expected.shape)
            torch.testing.assert_close(embeds.float(), expected.float(),
                                       atol=2e-5, rtol=2e-5)

    def test_interleaved_mrope_matches_transformers(self):
        rotary = self.model.full.model.language_model.rotary_emb
        pos = torch.tensor([[0, 1, 2, 2, 2, 3], [0, 1, 2, 2, 3, 3], [0, 1, 2, 3, 2, 3]])
        self.assertEqual(self.text_cfg.rope_parameters['mrope_section'], [2, 1, 1])
        rope_dim = int(self.text_cfg.head_dim
                       * self.text_cfg.rope_parameters['partial_rotary_factor'])
        cos, sin = mrope_tables(
            rope_dim, pos, self.text_cfg.rope_parameters['rope_theta'],
            tuple(self.text_cfg.rope_parameters['mrope_section']) + (0,),
            torch.device('cpu'), torch.float32, interleaved=True,
        )
        with torch.no_grad():
            want_cos, want_sin = rotary(torch.zeros(1, pos.shape[1], 1),
                                        pos[:, None, :])
        torch.testing.assert_close(cos, want_cos[0].float(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sin, want_sin[0].float(), atol=1e-6, rtol=1e-6)
        # Contiguous sections would give a different table for the same ids.
        flat_cos, _ = mrope_tables(
            rope_dim, pos, self.text_cfg.rope_parameters['rope_theta'],
            tuple(self.text_cfg.rope_parameters['mrope_section']) + (0,),
            torch.device('cpu'), torch.float32, interleaved=False,
        )
        self.assertFalse(torch.allclose(flat_cos, cos))

    def test_position_ids_match_get_rope_index(self):
        _, _, grid, ids = image_and_tokens()
        merged_h, merged_w = GRID_H // MERGE, GRID_W // MERGE
        actual = position_ids(ids, IMAGE_TOKEN, merged_h, merged_w, torch.device('cpu'))
        tokens = torch.tensor([ids])
        expected, _ = self.model.full.model.get_rope_index(
            tokens, (tokens == IMAGE_TOKEN).int(), image_grid_thw=grid)
        torch.testing.assert_close(actual, expected[:, 0, :].to(actual.dtype))

    def test_multimodal_prefill_matches_transformers(self):
        pixels, flat, grid, ids = image_and_tokens()
        tokens = torch.tensor([ids])
        with torch.no_grad():
            expected = self.model.full(
                input_ids=tokens, pixel_values=flat, image_grid_thw=grid,
                mm_token_type_ids=(tokens == IMAGE_TOKEN).int(),
            ).logits[0]
        for block_rows in (0, 3):
            with self.subTest(block_rows=block_rows), \
                    self.load(block_rows=block_rows) as ns, torch.no_grad():
                ns.tower.preprocess = lambda image, max_patches=0: (
                    image.to(ns.tower.device, ns.tower.dtype), GRID_H, GRID_W)
                embeds, _, mh, mw = ns.tower.encode(pixels)
                x = ns.model.embed(torch.tensor(ids)).clone()
                mask = torch.tensor(ids) == IMAGE_TOKEN
                x[mask] = embeds.to(x.dtype)
                pos = position_ids(ids, IMAGE_TOKEN, mh, mw, ns.device)
                actual = ns.model.forward(inputs_embeds=x, kv=KVCache(ns.cfg.n_layer),
                                          start_pos=0, pos_ids=pos, last_only=False)
                torch.testing.assert_close(actual, expected.float(),
                                           atol=2e-5, rtol=2e-5)

    def test_decode_after_image_continues_positions(self):
        """A token generated after the image must not reuse image positions."""
        pixels, flat, grid, ids = image_and_tokens()
        nxt = 9
        tokens = torch.tensor([ids + [nxt]])
        with torch.no_grad():
            expected = self.model.full(
                input_ids=tokens, pixel_values=flat, image_grid_thw=grid,
                mm_token_type_ids=(tokens == IMAGE_TOKEN).int(),
            ).logits[0, -1:]
        with self.load() as ns, torch.no_grad():
            ns.tower.preprocess = lambda image, max_patches=0: (
                image.to(ns.tower.device, ns.tower.dtype), GRID_H, GRID_W)
            embeds, _, mh, mw = ns.tower.encode(pixels)
            x = ns.model.embed(torch.tensor(ids)).clone()
            x[torch.tensor(ids) == IMAGE_TOKEN] = embeds.to(x.dtype)
            pos = position_ids(ids, IMAGE_TOKEN, mh, mw, ns.device)
            kv = KVCache(ns.cfg.n_layer)
            ns.model.forward(inputs_embeds=x, kv=kv, start_pos=0, pos_ids=pos)
            from neurostream.vision.prompt import next_position
            step = next_position(pos)
            actual = ns.model.forward(torch.tensor([nxt]), kv, start_pos=kv.length,
                                      pos_ids=step)
            torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)

    def test_generate_vl_runs_and_stops(self):
        pixels, _, _, _ = image_and_tokens()
        with self.load() as ns:
            ns.tower.preprocess = lambda image, max_patches=0: (
                image.to(ns.tower.device, ns.tower.dtype), GRID_H, GRID_W)
            pieces = list(ns.generate_vl(pixels, 'x5', max_tokens=4,
                                         enable_thinking=False))
            self.assertLessEqual(ns.stats.generated_tokens, 4)
            self.assertEqual(len(''.join(pieces)) > 0, ns.stats.generated_tokens > 0)
            self.assertGreater(ns.stats.prompt_tokens, 12)

    def test_rejects_bad_position_ids_and_deepstack(self):
        with self.load() as ns, torch.no_grad():
            with self.assertRaisesRegex(ValueError, 'shape'):
                ns.model.forward(torch.tensor([1, 2]), KVCache(ns.cfg.n_layer),
                                 pos_ids=torch.zeros(3, 5, dtype=torch.long))
            with self.assertRaisesRegex(ValueError, 'DeepStack'):
                ns.model.forward(torch.tensor([1]), KVCache(ns.cfg.n_layer),
                                 deepstack=[torch.zeros(1, ns.cfg.n_embd)])


if __name__ == '__main__':
    unittest.main()
