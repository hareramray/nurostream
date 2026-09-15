"""Qwen3.5 loaded straight from a HuggingFace checkpoint, versus Transformers.

Run: python tests/test_qwen35_safetensors.py

No GGUF anywhere: the checkpoint is written with the reference `safetensors`
writer, so this exercises the index parser against an independent producer.
What is really under test is the translation in format/safetensors.py — the
+1 RMSNorm shift, the negated A_log, and the tiled value heads — because a
checkpoint that skipped any of them would load happily and answer wrongly.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurostream.api import NeuroStream
from neurostream.format.gguf import GGMLType
from neurostream.format.safetensors import SafetensorsFile, looks_like_safetensors
from neurostream.model.qwen3 import KVCache
from neurostream.vision.prompt import position_ids
from neurostream.vision.tower import Qwen35VisionTower

from test_qwen35_vision import (
    GRID_H, GRID_W, IMAGE_TOKEN, MERGE, image_and_tokens, tiny_multimodal, vocabulary,
)

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised by the skip
    save_file = None


def write_checkpoint(root: Path, model, text_cfg, shards=1, dtype=torch.float32):
    """Save the reference model the way the Hugging Face repo ships it."""
    root.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(dtype).contiguous()
             for k, v in model.full.state_dict().items()}
    # Tied checkpoints omit lm_head entirely, as Qwen3.5-4B does; safetensors
    # refuses shared storage anyway.
    state.pop('lm_head.weight', None)
    # The MTP block is present in the real repo and must be ignored.
    state['mtp.layers.0.mlp.down_proj.weight'] = torch.zeros(
        text_cfg.hidden_size, text_cfg.intermediate_size, dtype=dtype)
    state['mtp.fc.weight'] = torch.zeros(text_cfg.hidden_size, 2 * text_cfg.hidden_size,
                                         dtype=dtype)

    names = sorted(state)
    groups = [names[i::shards] for i in range(shards)]
    weight_map = {}
    for i, group in enumerate(groups, start=1):
        fname = (f'model.safetensors-{i:05d}-of-{shards:05d}.safetensors'
                 if shards > 1 else 'model.safetensors')
        save_file({k: state[k] for k in group}, root / fname)
        weight_map.update({k: fname for k in group})
    if shards > 1:
        (root / 'model.safetensors.index.json').write_text(json.dumps(
            {'metadata': {'total_size': sum(v.numel() * v.element_size()
                                            for v in state.values())},
             'weight_map': weight_map}), encoding='utf-8')

    config = model.full.config.to_dict()
    config['dtype'] = str(dtype).replace('torch.', '')
    (root / 'config.json').write_text(json.dumps(config, default=str), encoding='utf-8')
    write_tokenizer(root, text_cfg.vocab_size)
    return root


def write_tokenizer(root: Path, size):
    tokens, types = vocabulary(size)
    added = [{'id': i, 'content': t, 'special': True}
             for i, (t, ty) in enumerate(zip(tokens, types)) if ty == 3]
    (root / 'tokenizer.json').write_text(json.dumps({
        'added_tokens': added,
        'model': {'type': 'BPE',
                  'vocab': {t: i for i, t in enumerate(tokens)},
                  'merges': [['x3', 'x4']]},
    }), encoding='utf-8')
    (root / 'tokenizer_config.json').write_text(json.dumps({
        'eos_token': '<|im_end|>', 'bos_token': None,
    }), encoding='utf-8')
    (root / 'chat_template.jinja').write_text(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        encoding='utf-8')


@unittest.skipIf(save_file is None, 'safetensors writer unavailable')
class SafetensorsCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.text_cfg, cls.vision_cfg = tiny_multimodal()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = write_checkpoint(Path(cls.tmp.name) / 'hf', cls.model, cls.text_cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, root=None, **kw):
        kw.setdefault('mem_budget', 65536)
        return NeuroStream.load(root or self.root, device='cpu', **kw)

    def test_index_is_lazy_and_translates_names(self):
        index = SafetensorsFile(self.root)
        self.assertTrue(looks_like_safetensors(self.root))
        self.assertEqual(index.arch(), 'qwen35')
        self.assertEqual(int(index.cfg('block_count')), self.text_cfg.num_hidden_layers)
        # MTP and vision tensors stay out of the text index.
        self.assertFalse([n for n in index.tensors if n.startswith(('v.', 'mm.'))])
        self.assertNotIn('output.weight', index.tensors)  # tied
        self.assertIn('blk.0.ssm_a', index.tensors)
        self.assertIn('blk.3.attn_q.weight', index.tensors)
        self.assertNotIn('blk.3.ssm_a', index.tensors)
        self.assertEqual(index.tensors['token_embd.weight'].torch_shape,
                         (self.text_cfg.vocab_size, self.text_cfg.hidden_size))
        self.assertEqual(index.tensors['blk.0.ssm_a'].dtype, GGMLType.F32)
        vision = SafetensorsFile(self.root, part='vision')
        self.assertEqual(vision.arch(), 'clip')
        self.assertIn('v.patch_embd.weight.1', vision.tensors)
        self.assertIn('mm.2.bias', vision.tensors)

    def test_translation_applies_converter_fixups(self):
        """The three transforms a naive reader would skip."""
        from neurostream.io.safetensors import SafetensorsSource
        from neurostream.format.safetensors import tiled_v_perm

        layer = self.model.full.model.language_model.layers
        with SafetensorsSource(self.root) as src:
            got = src.fetch('blk.0.attn_norm.weight')
            torch.testing.assert_close(got, layer[0].input_layernorm.weight + 1.0)
            # The gated SSM norm is the one norm that is NOT shifted.
            torch.testing.assert_close(src.fetch('blk.0.ssm_norm.weight'),
                                       layer[0].linear_attn.norm.weight)
            torch.testing.assert_close(src.fetch('blk.0.ssm_a'),
                                       -torch.exp(layer[0].linear_attn.A_log)[
                                           tiled_v_perm(2, 2, 1)])
            # Value heads arrive tiled by key head, not grouped.
            perm = tiled_v_perm(2, 2, self.vision_cfg.hidden_size // 4)
            expected = layer[0].linear_attn.in_proj_z.weight[
                tiled_v_perm(2, 2, self.text_cfg.linear_value_head_dim)]
            torch.testing.assert_close(src.fetch('blk.0.attn_gate.weight'), expected)
            del perm

    def test_row_blocks_match_whole_tensor_reads(self):
        """Streaming a permuted tensor by blocks must equal reading it whole."""
        from neurostream.io.safetensors import SafetensorsSource

        with SafetensorsSource(self.root) as src:
            for name in ('blk.0.attn_qkv.weight', 'blk.0.ssm_out.weight',
                         'blk.0.attn_gate.weight', 'blk.0.ssm_conv1d.weight',
                         'blk.3.attn_q.weight'):
                whole = src.fetch(name).reshape(-1, src.gguf.tensors[name].torch_shape[-1])
                for block in (1, 3, 7):
                    pieces = [src.fetch_rows(name, i, i + block)
                              for i in range(0, whole.shape[0], block)]
                    with self.subTest(name=name, block=block):
                        torch.testing.assert_close(torch.cat(pieces), whole)

    def test_text_logits_match_transformers(self):
        ids = [1, 5, 6, 7, 8, 2]
        with torch.no_grad():
            expected = self.model.full(input_ids=torch.tensor([ids])).logits[0]
        for block_rows in (0, 3):
            with self.subTest(block_rows=block_rows), \
                    self.load(block_rows=block_rows) as ns, torch.no_grad():
                actual = ns.model.forward(torch.tensor(ids), KVCache(ns.cfg.n_layer),
                                          last_only=False)
                torch.testing.assert_close(actual, expected.float(),
                                           atol=2e-5, rtol=2e-5)

    def test_tokenizer_and_chat_template_come_from_the_checkpoint(self):
        with self.load() as ns:
            self.assertEqual(ns.tokenizer.encode('<|im_start|>'), [1])
            self.assertEqual(ns.tokenizer.stop_ids, {0, 2})
            text = ns.tokenizer.apply_chat_template([{'role': 'user', 'content': 'x5'}])
            self.assertTrue(text.endswith('<|im_start|>assistant\n'))

    def test_vision_and_multimodal_prefill_match_transformers(self):
        pixels, flat, grid, ids = image_and_tokens()
        tokens = torch.tensor([ids])
        with torch.no_grad():
            expected = self.model.full(
                input_ids=tokens, pixel_values=flat, image_grid_thw=grid,
                mm_token_type_ids=(tokens == IMAGE_TOKEN).int(),
            ).logits[0]
        with self.load() as ns, torch.no_grad():
            # No mmproj argument: the tower is inside the same checkpoint.
            ns.attach_vision()
            self.assertIsInstance(ns.tower, Qwen35VisionTower)
            ns.tower.preprocess = lambda image, max_patches=0: (
                image.to(ns.tower.device, ns.tower.dtype), GRID_H, GRID_W)
            embeds, taps, mh, mw = ns.tower.encode(pixels)
            self.assertEqual(taps, [])
            want = self.model.full.model.visual(
                hidden_states=flat, grid_thw=grid).pooler_output
            torch.testing.assert_close(embeds.float(), want.float(), atol=2e-5, rtol=2e-5)

            x = ns.model.embed(torch.tensor(ids)).clone()
            x[torch.tensor(ids) == IMAGE_TOKEN] = embeds.to(x.dtype)
            pos = position_ids(ids, IMAGE_TOKEN, mh, mw, ns.device)
            actual = ns.model.forward(inputs_embeds=x, kv=KVCache(ns.cfg.n_layer),
                                      start_pos=0, pos_ids=pos, last_only=False)
            torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)

    def test_sharded_checkpoint_matches_single_file(self):
        root = write_checkpoint(Path(self.tmp.name) / 'sharded', self.model,
                                self.text_cfg, shards=3)
        index = SafetensorsFile(root)
        self.assertEqual(len(index.paths), 3)
        ids = [1, 5, 6, 7, 8, 2]
        with torch.no_grad():
            expected = self.model.full(input_ids=torch.tensor([ids])).logits[0]
        with self.load(root) as ns, torch.no_grad():
            actual = ns.model.forward(torch.tensor(ids), KVCache(ns.cfg.n_layer),
                                      last_only=False)
            torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)

    def test_bf16_checkpoint_streams_and_caches(self):
        root = write_checkpoint(Path(self.tmp.name) / 'bf16', self.model,
                                self.text_cfg, dtype=torch.bfloat16)
        index = SafetensorsFile(root)
        self.assertEqual(index.tensors['token_embd.weight'].dtype, GGMLType.BF16)
        ids = [1, 5, 6, 7, 8, 2]
        reference = tiny_multimodal()[0]
        # Compare against the same rounded weights the checkpoint now holds.
        with torch.no_grad():
            state = {k: v.to(torch.bfloat16).float()
                     for k, v in self.model.full.state_dict().items()}
            reference.full.load_state_dict(state)
            expected = reference.full(input_ids=torch.tensor([ids])).logits[0]
        for ram in (0, 1 << 20):
            with self.subTest(ram=ram), self.load(root, ram_budget=ram, block_rows=3) as ns, \
                    torch.no_grad():
                actual = ns.model.forward(torch.tensor(ids), KVCache(ns.cfg.n_layer),
                                          last_only=False)
                torch.testing.assert_close(actual, expected.float(), atol=3e-3, rtol=3e-3)
                if ram:
                    self.assertGreater(ns.cache.stats.ram_hits, 0)

    def test_rejects_unsupported_checkpoints(self):
        root = Path(self.tmp.name) / 'other'
        root.mkdir()
        save_file({'x': torch.zeros(4)}, root / 'model.safetensors')
        (root / 'config.json').write_text(json.dumps(
            {'model_type': 'llama', 'architectures': ['LlamaForCausalLM']}),
            encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unsupported safetensors checkpoint'):
            SafetensorsFile(root)
        (root / 'config.json').write_text(json.dumps(
            {'model_type': 'qwen3_5', 'text_config': {'num_experts': 128}}),
            encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'dense model'):
            SafetensorsFile(root)
        with self.assertRaises(FileNotFoundError):
            SafetensorsFile(Path(self.tmp.name) / 'nowhere')


if __name__ == '__main__':
    unittest.main()
