"""Qwen3.5 text math versus Transformers; tiny GGUFs require no model download.

Run: python tests/test_qwen35.py

The tiny GGUFs are written the way llama.cpp's conversion/qwen.py writes a real
one: RMSNorm weights pre-shifted by 1 (except the gated SSM norm), A_log stored
as -exp(A_log), and linear-attention value heads reordered from HF's grouped
layout to the tiled layout ggml broadcasts against. A test that skipped those
transforms would only prove the engine agrees with itself.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurostream.api import NeuroStream
from neurostream.format.gguf import GGMLType, GGUFFile
from neurostream.format.tokenizer import GGUFTokenizer, bytes_to_unicode
from neurostream.model.qwen3 import KVCache
from neurostream.model.qwen35 import Qwen35Config
from neurostream.residency.planner import plan

# 3 recurrent layers then 1 full-attention layer, the real interval-4 pattern.
LAYER_TYPES = ['linear_attention'] * 3 + ['full_attention']


def tiny_reference(**kwargs):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, layer_types=LAYER_TYPES,
        # Fewer key heads than value heads is what forces the tiled reorder.
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={'rope_type': 'default', 'rope_theta': 1e6,
                         'partial_rotary_factor': 0.25, 'mrope_section': [1, 1, 0],
                         'mrope_interleaved': True},
        **kwargs,
    )
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    model = Qwen3_5ForCausalLM(config).eval()
    # Non-zero norm weights expose a missing +1 shift; non-trivial dt_bias and
    # A_log expose a decay term that is dropped or applied in the wrong order.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('norm.weight'):
                param.uniform_(-0.3, 0.3)
            elif name.endswith('.dt_bias'):
                param.uniform_(-1.0, 1.0)
    return model, config


def reorder_v(t, dim, groups, per_group, head_dim):
    """Regroup the value-head axis, llama.cpp's _reorder_v_heads."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    t = t.reshape(*shape[:dim], groups, per_group, head_dim, *shape[dim + 1:])
    perm = list(range(t.dim()))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def write_reference(path, model, c, quantized=False,
                    quant_type=gguf.GGMLQuantizationType.Q8_0,
                    storage_type=gguf.GGMLQuantizationType.F32, n_mtp=0,
                    vocab=None):
    writer = gguf.GGUFWriter(path, 'qwen35')
    nk, nv = c.linear_num_key_heads, c.linear_num_value_heads
    hk, hv = c.linear_key_head_dim, c.linear_value_head_dim
    rep, key_dim = nv // nk, hk * c.linear_num_key_heads
    integers = {
        # The converter counts the MTP block in block_count, so the real depth
        # is block_count - nextn_predict_layers.
        'block_count': c.num_hidden_layers + n_mtp,
        'embedding_length': c.hidden_size,
        'feed_forward_length': c.intermediate_size,
        'context_length': c.max_position_embeddings,
        'attention.head_count': c.num_attention_heads,
        'attention.head_count_kv': c.num_key_value_heads,
        'attention.key_length': c.head_dim, 'attention.value_length': c.head_dim,
        'rope.dimension_count': int(c.head_dim * c.rope_parameters['partial_rotary_factor']),
        'ssm.conv_kernel': c.linear_conv_kernel_dim, 'ssm.state_size': hk,
        'ssm.group_count': nk, 'ssm.time_step_rank': nv, 'ssm.inner_size': hv * nv,
        'full_attention_interval': 4,
    }
    if n_mtp:
        integers['nextn_predict_layers'] = n_mtp
    for key, value in integers.items():
        writer.add_uint32('qwen35.' + key, value)
    for key, value in {'attention.layer_norm_rms_epsilon': c.rms_norm_eps,
                       'rope.freq_base': c.rope_parameters['rope_theta']}.items():
        writer.add_float32('qwen35.' + key, value)
    writer.add_array('qwen35.attention.recurrent_layers',
                     [t == 'linear_attention' for t in c.layer_types] + [False] * n_mtp)
    writer.add_array('qwen35.rope.dimension_sections',
                     list(c.rope_parameters['mrope_section']) + [0])

    def store(name, tensor):
        """Write a tensor and return the values the file actually holds."""
        array = tensor.detach().float().cpu().numpy().copy()
        # Quantized projections beside F32 norms and conv kernels, as in a real
        # mixed-type GGUF. Callers compare against exactly these values.
        for enabled, dtype in ((quantized, quant_type),
                               (storage_type != gguf.GGMLQuantizationType.F32, storage_type)):
            if enabled and array.ndim == 2 and array.shape[-1] % 32 == 0:
                raw = gguf.quantize(array, dtype)
                writer.add_tensor(name, raw, raw_dtype=dtype)
                return torch.from_numpy(gguf.dequantize(raw, dtype))
        writer.add_tensor(name, array)
        return tensor.detach().float()

    def add(name, param):
        with torch.no_grad():
            param.copy_(store(name, param))

    def add_norm(name, param):
        # HF applies (1 + w); ggml multiplies by w, so the file carries w + 1.
        with torch.no_grad():
            param.copy_(store(name, param + 1.0) - 1.0)

    def add_reordered(name, param, dim, head_dim):
        """Store in tiled value-head order, restore HF's grouped order."""
        with torch.no_grad():
            stored = store(name, reorder_v(param, dim, nk, rep, head_dim))
            param.copy_(reorder_v(stored, dim, rep, nk, head_dim))

    add('token_embd.weight', model.model.embed_tokens.weight)
    if not c.tie_word_embeddings:
        add('output.weight', model.lm_head.weight)
    add_norm('output_norm.weight', model.model.norm.weight)
    for i, layer in enumerate(model.model.layers):
        p = f'blk.{i}.'
        add_norm(p + 'attn_norm.weight', layer.input_layernorm.weight)
        add_norm(p + 'post_attention_norm.weight', layer.post_attention_layernorm.weight)
        for target, source in {'ffn_gate': 'gate_proj', 'ffn_up': 'up_proj',
                               'ffn_down': 'down_proj'}.items():
            add(p + target + '.weight', getattr(layer.mlp, source).weight)
        if layer.block_type == 'full_attention':
            attn = layer.self_attn
            for target, source in {'attn_q': 'q_proj', 'attn_k': 'k_proj',
                                   'attn_v': 'v_proj', 'attn_output': 'o_proj'}.items():
                add(p + target + '.weight', getattr(attn, source).weight)
            add_norm(p + 'attn_q_norm.weight', attn.q_norm.weight)
            add_norm(p + 'attn_k_norm.weight', attn.k_norm.weight)
            continue
        linear = layer.linear_attn
        # Only the value rows of the fused QKV projection are reordered.
        with torch.no_grad():
            qkv = linear.in_proj_qkv.weight
            stored = store(p + 'attn_qkv.weight', torch.cat(
                [qkv[:2 * key_dim], reorder_v(qkv[2 * key_dim:], 0, nk, rep, hv)]))
            qkv.copy_(torch.cat([stored[:2 * key_dim],
                                 reorder_v(stored[2 * key_dim:], 0, rep, nk, hv)]))
            conv = linear.conv1d.weight
            flat = conv.squeeze(1)
            stored = store(p + 'ssm_conv1d.weight', torch.cat(
                [flat[:2 * key_dim], reorder_v(flat[2 * key_dim:], 0, nk, rep, hv)]))
            conv.copy_(torch.cat([stored[:2 * key_dim],
                                  reorder_v(stored[2 * key_dim:], 0, rep, nk, hv)]).unsqueeze(1))
            # Stored already negated and exponentiated; per-head scalars only.
            store(p + 'ssm_a', -torch.exp(reorder_v(linear.A_log[:, None], 0, nk, rep, 1)[:, 0]))
            store(p + 'ssm_dt.bias', reorder_v(linear.dt_bias[:, None], 0, nk, rep, 1)[:, 0])
        add_reordered(p + 'attn_gate.weight', linear.in_proj_z.weight, 0, hv)
        add_reordered(p + 'ssm_beta.weight', linear.in_proj_b.weight, 0, 1)
        add_reordered(p + 'ssm_alpha.weight', linear.in_proj_a.weight, 0, 1)
        add_reordered(p + 'ssm_out.weight', linear.out_proj.weight, 1, hv)
        add(p + 'ssm_norm.weight', linear.norm.weight)

    # Real tokenizer metadata so the public loader runs too.
    writer.add_tokenizer_model('gpt2')
    writer.add_tokenizer_pre('qwen35')
    if vocab is None:
        base = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', 'a', 'b', 'ab']
        vocab = (base + [f'x{i}' for i in range(c.vocab_size - len(base))],
                 [3] * 3 + [1] * (c.vocab_size - 3))
    writer.add_token_list(vocab[0])
    writer.add_token_types(vocab[1])
    writer.add_token_merges(['a b'])
    writer.add_eos_token_id(2)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class Qwen35ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def _check_sequence(self, ns, model, tolerance, chunks=None):
        """Streamed chunks against a full recompute of the same prefix.

        Transformers is re-run over the whole prefix each time, so any drift in
        the carried convolution history or recurrent state shows up here.
        """
        kv, history = KVCache(ns.cfg.n_layer), []
        for ids in chunks or ([2, 8, 17, 21, 33, 9], [41], [5, 13, 12], [7]):
            tokens = torch.tensor(ids, device=ns.device)
            actual = ns.model.forward(tokens, kv, start_pos=len(history), last_only=False)
            history.extend(ids)
            expected = model(torch.tensor([history], device=ns.device),
                             use_cache=False).logits[0, -len(ids):]
            torch.testing.assert_close(actual, expected.float(), atol=tolerance, rtol=tolerance)
        return kv

    def test_prefill_chunked_decode_and_quantized_weights_match_transformers(self):
        for quantized, untied in [(False, False), (True, True)]:
            model, config = tiny_reference(tie_word_embeddings=not untied)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'qwen35.gguf'
                write_reference(path, model, config, quantized)
                for budget, rows in [(4096, 3), (32768, 32)]:
                    with self.subTest(quantized=quantized, budget=budget), NeuroStream.load(
                        path, device='cpu', mem_budget=budget, block_rows=rows, n_workers=2,
                    ) as ns, torch.no_grad():
                        self.assertEqual(ns.cfg.recurrent_layers, (True, True, True, False))
                        kv = self._check_sequence(ns, model, tolerance=2e-5)
                        self.assertLessEqual(ns.arena.stats.peak_bytes, budget)
                        # Recurrent layers keep fixed-size state, not a history.
                        self.assertIsNone(kv.k[0])
                        self.assertEqual(kv.recurrent[0].shape,
                                         (ns.cfg.linear_value_heads, ns.cfg.linear_key_dim,
                                          ns.cfg.linear_value_dim))
                        self.assertEqual(kv.conv[0].shape[1], ns.cfg.conv_kernel - 1)
                        self.assertIsNone(kv.recurrent[3])
                        self.assertEqual(kv.k[3].shape[1], kv.length)
                        with self.assertRaisesRegex(ValueError, 'position'):
                            ns.model.forward([2], kv, start_pos=0)
                        # Image input is exercised in tests/test_qwen35_vision.py.
                        with self.assertRaisesRegex(ValueError, 'DeepStack'):
                            ns.model.forward([2], KVCache(ns.cfg.n_layer),
                                             deepstack=[torch.zeros(1, ns.cfg.n_embd)])

    def test_bf16_gguf_streaming_and_ram_cache(self):
        """BF16 on disk, FP32 compute: compare against the same rounded weights."""
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bf16.gguf'
            write_reference(path, model, config, storage_type=gguf.GGMLQuantizationType.BF16)
            for ram in (0, 1 << 20):
                with self.subTest(ram=ram), NeuroStream.load(
                    path, device='cpu', mem_budget=2048, ram_budget=ram,
                    block_rows=3, n_workers=2,
                ) as ns, torch.no_grad():
                    self.assertEqual(ns.gguf.tensors['token_embd.weight'].dtype, GGMLType.BF16)
                    self.assertEqual({t.dtype for t in ns.gguf.tensors.values()},
                                     {GGMLType.BF16, GGMLType.F32})
                    # Larger than the I/O budget: exercises chunked resident loads.
                    self.assertGreater(ns.gguf.tensors['token_embd.weight'].nbytes, ns.arena.budget)
                    self._check_sequence(ns, model, tolerance=2e-5)
                    self.assertLessEqual(ns.arena.stats.peak_bytes, 2048)
                    self.assertEqual(ns.arena.used, 0)
                    if ram:
                        self.assertGreater(ns.cache.stats.ram_hits, 0)

    def test_mtp_block_count_and_config_validation(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mtp.gguf'
            # A real Qwen3.5-4B GGUF reports 33 blocks for 32 text layers.
            write_reference(path, model, config, n_mtp=1)
            g = GGUFFile(path)
            self.assertEqual(int(g.cfg('block_count')), config.num_hidden_layers + 1)
            cfg = Qwen35Config.from_gguf(g)
            self.assertEqual(cfg.n_layer, config.num_hidden_layers)
            self.assertEqual(cfg.recurrent_layers, (True, True, True, False))
            self.assertEqual(cfg.rope_dim, 4)
            self.assertEqual((cfg.linear_key_heads, cfg.linear_value_heads), (2, 4))
            self.assertIn('token_embd.weight', plan(g, cfg, vram_budget=1 << 20).vram)
            with NeuroStream.load(path, device='cpu', mem_budget=4096, n_workers=2) as ns, \
                    torch.no_grad():
                self.assertEqual(ns.cfg.n_layer, config.num_hidden_layers)
                self._check_sequence(ns, model, tolerance=2e-5, chunks=([2, 8, 17], [21]))

            g.metadata['qwen35.nextn_predict_layers'] = config.num_hidden_layers + 1
            with self.assertRaisesRegex(ValueError, 'MTP layer count'):
                Qwen35Config.from_gguf(g)
            g.metadata['qwen35.nextn_predict_layers'] = 1
            g.metadata['qwen35.attention.recurrent_layers'] = [True] * 3
            with self.assertRaisesRegex(ValueError, 'recurrent layer metadata'):
                Qwen35Config.from_gguf(g)
            del g.metadata['qwen35.attention.recurrent_layers']
            g.metadata['qwen35.expert_count'] = 128
            with self.assertRaisesRegex(ValueError, 'dense model'):
                Qwen35Config.from_gguf(g)

    def test_rejects_unsupported_rope_and_head_geometry(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.gguf'
            write_reference(path, model, config)
            for key, value, message in [
                ('qwen35.rope.dimension_count', 5, 'rotary dimension'),
                ('qwen35.rope.dimension_count', 0, 'rotary dimension'),
                ('qwen35.attention.value_length', 8, 'key/value dimensions'),
                ('qwen35.rope.scaling.type', 'yarn', 'native-context RoPE'),
                ('qwen35.ssm.conv_kernel', 0, 'convolution and head dimensions'),
                ('qwen35.full_attention_interval', 0, 'full_attention_interval'),
            ]:
                g = GGUFFile(path)
                if key == 'qwen35.full_attention_interval':
                    del g.metadata['qwen35.attention.recurrent_layers']
                g.metadata[key] = value
                with self.subTest(key=key, value=value), \
                        self.assertRaisesRegex(ValueError, message):
                    Qwen35Config.from_gguf(g)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_bf16_gguf_cuda_streaming_and_vram_cache(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bf16.gguf'
            write_reference(path, model, config, storage_type=gguf.GGMLQuantizationType.BF16)
            model.to(device='cuda', dtype=torch.bfloat16)
            for vram in (0, 1 << 20):
                with self.subTest(vram=vram), NeuroStream.load(
                    path, device='cuda', mem_budget=2048, vram_budget=vram,
                    reserve_vram=0, block_rows=3, n_workers=2,
                ) as ns, torch.no_grad():
                    self.assertEqual(ns.dtype, torch.bfloat16)
                    # Recurrent state stays float32 even when compute is bf16.
                    kv = self._check_sequence(ns, model, tolerance=0.06,
                                              chunks=([2, 8, 17, 21], [33], [5, 13]))
                    self.assertEqual(kv.recurrent[0].dtype, torch.float32)
                    self.assertLessEqual(ns.arena.stats.peak_bytes, 2048)
                    self.assertEqual(ns.arena.used, 0)
                    if vram:
                        self.assertGreater(ns.cache.stats.vram_hits, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_q4_streamed_and_resident(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q4.gguf'
            write_reference(path, model, config, quantized=True,
                            quant_type=gguf.GGMLQuantizationType.Q4_0)
            for dtype in (torch.float32, torch.bfloat16):
                model.to(device='cuda', dtype=dtype)
                for vram in (0, 1 << 20):
                    with self.subTest(dtype=dtype, vram=vram), NeuroStream.load(
                        path, device='cuda', compute_dtype=dtype, mem_budget=2048,
                        vram_budget=vram, reserve_vram=0, block_rows=8, n_workers=2,
                    ) as ns, torch.no_grad():
                        tolerance = 0.06 if dtype == torch.bfloat16 else 2e-5
                        # The second chunk is a single token: fused resident GEMV.
                        self._check_sequence(ns, model, tolerance=tolerance,
                                             chunks=([2, 8, 17, 21, 33, 9], [41]))


def small_tokenizer(**metadata):
    # Qwen is byte-level BPE: the base vocabulary is GPT-2's 256 mapped bytes,
    # so a multi-byte character is several tokens with no <0xXX> fallback.
    specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
    byte_tokens = [bytes_to_unicode()[b] for b in range(256)]
    tokens = specials + byte_tokens + ['ab']
    return GGUFTokenizer(SimpleNamespace(metadata={
        'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
        'tokenizer.ggml.tokens': tokens,
        'tokenizer.ggml.token_type': [3] * len(specials) + [1] * (len(tokens) - len(specials)),
        'tokenizer.ggml.merges': ['a b'], 'tokenizer.ggml.eos_token_id': 2,
        **metadata,
    }))


class Qwen35TokenizerTests(unittest.TestCase):
    def test_thinking_prompt_and_stop_ids(self):
        tok = small_tokenizer()
        messages = [{'role': 'user', 'content': 'ab'}]
        self.assertTrue(tok.apply_chat_template(messages).endswith(
            '<|im_start|>assistant\n<think>\n'))
        self.assertTrue(tok.apply_chat_template(messages, enable_thinking=False).endswith(
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        # Generation must stop on the turn marker as well as the declared EOS.
        self.assertEqual(tok.stop_ids, {0, 2})

    def test_jinja_template_wins_over_the_fallback(self):
        tok = small_tokenizer()
        tok.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}{{ enable_thinking }}"
        self.assertEqual(
            tok.apply_chat_template([{'role': 'user', 'content': 'ab'}], enable_thinking=False),
            'abFalse',
        )

    def test_nfc_normalization_and_specials(self):
        tok = small_tokenizer()
        # Qwen normalizes to NFC, so decomposed input must reach the composed id.
        self.assertEqual(tok.encode('é'), tok.encode('é'))
        self.assertEqual(tok.encode('<|im_start|>ab'), [1, tok.vocab['ab']])
        self.assertNotIn(1, tok.encode('<|im_start|>', allow_special=False))

    def test_generation_joins_split_utf8_and_stops(self):
        tok = small_tokenizer()
        ns = NeuroStream.__new__(NeuroStream)
        ns.cfg, ns.tokenizer = SimpleNamespace(n_layer=1), tok
        ns.model = SimpleNamespace(forward=lambda *a, **kw: torch.zeros(1, len(tok.tokens)))
        mapping = bytes_to_unicode()
        byte_ids = [tok.vocab[mapping[b]] for b in '\U0001f30d'.encode()]
        with patch.object(ns, '_sample', side_effect=byte_ids + [2]):
            self.assertEqual(''.join(ns.generate('ab', max_tokens=10)), '\U0001f30d')
        self.assertEqual(ns.stats.generated_tokens, 4)


if __name__ == '__main__':
    unittest.main()
