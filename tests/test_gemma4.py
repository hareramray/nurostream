"""Gemma 4 text math versus Transformers; tiny GGUFs require no model download.

Run: python tests/test_gemma4.py
Optional full-vocabulary parity: set NEUROSTREAM_GEMMA4_TOKENIZER_DIR to a
directory containing Google's tokenizer.json and chat_template.jinja.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
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
from neurostream.format.tokenizer import GGUFTokenizer
from neurostream.model.gemma4 import Gemma4Config
from neurostream.model.qwen3 import KVCache
from neurostream.residency.planner import plan


def tiny_reference(**kwargs):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    config = Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size=32, hidden_size_per_layer_input=8, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=16, num_kv_shared_layers=2,
        sliding_window=4, max_position_embeddings=128,
        layer_types=['sliding_attention', 'full_attention'] * 2,
        final_logit_softcapping=3.0, **kwargs,
    )
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    model = Gemma4ForCausalLM(config).eval()
    # Non-unit norms/scalars expose missing normalization and residual order.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'norm' in name:
                param.uniform_(0.7, 1.3)
        for layer in model.model.layers:
            layer.layer_scalar.fill_(0.83)
    return model, config


def write_reference(path, model, c, quantized=False, quant_type=gguf.GGMLQuantizationType.Q8_0,
                    storage_type=gguf.GGMLQuantizationType.F32):
    writer = gguf.GGUFWriter(path, 'gemma4')
    integers = {
        'block_count': c.num_hidden_layers, 'embedding_length': c.hidden_size,
        'embedding_length_per_layer_input': c.hidden_size_per_layer_input,
        'context_length': c.max_position_embeddings,
        'attention.head_count': c.num_attention_heads,
        'attention.head_count_kv': c.num_key_value_heads,
        'attention.key_length': 16, 'attention.value_length': 16,
        'attention.key_length_swa': model.model.layers[0].self_attn.head_dim,
        'attention.value_length_swa': model.model.layers[0].self_attn.head_dim,
        'attention.shared_kv_layers': c.num_kv_shared_layers,
        'attention.sliding_window': c.sliding_window,
    }
    for key, value in integers.items():
        writer.add_uint32('gemma4.' + key, value)
    writer.add_array('gemma4.feed_forward_length', [layer.mlp.up_proj.out_features for layer in model.model.layers])
    writer.add_array('gemma4.attention.sliding_window_pattern', [t == 'sliding_attention' for t in c.layer_types])
    for key, value in {
        'rope.freq_base': 1e6, 'rope.freq_base_swa': 1e4,
        'attention.layer_norm_rms_epsilon': c.rms_norm_eps,
        'final_logit_softcapping': c.final_logit_softcapping,
    }.items():
        writer.add_float32('gemma4.' + key, value)

    def add(name, value):
        array = value.detach().float().cpu().numpy().copy()
        # Q8 projections plus F32 norms/short-row PLE, like mixed-type GGUFs.
        if quantized and array.ndim == 2 and array.shape[-1] % 32 == 0:
            raw = gguf.quantize(array, quant_type)
            writer.add_tensor(name, raw, raw_dtype=quant_type)
            # Compare with exactly these dequantized weights in Transformers.
            with torch.no_grad():
                value.copy_(torch.from_numpy(gguf.dequantize(raw, quant_type)))
        elif array.ndim >= 2 and storage_type != gguf.GGMLQuantizationType.F32:
            raw = gguf.quantize(array, storage_type)
            writer.add_tensor(name, raw, raw_dtype=storage_type)
            with torch.no_grad():
                value.copy_(torch.from_numpy(gguf.dequantize(raw, storage_type)))
        else:
            writer.add_tensor(name, array)

    add('token_embd.weight', model.model.embed_tokens.weight)
    if not c.tie_word_embeddings:
        add('output.weight', model.lm_head.weight)
    for target, source in {
        'per_layer_token_embd': 'embed_tokens_per_layer',
        'per_layer_model_proj': 'per_layer_model_projection',
        'per_layer_proj_norm': 'per_layer_projection_norm', 'output_norm': 'norm',
    }.items():
        add(target + '.weight', getattr(model.model, source).weight)
    add('rope_freqs.weight', torch.tensor([1., 1.] + [1e30] * 6))
    for i, layer in enumerate(model.model.layers):
        p = f'blk.{i}.'
        for target, source in {
            'attn_norm': 'input_layernorm', 'post_attention_norm': 'post_attention_layernorm',
            'ffn_norm': 'pre_feedforward_layernorm', 'post_ffw_norm': 'post_feedforward_layernorm',
            'inp_gate': 'per_layer_input_gate', 'proj': 'per_layer_projection',
            'post_norm': 'post_per_layer_input_norm',
        }.items():
            add(p + target + '.weight', getattr(layer, source).weight)
        for target, source in {'attn_q': 'q_proj', 'attn_k': 'k_proj', 'attn_v': 'v_proj',
                               'attn_output': 'o_proj', 'attn_q_norm': 'q_norm', 'attn_k_norm': 'k_norm'}.items():
            linear = getattr(layer.self_attn, source, None)
            if linear is not None:
                add(p + target + '.weight', linear.weight)
        for target, source in {'ffn_gate': 'gate_proj', 'ffn_up': 'up_proj', 'ffn_down': 'down_proj'}.items():
            add(p + target + '.weight', getattr(layer.mlp, source).weight)
        add(p + 'layer_output_scale.weight', layer.layer_scalar)
    # Tiny real tokenizer metadata lets the public loader run as well.
    writer.add_tokenizer_model('gemma4')
    writer.add_token_list(['<pad>', '<eos>', '<bos>', '<unk>', '<turn|>', 'a', 'b', 'ab'] + [f'x{i}' for i in range(56)])
    writer.add_token_types([3] * 5 + [1] * 59)
    writer.add_token_merges(['a b'])
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class Gemma4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prefill_chunked_decode_and_quantized_weights_match_transformers(self):
        from transformers.cache_utils import DynamicCache
        for quantized, untied, alternative in [(False, False, False), (True, True, True)]:
            model, config = tiny_reference(tie_word_embeddings=not untied, attention_k_eq_v=alternative,
                                           use_double_wide_mlp=alternative)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'gemma4.gguf'
                write_reference(path, model, config, quantized)
                for budget, rows in [(4096, 3), (32768, 32)]:
                    with self.subTest(quantized=quantized, budget=budget), NeuroStream.load(
                        path, device='cpu', mem_budget=budget, block_rows=rows, n_workers=2,
                    ) as ns, torch.no_grad():
                        kv, hf_kv = KVCache(config.num_hidden_layers), DynamicCache(config=config)
                        pos = 0
                        # Cross the window during both prefill and a multi-token continuation.
                        for ids in ([2, 8, 17, 21, 33, 9], [41], [5, 13, 12], [7]):
                            tokens = torch.tensor(ids)
                            expected = model(tokens[None], past_key_values=hf_kv, use_cache=True).logits[0]
                            actual = ns.model.forward(tokens, kv, start_pos=pos, last_only=False)
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                            pos += len(ids)
                        self.assertLessEqual(ns.arena.stats.peak_bytes, budget)
                        self.assertEqual(kv.k[0].shape[1], config.sliding_window)
                        self.assertEqual(kv.k[1].shape[1], pos)
                        self.assertIsNone(kv.k[2])
                        self.assertIsNone(kv.k[3])
                        with self.assertRaisesRegex(ValueError, 'position'):
                            ns.model.forward([2], kv, start_pos=0)
                        with self.assertRaisesRegex(ValueError, 'text input'):
                            ns.attach_vision('unused.gguf')

    def test_config_validation_and_auxiliary_embedding_placement(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.gguf'
            write_reference(path, model, config)
            g = GGUFFile(path)
            cfg = Gemma4Config.from_gguf(g)
            placement = plan(g, cfg, vram_budget=g.tensors['token_embd.weight'].nbytes)
            self.assertNotIn('per_layer_token_embd.weight', placement.vram)
            g.metadata['gemma4.attention.shared_kv_layers'] = cfg.n_layer
            with self.assertRaisesRegex(ValueError, 'shared KV'):
                Gemma4Config.from_gguf(g)
            g.metadata['gemma4.expert_count'] = 128
            with self.assertRaisesRegex(ValueError, 'dense E4B'):
                Gemma4Config.from_gguf(g)

    def test_bf16_gguf_streaming_and_ram_cache(self):
        """BF16 on disk, FP32 compute: compare against the same rounded weights."""
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bf16.gguf'
            write_reference(path, model, config, storage_type=gguf.GGMLQuantizationType.BF16)
            for ram in (0, 1 << 20):
                with self.subTest(ram=ram), NeuroStream.load(
                    path, device='cpu', mem_budget=512, ram_budget=ram, block_rows=3, n_workers=2,
                ) as ns, torch.no_grad():
                    self.assertEqual(ns.gguf.tensors['token_embd.weight'].dtype, GGMLType.BF16)
                    self.assertEqual({t.dtype for t in ns.gguf.tensors.values()}, {GGMLType.BF16, GGMLType.F32})
                    self.assertGreater(ns.gguf.tensors['token_embd.weight'].nbytes, ns.arena.budget)
                    self._check_bf16_sequence(ns, model, tolerance=2e-5)
                    self.assertLessEqual(ns.arena.stats.peak_bytes, 512)
                    self.assertEqual(ns.arena.used, 0)
                    if ram:
                        self.assertGreater(ns.cache.stats.ram_hits, 0)

    def _check_bf16_sequence(self, ns, model, tolerance):
        kv = KVCache(ns.cfg.n_layer)
        history = []
        for ids in ([2, 8, 17, 21, 33, 9], [41], [5, 13, 12]):
            tokens = torch.tensor(ids, device=ns.device)
            actual = ns.model.forward(tokens, kv, start_pos=len(history), last_only=False)
            history.extend(ids)
            expected = model(torch.tensor([history], device=ns.device), use_cache=False).logits[0, -len(ids):]
            torch.testing.assert_close(actual, expected.float(), atol=tolerance, rtol=tolerance)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_bf16_gguf_cuda_streaming_and_vram_cache(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bf16.gguf'
            write_reference(path, model, config, storage_type=gguf.GGMLQuantizationType.BF16)
            model.to(device='cuda', dtype=torch.bfloat16)
            for vram in (0, 1 << 20):
                with self.subTest(vram=vram), NeuroStream.load(
                    path, device='cuda', mem_budget=512, vram_budget=vram,
                    reserve_vram=0, block_rows=3, n_workers=2,
                ) as ns, torch.no_grad():
                    self.assertEqual(ns.dtype, torch.bfloat16)
                    self._check_bf16_sequence(ns, model, tolerance=0.06)
                    self.assertLessEqual(ns.arena.stats.peak_bytes, 512)
                    self.assertEqual(ns.arena.used, 0)
                    if vram:
                        self.assertGreater(ns.cache.stats.vram_hits, 0)
                        self.assertIn('token_embd.weight', ns.placement.vram)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_q4_streamed_and_resident(self):
        model, config = tiny_reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q4.gguf'
            write_reference(path, model, config, quantized=True, quant_type=gguf.GGMLQuantizationType.Q4_0)
            for dtype in (torch.float32, torch.bfloat16):
                model.to(device='cuda', dtype=dtype)
                for vram in (0, 1 << 20):
                    with self.subTest(dtype=dtype, vram=vram), NeuroStream.load(
                        path, device='cuda', compute_dtype=dtype, mem_budget=512,
                        vram_budget=vram, reserve_vram=0, block_rows=8, n_workers=2,
                    ) as ns, torch.no_grad():
                        tokens = torch.tensor([2, 8, 17, 21, 33, 9], device='cuda')
                        expected = model(tokens[None], use_cache=False).logits[0, -1:]
                        kv = KVCache(config.num_hidden_layers)
                        actual = ns.model.forward(tokens, kv)
                        tolerance = 0.06 if dtype == torch.bfloat16 else 2e-5
                        torch.testing.assert_close(actual, expected.float(), atol=tolerance, rtol=tolerance)
                        # Single-token decode exercises fused resident/streamed GEMV.
                        expected = model(torch.cat((tokens, tokens[:1]))[None], use_cache=False).logits[0, -1:]
                        actual = ns.model.forward(tokens[:1], kv, start_pos=len(tokens))
                        torch.testing.assert_close(actual, expected.float(), atol=tolerance, rtol=tolerance)


def small_tokenizer():
    tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<turn|>', '<|turn>', '<|channel>',
              '<channel|>', '<|think|>', 'a', 'b', 'ab', '\u2581', '\u2581ab']
    types = [3] * 9 + [1] * 5
    # Gemma's visible channel markers are USER_DEFINED in GGUFs.
    types[6:8] = [4, 4]
    tokens += [f'<0x{b:02X}>' for b in range(256)]
    types += [6] * 256
    return GGUFTokenizer(SimpleNamespace(metadata={
        'tokenizer.ggml.model': 'gemma4', 'tokenizer.ggml.tokens': tokens,
        'tokenizer.ggml.token_type': types, 'tokenizer.ggml.merges': ['a b', '\u2581 ab'],
        'tokenizer.ggml.bos_token_id': 2, 'tokenizer.ggml.eos_token_id': 1,
    }))


class Gemma4TokenizerTests(unittest.TestCase):
    def test_unicode_bpe_specials_bos_and_byte_fallback(self):
        tok = small_tokenizer()
        self.assertEqual(tok.encode(' ab'), [2, 13])
        text = ' ab\n\t\U0001f30d\u0928\u092e\u0938\u094d\u0924\u0947'
        self.assertEqual(tok.decode(tok.encode(text)[1:]), text)
        self.assertEqual(tok.encode('<bos>ab'), [2, 11])
        self.assertEqual(tok.encode('<|channel>ab<channel|>'), [2, 6, 11, 7])
        self.assertEqual(tok.stop_ids, {1, 4})
        self.assertNotIn(6, tok.encode('<|channel>', allow_special=False))

    def test_chat_fallback_and_template_context(self):
        tok = small_tokenizer()
        expected = '<bos><|turn>user\nab<turn|>\n<|turn>model\n'
        self.assertEqual(tok.apply_chat_template([{'role': 'user', 'content': 'ab'}]), expected)
        thinking = tok.apply_chat_template([{'role': 'user', 'content': 'ab'}], enable_thinking=True)
        self.assertIn('<|turn>system\n<|think|>\n<turn|>', thinking)
        tok.chat_template = '{{ bos_token }}{{ messages[0].content }}{{ eos_token }}'
        self.assertEqual(tok.apply_chat_template([{'role': 'user', 'content': 'ab'}]), '<bos>ab<eos>')

    def test_generation_stops_at_turn_and_preserves_split_utf8(self):
        tok = small_tokenizer()
        ns = NeuroStream.__new__(NeuroStream)
        ns.cfg, ns.tokenizer = SimpleNamespace(n_layer=1), tok
        ns.model = SimpleNamespace(forward=lambda *args, **kw: torch.zeros(1, len(tok.tokens)))
        byte_ids = [tok.vocab[f'<0x{b:02X}>'] for b in '\U0001f30d'.encode()]
        with patch.object(ns, '_sample', side_effect=byte_ids + [tok.vocab['<turn|>']]):
            self.assertEqual(''.join(ns.generate('ab', max_tokens=10)), '\U0001f30d')
        self.assertEqual(ns.stats.generated_tokens, 4)

    @unittest.skipUnless(os.environ.get('NEUROSTREAM_GEMMA4_TOKENIZER_DIR'), 'optional Google tokenizer fixture absent')
    def test_google_tokenizer_and_chat_template_parity(self):
        from tokenizers import Tokenizer
        root = Path(os.environ['NEUROSTREAM_GEMMA4_TOKENIZER_DIR'])
        data = json.loads((root / 'tokenizer.json').read_text(encoding='utf-8'))
        reference = Tokenizer.from_file(str(root / 'tokenizer.json'))
        vocab = data['model']['vocab']
        tokens = [t for t, i in sorted(vocab.items(), key=lambda item: item[1])]
        types = [1] * len(tokens)
        for token in data['added_tokens']:
            types[token['id']] = 3 if token['special'] else 4
        tok = GGUFTokenizer(SimpleNamespace(metadata={
            'tokenizer.ggml.model': 'gemma4', 'tokenizer.ggml.tokens': tokens,
            'tokenizer.ggml.token_type': types,
            'tokenizer.ggml.merges': [' '.join(pair) for pair in data['model']['merges']],
            'tokenizer.ggml.bos_token_id': 2, 'tokenizer.ggml.eos_token_id': 1,
        }))
        samples = ['Hello world!', '  spaces\t tabs\n\n', '1234567890', '\u0928\u092e\u0938\u094d\u0924\u0947',
                   '\u4f60\u597d\U0001f30d', 'e\u0301', '<|channel>thought\nab<channel|>', '<bos><|turn>user\nHello<turn|>\n']
        rng = random.Random(42)
        alphabet = 'abcd 01239\n\t.,<|>\u00e9\u0301\u4f60\u0928\U0001f30d'
        samples += [''.join(rng.choices(alphabet, k=rng.randint(1, 150))) for _ in range(300)]
        for text in samples:
            expected = reference.encode(text, add_special_tokens=False).ids
            self.assertEqual(tok.encode(text, add_special_tokens=False), expected, repr(text))
            self.assertEqual(tok.decode(expected), reference.decode(expected, skip_special_tokens=False))
        messages = [{'role': 'user', 'content': 'Hello'}]
        fallback = [tok.apply_chat_template(messages, enable_thinking=b) for b in (False, True)]
        tok.chat_template = (root / 'chat_template.jinja').read_text(encoding='utf-8')
        for b, expected in zip((False, True), fallback):
            self.assertEqual(tok.apply_chat_template(messages, enable_thinking=b), expected)


if __name__ == '__main__':
    unittest.main()
