"""GPT-OSS math against Transformers and quantization against gguf-py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import gguf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurostream.compute.quant import dequantize
from neurostream.compute.triton_kernels import can_fuse, fused_gemv
from neurostream.format.gguf import GGMLType, GGUFFile
from neurostream.io.arena import Arena
from neurostream.io.stream import StreamingSource
from neurostream.model.gpt_oss import GptOssModel
from neurostream.model.qwen3 import KVCache
from neurostream.residency.cache import TieredCache


def write_reference(path, model, config):
    writer = gguf.GGUFWriter(path, 'gpt-oss')
    for key, value in {
        'block_count': config.num_hidden_layers, 'embedding_length': config.hidden_size,
        'feed_forward_length': config.intermediate_size, 'context_length': config.max_position_embeddings,
        'expert_feed_forward_length': config.intermediate_size, 'expert_count': config.num_local_experts,
        'expert_used_count': config.num_experts_per_tok, 'attention.head_count': config.num_attention_heads,
        'attention.head_count_kv': config.num_key_value_heads, 'attention.key_length': config.head_dim,
        'attention.sliding_window': config.sliding_window,
        'rope.scaling.original_context_length': 128,
    }.items():
        writer.add_uint32('gpt-oss.' + key, value)
    for key, value in {'rope.freq_base': 150000., 'rope.scaling.factor': 4.,
                       'attention.layer_norm_rms_epsilon': config.rms_norm_eps}.items():
        writer.add_float32('gpt-oss.' + key, value)
    def add(name, weight):
        writer.add_tensor(name, weight.detach().float().cpu().numpy().copy())
    add('token_embd.weight', model.model.embed_tokens.weight)
    add('output.weight', model.lm_head.weight)
    add('output_norm.weight', model.model.norm.weight)
    for i, layer in enumerate(model.model.layers):
        p = f'blk.{i}.'
        add(p + 'attn_norm.weight', layer.input_layernorm.weight)
        add(p + 'post_attention_norm.weight', layer.post_attention_layernorm.weight)
        add(p + 'attn_sinks.weight', layer.self_attn.sinks)
        for target, source in [('attn_q','q_proj'), ('attn_k','k_proj'),
                               ('attn_v','v_proj'), ('attn_output','o_proj')]:
            linear = getattr(layer.self_attn, source)
            add(p + target + '.weight', linear.weight)
            add(p + target + '.bias', linear.bias)
        add(p + 'ffn_gate_inp.weight', layer.mlp.router.weight)
        add(p + 'ffn_gate_inp.bias', layer.mlp.router.bias)
        experts = layer.mlp.experts
        gu = experts.gate_up_proj.transpose(1, 2)
        add(p + 'ffn_gate_exps.weight', gu[:, ::2])
        add(p + 'ffn_up_exps.weight', gu[:, 1::2])
        add(p + 'ffn_gate_exps.bias', experts.gate_up_proj_bias[:, ::2])
        add(p + 'ffn_up_exps.bias', experts.gate_up_proj_bias[:, 1::2])
        add(p + 'ffn_down_exps.weight', experts.down_proj.transpose(1, 2))
        add(p + 'ffn_down_exps.bias', experts.down_proj_bias)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def tiny_config():
    from transformers import GptOssConfig
    config = GptOssConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, vocab_size=64,
        max_position_embeddings=512, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention'],
        rope_parameters={'rope_type': 'yarn', 'rope_theta': 150000., 'factor': 4.,
                         'original_max_position_embeddings': 128, 'truncate': False},
    )
    config._attn_implementation = 'eager'
    return config


def build_tiny_gguf(tmp):
    """Write a tiny GPT-OSS reference and return its path."""
    from transformers import GptOssForCausalLM
    torch.manual_seed(42)
    config = tiny_config()
    model = GptOssForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name or 'sinks' in name:
                param.uniform_(-0.1, 0.1)
    path = Path(tmp) / 'tiny.gguf'
    write_reference(path, model, config)
    return path


class GptOssTests(unittest.TestCase):
    def test_mxfp4_matches_reference(self):
        rng = np.random.default_rng(42)
        raw = rng.integers(0, 256, (254, 17), dtype=np.uint8)
        raw[:, 0] = np.arange(254, dtype=np.uint8)
        expected = gguf.dequantize(raw, gguf.GGMLQuantizationType.MXFP4).reshape(-1)
        actual = dequantize(torch.from_numpy(raw.reshape(-1)), GGMLType.MXFP4, 254 * 32)
        np.testing.assert_array_equal(actual.numpy(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_fused_quant32_matches_reference(self):
        rng = np.random.default_rng(9)
        weights = rng.normal(0, 0.1, (19, 2880)).astype(np.float32)
        for dtype in (GGMLType.MXFP4, GGMLType.Q8_0):
            with self.subTest(dtype=dtype):
                raw = gguf.quantize(weights, gguf.GGMLQuantizationType(int(dtype)))
                expanded = gguf.dequantize(raw, gguf.GGMLQuantizationType(int(dtype)))
                x = torch.linspace(-1, 1, 2880, device='cuda').reshape(1, -1)
                self.assertTrue(can_fuse(dtype, x))
                actual = fused_gemv(torch.from_numpy(raw.reshape(-1)).cuda(), dtype, x,
                                    19, 2880, out_dtype=torch.float32)
                expected = x @ torch.from_numpy(expanded).cuda().T
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_streamed_model_matches_transformers_prefill_and_decode(self):
        from transformers import GptOssConfig, GptOssForCausalLM
        from transformers.cache_utils import DynamicCache
        torch.manual_seed(42)
        config = GptOssConfig(
            hidden_size=64, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            num_local_experts=4, num_experts_per_tok=2, vocab_size=64,
            max_position_embeddings=512, sliding_window=4,
            layer_types=['sliding_attention','full_attention'],
            rope_parameters={'rope_type':'yarn','rope_theta':150000.,'factor':4.,
                             'original_max_position_embeddings':128,'truncate':False},
        )
        config._attn_implementation = 'eager'
        model = GptOssForCausalLM(config).eval()
        # Nonzero biases and sinks exercise GPT-OSS-specific behavior.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'bias' in name or 'sinks' in name:
                    param.uniform_(-0.1, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.gguf'
            write_reference(path, model, config)
            with StreamingSource(path, Arena(96 << 10), n_workers=2) as source:
                streamed = GptOssModel(TieredCache(source), block_rows=16)
                kv = KVCache(2)
                hf_kv = DynamicCache(config=config)
                for tokens, pos in [([1, 2, 3, 4, 5, 6], 0), ([7], 6), ([8, 9], 7)]:
                    with torch.no_grad():
                        expected = model(torch.tensor([tokens]), past_key_values=hf_kv, use_cache=True).logits[0]
                        actual = streamed.forward(torch.tensor(tokens), kv, start_pos=pos, last_only=False)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                    self.assertEqual(source.arena.used, 0)
                self.assertLessEqual(source.arena.stats.peak_bytes, source.arena.budget)


class NeuronStreamingTests(unittest.TestCase):
    """The routed experts are walked a block of neurons at a time."""

    PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]

    def logits(self, path, budget, **kwargs):
        with StreamingSource(path, Arena(budget), n_workers=2) as source:
            model = GptOssModel(TieredCache(source), **kwargs)
            with torch.no_grad():
                out = model.forward(torch.tensor(self.PROMPT),
                                    KVCache(2), last_only=False)
            self.assertEqual(source.arena.used, 0)
            return out, source.arena.stats.peak_bytes

    def test_neuron_blocks_match_whole_expert_reads(self):
        """Splitting an expert into neuron blocks must not move the logits.

        The activation is elementwise per neuron and ffn_down accumulates
        over them, so any block decomposition is the same arithmetic in a
        different order -- exactly what makes the block size a free knob.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = build_tiny_gguf(tmp)
            whole, _ = self.logits(path, 1 << 20, block_rows=0)
            for block_rows in (1, 7, 8, 64, 4096):
                with self.subTest(block_rows=block_rows):
                    blocked, _ = self.logits(
                        path, 1 << 20, block_rows=block_rows,
                    )
                    torch.testing.assert_close(blocked, whole,
                                               rtol=1e-5, atol=1e-6)

    def test_budget_below_one_expert_still_runs(self):
        """Peak memory follows the block, not the expert.

        One expert matrix here is 64 x 64 floats = 16 KB. The arena is given
        half of that, so a forward pass can only complete if no expert is
        ever held whole.
        """
        expert_bytes = 64 * 64 * 4
        budget = expert_bytes // 2
        with tempfile.TemporaryDirectory() as tmp:
            path = build_tiny_gguf(tmp)
            reference, _ = self.logits(path, 1 << 20, block_rows=0)
            blocked, peak = self.logits(path, budget, block_rows=4096)
            torch.testing.assert_close(blocked, reference,
                                       rtol=1e-5, atol=1e-6)
            self.assertLessEqual(peak, budget)
            self.assertLess(peak, expert_bytes)

    def test_deeper_lookahead_keeps_the_budget(self):
        """Extra queue depth buys reads in flight, never extra peak bytes."""
        budget = 24 << 10
        with tempfile.TemporaryDirectory() as tmp:
            path = build_tiny_gguf(tmp)
            reference, _ = self.logits(path, 1 << 20, block_rows=0)
            for lookahead in (1, 2, 4, 8):
                with self.subTest(lookahead=lookahead):
                    blocked, peak = self.logits(
                        path, budget, block_rows=4096,
                        expert_lookahead=lookahead,
                    )
                    torch.testing.assert_close(blocked, reference,
                                               rtol=1e-5, atol=1e-6)
                    self.assertLessEqual(peak, budget)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
