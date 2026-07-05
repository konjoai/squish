"""tests/test_wave143_block_adapters.py

Wave 143 — block-kind adapter registry: resolve an architecture to the
right sequential-AWQ-calibration path, keyed by decoder-block *kind*
(structural shape) rather than architecture family name.

This pins the two-part safety net directly, against the real installed
mlx_lm 0.31.3:

- resolve_dense_architecture reuses mlx_lm.utils.MODEL_REMAPPING (the
  same table mlx_lm's own loader uses) rather than a hand-maintained
  list, so "mistral" correctly resolves to llama.py's classes (mistral
  has no dedicated mlx_lm/models/mistral.py), and genuinely non-dense
  model_types (mixtral, mamba — real MoE/SSM families) correctly return
  None because their modules lack a top-level TransformerBlock class.
- is_standard_dense_block is a structural (not name-based) check: it
  must correctly ACCEPT a real Llama block and REJECT synthetic
  MoE-shaped and SSM-shaped blocks even when they're given a
  deliberately misleading class name, since real mlx_lm architectures
  reuse "TransformerBlock" for at least one genuine MoE family (olmoe.py)
  and use a different name ("PhiDecoderLayer") for a genuinely dense one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs as LlamaArgs
from mlx_lm.models.llama import TransformerBlock as LlamaBlock

from squish.quant.block_adapters import is_standard_dense_block, resolve_dense_architecture

LLAMA_CONFIG = {
    "model_type": "llama",
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "rms_norm_eps": 1e-05,
    "vocab_size": 1000,
    "tie_word_embeddings": True,
}


class TestResolveDenseArchitecture:
    def test_llama_resolves_directly(self):
        resolved = resolve_dense_architecture("llama")
        assert resolved is not None
        assert resolved.module_name == "llama"
        assert resolved.transformer_block_cls.__name__ == "TransformerBlock"

    def test_qwen3_resolves_directly(self):
        resolved = resolve_dense_architecture("qwen3")
        assert resolved is not None
        assert resolved.module_name == "qwen3"

    def test_mistral_remaps_to_llama(self):
        # mlx_lm has no dedicated mistral.py — MODEL_REMAPPING sends it to
        # llama.py, since Mistral's architecture is Llama-compatible.
        resolved = resolve_dense_architecture("mistral")
        assert resolved is not None
        assert resolved.module_name == "llama"

    def test_mixtral_returns_none_no_transformer_block_class(self):
        # Real MoE family: mixtral.py defines MixtralDecoderLayer, not
        # TransformerBlock — resolution correctly fails rather than
        # guessing at a class that doesn't exist.
        assert resolve_dense_architecture("mixtral") is None

    def test_mamba_returns_none_different_block_shape_entirely(self):
        assert resolve_dense_architecture("mamba") is None

    def test_unknown_model_type_returns_none(self):
        assert resolve_dense_architecture("totally-fake-arch-xyz") is None


class TestIsStandardDenseBlock:
    def _real_llama_block(self):
        args = LlamaArgs.from_dict(LLAMA_CONFIG)
        return LlamaBlock(args=args, use_sliding=False)

    def test_accepts_real_llama_block(self):
        assert is_standard_dense_block(self._real_llama_block()) is True

    def test_rejects_object_without_named_modules(self):
        assert is_standard_dense_block(object()) is False

    def test_rejects_moe_shaped_block_even_with_dense_class_name(self):
        # Mirrors the real olmoe.py situation: a block whose own class is
        # literally named "TransformerBlock" but contains an MoE submodule
        # — must be rejected on structure, not waved through on name.
        class _FakeSparseMoeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 4)

        class TransformerBlock(nn.Module):  # deliberately reuse the dense name
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(8, 8)
                self.mlp = _FakeSparseMoeBlock()

        assert is_standard_dense_block(TransformerBlock()) is False

    def test_rejects_ssm_shaped_block(self):
        class _FakeMambaMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Linear(8, 8)

        class _FakeDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = _FakeMambaMixer()

        assert is_standard_dense_block(_FakeDecoderLayer()) is False

    def test_accepts_dense_block_under_a_different_class_name(self):
        # Mirrors the real phi.py situation: dense (attention+MLP, no MoE/
        # SSM submodules) but named "PhiDecoderLayer" not "TransformerBlock"
        # — must be accepted on structure, not rejected on name.
        class PhiDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(8, 8)
                self.mlp = nn.Linear(8, 8)

        assert is_standard_dense_block(PhiDecoderLayer()) is True

    def test_real_llama_block_forward_pass_still_works_after_check(self):
        # is_standard_dense_block must be read-only / non-mutating.
        block = self._real_llama_block()
        assert is_standard_dense_block(block) is True
        x = mx.random.normal((1, 4, 256))
        out = block(x, mask=None, cache=None)
        mx.eval(out)
        assert out.shape == x.shape
