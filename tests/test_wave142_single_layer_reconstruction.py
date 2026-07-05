"""tests/test_wave142_single_layer_reconstruction.py

Wave 142 — go/no-go spike: prove a standalone mlx_lm decoder-layer class
can be constructed and run in isolation with just its own weights and an
input tensor, without ever instantiating the full model.

This is the single biggest technical risk in the sequential-AWQ-calibration
plan (Waves 143-145): AWQ calibration needs each layer's real activation
statistics, one layer at a time, without loading the whole model. If
`mlx_lm`'s internal layer classes can't be reconstructed standalone — e.g.
if they turn out to have undocumented cross-layer state, or a constructor
signature that assumes a parent model — that whole plan doesn't work and
needs to be reconsidered before any further waves are built.

Verified directly against the installed mlx_lm 0.31.3 source
(site-packages/mlx_lm/models/llama.py) rather than assumed from general
transformer-architecture knowledge:
- `TransformerBlock(args: ModelArgs, use_sliding: bool = False)` takes only
  a config dataclass — no reference to a parent model or other layers
- `ModelArgs.from_dict(config_dict)` (via BaseModelArgs) silently drops any
  config.json keys the dataclass doesn't declare — safe to pass the whole
  config
- `block.load_weights(list_of_(name, array)_pairs, strict=True)` loads a
  layer's own weights (with the `model.layers.{i}.` prefix stripped, since
  the standalone block has no such prefix) with a hard shape/name check
- `block(x, mask=None, cache=None)` is a real "hidden state in, hidden
  state out" interface: output shape always matches input shape (residual
  block invariant) — the carried-forward tensor between layers in the
  eventual sequential calibration loop

Confirmed on both Llama (this test) and Qwen3 (same TransformerBlock(args)
/ __call__ shape in mlx_lm/models/qwen3.py) — the same reconstruction
approach covers both families with zero family-specific code, matching the
"adapter keyed by block kind, not family name" design for Wave 143.

GO: this spike succeeded. Proceeding to Wave 143.
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
from mlx_lm.models.llama import ModelArgs as LlamaArgs
from mlx_lm.models.llama import TransformerBlock as LlamaBlock
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3Block

# A real Llama-3.2-3B-Instruct config.json (architecture fields only —
# identical between the bf16 and quantized releases of the same base
# model; only tensor storage format differs, not the ModelArgs schema).
LLAMA_3_2_3B_CONFIG = {
    "model_type": "llama",
    "hidden_size": 3072,
    "num_hidden_layers": 28,
    "intermediate_size": 8192,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-05,
    "vocab_size": 128256,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
}

QWEN3_8B_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-06,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
}


def _synthetic_layer_weights(dim: int, n_heads: int, n_kv: int, hd: int, inter: int):
    """Layer weights with the standalone block's own naming (no
    model.layers.{i}. prefix — that's stripped when extracting one layer's
    tensors from a real shard's weight_map)."""
    return [
        ("self_attn.q_proj.weight", mx.random.normal((n_heads * hd, dim)) * 0.02),
        ("self_attn.k_proj.weight", mx.random.normal((n_kv * hd, dim)) * 0.02),
        ("self_attn.v_proj.weight", mx.random.normal((n_kv * hd, dim)) * 0.02),
        ("self_attn.o_proj.weight", mx.random.normal((dim, n_heads * hd)) * 0.02),
        ("mlp.gate_proj.weight", mx.random.normal((inter, dim)) * 0.02),
        ("mlp.up_proj.weight", mx.random.normal((inter, dim)) * 0.02),
        ("mlp.down_proj.weight", mx.random.normal((dim, inter)) * 0.02),
        ("input_layernorm.weight", mx.ones((dim,))),
        ("post_attention_layernorm.weight", mx.ones((dim,))),
    ]


class TestLlamaStandaloneBlockReconstruction:
    def test_model_args_builds_from_full_config_dict(self):
        args = LlamaArgs.from_dict(LLAMA_3_2_3B_CONFIG)
        assert args.hidden_size == 3072
        assert args.num_attention_heads == 24
        assert args.num_key_value_heads == 8

    def test_block_constructs_without_parent_model(self):
        args = LlamaArgs.from_dict(LLAMA_3_2_3B_CONFIG)
        block = LlamaBlock(args=args, use_sliding=False)
        assert block is not None

    def test_synthetic_weights_load_strictly(self):
        args = LlamaArgs.from_dict(LLAMA_3_2_3B_CONFIG)
        block = LlamaBlock(args=args, use_sliding=False)
        weights = _synthetic_layer_weights(
            args.hidden_size,
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.intermediate_size,
        )
        # strict=True: raises if any name/shape doesn't exactly match the
        # block's own parameters — proves the naming convention (prefix
        # stripped) is exactly right, not just "close enough".
        block.load_weights(weights, strict=True)

    def test_forward_pass_preserves_hidden_state_shape(self):
        args = LlamaArgs.from_dict(LLAMA_3_2_3B_CONFIG)
        block = LlamaBlock(args=args, use_sliding=False)
        block.load_weights(
            _synthetic_layer_weights(
                args.hidden_size,
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.intermediate_size,
            ),
            strict=True,
        )
        x = mx.random.normal((1, 8, args.hidden_size))
        out = block(x, mask=None, cache=None)
        mx.eval(out)
        # The residual-block invariant this whole design depends on: output
        # shape always matches input shape, so layer i's output can feed
        # directly into layer i+1 as the carried-forward activation.
        assert out.shape == x.shape

    def test_output_differs_from_input_not_a_passthrough_noop(self):
        args = LlamaArgs.from_dict(LLAMA_3_2_3B_CONFIG)
        block = LlamaBlock(args=args, use_sliding=False)
        block.load_weights(
            _synthetic_layer_weights(
                args.hidden_size,
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.intermediate_size,
            ),
            strict=True,
        )
        x = mx.random.normal((1, 8, args.hidden_size))
        out = block(x, mask=None, cache=None)
        mx.eval(out)
        assert not mx.allclose(out, x).item()


class TestQwen3StandaloneBlockReconstruction:
    """Same mechanism, different family — proves this isn't Llama-specific."""

    def test_block_constructs_and_runs_standalone(self):
        args = Qwen3Args.from_dict(QWEN3_8B_CONFIG)
        block = Qwen3Block(args=args)
        block.load_weights(
            _synthetic_layer_weights(
                args.hidden_size,
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.intermediate_size,
            ),
            strict=False,  # Qwen3 adds q_norm/k_norm this test doesn't supply
        )
        x = mx.random.normal((1, 8, args.hidden_size))
        out = block(x, mask=None, cache=None)
        mx.eval(out)
        assert out.shape == x.shape
