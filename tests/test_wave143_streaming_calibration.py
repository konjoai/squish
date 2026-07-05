"""tests/test_wave143_streaming_calibration.py

Wave 143 — collect_activation_scales_streaming: the sequential
(layer-at-a-time) counterpart to awq.py's full-load
collect_activation_scales, using Wave 142's proven single-layer
reconstruction and the block-kind adapter registry.

This pins, against a real synthetic sharded Llama-shaped model (real
mlx_lm TransformerBlock construction, real forward passes, real
safetensors shards split across two files with layer 0 in one shard and
layer 1 in the other):

- a full run produces one AWQ scale vector per nn.Linear submodule per
  layer, with shapes matching each submodule's real in_features
- scale vectors are non-degenerate (no NaN/Inf, no all-zero — the
  activation-hook mechanism is genuinely wired up, not silently inert)
- an unsupported architecture (mixtral — genuine MoE, no dense-block
  adapter) returns None rather than guessing, with a clear notice
- the function never holds more than one layer's raw weights at once —
  verified the same way as Wave 141's peak-RAM invariant test, by
  tracking shard-file reads per layer
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from safetensors.numpy import save_file

pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

from squish.quant.awq_streaming import collect_activation_scales_streaming
from squish.quant.shard_index import load_shard_index

HIDDEN = 64
N_HEADS = 4
N_KV = 2
HEAD_DIM = 16
INTER = 128
VOCAB = 500
N_LAYERS = 2


class _FakeTokenizer:
    """Deterministic small in-range token ids — no real vocab needed since
    this pins the calibration MECHANISM, not linguistic calibration quality
    (that's Wave 145's accuracy-gate job)."""

    def encode(self, text, add_special_tokens=True):
        return [(ord(c) % (VOCAB - 1)) + 1 for c in text[:12]]


def _build_synthetic_llama_model(model_dir: Path, model_type: str = "llama") -> None:
    config = {
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": True,
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    rng = np.random.default_rng(0)
    tensors = {}
    tensors["model.embed_tokens.weight"] = (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(
        np.float32
    )
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[p + "post_attention_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[p + "self_attn.q_proj.weight"] = (
            rng.standard_normal((N_HEADS * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.k_proj.weight"] = (
            rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.v_proj.weight"] = (
            rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.o_proj.weight"] = (
            rng.standard_normal((HIDDEN, N_HEADS * HEAD_DIM)) * 0.02
        ).astype(np.float32)
        tensors[p + "mlp.gate_proj.weight"] = (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(
            np.float32
        )
        tensors[p + "mlp.up_proj.weight"] = (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(
            np.float32
        )
        tensors[p + "mlp.down_proj.weight"] = (rng.standard_normal((HIDDEN, INTER)) * 0.02).astype(
            np.float32
        )
    tensors["model.norm.weight"] = np.ones((HIDDEN,), dtype=np.float32)

    # Split so layer 0 lands entirely in shard 1, layer 1 entirely in shard 2
    # — a clean, non-spanning split (the shard-spanning case is Wave 147's
    # streaming-pull eviction-timing concern, not calibration correctness).
    shard1 = {
        k: v
        for k, v in tensors.items()
        if k.startswith("model.embed_tokens") or k.startswith("model.layers.0.")
    }
    shard2 = {k: v for k, v in tensors.items() if k not in shard1}
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.fixture
def llama_model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    _build_synthetic_llama_model(d)
    return d


class TestStreamingCalibrationDenseArchitecture:
    def test_produces_one_scale_per_linear_submodule_per_layer(self, llama_model_dir):
        idx = load_shard_index(llama_model_dir)
        scales = collect_activation_scales_streaming(
            llama_model_dir, _FakeTokenizer(), idx, n_samples=4, seq_len=8, verbose=False
        )
        assert scales is not None
        # 7 nn.Linear submodules per layer (q/k/v/o + gate/up/down) x 2 layers
        assert len(scales) == 14
        for layer_idx in range(N_LAYERS):
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                assert f"model.layers.{layer_idx}.{name}" in scales

    def test_scale_shapes_match_in_features(self, llama_model_dir):
        idx = load_shard_index(llama_model_dir)
        scales = collect_activation_scales_streaming(
            llama_model_dir, _FakeTokenizer(), idx, n_samples=4, seq_len=8, verbose=False
        )
        assert scales["model.layers.0.self_attn.q_proj"].shape == (HIDDEN,)
        assert scales["model.layers.0.mlp.down_proj"].shape == (INTER,)

    def test_scales_are_non_degenerate(self, llama_model_dir):
        idx = load_shard_index(llama_model_dir)
        scales = collect_activation_scales_streaming(
            llama_model_dir, _FakeTokenizer(), idx, n_samples=4, seq_len=8, verbose=False
        )
        for name, s in scales.items():
            assert not np.isnan(s).any(), f"{name} has NaN"
            assert not np.isinf(s).any(), f"{name} has Inf"
            assert np.any(s > 0), f"{name} is all-zero — hooks likely never fired"

    def test_no_calibration_samples_after_tokenization_returns_none(self, llama_model_dir):
        idx = load_shard_index(llama_model_dir)

        class _EmptyTokenizer:
            def encode(self, text, add_special_tokens=True):
                return []

        scales = collect_activation_scales_streaming(
            llama_model_dir, _EmptyTokenizer(), idx, n_samples=2, seq_len=8, verbose=False
        )
        assert scales is None


class TestStreamingCalibrationFallback:
    def test_unsupported_moe_architecture_returns_none(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _build_synthetic_llama_model(model_dir, model_type="mixtral")
        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir, _FakeTokenizer(), idx, n_samples=2, seq_len=8, verbose=False
        )
        assert scales is None

    def test_missing_embed_tokens_returns_none(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _build_synthetic_llama_model(model_dir)
        # Rewrite the index without the embed_tokens entry.
        index_path = model_dir / "model.safetensors.index.json"
        data = json.loads(index_path.read_text())
        data["weight_map"] = {
            k: v for k, v in data["weight_map"].items() if "embed_tokens" not in k
        }
        index_path.write_text(json.dumps(data))

        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir, _FakeTokenizer(), idx, n_samples=2, seq_len=8, verbose=False
        )
        assert scales is None


class TestStreamingCalibrationPeakMemory:
    def test_only_one_layers_shard_data_resident_at_a_time(self, llama_model_dir, monkeypatch):
        load_calls = []
        from safetensors.numpy import load_file as real_st_load

        def _tracking_load(path):
            load_calls.append(Path(path).name)
            return real_st_load(path)

        monkeypatch.setattr("safetensors.numpy.load_file", _tracking_load)

        idx = load_shard_index(llama_model_dir)
        collect_activation_scales_streaming(
            llama_model_dir, _FakeTokenizer(), idx, n_samples=2, seq_len=8, verbose=False
        )

        # embed_tokens shard read once, plus once per layer needing a shard —
        # each shard file read exactly as many times as layers reference it,
        # never all shards loaded eagerly up front.
        assert load_calls.count("model-00001-of-00002.safetensors") == 2  # embed + layer 0
        assert load_calls.count("model-00002-of-00002.safetensors") == 1  # layer 1
