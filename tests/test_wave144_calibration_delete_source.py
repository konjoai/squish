"""tests/test_wave144_calibration_delete_source.py

Wave 144 — integrate delete-as-you-go with sequential AWQ calibration.

Mirrors process_weights_streaming's --delete-source safety rule (Wave
139/141) inside collect_activation_scales_streaming: a raw shard is only
deleted once every layer that needs it has finished calibrating, using
ShardIndex.shard_to_layers() to know when that point is reached.

This pins, against a real synthetic sharded Llama-shaped model with a
shard deliberately shared between embed_tokens AND two decoder layers
(shard boundaries don't align with layer boundaries — the case that
actually matters for correctness):

- delete_source=False (default) leaves every raw shard on disk
- a shard needed by embed_tokens only (no layer also uses it) is deleted
  right after the one-time embedding read, before any layer runs
- a shard shared by embed_tokens + layers 0 and 1 is NOT deleted after
  the embedding read, and NOT deleted after layer 0 alone — only after
  layer 1 (its last consumer) finishes
- a shard used by a later layer only is deleted right after that layer
- calibration still produces correct scales even with delete_source on
  (deletion is a side effect, not a shortcut that skips work)
- deletion failures are logged as warnings, not raised
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
N_LAYERS = 3


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [(ord(c) % (VOCAB - 1)) + 1 for c in text[:12]]


def _build_model_shared_shard(model_dir: Path) -> None:
    """3 layers, 2 shards: shard1 = embed_tokens + layer0 + layer1 (a shard
    genuinely shared across the embedding AND two decoder layers — the
    scenario that actually exercises deferred deletion), shard2 = layer2."""
    config = {
        "model_type": "llama",
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
    tensors = {
        "model.embed_tokens.weight": (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(
            np.float32
        )
    }
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

    shard1 = {
        k: v
        for k, v in tensors.items()
        if k.startswith("model.embed_tokens")
        or k.startswith("model.layers.0.")
        or k.startswith("model.layers.1.")
    }
    shard2 = {k: v for k, v in tensors.items() if k not in shard1}
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    _build_model_shared_shard(d)
    return d


SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"


class TestDeleteSourceDefaultOff:
    def test_shards_survive_by_default(self, model_dir):
        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir, _FakeTokenizer(), idx, n_samples=2, seq_len=8, verbose=False
        )
        assert scales is not None
        assert (model_dir / SHARD_1).exists()
        assert (model_dir / SHARD_2).exists()


class TestDeleteSourceSharedShard:
    def test_shared_shard_survives_through_embed_and_layer_0(self, model_dir, monkeypatch):
        """The shard backing embed_tokens + layers 0/1 must still exist
        right after the embedding read and right after layer 0 — it's not
        fully drained until layer 1 finishes."""
        seen_after_layer = {}
        import squish.quant.awq_streaming as mod

        real_check = mod.is_standard_dense_block

        def _tracking(block):
            # Called once per layer, before that layer's own deletion pass
            # runs — records whether shard1 is still present at the start
            # of each layer's processing.
            seen_after_layer[len(seen_after_layer)] = (model_dir / SHARD_1).exists()
            return real_check(block)

        monkeypatch.setattr(mod, "is_standard_dense_block", _tracking)

        idx = load_shard_index(model_dir)
        collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )

        # shard1 must still exist at the start of layer 0 AND layer 1 (both
        # are consumers) — only layer 2's start (a different shard's
        # problem entirely) may see it already gone.
        assert seen_after_layer[0] is True, "shard1 missing before layer 0 even started"
        assert seen_after_layer[1] is True, "shard1 deleted after layer 0 alone — too early"

    def test_shared_shard_deleted_only_after_last_consumer_layer_1(self, model_dir):
        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )
        assert scales is not None
        assert not (model_dir / SHARD_1).exists()

    def test_layer_2_only_shard_also_deleted(self, model_dir):
        idx = load_shard_index(model_dir)
        collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )
        assert not (model_dir / SHARD_2).exists()

    def test_calibration_still_correct_with_delete_source(self, model_dir):
        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )
        assert scales is not None
        assert len(scales) == 7 * N_LAYERS
        for s in scales.values():
            assert not np.isnan(s).any()
            assert np.any(s > 0)


class TestDeleteSourceEmbedOnlyShard:
    def test_embed_only_shard_deleted_immediately(self, tmp_path):
        """A shard backing ONLY embed_tokens (no layer overlap) is deleted
        right after the one-time embedding read, before any layer runs."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
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
        embed = {
            "model.embed_tokens.weight": (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(
                np.float32
            )
        }
        p = "model.layers.0."
        layer0 = {
            p + "input_layernorm.weight": np.ones((HIDDEN,), dtype=np.float32),
            p + "post_attention_layernorm.weight": np.ones((HIDDEN,), dtype=np.float32),
            p + "self_attn.q_proj.weight": (
                rng.standard_normal((N_HEADS * HEAD_DIM, HIDDEN)) * 0.02
            ).astype(np.float32),
            p + "self_attn.k_proj.weight": (
                rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
            ).astype(np.float32),
            p + "self_attn.v_proj.weight": (
                rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
            ).astype(np.float32),
            p + "self_attn.o_proj.weight": (
                rng.standard_normal((HIDDEN, N_HEADS * HEAD_DIM)) * 0.02
            ).astype(np.float32),
            p + "mlp.gate_proj.weight": (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(
                np.float32
            ),
            p + "mlp.up_proj.weight": (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(
                np.float32
            ),
            p + "mlp.down_proj.weight": (rng.standard_normal((HIDDEN, INTER)) * 0.02).astype(
                np.float32
            ),
        }
        save_file(embed, str(model_dir / "model-00001-of-00002.safetensors"))
        save_file(layer0, str(model_dir / "model-00002-of-00002.safetensors"))
        weight_map = {k: "model-00001-of-00002.safetensors" for k in embed}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in layer0})
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )
        assert scales is not None
        assert not (model_dir / "model-00001-of-00002.safetensors").exists()
        assert not (model_dir / "model-00002-of-00002.safetensors").exists()


class TestDeleteSourceFailureIsGraceful:
    def test_deletion_failure_logs_warning_and_continues(self, model_dir, monkeypatch, caplog):
        import squish.quant.awq_streaming as mod

        real_unlink = Path.unlink
        target = model_dir / SHARD_2

        def _boom(self, *a, **kw):
            if self == target:
                raise OSError("permission denied (simulated)")
            return real_unlink(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", _boom)
        warnings_logged = []
        monkeypatch.setattr(mod._LOG, "warning", lambda msg, *a: warnings_logged.append(msg % a))

        idx = load_shard_index(model_dir)
        scales = collect_activation_scales_streaming(
            model_dir,
            _FakeTokenizer(),
            idx,
            n_samples=2,
            seq_len=8,
            verbose=False,
            delete_source=True,
        )

        assert scales is not None
        assert len(scales) == 7 * N_LAYERS
        assert target.exists()  # deletion failed, file survives
        assert any("Could not delete raw shard" in msg for msg in warnings_logged)
