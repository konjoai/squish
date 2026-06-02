"""Tests for squish.kv.head_importance — per-head importance scoring."""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.head_importance import (
    HeadImportanceAnalyzer,
    HeadImportanceScores,
)


def _layer_samples(n_tokens: int, n_heads: int, head_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            for _ in range(n_tokens)]


class TestAnalyzerInit:
    def test_default_weights_normalised(self):
        a = HeadImportanceAnalyzer()
        w = a.weights
        assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
        assert set(w.keys()) == {"variance", "concentration", "magnitude"}

    def test_explicit_weights(self):
        a = HeadImportanceAnalyzer(weights={"variance": 2, "magnitude": 1})
        w = a.weights
        # absent key gets 0; others normalised to sum=1
        assert pytest.approx(w["variance"], abs=1e-9) == 2 / 3
        assert pytest.approx(w["concentration"], abs=1e-9) == 0.0
        assert pytest.approx(w["magnitude"], abs=1e-9) == 1 / 3

    def test_unknown_weight_key_raises(self):
        with pytest.raises(ValueError, match="unknown weight keys"):
            HeadImportanceAnalyzer(weights={"foo": 1.0})

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="sum to > 0"):
            HeadImportanceAnalyzer(
                weights={"variance": 0, "concentration": 0, "magnitude": 0},
            )


class TestScore:
    def test_score_shape_and_dtype(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(20, n_heads=4, head_dim=8, seed=i)
                   for i in range(3)]
        scores = a.score(samples)
        assert isinstance(scores, HeadImportanceScores)
        assert scores.per_layer.shape == (3, 4)
        assert scores.per_layer.dtype == np.float32
        assert scores.n_layers == 3
        assert scores.n_heads == 4

    def test_score_values_in_unit_interval(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(32, n_heads=8, head_dim=16, seed=i)
                   for i in range(2)]
        scores = a.score(samples)
        assert scores.per_layer.min() >= 0.0
        assert scores.per_layer.max() <= 1.0

    def test_high_variance_head_outranks_zero_head(self):
        # craft a layer where head 0 is all-zero and head 1 has high variance.
        rng = np.random.default_rng(42)
        n_tokens, n_heads, head_dim = 16, 2, 8
        tokens = []
        for _ in range(n_tokens):
            tok = np.zeros((n_heads, head_dim), dtype=np.float32)
            tok[1] = rng.standard_normal(head_dim) * 5.0
            tokens.append(tok)
        scores = HeadImportanceAnalyzer().score([tokens])
        assert scores.per_layer[0, 1] > scores.per_layer[0, 0]

    def test_constant_layer_normalises_to_half(self):
        # all heads identical → after min-max normalisation, all 0.5
        n_tokens, n_heads, head_dim = 8, 3, 4
        const = np.ones((n_heads, head_dim), dtype=np.float32)
        tokens = [const.copy() for _ in range(n_tokens)]
        scores = HeadImportanceAnalyzer().score([tokens])
        assert np.allclose(scores.per_layer[0], 0.5, atol=1e-6)

    def test_empty_layer_in_middle(self):
        a = HeadImportanceAnalyzer()
        samples = [
            _layer_samples(16, 4, 8, seed=1),
            [],                                 # empty middle layer
            _layer_samples(16, 4, 8, seed=2),
        ]
        scores = a.score(samples)
        # empty layer becomes a row of zeros (no raw signal) → normalised to 0.5
        assert scores.per_layer.shape == (3, 4)

    def test_n_heads_expected_mismatch_raises(self):
        a = HeadImportanceAnalyzer(n_heads_expected=8)
        samples = [_layer_samples(8, n_heads=4, head_dim=16)]
        with pytest.raises(ValueError, match="n_heads mismatch"):
            a.score(samples)

    def test_inconsistent_head_dim_across_layers_raises(self):
        a = HeadImportanceAnalyzer()
        samples = [
            _layer_samples(8, n_heads=4, head_dim=16, seed=1),
            _layer_samples(8, n_heads=4, head_dim=32, seed=2),
        ]
        with pytest.raises(ValueError, match="differs from"):
            a.score(samples)


class TestMaskAndPrune:
    def test_head_mask_threshold_invalid(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(8, 4, 8)]
        scores = a.score(samples)
        with pytest.raises(ValueError):
            scores.head_mask(1.5)

    def test_pruned_count_monotone_in_threshold(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(20, 4, 8, seed=i) for i in range(2)]
        scores = a.score(samples)
        assert scores.pruned_count(0.0) <= scores.pruned_count(0.5)
        assert scores.pruned_count(0.5) <= scores.pruned_count(1.0)

    def test_top_k_per_layer(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(16, 8, 8, seed=7)]
        scores = a.score(samples)
        keep = scores.top_k_per_layer(3)
        assert keep.sum(axis=-1)[0] == 3
        # k >= n_heads → all True
        keep_all = scores.top_k_per_layer(8)
        assert keep_all.all()

    def test_top_k_zero_raises(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(8, 4, 8)]
        scores = a.score(samples)
        with pytest.raises(ValueError):
            scores.top_k_per_layer(0)

    def test_prune_heads_returns_scores_and_mask(self):
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(16, 4, 8, seed=i) for i in range(2)]
        scores, mask = a.prune_heads(samples, threshold=0.3)
        assert mask.shape == scores.per_layer.shape
        assert mask.dtype == bool


class TestSerialisation:
    def test_to_json_roundtrip(self):
        import json
        a = HeadImportanceAnalyzer()
        samples = [_layer_samples(8, 4, 8, seed=i) for i in range(2)]
        scores = a.score(samples)
        encoded = json.dumps(scores.to_json())
        decoded = json.loads(encoded)
        assert decoded["n_layers"] == 2
        assert decoded["n_heads"] == 4
        assert len(decoded["per_layer"]) == 2
        assert len(decoded["per_layer"][0]) == 4
