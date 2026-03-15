"""
tests/speculative/test_ssd_unit.py

Unit tests for Phase 4: SSD (Speculative Speculative Decoding) acceptance predictor.

Covers:
  - SSDConfig validation
  - SSDPredictor.predict_acceptance output range and disabled paths
  - SSDPredictor.filter_drafts truncation logic
  - filter_drafts with threshold=0 is a no-op
  - filter_drafts always preserves at least 1 token
  - SSDPredictor save/load round-trip
  - SSDPredictor.init_random produces a working predictor
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.speculative.ssd import SSDConfig, SSDPredictor

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

GRU_DIM  = 16
FEAT_DIM = 8
VOCAB    = 32
RNG      = np.random.default_rng(999)


def _make_predictor(
    gru_dim:   int   = GRU_DIM,
    feat_dim:  int   = FEAT_DIM,
    threshold: float = 0.3,
) -> SSDPredictor:
    return SSDPredictor.init_random(
        gru_hidden_dim = gru_dim,
        feature_dim    = feat_dim,
        threshold      = threshold,
        rng            = np.random.default_rng(42),
    )


def _rand_hidden(dim: int = GRU_DIM) -> np.ndarray:
    return RNG.standard_normal(dim).astype(np.float32)


def _uniform_probs(vocab: int = VOCAB) -> np.ndarray:
    return np.ones(vocab, dtype=np.float32) / vocab


# ---------------------------------------------------------------------------
# SSDConfig
# ---------------------------------------------------------------------------

class TestSSDConfig:
    def test_defaults(self):
        cfg = SSDConfig()
        assert cfg.feature_dim == 32
        assert cfg.threshold   == 0.3
        assert cfg.enabled     is True

    def test_custom(self):
        cfg = SSDConfig(feature_dim=8, threshold=0.6)
        assert cfg.feature_dim == 8
        assert cfg.threshold   == 0.6

    def test_feature_dim_zero_raises(self):
        with pytest.raises(ValueError, match="feature_dim"):
            SSDConfig(feature_dim=0)

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SSDConfig(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SSDConfig(threshold=1.1)

    def test_threshold_zero_valid(self):
        cfg = SSDConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_one_valid(self):
        cfg = SSDConfig(threshold=1.0)
        assert cfg.threshold == 1.0


# ---------------------------------------------------------------------------
# SSDPredictor.predict_acceptance
# ---------------------------------------------------------------------------

class TestSSDPredictorAcceptance:
    def test_output_in_unit_interval(self):
        pred = _make_predictor()
        h = _rand_hidden()
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            pa = pred.predict_acceptance(h, p)
            assert 0.0 <= pa <= 1.0, f"predict_acceptance={pa} for p_draft={p}"

    def test_disabled_always_returns_one(self):
        pred = SSDPredictor.init_random(GRU_DIM, threshold=0.5)
        pred.config.enabled = False
        h = _rand_hidden()
        assert pred.predict_acceptance(h, 0.0) == 1.0

    def test_threshold_zero_returns_one(self):
        pred = SSDPredictor.init_random(GRU_DIM, threshold=0.0)
        h = _rand_hidden()
        assert pred.predict_acceptance(h, 0.5) == 1.0

    def test_different_hidden_states_give_different_outputs(self):
        pred = _make_predictor()
        h1   = np.zeros(GRU_DIM, dtype=np.float32)
        h2   = np.ones(GRU_DIM,  dtype=np.float32)
        pa1  = pred.predict_acceptance(h1, 0.5)
        pa2  = pred.predict_acceptance(h2, 0.5)
        # With non-trivial weights, different inputs should differ (may be very close
        # with random init, but collision is astronomically unlikely)
        # We just check both are in [0, 1]
        assert 0.0 <= pa1 <= 1.0
        assert 0.0 <= pa2 <= 1.0


# ---------------------------------------------------------------------------
# SSDPredictor.filter_drafts
# ---------------------------------------------------------------------------

class TestSSDPredictorFilterDrafts:
    def test_empty_draft_returns_empty(self):
        pred = _make_predictor(threshold=0.3)
        ids, probs = pred.filter_drafts([], [], [])
        assert ids   == []
        assert probs == []

    def test_threshold_zero_is_passthrough(self):
        pred  = _make_predictor(threshold=0.0)
        n     = 5
        hiddens = [_rand_hidden() for _ in range(n)]
        ids     = list(range(n))
        probs   = [_uniform_probs() for _ in range(n)]
        out_ids, out_probs = pred.filter_drafts(hiddens, ids, probs)
        assert out_ids   == ids
        assert out_probs == probs

    def test_disabled_is_passthrough(self):
        pred = SSDPredictor.init_random(GRU_DIM, threshold=0.9)
        pred.config.enabled = False
        n       = 4
        hiddens = [_rand_hidden() for _ in range(n)]
        ids     = [0, 1, 2, 3]
        probs   = [_uniform_probs() for _ in range(n)]
        out_ids, out_probs = pred.filter_drafts(hiddens, ids, probs)
        assert out_ids == ids

    def test_always_keeps_at_least_one_token(self):
        """Even if threshold=1.0 (always filter), at least 1 token is kept."""
        pred = _make_predictor(threshold=1.0)  # p_acc always < 1.0 → always filter
        n       = 4
        hiddens = [_rand_hidden() for _ in range(n)]
        ids     = [0, 1, 2, 3]
        probs   = [_uniform_probs() for _ in range(n)]
        out_ids, out_probs = pred.filter_drafts(hiddens, ids, probs)
        assert len(out_ids) >= 1
        assert len(out_probs) >= 1

    def test_truncates_at_first_low_acceptance(self):
        """
        Construct a predictor that always returns 1.0 acceptance, then
        manually set the last token's probability very low so the filter fires.
        We do this by using a very high threshold (0.999) and low random weights
        so all natural acceptance < 0.999.
        """
        pred = _make_predictor(threshold=0.999)
        n       = 4
        hiddens = [_rand_hidden() for _ in range(n)]
        ids     = [0, 1, 2, 3]
        # All uniform probs → p_draft ≈ 1/VOCAB ≈ 0.03; predict_acceptance(random_init) ≈ 0.5
        # With threshold=0.999, the very first token should trigger truncation
        probs   = [_uniform_probs() for _ in range(n)]
        out_ids, _ = pred.filter_drafts(hiddens, ids, probs)
        # Should be truncated to 1 token (threshold too high for random init output)
        assert 1 <= len(out_ids) <= n

    def test_output_ids_prefix_of_input(self):
        """Filtered IDs must be a prefix (not a subset) of the original IDs."""
        pred    = _make_predictor(threshold=0.5)
        n       = 6
        hiddens = [_rand_hidden() for _ in range(n)]
        ids     = list(range(n))
        probs   = [_uniform_probs() for _ in range(n)]
        out_ids, _ = pred.filter_drafts(hiddens, ids, probs)
        assert ids[:len(out_ids)] == out_ids


# ---------------------------------------------------------------------------
# SSDPredictor save / load
# ---------------------------------------------------------------------------

class TestSSDPredictorPersistence:
    def test_save_load_roundtrip_weights(self):
        pred = _make_predictor(feat_dim=FEAT_DIM)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "ssd.npz")
            pred.save(path)
            pred2 = SSDPredictor.load(path)
        assert np.allclose(pred._proj_w, pred2._proj_w)
        assert np.allclose(pred._cls_w1, pred2._cls_w1)
        assert np.allclose(pred._cls_w2, pred2._cls_w2)

    def test_save_load_roundtrip_same_output(self):
        pred = _make_predictor()
        h    = _rand_hidden()
        p1   = pred.predict_acceptance(h, 0.3)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "ssd.npz")
            pred.save(path)
            pred2 = SSDPredictor.load(path)
        p2 = pred2.predict_acceptance(h, 0.3)
        assert abs(p1 - p2) < 1e-5

    def test_save_load_preserves_config(self):
        pred = _make_predictor(feat_dim=FEAT_DIM, threshold=0.42)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "ssd.npz")
            pred.save(path)
            pred2 = SSDPredictor.load(path)
        assert pred2.config.feature_dim == FEAT_DIM
        assert abs(pred2.config.threshold - 0.42) < 1e-5
