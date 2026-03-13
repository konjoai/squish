"""
tests/test_layer_skip_unit.py

Unit tests for squish/layer_skip.py — 100% coverage.
"""

import numpy as np
import pytest

from squish.token.layer_skip import (
    ConfidenceEstimator,
    EarlyExitConfig,
    EarlyExitDecoder,
    EarlyExitStats,
)

# ---------------------------------------------------------------------------
# EarlyExitConfig
# ---------------------------------------------------------------------------

class TestEarlyExitConfig:
    def test_defaults(self):
        cfg = EarlyExitConfig()
        assert cfg.num_layers == 32
        assert cfg.exit_layer == 16
        assert 0 < cfg.confidence_threshold <= 1
        assert cfg.mode in ("early_exit", "self_speculative")
        assert cfg.gamma >= 1

    def test_both_modes(self):
        for mode in ("early_exit", "self_speculative"):
            cfg = EarlyExitConfig(mode=mode)
            assert cfg.mode == mode

    @pytest.mark.parametrize("kwargs, exc", [
        ({"num_layers": 1},               "num_layers"),
        ({"exit_layer": 0},               "exit_layer"),
        ({"exit_layer": 32},              "exit_layer"),
        ({"confidence_threshold": 0.0},   "confidence_threshold"),
        ({"confidence_threshold": 1.1},   "confidence_threshold"),
        ({"mode": "unknown"},             "mode"),
        ({"gamma": 0},                    "gamma"),
        ({"confidence_metric": "bad"},    "confidence_metric"),
    ])
    def test_validation_errors(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            EarlyExitConfig(**kwargs)

    def test_exit_layer_at_max_minus_one(self):
        # Should not raise
        cfg = EarlyExitConfig(num_layers=4, exit_layer=3)
        assert cfg.exit_layer == 3

    def test_exit_layer_equals_num_layers_raises(self):
        with pytest.raises(ValueError, match="exit_layer"):
            EarlyExitConfig(num_layers=4, exit_layer=4)


# ---------------------------------------------------------------------------
# ConfidenceEstimator
# ---------------------------------------------------------------------------

class TestConfidenceEstimator:
    def _logits(self, probs):
        return np.log(np.array(probs, dtype=np.float64) + 1e-12)

    def test_max_prob_certain(self):
        est = ConfidenceEstimator("max_prob")
        logits = np.array([100.0, -100.0, -100.0])
        assert est.estimate(logits) == pytest.approx(1.0, abs=1e-4)

    def test_max_prob_uniform(self):
        est = ConfidenceEstimator("max_prob")
        logits = np.zeros(4)  # uniform
        assert est.estimate(logits) == pytest.approx(0.25, abs=1e-4)

    def test_margin_high(self):
        est = ConfidenceEstimator("margin")
        logits = np.array([100.0, 0.0, 0.0])
        conf = est.estimate(logits)
        assert conf > 0.9

    def test_margin_uniform(self):
        est = ConfidenceEstimator("margin")
        logits = np.zeros(4)
        conf = est.estimate(logits)
        assert conf == pytest.approx(0.0, abs=1e-4)

    def test_neg_entropy_certain(self):
        est = ConfidenceEstimator("neg_entropy")
        logits = np.array([1000.0, -1000.0, -1000.0])
        conf = est.estimate(logits)
        assert conf > 0.95

    def test_neg_entropy_uniform(self):
        est = ConfidenceEstimator("neg_entropy")
        logits = np.zeros(8)
        conf = est.estimate(logits)
        assert conf < 0.05

    def test_neg_entropy_single_token(self):
        # Single-token vocab: entropy=0, h_max=1 → conf=1 (maximally certain)
        est = ConfidenceEstimator("neg_entropy")
        logits = np.array([1.0])
        conf = est.estimate(logits)
        assert conf >= 0.9

    def test_margin_single_token(self):
        # Single-token vocab: only one candidate, margin returns max_prob=1
        est = ConfidenceEstimator("margin")
        logits = np.array([5.0])
        conf = est.estimate(logits)
        assert conf == pytest.approx(1.0, abs=1e-4)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            ConfidenceEstimator("unknown")

    def test_empty_logits_raises(self):
        est = ConfidenceEstimator()
        with pytest.raises(ValueError, match="non-empty"):
            est.estimate(np.array([]))

    def test_non_1d_raises(self):
        est = ConfidenceEstimator()
        with pytest.raises(ValueError, match="1-D"):
            est.estimate(np.ones((3, 3)))

    def test_top_token(self):
        est = ConfidenceEstimator()
        logits = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        assert est.top_token(logits) == 3

    def test_confidence_in_range(self):
        rng = np.random.default_rng(7)
        for metric in ("max_prob", "margin", "neg_entropy"):
            est = ConfidenceEstimator(metric)
            for _ in range(20):
                logits = rng.standard_normal(50)
                c = est.estimate(logits)
                assert 0.0 <= c <= 1.0, f"{metric}: {c}"


# ---------------------------------------------------------------------------
# EarlyExitStats
# ---------------------------------------------------------------------------

class TestEarlyExitStats:
    def test_acceptance_rate_no_draft(self):
        s = EarlyExitStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate_with_data(self):
        s = EarlyExitStats(accepted_draft=3, rejected_draft=1)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_early_exit_rate(self):
        s = EarlyExitStats(total_tokens_generated=10, early_exits=7)
        assert s.early_exit_rate == pytest.approx(0.7)

    def test_early_exit_rate_zero(self):
        s = EarlyExitStats()
        assert s.early_exit_rate == 0.0


# ---------------------------------------------------------------------------
# EarlyExitDecoder — early_exit mode
# ---------------------------------------------------------------------------

class TestEarlyExitDecoderEarlyExit:
    """Test the decoder using a simple toy forward function."""

    def _make_forward(self, vocab=20, exit_layer=8, num_layers=16):
        """Return a forward callable that:
        - with layer_limit=exit_layer → returns confident logits for token 1
        - with layer_limit=None      → returns confident logits for token 2
        """
        def forward(ids, layer_limit=None):
            logits = np.full(vocab, -10.0)
            if layer_limit is not None and layer_limit < num_layers:
                logits[1] = 10.0   # early exit → always picks token 1
            else:
                logits[2] = 10.0   # full forward → always picks token 2
            return logits
        return forward

    def test_early_exit_generates_correct_length(self):
        cfg = EarlyExitConfig(
            num_layers=16, exit_layer=8,
            confidence_threshold=0.5,
            mode="early_exit",
        )
        fwd = self._make_forward()
        dec = EarlyExitDecoder(fwd, cfg)
        ids, stats = dec.generate([0], max_new_tokens=10)
        assert len(ids) == 11   # 1 prompt + 10 new
        assert stats.total_tokens_generated == 10

    def test_early_exit_path_when_confident(self):
        cfg = EarlyExitConfig(
            num_layers=16, exit_layer=8,
            confidence_threshold=0.1,   # low → always exits early
            mode="early_exit",
        )
        fwd = self._make_forward()
        dec = EarlyExitDecoder(fwd, cfg)
        ids, stats = dec.generate([0], max_new_tokens=5)
        assert stats.early_exits == 5
        assert stats.full_forwards == 0
        assert all(t == 1 for t in ids[1:])  # token 1 from early exit

    def test_full_forward_path_when_not_confident(self):
        """Force low confidence on early exit → full forward always taken."""
        vocab = 20

        def forward(ids, layer_limit=None):
            logits = np.zeros(vocab)  # uniform → max_prob=1/vocab ≈ 0.05
            if layer_limit is None:
                logits[2] = 100.0     # only full forward is confident
            return logits

        cfg = EarlyExitConfig(
            num_layers=16, exit_layer=8,
            confidence_threshold=0.99,  # near-certain required
            mode="early_exit",
        )
        dec = EarlyExitDecoder(forward, cfg)
        ids, stats = dec.generate([0], max_new_tokens=5)
        assert stats.full_forwards == 5
        assert stats.early_exits == 0
        assert all(t == 2 for t in ids[1:])


# ---------------------------------------------------------------------------
# EarlyExitDecoder — self_speculative mode
# ---------------------------------------------------------------------------

class TestEarlyExitDecoderSelfSpeculative:
    def _make_fwd(self, vocab=10, draft_tok=3, verify_tok=3):
        """Draft and verify always agree → all accepted."""
        def forward(ids, layer_limit=None):
            logits = np.full(vocab, -10.0)
            logits[draft_tok if layer_limit is not None else verify_tok] = 10.0
            return logits
        return forward

    def test_self_spec_all_accepted(self):
        cfg = EarlyExitConfig(
            num_layers=16, exit_layer=8,
            confidence_threshold=0.5,
            mode="self_speculative",
            gamma=3,
        )
        fwd = self._make_fwd(draft_tok=4, verify_tok=4)
        dec = EarlyExitDecoder(fwd, cfg)
        ids, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens_generated == 6
        assert stats.accepted_draft > 0

    def test_self_spec_rejection_inserts_verifier_token(self):
        """Draft picks token 3, verifier picks token 5 → mismatch → stats updated."""
        vocab = 10

        def fwd(ids, layer_limit=None):
            logits = np.full(vocab, -10.0)
            if layer_limit is not None:
                logits[3] = 10.0  # draft
            else:
                logits[5] = 10.0  # verify
            return logits

        cfg = EarlyExitConfig(
            num_layers=16, exit_layer=8,
            confidence_threshold=0.5,
            mode="self_speculative",
            gamma=2,
        )
        dec = EarlyExitDecoder(fwd, cfg)
        ids, stats = dec.generate([0], max_new_tokens=4)
        assert stats.rejected_draft > 0
        # Verifier's correction must appear in ids
        assert 5 in ids

    def test_self_spec_length(self):
        cfg = EarlyExitConfig(
            num_layers=8, exit_layer=4,
            mode="self_speculative",
            gamma=2,
        )
        fwd = self._make_fwd()
        dec = EarlyExitDecoder(fwd, cfg)
        ids, stats = dec.generate(list(range(3)), max_new_tokens=8)
        assert stats.total_tokens_generated == 8
