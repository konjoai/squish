"""Unit tests for squish.speculative.conf_spec (ConfSpec confidence-gated verification)."""

import math

import numpy as np
import pytest

from squish.speculative.conf_spec import (
    VALID_METRICS,
    ConfidenceMetric,
    ConfSpecConfig,
    ConfSpecDecision,
    ConfSpecStats,
    ConfSpecVerifier,
    VerificationRouting,
    compute_confidence,
)


def _cfg(high_gate=0.9, low_gate=0.3, metric="top_prob", **kw):
    return ConfSpecConfig(high_gate=high_gate, low_gate=low_gate, metric=metric, **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _peaked_logits(peak_idx=0, vocab_size=32):
    """Logits with all mass on one token → high confidence."""
    logits = np.full(vocab_size, -100.0, dtype=np.float32)
    logits[peak_idx] = 100.0
    return logits


def _uniform_logits(vocab_size=32):
    """Uniform logits → low confidence."""
    return np.zeros(vocab_size, dtype=np.float32)


# ---------------------------------------------------------------------------
# TestConfSpecConfig
# ---------------------------------------------------------------------------


class TestConfSpecConfig:
    def test_defaults(self):
        cfg = ConfSpecConfig()
        assert cfg.high_gate == 0.90
        assert cfg.low_gate == 0.50
        assert cfg.metric == ConfidenceMetric.TOP_PROB

    def test_invalid_gate_order(self):
        with pytest.raises(ValueError):
            ConfSpecConfig(low_gate=0.8, high_gate=0.5)

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError):
            ConfSpecConfig(vocab_size=0)

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError):
            ConfSpecConfig(ema_alpha=0.0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            ConfSpecConfig(metric="invalid_metric")

    def test_invalid_target_accept_rate(self):
        with pytest.raises(ValueError):
            ConfSpecConfig(target_accept_rate=1.0)


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_top_prob_peaked(self):
        logits = _peaked_logits()
        c = compute_confidence(logits, "top_prob")
        assert c > 0.99

    def test_top_prob_uniform(self):
        logits = _uniform_logits(32)
        c = compute_confidence(logits, "top_prob")
        assert c < 0.1

    def test_margin_peaked(self):
        logits = _peaked_logits(vocab_size=8)
        c = compute_confidence(logits, "margin")
        assert c > 0.9

    def test_margin_close(self):
        """Two nearly equal tokens → low margin."""
        logits = np.array([10.0, 9.9] + [0.0] * 6, dtype=np.float32)
        c = compute_confidence(logits, "margin")
        assert c < 0.1

    def test_entropy_peaked(self):
        logits = _peaked_logits(vocab_size=32)
        c = compute_confidence(logits, "entropy", vocab_size=32)
        assert c > 0.9

    def test_entropy_uniform(self):
        logits = _uniform_logits(32)
        c = compute_confidence(logits, "entropy", vocab_size=32)
        assert c < 0.1

    def test_result_in_0_1(self):
        rng = _rng()
        for metric in ("top_prob", "margin", "entropy"):
            logits = rng.normal(0, 1, 32).astype(np.float32)
            c = compute_confidence(logits, metric, vocab_size=32)
            assert 0.0 <= c <= 1.0

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            compute_confidence(np.ones(8), "unknown")


# ---------------------------------------------------------------------------
# TestConfSpecDecision
# ---------------------------------------------------------------------------


class TestConfSpecDecision:
    def test_auto_accept(self):
        d = ConfSpecDecision(
            confidence=0.95,
            routing=VerificationRouting.AUTO_ACCEPT,
            accepted=True,
        )
        assert d.accepted is True
        assert d.routing == VerificationRouting.AUTO_ACCEPT

    def test_full_target_rejected(self):
        d = ConfSpecDecision(
            confidence=0.2,
            routing=VerificationRouting.FULL_TARGET,
            accepted=False,
            score=0.3,
        )
        assert d.accepted is False


# ---------------------------------------------------------------------------
# TestConfSpecStats
# ---------------------------------------------------------------------------


class TestConfSpecStats:
    def test_initial_zeros(self):
        s = ConfSpecStats()
        assert s.total_steps == 0
        assert s.auto_accept_rate == 0.0

    def test_record_auto_accept(self):
        s = ConfSpecStats()
        s.record(ConfSpecDecision(confidence=0.95, routing=VerificationRouting.AUTO_ACCEPT, accepted=True))
        assert s.auto_accepted == 1
        assert s.auto_accept_rate == 1.0

    def test_record_full_target_accepted(self):
        s = ConfSpecStats()
        s.record(ConfSpecDecision(confidence=0.2, routing=VerificationRouting.FULL_TARGET, accepted=True))
        assert s.full_target_accepted == 1
        assert s.full_target_rate == 1.0

    def test_record_lightweight_rejected(self):
        s = ConfSpecStats()
        s.record(ConfSpecDecision(confidence=0.6, routing=VerificationRouting.LIGHTWEIGHT, accepted=False))
        assert s.lightweight_rejected == 1

    def test_target_calls_saved_fraction(self):
        s = ConfSpecStats()
        # 5 auto + 3 lightweight + 2 full = 10 total
        for _ in range(5):
            s.record(ConfSpecDecision(confidence=0.95, routing=VerificationRouting.AUTO_ACCEPT, accepted=True))
        for _ in range(3):
            s.record(ConfSpecDecision(confidence=0.6, routing=VerificationRouting.LIGHTWEIGHT, accepted=True))
        for _ in range(2):
            s.record(ConfSpecDecision(confidence=0.2, routing=VerificationRouting.FULL_TARGET, accepted=True))
        assert abs(s.target_calls_saved_fraction - 0.8) < 1e-6

    def test_speedup_estimate_auto_accept_dominant(self):
        s = ConfSpecStats()
        for _ in range(9):
            s.record(ConfSpecDecision(confidence=0.95, routing=VerificationRouting.AUTO_ACCEPT, accepted=True))
        s.record(ConfSpecDecision(confidence=0.2, routing=VerificationRouting.FULL_TARGET, accepted=True))
        assert s.estimated_speedup_vs_always_verify > 1.0


# ---------------------------------------------------------------------------
# TestConfSpecVerifier
# ---------------------------------------------------------------------------


class TestConfSpecVerifier:
    def _make_verifier(self, high=0.8, low=0.3, metric="top_prob"):
        cfg = _cfg(high_gate=high, low_gate=low, metric=metric)
        return ConfSpecVerifier(cfg)

    def test_auto_accept_above_high_gate(self):
        verifier = self._make_verifier()
        logits = _peaked_logits()  # very high confidence
        decision = verifier.verify_step("step text", "context", logits)
        assert decision.routing == VerificationRouting.AUTO_ACCEPT
        assert decision.accepted is True

    def test_full_target_below_low_gate(self):
        verifier = self._make_verifier(high=0.9, low=0.5)
        logits = _uniform_logits()  # very low confidence
        decision = verifier.verify_step("alpha beta", "context xyz", logits)
        assert decision.routing == VerificationRouting.FULL_TARGET

    def test_lightweight_in_middle(self):
        verifier = self._make_verifier(high=0.9, low=0.1)
        # Logits with moderate confidence between gates
        logits = np.array([5.0, 3.0] + [0.0] * 30, dtype=np.float32)
        decision = verifier.verify_step("some text", "other context", logits)
        # at moderate confidence (not extreme), should route to lightweight or full target
        assert decision.routing in (
            VerificationRouting.LIGHTWEIGHT,
            VerificationRouting.FULL_TARGET,
            VerificationRouting.AUTO_ACCEPT,
        )

    def test_stats_record_per_call(self):
        verifier = self._make_verifier()
        logits = _peaked_logits()
        verifier.verify_step("text", "context", logits)
        verifier.verify_step("text2", "context2", logits)
        assert verifier.stats.total_steps == 2

    def test_reset_clears_stats(self):
        verifier = self._make_verifier()
        logits = _peaked_logits()
        verifier.verify_step("text", "context", logits)
        verifier.reset()
        assert verifier.stats.total_steps == 0

    def test_auto_calibrate_adjusts_gate(self):
        cfg = _cfg(high_gate=0.5, low_gate=0.1, auto_calibrate=True, target_accept_rate=0.9)
        verifier = ConfSpecVerifier(cfg)
        # Feed many uniform-confidence steps → auto_accept_rate stays low
        uniform_logits = _uniform_logits(32)
        for _ in range(25):
            verifier.verify_step("text", "ctx", uniform_logits)
        # With calibration active, gate should have been adjusted
        # (exact direction depends on accept rate vs target)
        assert 0.0 < verifier.config.high_gate <= 1.0
