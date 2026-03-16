"""
tests/sampling/test_sampler.py

Unit tests for squish/sampling/sampler.py

Phase 6 — LLM-42 Determinism

Coverage:
  • SamplerConfig  — validation, field defaults
  • StructuredSampler  — lifecycle (reset, sample, update), determinism, pipeline steps
  • Standalone helpers — _softmax_f32, _apply_top_k, _apply_top_p, _apply_rep_penalty
  • Edge cases — zero temperature (greedy), all-inf logits, empty/oversize rep window
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from squish.sampling.sampler import (
    _SENTINEL,
    SamplerConfig,
    StructuredSampler,
    _apply_rep_penalty,
    _apply_top_k,
    _apply_top_p,
    _SentinelType,
    _softmax_f32,
)

# ---------------------------------------------------------------------------
# SamplerConfig — validation
# ---------------------------------------------------------------------------

class TestSamplerConfig:
    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.top_k == 0
        assert cfg.seed is None
        assert cfg.rep_penalty == 1.0
        assert cfg.rep_penalty_window == 64

    def test_valid_custom(self):
        cfg = SamplerConfig(temperature=0.7, top_p=0.9, top_k=50, seed=42,
                            rep_penalty=1.3, rep_penalty_window=32)
        assert cfg.temperature == 0.7
        assert cfg.seed == 42
        assert cfg.rep_penalty == 1.3

    def test_temperature_zero_allowed(self):
        cfg = SamplerConfig(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplerConfig(temperature=-0.1)

    def test_top_p_boundary_one_allowed(self):
        cfg = SamplerConfig(top_p=1.0)
        assert cfg.top_p == 1.0

    def test_top_p_zero_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplerConfig(top_p=0.0)

    def test_top_p_above_one_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplerConfig(top_p=1.001)

    def test_top_p_negative_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplerConfig(top_p=-0.5)

    def test_top_k_zero_allowed(self):
        cfg = SamplerConfig(top_k=0)
        assert cfg.top_k == 0

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplerConfig(top_k=-1)

    def test_rep_penalty_one_allowed(self):
        cfg = SamplerConfig(rep_penalty=1.0)
        assert cfg.rep_penalty == 1.0

    def test_rep_penalty_below_one_raises(self):
        with pytest.raises(ValueError, match="rep_penalty"):
            SamplerConfig(rep_penalty=0.9)

    def test_rep_penalty_window_zero_allowed(self):
        cfg = SamplerConfig(rep_penalty_window=0)
        assert cfg.rep_penalty_window == 0

    def test_rep_penalty_window_negative_raises(self):
        with pytest.raises(ValueError, match="rep_penalty_window"):
            SamplerConfig(rep_penalty_window=-1)


# ---------------------------------------------------------------------------
# _SentinelType singleton
# ---------------------------------------------------------------------------

class TestSentinel:
    def test_singleton(self):
        a = _SentinelType()
        b = _SentinelType()
        assert a is b

    def test_module_level_sentinel(self):
        assert _SENTINEL is _SentinelType()


# ---------------------------------------------------------------------------
# StructuredSampler — lifecycle
# ---------------------------------------------------------------------------

class TestStructuredSamplerLifecycle:
    def _make(self, **kwargs) -> StructuredSampler:
        return StructuredSampler(SamplerConfig(**kwargs))

    def test_initial_step_zero(self):
        s = self._make()
        assert s.step == 0

    def test_initial_rep_window_empty(self):
        s = self._make()
        assert s.rep_window == []

    def test_reset_clears_step(self):
        s = self._make(seed=1)
        logits = np.zeros(10, dtype=np.float32)
        s.sample(logits)
        s.reset()
        assert s.step == 0

    def test_reset_clears_rep_window(self):
        s = self._make(seed=1)
        s.update(5)
        s.update(9)
        assert len(s.rep_window) == 2
        s.reset()
        assert s.rep_window == []

    def test_update_appends_token(self):
        s = self._make()
        s.update(42)
        assert s.rep_window == [42]

    def test_update_trims_window(self):
        s = self._make(rep_penalty_window=3)
        for i in range(10):
            s.update(i)
        assert len(s.rep_window) == 3

    def test_update_zero_window_keeps_all(self):
        s = self._make(rep_penalty_window=0)
        for i in range(20):
            s.update(i)
        # rep_penalty_window=0 means unlimited
        assert len(s.rep_window) == 20

    def test_sample_increments_step(self):
        s = self._make(seed=7)
        logits = np.zeros(8, dtype=np.float32)
        assert s.step == 0
        s.sample(logits)
        assert s.step == 1
        s.sample(logits)
        assert s.step == 2

    def test_reset_seed_override(self):
        s = self._make(seed=42)
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        s.reset(seed=0)
        t1 = s.sample(logits)
        s.reset(seed=0)
        t2 = s.sample(logits)
        assert t1 == t2

    def test_reset_none_seed_override_non_deterministic(self):
        # reset with seed=None should be accepted (non-deterministic mode)
        s = self._make(seed=42)
        s.reset(seed=None)
        assert s.step == 0  # no crash

    def test_reset_omit_seed_uses_config(self):
        s = self._make(seed=99)
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        s.reset()  # omit → use config.seed = 99
        t1 = s.sample(logits)
        s.reset()
        t2 = s.sample(logits)
        assert t1 == t2  # deterministic


# ---------------------------------------------------------------------------
# StructuredSampler — determinism (LLM-42)
# ---------------------------------------------------------------------------

class TestStructuredSamplerDeterminism:
    def test_same_seed_same_first_token(self):
        logits = np.arange(16, dtype=np.float32)
        s = StructuredSampler(SamplerConfig(seed=42))
        s.reset()
        t1 = s.sample(logits)
        s.reset()
        t2 = s.sample(logits)
        assert t1 == t2

    def test_different_seeds_differ(self):
        logits = np.arange(100, dtype=np.float32)
        results = set()
        for seed in range(20):
            s = StructuredSampler(SamplerConfig(seed=seed))
            s.reset()
            results.add(s.sample(logits))
        # At least a few distinct tokens sampled — not all the same
        assert len(results) > 1

    def test_no_seed_non_deterministic(self):
        """Without a seed, two runs should differ (probabilistically)."""
        logits = np.zeros(100, dtype=np.float32)  # uniform distribution
        tokens = set()
        for _ in range(30):
            s = StructuredSampler(SamplerConfig(seed=None))
            s.reset()
            tokens.add(s.sample(logits))
        assert len(tokens) > 1

    def test_llm42_default_seed(self):
        """Seed 42 per-request produces identical multi-token sequences."""
        logits = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        s = StructuredSampler(SamplerConfig(seed=42))

        def _run():
            s.reset()
            return [s.sample(logits) for _ in range(5)]

        seq1 = _run()
        seq2 = _run()
        assert seq1 == seq2


# ---------------------------------------------------------------------------
# StructuredSampler — greedy (temperature=0)
# ---------------------------------------------------------------------------

class TestStructuredSamplerGreedy:
    def test_temperature_zero_returns_argmax(self):
        logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float32)
        s = StructuredSampler(SamplerConfig(temperature=0.0))
        s.reset()
        assert s.sample(logits) == 1  # index of max

    def test_temperature_zero_always_argmax(self):
        logits = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float32)
        s = StructuredSampler(SamplerConfig(temperature=0.0))
        for _ in range(10):
            s.reset()
            assert s.sample(logits) == 2


# ---------------------------------------------------------------------------
# StructuredSampler — repetition penalty
# ---------------------------------------------------------------------------

class TestStructuredSamplerRepPenalty:
    def test_rep_penalty_reduces_prob_of_recent_token(self):
        """A heavily penalised token should rarely be sampled."""
        rng = np.random.default_rng(0)
        vocab = 4
        # Logit of token 0 is highest, but it gets penalised
        logits = np.array([5.0, 1.0, 1.0, 1.0], dtype=np.float32)
        counts = [0, 0, 0, 0]
        s = StructuredSampler(SamplerConfig(seed=0, rep_penalty=5.0,
                                             rep_penalty_window=10))
        for _ in range(200):
            s.reset(seed=int(rng.integers(0, 10_000)))
            s.update(0)  # mark token 0 as recently seen
            t = s.sample(logits)
            counts[t] += 1
        # Token 0 (usually highest) should NOT dominate when penalised
        assert counts[0] < 150  # expected much lower than 200

    def test_penalty_window_limit_enforced(self):
        s = StructuredSampler(SamplerConfig(rep_penalty=2.0, rep_penalty_window=3))
        for i in range(10):
            s.update(i)
        assert len(s.rep_window) == 3
        assert s.rep_window == [7, 8, 9]


# ---------------------------------------------------------------------------
# StructuredSampler — top-k
# ---------------------------------------------------------------------------

class TestStructuredSamplerTopK:
    def test_top_k_restricts_vocabulary(self):
        """With top_k=1, the highest logit token must always be sampled."""
        logits = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        s = StructuredSampler(SamplerConfig(seed=42, top_k=1))
        for _ in range(5):
            s.reset()
            assert s.sample(logits) == 1


# ---------------------------------------------------------------------------
# StructuredSampler — top-p
# ---------------------------------------------------------------------------

class TestStructuredSamplerTopP:
    def test_top_p_small_restricts_to_high_mass_tokens(self):
        """With very small top_p, only the dominant token should appear."""
        logits = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s = StructuredSampler(SamplerConfig(seed=42, top_p=0.5))
        tokens = {s.sample(logits) for _ in range(20)}
        assert tokens == {0}


# ---------------------------------------------------------------------------
# _softmax_f32
# ---------------------------------------------------------------------------

class TestSoftmaxF32:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_dtype_float32(self):
        p = _softmax_f32(np.array([0.0, 1.0, 2.0]))
        assert p.dtype == np.float32

    def test_monotonic_order(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert p[0] < p[1] < p[2]

    def test_uniform_input_uniform_output(self):
        x = np.zeros(4, dtype=np.float32)
        p = _softmax_f32(x)
        np.testing.assert_allclose(p, [0.25, 0.25, 0.25, 0.25], atol=1e-6)

    def test_masked_minus_inf(self):
        x = np.array([1.0, -np.inf, 2.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert p[1] == 0.0
        assert abs(p.sum() - 1.0) < 1e-6

    def test_all_minus_inf_fallback_uniform(self):
        x = np.full(3, -np.inf, dtype=np.float32)
        p = _softmax_f32(x)
        np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_single_finite_among_inf(self):
        x = np.array([-np.inf, 3.0, -np.inf], dtype=np.float32)
        p = _softmax_f32(x)
        assert p[1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _apply_top_k
# ---------------------------------------------------------------------------

class TestApplyTopK:
    def test_keeps_top_k(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
        out = _apply_top_k(logits, k=2)
        # Top-2 are indices 1 (5.0) and 4 (4.0)
        assert out[1] == pytest.approx(5.0)
        assert out[4] == pytest.approx(4.0)
        assert out[0] == -np.inf
        assert out[2] == -np.inf
        assert out[3] == -np.inf

    def test_k_greater_than_vocab_keeps_all(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _apply_top_k(logits, k=100)
        np.testing.assert_array_equal(out, logits)

    def test_dtype_float32(self):
        out = _apply_top_k(np.array([1.0, 2.0]), k=1)
        assert out.dtype == np.float32

    def test_k_one_keeps_max(self):
        logits = np.array([3.0, 7.0, 5.0], dtype=np.float32)
        out = _apply_top_k(logits, k=1)
        assert out[1] == pytest.approx(7.0)
        assert out[0] == -np.inf
        assert out[2] == -np.inf


# ---------------------------------------------------------------------------
# _apply_top_p
# ---------------------------------------------------------------------------

class TestApplyTopP:
    def test_output_sums_to_one(self):
        probs = _softmax_f32(np.array([1.0, 3.0, 2.0, 0.5]))
        out = _apply_top_p(probs, 0.9)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_top_p_one_no_change(self):
        probs = _softmax_f32(np.arange(5, dtype=np.float32))
        out = _apply_top_p(probs, 1.0)
        np.testing.assert_allclose(out, probs, atol=1e-6)

    def test_dtype_float32(self):
        probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        out = _apply_top_p(probs, 0.9)
        assert out.dtype == np.float32

    def test_top_p_very_small_keeps_at_least_one(self):
        probs = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        out = _apply_top_p(probs, 0.01)
        assert (out > 0).sum() >= 1
        assert abs(out.sum() - 1.0) < 1e-6

    def test_zero_probs_fallback(self):
        # If filtered probs sum to 0, return input unchanged
        probs = np.zeros(4, dtype=np.float32)
        out = _apply_top_p(probs, 0.9)
        np.testing.assert_array_equal(out, probs)


# ---------------------------------------------------------------------------
# _apply_rep_penalty
# ---------------------------------------------------------------------------

class TestApplyRepPenalty:
    def test_positive_logit_divided(self):
        logits = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0], penalty=2.0)
        assert out[0] == pytest.approx(2.0)

    def test_negative_logit_multiplied(self):
        logits = np.array([-4.0, 0.0, 0.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0], penalty=2.0)
        assert out[0] == pytest.approx(-8.0)

    def test_unpenalised_tokens_unchanged(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0], penalty=2.0)
        assert out[1] == pytest.approx(2.0)
        assert out[2] == pytest.approx(3.0)

    def test_input_not_mutated(self):
        logits = np.array([4.0, 2.0], dtype=np.float32)
        original = logits.copy()
        _apply_rep_penalty(logits, [0, 1], penalty=3.0)
        np.testing.assert_array_equal(logits, original)

    def test_dtype_float32(self):
        logits = np.array([1.0, 2.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0], penalty=1.5)
        assert out.dtype == np.float32

    def test_empty_window_no_change(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [], penalty=2.0)
        np.testing.assert_array_equal(out, logits)

    def test_out_of_range_token_ignored(self):
        logits = np.array([1.0, 2.0], dtype=np.float32)
        # Token id 999 is out of range for vocab_size=2
        out = _apply_rep_penalty(logits, [999], penalty=2.0)
        np.testing.assert_array_equal(out, logits)

    def test_duplicate_tokens_in_window_treated_once(self):
        logits = np.array([4.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0, 0, 0], penalty=2.0)
        assert out[0] == pytest.approx(2.0)  # divided once, not three times

    def test_penalty_one_no_change(self):
        logits = np.array([3.0, -1.0], dtype=np.float32)
        out = _apply_rep_penalty(logits, [0, 1], penalty=1.0)
        np.testing.assert_allclose(out, logits, atol=1e-6)
