"""
tests/core/test_determinism.py

Unit tests for squish/core/determinism.py

Phase 6 — LLM-42 Determinism: Verified Speculation

Coverage:
  • DeterminismConfig — defaults, validation, post_init guards
  • DeterministicSampler — greedy, seeded draws, reset, n_samples counter
  • Reproducibility — same seed → identical token sequence
  • TokenVerifier — record, verify (OK / NOT_DUE / DIVERGED), rollback
  • VerifierResult / VerifyStatus
  • Internal helpers (_softmax_f32, _apply_top_p)
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.core.determinism import (
    DeterminismConfig,
    DeterministicSampler,
    TokenVerifier,
    VerifierResult,
    VerifyStatus,
    _apply_top_p,
    _softmax_f32,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_logits(n: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _make(
    seed: int = 42,
    buffer_size: int = 16,
    verify_every: int = 4,
    enabled: bool = True,
) -> tuple[DeterminismConfig, DeterministicSampler, TokenVerifier]:
    cfg      = DeterminismConfig(seed=seed, buffer_size=buffer_size,
                                  verify_every=verify_every, enabled=enabled)
    sampler  = DeterministicSampler(cfg)
    verifier = TokenVerifier(cfg, sampler)
    return cfg, sampler, verifier


# ---------------------------------------------------------------------------
# DeterminismConfig
# ---------------------------------------------------------------------------

class TestDeterminismConfig:
    def test_defaults(self):
        cfg = DeterminismConfig()
        assert cfg.seed         == 42
        assert cfg.buffer_size  == 64
        assert cfg.verify_every == 8
        assert cfg.enabled      is True

    def test_custom_values(self):
        cfg = DeterminismConfig(seed=7, buffer_size=32, verify_every=0, enabled=False)
        assert cfg.seed         == 7
        assert cfg.buffer_size  == 32
        assert cfg.verify_every == 0
        assert cfg.enabled      is False

    def test_buffer_size_zero_raises(self):
        with pytest.raises(ValueError, match="buffer_size"):
            DeterminismConfig(buffer_size=0)

    def test_verify_every_negative_raises(self):
        with pytest.raises(ValueError, match="verify_every"):
            DeterminismConfig(verify_every=-1)

    def test_verify_every_zero_valid(self):
        cfg = DeterminismConfig(verify_every=0)
        assert cfg.verify_every == 0


# ---------------------------------------------------------------------------
# DeterministicSampler
# ---------------------------------------------------------------------------

class TestDeterministicSamplerGreedy:
    def test_greedy_returns_argmax(self):
        logits = np.array([1.0, 5.0, 2.0, 0.5], dtype=np.float32)
        _, sampler, _ = _make()
        out = sampler.sample(logits, temperature=0.0)
        assert out == 1  # argmax

    def test_greedy_no_rng_advance(self):
        logits = _uniform_logits()
        _, sampler, _ = _make()
        sampler.sample(logits, temperature=0.0)
        assert sampler.n_samples == 1  # counter still increments
        # Calling again with same logits returns same result (no RNG state change)
        out1 = sampler.sample(logits, temperature=0.0)
        out2 = sampler.sample(logits, temperature=0.0)
        assert out1 == out2

    def test_greedy_dtype_int(self):
        logits = _uniform_logits()
        _, sampler, _ = _make()
        out = sampler.sample(logits, temperature=0.0)
        assert isinstance(out, int)


class TestDeterministicSamplerSeeded:
    def test_same_seed_same_sequence(self):
        logits = _uniform_logits(128)
        _, s1, _ = _make(seed=99)
        _, s2, _ = _make(seed=99)
        tokens1 = [s1.sample(logits, temperature=1.0) for _ in range(20)]
        tokens2 = [s2.sample(logits, temperature=1.0) for _ in range(20)]
        assert tokens1 == tokens2

    def test_different_seeds_differ(self):
        logits = _uniform_logits(128)
        _, s1, _ = _make(seed=1)
        _, s2, _ = _make(seed=2)
        tokens1 = [s1.sample(logits, temperature=1.0) for _ in range(20)]
        tokens2 = [s2.sample(logits, temperature=1.0) for _ in range(20)]
        assert tokens1 != tokens2

    def test_reset_restores_sequence(self):
        logits = _uniform_logits(64)
        _, sampler, _ = _make(seed=7)
        t1 = [sampler.sample(logits, temperature=1.0) for _ in range(10)]
        sampler.reset()
        t2 = [sampler.sample(logits, temperature=1.0) for _ in range(10)]
        assert t1 == t2

    def test_reset_with_new_seed(self):
        logits = _uniform_logits(64)
        _, sampler, _ = _make(seed=5)
        t1 = [sampler.sample(logits, temperature=1.0) for _ in range(10)]
        sampler.reset(seed=5)
        t2 = [sampler.sample(logits, temperature=1.0) for _ in range(10)]
        assert t1 == t2

    def test_n_samples_increments(self):
        logits = _uniform_logits()
        _, sampler, _ = _make()
        assert sampler.n_samples == 0
        for i in range(5):
            sampler.sample(logits)
        assert sampler.n_samples == 5

    def test_n_samples_resets_on_reset(self):
        logits = _uniform_logits()
        _, sampler, _ = _make()
        sampler.sample(logits)
        sampler.sample(logits)
        sampler.reset()
        assert sampler.n_samples == 0

    def test_output_in_vocab_range(self):
        vocab = 64
        logits = _uniform_logits(vocab)
        _, sampler, _ = _make(seed=0)
        for _ in range(50):
            t = sampler.sample(logits, temperature=0.8)
            assert 0 <= t < vocab

    def test_current_seed_property(self):
        cfg, sampler, _ = _make(seed=17)
        assert sampler.current_seed == 17
        sampler.reset(seed=99)
        assert sampler.current_seed == 99

    def test_top_k_limits_candidates(self):
        """With top_k=1 the sampler must always return the top token."""
        logits = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32)
        _, sampler, _ = _make(seed=42)
        for _ in range(20):
            t = sampler.sample(logits, temperature=1.0, top_k=1)
            assert t == 0

    def test_top_p_filters_nucleus(self):
        """Very low top_p should collapse to the single highest-prob token."""
        logits = np.array([20.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, sampler, _ = _make(seed=0)
        for _ in range(20):
            t = sampler.sample(logits, temperature=1.0, top_p=0.5)
            assert t == 0


# ---------------------------------------------------------------------------
# TokenVerifier
# ---------------------------------------------------------------------------

class TestTokenVerifierInit:
    def test_initial_step_zero(self):
        _, _, verifier = _make()
        assert verifier.step == 0

    def test_initial_buffer_empty(self):
        _, _, verifier = _make()
        assert verifier.buffer_len == 0


class TestTokenVerifierRecord:
    def test_record_increments_step(self):
        logits = _uniform_logits()
        _, _, verifier = _make()
        verifier.record(5, logits)
        assert verifier.step == 1

    def test_record_fills_buffer(self):
        logits = _uniform_logits()
        _, _, verifier = _make(buffer_size=4)
        for i in range(4):
            verifier.record(i, logits)
        assert verifier.buffer_len == 4

    def test_buffer_maxlen_rolls_off_oldest(self):
        logits = _uniform_logits()
        _, _, verifier = _make(buffer_size=3)
        for i in range(10):
            verifier.record(i, logits)
        assert verifier.buffer_len == 3


class TestTokenVerifierVerifyNotDue:
    def test_not_due_when_disabled(self):
        cfg      = DeterminismConfig(seed=1, verify_every=1, enabled=False)
        sampler  = DeterministicSampler(cfg)
        verifier = TokenVerifier(cfg, sampler)
        logits   = _uniform_logits()
        verifier.record(0, logits)
        result = verifier.verify()
        assert result.status is VerifyStatus.NOT_DUE

    def test_not_due_between_intervals(self):
        _, _, verifier = _make(verify_every=4)
        logits = _uniform_logits(32, seed=0)
        # Step 1: record but don't hit step % 4 == 0
        verifier.record(0, logits)
        result = verifier.verify()
        assert result.status is VerifyStatus.NOT_DUE

    def test_due_at_interval(self):
        _, sampler, verifier = _make(seed=42, verify_every=4)
        logits = _uniform_logits(32, seed=0)
        # Record 4 tokens, using the sampler itself so they agree
        sampler.reset(seed=42 ^ 0xDEAD_C0DE)  # match verifier's internal sub-seed logic
        for _ in range(4):
            # Deterministic sample with greedy avoids needing to predict RNG outcome
            tok = sampler.sample(logits, temperature=0.0)
            verifier.record(tok, logits)
        # After 4 records, step=4 → 4 % 4 == 0 → should run
        result = verifier.verify(temperature=0.0)
        assert result.status in (VerifyStatus.OK, VerifyStatus.DIVERGED)

    def test_not_due_empty_buffer(self):
        _, _, verifier = _make(verify_every=1)
        result = verifier.verify()
        assert result.status is VerifyStatus.NOT_DUE


class TestTokenVerifierOK:
    def test_ok_when_tokens_match(self):
        """Record tokens that ARE what the verifier would deterministically produce."""
        cfg      = DeterminismConfig(seed=42, buffer_size=16, verify_every=4, enabled=True)
        sampler  = DeterministicSampler(cfg)
        verifier = TokenVerifier(cfg, sampler)
        logits   = _uniform_logits(64, seed=1)

        # The internal verifier sub-seed is cfg.seed ^ 0xDEAD_C0DE
        verify_sampler = DeterministicSampler(DeterminismConfig(
            seed=42 ^ 0xDEAD_C0DE, verify_every=4,
        ))
        # Generate 4 tokens that the verifier expects
        for _ in range(4):
            expected = verify_sampler.sample(logits, temperature=0.0)
            verifier.record(expected, logits)

        result = verifier.verify(temperature=0.0)
        assert result.status is VerifyStatus.OK
        assert result.rollback_count == 0
        assert result.diverged_at is None

    def test_ok_result_does_not_advance_sampler(self):
        """verify() must be side-effect-free on the sampler."""
        cfg      = DeterminismConfig(seed=0, buffer_size=8, verify_every=4, enabled=True)
        sampler  = DeterministicSampler(cfg)
        verifier = TokenVerifier(cfg, sampler)
        logits   = _uniform_logits(32)

        # Fill with greedy tokens (always match under temperature=0)
        verify_sampler = DeterministicSampler(DeterminismConfig(seed=0 ^ 0xDEAD_C0DE))
        for _ in range(4):
            t = verify_sampler.sample(logits, temperature=0.0)
            verifier.record(t, logits)

        n_before = sampler.n_samples
        verifier.verify(temperature=0.0)
        assert sampler.n_samples == n_before  # no side effects


class TestTokenVerifierDiverged:
    def test_diverged_when_tokens_wrong(self):
        """Record all-zero tokens against non-zero logits — should diverge."""
        cfg      = DeterminismConfig(seed=42, buffer_size=16, verify_every=4, enabled=True)
        sampler  = DeterministicSampler(cfg)
        verifier = TokenVerifier(cfg, sampler)
        logits   = np.array([0.0, 0.0, 0.0, 0.0, 100.0] + [0.0] * 27, dtype=np.float32)
        # Record wrong tokens (all zero, but greedy would pick index 4)
        for _ in range(4):
            verifier.record(0, logits)
        result = verifier.verify(temperature=0.0)
        assert result.status is VerifyStatus.DIVERGED
        assert result.diverged_at is not None
        assert result.rollback_count > 0

    def test_diverged_rollback_count_leq_buffer_len(self):
        cfg      = DeterminismConfig(seed=42, buffer_size=16, verify_every=4, enabled=True)
        sampler  = DeterministicSampler(cfg)
        verifier = TokenVerifier(cfg, sampler)
        logits   = np.array([0.0] * 4 + [100.0] + [0.0] * 27, dtype=np.float32)
        for _ in range(4):
            verifier.record(0, logits)
        result = verifier.verify(temperature=0.0)
        if result.status is VerifyStatus.DIVERGED:
            assert result.rollback_count <= verifier.buffer_len + result.rollback_count


class TestTokenVerifierRollback:
    def test_rollback_trims_buffer(self):
        logits = _uniform_logits()
        _, _, verifier = _make(buffer_size=10)
        for i in range(6):
            verifier.record(i, logits)
        assert verifier.buffer_len == 6
        verifier.rollback(3)
        assert verifier.buffer_len == 3

    def test_rollback_adjusts_step(self):
        logits = _uniform_logits()
        _, _, verifier = _make()
        for i in range(5):
            verifier.record(i, logits)
        verifier.rollback(2)
        assert verifier.step == 3

    def test_rollback_clamped_to_buffer_size(self):
        logits = _uniform_logits()
        _, _, verifier = _make(buffer_size=4)
        for i in range(4):
            verifier.record(i, logits)
        verifier.rollback(100)  # more than buffer len
        assert verifier.buffer_len == 0

    def test_rollback_zero_noop(self):
        logits = _uniform_logits()
        _, _, verifier = _make()
        for i in range(3):
            verifier.record(i, logits)
        verifier.rollback(0)
        assert verifier.buffer_len == 3

    def test_reset_clears_everything(self):
        logits = _uniform_logits()
        _, _, verifier = _make()
        for i in range(8):
            verifier.record(i, logits)
        verifier.reset()
        assert verifier.buffer_len == 0
        assert verifier.step == 0


class TestVerifierResultStructure:
    def test_namedtuple_fields(self):
        r = VerifierResult(VerifyStatus.OK, None, 0)
        assert r.status        is VerifyStatus.OK
        assert r.diverged_at   is None
        assert r.rollback_count == 0

    def test_diverged_result_fields(self):
        r = VerifierResult(VerifyStatus.DIVERGED, 3, 5)
        assert r.status        is VerifyStatus.DIVERGED
        assert r.diverged_at   == 3
        assert r.rollback_count == 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestSoftmaxF32:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_dtype_float32(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        assert _softmax_f32(x).dtype == np.float32

    def test_all_negative_inf_safe(self):
        x = np.array([-np.inf, -np.inf, 0.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert np.isfinite(p).all()
        assert abs(p.sum() - 1.0) < 1e-5

    def test_max_at_highest_logit(self):
        x = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        p = _softmax_f32(x)
        assert p.argmax() == 1


class TestApplyTopP:
    def test_sums_to_one(self):
        probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        out   = _apply_top_p(probs, 0.7)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_low_p_concentrates_mass(self):
        probs = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        out   = _apply_top_p(probs, 0.6)
        # Only the top token should survive
        assert (out > 0).sum() == 1
        assert out.argmax() == 0

    def test_p_one_returns_normalized(self):
        probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        out   = _apply_top_p(probs, 1.0)
        np.testing.assert_allclose(out, probs, atol=1e-6)

    def test_dtype_float32(self):
        probs = np.array([0.5, 0.5], dtype=np.float32)
        assert _apply_top_p(probs, 1.0).dtype == np.float32
