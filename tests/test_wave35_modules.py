"""
tests/test_wave35_modules.py

Tests for Wave 35 (v14) modules:
  - AdaptiveDraftBudget   (squish/speculative/adaptive_draft_budget.py)
  - KVHeadQuantizer       (squish/kv/kv_quant_head.py)
  - PromptCompressor      (squish/token/prompt_compress.py)
  - RejectionSampleAligner (squish/speculative/rejection_sample_align.py)
  - NumpyMemPool          (squish/kernels/mem_pool.py)
  - EarlyExitSampler      (squish/token/early_exit_sampler.py)
"""

import threading
import time
import contextlib
import pytest
import numpy as np

from squish.speculative.adaptive_draft_budget import (
    AdaptiveDraftBudget,
    DraftBudgetConfig,
)
from squish.kv.kv_quant_head import KVHeadQuantizer, KVHeadQuantConfig
from squish.token.prompt_compress import PromptCompressor, PromptCompressorConfig
from squish.speculative.rejection_sample_align import (
    RejectionSampleAligner,
    RejectionSampleConfig,
)
from squish.kernels.mem_pool import NumpyMemPool, PoolConfig
from squish.token.early_exit_sampler import EarlyExitSampler, EarlyExitConfig


# ===========================================================================
# AdaptiveDraftBudget
# ===========================================================================


class TestDraftBudgetConfig:
    def test_default_construction(self):
        cfg = DraftBudgetConfig()
        assert cfg.min_k == 1
        assert cfg.max_k == 8
        assert cfg.n_arms == 8

    def test_n_arms_property(self):
        cfg = DraftBudgetConfig(min_k=2, max_k=6)
        assert cfg.n_arms == 5

    def test_invalid_min_k(self):
        with pytest.raises(ValueError, match="min_k"):
            DraftBudgetConfig(min_k=0)

    def test_invalid_max_k_lt_min_k(self):
        with pytest.raises(ValueError, match="max_k"):
            DraftBudgetConfig(min_k=4, max_k=2)

    def test_invalid_exploration_constant(self):
        with pytest.raises(ValueError, match="exploration_constant"):
            DraftBudgetConfig(exploration_constant=0.0)

    def test_invalid_reward_ema_alpha(self):
        with pytest.raises(ValueError, match="reward_ema_alpha"):
            DraftBudgetConfig(reward_ema_alpha=0.0)

    def test_warmup_rounds_negative_rejected(self):
        with pytest.raises(ValueError, match="warmup_rounds"):
            DraftBudgetConfig(warmup_rounds=-1)


class TestAdaptiveDraftBudget:
    def test_initial_state(self):
        b = AdaptiveDraftBudget()
        assert b.total_rounds == 0
        assert not b.is_warmed_up

    def test_select_returns_valid_k(self):
        cfg = DraftBudgetConfig(min_k=1, max_k=8, warmup_rounds=1)
        b = AdaptiveDraftBudget(cfg)
        k = b.select()
        assert cfg.min_k <= k <= cfg.max_k

    def test_warmup_covers_all_arms(self):
        cfg = DraftBudgetConfig(min_k=1, max_k=4, warmup_rounds=2, exploration_constant=1.41)
        b = AdaptiveDraftBudget(cfg)
        seen_depths = set()
        for _ in range(cfg.n_arms * cfg.warmup_rounds + 5):
            k = b.select()
            seen_depths.add(k)
            b.update(k, accepted_tokens=k - 1, elapsed_seconds=0.05)
        # All depths in range should be explored during warmup
        for k in range(cfg.min_k, cfg.max_k + 1):
            assert k in seen_depths

    def test_update_increments_total_rounds(self):
        b = AdaptiveDraftBudget()
        b.update(3, accepted_tokens=2, elapsed_seconds=0.01)
        assert b.total_rounds == 1

    def test_update_ignores_zero_elapsed(self):
        b = AdaptiveDraftBudget()
        b.update(3, accepted_tokens=2, elapsed_seconds=0.0)
        assert b.total_rounds == 0

    def test_best_k_before_any_updates(self):
        b = AdaptiveDraftBudget()
        k = b.best_k()
        assert b.config.min_k <= k <= b.config.max_k

    def test_best_k_reflects_highest_reward(self):
        cfg = DraftBudgetConfig(min_k=1, max_k=4, warmup_rounds=0)
        b = AdaptiveDraftBudget(cfg)
        # Arm k=3 should win: highest tok/s
        for _ in range(20):
            b.update(1, accepted_tokens=1, elapsed_seconds=0.1)  # 10 tok/s
            b.update(3, accepted_tokens=3, elapsed_seconds=0.05)  # 60 tok/s
            b.update(4, accepted_tokens=2, elapsed_seconds=0.2)  # 10 tok/s
        assert b.best_k() == 3

    def test_arm_stats_length(self):
        b = AdaptiveDraftBudget()
        stats = b.arm_stats()
        assert len(stats) == b.config.n_arms

    def test_arm_stats_structure(self):
        b = AdaptiveDraftBudget()
        stats = b.arm_stats()
        for entry in stats:
            assert "k" in entry
            assert "plays" in entry
            assert "reward_tps" in entry

    def test_reset_clears_state(self):
        b = AdaptiveDraftBudget()
        b.update(2, accepted_tokens=2, elapsed_seconds=0.05)
        b.reset()
        assert b.total_rounds == 0
        assert not b.is_warmed_up

    def test_ucb1_exploration_selects_unplayed_arms(self):
        # UCB1 must play every arm at least once before exploitation can win.
        # After priming arm k=1 with 50 plays, each select()+update() cycle
        # should cycle through unplayed arms k=2, k=3, k=4.
        cfg = DraftBudgetConfig(min_k=1, max_k=4, warmup_rounds=0, exploration_constant=2.0)
        b = AdaptiveDraftBudget(cfg)
        for _ in range(50):
            b.update(1, accepted_tokens=1, elapsed_seconds=1.0)
        # Now drive 5 select+update steps — unplayed arms should appear
        selected = set()
        for _ in range(5):
            k = b.select()
            selected.add(k)
            b.update(k, accepted_tokens=1, elapsed_seconds=1.0)
        # All three unplayed arms (k=2, k=3, k=4) must each be selected once
        assert 2 in selected
        assert 3 in selected
        assert 4 in selected

    def test_is_warmed_up_after_all_arms_played(self):
        cfg = DraftBudgetConfig(min_k=1, max_k=3, warmup_rounds=2)
        b = AdaptiveDraftBudget(cfg)
        for k in [1, 1, 2, 2, 3, 3]:
            b.update(k, accepted_tokens=1, elapsed_seconds=0.01)
        assert b.is_warmed_up

    def test_warmup_rounds_zero_is_always_warmed_up(self):
        cfg = DraftBudgetConfig(warmup_rounds=0)
        b = AdaptiveDraftBudget(cfg)
        assert b.is_warmed_up


# ===========================================================================
# KVHeadQuantizer
# ===========================================================================


class TestKVHeadQuantConfig:
    def test_default_construction(self):
        cfg = KVHeadQuantConfig()
        assert cfg.n_kv_heads == 8
        assert cfg.high_bits == 16

    def test_invalid_n_kv_heads(self):
        with pytest.raises(ValueError, match="n_kv_heads"):
            KVHeadQuantConfig(n_kv_heads=0)

    def test_invalid_bits_value(self):
        with pytest.raises(ValueError, match="high_bits"):
            KVHeadQuantConfig(high_bits=3)

    def test_entropy_threshold_ordering(self):
        with pytest.raises(ValueError, match="entropy_low_threshold"):
            KVHeadQuantConfig(entropy_high_threshold=1.0, entropy_low_threshold=2.0)

    def test_negative_calibration_steps_rejected(self):
        with pytest.raises(ValueError, match="calibration_steps"):
            KVHeadQuantConfig(calibration_steps=-1)


class TestKVHeadQuantizer:
    def _make_attn(self, n_heads, sq, sk):
        """Return a random normalised attention weight matrix."""
        raw = np.abs(np.random.randn(n_heads, sq, sk).astype(np.float64))
        raw /= raw.sum(axis=-1, keepdims=True) + 1e-9
        return raw

    def test_default_quantizer_uses_high_bits_before_calibration(self):
        q = KVHeadQuantizer(KVHeadQuantConfig(n_kv_heads=4, high_bits=16))
        for h in range(4):
            assert q.assigned_bits(h) == 16

    def test_quantize_dequantize_roundtrip_int8(self):
        cfg = KVHeadQuantConfig(n_kv_heads=2, high_bits=8, mid_bits=8, low_bits=8)
        q = KVHeadQuantizer(cfg)
        kv = np.random.randn(64).astype(np.float32)
        packed, scale, zero = q.quantize_head(kv, 0)
        restored = q.dequantize_head(packed, scale, zero, 0)
        # INT8 absmax: max error ≤ scale * 0.5
        assert np.allclose(kv, restored, atol=scale + 1e-4)

    def test_quantize_preserves_shape(self):
        q = KVHeadQuantizer()
        kv = np.random.randn(16, 64).astype(np.float32)
        packed, _, _ = q.quantize_head(kv, 0)
        assert packed.shape == (16, 64)

    def test_quantize_dequantize_exact_zeros(self):
        q = KVHeadQuantizer()
        kv = np.zeros(32, dtype=np.float32)
        packed, scale, zero = q.quantize_head(kv, 0)
        restored = q.dequantize_head(packed, scale, zero, 0)
        assert np.allclose(restored, 0.0, atol=1e-5)

    def test_calibrate_changes_precision_on_low_entropy(self):
        cfg = KVHeadQuantConfig(
            n_kv_heads=4,
            high_bits=16,
            mid_bits=8,
            low_bits=4,
            entropy_high_threshold=2.0,
            entropy_low_threshold=0.5,
            calibration_steps=1,
        )
        q = KVHeadQuantizer(cfg)
        # Create very sharp (low-entropy) attention — one winner per row
        n_heads, sq, sk = 4, 8, 16
        sharp = np.zeros((n_heads, sq, sk))
        sharp[:, :, 0] = 1.0  # all mass on token 0
        q.calibrate(sharp)
        q.force_calibrate()
        # Low entropy heads should be downgraded
        for h in range(n_heads):
            assert q.assigned_bits(h) in (4, 8, 16)

    def test_calibrate_high_entropy_keeps_high_bits(self):
        cfg = KVHeadQuantConfig(
            n_kv_heads=2,
            high_bits=16,
            mid_bits=8,
            low_bits=4,
            entropy_high_threshold=1.0,
            entropy_low_threshold=0.1,
            calibration_steps=1,
        )
        q = KVHeadQuantizer(cfg)
        n_heads, sq, sk = 2, 4, 4
        # Uniform attention = maximum entropy
        uniform = np.ones((n_heads, sq, sk)) / sk
        q.calibrate(uniform)
        q.force_calibrate()
        for h in range(n_heads):
            assert q.assigned_bits(h) == 16

    def test_invalid_head_idx_raises(self):
        q = KVHeadQuantizer(KVHeadQuantConfig(n_kv_heads=4))
        kv = np.ones(16, dtype=np.float32)
        with pytest.raises(IndexError):
            q.quantize_head(kv, 5)

    def test_compression_summary_structure(self):
        q = KVHeadQuantizer()
        summary = q.compression_summary()
        assert "head_bits" in summary
        assert "mean_bits" in summary
        assert "estimated_compression_ratio" in summary
        assert "calibrated" in summary

    def test_compression_ratio_le_one(self):
        q = KVHeadQuantizer()
        summary = q.compression_summary()
        assert 0.0 < summary["estimated_compression_ratio"] <= 1.0

    def test_reset_calibration(self):
        cfg = KVHeadQuantConfig(n_kv_heads=2, calibration_steps=1)
        q = KVHeadQuantizer(cfg)
        attn = np.ones((2, 4, 4)) / 4.0
        q.calibrate(attn)
        q.force_calibrate()
        q.reset_calibration()
        assert not q.compression_summary()["calibrated"]

    def test_calibrate_wrong_dimensionality_raises(self):
        q = KVHeadQuantizer()
        with pytest.raises(ValueError):
            q.calibrate(np.ones((8, 4)))  # 2-D


# ===========================================================================
# PromptCompressor
# ===========================================================================


class TestPromptCompressorConfig:
    def test_default_construction(self):
        cfg = PromptCompressorConfig()
        assert cfg.compression_ratio == 0.5
        assert len(cfg.score_weights) == 3

    def test_invalid_compression_ratio_zero(self):
        with pytest.raises(ValueError, match="compression_ratio"):
            PromptCompressorConfig(compression_ratio=0.0)

    def test_invalid_compression_ratio_above_one(self):
        with pytest.raises(ValueError, match="compression_ratio"):
            PromptCompressorConfig(compression_ratio=1.5)

    def test_score_weights_not_three_elements(self):
        with pytest.raises(ValueError, match="3 elements"):
            PromptCompressorConfig(score_weights=(0.5, 0.5))

    def test_score_weights_dont_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            PromptCompressorConfig(score_weights=(0.5, 0.3, 0.3))

    def test_preserve_boundary_frac_too_large(self):
        with pytest.raises(ValueError, match="preserve_boundary_frac"):
            PromptCompressorConfig(preserve_boundary_frac=0.6)


class TestPromptCompressor:
    def _token_ids(self, n: int) -> list:
        rng = np.random.default_rng(42)
        return rng.integers(0, 1000, n).tolist()

    def test_compress_returns_shorter_sequence(self):
        c = PromptCompressor(PromptCompressorConfig(compression_ratio=0.5, min_tokens=1))
        ids = self._token_ids(100)
        out = c.compress(ids)
        assert len(out) < len(ids)

    def test_compress_preserves_order(self):
        c = PromptCompressor()
        ids = list(range(200))
        out = c.compress(ids)
        for i in range(len(out) - 1):
            assert ids.index(out[i]) < ids.index(out[i + 1])

    def test_compress_ratio_one_is_identity(self):
        c = PromptCompressor(PromptCompressorConfig(compression_ratio=1.0))
        ids = self._token_ids(50)
        out = c.compress(ids)
        assert len(out) == len(ids)

    def test_min_tokens_enforced(self):
        c = PromptCompressor(PromptCompressorConfig(compression_ratio=0.1, min_tokens=20))
        ids = self._token_ids(80)
        out = c.compress(ids)
        assert len(out) >= 20

    def test_empty_input_returns_empty(self):
        c = PromptCompressor()
        assert c.compress([]) == []

    def test_actual_ratio_calculation(self):
        c = PromptCompressor()
        original = list(range(100))
        compressed = list(range(50))
        assert abs(c.actual_ratio(original, compressed) - 0.5) < 1e-9

    def test_actual_ratio_empty_original(self):
        c = PromptCompressor()
        assert c.actual_ratio([], []) == 1.0

    def test_score_length_matches_input(self):
        c = PromptCompressor()
        ids = self._token_ids(64)
        scores = c.score(ids)
        assert len(scores) == 64

    def test_score_returns_float32(self):
        c = PromptCompressor()
        scores = c.score(self._token_ids(32))
        assert scores.dtype == np.float32

    def test_target_ratio_override(self):
        c = PromptCompressor(PromptCompressorConfig(compression_ratio=0.5, min_tokens=1))
        ids = self._token_ids(100)
        out_less = c.compress(ids, target_ratio=0.3)
        out_more = c.compress(ids, target_ratio=0.7)
        assert len(out_less) < len(out_more)

    def test_boundary_tokens_always_kept(self):
        # keep_n=50, protected=10+10=20 positions → all 20 fit inside the 50 kept.
        cfg = PromptCompressorConfig(
            compression_ratio=0.5,
            min_tokens=1,
            preserve_boundary_frac=0.1,
        )
        c = PromptCompressor(cfg)
        ids = list(range(100))
        boundary_n = int(100 * 0.1)  # 10 start + 10 end protected
        out_set = set(c.compress(ids))
        # First boundary_n token IDs must always appear in output
        for t in range(boundary_n):
            assert t in out_set, f"token {t} should be in boundary-protected set"

    def test_single_token_sequence(self):
        c = PromptCompressor()
        out = c.compress([42])
        assert out == [42]


# ===========================================================================
# RejectionSampleAligner
# ===========================================================================


class TestRejectionSampleConfig:
    def test_default_construction(self):
        cfg = RejectionSampleConfig()
        assert cfg.temperature == 1.0

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RejectionSampleConfig(temperature=0.0)

    def test_max_vocab_size_positive(self):
        with pytest.raises(ValueError, match="max_vocab_size"):
            RejectionSampleConfig(max_vocab_size=0)


class TestRejectionSampleAligner:
    def _uniform_probs(self, vocab: int) -> np.ndarray:
        return np.ones(vocab) / vocab

    def _peaked_probs(self, vocab: int, peak: int, peak_mass: float = 0.9) -> np.ndarray:
        p = np.ones(vocab) * ((1 - peak_mass) / (vocab - 1))
        p[peak] = peak_mass
        return p

    def test_accept_token_high_ratio_always_accepts(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=0))
        vocab = 10
        # target >> draft for draft_token → should almost always accept
        p_d = self._peaked_probs(vocab, 3, 0.1)
        p_t = self._peaked_probs(vocab, 3, 0.9)
        acceptances = sum(
            aligner.accept_token(3, p_d.copy(), p_t.copy())[0]
            for _ in range(50)
        )
        assert acceptances >= 45  # ~100 % acceptance

    def test_accept_token_zero_ratio_never_accepts(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=1))
        vocab = 10
        # draft has all mass on token 3, target has none
        p_d = self._peaked_probs(vocab, 3, 0.99)
        p_t_arr = np.ones(vocab) / vocab
        p_t_arr[3] = 0.0
        p_t_arr /= p_t_arr.sum()
        accepts = sum(
            aligner.accept_token(3, p_d.copy(), p_t_arr.copy())[0]
            for _ in range(50)
        )
        assert accepts == 0

    def test_accept_token_returns_correction_on_reject(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=2))
        vocab = 8
        p_d = self._peaked_probs(vocab, 0, 0.99)
        p_t = self._peaked_probs(vocab, 7, 0.99)
        # Token 0 essentially never accepted by target
        for _ in range(30):
            accepted, correction = aligner.accept_token(0, p_d.copy(), p_t.copy())
            if not accepted:
                assert correction is not None
                assert 0 <= correction < vocab
                break

    def test_acceptance_rate_tracked(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=3))
        vocab = 4
        p_d = self._uniform_probs(vocab)
        p_t = self._uniform_probs(vocab)
        for t in range(vocab):
            aligner.accept_token(t, p_d.copy(), p_t.copy())
        rate = aligner.acceptance_rate
        assert 0.0 <= rate <= 1.0

    def test_reset_stats_clears_counts(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=4))
        vocab = 4
        p_d = self._uniform_probs(vocab)
        p_t = self._uniform_probs(vocab)
        aligner.accept_token(0, p_d, p_t)
        aligner.reset_stats()
        assert aligner.n_accepted == 0
        assert aligner.n_rejected == 0
        assert aligner.acceptance_rate == 0.0

    def test_verify_sequence_returns_lists(self):
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=5))
        vocab = 16
        k = 4
        draft_tokens = list(range(k))
        draft_logits = np.random.randn(k, vocab).astype(np.float32)
        target_logits = np.random.randn(k, vocab).astype(np.float32)
        accepted, correction = aligner.verify_sequence(
            draft_tokens, draft_logits, target_logits
        )
        assert isinstance(accepted, list)
        assert len(accepted) <= k
        # correction is an int or None
        assert correction is None or isinstance(correction, int)

    def test_verify_sequence_all_accepted_gives_bonus_token(self):
        # When target >> draft, all should be accepted → bonus token not None
        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=6))
        vocab = 8
        k = 3
        draft_tokens = [0, 0, 0]
        # Craft matching logits so target ≈ draft
        logits = np.zeros((k, vocab))
        logits[:, 0] = 10.0  # heavy mass on token 0
        accepted, correction = aligner.verify_sequence(
            draft_tokens, logits.copy(), logits.copy()
        )
        # All tokens accepted; bonus returned
        assert len(accepted) == k
        assert correction is not None

    def test_verify_sequence_shape_mismatch_raises(self):
        aligner = RejectionSampleAligner()
        with pytest.raises(ValueError, match="must match"):
            aligner.verify_sequence(
                [1, 2],
                np.zeros((3, 10)),  # k mismatch
                np.zeros((2, 10)),
            )

    def test_distribution_correctness_statistically(self):
        """Sampled tokens should approximate target distribution."""
        rng_fix = np.random.default_rng(99)
        vocab = 5
        p_target = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        p_draft = np.ones(vocab) / vocab  # uniform draft

        aligner = RejectionSampleAligner(RejectionSampleConfig(seed=99))
        counts = np.zeros(vocab)
        n_trials = 2000
        for _ in range(n_trials):
            draft_t = int(rng_fix.choice(vocab, p=p_draft))
            accepted, correction = aligner.accept_token(
                draft_t, p_draft.copy(), p_target.copy()
            )
            token = draft_t if accepted else correction
            counts[token] += 1

        empirical = counts / counts.sum()
        # Loose tolerance: chi-squared style per-bin
        for tok in range(vocab):
            assert abs(empirical[tok] - p_target[tok]) < 0.07, (
                f"token {tok}: empirical={empirical[tok]:.3f} expected={p_target[tok]:.3f}"
            )


# ===========================================================================
# NumpyMemPool
# ===========================================================================


class TestPoolConfig:
    def test_default_construction(self):
        cfg = PoolConfig()
        assert cfg.pool_size == 32
        assert cfg.max_elements > 0

    def test_invalid_pool_size(self):
        with pytest.raises(ValueError, match="pool_size"):
            PoolConfig(pool_size=0)

    def test_empty_max_shape_rejected(self):
        with pytest.raises(ValueError, match="max_shape"):
            PoolConfig(max_shape=())

    def test_zero_dim_in_max_shape_rejected(self):
        with pytest.raises(ValueError, match="max_shape"):
            PoolConfig(max_shape=(0, 256))

    def test_invalid_overflow_policy(self):
        with pytest.raises(ValueError, match="overflow_policy"):
            PoolConfig(overflow_policy="ignore")

    def test_max_elements_computed_correctly(self):
        cfg = PoolConfig(max_shape=(4, 8))
        assert cfg.max_elements == 32


class TestNumpyMemPool:
    def test_acquire_returns_correct_shape(self):
        pool = NumpyMemPool(PoolConfig(pool_size=4, max_shape=(1024,)))
        bid, arr = pool.acquire((64,))
        assert arr.shape == (64,)
        pool.release(bid)

    def test_acquire_2d_shape(self):
        pool = NumpyMemPool(PoolConfig(pool_size=4, max_shape=(128, 64)))
        bid, arr = pool.acquire((32, 16))
        assert arr.shape == (32, 16)
        pool.release(bid)

    def test_hit_counter_increments(self):
        pool = NumpyMemPool(PoolConfig(pool_size=4, max_shape=(256,)))
        bid, _ = pool.acquire((64,))
        assert pool.hits == 1
        pool.release(bid)

    def test_release_returns_to_pool(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(64,)))
        before_free = pool.free_count
        bid, _ = pool.acquire((32,))
        assert pool.free_count == before_free - 1
        pool.release(bid)
        assert pool.free_count == before_free

    def test_overflow_allocate_policy(self):
        pool = NumpyMemPool(PoolConfig(pool_size=1, max_shape=(64,), overflow_policy="allocate"))
        bid1, _ = pool.acquire((32,))
        # Pool exhausted — should return overflow buffer without error
        bid2, arr2 = pool.acquire((16,))
        assert bid2 == -1
        assert arr2.shape == (16,)
        pool.release(bid1)
        pool.release(bid2)  # no-op for overflow

    def test_overflow_raise_policy(self):
        pool = NumpyMemPool(PoolConfig(pool_size=1, max_shape=(64,), overflow_policy="raise"))
        bid, _ = pool.acquire((32,))
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire((16,))
        pool.release(bid)

    def test_double_release_raises(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(64,)))
        bid, _ = pool.acquire((32,))
        pool.release(bid)
        with pytest.raises(RuntimeError, match="released twice"):
            pool.release(bid)

    def test_shape_exceeds_max_elements_raises(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(64,)))
        with pytest.raises(ValueError, match="exceeds"):
            pool.acquire((128,))

    def test_context_manager_releases_on_exit(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(256,)))
        with pool.borrow((64,)) as buf:
            buf[:] = 1.0
            active_inside = pool.active_count
        assert pool.active_count == active_inside - 1

    def test_context_manager_releases_on_exception(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(256,)))
        try:
            with pool.borrow((64,)):
                raise RuntimeError("test exception")
        except RuntimeError:
            pass
        assert pool.active_count == 0

    def test_thread_safety(self):
        pool = NumpyMemPool(PoolConfig(pool_size=16, max_shape=(512,), overflow_policy="allocate"))
        errors = []

        def worker():
            try:
                bid, arr = pool.acquire((128,))
                arr[:] = 0.0
                time.sleep(0.001)
                pool.release(bid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"

    def test_reset_stats(self):
        pool = NumpyMemPool(PoolConfig(pool_size=2, max_shape=(64,)))
        bid, _ = pool.acquire((32,))
        pool.release(bid)
        assert pool.hits >= 1
        pool.reset_stats()
        assert pool.hits == 0
        assert pool.misses == 0

    def test_capacity_property(self):
        pool = NumpyMemPool(PoolConfig(pool_size=7, max_shape=(64,)))
        assert pool.capacity == 7


# ===========================================================================
# EarlyExitSampler
# ===========================================================================


class TestEarlyExitConfig:
    def test_default_construction(self):
        cfg = EarlyExitConfig()
        assert cfg.confidence_threshold == 0.9
        assert cfg.temperature == 1.0

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            EarlyExitConfig(confidence_threshold=-0.1)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            EarlyExitConfig(confidence_threshold=1.01)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            EarlyExitConfig(temperature=0.0)

    def test_invalid_top_k_negative(self):
        with pytest.raises(ValueError, match="top_k"):
            EarlyExitConfig(top_k=-1)

    def test_invalid_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            EarlyExitConfig(top_p=0.0)


class TestEarlyExitSampler:
    def _peaked_logits(self, vocab: int, peak: int, strength: float = 10.0) -> np.ndarray:
        logits = np.zeros(vocab, dtype=np.float32)
        logits[peak] = strength
        return logits

    def _uniform_logits(self, vocab: int) -> np.ndarray:
        return np.zeros(vocab, dtype=np.float32)

    def test_greedy_threshold_always_fast_path(self):
        cfg = EarlyExitConfig(confidence_threshold=0.0)  # always argmax
        # With threshold=0, fast path never triggers (requires softmax > 0.0 trivially)
        # Actually threshold=0 means: if max_prob >= 0 → always fast. Let's test 1.0 edge:
        cfg = EarlyExitConfig(confidence_threshold=1.0)
        s = EarlyExitSampler(cfg)
        logits = self._peaked_logits(100, 42)
        token = s.sample(logits)
        assert token == 42

    def test_peaked_logits_use_fast_path(self):
        cfg = EarlyExitConfig(confidence_threshold=0.9, seed=0)
        s = EarlyExitSampler(cfg)
        vocab = 32
        for _ in range(20):
            logits = self._peaked_logits(vocab, 5, strength=20.0)
            s.sample(logits)
        assert s.fast_path_rate > 0.9

    def test_uniform_logits_use_slow_path(self):
        cfg = EarlyExitConfig(confidence_threshold=0.9, seed=1)
        s = EarlyExitSampler(cfg)
        vocab = 32
        for _ in range(20):
            logits = self._uniform_logits(vocab)
            s.sample(logits)
        assert s.fast_path_rate < 0.1  # uniform: max_prob ~ 1/32 << 0.9

    def test_sample_returns_valid_index(self):
        s = EarlyExitSampler(EarlyExitConfig(seed=2))
        vocab = 128
        logits = np.random.randn(vocab).astype(np.float32)
        token = s.sample(logits)
        assert 0 <= token < vocab

    def test_sample_batch_returns_correct_length(self):
        s = EarlyExitSampler(EarlyExitConfig(seed=3))
        batch, vocab = 8, 64
        logits = np.random.randn(batch, vocab).astype(np.float32)
        tokens = s.sample_batch(logits)
        assert len(tokens) == batch

    def test_sample_batch_all_valid_indices(self):
        s = EarlyExitSampler(EarlyExitConfig(seed=4))
        batch, vocab = 4, 32
        logits = np.random.randn(batch, vocab).astype(np.float32)
        tokens = s.sample_batch(logits)
        for t in tokens:
            assert 0 <= t < vocab

    def test_sample_wrong_dimensionality_raises(self):
        s = EarlyExitSampler()
        with pytest.raises(ValueError, match="1-D"):
            s.sample(np.zeros((4, 8)))

    def test_sample_batch_wrong_dimensionality_raises(self):
        s = EarlyExitSampler()
        with pytest.raises(ValueError, match="2-D"):
            s.sample_batch(np.zeros(32))

    def test_stats_tracking(self):
        s = EarlyExitSampler(EarlyExitConfig(seed=5))
        vocab = 16
        for _ in range(10):
            s.sample(np.random.randn(vocab).astype(np.float32))
        assert s.total_sampled == 10
        assert s.n_fast + s.n_slow == 10

    def test_reset_stats(self):
        s = EarlyExitSampler(EarlyExitConfig(seed=6))
        for _ in range(5):
            s.sample(np.random.randn(64).astype(np.float32))
        s.reset_stats()
        assert s.total_sampled == 0
        assert s.fast_path_rate == 0.0

    def test_top_k_zero_disables_filtering(self):
        cfg = EarlyExitConfig(confidence_threshold=0.0, top_k=0, seed=7)
        s = EarlyExitSampler(cfg)
        vocab = 32
        # Should not raise
        for _ in range(5):
            s.sample(np.random.randn(vocab).astype(np.float32))

    def test_temperature_scaling_affects_distribution(self):
        """High temperature → flatter distribution → sampled tokens more varied."""
        vocab = 10
        logits = np.array([5.0, 3.0, 1.0] + [0.0] * 7, dtype=np.float32)
        cfg_hot = EarlyExitConfig(confidence_threshold=0.0, temperature=5.0, seed=8)
        cfg_cold = EarlyExitConfig(confidence_threshold=0.0, temperature=0.1, seed=9)
        hot_sampler = EarlyExitSampler(cfg_hot)
        cold_sampler = EarlyExitSampler(cfg_cold)

        hot_tokens = set(hot_sampler.sample(logits.copy()) for _ in range(200))
        cold_tokens = set(cold_sampler.sample(logits.copy()) for _ in range(200))

        # Cold sampler with logits[0]=5.0 and temp=0.1 should nearly always pick token 0
        assert len(cold_tokens) <= 3
        # Hot sampler should explore more
        assert len(hot_tokens) >= 3
