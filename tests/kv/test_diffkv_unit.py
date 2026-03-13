"""Unit tests for squish.kv.diffkv (DiffKV 3-level KV differentiation)."""

import numpy as np
import pytest

from squish.kv.diffkv import (
    IMPORTANCE_CRITICAL,
    IMPORTANCE_EVICT,
    IMPORTANCE_MARGINAL,
    PRECISION_TIERS,
    CompactedKVSlot,
    DiffKVConfig,
    DiffKVPolicy,
    DiffKVPolicyManager,
    DiffKVStats,
    HeadSparsityProfile,
    TokenImportanceTier,
    classify_tokens,
    compact_kv,
)


def _cfg(n_layers=4, n_heads=4, **kw):
    return DiffKVConfig(n_layers=n_layers, n_heads=n_heads, **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# TestDiffKVConfig
# ---------------------------------------------------------------------------


class TestDiffKVConfig:
    def test_defaults(self):
        cfg = DiffKVConfig()
        assert cfg.n_layers == 32
        assert cfg.n_heads == 32
        assert cfg.critical_k_bits == 8
        assert cfg.marginal_v_bits == 2

    def test_evict_fraction(self):
        cfg = _cfg(critical_fraction=0.20, marginal_fraction=0.40)
        assert abs(cfg.evict_fraction - 0.40) < 1e-6

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            DiffKVConfig(critical_k_bits=16)

    def test_invalid_fractions_sum(self):
        with pytest.raises(ValueError):
            DiffKVConfig(critical_fraction=0.6, marginal_fraction=0.6)

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            DiffKVConfig(n_layers=0)

    def test_invalid_sparsity_threshold(self):
        with pytest.raises(ValueError):
            DiffKVConfig(sparsity_threshold=0.0)


# ---------------------------------------------------------------------------
# TestHeadSparsityProfile
# ---------------------------------------------------------------------------


class TestHeadSparsityProfile:
    def test_initial_sparsity_zero(self):
        p = HeadSparsityProfile(layer_idx=0, head_idx=0)
        assert p.observed_sparsity == 0.0

    def test_update_increases_count(self):
        p = HeadSparsityProfile(layer_idx=0, head_idx=0)
        attn = np.zeros((8, 8), dtype=np.float32)
        p.update(attn)
        assert p.n_samples == 1

    def test_high_sparsity_detected(self):
        p = HeadSparsityProfile(layer_idx=0, head_idx=0)
        # All zeros → 100% near-zero
        attn = np.zeros((8, 8), dtype=np.float32)
        p.update(attn)
        assert p.observed_sparsity == 1.0

    def test_ema_averaging(self):
        p = HeadSparsityProfile(layer_idx=0, head_idx=0)
        attn_full = np.ones((4, 4)) * 0.5  # none near-zero (>= 0.01)
        attn_sparse = np.zeros((4, 4))  # all near-zero
        p.update(attn_full)
        p.update(attn_sparse)
        # should be between 0 and 1
        assert 0.0 <= p.observed_sparsity <= 1.0


# ---------------------------------------------------------------------------
# TestClassifyTokens
# ---------------------------------------------------------------------------


class TestClassifyTokens:
    def test_three_tiers_returned(self):
        cfg = _cfg()
        scores = np.ones(20, dtype=np.float32)
        tiers = classify_tokens(scores, cfg)
        assert len(tiers) == 3

    def test_tiers_cover_all_tokens(self):
        cfg = _cfg(critical_fraction=0.20, marginal_fraction=0.40)
        scores = np.arange(20, dtype=np.float32)
        tiers = classify_tokens(scores, cfg)
        total = sum(t.n_tokens for t in tiers)
        assert total == 20

    def test_critical_tokens_highest_score(self):
        cfg = _cfg()
        scores = np.arange(10, dtype=np.float32)  # 0..9; highest = 9,8,...
        tiers = classify_tokens(scores, cfg)
        critical_tier = tiers[0]
        assert critical_tier.tier == IMPORTANCE_CRITICAL
        # Indices of critical tokens should contain high-score indices
        n_crit = critical_tier.n_tokens
        assert n_crit >= 1

    def test_multi_head_scores_averaged(self):
        cfg = _cfg()
        scores = np.ones((4, 16), dtype=np.float32)
        tiers = classify_tokens(scores, cfg)
        assert len(tiers) == 3

    def test_evict_tier_exists(self):
        cfg = _cfg(critical_fraction=0.20, marginal_fraction=0.40)
        scores = np.arange(20, dtype=np.float32)
        tiers = classify_tokens(scores, cfg)
        evict_tier = tiers[2]
        assert evict_tier.tier == IMPORTANCE_EVICT
        assert evict_tier.n_tokens > 0


# ---------------------------------------------------------------------------
# TestDiffKVPolicy
# ---------------------------------------------------------------------------


class TestDiffKVPolicy:
    def _policy(self, boost=False):
        return DiffKVPolicy(
            layer_idx=0,
            head_idx=0,
            critical_k_bits=8,
            critical_v_bits=4,
            marginal_k_bits=4,
            marginal_v_bits=2,
            sparsity_boost_active=boost,
        )

    def test_critical_bits(self):
        p = self._policy()
        assert p.effective_k_bits(IMPORTANCE_CRITICAL) == 8
        assert p.effective_v_bits(IMPORTANCE_CRITICAL) == 4

    def test_marginal_bits(self):
        p = self._policy()
        assert p.effective_k_bits(IMPORTANCE_MARGINAL) == 4
        assert p.effective_v_bits(IMPORTANCE_MARGINAL) == 2

    def test_marginal_boost_reduces_bits(self):
        p = self._policy(boost=True)
        assert p.effective_k_bits(IMPORTANCE_MARGINAL) <= 4
        assert p.effective_v_bits(IMPORTANCE_MARGINAL) <= 2

    def test_evict_tier_returns_min_bits(self):
        p = self._policy()
        assert p.effective_k_bits(IMPORTANCE_EVICT) == 2


# ---------------------------------------------------------------------------
# TestDiffKVPolicyManager
# ---------------------------------------------------------------------------


class TestDiffKVPolicyManager:
    def test_all_policies_count(self):
        cfg = _cfg(n_layers=2, n_heads=3)
        manager = DiffKVPolicyManager(cfg)
        assert len(manager.all_policies()) == 6

    def test_get_policy_returns_policy(self):
        cfg = _cfg()
        manager = DiffKVPolicyManager(cfg)
        policy = manager.get_policy(0, 0)
        assert policy.layer_idx == 0

    def test_record_and_activate_boost(self):
        cfg = _cfg(sparsity_threshold=0.5)
        manager = DiffKVPolicyManager(cfg)
        # Record all-zero attention (100% sparse)
        attn = np.zeros((8, 8), dtype=np.float32)
        manager.record_attention(0, 0, attn)
        policy = manager.get_policy(0, 0)
        assert policy.sparsity_boost_active is True

    def test_no_boost_for_dense_head(self):
        cfg = _cfg(sparsity_threshold=0.99)
        manager = DiffKVPolicyManager(cfg)
        # Dense attention → sparsity near 0
        attn = np.ones((8, 8), dtype=np.float32) * 0.5
        manager.record_attention(0, 0, attn)
        policy = manager.get_policy(0, 0)
        assert policy.sparsity_boost_active is False


# ---------------------------------------------------------------------------
# TestCompactedKVSlot
# ---------------------------------------------------------------------------


class TestCompactedKVSlot:
    def _slot(self, n_crit=10, n_marg=20, n_evict=5, head_dim=64):
        policy = DiffKVPolicy(
            layer_idx=0, head_idx=0,
            critical_k_bits=8, critical_v_bits=4,
            marginal_k_bits=4, marginal_v_bits=2,
        )
        return CompactedKVSlot(
            layer_idx=0, head_idx=0,
            n_critical=n_crit, n_marginal=n_marg, n_evicted=n_evict,
            head_dim=head_dim, policy=policy,
        )

    def test_n_retained(self):
        slot = self._slot(n_crit=10, n_marg=20, n_evict=5)
        assert slot.n_retained == 30

    def test_compression_ratio_gt_1(self):
        slot = self._slot()
        # Mixed precision < FP16 cost → compression ratio > 1
        assert slot.compression_ratio > 1.0

    def test_bytes_fp16_gt_bytes_diffkv(self):
        slot = self._slot()
        assert slot.bytes_fp16_equivalent > slot.bytes_used


# ---------------------------------------------------------------------------
# TestCompactKV
# ---------------------------------------------------------------------------


class TestCompactKV:
    def test_returns_compacted_slot(self):
        cfg = _cfg()
        policy = DiffKVPolicy(
            layer_idx=0, head_idx=0,
            critical_k_bits=8, critical_v_bits=4,
            marginal_k_bits=4, marginal_v_bits=2,
        )
        scores = np.arange(20, dtype=np.float32)
        slot = compact_kv(scores, 0, 0, 64, policy, cfg)
        assert isinstance(slot, CompactedKVSlot)
        assert slot.n_critical + slot.n_marginal + slot.n_evicted == 20


# ---------------------------------------------------------------------------
# TestDiffKVStats
# ---------------------------------------------------------------------------


class TestDiffKVStats:
    def _slot(self):
        policy = DiffKVPolicy(
            layer_idx=0, head_idx=0,
            critical_k_bits=8, critical_v_bits=4,
            marginal_k_bits=4, marginal_v_bits=2,
        )
        return CompactedKVSlot(
            layer_idx=0, head_idx=0,
            n_critical=5, n_marginal=10, n_evicted=5,
            head_dim=64, policy=policy,
        )

    def test_record_increments_totals(self):
        stats = DiffKVStats()
        stats.record_slot(self._slot())
        assert stats.total_slots == 1
        assert stats.total_critical_tokens == 5

    def test_eviction_rate(self):
        stats = DiffKVStats()
        stats.record_slot(self._slot())
        total = 5 + 10 + 5
        assert abs(stats.eviction_rate - 5 / total) < 1e-6

    def test_compression_ratio_positive(self):
        stats = DiffKVStats()
        stats.record_slot(self._slot())
        assert stats.overall_compression_ratio > 0

    def test_estimated_throughput_multiplier_in_range(self):
        stats = DiffKVStats()
        for _ in range(10):
            stats.record_slot(self._slot())
        assert 1.0 <= stats.estimated_throughput_multiplier <= 6.0
