"""Unit tests for squish.smallkv (SmallKV saliency shift compensation)."""

import numpy as np
import pytest

from squish.smallkv import (
    MarginalVCache,
    SaliencyTracker,
    SmallKVCache,
    SmallKVConfig,
    SmallKVStats,
)


def _cfg(n_layers=4, kv_budget_fraction=0.5, **kw):
    return SmallKVConfig(n_layers=n_layers, kv_budget_fraction=kv_budget_fraction, **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# TestSmallKVConfig
# ---------------------------------------------------------------------------


class TestSmallKVConfig:
    def test_defaults(self):
        cfg = SmallKVConfig()
        assert cfg.n_layers == 32
        assert cfg.kv_budget_fraction == 0.10
        assert cfg.recall_top_k == 8

    def test_invalid_budget_fraction(self):
        with pytest.raises(ValueError):
            SmallKVConfig(kv_budget_fraction=0.0)

    def test_invalid_budget_fraction_gt1(self):
        with pytest.raises(ValueError):
            SmallKVConfig(kv_budget_fraction=1.1)

    def test_invalid_recall_top_k(self):
        with pytest.raises(ValueError):
            SmallKVConfig(recall_top_k=0)

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError):
            SmallKVConfig(score_ema_alpha=0.0)

    def test_invalid_proxy_weight(self):
        with pytest.raises(ValueError):
            SmallKVConfig(proxy_weight=1.5)

    def test_budget_plus_marginal_exceeds_one(self):
        with pytest.raises(ValueError):
            SmallKVConfig(kv_budget_fraction=0.7, marginal_v_only_fraction=0.7)


# ---------------------------------------------------------------------------
# TestSaliencyTracker
# ---------------------------------------------------------------------------


class TestSaliencyTracker:
    def test_initial_scores_none(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        assert tracker.current_scores is None

    def test_update_sets_scores(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        scores = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        tracker.update_scores(scores)
        assert tracker.current_scores is not None
        assert tracker.current_scores.size == 3

    def test_multi_head_scores_averaged(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        scores = np.ones((4, 8), dtype=np.float32) * 0.5
        tracker.update_scores(scores)
        assert tracker.current_scores.size == 8

    def test_mark_evicted_tracked(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        tracker.mark_evicted([0, 1, 2])
        assert tracker.n_evicted == 3

    def test_detect_no_shifts_without_updates(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        tracker.mark_evicted([0, 1])
        shifts = tracker.detect_saliency_shifts()
        assert shifts == []

    def test_detect_shift_above_threshold(self):
        cfg = _cfg(saliency_shift_threshold=0.05)
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        tracker.mark_evicted([0, 1])
        # Token 1 has high score → should be recalled
        scores = np.array([0.01, 0.8], dtype=np.float32)
        tracker.update_scores(scores)
        shifts = tracker.detect_saliency_shifts()
        assert 1 in shifts

    def test_recall_removes_from_evicted(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        tracker.mark_evicted([0, 1, 2])
        tracker.mark_recalled([1])
        assert tracker.n_evicted == 2

    def test_reset_clears_all(self):
        cfg = _cfg()
        tracker = SaliencyTracker(config=cfg, layer_idx=0)
        tracker.update_scores(np.ones(8))
        tracker.mark_evicted([0])
        tracker.reset()
        assert tracker.current_scores is None
        assert tracker.n_evicted == 0


# ---------------------------------------------------------------------------
# TestSmallKVCache
# ---------------------------------------------------------------------------


class TestSmallKVCache:
    def _make_cache(self):
        cfg = _cfg(kv_budget_fraction=0.5, marginal_v_only_fraction=0.2)
        return SmallKVCache(cfg), cfg

    def _data(self, seq=10, head_dim=16, seed=0):
        rng = _rng(seed)
        tokens = np.arange(seq)
        keys = rng.normal(0, 1, (seq, head_dim)).astype(np.float32)
        vals = rng.normal(0, 1, (seq, head_dim)).astype(np.float32)
        scores = rng.uniform(0, 1, seq).astype(np.float32)
        return tokens, keys, vals, scores

    def test_ingest_stores_critical(self):
        cache, _ = self._make_cache()
        tokens, keys, vals, scores = self._data()
        cache.ingest(0, tokens, keys, vals, scores)
        # At least some tokens should be in critical store
        assert len(cache._kv_store.get(0, {})) > 0

    def test_ingest_stats_updated(self):
        cache, _ = self._make_cache()
        tokens, keys, vals, scores = self._data()
        cache.ingest(0, tokens, keys, vals, scores)
        assert cache.stats.total_tokens > 0

    def test_get_kv_critical_returns_both(self):
        cfg = SmallKVConfig(n_layers=4, kv_budget_fraction=0.9, marginal_v_only_fraction=0.05)
        cache = SmallKVCache(cfg)
        tokens, keys, vals, scores = self._data(seq=5)
        cache.ingest(0, tokens, keys, vals, scores)
        # With 90% budget, first token should be critical
        k, v = cache.get_kv(0, int(tokens[np.argmax(scores)]))
        assert k is not None and v is not None

    def test_get_kv_evicted_returns_none(self):
        cfg = SmallKVConfig(n_layers=4, kv_budget_fraction=0.1, marginal_v_only_fraction=0.1)
        cache = SmallKVCache(cfg)
        tokens, keys, vals, scores = self._data(seq=20)
        cache.ingest(0, tokens, keys, vals, scores)
        k, v = cache.get_kv(0, 999)  # non-existent token
        assert k is None and v is None

    def test_check_and_recall_returns_list(self):
        cache, _ = self._make_cache()
        tokens, keys, vals, scores = self._data()
        cache.ingest(0, tokens, keys, vals, scores)
        recalled = cache.check_and_recall(0, np.ones(10) * 0.9)
        assert isinstance(recalled, list)

    def test_reset_layer_clears(self):
        cache, _ = self._make_cache()
        tokens, keys, vals, scores = self._data()
        cache.ingest(0, tokens, keys, vals, scores)
        cache.reset(layer_idx=0)
        assert len(cache._kv_store.get(0, {})) == 0

    def test_reset_all_clears(self):
        cache, _ = self._make_cache()
        for layer in range(3):
            tokens, keys, vals, scores = self._data(seed=layer)
            cache.ingest(layer, tokens, keys, vals, scores)
        cache.reset()
        assert all(len(cache._kv_store.get(l, {})) == 0 for l in range(3))


# ---------------------------------------------------------------------------
# TestSmallKVStats
# ---------------------------------------------------------------------------


class TestSmallKVStats:
    def test_initial_zeros(self):
        s = SmallKVStats()
        assert s.total_tokens == 0
        assert s.retention_rate == 0.0

    def test_record_ingest(self):
        s = SmallKVStats()
        s.record_ingest(n_critical=5, n_marginal=3, n_evicted=2)
        assert s.total_tokens == 10

    def test_retention_rate(self):
        s = SmallKVStats()
        s.record_ingest(n_critical=5, n_marginal=5, n_evicted=10)
        assert abs(s.retention_rate - 0.5) < 1e-6

    def test_full_kv_rate(self):
        s = SmallKVStats()
        s.record_ingest(n_critical=4, n_marginal=4, n_evicted=2)
        assert abs(s.full_kv_rate - 0.4) < 1e-6

    def test_recall_rate_no_evictions(self):
        s = SmallKVStats()
        assert s.recall_rate == 0.0

    def test_estimated_throughput_in_range(self):
        s = SmallKVStats()
        s.record_ingest(n_critical=1, n_marginal=1, n_evicted=8)
        assert 1.0 <= s.estimated_throughput_multiplier <= 3.0
