"""
tests/kv/test_q_filters.py

Unit tests for Phase 3: Q-Filters geometric KV cache compression.

Covers:
  - QFilterConfig validation
  - QFilterState calibration lifecycle
  - QFilterState.score_recent correctness
  - QFilterState.rebuild_after_eviction
  - _qfilter_evict rebuilds KVLayerCache correctly
  - QFilterManager end-to-end routing
  - QuantizedKVCache with qfilter_rank > 0 (slots, init, tick_qfilter)
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.kv_cache import KVLayerCache, QuantizedKVCache
from squish.kv.q_filters import (
    QFilterConfig,
    QFilterManager,
    QFilterState,
    _qfilter_evict,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

N_HEADS  = 4
HEAD_DIM = 16
RANK     = 8
RNG      = np.random.default_rng(42)


def _rand_key(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)


def _rand_value(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)


def _make_cfg(**kw) -> QFilterConfig:
    defaults = dict(rank=RANK, budget=10, anchor=4, min_tokens=RANK, evict_every=0)
    defaults.update(kw)
    return QFilterConfig(**defaults)


def _feed_state(state: QFilterState, n: int, rng_seed: int = 0) -> None:
    """Append ``n`` random keys to ``state``."""
    rng = np.random.default_rng(rng_seed)
    for _ in range(n):
        k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        state.append_key(k)


def _make_layer_with_n_tokens(n: int, window: int = 4) -> KVLayerCache:
    """Create a KVLayerCache already populated with ``n`` tokens."""
    layer = KVLayerCache(window=window)
    rng = np.random.default_rng(7)
    for i in range(n):
        k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        layer.append(k, v)
    return layer


# ---------------------------------------------------------------------------
# QFilterConfig
# ---------------------------------------------------------------------------

class TestQFilterConfig:
    def test_defaults(self):
        cfg = QFilterConfig()
        assert cfg.rank        == 32
        assert cfg.budget      == 2048
        assert cfg.anchor      == 64
        assert cfg.min_tokens  == 64
        assert cfg.evict_every == 16

    def test_custom(self):
        cfg = QFilterConfig(rank=8, budget=100, anchor=16, min_tokens=8, evict_every=4)
        assert cfg.rank == 8
        assert cfg.budget == 100

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError, match="rank"):
            QFilterConfig(rank=0)

    def test_budget_zero_raises(self):
        with pytest.raises(ValueError, match="budget"):
            QFilterConfig(budget=0)

    def test_anchor_negative_raises(self):
        with pytest.raises(ValueError, match="anchor"):
            QFilterConfig(anchor=-1)

    def test_min_tokens_less_than_rank_raises(self):
        with pytest.raises(ValueError, match="min_tokens"):
            QFilterConfig(rank=16, min_tokens=8)


# ---------------------------------------------------------------------------
# QFilterState — lifecycle
# ---------------------------------------------------------------------------

class TestQFilterStateLifecycle:
    def test_initial_not_calibrated(self):
        state = QFilterState(_make_cfg())
        assert not state.is_calibrated
        assert state.n_projected == 0

    def test_score_before_calibration_returns_none(self):
        state = QFilterState(_make_cfg())
        recent = RNG.standard_normal((N_HEADS, 4, HEAD_DIM)).astype(np.float32)
        assert state.score_recent(recent) is None

    def test_calibration_triggers_at_min_tokens(self):
        cfg   = _make_cfg(min_tokens=RANK)  # calibrate after RANK tokens
        state = QFilterState(cfg)
        _feed_state(state, RANK - 1)
        assert not state.is_calibrated, "one token short — should not have calibrated"
        state.append_key(_rand_key(seed=99))
        assert state.is_calibrated

    def test_basis_shape_after_calibration(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK)
        assert state._basis is not None
        assert state._basis.shape == (N_HEADS, RANK, HEAD_DIM)

    def test_kproj_populated_after_calibration(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK)
        # All calibration tokens are projected
        assert state._kproj is not None
        assert state._kproj.shape == (N_HEADS, RANK, RANK)

    def test_kproj_grows_per_postical_token(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK)
        n_before = state.n_projected
        state.append_key(_rand_key(seed=200))
        assert state.n_projected == n_before + 1

    def test_reset_clears_per_request_state(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK + 5)
        assert state.is_calibrated
        state.reset()
        assert not state.is_calibrated
        assert state.n_projected == 0
        assert state._calib_buf == []

    def test_reset_preserves_basis_by_default(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK)
        basis_before = state._basis.copy()
        state.reset()
        # basis preserved
        assert state._basis is not None
        assert np.allclose(state._basis, basis_before)

    def test_reset_clear_basis_removes_basis(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK)
        state.reset(clear_basis=True)
        assert state._basis is None


# ---------------------------------------------------------------------------
# QFilterState — scoring
# ---------------------------------------------------------------------------

class TestQFilterStateScoring:
    def _calibrated_state(self, n_extra: int = 0) -> QFilterState:
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK + n_extra)
        return state

    def test_score_shape(self):
        state  = self._calibrated_state(n_extra=5)
        n_tok  = state.n_projected
        recent = RNG.standard_normal((N_HEADS, 4, HEAD_DIM)).astype(np.float32)
        scores = state.score_recent(recent)
        assert scores is not None
        assert scores.shape == (n_tok,)

    def test_scores_in_valid_cosine_range(self):
        state  = self._calibrated_state(n_extra=10)
        recent = RNG.standard_normal((N_HEADS, 4, HEAD_DIM)).astype(np.float32)
        scores = state.score_recent(recent)
        assert scores is not None
        # cosine similarity is in [-1, 1]; mean over heads can be in [-1,1]
        assert np.all(scores >= -1.0 - 1e-6)
        assert np.all(scores <= 1.0  + 1e-6)

    def test_aligned_key_gets_higher_score_than_orthogonal(self):
        """
        If we feed a set of nearly-identical keys pointing in direction ``d``,
        the one token also pointing in ``d`` should score higher than a
        token pointing in an orthogonal direction.
        """
        cfg = QFilterConfig(rank=4, budget=10, anchor=2, min_tokens=4, evict_every=0)
        state = QFilterState(cfg)

        d = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
        d[:, 0] = 1.0  # all heads point along dim-0

        # Feed calibration tokens along d
        for _ in range(4):
            state.append_key((d + RNG.standard_normal(d.shape) * 0.01).astype(np.float16))

        # One token aligned with d, one orthogonal
        k_aligned    = d.astype(np.float16)
        k_orthogonal = np.zeros_like(d, dtype=np.float16)
        k_orthogonal[:, 1] = 1.0  # all heads in dim-1
        state.append_key(k_aligned)
        state.append_key(k_orthogonal)

        # Recent proxies pointing along d
        recent = d[:, np.newaxis, :].repeat(2, axis=1).astype(np.float32)
        scores = state.score_recent(recent)
        assert scores is not None
        n = state.n_projected
        # Last two tokens: aligned and orthogonal
        assert scores[n - 2] > scores[n - 1], (
            f"aligned score {scores[n-2]:.4f} should be > orthogonal {scores[n-1]:.4f}"
        )

    def test_score_uses_mean_of_recent_anchor(self):
        """score_recent should accept multi-token recent arrays."""
        state  = self._calibrated_state(n_extra=8)
        recent = RNG.standard_normal((N_HEADS, 6, HEAD_DIM)).astype(np.float16)
        scores = state.score_recent(recent)
        assert scores is not None  # multi-token anchor accepted OK


# ---------------------------------------------------------------------------
# QFilterState — rebuild_after_eviction
# ---------------------------------------------------------------------------

class TestQFilterStateRebuild:
    def test_rebuild_trims_projected_buffer(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK + 10)   # 10 post-calibration tokens
        n_before = state.n_projected    # should be 10

        keep = np.array([0, 2, 4, 6, 8])
        state.rebuild_after_eviction(keep)
        assert state.n_projected == len(keep)

    def test_rebuild_preserves_correct_projections(self):
        cfg   = _make_cfg(min_tokens=RANK)
        state = QFilterState(cfg)
        _feed_state(state, RANK + 6)

        kproj_before = state._kproj.copy()  # (n_heads, 6, rank)
        keep = np.array([1, 3, 5])
        state.rebuild_after_eviction(keep)

        assert np.allclose(state._kproj, kproj_before[:, keep, :])

    def test_rebuild_with_no_kproj_is_safe(self):
        state = QFilterState(_make_cfg())
        # Not yet calibrated — kproj is None
        state.rebuild_after_eviction(np.array([0, 1, 2]))  # should not raise


# ---------------------------------------------------------------------------
# _qfilter_evict
# ---------------------------------------------------------------------------

class TestQFilterEvict:
    def test_evict_reduces_token_count(self):
        layer = _make_layer_with_n_tokens(20)
        n_before = layer.n_tokens
        keep = np.arange(10)  # keep first 10
        _qfilter_evict(layer, keep)
        assert layer.n_tokens == 10

    def test_evict_preserves_selected_keys(self):
        layer = _make_layer_with_n_tokens(20)
        full_k, full_v = layer.get_full_kv()
        sel_k = full_k[:, 3:8, :]   # keep positions 3..7

        keep = np.array([3, 4, 5, 6, 7])
        _qfilter_evict(layer, keep)

        rebuilt_k, rebuilt_v = layer.get_full_kv()
        assert rebuilt_k.shape[1] == 5
        assert np.allclose(rebuilt_k.astype(np.float32),
                           sel_k.astype(np.float32), atol=0.02), \
            "evicted cache should reproduce selected keys (within INT8 quant tolerance)"

    def test_evict_all_tokens_is_safe(self):
        layer = _make_layer_with_n_tokens(5)
        keep = np.arange(5)
        _qfilter_evict(layer, keep)
        assert layer.n_tokens == 5  # no change

    def test_evict_empty_layer_is_safe(self):
        layer = KVLayerCache(window=4)
        _qfilter_evict(layer, np.array([]))  # nothing to evict


# ---------------------------------------------------------------------------
# QFilterManager
# ---------------------------------------------------------------------------

class TestQFilterManager:
    def _make_manager(self, n_layers: int = 2) -> QFilterManager:
        cfg = _make_cfg(min_tokens=RANK, budget=6, anchor=2)
        return QFilterManager(cfg, n_layers)

    def test_init_creates_correct_number_of_states(self):
        mgr = self._make_manager(n_layers=4)
        assert len(mgr._states) == 4

    def test_notify_key_delegates_to_correct_layer(self):
        mgr = self._make_manager(n_layers=2)
        k   = _rand_key()
        for _ in range(RANK):
            mgr.notify_key(0, k)
        assert mgr._states[0].is_calibrated
        assert not mgr._states[1].is_calibrated

    def test_reset_clears_all_layers(self):
        mgr = self._make_manager(n_layers=2)
        for _ in range(RANK + 2):
            mgr.notify_key(0, _rand_key())
        mgr.reset()
        assert not mgr._states[0].is_calibrated

    def test_maybe_evict_no_op_when_under_budget(self):
        mgr   = self._make_manager()
        layer = _make_layer_with_n_tokens(5)  # 5 < budget=6
        for i in range(5):
            mgr.notify_key(0, _rand_key(seed=i))
        evicted = mgr.maybe_evict(0, layer, step=0)
        assert not evicted
        assert layer.n_tokens == 5

    def test_maybe_evict_fires_when_over_budget(self):
        cfg   = _make_cfg(min_tokens=RANK, budget=6, anchor=2, evict_every=0)
        mgr   = QFilterManager(cfg, n_layers=1)
        layer = KVLayerCache(window=4)
        # Feed BOTH layer AND manager in sync so n_tokens == n_projected after calibration
        rng   = np.random.default_rng(99)
        n_total = RANK + 6   # calibrate then 6 more (total 14 > budget=6)
        for i in range(n_total):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            layer.append(k, v)
            mgr.notify_key(0, k)

        assert mgr._states[0].is_calibrated
        assert layer.n_tokens == n_total
        assert mgr._states[0].n_projected == n_total

        evicted = mgr.maybe_evict(0, layer, step=0)
        assert evicted
        assert layer.n_tokens <= 6

    def test_maybe_evict_respects_evict_every(self):
        cfg   = _make_cfg(min_tokens=RANK, budget=4, anchor=2, evict_every=4)
        mgr   = QFilterManager(cfg, n_layers=1)
        layer = KVLayerCache(window=4)
        rng   = np.random.default_rng(77)
        n_total = RANK + 6   # calibrate then add 6 more (total > budget=4)
        for i in range(n_total):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            layer.append(k, v)
            mgr.notify_key(0, k)

        # step=1 is NOT divisible by 4 → no eviction
        evicted = mgr.maybe_evict(0, layer, step=1)
        assert not evicted

        # step=4 IS divisible by 4 → eviction fires
        evicted = mgr.maybe_evict(0, layer, step=4)
        assert evicted


# ---------------------------------------------------------------------------
# QuantizedKVCache integration — qfilter_rank > 0
# ---------------------------------------------------------------------------

class TestQuantizedKVCacheQFilter:
    def test_qfilter_cfg_set_when_rank_nonzero(self):
        cache = QuantizedKVCache(n_layers=2, qfilter_rank=RANK, qfilter_budget=8,
                                 qfilter_anchor=2, qfilter_evict_every=0)
        assert cache._qfilter_cfg is not None
        assert cache._qfilter_cfg.rank == RANK

    def test_qfilter_state_attached_to_each_layer(self):
        cache = QuantizedKVCache(n_layers=3, qfilter_rank=RANK)
        for layer in cache._layers:
            assert layer._qfilter is not None
            assert isinstance(layer._qfilter, QFilterState)

    def test_qfilter_cfg_none_when_rank_zero(self):
        cache = QuantizedKVCache(n_layers=2)
        assert cache._qfilter_cfg is None
        for layer in cache._layers:
            assert layer._qfilter is None

    def test_append_feeds_qfilter(self):
        cache = QuantizedKVCache(n_layers=2, qfilter_rank=RANK,
                                 qfilter_budget=8, qfilter_evict_every=0)
        # QuantizedKVCache sets min_tokens = max(rank, 64); need 64 appends to calibrate
        n_cal = 64
        layer = cache._layers[0]
        for i in range(n_cal):
            layer.append(_rand_key(seed=i), _rand_value(seed=i))
        assert layer._qfilter.is_calibrated

    def test_reset_clears_qfilter_state(self):
        cache = QuantizedKVCache(n_layers=2, qfilter_rank=RANK, qfilter_evict_every=0)
        layer = cache._layers[0]
        # min_tokens = max(RANK, 64) = 64; feed 70 tokens to ensure calibration
        for i in range(70):
            layer.append(_rand_key(seed=i), _rand_value(seed=i))
        assert layer._qfilter.is_calibrated
        cache.reset()
        assert not layer._qfilter.is_calibrated
        assert layer._qfilter.n_projected == 0

    def test_tick_qfilter_is_noop_when_disabled(self):
        cache = QuantizedKVCache(n_layers=1)  # qfilter_rank=0
        layer = _make_layer_with_n_tokens(20)
        cache._layers[0] = layer
        # Should not raise
        cache.tick_qfilter(step=0)

    def test_tick_qfilter_evicts_when_over_budget(self):
        budget = 8
        cache  = QuantizedKVCache(
            n_layers=1, qfilter_rank=RANK,
            qfilter_budget=budget, qfilter_anchor=2,
            qfilter_evict_every=0,
        )
        layer = cache._layers[0]

        # Calibrate first (min_tokens = max(RANK, 64) = 64), then add extras
        n_tokens = 64 + budget + 2   # 74 > budget=8; first 64 are for calibration
        for i in range(n_tokens):
            layer.append(_rand_key(seed=i), _rand_value(seed=i))

        assert layer._qfilter.is_calibrated, "must be calibrated before eviction can fire"
        cache.tick_qfilter(step=0)
        assert layer.n_tokens <= budget

    def test_tick_qfilter_before_calibration_is_safe(self):
        cache = QuantizedKVCache(n_layers=1, qfilter_rank=RANK, qfilter_evict_every=0)
        layer = cache._layers[0]
        # Only 2 tokens — not yet calibrated
        layer.append(_rand_key(seed=0), _rand_value(seed=0))
        layer.append(_rand_key(seed=1), _rand_value(seed=1))
        # Should not raise even though not calibrated
        cache.tick_qfilter(step=0)

    def test_stats_includes_qfilter_keys(self):
        cache = QuantizedKVCache(n_layers=2, qfilter_rank=RANK)
        s = cache.stats()
        assert "qfilter_rank"       in s
        assert "qfilter_budget"     in s
        assert "qfilter_calibrated" in s
        assert s["qfilter_rank"] == RANK

    def test_stats_no_qfilter_keys_when_disabled(self):
        cache = QuantizedKVCache(n_layers=2)
        s = cache.stats()
        assert "qfilter_rank" not in s

    def test_kproj_buffer_matches_token_count_after_eviction(self):
        """After eviction, kproj buffer and n_tokens stay in sync."""
        cache = QuantizedKVCache(
            n_layers=1, qfilter_rank=RANK,
            qfilter_budget=6, qfilter_anchor=2,
            qfilter_evict_every=0,
        )
        layer = cache._layers[0]
        # Calibrate (64 tokens needed) then add extras to exceed budget
        n_tokens = 64 + 6   # 70 total; after calibration n_projected==n_tokens
        for i in range(n_tokens):
            layer.append(_rand_key(seed=i), _rand_value(seed=i))

        assert layer._qfilter.is_calibrated
        cache.tick_qfilter(step=0)

        # The kproj buffer should reflect the surviving token count
        surviving = layer.n_tokens
        assert surviving <= 6
        assert layer._qfilter.n_projected == surviving
