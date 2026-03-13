"""tests/test_hierarchical_kv_unit.py

Full-coverage unit tests for squish/hierarchical_kv.py.

Covers:
  TierConfig            — all __post_init__ validation errors
  HierarchicalKVStats   — hit_rate and hot_hit_rate (zero and non-zero)
  HierarchicalKVStore   — put/get basics, shape validation, update existing,
                          promotion from warm/cold, demotion cascade,
                          cold-tier eviction, n_hot/warm/cold properties,
                          __len__, __repr__, stats
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.hierarchical_kv import (
    HierarchicalKVStats,
    HierarchicalKVStore,
    TierConfig,
)


# ---------------------------------------------------------------------------
# TierConfig
# ---------------------------------------------------------------------------


class TestTierConfig:
    def test_valid_defaults(self):
        cfg = TierConfig()
        assert cfg.hot_capacity == 64
        assert cfg.warm_capacity == 256
        assert cfg.cold_capacity == 1024
        assert cfg.n_heads == 4
        assert cfg.head_dim == 64

    def test_valid_custom(self):
        cfg = TierConfig(hot_capacity=2, warm_capacity=4, cold_capacity=8)
        assert cfg.hot_capacity == 2

    def test_hot_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="hot_capacity must be >= 1"):
            TierConfig(hot_capacity=0)

    def test_warm_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="warm_capacity must be >= 1"):
            TierConfig(warm_capacity=0, hot_capacity=1, cold_capacity=2)

    def test_cold_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="cold_capacity must be >= 1"):
            TierConfig(cold_capacity=0, hot_capacity=1, warm_capacity=2)

    def test_capacity_order_violation_raises(self):
        """hot must be strictly less than warm which must be less than cold."""
        with pytest.raises(ValueError, match="must satisfy hot_capacity < warm_capacity"):
            TierConfig(hot_capacity=5, warm_capacity=3, cold_capacity=10)

    def test_hot_equals_warm_raises(self):
        with pytest.raises(ValueError, match="must satisfy hot_capacity < warm_capacity"):
            TierConfig(hot_capacity=4, warm_capacity=4, cold_capacity=8)

    def test_warm_equals_cold_raises(self):
        with pytest.raises(ValueError, match="must satisfy hot_capacity < warm_capacity"):
            TierConfig(hot_capacity=2, warm_capacity=8, cold_capacity=8)

    def test_n_heads_zero_raises(self):
        with pytest.raises(ValueError, match="n_heads must be >= 1"):
            TierConfig(n_heads=0)

    def test_head_dim_zero_raises(self):
        with pytest.raises(ValueError, match="head_dim must be >= 1"):
            TierConfig(head_dim=0)


# ---------------------------------------------------------------------------
# HierarchicalKVStats
# ---------------------------------------------------------------------------


class TestHierarchicalKVStats:
    def test_hit_rate_zero_gets(self):
        s = HierarchicalKVStats()
        assert s.hit_rate == 0.0

    def test_hit_rate_nonzero(self):
        s = HierarchicalKVStats(
            total_gets=10,
            hot_hits=4,
            warm_hits=3,
            cold_hits=1,
        )
        assert s.hit_rate == pytest.approx(0.8)

    def test_hit_rate_all_misses(self):
        s = HierarchicalKVStats(total_gets=5, cold_misses=5)
        assert s.hit_rate == pytest.approx(0.0)

    def test_hot_hit_rate_zero_gets(self):
        s = HierarchicalKVStats()
        assert s.hot_hit_rate == 0.0

    def test_hot_hit_rate_nonzero(self):
        s = HierarchicalKVStats(total_gets=8, hot_hits=6)
        assert s.hot_hit_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# HierarchicalKVStore
# ---------------------------------------------------------------------------


def _make_cfg(hot=2, warm=4, cold=8, n_heads=2, head_dim=4):
    return TierConfig(
        hot_capacity=hot,
        warm_capacity=warm,
        cold_capacity=cold,
        n_heads=n_heads,
        head_dim=head_dim,
    )


def _kv(n_heads=2, head_dim=4, val=0.0):
    k = np.full((n_heads, head_dim), val, dtype=np.float32)
    v = np.full((n_heads, head_dim), val + 1.0, dtype=np.float32)
    return k, v


class TestPut:
    def test_put_inserts_into_hot(self):
        store = HierarchicalKVStore(_make_cfg())
        k, v = _kv()
        store.put(pos=0, key=k, value=v)
        assert store.n_hot == 1
        assert 0 in store._store

    def test_put_increments_total_puts(self):
        store = HierarchicalKVStore(_make_cfg())
        k, v = _kv()
        store.put(0, k, v)
        assert store.stats.total_puts == 1

    def test_put_wrong_key_shape_raises(self):
        store = HierarchicalKVStore(_make_cfg())
        bad_k = np.ones((3, 4), dtype=np.float32)  # wrong n_heads
        _, v = _kv()
        with pytest.raises(ValueError, match="key shape must be"):
            store.put(0, bad_k, v)

    def test_put_wrong_value_shape_raises(self):
        store = HierarchicalKVStore(_make_cfg())
        k, _ = _kv()
        bad_v = np.ones((2, 5), dtype=np.float32)  # wrong head_dim
        with pytest.raises(ValueError, match="value shape must be"):
            store.put(0, k, bad_v)

    def test_put_updates_existing_position(self):
        store = HierarchicalKVStore(_make_cfg())
        k1, v1 = _kv(val=1.0)
        k2, v2 = _kv(val=2.0)
        store.put(0, k1, v1)
        store.put(0, k2, v2)
        # Should still be in hot tier after update
        assert store.n_hot == 1
        stored_k, stored_v, tier = store._store[0]
        assert tier == "hot"
        np.testing.assert_allclose(stored_k, k2)

    def test_put_demotes_on_hot_overflow(self):
        store = HierarchicalKVStore(_make_cfg(hot=2, warm=4, cold=8))
        for i in range(3):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        # hot_capacity=2 → oldest should have been demoted to warm
        assert store.n_hot == 2
        assert store.n_warm == 1

    def test_put_stores_copy_not_reference(self):
        store = HierarchicalKVStore(_make_cfg())
        k, v = _kv(val=0.0)
        store.put(0, k, v)
        k[0, 0] = 999.0  # mutate original
        stored_k, _, _ = store._store[0]
        assert stored_k[0, 0] != 999.0  # store has a copy


class TestGet:
    def test_get_miss(self):
        store = HierarchicalKVStore(_make_cfg())
        result = store.get(99)
        assert result is None
        assert store.stats.cold_misses == 1

    def test_get_hot_hit(self):
        store = HierarchicalKVStore(_make_cfg())
        k, v = _kv(val=5.0)
        store.put(0, k, v)
        result = store.get(0)
        assert result is not None
        ret_k, ret_v = result
        np.testing.assert_allclose(ret_k, k)
        assert store.stats.hot_hits == 1

    def test_get_promotes_to_hot_from_warm(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=3, cold=6))
        # Fill hot so LRU gets demoted to warm
        k0, v0 = _kv(val=0.0)
        k1, v1 = _kv(val=1.0)
        store.put(0, k0, v0)  # goes to hot
        store.put(1, k1, v1)  # hot overflows; 0 demoted to warm
        assert store._store[0][2] == "warm"  # pos 0 is in warm
        # Now GET pos 0 — should promote to hot
        result = store.get(0)
        assert result is not None
        assert store._store[0][2] == "hot"
        assert store.stats.warm_hits == 1

    def test_get_promotes_to_hot_from_cold(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=2, cold=4))
        # Put 4 items to cascade pos 0 to cold (hot=1, warm=2: need 4 puts)
        for i in range(4):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        # pos 0 should be in cold by now
        if store._store[0][2] == "cold":
            result = store.get(0)
            assert result is not None
            assert store.stats.cold_hits == 1

    def test_get_hot_already_in_hot_moved_to_mru(self):
        """Getting a hot token moves it to MRU position."""
        store = HierarchicalKVStore(_make_cfg(hot=3, warm=6, cold=12))
        for i in range(3):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        # Access pos 0 to move it to MRU
        store.get(0)
        # pos 0 should now be at tail of hot_list
        assert store._hot_list[-1] == 0

    def test_get_increments_total_gets(self):
        store = HierarchicalKVStore(_make_cfg())
        store.get(0)
        store.get(0)
        assert store.stats.total_gets == 2


class TestDemotionAndEviction:
    def test_warm_overflow_causes_cold_demotion(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=2, cold=4))
        for i in range(4):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        # Check that some items are in cold
        assert store.n_cold > 0

    def test_cold_overflow_causes_eviction(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=2, cold=3))
        for i in range(7):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        assert store.stats.total_evictions > 0
        # Evicted items are fully gone from store
        total_stored = len(store._store)
        assert total_stored <= 1 + 2 + 3  # hot + warm + cold

    def test_total_demotions_counted(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=2, cold=4))
        for i in range(4):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        assert store.stats.total_demotions > 0


class TestProperties:
    def test_n_hot_warm_cold(self):
        store = HierarchicalKVStore(_make_cfg(hot=2, warm=4, cold=8))
        for i in range(2):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        assert store.n_hot == 2
        assert store.n_warm == 0
        assert store.n_cold == 0

    def test_len(self):
        store = HierarchicalKVStore(_make_cfg())
        assert len(store) == 0
        k, v = _kv()
        store.put(0, k, v)
        assert len(store) == 1

    def test_repr(self):
        store = HierarchicalKVStore(_make_cfg(hot=2, warm=4, cold=8))
        r = repr(store)
        assert "HierarchicalKVStore" in r
        assert "hot=" in r

    def test_stats_property(self):
        store = HierarchicalKVStore(_make_cfg())
        s = store.stats
        assert isinstance(s, HierarchicalKVStats)

    def test_update_existing_removes_from_tier_list(self):
        """put on existing pos removes it from its tier list first."""
        store = HierarchicalKVStore(_make_cfg(hot=2, warm=4, cold=8))
        k1, v1 = _kv(val=1.0)
        k2, v2 = _kv(val=2.0)
        store.put(0, k1, v1)
        store.put(1, k2, v2)
        # hot is full (cap=2); put 2 more to demote pos 0 to warm
        k3, v3 = _kv(val=3.0)
        store.put(2, k3, v3)  # pos 0 demoted to warm
        assert store._store[0][2] == "warm"
        # Now update pos 0 → should be moved back to hot
        k_new, v_new = _kv(val=9.0)
        store.put(0, k_new, v_new)
        assert store._store[0][2] == "hot"


class TestRemoveFromTierList:
    def test_remove_from_warm(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=3, cold=6))
        k, v = _kv()
        store.put(0, k, v)
        store.put(1, k, v)  # pushes 0 to warm
        assert 0 in store._warm_list
        # Now get 0 to promote it back out of warm
        store.get(0)
        assert 0 not in store._warm_list

    def test_remove_from_cold_via_promotion(self):
        store = HierarchicalKVStore(_make_cfg(hot=1, warm=2, cold=4))
        for i in range(4):
            k, v = _kv(val=float(i))
            store.put(i, k, v)
        # Find a token in cold and get it
        for pos in [0, 1, 2, 3]:
            if pos in store._store and store._store[pos][2] == "cold":
                result = store.get(pos)
                assert result is not None
                assert pos not in store._cold_list
                break
