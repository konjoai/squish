"""Tests for ModelShardLoader — three-tier (HOT/WARM/COLD) weight paging."""
from __future__ import annotations

import numpy as np
import pytest

from squish.io.model_shard_loader import (
    LayerShard,
    ModelShardLoader,
    ShardConfig,
    ShardTier,
)


def _layers(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.standard_normal((dim, dim)).astype(np.float32) for i in range(n)}


def _loader(n_layers=8, hot=2, warm=3, lookahead=1):
    cfg = ShardConfig(n_layers=n_layers, hot_layers=hot, warm_layers=warm, lookahead=lookahead)
    ld = ModelShardLoader(cfg)
    ld.load_model(_layers(n_layers))
    return ld


class TestShardConfig:
    def test_defaults(self):
        c = ShardConfig()
        assert (c.n_layers, c.hot_layers, c.warm_layers, c.lookahead) == (32, 4, 8, 2)

    @pytest.mark.parametrize("kw", [
        {"n_layers": 0},
        {"hot_layers": 0},
        {"warm_layers": -1},
        {"lookahead": -1},
        {"n_layers": 2, "hot_layers": 5},  # hot > n_layers
    ])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            ShardConfig(**kw)


class TestLayerShard:
    def test_is_resident(self):
        assert LayerShard(0, ShardTier.HOT, np.zeros(4, np.float32), 16).is_resident
        assert not LayerShard(0, ShardTier.COLD, None, 16).is_resident


class TestLoadModel:
    def test_tier_assignment_and_cold_eviction(self):
        ld = _loader(n_layers=8, hot=2, warm=3)
        # first 2 HOT, next 3 WARM, rest COLD (data freed)
        assert [ld.tier_of(i) for i in range(8)] == [
            ShardTier.HOT, ShardTier.HOT,
            ShardTier.WARM, ShardTier.WARM, ShardTier.WARM,
            ShardTier.COLD, ShardTier.COLD, ShardTier.COLD,
        ]
        assert ld.get_layer(0).shape == (8, 8)        # HOT resident
        with pytest.raises(RuntimeError):
            ld.get_layer(7)                            # COLD → not resident

    def test_missing_layer_data_is_zero_bytes(self):
        cfg = ShardConfig(n_layers=3, hot_layers=1, warm_layers=1)
        ld = ModelShardLoader(cfg)
        ld.load_model({0: np.ones((4, 4), np.float32)})  # layers 1,2 absent
        rep = ld.memory_report()
        assert rep["total_layers"] == 3
        # layer 1 would be WARM but had no data → size 0
        assert ld._shards[1].size_bytes == 0

    def test_len(self):
        assert len(_loader(n_layers=6)) == 6


class TestDataAccess:
    def test_get_unregistered_raises_keyerror(self):
        with pytest.raises(KeyError):
            _loader().get_layer(999)

    def test_get_hot_returns_array(self):
        ld = _loader()
        np.testing.assert_array_equal(ld.get_layer(0), ld._shards[0].data)


class TestTierManagement:
    def test_promote_to_warm_restores_zeros_after_cold(self):
        ld = _loader(n_layers=4, hot=1, warm=2)
        ld.evict_to_cold(0)
        assert ld.tier_of(0) == ShardTier.COLD
        ld.promote_to_warm(0)
        assert ld.tier_of(0) == ShardTier.WARM
        # data restored (zeros stub, matching original byte size)
        assert ld.get_layer(0) is not None
        assert np.count_nonzero(ld.get_layer(0)) == 0

    def test_promote_to_warm_already_hot_is_noop(self):
        ld = _loader(n_layers=4, hot=2, warm=2)
        ld.promote_to_warm(0)  # 0 is HOT
        assert ld.tier_of(0) == ShardTier.HOT

    def test_promote_to_hot_demotes_oldest_when_full(self):
        ld = _loader(n_layers=6, hot=2, warm=3)
        # HOT is {0,1}; promoting 2 (WARM) must demote layer 0 to WARM.
        ld.promote_to_hot(2)
        assert ld.tier_of(2) == ShardTier.HOT
        assert ld.tier_of(0) == ShardTier.WARM

    def test_promote_to_hot_cold_data_raises(self):
        ld = _loader(n_layers=4, hot=1, warm=1)
        ld.evict_to_cold(0)
        with pytest.raises(RuntimeError):
            ld.promote_to_hot(0)

    def test_evict_and_tier_keyerrors(self):
        ld = _loader()
        for fn in (ld.evict_to_cold, ld.promote_to_hot, ld.promote_to_warm, ld.tier_of):
            with pytest.raises(KeyError):
                fn(999)

    def test_prefetch_promotes_to_warm(self):
        ld = _loader(n_layers=8, hot=1, warm=4)
        ld.prefetch([5, 6])
        assert ld.tier_of(5) == ShardTier.WARM
        assert ld.tier_of(6) == ShardTier.WARM

    def test_warm_capacity_evicts_oldest_to_cold(self):
        # warm=1: promoting a second layer to WARM evicts the older WARM to COLD.
        ld = _loader(n_layers=6, hot=1, warm=1)
        # After load: HOT={0}, WARM={1}, COLD={2..5}
        ld.promote_to_warm(2)
        assert ld.tier_of(2) == ShardTier.WARM
        assert ld.tier_of(1) == ShardTier.COLD  # evicted to make room


class TestAdvanceWindow:
    def test_promotes_current_and_prefetches_lookahead(self):
        ld = _loader(n_layers=8, hot=2, warm=4, lookahead=2)
        ld.advance_window(3)
        assert ld.tier_of(3) == ShardTier.HOT
        # lookahead 4,5 → WARM
        assert ld.tier_of(4) == ShardTier.WARM
        assert ld.tier_of(5) == ShardTier.WARM

    def test_evicts_far_behind_layers(self):
        ld = _loader(n_layers=10, hot=2, warm=5, lookahead=1)
        ld.advance_window(7)
        # layers before 7 - 2 = 5 must be COLD
        for i in range(5):
            assert ld.tier_of(i) == ShardTier.COLD

    def test_advance_handles_layer_without_data(self):
        cfg = ShardConfig(n_layers=4, hot_layers=1, warm_layers=2, lookahead=1)
        ld = ModelShardLoader(cfg)
        ld.load_model({0: np.ones((4, 4), np.float32)})  # layers 1-3 have no data
        # Advancing to a data-less layer must not raise (promote_to_hot swallowed).
        ld.advance_window(2)
        assert ld.tier_of(2) in (ShardTier.WARM, ShardTier.COLD)

    def test_advance_to_unregistered_layer_is_safe(self):
        # current_layer not in shards → the promote block is skipped entirely.
        ld = _loader(n_layers=4, hot=1, warm=2, lookahead=1)
        ld.advance_window(999)  # must not raise
        assert len(ld) == 4

    def test_advance_at_last_layer_skips_out_of_range_lookahead(self):
        # lookahead target beyond n_layers is not in shards → prefetch skips it.
        ld = _loader(n_layers=4, hot=1, warm=2, lookahead=2)
        ld.advance_window(3)  # targets 4,5 don't exist
        assert ld.tier_of(3) == ShardTier.HOT


class TestReporting:
    def test_memory_report_structure(self):
        ld = _loader(n_layers=8, hot=2, warm=3, lookahead=1)
        rep = ld.memory_report()
        assert rep["hot_count"] == 2 and rep["warm_count"] == 3 and rep["cold_count"] == 3
        assert rep["hot_layers"] == [0, 1]
        assert rep["hot_bytes"] > 0 and rep["warm_bytes"] > 0
        assert rep["total_layers"] == 8
        assert rep["config"]["n_layers"] == 8

    def test_iter_hot_yields_resident_in_order(self):
        ld = _loader(n_layers=6, hot=3, warm=2)
        hot = list(ld.iter_hot())
        assert [idx for idx, _ in hot] == [0, 1, 2]
        assert all(isinstance(arr, np.ndarray) for _, arr in hot)
