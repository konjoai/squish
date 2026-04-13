"""tests/streaming/test_streaming_sink_unit.py

Unit tests for squish/streaming/streaming_sink.py.

Taxonomy: pure unit — no I/O, no MLX, deterministic.

Coverage:
  SinkConfig.__post_init__
    - n_sink_tokens < 0  → ValueError
    - window_size < 1    → ValueError
    - bad dtype string   → ValueError
    - non-float dtype    → ValueError

  SinkKVCache constructor
    - n_heads < 1        → ValueError
    - head_dim < 1       → ValueError

  SinkKVCache.add_kv
    - wrong key shape    → ValueError
    - wrong value shape  → ValueError
    - first n_sink_tokens go to sink region, rest to rolling window

  SinkKVCache.get_kv
    - empty cache        → zero-length tensors
    - sink only (< n_sink_tokens added)
    - sink + recent combined shape

  SinkKVCache eviction behaviour
    - window fills and evicts oldest recent tokens
    - n_sink tokens are NEVER evicted

  SinkKVCache.reset clears everything

  SinkStats.util_fraction and total_tokens_held

  __repr__ smoke test
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.streaming.streaming_sink import SinkConfig, SinkKVCache, SinkStats


# ── SinkConfig validation ─────────────────────────────────────────────────────


class TestSinkConfigValidation:
    def test_negative_n_sink_raises(self):
        with pytest.raises(ValueError, match="n_sink_tokens"):
            SinkConfig(n_sink_tokens=-1)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            SinkConfig(window_size=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            SinkConfig(window_size=-5)

    def test_invalid_dtype_raises(self):
        with pytest.raises((ValueError, TypeError)):
            SinkConfig(dtype="not_a_dtype")

    def test_integer_dtype_raises(self):
        with pytest.raises(ValueError, match="floating-point"):
            SinkConfig(dtype="int32")

    def test_default_config_is_valid(self):
        cfg = SinkConfig()
        assert cfg.n_sink_tokens == 4
        assert cfg.window_size == 256
        assert cfg.dtype == "float32"

    def test_zero_sink_tokens_is_valid(self):
        # All tokens go to rolling window
        cfg = SinkConfig(n_sink_tokens=0, window_size=8)
        assert cfg.n_sink_tokens == 0

    def test_float16_dtype_is_valid(self):
        cfg = SinkConfig(dtype="float16")
        assert cfg.dtype == "float16"


# ── SinkKVCache constructor ───────────────────────────────────────────────────


class TestSinkKVCacheConstructor:
    def test_n_heads_zero_raises(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=4)
        with pytest.raises(ValueError, match="n_heads"):
            SinkKVCache(cfg, n_heads=0, head_dim=8)

    def test_head_dim_zero_raises(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=4)
        with pytest.raises(ValueError, match="head_dim"):
            SinkKVCache(cfg, n_heads=4, head_dim=0)

    def test_valid_construction(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=4)
        cache = SinkKVCache(cfg, n_heads=4, head_dim=8)
        assert cache.n_sink == 0
        assert cache.n_recent == 0
        assert cache.total_tokens == 0


# ── Shape validation in add_kv ────────────────────────────────────────────────


class TestAddKVShapeValidation:
    def _make_cache(self):
        return SinkKVCache(
            SinkConfig(n_sink_tokens=2, window_size=4), n_heads=4, head_dim=8
        )

    def _kv(self, n_heads=4, head_dim=8):
        rng = np.random.default_rng(0)
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        return k, v

    def test_wrong_key_shape_raises(self):
        cache = self._make_cache()
        k, v = self._kv()
        bad_k = np.ones((3, 8), dtype=np.float32)  # wrong n_heads
        with pytest.raises(ValueError, match="key shape"):
            cache.add_kv(bad_k, v)

    def test_wrong_value_shape_raises(self):
        cache = self._make_cache()
        k, v = self._kv()
        bad_v = np.ones((4, 7), dtype=np.float32)  # wrong head_dim
        with pytest.raises(ValueError, match="value shape"):
            cache.add_kv(k, bad_v)


# ── Sink vs recent routing ────────────────────────────────────────────────────


class TestSinkVsRecentRouting:
    def _make_cache(self, n_sink=2, window=4, n_heads=2, head_dim=4):
        cfg = SinkConfig(n_sink_tokens=n_sink, window_size=window)
        return SinkKVCache(cfg, n_heads=n_heads, head_dim=head_dim)

    def _rand_kv(self, rng, n_heads=2, head_dim=4):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        return k, v

    def test_first_tokens_go_to_sink(self):
        cache = self._make_cache(n_sink=2, window=4)
        rng = np.random.default_rng(1)
        for _ in range(2):
            cache.add_kv(*self._rand_kv(rng))
        assert cache.n_sink == 2
        assert cache.n_recent == 0

    def test_tokens_past_sink_go_to_recent(self):
        cache = self._make_cache(n_sink=2, window=4)
        rng = np.random.default_rng(2)
        for _ in range(5):
            cache.add_kv(*self._rand_kv(rng))
        assert cache.n_sink == 2
        assert cache.n_recent == 3   # tokens 3,4,5

    def test_zero_sink_all_go_to_recent(self):
        cache = self._make_cache(n_sink=0, window=4)
        rng = np.random.default_rng(3)
        for _ in range(3):
            cache.add_kv(*self._rand_kv(rng))
        assert cache.n_sink == 0
        assert cache.n_recent == 3


# ── get_kv shapes and content ─────────────────────────────────────────────────


class TestGetKV:
    def _make_cache(self, n_sink=2, window=4, n_heads=2, head_dim=4):
        cfg = SinkConfig(n_sink_tokens=n_sink, window_size=window)
        return SinkKVCache(cfg, n_heads=n_heads, head_dim=head_dim)

    def _rand_kv(self, rng, n_heads=2, head_dim=4):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        return k, v

    def test_empty_cache_returns_zero_length(self):
        cache = self._make_cache()
        keys, vals = cache.get_kv()
        assert keys.shape == (0, 2, 4)
        assert vals.shape == (0, 2, 4)

    def test_sink_only_shape(self):
        cache = self._make_cache(n_sink=2)
        rng = np.random.default_rng(4)
        cache.add_kv(*self._rand_kv(rng))
        keys, vals = cache.get_kv()
        assert keys.shape == (1, 2, 4)

    def test_sink_plus_recent_shape(self):
        cache = self._make_cache(n_sink=2)
        rng = np.random.default_rng(5)
        for _ in range(5):
            cache.add_kv(*self._rand_kv(rng))
        # sink=2, recent=3
        keys, vals = cache.get_kv()
        assert keys.shape == (5, 2, 4)
        assert vals.shape == (5, 2, 4)

    def test_sink_tokens_preserved_exactly(self):
        """First n_sink_tokens must appear at the front of get_kv output."""
        cache = self._make_cache(n_sink=2)
        rng = np.random.default_rng(6)
        first_k, first_v = self._rand_kv(rng)
        second_k, second_v = self._rand_kv(rng)
        cache.add_kv(first_k, first_v)
        cache.add_kv(second_k, second_v)
        # Add more tokens to fill window + trigger evictions
        for _ in range(6):
            cache.add_kv(*self._rand_kv(rng))
        keys, vals = cache.get_kv()
        # Sinks: index 0 and 1 in output
        np.testing.assert_array_equal(keys[0], first_k)
        np.testing.assert_array_equal(keys[1], second_k)
        np.testing.assert_array_equal(vals[0], first_v)
        np.testing.assert_array_equal(vals[1], second_v)


# ── Window eviction behaviour ─────────────────────────────────────────────────


class TestEviction:
    def _make_cache(self, n_sink=2, window=3, n_heads=1, head_dim=2):
        cfg = SinkConfig(n_sink_tokens=n_sink, window_size=window)
        return SinkKVCache(cfg, n_heads=n_heads, head_dim=head_dim)

    def test_window_does_not_exceed_max(self):
        cache = self._make_cache(n_sink=1, window=3)
        rng = np.random.default_rng(7)
        for _ in range(10):
            k = rng.standard_normal((1, 2)).astype(np.float32)
            v = rng.standard_normal((1, 2)).astype(np.float32)
            cache.add_kv(k, v)
        assert cache.n_recent <= 3
        assert cache.n_sink == 1

    def test_eviction_count_increments(self):
        cache = self._make_cache(n_sink=1, window=2)
        rng = np.random.default_rng(8)
        for _ in range(5):
            k = rng.standard_normal((1, 2)).astype(np.float32)
            v = rng.standard_normal((1, 2)).astype(np.float32)
            cache.add_kv(k, v)
        stats = cache.get_stats()
        # tokens 2,3,4 went to recent (window=2); 3rd-onwards evicted oldest
        # evictions = max(0, n_recent_adds - window_size)
        # n_recent_adds = 5 - 1 (sink) = 4; evictions = 4 - 2 = 2
        assert stats.n_evictions == 2
        assert stats.n_tokens_seen == 5

    def test_sink_tokens_never_evicted(self):
        cache = self._make_cache(n_sink=2, window=2)
        rng = np.random.default_rng(9)
        # Add 20 tokens; sink only holds 2
        for _ in range(20):
            k = rng.standard_normal((1, 2)).astype(np.float32)
            v = rng.standard_normal((1, 2)).astype(np.float32)
            cache.add_kv(k, v)
        assert cache.n_sink == 2  # never drops below 2


# ── Reset ─────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_regions(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=4)
        cache = SinkKVCache(cfg, n_heads=2, head_dim=4)
        rng = np.random.default_rng(10)
        for _ in range(6):
            k = rng.standard_normal((2, 4)).astype(np.float32)
            v = rng.standard_normal((2, 4)).astype(np.float32)
            cache.add_kv(k, v)
        assert cache.total_tokens > 0
        cache.reset()
        assert cache.n_sink == 0
        assert cache.n_recent == 0
        assert cache.total_tokens == 0
        stats = cache.get_stats()
        assert stats.n_tokens_seen == 0
        assert stats.n_evictions == 0

    def test_cache_usable_after_reset(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=4)
        cache = SinkKVCache(cfg, n_heads=2, head_dim=4)
        rng = np.random.default_rng(11)
        for _ in range(3):
            k = rng.standard_normal((2, 4)).astype(np.float32)
            v = rng.standard_normal((2, 4)).astype(np.float32)
            cache.add_kv(k, v)
        cache.reset()
        k = rng.standard_normal((2, 4)).astype(np.float32)
        v = rng.standard_normal((2, 4)).astype(np.float32)
        cache.add_kv(k, v)
        keys, _ = cache.get_kv()
        assert keys.shape == (1, 2, 4)


# ── SinkStats ────────────────────────────────────────────────────────────────


class TestSinkStats:
    def test_util_fraction_zero_tokens_seen(self):
        stats = SinkStats(n_tokens_seen=0, n_evictions=0)
        # util_fraction uses n_tokens_seen as both numerator and denominator → 1.0
        # for n_tokens_seen=0, the max()=1 path → 0/1 = 0.0
        assert stats.util_fraction == 0.0

    def test_util_fraction_nonzero(self):
        stats = SinkStats(n_tokens_seen=5, n_evictions=0)
        assert stats.util_fraction == 1.0  # clamped to 1.0

    def test_total_tokens_held_no_evictions(self):
        stats = SinkStats(n_tokens_seen=10, n_evictions=0)
        assert stats.total_tokens_held == 10

    def test_total_tokens_held_with_evictions(self):
        stats = SinkStats(n_tokens_seen=10, n_evictions=3)
        assert stats.total_tokens_held == 7


# ── dtype propagation ────────────────────────────────────────────────────────


class TestDtypePropagation:
    def test_float16_stored_as_float16(self):
        cfg = SinkConfig(n_sink_tokens=1, window_size=4, dtype="float16")
        cache = SinkKVCache(cfg, n_heads=2, head_dim=4)
        k = np.ones((2, 4), dtype=np.float32)  # pass float32, expect cast
        v = np.ones((2, 4), dtype=np.float32)
        cache.add_kv(k, v)
        keys, _ = cache.get_kv()
        assert keys.dtype == np.float16


# ── __repr__ smoke ────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_key_fields(self):
        cfg = SinkConfig(n_sink_tokens=4, window_size=16)
        cache = SinkKVCache(cfg, n_heads=8, head_dim=64)
        r = repr(cache)
        assert "n_sink=" in r
        assert "n_recent=" in r
        assert "evictions=" in r
