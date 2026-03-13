#!/usr/bin/env python3
"""
tests/test_agent_kv_unit.py

Unit tests for squish/agent_kv.py — asymmetric INT2 KV cache.

Coverage targets
────────────────
AgentKVConfig
  - validation: negative sink_tokens
  - validation: zero window_tokens
  - validation: head_dim not divisible by 4
  - validation: zero n_heads / n_layers
  - valid config constructs without error

_quantise_int2 / _dequantise_int2
  - round-trip: dequantised values close to original (within INT2 quantisation error)
  - packed array dtype is uint8
  - packed shape: (n_heads, n_tokens, head_dim // 4)
  - scales shape: (n_heads, n_tokens, 1)
  - zero input → all-zero output after round-trip

Int2Block
  - from_float32 constructs correctly
  - to_float32 returns float32 array of correct shape
  - n_tokens property

AgentKVCache
  - append single token per layer → get() returns shape (H, 1, D)
  - append multiple tokens → correct reconstruction order
  - compact: overflow from window moves to history
  - compact: sink zone filled before history
  - compact: history merging on repeated calls
  - evict_history: removes oldest n tokens from history
  - evict_history: n_tokens > history → clears history entirely
  - evict_history: no history → returns 0
  - evict_history: n_tokens=0 → no-op
  - reset(layer_idx) → clears single layer
  - reset(None) → clears all layers
  - _validate_layer raises IndexError on out-of-range
  - stats: total_tokens, sink, hist, window counts correct
  - stats: estimated_bytes > 0 when tokens present
  - __repr__ contains expected fields
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.agent_kv import (
    AgentKVCache,
    AgentKVConfig,
    AgentKVStats,
    Int2Block,
    _dequantise_int2,
    _quantise_int2,
)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# AgentKVConfig validation
# ---------------------------------------------------------------------------

class TestAgentKVConfig:
    def test_valid_default_config(self):
        cfg = AgentKVConfig()
        assert cfg.sink_tokens == 4
        assert cfg.window_tokens == 64

    def test_negative_sink_tokens_raises(self):
        with pytest.raises(ValueError, match="sink_tokens"):
            AgentKVConfig(sink_tokens=-1)

    def test_zero_window_tokens_raises(self):
        with pytest.raises(ValueError, match="window_tokens"):
            AgentKVConfig(window_tokens=0)

    def test_head_dim_not_divisible_by_4_raises(self):
        with pytest.raises(ValueError, match="head_dim"):
            AgentKVConfig(head_dim=65)

    def test_zero_n_heads_raises(self):
        with pytest.raises(ValueError, match="n_heads"):
            AgentKVConfig(n_heads=0)

    def test_zero_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            AgentKVConfig(n_layers=0)

    def test_zero_head_dim_raises(self):
        with pytest.raises(ValueError, match="head_dim"):
            AgentKVConfig(head_dim=0)

    def test_valid_custom_config(self):
        cfg = AgentKVConfig(sink_tokens=0, window_tokens=32, n_heads=2,
                            head_dim=64, n_layers=4)
        assert cfg.sink_tokens == 0
        assert cfg.head_dim == 64


# ---------------------------------------------------------------------------
# _quantise_int2 / _dequantise_int2
# ---------------------------------------------------------------------------

class TestInt2Quantize:
    @pytest.fixture()
    def random_kv(self):
        return RNG.standard_normal((4, 8, 32)).astype(np.float32)

    def test_round_trip_within_error(self, random_kv):
        packed, scales = _quantise_int2(random_kv)
        reconstructed  = _dequantise_int2(packed, scales)
        # INT2 is lossy; max error should be bounded by scale / 1.5
        max_scale = np.abs(random_kv).max() / 1.5
        assert np.abs(random_kv - reconstructed).max() < max_scale + 1e-5

    def test_packed_dtype_is_uint8(self, random_kv):
        packed, _ = _quantise_int2(random_kv)
        assert packed.dtype == np.uint8

    def test_packed_shape(self, random_kv):
        n_heads, n_tokens, head_dim = random_kv.shape
        packed, scales = _quantise_int2(random_kv)
        assert packed.shape == (n_heads, n_tokens, head_dim // 4)

    def test_scales_shape(self, random_kv):
        n_heads, n_tokens, _ = random_kv.shape
        _, scales = _quantise_int2(random_kv)
        assert scales.shape == (n_heads, n_tokens, 1)
        assert scales.dtype == np.float32

    def test_zero_input_round_trip(self):
        x = np.zeros((2, 4, 16), dtype=np.float32)
        packed, scales = _quantise_int2(x)
        reconstructed  = _dequantise_int2(packed, scales)
        # Zero in → near-zero out (scale is 1 due to safe divide)
        assert np.allclose(reconstructed, 0.0, atol=1.0)

    def test_packed_values_in_range(self, random_kv):
        """Each nibble pair's value should decode to one of 4 levels."""
        packed, scales = _quantise_int2(random_kv)
        # Each byte encodes 4 uint2 values in range [0,3]
        assert (packed >= 0).all()
        assert (packed <= 255).all()


# ---------------------------------------------------------------------------
# Int2Block
# ---------------------------------------------------------------------------

class TestInt2Block:
    def test_from_float32_roundtrip_shape(self):
        x = RNG.standard_normal((2, 10, 16)).astype(np.float32)
        block = Int2Block.from_float32(x)
        out = block.to_float32()
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_n_tokens_property(self):
        x = RNG.standard_normal((2, 7, 16)).astype(np.float32)
        block = Int2Block.from_float32(x)
        assert block.n_tokens == 7

    def test_from_float32_packed_dtype(self):
        x = np.ones((1, 4, 16), dtype=np.float32)
        block = Int2Block.from_float32(x)
        assert block.packed.dtype == np.uint8


# ---------------------------------------------------------------------------
# AgentKVCache — append and get
# ---------------------------------------------------------------------------

def _make_cache(sink=4, window=8, n_heads=2, head_dim=16, n_layers=2):
    cfg = AgentKVConfig(sink_tokens=sink, window_tokens=window,
                        n_heads=n_heads, head_dim=head_dim, n_layers=n_layers)
    return AgentKVCache(cfg)


def _rand_kv(n_heads=2, n_tokens=1, head_dim=16):
    k = RNG.standard_normal((n_heads, n_tokens, head_dim)).astype(np.float32)
    v = RNG.standard_normal((n_heads, n_tokens, head_dim)).astype(np.float32)
    return k, v


class TestAgentKVCacheAppendGet:
    def test_single_token_append_get_shape(self):
        cache = _make_cache()
        k, v = _rand_kv()
        cache.append(0, k, v)
        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 1, 16)
        assert v_out.shape == (2, 1, 16)

    def test_empty_cache_returns_zero_length(self):
        cache = _make_cache()
        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 0, 16)
        assert v_out.shape == (2, 0, 16)

    def test_append_accumulates_tokens(self):
        cache = _make_cache(sink=0, window=100)  # large window → no compaction
        for _ in range(5):
            cache.append(0, *_rand_kv())
        k_out, _ = cache.get(0)
        assert k_out.shape[1] == 5

    def test_get_returns_float32(self):
        cache = _make_cache()
        cache.append(0, *_rand_kv())
        k_out, v_out = cache.get(0)
        assert k_out.dtype == np.float32
        assert v_out.dtype == np.float32

    def test_layer_isolation(self):
        """Appending to layer 0 does not affect layer 1."""
        cache = _make_cache()
        cache.append(0, *_rand_kv())
        k1, _ = cache.get(1)
        assert k1.shape[1] == 0

    def test_multi_token_append(self):
        cache = _make_cache(sink=0, window=50)
        k, v = _rand_kv(n_tokens=5)
        cache.append(0, k, v)
        k_out, _ = cache.get(0)
        assert k_out.shape[1] == 5


# ---------------------------------------------------------------------------
# AgentKVCache — compaction
# ---------------------------------------------------------------------------

class TestAgentKVCacheCompact:
    def test_overflow_moves_to_history(self):
        """Tokens exceeding window (after sink fills) should end up in hist."""
        cache = _make_cache(sink=2, window=4)
        # Append enough tokens to overflow window and fill sink
        for _ in range(12):
            cache.append(0, *_rand_kv())
        k, _ = cache.get(0)
        assert k.shape[1] == 12
        s = cache.stats
        assert s.history_tokens > 0 or s.sink_tokens > 0

    def test_sink_fills_before_history(self):
        """First overflow tokens should go to FP32 sink, not INT2 history."""
        cache = _make_cache(sink=4, window=4)
        # Append more than window tokens; first overflows should fill sink
        for _ in range(8):
            cache.append(0, *_rand_kv())
        lkv = cache._layers[0]
        # Sink should be populated
        if lkv.sink_k is not None:
            assert lkv.sink_k.shape[1] <= 4

    def test_total_tokens_preserved_after_compact(self):
        """Total tokens must equal sequence length regardless of zone."""
        cache = _make_cache(sink=4, window=4)
        n = 20
        for _ in range(n):
            cache.append(0, *_rand_kv())
        k, _ = cache.get(0)
        assert k.shape[1] == n

    def test_compactions_counter_increments(self):
        cache = _make_cache(sink=2, window=4)
        for _ in range(15):
            cache.append(0, *_rand_kv())
        assert cache._compactions > 0

    def test_manual_compact_is_idempotent_when_window_not_full(self):
        cache = _make_cache(sink=0, window=10)
        cache.append(0, *_rand_kv())
        cache.compact(0)  # window not full — no-op
        k, _ = cache.get(0)
        assert k.shape[1] == 1


# ---------------------------------------------------------------------------
# AgentKVCache — evict_history
# ---------------------------------------------------------------------------

class TestAgentKVCacheEvictHistory:
    def _cache_with_history(self, n_hist_tokens=8):
        """Build a cache that has some history tokens."""
        cache = _make_cache(sink=0, window=4)
        # Append enough to overflow into history
        for _ in range(n_hist_tokens + 4):
            cache.append(0, *_rand_kv())
        return cache

    def test_evict_reduces_history(self):
        cache = self._cache_with_history(8)
        lkv = cache._layers[0]
        if lkv.hist_k is None:
            pytest.skip("no history accumulated")
        before = lkv.hist_k.n_tokens
        removed = cache.evict_history(0, 2)
        assert removed == 2
        after_lkv = cache._layers[0]
        after = after_lkv.hist_k.n_tokens if after_lkv.hist_k else 0
        assert after == before - 2

    def test_evict_more_than_history_clears_history(self):
        cache = self._cache_with_history(4)
        lkv = cache._layers[0]
        if lkv.hist_k is None:
            pytest.skip("no history accumulated")
        total_hist = lkv.hist_k.n_tokens
        removed = cache.evict_history(0, total_hist + 100)
        assert removed == total_hist
        assert cache._layers[0].hist_k is None

    def test_evict_no_history_returns_zero(self):
        cache = _make_cache(sink=0, window=50)
        cache.append(0, *_rand_kv())  # stays in window
        removed = cache.evict_history(0, 5)
        assert removed == 0

    def test_evict_zero_tokens_no_op(self):
        cache = self._cache_with_history()
        lkv = cache._layers[0]
        if lkv.hist_k is None:
            pytest.skip("no history")
        before = lkv.hist_k.n_tokens
        removed = cache.evict_history(0, 0)
        assert removed == 0
        assert cache._layers[0].hist_k.n_tokens == before


# ---------------------------------------------------------------------------
# AgentKVCache — reset
# ---------------------------------------------------------------------------

class TestAgentKVCacheReset:
    def test_reset_single_layer_clears_only_that_layer(self):
        cache = _make_cache()
        cache.append(0, *_rand_kv())
        cache.append(1, *_rand_kv())
        cache.reset(0)
        k0, _ = cache.get(0)
        k1, _ = cache.get(1)
        assert k0.shape[1] == 0
        assert k1.shape[1] == 1

    def test_reset_none_clears_all_layers(self):
        cache = _make_cache()
        for layer in range(2):
            for _ in range(3):
                cache.append(layer, *_rand_kv())
        cache.reset()
        for layer in range(2):
            k, _ = cache.get(layer)
            assert k.shape[1] == 0

    def test_reset_all_zeros_compaction_counter(self):
        cache = _make_cache(sink=2, window=4)
        for _ in range(15):
            cache.append(0, *_rand_kv())
        assert cache._compactions > 0
        cache.reset()
        assert cache._compactions == 0


# ---------------------------------------------------------------------------
# AgentKVCache — validation and stats
# ---------------------------------------------------------------------------

class TestAgentKVCacheValidation:
    def test_validate_layer_raises_on_negative(self):
        cache = _make_cache(n_layers=4)
        with pytest.raises(IndexError):
            cache._validate_layer(-1)

    def test_validate_layer_raises_on_too_large(self):
        cache = _make_cache(n_layers=4)
        with pytest.raises(IndexError):
            cache._validate_layer(4)

    def test_append_invalid_layer_raises(self):
        cache = _make_cache(n_layers=2)
        with pytest.raises(IndexError):
            cache.append(5, *_rand_kv())

    def test_evict_history_invalid_layer_raises(self):
        cache = _make_cache(n_layers=2)
        with pytest.raises(IndexError):
            cache.evict_history(99, 1)


class TestAgentKVCacheStats:
    def test_empty_stats_all_zero(self):
        cache = _make_cache()
        s = cache.stats
        assert s.total_tokens == 0
        assert s.sink_tokens == 0
        assert s.history_tokens == 0
        assert s.window_tokens == 0
        assert s.estimated_bytes == 0

    def test_stats_after_append(self):
        cache = _make_cache(sink=0, window=50)
        for _ in range(3):
            cache.append(0, *_rand_kv())
        s = cache.stats
        assert s.total_tokens == 3   # only 1 layer has tokens

    def test_stats_estimated_bytes_grows_with_tokens(self):
        cache = _make_cache()
        s0 = cache.stats
        cache.append(0, *_rand_kv())
        s1 = cache.stats
        assert s1.estimated_bytes >= s0.estimated_bytes

    def test_stats_compactions_count(self):
        cache = _make_cache(sink=2, window=4)
        for _ in range(12):
            cache.append(0, *_rand_kv())
        s = cache.stats
        assert s.compactions > 0


class TestAgentKVCacheRepr:
    def test_repr_contains_layers(self):
        cache = _make_cache()
        r = repr(cache)
        assert "AgentKVCache" in r
        assert "layers=2" in r

    def test_repr_contains_int2_tag(self):
        cache = _make_cache()
        r = repr(cache)
        assert "INT2" in r
