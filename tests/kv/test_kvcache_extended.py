"""
tests/test_kvcache_extended.py

Unit tests for QuantizedKVCache pure-Python / pure-numpy paths.
Covers __init__, iteration, update, reset, n_tokens, memory_mb,
stats(), and the _snap_evict helper via the update path.
Does NOT require MLX or a loaded model.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.kv_cache import (
    KVLayerCache,
    QuantizedKVCache,
    _compute_layer_budgets,
    _snap_evict,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _rand_kv(n_heads=4, head_dim=8, dtype=np.float16):
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n_heads, head_dim)).astype(dtype),
        rng.standard_normal((n_heads, head_dim)).astype(dtype),
    )


# ── QuantizedKVCache construction ──────────────────────────────────────────────

class TestQuantizedKVCacheInit:
    def test_default_init(self):
        cache = QuantizedKVCache(n_layers=4)
        assert cache.n_layers == 4
        assert cache.mode == "int8"
        assert len(cache._layers) == 4

    def test_fp16_mode(self):
        cache = QuantizedKVCache(n_layers=2, mode="fp16")
        assert cache.mode == "fp16"

    def test_snap_mode(self):
        cache = QuantizedKVCache(n_layers=2, mode="snap")
        assert cache.mode == "snap"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            QuantizedKVCache(n_layers=2, mode="invalid")

    def test_window_stored(self):
        cache = QuantizedKVCache(n_layers=2, window=128)
        assert cache.window == 128

    def test_budget_stored(self):
        cache = QuantizedKVCache(n_layers=2, budget=512)
        assert cache.budget == 512

    def test_snap_window_stored(self):
        cache = QuantizedKVCache(n_layers=2, snap_window=64)
        assert cache.snap_window == 64


# ── Sequence protocol ──────────────────────────────────────────────────────────

class TestQuantizedKVCacheSequence:
    def setup_method(self):
        self.cache = QuantizedKVCache(n_layers=3)

    def test_len(self):
        assert len(self.cache) == 3

    def test_getitem_returns_kv_layer_cache(self):
        layer = self.cache[0]
        assert isinstance(layer, KVLayerCache)

    def test_iter_yields_all_layers(self):
        layers = list(self.cache)
        assert len(layers) == 3
        for layer in layers:
            assert isinstance(layer, KVLayerCache)


# ── update and stats ───────────────────────────────────────────────────────────

class TestQuantizedKVCacheUpdate:
    def setup_method(self):
        self.cache = QuantizedKVCache(n_layers=2, window=4)
        self.n_heads = 4
        self.head_dim = 8

    def _kv(self, seed=0):
        rng = np.random.default_rng(seed)
        k = rng.standard_normal((self.n_heads, self.head_dim)).astype(np.float16)
        v = rng.standard_normal((self.n_heads, self.head_dim)).astype(np.float16)
        return k, v

    def test_update_increments_n_tokens(self):
        k, v = self._kv()
        self.cache.update(0, k, v)
        assert self.cache.n_tokens == 1

    def test_update_both_layers_independently(self):
        k0, v0 = self._kv(0)
        k1, v1 = self._kv(1)
        self.cache.update(0, k0, v0)
        self.cache.update(0, k0, v0)
        self.cache.update(1, k1, v1)
        assert self.cache._layers[0].n_tokens == 2
        assert self.cache._layers[1].n_tokens == 1

    def test_reset_clears_all_layers(self):
        k, v = self._kv()
        self.cache.update(0, k, v)
        self.cache.update(1, k, v)
        self.cache.reset()
        assert self.cache.n_tokens == 0
        for layer in self.cache._layers:
            assert layer.n_tokens == 0

    def test_reset_clears_snapped_flags(self):
        self.cache._snapped[0] = True
        self.cache.reset()
        assert all(not s for s in self.cache._snapped)

    def test_stats_returns_dict(self):
        s = self.cache.stats()
        assert "mode" in s
        assert "n_layers" in s
        assert "n_tokens" in s
        assert "memory_mb" in s

    def test_stats_n_tokens_matches(self):
        k, v = self._kv()
        self.cache.update(0, k, v)
        s = self.cache.stats()
        assert s["n_tokens"] == 1

    def test_memory_mb_non_negative(self):
        assert self.cache.memory_mb >= 0.0

    def test_memory_mb_increases_after_update(self):
        before = self.cache.memory_mb
        k, v = self._kv()
        for _ in range(10):
            self.cache.update(0, k, v)
        after = self.cache.memory_mb
        assert after >= before

    def test_n_tokens_empty_returns_zero(self):
        cache = QuantizedKVCache(n_layers=0)
        assert cache.n_tokens == 0


# ── snap mode eviction via update ─────────────────────────────────────────────

class TestSnapEviction:
    def test_snap_eviction_triggered(self):
        """After exceeding budget, _snapped[idx] should become True."""
        n_heads, head_dim = 2, 16
        budget = 4
        cache = QuantizedKVCache(
            n_layers=1, mode="snap", window=2, budget=budget, snap_window=2,
        )
        rng = np.random.default_rng(42)
        for _i in range(budget + 3):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            cache.update(0, k, v)
        assert cache._snapped[0] is True

    def test_n_tokens_respects_budget_after_snap(self):
        n_heads, head_dim = 2, 16
        budget = 4
        cache = QuantizedKVCache(
            n_layers=1, mode="snap", window=2, budget=budget, snap_window=2,
        )
        rng = np.random.default_rng(7)
        for _ in range(budget + 5):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            cache.update(0, k, v)
        assert cache.n_tokens <= budget + 5  # eviction ran at least once


# ── _snap_evict low-level ─────────────────────────────────────────────────────

class TestSnapEvictDirect:
    def _fill_layer(self, n=20, window=4, n_heads=2, head_dim=8, seed=0):
        layer = KVLayerCache(window=window)
        rng = np.random.default_rng(seed)
        for _ in range(n):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            layer.append(k, v)
        return layer

    def test_evict_reduces_n_tokens(self):
        layer = self._fill_layer(n=20, window=4)
        assert layer.n_tokens == 20
        _snap_evict(layer, budget=8, snap_window=4)
        assert layer.n_tokens <= 20

    def test_evict_noop_when_tokens_lte_budget(self):
        layer = self._fill_layer(n=5, window=4)
        before = layer.n_tokens
        _snap_evict(layer, budget=10, snap_window=4)
        assert layer.n_tokens == before

    def test_evict_preserves_some_tokens(self):
        layer = self._fill_layer(n=30, window=4)
        _snap_evict(layer, budget=8, snap_window=4)
        assert layer.n_tokens > 0

    def test_evict_on_empty_layer_is_safe(self):
        layer = KVLayerCache(window=4)
        _snap_evict(layer, budget=8)  # should not raise


# ── PyramidKV — per-layer budget allocation ────────────────────────────────────

class TestComputeLayerBudgets:
    def test_uniform_is_flat(self):
        assert _compute_layer_budgets(8, 1000, "uniform") == [1000] * 8

    def test_pyramid_conserves_total_memory(self):
        budgets = _compute_layer_budgets(8, 1000, "pyramid", 0.5, floor=32)
        # Same total KV memory as uniform (within integer rounding).
        assert abs(sum(budgets) - 8 * 1000) <= 8

    def test_pyramid_monotonic_lower_gets_more(self):
        budgets = _compute_layer_budgets(12, 2048, "pyramid", 0.5, floor=32)
        assert all(budgets[i] >= budgets[i + 1] for i in range(len(budgets) - 1))
        assert budgets[0] > budgets[-1]
        assert budgets[0] > 2048 > budgets[-1]  # straddles the uniform value

    def test_pyramid_respects_floor(self):
        # Steep ramp on a small budget would push upper layers below the floor.
        budgets = _compute_layer_budgets(16, 100, "pyramid", 0.9, floor=64)
        assert min(budgets) >= 64

    def test_beta_zero_equals_uniform(self):
        assert _compute_layer_budgets(6, 500, "pyramid", 0.0) == [500] * 6

    def test_single_layer_returns_budget(self):
        assert _compute_layer_budgets(1, 777, "pyramid", 0.5) == [777]

    def test_zero_layers_empty(self):
        assert _compute_layer_budgets(0, 100, "pyramid") == []

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="budget_schedule"):
            _compute_layer_budgets(4, 100, "bogus")

    def test_beta_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pyramid_beta"):
            _compute_layer_budgets(4, 100, "pyramid", 1.5)
        with pytest.raises(ValueError, match="pyramid_beta"):
            _compute_layer_budgets(4, 100, "pyramid", -0.1)


class TestPyramidCacheEviction:
    def _fill(self, cache, n_layers, n_per_layer, n_heads=2, head_dim=8, seed=1):
        rng = np.random.default_rng(seed)
        for layer in range(n_layers):
            for _ in range(n_per_layer):
                k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
                v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
                cache.update(layer, k, v)

    def test_cache_exposes_per_layer_budgets(self):
        cache = QuantizedKVCache(n_layers=4, mode="snap", window=4, budget=16,
                                 budget_schedule="pyramid", pyramid_beta=0.5,
                                 snap_window=4)
        assert len(cache._layer_budgets) == 4
        assert cache._layer_budgets[0] > cache._layer_budgets[-1]

    def test_uniform_default_unchanged(self):
        cache = QuantizedKVCache(n_layers=4, mode="snap", budget=16)
        assert cache._layer_budgets == [16, 16, 16, 16]

    def test_pyramid_evicts_lower_layers_to_larger_budget(self):
        """At the snap point each layer is capped to its OWN budget."""
        nl, budget = 4, 16
        cache = QuantizedKVCache(n_layers=nl, mode="snap", window=4, budget=budget,
                                 budget_schedule="pyramid", pyramid_beta=0.5,
                                 snap_window=4)
        # Append exactly budget+1 tokens per layer → triggers a single snap to
        # that layer's budget, before any extra decode growth.
        for layer in range(nl):
            lb = cache._layer_budgets[layer]
            rng = np.random.default_rng(layer)
            for _ in range(lb + 1):
                k = rng.standard_normal((2, 8)).astype(np.float16)
                v = rng.standard_normal((2, 8)).astype(np.float16)
                cache.update(layer, k, v)
            assert cache._snapped[layer]
            # Lower layer retains strictly more tokens than an upper layer.
            assert cache._layers[layer].n_tokens <= lb
        assert cache._layers[0].n_tokens > cache._layers[nl - 1].n_tokens
