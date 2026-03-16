"""
tests/kv/test_fast_weights.py

Unit tests for Phase 5: TTT Fast Weights (squish/kv/fast_weights.py).

Covers:
  - FastWeightConfig validation
  - FastWeightLayer lazy initialization
  - FastWeightLayer absorb — outer product update, decay application
  - FastWeightLayer query — correct math, zero before init, shape
  - FastWeightLayer decay_step — standalone decay without absorb
  - FastWeightLayer reset — zeros W_f and n_absorptions
  - FastWeightLayer.weight_norm
  - FastWeightLayer 2D (flat) key/value absorb
  - FastWeightManager per-layer routing
  - FastWeightManager reset and stats
  - QuantizedKVCache integration (fast_weight_lr > 0 creates manager)
  - kv_cache.reset() resets fast weights
  - kv_cache.stats() includes fast_weight keys when enabled
  - tick_qfilter absorbs evicted KVs into fast weights (integration test)
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.fast_weights import FastWeightConfig, FastWeightLayer, FastWeightManager

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

N_HEADS  = 4
HEAD_DIM = 8
N_LAYERS = 3
RNG      = np.random.default_rng(77)


def _rand_kv(n_heads: int = N_HEADS, n_tokens: int = 1, head_dim: int = HEAD_DIM):
    k = RNG.standard_normal((n_heads, n_tokens, head_dim)).astype(np.float32)
    v = RNG.standard_normal((n_heads, n_tokens, head_dim)).astype(np.float32)
    return k, v


def _rand_q(n_heads: int = N_HEADS, head_dim: int = HEAD_DIM):
    return RNG.standard_normal((n_heads, head_dim)).astype(np.float32)


def _make_layer(lr: float = 0.01, decay: float = 0.999) -> FastWeightLayer:
    return FastWeightLayer(FastWeightConfig(lr=lr, decay=decay))


def _make_manager(n_layers: int = N_LAYERS, lr: float = 0.01, decay: float = 0.99) -> FastWeightManager:
    return FastWeightManager(FastWeightConfig(lr=lr, decay=decay), n_layers=n_layers)


# ---------------------------------------------------------------------------
# FastWeightConfig
# ---------------------------------------------------------------------------

class TestFastWeightConfig:
    def test_defaults(self):
        cfg = FastWeightConfig()
        assert cfg.lr    == 0.01
        assert cfg.decay == 0.999

    def test_custom(self):
        cfg = FastWeightConfig(lr=0.1, decay=0.95)
        assert cfg.lr    == 0.1
        assert cfg.decay == 0.95

    def test_lr_zero_raises(self):
        with pytest.raises(ValueError, match="lr"):
            FastWeightConfig(lr=0.0)

    def test_lr_negative_raises(self):
        with pytest.raises(ValueError, match="lr"):
            FastWeightConfig(lr=-0.1)

    def test_decay_negative_raises(self):
        with pytest.raises(ValueError, match="decay"):
            FastWeightConfig(decay=-0.1)

    def test_decay_above_one_raises(self):
        with pytest.raises(ValueError, match="decay"):
            FastWeightConfig(decay=1.001)

    def test_decay_zero_valid(self):
        cfg = FastWeightConfig(decay=0.0)
        assert cfg.decay == 0.0

    def test_decay_one_valid(self):
        cfg = FastWeightConfig(decay=1.0)
        assert cfg.decay == 1.0


# ---------------------------------------------------------------------------
# FastWeightLayer — initialization
# ---------------------------------------------------------------------------

class TestFastWeightLayerInit:
    def test_not_initialized_before_absorb(self):
        layer = _make_layer()
        assert not layer.is_initialized
        assert layer.n_absorptions == 0

    def test_initialized_after_first_absorb(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        assert layer.is_initialized

    def test_weight_shape_correct_after_init(self):
        layer = _make_layer()
        k, v  = _rand_kv(n_heads=N_HEADS, n_tokens=2, head_dim=HEAD_DIM)
        layer.absorb(k, v)
        assert layer._W_f.shape == (N_HEADS, HEAD_DIM, HEAD_DIM)

    def test_weight_dtype_float32(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        assert layer._W_f.dtype == np.float32


# ---------------------------------------------------------------------------
# FastWeightLayer — absorb
# ---------------------------------------------------------------------------

class TestFastWeightLayerAbsorb:
    def test_absorb_changes_weight(self):
        layer = _make_layer(decay=1.0)  # no decay so we can isolate update
        k, v  = _rand_kv()
        layer.absorb(k, v)
        # W_f should be non-zero after absorbing non-zero KV
        assert layer.weight_norm > 0.0

    def test_n_absorptions_increments(self):
        layer   = _make_layer()
        k1, v1  = _rand_kv(n_tokens=3)
        k2, v2  = _rand_kv(n_tokens=2)
        layer.absorb(k1, v1)
        assert layer.n_absorptions == 3
        layer.absorb(k2, v2)
        assert layer.n_absorptions == 5

    def test_absorb_2d_input(self):
        """Pass (n_heads, head_dim) instead of (n_heads, 1, head_dim)."""
        layer = _make_layer(decay=1.0)
        k = RNG.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32)
        v = RNG.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32)
        layer.absorb(k, v)
        assert layer.n_absorptions == 1
        assert layer._W_f.shape == (N_HEADS, HEAD_DIM, HEAD_DIM)

    def test_decay_applied_before_update(self):
        """With decay=0.0, the second absorb should overwrite the first."""
        layer  = _make_layer(lr=1.0, decay=0.0)
        k1, v1 = _rand_kv()
        k2, v2 = _rand_kv()
        layer.absorb(k1, v1)
        w_after_first = layer._W_f.copy()
        layer.absorb(k2, v2)
        w_after_second = layer._W_f.copy()
        # After second absorb with decay=0: W_f = 0 + lr * outer(v2, k2)
        # So W_f should be entirely determined by k2, v2 (not k1, v1)
        expected = np.einsum("hte,htd->hed", v2, k2, dtype=np.float32)
        assert np.allclose(w_after_second, expected, atol=1e-5), (
            "With decay=0.0, second absorb should overwrite first"
        )

    def test_no_decay_accumulates(self):
        """With decay=1.0, absorbs should accumulate indefinitely."""
        layer = _make_layer(lr=1.0, decay=1.0)
        k, v  = np.ones((N_HEADS, 1, HEAD_DIM), dtype=np.float32), \
                np.ones((N_HEADS, 1, HEAD_DIM), dtype=np.float32)
        layer.absorb(k, v)
        n1 = layer.weight_norm
        layer.absorb(k, v)
        n2 = layer.weight_norm
        assert n2 > n1, "With decay=1.0 and same inputs, norm should grow"

    def test_outer_product_math(self):
        """
        Verify the outer-product update formula exactly.
        W_f_new = decay * W_f + lr * einsum("hte,htd->hed", v, k)
        Using decay=1.0, lr=1.0, 1 head, 1 token.
        """
        layer = FastWeightLayer(FastWeightConfig(lr=1.0, decay=1.0))
        k = np.array([[[1.0, 2.0]]], dtype=np.float32)  # (1, 1, 2)
        v = np.array([[[3.0, 4.0]]], dtype=np.float32)  # (1, 1, 2)
        layer.absorb(k, v)
        # Expected: outer(v[0,0], k[0,0]) = [[3*1, 3*2], [4*1, 4*2]]
        expected = np.array([[[3.0, 6.0], [4.0, 8.0]]], dtype=np.float32)
        assert np.allclose(layer._W_f, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# FastWeightLayer — query
# ---------------------------------------------------------------------------

class TestFastWeightLayerQuery:
    def test_query_before_init_returns_zeros(self):
        layer = _make_layer()
        q     = _rand_q()
        out   = layer.query(q)
        assert out.shape == (N_HEADS, HEAD_DIM)
        assert np.allclose(out, 0.0)

    def test_query_shape_correct(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        out   = layer.query(_rand_q())
        assert out.shape == (N_HEADS, HEAD_DIM)

    def test_query_dtype_float32(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        out   = layer.query(_rand_q())
        assert out.dtype == np.float32

    def test_query_math(self):
        """
        Verify W_f @ q per head with lr=1.0, decay=1.0, 1 head.
        """
        layer = FastWeightLayer(FastWeightConfig(lr=1.0, decay=1.0))
        k = np.array([[[1.0, 0.0]]], dtype=np.float32)  # (1, 1, 2)
        v = np.array([[[0.0, 1.0]]], dtype=np.float32)
        layer.absorb(k, v)
        # W_f = [[0*1, 0*0], [1*1, 1*0]] = [[0, 0], [1, 0]]
        q     = np.array([[1.0, 1.0]], dtype=np.float32)  # (1, 2)
        out   = layer.query(q)
        # out = W_f @ q = [[0, 0], [1, 0]] @ [1, 1] = [0, 1]
        expected = np.array([[0.0, 1.0]], dtype=np.float32)
        assert np.allclose(out, expected, atol=1e-6)

    def test_query_non_zero_after_absorb(self):
        layer = _make_layer(lr=0.1, decay=1.0)
        k, v  = _rand_kv(n_tokens=4)
        layer.absorb(k, v)
        q   = _rand_q()
        out = layer.query(q)
        # With random k/v and random q, result should generally be non-zero
        assert not np.allclose(out, 0.0)


# ---------------------------------------------------------------------------
# FastWeightLayer — decay_step and reset
# ---------------------------------------------------------------------------

class TestFastWeightLayerDecayAndReset:
    def test_decay_step_reduces_norm(self):
        layer = _make_layer(lr=1.0, decay=0.5)
        k, v  = _rand_kv()
        layer.absorb(k, v)
        norm_before = layer.weight_norm
        layer.decay_step()
        norm_after  = layer.weight_norm
        assert norm_after < norm_before

    def test_decay_step_before_init_noop(self):
        layer = _make_layer()
        layer.decay_step()  # Should not raise
        assert not layer.is_initialized

    def test_reset_zeros_weight(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        layer.reset()
        assert np.allclose(layer._W_f, 0.0)

    def test_reset_clears_absorption_count(self):
        layer = _make_layer()
        k, v  = _rand_kv(n_tokens=5)
        layer.absorb(k, v)
        assert layer.n_absorptions == 5
        layer.reset()
        assert layer.n_absorptions == 0

    def test_reset_preserves_shape(self):
        layer = _make_layer()
        k, v  = _rand_kv()
        layer.absorb(k, v)
        shape = layer._W_f.shape
        layer.reset()
        assert layer._W_f.shape == shape

    def test_weight_norm_zero_before_init(self):
        layer = _make_layer()
        assert layer.weight_norm == 0.0


# ---------------------------------------------------------------------------
# FastWeightManager
# ---------------------------------------------------------------------------

class TestFastWeightManager:
    def test_absorb_routes_to_correct_layer(self):
        mgr   = _make_manager(n_layers=3)
        k, v  = _rand_kv()
        mgr.absorb_layer(1, k, v)
        # Layer 1 should be initialized; layers 0 and 2 should not
        assert mgr._layers[1].is_initialized
        assert not mgr._layers[0].is_initialized
        assert not mgr._layers[2].is_initialized

    def test_query_routes_to_correct_layer(self):
        mgr   = _make_manager(n_layers=3)
        k, v  = _rand_kv()
        mgr.absorb_layer(0, k, v)
        q     = _rand_q()
        out0  = mgr.query_layer(0, q)
        out1  = mgr.query_layer(1, q)  # layer 1 not initialized → zeros
        assert not np.allclose(out0, 0.0)
        assert np.allclose(out1, 0.0)

    def test_reset_clears_all_layers(self):
        mgr   = _make_manager(n_layers=2)
        k, v  = _rand_kv()
        mgr.absorb_layer(0, k, v)
        mgr.absorb_layer(1, k, v)
        mgr.reset()
        assert mgr._layers[0].n_absorptions == 0
        assert mgr._layers[1].n_absorptions == 0

    def test_stats_returns_expected_keys(self):
        mgr   = _make_manager(n_layers=2)
        stats = mgr.stats()
        assert "fast_weight_lr"            in stats
        assert "fast_weight_decay"         in stats
        assert "fast_weight_absorptions"   in stats
        assert "fast_weight_total_absorbed" in stats
        assert "fast_weight_norms"         in stats

    def test_stats_total_absorbed_sums_layers(self):
        mgr   = _make_manager(n_layers=3)
        k, v  = _rand_kv(n_tokens=3)
        mgr.absorb_layer(0, k, v)
        mgr.absorb_layer(2, k, v)
        stats = mgr.stats()
        assert stats["fast_weight_total_absorbed"] == 6  # 3 + 3

    def test_decay_all_reduces_all_norms(self):
        mgr = _make_manager(n_layers=2, lr=1.0, decay=0.5)
        k, v = _rand_kv()
        mgr.absorb_layer(0, k, v)
        mgr.absorb_layer(1, k, v)
        norms_before = [layer.weight_norm for layer in mgr._layers]
        mgr.decay_all()
        norms_after  = [layer.weight_norm for layer in mgr._layers]
        assert all(na < nb for na, nb in zip(norms_after, norms_before))


# ---------------------------------------------------------------------------
# QuantizedKVCache integration
# ---------------------------------------------------------------------------

class TestQuantizedKVCacheFastWeightIntegration:
    """Integration tests using QuantizedKVCache with fast_weight_lr > 0."""

    def _make_cache(
        self,
        n_layers: int = 2,
        fast_weight_lr: float = 0.01,
        fast_weight_decay: float = 0.99,
        qfilter_rank: int = 0,
    ):
        from squish.kv.kv_cache import QuantizedKVCache
        return QuantizedKVCache(
            n_layers          = n_layers,
            mode              = "int8",
            window            = 8,
            fast_weight_lr    = fast_weight_lr,
            fast_weight_decay = fast_weight_decay,
            qfilter_rank      = qfilter_rank,
            qfilter_budget    = 10 if qfilter_rank > 0 else 4096,
        )

    def test_fw_manager_created_when_lr_positive(self):
        cache = self._make_cache(fast_weight_lr=0.01)
        assert cache._fw_manager is not None

    def test_fw_manager_not_created_when_lr_zero(self):
        cache = self._make_cache(fast_weight_lr=0.0)
        assert cache._fw_manager is None

    def test_reset_resets_fw_manager(self):
        cache = self._make_cache(fast_weight_lr=0.01)
        # Manually absorb something
        k, v = _rand_kv(n_heads=4, n_tokens=1, head_dim=8)
        cache._fw_manager.absorb_layer(0, k, v)
        assert cache._fw_manager._layers[0].n_absorptions > 0
        cache.reset()
        assert cache._fw_manager._layers[0].n_absorptions == 0

    def test_stats_includes_fast_weight_keys(self):
        cache = self._make_cache(fast_weight_lr=0.01)
        stats = cache.stats()
        assert "fast_weight_lr"       in stats
        assert "fast_weight_absorbed" in stats

    def test_stats_no_fast_weight_keys_when_disabled(self):
        cache = self._make_cache(fast_weight_lr=0.0)
        stats = cache.stats()
        assert "fast_weight_lr"       not in stats
        assert "fast_weight_absorbed" not in stats

    def test_fw_config_stored_correctly(self):
        cache = self._make_cache(fast_weight_lr=0.05, fast_weight_decay=0.98)
        assert abs(cache._fw_manager.config.lr    - 0.05) < 1e-7
        assert abs(cache._fw_manager.config.decay - 0.98) < 1e-7


# ---------------------------------------------------------------------------
# tick_qfilter + fast weights absorption integration
# ---------------------------------------------------------------------------

class TestTickQFilterWithFastWeights:
    """
    Verify that tick_qfilter absorbs evicted tokens into fast weights.
    """

    def _feed_tokens(self, cache, n_tokens: int, n_heads: int = 4, head_dim: int = 8):
        """Append n_tokens to cache layer 0 using fake key/value data."""
        rng = np.random.default_rng(0)
        for _ in range(n_tokens):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            cache._layers[0].append(k, v)

    def test_eviction_with_fw_absorbs_tokens(self):
        """After eviction, fast weight manager should have non-zero absorptions."""
        from squish.kv.kv_cache import QuantizedKVCache
        N_H, H_D = 4, 8
        cache = QuantizedKVCache(
            n_layers          = 1,
            mode              = "int8",
            window            = 8,
            qfilter_rank      = 4,
            qfilter_budget    = 10,  # small budget to trigger eviction
            qfilter_anchor    = 4,
            qfilter_evict_every = 1,
            fast_weight_lr    = 0.01,
            fast_weight_decay = 0.99,
        )
        # Feed more tokens than budget AND past calibration threshold (min_tokens=64)
        self._feed_tokens(cache, n_tokens=70, n_heads=N_H, head_dim=H_D)
        # Trigger tick_qfilter
        cache.tick_qfilter(step=1)
        # Fast weight manager should have absorbed some evicted tokens
        total_absorbed = cache._fw_manager.stats()["fast_weight_total_absorbed"]
        assert total_absorbed > 0, (
            f"Expected fast weights to absorb evicted tokens, got {total_absorbed}"
        )

    def test_no_eviction_no_absorption(self):
        """If cache is below budget, no eviction and no fast weight absorption."""
        from squish.kv.kv_cache import QuantizedKVCache
        cache = QuantizedKVCache(
            n_layers          = 1,
            mode              = "int8",
            window            = 8,
            qfilter_rank      = 4,
            qfilter_budget    = 1000,  # large budget — never evicts
            fast_weight_lr    = 0.01,
        )
        self._feed_tokens(cache, n_tokens=5)
        cache.tick_qfilter(step=1)
        total_absorbed = cache._fw_manager.stats()["fast_weight_total_absorbed"]
        assert total_absorbed == 0
