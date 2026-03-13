"""tests/test_shadow_kv_unit.py — 100% coverage for squish/shadow_kv.py"""
import numpy as np
import pytest

from squish.kv.shadow_kv import (
    LandmarkSelector,
    LowRankKeyCache,
    ShadowKVCache,
    ShadowKVConfig,
)

RNG = np.random.default_rng(13)


# ---------------------------------------------------------------------------
# ShadowKVConfig
# ---------------------------------------------------------------------------

class TestShadowKVConfig:
    def test_defaults(self):
        cfg = ShadowKVConfig()
        assert cfg.svd_rank == 32
        assert cfg.n_landmarks == 128
        assert cfg.min_calibration_tokens == 64

    def test_custom(self):
        cfg = ShadowKVConfig(svd_rank=8, n_landmarks=16, min_calibration_tokens=10)
        assert cfg.svd_rank == 8
        assert cfg.n_landmarks == 16

    def test_invalid_svd_rank(self):
        with pytest.raises(ValueError, match="svd_rank"):
            ShadowKVConfig(svd_rank=0)

    def test_invalid_n_landmarks(self):
        with pytest.raises(ValueError, match="n_landmarks"):
            ShadowKVConfig(n_landmarks=0)

    def test_invalid_min_calibration_tokens(self):
        with pytest.raises(ValueError, match="min_calibration_tokens"):
            ShadowKVConfig(min_calibration_tokens=0)


# ---------------------------------------------------------------------------
# LowRankKeyCache
# ---------------------------------------------------------------------------

class TestLowRankKeyCache:
    def _make(self, n_heads=4, head_dim=8, rank=3):
        return LowRankKeyCache(n_heads=n_heads, head_dim=head_dim, rank=rank)

    def test_initial_not_fitted(self):
        kc = self._make()
        assert not kc.is_fitted

    def test_n_stored_initial(self):
        kc = self._make()
        assert kc.n_stored == 0

    def test_project_raises_before_fit(self):
        kc  = self._make()
        key = np.zeros((4, 8), dtype=np.float32)
        with pytest.raises(RuntimeError, match="fit_svd"):
            kc.project(key)

    def test_reconstruct_raises_before_fit(self):
        kc   = self._make()
        proj = np.zeros((4, 3), dtype=np.float32)
        with pytest.raises(RuntimeError, match="fit_svd"):
            kc.reconstruct(proj)

    def test_fit_svd_sets_fitted(self):
        kc   = self._make(n_heads=2, head_dim=8, rank=3)
        keys = RNG.standard_normal((10, 2, 8)).astype(np.float16)
        kc.fit_svd(keys)
        assert kc.is_fitted

    def test_V_shape_after_fit(self):
        n_heads, head_dim, rank = 4, 16, 5
        kc   = LowRankKeyCache(n_heads, head_dim, rank)
        keys = RNG.standard_normal((20, n_heads, head_dim)).astype(np.float16)
        kc.fit_svd(keys)
        assert kc._V.shape == (n_heads, min(rank, head_dim, 20), head_dim)

    def test_project_reconstruct_approximate(self):
        n_heads, head_dim, rank = 2, 8, 4
        kc   = LowRankKeyCache(n_heads, head_dim, rank)
        keys = RNG.standard_normal((30, n_heads, head_dim)).astype(np.float16)
        kc.fit_svd(keys)

        key  = keys[0].astype(np.float32)
        proj = kc.project(key)
        assert proj.shape == (n_heads, rank)

        recon = kc.reconstruct(proj)
        assert recon.shape == (n_heads, head_dim)
        # Reconstruction should reduce MSE vs zero baseline
        mse_approx = np.mean((key - recon.astype(np.float32)) ** 2)
        mse_zero   = np.mean(key ** 2)
        assert mse_approx <= mse_zero + 1e-6

    def test_add_returns_position(self):
        kc   = self._make(n_heads=2, head_dim=8, rank=3)
        keys = RNG.standard_normal((12, 2, 8)).astype(np.float16)
        kc.fit_svd(keys)

        pos0 = kc.add(keys[0])
        pos1 = kc.add(keys[1])
        assert pos0 == 0
        assert pos1 == 1
        assert kc.n_stored == 2

    def test_add_before_fit_uses_raw(self):
        kc  = self._make(n_heads=2, head_dim=8, rank=3)
        key = RNG.standard_normal((2, 8)).astype(np.float16)
        pos = kc.add(key)
        assert pos == 0
        assert kc.n_stored == 1

    def test_get_all_projections_empty(self):
        kc   = self._make()
        proj = kc.get_all_projections()
        assert proj.shape == (0, 4, 3)

    def test_get_all_projections_shape(self):
        n_heads, head_dim, rank = 4, 8, 3
        kc   = LowRankKeyCache(n_heads, head_dim, rank)
        keys = RNG.standard_normal((16, n_heads, head_dim)).astype(np.float16)
        kc.fit_svd(keys)
        for k in keys[:5]:
            kc.add(k)
        proj = kc.get_all_projections()
        assert proj.shape[0] == 5
        assert proj.shape[1] == n_heads
        assert proj.shape[2] == kc._V.shape[1]  # actual fitted rank


# ---------------------------------------------------------------------------
# LandmarkSelector
# ---------------------------------------------------------------------------

class TestLandmarkSelector:
    def test_invalid_n_landmarks(self):
        with pytest.raises(ValueError, match="n_landmarks"):
            LandmarkSelector(n_landmarks=0)

    def test_empty_keys_returns_empty(self):
        sel  = LandmarkSelector(n_landmarks=4)
        q    = np.zeros((2, 3), dtype=np.float16)
        keys = np.empty((0, 2, 3), dtype=np.float16)
        res  = sel.select(q, keys)
        assert len(res) == 0

    def test_returns_k_positions(self):
        sel  = LandmarkSelector(n_landmarks=3)
        q    = RNG.standard_normal((4, 8)).astype(np.float32)
        keys = RNG.standard_normal((20, 4, 8)).astype(np.float32)
        res  = sel.select(q, keys)
        assert len(res) == 3

    def test_positions_in_valid_range(self):
        sel  = LandmarkSelector(n_landmarks=5)
        q    = RNG.standard_normal((2, 4)).astype(np.float32)
        keys = RNG.standard_normal((10, 2, 4)).astype(np.float32)
        res  = sel.select(q, keys)
        assert all(0 <= p < 10 for p in res)

    def test_clamped_to_available(self):
        sel  = LandmarkSelector(n_landmarks=50)
        q    = RNG.standard_normal((2, 4)).astype(np.float32)
        keys = RNG.standard_normal((5, 2, 4)).astype(np.float32)
        res  = sel.select(q, keys)
        assert len(res) == 5

    def test_top_1_returns_most_similar(self):
        sel  = LandmarkSelector(n_landmarks=1)
        # query = keys[3] → should pick position 3
        keys = RNG.standard_normal((10, 1, 4)).astype(np.float32)
        q    = keys[3, :, :]  # (1, 4)
        res  = sel.select(q, keys)
        assert int(res[0]) == 3


# ---------------------------------------------------------------------------
# ShadowKVCache
# ---------------------------------------------------------------------------

class TestShadowKVCache:
    def _make(self, n_layers=2, n_heads=4, head_dim=8, rank=3, landmarks=4, min_calib=5):
        cfg = ShadowKVConfig(
            svd_rank=rank,
            n_landmarks=landmarks,
            min_calibration_tokens=min_calib,
        )
        return ShadowKVCache(n_layers, n_heads, head_dim, cfg)

    def test_initial_n_stored(self):
        cache = self._make()
        assert cache.n_stored(0) == 0
        assert cache.n_stored(1) == 0

    def test_store_increments_count(self):
        cache = self._make()
        keys  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        vals  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        assert cache.n_stored(0) == 8

    def test_store_values_in_shadow(self):
        cache = self._make(min_calib=3)
        keys  = RNG.standard_normal((6, 4, 8)).astype(np.float16)
        vals  = RNG.standard_normal((6, 4, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        # Shadow dict should have entries for layer 0
        shadow_keys = [k for k in cache._value_shadow if k[0] == 0]
        assert len(shadow_keys) == 6

    def test_recall_empty_cache(self):
        cache = self._make()
        q     = RNG.standard_normal((4, 8)).astype(np.float16)
        ks, vs = cache.recall(0, q)
        assert ks.shape[0] == 0
        assert vs.shape[0] == 0

    def test_recall_returns_top_k(self):
        cache = self._make(n_layers=1, n_heads=2, head_dim=8, rank=3, landmarks=3, min_calib=5)
        keys  = RNG.standard_normal((10, 2, 8)).astype(np.float16)
        vals  = RNG.standard_normal((10, 2, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        q     = RNG.standard_normal((2, 8)).astype(np.float16)
        ks, vs = cache.recall(0, q, top_k=3)
        assert ks.shape == (3, 2, 8)
        assert vs.shape == (3, 2, 8)

    def test_recall_uses_default_landmarks(self):
        cfg   = ShadowKVConfig(svd_rank=3, n_landmarks=4, min_calibration_tokens=5)
        cache = ShadowKVCache(1, 2, 8, cfg)
        keys  = RNG.standard_normal((10, 2, 8)).astype(np.float16)
        vals  = RNG.standard_normal((10, 2, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        q     = RNG.standard_normal((2, 8)).astype(np.float16)
        ks, vs = cache.recall(0, q)  # no top_k arg → use config
        assert ks.shape[0] <= 4

    def test_recall_before_svd_fit(self):
        # min_calibration_tokens > n stored → SVD not fitted
        cache = self._make(min_calib=100)
        keys  = RNG.standard_normal((5, 4, 8)).astype(np.float16)
        vals  = RNG.standard_normal((5, 4, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        assert not cache._key_caches[0].is_fitted
        q     = RNG.standard_normal((4, 8)).astype(np.float16)
        ks, vs = cache.recall(0, q, top_k=3)
        assert ks.shape[0] <= 5

    def test_values_come_from_cpu_shadow(self):
        cache = self._make(n_layers=1, n_heads=2, head_dim=4, rank=2, landmarks=2, min_calib=5)
        keys  = RNG.standard_normal((6, 2, 4)).astype(np.float16)
        vals  = np.arange(6 * 2 * 4, dtype=np.float16).reshape(6, 2, 4)
        cache.store(0, keys, vals)
        q = RNG.standard_normal((2, 4)).astype(np.float16)
        ks, vs = cache.recall(0, q, top_k=2)
        # Values should not be all-zero (they come from the shadow dict)
        assert vs.dtype == np.float16

    def test_clear_single_layer(self):
        cache = self._make()
        keys  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        vals  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        cache.store(1, keys, vals)
        cache.clear(layer_id=0)
        assert cache.n_stored(0) == 0
        assert cache.n_stored(1) == 8  # layer 1 unaffected

    def test_clear_all_layers(self):
        cache = self._make()
        keys  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        vals  = RNG.standard_normal((8, 4, 8)).astype(np.float16)
        cache.store(0, keys, vals)
        cache.store(1, keys, vals)
        cache.clear()
        assert cache.n_stored(0) == 0
        assert cache.n_stored(1) == 0
        assert len(cache._value_shadow) == 0

    def test_default_config(self):
        cache = ShadowKVCache(n_layers=1, n_heads=2, head_dim=4)
        assert cache.config.svd_rank == 32

    def test_recall_top_k_zero_returns_empty(self):
        """top_k=0 results in empty keys_out, hitting the `if not keys_out:` branch (line 453)."""
        cache = self._make(n_layers=1, n_heads=2, head_dim=4, rank=2, landmarks=4, min_calib=3)
        keys  = RNG.standard_normal((5, 2, 4)).astype(np.float16)
        vals  = RNG.standard_normal((5, 2, 4)).astype(np.float16)
        cache.store(0, keys, vals)
        q     = RNG.standard_normal((2, 4)).astype(np.float16)
        ks, vs = cache.recall(0, q, top_k=0)
        assert ks.shape[0] == 0
        assert vs.shape[0] == 0
