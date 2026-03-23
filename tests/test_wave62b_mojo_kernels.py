"""tests/test_wave62b_mojo_kernels.py — Wave 62b Mojo kernel wrapper unit tests.

Tests all six Wave 62b Mojo-backed Python wrappers:
  MojoSVDqHead, MojoShadowKVFit, MojoClusterKV, MojoAny4Lloyd,
  MojoOuroborosNgram, MojoPyramidKVBudget.

All tests exercise the NumPy fallback path (Mojo bridge returns None for
kernels in CI), validating shapes, dtypes, numerical correctness, error
handling, configuration, and backend reporting.
"""

import unittest

import numpy as np

from squish.kernels.mojo.svdq_head_mojo import SVDqHeadMojoConfig, MojoSVDqHead
from squish.kernels.mojo.shadow_kv_fit_mojo import ShadowKVFitMojoConfig, MojoShadowKVFit
from squish.kernels.mojo.cluster_kv_mojo import ClusterKVMojoConfig, MojoClusterKV
from squish.kernels.mojo.any4_lloyd_mojo import Any4LloydMojoConfig, MojoAny4Lloyd
from squish.kernels.mojo.ouroboros_ngram_mojo import OuroborosNgramMojoConfig, MojoOuroborosNgram
from squish.kernels.mojo.pyramid_kv_budget_mojo import PyramidKVBudgetMojoConfig, MojoPyramidKVBudget


# ── SVDqHeadMojoConfig ────────────────────────────────────────────────────────

class TestSVDqHeadMojoConfig(unittest.TestCase):
    def test_default_rank_threshold(self):
        self.assertAlmostEqual(SVDqHeadMojoConfig().rank_threshold, 0.01)

    def test_custom_rank_threshold(self):
        cfg = SVDqHeadMojoConfig(rank_threshold=0.05)
        self.assertAlmostEqual(cfg.rank_threshold, 0.05)

    def test_is_dataclass(self):
        from dataclasses import fields
        self.assertEqual(len(fields(SVDqHeadMojoConfig())), 1)


# ── MojoSVDqHead ──────────────────────────────────────────────────────────────

class TestMojoSVDqHead(unittest.TestCase):
    def _keys(self, L=2, H=3, T=8, D=8):
        return np.random.default_rng(0).standard_normal((L, H, T, D)).astype(np.float32)

    def test_rank_profile_shape(self):
        keys = self._keys()
        L, H, T, D = keys.shape
        out = MojoSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape, (L, H, min(T, D)))

    def test_rank_profile_dtype(self):
        out = MojoSVDqHead().rank_profile(self._keys())
        self.assertEqual(out.dtype, np.float32)

    def test_rank_profile_non_negative(self):
        out = MojoSVDqHead().rank_profile(self._keys())
        self.assertTrue(np.all(out >= 0))

    def test_rank_profile_rejects_3d(self):
        with self.assertRaises(ValueError):
            MojoSVDqHead().rank_profile(np.zeros((4, 8, 8), dtype=np.float32))

    def test_rank_profile_T_lt_D(self):
        keys = self._keys(L=1, H=1, T=3, D=8)
        out = MojoSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape[-1], 3)

    def test_rank_profile_D_lt_T(self):
        keys = self._keys(L=1, H=1, T=8, D=4)
        out = MojoSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape[-1], 4)

    def test_rank_per_head_shape(self):
        keys = self._keys()
        L, H = keys.shape[:2]
        out = MojoSVDqHead().rank_per_head(keys)
        self.assertEqual(out.shape, (L, H))

    def test_rank_per_head_dtype(self):
        out = MojoSVDqHead().rank_per_head(self._keys())
        self.assertEqual(out.dtype, np.int32)

    def test_rank_per_head_values_bounded(self):
        keys = self._keys()
        L, H, T, D = keys.shape
        out = MojoSVDqHead().rank_per_head(keys)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= min(T, D)))

    def test_backend_str(self):
        self.assertIn(MojoSVDqHead().backend(), ("mojo", "numpy"))

    def test_no_nan_all_ones(self):
        keys = np.ones((2, 2, 4, 4), dtype=np.float32)
        out = MojoSVDqHead().rank_profile(keys)
        self.assertFalse(np.any(np.isnan(out)))

    def test_rank_per_head_custom_threshold_high(self):
        keys = self._keys()
        out = MojoSVDqHead().rank_per_head(keys, threshold=0.99)
        self.assertTrue(np.all(out >= 0))

    def test_rank_per_head_threshold_zero_all_kept(self):
        keys = self._keys()
        out = MojoSVDqHead().rank_per_head(keys, threshold=0.0)
        _, _, T, D = keys.shape
        self.assertTrue(np.all(out == min(T, D)))


# ── ShadowKVFitMojoConfig ─────────────────────────────────────────────────────

class TestShadowKVFitMojoConfig(unittest.TestCase):
    def test_default_rank(self):
        self.assertEqual(ShadowKVFitMojoConfig().rank, 16)

    def test_custom_rank(self):
        self.assertEqual(ShadowKVFitMojoConfig(rank=8).rank, 8)


# ── MojoShadowKVFit ───────────────────────────────────────────────────────────

class TestMojoShadowKVFit(unittest.TestCase):
    def _keys(self, H=4, T=12, D=8):
        return np.random.default_rng(1).standard_normal((H, T, D)).astype(np.float32)

    def test_fit_svd_shape(self):
        keys = self._keys()
        H, T, D = keys.shape
        rank = 4
        out = MojoShadowKVFit(ShadowKVFitMojoConfig(rank=rank)).fit_svd(keys)
        self.assertEqual(out.shape, (H, rank, D))

    def test_fit_svd_dtype(self):
        out = MojoShadowKVFit().fit_svd(self._keys())
        self.assertEqual(out.dtype, np.float32)

    def test_fit_svd_rejects_2d(self):
        with self.assertRaises(ValueError):
            MojoShadowKVFit().fit_svd(np.zeros((4, 8), dtype=np.float32))

    def test_fit_svd_rank_clamped(self):
        keys = self._keys(H=2, T=3, D=8)
        out = MojoShadowKVFit(ShadowKVFitMojoConfig(rank=16)).fit_svd(keys)
        self.assertEqual(out.shape[1], 3)

    def test_store_batch_shape(self):
        keys = self._keys(H=3, T=10, D=8)
        rank = 4
        v_mat = MojoShadowKVFit(ShadowKVFitMojoConfig(rank=rank)).fit_svd(keys)
        out = MojoShadowKVFit().store_batch(keys, v_mat)
        self.assertEqual(out.shape, (3, 10, rank))

    def test_store_batch_dtype(self):
        keys = self._keys()
        rank = 4
        v_mat = MojoShadowKVFit(ShadowKVFitMojoConfig(rank=rank)).fit_svd(keys)
        out = MojoShadowKVFit().store_batch(keys, v_mat)
        self.assertEqual(out.dtype, np.float32)

    def test_store_batch_rejects_2d_keys(self):
        with self.assertRaises(ValueError):
            MojoShadowKVFit().store_batch(
                np.zeros((4, 8), dtype=np.float32),
                np.zeros((4, 4, 8), dtype=np.float32),
            )

    def test_no_nan_fit(self):
        keys = np.ones((2, 4, 4), dtype=np.float32)
        out = MojoShadowKVFit(ShadowKVFitMojoConfig(rank=2)).fit_svd(keys)
        self.assertFalse(np.any(np.isnan(out)))

    def test_backend_str(self):
        self.assertIn(MojoShadowKVFit().backend(), ("mojo", "numpy"))


# ── ClusterKVMojoConfig ───────────────────────────────────────────────────────

class TestClusterKVMojoConfig(unittest.TestCase):
    def test_default_evict_ratio(self):
        self.assertAlmostEqual(ClusterKVMojoConfig().evict_ratio, 0.5)

    def test_custom_evict_ratio(self):
        self.assertAlmostEqual(ClusterKVMojoConfig(evict_ratio=0.25).evict_ratio, 0.25)


# ── MojoClusterKV ─────────────────────────────────────────────────────────────

class TestMojoClusterKV(unittest.TestCase):
    def _data(self, S=20, C=5):
        rng = np.random.default_rng(2)
        return (
            rng.integers(0, C, size=S).astype(np.int32),
            rng.random(S).astype(np.float32),
        )

    def test_score_clusters_shape(self):
        assign, attn = self._data()
        out = MojoClusterKV().score_clusters(assign, attn, 5)
        self.assertEqual(out.shape, (5,))

    def test_score_clusters_dtype(self):
        assign, attn = self._data()
        out = MojoClusterKV().score_clusters(assign, attn, 5)
        self.assertEqual(out.dtype, np.float32)

    def test_score_clusters_non_negative(self):
        assign, attn = self._data()
        out = MojoClusterKV().score_clusters(assign, attn, 5)
        self.assertTrue(np.all(out >= 0))

    def test_score_clusters_sum_close_to_attn_sum(self):
        S, C = 100, 6
        assign, attn = self._data(S, C)
        out = MojoClusterKV().score_clusters(assign, attn, C)
        self.assertAlmostEqual(float(out.sum()), float(attn.sum()), places=4)

    def test_score_clusters_shape_mismatch_error(self):
        with self.assertRaises(ValueError):
            MojoClusterKV().score_clusters(
                np.zeros(5, dtype=np.int32),
                np.zeros(6, dtype=np.float32),
                3,
            )

    def test_evict_mask_shape(self):
        assign, attn = self._data(20, 5)
        mask = MojoClusterKV().evict_mask(assign, attn, 5)
        self.assertEqual(mask.shape, (20,))

    def test_evict_mask_dtype(self):
        assign, attn = self._data()
        mask = MojoClusterKV().evict_mask(assign, attn, 5)
        self.assertEqual(mask.dtype, bool)

    def test_evict_ratio_zero_no_eviction(self):
        assign, attn = self._data()
        mask = MojoClusterKV().evict_mask(assign, attn, 5, evict_ratio=0.0)
        self.assertFalse(np.any(mask))

    def test_evict_ratio_one_all_evicted(self):
        assign, attn = self._data()
        mask = MojoClusterKV().evict_mask(assign, attn, 5, evict_ratio=1.0)
        self.assertTrue(np.all(mask))

    def test_backend_str(self):
        self.assertIn(MojoClusterKV().backend(), ("mojo", "numpy"))


# ── Any4LloydMojoConfig ───────────────────────────────────────────────────────

class TestAny4LloydMojoConfig(unittest.TestCase):
    def test_default_n_iter(self):
        self.assertEqual(Any4LloydMojoConfig().n_iter, 100)

    def test_default_codebook_k(self):
        self.assertEqual(Any4LloydMojoConfig().codebook_k, 16)

    def test_custom(self):
        cfg = Any4LloydMojoConfig(n_iter=20, codebook_k=4)
        self.assertEqual(cfg.n_iter, 20)
        self.assertEqual(cfg.codebook_k, 4)


# ── MojoAny4Lloyd ─────────────────────────────────────────────────────────────

class TestMojoAny4Lloyd(unittest.TestCase):
    def _vc(self, N=64, k=8):
        rng = np.random.default_rng(3)
        v = rng.standard_normal(N).astype(np.float32)
        c = v[rng.choice(N, size=k, replace=False)].copy()
        return v, c

    def test_lloyd_step_shape(self):
        v, c = self._vc()
        out = MojoAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertEqual(out.shape, (8,))

    def test_lloyd_step_dtype(self):
        v, c = self._vc()
        out = MojoAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertEqual(out.dtype, np.float32)

    def test_lloyd_step_no_nan(self):
        v, c = self._vc()
        out = MojoAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertFalse(np.any(np.isnan(out)))

    def test_lloyd_step_centroids_in_data_range(self):
        v, c = self._vc(100, 8)
        out = MojoAny4Lloyd().lloyd_step(v, c, n_iter=10)
        margin = 1e-2
        self.assertTrue(float(out.min()) >= float(v.min()) - margin)
        self.assertTrue(float(out.max()) <= float(v.max()) + margin)

    def test_quantize_indices_shape(self):
        v, _ = self._vc(64, 8)
        idx, cb = MojoAny4Lloyd(Any4LloydMojoConfig(codebook_k=8)).quantize(v)
        self.assertEqual(idx.shape, (64,))

    def test_quantize_codebook_shape(self):
        v, _ = self._vc(64, 8)
        idx, cb = MojoAny4Lloyd(Any4LloydMojoConfig(codebook_k=8)).quantize(v)
        self.assertEqual(cb.shape, (8,))

    def test_quantize_indices_dtype(self):
        v, _ = self._vc()
        idx, cb = MojoAny4Lloyd(Any4LloydMojoConfig(codebook_k=8)).quantize(v)
        self.assertEqual(idx.dtype, np.int32)

    def test_quantize_in_range(self):
        v, _ = self._vc(64, 8)
        idx, cb = MojoAny4Lloyd(Any4LloydMojoConfig(codebook_k=8)).quantize(v)
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 8))

    def test_quantize_with_centroids(self):
        v = np.linspace(-1, 1, 32, dtype=np.float32)
        c = np.linspace(-1, 1, 4, dtype=np.float32)
        idx, cb = MojoAny4Lloyd().quantize(v, centroids=c)
        self.assertEqual(cb.shape, (4,))

    def test_backend_str(self):
        self.assertIn(MojoAny4Lloyd().backend(), ("mojo", "numpy"))


# ── OuroborosNgramMojoConfig ──────────────────────────────────────────────────

class TestOuroborosNgramMojoConfig(unittest.TestCase):
    def test_default_order(self):
        self.assertEqual(OuroborosNgramMojoConfig().order, 4)

    def test_default_depth(self):
        self.assertEqual(OuroborosNgramMojoConfig().depth, 8)

    def test_default_max_entries(self):
        self.assertEqual(OuroborosNgramMojoConfig().max_entries, 65536)

    def test_default_temperature(self):
        self.assertAlmostEqual(OuroborosNgramMojoConfig().temperature, 1.0)

    def test_custom(self):
        cfg = OuroborosNgramMojoConfig(order=3, max_entries=50, depth=4, temperature=0.5)
        self.assertEqual(cfg.order, 3)
        self.assertEqual(cfg.max_entries, 50)
        self.assertEqual(cfg.depth, 4)
        self.assertAlmostEqual(cfg.temperature, 0.5)


# ── MojoOuroborosNgram ────────────────────────────────────────────────────────

class TestMojoOuroborosNgram(unittest.TestCase):
    def _tokens(self, T=20):
        return np.arange(T, dtype=np.int32) % 10

    def test_build_returns_ndarray(self):
        out = MojoOuroborosNgram().build(self._tokens())
        self.assertIsInstance(out, np.ndarray)

    def test_build_dtype(self):
        out = MojoOuroborosNgram().build(self._tokens())
        self.assertEqual(out.dtype, np.int32)

    def test_build_col_width(self):
        order = 3
        out = MojoOuroborosNgram(OuroborosNgramMojoConfig(order=order)).build(
            self._tokens(), order=order
        )
        if len(out) > 0:
            self.assertEqual(out.shape[1], order + 1)

    def test_build_max_entries(self):
        out = MojoOuroborosNgram().build(self._tokens(50), max_entries=5)
        self.assertLessEqual(len(out), 5)

    def test_build_empty_too_short(self):
        out = MojoOuroborosNgram(OuroborosNgramMojoConfig(order=5)).build(
            np.array([1, 2, 3], dtype=np.int32), order=5
        )
        self.assertEqual(len(out), 0)

    def test_build_order_2(self):
        toks = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
        out = MojoOuroborosNgram().build(toks, order=2)
        self.assertGreater(len(out), 0)

    def test_lookahead_shape(self):
        depth, vocab = 4, 32
        logits = np.random.default_rng(5).standard_normal((depth, vocab)).astype(np.float32)
        out = MojoOuroborosNgram().lookahead(logits)
        self.assertEqual(out.shape, (depth,))

    def test_lookahead_dtype(self):
        out = MojoOuroborosNgram().lookahead(np.ones((4, 32), dtype=np.float32))
        self.assertEqual(out.dtype, np.int32)

    def test_lookahead_in_vocab(self):
        depth, vocab = 6, 50
        logits = np.random.default_rng(6).standard_normal((depth, vocab)).astype(np.float32)
        out = MojoOuroborosNgram().lookahead(logits)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out < vocab))

    def test_lookahead_rejects_1d(self):
        with self.assertRaises(ValueError):
            MojoOuroborosNgram().lookahead(np.ones(10, dtype=np.float32))

    def test_lookahead_deterministic(self):
        logits = np.random.default_rng(7).standard_normal((4, 32)).astype(np.float32)
        a = MojoOuroborosNgram().lookahead(logits, seed=42)
        b = MojoOuroborosNgram().lookahead(logits, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_backend_str(self):
        self.assertIn(MojoOuroborosNgram().backend(), ("mojo", "numpy"))


# ── PyramidKVBudgetMojoConfig ─────────────────────────────────────────────────

class TestPyramidKVBudgetMojoConfig(unittest.TestCase):
    def test_default_alpha(self):
        self.assertAlmostEqual(PyramidKVBudgetMojoConfig().alpha, 0.5)

    def test_default_min_budget(self):
        self.assertEqual(PyramidKVBudgetMojoConfig().min_budget, 32)

    def test_custom(self):
        cfg = PyramidKVBudgetMojoConfig(alpha=0.7, min_budget=16)
        self.assertAlmostEqual(cfg.alpha, 0.7)
        self.assertEqual(cfg.min_budget, 16)


# ── MojoPyramidKVBudget ───────────────────────────────────────────────────────

class TestMojoPyramidKVBudget(unittest.TestCase):
    def test_compute_shape(self):
        out = MojoPyramidKVBudget().compute(512.0, 32)
        self.assertEqual(out.shape, (32,))

    def test_compute_dtype(self):
        out = MojoPyramidKVBudget().compute(512.0, 32)
        self.assertEqual(out.dtype, np.int32)

    def test_compute_first_layer(self):
        out = MojoPyramidKVBudget().compute(512.0, 32)
        self.assertAlmostEqual(float(out[0]), 512.0, delta=1.0)

    def test_compute_monotone(self):
        out = MojoPyramidKVBudget().compute(512.0, 32)
        self.assertTrue(np.all(np.diff(out.astype(np.float32)) <= 0))

    def test_compute_min_budget_floor(self):
        cfg = PyramidKVBudgetMojoConfig(min_budget=64)
        out = MojoPyramidKVBudget(cfg).compute(512.0, 32)
        self.assertTrue(np.all(out >= 64))

    def test_compute_alpha_zero_flat(self):
        cfg = PyramidKVBudgetMojoConfig(alpha=0.0)
        out = MojoPyramidKVBudget(cfg).compute(100.0, 8)
        self.assertTrue(np.all(out == out[0]))

    def test_compute_single_layer(self):
        out = MojoPyramidKVBudget().compute(256.0, 1)
        self.assertEqual(len(out), 1)

    def test_rejects_n_layers_zero(self):
        with self.assertRaises(ValueError):
            MojoPyramidKVBudget().compute(256.0, 0)

    def test_rejects_negative_base(self):
        with self.assertRaises(ValueError):
            MojoPyramidKVBudget().compute(-10.0, 8)

    def test_total(self):
        planner = MojoPyramidKVBudget()
        budgets = planner.compute(512.0, 8)
        self.assertEqual(planner.total(512.0, 8), int(budgets.sum()))

    def test_alpha_one_last_layer_min_budget(self):
        mb = 16
        cfg = PyramidKVBudgetMojoConfig(alpha=1.0, min_budget=mb)
        out = MojoPyramidKVBudget(cfg).compute(128.0, 8)
        self.assertGreaterEqual(int(out[-1]), mb)

    def test_backend_str(self):
        self.assertIn(MojoPyramidKVBudget().backend(), ("mojo", "numpy"))


if __name__ == "__main__":
    unittest.main()
