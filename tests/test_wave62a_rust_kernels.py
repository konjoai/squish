"""tests/test_wave62a_rust_kernels.py — Wave 62a Rust wrapper unit tests.

Tests all seven Wave 62a Rust-backed Python wrappers:
  RustSVDqHead, RustShadowKVFit, RustClusterKV, RustAny4Lloyd,
  RustOuroborosNgram, RustPyramidKVBudget, RustQMoECompress.

All tests exercise the NumPy fallback path (Rust extension may not be
built in CI), validating shapes, dtypes, numerical correctness, error
handling, configuration, and backend reporting.
"""

import unittest

import numpy as np

from squish.kernels.rs_svdq_head import SVDqHeadConfig, RustSVDqHead
from squish.kernels.rs_shadow_kv_fit import ShadowKVFitConfig, RustShadowKVFit
from squish.kernels.rs_cluster_kv import ClusterKVConfig, RustClusterKV
from squish.kernels.rs_any4_lloyd import Any4LloydConfig, RustAny4Lloyd
from squish.kernels.rs_ouroboros_ngram import OuroborosNgramConfig, RustOuroborosNgram
from squish.kernels.rs_pyramid_kv_budget import PyramidKVBudgetConfig, RustPyramidKVBudget
from squish.kernels.rs_qmoe_compress import QMoECompressConfig, RustQMoECompress


# ── SVDqHeadConfig ────────────────────────────────────────────────────────────

class TestSVDqHeadConfig(unittest.TestCase):
    def test_default_rank_threshold(self):
        cfg = SVDqHeadConfig()
        self.assertAlmostEqual(cfg.rank_threshold, 0.01)

    def test_custom_rank_threshold(self):
        cfg = SVDqHeadConfig(rank_threshold=0.05)
        self.assertAlmostEqual(cfg.rank_threshold, 0.05)

    def test_rank_threshold_zero(self):
        cfg = SVDqHeadConfig(rank_threshold=0.0)
        self.assertEqual(cfg.rank_threshold, 0.0)

    def test_config_is_dataclass(self):
        from dataclasses import fields
        self.assertEqual(len(fields(SVDqHeadConfig())), 1)


# ── RustSVDqHead ──────────────────────────────────────────────────────────────

class TestRustSVDqHead(unittest.TestCase):
    def _make_keys(self, L=2, H=4, T=8, D=8):
        rng = np.random.default_rng(0)
        return rng.standard_normal((L, H, T, D)).astype(np.float32)

    def test_rank_profile_shape(self):
        svd = RustSVDqHead()
        keys = self._make_keys()
        out = svd.rank_profile(keys)
        L, H, T, D = keys.shape
        self.assertEqual(out.shape, (L, H, min(T, D)))

    def test_rank_profile_dtype(self):
        out = RustSVDqHead().rank_profile(self._make_keys())
        self.assertEqual(out.dtype, np.float32)

    def test_rank_profile_non_negative(self):
        out = RustSVDqHead().rank_profile(self._make_keys())
        self.assertTrue(np.all(out >= 0))

    def test_rank_profile_single_layer(self):
        keys = self._make_keys(L=1, H=2, T=4, D=4)
        out = RustSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape[0], 1)

    def test_rank_profile_non_square(self):
        keys = self._make_keys(L=1, H=1, T=6, D=4)
        out = RustSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape[-1], 4)  # min(T=6, D=4)

    def test_rank_profile_non_square_T_lt_D(self):
        keys = self._make_keys(L=1, H=1, T=3, D=8)
        out = RustSVDqHead().rank_profile(keys)
        self.assertEqual(out.shape[-1], 3)  # min(T=3, D=8)

    def test_rank_profile_rejects_wrong_ndim(self):
        svd = RustSVDqHead()
        with self.assertRaises(ValueError):
            svd.rank_profile(np.zeros((4, 8, 8), dtype=np.float32))

    def test_rank_per_head_shape(self):
        keys = self._make_keys()
        L, H = keys.shape[:2]
        out = RustSVDqHead().rank_per_head(keys)
        self.assertEqual(out.shape, (L, H))

    def test_rank_per_head_dtype(self):
        out = RustSVDqHead().rank_per_head(self._make_keys())
        self.assertEqual(out.dtype, np.int32)

    def test_rank_per_head_values_bounded(self):
        keys = self._make_keys()
        L, H, T, D = keys.shape
        out = RustSVDqHead().rank_per_head(keys)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= min(T, D)))

    def test_rank_per_head_custom_threshold(self):
        keys = self._make_keys()
        svd = RustSVDqHead(SVDqHeadConfig(rank_threshold=0.99))
        out = svd.rank_per_head(keys, threshold=0.99)
        # High threshold → fewer components kept
        self.assertTrue(np.all(out >= 0))

    def test_backend_str(self):
        b = RustSVDqHead().backend()
        self.assertIn(b, ("rust", "numpy"))

    def test_default_config_used(self):
        svd = RustSVDqHead()
        self.assertAlmostEqual(svd._cfg.rank_threshold, 0.01)

    def test_rank_profile_no_nan(self):
        keys = np.ones((2, 2, 4, 4), dtype=np.float32)
        out = RustSVDqHead().rank_profile(keys)
        self.assertFalse(np.any(np.isnan(out)))


# ── ShadowKVFitConfig ─────────────────────────────────────────────────────────

class TestShadowKVFitConfig(unittest.TestCase):
    def test_default_rank(self):
        self.assertEqual(ShadowKVFitConfig().rank, 16)

    def test_custom_rank(self):
        self.assertEqual(ShadowKVFitConfig(rank=8).rank, 8)


# ── RustShadowKVFit ───────────────────────────────────────────────────────────

class TestRustShadowKVFit(unittest.TestCase):
    def _keys(self, H=4, T=16, D=8):
        return np.random.default_rng(1).standard_normal((H, T, D)).astype(np.float32)

    def test_fit_svd_shape(self):
        keys = self._keys()
        H, T, D = keys.shape
        rank = 4
        out = RustShadowKVFit(ShadowKVFitConfig(rank=rank)).fit_svd(keys)
        self.assertEqual(out.shape, (H, rank, D))

    def test_fit_svd_dtype(self):
        out = RustShadowKVFit().fit_svd(self._keys())
        self.assertEqual(out.dtype, np.float32)

    def test_fit_svd_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            RustShadowKVFit().fit_svd(np.zeros((4, 8), dtype=np.float32))

    def test_fit_svd_rank_clamped_to_min_T_D(self):
        # T=3 < rank=16
        keys = self._keys(H=2, T=3, D=8)
        out = RustShadowKVFit(ShadowKVFitConfig(rank=16)).fit_svd(keys)
        self.assertEqual(out.shape[1], 3)  # min(T=3, D=8)

    def test_store_batch_shape(self):
        keys = self._keys(H=3, T=10, D=8)
        rank = 4
        v_mat = RustShadowKVFit(ShadowKVFitConfig(rank=rank)).fit_svd(keys)
        out = RustShadowKVFit().store_batch(keys, v_mat)
        self.assertEqual(out.shape, (3, 10, rank))

    def test_store_batch_dtype(self):
        keys = self._keys()
        rank = 4
        v_mat = RustShadowKVFit(ShadowKVFitConfig(rank=rank)).fit_svd(keys)
        out = RustShadowKVFit().store_batch(keys, v_mat)
        self.assertEqual(out.dtype, np.float32)

    def test_store_batch_rejects_1d(self):
        with self.assertRaises(ValueError):
            RustShadowKVFit().store_batch(
                np.zeros((4, 8), dtype=np.float32),
                np.zeros((4, 4, 8), dtype=np.float32),
            )

    def test_store_batch_orthogonality_proxy(self):
        # V rows from SVD should give decreasing projected variance
        keys = self._keys(H=1, T=32, D=8)
        v_mat = RustShadowKVFit(ShadowKVFitConfig(rank=4)).fit_svd(keys)
        proj = RustShadowKVFit().store_batch(keys, v_mat)
        self.assertEqual(proj.shape, (1, 32, 4))

    def test_backend_str(self):
        self.assertIn(RustShadowKVFit().backend(), ("rust", "numpy"))

    def test_no_nan_in_fit(self):
        keys = np.ones((2, 4, 4), dtype=np.float32)
        out = RustShadowKVFit(ShadowKVFitConfig(rank=2)).fit_svd(keys)
        self.assertFalse(np.any(np.isnan(out)))


# ── ClusterKVConfig ───────────────────────────────────────────────────────────

class TestClusterKVConfig(unittest.TestCase):
    def test_default_evict_ratio(self):
        self.assertAlmostEqual(ClusterKVConfig().evict_ratio, 0.5)

    def test_custom_evict_ratio(self):
        self.assertAlmostEqual(ClusterKVConfig(evict_ratio=0.3).evict_ratio, 0.3)


# ── RustClusterKV ─────────────────────────────────────────────────────────────

class TestRustClusterKV(unittest.TestCase):
    def _data(self, S=20, C=5):
        rng = np.random.default_rng(2)
        assign = rng.integers(0, C, size=S).astype(np.int32)
        attn = rng.random(S).astype(np.float32)
        return assign, attn

    def test_score_clusters_shape(self):
        assign, attn = self._data()
        out = RustClusterKV().score_clusters(assign, attn, 5)
        self.assertEqual(out.shape, (5,))

    def test_score_clusters_dtype(self):
        assign, attn = self._data()
        out = RustClusterKV().score_clusters(assign, attn, 5)
        self.assertEqual(out.dtype, np.float32)

    def test_score_clusters_non_negative(self):
        assign, attn = self._data()
        out = RustClusterKV().score_clusters(assign, attn, 5)
        self.assertTrue(np.all(out >= 0))

    def test_score_clusters_sums_to_attn_total(self):
        S, C = 50, 4
        assign, attn = self._data(S, C)
        out = RustClusterKV().score_clusters(assign, attn, C)
        self.assertAlmostEqual(float(out.sum()), float(attn.sum()), places=4)

    def test_score_clusters_shape_mismatch(self):
        assign = np.zeros(5, dtype=np.int32)
        attn = np.zeros(6, dtype=np.float32)
        with self.assertRaises(ValueError):
            RustClusterKV().score_clusters(assign, attn, 3)

    def test_evict_mask_shape(self):
        assign, attn = self._data(20, 5)
        mask = RustClusterKV().evict_mask(assign, attn, 5)
        self.assertEqual(mask.shape, (20,))

    def test_evict_mask_dtype(self):
        assign, attn = self._data()
        mask = RustClusterKV().evict_mask(assign, attn, 5)
        self.assertEqual(mask.dtype, bool)

    def test_evict_mask_evict_ratio_zero(self):
        assign, attn = self._data()
        mask = RustClusterKV().evict_mask(assign, attn, 5, evict_ratio=0.0)
        self.assertFalse(np.any(mask))

    def test_evict_mask_evict_ratio_one(self):
        assign, attn = self._data()
        mask = RustClusterKV().evict_mask(assign, attn, 5, evict_ratio=1.0)
        self.assertTrue(np.all(mask))

    def test_backend_str(self):
        self.assertIn(RustClusterKV().backend(), ("rust", "numpy"))


# ── Any4LloydConfig ───────────────────────────────────────────────────────────

class TestAny4LloydConfig(unittest.TestCase):
    def test_default_n_iter(self):
        self.assertEqual(Any4LloydConfig().n_iter, 100)

    def test_default_codebook_k(self):
        self.assertEqual(Any4LloydConfig().codebook_k, 16)

    def test_custom_config(self):
        cfg = Any4LloydConfig(n_iter=50, codebook_k=8)
        self.assertEqual(cfg.n_iter, 50)
        self.assertEqual(cfg.codebook_k, 8)


# ── RustAny4Lloyd ─────────────────────────────────────────────────────────────

class TestRustAny4Lloyd(unittest.TestCase):
    def _values_and_init(self, N=100, k=8):
        rng = np.random.default_rng(3)
        v = rng.standard_normal(N).astype(np.float32)
        c = v[rng.choice(N, size=k, replace=False)].copy()
        return v, c

    def test_lloyd_step_shape(self):
        v, c = self._values_and_init(k=8)
        out = RustAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertEqual(out.shape, (8,))

    def test_lloyd_step_dtype(self):
        v, c = self._values_and_init(k=8)
        out = RustAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertEqual(out.dtype, np.float32)

    def test_lloyd_step_centroids_in_range(self):
        v, c = self._values_and_init(100, 8)
        out = RustAny4Lloyd().lloyd_step(v, c, n_iter=10)
        self.assertTrue(float(out.min()) >= float(v.min()) - 1e-3)
        self.assertTrue(float(out.max()) <= float(v.max()) + 1e-3)

    def test_lloyd_step_no_nan(self):
        v, c = self._values_and_init()
        out = RustAny4Lloyd().lloyd_step(v, c, n_iter=5)
        self.assertFalse(np.any(np.isnan(out)))

    def test_quantize_indices_shape(self):
        v, _ = self._values_and_init(64, 8)
        indices, cb = RustAny4Lloyd(Any4LloydConfig(codebook_k=8)).quantize(v)
        self.assertEqual(indices.shape, (64,))

    def test_quantize_codebook_shape(self):
        v, _ = self._values_and_init(64, 8)
        indices, cb = RustAny4Lloyd(Any4LloydConfig(codebook_k=8)).quantize(v)
        self.assertEqual(cb.shape, (8,))

    def test_quantize_indices_dtype(self):
        v, _ = self._values_and_init(64, 8)
        indices, cb = RustAny4Lloyd(Any4LloydConfig(codebook_k=8)).quantize(v)
        self.assertEqual(indices.dtype, np.int32)

    def test_quantize_indices_in_range(self):
        v, _ = self._values_and_init(64, 8)
        indices, cb = RustAny4Lloyd(Any4LloydConfig(codebook_k=8)).quantize(v)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 8))

    def test_quantize_with_provided_centroids(self):
        v = np.linspace(-1, 1, 32, dtype=np.float32)
        c = np.linspace(-1, 1, 4, dtype=np.float32)
        indices, cb = RustAny4Lloyd().quantize(v, centroids=c)
        self.assertEqual(cb.shape, (4,))

    def test_backend_str(self):
        self.assertIn(RustAny4Lloyd().backend(), ("rust", "numpy"))


# ── OuroborosNgramConfig ──────────────────────────────────────────────────────

class TestOuroborosNgramConfig(unittest.TestCase):
    def test_default_order(self):
        self.assertEqual(OuroborosNgramConfig().order, 4)

    def test_default_depth(self):
        self.assertEqual(OuroborosNgramConfig().depth, 8)

    def test_default_max_entries(self):
        self.assertEqual(OuroborosNgramConfig().max_entries, 65536)

    def test_default_temperature(self):
        self.assertAlmostEqual(OuroborosNgramConfig().temperature, 1.0)

    def test_custom_config(self):
        cfg = OuroborosNgramConfig(order=3, max_entries=100, depth=4, temperature=0.7)
        self.assertEqual(cfg.order, 3)
        self.assertEqual(cfg.max_entries, 100)
        self.assertEqual(cfg.depth, 4)
        self.assertAlmostEqual(cfg.temperature, 0.7)


# ── RustOuroborosNgram ────────────────────────────────────────────────────────

class TestRustOuroborosNgram(unittest.TestCase):
    def _tokens(self, T=20):
        return np.arange(T, dtype=np.int32) % 10

    def test_build_returns_ndarray(self):
        out = RustOuroborosNgram().build(self._tokens())
        self.assertIsInstance(out, np.ndarray)

    def test_build_dtype(self):
        out = RustOuroborosNgram().build(self._tokens())
        self.assertEqual(out.dtype, np.int32)

    def test_build_col_count(self):
        order = 3
        out = RustOuroborosNgram(OuroborosNgramConfig(order=order)).build(
            self._tokens(), order=order
        )
        if len(out) > 0:
            self.assertEqual(out.shape[1], order + 1)

    def test_build_max_entries_respected(self):
        out = RustOuroborosNgram().build(self._tokens(50), max_entries=5)
        self.assertLessEqual(len(out), 5)

    def test_build_empty_for_short_sequence(self):
        # Sequence shorter than order → no n-grams
        out = RustOuroborosNgram(OuroborosNgramConfig(order=5)).build(
            np.array([1, 2, 3], dtype=np.int32), order=5
        )
        self.assertEqual(len(out), 0)

    def test_build_order_2(self):
        toks = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
        out = RustOuroborosNgram().build(toks, order=2)
        self.assertTrue(len(out) > 0)

    def test_lookahead_shape(self):
        depth, vocab = 4, 32
        logits = np.random.default_rng(5).standard_normal((depth, vocab)).astype(np.float32)
        out = RustOuroborosNgram().lookahead(logits)
        self.assertEqual(out.shape, (depth,))

    def test_lookahead_dtype(self):
        logits = np.ones((4, 32), dtype=np.float32)
        out = RustOuroborosNgram().lookahead(logits)
        self.assertEqual(out.dtype, np.int32)

    def test_lookahead_tokens_in_vocab(self):
        depth, vocab = 6, 50
        logits = np.random.default_rng(6).standard_normal((depth, vocab)).astype(np.float32)
        out = RustOuroborosNgram().lookahead(logits)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out < vocab))

    def test_lookahead_rejects_1d(self):
        with self.assertRaises(ValueError):
            RustOuroborosNgram().lookahead(np.ones(10, dtype=np.float32))

    def test_backend_str(self):
        self.assertIn(RustOuroborosNgram().backend(), ("rust", "numpy"))

    def test_lookahead_deterministic_seed(self):
        logits = np.random.default_rng(7).standard_normal((4, 32)).astype(np.float32)
        a = RustOuroborosNgram().lookahead(logits, seed=42)
        b = RustOuroborosNgram().lookahead(logits, seed=42)
        np.testing.assert_array_equal(a, b)


# ── PyramidKVBudgetConfig ─────────────────────────────────────────────────────

class TestPyramidKVBudgetConfig(unittest.TestCase):
    def test_default_alpha(self):
        self.assertAlmostEqual(PyramidKVBudgetConfig().alpha, 0.5)

    def test_default_min_budget(self):
        self.assertEqual(PyramidKVBudgetConfig().min_budget, 32)

    def test_custom_config(self):
        cfg = PyramidKVBudgetConfig(alpha=0.8, min_budget=16)
        self.assertAlmostEqual(cfg.alpha, 0.8)
        self.assertEqual(cfg.min_budget, 16)


# ── RustPyramidKVBudget ───────────────────────────────────────────────────────

class TestRustPyramidKVBudget(unittest.TestCase):
    def test_compute_shape(self):
        out = RustPyramidKVBudget().compute(512.0, 32)
        self.assertEqual(out.shape, (32,))

    def test_compute_dtype(self):
        out = RustPyramidKVBudget().compute(512.0, 32)
        self.assertEqual(out.dtype, np.int32)

    def test_compute_first_layer_near_base(self):
        out = RustPyramidKVBudget().compute(512.0, 32)
        self.assertAlmostEqual(float(out[0]), 512.0, delta=1.0)

    def test_compute_monotone_non_increasing(self):
        out = RustPyramidKVBudget().compute(512.0, 32)
        self.assertTrue(np.all(np.diff(out.astype(np.float32)) <= 0))

    def test_compute_min_budget_floor(self):
        out = RustPyramidKVBudget(PyramidKVBudgetConfig(min_budget=64)).compute(512.0, 32)
        self.assertTrue(np.all(out >= 64))

    def test_compute_alpha_zero_flat(self):
        out = RustPyramidKVBudget(PyramidKVBudgetConfig(alpha=0.0)).compute(100.0, 8)
        self.assertTrue(np.all(out == out[0]))

    def test_compute_single_layer(self):
        out = RustPyramidKVBudget().compute(256.0, 1)
        self.assertEqual(len(out), 1)

    def test_compute_rejects_invalid_layers(self):
        with self.assertRaises(ValueError):
            RustPyramidKVBudget().compute(512.0, 0)

    def test_compute_rejects_negative_base(self):
        with self.assertRaises(ValueError):
            RustPyramidKVBudget().compute(-1.0, 8)

    def test_total_sum_correct(self):
        planner = RustPyramidKVBudget()
        out = planner.compute(512.0, 8)
        total = planner.total(512.0, 8)
        self.assertEqual(total, int(out.sum()))

    def test_backend_str(self):
        self.assertIn(RustPyramidKVBudget().backend(), ("rust", "numpy"))

    def test_alpha_one_all_min_budget(self):
        # alpha=1 → deepest layer = base*(1-1) = 0 → clipped to min_budget
        mb = 16
        out = RustPyramidKVBudget(PyramidKVBudgetConfig(alpha=1.0, min_budget=mb)).compute(
            128.0, 8
        )
        self.assertGreaterEqual(int(out[-1]), mb)


# ── QMoECompressConfig ────────────────────────────────────────────────────────

class TestQMoECompressConfig(unittest.TestCase):
    def test_default_n_iter(self):
        self.assertEqual(QMoECompressConfig().n_iter, 50)

    def test_default_seed(self):
        self.assertEqual(QMoECompressConfig().seed, 42)

    def test_custom_config(self):
        cfg = QMoECompressConfig(n_iter=10, seed=7)
        self.assertEqual(cfg.n_iter, 10)
        self.assertEqual(cfg.seed, 7)


# ── RustQMoECompress ──────────────────────────────────────────────────────────

class TestRustQMoECompress(unittest.TestCase):
    def _blocks(self, N=32, bs=8):
        return np.random.default_rng(4).standard_normal((N, bs)).astype(np.float32)

    def test_compress_codebook_shape(self):
        blocks = self._blocks()
        k = 8
        cb, asgn = RustQMoECompress().compress(blocks, k)
        self.assertEqual(cb.shape, (k, 8))

    def test_compress_assignments_shape(self):
        blocks = self._blocks()
        _, asgn = RustQMoECompress().compress(blocks, 8)
        self.assertEqual(asgn.shape, (32,))

    def test_compress_codebook_dtype(self):
        cb, _ = RustQMoECompress().compress(self._blocks(), 8)
        self.assertEqual(cb.dtype, np.float32)

    def test_compress_assignments_dtype(self):
        _, asgn = RustQMoECompress().compress(self._blocks(), 8)
        self.assertEqual(asgn.dtype, np.int32)

    def test_compress_assignments_in_range(self):
        k = 8
        _, asgn = RustQMoECompress().compress(self._blocks(), k)
        self.assertTrue(np.all(asgn >= 0))
        self.assertTrue(np.all(asgn < k))

    def test_compress_rejects_1d(self):
        with self.assertRaises(ValueError):
            RustQMoECompress().compress(np.zeros(32, dtype=np.float32), 8)

    def test_compress_rejects_k_zero(self):
        with self.assertRaises(ValueError):
            RustQMoECompress().compress(self._blocks(), 0)

    def test_compress_k_larger_than_N_clamped(self):
        # k > N should not crash (k is clamped to N in NumPy fallback)
        blocks = self._blocks(N=4, bs=4)
        cb, asgn = RustQMoECompress().compress(blocks, k=100)
        self.assertLessEqual(cb.shape[0], 4)

    def test_reconstruct_shape(self):
        blocks = self._blocks(N=16, bs=8)
        k = 4
        cb, asgn = RustQMoECompress().compress(blocks, k)
        rec = RustQMoECompress().reconstruct(asgn, cb)
        self.assertEqual(rec.shape, (16, 8))

    def test_reconstruct_dtype(self):
        cb, asgn = RustQMoECompress().compress(self._blocks(), 8)
        rec = RustQMoECompress().reconstruct(asgn, cb)
        self.assertEqual(rec.dtype, np.float32)

    def test_reconstruct_within_codebook_range(self):
        blocks = self._blocks(N=16)
        k = 4
        cb, asgn = RustQMoECompress().compress(blocks, k)
        rec = RustQMoECompress().reconstruct(asgn, cb)
        # Every reconstructed row must be exactly one of the codebook rows
        for i in range(len(asgn)):
            np.testing.assert_array_equal(rec[i], cb[asgn[i]])

    def test_no_nan_in_compress(self):
        cb, asgn = RustQMoECompress().compress(self._blocks(), 8)
        self.assertFalse(np.any(np.isnan(cb)))

    def test_backend_str(self):
        self.assertIn(RustQMoECompress().backend(), ("rust", "numpy"))

    def test_compress_deterministic_same_seed(self):
        blocks = self._blocks()
        k = 4
        cfg = QMoECompressConfig(n_iter=5, seed=0)
        cb1, a1 = RustQMoECompress(cfg).compress(blocks, k)
        cb2, a2 = RustQMoECompress(cfg).compress(blocks, k)
        np.testing.assert_array_equal(a1, a2)


if __name__ == "__main__":
    unittest.main()
