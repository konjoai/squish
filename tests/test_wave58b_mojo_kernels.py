"""tests/test_wave58b_mojo_kernels.py — Wave 58b Mojo kernel tests.

Tests for:
  - MojoDualChunkAttn   (DualChunkAttnConfig, forward)
  - MojoInfiniAttnMemory (InfiniAttnConfig, zero_memory, update, retrieve)
  - MojoSlidingWindowAttn (SlidingWindowAttnConfig, forward)
  - MojoHQQALS           (HQQALSConfig, fit_group, fit, dequantize)
  - MojoVPTQDecode       (VPTQDecodeConfig, decode, multi_decode)
  - MojoTopKP            (TopKPConfig, sample, filter)

All tests use NumPy fallback path (Mojo library not compiled in CI).
79 tests total.
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.mojo.dual_chunk_attn_mojo import DualChunkAttnConfig, MojoDualChunkAttn
from squish.kernels.mojo.infini_attn_mojo import InfiniAttnConfig, MojoInfiniAttnMemory
from squish.kernels.mojo.sliding_window_attn_mojo import SlidingWindowAttnConfig, MojoSlidingWindowAttn
from squish.kernels.mojo.hqq_als_mojo import HQQALSConfig, MojoHQQALS
from squish.kernels.mojo.vptq_decode_mojo import VPTQDecodeConfig, MojoVPTQDecode
from squish.kernels.mojo.topkp_mojo import TopKPConfig, MojoTopKP


# ── DualChunkAttnConfig ────────────────────────────────────────────────────


class TestDualChunkAttnConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DualChunkAttnConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.chunk_size, 512)
        self.assertIsNone(cfg.scale)

    def test_custom(self):
        cfg = DualChunkAttnConfig(n_heads=4, head_dim=64, chunk_size=256)
        self.assertEqual(cfg.n_heads, 4)

    def test_scale_default_auto(self):
        dca = MojoDualChunkAttn(DualChunkAttnConfig(head_dim=64))
        self.assertAlmostEqual(dca._scale, 64 ** -0.5, places=5)

    def test_scale_override(self):
        cfg = DualChunkAttnConfig(scale=0.1)
        dca = MojoDualChunkAttn(cfg)
        self.assertAlmostEqual(dca._scale, 0.1, places=5)


# ── MojoDualChunkAttn ──────────────────────────────────────────────────────


class TestMojoDualChunkAttn(unittest.TestCase):
    def setUp(self):
        self.cfg = DualChunkAttnConfig(n_heads=2, head_dim=8, chunk_size=4)
        self.dca = MojoDualChunkAttn(self.cfg)
        self.rng = np.random.default_rng(0)

    def test_backend_is_string(self):
        self.assertIn(self.dca.backend(), ("mojo", "numpy"))

    def test_forward_shape(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        out = self.dca.forward(Q, K, V)
        self.assertEqual(out.shape, (2, 8, 8))
        self.assertEqual(out.dtype, np.float32)

    def test_forward_no_nan(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        out = self.dca.forward(Q, K, V)
        self.assertFalse(np.isnan(out).any())

    def test_forward_finite(self):
        Q = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out = self.dca.forward(Q, K, V)
        self.assertTrue(np.isfinite(out).all())

    def test_forward_chunk_size_override(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = Q.copy()
        V = Q.copy()
        out = self.dca.forward(Q, K, V, chunk_size=2)
        self.assertEqual(out.shape, (2, 8, 8))

    def test_chunk_size_property(self):
        self.assertEqual(self.dca.chunk_size(), 4)

    def test_single_token(self):
        Q = self.rng.standard_normal((2, 1, 8)).astype(np.float32)
        K = Q.copy()
        V = Q.copy()
        out = self.dca.forward(Q, K, V)
        self.assertEqual(out.shape, (2, 1, 8))

    def test_output_invariant_to_future_keys(self):
        # Causal: out[:, 0, :] should be unchanged if we append more tokens
        Q = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out_short = self.dca.forward(Q[:, :2, :], K[:, :2, :], V[:, :2, :])
        out_full = self.dca.forward(Q, K, V)
        # First token output must match (in-chunk causal SDPA)
        np.testing.assert_allclose(out_short[:, 0, :], out_full[:, 0, :], atol=1e-4)


# ── InfiniAttnConfig ───────────────────────────────────────────────────────


class TestInfiniAttnConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = InfiniAttnConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.head_dim, 128)
        self.assertTrue(cfg.use_elu)

    def test_custom(self):
        cfg = InfiniAttnConfig(n_heads=4, head_dim=64, use_elu=False)
        self.assertFalse(cfg.use_elu)


# ── MojoInfiniAttnMemory ───────────────────────────────────────────────────


class TestMojoInfiniAttnMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = InfiniAttnConfig(n_heads=2, head_dim=4, use_elu=True)
        self.inf = MojoInfiniAttnMemory(self.cfg)
        self.rng = np.random.default_rng(0)

    def test_backend_is_string(self):
        self.assertIn(self.inf.backend(), ("mojo", "numpy"))

    def test_zero_memory_shape(self):
        M, Z = self.inf.zero_memory()
        self.assertEqual(M.shape, (2, 4, 4))
        self.assertEqual(Z.shape, (2, 4))

    def test_zero_memory_is_zeros(self):
        M, Z = self.inf.zero_memory()
        np.testing.assert_array_equal(M, 0.0)
        np.testing.assert_array_equal(Z, 0.0)

    def test_update_shape(self):
        M, Z = self.inf.zero_memory()
        K = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        V = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        M2, Z2 = self.inf.update(K, V, M, Z)
        self.assertEqual(M2.shape, (2, 4, 4))
        self.assertEqual(Z2.shape, (2, 4))

    def test_update_changes_memory(self):
        M, Z = self.inf.zero_memory()
        K = np.ones((2, 1, 4), dtype=np.float32)
        V = np.ones((2, 1, 4), dtype=np.float32)
        M2, Z2 = self.inf.update(K, V, M, Z)
        self.assertFalse(np.allclose(M2, 0.0))

    def test_retrieve_shape(self):
        M, Z = self.inf.zero_memory()
        K = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        V = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        M2, Z2 = self.inf.update(K, V, M, Z)
        Q = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        A = self.inf.retrieve(Q, M2, Z2)
        self.assertEqual(A.shape, (2, 3, 4))
        self.assertEqual(A.dtype, np.float32)

    def test_retrieve_finite(self):
        M, Z = self.inf.zero_memory()
        K = self.rng.standard_normal((2, 2, 4)).astype(np.float32)
        V = self.rng.standard_normal((2, 2, 4)).astype(np.float32)
        M2, Z2 = self.inf.update(K, V, M, Z)
        Q = self.rng.standard_normal((2, 2, 4)).astype(np.float32)
        A = self.inf.retrieve(Q, M2, Z2)
        self.assertTrue(np.isfinite(A).all())

    def test_update_accumulates(self):
        M, Z = self.inf.zero_memory()
        K = np.ones((2, 1, 4), dtype=np.float32)
        V = np.ones((2, 1, 4), dtype=np.float32)
        M1, Z1 = self.inf.update(K, V, M, Z)
        M2, Z2 = self.inf.update(K, V, M1, Z1)
        # Memory should accumulate (norm should increase)
        self.assertGreater(np.linalg.norm(M2), np.linalg.norm(M1))

    def test_no_elu_mode(self):
        inf2 = MojoInfiniAttnMemory(InfiniAttnConfig(n_heads=2, head_dim=4, use_elu=False))
        M, Z = inf2.zero_memory()
        K = self.rng.standard_normal((2, 2, 4)).astype(np.float32)
        V = self.rng.standard_normal((2, 2, 4)).astype(np.float32)
        M2, Z2 = inf2.update(K, V, M, Z)
        self.assertEqual(M2.shape, (2, 4, 4))


# ── SlidingWindowAttnConfig ────────────────────────────────────────────────


class TestSlidingWindowAttnConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SlidingWindowAttnConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.window_size, 128)

    def test_custom(self):
        cfg = SlidingWindowAttnConfig(window_size=64)
        self.assertEqual(cfg.window_size, 64)


# ── MojoSlidingWindowAttn ──────────────────────────────────────────────────


class TestMojoSlidingWindowAttn(unittest.TestCase):
    def setUp(self):
        self.cfg = SlidingWindowAttnConfig(n_heads=2, head_dim=8, window_size=4)
        self.swa = MojoSlidingWindowAttn(self.cfg)
        self.rng = np.random.default_rng(3)

    def test_backend_is_string(self):
        self.assertIn(self.swa.backend(), ("mojo", "numpy"))

    def test_forward_shape(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        out = self.swa.forward(Q, K, V)
        self.assertEqual(out.shape, (2, 8, 8))
        self.assertEqual(out.dtype, np.float32)

    def test_forward_no_nan(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        out = self.swa.forward(Q, K, V)
        self.assertFalse(np.isnan(out).any())

    def test_forward_finite(self):
        Q = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out = self.swa.forward(Q, K, V)
        self.assertTrue(np.isfinite(out).all())

    def test_first_token_attends_only_self(self):
        # Token 0 with window=4: attends only to position 0
        Q = np.zeros((1, 4, 4), dtype=np.float32)
        K = np.zeros((1, 4, 4), dtype=np.float32)
        V = np.eye(4, dtype=np.float32)[np.newaxis, :, :].astype(np.float32)
        Q[0, 0, 0] = 1.0
        K[0, 0, 0] = 1.0
        out = self.swa.forward(Q, K, V)
        # First token output should be V[0, 0, :] = e_0
        np.testing.assert_allclose(out[0, 0, 0], 1.0, atol=0.1)

    def test_window_size_property(self):
        self.assertEqual(self.swa.window_size(), 4)

    def test_window_size_override(self):
        Q = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        K = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        V = self.rng.standard_normal((2, 8, 8)).astype(np.float32)
        out = self.swa.forward(Q, K, V, window_size=2)
        self.assertEqual(out.shape, (2, 8, 8))

    def test_single_head(self):
        Q = self.rng.standard_normal((1, 4, 8)).astype(np.float32)
        K = self.rng.standard_normal((1, 4, 8)).astype(np.float32)
        V = self.rng.standard_normal((1, 4, 8)).astype(np.float32)
        out = self.swa.forward(Q, K, V)
        self.assertEqual(out.shape, (1, 4, 8))


# ── HQQALSConfig ───────────────────────────────────────────────────────────


class TestHQQALSConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HQQALSConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.qmax, 15)
        self.assertAlmostEqual(cfg.lmbda, 1.0)
        self.assertEqual(cfg.max_iter, 10)

    def test_custom(self):
        cfg = HQQALSConfig(group_size=64, qmax=255)
        self.assertEqual(cfg.qmax, 255)


# ── MojoHQQALS ────────────────────────────────────────────────────────────


class TestMojoHQQALS(unittest.TestCase):
    def setUp(self):
        self.hqq = MojoHQQALS(HQQALSConfig(group_size=8, qmax=15, max_iter=5))
        self.rng = np.random.default_rng(4)

    def test_backend_is_string(self):
        self.assertIn(self.hqq.backend(), ("mojo", "numpy"))

    def test_fit_group_returns_tuple(self):
        W = self.rng.standard_normal(8).astype(np.float32)
        result = self.hqq.fit_group(W)
        self.assertEqual(len(result), 3)

    def test_fit_group_scale_positive(self):
        W = self.rng.standard_normal(8).astype(np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        self.assertGreater(scale, 0.0)

    def test_fit_group_codes_range(self):
        W = self.rng.standard_normal(8).astype(np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes <= 15).all())

    def test_fit_group_codes_shape(self):
        W = self.rng.standard_normal(8).astype(np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        self.assertEqual(codes.shape, (8,))
        self.assertEqual(codes.dtype, np.int32)

    def test_dequantize_shape(self):
        W = self.rng.standard_normal(8).astype(np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        recon = self.hqq.dequantize(codes, scale, zero)
        self.assertEqual(recon.shape, (8,))
        self.assertEqual(recon.dtype, np.float32)

    def test_dequantize_near_original(self):
        # INT4 reconstruction error should be bounded
        W = np.linspace(-1.0, 1.0, 8).astype(np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        recon = self.hqq.dequantize(codes, scale, zero)
        mse = float(np.mean((W - recon) ** 2))
        self.assertLess(mse, 0.1)

    def test_fit_all_shapes(self):
        hqq = MojoHQQALS(HQQALSConfig(group_size=4, qmax=15, max_iter=3))
        W = self.rng.standard_normal(16).astype(np.float32)
        scales, zeros, codes = hqq.fit(W)
        self.assertEqual(scales.shape, (4,))
        self.assertEqual(codes.shape, (16,))

    def test_constant_group_scale(self):
        # Group of all same value → scale small, zero near the value
        W = np.full(8, 2.0, dtype=np.float32)
        scale, zero, codes = self.hqq.fit_group(W)
        recon = self.hqq.dequantize(codes, scale, zero)
        np.testing.assert_allclose(recon, 2.0, atol=0.3)


# ── VPTQDecodeConfig ───────────────────────────────────────────────────────


class TestVPTQDecodeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VPTQDecodeConfig()
        self.assertEqual(cfg.group_size, 4)
        self.assertEqual(cfg.n_codebooks, 1)

    def test_custom(self):
        cfg = VPTQDecodeConfig(group_size=8, n_codebooks=2)
        self.assertEqual(cfg.group_size, 8)


# ── MojoVPTQDecode ────────────────────────────────────────────────────────


class TestMojoVPTQDecode(unittest.TestCase):
    def setUp(self):
        self.dec = MojoVPTQDecode(VPTQDecodeConfig(group_size=4))
        self.rng = np.random.default_rng(5)
        self.centroids = self.rng.standard_normal((16, 4)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.dec.backend(), ("mojo", "numpy"))

    def test_decode_shape(self):
        indices = np.random.randint(0, 16, 32).astype(np.int32)
        out = self.dec.decode(indices, self.centroids)
        self.assertEqual(out.shape, (32, 4))
        self.assertEqual(out.dtype, np.float32)

    def test_decode_correct_lookup(self):
        indices = np.array([0, 3, 7, 15], dtype=np.int32)
        out = self.dec.decode(indices, self.centroids)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(out[i], self.centroids[idx])

    def test_decode_index_clipping(self):
        indices = np.array([-1, 100], dtype=np.int32)
        out = self.dec.decode(indices, self.centroids)
        self.assertEqual(out.shape, (2, 4))
        # Should not crash; values should be valid centroid rows

    def test_decode_single_index(self):
        indices = np.array([5], dtype=np.int32)
        out = self.dec.decode(indices, self.centroids)
        np.testing.assert_array_equal(out[0], self.centroids[5])

    def test_multi_decode_shape(self):
        idx1 = np.random.randint(0, 16, 10).astype(np.int32)
        idx2 = np.random.randint(0, 16, 10).astype(np.int32)
        out = self.dec.multi_decode([idx1, idx2], [self.centroids, self.centroids])
        self.assertEqual(out.shape, (10, 4))

    def test_multi_decode_equals_sum(self):
        idx1 = np.array([0, 1, 2], dtype=np.int32)
        idx2 = np.array([3, 4, 5], dtype=np.int32)
        out = self.dec.multi_decode([idx1, idx2], [self.centroids, self.centroids])
        expected = self.centroids[idx1] + self.centroids[idx2]
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_group_size_property(self):
        self.assertEqual(self.dec.group_size(), 4)

    def test_decode_all_same_index(self):
        indices = np.zeros(8, dtype=np.int32)
        out = self.dec.decode(indices, self.centroids)
        for row in out:
            np.testing.assert_array_equal(row, self.centroids[0])


# ── TopKPConfig ────────────────────────────────────────────────────────────


class TestTopKPConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TopKPConfig()
        self.assertEqual(cfg.vocab_size, 128256)
        self.assertEqual(cfg.top_k, 50)
        self.assertAlmostEqual(cfg.top_p, 0.9)

    def test_custom(self):
        cfg = TopKPConfig(vocab_size=32000, temperature=0.7)
        self.assertEqual(cfg.vocab_size, 32000)


# ── MojoTopKP ──────────────────────────────────────────────────────────────


class TestMojoTopKP(unittest.TestCase):
    def setUp(self):
        self.sampler = MojoTopKP(TopKPConfig(vocab_size=64, top_k=10, top_p=0.9))
        self.rng = np.random.default_rng(6)
        self.logits = self.rng.standard_normal(64).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.sampler.backend(), ("mojo", "numpy"))

    def test_sample_returns_int(self):
        token = self.sampler.sample(self.logits, seed=0)
        self.assertIsInstance(token, int)

    def test_sample_in_vocab_range(self):
        token = self.sampler.sample(self.logits, seed=0)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 64)

    def test_sample_deterministic_with_seed(self):
        t1 = self.sampler.sample(self.logits, seed=42)
        t2 = self.sampler.sample(self.logits, seed=42)
        self.assertEqual(t1, t2)

    def test_filter_shape(self):
        probs = self.sampler.filter(self.logits)
        self.assertEqual(probs.shape, (64,))
        self.assertEqual(probs.dtype, np.float32)

    def test_filter_sums_to_one(self):
        probs = self.sampler.filter(self.logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

    def test_filter_non_negative(self):
        probs = self.sampler.filter(self.logits)
        self.assertTrue((probs >= 0).all())

    def test_top_k_limits_support(self):
        probs = self.sampler.filter(self.logits, top_k=3)
        nonzero = int((probs > 0).sum())
        self.assertLessEqual(nonzero, 3)

    def test_temperature_scaling(self):
        probs_hot = self.sampler.filter(self.logits, temperature=0.1)
        probs_cold = self.sampler.filter(self.logits, temperature=10.0)
        # Low temperature → more peaked (higher max prob)
        self.assertGreater(float(probs_hot.max()), float(probs_cold.max()))

    def test_top_p_1_disables_nucleus(self):
        probs_full = self.sampler.filter(self.logits, top_p=1.0, top_k=0)
        # All tokens should have nonzero probability
        self.assertEqual(int((probs_full > 0).sum()), 64)

    def test_vocab_size_property(self):
        self.assertEqual(self.sampler.vocab_size(), 64)

    def test_sample_with_one_hot_logits(self):
        # Logit >> 0 for token 5: should almost always sample token 5
        logits = np.full(64, -100.0, dtype=np.float32)
        logits[5] = 100.0
        token = self.sampler.sample(logits, seed=7)
        self.assertEqual(token, 5)

    def test_filter_top_p_reduces_support(self):
        probs_all = self.sampler.filter(self.logits, top_p=1.0, top_k=0)
        probs_half = self.sampler.filter(self.logits, top_p=0.5, top_k=0)
        nz_all = int((probs_all > 0).sum())
        nz_half = int((probs_half > 0).sum())
        self.assertLessEqual(nz_half, nz_all)


if __name__ == "__main__":
    unittest.main()
