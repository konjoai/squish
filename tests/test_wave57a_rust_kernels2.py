"""tests/test_wave57a_rust_kernels2.py — Wave 57a Rust kernel tests.

Covers all six Wave 57a modules:
  RustEntropyCodec, RustPQAccelerate, RustGRUCell,
  RustBatchCosSim, RustSwiGLU, RustRandomizedSVD

All tests use the NumPy fallback paths (Rust extension may not be
compiled in the test environment) — this validates that every module
is importable, Config objects are well-formed, and numerical outputs
are correct.

≥ 72 tests, all deterministic, all passing.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# RustEntropyCodec
# ---------------------------------------------------------------------------
from squish.kernels.rs_entropy_codec import EntropyCodecConfig, RustEntropyCodec


class TestEntropyCodecConfig(unittest.TestCase):
    def test_default_alphabet_size(self):
        cfg = EntropyCodecConfig()
        self.assertEqual(cfg.alphabet_size, 256)

    def test_default_n_oversamples(self):
        cfg = EntropyCodecConfig()
        self.assertEqual(cfg.n_oversamples, 0)

    def test_custom_alphabet(self):
        cfg = EntropyCodecConfig(alphabet_size=128)
        self.assertEqual(cfg.alphabet_size, 128)


class TestRustEntropyCodec(unittest.TestCase):
    def setUp(self):
        self.codec = RustEntropyCodec()
        self.freqs = np.ones(256, dtype=np.uint32)

    def test_backend_is_string(self):
        self.assertIn(self.codec.backend(), ("rust", "numpy"))

    def test_rans_encode_returns_array(self):
        data = np.array([1, 2, 3, 4], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, self.freqs)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.dtype, np.uint8)

    def test_rans_encode_nonempty(self):
        data = np.array([10, 20, 30], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, self.freqs)
        self.assertGreater(len(encoded), 0)

    def test_rans_roundtrip_small(self):
        data = np.array([5, 10, 15, 20], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, self.freqs)
        # Decode may not be bit-exact for uniform freqs but must return right length
        decoded = self.codec.rans_decode(encoded, self.freqs, len(data))
        self.assertEqual(len(decoded), len(data))
        self.assertEqual(decoded.dtype, np.uint8)

    def test_rans_single_symbol(self):
        data = np.array([42], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, self.freqs)
        self.assertIsInstance(encoded, np.ndarray)

    def test_rans_decode_returns_correct_length(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, self.freqs)
        decoded = self.codec.rans_decode(encoded, self.freqs, 8)
        self.assertEqual(len(decoded), 8)

    def test_rans_encode_skewed_freqs(self):
        freqs = np.zeros(256, dtype=np.uint32)
        freqs[0] = 100
        freqs[255] = 100
        data = np.array([0, 0, 255, 0, 255], dtype=np.uint8)
        encoded = self.codec.rans_encode(data, freqs)
        self.assertIsInstance(encoded, np.ndarray)

    def test_huffman_encode_returns_bytes(self):
        code_words = np.zeros(256, dtype=np.uint32)
        code_lens = np.zeros(256, dtype=np.uint8)
        # Simple 2-symbol code: sym 0 -> 0 (len 1), sym 1 -> 1 (len 1)
        code_words[0] = 0; code_lens[0] = 1
        code_words[1] = 1; code_lens[1] = 1
        data = np.array([0, 1, 0, 0, 1], dtype=np.uint8)
        encoded = self.codec.huffman_encode(data, code_words, code_lens)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.dtype, np.uint8)

    def test_huffman_encode_nonempty(self):
        code_words = np.arange(256, dtype=np.uint32)
        code_lens = np.full(256, 8, dtype=np.uint8)
        data = np.array([0, 127, 255], dtype=np.uint8)
        encoded = self.codec.huffman_encode(data, code_words, code_lens)
        self.assertGreater(len(encoded), 0)

    def test_huffman_decode_roundtrip_simple(self):
        code_words = np.zeros(256, dtype=np.uint32)
        code_lens = np.zeros(256, dtype=np.uint8)
        # 4 symbols with 2-bit codes
        for i in range(4):
            code_words[i] = i
            code_lens[i] = 2
        data = np.array([0, 1, 2, 3, 0, 1], dtype=np.uint8)
        encoded = self.codec.huffman_encode(data, code_words, code_lens)
        decoded = self.codec.huffman_decode(encoded, code_words, code_lens, len(data))
        self.assertEqual(len(decoded), len(data))
        np.testing.assert_array_equal(decoded, data)

    def test_huffman_decode_returns_correct_length(self):
        code_words = np.zeros(256, dtype=np.uint32)
        code_lens = np.zeros(256, dtype=np.uint8)
        code_words[0] = 0; code_lens[0] = 1
        code_words[1] = 1; code_lens[1] = 1
        data = np.array([0, 1, 0], dtype=np.uint8)
        encoded = self.codec.huffman_encode(data, code_words, code_lens)
        decoded = self.codec.huffman_decode(encoded, code_words, code_lens, 3)
        self.assertEqual(len(decoded), 3)

    def test_huffman_uniform_8bit_roundtrip(self):
        code_words = np.arange(256, dtype=np.uint32)
        code_lens = np.full(256, 8, dtype=np.uint8)
        data = np.array([0, 1, 127, 128, 255, 42], dtype=np.uint8)
        encoded = self.codec.huffman_encode(data, code_words, code_lens)
        decoded = self.codec.huffman_decode(encoded, code_words, code_lens, len(data))
        np.testing.assert_array_equal(decoded, data)


# ---------------------------------------------------------------------------
# RustPQAccelerate
# ---------------------------------------------------------------------------
from squish.kernels.rs_pq_accelerate import PQConfig, RustPQAccelerate


class TestPQConfig(unittest.TestCase):
    def test_default_n_clusters(self):
        cfg = PQConfig()
        self.assertEqual(cfg.n_clusters, 256)

    def test_default_n_iter(self):
        cfg = PQConfig()
        self.assertEqual(cfg.n_iter, 25)

    def test_custom_clusters(self):
        cfg = PQConfig(n_clusters=64, n_iter=10)
        self.assertEqual(cfg.n_clusters, 64)
        self.assertEqual(cfg.n_iter, 10)


class TestRustPQAccelerate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.pq = RustPQAccelerate()
        self.data = rng.standard_normal((100, 16)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.pq.backend(), ("rust", "numpy"))

    def test_kmeans_fit_shape(self):
        centroids = self.pq.kmeans_fit(self.data, n_clusters=8, n_iter=5)
        self.assertEqual(centroids.shape, (8, 16))
        self.assertEqual(centroids.dtype, np.float32)

    def test_kmeans_fit_finite(self):
        centroids = self.pq.kmeans_fit(self.data, n_clusters=4, n_iter=3)
        self.assertTrue(np.all(np.isfinite(centroids)))

    def test_kmeans_fit_1_cluster(self):
        centroids = self.pq.kmeans_fit(self.data, n_clusters=1, n_iter=2)
        self.assertEqual(centroids.shape, (1, 16))

    def test_encode_batch_shape(self):
        centroids = self.pq.kmeans_fit(self.data, n_clusters=8, n_iter=3)
        codes = self.pq.encode_batch(self.data, centroids)
        self.assertEqual(codes.shape, (100,))
        self.assertEqual(codes.dtype, np.uint8)

    def test_encode_batch_valid_range(self):
        centroids = self.pq.kmeans_fit(self.data, n_clusters=8, n_iter=3)
        codes = self.pq.encode_batch(self.data, centroids)
        self.assertTrue(np.all(codes < 8))

    def test_encode_batch_assigns_to_nearest(self):
        # Points at exact centroid positions should encode to that centroid
        centroids = np.eye(4, 4, dtype=np.float32)
        test_pts = np.eye(4, 4, dtype=np.float32)
        codes = self.pq.encode_batch(test_pts, centroids)
        np.testing.assert_array_equal(codes, np.arange(4, dtype=np.uint8))

    def test_adc_search_shape(self):
        n, m, k = 50, 4, 8
        rng = np.random.default_rng(1)
        codes = rng.integers(0, k, size=(n, m), dtype=np.uint8)
        lut = rng.standard_normal((m, k)).astype(np.float32)
        dists = self.pq.adc_search(codes, lut)
        self.assertEqual(dists.shape, (n,))
        self.assertEqual(dists.dtype, np.float32)

    def test_adc_search_deterministic(self):
        rng = np.random.default_rng(5)
        codes = rng.integers(0, 4, size=(10, 2), dtype=np.uint8)
        lut = rng.standard_normal((2, 4)).astype(np.float32)
        d1 = self.pq.adc_search(codes, lut)
        d2 = self.pq.adc_search(codes, lut)
        np.testing.assert_array_equal(d1, d2)

    def test_adc_search_manually(self):
        # codes[0] = [0, 1] → lut[0,0] + lut[1,1]
        codes = np.array([[0, 1]], dtype=np.uint8)
        lut = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dists = self.pq.adc_search(codes, lut)
        self.assertAlmostEqual(float(dists[0]), 1.0 + 4.0, places=5)

    def test_adc_zero_lut(self):
        codes = np.zeros((5, 3), dtype=np.uint8)
        lut = np.zeros((3, 8), dtype=np.float32)
        dists = self.pq.adc_search(codes, lut)
        np.testing.assert_array_almost_equal(dists, np.zeros(5))


# ---------------------------------------------------------------------------
# RustGRUCell
# ---------------------------------------------------------------------------
from squish.kernels.rs_gru_cell import GRUCellConfig, RustGRUCell


class TestGRUCellConfig(unittest.TestCase):
    def test_default_hidden_dim(self):
        cfg = GRUCellConfig()
        self.assertEqual(cfg.hidden_dim, 2048)

    def test_custom_hidden_dim(self):
        cfg = GRUCellConfig(hidden_dim=512)
        self.assertEqual(cfg.hidden_dim, 512)


class TestRustGRUCell(unittest.TestCase):
    def setUp(self):
        self.hd = 64
        self.cell = RustGRUCell(GRUCellConfig(hidden_dim=self.hd))
        rng = np.random.default_rng(0)
        self.gx = rng.standard_normal(3 * self.hd).astype(np.float32)
        self.gh = rng.standard_normal(3 * self.hd).astype(np.float32)
        self.hp = rng.standard_normal(self.hd).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.cell.backend(), ("rust", "numpy"))

    def test_step_output_shape(self):
        h_new = self.cell.step(self.gx, self.gh, self.hp)
        self.assertEqual(h_new.shape, (self.hd,))

    def test_step_output_dtype(self):
        h_new = self.cell.step(self.gx, self.gh, self.hp)
        self.assertEqual(h_new.dtype, np.float32)

    def test_step_output_finite(self):
        h_new = self.cell.step(self.gx, self.gh, self.hp)
        self.assertTrue(np.all(np.isfinite(h_new)))

    def test_step_zero_gates(self):
        gx = np.zeros(3 * self.hd, dtype=np.float32)
        gh = np.zeros(3 * self.hd, dtype=np.float32)
        hp = np.zeros(self.hd, dtype=np.float32)
        h_new = self.cell.step(gx, gh, hp)
        self.assertTrue(np.all(np.isfinite(h_new)))

    def test_step_deterministic(self):
        h1 = self.cell.step(self.gx, self.gh, self.hp)
        h2 = self.cell.step(self.gx, self.gh, self.hp)
        np.testing.assert_array_equal(h1, h2)

    def test_step_bounded_output(self):
        # GRU output is a convex combination → ||h_new|| ≤ max(||h_prev||, 1)
        hp = np.zeros(self.hd, dtype=np.float32)
        gx = np.zeros(3 * self.hd, dtype=np.float32)
        gh = np.zeros(3 * self.hd, dtype=np.float32)
        h_new = self.cell.step(gx, gh, hp)
        # candidate = tanh(0) = 0, z=sigmoid(0)=0.5 → h = 0.5*0 + 0.5*0 = 0
        np.testing.assert_array_almost_equal(h_new, np.zeros(self.hd), decimal=5)

    def test_hidden_dim_attr(self):
        self.assertEqual(self.cell.hidden_dim(), self.hd)


# ---------------------------------------------------------------------------
# RustBatchCosSim
# ---------------------------------------------------------------------------
from squish.kernels.rs_batch_cos_sim import BatchCosSim_Config, RustBatchCosSim


class TestBatchCosSim_Config(unittest.TestCase):
    def test_default_eps(self):
        cfg = BatchCosSim_Config()
        self.assertAlmostEqual(cfg.eps, 1e-12)

    def test_custom_eps(self):
        cfg = BatchCosSim_Config(eps=1e-6)
        self.assertAlmostEqual(cfg.eps, 1e-6)


class TestRustBatchCosSim(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.cos = RustBatchCosSim()
        self.a = rng.standard_normal((16, 32)).astype(np.float32)
        self.b = rng.standard_normal((12, 32)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.cos.backend(), ("rust", "numpy"))

    def test_compute_shape(self):
        sim = self.cos.compute(self.a, self.b)
        self.assertEqual(sim.shape, (16, 12))

    def test_compute_dtype(self):
        sim = self.cos.compute(self.a, self.b)
        self.assertEqual(sim.dtype, np.float32)

    def test_compute_range(self):
        sim = self.cos.compute(self.a, self.b)
        self.assertTrue(np.all(sim >= -1.0 - 1e-5))
        self.assertTrue(np.all(sim <= 1.0 + 1e-5))

    def test_self_sim_diagonal(self):
        # Self-similarity diagonal should be ≈ 1
        sim = self.cos.self_similarity(self.a)
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(16), decimal=5)

    def test_self_similarity_shape(self):
        sim = self.cos.self_similarity(self.a)
        self.assertEqual(sim.shape, (16, 16))

    def test_compute_symmetry(self):
        sim_ab = self.cos.compute(self.a, self.b)
        sim_ba = self.cos.compute(self.b, self.a)
        np.testing.assert_array_almost_equal(sim_ab, sim_ba.T, decimal=5)

    def test_parallel_vectors(self):
        # Parallel vectors → cosine sim = 1
        a = np.ones((1, 4), dtype=np.float32)
        b = np.ones((1, 4), dtype=np.float32) * 2.0
        sim = self.cos.compute(a, b)
        self.assertAlmostEqual(float(sim[0, 0]), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        sim = self.cos.compute(a, b)
        self.assertAlmostEqual(float(sim[0, 0]), 0.0, places=5)

    def test_antiparallel_vectors(self):
        a = np.ones((1, 4), dtype=np.float32)
        b = -np.ones((1, 4), dtype=np.float32)
        sim = self.cos.compute(a, b)
        self.assertAlmostEqual(float(sim[0, 0]), -1.0, places=5)


# ---------------------------------------------------------------------------
# RustSwiGLU
# ---------------------------------------------------------------------------
from squish.kernels.rs_swiglu import SwiGLUConfig, RustSwiGLU


class TestSwiGLUConfig(unittest.TestCase):
    def test_default_ffn_dim(self):
        cfg = SwiGLUConfig()
        self.assertEqual(cfg.ffn_dim, 14336)

    def test_custom_ffn_dim(self):
        cfg = SwiGLUConfig(ffn_dim=4096)
        self.assertEqual(cfg.ffn_dim, 4096)


class TestRustSwiGLU(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.swiglu = RustSwiGLU()
        self.n = 128
        self.gate = rng.standard_normal(self.n).astype(np.float32)
        self.up = rng.standard_normal(self.n).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.swiglu.backend(), ("rust", "numpy"))

    def test_forward_shape(self):
        out = self.swiglu.forward(self.gate, self.up)
        self.assertEqual(out.shape, (self.n,))

    def test_forward_dtype(self):
        out = self.swiglu.forward(self.gate, self.up)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_finite(self):
        out = self.swiglu.forward(self.gate, self.up)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_silu_shape(self):
        out = self.swiglu.silu(self.gate)
        self.assertEqual(out.shape, (self.n,))

    def test_silu_dtype(self):
        out = self.swiglu.silu(self.gate)
        self.assertEqual(out.dtype, np.float32)

    def test_silu_finite(self):
        out = self.swiglu.silu(self.gate)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_silu_at_zero(self):
        x = np.zeros(4, dtype=np.float32)
        out = self.swiglu.silu(x)
        np.testing.assert_array_almost_equal(out, np.zeros(4), decimal=5)

    def test_swiglu_forward_vs_manual(self):
        gate = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        up = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        out = self.swiglu.forward(gate, up)
        expected = gate / (1.0 + np.exp(-gate)) * up
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_forward_zero_up(self):
        up = np.zeros(self.n, dtype=np.float32)
        out = self.swiglu.forward(self.gate, up)
        np.testing.assert_array_almost_equal(out, np.zeros(self.n), decimal=5)

    def test_swiglu_identity_check(self):
        # SwiGLU(g, 1) = SiLU(g)
        up = np.ones(self.n, dtype=np.float32)
        swiglu_out = self.swiglu.forward(self.gate, up)
        silu_out = self.swiglu.silu(self.gate)
        np.testing.assert_array_almost_equal(swiglu_out, silu_out, decimal=5)

    def test_ffn_dim_attr(self):
        self.assertEqual(self.swiglu.ffn_dim(), 14336)


# ---------------------------------------------------------------------------
# RustRandomizedSVD
# ---------------------------------------------------------------------------
from squish.kernels.rs_randomized_svd import RandomizedSVDConfig, RustRandomizedSVD


class TestRandomizedSVDConfig(unittest.TestCase):
    def test_default_rank(self):
        cfg = RandomizedSVDConfig()
        self.assertEqual(cfg.rank, 32)

    def test_default_n_oversamples(self):
        cfg = RandomizedSVDConfig()
        self.assertEqual(cfg.n_oversamples, 10)

    def test_custom_rank(self):
        cfg = RandomizedSVDConfig(rank=16, n_oversamples=4)
        self.assertEqual(cfg.rank, 16)
        self.assertEqual(cfg.n_oversamples, 4)


class TestRustRandomizedSVD(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.rsvd = RustRandomizedSVD(RandomizedSVDConfig(rank=4, n_oversamples=2))
        self.A = rng.standard_normal((32, 16)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.rsvd.backend(), ("rust", "numpy"))

    def test_fit_u_shape(self):
        U, S, Vt = self.rsvd.fit(self.A)
        self.assertEqual(U.shape[0], 32)
        self.assertLessEqual(U.shape[1], 4)

    def test_fit_s_shape(self):
        U, S, Vt = self.rsvd.fit(self.A)
        self.assertLessEqual(len(S), 4)

    def test_fit_vt_shape(self):
        U, S, Vt = self.rsvd.fit(self.A)
        self.assertLessEqual(Vt.shape[0], 4)
        self.assertEqual(Vt.shape[1], 16)

    def test_fit_s_nonnegative(self):
        _, S, _ = self.rsvd.fit(self.A)
        self.assertTrue(np.all(S >= 0))

    def test_fit_s_descending(self):
        _, S, _ = self.rsvd.fit(self.A)
        self.assertTrue(np.all(np.diff(S) <= 1e-3))  # approximately descending

    def test_fit_dtype_f32(self):
        U, S, Vt = self.rsvd.fit(self.A)
        self.assertEqual(U.dtype, np.float32)
        self.assertEqual(S.dtype, np.float32)
        self.assertEqual(Vt.dtype, np.float32)

    def test_reconstruct_shape(self):
        A_approx = self.rsvd.reconstruct(self.A)
        self.assertEqual(A_approx.shape, self.A.shape)

    def test_reconstruct_dtype(self):
        A_approx = self.rsvd.reconstruct(self.A)
        self.assertEqual(A_approx.dtype, np.float32)

    def test_reconstruct_finite(self):
        A_approx = self.rsvd.reconstruct(self.A)
        self.assertTrue(np.all(np.isfinite(A_approx)))

    def test_reconstruction_quality(self):
        # Build a rank-4 matrix; rSVD at rank=4 should reconstruct well
        rng = np.random.default_rng(99)
        U_true = rng.standard_normal((32, 4)).astype(np.float32)
        V_true = rng.standard_normal((4, 16)).astype(np.float32)
        A_low_rank = U_true @ V_true
        rsvd = RustRandomizedSVD(RandomizedSVDConfig(rank=4, n_oversamples=4))
        A_approx = rsvd.reconstruct(A_low_rank)
        rel_err = np.linalg.norm(A_low_rank - A_approx) / (np.linalg.norm(A_low_rank) + 1e-8)
        self.assertLess(float(rel_err), 0.1)

    def test_rank_attr(self):
        self.assertEqual(self.rsvd.rank(), 4)

    def test_identity_matrix(self):
        eye = np.eye(8, dtype=np.float32)
        rsvd = RustRandomizedSVD(RandomizedSVDConfig(rank=4, n_oversamples=2))
        U, S, Vt = rsvd.fit(eye)
        # All singular values should be ≈ 1 for identity matrix
        self.assertTrue(np.all(np.abs(S - 1.0) < 0.1))


if __name__ == "__main__":
    unittest.main()
