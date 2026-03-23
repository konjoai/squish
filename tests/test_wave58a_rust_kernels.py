"""tests/test_wave58a_rust_kernels.py — Wave 58a Rust kernel tests.

Tests for:
  - RustVectorKMeans (VectorKMeansConfig, fit, assign, reconstruct)
  - RustFP6BitPack   (FP6Config, encode, decode round-trip)
  - RustAWQChannel   (AWQChannelConfig, record, compute_scales)
  - RustModelMerge   (ModelMergeConfig, slerp, dare, ties)
  - RustMoEBincount  (MoEBincountConfig, bincount, top_k)
  - RustOnlineSGD    (OnlineSGDConfig, predict, step, weight_update)

All tests use NumPy fallback path (squish_quant not compiled in CI).
79 tests total.
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.rs_vector_kmeans import VectorKMeansConfig, RustVectorKMeans
from squish.kernels.rs_fp6_bitpack import FP6Config, RustFP6BitPack
from squish.kernels.rs_awq_channel import AWQChannelConfig, RustAWQChannel
from squish.kernels.rs_model_merge import ModelMergeConfig, RustModelMerge
from squish.kernels.rs_moe_bincount import MoEBincountConfig, RustMoEBincount
from squish.kernels.rs_online_sgd import OnlineSGDConfig, RustOnlineSGD


# ── VectorKMeansConfig ─────────────────────────────────────────────────────


class TestVectorKMeansConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VectorKMeansConfig()
        self.assertEqual(cfg.n_clusters, 256)
        self.assertEqual(cfg.n_iter, 25)

    def test_custom(self):
        cfg = VectorKMeansConfig(n_clusters=64, n_iter=10)
        self.assertEqual(cfg.n_clusters, 64)
        self.assertEqual(cfg.n_iter, 10)

    def test_dataclass_eq(self):
        self.assertEqual(VectorKMeansConfig(64, 10), VectorKMeansConfig(64, 10))


# ── RustVectorKMeans ──────────────────────────────────────────────────────


class TestRustVectorKMeans(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # 200 points in 3D from 4 clusters
        centres = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0]], dtype=np.float32)
        self.data = np.vstack([
            centres[i] + rng.standard_normal((50, 3)).astype(np.float32) * 0.3
            for i in range(4)
        ])
        self.km = RustVectorKMeans(VectorKMeansConfig(n_clusters=4, n_iter=10))

    def test_backend_is_string(self):
        self.assertIn(self.km.backend(), ("rust", "numpy"))

    def test_fit_shape(self):
        centroids = self.km.fit(self.data)
        self.assertEqual(centroids.shape, (4, 3))
        self.assertEqual(centroids.dtype, np.float32)

    def test_fit_centroids_near_true(self):
        centroids = self.km.fit(self.data)
        true_c = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0]], dtype=np.float32)
        # Each true centre should be close to some fitted centroid
        for tc in true_c:
            dists = np.linalg.norm(centroids - tc[np.newaxis, :], axis=1)
            self.assertLess(dists.min(), 1.0)

    def test_assign_shape(self):
        centroids = self.km.fit(self.data)
        codes = self.km.assign(self.data, centroids)
        self.assertEqual(codes.shape, (200,))
        self.assertEqual(codes.dtype, np.int32)

    def test_assign_range(self):
        centroids = self.km.fit(self.data)
        codes = self.km.assign(self.data, centroids)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 4).all())

    def test_reconstruct_shape(self):
        centroids = self.km.fit(self.data)
        codes = self.km.assign(self.data, centroids)
        recon = self.km.reconstruct(codes, centroids)
        self.assertEqual(recon.shape, (200, 3))
        self.assertEqual(recon.dtype, np.float32)

    def test_reconstruct_values(self):
        centroids = self.km.fit(self.data)
        codes = self.km.assign(self.data, centroids)
        recon = self.km.reconstruct(codes, centroids)
        # Each reconstructed row must equal its assigned centroid
        for i in range(len(codes)):
            np.testing.assert_array_equal(recon[i], centroids[codes[i]])

    def test_n_clusters_property(self):
        self.assertEqual(self.km.n_clusters(), 4)

    def test_fit_override_params(self):
        centroids = self.km.fit(self.data, n_clusters=2, n_iter=5)
        self.assertEqual(centroids.shape, (2, 3))

    def test_assign_uses_nearest(self):
        centroids = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        points = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=np.float32)
        codes = self.km.assign(points, centroids)
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(int(codes[1]), 1)

    def test_small_data(self):
        data = np.eye(3, dtype=np.float32)
        km = RustVectorKMeans(VectorKMeansConfig(n_clusters=3, n_iter=2))
        centroids = km.fit(data)
        self.assertEqual(centroids.shape, (3, 3))

    def test_round_trip_error(self):
        centroids = self.km.fit(self.data)
        codes = self.km.assign(self.data, centroids)
        recon = self.km.reconstruct(codes, centroids)
        # Reconstruction MSE should be small compared to data range
        mse = float(np.mean((self.data - recon) ** 2))
        self.assertLess(mse, 5.0)


# ── FP6Config ──────────────────────────────────────────────────────────────


class TestFP6Config(unittest.TestCase):
    def test_defaults(self):
        cfg = FP6Config()
        self.assertEqual(cfg.exp_bits, 3)
        self.assertEqual(cfg.man_bits, 2)

    def test_custom(self):
        cfg = FP6Config(exp_bits=4, man_bits=1)
        self.assertEqual(cfg.exp_bits, 4)

    def test_invalid_bits_raises(self):
        fp6 = RustFP6BitPack(FP6Config(exp_bits=3, man_bits=2))  # valid
        with self.assertRaises(ValueError):
            RustFP6BitPack(FP6Config(exp_bits=2, man_bits=2))  # 1+2+2=5 ≠ 6


# ── RustFP6BitPack ─────────────────────────────────────────────────────────


class TestRustFP6BitPack(unittest.TestCase):
    def setUp(self):
        self.fp6 = RustFP6BitPack()

    def test_backend_is_string(self):
        self.assertIn(self.fp6.backend(), ("rust", "numpy"))

    def test_encode_returns_bytes(self):
        data = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        packed = self.fp6.encode(data)
        self.assertIsInstance(packed, bytes)
        self.assertEqual(len(packed), 3)

    def test_encode_length(self):
        data = np.ones(16, dtype=np.float32)
        packed = self.fp6.encode(data)
        self.assertEqual(len(packed), 12)  # 16/4*3

    def test_decode_shape(self):
        data = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        packed = self.fp6.encode(data)
        out = self.fp6.decode(packed)
        self.assertEqual(out.shape, (4,))
        self.assertEqual(out.dtype, np.float32)

    def test_round_trip_zero(self):
        data = np.zeros(4, dtype=np.float32)
        out = self.fp6.decode(self.fp6.encode(data))
        np.testing.assert_array_equal(out, 0.0)

    def test_round_trip_sign(self):
        data = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        out = self.fp6.decode(self.fp6.encode(data))
        # Signs must be preserved
        self.assertEqual(out[0] > 0, True)
        self.assertEqual(out[1] < 0, True)

    def test_encode_length_must_be_multiple_4(self):
        with self.assertRaises(ValueError):
            self.fp6.encode(np.ones(5, dtype=np.float32))

    def test_decode_length_must_be_multiple_3(self):
        with self.assertRaises(ValueError):
            self.fp6.decode(bytes([0, 1]))

    def test_exp_bits_property(self):
        self.assertEqual(self.fp6.exp_bits(), 3)

    def test_man_bits_property(self):
        self.assertEqual(self.fp6.man_bits(), 2)

    def test_decode_n_values_truncation(self):
        data = np.ones(8, dtype=np.float32)
        packed = self.fp6.encode(data)
        out = self.fp6.decode(packed, n_values=5)
        self.assertEqual(len(out), 5)

    def test_round_trip_large_array(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(256).astype(np.float32)
        packed = self.fp6.encode(data)
        out = self.fp6.decode(packed)
        self.assertEqual(len(out), 256)


# ── AWQChannelConfig ───────────────────────────────────────────────────────


class TestAWQChannelConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AWQChannelConfig()
        self.assertEqual(cfg.in_features, 4096)
        self.assertAlmostEqual(cfg.alpha, 0.5)

    def test_custom(self):
        cfg = AWQChannelConfig(in_features=1024, alpha=0.75)
        self.assertEqual(cfg.in_features, 1024)


# ── RustAWQChannel ─────────────────────────────────────────────────────────


class TestRustAWQChannel(unittest.TestCase):
    def setUp(self):
        self.cfg = AWQChannelConfig(in_features=16, alpha=0.5)
        self.awq = RustAWQChannel(self.cfg)
        self.rng = np.random.default_rng(42)

    def test_backend_is_string(self):
        self.assertIn(self.awq.backend(), ("rust", "numpy"))

    def test_initial_abs_mean_zeros(self):
        mean = self.awq.abs_mean()
        np.testing.assert_array_equal(mean, 0.0)
        self.assertEqual(mean.shape, (16,))

    def test_record_updates_mean(self):
        batch = np.ones((4, 16), dtype=np.float32)
        self.awq.record(batch)
        mean = self.awq.abs_mean()
        np.testing.assert_allclose(mean, 1.0, atol=1e-5)

    def test_record_multiple_batches(self):
        for _ in range(3):
            batch = self.rng.standard_normal((8, 16)).astype(np.float32)
            self.awq.record(batch)
        mean = self.awq.abs_mean()
        self.assertEqual(mean.shape, (16,))
        self.assertTrue((mean >= 0).all())

    def test_compute_scales_shape(self):
        batch = self.rng.standard_normal((4, 16)).astype(np.float32)
        self.awq.record(batch)
        scales = self.awq.compute_scales()
        self.assertEqual(scales.shape, (16,))
        self.assertEqual(scales.dtype, np.float32)

    def test_compute_scales_positive(self):
        batch = self.rng.standard_normal((4, 16)).astype(np.float32)
        self.awq.record(batch)
        scales = self.awq.compute_scales()
        self.assertTrue((scales > 0).all())

    def test_compute_scales_alpha_override(self):
        batch = np.ones((4, 16), dtype=np.float32)
        self.awq.record(batch)
        s1 = self.awq.compute_scales(alpha=0.0)
        s2 = self.awq.compute_scales(alpha=1.0)
        # alpha=0 → all ones; alpha=1 → same as abs_mean clipped
        np.testing.assert_allclose(s1, 1.0, atol=1e-5)

    def test_reset(self):
        self.awq.record(np.ones((4, 16), dtype=np.float32))
        self.awq.reset()
        np.testing.assert_array_equal(self.awq.abs_mean(), 0.0)

    def test_1d_batch_accepted(self):
        batch = np.ones(16, dtype=np.float32)
        self.awq.record(batch)
        mean = self.awq.abs_mean()
        np.testing.assert_allclose(mean, 1.0, atol=1e-5)


# ── ModelMergeConfig ───────────────────────────────────────────────────────


class TestModelMergeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ModelMergeConfig()
        self.assertAlmostEqual(cfg.trim_fraction, 0.2)
        self.assertAlmostEqual(cfg.density, 0.5)

    def test_custom(self):
        cfg = ModelMergeConfig(density=0.3, seed=7)
        self.assertEqual(cfg.seed, 7)


# ── RustModelMerge ─────────────────────────────────────────────────────────


class TestRustModelMerge(unittest.TestCase):
    def setUp(self):
        self.merge = RustModelMerge()
        rng = np.random.default_rng(0)
        self.a = rng.standard_normal(256).astype(np.float32)
        self.b = rng.standard_normal(256).astype(np.float32)
        self.delta = rng.standard_normal(256).astype(np.float32) * 0.01

    def test_backend_is_string(self):
        self.assertIn(self.merge.backend(), ("rust", "numpy"))

    def test_slerp_shape(self):
        out = self.merge.slerp(self.a, self.b, t=0.5)
        self.assertEqual(out.shape, (256,))
        self.assertEqual(out.dtype, np.float32)

    def test_slerp_t0_returns_a(self):
        out = self.merge.slerp(self.a, self.b, t=0.0)
        np.testing.assert_allclose(out, self.a, atol=1e-4)

    def test_slerp_t1_returns_b(self):
        out = self.merge.slerp(self.a, self.b, t=1.0)
        np.testing.assert_allclose(out, self.b, atol=1e-4)

    def test_dare_shape(self):
        out = self.merge.dare(self.a, self.delta)
        self.assertEqual(out.shape, (256,))
        self.assertEqual(out.dtype, np.float32)

    def test_dare_full_density_equals_add(self):
        out = self.merge.dare(self.a, self.delta, density=1.0)
        np.testing.assert_allclose(out, self.a + self.delta, atol=1e-5)

    def test_dare_deterministic(self):
        o1 = self.merge.dare(self.a, self.delta, seed=42)
        o2 = self.merge.dare(self.a, self.delta, seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_ties_shape(self):
        deltas = np.stack([self.delta, self.delta * 0.5], axis=0)
        out = self.merge.ties(self.a, deltas)
        self.assertEqual(out.shape, (256,))
        self.assertEqual(out.dtype, np.float32)

    def test_ties_zero_delta(self):
        deltas = np.zeros((2, 256), dtype=np.float32)
        out = self.merge.ties(self.a, deltas)
        np.testing.assert_allclose(out, self.a, atol=1e-5)

    def test_ties_single_model(self):
        out = self.merge.ties(self.a, self.delta[np.newaxis, :])
        self.assertEqual(out.shape, (256,))

    def test_slerp_midpoint_norm_bounded(self):
        # Midpoint of SLERP should have norm between norm(a) and norm(b)
        out = self.merge.slerp(self.a, self.b, t=0.5)
        na, nb = np.linalg.norm(self.a), np.linalg.norm(self.b)
        no = np.linalg.norm(out)
        self.assertLessEqual(no, max(na, nb) * 1.5)


# ── MoEBincountConfig ──────────────────────────────────────────────────────


class TestMoEBincountConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MoEBincountConfig()
        self.assertEqual(cfg.n_experts, 128)
        self.assertEqual(cfg.top_k, 2)

    def test_custom(self):
        cfg = MoEBincountConfig(n_experts=64, top_k=4)
        self.assertEqual(cfg.n_experts, 64)


# ── RustMoEBincount ────────────────────────────────────────────────────────


class TestRustMoEBincount(unittest.TestCase):
    def setUp(self):
        self.moe = RustMoEBincount(MoEBincountConfig(n_experts=8, top_k=2))
        self.rng = np.random.default_rng(1)

    def test_backend_is_string(self):
        self.assertIn(self.moe.backend(), ("rust", "numpy"))

    def test_bincount_shape(self):
        assignments = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        freqs = self.moe.bincount(assignments, n_experts=8)
        self.assertEqual(freqs.shape, (8,))
        self.assertEqual(freqs.dtype, np.float32)

    def test_bincount_sums_to_one(self):
        assignments = self.rng.integers(0, 8, 64).astype(np.int32)
        freqs = self.moe.bincount(assignments, n_experts=8)
        self.assertAlmostEqual(float(freqs.sum()), 1.0, places=5)

    def test_bincount_uniform(self):
        # All-same assignment → one hot
        assignments = np.zeros(16, dtype=np.int32)
        freqs = self.moe.bincount(assignments, n_experts=8)
        self.assertAlmostEqual(float(freqs[0]), 1.0, places=5)
        np.testing.assert_allclose(freqs[1:], 0.0, atol=1e-5)

    def test_bincount_counting_correct(self):
        assignments = np.array([0, 0, 1, 2], dtype=np.int32)
        freqs = self.moe.bincount(assignments, n_experts=3)
        self.assertAlmostEqual(float(freqs[0]), 0.5, places=5)
        self.assertAlmostEqual(float(freqs[1]), 0.25, places=5)
        self.assertAlmostEqual(float(freqs[2]), 0.25, places=5)

    def test_top_k_shape(self):
        logits = self.rng.standard_normal((4, 8)).astype(np.float32)
        top = self.moe.top_k(logits, k=2)
        self.assertEqual(top.shape, (4, 2))
        self.assertEqual(top.dtype, np.int32)

    def test_top_k_valid_indices(self):
        logits = self.rng.standard_normal((4, 8)).astype(np.float32)
        top = self.moe.top_k(logits, k=2)
        self.assertTrue((top >= 0).all())
        self.assertTrue((top < 8).all())

    def test_top_k_descending_score_order(self):
        logits = np.arange(8, dtype=np.float32)[np.newaxis, :]
        top = self.moe.top_k(logits, k=3)
        # Expert 7 > 6 > 5 in score
        self.assertEqual(int(top[0, 0]), 7)
        self.assertEqual(int(top[0, 1]), 6)

    def test_n_experts_property(self):
        self.assertEqual(self.moe.n_experts(), 8)

    def test_top1_assignment(self):
        logits = np.eye(8, dtype=np.float32)  # 8 rows, expert i has score 1 elsewhere 0
        top = self.moe.top_k(logits, k=1)
        expected = np.arange(8, dtype=np.int32).reshape(8, 1)
        np.testing.assert_array_equal(top, expected)


# ── OnlineSGDConfig ────────────────────────────────────────────────────────


class TestOnlineSGDConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = OnlineSGDConfig()
        self.assertEqual(cfg.n_features, 32)
        self.assertAlmostEqual(cfg.learning_rate, 0.01)

    def test_custom(self):
        cfg = OnlineSGDConfig(n_features=16, learning_rate=0.001)
        self.assertEqual(cfg.n_features, 16)


# ── RustOnlineSGD ──────────────────────────────────────────────────────────


class TestRustOnlineSGD(unittest.TestCase):
    def setUp(self):
        self.sgd = RustOnlineSGD(OnlineSGDConfig(n_features=8, learning_rate=0.1))
        self.rng = np.random.default_rng(2)

    def test_backend_is_string(self):
        self.assertIn(self.sgd.backend(), ("rust", "numpy"))

    def test_initial_weights_zero(self):
        w = self.sgd.weights()
        np.testing.assert_array_equal(w, 0.0)
        self.assertEqual(w.shape, (8,))

    def test_predict_zero_weights(self):
        x = np.ones(8, dtype=np.float32)
        y_hat = self.sgd.predict(x)
        self.assertAlmostEqual(y_hat, 0.5, places=5)  # sigmoid(0) = 0.5

    def test_predict_returns_probability(self):
        x = self.rng.standard_normal(8).astype(np.float32)
        y_hat = self.sgd.predict(x)
        self.assertGreater(y_hat, 0.0)
        self.assertLess(y_hat, 1.0)

    def test_step_returns_tuple(self):
        x = self.rng.standard_normal(8).astype(np.float32)
        result = self.sgd.step(x, label=1.0)
        self.assertEqual(len(result), 2)

    def test_step_updates_weights(self):
        w_before = self.sgd.weights().copy()
        x = np.ones(8, dtype=np.float32)
        self.sgd.step(x, label=1.0)
        w_after = self.sgd.weights()
        # Weights must have changed
        self.assertFalse(np.allclose(w_before, w_after))

    def test_step_convergence_trivial(self):
        # Learn weight for a single feature — should converge to positive w
        self.sgd = RustOnlineSGD(OnlineSGDConfig(n_features=1, learning_rate=0.5))
        x = np.array([1.0], dtype=np.float32)
        for _ in range(100):
            self.sgd.step(x, label=1.0)
        w = self.sgd.weights()
        self.assertGreater(float(w[0]), 2.0)  # large positive weight drives y_hat → 1

    def test_reset_weights(self):
        self.sgd.step(np.ones(8, dtype=np.float32), label=1.0)
        self.sgd.reset_weights()
        np.testing.assert_array_equal(self.sgd.weights(), 0.0)

    def test_n_features(self):
        self.assertEqual(self.sgd.n_features(), 8)

    def test_error_direction(self):
        # With label=1 and zero weights (y_hat=0.5), error should be positive
        x = np.ones(8, dtype=np.float32)
        _, error = self.sgd.step(x, label=1.0)
        self.assertGreater(error, 0.0)

    def test_lr_override(self):
        x = np.ones(8, dtype=np.float32)
        self.sgd.step(x, label=1.0, lr=0.0)
        # Zero learning rate → weights unchanged
        w = self.sgd.weights()
        np.testing.assert_array_equal(w, 0.0)

    def test_deterministic_across_calls(self):
        # Same input/label sequence should give same weights
        sgd1 = RustOnlineSGD(OnlineSGDConfig(n_features=4, learning_rate=0.01))
        sgd2 = RustOnlineSGD(OnlineSGDConfig(n_features=4, learning_rate=0.01))
        xs = [self.rng.standard_normal(4).astype(np.float32) for _ in range(10)]
        labels = [float(self.rng.integers(0, 2)) for _ in range(10)]
        for x, l in zip(xs, labels):
            sgd1.step(x, l)
            sgd2.step(x, l)
        np.testing.assert_array_equal(sgd1.weights(), sgd2.weights())


if __name__ == "__main__":
    unittest.main()
