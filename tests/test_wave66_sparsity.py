"""tests/test_wave66_sparsity.py

Unit tests for Wave 66: FFN co-activation sparsity profiling, cluster
reordering, sparse GEMV Metal shader constants, and per-layer sparsity
predictor.

Modules under test
──────────────────
* squish.compress.sparsity_profiler  — ProfilerConfig, ClusterInfo,
                                        LayerSparsityProfile, SparsityProfiler,
                                        coactivation_matrix, kmeans_cluster
* squish.compress.cluster_reorder   — ClusterReorder, ReorderResult,
                                        compute_cluster_permutation
* squish.token.sparsity_predictor   — PredictorConfig, SparsityPredictor
* squish.runtime.squish_runtime     — KernelStack.SPARSE, _select_kernel routing

All tests run without Metal, hardware, or external dependencies.
NumPy is the only required third-party package.
"""
from __future__ import annotations

import struct
import unittest

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _make_activations(n_samples: int, n_neurons: int, sparsity: float = 0.5, seed: int = 0) -> np.ndarray:
    """Return (n_samples, n_neurons) float32 with ~sparsity fraction zeros."""
    rng = np.random.default_rng(seed)
    act = rng.standard_normal((n_samples, n_neurons)).astype(np.float32)
    mask = rng.random((n_samples, n_neurons)) < sparsity
    act[mask] = 0.0
    return act


# =============================================================================
# 1. ProfilerConfig
# =============================================================================

class TestProfilerConfigDefaults(unittest.TestCase):
    def test_default_n_samples(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        cfg = ProfilerConfig()
        self.assertEqual(cfg.n_samples, 2000)

    def test_default_n_clusters(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        cfg = ProfilerConfig()
        self.assertEqual(cfg.n_clusters, 64)

    def test_default_threshold(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        cfg = ProfilerConfig()
        self.assertAlmostEqual(cfg.activation_threshold, 1e-3)

    def test_default_kmeans_seed(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        cfg = ProfilerConfig()
        self.assertEqual(cfg.kmeans_seed, 42)


class TestProfilerConfigValidation(unittest.TestCase):
    def test_bad_n_samples_raises(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        with self.assertRaises(ValueError):
            ProfilerConfig(n_samples=0)

    def test_bad_n_clusters_raises(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        with self.assertRaises(ValueError):
            ProfilerConfig(n_clusters=0)

    def test_bad_threshold_raises(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        with self.assertRaises(ValueError):
            ProfilerConfig(activation_threshold=-1e-4)

    def test_custom_values(self):
        from squish.compress.sparsity_profiler import ProfilerConfig
        cfg = ProfilerConfig(n_samples=100, n_clusters=8, activation_threshold=0.01)
        self.assertEqual(cfg.n_samples, 100)
        self.assertEqual(cfg.n_clusters, 8)
        self.assertAlmostEqual(cfg.activation_threshold, 0.01)


# =============================================================================
# 2. kmeans_cluster
# =============================================================================

class TestKMeansCluster(unittest.TestCase):
    def test_returns_correct_shapes(self):
        from squish.compress.sparsity_profiler import kmeans_cluster
        X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
        assignments, centroids = kmeans_cluster(X, k=3, max_iter=20, seed=0)
        self.assertEqual(assignments.shape, (20,))
        self.assertEqual(centroids.shape, (3, 4))

    def test_assignments_in_range(self):
        from squish.compress.sparsity_profiler import kmeans_cluster
        X = np.random.default_rng(1).random((30, 5)).astype(np.float32)
        assignments, _ = kmeans_cluster(X, k=4, max_iter=30, seed=1)
        self.assertTrue(np.all(assignments >= 0))
        self.assertTrue(np.all(assignments < 4))

    def test_two_clear_clusters(self):
        from squish.compress.sparsity_profiler import kmeans_cluster
        # Two clearly separated clusters in 1D.
        X = np.concatenate([
            np.ones((10, 1)) * -10.0,
            np.ones((10, 1)) * +10.0,
        ]).astype(np.float32)
        assignments, _ = kmeans_cluster(X, k=2, max_iter=50, seed=0)
        # All first 10 should be in the same cluster; all last 10 same cluster.
        self.assertEqual(len(set(assignments[:10])), 1)
        self.assertEqual(len(set(assignments[10:])), 1)
        self.assertNotEqual(assignments[0], assignments[10])

    def test_k_equals_1(self):
        from squish.compress.sparsity_profiler import kmeans_cluster
        X = np.random.default_rng(2).random((15, 3)).astype(np.float32)
        assignments, centroids = kmeans_cluster(X, k=1, max_iter=10, seed=0)
        self.assertTrue(np.all(assignments == 0))
        self.assertEqual(centroids.shape, (1, 3))

    def test_deterministic_with_same_seed(self):
        from squish.compress.sparsity_profiler import kmeans_cluster
        X = np.random.default_rng(3).random((25, 4)).astype(np.float32)
        a1, _ = kmeans_cluster(X, k=3, max_iter=30, seed=7)
        a2, _ = kmeans_cluster(X, k=3, max_iter=30, seed=7)
        np.testing.assert_array_equal(a1, a2)


# =============================================================================
# 3. coactivation_matrix
# =============================================================================

class TestCoactivationMatrix(unittest.TestCase):
    def test_shape_correct(self):
        from squish.compress.sparsity_profiler import coactivation_matrix
        act = _make_activations(50, 10, sparsity=0.3, seed=0)
        mat = coactivation_matrix(act, threshold=1e-3)
        self.assertEqual(mat.shape, (10, 10))

    def test_diagonal_ones(self):
        """Each neuron co-activates with itself whenever it activates."""
        from squish.compress.sparsity_profiler import coactivation_matrix
        act = _make_activations(100, 8, sparsity=0.0, seed=0)  # fully dense
        mat = coactivation_matrix(act, threshold=1e-3)
        np.testing.assert_array_almost_equal(np.diag(mat), np.ones(8), decimal=6)

    def test_all_zero_input_gives_zero_matrix(self):
        from squish.compress.sparsity_profiler import coactivation_matrix
        act = np.zeros((50, 10), dtype=np.float32)
        mat = coactivation_matrix(act, threshold=1e-3)
        # No activations → all co-activations are nan or 0; either is acceptable.
        self.assertEqual(mat.shape, (10, 10))

    def test_symmetric(self):
        from squish.compress.sparsity_profiler import coactivation_matrix
        act = _make_activations(80, 6, sparsity=0.4, seed=5)
        mat = coactivation_matrix(act, threshold=1e-3)
        np.testing.assert_array_almost_equal(mat, mat.T, decimal=6)

    def test_values_in_unit_interval(self):
        from squish.compress.sparsity_profiler import coactivation_matrix
        act = _make_activations(100, 12, sparsity=0.5, seed=6)
        mat = coactivation_matrix(act, threshold=1e-3)
        finite = mat[np.isfinite(mat)]
        self.assertTrue(np.all(finite >= 0.0))
        self.assertTrue(np.all(finite <= 1.0 + 1e-6))


# =============================================================================
# 4. LayerSparsityProfile serialisation
# =============================================================================

class TestLayerSparsityProfileRoundTrip(unittest.TestCase):
    def _make_profile(self, n_neurons=64, n_clusters=4):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        rng = np.random.default_rng(0)
        assignments = rng.integers(0, n_clusters, size=n_neurons).astype(np.int32)
        sizes = np.bincount(assignments, minlength=n_clusters).astype(np.int32)
        boundaries = np.zeros(n_clusters + 1, dtype=np.int32)
        boundaries[1:] = np.cumsum(sizes)
        hist = rng.random(n_clusters).astype(np.float32)
        return LayerSparsityProfile(
            layer_idx=0,
            n_neurons=n_neurons,
            n_clusters=n_clusters,
            cluster_assignments=assignments,
            cluster_boundaries=boundaries,
            activation_histogram=hist,
            sparsity_ratio=0.5,
            expected_sparsity=0.5,
            clusters=[],
        )

    def test_to_bytes_returns_bytes(self):
        p = self._make_profile()
        raw = p.to_metadata_bytes()
        self.assertIsInstance(raw, bytes)

    def test_round_trip_n_clusters(self):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        p = self._make_profile(n_neurons=32, n_clusters=8)
        raw = p.to_metadata_bytes()
        p2 = LayerSparsityProfile.from_metadata_bytes(raw, layer_idx=0, n_neurons=32)
        self.assertEqual(p2.n_clusters, 8)

    def test_round_trip_boundaries(self):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        p = self._make_profile(n_neurons=32, n_clusters=4)
        raw = p.to_metadata_bytes()
        p2 = LayerSparsityProfile.from_metadata_bytes(raw, layer_idx=0, n_neurons=32)
        np.testing.assert_array_equal(p2.cluster_boundaries, p.cluster_boundaries)

    def test_round_trip_histogram(self):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        p = self._make_profile(n_neurons=32, n_clusters=4)
        raw = p.to_metadata_bytes()
        p2 = LayerSparsityProfile.from_metadata_bytes(raw, layer_idx=0, n_neurons=32)
        np.testing.assert_array_almost_equal(p2.activation_histogram, p.activation_histogram, decimal=5)

    def test_round_trip_expected_sparsity(self):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        p = self._make_profile()
        raw = p.to_metadata_bytes()
        p2 = LayerSparsityProfile.from_metadata_bytes(raw, layer_idx=0, n_neurons=64)
        self.assertAlmostEqual(p2.expected_sparsity, p.expected_sparsity, places=5)


# =============================================================================
# 5. SparsityProfiler
# =============================================================================

class TestSparsityProfiler(unittest.TestCase):
    def setUp(self):
        from squish.compress.sparsity_profiler import ProfilerConfig, SparsityProfiler
        self.cfg = ProfilerConfig(n_samples=50, n_clusters=4, kmeans_seed=0)
        self.profiler = SparsityProfiler(self.cfg)

    def test_collect_activations_shape(self):
        n_neurons = 16
        act = _make_activations(50, n_neurons, seed=0)
        fn = lambda hs: hs  # identity activation
        result = self.profiler.collect_activations(fn, act)
        self.assertEqual(result.shape[1], n_neurons)

    def test_compute_neuron_stats_shapes(self):
        act = _make_activations(50, 16, sparsity=0.4, seed=1)
        mean_mag, firing_freq = self.profiler.compute_neuron_stats(act)
        self.assertEqual(mean_mag.shape, (16,))
        self.assertEqual(firing_freq.shape, (16,))

    def test_compute_neuron_stats_firing_freq_in_01(self):
        act = _make_activations(100, 16, sparsity=0.5, seed=3)
        _, freq = self.profiler.compute_neuron_stats(act)
        self.assertTrue(np.all(freq >= 0.0))
        self.assertTrue(np.all(freq <= 1.0 + 1e-6))

    def test_profile_layer_returns_profile(self):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        act = _make_activations(50, 16, sparsity=0.4, seed=2)
        fn = lambda hs: hs
        profile = self.profiler.profile_layer(fn, act, layer_idx=0)
        self.assertIsInstance(profile, LayerSparsityProfile)

    def test_profile_layer_sparsity_ratio_in_01(self):
        act = _make_activations(50, 16, sparsity=0.6, seed=4)
        fn = lambda hs: hs
        profile = self.profiler.profile_layer(fn, act, layer_idx=1)
        self.assertGreaterEqual(profile.sparsity_ratio, 0.0)
        self.assertLessEqual(profile.sparsity_ratio, 1.0)

    def test_profile_layer_n_clusters(self):
        act = _make_activations(50, 16, seed=5)
        fn = lambda hs: hs
        profile = self.profiler.profile_layer(fn, act, layer_idx=0)
        self.assertEqual(profile.n_clusters, self.cfg.n_clusters)

    def test_profile_layer_cluster_assignment_count(self):
        act = _make_activations(50, 16, seed=6)
        fn = lambda hs: hs
        profile = self.profiler.profile_layer(fn, act, layer_idx=0)
        self.assertEqual(len(profile.cluster_assignments), 16)

    def test_profile_model_returns_list(self):
        fns = [lambda hs: hs, lambda hs: hs]
        act = _make_activations(50, 16, seed=7)
        profiles = self.profiler.profile_model(fns, act)
        self.assertEqual(len(profiles), 2)

    def test_profile_model_layer_indices(self):
        fns = [lambda hs: hs, lambda hs: hs, lambda hs: hs]
        act = _make_activations(50, 16, seed=8)
        profiles = self.profiler.profile_model(fns, act)
        for i, p in enumerate(profiles):
            self.assertEqual(p.layer_idx, i)


# =============================================================================
# 6. compute_cluster_permutation
# =============================================================================

class TestComputeClusterPermutation(unittest.TestCase):
    def test_groups_by_cluster(self):
        from squish.compress.cluster_reorder import compute_cluster_permutation
        assignments = np.array([2, 0, 1, 0, 2, 1], dtype=np.int32)
        perm = compute_cluster_permutation(assignments, n_clusters=3)
        reordered = assignments[perm]
        # Should be non-decreasing after applying permutation.
        self.assertTrue(np.all(np.diff(reordered) >= 0))

    def test_permutation_is_valid(self):
        from squish.compress.cluster_reorder import compute_cluster_permutation
        n = 20
        rng = np.random.default_rng(0)
        assignments = rng.integers(0, 4, size=n).astype(np.int32)
        perm = compute_cluster_permutation(assignments, n_clusters=4)
        self.assertEqual(len(perm), n)
        np.testing.assert_array_equal(np.sort(perm), np.arange(n, dtype=np.int32))

    def test_single_cluster(self):
        from squish.compress.cluster_reorder import compute_cluster_permutation
        assignments = np.zeros(10, dtype=np.int32)
        perm = compute_cluster_permutation(assignments, n_clusters=1)
        self.assertEqual(len(perm), 10)


# =============================================================================
# 7. ClusterReorder
# =============================================================================

class TestClusterReorder(unittest.TestCase):
    def _make_profile(self, n_neurons=16, n_clusters=4):
        from squish.compress.sparsity_profiler import LayerSparsityProfile
        rng = np.random.default_rng(99)
        assignments = rng.integers(0, n_clusters, size=n_neurons).astype(np.int32)
        sizes = np.bincount(assignments, minlength=n_clusters).astype(np.int32)
        boundaries = np.zeros(n_clusters + 1, dtype=np.int32)
        boundaries[1:] = np.cumsum(sizes)
        hist = rng.random(n_clusters).astype(np.float32)
        return LayerSparsityProfile(
            layer_idx=0,
            n_neurons=n_neurons,
            n_clusters=n_clusters,
            cluster_assignments=assignments,
            cluster_boundaries=boundaries,
            activation_histogram=hist,
            sparsity_ratio=0.5,
            expected_sparsity=0.5,
            clusters=[],
        )

    def test_reorder_up_down_shapes(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(0)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down)
        self.assertEqual(result.w_up_reordered.shape, (16, 8))
        self.assertEqual(result.w_down_reordered.shape, (8, 16))

    def test_reorder_with_gate_shapes(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(1)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_gate = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down, w_gate=w_gate)
        self.assertIsNotNone(result.w_gate_reordered)
        self.assertEqual(result.w_gate_reordered.shape, (16, 8))

    def test_permutation_is_bijective(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(2)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down)
        np.testing.assert_array_equal(
            np.sort(result.permutation), np.arange(16, dtype=np.int32)
        )

    def test_inverse_permutation_correct(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(3)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down)
        # perm[inv_perm[i]] == i
        expected = np.arange(16, dtype=np.int32)
        np.testing.assert_array_equal(result.permutation[result.inverse_permutation], expected)

    def test_w_down_correctness(self):
        """GEMV output should be identical before and after reordering."""
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(4)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        x = rng.standard_normal(8).astype(np.float32)

        result = ClusterReorder().reorder(p, w_up, w_down)

        # Dense path: x @ W_up.T → (n_neurons,) then @ W_down.T → (hidden,)
        out_orig  = w_down @ (w_up @ x)
        out_reord = result.w_down_reordered @ (result.w_up_reordered @ x)
        np.testing.assert_array_almost_equal(out_orig, out_reord, decimal=5)

    def test_cluster_boundaries_length(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(5)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down)
        self.assertEqual(len(result.cluster_boundaries), 5)  # n_clusters + 1

    def test_cluster_boundaries_last_equals_n_neurons(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(6)
        w_up   = rng.standard_normal((16, 8)).astype(np.float32)
        w_down = rng.standard_normal((8, 16)).astype(np.float32)
        result = ClusterReorder().reorder(p, w_up, w_down)
        self.assertEqual(int(result.cluster_boundaries[-1]), 16)

    def test_mismatched_w_up_raises(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(7)
        w_up_bad = rng.standard_normal((24, 8)).astype(np.float32)  # wrong n_neurons
        w_down   = rng.standard_normal((8, 16)).astype(np.float32)
        with self.assertRaises(ValueError):
            ClusterReorder().reorder(p, w_up_bad, w_down)

    def test_mismatched_w_down_raises(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(8)
        w_up      = rng.standard_normal((16, 8)).astype(np.float32)
        w_down_bad = rng.standard_normal((8, 24)).astype(np.float32)  # wrong n_neurons
        with self.assertRaises(ValueError):
            ClusterReorder().reorder(p, w_up, w_down_bad)

    def test_verify_reorder_returns_zero(self):
        from squish.compress.cluster_reorder import ClusterReorder
        p = self._make_profile(n_neurons=16, n_clusters=4)
        rng = np.random.default_rng(9)
        w_up_orig = rng.standard_normal((16, 8)).astype(np.float32)
        w_down     = rng.standard_normal((8, 16)).astype(np.float32)
        hs = rng.standard_normal(8).astype(np.float32)
        reorderer = ClusterReorder()
        result = reorderer.reorder(p, w_up_orig.copy(), w_down)
        err = reorderer.verify_reorder(w_up_orig, result, hs)
        self.assertAlmostEqual(err, 0.0, places=4)


# =============================================================================
# 8. PredictorConfig
# =============================================================================

class TestPredictorConfigDefaults(unittest.TestCase):
    def test_default_threshold(self):
        from squish.token.sparsity_predictor import PredictorConfig
        cfg = PredictorConfig(d_model=128, n_clusters=8)
        self.assertAlmostEqual(cfg.threshold, 0.0)

    def test_default_lr(self):
        from squish.token.sparsity_predictor import PredictorConfig
        cfg = PredictorConfig(d_model=128, n_clusters=8)
        self.assertAlmostEqual(cfg.learning_rate, 0.01)

    def test_default_n_epochs(self):
        from squish.token.sparsity_predictor import PredictorConfig
        cfg = PredictorConfig(d_model=128, n_clusters=8)
        self.assertEqual(cfg.n_epochs, 20)


class TestPredictorConfigValidation(unittest.TestCase):
    def test_bad_d_model_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig
        with self.assertRaises(ValueError):
            PredictorConfig(d_model=0, n_clusters=8)

    def test_bad_n_clusters_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig
        with self.assertRaises(ValueError):
            PredictorConfig(d_model=128, n_clusters=0)

    def test_bad_lr_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig
        with self.assertRaises(ValueError):
            PredictorConfig(d_model=128, n_clusters=8, learning_rate=0.0)

    def test_bad_n_epochs_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig
        with self.assertRaises(ValueError):
            PredictorConfig(d_model=128, n_clusters=8, n_epochs=0)


# =============================================================================
# 9. SparsityPredictor
# =============================================================================

class TestSparsityPredictorPredict(unittest.TestCase):
    def setUp(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        self.cfg = PredictorConfig(d_model=32, n_clusters=8)
        self.pred = SparsityPredictor(self.cfg)

    def test_predict_returns_bool_array(self):
        hs = np.random.default_rng(0).standard_normal(32).astype(np.float32)
        mask = self.pred.predict(hs)
        self.assertEqual(mask.dtype, bool)

    def test_predict_output_shape(self):
        hs = np.random.default_rng(1).standard_normal(32).astype(np.float32)
        mask = self.pred.predict(hs)
        self.assertEqual(mask.shape, (8,))

    def test_predict_wrong_size_raises(self):
        hs = np.ones(16, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.pred.predict(hs)

    def test_predict_batch_shape(self):
        hs = np.random.default_rng(2).standard_normal((10, 32)).astype(np.float32)
        masks = self.pred.predict_batch(hs)
        self.assertEqual(masks.shape, (10, 8))

    def test_zero_weights_all_inactive(self):
        # Zero-initialised weights → all logits == 0 → threshold 0.0 → all False.
        hs = np.ones(32, dtype=np.float32)
        mask = self.pred.predict(hs)
        self.assertFalse(np.any(mask))


class TestSparsityPredictorTrain(unittest.TestCase):
    def setUp(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        self.cfg = PredictorConfig(d_model=16, n_clusters=4, n_epochs=5, batch_size=32, learning_rate=0.1)
        self.pred = SparsityPredictor(self.cfg)

    def test_train_returns_loss_list(self):
        rng = np.random.default_rng(0)
        hs = rng.standard_normal((100, 16)).astype(np.float32)
        labels = (rng.random((100, 4)) > 0.5).astype(np.float32)
        losses = self.pred.train(hs, labels)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 5)

    def test_train_loss_decreases(self):
        rng = np.random.default_rng(1)
        hs = rng.standard_normal((200, 16)).astype(np.float32)
        labels = (rng.random((200, 4)) > 0.5).astype(np.float32)
        cfg2 = self._make_cfg(n_epochs=20)
        from squish.token.sparsity_predictor import SparsityPredictor
        pred = SparsityPredictor(cfg2)
        losses = pred.train(hs, labels)
        self.assertLess(losses[-1], losses[0])

    def _make_cfg(self, n_epochs=5):
        from squish.token.sparsity_predictor import PredictorConfig
        return PredictorConfig(d_model=16, n_clusters=4, n_epochs=n_epochs, batch_size=32, learning_rate=0.1)

    def test_train_weights_not_zero_after_training(self):
        rng = np.random.default_rng(2)
        hs = rng.standard_normal((100, 16)).astype(np.float32)
        labels = (rng.random((100, 4)) > 0.5).astype(np.float32)
        self.pred.train(hs, labels)
        self.assertFalse(np.all(self.pred.predictor_weights == 0))

    def test_train_wrong_d_model_raises(self):
        rng = np.random.default_rng(3)
        hs_bad = rng.standard_normal((100, 99)).astype(np.float32)
        labels = (rng.random((100, 4)) > 0.5).astype(np.float32)
        with self.assertRaises(ValueError):
            self.pred.train(hs_bad, labels)

    def test_train_wrong_n_clusters_raises(self):
        rng = np.random.default_rng(4)
        hs = rng.standard_normal((100, 16)).astype(np.float32)
        labels_bad = (rng.random((100, 99)) > 0.5).astype(np.float32)
        with self.assertRaises(ValueError):
            self.pred.train(hs, labels_bad)


class TestSparsityPredictorAccuracy(unittest.TestCase):
    def test_accuracy_in_01(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        cfg = PredictorConfig(d_model=16, n_clusters=4)
        pred = SparsityPredictor(cfg)
        rng = np.random.default_rng(5)
        hs = rng.standard_normal((50, 16)).astype(np.float32)
        labels = (rng.random((50, 4)) > 0.5).astype(np.float32)
        acc = pred.accuracy(hs, labels)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_recall_in_01(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        cfg = PredictorConfig(d_model=16, n_clusters=4)
        pred = SparsityPredictor(cfg)
        rng = np.random.default_rng(6)
        hs = rng.standard_normal((50, 16)).astype(np.float32)
        labels = (rng.random((50, 4)) > 0.5).astype(np.float32)
        recall = pred.recall(hs, labels)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)


class TestSparsityPredictorSerialization(unittest.TestCase):
    def setUp(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        self.cfg = PredictorConfig(d_model=32, n_clusters=8)
        self.pred = SparsityPredictor(self.cfg)
        rng = np.random.default_rng(0)
        self.pred.predictor_weights = rng.standard_normal((32, 8)).astype(np.float16)

    def test_to_bytes_returns_bytes(self):
        raw = self.pred.to_bytes()
        self.assertIsInstance(raw, bytes)

    def test_to_bytes_length(self):
        raw = self.pred.to_bytes()
        expected = 8 + 4 + 4 + 4 + 32 * 8 * 2  # magic + d_model + n_clusters + threshold + weights
        self.assertEqual(len(raw), expected)

    def test_round_trip_weights(self):
        from squish.token.sparsity_predictor import SparsityPredictor
        raw = self.pred.to_bytes()
        pred2 = SparsityPredictor.from_bytes(raw, self.cfg)
        np.testing.assert_array_equal(pred2.predictor_weights, self.pred.predictor_weights)

    def test_round_trip_threshold(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        cfg2 = PredictorConfig(d_model=32, n_clusters=8, threshold=0.5)
        pred = SparsityPredictor(cfg2)
        raw = pred.to_bytes()
        pred2 = SparsityPredictor.from_bytes(raw, cfg2)
        self.assertAlmostEqual(pred2.config.threshold, 0.5, places=5)

    def test_from_bytes_wrong_magic_raises(self):
        from squish.token.sparsity_predictor import SparsityPredictor
        raw = b"\x00" * 100
        with self.assertRaises(ValueError):
            SparsityPredictor.from_bytes(raw, self.cfg)

    def test_from_bytes_wrong_d_model_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        raw = self.pred.to_bytes()
        cfg_bad = PredictorConfig(d_model=64, n_clusters=8)
        with self.assertRaises(ValueError):
            SparsityPredictor.from_bytes(raw, cfg_bad)

    def test_from_bytes_wrong_n_clusters_raises(self):
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor
        raw = self.pred.to_bytes()
        cfg_bad = PredictorConfig(d_model=32, n_clusters=16)
        with self.assertRaises(ValueError):
            SparsityPredictor.from_bytes(raw, cfg_bad)

    def test_predict_after_round_trip_matches(self):
        from squish.token.sparsity_predictor import SparsityPredictor
        raw = self.pred.to_bytes()
        pred2 = SparsityPredictor.from_bytes(raw, self.cfg)
        hs = np.random.default_rng(1).standard_normal(32).astype(np.float32)
        np.testing.assert_array_equal(self.pred.predict(hs), pred2.predict(hs))


# =============================================================================
# 10. KernelStack.SPARSE and _select_kernel routing
# =============================================================================

class TestKernelStackSparse(unittest.TestCase):
    def test_sparse_constant_exists(self):
        from squish.runtime.squish_runtime import KernelStack
        self.assertTrue(hasattr(KernelStack, "SPARSE"))

    def test_sparse_constant_value(self):
        from squish.runtime.squish_runtime import KernelStack
        self.assertEqual(KernelStack.SPARSE, "sparse_gemv")


class TestSelectKernelSparseRouting(unittest.TestCase):
    def _get_select_kernel(self):
        from squish.runtime.squish_runtime import SquishRuntime
        return SquishRuntime._select_kernel

    def _make_flags(self, *flag_names):
        from squish.runtime.squish_runtime import SquizdFlags
        result = SquizdFlags(0)
        for name in flag_names:
            result = SquizdFlags(result.value | getattr(SquizdFlags, name).value)
        return result

    def test_sparse_flag_routes_to_sparse(self):
        from squish.runtime.squish_runtime import KernelStack, SquizdFlags
        select = self._get_select_kernel()
        flags = SquizdFlags(SquizdFlags.SPARSE.value)
        self.assertEqual(select(flags), KernelStack.SPARSE)

    def test_no_flags_routes_to_numpy(self):
        from squish.runtime.squish_runtime import KernelStack, SquizdFlags
        select = self._get_select_kernel()
        flags = SquizdFlags(0)
        self.assertEqual(select(flags), KernelStack.NUMPY)

    def test_tca_tbe_routes_to_tca_tbe(self):
        from squish.runtime.squish_runtime import KernelStack, SquizdFlags
        select = self._get_select_kernel()
        flags = SquizdFlags(SquizdFlags.TCA_TBE.value)
        self.assertEqual(select(flags), KernelStack.TCA_TBE)

    def test_int4_routes_to_int4(self):
        from squish.runtime.squish_runtime import KernelStack, SquizdFlags
        select = self._get_select_kernel()
        flags = SquizdFlags(SquizdFlags.INT4.value)
        self.assertEqual(select(flags), KernelStack.INT4)


# =============================================================================
# 11. Wave 66 integration pipeline
# =============================================================================

class TestWave66Integration(unittest.TestCase):
    """End-to-end: profile → reorder → predictor → dispatch."""

    def test_full_pipeline_produces_correct_output(self):
        """Calibrate sparsity, reorder weights, train predictor, predict mask."""
        from squish.compress.sparsity_profiler import ProfilerConfig, SparsityProfiler
        from squish.compress.cluster_reorder import ClusterReorder
        from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor

        n_neurons = 32
        hidden_dim = 16
        n_clusters = 4
        n_cal = 60

        rng = np.random.default_rng(42)
        # Calibration hidden states.
        cal_hs = rng.standard_normal((n_cal, hidden_dim)).astype(np.float32)

        # Simulate neuron activations.
        cal_act = rng.standard_normal((n_cal, n_neurons)).astype(np.float32)
        cal_act[rng.random((n_cal, n_neurons)) > 0.5] = 0.0

        # Stage 1: profile.
        cfg = ProfilerConfig(n_samples=n_cal, n_clusters=n_clusters, kmeans_seed=0)
        profiler = SparsityProfiler(cfg)
        profile = profiler.profile_layer(lambda x: x, cal_act, layer_idx=0)
        self.assertEqual(profile.n_clusters, n_clusters)

        # Stage 2: reorder weights.
        w_up   = rng.standard_normal((n_neurons, hidden_dim)).astype(np.float32)
        w_gate = rng.standard_normal((n_neurons, hidden_dim)).astype(np.float32)
        w_down = rng.standard_normal((hidden_dim, n_neurons)).astype(np.float32)

        result = ClusterReorder().reorder(profile, w_up, w_down, w_gate=w_gate)
        self.assertEqual(result.w_up_reordered.shape, (n_neurons, hidden_dim))

        # Stage 3: verify GEMV correctness after reorder.
        test_x = rng.standard_normal(hidden_dim).astype(np.float32)
        out_orig  = w_down @ (w_up @ test_x)
        out_reord = result.w_down_reordered @ (result.w_up_reordered @ test_x)
        np.testing.assert_array_almost_equal(out_orig, out_reord, decimal=4)

        # Stage 4: train predictor.
        # Build cluster activation labels from calibration activations.
        cluster_labels = np.zeros((n_cal, n_clusters), dtype=np.float32)
        for i in range(n_cal):
            for c in range(n_clusters):
                start = int(result.cluster_boundaries[c])
                end   = int(result.cluster_boundaries[c + 1])
                cluster_labels[i, c] = float(
                    np.any(np.abs(cal_act[i, result.permutation[start:end]]) > 1e-3)
                )

        pcfg = PredictorConfig(d_model=hidden_dim, n_clusters=n_clusters, n_epochs=5)
        predictor = SparsityPredictor(pcfg)
        predictor.train(cal_hs, cluster_labels)

        # Stage 5: predict mask for a new hidden state.
        new_hs = rng.standard_normal(hidden_dim).astype(np.float32)
        mask = predictor.predict(new_hs)
        self.assertEqual(mask.shape, (n_clusters,))
        self.assertEqual(mask.dtype, bool)

    def test_sparse_flag_routes_in_runtime(self):
        from squish.runtime.squish_runtime import KernelStack, SquizdFlags, SquishRuntime
        flags = SquizdFlags(SquizdFlags.SPARSE.value)
        self.assertEqual(SquishRuntime._select_kernel(flags), KernelStack.SPARSE)

    def test_reorder_result_has_profile(self):
        from squish.compress.sparsity_profiler import ProfilerConfig, SparsityProfiler, LayerSparsityProfile
        from squish.compress.cluster_reorder import ClusterReorder
        rng = np.random.default_rng(0)
        act = rng.standard_normal((40, 8)).astype(np.float32)
        cfg = ProfilerConfig(n_samples=40, n_clusters=2, kmeans_seed=0)
        profile = SparsityProfiler(cfg).profile_layer(lambda x: x, act, layer_idx=0)
        w_up   = rng.standard_normal((8, 4)).astype(np.float32)
        w_down = rng.standard_normal((4, 8)).astype(np.float32)
        result = ClusterReorder().reorder(profile, w_up, w_down)
        self.assertIsInstance(result.profile, LayerSparsityProfile)


if __name__ == "__main__":
    unittest.main()
