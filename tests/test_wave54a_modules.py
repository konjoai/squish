"""tests/test_wave54a_modules.py

Unit tests for Wave 54a MoE modules:
  - SharedExpertMoE
  - FineGrainedMoERouter
  - ExpertOffloader
  - ExpertMerger
  - LazyExpertLoader
  - ExpertActivationCache
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from squish.moe.shared_expert import SharedExpertConfig, SharedExpertMoE
from squish.moe.fine_grained_router import (
    FineGrainedRouterConfig,
    RouterBiasState,
    FineGrainedMoERouter,
)
from squish.moe.expert_offload import (
    ExpertOffloadConfig,
    OffloadState,
    ExpertOffloader,
)
from squish.moe.expert_merge import ExpertMergeConfig, ExpertMerger
from squish.moe.lazy_expert_load import (
    LazyExpertConfig,
    LazyExpertState,
    LazyExpertLoader,
)
from squish.moe.expert_cache import (
    ExpertCacheConfig,
    ExpertCacheState,
    ExpertActivationCache,
)


# ===========================================================================
# SharedExpertMoE Tests
# ===========================================================================

class TestSharedExpertConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SharedExpertConfig()
        self.assertEqual(cfg.d_model, 256)
        self.assertEqual(cfg.n_routed, 8)
        self.assertEqual(cfg.n_shared, 2)
        self.assertEqual(cfg.top_k, 2)
        self.assertAlmostEqual(cfg.expand_factor, 4.0)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(d_model=0)

    def test_invalid_n_routed(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(n_routed=0)

    def test_invalid_n_shared(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(n_shared=0)

    def test_invalid_top_k_zero(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(top_k=0)

    def test_invalid_top_k_exceeds_n_routed(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(n_routed=4, top_k=5)

    def test_invalid_expand_factor(self):
        with self.assertRaises(ValueError):
            SharedExpertConfig(expand_factor=0.0)


class TestSharedExpertMoE(unittest.TestCase):
    def setUp(self):
        self.cfg = SharedExpertConfig(d_model=32, n_routed=4, n_shared=2, top_k=2, seed=0)
        self.moe = SharedExpertMoE(self.cfg)

    def test_forward_output_shape(self):
        x = np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)
        (out,) = self.moe.forward(x)
        self.assertEqual(out.shape, (8, 32))

    def test_forward_single_token(self):
        x = np.ones((1, 32), dtype=np.float32)
        (out,) = self.moe.forward(x)
        self.assertEqual(out.shape, (1, 32))

    def test_forward_returns_tuple(self):
        x = np.zeros((4, 32), dtype=np.float32)
        result = self.moe.forward(x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_forward_output_dtype(self):
        x = np.ones((4, 32), dtype=np.float32)
        (out,) = self.moe.forward(x)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_invalid_shape(self):
        x = np.ones((4, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.moe.forward(x)

    def test_forward_1d_raises(self):
        x = np.ones(32, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.moe.forward(x)

    def test_router_indices_in_range(self):
        x = np.random.default_rng(7).standard_normal((16, 32)).astype(np.float32)
        indices, weights = self.moe._router(x)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.cfg.n_routed))

    def test_router_weights_sum_to_one(self):
        x = np.random.default_rng(7).standard_normal((16, 32)).astype(np.float32)
        _, weights = self.moe._router(x)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(16), atol=1e-5)

    def test_router_shape(self):
        x = np.random.default_rng(8).standard_normal((10, 32)).astype(np.float32)
        indices, weights = self.moe._router(x)
        self.assertEqual(indices.shape, (10, self.cfg.top_k))
        self.assertEqual(weights.shape, (10, self.cfg.top_k))

    def test_deterministic_with_same_seed(self):
        cfg = SharedExpertConfig(d_model=32, n_routed=4, n_shared=2, top_k=2, seed=42)
        m1 = SharedExpertMoE(cfg)
        m2 = SharedExpertMoE(cfg)
        x = np.ones((4, 32), dtype=np.float32)
        (o1,) = m1.forward(x)
        (o2,) = m2.forward(x)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        cfg1 = SharedExpertConfig(d_model=32, n_routed=4, n_shared=2, top_k=2, seed=0)
        cfg2 = SharedExpertConfig(d_model=32, n_routed=4, n_shared=2, top_k=2, seed=99)
        m1, m2 = SharedExpertMoE(cfg1), SharedExpertMoE(cfg2)
        x = np.ones((4, 32), dtype=np.float32)
        (o1,) = m1.forward(x)
        (o2,) = m2.forward(x)
        self.assertFalse(np.allclose(o1, o2))

    def test_zero_input(self):
        x = np.zeros((4, 32), dtype=np.float32)
        (out,) = self.moe.forward(x)
        self.assertEqual(out.shape, (4, 32))


# ===========================================================================
# FineGrainedMoERouter Tests
# ===========================================================================

class TestFineGrainedRouterConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FineGrainedRouterConfig()
        self.assertEqual(cfg.n_experts, 64)
        self.assertEqual(cfg.top_k, 6)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            FineGrainedRouterConfig(d_model=0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            FineGrainedRouterConfig(n_experts=8, top_k=9)

    def test_invalid_n_groups(self):
        with self.assertRaises(ValueError):
            FineGrainedRouterConfig(n_experts=8, n_groups=9)

    def test_invalid_bias_lr(self):
        with self.assertRaises(ValueError):
            FineGrainedRouterConfig(bias_update_lr=0.0)

    def test_invalid_balance_window(self):
        with self.assertRaises(ValueError):
            FineGrainedRouterConfig(balance_window=0)


class TestFineGrainedMoERouter(unittest.TestCase):
    def setUp(self):
        self.cfg = FineGrainedRouterConfig(d_model=32, n_experts=8, top_k=2, n_groups=2, seed=0)
        self.router = FineGrainedMoERouter(self.cfg)

    def test_new_state_bias_zeros(self):
        state = self.router.new_state()
        np.testing.assert_array_equal(state.bias, np.zeros(8))

    def test_new_state_n_steps_zero(self):
        state = self.router.new_state()
        self.assertEqual(state.n_steps, 0)

    def test_route_shapes(self):
        x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
        state = self.router.new_state()
        indices, weights, _ = self.router.route(x, state)
        self.assertEqual(indices.shape, (16, 2))
        self.assertEqual(weights.shape, (16, 2))

    def test_route_weights_sum_to_one(self):
        x = np.random.default_rng(2).standard_normal((16, 32)).astype(np.float32)
        state = self.router.new_state()
        _, weights, _ = self.router.route(x, state)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(16), atol=1e-5)

    def test_route_indices_in_range(self):
        x = np.random.default_rng(3).standard_normal((16, 32)).astype(np.float32)
        state = self.router.new_state()
        indices, _, _ = self.router.route(x, state)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 8))

    def test_route_state_n_steps_increments(self):
        x = np.random.default_rng(4).standard_normal((4, 32)).astype(np.float32)
        state = self.router.new_state()
        _, _, state2 = self.router.route(x, state)
        self.assertEqual(state2.n_steps, 1)

    def test_update_bias_changes_bias(self):
        state = self.router.new_state()
        load = np.ones(8, dtype=np.float32) * 0.2  # uniform overload
        state2 = self.router.update_bias(load, state)
        self.assertFalse(np.allclose(state2.bias, state.bias))

    def test_expert_utilization_shape(self):
        state = self.router.new_state()
        util = self.router.expert_utilization(state)
        self.assertEqual(util.shape, (8,))

    def test_route_invalid_shape(self):
        state = self.router.new_state()
        x = np.ones((4, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.router.route(x, state)


# ===========================================================================
# ExpertOffloader Tests
# ===========================================================================

class TestExpertOffloadConfig(unittest.TestCase):
    def test_max_resident_exceeds_n_experts(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig(n_experts=4, max_resident=5)

    def test_max_resident_zero(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig(max_resident=0)

    def test_n_experts_zero(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig(n_experts=0)


class TestExpertOffloader(unittest.TestCase):
    def setUp(self):
        self.cfg = ExpertOffloadConfig(n_experts=8, expert_dim=16, ffn_dim=32, max_resident=3, seed=0)
        self.offloader = ExpertOffloader(self.cfg)

    def test_new_state_empty(self):
        s = self.offloader.new_state()
        self.assertEqual(len(s.resident), 0)
        self.assertEqual(s.n_fetches, 0)

    def test_get_expert_returns_weights(self):
        s = self.offloader.new_state()
        W_up, W_down, s2 = self.offloader.get_expert(0, s)
        self.assertEqual(W_up.shape, (32, 16))
        self.assertEqual(W_down.shape, (16, 32))

    def test_get_expert_increments_fetches(self):
        s = self.offloader.new_state()
        _, _, s2 = self.offloader.get_expert(0, s)
        self.assertEqual(s2.n_fetches, 1)

    def test_get_expert_cache_hit_no_extra_fetch(self):
        s = self.offloader.new_state()
        _, _, s2 = self.offloader.get_expert(0, s)
        _, _, s3 = self.offloader.get_expert(0, s2)
        self.assertEqual(s3.n_fetches, 1)

    def test_eviction_when_full(self):
        s = self.offloader.new_state()
        for i in range(4):  # 4 fetches into max_resident=3 cache
            _, _, s = self.offloader.get_expert(i, s)
        self.assertLessEqual(len(s.resident), 3)
        self.assertGreater(s.n_evictions, 0)

    def test_evict_lru_removes_oldest(self):
        s = self.offloader.new_state()
        for i in range(3):
            _, _, s = self.offloader.get_expert(i, s)
        s2 = self.offloader.evict_lru(s)
        self.assertEqual(len(s2.resident), 2)
        self.assertEqual(s2.n_evictions, 1)

    def test_out_of_range_expert(self):
        s = self.offloader.new_state()
        with self.assertRaises(IndexError):
            self.offloader.get_expert(100, s)

    def test_stats_keys(self):
        s = self.offloader.new_state()
        _, _, s = self.offloader.get_expert(0, s)
        stats = self.offloader.stats(s)
        self.assertIn("n_resident", stats)
        self.assertIn("n_fetches", stats)


# ===========================================================================
# ExpertMerger Tests
# ===========================================================================

class TestExpertMergeConfig(unittest.TestCase):
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            ExpertMergeConfig(similarity_threshold=1.5)

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            ExpertMergeConfig(target_ratio=0.0)


class TestExpertMerger(unittest.TestCase):
    def setUp(self):
        self.cfg = ExpertMergeConfig(similarity_threshold=0.8, target_ratio=0.5, seed=0)
        self.merger = ExpertMerger(self.cfg)

    def test_merge_reduces_count(self):
        rng = np.random.default_rng(0)
        weights = [rng.standard_normal((64, 64)).astype(np.float32) for _ in range(8)]
        merged, memo = self.merger.merge(weights)
        self.assertLessEqual(len(merged), len(weights))

    def test_merge_empty_list(self):
        merged, memo = self.merger.merge([])
        self.assertEqual(merged, [])
        self.assertEqual(memo, {})

    def test_similarity_matrix_shape(self):
        rng = np.random.default_rng(1)
        weights = [rng.standard_normal(32).astype(np.float32) for _ in range(5)]
        sim = self.merger.similarity_matrix(weights)
        self.assertEqual(sim.shape, (5, 5))

    def test_similarity_matrix_diagonal_ones(self):
        w = [np.ones(16, dtype=np.float32) * 2.0]
        sim = self.merger.similarity_matrix(w)
        self.assertAlmostEqual(float(sim[0, 0]), 1.0, places=5)

    def test_similarity_matrix_symmetric(self):
        rng = np.random.default_rng(2)
        weights = [rng.standard_normal(32).astype(np.float32) for _ in range(4)]
        sim = self.merger.similarity_matrix(weights)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_compression_ratio(self):
        ratio = ExpertMerger.compression_ratio(8, 6)
        self.assertAlmostEqual(ratio, 0.75, places=5)

    def test_compression_ratio_zero_original(self):
        with self.assertRaises(ValueError):
            ExpertMerger.compression_ratio(0, 0)

    def test_merge_map_values_are_original_indices(self):
        rng = np.random.default_rng(3)
        # Create two very similar experts and one dissimilar
        w0 = np.ones(32, dtype=np.float32)
        w1 = np.ones(32, dtype=np.float32) + 1e-6
        w2 = np.zeros(32, dtype=np.float32)
        w2[0] = 1.0
        cfg = ExpertMergeConfig(similarity_threshold=0.0, target_ratio=0.6)
        merger = ExpertMerger(cfg)
        merged, memo = merger.merge([w0, w1, w2])
        for dst in memo.values():
            self.assertIn(dst, [0, 1, 2])

    def test_similarity_matrix_empty(self):
        sim = self.merger.similarity_matrix([])
        self.assertEqual(sim.shape, (0, 0))


# ===========================================================================
# LazyExpertLoader Tests
# ===========================================================================

class TestLazyExpertConfig(unittest.TestCase):
    def test_invalid_n_experts(self):
        with self.assertRaises(ValueError):
            LazyExpertConfig(n_experts=0)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            LazyExpertConfig(activation_threshold=-0.1)

    def test_invalid_idle_steps(self):
        with self.assertRaises(ValueError):
            LazyExpertConfig(idle_evict_steps=0)


class TestLazyExpertLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = LazyExpertConfig(
            n_experts=8, expert_dim=16, ffn_dim=32,
            activation_threshold=0.1, idle_evict_steps=5, seed=0
        )
        self.loader = LazyExpertLoader(self.cfg)

    def test_new_state_empty(self):
        s = self.loader.new_state()
        self.assertEqual(len(s.materialized), 0)
        self.assertEqual(s.step, 0)

    def test_forward_below_threshold_returns_zeros(self):
        s = self.loader.new_state()
        x = np.ones(16, dtype=np.float32)
        out, s2 = self.loader.forward(x, expert_idx=0, score=0.05, state=s)
        np.testing.assert_array_equal(out, np.zeros(16, dtype=np.float32))
        self.assertEqual(len(s2.materialized), 0)

    def test_forward_above_threshold_materializes(self):
        s = self.loader.new_state()
        x = np.ones(16, dtype=np.float32)
        _, s2 = self.loader.forward(x, expert_idx=0, score=0.5, state=s)
        self.assertIn(0, s2.materialized)
        self.assertEqual(s2.n_materializations, 1)

    def test_forward_output_shape_1d(self):
        s = self.loader.new_state()
        x = np.ones(16, dtype=np.float32)
        out, _ = self.loader.forward(x, expert_idx=0, score=0.5, state=s)
        self.assertEqual(out.shape, (16,))

    def test_forward_output_shape_2d(self):
        s = self.loader.new_state()
        x = np.ones((4, 16), dtype=np.float32)
        out, _ = self.loader.forward(x, expert_idx=0, score=0.5, state=s)
        self.assertEqual(out.shape, (4, 16))

    def test_idle_eviction(self):
        s = self.loader.new_state()
        x = np.ones(16, dtype=np.float32)
        # Materialise expert 0 at step 0
        _, s = self.loader.forward(x, expert_idx=0, score=0.5, state=s)
        # Advance step beyond idle_evict_steps (5) without activating
        for _ in range(6):
            _, s = self.loader.forward(x, expert_idx=1, score=0.05, state=s)
        self.assertNotIn(0, s.materialized)
        self.assertGreater(s.n_evictions, 0)

    def test_materialize_public(self):
        s = self.loader.new_state()
        s2 = self.loader._materialize(3, s)
        self.assertIn(3, s2.materialized)

    def test_materialize_idempotent(self):
        s = self.loader.new_state()
        s2 = self.loader._materialize(3, s)
        s3 = self.loader._materialize(3, s2)
        self.assertEqual(s3.n_materializations, 1)


# ===========================================================================
# ExpertActivationCache Tests
# ===========================================================================

class TestExpertCacheConfig(unittest.TestCase):
    def test_invalid_max_entries(self):
        with self.assertRaises(ValueError):
            ExpertCacheConfig(max_entries=0)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            ExpertCacheConfig(similarity_threshold=1.1)

    def test_invalid_expert_dim(self):
        with self.assertRaises(ValueError):
            ExpertCacheConfig(expert_dim=0)


class TestExpertActivationCache(unittest.TestCase):
    def setUp(self):
        self.cfg = ExpertCacheConfig(max_entries=4, similarity_threshold=0.95, expert_dim=16, seed=0)
        self.cache = ExpertActivationCache(self.cfg)

    def test_new_state_empty(self):
        s = self.cache.new_state()
        self.assertEqual(len(s.entries), 0)
        self.assertEqual(s.hits, 0)
        self.assertEqual(s.misses, 0)

    def test_lookup_miss_empty_cache(self):
        s = self.cache.new_state()
        x = np.ones(16, dtype=np.float32)
        out, s2 = self.cache.lookup(0, x, s)
        self.assertIsNone(out)
        self.assertEqual(s2.misses, 1)

    def test_store_then_lookup_hit(self):
        s = self.cache.new_state()
        x = np.array([1.0] * 16, dtype=np.float32)
        y = np.array([2.0] * 16, dtype=np.float32)
        s = self.cache.store(0, x, y, s)
        # Query with same vector
        cached, s2 = self.cache.lookup(0, x, s)
        self.assertIsNotNone(cached)
        self.assertEqual(s2.hits, 1)

    def test_store_then_lookup_different_expert(self):
        s = self.cache.new_state()
        x = np.ones(16, dtype=np.float32)
        y = np.ones(16, dtype=np.float32) * 2
        s = self.cache.store(0, x, y, s)
        cached, s2 = self.cache.lookup(1, x, s)  # wrong expert
        self.assertIsNone(cached)

    def test_eviction_at_capacity(self):
        s = self.cache.new_state()
        rng = np.random.default_rng(0)
        for i in range(6):
            x = rng.standard_normal(16).astype(np.float32)
            y = rng.standard_normal(16).astype(np.float32)
            s = self.cache.store(i, x, y, s)
        self.assertLessEqual(len(s.entries), 4)

    def test_hit_rate_zero_initially(self):
        s = self.cache.new_state()
        self.assertEqual(ExpertActivationCache.hit_rate(s), 0.0)

    def test_hit_rate_after_hits(self):
        s = self.cache.new_state()
        x = np.ones(16, dtype=np.float32)
        y = np.ones(16, dtype=np.float32) * 2
        s = self.cache.store(0, x, y, s)
        _, s = self.cache.lookup(0, x, s)   # hit
        _, s = self.cache.lookup(0, x * 100, s)  # miss (different direction)
        r = ExpertActivationCache.hit_rate(s)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_stats_keys(self):
        s = self.cache.new_state()
        stats = ExpertActivationCache.stats(s)
        self.assertIn("n_entries", stats)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)

    def test_low_similarity_is_cache_miss(self):
        s = self.cache.new_state()
        x = np.array([1.0, 0.0] + [0.0] * 14, dtype=np.float32)
        y = np.array([0.0, 1.0] + [0.0] * 14, dtype=np.float32)
        out_val = np.ones(16, dtype=np.float32)
        s = self.cache.store(0, x, out_val, s)
        # y is orthogonal to x — cosine sim = 0; should miss
        cached, s2 = self.cache.lookup(0, y, s)
        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
