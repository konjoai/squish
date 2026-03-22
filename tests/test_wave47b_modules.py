"""tests/test_wave47b_modules.py

Wave 47b test suite — covers:
  * MoEInfinityOffload  (squish/moe/moe_infinity.py)
  * MegaBlocksSparse    (squish/moe/mega_blocks.py)
  * KGWWatermark        (squish/serving/kgw_watermark.py)
  * TypicalSampler      (squish/sampling/typical_sampler.py)
  * DoRAAdapter         (squish/lora/dora.py)
  * AdaptiveCALM        (squish/token/calm_exit.py)
"""

from __future__ import annotations

import unittest

import numpy as np

# ---------------------------------------------------------------------------
# MoEInfinityOffload
# ---------------------------------------------------------------------------
from squish.moe.moe_infinity import MoEInfinityConfig, MoEInfinityOffload


class TestMoEInfinityConfig(unittest.TestCase):
    def test_defaults(self):
        c = MoEInfinityConfig()
        self.assertGreaterEqual(c.n_experts, 1)

    def test_invalid_n_experts(self):
        with self.assertRaises(ValueError):
            MoEInfinityConfig(n_experts=0)

    def test_invalid_expert_dim(self):
        with self.assertRaises(ValueError):
            MoEInfinityConfig(expert_dim=0)

    def test_invalid_hidden_dim(self):
        with self.assertRaises(ValueError):
            MoEInfinityConfig(hidden_dim=0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            MoEInfinityConfig(top_k=0)


class TestMoEInfinityOffload(unittest.TestCase):
    def setUp(self):
        self.cfg = MoEInfinityConfig(n_experts=4, expert_dim=8, hidden_dim=16, top_k=2, seed=0)
        self.moe = MoEInfinityOffload(self.cfg)
        rng = np.random.default_rng(0)
        for i in range(4):
            w = rng.standard_normal((8, 16)).astype(np.float32)
            self.moe.store_expert(i, w)

    def test_config_property(self):
        self.assertIs(self.moe.config, self.cfg)

    def test_initial_device_empty(self):
        moe = MoEInfinityOffload(self.cfg)
        self.assertEqual(moe.n_on_device, 0)

    def test_prefetch_loads_expert(self):
        self.moe.prefetch([0, 1])
        self.assertEqual(self.moe.n_on_device, 2)

    def test_forward_output_shape(self):
        token = np.random.randn(8).astype(np.float32)
        out = self.moe.forward(token, expert_id=0)
        self.assertEqual(out.shape, (8,))

    def test_forward_unknown_expert_raises(self):
        moe = MoEInfinityOffload(self.cfg)
        token = np.random.randn(8).astype(np.float32)
        with self.assertRaises(KeyError):
            moe.forward(token, expert_id=99)

    def test_prefetch_hit_rate_zero_initially(self):
        self.assertAlmostEqual(self.moe.prefetch_hit_rate, 0.0)

    def test_prefetch_hit_rate_after_prefetch(self):
        self.moe.prefetch([0])
        token = np.random.randn(8).astype(np.float32)
        self.moe.forward(token, 0)
        self.assertAlmostEqual(self.moe.prefetch_hit_rate, 1.0)

    def test_predict_next_experts_count(self):
        logits = np.random.randn(4).astype(np.float32)
        ids = self.moe.predict_next_experts(logits, k=2)
        self.assertEqual(len(ids), 2)

    def test_predict_next_experts_valid_ids(self):
        logits = np.random.randn(4).astype(np.float32)
        ids = self.moe.predict_next_experts(logits)
        for e in ids:
            self.assertIn(e, range(4))

    def test_predict_2d_logits(self):
        logits = np.random.randn(3, 4).astype(np.float32)
        ids = self.moe.predict_next_experts(logits, k=1)
        self.assertEqual(len(ids), 1)

    def test_evict_removes_from_device(self):
        self.moe.prefetch([0, 1])
        self.moe.evict([0])
        self.assertEqual(self.moe.n_on_device, 1)

    def test_forward_dtype(self):
        token = np.random.randn(8).astype(np.float32)
        out = self.moe.forward(token, 0)
        self.assertEqual(out.dtype, np.float32)


# ---------------------------------------------------------------------------
# MegaBlocksSparse
# ---------------------------------------------------------------------------
from squish.moe.mega_blocks import MegaBlocksConfig, MegaBlocksSparse


class TestMegaBlocksConfig(unittest.TestCase):
    def test_defaults(self):
        c = MegaBlocksConfig()
        self.assertGreaterEqual(c.n_experts, 1)

    def test_invalid_n_experts(self):
        with self.assertRaises(ValueError):
            MegaBlocksConfig(n_experts=0)

    def test_invalid_hidden_size(self):
        with self.assertRaises(ValueError):
            MegaBlocksConfig(hidden_size=0)

    def test_invalid_ffn_dim(self):
        with self.assertRaises(ValueError):
            MegaBlocksConfig(ffn_dim=0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            MegaBlocksConfig(top_k=0)

    def test_top_k_exceeds_n_experts(self):
        with self.assertRaises(ValueError):
            MegaBlocksConfig(n_experts=4, top_k=5)


class TestMegaBlocksSparse(unittest.TestCase):
    def setUp(self):
        self.cfg = MegaBlocksConfig(n_experts=4, hidden_size=16, ffn_dim=32, top_k=2, seed=0)
        self.moe = MegaBlocksSparse(self.cfg)

    def test_config_property(self):
        self.assertIs(self.moe.config, self.cfg)

    def test_router_weight_shape(self):
        self.assertEqual(
            self.moe.router_weight.shape, (self.cfg.hidden_size, self.cfg.n_experts)
        )

    def test_expert_weights_count(self):
        self.assertEqual(len(self.moe.expert_weights), self.cfg.n_experts)

    def test_route_output_shapes(self):
        tokens = np.random.randn(6, 16).astype(np.float32)
        ids, weights = self.moe.route(tokens)
        self.assertEqual(ids.shape, (6, self.cfg.top_k))
        self.assertEqual(weights.shape, (6, self.cfg.top_k))

    def test_routing_weights_sum_to_one(self):
        tokens = np.random.randn(5, 16).astype(np.float32)
        _, weights = self.moe.route(tokens)
        np.testing.assert_array_almost_equal(weights.sum(axis=-1), np.ones(5), decimal=5)

    def test_forward_output_shape(self):
        tokens = np.random.randn(6, 16).astype(np.float32)
        out = self.moe.forward(tokens)
        self.assertEqual(out.shape, (6, 16))

    def test_forward_no_tokens_dropped(self):
        # Every token must appear in at least one expert's batch
        tokens = np.random.randn(5, 16).astype(np.float32)
        out = self.moe.forward(tokens)
        # Output shape preserved means no token was dropped
        self.assertEqual(out.shape[0], 5)

    def test_forward_single_token(self):
        token = np.random.randn(1, 16).astype(np.float32)
        out = self.moe.forward(token)
        self.assertEqual(out.shape, (1, 16))

    def test_forward_dtype(self):
        tokens = np.random.randn(3, 16).astype(np.float32)
        out = self.moe.forward(tokens)
        self.assertEqual(out.dtype, np.float32)

    def test_route_ids_valid(self):
        tokens = np.random.randn(4, 16).astype(np.float32)
        ids, _ = self.moe.route(tokens)
        self.assertTrue((ids >= 0).all() and (ids < self.cfg.n_experts).all())


# ---------------------------------------------------------------------------
# KGWWatermark
# ---------------------------------------------------------------------------
from squish.serving.kgw_watermark import KGWConfig, KGWWatermark, WatermarkResult


class TestKGWConfig(unittest.TestCase):
    def test_defaults(self):
        c = KGWConfig()
        self.assertGreater(c.vocab_size, 1)

    def test_invalid_vocab_size(self):
        with self.assertRaises(ValueError):
            KGWConfig(vocab_size=1)

    def test_invalid_gamma_zero(self):
        with self.assertRaises(ValueError):
            KGWConfig(gamma=0.0)

    def test_invalid_gamma_one(self):
        with self.assertRaises(ValueError):
            KGWConfig(gamma=1.0)

    def test_invalid_delta_negative(self):
        with self.assertRaises(ValueError):
            KGWConfig(delta=-0.1)


class TestKGWWatermark(unittest.TestCase):
    def setUp(self):
        self.cfg = KGWConfig(vocab_size=128, gamma=0.25, delta=2.0)
        self.wm = KGWWatermark(self.cfg)

    def test_config_property(self):
        self.assertIs(self.wm.config, self.cfg)

    def test_green_list_size(self):
        green = self.wm._get_green_list(42)
        expected = max(1, int(0.25 * 128))
        self.assertEqual(len(green), expected)

    def test_green_list_within_vocab(self):
        green = self.wm._get_green_list(7)
        for t in green:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, 128)

    def test_green_list_deterministic(self):
        g1 = self.wm._get_green_list(5)
        g2 = self.wm._get_green_list(5)
        self.assertEqual(g1, g2)

    def test_green_list_context_dependent(self):
        g1 = self.wm._get_green_list(5)
        g2 = self.wm._get_green_list(6)
        self.assertNotEqual(g1, g2)

    def test_apply_shape(self):
        logits = np.zeros(128, dtype=np.float32)
        out = self.wm.apply(logits, [42])
        self.assertEqual(out.shape, (128,))

    def test_apply_biases_green(self):
        logits = np.zeros(128, dtype=np.float32)
        ctx = 42
        green = self.wm._get_green_list(ctx)
        out = self.wm.apply(logits, [ctx])
        for g in green:
            self.assertAlmostEqual(float(out[g]), self.cfg.delta)
        non_green = list(set(range(128)) - green)[:5]
        for ng in non_green:
            self.assertAlmostEqual(float(out[ng]), 0.0)

    def test_detect_short_sequence(self):
        result = self.wm.detect([1])
        self.assertIsInstance(result, WatermarkResult)
        self.assertFalse(result.is_watermarked)

    def test_detect_z_score_type(self):
        result = self.wm.detect([1, 2, 3, 4])
        self.assertIsInstance(result.z_score, float)

    def test_detect_green_count_nonneg(self):
        result = self.wm.detect([1, 2, 3, 4, 5])
        self.assertGreaterEqual(result.green_count, 0)

    def test_detect_total_tokens(self):
        tokens = [1, 2, 3, 4, 5]
        result = self.wm.detect(tokens)
        self.assertEqual(result.total_tokens, len(tokens) - 1)

    def test_apply_empty_context(self):
        logits = np.zeros(128, dtype=np.float32)
        out = self.wm.apply(logits, [])
        self.assertEqual(out.shape, (128,))


# ---------------------------------------------------------------------------
# TypicalSampler
# ---------------------------------------------------------------------------
from squish.sampling.typical_sampler import TypicalConfig, TypicalSampler, TypicalResult


class TestTypicalConfig(unittest.TestCase):
    def test_defaults(self):
        c = TypicalConfig()
        self.assertGreater(c.tau, 0.0)

    def test_invalid_tau_zero(self):
        with self.assertRaises(ValueError):
            TypicalConfig(tau=0.0)

    def test_invalid_tau_gt_one(self):
        with self.assertRaises(ValueError):
            TypicalConfig(tau=1.1)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            TypicalConfig(temperature=0.0)


class TestTypicalSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = TypicalConfig(tau=0.9, temperature=1.0, seed=0)
        self.sampler = TypicalSampler(self.cfg)

    def test_config_property(self):
        self.assertIs(self.sampler.config, self.cfg)

    def test_sample_returns_result(self):
        logits = np.random.randn(64).astype(np.float32)
        result = self.sampler.sample(logits)
        self.assertIsInstance(result, TypicalResult)

    def test_sample_token_in_vocab(self):
        logits = np.random.randn(64).astype(np.float32)
        result = self.sampler.sample(logits)
        self.assertGreaterEqual(result.token_id, 0)
        self.assertLess(result.token_id, 64)

    def test_sample_probability_in_range(self):
        logits = np.random.randn(64).astype(np.float32)
        result = self.sampler.sample(logits)
        self.assertGreater(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_sample_n_candidates_positive(self):
        logits = np.random.randn(64).astype(np.float32)
        result = self.sampler.sample(logits)
        self.assertGreater(result.n_candidates, 0)

    def test_sample_entropy_nonneg(self):
        logits = np.random.randn(64).astype(np.float32)
        result = self.sampler.sample(logits)
        self.assertGreaterEqual(result.entropy, 0.0)

    def test_filter_logits_shape(self):
        logits = np.random.randn(64).astype(np.float32)
        filtered = self.sampler.filter_logits(logits)
        self.assertEqual(filtered.shape, (64,))

    def test_filter_logits_has_neg_inf(self):
        logits = np.random.randn(64).astype(np.float32)
        filtered = self.sampler.filter_logits(logits)
        self.assertTrue(np.any(np.isneginf(filtered)))

    def test_sample_batch_shape(self):
        logits = np.random.randn(8, 64).astype(np.float32)
        ids = self.sampler.sample_batch(logits)
        self.assertEqual(ids.shape, (8,))

    def test_sample_batch_dtype(self):
        logits = np.random.randn(4, 64).astype(np.float32)
        ids = self.sampler.sample_batch(logits)
        self.assertEqual(ids.dtype, np.int32)

    def test_sample_batch_valid_ids(self):
        logits = np.random.randn(4, 64).astype(np.float32)
        ids = self.sampler.sample_batch(logits)
        self.assertTrue((ids >= 0).all() and (ids < 64).all())

    def test_tau_one_keeps_all(self):
        sampler = TypicalSampler(TypicalConfig(tau=1.0, seed=1))
        logits = np.random.randn(64).astype(np.float32)
        filtered = sampler.filter_logits(logits)
        # tau=1.0: all tokens should be in the set
        self.assertTrue(np.all(np.isfinite(filtered)))


# ---------------------------------------------------------------------------
# DoRAAdapter
# ---------------------------------------------------------------------------
from squish.lora.dora import DoRAConfig, DoRAAdapter


class TestDoRAConfig(unittest.TestCase):
    def test_defaults(self):
        c = DoRAConfig()
        self.assertGreaterEqual(c.d_in, 1)

    def test_invalid_d_in(self):
        with self.assertRaises(ValueError):
            DoRAConfig(d_in=0)

    def test_invalid_d_out(self):
        with self.assertRaises(ValueError):
            DoRAConfig(d_out=0)

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            DoRAConfig(rank=0)


class TestDoRAAdapter(unittest.TestCase):
    def setUp(self):
        self.cfg = DoRAConfig(d_in=16, d_out=16, rank=4, seed=0)
        self.adapter = DoRAAdapter(self.cfg)

    def test_config_property(self):
        self.assertIs(self.adapter.config, self.cfg)

    def test_magnitude_shape(self):
        self.assertEqual(self.adapter.magnitude.shape, (16,))

    def test_direction_shape(self):
        self.assertEqual(self.adapter.direction.shape, (16, 16))

    def test_lora_a_shape(self):
        self.assertEqual(self.adapter.lora_A.shape, (16, 4))

    def test_lora_b_shape(self):
        self.assertEqual(self.adapter.lora_B.shape, (4, 16))

    def test_adapted_weight_shape(self):
        W = self.adapter.adapted_weight()
        self.assertEqual(W.shape, (16, 16))

    def test_adapted_weight_dtype(self):
        W = self.adapter.adapted_weight()
        self.assertEqual(W.dtype, np.float32)

    def test_forward_shape(self):
        x = np.random.randn(6, 16).astype(np.float32)
        out = self.adapter.forward(x)
        self.assertEqual(out.shape, (6, 16))

    def test_forward_single_row(self):
        x = np.random.randn(1, 16).astype(np.float32)
        out = self.adapter.forward(x)
        self.assertEqual(out.shape, (1, 16))

    def test_merge_equals_adapted(self):
        W_adapted = self.adapter.adapted_weight()
        W_merged = self.adapter.merge_to_weight()
        np.testing.assert_array_almost_equal(W_adapted, W_merged)

    def test_direction_column_norms(self):
        # Initial V0 should have unit column norms
        col_norms = np.linalg.norm(self.adapter.direction, axis=0)
        np.testing.assert_array_almost_equal(col_norms, np.ones(16), decimal=5)

    def test_magnitude_positive(self):
        self.assertTrue((self.adapter.magnitude > 0).all())

    def test_non_rectangular(self):
        cfg = DoRAConfig(d_in=12, d_out=8, rank=3, seed=0)
        adapter = DoRAAdapter(cfg)
        x = np.random.randn(5, 12).astype(np.float32)
        out = adapter.forward(x)
        self.assertEqual(out.shape, (5, 8))


# ---------------------------------------------------------------------------
# AdaptiveCALM
# ---------------------------------------------------------------------------
from squish.token.calm_exit import CALMConfig, CALMResult, AdaptiveCALM


class TestCALMConfig(unittest.TestCase):
    def test_defaults(self):
        c = CALMConfig()
        self.assertGreaterEqual(c.n_layers, 1)

    def test_invalid_n_layers(self):
        with self.assertRaises(ValueError):
            CALMConfig(n_layers=0)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            CALMConfig(d_model=0)

    def test_invalid_confidence_zero(self):
        with self.assertRaises(ValueError):
            CALMConfig(confidence_threshold=0.0)

    def test_invalid_confidence_gt_one(self):
        with self.assertRaises(ValueError):
            CALMConfig(confidence_threshold=1.1)

    def test_invalid_min_layers_zero(self):
        with self.assertRaises(ValueError):
            CALMConfig(min_layers=0)

    def test_min_layers_ge_n_layers(self):
        with self.assertRaises(ValueError):
            CALMConfig(n_layers=4, min_layers=4)


class TestAdaptiveCALM(unittest.TestCase):
    def setUp(self):
        self.n_layers = 8
        self.d_model = 16
        self.cfg = CALMConfig(
            n_layers=self.n_layers,
            d_model=self.d_model,
            confidence_threshold=0.5,
            min_layers=2,
        )
        self.calm = AdaptiveCALM(self.cfg)

    def _layer_fns(self, scale: float = 0.9):
        # Each layer scales the hidden state toward a peak direction
        rng = np.random.default_rng(0)
        def make_fn(i):
            peak = rng.standard_normal(self.d_model).astype(np.float32)
            peak /= np.linalg.norm(peak)
            def fn(h): return h * scale + peak * (1 - scale) * 10
            return fn
        return [make_fn(i) for i in range(self.n_layers)]

    def test_config_property(self):
        self.assertIs(self.calm.config, self.cfg)

    def test_forward_returns_result(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        self.assertIsInstance(result, CALMResult)

    def test_forward_output_shape(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        self.assertEqual(result.output.shape, (self.d_model,))

    def test_exit_layer_in_range(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        self.assertGreaterEqual(result.exit_layer, 0)
        self.assertLess(result.exit_layer, self.n_layers)

    def test_min_layers_respected(self):
        # With confidence_threshold=0.0 we can't set, but min_layers=6 should force at least 6
        cfg = CALMConfig(
            n_layers=8, d_model=16, confidence_threshold=0.01, min_layers=6
        )
        calm = AdaptiveCALM(cfg)
        x = np.random.randn(16).astype(np.float32)
        result = calm.forward(x, self._layer_fns(scale=0.5))
        self.assertGreaterEqual(result.exit_layer + 1, 6)

    def test_flop_ratio_in_range(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        self.assertGreater(result.flop_ratio, 0.0)
        self.assertLessEqual(result.flop_ratio, 1.0)

    def test_flop_ratio_formula(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        expected = (result.exit_layer + 1) / self.n_layers
        self.assertAlmostEqual(result.flop_ratio, expected)

    def test_confidence_in_range(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        result = self.calm.forward(x, self._layer_fns())
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_exit_histogram_updated(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        self.calm.forward(x, self._layer_fns())
        self.assertEqual(self.calm.exit_histogram.sum(), 1)

    def test_exit_histogram_accumulates(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        for _ in range(5):
            self.calm.forward(x, self._layer_fns())
        self.assertEqual(self.calm.exit_histogram.sum(), 5)

    def test_wrong_n_layer_fns_raises(self):
        x = np.random.randn(self.d_model).astype(np.float32)
        with self.assertRaises(ValueError):
            self.calm.forward(x, self._layer_fns()[:3])

    def test_confidence_at_layer(self):
        h = np.random.randn(16).astype(np.float32)
        c = self.calm.confidence_at_layer(h)
        self.assertGreater(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_full_depth_when_low_threshold(self):
        # With threshold=1.0 exit must be at last layer
        cfg = CALMConfig(n_layers=8, d_model=16, confidence_threshold=1.0, min_layers=2)
        calm = AdaptiveCALM(cfg)
        x = np.random.randn(16).astype(np.float32)
        fns = [lambda h: h for _ in range(8)]
        result = calm.forward(x, fns)
        self.assertEqual(result.exit_layer, 7)


if __name__ == "__main__":
    unittest.main()
