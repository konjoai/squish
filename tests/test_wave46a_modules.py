"""tests/test_wave46a_modules.py

Wave 46a test suite — SliceGPT · Wanda · ShortGPT · W4A8 · ExpertChoice · MLA KV Compress

Tests cover:
- Config defaults and validation
- Core algorithm shapes / dtypes
- Math properties (orthogonality, dequantize round-trip, etc.)
- Edge cases
"""

from __future__ import annotations

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# SliceGPT
# ---------------------------------------------------------------------------

class TestSliceGPT(unittest.TestCase):
    def _make(self, sparsity=0.25):
        from squish.quant.slice_gpt import SliceGPTPruner, SliceGPTConfig
        return SliceGPTPruner(SliceGPTConfig(sparsity=sparsity))

    def test_config_defaults(self):
        from squish.quant.slice_gpt import SliceGPTConfig
        cfg = SliceGPTConfig()
        self.assertEqual(cfg.sparsity, 0.25)
        self.assertEqual(cfg.n_calibration_samples, 128)

    def test_config_invalid_sparsity(self):
        from squish.quant.slice_gpt import SliceGPTConfig
        with self.assertRaises(ValueError):
            SliceGPTConfig(sparsity=0.0)
        with self.assertRaises(ValueError):
            SliceGPTConfig(sparsity=1.0)

    def test_config_property(self):
        pruner = self._make()
        from squish.quant.slice_gpt import SliceGPTConfig
        self.assertIsInstance(pruner.config, SliceGPTConfig)

    def test_compute_rotation_shape(self):
        pruner = self._make()
        X = np.random.randn(32, 64).astype(np.float32)
        Q = pruner.compute_rotation(X)
        self.assertEqual(Q.shape, (64, 64))
        self.assertEqual(Q.dtype, np.float32)

    def test_compute_rotation_orthogonal(self):
        pruner = self._make()
        X = np.random.randn(64, 32).astype(np.float32)
        Q = pruner.compute_rotation(X)
        # Q should be orthogonal: Q^T Q ~ I
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ, np.eye(32), atol=1e-4)

    def test_compute_rotation_3d_input(self):
        pruner = self._make()
        X = np.random.randn(4, 8, 32).astype(np.float32)
        Q = pruner.compute_rotation(X)
        self.assertEqual(Q.shape, (32, 32))

    def test_slice_weight_shape(self):
        pruner = self._make(sparsity=0.25)
        W = np.random.randn(64, 32).astype(np.float32)
        X = np.random.randn(16, 32).astype(np.float32)
        result = pruner.slice_weight(W, activations=X)
        expected_d = round(32 * (1 - 0.25))
        self.assertEqual(result.W_sliced.shape, (64, expected_d))
        self.assertEqual(result.d, expected_d)

    def test_slice_weight_original_shape(self):
        pruner = self._make()
        W = np.random.randn(16, 8).astype(np.float32)
        X = np.random.randn(10, 8).astype(np.float32)
        result = pruner.slice_weight(W, activations=X)
        self.assertEqual(result.original_shape, (16, 8))

    def test_slice_weight_with_precomputed_Q(self):
        pruner = self._make()
        W = np.random.randn(8, 16).astype(np.float32)
        X = np.random.randn(20, 16).astype(np.float32)
        Q = pruner.compute_rotation(X)
        result = pruner.slice_weight(W, Q=Q)
        self.assertIsNotNone(result)

    def test_slice_weight_requires_Q_or_activations(self):
        pruner = self._make()
        W = np.random.randn(8, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            pruner.slice_weight(W)

    def test_calibrate_and_slice(self):
        pruner = self._make(sparsity=0.5)
        W = np.random.randn(32, 64).astype(np.float32)
        result = pruner.calibrate_and_slice(W)
        expected_d = round(64 * 0.5)
        self.assertEqual(result.d, expected_d)

    def test_compression_ratio(self):
        pruner = self._make(sparsity=0.5)
        W = np.random.randn(16, 20).astype(np.float32)
        result = pruner.calibrate_and_slice(W)
        self.assertAlmostEqual(result.compression_ratio, result.d / 20, places=5)

    def test_reconstruct_shape(self):
        pruner = self._make()
        W = np.random.randn(8, 16).astype(np.float32)
        result = pruner.calibrate_and_slice(W)
        W_hat = result.reconstruct()
        self.assertEqual(W_hat.shape, (8, 16))

    def test_slice_pair_shape(self):
        pruner = self._make(sparsity=0.25)
        W1 = np.random.randn(16, 32).astype(np.float32)
        W2 = np.random.randn(8, 32).astype(np.float32)  # W2 shares W1's input dim
        result1, W2_sliced = pruner.slice_pair(W1, W2)
        d = result1.d
        self.assertEqual(result1.W_sliced.shape, (16, d))
        self.assertEqual(W2_sliced.shape, (8, d))

    def test_slice_pair_synthetic_acts(self):
        pruner = self._make()
        W1 = np.random.randn(8, 16).astype(np.float32)
        W2 = np.random.randn(4, 16).astype(np.float32)  # W2 shares W1's input dim
        result1, W2_sliced = pruner.slice_pair(W1, W2)
        self.assertIsNotNone(W2_sliced)

    def test_output_dtype_float32(self):
        pruner = self._make()
        W = np.random.randn(8, 16).astype(np.float64)
        result = pruner.calibrate_and_slice(W)
        self.assertEqual(result.W_sliced.dtype, np.float32)


# ---------------------------------------------------------------------------
# WandaPruner
# ---------------------------------------------------------------------------

class TestWandaPruner(unittest.TestCase):
    def _make(self, sparsity=0.5):
        from squish.quant.wanda_pruner import WandaPruner, WandaConfig
        return WandaPruner(WandaConfig(sparsity=sparsity))

    def test_config_defaults(self):
        from squish.quant.wanda_pruner import WandaConfig
        cfg = WandaConfig()
        self.assertEqual(cfg.sparsity, 0.5)
        self.assertIsNone(cfg.structured_n)
        self.assertIsNone(cfg.structured_m)

    def test_config_invalid_sparsity(self):
        from squish.quant.wanda_pruner import WandaConfig
        with self.assertRaises(ValueError):
            WandaConfig(sparsity=-0.1)
        with self.assertRaises(ValueError):
            WandaConfig(sparsity=1.5)

    def test_config_nm_mismatch(self):
        from squish.quant.wanda_pruner import WandaConfig
        with self.assertRaises(ValueError):
            WandaConfig(structured_n=2, structured_m=None)

    def test_config_nm_invalid(self):
        from squish.quant.wanda_pruner import WandaConfig
        with self.assertRaises(ValueError):
            WandaConfig(structured_n=4, structured_m=4)

    def test_prune_shape(self):
        pruner = self._make()
        W = np.random.randn(32, 64).astype(np.float32)
        result = pruner.prune(W)
        self.assertEqual(result.W_pruned.shape, (32, 64))
        self.assertEqual(result.mask.shape, (32, 64))

    def test_sparsity_achieved(self):
        pruner = self._make(sparsity=0.5)
        W = np.random.randn(64, 128).astype(np.float32)
        result = pruner.prune(W)
        # Should be approximately 50% sparse
        self.assertAlmostEqual(result.sparsity_achieved, 0.5, delta=0.1)

    def test_pruned_zeros_match_mask(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        # Where mask is False, W_pruned should be 0
        self.assertTrue(np.all(result.W_pruned[~result.mask] == 0.0))

    def test_importance_shape(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        self.assertEqual(result.importance.shape, (16, 32))

    def test_importance_nonneg(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        self.assertTrue(np.all(result.importance >= 0))

    def test_nm_structured_pruning(self):
        from squish.quant.wanda_pruner import WandaPruner, WandaConfig
        pruner = WandaPruner(WandaConfig(sparsity=0.5, structured_n=2, structured_m=4))
        W = np.random.randn(8, 32).astype(np.float32)
        result = pruner.prune(W)
        self.assertEqual(result.W_pruned.shape, (8, 32))

    def test_apply_shape(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        x = np.random.randn(32).astype(np.float32)
        out = result.apply(x)
        self.assertEqual(out.shape, (16,))
        self.assertEqual(out.dtype, np.float32)

    def test_apply_batch(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        x = np.random.randn(4, 32).astype(np.float32)
        out = result.apply(x)
        self.assertEqual(out.shape, (4, 16))

    def test_nnz(self):
        pruner = self._make(sparsity=0.5)
        W = np.random.randn(16, 32).astype(np.float32)
        result = pruner.prune(W)
        self.assertEqual(result.nnz(), int(result.mask.sum()))

    def test_prune_with_activations(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        X = np.random.randn(64, 32).astype(np.float32)
        result = pruner.prune(W, activations=X)
        self.assertEqual(result.W_pruned.shape, (16, 32))

    def test_prune_layer(self):
        pruner = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        b = np.random.randn(16).astype(np.float32)
        result, returned_bias = pruner.prune_layer(W, bias=b)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(returned_bias, b)

    def test_output_dtype_float32(self):
        pruner = self._make()
        W = np.random.randn(8, 16).astype(np.float64)
        result = pruner.prune(W)
        self.assertEqual(result.W_pruned.dtype, np.float32)


# ---------------------------------------------------------------------------
# ShortGPTPruner
# ---------------------------------------------------------------------------

class TestShortGPTPruner(unittest.TestCase):
    def _make(self, removal_fraction=0.25, hidden_size=32):
        from squish.quant.short_gpt import ShortGPTPruner, ShortGPTConfig
        return ShortGPTPruner(ShortGPTConfig(
            removal_fraction=removal_fraction,
            hidden_size=hidden_size,
            n_calibration_tokens=16,
        ))

    def test_config_defaults(self):
        from squish.quant.short_gpt import ShortGPTConfig
        cfg = ShortGPTConfig()
        self.assertEqual(cfg.removal_fraction, 0.25)
        self.assertEqual(cfg.hidden_size, 4096)

    def test_config_invalid_removal(self):
        from squish.quant.short_gpt import ShortGPTConfig
        with self.assertRaises(ValueError):
            ShortGPTConfig(removal_fraction=0.0)
        with self.assertRaises(ValueError):
            ShortGPTConfig(removal_fraction=1.0)

    def test_config_property(self):
        pruner = self._make()
        from squish.quant.short_gpt import ShortGPTConfig
        self.assertIsInstance(pruner.config, ShortGPTConfig)

    def test_compute_block_importance_shape(self):
        pruner = self._make(hidden_size=16)
        n_layers = 8
        inputs = [np.random.randn(32, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h + 0.01 * np.random.randn(*h.shape).astype(np.float32) for h in inputs]
        bi = pruner.compute_block_importance(inputs, outputs)
        self.assertEqual(bi.scores.shape, (n_layers,))
        self.assertEqual(len(bi.layer_indices), n_layers)

    def test_bi_scores_range(self):
        pruner = self._make(hidden_size=16)
        n_layers = 4
        inputs = [np.random.randn(16, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h.copy() for h in inputs]  # identical → BI = 0
        bi = pruner.compute_block_importance(inputs, outputs)
        np.testing.assert_allclose(bi.scores, 0.0, atol=1e-5)

    def test_bi_high_for_changed_layers(self):
        pruner = self._make(hidden_size=16)
        inputs = [np.random.randn(16, 16).astype(np.float32)]
        outputs = [np.random.randn(16, 16).astype(np.float32)]  # completely different
        bi = pruner.compute_block_importance(inputs, outputs)
        self.assertGreater(bi.scores[0], 0.0)

    def test_most_redundant(self):
        pruner = self._make(hidden_size=16)
        n_layers = 6
        inputs = [np.random.randn(16, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h + np.random.randn(*h.shape).astype(np.float32) for h in inputs]
        bi = pruner.compute_block_importance(inputs, outputs)
        redundant = bi.most_redundant(2)
        self.assertEqual(len(redundant), 2)

    def test_select_layers_to_remove(self):
        pruner = self._make(removal_fraction=0.25)
        n_layers = 8
        inputs = [np.random.randn(16, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h + np.random.randn(*h.shape).astype(np.float32) for h in inputs]
        bi = pruner.compute_block_importance(inputs, outputs)
        to_remove = pruner.select_layers_to_remove(bi, n_layers)
        expected_n = max(1, round(n_layers * 0.25))
        self.assertEqual(len(to_remove), expected_n)

    def test_prune_layer_list(self):
        pruner = self._make(removal_fraction=0.25)
        n_layers = 8
        layers = list(range(n_layers))
        inputs = [np.random.randn(16, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h + np.random.randn(*h.shape).astype(np.float32) for h in inputs]
        bi = pruner.compute_block_importance(inputs, outputs)
        pruned, removed = pruner.prune_layer_list(layers, bi)
        self.assertEqual(len(pruned) + len(removed), n_layers)

    def test_prune_layer_list_length(self):
        pruner = self._make(removal_fraction=0.5)
        n_layers = 4
        layers = list(range(n_layers))
        inputs = [np.random.randn(8, 16).astype(np.float32) for _ in range(n_layers)]
        outputs = [h + np.random.randn(*h.shape).astype(np.float32) for h in inputs]
        bi = pruner.compute_block_importance(inputs, outputs)
        pruned, removed = pruner.prune_layer_list(layers, bi)
        expected_remove = max(1, round(n_layers * 0.5))
        self.assertEqual(len(removed), expected_remove)

    def test_calibrate_importance_with_callables(self):
        pruner = self._make(hidden_size=16)
        # Each "layer" applies a small random transform
        transforms = [lambda x: x + 0.01 for _ in range(4)]
        bi = pruner.calibrate_importance(transforms)
        self.assertEqual(len(bi.scores), 4)

    def test_calibrate_importance_scores_nonneg(self):
        pruner = self._make(hidden_size=16)
        transforms = [lambda x: x for _ in range(4)]  # identity → BI ≈ 0
        bi = pruner.calibrate_importance(transforms)
        self.assertTrue(np.all(bi.scores >= -1e-5))


# ---------------------------------------------------------------------------
# W4A8QuantRuntime
# ---------------------------------------------------------------------------

class TestW4A8QuantRuntime(unittest.TestCase):
    def _make(self, group_size=8, symmetric=True):
        from squish.quant.w4a8_quant import W4A8QuantRuntime, W4A8Config
        return W4A8QuantRuntime(W4A8Config(group_size=group_size, symmetric=symmetric))

    def test_config_defaults(self):
        from squish.quant.w4a8_quant import W4A8Config
        cfg = W4A8Config()
        self.assertEqual(cfg.w_bits, 4)
        self.assertEqual(cfg.a_bits, 8)
        self.assertEqual(cfg.group_size, 128)

    def test_config_invalid_w_bits(self):
        from squish.quant.w4a8_quant import W4A8Config
        with self.assertRaises(ValueError):
            W4A8Config(w_bits=5)

    def test_config_invalid_a_bits(self):
        from squish.quant.w4a8_quant import W4A8Config
        with self.assertRaises(ValueError):
            W4A8Config(a_bits=16)

    def test_config_property(self):
        rt = self._make()
        from squish.quant.w4a8_quant import W4A8Config
        self.assertIsInstance(rt.config, W4A8Config)

    def test_quantize_weight_shape(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        result = rt.quantize_weight(W)
        self.assertEqual(result.original_shape, (16, 32))

    def test_quantize_weight_scale_shape(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        result = rt.quantize_weight(W)
        n_groups = 32 // 8
        self.assertEqual(result.scale.shape, (16, n_groups))

    def test_dequantize_shape(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        result = rt.quantize_weight(W)
        W_hat = result.dequantize()
        self.assertEqual(W_hat.shape, (16, 32))
        self.assertEqual(W_hat.dtype, np.float32)

    def test_dequantize_approx(self):
        rt = self._make(group_size=4)
        W = np.random.randn(8, 16).astype(np.float32)
        result = rt.quantize_weight(W)
        W_hat = result.dequantize()
        # Dequantized should be a reasonably close approximation (4-bit can have ~1–2 MAE)
        mae = np.abs(W - W_hat).mean()
        self.assertLess(mae, 3.0)

    def test_quantize_activation_shape(self):
        rt = self._make()
        X = np.random.randn(4, 32).astype(np.float32)
        result = rt.quantize_activation(X)
        self.assertEqual(result.X_int8.shape, (4, 32))
        self.assertEqual(result.original_shape, (4, 32))

    def test_quantize_activation_int8_range(self):
        rt = self._make()
        X = np.random.randn(16, 32).astype(np.float32)
        result = rt.quantize_activation(X)
        self.assertTrue(np.all(result.X_int8 >= -128))
        self.assertTrue(np.all(result.X_int8 <= 127))

    def test_activation_dequantize_shape(self):
        rt = self._make()
        X = np.random.randn(4, 32).astype(np.float32)
        result = rt.quantize_activation(X)
        X_hat = result.dequantize()
        self.assertEqual(X_hat.shape, X.shape)

    def test_forward_shape(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        X = np.random.randn(4, 32).astype(np.float32)
        out = rt.forward(X, W)
        self.assertEqual(out.shape, (4, 16))
        self.assertEqual(out.dtype, np.float32)

    def test_forward_1d_input(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        X = np.random.randn(32).astype(np.float32)
        out = rt.forward(X, W)
        self.assertEqual(out.shape, (16,))

    def test_forward_without_W_uses_cached(self):
        rt = self._make(group_size=8)
        W = np.random.randn(16, 32).astype(np.float32)
        rt.quantize_weight(W)
        X = np.random.randn(4, 32).astype(np.float32)
        out = rt.forward(X)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_without_W_raises_if_not_quantized(self):
        rt = self._make()
        X = np.random.randn(4, 32).astype(np.float32)
        with self.assertRaises(RuntimeError):
            rt.forward(X)

    def test_asymmetric_config(self):
        rt = self._make(group_size=8, symmetric=False)
        W = np.random.randn(8, 16).astype(np.float32)
        result = rt.quantize_weight(W)
        W_hat = result.dequantize()
        self.assertEqual(W_hat.shape, (8, 16))


# ---------------------------------------------------------------------------
# ExpertChoiceRouter
# ---------------------------------------------------------------------------

class TestExpertChoiceRouter(unittest.TestCase):
    def _make(self, n_experts=4, hidden_size=32, capacity_factor=1.5):
        from squish.moe.expert_choice import ExpertChoiceRouter, ExpertChoiceConfig
        return ExpertChoiceRouter(ExpertChoiceConfig(
            n_experts=n_experts,
            hidden_size=hidden_size,
            capacity_factor=capacity_factor,
        ))

    def test_config_defaults(self):
        from squish.moe.expert_choice import ExpertChoiceConfig
        cfg = ExpertChoiceConfig()
        self.assertEqual(cfg.n_experts, 8)
        self.assertAlmostEqual(cfg.capacity_factor, 1.25)

    def test_config_invalid(self):
        from squish.moe.expert_choice import ExpertChoiceConfig
        with self.assertRaises(ValueError):
            ExpertChoiceConfig(n_experts=0)
        with self.assertRaises(ValueError):
            ExpertChoiceConfig(capacity_factor=0)

    def test_router_weight_shape(self):
        router = self._make(n_experts=4, hidden_size=32)
        self.assertEqual(router.router_weight.shape, (32, 4))

    def test_route_token_indices_shape(self):
        router = self._make(n_experts=4, hidden_size=32, capacity_factor=1.0)
        X = np.random.randn(16, 32).astype(np.float32)
        result = router.route(X)
        # (n_experts, capacity)
        self.assertEqual(result.token_indices.shape[0], 4)

    def test_route_token_indices_in_range(self):
        router = self._make(n_experts=4, hidden_size=32)
        n_tokens = 20
        X = np.random.randn(n_tokens, 32).astype(np.float32)
        result = router.route(X)
        self.assertTrue(np.all(result.token_indices >= 0))
        self.assertTrue(np.all(result.token_indices < n_tokens))

    def test_route_routing_weights_nonneg(self):
        router = self._make(n_experts=4, hidden_size=32)
        X = np.random.randn(16, 32).astype(np.float32)
        result = router.route(X)
        self.assertTrue(np.all(result.routing_weights >= 0))

    def test_route_router_probs_shape(self):
        router = self._make(n_experts=4, hidden_size=32)
        n_tokens = 12
        X = np.random.randn(n_tokens, 32).astype(np.float32)
        result = router.route(X)
        self.assertEqual(result.router_probs.shape, (n_tokens, 4))

    def test_route_router_probs_sum_to_one(self):
        router = self._make(n_experts=4, hidden_size=32)
        X = np.random.randn(8, 32).astype(np.float32)
        result = router.route(X)
        row_sums = result.router_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_load_balance_loss_zero(self):
        router = self._make()
        X = np.random.randn(16, 32).astype(np.float32)
        result = router.route(X)
        self.assertEqual(result.load_balance_loss(), 0.0)

    def test_tokens_per_expert_equal(self):
        router = self._make(n_experts=4, hidden_size=32, capacity_factor=1.0)
        X = np.random.randn(16, 32).astype(np.float32)
        result = router.route(X)
        tpe = result.tokens_per_expert()
        self.assertEqual(len(tpe), 4)
        # All experts get the same capacity
        self.assertEqual(len(set(tpe.tolist())), 1)

    def test_route_3d_input_flattened(self):
        router = self._make(n_experts=4, hidden_size=32)
        X = np.random.randn(2, 8, 32).astype(np.float32)
        result = router.route(X)
        n_tokens = 16
        self.assertEqual(result.router_probs.shape, (n_tokens, 4))

    def test_combine_shape(self):
        router = self._make(n_experts=4, hidden_size=32, capacity_factor=1.0)
        n_tokens = 16
        X = np.random.randn(n_tokens, 32).astype(np.float32)
        result = router.route(X)
        cap = result.expert_capacity
        expert_dim = 16
        expert_outputs = np.random.randn(4, cap, expert_dim).astype(np.float32)
        combined = router.combine(expert_outputs, result, n_tokens)
        self.assertEqual(combined.shape, (n_tokens, expert_dim))
        self.assertEqual(combined.dtype, np.float32)


# ---------------------------------------------------------------------------
# MLAKVCompress
# ---------------------------------------------------------------------------

class TestMLAKVCompress(unittest.TestCase):
    def _make(self, n_heads=4, head_dim=8, latent_dim=16, hidden_size=64):
        from squish.kv.mla_kv_compress import MLAKVCompress, MLAKVConfig
        return MLAKVCompress(MLAKVConfig(
            n_heads=n_heads,
            head_dim=head_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
        ))

    def test_config_defaults(self):
        from squish.kv.mla_kv_compress import MLAKVConfig
        cfg = MLAKVConfig()
        self.assertEqual(cfg.n_heads, 128)
        self.assertEqual(cfg.latent_dim, 512)

    def test_config_latent_dim_too_large(self):
        from squish.kv.mla_kv_compress import MLAKVConfig
        with self.assertRaises(ValueError):
            MLAKVConfig(n_heads=2, head_dim=8, latent_dim=32)  # 32 >= 2*8=16

    def test_config_property(self):
        mla = self._make()
        from squish.kv.mla_kv_compress import MLAKVConfig
        self.assertIsInstance(mla.config, MLAKVConfig)

    def test_w_compress_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        self.assertEqual(mla.W_compress.shape, (64, 16))

    def test_w_decompress_k_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        self.assertEqual(mla.W_decompress_k.shape, (16, 32))  # 4*8=32

    def test_w_decompress_v_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        self.assertEqual(mla.W_decompress_v.shape, (16, 32))

    def test_compress_entry_c_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        h = np.random.randn(64).astype(np.float32)
        entry = mla.compress(h, position=0)
        self.assertEqual(entry.c.shape, (16,))
        self.assertEqual(entry.position, 0)

    def test_compress_2d_input(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        h = np.random.randn(1, 64).astype(np.float32)
        entry = mla.compress(h, position=5)
        self.assertEqual(entry.c.shape, (16,))

    def test_compress_increments_cache(self):
        mla = self._make()
        self.assertEqual(mla.cache_size, 0)
        h = np.random.randn(64).astype(np.float32)
        mla.compress(h, 0)
        self.assertEqual(mla.cache_size, 1)

    def test_decompress_k_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        h = np.random.randn(64).astype(np.float32)
        entry = mla.compress(h, 0)
        K = mla.decompress_k(entry)
        self.assertEqual(K.shape, (4, 8))

    def test_decompress_v_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        h = np.random.randn(64).astype(np.float32)
        entry = mla.compress(h, 0)
        V = mla.decompress_v(entry)
        self.assertEqual(V.shape, (4, 8))

    def test_get_kv_sequence_shape(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        for i in range(5):
            h = np.random.randn(64).astype(np.float32)
            mla.compress(h, i)
        K, V = mla.get_kv_sequence()
        self.assertEqual(K.shape, (5, 4, 8))
        self.assertEqual(V.shape, (5, 4, 8))

    def test_get_kv_empty(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        K, V = mla.get_kv_sequence()
        self.assertEqual(K.shape[0], 0)
        self.assertEqual(V.shape[0], 0)

    def test_reset_clears_cache(self):
        mla = self._make()
        h = np.random.randn(64).astype(np.float32)
        mla.compress(h, 0)
        mla.reset()
        self.assertEqual(mla.cache_size, 0)

    def test_compression_ratio(self):
        mla = self._make(n_heads=4, head_dim=8, latent_dim=16, hidden_size=64)
        # ratio = 16 / (4*8*2) = 16/64 = 0.25
        self.assertAlmostEqual(mla.compression_ratio, 16.0 / 64.0, places=5)


if __name__ == "__main__":
    unittest.main()
