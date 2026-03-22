"""Tests for Wave 50a inference modules: SparseGPT, MixtureOfDepths, LeanKV."""

from __future__ import annotations

import unittest

import numpy as np

from squish.model.sparse_gpt import (
    SparseGPTConfig,
    SparseGPTPruner,
    SparseGPTResult,
)
from squish.model.mix_of_depths import (
    MixtureOfDepths,
    MixtureOfDepthsConfig,
    MoDLayerResult,
)
from squish.kv.lean_kv import (
    LeanKVConfig,
    LeanKVQuant,
    LeanKVState,
)


# ---------------------------------------------------------------------------
# SparseGPTConfig
# ---------------------------------------------------------------------------


class TestSparseGPTConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SparseGPTConfig()
        self.assertAlmostEqual(cfg.sparsity_ratio, 0.5)
        self.assertEqual(cfg.block_size, 128)
        self.assertTrue(cfg.update_weights)
        self.assertFalse(cfg.structured)
        self.assertAlmostEqual(cfg.damp_pct, 0.01)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = SparseGPTConfig(sparsity_ratio=0.3, block_size=64, update_weights=False, seed=42)
        self.assertAlmostEqual(cfg.sparsity_ratio, 0.3)
        self.assertEqual(cfg.block_size, 64)
        self.assertFalse(cfg.update_weights)

    def test_sparsity_ratio_zero_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(sparsity_ratio=0.0)

    def test_sparsity_ratio_one_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(sparsity_ratio=1.0)

    def test_sparsity_ratio_negative_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(sparsity_ratio=-0.1)

    def test_block_size_zero_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(block_size=0)

    def test_damp_pct_zero_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(damp_pct=0.0)

    def test_damp_pct_one_raises(self):
        with self.assertRaises(ValueError):
            SparseGPTConfig(damp_pct=1.0)

    def test_structured_flag(self):
        cfg = SparseGPTConfig(structured=True)
        self.assertTrue(cfg.structured)


# ---------------------------------------------------------------------------
# SparseGPTResult
# ---------------------------------------------------------------------------


class TestSparseGPTResult(unittest.TestCase):
    def test_defaults(self):
        r = SparseGPTResult(sparsity_achieved=0.5, n_params_pruned=50, n_params_total=100)
        self.assertEqual(r.layer_name, "")

    def test_compression_ratio_property(self):
        r = SparseGPTResult(sparsity_achieved=0.5, n_params_pruned=50, n_params_total=100)
        self.assertAlmostEqual(r.compression_ratio, 0.5)

    def test_compression_ratio_full_density(self):
        r = SparseGPTResult(sparsity_achieved=0.0, n_params_pruned=0, n_params_total=100)
        self.assertAlmostEqual(r.compression_ratio, 1.0)

    def test_layer_name_set(self):
        r = SparseGPTResult(sparsity_achieved=0.3, n_params_pruned=30, n_params_total=100, layer_name="attn.q")
        self.assertEqual(r.layer_name, "attn.q")

    def test_n_params_consistency(self):
        r = SparseGPTResult(sparsity_achieved=0.4, n_params_pruned=40, n_params_total=100)
        self.assertEqual(r.n_params_pruned + int(r.n_params_total * r.compression_ratio), r.n_params_total)


# ---------------------------------------------------------------------------
# SparseGPTPruner
# ---------------------------------------------------------------------------


class TestSparseGPTPruner(unittest.TestCase):
    def _make_W(self, rows=32, cols=64, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((rows, cols)).astype(np.float32)

    def test_prune_weight_returns_tuple(self):
        W = self._make_W()
        pruner = SparseGPTPruner(SparseGPTConfig(sparsity_ratio=0.5))
        W_p, result = pruner.prune_weight(W)
        self.assertIsInstance(W_p, np.ndarray)
        self.assertIsInstance(result, SparseGPTResult)

    def test_prune_weight_shape_preserved(self):
        W = self._make_W(8, 16)
        pruner = SparseGPTPruner(SparseGPTConfig())
        W_p, _ = pruner.prune_weight(W)
        self.assertEqual(W_p.shape, W.shape)

    def test_prune_weight_sparsity_roughly_correct(self):
        W = self._make_W(64, 256, seed=7)
        pruner = SparseGPTPruner(SparseGPTConfig(sparsity_ratio=0.5, block_size=32))
        W_p, result = pruner.prune_weight(W)
        self.assertGreater(result.sparsity_achieved, 0.0)
        self.assertLessEqual(result.sparsity_achieved, 1.0)

    def test_prune_weight_non_negative_pruned_count(self):
        W = self._make_W()
        pruner = SparseGPTPruner(SparseGPTConfig())
        _, result = pruner.prune_weight(W)
        self.assertGreaterEqual(result.n_params_pruned, 0)
        self.assertEqual(result.n_params_total, W.size)

    def test_prune_weight_non_2d_raises(self):
        pruner = SparseGPTPruner(SparseGPTConfig())
        with self.assertRaises(ValueError):
            pruner.prune_weight(np.ones((4, 4, 4)))

    def test_prune_weight_with_custom_hessian(self):
        W = self._make_W(8, 16)
        H = np.eye(16, dtype=np.float32)
        pruner = SparseGPTPruner(SparseGPTConfig())
        W_p, result = pruner.prune_weight(W, H=H)
        self.assertEqual(W_p.shape, W.shape)

    def test_prune_weight_no_update_weights(self):
        W = self._make_W(16, 32)
        pruner = SparseGPTPruner(SparseGPTConfig(update_weights=False, sparsity_ratio=0.5))
        W_p, result = pruner.prune_weight(W)
        self.assertGreater(result.n_params_pruned, 0)

    def test_prune_model_dict_input(self):
        weights = {"layer1": self._make_W(8, 16), "layer2": self._make_W(8, 16)}
        pruner = SparseGPTPruner(SparseGPTConfig())
        pruned, results = pruner.prune_model(weights)
        self.assertEqual(set(pruned.keys()), set(weights.keys()))
        self.assertEqual(len(results), 2)

    def test_prune_model_layer_name_set(self):
        weights = {"attn.q": self._make_W(8, 16)}
        pruner = SparseGPTPruner(SparseGPTConfig())
        _, results = pruner.prune_model(weights)
        self.assertEqual(results[0].layer_name, "attn.q")

    def test_prune_model_with_hessians(self):
        W = self._make_W(8, 16)
        weights = {"x": W}
        hessians = {"x": np.eye(16, dtype=np.float32)}
        pruner = SparseGPTPruner(SparseGPTConfig())
        pruned, _ = pruner.prune_model(weights, hessians=hessians)
        self.assertEqual(pruned["x"].shape, W.shape)

    def test_sparsity_report_all_zeros(self):
        W = np.zeros((8, 16), dtype=np.float32)
        pruner = SparseGPTPruner(SparseGPTConfig())
        report = pruner.sparsity_report({"x": W})
        self.assertAlmostEqual(report["x"], 1.0)

    def test_sparsity_report_all_nonzero(self):
        W = np.ones((8, 16), dtype=np.float32)
        pruner = SparseGPTPruner(SparseGPTConfig())
        report = pruner.sparsity_report({"x": W})
        self.assertAlmostEqual(report["x"], 0.0)

    def test_structured_pruning_returns_correct_shape(self):
        W = self._make_W(8, 32)
        pruner = SparseGPTPruner(SparseGPTConfig(structured=True, sparsity_ratio=0.5))
        W_p, result = pruner.prune_weight(W)
        self.assertEqual(W_p.shape[0], W.shape[0])

    def test_high_sparsity(self):
        W = self._make_W(16, 64)
        pruner = SparseGPTPruner(SparseGPTConfig(sparsity_ratio=0.9))
        W_p, result = pruner.prune_weight(W)
        self.assertGreater(result.sparsity_achieved, 0.0)

    def test_small_matrix(self):
        W = self._make_W(2, 4)
        pruner = SparseGPTPruner(SparseGPTConfig(sparsity_ratio=0.5, block_size=2))
        W_p, result = pruner.prune_weight(W)
        self.assertEqual(W_p.shape, W.shape)

    def test_result_params_total_matches_weight(self):
        W = self._make_W(12, 24)
        pruner = SparseGPTPruner(SparseGPTConfig())
        _, result = pruner.prune_weight(W)
        self.assertEqual(result.n_params_total, 12 * 24)


# ---------------------------------------------------------------------------
# MixtureOfDepthsConfig
# ---------------------------------------------------------------------------


class TestMixtureOfDepthsConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MixtureOfDepthsConfig()
        self.assertEqual(cfg.n_layers, 32)
        self.assertAlmostEqual(cfg.skip_ratio, 0.5)
        self.assertEqual(cfg.router_dim, 64)
        self.assertEqual(cfg.router_type, "linear")
        self.assertEqual(cfg.min_active_tokens, 1)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = MixtureOfDepthsConfig(n_layers=12, skip_ratio=0.25, router_type="threshold")
        self.assertEqual(cfg.n_layers, 12)
        self.assertAlmostEqual(cfg.skip_ratio, 0.25)

    def test_n_layers_zero_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(n_layers=0)

    def test_skip_ratio_negative_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(skip_ratio=-0.1)

    def test_skip_ratio_one_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(skip_ratio=1.0)

    def test_router_dim_zero_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(router_dim=0)

    def test_invalid_router_type_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(router_type="unknown")

    def test_min_active_tokens_zero_raises(self):
        with self.assertRaises(ValueError):
            MixtureOfDepthsConfig(min_active_tokens=0)

    def test_skip_ratio_zero_is_valid(self):
        cfg = MixtureOfDepthsConfig(skip_ratio=0.0)
        self.assertAlmostEqual(cfg.skip_ratio, 0.0)


# ---------------------------------------------------------------------------
# MoDLayerResult
# ---------------------------------------------------------------------------


class TestMoDLayerResult(unittest.TestCase):
    def _make_result(self, n_tokens=10, n_active=5):
        mask = np.zeros(n_tokens, dtype=bool)
        mask[:n_tokens - n_active] = True
        return MoDLayerResult(
            layer_idx=0,
            n_tokens=n_tokens,
            n_active=n_active,
            n_skipped=n_tokens - n_active,
            skip_mask=mask,
        )

    def test_active_ratio_property(self):
        r = self._make_result(10, 6)
        self.assertAlmostEqual(r.active_ratio, 0.6)

    def test_active_ratio_all_active(self):
        r = self._make_result(10, 10)
        self.assertAlmostEqual(r.active_ratio, 1.0)

    def test_layer_idx(self):
        r = self._make_result()
        self.assertEqual(r.layer_idx, 0)

    def test_skip_mask_shape(self):
        r = self._make_result(8, 4)
        self.assertEqual(r.skip_mask.shape, (8,))

    def test_n_active_plus_n_skipped_equals_n_tokens(self):
        r = self._make_result(12, 7)
        self.assertEqual(r.n_active + r.n_skipped, r.n_tokens)


# ---------------------------------------------------------------------------
# MixtureOfDepths
# ---------------------------------------------------------------------------


class TestMixtureOfDepths(unittest.TestCase):
    def _make_mod(self, n_layers=4, skip_ratio=0.5, router_type="linear"):
        cfg = MixtureOfDepthsConfig(n_layers=n_layers, skip_ratio=skip_ratio,
                                    router_dim=8, router_type=router_type, seed=0)
        return MixtureOfDepths(cfg)

    def _make_hs(self, n_tokens=10, hidden_dim=8, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32)

    def test_route_returns_result(self):
        mod = self._make_mod()
        hs = self._make_hs()
        result = mod.route(hs, 0)
        self.assertIsInstance(result, MoDLayerResult)

    def test_route_skip_mask_shape(self):
        mod = self._make_mod()
        hs = self._make_hs(10)
        result = mod.route(hs, 0)
        self.assertEqual(result.skip_mask.shape, (10,))

    def test_route_n_active_plus_n_skipped_equals_n_tokens(self):
        mod = self._make_mod()
        hs = self._make_hs(10)
        result = mod.route(hs, 0)
        self.assertEqual(result.n_active + result.n_skipped, result.n_tokens)

    def test_route_min_active_tokens_respected(self):
        cfg = MixtureOfDepthsConfig(n_layers=4, skip_ratio=0.99, router_dim=8, min_active_tokens=2, seed=0)
        mod = MixtureOfDepths(cfg)
        hs = self._make_hs(10)
        result = mod.route(hs, 0)
        self.assertGreaterEqual(result.n_active, 2)

    def test_route_invalid_layer_idx_raises(self):
        mod = self._make_mod(n_layers=4)
        hs = self._make_hs()
        with self.assertRaises(ValueError):
            mod.route(hs, 10)

    def test_apply_layer_shape_preserved(self):
        mod = self._make_mod()
        hs = self._make_hs(10, 8)
        result = mod.route(hs, 0)
        layer_out = np.zeros_like(hs[~result.skip_mask])
        out = mod.apply_layer(hs, layer_out, result)
        self.assertEqual(out.shape, hs.shape)

    def test_apply_layer_skipped_unchanged(self):
        mod = self._make_mod()
        hs = self._make_hs(10, 8)
        result = mod.route(hs, 0)
        layer_out = np.zeros_like(hs[~result.skip_mask])
        out = mod.apply_layer(hs, layer_out, result)
        np.testing.assert_array_equal(out[result.skip_mask], hs[result.skip_mask])

    def test_apply_layer_active_updated(self):
        mod = self._make_mod()
        hs = self._make_hs(10, 8)
        result = mod.route(hs, 0)
        layer_out = np.ones_like(hs[~result.skip_mask]) * 99.0
        out = mod.apply_layer(hs, layer_out, result)
        np.testing.assert_array_almost_equal(out[~result.skip_mask], 99.0)

    def test_apply_layer_mismatch_raises(self):
        mod = self._make_mod()
        hs = self._make_hs(10, 8)
        result = mod.route(hs, 0)
        wrong_out = np.zeros((result.n_active + 1, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            mod.apply_layer(hs, wrong_out, result)

    def test_expected_flop_ratio(self):
        mod = self._make_mod(skip_ratio=0.5)
        self.assertAlmostEqual(mod.expected_flop_ratio(), 0.5)

    def test_stats_accumulation(self):
        mod = self._make_mod(n_layers=2)
        hs = self._make_hs(10)
        mod.route(hs, 0)
        mod.route(hs, 0)
        stats = mod.layer_stats()
        self.assertIn(0, stats)
        self.assertGreater(stats[0], 0.0)

    def test_reset_stats(self):
        mod = self._make_mod()
        hs = self._make_hs(10)
        mod.route(hs, 0)
        mod.reset_stats()
        stats = mod.layer_stats()
        self.assertEqual(stats[0], 0.0)

    def test_router_weight_shape(self):
        cfg = MixtureOfDepthsConfig(n_layers=4, router_dim=16, seed=0)
        mod = MixtureOfDepths(cfg)
        w = mod.router_weight(0)
        self.assertEqual(w.shape, (16,))

    def test_threshold_router_type(self):
        mod = self._make_mod(router_type="threshold")
        hs = self._make_hs(10)
        result = mod.route(hs, 0)
        self.assertIsInstance(result, MoDLayerResult)

    def test_skip_ratio_zero_all_active(self):
        mod = self._make_mod(skip_ratio=0.0)
        hs = self._make_hs(10)
        result = mod.route(hs, 0)
        self.assertEqual(result.n_skipped, 0)


# ---------------------------------------------------------------------------
# LeanKVConfig
# ---------------------------------------------------------------------------


class TestLeanKVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LeanKVConfig()
        self.assertEqual(cfg.k_bits, 4)
        self.assertEqual(cfg.v_bits, 8)
        self.assertEqual(cfg.group_size, 32)
        self.assertFalse(cfg.per_tensor)
        self.assertFalse(cfg.symmetric)
        self.assertEqual(cfg.seed, 0)

    def test_valid_bits(self):
        cfg = LeanKVConfig(k_bits=2, v_bits=6)
        self.assertEqual(cfg.k_bits, 2)
        self.assertEqual(cfg.v_bits, 6)

    def test_k_bits_invalid_raises(self):
        with self.assertRaises(ValueError):
            LeanKVConfig(k_bits=1)

    def test_v_bits_invalid_raises(self):
        with self.assertRaises(ValueError):
            LeanKVConfig(v_bits=9)

    def test_group_size_negative_raises(self):
        with self.assertRaises(ValueError):
            LeanKVConfig(group_size=-1)

    def test_per_tensor_flag(self):
        cfg = LeanKVConfig(per_tensor=True)
        self.assertTrue(cfg.per_tensor)

    def test_symmetric_flag(self):
        cfg = LeanKVConfig(symmetric=True)
        self.assertTrue(cfg.symmetric)

    def test_group_size_zero_valid(self):
        cfg = LeanKVConfig(group_size=0)
        self.assertEqual(cfg.group_size, 0)


# ---------------------------------------------------------------------------
# LeanKVState
# ---------------------------------------------------------------------------


class TestLeanKVState(unittest.TestCase):
    def _make_state(self, n=64, k_bits=4, v_bits=8):
        k_q = np.zeros(n, dtype=np.uint8)
        v_q = np.zeros(n, dtype=np.uint8)
        k_sc = np.ones(2, dtype=np.float32)
        k_zp = np.zeros(2, dtype=np.float32)
        v_sc = np.ones(2, dtype=np.float32)
        v_zp = np.zeros(2, dtype=np.float32)
        return LeanKVState(k_q, v_q, k_sc, k_zp, v_sc, v_zp, k_bits, v_bits, (n,))

    def test_k_bytes_property(self):
        state = self._make_state(64, k_bits=4)
        self.assertEqual(state.k_bytes, 32)  # 64 * 4 / 8

    def test_v_bytes_property(self):
        state = self._make_state(64, v_bits=8)
        self.assertEqual(state.v_bytes, 64)  # 64 * 8 / 8

    def test_fp16_bytes_property(self):
        state = self._make_state(64)
        self.assertEqual(state.fp16_bytes, 128)  # 64 * 2

    def test_compression_ratio_positive(self):
        state = self._make_state(64)
        self.assertGreater(state.compression_ratio, 0.0)

    def test_compression_ratio_lean_better_than_fp16(self):
        state = self._make_state(64, k_bits=4, v_bits=8)
        # k=4 bits, v=8 bits → total 96 bits vs fp16 total = 256 bits
        self.assertGreater(state.compression_ratio, 1.0)


# ---------------------------------------------------------------------------
# LeanKVQuant
# ---------------------------------------------------------------------------


class TestLeanKVQuant(unittest.TestCase):
    def _kv(self, shape=(2, 16, 32), seed=0):
        rng = np.random.default_rng(seed)
        k = rng.standard_normal(shape).astype(np.float32)
        v = rng.standard_normal(shape).astype(np.float32)
        return k, v

    def test_quantize_kv_returns_state(self):
        lkv = LeanKVQuant(LeanKVConfig())
        k, v = self._kv()
        state = lkv.quantize_kv(k, v)
        self.assertIsInstance(state, LeanKVState)

    def test_quantize_kv_shape_mismatch_raises(self):
        lkv = LeanKVQuant(LeanKVConfig())
        k = np.ones((4, 4), dtype=np.float32)
        v = np.ones((4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            lkv.quantize_kv(k, v)

    def test_dequantize_kv_shape_preserved(self):
        lkv = LeanKVQuant(LeanKVConfig())
        k, v = self._kv((2, 8, 16))
        state = lkv.quantize_kv(k, v)
        k_rec, v_rec = lkv.dequantize_kv(state)
        self.assertEqual(k_rec.shape, k.shape)
        self.assertEqual(v_rec.shape, v.shape)

    def test_dequantize_kv_dtype_float32(self):
        lkv = LeanKVQuant(LeanKVConfig())
        k, v = self._kv()
        state = lkv.quantize_kv(k, v)
        k_rec, v_rec = lkv.dequantize_kv(state)
        self.assertEqual(k_rec.dtype, np.float32)
        self.assertEqual(v_rec.dtype, np.float32)

    def test_quantize_k_returns_triple(self):
        lkv = LeanKVQuant(LeanKVConfig())
        k = np.ones((4, 32), dtype=np.float32)
        result = lkv.quantize_k(k)
        self.assertEqual(len(result), 3)

    def test_quantize_v_returns_triple(self):
        lkv = LeanKVQuant(LeanKVConfig())
        v = np.ones((4, 32), dtype=np.float32)
        result = lkv.quantize_v(v)
        self.assertEqual(len(result), 3)

    def test_dequantize_k_shape(self):
        lkv = LeanKVQuant(LeanKVConfig(k_bits=4))
        k = np.random.default_rng(0).standard_normal((32,)).astype(np.float32)
        k_q, sc, zp = lkv.quantize_k(k)
        k_rec = lkv.dequantize_k(k_q, sc, zp)
        self.assertEqual(k_rec.shape, k_q.shape)

    def test_per_tensor_quantize(self):
        lkv = LeanKVQuant(LeanKVConfig(per_tensor=True))
        k, v = self._kv((4, 32))
        state = lkv.quantize_kv(k, v)
        k_rec, _ = lkv.dequantize_kv(state)
        self.assertEqual(k_rec.shape, k.shape)

    def test_symmetric_quantize(self):
        lkv = LeanKVQuant(LeanKVConfig(symmetric=True))
        k, v = self._kv((4, 32))
        state = lkv.quantize_kv(k, v)
        k_rec, _ = lkv.dequantize_kv(state)
        self.assertEqual(k_rec.shape, k.shape)

    def test_memory_bytes_keys(self):
        lkv = LeanKVQuant(LeanKVConfig(k_bits=4, v_bits=8))
        result = lkv.memory_bytes(n_heads=8, seq_len=128, head_dim=64)
        for key in ("k_bytes", "v_bytes", "total_bytes", "compression_ratio"):
            self.assertIn(key, result)

    def test_memory_bytes_compression_positive(self):
        lkv = LeanKVQuant(LeanKVConfig())
        result = lkv.memory_bytes(n_heads=8, seq_len=128, head_dim=64)
        self.assertGreater(result["compression_ratio"], 0.0)

    def test_v8_less_error_than_v4(self):
        """V with more bits should have lower quantization error."""
        k, v = self._kv((1, 128))
        lkv8 = LeanKVQuant(LeanKVConfig(v_bits=8, k_bits=4, per_tensor=True))
        lkv4 = LeanKVQuant(LeanKVConfig(v_bits=4, k_bits=4, per_tensor=True))
        s8 = lkv8.quantize_kv(k, v)
        s4 = lkv4.quantize_kv(k, v)
        _, v_rec8 = lkv8.dequantize_kv(s8)
        _, v_rec4 = lkv4.dequantize_kv(s4)
        err8 = float(np.mean((v_rec8 - v) ** 2))
        err4 = float(np.mean((v_rec4 - v) ** 2))
        self.assertLessEqual(err8, err4 + 1e-3)  # allow tiny tolerance


if __name__ == "__main__":
    unittest.main()
