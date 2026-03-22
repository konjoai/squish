"""Tests for Wave 48a modules: SpQRQuantizer, AutoRoundQuantizer, OWQQuantizer."""

from __future__ import annotations

import unittest

import numpy as np

from squish.quant.spqr import SpQRConfig, SpQRQuantizer, SpQRResult
from squish.quant.auto_round import AutoRoundConfig, AutoRoundQuantizer, AutoRoundResult
from squish.quant.owq import OWQConfig, OWQQuantizer, OWQResult


# ---------------------------------------------------------------------------
# SpQR tests
# ---------------------------------------------------------------------------


class TestSpQRConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SpQRConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertAlmostEqual(cfg.outlier_threshold, 3.0)
        self.assertAlmostEqual(cfg.outlier_fraction, 0.02)
        self.assertEqual(cfg.bits, 3)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = SpQRConfig(group_size=16, bits=4, outlier_fraction=0.1)
        self.assertEqual(cfg.group_size, 16)
        self.assertEqual(cfg.bits, 4)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            SpQRConfig(group_size=0)

    def test_invalid_bits_low(self):
        with self.assertRaises(ValueError):
            SpQRConfig(bits=0)

    def test_invalid_bits_high(self):
        with self.assertRaises(ValueError):
            SpQRConfig(bits=9)

    def test_invalid_outlier_fraction_negative(self):
        with self.assertRaises(ValueError):
            SpQRConfig(outlier_fraction=-0.01)

    def test_invalid_outlier_fraction_one(self):
        with self.assertRaises(ValueError):
            SpQRConfig(outlier_fraction=1.0)

    def test_outlier_fraction_zero_is_valid(self):
        cfg = SpQRConfig(outlier_fraction=0.0)
        self.assertEqual(cfg.outlier_fraction, 0.0)


class TestSpQRResult(unittest.TestCase):
    def _make_result(self, rows=4, cols=8, n_outlier=2, group_size=4):
        q = np.zeros((rows, cols), dtype=np.int8)
        n_groups = (rows * cols + group_size - 1) // group_size
        scales = np.ones(n_groups, dtype=np.float32)
        zeros = np.zeros(n_groups, dtype=np.float32)
        sparse_indices = np.zeros((n_outlier, 2), dtype=np.int32)
        sparse_values = np.zeros(n_outlier, dtype=np.float32)
        return SpQRResult(
            quantized=q,
            scales=scales,
            zeros=zeros,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            shape=(rows, cols),
            group_size=group_size,
        )

    def test_effective_bits_is_float(self):
        result = self._make_result()
        self.assertIsInstance(result.effective_bits, float)

    def test_effective_bits_positive(self):
        result = self._make_result()
        self.assertGreater(result.effective_bits, 0)


class TestSpQRQuantizerConfig(unittest.TestCase):
    def test_default_config(self):
        q = SpQRQuantizer()
        self.assertIsInstance(q.config, SpQRConfig)

    def test_custom_config(self):
        cfg = SpQRConfig(bits=2, group_size=32)
        q = SpQRQuantizer(cfg)
        self.assertEqual(q.config.bits, 2)
        self.assertEqual(q.config.group_size, 32)


class TestSpQRQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.W = rng.standard_normal((16, 32)).astype(np.float32)
        self.cfg = SpQRConfig(group_size=16, outlier_fraction=0.1, bits=3, seed=0)
        self.quant = SpQRQuantizer(self.cfg)

    def test_returns_spqr_result(self):
        result = self.quant.quantize(self.W)
        self.assertIsInstance(result, SpQRResult)

    def test_result_shape(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.shape, (16, 32))

    def test_quantized_dtype_int8(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.quantized.dtype, np.int8)

    def test_scales_dtype_float32(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.scales.dtype, np.float32)

    def test_sparse_indices_is_2d(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.sparse_indices.ndim, 2)
        self.assertEqual(result.sparse_indices.shape[1], 2)

    def test_sparse_values_length_matches_indices(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.sparse_indices.shape[0], result.sparse_values.shape[0])

    def test_group_size_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.group_size, self.cfg.group_size)


class TestSpQRDequantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = SpQRQuantizer(SpQRConfig(group_size=8, bits=3, outlier_fraction=0.05))

    def test_dequantize_shape(self):
        result = self.quant.quantize(self.W)
        W_hat = self.quant.dequantize(result)
        self.assertEqual(W_hat.shape, (8, 16))

    def test_dequantize_dtype(self):
        result = self.quant.quantize(self.W)
        W_hat = self.quant.dequantize(result)
        self.assertEqual(W_hat.dtype, np.float32)

    def test_reconstruction_close(self):
        result = self.quant.quantize(self.W)
        W_hat = self.quant.dequantize(result)
        err = np.mean(np.abs(W_hat - self.W))
        self.assertLess(err, 1.0)


class TestSpQRForward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = SpQRQuantizer(SpQRConfig(group_size=8, bits=3, outlier_fraction=0.05))
        self.result = self.quant.quantize(self.W)

    def test_forward_output_shape(self):
        x = np.random.randn(4, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (4, 16))

    def test_forward_batched(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (2, 3, 16))

    def test_forward_1d(self):
        x = np.random.randn(8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (16,))


class TestSpQRIntQuant(unittest.TestCase):
    def test_int3_quant_group_returns_tuple(self):
        quant = SpQRQuantizer(SpQRConfig(group_size=16, bits=3))
        g = np.linspace(-1, 1, 16).astype(np.float32)
        q, scale, zero = quant._int3_quant_group(g)
        self.assertEqual(q.dtype, np.int8)
        self.assertIsInstance(scale, float)
        self.assertIsInstance(zero, float)

    def test_int3_quant_group_range(self):
        quant = SpQRQuantizer(SpQRConfig(bits=3))
        g = np.linspace(-2, 2, 64).astype(np.float32)
        q, _, _ = quant._int3_quant_group(g)
        self.assertTrue(np.all(q >= -4))
        self.assertTrue(np.all(q <= 3))


# ---------------------------------------------------------------------------
# AutoRound tests
# ---------------------------------------------------------------------------


class TestAutoRoundConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AutoRoundConfig()
        self.assertEqual(cfg.bits, 3)
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.n_steps, 512)
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = AutoRoundConfig(bits=2, group_size=64, n_steps=100, lr=5e-4)
        self.assertEqual(cfg.bits, 2)
        self.assertEqual(cfg.n_steps, 100)

    def test_invalid_bits_low(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(bits=0)

    def test_invalid_bits_high(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(bits=9)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(group_size=0)

    def test_invalid_n_steps(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(n_steps=0)

    def test_invalid_lr(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(lr=0.0)

    def test_invalid_lr_negative(self):
        with self.assertRaises(ValueError):
            AutoRoundConfig(lr=-1e-3)


class TestAutoRoundQuantizerConfig(unittest.TestCase):
    def test_default_config(self):
        q = AutoRoundQuantizer()
        self.assertIsInstance(q.config, AutoRoundConfig)

    def test_custom_config(self):
        cfg = AutoRoundConfig(bits=2, n_steps=10)
        q = AutoRoundQuantizer(cfg)
        self.assertEqual(q.config.bits, 2)
        self.assertEqual(q.config.n_steps, 10)


class TestAutoRoundQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = AutoRoundQuantizer(
            AutoRoundConfig(bits=3, group_size=8, n_steps=5, lr=1e-3, seed=0)
        )

    def test_returns_result(self):
        result = self.quant.quantize(self.W)
        self.assertIsInstance(result, AutoRoundResult)

    def test_result_shape(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.shape, (8, 16))

    def test_quantized_dtype(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.quantized.dtype, np.int8)

    def test_scales_shape(self):
        result = self.quant.quantize(self.W)
        n_groups = 8 * (16 // 8)
        self.assertEqual(result.scales.shape[0], n_groups)

    def test_loss_history_length(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(len(result.loss_history), 5)

    def test_loss_history_finite(self):
        result = self.quant.quantize(self.W)
        self.assertTrue(all(np.isfinite(v) for v in result.loss_history))

    def test_quantize_with_calibration(self):
        rng = np.random.default_rng(99)
        # calibration_data shape: (n_samples, rows) so out = cal @ W is (n_samples, cols)
        cal = rng.standard_normal((4, 8)).astype(np.float32)  # W is (8, 16), rows=8
        result = self.quant.quantize(self.W, calibration_data=cal)
        self.assertIsInstance(result, AutoRoundResult)

    def test_bits_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.bits, 3)

    def test_group_size_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.group_size, 8)


class TestAutoRoundDequantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77)
        self.W = rng.standard_normal((4, 8)).astype(np.float32)
        self.quant = AutoRoundQuantizer(
            AutoRoundConfig(bits=4, group_size=4, n_steps=3, seed=0)
        )
        self.result = self.quant.quantize(self.W)

    def test_dequantize_shape(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.shape, (4, 8))

    def test_dequantize_dtype(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.dtype, np.float32)

    def test_reconstruction_reasonable(self):
        W_hat = self.quant.dequantize(self.result)
        err = float(np.mean(np.abs(W_hat - self.W)))
        self.assertLess(err, 2.0)


class TestAutoRoundForward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(55)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = AutoRoundQuantizer(
            AutoRoundConfig(bits=3, group_size=8, n_steps=2, seed=0)
        )
        self.result = self.quant.quantize(self.W)

    def test_forward_shape_2d(self):
        x = np.random.randn(3, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (3, 16))

    def test_forward_shape_1d(self):
        x = np.random.randn(8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (16,))

    def test_forward_batched(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (2, 4, 16))


# ---------------------------------------------------------------------------
# OWQ tests
# ---------------------------------------------------------------------------


class TestOWQConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = OWQConfig()
        self.assertEqual(cfg.bits, 3)
        self.assertEqual(cfg.promoted_bits, 4)
        self.assertEqual(cfg.group_size, 128)
        self.assertAlmostEqual(cfg.promotion_fraction, 0.05)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = OWQConfig(bits=2, promoted_bits=4, group_size=64, promotion_fraction=0.1)
        self.assertEqual(cfg.bits, 2)
        self.assertEqual(cfg.promoted_bits, 4)

    def test_invalid_bits_zero(self):
        with self.assertRaises(ValueError):
            OWQConfig(bits=0)

    def test_promoted_bits_not_greater(self):
        with self.assertRaises(ValueError):
            OWQConfig(bits=4, promoted_bits=4)

    def test_promoted_bits_less_than_bits(self):
        with self.assertRaises(ValueError):
            OWQConfig(bits=4, promoted_bits=3)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            OWQConfig(group_size=0)

    def test_promotion_fraction_zero(self):
        with self.assertRaises(ValueError):
            OWQConfig(promotion_fraction=0.0)

    def test_promotion_fraction_one(self):
        with self.assertRaises(ValueError):
            OWQConfig(promotion_fraction=1.0)

    def test_promotion_fraction_gt_one(self):
        with self.assertRaises(ValueError):
            OWQConfig(promotion_fraction=1.5)


class TestOWQQuantizerConfig(unittest.TestCase):
    def test_default_config(self):
        q = OWQQuantizer()
        self.assertIsInstance(q.config, OWQConfig)

    def test_custom_config_attached(self):
        cfg = OWQConfig(bits=2, promoted_bits=3, group_size=32)
        q = OWQQuantizer(cfg)
        self.assertEqual(q.config.group_size, 32)


class TestOWQActivationVariance(unittest.TestCase):
    def test_returns_correct_shape(self):
        q = OWQQuantizer()
        rng = np.random.default_rng(1)
        acts = rng.standard_normal((100, 32)).astype(np.float32)
        var = q.compute_activation_variance(acts)
        self.assertEqual(var.shape, (32,))

    def test_variance_non_negative(self):
        q = OWQQuantizer()
        rng = np.random.default_rng(2)
        acts = rng.standard_normal((50, 16)).astype(np.float32)
        var = q.compute_activation_variance(acts)
        self.assertTrue(np.all(var >= 0))

    def test_constant_column_zero_variance(self):
        q = OWQQuantizer()
        acts = np.ones((10, 4), dtype=np.float32)
        var = q.compute_activation_variance(acts)
        np.testing.assert_allclose(var, 0.0, atol=1e-6)


class TestOWQQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.cfg = OWQConfig(bits=3, promoted_bits=4, group_size=8, promotion_fraction=0.1)
        self.quant = OWQQuantizer(self.cfg)

    def test_returns_owq_result(self):
        result = self.quant.quantize(self.W)
        self.assertIsInstance(result, OWQResult)

    def test_result_shape(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.shape, (8, 16))

    def test_quantized_dtype(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.quantized.dtype, np.int8)

    def test_promoted_cols_count(self):
        result = self.quant.quantize(self.W)
        n_expected = max(1, int(np.ceil(16 * 0.1)))
        self.assertEqual(result.promoted_cols.shape[0], n_expected)

    def test_promoted_values_shape(self):
        result = self.quant.quantize(self.W)
        n_promoted = result.promoted_cols.shape[0]
        self.assertEqual(result.promoted_values.shape, (8, n_promoted))

    def test_quantize_with_activation_stats(self):
        rng = np.random.default_rng(3)
        acts = rng.standard_normal((50, 16)).astype(np.float32)
        stats = self.quant.compute_activation_variance(acts)
        result = self.quant.quantize(self.W, activation_stats=stats)
        self.assertIsInstance(result, OWQResult)

    def test_activation_stats_wrong_size_raises(self):
        with self.assertRaises(ValueError):
            self.quant.quantize(self.W, activation_stats=np.ones(10, dtype=np.float32))

    def test_bits_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.bits, 3)

    def test_group_size_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.group_size, 8)


class TestOWQDequantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.W = rng.standard_normal((6, 12)).astype(np.float32)
        self.quant = OWQQuantizer(
            OWQConfig(bits=3, promoted_bits=4, group_size=6, promotion_fraction=0.1)
        )
        self.result = self.quant.quantize(self.W)

    def test_dequantize_shape(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.shape, (6, 12))

    def test_dequantize_dtype(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.dtype, np.float32)

    def test_promoted_cols_exact(self):
        """Promoted columns should be reconstructed exactly."""
        W_hat = self.quant.dequantize(self.result)
        for ci in self.result.promoted_cols:
            np.testing.assert_allclose(
                W_hat[:, ci], self.W[:, ci], atol=1e-5
            )


class TestOWQForward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = OWQQuantizer(
            OWQConfig(bits=3, promoted_bits=4, group_size=8, promotion_fraction=0.1)
        )
        self.result = self.quant.quantize(self.W)

    def test_forward_2d(self):
        x = np.random.randn(3, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (3, 16))

    def test_forward_1d(self):
        x = np.random.randn(8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (16,))

    def test_forward_batched(self):
        x = np.random.randn(2, 5, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
