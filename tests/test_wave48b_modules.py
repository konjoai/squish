"""Tests for Wave 48b modules: BitDistillerQuant, ZipLMMixedPrecision, GGUFMixedQuantizer."""

from __future__ import annotations

import unittest

import numpy as np

from squish.quant.bit_distiller import (
    BitDistillerConfig,
    BitDistillerQuant,
    BitDistillerResult,
)
from squish.quant.zip_lm import ZipLMConfig, ZipLMMixedPrecision, ZipLMResult
from squish.quant.gguf_mixed import GGUFConfig, GGUFMixedQuantizer, GGUFTensor


# ---------------------------------------------------------------------------
# BitDistiller tests
# ---------------------------------------------------------------------------


class TestBitDistillerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = BitDistillerConfig()
        self.assertEqual(cfg.bits, 2)
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.n_steps, 512)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = BitDistillerConfig(bits=3, group_size=64, n_steps=100, temperature=2.0)
        self.assertEqual(cfg.bits, 3)
        self.assertAlmostEqual(cfg.temperature, 2.0)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            BitDistillerConfig(bits=0)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            BitDistillerConfig(group_size=0)

    def test_invalid_n_steps(self):
        with self.assertRaises(ValueError):
            BitDistillerConfig(n_steps=0)

    def test_invalid_temperature_zero(self):
        with self.assertRaises(ValueError):
            BitDistillerConfig(temperature=0.0)

    def test_invalid_temperature_negative(self):
        with self.assertRaises(ValueError):
            BitDistillerConfig(temperature=-1.0)


class TestBitDistillerQuantConfig(unittest.TestCase):
    def test_default_config(self):
        q = BitDistillerQuant()
        self.assertIsInstance(q.config, BitDistillerConfig)

    def test_custom_config(self):
        cfg = BitDistillerConfig(bits=3, n_steps=10)
        q = BitDistillerQuant(cfg)
        self.assertEqual(q.config.bits, 3)
        self.assertEqual(q.config.n_steps, 10)


class TestBitDistillerKLDivergence(unittest.TestCase):
    def test_identical_distributions_zero_kl(self):
        q = BitDistillerQuant()
        p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        kl = q._kl_divergence(p, p)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_different_distributions_positive(self):
        q = BitDistillerQuant()
        p = np.array([0.6, 0.4], dtype=np.float32)
        r = np.array([0.5, 0.5], dtype=np.float32)
        kl = q._kl_divergence(p, r)
        self.assertGreater(kl, 0.0)

    def test_returns_scalar(self):
        q = BitDistillerQuant()
        p = np.array([0.3, 0.7], dtype=np.float32)
        r = np.array([0.4, 0.6], dtype=np.float32)
        kl = q._kl_divergence(p, r)
        self.assertIsInstance(kl, float)


class TestBitDistillerQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(44)
        self.W = rng.standard_normal((4, 8)).astype(np.float32)
        self.quant = BitDistillerQuant(
            BitDistillerConfig(bits=2, group_size=4, n_steps=3, temperature=1.0, seed=0)
        )

    def test_returns_result(self):
        result = self.quant.quantize(self.W)
        self.assertIsInstance(result, BitDistillerResult)

    def test_result_shape(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.shape, (4, 8))

    def test_quantized_dtype(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.quantized.dtype, np.int8)

    def test_kl_loss_history_length(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(len(result.kl_loss_history), 3)

    def test_kl_loss_finite(self):
        result = self.quant.quantize(self.W)
        self.assertTrue(all(np.isfinite(v) for v in result.kl_loss_history))

    def test_kl_loss_first_non_negative(self):
        result = self.quant.quantize(self.W)
        self.assertGreaterEqual(result.kl_loss_history[0], 0.0)

    def test_quantize_with_teacher(self):
        rng = np.random.default_rng(55)
        teacher_W = rng.standard_normal((4, 8)).astype(np.float32)
        result = self.quant.quantize(self.W, teacher_W=teacher_W)
        self.assertIsInstance(result, BitDistillerResult)

    def test_bits_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.bits, 2)

    def test_group_size_stored(self):
        result = self.quant.quantize(self.W)
        self.assertEqual(result.group_size, 4)


class TestBitDistillerDequantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        self.W = rng.standard_normal((4, 8)).astype(np.float32)
        self.quant = BitDistillerQuant(
            BitDistillerConfig(bits=2, group_size=4, n_steps=2, seed=0)
        )
        self.result = self.quant.quantize(self.W)

    def test_dequantize_shape(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.shape, (4, 8))

    def test_dequantize_dtype(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertEqual(W_hat.dtype, np.float32)

    def test_dequantize_finite(self):
        W_hat = self.quant.dequantize(self.result)
        self.assertTrue(np.all(np.isfinite(W_hat)))


class TestBitDistillerForward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(66)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = BitDistillerQuant(
            BitDistillerConfig(bits=2, group_size=8, n_steps=2, seed=0)
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
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = self.quant.forward(x, self.result)
        self.assertEqual(y.shape, (2, 4, 16))


# ---------------------------------------------------------------------------
# ZipLM tests
# ---------------------------------------------------------------------------


class TestZipLMConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ZipLMConfig()
        self.assertAlmostEqual(cfg.memory_budget_gb, 7.0)
        self.assertEqual(cfg.min_bits, 2)
        self.assertEqual(cfg.max_bits, 4)
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.seed, 0)

    def test_custom_valid(self):
        cfg = ZipLMConfig(memory_budget_gb=8.0, min_bits=2, max_bits=4)
        self.assertAlmostEqual(cfg.memory_budget_gb, 8.0)

    def test_invalid_budget_zero(self):
        with self.assertRaises(ValueError):
            ZipLMConfig(memory_budget_gb=0.0)

    def test_invalid_budget_negative(self):
        with self.assertRaises(ValueError):
            ZipLMConfig(memory_budget_gb=-1.0)

    def test_invalid_min_bits(self):
        with self.assertRaises(ValueError):
            ZipLMConfig(min_bits=0)

    def test_max_bits_less_than_min_bits(self):
        with self.assertRaises(ValueError):
            ZipLMConfig(min_bits=4, max_bits=3)

    def test_min_equals_max_bits_valid(self):
        cfg = ZipLMConfig(min_bits=3, max_bits=3)
        self.assertEqual(cfg.min_bits, 3)
        self.assertEqual(cfg.max_bits, 3)


class TestZipLMMixedPrecisionConfig(unittest.TestCase):
    def test_default_config(self):
        planner = ZipLMMixedPrecision()
        self.assertIsInstance(planner.config, ZipLMConfig)

    def test_custom_config(self):
        cfg = ZipLMConfig(memory_budget_gb=4.0, min_bits=2, max_bits=3)
        planner = ZipLMMixedPrecision(cfg)
        self.assertAlmostEqual(planner.config.memory_budget_gb, 4.0)


class TestZipLMEstimateMemory(unittest.TestCase):
    def test_single_layer(self):
        planner = ZipLMMixedPrecision()
        shapes = [(1000, 1000)]
        bits = [4]
        expected = 1000 * 1000 * 4 / 8 / 1e9
        result = planner.estimate_memory_gb(shapes, bits)
        self.assertAlmostEqual(result, expected, places=10)

    def test_multiple_layers(self):
        planner = ZipLMMixedPrecision()
        shapes = [(100, 200), (300, 400)]
        bits = [2, 4]
        expected = (100 * 200 * 2 + 300 * 400 * 4) / 8 / 1e9
        result = planner.estimate_memory_gb(shapes, bits)
        self.assertAlmostEqual(result, expected, places=10)

    def test_returns_float(self):
        planner = ZipLMMixedPrecision()
        result = planner.estimate_memory_gb([(10, 10)], [3])
        self.assertIsInstance(result, float)


class TestZipLMAssignBits(unittest.TestCase):
    def setUp(self):
        self.cfg = ZipLMConfig(memory_budget_gb=1000.0, min_bits=2, max_bits=4, seed=0)
        self.planner = ZipLMMixedPrecision(self.cfg)

    def test_returns_correct_length(self):
        shapes = [(100, 200)] * 5
        sens = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        bits = self.planner.assign_bits(5, shapes, sens)
        self.assertEqual(len(bits), 5)

    def test_bits_within_range(self):
        shapes = [(100, 100)] * 4
        sens = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        bits = self.planner.assign_bits(4, shapes, sens)
        for b in bits:
            self.assertGreaterEqual(b, self.cfg.min_bits)
            self.assertLessEqual(b, self.cfg.max_bits)

    def test_tight_budget_stays_at_min(self):
        # Budget so tight that no layer can be promoted
        cfg = ZipLMConfig(memory_budget_gb=1e-20, min_bits=2, max_bits=4, seed=0)
        planner = ZipLMMixedPrecision(cfg)
        shapes = [(100, 100)] * 3
        sens = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        bits = planner.assign_bits(3, shapes, sens)
        self.assertTrue(all(b == 2 for b in bits))


class TestZipLMPlan(unittest.TestCase):
    def setUp(self):
        self.cfg = ZipLMConfig(memory_budget_gb=100.0, min_bits=2, max_bits=4, seed=42)
        self.planner = ZipLMMixedPrecision(self.cfg)
        self.shapes = [(64, 128)] * 6

    def test_returns_ziplm_result(self):
        result = self.planner.plan(self.shapes)
        self.assertIsInstance(result, ZipLMResult)

    def test_bit_schedule_length(self):
        result = self.planner.plan(self.shapes)
        self.assertEqual(len(result.bit_schedule), 6)

    def test_plan_with_sensitivities(self):
        sens = [0.1 * i for i in range(1, 7)]
        result = self.planner.plan(self.shapes, layer_sensitivities=sens)
        self.assertEqual(len(result.bit_schedule), 6)

    def test_total_bits_est_positive(self):
        result = self.planner.plan(self.shapes)
        self.assertGreater(result.total_bits_est, 0.0)

    def test_layers_by_bits_covers_all(self):
        result = self.planner.plan(self.shapes)
        all_layers = []
        for indices in result.layers_by_bits.values():
            all_layers.extend(indices)
        self.assertEqual(sorted(all_layers), list(range(6)))

    def test_effective_bits_in_range(self):
        result = self.planner.plan(self.shapes)
        self.assertGreaterEqual(result.effective_bits, self.cfg.min_bits)
        self.assertLessEqual(result.effective_bits, self.cfg.max_bits)

    def test_effective_bits_empty_schedule(self):
        result = ZipLMResult(bit_schedule=[], total_bits_est=0.0, layers_by_bits={})
        self.assertAlmostEqual(result.effective_bits, 0.0)


# ---------------------------------------------------------------------------
# GGUF Mixed Quantizer tests
# ---------------------------------------------------------------------------


class TestGGUFConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GGUFConfig()
        self.assertEqual(cfg.quant_type, "Q3_K")
        self.assertEqual(cfg.group_size, 32)
        self.assertEqual(cfg.seed, 0)

    def test_all_valid_quant_types(self):
        for qt in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q8_0"):
            cfg = GGUFConfig(quant_type=qt)
            self.assertEqual(cfg.quant_type, qt)

    def test_invalid_quant_type(self):
        with self.assertRaises(ValueError):
            GGUFConfig(quant_type="Q6_K")

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            GGUFConfig(group_size=0)


class TestGGUFTensor(unittest.TestCase):
    def _make_tensor(self, quant_type="Q3_K", rows=4, cols=8, group_size=4):
        padded_cols = cols + ((-cols) % group_size)
        n_blocks = rows * (padded_cols // group_size)
        n_super = max(1, (n_blocks + 7) // 8)
        return GGUFTensor(
            quant_type=quant_type,
            quantized=np.zeros((rows, padded_cols), dtype=np.int8),
            scales=np.ones(n_blocks, dtype=np.float32),
            mins=np.zeros(n_blocks, dtype=np.float32),
            super_scales=np.ones(n_super, dtype=np.float32),
            shape=(rows, cols),
        )

    def test_quant_bits_q2k(self):
        t = self._make_tensor("Q2_K")
        self.assertEqual(t.quant_bits, 2)

    def test_quant_bits_q3k(self):
        t = self._make_tensor("Q3_K")
        self.assertEqual(t.quant_bits, 3)

    def test_quant_bits_q4k(self):
        t = self._make_tensor("Q4_K")
        self.assertEqual(t.quant_bits, 4)

    def test_quant_bits_q5k(self):
        t = self._make_tensor("Q5_K")
        self.assertEqual(t.quant_bits, 5)

    def test_quant_bits_q80(self):
        t = self._make_tensor("Q8_0")
        self.assertEqual(t.quant_bits, 8)


class TestGGUFMixedQuantizerConfig(unittest.TestCase):
    def test_default_config(self):
        q = GGUFMixedQuantizer()
        self.assertIsInstance(q.config, GGUFConfig)

    def test_custom_config(self):
        cfg = GGUFConfig(quant_type="Q4_K", group_size=16)
        q = GGUFMixedQuantizer(cfg)
        self.assertEqual(q.config.quant_type, "Q4_K")

    def test_quant_bits_helper(self):
        q = GGUFMixedQuantizer()
        self.assertEqual(q._quant_bits_for_type("Q2_K"), 2)
        self.assertEqual(q._quant_bits_for_type("Q8_0"), 8)


class TestGGUFQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)

    def _make_quant(self, quant_type="Q3_K", group_size=4):
        return GGUFMixedQuantizer(GGUFConfig(quant_type=quant_type, group_size=group_size))

    def test_returns_tensor(self):
        q = self._make_quant()
        tensor = q.quantize(self.W)
        self.assertIsInstance(tensor, GGUFTensor)

    def test_tensor_shape(self):
        q = self._make_quant()
        tensor = q.quantize(self.W)
        self.assertEqual(tensor.shape, (8, 16))

    def test_quantized_dtype(self):
        q = self._make_quant()
        tensor = q.quantize(self.W)
        self.assertEqual(tensor.quantized.dtype, np.int8)

    def test_scales_float32(self):
        q = self._make_quant()
        tensor = q.quantize(self.W)
        self.assertEqual(tensor.scales.dtype, np.float32)

    def test_super_scales_present(self):
        q = self._make_quant()
        tensor = q.quantize(self.W)
        self.assertGreater(len(tensor.super_scales), 0)

    def test_quant_type_stored(self):
        q = self._make_quant("Q4_K")
        tensor = q.quantize(self.W)
        self.assertEqual(tensor.quant_type, "Q4_K")

    def test_all_quant_types(self):
        for qt in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q8_0"):
            q = self._make_quant(quant_type=qt)
            tensor = q.quantize(self.W)
            self.assertEqual(tensor.quant_type, qt)


class TestGGUFDequantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(88)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = GGUFMixedQuantizer(GGUFConfig(quant_type="Q3_K", group_size=4))
        self.tensor = self.quant.quantize(self.W)

    def test_dequantize_shape(self):
        W_hat = self.quant.dequantize(self.tensor)
        self.assertEqual(W_hat.shape, (8, 16))

    def test_dequantize_dtype(self):
        W_hat = self.quant.dequantize(self.tensor)
        self.assertEqual(W_hat.dtype, np.float32)

    def test_reconstruction_reasonable(self):
        W_hat = self.quant.dequantize(self.tensor)
        err = float(np.mean(np.abs(W_hat - self.W)))
        self.assertLess(err, 2.0)

    def test_q8_0_very_close(self):
        quant = GGUFMixedQuantizer(GGUFConfig(quant_type="Q8_0", group_size=4))
        tensor = quant.quantize(self.W)
        W_hat = quant.dequantize(tensor)
        err = float(np.mean(np.abs(W_hat - self.W)))
        self.assertLess(err, 0.1)


class TestGGUFForward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.W = rng.standard_normal((8, 16)).astype(np.float32)
        self.quant = GGUFMixedQuantizer(GGUFConfig(quant_type="Q4_K", group_size=4))
        self.tensor = self.quant.quantize(self.W)

    def test_forward_2d(self):
        x = np.random.randn(3, 8).astype(np.float32)
        y = self.quant.forward(x, self.tensor)
        self.assertEqual(y.shape, (3, 16))

    def test_forward_1d(self):
        x = np.random.randn(8).astype(np.float32)
        y = self.quant.forward(x, self.tensor)
        self.assertEqual(y.shape, (16,))

    def test_forward_batched(self):
        x = np.random.randn(2, 5, 8).astype(np.float32)
        y = self.quant.forward(x, self.tensor)
        self.assertEqual(y.shape, (2, 5, 16))


class TestGGUFEncodeDecodeRoundtrip(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(111)
        self.W = rng.standard_normal((6, 12)).astype(np.float32)
        self.quant = GGUFMixedQuantizer(GGUFConfig(quant_type="Q3_K", group_size=4))
        self.tensor = self.quant.quantize(self.W)

    def test_encode_returns_bytes(self):
        data = self.quant.encode_to_bytes(self.tensor)
        self.assertIsInstance(data, bytes)

    def test_encode_non_empty(self):
        data = self.quant.encode_to_bytes(self.tensor)
        self.assertGreater(len(data), 0)

    def test_roundtrip_quant_type(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        self.assertEqual(decoded.quant_type, self.tensor.quant_type)

    def test_roundtrip_shape(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        self.assertEqual(decoded.shape, self.tensor.shape)

    def test_roundtrip_scales(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        np.testing.assert_array_equal(decoded.scales, self.tensor.scales)

    def test_roundtrip_mins(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        np.testing.assert_array_equal(decoded.mins, self.tensor.mins)

    def test_roundtrip_quantized(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        np.testing.assert_array_equal(decoded.quantized, self.tensor.quantized)

    def test_dequantize_after_decode_matches_original(self):
        data = self.quant.encode_to_bytes(self.tensor)
        decoded = self.quant.decode_from_bytes(data, self.tensor.shape)
        W_hat_orig = self.quant.dequantize(self.tensor)
        W_hat_dec = self.quant.dequantize(decoded)
        np.testing.assert_array_equal(W_hat_orig, W_hat_dec)

    def test_bad_magic_raises(self):
        data = self.quant.encode_to_bytes(self.tensor)
        bad = b"BAD!" + data[4:]
        with self.assertRaises(ValueError):
            self.quant.decode_from_bytes(bad, self.tensor.shape)

    def test_shape_mismatch_raises(self):
        data = self.quant.encode_to_bytes(self.tensor)
        with self.assertRaises(ValueError):
            self.quant.decode_from_bytes(data, (99, 99))

    def test_roundtrip_all_quant_types(self):
        for qt in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q8_0"):
            quant = GGUFMixedQuantizer(GGUFConfig(quant_type=qt, group_size=4))
            tensor = quant.quantize(self.W)
            data = quant.encode_to_bytes(tensor)
            decoded = quant.decode_from_bytes(data, tensor.shape)
            self.assertEqual(decoded.quant_type, qt)


if __name__ == "__main__":
    unittest.main()
