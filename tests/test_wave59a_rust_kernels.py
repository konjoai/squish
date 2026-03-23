"""tests/test_wave59a_rust_kernels.py — Wave 59a Rust kernel tests.

Tests for:
  - RustGPTQColumnSolve  (GPTQConfig, solve)
  - RustQuaRotGroup      (QuaRotGroupConfig, quantize, dequantize)
  - RustCalibScale       (CalibScaleConfig, compute_scales)
  - RustFlashDecodeKernel (FlashDecodeConfig, compute_split)
  - RustBF16Cast         (BF16CastConfig, to_float32, to_bf16)
  - RustSparseActGEMV    (SparseActGEMVConfig, gemv)

All tests use NumPy fallback path (squish_quant_rs not compiled in CI).
79 tests total.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from squish.kernels.rs_gptq_solve import GPTQConfig, RustGPTQColumnSolve
from squish.kernels.rs_quarot_group import QuaRotGroupConfig, RustQuaRotGroup
from squish.kernels.rs_calib_scale import CalibScaleConfig, RustCalibScale
from squish.kernels.rs_flash_decode import FlashDecodeConfig, RustFlashDecodeKernel
from squish.kernels.rs_bf16_cast import BF16CastConfig, RustBF16Cast
from squish.kernels.rs_sparse_act_gemv import SparseActGEMVConfig, RustSparseActGEMV


# ── GPTQConfig ─────────────────────────────────────────────────────────────


class TestGPTQConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GPTQConfig()
        self.assertAlmostEqual(cfg.q_max, 7.0)
        self.assertEqual(cfg.block_size, 128)

    def test_custom(self):
        cfg = GPTQConfig(q_max=15.0, block_size=64)
        self.assertAlmostEqual(cfg.q_max, 15.0)
        self.assertEqual(cfg.block_size, 64)


# ── RustGPTQColumnSolve ────────────────────────────────────────────────────


class TestRustGPTQColumnSolve(unittest.TestCase):
    def setUp(self):
        self.solver = RustGPTQColumnSolve(GPTQConfig(q_max=7.0, block_size=4))
        self.rng = np.random.default_rng(0)

    def test_backend_is_string(self):
        self.assertIn(self.solver.backend(), ("rust", "numpy"))

    def test_solve_codes_shape(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        self.assertEqual(codes.shape, (4, 8))
        self.assertEqual(codes.dtype, np.int32)

    def test_solve_scales_shape(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        self.assertEqual(scales.shape, (8,))
        self.assertEqual(scales.dtype, np.float32)

    def test_solve_codes_in_range(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        self.assertTrue((np.abs(codes) <= 7).all())

    def test_solve_scales_positive(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        self.assertTrue((scales > 0).all())

    def test_dequantize_near_original(self):
        W = np.linspace(-1.0, 1.0, 32).reshape(4, 8).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        recon = codes.astype(np.float32) * scales[np.newaxis, :]
        mse = float(np.mean((W - recon) ** 2))
        self.assertLess(mse, 0.1)

    def test_zero_weight_scales_to_one(self):
        W = np.zeros((4, 8), dtype=np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        np.testing.assert_array_equal(codes, 0)
        np.testing.assert_allclose(scales, 1.0, atol=1e-5)

    def test_q_max_override(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        h = np.ones(8, dtype=np.float32)
        codes, _ = self.solver.solve(W, h, q_max=3.0)
        self.assertTrue((np.abs(codes) <= 3).all())

    def test_block_size_override(self):
        W = self.rng.standard_normal((4, 16)).astype(np.float32)
        h = np.ones(16, dtype=np.float32)
        codes, scales = self.solver.solve(W, h, block_size=8)
        self.assertEqual(codes.shape, (4, 16))

    def test_q_max_property(self):
        self.assertAlmostEqual(self.solver.q_max(), 7.0)

    def test_block_size_property(self):
        self.assertEqual(self.solver.block_size(), 4)

    def test_single_column(self):
        W = self.rng.standard_normal((8, 1)).astype(np.float32)
        h = np.ones(1, dtype=np.float32)
        codes, scales = self.solver.solve(W, h)
        self.assertEqual(codes.shape, (8, 1))


# ── QuaRotGroupConfig ──────────────────────────────────────────────────────


class TestQuaRotGroupConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = QuaRotGroupConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertAlmostEqual(cfg.q_max, 7.0)
        self.assertTrue(cfg.symmetric)

    def test_custom(self):
        cfg = QuaRotGroupConfig(group_size=64, q_max=15.0, symmetric=False)
        self.assertFalse(cfg.symmetric)


# ── RustQuaRotGroup ────────────────────────────────────────────────────────


class TestRustQuaRotGroup(unittest.TestCase):
    def setUp(self):
        self.qrg = RustQuaRotGroup(QuaRotGroupConfig(group_size=4, q_max=7.0))
        self.rng = np.random.default_rng(1)

    def test_backend_is_string(self):
        self.assertIn(self.qrg.backend(), ("rust", "numpy"))

    def test_quantize_codes_shape(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        self.assertEqual(codes.shape, (4, 8))

    def test_quantize_codes_in_range_sym(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        self.assertTrue((np.abs(codes) <= 7).all())

    def test_quantize_scales_positive(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        self.assertTrue((scales > 0).all())

    def test_quantize_zeros_sym(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        np.testing.assert_allclose(zeros, 0.0, atol=1e-5)

    def test_quantize_asym_codes_non_negative(self):
        qrg_asym = RustQuaRotGroup(QuaRotGroupConfig(group_size=4, q_max=7.0, symmetric=False))
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = qrg_asym.quantize(W)
        self.assertTrue((codes >= 0).all())

    def test_dequantize_shape(self):
        W = self.rng.standard_normal((4, 8)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        recon = self.qrg.dequantize(codes, scales, zeros)
        self.assertEqual(recon.shape, (4, 8))
        self.assertEqual(recon.dtype, np.float32)

    def test_round_trip_symmetric(self):
        W = np.linspace(-1.0, 1.0, 32).reshape(4, 8).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        recon = self.qrg.dequantize(codes, scales, zeros)
        mse = float(np.mean((W - recon) ** 2))
        self.assertLess(mse, 0.05)

    def test_round_trip_asymmetric(self):
        qrg_asym = RustQuaRotGroup(QuaRotGroupConfig(group_size=4, q_max=7.0, symmetric=False))
        W = self.rng.uniform(0.1, 2.0, (4, 8)).astype(np.float32)
        codes, scales, zeros = qrg_asym.quantize(W)
        recon = qrg_asym.dequantize(codes, scales, zeros)
        mse = float(np.mean((W - recon) ** 2))
        self.assertLess(mse, 0.1)

    def test_group_size_property(self):
        self.assertEqual(self.qrg.group_size(), 4)

    def test_q_max_property(self):
        self.assertAlmostEqual(self.qrg.q_max(), 7.0)

    def test_n_groups_correct(self):
        W = self.rng.standard_normal((4, 12)).astype(np.float32)
        codes, scales, zeros = self.qrg.quantize(W)
        # group_size=4, cols=12 → 3 groups
        self.assertEqual(scales.shape[0], 3)


# ── CalibScaleConfig ───────────────────────────────────────────────────────


class TestCalibScaleConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CalibScaleConfig()
        self.assertEqual(cfg.method, "absmax")
        self.assertAlmostEqual(cfg.percentile, 99.9)
        self.assertEqual(cfg.n_levels, 256)

    def test_custom(self):
        cfg = CalibScaleConfig(method="aciq", n_levels=16)
        self.assertEqual(cfg.method, "aciq")


# ── RustCalibScale ─────────────────────────────────────────────────────────


class TestRustCalibScale(unittest.TestCase):
    def setUp(self):
        self.calib = RustCalibScale(CalibScaleConfig(method="absmax"))
        self.rng = np.random.default_rng(2)
        self.acts = self.rng.standard_normal((32, 16)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.calib.backend(), ("rust", "numpy"))

    def test_absmax_shape(self):
        scales = self.calib.compute_scales(self.acts)
        self.assertEqual(scales.shape, (16,))
        self.assertEqual(scales.dtype, np.float32)

    def test_absmax_values_correct(self):
        scales = self.calib.compute_scales(self.acts, method="absmax")
        expected = np.abs(self.acts).max(axis=0).astype(np.float32)
        np.testing.assert_allclose(scales, expected, atol=1e-5)

    def test_percentile_shape(self):
        calib = RustCalibScale(CalibScaleConfig(method="percentile"))
        scales = calib.compute_scales(self.acts, method="percentile", percentile=99.0)
        self.assertEqual(scales.shape, (16,))

    def test_percentile_leq_absmax(self):
        calib = RustCalibScale(CalibScaleConfig())
        absmax = calib.compute_scales(self.acts, method="absmax")
        pct = calib.compute_scales(self.acts, method="percentile", percentile=90.0)
        self.assertTrue((pct <= absmax + 1e-4).all())

    def test_aciq_shape(self):
        calib = RustCalibScale(CalibScaleConfig(method="aciq"))
        scales = calib.compute_scales(self.acts, method="aciq", n_levels=256)
        self.assertEqual(scales.shape, (16,))

    def test_aciq_positive(self):
        calib = RustCalibScale(CalibScaleConfig(method="aciq"))
        scales = calib.compute_scales(self.acts, method="aciq")
        self.assertTrue((scales > 0).all())

    def test_method_override(self):
        scales_abs = self.calib.compute_scales(self.acts, method="absmax")
        scales_pct = self.calib.compute_scales(self.acts, method="percentile", percentile=100.0)
        # percentile at 100% == absmax
        np.testing.assert_allclose(scales_abs, scales_pct, atol=1e-4)

    def test_method_property(self):
        self.assertEqual(self.calib.method(), "absmax")

    def test_single_sample(self):
        acts = self.rng.standard_normal((1, 8)).astype(np.float32)
        scales = self.calib.compute_scales(acts)
        self.assertEqual(scales.shape, (8,))


# ── FlashDecodeConfig ──────────────────────────────────────────────────────


class TestFlashDecodeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FlashDecodeConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.gqa_group, 4)

    def test_custom(self):
        cfg = FlashDecodeConfig(n_heads=8, head_dim=64)
        self.assertEqual(cfg.head_dim, 64)


# ── RustFlashDecodeKernel ──────────────────────────────────────────────────


class TestRustFlashDecodeKernel(unittest.TestCase):
    def setUp(self):
        self.fdk = RustFlashDecodeKernel(
            FlashDecodeConfig(n_heads=4, head_dim=8, gqa_group=2)
        )
        self.rng = np.random.default_rng(3)

    def _make_qkv(self, n_heads=4, n_kv_heads=2, split_len=6, head_dim=8):
        q = self.rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        k = self.rng.standard_normal((n_kv_heads, split_len, head_dim)).astype(np.float32)
        v = self.rng.standard_normal((n_kv_heads, split_len, head_dim)).astype(np.float32)
        return q, k, v

    def test_backend_is_string(self):
        self.assertIn(self.fdk.backend(), ("rust", "numpy"))

    def test_output_shape(self):
        q, k, v = self._make_qkv()
        out, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(out.dtype, np.float32)

    def test_lse_shape(self):
        q, k, v = self._make_qkv()
        _, lse, _ = self.fdk.compute_split(q, k, v)
        self.assertEqual(lse.shape, (4,))

    def test_max_score_shape(self):
        q, k, v = self._make_qkv()
        _, _, ms = self.fdk.compute_split(q, k, v)
        self.assertEqual(ms.shape, (4,))

    def test_output_no_nan(self):
        q, k, v = self._make_qkv()
        out, _, _ = self.fdk.compute_split(q, k, v)
        self.assertFalse(np.isnan(out).any())

    def test_output_finite(self):
        q, k, v = self._make_qkv()
        out, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(lse).all())
        self.assertTrue(np.isfinite(ms).all())

    def test_single_kv_token(self):
        q = self.rng.standard_normal((4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 1, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 1, 8)).astype(np.float32)
        out, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertEqual(out.shape, (4, 8))

    def test_gqa_group_override(self):
        q = self.rng.standard_normal((8, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out, _, _ = self.fdk.compute_split(q, k, v, gqa_group=4)
        self.assertEqual(out.shape, (8, 8))

    def test_n_heads_property(self):
        self.assertEqual(self.fdk.n_heads(), 4)

    def test_lse_geq_max_score(self):
        q, k, v = self._make_qkv()
        _, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertTrue((lse >= ms - 1e-4).all())


# ── BF16CastConfig ─────────────────────────────────────────────────────────


class TestBF16CastConfig(unittest.TestCase):
    def test_instantiate(self):
        cfg = BF16CastConfig()
        self.assertIsInstance(cfg, BF16CastConfig)


# ── RustBF16Cast ───────────────────────────────────────────────────────────


class TestRustBF16Cast(unittest.TestCase):
    def setUp(self):
        self.cast = RustBF16Cast()

    def test_backend_is_string(self):
        self.assertIn(self.cast.backend(), ("rust", "numpy"))

    def test_to_float32_shape(self):
        bits = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
        result = self.cast.to_float32(bits)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.float32)

    def test_to_float32_known_values(self):
        # BF16 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040
        bits = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
        result = self.cast.to_float32(bits)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-4)

    def test_to_bf16_shape(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.cast.to_bf16(values)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.uint16)

    def test_round_trip(self):
        # F32 → BF16 → F32: round-trip should be close
        values = np.array([1.0, -2.5, 0.125, 100.0], dtype=np.float32)
        bf16 = self.cast.to_bf16(values)
        back = self.cast.to_float32(bf16)
        np.testing.assert_allclose(back, values, rtol=1e-2)

    def test_zero_roundtrip(self):
        values = np.zeros(8, dtype=np.float32)
        bf16 = self.cast.to_bf16(values)
        back = self.cast.to_float32(bf16)
        np.testing.assert_allclose(back, values, atol=1e-6)

    def test_preserves_shape_2d(self):
        vals = np.ones((4, 4), dtype=np.float32)
        bf16 = self.cast.to_bf16(vals)
        back = self.cast.to_float32(bf16)
        self.assertEqual(back.shape, (4, 4))

    def test_negative_values(self):
        values = np.array([-1.0, -2.0, -4.0], dtype=np.float32)
        bf16 = self.cast.to_bf16(values)
        back = self.cast.to_float32(bf16)
        self.assertTrue((back < 0).all())

    def test_bf16_zero_bits(self):
        # 0x0000 = +0.0 in bf16
        bits = np.array([0x0000], dtype=np.uint16)
        result = self.cast.to_float32(bits)
        self.assertAlmostEqual(float(result[0]), 0.0, places=6)


# ── SparseActGEMVConfig ────────────────────────────────────────────────────


class TestSparseActGEMVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SparseActGEMVConfig()
        self.assertAlmostEqual(cfg.threshold, 0.0)

    def test_custom(self):
        cfg = SparseActGEMVConfig(threshold=0.01)
        self.assertAlmostEqual(cfg.threshold, 0.01)


# ── RustSparseActGEMV ──────────────────────────────────────────────────────


class TestRustSparseActGEMV(unittest.TestCase):
    def setUp(self):
        self.gemv = RustSparseActGEMV(SparseActGEMVConfig(threshold=0.0))
        self.rng = np.random.default_rng(4)

    def test_backend_is_string(self):
        self.assertIn(self.gemv.backend(), ("rust", "numpy"))

    def test_output_shape(self):
        W = self.rng.standard_normal((8, 4)).astype(np.float32)
        a = self.rng.standard_normal(4).astype(np.float32)
        out = self.gemv.gemv(W, a)
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, np.float32)

    def test_matches_dense_matmul(self):
        W = self.rng.standard_normal((8, 4)).astype(np.float32)
        a = self.rng.standard_normal(4).astype(np.float32)
        out_sparse = self.gemv.gemv(W, a)
        out_dense = W @ a
        np.testing.assert_allclose(out_sparse, out_dense, atol=1e-5)

    def test_zero_activation_gives_zero_output(self):
        W = self.rng.standard_normal((8, 4)).astype(np.float32)
        a = np.zeros(4, dtype=np.float32)
        out = self.gemv.gemv(W, a)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_threshold_skips_small_values(self):
        # With threshold > max_act, output should be all zeros
        W = np.ones((4, 4), dtype=np.float32)
        a = np.full(4, 0.001, dtype=np.float32)
        out = self.gemv.gemv(W, a, threshold=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_threshold_override(self):
        W = self.rng.standard_normal((6, 4)).astype(np.float32)
        a = self.rng.standard_normal(4).astype(np.float32)
        out = self.gemv.gemv(W, a, threshold=0.5)
        self.assertEqual(out.shape, (6,))

    def test_sparsity_all_zeros_is_one(self):
        a = np.zeros(4, dtype=np.float32)
        sp = self.gemv.sparsity(a)
        self.assertAlmostEqual(sp, 1.0, places=5)

    def test_sparsity_all_nonzero_is_zero(self):
        a = np.ones(4, dtype=np.float32)
        sp = self.gemv.sparsity(a)
        self.assertAlmostEqual(sp, 0.0, places=5)

    def test_threshold_property(self):
        self.assertAlmostEqual(self.gemv.threshold(), 0.0)

    def test_shape_error_on_mismatch(self):
        W = self.rng.standard_normal((8, 4)).astype(np.float32)
        a = self.rng.standard_normal(5).astype(np.float32)
        with self.assertRaises(ValueError):
            self.gemv.gemv(W, a)


if __name__ == "__main__":
    unittest.main()
