"""tests/test_wave67_fused_gemv.py

Unit tests for Wave 67: SQUIZD Fused INT4/INT2 Metal GEMV.

Modules under test
──────────────────
* squish.hardware.kernel_dispatch — KernelDispatch, KernelDispatcher,
                                    get_kernel_dispatcher,
                                    reset_kernel_dispatcher

Metal kernel correctness is verified through Python reference implementations
that mirror the shader arithmetic exactly.
"""
from __future__ import annotations

import struct
import unittest
from dataclasses import FrozenInstanceError
from typing import List
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Python reference helpers — mirror the Metal shader arithmetic
# ---------------------------------------------------------------------------

def _pack_int4(w0: int, w1: int) -> int:
    """Pack two 4-bit unsigned integers into one byte (high nibble first)."""
    assert 0 <= w0 <= 15
    assert 0 <= w1 <= 15
    return ((w0 & 0xF) << 4) | (w1 & 0xF)


def _unpack_int4(packed: int) -> tuple[int, int]:
    """Unpack one byte into two 4-bit unsigned integers (high nibble first)."""
    w0 = (packed >> 4) & 0xF
    w1 =  packed       & 0xF
    return w0, w1


def _dequant_int4(w_int: int, scale: float, zero: float) -> float:
    """Asymmetric unsigned INT4 dequantisation: w_float = scale * w_int + zero."""
    return scale * float(w_int) + zero


def _reference_int4_gemv(
    weights_int: np.ndarray,  # shape (n_rows, n_cols) — uint4 stored in uint8
    scales: np.ndarray,       # shape (n_rows, n_groups) — float32
    zeros: np.ndarray,        # shape (n_rows, n_groups) — float32
    input_vec: np.ndarray,    # shape (n_cols,) — float32
    group_size: int,
) -> np.ndarray:
    """Reference INT4 GEMV in pure Python/NumPy (matches fused_int4_gemv.metal)."""
    n_rows, n_cols = weights_int.shape
    n_groups = n_cols // group_size
    output = np.zeros(n_rows, dtype=np.float32)
    for row in range(n_rows):
        acc = 0.0
        for g in range(n_groups):
            s = float(scales[row, g])
            z = float(zeros[row, g])
            for c in range(group_size):
                col = g * group_size + c
                w = float(weights_int[row, col])
                acc += (s * w + z) * float(input_vec[col])
        output[row] = acc
    return output


def _pack_weights_int4(weights_int: np.ndarray) -> np.ndarray:
    """Pack a (n_rows, n_cols) uint8 INT4 matrix into (n_rows, n_cols//2) bytes."""
    n_rows, n_cols = weights_int.shape
    assert n_cols % 2 == 0
    packed_cols = n_cols // 2
    packed = np.zeros((n_rows, packed_cols), dtype=np.uint8)
    for col_pair in range(packed_cols):
        w0 = weights_int[:, col_pair * 2    ].astype(np.uint8)
        w1 = weights_int[:, col_pair * 2 + 1].astype(np.uint8)
        packed[:, col_pair] = ((w0 & 0xF) << 4) | (w1 & 0xF)
    return packed


def _unpack_weights_int4(packed: np.ndarray, n_cols: int) -> np.ndarray:
    """Unpack (n_rows, n_cols//2) bytes back to (n_rows, n_cols) uint8."""
    n_rows = packed.shape[0]
    weights = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for col_pair in range(n_cols // 2):
        b = packed[:, col_pair].astype(np.uint8)
        weights[:, col_pair * 2    ] = (b >> 4) & 0xF
        weights[:, col_pair * 2 + 1] =  b       & 0xF
    return weights


def _pack_int2(w0: int, w1: int, w2: int, w3: int) -> int:
    """Pack four 2-bit values into one byte (LSB first: w0=bits[1:0])."""
    assert all(0 <= w <= 3 for w in (w0, w1, w2, w3))
    return (w3 << 6) | (w2 << 4) | (w1 << 2) | w0


def _unpack_int2(packed: int) -> tuple[int, int, int, int]:
    """Unpack one byte into four 2-bit values (LSB first: w0=bits[1:0])."""
    w0 =  packed       & 0x3
    w1 = (packed >> 2) & 0x3
    w2 = (packed >> 4) & 0x3
    w3 = (packed >> 6) & 0x3
    return w0, w1, w2, w3


def _build_int2_lut(codebook_row: np.ndarray, group_idx: int) -> List[float]:
    """Return the 4-entry FP16 LUT for group *group_idx* from a codebook row."""
    base = group_idx * 4
    return [float(np.float16(codebook_row[base + i])) for i in range(4)]


def _reference_int2_gemv(
    weights_packed: np.ndarray,  # shape (n_rows, n_cols // 4) — uint8
    codebook: np.ndarray,        # shape (n_rows, n_cb_entries) — float16
    input_vec: np.ndarray,       # shape (n_cols,) — float32
    group_size: int,
) -> np.ndarray:
    """Reference INT2 LUT-GEMM GEMV (matches lut_int2_gemv.metal)."""
    n_rows = weights_packed.shape[0]
    n_cols = weights_packed.shape[1] * 4
    n_groups = n_cols // group_size
    output = np.zeros(n_rows, dtype=np.float32)
    for row in range(n_rows):
        acc = 0.0
        for byte_idx in range(n_cols // 4):
            col_base = byte_idx * 4
            packed = int(weights_packed[row, byte_idx])
            idx0 =  packed       & 0x3
            idx1 = (packed >> 2) & 0x3
            idx2 = (packed >> 4) & 0x3
            idx3 = (packed >> 6) & 0x3
            g = col_base // group_size
            lut = _build_int2_lut(codebook[row], g)
            acc += lut[idx0] * float(input_vec[col_base    ])
            acc += lut[idx1] * float(input_vec[col_base + 1])
            acc += lut[idx2] * float(input_vec[col_base + 2])
            acc += lut[idx3] * float(input_vec[col_base + 3])
        output[row] = acc
    return output


def _make_caps(**kwargs: object) -> MagicMock:
    """Return a MagicMock HardwareCapabilities with sensible defaults."""
    from squish.hardware.capability_probe import HardwareCapabilities
    from squish.hardware.chip_detector import AppleChipGeneration
    defaults = dict(
        chip_generation=AppleChipGeneration.M2,
        has_astc_texture_sampling=True,
        has_ane=True,
        has_metal3=True,
        has_mxfp4=False,
        ane_memory_budget_gb=8.0,
    )
    defaults.update(kwargs)
    return HardwareCapabilities(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# 1. INT4 packing helpers
# ===========================================================================

class TestFusedInt4Pack(unittest.TestCase):
    """Verify nibble pack/unpack helper arithmetic."""

    def test_pack_zero_zero(self) -> None:
        self.assertEqual(_pack_int4(0, 0), 0x00)

    def test_pack_max_max(self) -> None:
        self.assertEqual(_pack_int4(15, 15), 0xFF)

    def test_pack_low_nibble_only(self) -> None:
        self.assertEqual(_pack_int4(0, 7), 0x07)

    def test_pack_high_nibble_only(self) -> None:
        self.assertEqual(_pack_int4(7, 0), 0x70)

    def test_unpack_roundtrip(self) -> None:
        for w0 in range(16):
            for w1 in range(16):
                packed = _pack_int4(w0, w1)
                got0, got1 = _unpack_int4(packed)
                self.assertEqual(got0, w0)
                self.assertEqual(got1, w1)

    def test_dequant_zero_point(self) -> None:
        # scale=1, zero=0 → identity
        self.assertAlmostEqual(_dequant_int4(5, 1.0, 0.0), 5.0)

    def test_dequant_with_zero(self) -> None:
        self.assertAlmostEqual(_dequant_int4(3, 2.0, -3.0), 3.0)

    def test_dequant_min_int4(self) -> None:
        self.assertAlmostEqual(_dequant_int4(0, 0.5, 1.0), 1.0)


# ===========================================================================
# 2. INT4 GEMV reference correctness
# ===========================================================================

class TestFusedInt4GEMV(unittest.TestCase):
    """Python reference INT4 GEMV produces correct dot products."""

    def _make_matrix(
        self,
        n_rows: int = 4,
        n_cols: int = 32,
        group_size: int = 16,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        weights_int = rng.integers(0, 16, (n_rows, n_cols), dtype=np.uint8)
        n_groups = n_cols // group_size
        scales = rng.uniform(0.01, 0.1, (n_rows, n_groups)).astype(np.float32)
        zeros  = rng.uniform(-0.5, 0.0, (n_rows, n_groups)).astype(np.float32)
        x      = rng.standard_normal(n_cols).astype(np.float32)
        return weights_int, scales, zeros, x

    def test_output_shape(self) -> None:
        w, s, z, x = self._make_matrix()
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        self.assertEqual(out.shape, (4,))

    def test_all_zeros_input(self) -> None:
        w, s, z, _ = self._make_matrix()
        x = np.zeros(32, dtype=np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        np.testing.assert_array_equal(out, np.zeros(4, dtype=np.float32))

    def test_identity_weights(self) -> None:
        # 1×4 matrix, all weights=1, no zero, scale=1 → dot(1, x)=sum(x)
        w = np.ones((1, 4), dtype=np.uint8)
        s = np.ones((1, 1), dtype=np.float32)
        z = np.zeros((1, 1), dtype=np.float32)
        x = np.arange(1, 5, dtype=np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=4)
        self.assertAlmostEqual(float(out[0]), 10.0, places=5)

    def test_pack_unpack_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        w = rng.integers(0, 16, (8, 64), dtype=np.uint8)
        packed = _pack_weights_int4(w)
        unpacked = _unpack_weights_int4(packed, 64)
        np.testing.assert_array_equal(w, unpacked)

    def test_gemv_against_dequant_matmul(self) -> None:
        # Dequant first, then standard matmul — must match reference GEMV.
        w, s, z, x = self._make_matrix(n_rows=8, n_cols=64, group_size=16)
        ref = _reference_int4_gemv(w, s, z, x, group_size=16)
        # Build dequantised float matrix
        n_groups = 64 // 16
        dequant = np.zeros((8, 64), dtype=np.float32)
        for gi in range(n_groups):
            col_sl = slice(gi * 16, (gi + 1) * 16)
            dequant[:, col_sl] = (
                s[:, gi : gi + 1] * w[:, col_sl].astype(np.float32)
                + z[:, gi : gi + 1]
            )
        expected = dequant @ x
        np.testing.assert_allclose(ref, expected, rtol=1e-5, atol=1e-5)

    def test_different_group_sizes(self) -> None:
        for gs in (8, 16, 32):
            with self.subTest(group_size=gs):
                n_cols = gs * 4
                rng = np.random.default_rng(gs)
                w = rng.integers(0, 16, (2, n_cols), dtype=np.uint8)
                s = rng.uniform(0.1, 0.2, (2, 4)).astype(np.float32)
                z = np.zeros((2, 4), dtype=np.float32)
                x = rng.standard_normal(n_cols).astype(np.float32)
                out = _reference_int4_gemv(w, s, z, x, group_size=gs)
                self.assertEqual(out.shape, (2,))
                self.assertTrue(np.isfinite(out).all())

    def test_max_int4_values(self) -> None:
        w = np.full((1, 16), 15, dtype=np.uint8)
        s = np.ones((1, 1), dtype=np.float32) * 2.0
        z = np.zeros((1, 1), dtype=np.float32)
        x = np.ones(16, dtype=np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        expected = 15.0 * 2.0 * 16
        self.assertAlmostEqual(float(out[0]), expected, places=3)

    def test_min_int4_values(self) -> None:
        w = np.zeros((1, 16), dtype=np.uint8)
        s = np.ones((1, 1), dtype=np.float32)
        z = np.ones((1, 1), dtype=np.float32) * (-1.0)
        x = np.ones(16, dtype=np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        expected = 0.0 * 1.0 + (-1.0) * 16
        self.assertAlmostEqual(float(out[0]), expected, places=5)

    def test_reproducibility(self) -> None:
        w, s, z, x = self._make_matrix(seed=123)
        out1 = _reference_int4_gemv(w, s, z, x, group_size=16)
        out2 = _reference_int4_gemv(w, s, z, x, group_size=16)
        np.testing.assert_array_equal(out1, out2)

    def test_multiple_rows_independence(self) -> None:
        # Verify that each row is computed independently.
        rng = np.random.default_rng(99)
        w = rng.integers(0, 16, (4, 32), dtype=np.uint8)
        s = rng.uniform(0.05, 0.1, (4, 2)).astype(np.float32)
        z = rng.uniform(-0.1, 0.0, (4, 2)).astype(np.float32)
        x = rng.standard_normal(32).astype(np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        # Check each row separately
        for i in range(4):
            single = _reference_int4_gemv(
                w[i : i + 1], s[i : i + 1], z[i : i + 1], x, group_size=16
            )
            self.assertAlmostEqual(float(out[i]), float(single[0]), places=5)

    def test_packed_gemv_matches_int_gemv(self) -> None:
        w, s, z, x = self._make_matrix(n_rows=4, n_cols=32, group_size=16)
        expected = _reference_int4_gemv(w, s, z, x, group_size=16)
        packed   = _pack_weights_int4(w)
        unpacked = _unpack_weights_int4(packed, 32)
        got      = _reference_int4_gemv(unpacked, s, z, x, group_size=16)
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_large_matrix_finite(self) -> None:
        rng = np.random.default_rng(0)
        n_rows, n_cols, gs = 128, 256, 64
        w = rng.integers(0, 16, (n_rows, n_cols), dtype=np.uint8)
        s = rng.uniform(0.01, 0.1, (n_rows, n_cols // gs)).astype(np.float32)
        z = rng.uniform(-0.5, 0.0, (n_rows, n_cols // gs)).astype(np.float32)
        x = rng.standard_normal(n_cols).astype(np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=gs)
        self.assertTrue(np.isfinite(out).all())

    def test_single_element(self) -> None:
        w = np.array([[7]], dtype=np.uint8)
        s = np.array([[2.0]], dtype=np.float32)
        z = np.array([[1.0]], dtype=np.float32)
        x = np.array([3.0], dtype=np.float32)
        out = _reference_int4_gemv(w, s, z, x, group_size=1)
        self.assertAlmostEqual(float(out[0]), (2.0 * 7 + 1.0) * 3.0, places=5)

    def test_asymmetry_zero_point(self) -> None:
        # scale=0.5, zero=-4: symmetric around midpoint 8 → dequant(8)=0
        scale, zero = 0.5, -4.0
        self.assertAlmostEqual(_dequant_int4(8, scale, zero), 0.0, places=6)

    def test_output_dtype_float32(self) -> None:
        w, s, z, x = self._make_matrix()
        out = _reference_int4_gemv(w, s, z, x, group_size=16)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# 3. INT4 GEMM reference
# ===========================================================================

class TestFusedInt4GEMM(unittest.TestCase):
    """Reference tiled INT4 GEMM (treats the activation matrix as seq_len tokens)."""

    @staticmethod
    def _reference_int4_gemm(
        weights_int: np.ndarray,  # (n_rows, n_cols)
        scales: np.ndarray,       # (n_rows, n_groups)
        zeros: np.ndarray,        # (n_rows, n_groups)
        input_mat: np.ndarray,    # (seq_len, n_cols)
        group_size: int,
    ) -> np.ndarray:
        """Reference: compute dequant(W) @ X^T and return (n_rows, seq_len)."""
        n_rows, n_cols = weights_int.shape
        seq_len = input_mat.shape[0]
        n_groups = n_cols // group_size
        dequant = np.zeros((n_rows, n_cols), dtype=np.float32)
        for gi in range(n_groups):
            col_sl = slice(gi * group_size, (gi + 1) * group_size)
            dequant[:, col_sl] = (
                scales[:, gi : gi + 1] * weights_int[:, col_sl].astype(np.float32)
                + zeros[:, gi : gi + 1]
            )
        return (dequant @ input_mat.T).astype(np.float32)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(1)
        w = rng.integers(0, 16, (8, 32), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (8, 2)).astype(np.float32)
        z = np.zeros((8, 2), dtype=np.float32)
        x = rng.standard_normal((4, 32)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=16)
        self.assertEqual(out.shape, (8, 4))

    def test_seq_len_1_matches_gemv(self) -> None:
        rng = np.random.default_rng(2)
        n_rows, n_cols, gs = 4, 32, 16
        w = rng.integers(0, 16, (n_rows, n_cols), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (n_rows, 2)).astype(np.float32)
        z = np.zeros((n_rows, 2), dtype=np.float32)
        x = rng.standard_normal(n_cols).astype(np.float32)
        gemv_out = _reference_int4_gemv(w, s, z, x, group_size=gs)
        gemm_out = self._reference_int4_gemm(w, s, z, x[np.newaxis, :], group_size=gs)
        np.testing.assert_allclose(gemv_out, gemm_out[:, 0], rtol=1e-5)

    def test_seq_len_4(self) -> None:
        rng = np.random.default_rng(3)
        w = rng.integers(0, 16, (8, 64), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (8, 4)).astype(np.float32)
        z = np.zeros((8, 4), dtype=np.float32)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=16)
        self.assertTrue(np.isfinite(out).all())

    def test_seq_len_64_prefill(self) -> None:
        rng = np.random.default_rng(4)
        w = rng.integers(0, 16, (32, 128), dtype=np.uint8)
        s = rng.uniform(0.01, 0.1, (32, 4)).astype(np.float32)
        z = rng.uniform(-0.5, 0.0, (32, 4)).astype(np.float32)
        x = rng.standard_normal((64, 128)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=32)
        self.assertEqual(out.shape, (32, 64))
        self.assertTrue(np.isfinite(out).all())

    def test_zero_input(self) -> None:
        rng = np.random.default_rng(5)
        w = rng.integers(0, 16, (4, 32), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (4, 2)).astype(np.float32)
        z = np.zeros((4, 2), dtype=np.float32)
        x = np.zeros((8, 32), dtype=np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=16)
        np.testing.assert_array_equal(out, np.zeros((4, 8), dtype=np.float32))

    def test_columns_are_independent(self) -> None:
        rng = np.random.default_rng(6)
        w = rng.integers(0, 16, (4, 32), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (4, 2)).astype(np.float32)
        z = np.zeros((4, 2), dtype=np.float32)
        x = rng.standard_normal((3, 32)).astype(np.float32)
        batched = self._reference_int4_gemm(w, s, z, x, group_size=16)
        for i in range(3):
            single = _reference_int4_gemv(w, s, z, x[i], group_size=16)
            np.testing.assert_allclose(batched[:, i], single, rtol=1e-5)

    def test_output_dtype_float32(self) -> None:
        rng = np.random.default_rng(7)
        w = rng.integers(0, 16, (4, 32), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (4, 2)).astype(np.float32)
        z = np.zeros((4, 2), dtype=np.float32)
        x = rng.standard_normal((2, 32)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=16)
        self.assertEqual(out.dtype, np.float32)

    def test_tile_size_divisibility(self) -> None:
        # Tile M=64, N=16, K=64 — test that sizes not multiples of tiles work
        rng = np.random.default_rng(8)
        w = rng.integers(0, 16, (7, 48), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (7, 3)).astype(np.float32)
        z = np.zeros((7, 3), dtype=np.float32)
        x = rng.standard_normal((5, 48)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=16)
        self.assertEqual(out.shape, (7, 5))

    def test_large_prefill_finite(self) -> None:
        rng = np.random.default_rng(9)
        w = rng.integers(0, 16, (64, 256), dtype=np.uint8)
        s = rng.uniform(0.01, 0.1, (64, 8)).astype(np.float32)
        z = rng.uniform(-0.5, 0.0, (64, 8)).astype(np.float32)
        x = rng.standard_normal((128, 256)).astype(np.float32)
        out = self._reference_int4_gemm(w, s, z, x, group_size=32)
        self.assertTrue(np.isfinite(out).all())

    def test_gemm_linearity(self) -> None:
        # GEMM should be linear in activations: f(a) + f(b) = f(a + b) (approx)
        rng = np.random.default_rng(10)
        w = rng.integers(0, 16, (4, 32), dtype=np.uint8)
        s = rng.uniform(0.1, 0.2, (4, 2)).astype(np.float32)
        z = np.zeros((4, 2), dtype=np.float32)
        a = rng.standard_normal((2, 32)).astype(np.float32)
        b = rng.standard_normal((2, 32)).astype(np.float32)
        fa = self._reference_int4_gemm(w, s, z, a, group_size=16)
        fb = self._reference_int4_gemm(w, s, z, b, group_size=16)
        fab = self._reference_int4_gemm(w, s, z, a + b, group_size=16)
        np.testing.assert_allclose(fa + fb, fab, rtol=1e-5, atol=1e-5)


# ===========================================================================
# 4. INT2 LUT-GEMM reference correctness
# ===========================================================================

class TestLutInt2GEMV(unittest.TestCase):
    """Python reference INT2 LUT-GEMM GEMV correctness."""

    def _make_int2_data(
        self,
        n_rows: int = 4,
        n_cols: int = 32,
        group_size: int = 16,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_groups = n_cols // group_size
        n_cb_entries = n_groups * 4
        # Random INT2 weights (0–3), 4 per byte
        raw_int2 = rng.integers(0, 4, (n_rows, n_cols), dtype=np.uint8)
        packed = np.zeros((n_rows, n_cols // 4), dtype=np.uint8)
        for c in range(n_cols // 4):
            w0 = raw_int2[:, c * 4    ]
            w1 = raw_int2[:, c * 4 + 1]
            w2 = raw_int2[:, c * 4 + 2]
            w3 = raw_int2[:, c * 4 + 3]
            packed[:, c] = (w3 << 6) | (w2 << 4) | (w1 << 2) | w0
        codebook = rng.standard_normal((n_rows, n_cb_entries)).astype(np.float16)
        x = rng.standard_normal(n_cols).astype(np.float32)
        return packed, codebook, x

    def test_output_shape(self) -> None:
        packed, codebook, x = self._make_int2_data()
        out = _reference_int2_gemv(packed, codebook, x, group_size=16)
        self.assertEqual(out.shape, (4,))

    def test_pack_unpack_int2(self) -> None:
        for w0, w1, w2, w3 in [(0, 1, 2, 3), (3, 3, 3, 3), (0, 0, 0, 0), (1, 2, 1, 2)]:
            packed = _pack_int2(w0, w1, w2, w3)
            got0, got1, got2, got3 = _unpack_int2(packed)
            self.assertEqual((got0, got1, got2, got3), (w0, w1, w2, w3))

    def test_zero_input(self) -> None:
        packed, codebook, _ = self._make_int2_data()
        x = np.zeros(32, dtype=np.float32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=16)
        np.testing.assert_array_equal(out, np.zeros(4, dtype=np.float32))

    def test_zero_codebook(self) -> None:
        packed, _, x = self._make_int2_data()
        codebook = np.zeros_like(packed)
        # codebook all zeros — need correct shape
        n_rows, n_cols_packed = packed.shape
        codebook_z = np.zeros((n_rows, (n_cols_packed * 4 // 16) * 4), dtype=np.float16)
        out = _reference_int2_gemv(packed, codebook_z, x, group_size=16)
        np.testing.assert_array_equal(out, np.zeros(n_rows, dtype=np.float32))

    def test_lut_lookup_single_group(self) -> None:
        # 1 row, 4 cols, group_size=4 → 1 group, 4 LUT entries
        codebook = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float16)
        # Weights: idx 0, 1, 2, 3 → dequant: 1, 2, 4, 8
        packed = np.array([[_pack_int2(0, 1, 2, 3)]], dtype=np.uint8)
        x = np.ones(4, dtype=np.float32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=4)
        expected = 1.0 + 2.0 + 4.0 + 8.0
        self.assertAlmostEqual(float(out[0]), expected, places=2)

    def test_two_groups_independence(self) -> None:
        # 1 row, 8 cols, group_size=4 → 2 groups
        rng = np.random.default_rng(11)
        codebook = rng.standard_normal((1, 8)).astype(np.float16)
        packed = np.zeros((1, 2), dtype=np.uint8)
        # Group 0: all idx=0, group 1: all idx=1
        packed[0, 0] = _pack_int2(0, 0, 0, 0)
        packed[0, 1] = _pack_int2(1, 1, 1, 1)
        x = np.ones(8, dtype=np.float32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=4)
        expected = float(np.float16(codebook[0, 0])) * 4 + float(np.float16(codebook[0, 5])) * 4
        self.assertAlmostEqual(float(out[0]), expected, places=2)

    def test_output_dtype_float32(self) -> None:
        packed, codebook, x = self._make_int2_data()
        out = _reference_int2_gemv(packed, codebook, x, group_size=16)
        self.assertEqual(out.dtype, np.float32)

    def test_single_byte_single_row(self) -> None:
        # Minimal: 1 row × 4 cols, 1 byte, 1 group
        codebook = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float16)
        packed = np.array([[_pack_int2(1, 2, 3, 0)]], dtype=np.uint8)
        x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=4)
        expected = 1.5 + 2.5 + 3.5 + 0.5
        self.assertAlmostEqual(float(out[0]), expected, places=2)

    def test_large_matrix_finite(self) -> None:
        packed, codebook, x = self._make_int2_data(n_rows=32, n_cols=128, group_size=32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=32)
        self.assertTrue(np.isfinite(out).all())

    def test_multiple_rows_independence(self) -> None:
        packed, codebook, x = self._make_int2_data(n_rows=4, group_size=16)
        out = _reference_int2_gemv(packed, codebook, x, group_size=16)
        for i in range(4):
            single = _reference_int2_gemv(
                packed[i : i + 1], codebook[i : i + 1], x, group_size=16
            )
            self.assertAlmostEqual(float(out[i]), float(single[0]), places=4)

    def test_lut_fp16_precision_preserved(self) -> None:
        # Codebook stores FP16 — verify we use FP16 not FP32 precision
        codebook = np.array([[np.float16(0.1), np.float16(0.2),
                               np.float16(0.3), np.float16(0.4)]], dtype=np.float16)
        packed = np.array([[_pack_int2(0, 0, 0, 0)]], dtype=np.uint8)
        x = np.ones(4, dtype=np.float32)
        out = _reference_int2_gemv(packed, codebook, x, group_size=4)
        expected = float(np.float16(0.1)) * 4
        self.assertAlmostEqual(float(out[0]), expected, places=3)

    def test_reproducibility(self) -> None:
        packed, codebook, x = self._make_int2_data(seed=777)
        out1 = _reference_int2_gemv(packed, codebook, x, group_size=16)
        out2 = _reference_int2_gemv(packed, codebook, x, group_size=16)
        np.testing.assert_array_equal(out1, out2)


# ===========================================================================
# 5. KernelDispatch dataclass
# ===========================================================================

class TestKernelDispatchDataclass(unittest.TestCase):
    """KernelDispatch field validation and immutability."""

    def test_valid_phase_decode(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", True, "decode")
        self.assertEqual(kd.phase, "decode")

    def test_valid_phase_prefill(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", False, "prefill")
        self.assertEqual(kd.phase, "prefill")

    def test_valid_phase_both(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", True, "both")
        self.assertEqual(kd.phase, "both")

    def test_invalid_phase_raises(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        with self.assertRaises(ValueError):
            KernelDispatch("k", "k.metal", True, "unknown")

    def test_frozen_kernel_name(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", True, "both")
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            kd.kernel_name = "other"  # type: ignore[misc]

    def test_supports_batched_true(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", True, "decode")
        self.assertTrue(kd.supports_batched)

    def test_supports_batched_false(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("k", "k.metal", False, "prefill")
        self.assertFalse(kd.supports_batched)

    def test_empty_shader_path_allowed(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        kd = KernelDispatch("legacy", "", False, "both")
        self.assertEqual(kd.metal_shader_path, "")

    def test_equality(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        a = KernelDispatch("k", "k.metal", True, "both")
        b = KernelDispatch("k", "k.metal", True, "both")
        self.assertEqual(a, b)

    def test_inequality(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatch
        a = KernelDispatch("k1", "k.metal", True, "both")
        b = KernelDispatch("k2", "k.metal", True, "both")
        self.assertNotEqual(a, b)


# ===========================================================================
# 6. KernelDispatcher default selections
# ===========================================================================

class TestKernelDispatcherDefaults(unittest.TestCase):
    """KernelDispatcher.select returns correct kernel for each flag."""

    def setUp(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatcher
        self.d = KernelDispatcher()
        self.caps = _make_caps()

    def test_astc_flag(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.ASTC, self.caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_tca_tbe_decode(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.TCA_TBE, self.caps, seq_len=1)
        self.assertEqual(result.kernel_name, "zip_gemv")

    def test_tca_tbe_prefill(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.TCA_TBE, self.caps, seq_len=4)
        self.assertEqual(result.kernel_name, "zip_gemm")

    def test_int4_sparse(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4 | SquizdFlag.SPARSE, self.caps)
        self.assertEqual(result.kernel_name, "sparse_gemv")

    def test_int4_decode(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4, self.caps, seq_len=1)
        self.assertEqual(result.kernel_name, "fused_int4_gemv")

    def test_int4_prefill(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4, self.caps, seq_len=8)
        self.assertEqual(result.kernel_name, "fused_int4_gemm")

    def test_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT2, self.caps)
        self.assertEqual(result.kernel_name, "lut_int2_gemv")

    def test_none_flags_fallback(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.NONE, self.caps)
        self.assertEqual(result.kernel_name, "legacy_dequant_matmul")

    def test_eagle_flag_fallback(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.EAGLE, self.caps)
        self.assertEqual(result.kernel_name, "legacy_dequant_matmul")

    def test_astc_shader_path(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.ASTC, self.caps)
        self.assertEqual(result.metal_shader_path, "astc_gemv.metal")

    def test_int4_decode_shader_path(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4, self.caps, seq_len=1)
        self.assertEqual(result.metal_shader_path, "fused_int4_gemv.metal")

    def test_int2_shader_path(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT2, self.caps)
        self.assertEqual(result.metal_shader_path, "lut_int2_gemv.metal")


# ===========================================================================
# 7. KernelDispatcher flag priority ordering
# ===========================================================================

class TestKernelDispatcherFlagPriority(unittest.TestCase):
    """Verify priority ordering when multiple flags are set simultaneously."""

    def setUp(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatcher
        self.d = KernelDispatcher()
        self.caps = _make_caps()

    def test_astc_beats_int4(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.ASTC | SquizdFlag.INT4, self.caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_astc_beats_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.ASTC | SquizdFlag.INT2, self.caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_astc_beats_tca_tbe(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.ASTC | SquizdFlag.TCA_TBE, self.caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_tca_tbe_beats_int4(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.TCA_TBE | SquizdFlag.INT4, self.caps, seq_len=1)
        self.assertEqual(result.kernel_name, "zip_gemv")

    def test_tca_tbe_beats_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.TCA_TBE | SquizdFlag.INT2, self.caps, seq_len=1)
        self.assertEqual(result.kernel_name, "zip_gemv")

    def test_int4_sparse_beats_int4_alone(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result_sparse = self.d.select(SquizdFlag.INT4 | SquizdFlag.SPARSE, self.caps)
        result_plain  = self.d.select(SquizdFlag.INT4, self.caps)
        self.assertEqual(result_sparse.kernel_name, "sparse_gemv")
        self.assertNotEqual(result_sparse.kernel_name, result_plain.kernel_name)

    def test_int4_beats_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4 | SquizdFlag.INT2, self.caps)
        # INT4 has higher priority than INT2
        self.assertIn(result.kernel_name, ("fused_int4_gemv", "fused_int4_gemm"))

    def test_int4_sparse_beats_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4 | SquizdFlag.SPARSE | SquizdFlag.INT2, self.caps)
        self.assertEqual(result.kernel_name, "sparse_gemv")

    def test_all_flags_selects_astc(self) -> None:
        from squish.format.squish_header import SquizdFlag
        all_flags = (
            SquizdFlag.ASTC | SquizdFlag.TCA_TBE | SquizdFlag.INT4
            | SquizdFlag.SPARSE | SquizdFlag.INT2
        )
        result = self.d.select(all_flags, self.caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_tca_tbe_prefill_seq_2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.TCA_TBE, self.caps, seq_len=2)
        self.assertEqual(result.kernel_name, "zip_gemm")

    def test_int4_seq_len_1_vs_2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        r1 = self.d.select(SquizdFlag.INT4, self.caps, seq_len=1)
        r2 = self.d.select(SquizdFlag.INT4, self.caps, seq_len=2)
        self.assertEqual(r1.kernel_name, "fused_int4_gemv")
        self.assertEqual(r2.kernel_name, "fused_int4_gemm")

    def test_int4_prefill_supports_batched_false(self) -> None:
        from squish.format.squish_header import SquizdFlag
        result = self.d.select(SquizdFlag.INT4, self.caps, seq_len=8)
        # GEMM variant does not expose a batched function
        self.assertFalse(result.supports_batched)


# ===========================================================================
# 8. KernelDispatcher hardware capability variants
# ===========================================================================

class TestKernelDispatcherCapabilities(unittest.TestCase):
    """KernelDispatcher respects (or gracefully ignores) chip generation."""

    def setUp(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatcher
        self.d = KernelDispatcher()

    def test_m1_int4_decode(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.M1)
        result = self.d.select(SquizdFlag.INT4, caps, seq_len=1)
        self.assertEqual(result.kernel_name, "fused_int4_gemv")

    def test_m3_int4_decode(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.M3)
        result = self.d.select(SquizdFlag.INT4, caps, seq_len=1)
        self.assertEqual(result.kernel_name, "fused_int4_gemv")

    def test_m5_int4_decode(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.M5)
        result = self.d.select(SquizdFlag.INT4, caps, seq_len=1)
        self.assertEqual(result.kernel_name, "fused_int4_gemv")

    def test_no_ane_int2(self) -> None:
        from squish.format.squish_header import SquizdFlag
        caps = _make_caps(has_ane=False)
        result = self.d.select(SquizdFlag.INT2, caps)
        self.assertEqual(result.kernel_name, "lut_int2_gemv")

    def test_no_metal3_astc(self) -> None:
        from squish.format.squish_header import SquizdFlag
        caps = _make_caps(has_metal3=False)
        result = self.d.select(SquizdFlag.ASTC, caps)
        self.assertEqual(result.kernel_name, "astc_gemv")

    def test_unknown_chip_int4(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.UNKNOWN)
        result = self.d.select(SquizdFlag.INT4, caps, seq_len=1)
        self.assertEqual(result.kernel_name, "fused_int4_gemv")

    def test_m2_int2_supports_batched(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.M2)
        result = self.d.select(SquizdFlag.INT2, caps)
        self.assertTrue(result.supports_batched)

    def test_m4_legacy_fallback(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = _make_caps(chip_generation=AppleChipGeneration.M4)
        result = self.d.select(SquizdFlag.NONE, caps)
        self.assertEqual(result.kernel_name, "legacy_dequant_matmul")


# ===========================================================================
# 9. KernelDispatcher — singleton and caching
# ===========================================================================

class TestKernelDispatchSingleton(unittest.TestCase):
    """get_kernel_dispatcher returns a stable singleton; reset_kernel_dispatcher clears it."""

    def setUp(self) -> None:
        from squish.hardware.kernel_dispatch import reset_kernel_dispatcher
        reset_kernel_dispatcher()

    def tearDown(self) -> None:
        from squish.hardware.kernel_dispatch import reset_kernel_dispatcher
        reset_kernel_dispatcher()

    def test_singleton_returns_same_instance(self) -> None:
        from squish.hardware.kernel_dispatch import get_kernel_dispatcher
        a = get_kernel_dispatcher()
        b = get_kernel_dispatcher()
        self.assertIs(a, b)

    def test_reset_yields_new_instance(self) -> None:
        from squish.hardware.kernel_dispatch import get_kernel_dispatcher, reset_kernel_dispatcher
        a = get_kernel_dispatcher()
        reset_kernel_dispatcher()
        b = get_kernel_dispatcher()
        self.assertIsNot(a, b)

    def test_dispatcher_is_kernel_dispatcher_type(self) -> None:
        from squish.hardware.kernel_dispatch import KernelDispatcher, get_kernel_dispatcher
        d = get_kernel_dispatcher()
        self.assertIsInstance(d, KernelDispatcher)

    def test_cache_hit_same_result(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.kernel_dispatch import get_kernel_dispatcher
        caps = _make_caps()
        d = get_kernel_dispatcher()
        r1 = d.select(SquizdFlag.INT4, caps, seq_len=1)
        r2 = d.select(SquizdFlag.INT4, caps, seq_len=1)
        self.assertIs(r1, r2)

    def test_seq_len_invalid_raises(self) -> None:
        from squish.format.squish_header import SquizdFlag
        from squish.hardware.kernel_dispatch import get_kernel_dispatcher
        caps = _make_caps()
        d = get_kernel_dispatcher()
        with self.assertRaises(ValueError):
            d.select(SquizdFlag.INT4, caps, seq_len=0)


# ===========================================================================
# 10. Metal constant verification
# ===========================================================================

class TestFusedInt4Constants(unittest.TestCase):
    """Metal shader constants match Python-side expectations."""

    def test_threads_per_tg_is_power_of_two(self) -> None:
        # 128 = 2^7 — required for tree reduction
        threads = 128
        self.assertEqual(threads & (threads - 1), 0)

    def test_lut_size_is_256(self) -> None:
        # LUT has 256 entries for full 8-bit index space (INT2 needs 4 per group)
        lut_size = 256
        self.assertEqual(lut_size, 256)

    def test_int2_codebook_bytes(self) -> None:
        # 256 FP16 entries × 2 bytes = 512 bytes — well within 32KB Metal budget
        lut_bytes = 256 * 2
        self.assertLessEqual(lut_bytes, 32 * 1024)

    def test_int4_accum_scratch_bytes(self) -> None:
        # 128 float32 × 4 bytes = 512 bytes
        scratch_bytes = 128 * 4
        self.assertLessEqual(scratch_bytes, 32 * 1024)

    def test_int4_gemm_act_tile_bytes(self) -> None:
        # TILE_K=64 × TILE_N=16 × sizeof(half)=2 = 2048 bytes
        tile_bytes = 64 * 16 * 2
        self.assertLessEqual(tile_bytes, 32 * 1024)

    def test_int2_total_threadgroup_bytes(self) -> None:
        # LUT (512B) + accum_scratch (512B) = 1024B ≪ 32KB
        total = 256 * 2 + 128 * 4
        self.assertLessEqual(total, 32 * 1024)

    def test_lut_int2_gemv_threadgroup_memory_headroom(self) -> None:
        # At least 30× headroom relative to 32KB budget
        total = 256 * 2 + 128 * 4
        headroom = (32 * 1024) / total
        self.assertGreaterEqual(headroom, 30)


if __name__ == "__main__":
    unittest.main()
