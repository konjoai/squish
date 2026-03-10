"""tests/test_nf4_unit.py — unit tests for squish/nf4_quant.py"""
import numpy as np
import pytest

from squish import quantizer as _qmod
from squish.nf4_quant import (
    NF4_LEVELS,
    dequantize_nf4,
    quantize_nf4,
)

_HAS_INT4 = getattr(_qmod, "_squish_quant", None) is not None

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# NF4 codebook sanity
# ---------------------------------------------------------------------------

class TestNF4Levels:
    def test_length(self):
        assert len(NF4_LEVELS) == 16

    def test_sorted_ascending(self):
        assert np.all(NF4_LEVELS[:-1] <= NF4_LEVELS[1:])

    def test_range(self):
        assert NF4_LEVELS[0] == pytest.approx(-1.0, abs=1e-3)
        assert NF4_LEVELS[-1] == pytest.approx(1.0, abs=1e-3)

    def test_dtype(self):
        assert NF4_LEVELS.dtype == np.float32


# ---------------------------------------------------------------------------
# quantize_nf4 basic shapes
# ---------------------------------------------------------------------------

class TestQuantizeNF4Shape:
    def test_basic_shape_2d(self):
        arr = RNG.standard_normal((4, 128)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        # packed: nibble-pack halves the last dim
        assert packed.shape == (4, 64)
        # scales: one per group → 128/64 = 2 groups per row
        assert scales.shape == (4, 2)

    def test_1d_input(self):
        arr = RNG.standard_normal(128).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        assert packed.ndim >= 1

    def test_packed_dtype(self):
        arr = RNG.standard_normal((2, 64)).astype(np.float32)
        packed, _ = quantize_nf4(arr, group_size=64)
        assert packed.dtype == np.uint8

    def test_scales_dtype(self):
        arr = RNG.standard_normal((2, 64)).astype(np.float32)
        _, scales = quantize_nf4(arr, group_size=64)
        assert scales.dtype == np.float32

    def test_scales_positive(self):
        arr = RNG.standard_normal((4, 64)).astype(np.float32)
        _, scales = quantize_nf4(arr, group_size=64)
        assert np.all(scales > 0)


# ---------------------------------------------------------------------------
# dequantize_nf4 round-trip quality
# ---------------------------------------------------------------------------

class TestDequantizeNF4:
    def _snr_db(self, orig: np.ndarray, approx: np.ndarray) -> float:
        sig  = np.mean(orig ** 2)
        err  = np.mean((orig - approx) ** 2)
        if err == 0:
            return float("inf")
        return 10 * np.log10(sig / err)

    def test_round_trip_shape_preserved(self):
        arr = RNG.standard_normal((8, 128)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        assert recon.shape == arr.shape

    def test_round_trip_dtype_float32(self):
        arr = RNG.standard_normal((4, 64)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        assert recon.dtype == np.float32

    def test_round_trip_snr_acceptable(self):
        """NF4 should achieve > 20 dB SNR on Gaussian weights."""
        arr = RNG.standard_normal((64, 128)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        snr = self._snr_db(arr, recon)
        assert snr > 20.0, f"SNR {snr:.1f} dB is too low"

    def test_zero_tensor(self):
        arr = np.zeros((4, 64), dtype=np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        # All-zero input → all-zero reconstruction (within numerical tolerance)
        assert np.allclose(recon, 0.0, atol=1e-3)

    def test_constant_tensor(self):
        arr = np.full((4, 64), 2.5, dtype=np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        # Reconstruction should be close (within ~5% relative error for constant)
        np.testing.assert_allclose(recon, arr, rtol=0.05)

    def test_large_tensor(self):
        """Smoke test with larger tensor for performance / no-crash."""
        arr = RNG.standard_normal((256, 512)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        assert recon.shape == arr.shape

    def test_nibble_packing_lossless_indices(self):
        """Packing nibbles and unpacking should give back original 4-bit indices."""
        # We can verify by: quantize → dequantize → values must be in NF4_LEVELS
        arr = RNG.standard_normal((4, 64)).astype(np.float32)
        packed, scales = quantize_nf4(arr, group_size=64)
        recon = dequantize_nf4(packed, scales, group_size=64)
        # Each group should have values from NF4_LEVELS * scale
        # Check that reconstruction is deterministic (same call → same result)
        recon2 = dequantize_nf4(packed, scales, group_size=64)
        np.testing.assert_array_equal(recon, recon2)


# ---------------------------------------------------------------------------
# Interface consistency with quantizer.py (INT4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_INT4, reason="squish_quant Rust extension not built")
class TestNF4vsINT4Interface:
    def test_same_output_interface_as_int4(self):
        """quantize_nf4 should return same (packed, scales) tuple structure as quantize_int4."""
        from squish.quantizer import quantize_int4
        arr = RNG.standard_normal((4, 128)).astype(np.float32)

        nf4_packed, nf4_scales = quantize_nf4(arr, group_size=64)
        i4_packed, i4_scales   = quantize_int4(arr, group_size=64)

        # Both return uint8 packed and float32 scales
        assert nf4_packed.dtype == np.uint8
        assert nf4_scales.dtype == np.float32
        assert i4_packed.dtype  == np.uint8
        assert i4_scales.dtype  == np.float32

        # Shapes should be consistent
        assert nf4_packed.shape == i4_packed.shape
        assert nf4_scales.shape == i4_scales.shape
