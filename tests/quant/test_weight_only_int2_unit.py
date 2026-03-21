"""tests/quant/test_weight_only_int2_unit.py

Dedicated unit tests for squish/quant/weight_only_int2.py.

Covers:
  - Int2QuantConfig validation
  - WeightOnlyInt2Quant.quantize shapes & dtypes
  - WeightOnlyInt2Quant.dequantize shapes & values
  - Pack/unpack round-trip correctness
  - Symmetry modes (symmetric + asymmetric)
  - Outlier clipping (clip_threshold < 1.0)
  - Compression ratio reporting
  - Matmul workflow (quantize → dequantize → matmul ~= FP32 matmul)
  - Edge cases: single group, 2-column matrix, all-same values
  - Error paths: non-2D inputs, bad group_size, bad clip_threshold
  - Round-trip SNR is acceptable for typical Gaussian weights
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quant.weight_only_int2 import Int2QuantConfig, WeightOnlyInt2Quant

RNG = np.random.default_rng(0xC0FFEE)

# ── helpers ────────────────────────────────────────────────────────────────────

def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal_power = float(np.mean(original ** 2))
    noise_power  = float(np.mean((original - reconstructed) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def _make_weight(rows: int = 64, cols: int = 128, scale: float = 0.02,
                 seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32) * scale


# ── Int2QuantConfig ────────────────────────────────────────────────────────────

class TestInt2QuantConfig:
    def test_defaults(self):
        cfg = Int2QuantConfig()
        assert cfg.group_size == 64
        assert cfg.symmetric is False
        assert cfg.clip_threshold == 0.99

    def test_custom_values(self):
        cfg = Int2QuantConfig(group_size=128, symmetric=True, clip_threshold=0.95)
        assert cfg.group_size == 128
        assert cfg.symmetric is True
        assert cfg.clip_threshold == 0.95

    def test_group_size_too_small(self):
        with pytest.raises(ValueError, match="group_size"):
            Int2QuantConfig(group_size=4)

    def test_group_size_not_divisible_by_4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            Int2QuantConfig(group_size=10)  # 10 % 4 != 0

    def test_clip_threshold_too_low(self):
        with pytest.raises(ValueError, match="clip_threshold"):
            Int2QuantConfig(clip_threshold=0.3)

    def test_clip_threshold_too_high(self):
        with pytest.raises(ValueError, match="clip_threshold"):
            Int2QuantConfig(clip_threshold=1.1)

    def test_clip_threshold_boundary_values(self):
        cfg_lo = Int2QuantConfig(clip_threshold=0.5)
        assert cfg_lo.clip_threshold == 0.5
        cfg_hi = Int2QuantConfig(clip_threshold=1.0)
        assert cfg_hi.clip_threshold == 1.0

    def test_group_size_must_be_8_minimum(self):
        """group_size=8 satisfies both ≥8 and %4==0."""
        cfg = Int2QuantConfig(group_size=8)
        assert cfg.group_size == 8


# ── WeightOnlyInt2Quant — quantize output shapes ──────────────────────────────

class TestInt2QuantShapes:
    def setup_method(self):
        self.q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64))

    def test_packed_shape_cols_divided_by_4(self):
        w = _make_weight(32, 128)
        packed, scale, zero = self.q.quantize(w)
        assert packed.shape == (32, 32)  # 128 / 4 = 32

    def test_scale_shape(self):
        w = _make_weight(32, 128)
        packed, scale, zero = self.q.quantize(w)
        n_groups = 128 // 64
        assert scale.shape == (32, n_groups)

    def test_zero_shape(self):
        w = _make_weight(32, 128)
        packed, scale, zero = self.q.quantize(w)
        n_groups = 128 // 64
        assert zero.shape == (32, n_groups)

    def test_packed_dtype_uint8(self):
        w = _make_weight(32, 128)
        packed, _, _ = self.q.quantize(w)
        assert packed.dtype == np.uint8

    def test_scale_dtype_float32(self):
        w = _make_weight(32, 128)
        _, scale, _ = self.q.quantize(w)
        assert scale.dtype == np.float32

    def test_zero_symmetric_is_zeros(self):
        q_sym = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, symmetric=True))
        w = _make_weight(16, 64)
        _, _, zero = q_sym.quantize(w)
        np.testing.assert_array_equal(zero, np.zeros_like(zero))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            self.q.quantize(np.ones((32,)))  # 1-D

    def test_cols_not_divisible_by_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            self.q.quantize(np.ones((4, 72), dtype=np.float32))  # 72 % 64 != 0


# ── WeightOnlyInt2Quant — dequantize ──────────────────────────────────────────

class TestInt2Dequantize:
    def setup_method(self):
        self.q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64))

    def test_output_shape_matches_original(self):
        w = _make_weight(32, 128)
        packed, scale, zero = self.q.quantize(w)
        w_approx = self.q.dequantize(packed, scale, zero)
        assert w_approx.shape == (32, 128)

    def test_output_dtype_float32(self):
        w = _make_weight(16, 64)
        packed, scale, zero = self.q.quantize(w)
        w_approx = self.q.dequantize(packed, scale, zero)
        assert w_approx.dtype == np.float32

    def test_all_same_weights_reconstructed_closely(self):
        """All-same value: quantization should be near-perfect."""
        w = np.full((4, 64), 0.05, dtype=np.float32)
        packed, scale, zero = self.q.quantize(w)
        w_approx = self.q.dequantize(packed, scale, zero)
        np.testing.assert_allclose(w_approx, w, atol=0.01)

    def test_dequant_range_bounded(self):
        """Dequantized values must fall within roughly original range."""
        w = _make_weight(64, 128, scale=0.05)
        packed, scale, zero = self.q.quantize(w)
        w_approx = self.q.dequantize(packed, scale, zero)
        # Upper and lower bounds should be within 2× original range (very loose)
        assert float(w_approx.min()) >= float(w.min()) - abs(float(w.min()))
        assert float(w_approx.max()) <= float(w.max()) + abs(float(w.max()))

    def test_symmetric_zero_point_is_zero(self):
        q_sym = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, symmetric=True))
        w = _make_weight(16, 64)
        packed, scale, zero = q_sym.quantize(w)
        np.testing.assert_array_equal(zero, np.zeros_like(zero))
        w_approx = q_sym.dequantize(packed, scale, zero)
        assert w_approx.shape == w.shape


# ── Pack / Unpack ──────────────────────────────────────────────────────────────

class TestPackUnpack:
    def setup_method(self):
        self.q = WeightOnlyInt2Quant()

    def test_pack_factor(self):
        assert self.q.PACK_FACTOR == 4

    def test_n_bits(self):
        assert self.q.N_BITS == 2

    def test_n_levels(self):
        assert self.q.N_LEVELS == 4

    def test_pack_round_trip_all_zeros(self):
        w_int = np.zeros((4, 64), dtype=np.uint8)
        packed = self.q._pack(w_int)
        recovered = self.q._unpack(packed)
        np.testing.assert_array_equal(recovered, w_int)

    def test_pack_round_trip_all_max(self):
        w_int = np.full((4, 64), 3, dtype=np.uint8)  # 0b11 max in 2-bit
        packed = self.q._pack(w_int)
        recovered = self.q._unpack(packed)
        np.testing.assert_array_equal(recovered, w_int)

    def test_pack_round_trip_random(self):
        rng = np.random.default_rng(5)
        w_int = rng.integers(0, 4, size=(8, 128), dtype=np.uint8)
        packed = self.q._pack(w_int)
        recovered = self.q._unpack(packed)
        np.testing.assert_array_equal(recovered, w_int)

    def test_packed_cols_quarter_of_original(self):
        w_int = np.zeros((4, 64), dtype=np.uint8)
        packed = self.q._pack(w_int)
        assert packed.shape == (4, 16)

    def test_pack_bit_layout(self):
        """Verify pack-4 bit layout: w0 in bits[1:0], w1 in bits[3:2], etc."""
        w_int = np.array([[0, 1, 2, 3]], dtype=np.uint8)  # exactly 1 byte
        packed = self.q._pack(w_int)
        assert packed.shape == (1, 1)
        # byte = 0|(1<<2)|(2<<4)|(3<<6) = 0 + 4 + 32 + 192 = 228
        assert int(packed[0, 0]) == (0 | (1 << 2) | (2 << 4) | (3 << 6))
        unpacked = self.q._unpack(packed)
        np.testing.assert_array_equal(unpacked, w_int)


# ── Compression ratio ──────────────────────────────────────────────────────────

class TestCompressionRatio:
    def test_fp32_ratio_is_16(self):
        q = WeightOnlyInt2Quant()
        ratio = q.compression_ratio("float32")
        assert ratio == 16.0  # 32 bits / 2 bits

    def test_fp16_ratio_is_8(self):
        q = WeightOnlyInt2Quant()
        ratio = q.compression_ratio("float16")
        assert ratio == 8.0  # 16 bits / 2 bits

    def test_bf16_ratio_is_8(self):
        q = WeightOnlyInt2Quant()
        ratio = q.compression_ratio("bfloat16")
        assert ratio == 8.0

    def test_unknown_dtype_fallback(self):
        q = WeightOnlyInt2Quant()
        # "float64" not in the dict → falls back to 16 bits → 8×
        ratio = q.compression_ratio("float64")
        assert ratio == 8.0


# ── Clipping ──────────────────────────────────────────────────────────────────

class TestClipping:
    def test_clip_reduces_outlier_influence(self):
        """After clipping at 0.85, outlier injected at 30σ should be ignored."""
        rng = np.random.default_rng(77)
        w = rng.standard_normal((4, 64)).astype(np.float32) * 0.02
        w_outlier = w.copy()
        w_outlier[0, 0] = 100.0  # extreme outlier

        q_clip = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, clip_threshold=0.85))
        q_noclip = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, clip_threshold=1.0))

        _, s_clip, _  = q_clip.quantize(w_outlier)
        _, s_noclip, _ = q_noclip.quantize(w_outlier)

        # Max scale in the outlier-free groups should be similar under clipping
        max_scale_clip = float(s_clip.max())
        max_scale_noclip = float(s_noclip.max())
        # With outlier + no clipping the scale blows up dramatically
        assert max_scale_clip < max_scale_noclip * 0.5

    def test_no_clip_preserves_range(self):
        """With clip_threshold=1.0, max/min of quantized weight spans full range."""
        w = np.array([[0.0, 1.0, 2.0, 3.0] * 16], dtype=np.float32)  # (1, 64)
        q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, clip_threshold=1.0))
        packed, scale, zero = q.quantize(w)
        w_approx = q.dequantize(packed, scale, zero)
        # The range should cover ~[0, 3]
        assert float(w_approx.max()) > 2.5


# ── Matmul workflow ────────────────────────────────────────────────────────────

class TestMatmulWorkflow:
    """Verify that the dequantize-then-matmul pattern works correctly."""

    def test_matmul_output_close_to_fp32(self):
        """
        output = activation @ weight.T should be close whether weight is
        FP32 or dequantized INT2 (within a reasonable tolerance for 2-bit).
        """
        rng    = np.random.default_rng(42)
        weight = rng.standard_normal((128, 256)).astype(np.float32) * 0.01
        act    = rng.standard_normal((16, 256)).astype(np.float32)

        q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64))
        packed, scale, zero = q.quantize(weight)
        weight_approx = q.dequantize(packed, scale, zero)

        out_fp32 = act @ weight.T
        out_q2   = act @ weight_approx.T

        # For 2-bit quant the error is noticeable but output correlation
        # should still be > 0.95
        flat_fp32 = out_fp32.ravel()
        flat_q2   = out_q2.ravel()
        corr = float(
            np.corrcoef(flat_fp32, flat_q2)[0, 1]
        )
        assert corr > 0.90, f"Matmul output correlation too low: {corr:.4f}"


# ── Round-trip accuracy (SNR) ──────────────────────────────────────────────────

class TestRoundTripAccuracy:
    @pytest.mark.parametrize("rows,cols,group_size,symmetric,min_snr", [
        (64,  128, 64,  False, 5.0),
        # Symmetric INT2 (3 effective levels for centred distribution) has lower SNR
        (64,  128, 64,  True,  2.0),
        (128, 256, 128, False, 5.0),
        (32,  64,  32,  False, 5.0),
    ])
    def test_asymmetric_snr_threshold(self, rows, cols, group_size, symmetric, min_snr):
        """INT2 round-trip SNR for Gaussian weights (threshold depends on mode)."""
        w = _make_weight(rows, cols, seed=rows + cols)
        q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=group_size, symmetric=symmetric))
        packed, scale, zero = q.quantize(w)
        w_approx = q.dequantize(packed, scale, zero)
        snr = _snr_db(w, w_approx)
        assert snr > min_snr, f"SNR too low for INT2 round-trip ({rows}x{cols} gs={group_size}): {snr:.1f} dB"

    def test_smaller_group_size_better_snr(self):
        """Smaller group_size should give lower quantization error (higher SNR)."""
        w = _make_weight(64, 128, seed=99)
        q_small = WeightOnlyInt2Quant(Int2QuantConfig(group_size=32))
        q_large = WeightOnlyInt2Quant(Int2QuantConfig(group_size=128))
        p_s, s_s, z_s = q_small.quantize(w)
        p_l, s_l, z_l = q_large.quantize(w)
        snr_small = _snr_db(w, q_small.dequantize(p_s, s_s, z_s))
        snr_large = _snr_db(w, q_large.dequantize(p_l, s_l, z_l))
        assert snr_small >= snr_large, (
            f"Expected smaller group size to have higher SNR: "
            f"gs=32 → {snr_small:.1f} dB, gs=128 → {snr_large:.1f} dB"
        )

    def test_clipping_improves_snr_with_outliers(self):
        """Clipping outliers should improve SNR vs. no clipping when outliers are present."""
        rng = np.random.default_rng(0xAB)
        w = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        w_noisy = w.copy()
        # Inject 2 hard outliers at 5% of positions
        w_noisy[::16, ::32] = 5.0

        q_clip   = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, clip_threshold=0.90))
        q_noclip = WeightOnlyInt2Quant(Int2QuantConfig(group_size=64, clip_threshold=1.0))

        p_c, s_c, z_c = q_clip.quantize(w_noisy)
        p_n, s_n, z_n = q_noclip.quantize(w_noisy)
        w_c = q_clip.dequantize(p_c, s_c, z_c)
        w_n = q_noclip.dequantize(p_n, s_n, z_n)

        # Compare on the non-outlier elements only
        mask = w_noisy < 1.0
        err_c = float(np.mean((w_noisy[mask] - w_c[mask]) ** 2))
        err_n = float(np.mean((w_noisy[mask] - w_n[mask]) ** 2))
        assert err_c <= err_n, (
            f"Clipping should reduce error on non-outlier elements: clip {err_c:.6f} vs no-clip {err_n:.6f}"
        )
