"""tests/quant/test_ternary_quant_extended.py

Extended unit tests for AsymmetricTernaryQuantizer and AsymmetricTernaryResult
in squish/quant/ternary_quant.py.
"""
import numpy as np
import pytest

from squish.quant.ternary_quant import (
    AsymmetricTernaryQuantizer,
    AsymmetricTernaryResult,
    TernaryConfig,
    TernaryQuantizer,
)

RNG = np.random.default_rng(7)


def _aqtz(zero_threshold=0.5):
    return AsymmetricTernaryQuantizer(TernaryConfig(zero_threshold=zero_threshold))


# ---------------------------------------------------------------------------
# AsymmetricTernaryResult — field dtypes
# ---------------------------------------------------------------------------

class TestAsymmetricTernaryResultDtypes:
    def test_ternary_is_int8(self):
        qtz = _aqtz()
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w)
        assert result.ternary.dtype == np.int8

    def test_preserved_fp16_is_float16(self):
        qtz = _aqtz()
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[0, 1])
        assert result.preserved_fp16.dtype == np.float16

    def test_preserved_cols_is_int32(self):
        qtz = _aqtz()
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[5, 10])
        assert result.preserved_cols.dtype == np.int32

    def test_scale_is_float(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 32)).astype(np.float32)
        result = qtz.quantize(w)
        assert isinstance(result.scale, float)

    def test_original_shape_stored(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 16, 32)).astype(np.float32)
        result = qtz.quantize(w)
        assert result.original_shape == (4, 16, 32)


# ---------------------------------------------------------------------------
# quantize — no protected columns (parity with TernaryQuantizer)
# ---------------------------------------------------------------------------

class TestQuantizeNoCols:
    """With no protected columns, result must match TernaryQuantizer output."""

    def _compare(self, shape):
        w = RNG.standard_normal(shape).astype(np.float32)
        cfg = TernaryConfig(zero_threshold=0.5)

        sym = TernaryQuantizer(cfg)
        sym_tern, sym_scale = sym.quantize(w.copy())

        asym = AsymmetricTernaryQuantizer(cfg)
        result = asym.quantize(w.copy(), protected_cols=None)

        np.testing.assert_array_equal(
            result.ternary.reshape(sym_tern.shape), sym_tern,
            err_msg="Ternary codes differ from TernaryQuantizer baseline"
        )
        assert abs(result.scale - sym_scale) < 1e-5

    def test_2d_parity(self):
        self._compare((32, 128))

    def test_1d_parity(self):
        self._compare((256,))

    def test_empty_protected_list_parity(self):
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        cfg = TernaryConfig(zero_threshold=0.5)
        asym = AsymmetricTernaryQuantizer(cfg)
        result = asym.quantize(w, protected_cols=[])
        assert result.preserved_cols.size == 0
        assert result.preserved_fp16.shape[1] == 0


# ---------------------------------------------------------------------------
# quantize — with protected columns
# ---------------------------------------------------------------------------

class TestQuantizeWithCols:
    def test_protected_cols_zeroed_in_ternary(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        protected = [0, 7, 32]
        result = qtz.quantize(w, protected_cols=protected)
        for col in protected:
            np.testing.assert_array_equal(
                result.ternary[:, col],
                np.zeros(8, dtype=np.int8),
                err_msg=f"Column {col} not zeroed in ternary output"
            )

    def test_non_protected_cols_not_zeroed(self):
        qtz = _aqtz(zero_threshold=0.1)   # low threshold → few zeros
        w = RNG.standard_normal((8, 64)).astype(np.float32) * 10
        result = qtz.quantize(w, protected_cols=[0])
        # Some non-zero ternary values must exist among the unprotected columns
        unprotected_tern = result.ternary[:, 1:]
        assert unprotected_tern.any()

    def test_preserved_fp16_shape(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[3, 7, 20])
        assert result.preserved_fp16.shape == (8, 3)

    def test_preserved_fp16_values_match_original(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        protected = [5, 15]
        result = qtz.quantize(w, protected_cols=protected)
        for i, col in enumerate(protected):
            expected_fp16 = w[:, col].astype(np.float16)
            np.testing.assert_array_equal(result.preserved_fp16[:, i], expected_fp16)

    def test_preserved_cols_array_sorted(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[20, 5, 40, 1])
        expected = np.array(sorted([20, 5, 40, 1]), dtype=np.int32)
        np.testing.assert_array_equal(result.preserved_cols, expected)

    def test_scale_computed_from_unprotected_only(self):
        """Scale must not be inflated by extreme values in protected columns."""
        qtz = _aqtz()
        w = np.full((8, 64), 0.01, dtype=np.float32)
        w[:, 0] = 1_000.0   # enormous outlier in protected column
        result_with_sw = qtz.quantize(w, protected_cols=[0])
        # Scale should reflect the 0.01 background, not the 1000 outlier
        assert result_with_sw.scale < 1.0

    def test_out_of_range_cols_silently_dropped(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[9999, -1, 0])
        assert list(result.preserved_cols) == [0]   # only col 0 is valid

    def test_duplicate_cols_deduped(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[5, 5, 10, 10])
        assert list(result.preserved_cols) == [5, 10]


# ---------------------------------------------------------------------------
# quantize — 1-D and 3-D reshape handling
# ---------------------------------------------------------------------------

class TestReshapeHandling:
    def test_1d_input(self):
        qtz = _aqtz()
        w = RNG.standard_normal(256).astype(np.float32)
        result = qtz.quantize(w)
        assert result.original_shape == (256,)
        assert result.ternary.ndim == 2   # internal: (1, 256)

    def test_3d_input(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 8, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[0, 1])
        assert result.original_shape == (4, 8, 64)

    def test_ternary_values_restricted(self):
        qtz = _aqtz()
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w)
        unique_vals = np.unique(result.ternary)
        assert set(unique_vals.tolist()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# dequantize — round-trip accuracy
# ---------------------------------------------------------------------------

class TestDequantize:
    def test_output_shape_matches_original(self):
        qtz = _aqtz()
        w = RNG.standard_normal((4, 16, 32)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[0])
        recon = qtz.dequantize(result)
        assert recon.shape == w.shape

    def test_output_is_float32(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        result = qtz.quantize(w)
        recon = qtz.dequantize(result)
        assert recon.dtype == np.float32

    def test_protected_cols_reconstructed_exactly_to_fp16(self):
        """Protected columns should be reproduced at FP16 precision."""
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        protected = [3, 7]
        result = qtz.quantize(w, protected_cols=protected)
        recon = qtz.dequantize(result)
        for col in protected:
            expected = w[:, col].astype(np.float16).astype(np.float32)
            np.testing.assert_array_almost_equal(recon[:, col], expected, decimal=3)

    def test_ternary_cols_have_higher_error_than_protected(self):
        """Ternary error should dominate compared to FP16 protected columns."""
        qtz = _aqtz()
        # Large-magnitude weights to exaggerate quantisation error
        w = RNG.standard_normal((32, 128)).astype(np.float32) * 5
        protected = [0, 1, 64, 127]
        result = qtz.quantize(w, protected_cols=protected)
        recon = qtz.dequantize(result)

        protected_err = float(np.abs(recon[:, protected] - w[:, protected]).mean())
        ternary_mask = np.ones(128, dtype=bool)
        for col in protected:
            ternary_mask[col] = False
        ternary_err = float(np.abs(recon[:, ternary_mask] - w[:, ternary_mask]).mean())

        # Protected columns (FP16) should have strictly lower error
        assert protected_err < ternary_err

    def test_snr_protected_cols_near_zero_error(self):
        """Protected columns should have near-zero reconstruction error (FP16)."""
        qtz = _aqtz()
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        protected = [10, 20, 30]
        result = qtz.quantize(w, protected_cols=protected)
        recon = qtz.dequantize(result)

        for col in protected:
            signal_power = float(np.mean(w[:, col] ** 2))
            noise = w[:, col].astype(np.float16).astype(np.float32) - recon[:, col]
            noise_power = float(np.mean(noise ** 2))
            if signal_power > 0 and noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                assert snr_db > 40, f"SNR of protected col {col} too low: {snr_db:.1f} dB"

    def test_no_protected_cols_dequantize(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[])
        recon = qtz.dequantize(result)
        assert recon.shape == w.shape

    def test_1d_round_trip_shape(self):
        qtz = _aqtz()
        w = RNG.standard_normal(128).astype(np.float32)
        result = qtz.quantize(w)
        recon = qtz.dequantize(result)
        assert recon.shape == (128,)

    def test_3d_round_trip_shape(self):
        qtz = _aqtz()
        w = RNG.standard_normal((2, 4, 64)).astype(np.float32)
        result = qtz.quantize(w, protected_cols=[0])
        recon = qtz.dequantize(result)
        assert recon.shape == (2, 4, 64)


# ---------------------------------------------------------------------------
# Integration: full quantize → dequantize pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_all_zero_weight_tensor(self):
        qtz = _aqtz()
        w = np.zeros((8, 64), dtype=np.float32)
        result = qtz.quantize(w, protected_cols=[0])
        recon = qtz.dequantize(result)
        # All-zero input → all-zero ternary → all-zero output (protected col is also 0)
        np.testing.assert_array_equal(recon, np.zeros_like(w))

    def test_binary_weights_no_dead_zone(self):
        """With zero_threshold near 0, almost nothing should be zeroed."""
        qtz = AsymmetricTernaryQuantizer(TernaryConfig(zero_threshold=0.001))
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w)
        n_zeros = int(np.sum(result.ternary == 0))
        total = result.ternary.size
        # Very few zeros expected
        assert n_zeros / total < 0.05

    def test_high_threshold_many_zeros(self):
        """With high zero_threshold, most weights should be zeroed."""
        qtz = AsymmetricTernaryQuantizer(TernaryConfig(zero_threshold=2.0))
        w = RNG.standard_normal((16, 64)).astype(np.float32)
        result = qtz.quantize(w)
        n_zeros = int(np.sum(result.ternary == 0))
        total = result.ternary.size
        assert n_zeros / total > 0.5

    def test_ternary_only_values_in_ternary_cols(self):
        qtz = _aqtz()
        w = RNG.standard_normal((8, 64)).astype(np.float32)
        protected = [5]
        result = qtz.quantize(w, protected_cols=protected)
        tern_cols = [c for c in range(64) if c not in protected]
        assert set(np.unique(result.ternary[:, tern_cols]).tolist()).issubset({-1, 0, 1})
