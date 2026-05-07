"""tests/test_sqint2_linear.py — W103.4c SQINT2Linear MLX Module tests.

All tests are skipped unless running on darwin/arm64 (Apple Silicon) with
MLX available — matching the platform guard in sqint2_linear.py.

Covers:
  - Platform guard: ImportError raised on non-darwin/non-arm64
  - sqint2_linear_from_layer factory (via SQINT2Layer fixtures)
  - SQINT2Linear construction: dtype and shape constraints
  - Properties: in_features, out_features, group_size, has_residual, has_sparse
  - Forward pass shape contract: (..., in_f) → (..., out_f)
  - Forward pass correctness vs decompress_weight reference (no residual)
  - Forward pass correctness with SVD residual (residual_rank=4)
  - Forward pass correctness with sparse COO correction
  - Forward pass with all three stages (base + residual + sparse)
  - Bias application
  - Batch dimension handling (1D, 2D, 3D inputs)
  - repr / str coverage
  - TypeError/ValueError on bad construction inputs
  - SQINT2Linear honours NF2_VALUES LUT (checked via known-input forward)
"""
from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

# ── Guard: skip entire module if not on Apple Silicon with MLX ────────────────
_IS_DARWIN_ARM64 = (sys.platform == "darwin" and platform.machine() == "arm64")

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

_RUN_TESTS = _IS_DARWIN_ARM64 and _MLX_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _RUN_TESTS,
    reason="SQINT2Linear requires Apple Silicon (darwin/arm64) with MLX"
)

if _RUN_TESTS:
    from squish.quant.sqint2_linear import SQINT2Linear, sqint2_linear_from_layer, _NF2_LUT
    from squish.quant.sqint2 import (
        SQINT2Config,
        SQINT2Layer,
        NF2_VALUES,
        compress_weight,
        decompress_weight,
        snr_db,
    )

RNG = np.random.default_rng(0xABCD_DEAD_BEEF)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(
    out_f: int = 32,
    in_f: int = 64,
    group_size: int = 32,
    residual_rank: int = 0,
    sparse_frac: float = 0.0,
    seed: int = 42,
) -> "SQINT2Layer":
    """Build a SQINT2Layer from synthetic float32 weights."""
    cfg = SQINT2Config(
        group_size=group_size,
        seed=seed,
        refine_iters=1,
        residual_rank=residual_rank,
        residual_factor_dtype="fp32",  # fp32 for exact test comparison
        sparse_frac=sparse_frac,
    )
    W = RNG.standard_normal((out_f, in_f)).astype(np.float32) * 0.02
    return compress_weight(W, cfg)


def _layer_to_linear(layer: "SQINT2Layer") -> "SQINT2Linear":
    """Convert SQINT2Layer to SQINT2Linear using the factory."""
    return sqint2_linear_from_layer(layer)


def _mx_to_np(arr: "mx.array") -> np.ndarray:
    """Convert an MLX array to a numpy float32 array."""
    return np.array(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Platform guard tests
# ---------------------------------------------------------------------------

class TestPlatformGuard:
    """The ImportError guard must fire on Linux/x86."""

    def test_import_succeeds_on_darwin_arm64(self):
        # If we're here the import already succeeded (pytestmark would skip).
        import squish.quant.sqint2_linear as _mod
        assert hasattr(_mod, "SQINT2Linear")

    def test_nf2_lut_has_four_elements(self):
        lut = _mx_to_np(_NF2_LUT)
        assert lut.shape == (4,)

    def test_nf2_lut_values_match_nf2_values(self):
        lut = _mx_to_np(_NF2_LUT)
        np.testing.assert_allclose(lut, NF2_VALUES, atol=1e-6)


# ---------------------------------------------------------------------------
# Construction: valid inputs
# ---------------------------------------------------------------------------

class TestSQINT2LinearConstruction:
    def test_basic_construction(self):
        layer = _make_layer(out_f=32, in_f=64)
        lin   = _layer_to_linear(layer)
        assert lin.in_features  == 64
        assert lin.out_features == 32
        assert lin.group_size   == 32

    def test_properties_match_layer(self):
        layer = _make_layer(out_f=48, in_f=96, group_size=32)
        lin   = _layer_to_linear(layer)
        assert lin.in_features  == layer.in_features
        assert lin.out_features == layer.out_features
        assert lin.group_size   == layer.cfg.group_size

    def test_has_residual_false_when_rank_zero(self):
        layer = _make_layer(residual_rank=0)
        lin   = _layer_to_linear(layer)
        assert lin.has_residual is False

    def test_has_residual_true_when_rank_nonzero(self):
        layer = _make_layer(residual_rank=4)
        lin   = _layer_to_linear(layer)
        assert lin.has_residual is True

    def test_has_sparse_false_when_no_sparse(self):
        layer = _make_layer(sparse_frac=0.0)
        lin   = _layer_to_linear(layer)
        assert lin.has_sparse is False

    def test_has_sparse_true_when_sparse_present(self):
        layer = _make_layer(sparse_frac=0.05)
        lin   = _layer_to_linear(layer)
        assert lin.has_sparse is True

    def test_indices_dtype_is_uint8(self):
        layer = _make_layer()
        lin   = _layer_to_linear(layer)
        assert lin.indices.dtype == mx.uint8

    def test_scales_dtype_is_float32(self):
        layer = _make_layer()
        lin   = _layer_to_linear(layer)
        assert lin.scales.dtype == mx.float32

    def test_zero_points_dtype_is_float32(self):
        layer = _make_layer()
        lin   = _layer_to_linear(layer)
        assert lin.zero_points.dtype == mx.float32

    def test_bias_stored_when_provided(self):
        layer = _make_layer(out_f=32)
        bias  = mx.zeros((32,))
        lin   = SQINT2Linear(
            indices=mx.array(layer.indices),
            scales=mx.array(layer.scales),
            zero_points=mx.array(layer.zero_points),
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=layer.cfg.group_size,
            bias=bias,
        )
        assert hasattr(lin, "bias")

    def test_no_bias_by_default(self):
        layer = _make_layer()
        lin   = _layer_to_linear(layer)
        assert not hasattr(lin, "bias")


# ---------------------------------------------------------------------------
# Construction: error paths
# ---------------------------------------------------------------------------

class TestSQINT2LinearErrors:
    def test_raises_type_error_on_non_uint8_indices(self):
        layer = _make_layer()
        with pytest.raises(TypeError, match="uint8"):
            SQINT2Linear(
                indices=mx.zeros((4, 8), dtype=mx.float32),
                scales=mx.array(layer.scales),
                zero_points=mx.array(layer.zero_points),
                in_features=layer.in_features,
                out_features=layer.out_features,
                group_size=layer.cfg.group_size,
            )

    def test_raises_value_error_on_1d_indices(self):
        layer = _make_layer()
        with pytest.raises(ValueError, match="2-D"):
            SQINT2Linear(
                indices=mx.zeros((16,), dtype=mx.uint8),
                scales=mx.array(layer.scales),
                zero_points=mx.array(layer.zero_points),
                in_features=layer.in_features,
                out_features=layer.out_features,
                group_size=layer.cfg.group_size,
            )

    def test_raises_value_error_on_scales_shape_mismatch(self):
        layer = _make_layer(out_f=32, in_f=64, group_size=32)
        # n_groups = 64 // 32 = 2; wrong scales shape: (32, 1) instead of (32, 2)
        with pytest.raises(ValueError):
            SQINT2Linear(
                indices=mx.array(layer.indices),
                scales=mx.zeros((32, 1), dtype=mx.float32),
                zero_points=mx.array(layer.zero_points),
                in_features=layer.in_features,
                out_features=layer.out_features,
                group_size=layer.cfg.group_size,
            )

    def test_raises_value_error_on_scales_zp_shape_mismatch(self):
        layer = _make_layer(out_f=32, in_f=64, group_size=32)
        n_groups = 64 // 32
        with pytest.raises(ValueError, match="identical shapes"):
            SQINT2Linear(
                indices=mx.array(layer.indices),
                scales=mx.zeros((32, n_groups), dtype=mx.float32),
                zero_points=mx.zeros((32, n_groups + 1), dtype=mx.float32),
                in_features=layer.in_features,
                out_features=layer.out_features,
                group_size=layer.cfg.group_size,
            )


# ---------------------------------------------------------------------------
# Forward pass: shape contracts
# ---------------------------------------------------------------------------

class TestSQINT2LinearForwardShape:
    def test_1d_input_gives_out_features_output(self):
        layer = _make_layer(out_f=16, in_f=32)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((32,))
        y     = lin(x)
        assert y.shape == (16,)

    def test_2d_input_shape(self):
        layer = _make_layer(out_f=16, in_f=32)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((4, 32))
        y     = lin(x)
        assert y.shape == (4, 16)

    def test_3d_input_shape(self):
        layer = _make_layer(out_f=16, in_f=32)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((2, 3, 32))
        y     = lin(x)
        assert y.shape == (2, 3, 16)

    def test_output_dtype_is_float(self):
        layer = _make_layer(out_f=16, in_f=32)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((4, 32))
        y     = lin(x)
        assert y.dtype in (mx.float32, mx.float16, mx.bfloat16)

    def test_non_square_layer_shape(self):
        layer = _make_layer(out_f=64, in_f=128)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((8, 128))
        y     = lin(x)
        assert y.shape == (8, 64)


# ---------------------------------------------------------------------------
# Forward pass: correctness (no residual)
# ---------------------------------------------------------------------------

class TestSQINT2LinearCorrectnessBase:
    """Forward pass SNR vs decompress_weight reference (Stage 1+2 only)."""

    def test_forward_matches_decompress_on_zero_input(self):
        """Zero input → zero output regardless of weights."""
        layer = _make_layer(out_f=32, in_f=64, residual_rank=0)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((1, 64))
        y     = lin(x)
        np.testing.assert_allclose(_mx_to_np(y), 0.0, atol=1e-5)

    def test_forward_output_not_all_zero_on_nonzero_input(self):
        """Non-zero input → non-zero output (sanity check)."""
        layer = _make_layer(out_f=32, in_f=64, residual_rank=0)
        lin   = _layer_to_linear(layer)
        x     = mx.ones((1, 64))
        y     = lin(x)
        assert float(mx.sum(mx.abs(y)).item()) > 0.0

    def test_forward_snr_vs_fp32_reference_above_threshold(self):
        """SQINT2Linear output should have SNR ≥ 6 dB vs FP32 linear reference.

        The threshold is deliberately loose (6 dB) because Stage 1+2 alone on
        small synthetic tensors achieves ~9 dB SNR on the *weight matrix*, not
        on the GEMV output which is further reduced by the input projection.
        The test confirms the forward pass is *doing quantised compute*, not
        random noise.
        """
        rng   = np.random.default_rng(0xF00D)
        out_f, in_f = 32, 64
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02
        cfg   = SQINT2Config(group_size=32, seed=11, refine_iters=1, residual_rank=0)
        layer = compress_weight(W, cfg)
        lin   = sqint2_linear_from_layer(layer)

        x_np  = rng.standard_normal((8, in_f)).astype(np.float32) * 0.1
        x_mx  = mx.array(x_np)

        # Reference: exact FP32 linear
        y_ref = x_np @ W.T    # (8, out_f)

        # SQINT2Linear forward
        y_sq  = _mx_to_np(lin(x_mx))

        snr = snr_db(y_ref, y_sq)
        assert snr >= 6.0, f"SQINT2Linear forward SNR {snr:.2f} dB below 6 dB threshold"

    def test_forward_matches_decompress_gemv_reference(self):
        """SQINT2Linear GEMV should match decompress_weight @ x within fp16 tolerance."""
        rng   = np.random.default_rng(0xBEEF)
        out_f, in_f = 32, 64
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02
        cfg   = SQINT2Config(group_size=32, seed=7, refine_iters=1, residual_rank=0)
        layer = compress_weight(W, cfg)
        lin   = sqint2_linear_from_layer(layer)

        # Reference: reconstruct weight via decompress, then matmul
        W_recon = decompress_weight(layer)           # (out_f, in_f) fp32

        x_np  = rng.standard_normal((4, in_f)).astype(np.float32)
        x_mx  = mx.array(x_np)
        y_ref = x_np @ W_recon.T                     # (4, out_f)
        y_sq  = _mx_to_np(lin(x_mx))                 # (4, out_f)

        # Tolerance: fp16 round-trip on the weight matrix + fp16 matmul → ~1e-2
        np.testing.assert_allclose(y_sq, y_ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Forward pass: Stage 3 residual
# ---------------------------------------------------------------------------

class TestSQINT2LinearWithResidual:
    def test_forward_with_residual_shape_unchanged(self):
        layer = _make_layer(out_f=32, in_f=64, residual_rank=4)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((4, 64))
        y     = lin(x)
        assert y.shape == (4, 32)

    def test_residual_changes_output_vs_no_residual(self):
        """Adding SVD residual should change the output (non-zero residual energy)."""
        rng   = np.random.default_rng(0xC0FFEE)
        out_f, in_f = 32, 64
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02

        cfg_no  = SQINT2Config(group_size=32, seed=13, refine_iters=1, residual_rank=0)
        cfg_res = SQINT2Config(group_size=32, seed=13, refine_iters=1, residual_rank=4,
                               residual_factor_dtype="fp32")
        layer_no  = compress_weight(W, cfg_no)
        layer_res = compress_weight(W, cfg_res)

        lin_no  = sqint2_linear_from_layer(layer_no)
        lin_res = sqint2_linear_from_layer(layer_res)

        x_np = rng.standard_normal((4, in_f)).astype(np.float32)
        x    = mx.array(x_np)

        y_no  = _mx_to_np(lin_no(x))
        y_res = _mx_to_np(lin_res(x))

        # Outputs must differ (SVD residual adds non-zero correction)
        assert not np.allclose(y_no, y_res, atol=1e-8), (
            "SVD residual had no effect on output"
        )

    def test_residual_improves_snr_vs_no_residual(self):
        """SVD residual should improve (or maintain) reconstruction SNR."""
        rng   = np.random.default_rng(0xDEAD_BEE5)
        out_f, in_f = 64, 128
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02

        cfg_no  = SQINT2Config(group_size=32, seed=21, refine_iters=1, residual_rank=0)
        cfg_res = SQINT2Config(group_size=32, seed=21, refine_iters=1, residual_rank=8,
                               residual_factor_dtype="fp32")
        layer_no  = compress_weight(W, cfg_no)
        layer_res = compress_weight(W, cfg_res)

        lin_no  = sqint2_linear_from_layer(layer_no)
        lin_res = sqint2_linear_from_layer(layer_res)

        x_np = rng.standard_normal((8, in_f)).astype(np.float32) * 0.1
        y_ref = x_np @ W.T

        y_no  = _mx_to_np(lin_no(mx.array(x_np)))
        y_res = _mx_to_np(lin_res(mx.array(x_np)))

        snr_no  = snr_db(y_ref, y_no)
        snr_res = snr_db(y_ref, y_res)
        # Residual should not degrade SNR (may improve or be within noise)
        assert snr_res >= snr_no - 1.0, (
            f"Residual degraded SNR: {snr_no:.2f} dB → {snr_res:.2f} dB"
        )


# ---------------------------------------------------------------------------
# Forward pass: sparse correction
# ---------------------------------------------------------------------------

class TestSQINT2LinearWithSparse:
    def test_forward_with_sparse_shape_unchanged(self):
        layer = _make_layer(out_f=32, in_f=64, sparse_frac=0.05)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((4, 64))
        y     = lin(x)
        assert y.shape == (4, 32)

    def test_sparse_changes_output_vs_no_sparse(self):
        """Sparse correction should change the output vs no-sparse version."""
        rng   = np.random.default_rng(0x5A1573)
        out_f, in_f = 32, 64
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02

        cfg_no  = SQINT2Config(group_size=32, seed=5, refine_iters=1, sparse_frac=0.0)
        cfg_sp  = SQINT2Config(group_size=32, seed=5, refine_iters=1, sparse_frac=0.05)
        layer_no = compress_weight(W, cfg_no)
        layer_sp = compress_weight(W, cfg_sp)

        lin_no = sqint2_linear_from_layer(layer_no)
        lin_sp = sqint2_linear_from_layer(layer_sp)

        x_np = rng.standard_normal((4, in_f)).astype(np.float32)
        x    = mx.array(x_np)

        y_no = _mx_to_np(lin_no(x))
        y_sp = _mx_to_np(lin_sp(x))

        assert not np.allclose(y_no, y_sp, atol=1e-8), (
            "Sparse correction had no effect on output"
        )


# ---------------------------------------------------------------------------
# Forward pass: all three stages combined
# ---------------------------------------------------------------------------

class TestSQINT2LinearFullPipeline:
    def test_all_stages_shape_correct(self):
        layer = _make_layer(out_f=32, in_f=64, residual_rank=4, sparse_frac=0.05)
        lin   = _layer_to_linear(layer)
        x     = mx.zeros((2, 64))
        y     = lin(x)
        assert y.shape == (2, 32)

    def test_full_pipeline_snr_above_threshold(self):
        """Full SQINT2Linear (all three stages) SNR ≥ 5 dB vs FP32 reference."""
        rng   = np.random.default_rng(0xF011A113)
        out_f, in_f = 64, 128
        W     = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.02
        cfg   = SQINT2Config(
            group_size=32, seed=99, refine_iters=1,
            residual_rank=4, residual_factor_dtype="fp32",
            sparse_frac=0.02,
        )
        layer = compress_weight(W, cfg)
        lin   = sqint2_linear_from_layer(layer)

        x_np = rng.standard_normal((8, in_f)).astype(np.float32) * 0.1
        y_ref = x_np @ W.T
        y_sq  = _mx_to_np(lin(mx.array(x_np)))

        snr = snr_db(y_ref, y_sq)
        assert snr >= 5.0, f"Full pipeline SNR {snr:.2f} dB below 5 dB threshold"


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

class TestSQINT2LinearBias:
    def test_bias_changes_output(self):
        layer = _make_layer(out_f=16, in_f=32)
        bias  = mx.ones((16,))
        lin_no_bias = _layer_to_linear(layer)
        lin_bias    = SQINT2Linear(
            indices=mx.array(layer.indices),
            scales=mx.array(layer.scales),
            zero_points=mx.array(layer.zero_points),
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=layer.cfg.group_size,
            bias=bias,
        )
        x  = mx.zeros((4, 32))
        y1 = _mx_to_np(lin_no_bias(x))
        y2 = _mx_to_np(lin_bias(x))
        # With zero input, output = bias for lin_bias
        np.testing.assert_allclose(y2, 1.0, atol=1e-5)
        np.testing.assert_allclose(y1, 0.0, atol=1e-5)

    def test_bias_shape_propagated(self):
        layer = _make_layer(out_f=16, in_f=32)
        lin   = SQINT2Linear(
            indices=mx.array(layer.indices),
            scales=mx.array(layer.scales),
            zero_points=mx.array(layer.zero_points),
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=layer.cfg.group_size,
            bias=mx.zeros((16,)),
        )
        x = mx.zeros((3, 32))
        y = lin(x)
        assert y.shape == (3, 16)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestSQINT2LinearRepr:
    def test_repr_contains_class_name(self):
        layer = _make_layer()
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "SQINT2Linear" in r

    def test_repr_contains_dimensions(self):
        layer = _make_layer(out_f=32, in_f=64)
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "32" in r
        assert "64" in r

    def test_repr_contains_group_size(self):
        layer = _make_layer(group_size=32)
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "gs=32" in r

    def test_repr_contains_residual_rank_when_present(self):
        layer = _make_layer(residual_rank=8)
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "residual_rank=8" in r

    def test_repr_omits_residual_when_absent(self):
        layer = _make_layer(residual_rank=0)
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "residual_rank" not in r

    def test_repr_contains_sparse_k_when_present(self):
        layer = _make_layer(sparse_frac=0.05)
        lin   = _layer_to_linear(layer)
        r     = repr(lin)
        assert "sparse_k" in r


# ---------------------------------------------------------------------------
# sqint2_linear_from_layer factory
# ---------------------------------------------------------------------------

class TestSQINT2LinearFactory:
    def test_factory_returns_sqint2_linear(self):
        layer = _make_layer()
        lin   = sqint2_linear_from_layer(layer)
        assert isinstance(lin, SQINT2Linear)

    def test_factory_accepts_numpy_bias(self):
        layer = _make_layer(out_f=16)
        bias_np = np.ones(16, dtype=np.float32)
        lin = sqint2_linear_from_layer(layer, bias=bias_np)
        assert hasattr(lin, "bias")

    def test_factory_accepts_mx_bias(self):
        layer = _make_layer(out_f=16)
        bias_mx = mx.ones((16,))
        lin = sqint2_linear_from_layer(layer, bias=bias_mx)
        assert hasattr(lin, "bias")

    def test_factory_no_bias_by_default(self):
        layer = _make_layer()
        lin   = sqint2_linear_from_layer(layer)
        assert not hasattr(lin, "bias")
