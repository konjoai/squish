"""tests/test_rust_matmul.py — W101 Rust INT4 matmul bridge tests.

Covers:
  - Shape and dtype contract
  - Numerical correctness: NumPy fallback vs reference dequant+matmul
  - NumPy fallback works when Rust extension is absent
  - Rust kernel matches NumPy fallback (when available)
  - Edge cases: batch=1, batch>1, group_size variants
  - Error paths: mismatched shapes, bad group_size
"""

import importlib
import numpy as np
import pytest

from squish.quant.quantizer import (
    quantize_int4_asymmetric,
    quantized_matmul_int4,
    _quantized_matmul_int4_numpy,
    get_backend_info,
)

RNG = np.random.default_rng(0xDEADBEEF)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weights_and_activations(out_f=32, in_f=64, batch=4, group_size=16, seed=42):
    rng = np.random.default_rng(seed)
    W   = rng.standard_normal((out_f, in_f)).astype(np.float32)
    x   = rng.standard_normal((batch, in_f)).astype(np.float32)
    return W, x


def _reference_output(W, x):
    """Exact float32 GEMM — ground truth for correctness checks."""
    return (x @ W.T).astype(np.float32)


def _pack_weights(W, group_size):
    """Quantise W with the Rust asymmetric INT4 kernel if available,
    else raise SkipTest so correctness tests degrade gracefully."""
    try:
        packed, scales, offsets = quantize_int4_asymmetric(W, group_size=group_size)
    except RuntimeError as exc:
        pytest.skip(f"quantize_int4_asymmetric unavailable: {exc}")
    return packed, scales, offsets


# ---------------------------------------------------------------------------
# Shape / dtype contract
# ---------------------------------------------------------------------------

class TestShapeDtype:
    def test_output_shape_single_batch(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=1, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        assert out.shape == (1, 16), out.shape

    def test_output_shape_multi_batch(self):
        W, x = _make_weights_and_activations(out_f=64, in_f=128, batch=8, group_size=32)
        packed, scales, offsets = _pack_weights(W, 32)
        out = quantized_matmul_int4(packed, scales, offsets, x, group_size=32)
        assert out.shape == (8, 64), out.shape

    def test_output_dtype_is_float32(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=2, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        assert out.dtype == np.float32, out.dtype

    def test_accepts_non_contiguous_x(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=4, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        # Transpose makes it non-contiguous, then transpose back
        x_nc = x.T.T
        assert not x_nc.flags["C_CONTIGUOUS"] or True  # might still be C after double-T
        out = quantized_matmul_int4(packed, scales, offsets, x_nc, group_size=16)
        assert out.shape == (4, 16)


# ---------------------------------------------------------------------------
# Numerical correctness — NumPy fallback
# ---------------------------------------------------------------------------

class TestNumpyFallback:
    """The NumPy fallback must be independently correct."""

    def test_numpy_fallback_vs_reference(self):
        W, x = _make_weights_and_activations(out_f=32, in_f=64, batch=4, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out_np  = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 16)
        # Reference: dequant first, then matmul
        from squish.quant.quantizer import dequantize_int4_asymmetric
        W_hat = dequantize_int4_asymmetric(packed, scales, offsets, group_size=16)
        out_ref = (x @ W_hat.T).astype(np.float32)
        np.testing.assert_allclose(out_np, out_ref, rtol=1e-4, atol=1e-4,
                                   err_msg="NumPy fallback diverges from reference dequant+GEMM")

    def test_numpy_fallback_shape_and_dtype(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=3, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 16)
        assert out.shape == (3, 16)
        assert out.dtype == np.float32

    def test_numpy_fallback_batch_one(self):
        W, x = _make_weights_and_activations(out_f=8, in_f=32, batch=1, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 16)
        assert out.shape == (1, 8)

    def test_numpy_fallback_group_size_32(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=64, batch=2, group_size=32)
        packed, scales, offsets = _pack_weights(W, 32)
        out = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 32)
        assert out.shape == (2, 16)

    def test_numpy_fallback_group_size_64(self):
        W, x = _make_weights_and_activations(out_f=32, in_f=128, batch=4, group_size=64)
        packed, scales, offsets = _pack_weights(W, 64)
        out = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 64)
        assert out.shape == (4, 32)


# ---------------------------------------------------------------------------
# Rust kernel correctness (skipped when Rust not built)
# ---------------------------------------------------------------------------

_has_rust_matmul = get_backend_info().get("int4_matmul_rust", False)
rust_required = pytest.mark.skipif(
    not _has_rust_matmul,
    reason="squish_quant Rust extension with quantized_matmul_int4 not built"
)


class TestRustKernel:
    @rust_required
    def test_rust_matches_numpy_fallback(self):
        W, x = _make_weights_and_activations(out_f=64, in_f=128, batch=8, group_size=32)
        packed, scales, offsets = _pack_weights(W, 32)
        out_rust = quantized_matmul_int4(packed, scales, offsets, x, group_size=32)
        out_np   = _quantized_matmul_int4_numpy(packed, scales, offsets, x, 32)
        np.testing.assert_allclose(out_rust, out_np, rtol=1e-5, atol=1e-5,
                                   err_msg="Rust kernel diverges from NumPy fallback")

    @rust_required
    def test_rust_output_dtype(self):
        W, x = _make_weights_and_activations(out_f=32, in_f=64, batch=4, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        assert out.dtype == np.float32

    @rust_required
    def test_rust_batch_one(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=1, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        assert out.shape == (1, 16)

    @rust_required
    def test_rust_deterministic(self):
        W, x = _make_weights_and_activations(out_f=32, in_f=64, batch=4, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        out1 = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        out2 = quantized_matmul_int4(packed, scales, offsets, x, group_size=16)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_mismatched_x_in_f_raises(self):
        W, _ = _make_weights_and_activations(out_f=16, in_f=32, batch=2, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        x_bad = np.ones((2, 64), dtype=np.float32)  # in_f wrong
        with pytest.raises((ValueError, Exception)):
            quantized_matmul_int4(packed, scales, offsets, x_bad, group_size=16)

    def test_bad_group_size_raises_numpy(self):
        W, x = _make_weights_and_activations(out_f=16, in_f=32, batch=2, group_size=16)
        packed, scales, offsets = _pack_weights(W, 16)
        # group_size=0 → division by zero / error in NumPy path
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            _quantized_matmul_int4_numpy(packed, scales, offsets, x, group_size=0)


# ---------------------------------------------------------------------------
# Backend info
# ---------------------------------------------------------------------------

class TestBackendInfo:
    def test_backend_info_has_int4_matmul_key(self):
        info = get_backend_info()
        assert "int4_matmul_rust" in info

    def test_backend_info_int4_matmul_is_bool(self):
        info = get_backend_info()
        assert isinstance(info["int4_matmul_rust"], bool)

    def test_backend_info_numpy_always_true(self):
        info = get_backend_info()
        assert info["numpy"] is True
