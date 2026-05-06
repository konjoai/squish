"""Tests for squish.quant.sqint2_linear (W103.4c)."""
import sys
import numpy as np
import pytest
from squish.quant.sqint2_linear import (
    NF2_CODEBOOK,
    SQINT2LinearNumPy,
    SQINT2LinearMLX,
    _nf2_dequantize_numpy,
    _residual_gemv_numpy,
    get_sqint2_linear_info,
)


# ---------------------------------------------------------------------------
# NF2 codebook
# ---------------------------------------------------------------------------

def test_codebook_has_four_levels():
    assert len(NF2_CODEBOOK) == 4


def test_codebook_values():
    expected = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    np.testing.assert_array_equal(NF2_CODEBOOK, expected)


# ---------------------------------------------------------------------------
# _nf2_dequantize_numpy
# ---------------------------------------------------------------------------

def test_dequantize_all_indices():
    packed = np.array([0, 1, 2, 3], dtype=np.uint8)
    out = _nf2_dequantize_numpy(packed)
    np.testing.assert_array_almost_equal(out, [-1.5, -0.5, 0.5, 1.5])


def test_dequantize_with_scale():
    packed = np.array([2], dtype=np.uint8)  # codebook[2] = 0.5
    out = _nf2_dequantize_numpy(packed, scale=2.0)
    assert abs(out[0] - 1.0) < 1e-6


def test_dequantize_output_float32():
    packed = np.array([0, 1, 2, 3], dtype=np.uint8)
    assert _nf2_dequantize_numpy(packed).dtype == np.float32


def test_dequantize_masks_to_2bit():
    packed = np.array([0xFF], dtype=np.uint8)  # all bits set → index 3
    out = _nf2_dequantize_numpy(packed)
    assert abs(out[0] - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# _residual_gemv_numpy
# ---------------------------------------------------------------------------

def test_residual_gemv_shape():
    x = np.ones((1, 4), dtype=np.float32)
    L = np.ones((8, 2), dtype=np.float32)
    R = np.ones((2, 4), dtype=np.float32)
    out = _residual_gemv_numpy(x, L, R)
    assert out.shape == (1, 8)


def test_residual_gemv_values():
    x = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    L = np.eye(2, dtype=np.float32)
    R = np.ones((2, 4), dtype=np.float32)
    out = _residual_gemv_numpy(x, L, R)
    np.testing.assert_array_almost_equal(out, [[1.0, 1.0]])


# ---------------------------------------------------------------------------
# SQINT2LinearNumPy
# ---------------------------------------------------------------------------

def _make_layer(out=4, in_=8, rank=None):
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 4, (out, in_), dtype=np.uint8)
    L = rng.standard_normal((out, rank)).astype(np.float32) if rank else None
    R = rng.standard_normal((rank, in_)).astype(np.float32) if rank else None
    return SQINT2LinearNumPy(packed, scale=0.1, zero_point=0.0, residual_L=L, residual_R=R)


def test_numpy_layer_forward_shape():
    layer = _make_layer(out=4, in_=8)
    x = np.ones((1, 8), dtype=np.float32)
    assert layer(x).shape == (1, 4)


def test_numpy_layer_weight_property():
    layer = _make_layer(out=4, in_=8)
    assert layer.weight.shape == (4, 8)
    assert layer.weight.dtype == np.float32


def test_numpy_layer_with_residual():
    layer = _make_layer(out=4, in_=8, rank=2)
    x = np.ones((3, 8), dtype=np.float32)
    out = layer(x)
    assert out.shape == (3, 4)


def test_numpy_layer_no_residual():
    layer = _make_layer(out=4, in_=8, rank=None)
    x = np.ones((2, 8), dtype=np.float32)
    assert layer(x).shape == (2, 4)


# ---------------------------------------------------------------------------
# SQINT2LinearMLX
# ---------------------------------------------------------------------------

def test_mlx_raises_on_non_darwin():
    if sys.platform == "darwin":
        pytest.skip("On Darwin MLX may be available")
    with pytest.raises(ImportError):
        SQINT2LinearMLX(np.zeros((4, 8), dtype=np.uint8))


# ---------------------------------------------------------------------------
# get_sqint2_linear_info
# ---------------------------------------------------------------------------

def test_info_numpy_always_true():
    info = get_sqint2_linear_info()
    assert info["sqint2_linear_numpy"] is True


def test_info_has_platform():
    info = get_sqint2_linear_info()
    assert "platform" in info


def test_info_mlx_false_on_linux():
    if sys.platform == "darwin":
        pytest.skip("platform is darwin")
    info = get_sqint2_linear_info()
    assert info["sqint2_linear_mlx"] is False
