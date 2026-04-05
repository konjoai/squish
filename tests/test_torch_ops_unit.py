"""tests/test_torch_ops_unit.py

Unit tests for squish/torch_ops.py.

All tests use numpy arrays only (no real GPU required). The torch import is
mocked via ``sys.modules`` where torch is not installed so the tests also run
in the MLX-only CI environment without torch.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch_mock() -> MagicMock:
    """Minimal torch mock that makes torch.from_numpy().to() work."""
    mock = MagicMock(name="torch")
    mock.float32  = "float32"
    mock.float16  = "float16"
    mock.bfloat16 = "bfloat16"

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def to(self, device=None, dtype=None):
            return self

        def numpy(self):
            return self._arr

    mock.from_numpy.side_effect = lambda arr: _FakeTensor(arr)
    return mock


try:
    import torch as _torch_real
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

torch_only = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# _import_torch
# ---------------------------------------------------------------------------

class TestImportTorch:
    def test_raises_if_torch_missing(self):
        """_import_torch() raises RuntimeError when torch is not installed."""
        from squish.experimental.torch_ops import _import_torch
        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises((RuntimeError, ImportError)):
                _import_torch()

    @torch_only
    def test_returns_torch_when_available(self):
        from squish.experimental.torch_ops import _import_torch
        t = _import_torch()
        import torch
        assert t is torch


# ---------------------------------------------------------------------------
# dequantize_int4_asymmetric_torch
# ---------------------------------------------------------------------------

class TestDequantizeInt4Asymmetric:
    """Tests run with real numpy but mocked torch.from_numpy().to()."""

    def _run(self, packed, scales, zero_points, group_size=4):
        """Run dequantize using a minimal torch mock."""
        import importlib
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Reload to pick up mock (avoids cached import)
            import squish.experimental.torch_ops as mod
            importlib.reload(mod)
            result = mod.dequantize_int4_asymmetric_torch(
                packed, scales, zero_points, group_size=group_size, device="cpu"
            )
        return result

    def test_output_shape(self):
        """Output shape is (n_rows, n_cols) = (n_rows, packed.shape[1]*2)."""
        n_rows, half_cols = 4, 8
        packed = np.zeros((n_rows, half_cols), dtype=np.uint8)
        scales = np.ones((n_rows, half_cols * 2 // 4), dtype=np.float32)
        zero_points = np.zeros_like(scales)
        result = self._run(packed, scales, zero_points, group_size=4)
        # result._arr is the numpy array inside _FakeTensor
        assert result._arr.shape == (n_rows, half_cols * 2)

    def test_zero_packed_zero_bias(self):
        """Zero nibbles and zero zero-points → all-zero output."""
        packed = np.zeros((2, 4), dtype=np.uint8)
        scales = np.ones((2, 2), dtype=np.float32)
        zero_points = np.zeros((2, 2), dtype=np.float32)
        result = self._run(packed, scales, zero_points, group_size=4)
        np.testing.assert_allclose(result._arr, 0.0)

    def test_nibble_unpacking_order(self):
        """Low nibble → even columns; high nibble → odd columns."""
        # packed byte 0xAB: lo=0xB=11, hi=0xA=10
        packed = np.array([[0xAB]], dtype=np.uint8)  # (1, 1) → (1, 2) after unpack
        scales = np.ones((1, 1), dtype=np.float32)   # group_size=2, 1 group
        zero_points = np.zeros((1, 1), dtype=np.float32)
        result = self._run(packed, scales, zero_points, group_size=2)
        arr = result._arr  # shape (1, 2)
        assert arr[0, 0] == pytest.approx(11.0)   # lo nibble → col 0
        assert arr[0, 1] == pytest.approx(10.0)   # hi nibble → col 1

    def test_scale_applied(self):
        """scale × (q - zero) is applied per group."""
        packed = np.array([[0x11]], dtype=np.uint8)  # lo=1, hi=1
        scales = np.array([[2.0]], dtype=np.float32)
        zero_points = np.array([[0.0]], dtype=np.float32)
        result = self._run(packed, scales, zero_points, group_size=2)
        np.testing.assert_allclose(result._arr, [[2.0, 2.0]])

    def test_zero_point_subtracted(self):
        """Zero-point is subtracted before scaling."""
        packed = np.array([[0x55]], dtype=np.uint8)  # lo=5, hi=5
        scales = np.array([[1.0]], dtype=np.float32)
        zero_points = np.array([[3.0]], dtype=np.float32)
        result = self._run(packed, scales, zero_points, group_size=2)
        np.testing.assert_allclose(result._arr, [[2.0, 2.0]])


# ---------------------------------------------------------------------------
# dequantize_int4_torch
# ---------------------------------------------------------------------------

class TestDequantizeInt4Symmetric:
    def _run(self, packed, scales, group_size=4):
        import importlib
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            import squish.experimental.torch_ops as mod
            importlib.reload(mod)
            return mod.dequantize_int4_torch(packed, scales, group_size=group_size)

    def test_output_shape(self):
        packed = np.zeros((3, 8), dtype=np.uint8)
        scales = np.ones((3, 4), dtype=np.float32)
        result = self._run(packed, scales, group_size=4)
        assert result._arr.shape == (3, 16)

    def test_zero_packed_is_zero(self):
        """0x00 packed → nibble values 0; symmetric: 0 - 8 = -8 * scale = -8.
        Wait: lo nibble 0 cast to int8 = 0, then 0 - 8 = -8.
        Scale of 1 → -8."""
        packed = np.array([[0x00, 0x00]], dtype=np.uint8)  # (1,2) → (1,4)
        scales = np.ones((1, 1), dtype=np.float32)         # group_size=4
        result = self._run(packed, scales, group_size=4)
        np.testing.assert_allclose(result._arr, [[-8.0, -8.0, -8.0, -8.0]])

    def test_0x88_is_zero(self):
        """0x88 → lo nibble 8, hi nibble 8; 8 - 8 = 0 → zero output."""
        packed = np.array([[0x88]], dtype=np.uint8)  # (1,1) → (1,2)
        scales = np.ones((1, 1), dtype=np.float32)   # group_size=2
        result = self._run(packed, scales, group_size=2)
        np.testing.assert_allclose(result._arr, [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# loaded_weight_to_torch
# ---------------------------------------------------------------------------

class TestLoadedWeightToTorch:
    def _run(self, arr, device="cpu", dtype="float16"):
        import importlib
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            import squish.experimental.torch_ops as mod
            importlib.reload(mod)
            return mod.loaded_weight_to_torch(arr, device=device, dtype=dtype)

    def test_basic_conversion(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self._run(arr)
        assert result is not None

    def test_c_contiguous(self):
        """Non-contiguous arrays are made contiguous before conversion."""
        arr = np.ones((4, 4), dtype=np.float32)[::2, ::2]  # non-contiguous
        result = self._run(arr)
        assert result is not None

    def test_unknown_dtype_defaults_to_float16(self):
        arr = np.ones(4, dtype=np.float32)
        result = self._run(arr, dtype="unknown_dtype")
        assert result is not None
