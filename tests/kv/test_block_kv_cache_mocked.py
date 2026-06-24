"""Host-agnostic coverage for block_kv_cache mlx paths via a mocked mlx.core.

The sibling test_block_kv_cache_coverage.py importorskips mlx (skips on the Linux
coverage runner). These inject a fake mlx.core so the dtype-coercion helpers and
restore_blocks_to_cache run anywhere.
"""

import sys
import types

import numpy as np

from squish.kv import block_kv_cache as bkc


def _install_fake_mlx(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.float32 = "f32"
    mx.eval = lambda *a: None

    class _A:
        def __init__(self, data):
            self.data = data

        def astype(self, dt):
            return _A(("cast", dt, self.data))

    mx.array = lambda data: _A(data)
    mx.concatenate = lambda arrs, axis: _A(("concat", axis, tuple(a.data for a in arrs)))
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


class _Block:
    def __init__(self, n_layers, n_tokens):
        self.n_layers = n_layers
        self.n_tokens = n_tokens
        self.keys = [f"k{i}" for i in range(n_layers)]
        self.values = [f"v{i}" for i in range(n_layers)]


class _Layer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0


def test_to_numpy_f16_ndarray_list_and_astype_fallback(monkeypatch):
    _install_fake_mlx(monkeypatch)
    assert bkc._to_numpy_f16(np.ones(2, np.float32)).dtype == np.float16  # ndarray
    assert bkc._to_numpy_f16([1.0, 2.0]).dtype == np.float16  # np.array success
    fallback = types.SimpleNamespace(astype=lambda dt: [1.0, 2.0])  # forces except branch
    assert bkc._to_numpy_f16(fallback).dtype == np.float16


def test_to_numpy_f32_ndarray_list_and_astype_fallback(monkeypatch):
    _install_fake_mlx(monkeypatch)
    assert bkc._to_numpy_f32(np.ones(2, np.float16)).dtype == np.float32
    assert bkc._to_numpy_f32([1.0, 2.0]).dtype == np.float32
    fallback = types.SimpleNamespace(astype=lambda dt: [1.0, 2.0])
    assert bkc._to_numpy_f32(fallback).dtype == np.float32


def test_restore_blocks_success_without_dtype(monkeypatch):
    _install_fake_mlx(monkeypatch)
    cache = [_Layer(), _Layer()]
    out = bkc.restore_blocks_to_cache(cache, [_Block(2, 3)], target_dtype=None)
    assert out == (2, 3)
    assert cache[0].keys is not None and cache[0].offset == 3


def test_restore_blocks_success_with_target_dtype(monkeypatch):
    _install_fake_mlx(monkeypatch)
    cache = [_Layer(), _Layer()]
    out = bkc.restore_blocks_to_cache(cache, [_Block(2, 5)], target_dtype="f16")
    assert out == (2, 5)


def test_restore_blocks_layer_mismatch_returns_none(monkeypatch):
    _install_fake_mlx(monkeypatch)
    # cache has 1 layer, block claims 2 → mismatch warning branch
    assert bkc.restore_blocks_to_cache([_Layer()], [_Block(2, 1)]) is None


def test_restore_blocks_unsupported_layer_returns_none(monkeypatch):
    _install_fake_mlx(monkeypatch)
    # layer object lacks keys/values attrs → unsupported cache type
    assert bkc.restore_blocks_to_cache([object()], [_Block(1, 1)]) is None


def test_to_numpy_import_error_raises():
    # No fake mlx installed → `import mlx.core` raises, so both helpers re-raise as
    # TypeError (a non-ndarray value can't be converted without mlx).
    import pytest

    with pytest.raises(TypeError, match="mlx not available"):
        bkc._to_numpy_f16([1.0, 2.0])
    with pytest.raises(TypeError, match="mlx not available"):
        bkc._to_numpy_f32([1.0, 2.0])


def test_restore_blocks_layer_without_offset_attr(monkeypatch):
    _install_fake_mlx(monkeypatch)

    class _LayerNoOffset:
        def __init__(self):
            self.keys = None
            self.values = None

    cache = [_LayerNoOffset(), _LayerNoOffset()]
    out = bkc.restore_blocks_to_cache(cache, [_Block(2, 4)])
    assert out == (2, 4)
    assert cache[0].keys is not None  # offset branch skipped, keys still set
