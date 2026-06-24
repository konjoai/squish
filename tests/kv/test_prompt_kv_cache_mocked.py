"""Host-agnostic coverage for prompt_kv_cache mlx paths via a mocked mlx.core.

The sibling test_prompt_kv_cache_coverage.py importorskips mlx (skips on Linux).
A fake mlx.core whose ``array`` is a real class (so ``isinstance`` and
construction both work) drives _to_numpy, infer_kv_dtype and restore_kv_state.
"""

import sys
import types

import numpy as np
import pytest

from squish.kv import prompt_kv_cache as pkc


class _MxArray:
    def __init__(self, data, dtype="f16"):
        self.data = data
        self.dtype = dtype

    def astype(self, dt):
        return _MxArray(self.data, dt)

    def __array__(self, dtype=None, copy=None):
        if self.dtype == "bf16":  # bf16 has no numpy buffer → forces the f32 fallback
            raise RuntimeError("bf16 buffer mismatch")
        return np.asarray(self.data, dtype=dtype)


def _install_mlx(monkeypatch, with_utils=False):
    mx = types.ModuleType("mlx.core")
    mx.array = _MxArray
    mx.float16, mx.bfloat16, mx.float32 = "f16", "bf16", "f32"
    mx.eval = lambda *a: None
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    if with_utils:
        utils = types.ModuleType("mlx.utils")
        utils.tree_flatten = lambda d: list(d.items()) if isinstance(d, dict) else list(d)
        pkg.utils = utils
        monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    return mx


class _Layer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0


def _entry(n_layers=2, keys=None, values=None, offset=5, lazy_dir=None):
    return types.SimpleNamespace(
        n_layers=n_layers,
        keys=keys if keys is not None else [[1.0], [2.0]],
        values=values if values is not None else [[3.0], [4.0]],
        offset=offset,
        _lazy_kv_dir=lazy_dir,
    )


# ── _to_numpy ──────────────────────────────────────────────────────────────────


def test_to_numpy_ndarray_shortcut():
    assert pkc._to_numpy(np.ones(2, np.float32)).dtype == np.float16


def test_to_numpy_mlx_success_and_bf16_fallback(monkeypatch):
    _install_mlx(monkeypatch)
    assert pkc._to_numpy(_MxArray([1.0, 2.0], "f16")).dtype == np.float16
    assert pkc._to_numpy(_MxArray([1.0, 2.0], "bf16")).dtype == np.float16  # fallback


def test_to_numpy_rejects_non_mlx_type(monkeypatch):
    _install_mlx(monkeypatch)
    with pytest.raises(TypeError, match="expected np.ndarray or mlx"):
        pkc._to_numpy("not an array")


def test_to_numpy_raises_when_mlx_unavailable():
    with pytest.raises(TypeError, match="mlx not available"):
        pkc._to_numpy(object())


# ── infer_kv_dtype ─────────────────────────────────────────────────────────────


def test_infer_kv_dtype_returns_param_dtype(monkeypatch):
    _install_mlx(monkeypatch, with_utils=True)
    model = types.SimpleNamespace(parameters=lambda: {"w": _MxArray([1.0], "bf16")})
    assert pkc.infer_kv_dtype(model) == "bf16"


def test_infer_kv_dtype_defaults_to_float16(monkeypatch):
    _install_mlx(monkeypatch, with_utils=True)
    model = types.SimpleNamespace(parameters=lambda: {"w": _MxArray([1.0], "int8")})
    assert pkc.infer_kv_dtype(model) == "f16"  # no float param → default


# ── restore_kv_state ───────────────────────────────────────────────────────────


def test_restore_none_cache():
    assert pkc.restore_kv_state(None, _entry()) is False


def test_restore_non_list_cache():
    assert pkc.restore_kv_state("not-a-list", _entry()) is False


def test_restore_layer_count_mismatch():
    assert pkc.restore_kv_state([_Layer()], _entry(n_layers=2)) is False


def test_restore_returns_false_without_mlx():
    assert pkc.restore_kv_state([_Layer(), _Layer()], _entry(n_layers=2)) is False


def test_restore_success_and_offset(monkeypatch):
    _install_mlx(monkeypatch)
    cache = [_Layer(), _Layer()]
    assert pkc.restore_kv_state(cache, _entry(n_layers=2, offset=7)) is True
    assert cache[0].keys is not None and cache[0].offset == 7


def test_restore_with_target_dtype(monkeypatch):
    _install_mlx(monkeypatch)
    cache = [_Layer(), _Layer()]
    assert pkc.restore_kv_state(cache, _entry(n_layers=2), target_dtype="f32") is True


def test_restore_unsupported_layer_type(monkeypatch):
    _install_mlx(monkeypatch)
    assert pkc.restore_kv_state([object(), object()], _entry(n_layers=2)) is False


def test_restore_lazy_load_from_disk(monkeypatch, tmp_path):
    _install_mlx(monkeypatch)
    for i in range(2):
        np.save(tmp_path / f"k_{i}.npy", np.ones(2, np.float32))
        np.save(tmp_path / f"v_{i}.npy", np.ones(2, np.float32))
    entry = _entry(n_layers=2, keys=[None, None], values=[None, None], lazy_dir=tmp_path)
    cache = [_Layer(), _Layer()]
    assert pkc.restore_kv_state(cache, entry) is True


def test_restore_lazy_load_failure(monkeypatch, tmp_path):
    _install_mlx(monkeypatch)
    # no .npy files on disk → np.load raises OSError → swallowed → False
    entry = _entry(n_layers=2, keys=[None, None], values=[None, None], lazy_dir=tmp_path)
    assert pkc.restore_kv_state([_Layer(), _Layer()], entry) is False


def test_restore_outer_except_returns_false(monkeypatch):
    _install_mlx(monkeypatch)

    class _BadKeys:
        def __getitem__(self, _i):
            raise ValueError("boom")

    entry = types.SimpleNamespace(
        n_layers=2, keys=_BadKeys(), values=[[1.0], [2.0]], offset=1, _lazy_kv_dir=None
    )
    assert pkc.restore_kv_state([_Layer(), _Layer()], entry) is False


def test_restore_layer_without_offset_attr(monkeypatch):
    _install_mlx(monkeypatch)

    class _LayerNoOffset:
        def __init__(self):
            self.keys = None
            self.values = None

    cache = [_LayerNoOffset(), _LayerNoOffset()]
    assert pkc.restore_kv_state(cache, _entry(n_layers=2)) is True
    assert cache[0].keys is not None  # offset assignment skipped
