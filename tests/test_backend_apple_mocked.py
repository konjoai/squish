"""_AppleBackend coverage on non-Apple machines via a mocked ``mlx.core``.

The real-MLX _AppleBackend tests in test_backend_unit.py are gated behind
``@mlx_only`` and skip on the Linux coverage runner, leaving the Apple delegation
paths uncovered there. These tests inject a minimal fake ``mlx.core`` (mirroring
how _TorchBackend is covered with a mocked torch) so the same logic is exercised
in any environment.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture()
def fake_mx(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.int32, mx.float32, mx.float16, mx.bfloat16 = "i32", "f32", "f16", "bf16"
    mx._array_calls = []
    mx._evaled = []

    def _array(data, dtype=None):
        mx._array_calls.append((data, dtype))
        return ("ARR", data, dtype)

    mx.array = _array
    mx.eval = lambda *tensors: mx._evaled.extend(tensors)
    mx.save_safetensors = lambda path, d: setattr(mx, "_saved", (path, d))
    mx.load = lambda path: {"w": ("LOADED", path)}
    mx.metal = types.SimpleNamespace(
        set_memory_limit=lambda *a, **k: setattr(mx, "_memlimit", (a, k))
    )

    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


def _apple():
    from squish.backend import _AppleBackend

    return _AppleBackend()


def test_array_maps_known_and_unknown_dtypes(fake_mx):
    be = _apple()
    assert be.array([1, 2], "bfloat16") == ("ARR", [1, 2], "bf16")
    # unknown dtype falls back to int32
    be.array([1], "weird")
    assert fake_mx._array_calls[-1] == ([1], "i32")


def test_eval_skips_none(fake_mx):
    be = _apple()
    a, b = object(), object()
    be.eval(a, None, b)
    assert fake_mx._evaled == [a, b]


def test_to_numpy_casts_via_mx_float32(fake_mx):
    be = _apple()
    tensor = types.SimpleNamespace(astype=lambda dt: np.array([1.0, 2.0], dtype=np.float32))
    out = be.to_numpy(tensor)
    assert out.dtype == np.float32 and list(out) == [1.0, 2.0]


def test_forward_with_and_without_cache(fake_mx):
    be = _apple()
    model = lambda ids, cache=None: {"ids": ids, "cache": cache}  # noqa: E731
    assert be.forward(model, [1, 2]) == {"ids": [1, 2], "cache": None}
    assert be.forward(model, [1], cache="C") == {"ids": [1], "cache": "C"}


def test_forward_np_evaluates_and_returns_numpy(fake_mx):
    be = _apple()
    out_tensor = types.SimpleNamespace(astype=lambda dt: np.array([0.5], dtype=np.float32))
    model = lambda ids, cache=None: out_tensor  # noqa: E731
    out = be.forward_np(model, [1])
    assert out.dtype == np.float32 and out[0] == np.float32(0.5)
    assert out_tensor in fake_mx._evaled


def test_save_and_load_tensors_delegate_to_mx(fake_mx):
    be = _apple()
    be.save_tensors("/tmp/x.safetensors", {"w": 1})
    assert fake_mx._saved == ("/tmp/x.safetensors", {"w": 1})
    assert be.load_tensors("/tmp/x.safetensors") == {"w": ("LOADED", "/tmp/x.safetensors")}


def test_configure_memory_rejects_out_of_range_fraction(fake_mx):
    be = _apple()
    be.configure_memory(0.1)  # < 0.5 → early return, no metal call
    assert not hasattr(fake_mx, "_memlimit")


def test_configure_memory_swallows_non_macos_sysctl_failure(fake_mx):
    # On Linux ctypes.CDLL("libSystem.dylib") raises OSError; configure_memory
    # must swallow it rather than propagate (the except branch).
    be = _apple()
    be.configure_memory(0.9)
    assert not hasattr(fake_mx, "_memlimit")
