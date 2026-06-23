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
from unittest.mock import MagicMock, patch


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


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="exercises the off-macOS OSError branch; on macOS libSystem loads and "
    "the sysctl-success path runs instead (covered by test_backend_unit.py)",
)
def test_configure_memory_swallows_non_macos_sysctl_failure(fake_mx):
    # On Linux ctypes.CDLL("libSystem.dylib") raises OSError; configure_memory
    # must swallow it rather than propagate (the except branch).
    be = _apple()
    be.configure_memory(0.9)
    assert not hasattr(fake_mx, "_memlimit")


def test_configure_memory_sets_limit_on_sysctl_success(fake_mx, monkeypatch):
    """When libSystem loads and sysctlbyname succeeds (the macOS path), the Metal
    memory limit is set. Mock ctypes.CDLL so this branch runs on any OS."""
    import ctypes

    class _FakeLibc:
        def sysctlbyname(self, *_args):
            return 0  # success; leaves memsize at 0, which is fine for the test

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: _FakeLibc())
    be = _apple()
    be.configure_memory(0.9)
    assert hasattr(fake_mx, "_memlimit")


# ── _TorchBackend / create_backend (mocked torch) ──────────────────────────────


def _torch_mock(cuda: bool = False) -> MagicMock:
    m = MagicMock(name="torch")
    m.cuda.is_available.return_value = cuda
    m.device.side_effect = lambda s: s
    m.float16 = "f16"
    return m


def _save_file_recorder():
    recorded = {}
    mod = types.ModuleType("safetensors.torch")
    mod.save_file = lambda d, p: recorded.update({"dict": d, "path": p})
    mod.load_file = lambda p: {}
    return mod, recorded


def test_save_tensors_routes_tensor_and_numpy(monkeypatch, tmp_path):
    mock_torch = _torch_mock()

    class _T:  # stand-in for torch.Tensor
        def __init__(self, v):
            self.v = v

        def contiguous(self):
            return ("CONTIG", self.v)

    mock_torch.Tensor = _T
    mock_torch.from_numpy = lambda a: ("FROMNP", a)
    st_mod, recorded = _save_file_recorder()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st_mod)

    from squish.backend import _TorchBackend

    tb = _TorchBackend()
    tb.save_tensors(str(tmp_path / "w.safetensors"), {"a": _T(1), "b": [1.0, 2.0]})
    # tensor → .contiguous(); non-tensor → from_numpy
    assert recorded["dict"]["a"] == ("CONTIG", 1)
    assert recorded["dict"]["b"][0] == "FROMNP"


def test_create_backend_returns_apple_when_is_apple(monkeypatch):
    from squish import backend as be_mod

    monkeypatch.setattr(be_mod, "_IS_APPLE", True)
    assert isinstance(be_mod.create_backend(), be_mod._AppleBackend)


def test_create_backend_cpu_and_cuda_paths(monkeypatch):
    from squish import backend as be_mod

    monkeypatch.setattr(be_mod, "_IS_APPLE", False)
    monkeypatch.setitem(sys.modules, "torch", _torch_mock(cuda=True))
    cpu = be_mod.create_backend(device="cpu")
    assert cpu.device == "cpu"
    cuda = be_mod.create_backend(device="cuda")
    assert cuda.device == "cuda"


def test_create_backend_cuda_requested_but_unavailable_raises(monkeypatch):
    from squish import backend as be_mod

    monkeypatch.setattr(be_mod, "_IS_APPLE", False)
    monkeypatch.setitem(sys.modules, "torch", _torch_mock(cuda=False))
    # _TorchBackend builds (cpu), then the explicit cuda request fails the guard.
    with pytest.raises(RuntimeError, match="cuda requested"):
        be_mod.create_backend(device="cuda")


def test_create_backend_falls_back_to_stub_without_torch(monkeypatch):
    from squish import backend as be_mod

    monkeypatch.setattr(be_mod, "_IS_APPLE", False)
    # Force `import torch` inside _TorchBackend.__init__ to fail.
    monkeypatch.setitem(sys.modules, "torch", None)
    stub = be_mod.create_backend()
    assert isinstance(stub, be_mod._StubBackend)
    with pytest.raises(RuntimeError, match="no compute backend"):
        stub.array([1])


def test_configure_memory_skips_limit_when_sysctl_fails(fake_mx, monkeypatch):
    """If sysctlbyname returns non-zero, the Metal limit is not set (the
    ``ret == 0`` guard's false branch)."""
    import ctypes

    class _FailLibc:
        def sysctlbyname(self, *_args):
            return 1  # non-zero → failure

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: _FailLibc())
    be = _apple()
    be.configure_memory(0.9)
    assert not hasattr(fake_mx, "_memlimit")
