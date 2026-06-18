"""Tests for PlatformRouter — priority-ordered backend routing with CPU fallback."""
from __future__ import annotations

import pytest

from squish.platform import platform_router as pr
from squish.platform.platform_router import (
    BackendPriority,
    PlatformRouter,
    PlatformRouterConfig,
    RoutedBackend,
    get_inference_backend,
)


class TestPriorityAndConfig:
    def test_priority_ordering(self):
        assert BackendPriority.ANE < BackendPriority.CUDA < BackendPriority.ROCM
        assert BackendPriority.METAL < BackendPriority.DIRECTML < BackendPriority.CPU
        assert int(BackendPriority.CPU) == 99

    @pytest.mark.parametrize("kw", [
        {"cuda_device_index": -1},
        {"rocm_device_index": -1},
        {"ane_model_size_gb": 0},
        {"ane_model_size_gb": -2.0},
    ])
    def test_config_validation(self, kw):
        with pytest.raises(ValueError):
            PlatformRouterConfig(**kw)

    def test_config_defaults(self):
        c = PlatformRouterConfig()
        assert c.cuda_device_index == 0 and c.dml_adapter_index == -1


def _force_probes(router, *, ane=False, cuda=False, rocm=False, mlx=False, directml=False):
    """Pin every probe deterministically so routing tests are host-independent
    (the CI 'Test (Python 3.x)' jobs run on macOS where mlx/ANE are live)."""
    router._probe_ane = lambda: ane
    router._probe_cuda = (cuda if callable(cuda) else (lambda: cuda))
    router._probe_rocm = lambda: rocm
    router._probe_mlx = lambda: mlx
    router._probe_directml = lambda: directml


class TestRouting:
    def test_cpu_fallback_when_no_backend_live(self):
        # All probes unavailable → always resolves to CPU, never None.
        r = PlatformRouter()
        _force_probes(r)
        b = r.route()
        assert isinstance(b, RoutedBackend)
        assert b.name == "cpu" and b.device == "cpu"
        assert b.priority == int(BackendPriority.CPU)
        assert b.kernel_path_hint == "FP32_CPU"

    def test_route_caches_result(self):
        r = PlatformRouter()
        _force_probes(r)
        first = r.route()
        second = r.route()
        assert first is second
        assert r.stats.route_calls == 2 and r.stats.cache_hits == 1

    def test_reset_forces_rerouting(self):
        r = PlatformRouter()
        _force_probes(r)
        r.route()
        r.reset()
        assert r._result is None and r._chain is None
        r.route()
        assert r.stats.route_calls == 2 and r.stats.cache_hits == 0

    def test_highest_priority_live_backend_wins(self):
        r = PlatformRouter()
        _force_probes(r, cuda=True)  # CUDA live; ANE (higher priority) unavailable
        b = r.route()
        assert b.name == "cuda" and b.priority == int(BackendPriority.CUDA)
        assert b.device == "cuda:0"

    def test_ane_outranks_cuda_when_both_live(self):
        r = PlatformRouter()
        _force_probes(r, ane=True, cuda=True)
        assert r.route().name == "ane"

    def test_probe_exception_is_treated_as_unavailable(self):
        def _raise():
            raise RuntimeError("probe blew up")

        r = PlatformRouter()
        _force_probes(r, cuda=_raise, rocm=True)
        # CUDA probe raised → skipped; ROCm (next priority) selected.
        assert r.route().name == "rocm"

    def test_build_chain_lists_all_backends_and_caches(self):
        r = PlatformRouter()
        chain = r.build_chain()
        names = {e.name for e in chain}
        assert names == {"ane", "cuda", "rocm", "mlx", "directml"}
        assert r.build_chain() is chain  # cached

    def test_stats_and_repr(self):
        r = PlatformRouter()
        assert repr(r) == "PlatformRouter(unresolved)"
        _force_probes(r, cuda=True)
        r.route()
        assert r.stats.selected_name == "cuda"
        assert r.stats.probes_fired >= 2  # ANE (miss) then CUDA (hit)
        assert "selected='cuda'" in repr(r)


def _install_backend(monkeypatch, mod_name, cls_name, cfg_name, *, available, kernel="W8A8"):
    """Inject a stub backend module so a probe's success body can run.

    cuda_backend / rocm_backend / windows_backend are forward-looking and not
    shipped, so these stubs exercise the probe glue (construct cfg+backend, call
    is_available / get_kernel_path) the same way the real modules would.
    """
    import sys
    import types

    mod = types.ModuleType(mod_name)

    class _Cfg:
        def __init__(self, **kw):
            self.kw = kw

    class _Backend:
        def __init__(self, cfg):
            self.cfg = cfg

        def is_available(self):
            return available

        def get_kernel_path(self):
            return types.SimpleNamespace(name=kernel)

    setattr(mod, cls_name, _Backend)
    setattr(mod, cfg_name, _Cfg)
    monkeypatch.setitem(sys.modules, mod_name, mod)


class TestMlxProbe:
    def test_mlx_probe_false_without_mlx_and_non_darwin(self, monkeypatch):
        # Force the ImportError path + non-darwin platform → deterministic False
        # on any host (the macOS CI runners have real mlx installed).
        import sys
        monkeypatch.setitem(sys.modules, "mlx.core", None)  # import mlx.core → ImportError
        monkeypatch.setattr(sys, "platform", "linux")
        assert PlatformRouter()._probe_mlx() is False

    def test_mlx_probe_darwin_platform_fallback(self, monkeypatch):
        # ImportError path but darwin → platform fallback returns True.
        import sys
        monkeypatch.setitem(sys.modules, "mlx.core", None)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert PlatformRouter()._probe_mlx() is True

    def test_mlx_probe_true_when_mlx_importable(self, monkeypatch):
        import sys
        import types
        monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
        monkeypatch.setitem(sys.modules, "mlx.core", types.ModuleType("mlx.core"))
        assert PlatformRouter()._probe_mlx() is True


class TestProbeBodiesWithStubBackends:
    def test_cuda_probe_and_kernel_hint(self, monkeypatch):
        _install_backend(monkeypatch, "squish.platform.cuda_backend",
                         "CUDABackend", "CUDAConfig", available=True, kernel="W8A8_SQ")
        r = PlatformRouter()
        assert r._probe_cuda() is True
        # _cuda_kernel_hint runs while building the chain → picks up the stub name.
        chain = r.build_chain()
        cuda_entry = next(e for e in chain if e.name == "cuda")
        assert cuda_entry.kernel_path_hint == "W8A8_SQ"

    def test_cuda_kernel_hint_unavailable_falls_back(self, monkeypatch):
        _install_backend(monkeypatch, "squish.platform.cuda_backend",
                         "CUDABackend", "CUDAConfig", available=False)
        assert PlatformRouter()._cuda_kernel_hint() == "FP16_BASELINE"

    def test_rocm_probe(self, monkeypatch):
        _install_backend(monkeypatch, "squish.platform.rocm_backend",
                         "ROCmBackend", "ROCmConfig", available=True)
        assert PlatformRouter()._probe_rocm() is True

    def test_directml_probe(self, monkeypatch):
        _install_backend(monkeypatch, "squish.platform.windows_backend",
                         "WindowsBackend", "WindowsConfig", available=True)
        assert PlatformRouter()._probe_directml() is True

    def test_real_probes_without_backends_return_false(self):
        # cuda_backend / rocm_backend / windows_backend are not shipped (any OS),
        # so the real probes hit the ImportError except path → False.
        r = PlatformRouter()
        assert r._probe_cuda() is False
        assert r._probe_rocm() is False
        assert r._probe_directml() is False
        assert r._cuda_kernel_hint() == "FP16_BASELINE"  # except → fallback hint
        # ane_router IS shipped → body runs; value is host-dependent (True on ANE).
        assert isinstance(r._probe_ane(), bool)

    def test_ane_probe_available_with_stub(self, monkeypatch):
        import sys
        import types
        mod = types.ModuleType("squish.platform.ane_router")

        class ANERouter:
            def is_available(self):
                return True

        mod.ANERouter = ANERouter
        monkeypatch.setitem(sys.modules, "squish.platform.ane_router", mod)
        # default ane_model_size_gb (8.0) <= 8.0 → True
        assert PlatformRouter()._probe_ane() is True

    def test_ane_probe_handles_raising_router(self, monkeypatch):
        import sys
        import types
        mod = types.ModuleType("squish.platform.ane_router")

        class ANERouter:
            def __init__(self):
                raise RuntimeError("no ANE here")

        mod.ANERouter = ANERouter
        monkeypatch.setitem(sys.modules, "squish.platform.ane_router", mod)
        assert PlatformRouter()._probe_ane() is False  # except path


class TestGetInferenceBackend:
    class _Plat:
        def __init__(self, **kw):
            self.is_apple_silicon = kw.get("apple", False)
            self.is_cuda = kw.get("cuda", False)
            self.has_cuda = kw.get("cuda", False)
            self.has_rocm = kw.get("rocm", False)

    def test_apple(self):
        assert get_inference_backend(self._Plat(apple=True)) == "mlx"

    def test_cuda(self):
        assert get_inference_backend(self._Plat(cuda=True)) == "torch_cuda"

    def test_rocm(self):
        assert get_inference_backend(self._Plat(rocm=True)) == "torch_rocm"

    def test_cpu_default(self):
        assert get_inference_backend(self._Plat()) == "torch_cpu"

    def test_missing_attrs_defaults_to_cpu(self):
        assert get_inference_backend(object()) == "torch_cpu"
