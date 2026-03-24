"""tests/test_wave71_cross_platform.py — Wave 71 cross-platform tests.

Tests for:
  - squish/platform/cuda_backend.py    (CUDABackend)
  - squish/platform/windows_backend.py (WindowsBackend)
  - squish/platform/platform_router.py (PlatformRouter)

All tests are deterministic and do NOT require physical CUDA/DirectML/ANE
hardware.  GPU-dependent logic is tested via mocking.
"""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CUDABackend tests
# ---------------------------------------------------------------------------
from squish.platform.cuda_backend import (
    CUDABackend,
    CUDAConfig,
    CUDABackendStats,
    CUDADeviceInfo,
    CUDAKernelPath,
)


# -- CUDAConfig validation ---------------------------------------------------

class TestCUDAConfig:
    def test_defaults(self):
        cfg = CUDAConfig()
        assert cfg.device_index == 0
        assert cfg.memory_fraction == 0.85
        assert cfg.prefer_tensorrt is False

    def test_custom_values(self):
        cfg = CUDAConfig(device_index=2, memory_fraction=0.5, prefer_tensorrt=True)
        assert cfg.device_index == 2
        assert cfg.memory_fraction == 0.5
        assert cfg.prefer_tensorrt is True

    def test_negative_device_index_raises(self):
        with pytest.raises(ValueError, match="device_index"):
            CUDAConfig(device_index=-1)

    def test_zero_memory_fraction_raises(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            CUDAConfig(memory_fraction=0.0)

    def test_above_one_memory_fraction_raises(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            CUDAConfig(memory_fraction=1.1)

    def test_memory_fraction_one_is_valid(self):
        cfg = CUDAConfig(memory_fraction=1.0)
        assert cfg.memory_fraction == 1.0


# -- CUDADeviceInfo immutability --------------------------------------------

class TestCUDADeviceInfo:
    def test_frozen(self):
        info = CUDADeviceInfo(
            device_name="RTX 4090",
            vram_gb=24.0,
            cuda_version="12.1",
            driver_version="535.86.10",
            compute_capability="8.9",
            sm_count=128,
            is_available=True,
            has_bf16=True,
            has_fp8=True,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            info.vram_gb = 99.0  # type: ignore[misc]

    def test_fields_accessible(self):
        info = CUDADeviceInfo(
            device_name="Test GPU",
            vram_gb=8.0,
            cuda_version="11.8",
            driver_version="520.0",
            compute_capability="7.5",
            sm_count=40,
            is_available=True,
            has_bf16=False,
            has_fp8=False,
        )
        assert info.device_name == "Test GPU"
        assert info.vram_gb == 8.0
        assert info.has_bf16 is False


# -- CUDABackend no-CUDA scenarios ------------------------------------------

class TestCUDABackendNoCUDA:
    def test_not_available_without_torch_cuda(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            assert backend.is_available() is False

    def test_detect_returns_not_available(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            info = backend.detect()
            assert info.is_available is False
            assert info.vram_gb == 0.0

    def test_kernel_path_unknown_when_no_cuda(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            assert backend.get_kernel_path() == CUDAKernelPath.UNKNOWN

    def test_recommended_config_cpu_fallback(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            cfg = backend.get_recommended_config()
            assert cfg["device"] == "cpu"
            assert cfg["dtype"] == "float32"
            assert cfg["batch_size"] == 1

    def test_stats_incremented_on_detect(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            backend.detect()
            backend.detect()
            assert backend.stats.detection_calls == 2
            assert backend.stats.cache_hits == 1

    def test_repr_detected_false(self):
        backend = CUDABackend.__new__(CUDABackend)
        backend._cfg   = CUDAConfig()
        backend.stats  = CUDABackendStats()
        backend._info  = None
        assert "detected=False" in repr(backend)

    def test_reset_clears_cache(self):
        with patch.object(CUDABackend, "_check_cuda_present", return_value=False):
            backend = CUDABackend()
            _ = backend.detect()
            assert backend._info is not None
            backend.reset()
            assert backend._info is None


# -- CUDABackend with mocked CUDA device ------------------------------------

def _make_mock_info(cc_major=8, cc_minor=0, vram_gb=24.0) -> CUDADeviceInfo:
    return CUDADeviceInfo(
        device_name="Mock RTX A100",
        vram_gb=vram_gb,
        cuda_version="12.1",
        driver_version="535.86.10",
        compute_capability=f"{cc_major}.{cc_minor}",
        sm_count=108,
        is_available=True,
        has_bf16=cc_major >= 8,
        has_fp8=cc_major >= 9 or (cc_major == 8 and cc_minor >= 9),
    )


class TestCUDABackendWithGPU:
    def _backend_with_info(self, info: CUDADeviceInfo) -> CUDABackend:
        backend = CUDABackend()
        backend._info = info
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            pass
        return backend

    def test_smoothquant_path_ampere_16gb(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=8, cc_minor=0, vram_gb=24.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            path = backend.get_kernel_path()
        assert path == CUDAKernelPath.W8A8_SMOOTHQUANT

    def test_int4_path_low_vram(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=7, cc_minor=5, vram_gb=8.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            path = backend.get_kernel_path()
        assert path == CUDAKernelPath.INT4_GROUPWISE

    def test_fp16_baseline_legacy(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=6, cc_minor=1, vram_gb=30.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            path = backend.get_kernel_path()
        # vram > INT4_MAX_VRAM but cc < smoothquant min CC → FP16 baseline
        assert path == CUDAKernelPath.FP16_BASELINE

    def test_tensorrt_preferred_ampere(self):
        backend = CUDABackend(CUDAConfig(prefer_tensorrt=True))
        backend._info = _make_mock_info(cc_major=8, cc_minor=0, vram_gb=24.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True), \
             patch.object(CUDABackend, "_tensorrt_available", return_value=True):
            path = backend.get_kernel_path()
        assert path == CUDAKernelPath.FP16_TENSORRT

    def test_recommended_config_bf16_ampere(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=8, cc_minor=0, vram_gb=24.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            cfg = backend.get_recommended_config()
        assert cfg["dtype"] == "bfloat16"
        assert cfg["device"] == "cuda:0"
        assert cfg["flash_attn_support"] is True

    def test_recommended_config_no_cuda_graphs_turing(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=7, cc_minor=5, vram_gb=8.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            cfg = backend.get_recommended_config()
        assert cfg["cuda_graphs"] is False

    def test_batch_size_scaling(self):
        cases = [
            (80.0, 64), (40.0, 32), (24.0, 16), (16.0, 8), (8.0, 4), (4.0, 1),
        ]
        for vram, expected_batch in cases:
            backend = CUDABackend()
            backend._info = _make_mock_info(vram_gb=vram)
            with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
                cfg = backend.get_recommended_config()
            assert cfg["batch_size"] == expected_batch, f"vram={vram}"

    def test_has_fp8_hopper(self):
        info = _make_mock_info(cc_major=9, cc_minor=0)
        assert info.has_fp8 is True

    def test_has_fp8_false_ampere(self):
        info = _make_mock_info(cc_major=8, cc_minor=0)
        assert info.has_fp8 is False

    def test_repr_available(self):
        backend = CUDABackend()
        backend._info = _make_mock_info(cc_major=8, cc_minor=0, vram_gb=24.0)
        with patch.object(CUDABackend, "_check_cuda_present", return_value=True):
            r = repr(backend)
        assert "CUDABackend(" in r
        assert "8.0" in r


# -- CUDAKernelPath enum -----------------------------------------------------

class TestCUDAKernelPath:
    def test_all_paths_unique(self):
        values = [p.value for p in CUDAKernelPath]
        assert len(values) == len(set(values))

    def test_unknown_exists(self):
        assert CUDAKernelPath.UNKNOWN is not None

    def test_smoothquant_name(self):
        assert CUDAKernelPath.W8A8_SMOOTHQUANT.name == "W8A8_SMOOTHQUANT"


# ---------------------------------------------------------------------------
# WindowsBackend tests
# ---------------------------------------------------------------------------
from squish.platform.windows_backend import (
    WindowsBackend,
    WindowsConfig,
    WindowsBackendStats,
    DMLAdapterInfo,
)


class TestWindowsConfig:
    def test_defaults(self):
        cfg = WindowsConfig()
        assert cfg.adapter_index == -1
        assert cfg.memory_fraction == 0.85
        assert cfg.allow_wsl2_fallback is True

    def test_invalid_adapter_index(self):
        with pytest.raises(ValueError, match="adapter_index"):
            WindowsConfig(adapter_index=-2)

    def test_invalid_memory_fraction(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            WindowsConfig(memory_fraction=0.0)

    def test_adapter_index_zero_valid(self):
        cfg = WindowsConfig(adapter_index=0)
        assert cfg.adapter_index == 0


class TestDMLAdapterInfo:
    def test_frozen(self):
        info = DMLAdapterInfo(
            adapter_index=0, adapter_name="NVIDIA RTX 4090",
            vram_gb=24.0, is_discrete=True, vendor_id=0x10DE, is_available=True,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            info.vram_gb = 99.0  # type: ignore[misc]

    def test_fields(self):
        info = DMLAdapterInfo(
            adapter_index=1, adapter_name="Intel UHD",
            vram_gb=0.0, is_discrete=False, vendor_id=0x8086, is_available=True,
        )
        assert info.vendor_id == 0x8086
        assert info.is_discrete is False


class TestWindowsBackendNoDML:
    def test_not_available_non_windows(self):
        with patch("squish.platform.windows_backend.WindowsBackend._check_dml_present",
                   return_value=False), \
             patch("squish.platform.windows_backend.WindowsBackend._is_wsl2",
                   return_value=False):
            backend = WindowsBackend()
            assert backend.is_available() is False

    def test_wsl2_fallback_disables_dml(self):
        cfg = WindowsConfig(allow_wsl2_fallback=True)
        with patch("squish.platform.windows_backend.WindowsBackend._is_wsl2",
                   return_value=True):
            backend = WindowsBackend(cfg)
            assert backend.is_available() is False

    def test_wsl2_fallback_disabled(self):
        cfg = WindowsConfig(allow_wsl2_fallback=False)
        with patch("squish.platform.windows_backend.WindowsBackend._check_dml_present",
                   return_value=True), \
             patch("squish.platform.windows_backend.WindowsBackend._is_wsl2",
                   return_value=True), \
             patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=[]):
            backend = WindowsBackend(cfg)
            assert backend.is_available() is True

    def test_enumerate_empty_no_dml(self):
        with patch("squish.platform.windows_backend.WindowsBackend._check_dml_present",
                   return_value=False), \
             patch("squish.platform.windows_backend.WindowsBackend._is_wsl2",
                   return_value=False), \
             patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=[]):
            backend = WindowsBackend()
            adapters = backend.enumerate_adapters()
            assert adapters == []

    def test_best_adapter_returns_no_adapter_when_empty(self):
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=[]):
            backend = WindowsBackend()
            best = backend.best_adapter()
            assert best.is_available is False

    def test_recommended_config_cpu_fallback(self):
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=[]):
            backend = WindowsBackend()
            cfg = backend.get_recommended_config()
            assert cfg["device"] == "cpu"

    def test_repr_not_detected(self):
        backend = WindowsBackend.__new__(WindowsBackend)
        backend._cfg      = WindowsConfig()
        backend.stats     = WindowsBackendStats()
        backend._adapters = None
        assert "detected=False" in repr(backend)


class TestWindowsBackendWithAdapters:
    def _make_adapters(self) -> List[DMLAdapterInfo]:
        return [
            DMLAdapterInfo(0, "NVIDIA RTX 4090",  24.0, True,  0x10DE, True),
            DMLAdapterInfo(1, "Intel UHD 770",     0.0, False, 0x8086, True),
        ]

    def test_best_adapter_auto_discrete(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            best = backend.best_adapter()
        assert best.is_discrete is True
        assert best.adapter_name == "NVIDIA RTX 4090"

    def test_best_adapter_honours_specific_index(self):
        adapters = self._make_adapters()
        cfg = WindowsConfig(adapter_index=1)
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend(cfg)
            best = backend.best_adapter()
        assert best.adapter_index == 1

    def test_device_map_correct_index(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            dm = backend.get_device_map()
        assert "privateuseone:0" in dm[""]

    def test_recommended_config_fields(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            cfg = backend.get_recommended_config()
        assert cfg["dtype"] == "float16"
        assert "privateuseone" in cfg["device"]
        assert cfg["is_discrete_gpu"] is True

    def test_stats_tracking(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            backend.enumerate_adapters()
            backend.enumerate_adapters()
        assert backend.stats.detection_calls == 2
        assert backend.stats.cache_hits == 1

    def test_reset_clears_cache(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            backend.enumerate_adapters()
            assert backend._adapters is not None
            backend.reset()
            assert backend._adapters is None

    def test_repr_with_adapters(self):
        adapters = self._make_adapters()
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            backend.enumerate_adapters()
            r = repr(backend)
        assert "WindowsBackend(" in r
        assert "2" in r   # 2 adapters

    def test_batch_size_high_vram(self):
        adapters = [DMLAdapterInfo(0, "Big GPU", 24.0, True, 0x10DE, True)]
        with patch("squish.platform.windows_backend.WindowsBackend._run_enumeration",
                   return_value=adapters):
            backend = WindowsBackend()
            cfg = backend.get_recommended_config()
        assert cfg["batch_size"] >= 8


# ---------------------------------------------------------------------------
# PlatformRouter tests
# ---------------------------------------------------------------------------
from squish.platform.platform_router import (
    BackendChainEntry,
    BackendPriority,
    PlatformRouter,
    PlatformRouterConfig,
    PlatformRouterStats,
    RoutedBackend,
)


class TestPlatformRouterConfig:
    def test_defaults(self):
        cfg = PlatformRouterConfig()
        assert cfg.cuda_device_index == 0
        assert cfg.rocm_device_index == 0
        assert cfg.dml_adapter_index == -1
        assert cfg.ane_model_size_gb == 8.0

    def test_invalid_cuda_index(self):
        with pytest.raises(ValueError, match="cuda_device_index"):
            PlatformRouterConfig(cuda_device_index=-1)

    def test_invalid_ane_size(self):
        with pytest.raises(ValueError, match="ane_model_size_gb"):
            PlatformRouterConfig(ane_model_size_gb=0.0)

    def test_custom_values(self):
        cfg = PlatformRouterConfig(cuda_device_index=1, ane_model_size_gb=4.0)
        assert cfg.cuda_device_index == 1
        assert cfg.ane_model_size_gb == 4.0


class TestRoutedBackend:
    def test_frozen(self):
        rb = RoutedBackend(
            name="cuda", device="cuda:0",
            kernel_path_hint="W8A8_SMOOTHQUANT", priority=2, latency_ms=5.0,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            rb.name = "cpu"  # type: ignore[misc]

    def test_fields(self):
        rb = RoutedBackend(
            name="mlx", device="mlx:0",
            kernel_path_hint="METAL_GEMV", priority=4, latency_ms=1.2,
        )
        assert rb.name == "mlx"
        assert rb.priority == 4


class TestBackendPriority:
    def test_ane_highest_priority(self):
        assert BackendPriority.ANE < BackendPriority.CUDA
        assert BackendPriority.ANE < BackendPriority.ROCM
        assert BackendPriority.ANE < BackendPriority.METAL
        assert BackendPriority.ANE < BackendPriority.DIRECTML
        assert BackendPriority.ANE < BackendPriority.CPU

    def test_cpu_lowest_priority(self):
        for p in [BackendPriority.ANE, BackendPriority.CUDA,
                  BackendPriority.ROCM, BackendPriority.METAL,
                  BackendPriority.DIRECTML]:
            assert p < BackendPriority.CPU

    def test_cuda_before_rocm(self):
        assert BackendPriority.CUDA < BackendPriority.ROCM


class TestPlatformRouterAllFail:
    def _all_fail_router(self) -> PlatformRouter:
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        return router

    def test_falls_back_to_cpu(self):
        router = self._all_fail_router()
        result = router.route()
        assert result.name == "cpu"
        assert result.device == "cpu"

    def test_cpu_fallback_priority(self):
        router = self._all_fail_router()
        result = router.route()
        assert result.priority == BackendPriority.CPU

    def test_result_cached(self):
        router = self._all_fail_router()
        r1 = router.route()
        r2 = router.route()
        assert router.stats.route_calls == 2
        assert router.stats.cache_hits == 1
        assert r1 is r2

    def test_reset_clears_result(self):
        router = self._all_fail_router()
        router.route()
        router.reset()
        assert router._result is None
        assert router._chain is None

    def test_repr_unresolved(self):
        router = PlatformRouter.__new__(PlatformRouter)
        router._cfg     = PlatformRouterConfig()
        router.stats    = PlatformRouterStats()
        router._result  = None
        router._chain   = None
        assert "unresolved" in repr(router)


class TestPlatformRouterCUDAAvailable:
    def test_routes_to_cuda_over_cpu(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: True
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        result = router.route()
        assert result.name == "cuda"

    def test_cuda_device_spec_includes_index(self):
        cfg = PlatformRouterConfig(cuda_device_index=2)
        router = PlatformRouter(cfg)
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: True
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        result = router.route()
        assert "2" in result.device


class TestPlatformRouterANEPriority:
    def test_ane_wins_over_cuda(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: True
        router._probe_cuda     = lambda: True
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        result = router.route()
        assert result.name == "ane"

    def test_ane_kernel_hint(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: True
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        result = router.route()
        assert "ANE" in result.kernel_path_hint or "COREML" in result.kernel_path_hint


class TestPlatformRouterChain:
    def test_build_chain_has_all_backends(self):
        router = PlatformRouter()
        chain  = router.build_chain()
        names  = {e.name for e in chain}
        assert {"ane", "cuda", "rocm", "mlx", "directml"} <= names

    def test_chain_cached(self):
        router = PlatformRouter()
        c1 = router.build_chain()
        c2 = router.build_chain()
        assert c1 is c2

    def test_probe_exception_handled(self):
        """A probe that raises should NOT propagate up — treated as unavailable."""
        router = PlatformRouter()
        router._probe_ane      = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        result = router.route()
        assert result.name == "cpu"   # fallback, no crash

    def test_stats_probes_fired(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        router.route()
        assert router.stats.probes_fired >= 5

    def test_mlx_available_routes_to_mlx(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: True
        router._probe_directml = lambda: False
        result = router.route()
        assert result.name == "mlx"

    def test_directml_available_routes_to_directml(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: True
        result = router.route()
        assert result.name == "directml"

    def test_rocm_before_metal(self):
        # ROCm priority < Metal priority
        assert BackendPriority.ROCM < BackendPriority.METAL

    def test_repr_with_result(self):
        router = PlatformRouter()
        router._probe_ane      = lambda: False
        router._probe_cuda     = lambda: False
        router._probe_rocm     = lambda: False
        router._probe_mlx      = lambda: False
        router._probe_directml = lambda: False
        router.route()
        r = repr(router)
        assert "selected='cpu'" in r
