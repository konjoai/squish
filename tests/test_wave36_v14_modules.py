"""tests/test_wave36_v14_modules.py — v14 Wave 36: Cross-Platform Serving Parity tests.

Tests for:
  - squish.kernels.universal_attn       (UniversalAttention)
  - squish.serving.linux_server_init    (LinuxServerInit)
  - squish.platform.rocm_backend        (ROCmBackend)
  - squish.platform.wsl_detector        (WSLDetector)
  - squish.quant.cross_platform_loader  (CrossPlatformModelLoader)
  - squish.install.dependency_resolver  (DependencyResolver)

All tests run on macOS and Linux without CUDA / ROCm hardware.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# ============================================================
# squish.kernels.universal_attn
# ============================================================
from squish.kernels.universal_attn import (
    UniversalAttnConfig,
    UniversalAttention,
    UniversalAttnStats,
)


class TestUniversalAttnConfig:
    def test_defaults(self):
        cfg = UniversalAttnConfig()
        assert cfg.causal is True
        assert cfg.prefer_implementation == "auto"
        assert cfg.dropout == 0.0

    def test_invalid_prefer(self):
        with pytest.raises(ValueError, match="prefer_implementation"):
            UniversalAttnConfig(prefer_implementation="invalid")

    def test_invalid_dropout_one(self):
        with pytest.raises(ValueError, match="dropout"):
            UniversalAttnConfig(dropout=1.0)

    def test_valid_prefer_values(self):
        for p in ("auto", "metal", "cuda", "numpy"):
            assert UniversalAttnConfig(prefer_implementation=p).prefer_implementation == p


class TestUniversalAttention:
    @staticmethod
    def _qkv(seq=8, heads=4, d=16) -> tuple:
        rng = np.random.default_rng(7)
        q = rng.standard_normal((seq, heads, d)).astype(np.float32)
        k = rng.standard_normal((seq, heads, d)).astype(np.float32)
        v = rng.standard_normal((seq, heads, d)).astype(np.float32)
        return q, k, v

    def test_forward_3d_shape(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        out, lse = attn.forward(q, k, v)
        assert out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[1]) or lse.shape == (q.shape[0],)

    def test_forward_2d_single_head(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q = np.random.randn(8, 16).astype(np.float32)
        out, lse = attn.forward(q, q, q)
        assert out.shape == (8, 16)

    def test_output_float32(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        out, lse = attn.forward(q, k, v)
        assert out.dtype == np.float32

    def test_calls_tracked(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        attn.forward(q, k, v)
        attn.forward(q, k, v)
        assert attn.stats.total_calls == 2

    def test_numpy_calls_tracked(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        attn.forward(q, k, v)
        assert attn.stats.numpy_calls == 1

    def test_active_backend_after_calls(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        attn.forward(q, k, v)
        # numpy_calls >= 1, so active_backend should be "numpy"
        assert attn.stats.active_backend == "numpy"

    def test_reset_stats(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        attn.forward(q, k, v)
        attn.reset_stats()
        assert attn.stats.total_calls == 0

    def test_causal_no_nans(self):
        attn = UniversalAttention(UniversalAttnConfig(causal=True, prefer_implementation="numpy"))
        q, k, v = self._qkv(seq=4)
        out, _ = attn.forward(q, k, v)
        assert np.all(np.isfinite(out))

    def test_non_causal_shape(self):
        attn = UniversalAttention(UniversalAttnConfig(causal=False, prefer_implementation="numpy"))
        q, k, v = self._qkv()
        out, _ = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_backend_name_property(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        assert isinstance(attn.backend_name, str)

    def test_repr(self):
        attn = UniversalAttention()
        assert "UniversalAttention" in repr(attn)

    def test_last_call_ms_recorded(self):
        attn = UniversalAttention(UniversalAttnConfig(prefer_implementation="numpy"))
        q, k, v = self._qkv()
        attn.forward(q, k, v)
        assert attn.stats.last_call_ms >= 0.0


# ============================================================
# squish.serving.linux_server_init
# ============================================================
from squish.serving.linux_server_init import (
    LinuxInitResult,
    LinuxServerConfig,
    LinuxServerInit,
    LinuxServerStats,
)


class TestLinuxServerConfig:
    def test_defaults(self):
        cfg = LinuxServerConfig()
        assert cfg.cuda_device == "auto"
        assert cfg.memory_fraction == 0.90
        assert cfg.num_cpu_threads is None
        assert cfg.enable_tf32 is True

    def test_invalid_memory_fraction_zero(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            LinuxServerConfig(memory_fraction=0.0)

    def test_invalid_memory_fraction_over_one(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            LinuxServerConfig(memory_fraction=1.1)

    def test_invalid_num_cpu_threads(self):
        with pytest.raises(ValueError, match="num_cpu_threads"):
            LinuxServerConfig(num_cpu_threads=0)

    def test_valid_cpu_device(self):
        cfg = LinuxServerConfig(cuda_device="cpu")
        assert cfg.cuda_device == "cpu"


class TestLinuxServerInit:
    def test_initialize_returns_result(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        result = init.initialize()
        assert isinstance(result, LinuxInitResult)

    def test_cpu_mode_device(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        result = init.initialize()
        assert result.device == "cpu"
        assert result.backend_name == "cpu"

    def test_cpu_mode_memory_limit_zero(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        result = init.initialize()
        assert result.memory_limit_gb == 0.0

    def test_num_threads_positive(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        result = init.initialize()
        assert result.num_threads >= 1

    def test_init_ms_nonnegative(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        result = init.initialize()
        assert result.init_ms >= 0.0

    def test_stats_cpu_inits(self):
        init = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        init.initialize()
        assert init.stats.cpu_inits == 1
        assert init.stats.total_inits == 1

    def test_recommended_batch_size_cpu(self):
        init = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        init.initialize()
        bs = init.get_recommended_batch_size()
        assert bs >= 1

    def test_get_recommended_batch_size_calls_initialize(self):
        init = LinuxServerInit(LinuxServerConfig(cuda_device="cpu"))
        bs   = init.get_recommended_batch_size()
        assert bs >= 1

    def test_explicit_thread_count(self):
        init   = LinuxServerInit(LinuxServerConfig(cuda_device="cpu", num_cpu_threads=2))
        result = init.initialize()
        assert result.num_threads == 2

    def test_repr(self):
        init = LinuxServerInit()
        assert "LinuxServerInit" in repr(init)


# ============================================================
# squish.platform.rocm_backend
# ============================================================
from squish.platform.rocm_backend import (
    ROCmBackend,
    ROCmConfig,
    ROCmDeviceInfo,
    ROCmStats,
)


class TestROCmConfig:
    def test_defaults(self):
        cfg = ROCmConfig()
        assert cfg.device_index == 0
        assert cfg.memory_fraction == 0.85

    def test_invalid_device_index(self):
        with pytest.raises(ValueError, match="device_index"):
            ROCmConfig(device_index=-1)

    def test_invalid_memory_fraction_zero(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            ROCmConfig(memory_fraction=0.0)

    def test_invalid_memory_fraction_over(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            ROCmConfig(memory_fraction=1.1)


class TestROCmBackend:
    def test_not_available_on_macos(self):
        if sys.platform != "darwin":
            pytest.skip("macOS only")
        backend = ROCmBackend()
        assert backend.is_available() is False

    def test_detect_returns_device_info(self):
        backend = ROCmBackend()
        info    = backend.detect()
        assert isinstance(info, ROCmDeviceInfo)

    def test_detect_is_cached(self):
        backend = ROCmBackend()
        info1   = backend.detect()
        info2   = backend.detect()
        assert info1 is info2

    def test_detect_stats_increment(self):
        backend = ROCmBackend()
        backend.detect()
        backend.detect()
        assert backend.stats.detection_calls == 2
        assert backend.stats.cache_hits == 1

    def test_reset_clears_cache(self):
        backend = ROCmBackend()
        backend.detect()
        backend.reset()
        backend.detect()
        assert backend.stats.detection_calls == 2
        assert backend.stats.cache_hits == 0

    def test_not_available_returns_zeroed_info(self):
        if ROCmBackend().is_available():
            pytest.skip("ROCm found")
        info = ROCmBackend().detect()
        assert info.is_available is False
        assert info.vram_gb == 0.0

    def test_recommended_config_is_dict(self):
        backend = ROCmBackend()
        cfg     = backend.get_recommended_config()
        assert isinstance(cfg, dict)
        assert "device" in cfg
        assert "dtype" in cfg

    def test_recommended_config_cpu_fallback(self):
        if ROCmBackend().is_available():
            pytest.skip("ROCm found")
        cfg = ROCmBackend().get_recommended_config()
        assert cfg["device"] == "cpu"

    def test_repr(self):
        assert "ROCmBackend" in repr(ROCmBackend())

    def test_last_detect_ms(self):
        backend = ROCmBackend()
        backend.detect()
        assert backend.stats.last_detect_ms >= 0.0


# ============================================================
# squish.platform.wsl_detector
# ============================================================
from squish.platform.wsl_detector import (
    WSLConfig,
    WSLDetector,
    WSLDetectorStats,
    WSLInfo,
)


class TestWSLConfig:
    def test_defaults(self):
        cfg = WSLConfig()
        assert cfg.check_virtio_gpu is True
        assert cfg.check_memory_limit is True


class TestWSLDetector:
    def test_detect_returns_wsl_info(self):
        det  = WSLDetector()
        info = det.detect()
        assert isinstance(info, WSLInfo)

    def test_detect_is_cached(self):
        det = WSLDetector()
        assert det.detect() is det.detect()

    def test_stats_increment(self):
        det = WSLDetector()
        det.detect(); det.detect()
        assert det.stats.detection_calls == 2
        assert det.stats.cache_hits == 1

    def test_not_wsl_on_macos(self):
        if sys.platform != "darwin":
            pytest.skip("macOS only")
        info = WSLDetector().detect()
        assert info.is_wsl is False

    def test_reset_clears_cache(self):
        det = WSLDetector()
        det.detect()
        det.reset()
        det.detect()
        assert det.stats.detection_calls == 2
        assert det.stats.cache_hits == 0

    def test_get_memory_limit_gb_type(self):
        det = WSLDetector()
        assert isinstance(det.get_memory_limit_gb(), float)

    def test_has_gpu_access_type(self):
        det = WSLDetector()
        assert isinstance(det.has_gpu_access(), bool)

    def test_detection_time_recorded(self):
        det = WSLDetector()
        det.detect()
        assert det.stats.last_detect_ms >= 0.0

    def test_repr_before_detect(self):
        assert "not yet detected" in repr(WSLDetector())

    def test_repr_after_detect(self):
        det = WSLDetector()
        det.detect()
        assert "WSLDetector" in repr(det)

    def test_wsl_info_fields(self):
        info = WSLDetector().detect()
        assert hasattr(info, "is_wsl")
        assert hasattr(info, "wsl_version")
        assert hasattr(info, "distro_name")
        assert hasattr(info, "has_gpu")
        assert hasattr(info, "memory_limit_gb")
        assert hasattr(info, "kernel_version")


# ============================================================
# squish.quant.cross_platform_loader
# ============================================================
from squish.quant.cross_platform_loader import (
    CrossPlatformLoaderConfig,
    CrossPlatformModelLoader,
    CrossPlatformLoaderStats,
    LoadResult,
)


class TestCrossPlatformLoaderConfig:
    def test_defaults(self):
        cfg = CrossPlatformLoaderConfig()
        assert cfg.prefer_quantized is True
        assert cfg.fallback_to_fp16 is True
        assert cfg.max_memory_gb is None
        assert cfg.strategy == "auto"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            CrossPlatformLoaderConfig(strategy="bad")

    def test_invalid_max_memory(self):
        with pytest.raises(ValueError, match="max_memory_gb"):
            CrossPlatformLoaderConfig(max_memory_gb=0.0)

    def test_valid_strategies(self):
        for s in ("auto", "mlx", "torch_bnb", "torch_fp16", "torch_fp32"):
            assert CrossPlatformLoaderConfig(strategy=s).strategy == s


class TestCrossPlatformModelLoader:
    def test_select_loader_returns_string(self):
        loader = CrossPlatformModelLoader()
        result  = loader.select_loader(".")
        assert isinstance(result, str)

    def test_explicit_strategy_honored(self):
        loader = CrossPlatformModelLoader(CrossPlatformLoaderConfig(strategy="torch_fp32"))
        assert loader.select_loader(".") == "torch_fp32"

    def test_load_nonexistent_path_returns_result(self):
        loader = CrossPlatformModelLoader()
        result = loader.load("nonexistent_model_path_xyz")
        assert isinstance(result, LoadResult)
        assert result.model_path == "nonexistent_model_path_xyz"

    def test_load_result_loader_used(self):
        loader = CrossPlatformModelLoader(CrossPlatformLoaderConfig(strategy="torch_fp32"))
        result = loader.load(".")
        assert result.loader_used == "torch_fp32"

    def test_estimate_memory_zero_nonexistent(self):
        loader = CrossPlatformModelLoader()
        assert loader.estimate_memory("nonexistent_path_xyz") == 0.0

    def test_estimate_memory_dir(self):
        with tempfile.TemporaryDirectory() as td:
            arr = np.zeros(1000, dtype=np.float32)
            np.save(os.path.join(td, "weights.npy"), arr)
            loader = CrossPlatformModelLoader()
            mem = loader.estimate_memory(td)
            assert mem >= 0.0

    def test_stats_total_loads(self):
        loader = CrossPlatformModelLoader()
        loader.load("a")
        loader.load("b")
        assert loader.stats.total_loads == 2

    def test_max_memory_raises(self):
        with tempfile.TemporaryDirectory() as td:
            arr = np.zeros(100_000, dtype=np.float32)
            np.save(os.path.join(td, "big.npy"), arr)
            cfg    = CrossPlatformLoaderConfig(max_memory_gb=0.00001)
            loader = CrossPlatformModelLoader(cfg)
            with pytest.raises(MemoryError, match="max_memory_gb"):
                loader.load(td)

    def test_quantized_flag_in_result(self):
        loader = CrossPlatformModelLoader(CrossPlatformLoaderConfig(strategy="torch_fp32"))
        result = loader.load(".")
        assert result.quantized is False

    def test_mlx_strategy_quantized_true(self):
        loader = CrossPlatformModelLoader(CrossPlatformLoaderConfig(strategy="mlx"))
        result = loader.load(".")
        assert result.quantized is True

    def test_repr(self):
        assert "CrossPlatformModelLoader" in repr(CrossPlatformModelLoader())


# ============================================================
# squish.install.dependency_resolver
# ============================================================
from squish.install.dependency_resolver import (
    DependencyGroup,
    DependencyResolver,
    DependencyResolverConfig,
    DependencyResolverStats,
    InstallSpec,
)
from squish.platform.detector import PlatformInfo, PlatformKind


def _linux_cuda_info() -> PlatformInfo:
    return PlatformInfo(
        kind=PlatformKind.LINUX_CUDA, os_name="linux",
        python_version="3.12", arch="x86_64",
        has_mlx=False, has_cuda=True, has_rocm=False, is_wsl=False,
        cuda_info=None, apple_chip="", ram_gb=32.0,
    )


def _macos_info() -> PlatformInfo:
    return PlatformInfo(
        kind=PlatformKind.MACOS_APPLE_SILICON, os_name="darwin",
        python_version="3.12", arch="arm64",
        has_mlx=True, has_cuda=False, has_rocm=False, is_wsl=False,
        cuda_info=None, apple_chip="M3 Pro", ram_gb=36.0,
    )


class TestInstallSpec:
    def test_pip_token_simple(self):
        spec = InstallSpec("numpy", ">=1.24.0")
        assert spec.pip_token == "numpy>=1.24.0"

    def test_pip_token_with_extras(self):
        spec = InstallSpec("torch", ">=2.1.0", extras=("cu121",))
        assert spec.pip_token == "torch[cu121]>=2.1.0"

    def test_pip_token_no_version(self):
        spec = InstallSpec("tqdm")
        assert spec.pip_token == "tqdm"

    def test_optional_default_false(self):
        assert InstallSpec("torch").optional is False


class TestDependencyResolverConfig:
    def test_defaults(self):
        cfg = DependencyResolverConfig()
        assert cfg.auto_install is False
        assert cfg.pip_extra_index is None
        assert cfg.include_optional is True


class TestDependencyResolver:
    def test_resolve_macos_includes_mlx(self):
        resolver = DependencyResolver(platform_info=_macos_info())
        specs    = resolver.resolve()
        packages = [s.package for s in specs]
        assert "mlx" in packages

    def test_resolve_macos_excludes_torch_cuda(self):
        resolver = DependencyResolver(platform_info=_macos_info())
        specs    = resolver.resolve()
        cuda_specs = [s for s in specs if "cu121" in s.extras]
        assert len(cuda_specs) == 0

    def test_resolve_linux_cuda_includes_torch(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        specs    = resolver.resolve()
        packages = [s.package for s in specs]
        assert "torch" in packages

    def test_resolve_linux_cuda_includes_bitsandbytes(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        specs    = resolver.resolve()
        packages = [s.package for s in specs]
        assert "bitsandbytes" in packages

    def test_resolve_increments_stats(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        resolver.resolve()
        resolver.resolve()
        assert resolver.stats.resolve_calls == 2

    def test_validate_returns_dict(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        result   = resolver.validate()
        assert isinstance(result, dict)
        # numpy is always resolvable in test env
        assert "numpy" in result

    def test_validate_numpy_importable(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        result   = resolver.validate()
        assert result.get("numpy") is True

    def test_get_install_command_is_string(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        cmd      = resolver.get_install_command()
        assert isinstance(cmd, str)
        assert cmd.startswith("pip install")

    def test_get_install_command_includes_extra_index(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        cmd      = resolver.get_install_command()
        assert "--extra-index-url" in cmd

    def test_check_missing_returns_list(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        missing  = resolver.check_missing()
        assert isinstance(missing, list)

    def test_check_missing_stats(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        resolver.check_missing()
        assert resolver.stats.validation_calls == 1

    def test_exclude_optional_packages(self):
        cfg      = DependencyResolverConfig(include_optional=False)
        resolver = DependencyResolver(config=cfg, platform_info=_linux_cuda_info())
        specs    = resolver.resolve()
        assert all(not s.optional for s in specs)

    def test_repr(self):
        resolver = DependencyResolver(platform_info=_linux_cuda_info())
        assert "DependencyResolver" in repr(resolver)

    def test_resolve_core_always_present(self):
        for kind_name in ("LINUX_CUDA", "LINUX_CPU", "MACOS_APPLE_SILICON"):
            info = PlatformInfo(
                kind=PlatformKind[kind_name], os_name="linux",
                python_version="3.12", arch="x86_64",
                has_mlx=False, has_cuda=False, has_rocm=False, is_wsl=False,
                cuda_info=None, apple_chip="", ram_gb=16.0,
            )
            resolver = DependencyResolver(platform_info=info)
            packages = [s.package for s in resolver.resolve()]
            assert "numpy" in packages, f"numpy missing for {kind_name}"
