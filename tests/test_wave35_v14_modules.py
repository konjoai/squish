"""tests/test_wave35_v14_modules.py — v14 Wave 35: Linux/CUDA Foundation tests.

Tests for:
  - squish.platform.detector      (UnifiedPlatformDetector)
  - squish.platform.memory_linux  (LinuxMemGovernor)
  - squish.kernels.cuda_flash_attn (CUDAFlashAttention)
  - squish.quant.bnb_quant         (BitsAndBytesQuantizer)
  - squish.io.mmap_loader          (CrossPlatformMmapLoader)
  - squish.platform.feature_registry (PlatformFeatureRegistry)

All tests run on macOS and Linux without CUDA hardware.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# ============================================================
# squish.platform.detector
# ============================================================
from squish.platform.detector import (
    CUDAInfo,
    PlatformDetectorStats,
    PlatformInfo,
    PlatformKind,
    UnifiedPlatformDetector,
)


class TestPlatformKind:
    def test_enum_has_required_values(self):
        names = {k.name for k in PlatformKind}
        for expected in (
            "MACOS_APPLE_SILICON", "LINUX_CUDA", "LINUX_ROCM",
            "LINUX_CPU", "WINDOWS_WSL", "WINDOWS_NATIVE", "UNKNOWN",
        ):
            assert expected in names

    def test_seven_kinds(self):
        assert len(PlatformKind) == 7


class TestCUDAInfo:
    def test_instantiation(self):
        info = CUDAInfo(
            device_count=1, device_name="A100",
            total_memory_gb=80.0, compute_capability="8.0",
            is_available=True,
        )
        assert info.device_count == 1
        assert info.device_name == "A100"
        assert info.total_memory_gb == 80.0

    def test_frozen(self):
        info = CUDAInfo(1, "A100", 80.0, "8.0", True)
        with pytest.raises((AttributeError, TypeError)):
            info.device_count = 2  # type: ignore[misc]


class TestPlatformInfo:
    def _make(self, kind=PlatformKind.LINUX_CPU):
        return PlatformInfo(
            kind=kind, os_name="linux", python_version="3.12.0",
            arch="x86_64", has_mlx=False, has_cuda=False,
            has_rocm=False, is_wsl=False, cuda_info=None,
            apple_chip="", ram_gb=16.0,
        )

    def test_instantiation(self):
        info = self._make()
        assert info.kind == PlatformKind.LINUX_CPU
        assert info.ram_gb == 16.0

    def test_frozen(self):
        info = self._make()
        with pytest.raises((AttributeError, TypeError)):
            info.ram_gb = 32.0  # type: ignore[misc]


class TestUnifiedPlatformDetector:
    def test_detect_returns_platform_info(self):
        det = UnifiedPlatformDetector()
        assert isinstance(det.detect(), PlatformInfo)

    def test_detect_is_cached(self):
        det = UnifiedPlatformDetector()
        assert det.detect() is det.detect()

    def test_stats_increment(self):
        det = UnifiedPlatformDetector()
        det.detect()
        det.detect()
        assert det.stats.detection_calls == 2
        assert det.stats.cache_hits == 1

    def test_cache_hit_rate(self):
        det = UnifiedPlatformDetector()
        for _ in range(3):
            det.detect()
        assert abs(det.stats.cache_hit_rate - 2 / 3) < 1e-6

    def test_zero_calls_cache_rate(self):
        det = UnifiedPlatformDetector()
        assert det.stats.cache_hit_rate == 0.0

    def test_reset_clears_cache(self):
        det = UnifiedPlatformDetector()
        det.detect()
        det.reset()
        info2 = det.detect()
        assert isinstance(info2, PlatformInfo)
        assert det.stats.detection_calls == 2

    def test_detection_time_recorded(self):
        det = UnifiedPlatformDetector()
        det.detect()
        assert det.stats.last_detection_ms >= 0.0

    def test_platform_kind_matches_os(self):
        det  = UnifiedPlatformDetector()
        info = det.detect()
        if sys.platform == "darwin":
            assert info.kind == PlatformKind.MACOS_APPLE_SILICON
        elif sys.platform.startswith("linux"):
            assert info.kind in {
                PlatformKind.LINUX_CUDA, PlatformKind.LINUX_ROCM,
                PlatformKind.LINUX_CPU, PlatformKind.WINDOWS_WSL,
            }

    def test_python_version_populated(self):
        info = UnifiedPlatformDetector().detect()
        assert len(info.python_version.split(".")) >= 2

    def test_arch_populated(self):
        info = UnifiedPlatformDetector().detect()
        assert info.arch

    def test_repr_before_detect(self):
        assert "not yet detected" in repr(UnifiedPlatformDetector())

    def test_repr_after_detect(self):
        det = UnifiedPlatformDetector()
        det.detect()
        assert "UnifiedPlatformDetector" in repr(det)

    def test_ram_gb_nonnegative(self):
        info = UnifiedPlatformDetector().detect()
        assert info.ram_gb >= 0.0


# ============================================================
# squish.platform.memory_linux
# ============================================================
from squish.platform.memory_linux import (
    LinuxMemConfig,
    LinuxMemGovernor,
    LinuxMemGovernorStats,
    LinuxMemLevel,
    LinuxMemSnapshot,
)


class TestLinuxMemLevel:
    def test_ordering(self):
        assert LinuxMemLevel.OK < LinuxMemLevel.MODERATE
        assert LinuxMemLevel.MODERATE < LinuxMemLevel.HIGH
        assert LinuxMemLevel.HIGH < LinuxMemLevel.CRITICAL

    def test_four_levels(self):
        assert len(LinuxMemLevel) == 4


class TestLinuxMemConfig:
    def test_defaults(self):
        cfg = LinuxMemConfig()
        assert cfg.poll_interval_s == 1.0
        assert cfg.moderate_threshold == 0.65
        assert cfg.high_threshold == 0.80
        assert cfg.critical_threshold == 0.92

    def test_invalid_poll_interval_zero(self):
        with pytest.raises(ValueError, match="poll_interval_s"):
            LinuxMemConfig(poll_interval_s=0)

    def test_invalid_poll_interval_negative(self):
        with pytest.raises(ValueError, match="poll_interval_s"):
            LinuxMemConfig(poll_interval_s=-1.0)

    def test_threshold_out_of_range_zero(self):
        with pytest.raises(ValueError):
            LinuxMemConfig(moderate_threshold=0.0)

    def test_threshold_out_of_range_one(self):
        with pytest.raises(ValueError):
            LinuxMemConfig(critical_threshold=1.0)

    def test_threshold_order_violation(self):
        with pytest.raises(ValueError, match="moderate < high < critical"):
            LinuxMemConfig(moderate_threshold=0.80, high_threshold=0.65)

    def test_custom_valid_config(self):
        cfg = LinuxMemConfig(poll_interval_s=0.5, moderate_threshold=0.60,
                             high_threshold=0.75, critical_threshold=0.90)
        assert cfg.poll_interval_s == 0.5


class TestLinuxMemGovernor:
    def test_instantiation(self):
        gov = LinuxMemGovernor()
        assert isinstance(gov.stats, LinuxMemGovernorStats)

    def test_snapshot_returns_snapshot(self):
        gov  = LinuxMemGovernor()
        snap = gov.snapshot()
        assert isinstance(snap, LinuxMemSnapshot)

    def test_snapshot_noop_on_non_linux(self):
        if sys.platform == "linux":
            pytest.skip("Linux platform")
        gov  = LinuxMemGovernor()
        snap = gov.snapshot()
        assert snap.level == LinuxMemLevel.OK
        assert snap.total_gb == 0.0

    def test_start_stop_noop_on_non_linux(self):
        if sys.platform == "linux":
            pytest.skip("Linux platform")
        gov = LinuxMemGovernor()
        gov.start()
        gov.stop()

    def test_register_handler_no_error(self):
        gov = LinuxMemGovernor()
        gov.register_handler(LinuxMemLevel.OK, lambda s: None)

    def test_current_level_default_ok(self):
        assert LinuxMemGovernor().current_level == LinuxMemLevel.OK

    def test_snapshot_pressure_pct(self):
        snap = LinuxMemSnapshot(
            total_gb=16.0, available_gb=8.0, used_gb=8.0, usage_ratio=0.5,
            cgroup_limit_gb=None, cgroup_usage_gb=None,
            swap_total_gb=0.0, swap_free_gb=0.0, level=LinuxMemLevel.OK,
        )
        assert snap.pressure_pct == pytest.approx(50.0)

    def test_repr(self):
        assert "LinuxMemGovernor" in repr(LinuxMemGovernor())

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_linux_snapshot_nonzero_total(self):
        gov  = LinuxMemGovernor()
        snap = gov.snapshot()
        assert snap.total_gb > 0.0

    def test_stats_instance(self):
        gov = LinuxMemGovernor()
        assert isinstance(gov.stats, LinuxMemGovernorStats)


# ============================================================
# squish.kernels.cuda_flash_attn
# ============================================================
from squish.kernels.cuda_flash_attn import (
    CUDAFlashAttention,
    CUDAFlashConfig,
    CUDAFlashStats,
)


def _make_qkv(seq=8, heads=4, d=16, seed=42) -> tuple:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq, heads, d)).astype(np.float32)
    k = rng.standard_normal((seq, heads, d)).astype(np.float32)
    v = rng.standard_normal((seq, heads, d)).astype(np.float32)
    return q, k, v


class TestCUDAFlashConfig:
    def test_defaults(self):
        cfg = CUDAFlashConfig()
        assert cfg.causal is True
        assert cfg.dropout == 0.0
        assert cfg.scale is None
        assert cfg.implementation == "auto"

    def test_invalid_implementation(self):
        with pytest.raises(ValueError, match="implementation"):
            CUDAFlashConfig(implementation="bad")

    def test_invalid_dropout_negative(self):
        with pytest.raises(ValueError, match="dropout"):
            CUDAFlashConfig(dropout=-0.1)

    def test_invalid_dropout_one(self):
        with pytest.raises(ValueError, match="dropout"):
            CUDAFlashConfig(dropout=1.0)

    def test_all_valid_implementations(self):
        for impl in ("auto", "flash_attn", "xformers", "torch_sdpa", "numpy"):
            assert CUDAFlashConfig(implementation=impl).implementation == impl


class TestCUDAFlashAttention:
    def test_forward_3d_shape(self):
        cfg  = CUDAFlashConfig(implementation="numpy")
        attn = CUDAFlashAttention(cfg)
        q, k, v = _make_qkv()
        out, lse = attn.forward(q, k, v)
        assert out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[1])

    def test_forward_2d_single_head(self):
        cfg  = CUDAFlashConfig(implementation="numpy")
        attn = CUDAFlashAttention(cfg)
        q = np.random.randn(8, 16).astype(np.float32)
        out, lse = attn.forward(q, q, q)
        assert out.shape == (8, 16)
        assert lse.shape == (8,)

    def test_output_is_float32(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(implementation="numpy"))
        q, k, v = _make_qkv()
        out, lse = attn.forward(q, k, v)
        assert out.dtype == np.float32
        assert lse.dtype == np.float32

    def test_stats_calls_increment(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(implementation="numpy"))
        q, k, v = _make_qkv(seq=4)
        attn.forward(q, k, v)
        attn.forward(q, k, v)
        assert attn.stats.total_forward_calls == 2
        assert attn.stats.total_query_tokens  == 8

    def test_avg_tokens_per_call(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(implementation="numpy"))
        q, k, v = _make_qkv(seq=6)
        attn.forward(q, k, v)
        assert attn.stats.avg_tokens_per_call == pytest.approx(6.0)

    def test_reset_stats(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(implementation="numpy"))
        q, k, v = _make_qkv()
        attn.forward(q, k, v)
        attn.reset_stats()
        assert attn.stats.total_forward_calls == 0

    def test_causal_no_nans(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(causal=True, implementation="numpy"))
        q, k, v = _make_qkv(seq=4)
        out, _ = attn.forward(q, k, v)
        assert np.all(np.isfinite(out))

    def test_non_causal_shape(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(causal=False, implementation="numpy"))
        q, k, v = _make_qkv()
        out, lse = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_custom_scale(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(scale=0.1, implementation="numpy"))
        q, k, v = _make_qkv()
        out, _ = attn.forward(q, k, v)
        assert np.all(np.isfinite(out))

    def test_numpy_fallback_calls_tracked(self):
        attn = CUDAFlashAttention(CUDAFlashConfig(implementation="numpy"))
        q, k, v = _make_qkv()
        attn.forward(q, k, v)
        assert attn.stats.numpy_fallback_calls == 1

    def test_repr(self):
        assert "CUDAFlashAttention" in repr(CUDAFlashAttention())


# ============================================================
# squish.quant.bnb_quant
# ============================================================
from squish.quant.bnb_quant import (
    BitsAndBytesQuantizer,
    BnbConfig,
    BnbQuantized,
    BnbStats,
)


def _rand_weight(rows=64, cols=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)


class TestBnbConfig:
    def test_defaults(self):
        cfg = BnbConfig()
        assert cfg.quant_type == "nf4"
        assert cfg.compute_dtype == "float16"
        assert cfg.use_double_quant is True
        assert cfg.group_size == 64

    def test_invalid_quant_type(self):
        with pytest.raises(ValueError, match="quant_type"):
            BnbConfig(quant_type="q8_0")

    def test_invalid_compute_dtype(self):
        with pytest.raises(ValueError, match="compute_dtype"):
            BnbConfig(compute_dtype="bf16")

    def test_invalid_group_size_zero(self):
        with pytest.raises(ValueError, match="group_size"):
            BnbConfig(group_size=0)

    def test_valid_quant_types(self):
        for qt in ("nf4", "int8", "fp4"):
            assert BnbConfig(quant_type=qt).quant_type == qt


class TestBitsAndBytesQuantizer:
    def test_int8_round_trip_shape(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="int8"))
        w  = _rand_weight()
        w2 = q.dequantize(q.quantize(w))
        assert w2.shape == w.shape

    def test_int8_round_trip_accuracy(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="int8"))
        w  = _rand_weight()
        w2 = q.dequantize(q.quantize(w))
        rel = np.abs(w - w2).mean() / (np.abs(w).mean() + 1e-8)
        assert rel < 0.02

    def test_nf4_round_trip_finite(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="nf4"))
        w  = _rand_weight(32, 64)
        w2 = q.dequantize(q.quantize(w))
        assert np.all(np.isfinite(w2))

    def test_fp4_round_trip_shape(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="fp4"))
        w  = _rand_weight(32, 32)
        w2 = q.dequantize(q.quantize(w))
        assert w2.shape == w.shape

    def test_stats_total_params(self):
        q = BitsAndBytesQuantizer()
        q.quantize(_rand_weight(16, 16))
        assert q.stats.total_params == 256

    def test_stats_quantized_tensors(self):
        q = BitsAndBytesQuantizer()
        q.quantize(_rand_weight(8, 8))
        q.quantize(_rand_weight(8, 8))
        assert q.stats.total_quantized_tensors == 2

    def test_dequantize_stats(self):
        q  = BitsAndBytesQuantizer()
        qw = q.quantize(_rand_weight(16, 16))
        q.dequantize(qw)
        assert q.stats.total_dequantized_tensors == 1

    def test_1d_weight(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="int8"))
        w  = np.random.randn(128).astype(np.float32)
        w2 = q.dequantize(q.quantize(w))
        assert w2.shape == (128,)

    def test_original_shape_preserved(self):
        q  = BitsAndBytesQuantizer()
        w  = _rand_weight(10, 20)
        qw = q.quantize(w)
        assert qw.original_shape == (10, 20)

    def test_quant_type_on_quantized(self):
        q  = BitsAndBytesQuantizer(BnbConfig(quant_type="nf4"))
        qw = q.quantize(_rand_weight(8, 8))
        assert qw.quant_type == "nf4"

    def test_is_bnb_native_false_on_macos(self):
        if sys.platform == "linux":
            pytest.skip("may have bitsandbytes")
        q  = BitsAndBytesQuantizer()
        qw = q.quantize(_rand_weight(8, 8))
        assert qw.is_bnb_native is False

    def test_repr(self):
        assert "BitsAndBytesQuantizer" in repr(BitsAndBytesQuantizer())


# ============================================================
# squish.io.mmap_loader
# ============================================================
from squish.io.mmap_loader import (
    CrossPlatformMmapLoader,
    MmapLoaderConfig,
    MmapLoaderStats,
)


class TestMmapLoaderConfig:
    def test_defaults(self):
        cfg = MmapLoaderConfig()
        assert cfg.mode == "auto"
        assert cfg.prefetch is True
        assert cfg.max_map_size_gb == 16.0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            MmapLoaderConfig(mode="bad")

    def test_invalid_max_map_size(self):
        with pytest.raises(ValueError, match="max_map_size_gb"):
            MmapLoaderConfig(max_map_size_gb=0.0)

    def test_all_valid_modes(self):
        for mode in ("auto", "mmap", "copy", "metal"):
            assert MmapLoaderConfig(mode=mode).mode == mode


class TestCrossPlatformMmapLoader:
    @staticmethod
    def _write_npy(tmpdir: str, name: str, arr: np.ndarray) -> str:
        path = os.path.join(tmpdir, f"{name}.npy")
        np.save(path, arr)
        return path

    def test_load_npy_array(self):
        with tempfile.TemporaryDirectory() as td:
            arr    = np.arange(16, dtype=np.float32)
            path   = self._write_npy(td, "test", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            np.testing.assert_array_equal(loader.load(path), arr)
            loader.close()

    def test_load_dir_finds_all_npy(self):
        with tempfile.TemporaryDirectory() as td:
            arr1 = np.ones(4, dtype=np.float32)
            arr2 = np.zeros(4, dtype=np.float32)
            self._write_npy(td, "a", arr1)
            self._write_npy(td, "b", arr2)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            arrays = loader.load_dir(td)
            assert set(arrays.keys()) == {"a", "b"}
            loader.close()

    def test_cache_hit_on_second_load(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.array([1.0, 2.0], dtype=np.float32)
            path = self._write_npy(td, "x", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            loader.load(path); loader.load(path)
            assert loader.stats.cache_hits == 1
            loader.close()

    def test_no_cache_hit_with_cache_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.array([1.0], dtype=np.float32)
            path = self._write_npy(td, "y", arr)
            cfg  = MmapLoaderConfig(mode="copy", cache_arrays=False)
            loader = CrossPlatformMmapLoader(cfg)
            loader.load(path); loader.load(path)
            assert loader.stats.cache_hits == 0
            loader.close()

    def test_stats_total_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.zeros(100, dtype=np.float32)
            path = self._write_npy(td, "z", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            loader.load(path)
            assert loader.stats.total_bytes > 0
            loader.close()

    def test_close_idempotent(self):
        loader = CrossPlatformMmapLoader()
        loader.close()
        loader.close()

    def test_files_loaded_counter(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.ones(4, dtype=np.float32)
            path = self._write_npy(td, "f", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            loader.load(path)
            assert loader.stats.files_loaded == 1
            loader.close()

    def test_copy_loads_tracked(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.ones(4, dtype=np.float32)
            path = self._write_npy(td, "c", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            loader.load(path)
            assert loader.stats.copy_loads == 1
            loader.close()

    def test_mmap_hit_rate_copy_only(self):
        with tempfile.TemporaryDirectory() as td:
            arr  = np.ones(8, dtype=np.float32)
            path = self._write_npy(td, "h", arr)
            loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="copy"))
            loader.load(path)
            assert loader.stats.mmap_hit_rate == 0.0
            loader.close()

    def test_repr(self):
        assert "CrossPlatformMmapLoader" in repr(CrossPlatformMmapLoader())

    def test_avg_load_ms_zero_before_loads(self):
        loader = CrossPlatformMmapLoader()
        assert loader.stats.avg_load_ms == 0.0
        loader.close()


# ============================================================
# squish.platform.feature_registry
# ============================================================
from squish.platform.feature_registry import (
    FeatureRegistryStats,
    FeatureSupport,
    PlatformFeature,
    PlatformFeatureRegistry,
)


def _p(kind_name: str, **kw) -> PlatformInfo:
    defaults = dict(
        os_name="linux", python_version="3.12", arch="x86_64",
        has_mlx=False, has_cuda=False, has_rocm=False, is_wsl=False,
        cuda_info=None, apple_chip="", ram_gb=16.0,
    )
    defaults.update(kw)
    return PlatformInfo(kind=PlatformKind[kind_name], **defaults)


class TestPlatformFeature:
    def test_ten_features(self):
        assert len(PlatformFeature) == 10

    def test_required_names(self):
        names = {f.name for f in PlatformFeature}
        for n in (
            "FLASH_ATTENTION", "METAL_DISPATCH", "CUDA_GRAPHS",
            "INT4_QUANT", "INT8_QUANT", "SPECULATIVE_DECODE",
            "LAYER_SKIP", "TOKEN_PIPELINE", "MMAP_WEIGHTS", "BNB_QUANT",
        ):
            assert n in names


class TestFeatureSupport:
    def test_three_levels(self):
        assert len(FeatureSupport) == 3
        levels = {s.name for s in FeatureSupport}
        assert levels == {"NATIVE", "EMULATED", "UNSUPPORTED"}


class TestPlatformFeatureRegistry:
    def test_macos_flash_native(self):
        info = _p("MACOS_APPLE_SILICON", os_name="darwin", arch="arm64",
                  has_mlx=True, apple_chip="M3 Pro", ram_gb=36.0)
        reg  = PlatformFeatureRegistry(info)
        assert reg.support_level(PlatformFeature.FLASH_ATTENTION) == FeatureSupport.NATIVE

    def test_macos_metal_native(self):
        info = _p("MACOS_APPLE_SILICON", os_name="darwin", arch="arm64",
                  has_mlx=True, apple_chip="M3 Pro", ram_gb=36.0)
        reg  = PlatformFeatureRegistry(info)
        assert reg.support_level(PlatformFeature.METAL_DISPATCH) == FeatureSupport.NATIVE

    def test_macos_bnb_unsupported(self):
        info = _p("MACOS_APPLE_SILICON", os_name="darwin", arch="arm64",
                  has_mlx=True, apple_chip="M3 Pro", ram_gb=36.0)
        reg  = PlatformFeatureRegistry(info)
        assert reg.support_level(PlatformFeature.BNB_QUANT) == FeatureSupport.UNSUPPORTED

    def test_linux_cuda_bnb_native(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert reg.support_level(PlatformFeature.BNB_QUANT) == FeatureSupport.NATIVE

    def test_linux_cuda_cuda_graphs_native(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert reg.support_level(PlatformFeature.CUDA_GRAPHS) == FeatureSupport.NATIVE

    def test_linux_cuda_metal_unsupported(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert reg.support_level(PlatformFeature.METAL_DISPATCH) == FeatureSupport.UNSUPPORTED

    def test_is_supported_native(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert reg.is_supported(PlatformFeature.FLASH_ATTENTION) is True

    def test_is_supported_emulated(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CPU"))
        assert reg.is_supported(PlatformFeature.FLASH_ATTENTION) is True

    def test_is_supported_false(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert reg.is_supported(PlatformFeature.METAL_DISPATCH) is False

    def test_query_stats_increment(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        reg.support_level(PlatformFeature.FLASH_ATTENTION)
        reg.support_level(PlatformFeature.INT4_QUANT)
        assert reg.stats.query_count == 2

    def test_supported_features_nonempty(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert len(reg.supported_features()) > 0

    def test_native_subset_of_supported(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert set(reg.native_features()).issubset(set(reg.supported_features()))

    def test_best_fallback_returns_string(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CPU"))
        fb = reg.best_fallback(PlatformFeature.FLASH_ATTENTION)
        assert isinstance(fb, str) and len(fb) > 0

    def test_summary_keys(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        s   = reg.summary()
        assert "FLASH_ATTENTION" in s
        assert "BNB_QUANT" in s

    def test_repr_contains_class_name(self):
        reg = PlatformFeatureRegistry(_p("LINUX_CUDA"))
        assert "PlatformFeatureRegistry" in repr(reg)
