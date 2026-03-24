"""squish/platform/cuda_backend.py — NVIDIA CUDA GPU backend.

Detects NVIDIA CUDA presence and provides Ampere/Hopper/Turing architecture
info and recommended serving configuration for CUDA-accelerated inference.

Classes
───────
CUDAConfig          — Configuration dataclass.
CUDADeviceInfo      — Detected CUDA device properties.
CUDAKernelPath      — Recommended kernel path enum.
CUDABackendStats    — Runtime statistics.
CUDABackend         — Main backend class.

Usage::

    backend = CUDABackend()
    if backend.is_available():
        info = backend.detect()
        cfg  = backend.get_recommended_config()
        path = backend.get_kernel_path()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CUDAKernelPath(Enum):
    """Recommended quantization / kernel path for CUDA serving."""
    W8A8_SMOOTHQUANT  = auto()  # Ampere+ (sm_80+): SmoothQuant INT8 GEMM
    INT4_GROUPWISE    = auto()  # Pascal–Ampere limited VRAM: bitsandbytes INT4
    FP16_BASELINE     = auto()  # Older / low-VRAM: plain FP16 GEMM
    FP16_TENSORRT     = auto()  # Ampere+ with TensorRT: compiled FP16 engine
    UNKNOWN           = auto()  # Fallback / detection not possible


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CUDAConfig:
    """Configuration for CUDABackend.

    Attributes
    ----------
    device_index:
        CUDA device index to interrogate. Default 0.
    memory_fraction:
        GPU memory fraction reserved for serving. Range (0, 1]. Default 0.85.
    prefer_tensorrt:
        Prefer TensorRT kernel path when the library is available. Default False.
    """
    device_index:     int   = 0
    memory_fraction:  float = 0.85
    prefer_tensorrt:  bool  = False

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError(
                f"device_index must be >= 0, got {self.device_index}"
            )
        if not (0.0 < self.memory_fraction <= 1.0):
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )


@dataclass(frozen=True)
class CUDADeviceInfo:
    """Properties of a detected NVIDIA CUDA device."""
    device_name:        str
    vram_gb:            float
    cuda_version:       str       # e.g. "12.1"
    driver_version:     str       # e.g. "535.86.10"
    compute_capability: str       # e.g. "8.0" for Ampere, "9.0" for Hopper
    sm_count:           int       # number of streaming multiprocessors
    is_available:       bool
    has_bf16:           bool      # BF16 tensor core support (Ampere+)
    has_fp8:            bool      # FP8 support (Hopper, H100)


@dataclass
class CUDABackendStats:
    """Runtime statistics for CUDABackend."""
    detection_calls: int   = 0
    cache_hits:      int   = 0
    last_detect_ms:  float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CUDABackend:
    """Detect and describe NVIDIA CUDA GPU for Squish inference.

    On non-CUDA machines (macOS, CPU-only Linux, AMD ROCm) this class reports
    ``is_available() → False`` and returns a zeroed-out ``CUDADeviceInfo``.

    The ``get_kernel_path()`` method returns a ``CUDAKernelPath`` that the
    unified platform router and ``squish/hardware/kernel_dispatch.py`` use to
    select the appropriate quantization and GEMM strategy.

    Usage::

        backend = CUDABackend(CUDAConfig(device_index=0))
        if backend.is_available():
            info = backend.detect()
            print(info.compute_capability, info.vram_gb)
            path = backend.get_kernel_path()
    """

    _NOT_AVAILABLE = CUDADeviceInfo(
        device_name="N/A",
        vram_gb=0.0,
        cuda_version="0.0",
        driver_version="0.0",
        compute_capability="0.0",
        sm_count=0,
        is_available=False,
        has_bf16=False,
        has_fp8=False,
    )

    # Minimum compute capability to enable W8A8 SmoothQuant path
    _SMOOTHQUANT_MIN_CC: float = 8.0  # Ampere (A100, RTX 3xxx)
    # Minimum VRAM (GB) for W8A8 path (activation buffers are large)
    _SMOOTHQUANT_MIN_VRAM_GB: float = 16.0
    # VRAM threshold below which we fall back to INT4 groupwise
    _INT4_MAX_VRAM_GB: float = 24.0

    def __init__(self, config: Optional[CUDAConfig] = None) -> None:
        self._cfg   = config or CUDAConfig()
        self.stats  = CUDABackendStats()
        self._info: Optional[CUDADeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if a CUDA device is present and torch.cuda is active.

        This explicitly skips ROCm devices: even though ROCm exposes a
        CUDA-compatible API, ``torch.version.hip`` being set indicates a ROCm
        build.  ROCm devices are handled by ``ROCmBackend`` instead.
        """
        return self._check_cuda_present()

    def detect(self) -> CUDADeviceInfo:
        """Detect and return CUDA device info (cached after first call)."""
        self.stats.detection_calls += 1
        if self._info is not None:
            self.stats.cache_hits += 1
            return self._info

        t0 = time.perf_counter()
        self._info = self._run_detection()
        self.stats.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        return self._info

    def get_kernel_path(self) -> CUDAKernelPath:
        """Return the recommended kernel/quantization path for this device.

        Priority order (highest to lowest):
        1. W8A8_SMOOTHQUANT — Ampere+ with ≥ 16 GB VRAM
        2. FP16_TENSORRT   — Ampere+ with TensorRT available (if prefer_tensorrt)
        3. INT4_GROUPWISE   — Older GPU or < 16 GB VRAM
        4. FP16_BASELINE    — Legacy / very low VRAM
        """
        info = self.detect()
        if not info.is_available:
            return CUDAKernelPath.UNKNOWN

        cc = float(info.compute_capability)

        if cc >= self._SMOOTHQUANT_MIN_CC and info.vram_gb >= self._SMOOTHQUANT_MIN_VRAM_GB:
            if self._cfg.prefer_tensorrt and self._tensorrt_available():
                return CUDAKernelPath.FP16_TENSORRT
            return CUDAKernelPath.W8A8_SMOOTHQUANT

        if cc >= 6.1 and info.vram_gb <= self._INT4_MAX_VRAM_GB:
            return CUDAKernelPath.INT4_GROUPWISE

        if cc >= 7.0:
            if self._cfg.prefer_tensorrt and self._tensorrt_available():
                return CUDAKernelPath.FP16_TENSORRT
            return CUDAKernelPath.FP16_BASELINE

        return CUDAKernelPath.FP16_BASELINE

    def get_recommended_config(self) -> dict:
        """Return a dict of recommended serving parameters for this device."""
        info = self.detect()
        if not info.is_available:
            return {"device": "cpu", "dtype": "float32", "batch_size": 1}

        cc    = float(info.compute_capability)
        dtype = "float16"
        if info.has_bf16:
            dtype = "bfloat16"

        path  = self.get_kernel_path()
        batch = self._recommend_batch_size(info.vram_gb)

        return {
            "device":             f"cuda:{self._cfg.device_index}",
            "dtype":              dtype,
            "batch_size":         batch,
            "memory_fraction":    self._cfg.memory_fraction,
            "compute_capability": info.compute_capability,
            "kernel_path":        path.name,
            "flash_attn_support": cc >= 7.5,
            "cuda_graphs":        cc >= 8.0,
        }

    def reset(self) -> None:
        """Clear cached detection result (forces re-detection on next call)."""
        self._info = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cuda_present() -> bool:
        """Return True for NVIDIA CUDA, False for ROCm or no-GPU."""
        try:
            import torch
            # ROCm masquerades as CUDA; exclude it
            is_rocm = (
                hasattr(torch.version, "hip")
                and torch.version.hip is not None
            )
            return not is_rocm and torch.cuda.is_available()
        except Exception:
            return False

    def _run_detection(self) -> CUDADeviceInfo:
        if not self._check_cuda_present():
            return self._NOT_AVAILABLE
        try:
            import torch
            idx   = self._cfg.device_index
            props = torch.cuda.get_device_properties(idx)
            vram  = round(props.total_memory / 1e9, 1)
            name  = props.name

            # Compute capability as "major.minor" string
            cc = f"{props.major}.{props.minor}"

            # CUDA runtime version
            cuda_ver = torch.version.cuda or "0.0"

            # Driver version (Linux: sysfs; fallback to "unknown")
            driver_ver = self._read_driver_version()

            # BF16 requires sm_80+ (Ampere)
            has_bf16 = props.major >= 8

            # FP8 requires sm_89+ (Ada Lovelace) or sm_90+ (Hopper)
            has_fp8 = props.major >= 9 or (props.major == 8 and props.minor >= 9)

            sm_count = props.multi_processor_count

            return CUDADeviceInfo(
                device_name=name,
                vram_gb=vram,
                cuda_version=cuda_ver,
                driver_version=driver_ver,
                compute_capability=cc,
                sm_count=sm_count,
                is_available=True,
                has_bf16=has_bf16,
                has_fp8=has_fp8,
            )
        except Exception as exc:
            return CUDADeviceInfo(
                device_name=f"error:{exc}",
                vram_gb=0.0,
                cuda_version="0.0",
                driver_version="0.0",
                compute_capability="0.0",
                sm_count=0,
                is_available=False,
                has_bf16=False,
                has_fp8=False,
            )

    @staticmethod
    def _read_driver_version() -> str:
        """Read NVIDIA driver version from /proc/driver/nvidia/version if present."""
        import os
        path = "/proc/driver/nvidia/version"
        try:
            if os.path.exists(path):
                line = open(path).readline()
                # "NVRM version: NVIDIA UNIX Open Kernel Module  535.86.10 ..."
                for tok in line.split():
                    parts = tok.split(".")
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        return tok
        except OSError:
            pass
        return "unknown"

    @staticmethod
    def _tensorrt_available() -> bool:
        """Return True if TensorRT Python bindings are importable."""
        try:
            import tensorrt  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _recommend_batch_size(vram_gb: float) -> int:
        if vram_gb >= 80:
            return 64
        if vram_gb >= 40:
            return 32
        if vram_gb >= 24:
            return 16
        if vram_gb >= 16:
            return 8
        if vram_gb >= 8:
            return 4
        return 1

    def __repr__(self) -> str:
        info = self._info
        if info is None:
            return "CUDABackend(detected=False)"
        return (
            f"CUDABackend("
            f"cc={info.compute_capability}, "
            f"vram={info.vram_gb}GB, "
            f"available={info.is_available}, "
            f"path={self.get_kernel_path().name})"
        )
