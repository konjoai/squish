"""squish/platform/detector.py — Unified platform and hardware detector.

Detects the host platform (macOS Apple Silicon, Linux CUDA, Linux ROCm,
Linux CPU-only, Windows/WSL2) and exposes a structured PlatformInfo
dataclass used by all downstream platform-adaptive code paths.

Classes
───────
PlatformKind              — Enum of supported platform kinds.
CUDAInfo                  — CUDA device properties.
PlatformInfo              — Complete platform description (frozen dataclass).
PlatformDetectorStats     — Detection latency and cache stats.
UnifiedPlatformDetector   — Main detector; call .detect() to populate.

Usage::

    from squish.platform.detector import UnifiedPlatformDetector

    detector = UnifiedPlatformDetector()
    info = detector.detect()
    print(info.kind)       # PlatformKind.MACOS_APPLE_SILICON
    print(info.has_cuda)   # False on Apple
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Platform kind enum
# ---------------------------------------------------------------------------

class PlatformKind(Enum):
    """Enumeration of platform kinds squish can run on."""
    MACOS_APPLE_SILICON = auto()   # macOS + MLX + Metal
    LINUX_CUDA          = auto()   # Linux + NVIDIA CUDA
    LINUX_ROCM          = auto()   # Linux + AMD ROCm
    LINUX_CPU           = auto()   # Linux, CPU only
    WINDOWS_WSL         = auto()   # Windows Subsystem for Linux
    WINDOWS_NATIVE      = auto()   # Windows native Python
    UNKNOWN             = auto()   # Fallback / unsupported


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAInfo:
    """Properties of a detected CUDA device (first device, index 0)."""
    device_count:        int
    device_name:         str
    total_memory_gb:     float
    compute_capability:  str     # e.g. "8.0" for Ampere
    is_available:        bool


@dataclass(frozen=True)
class PlatformInfo:
    """Comprehensive, immutable platform description."""
    kind:            PlatformKind
    os_name:         str             # sys.platform string
    python_version:  str
    arch:            str             # "arm64" / "x86_64" / "amd64"
    has_mlx:         bool
    has_cuda:        bool
    has_rocm:        bool
    is_wsl:          bool
    cuda_info:       Optional[CUDAInfo]
    apple_chip:      str             # "Apple M3 Pro" or "" on non-Apple
    ram_gb:          float           # Total system RAM GB

    @property
    def is_apple_silicon(self) -> bool:
        """True when running on macOS Apple Silicon with MLX available."""
        return self.kind == PlatformKind.MACOS_APPLE_SILICON

    @property
    def is_cuda(self) -> bool:
        """True when a CUDA device is available (alias for has_cuda)."""
        return self.has_cuda

    @property
    def name(self) -> str:
        """Human-readable platform name, e.g. 'macos_apple_silicon'."""
        return self.kind.name.lower()

    @property
    def platform_name(self) -> str:
        """Descriptive platform string, e.g. 'Apple Silicon (M3 Pro)'."""
        if self.is_apple_silicon:
            chip = self.apple_chip or "Apple Silicon"
            return f"Apple Silicon ({chip})"
        if self.has_cuda:
            device = self.cuda_info.device_name if self.cuda_info else "CUDA"
            return f"Linux CUDA ({device})"
        if self.has_rocm:
            return "Linux ROCm (AMD)"
        if self.is_wsl:
            return "Windows (WSL2)"
        if self.os_name == "win32":
            return "Windows (native)"
        return f"Unknown ({self.os_name}/{self.arch})"


@dataclass
class PlatformDetectorStats:
    """Runtime statistics for UnifiedPlatformDetector calls."""
    detection_calls:    int   = 0
    cache_hits:         int   = 0
    last_detection_ms:  float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        if self.detection_calls == 0:
            return 0.0
        return self.cache_hits / self.detection_calls


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class UnifiedPlatformDetector:
    """Detect and describe the runtime hardware/software platform.

    The detection result is cached after the first call; subsequent calls
    are O(1).  Call ``reset()`` to force re-detection.

    Usage::

        detector = UnifiedPlatformDetector()
        info = detector.detect()

        if info.kind == PlatformKind.LINUX_CUDA:
            print("CUDA:", info.cuda_info.device_name)
        elif info.kind == PlatformKind.MACOS_APPLE_SILICON:
            print("Apple chip:", info.apple_chip)
    """

    def __init__(self) -> None:
        self._info:  Optional[PlatformInfo] = None
        self.stats = PlatformDetectorStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> PlatformInfo:
        """Return PlatformInfo (cached after first detection)."""
        self.stats.detection_calls += 1
        if self._info is not None:
            self.stats.cache_hits += 1
            return self._info
        t0 = time.perf_counter()
        self._info = self._run_detection()
        self.stats.last_detection_ms = (time.perf_counter() - t0) * 1000.0
        return self._info

    def reset(self) -> None:
        """Clear cached result, forcing re-detection on next call."""
        self._info = None

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _run_detection(self) -> PlatformInfo:
        import platform as _plat

        os_name  = sys.platform
        arch     = _plat.machine()
        py_ver   = _plat.python_version()

        has_mlx             = self._check_mlx()
        has_cuda, cuda_info = self._check_cuda()
        has_rocm            = self._check_rocm()
        is_wsl              = self._check_wsl()
        apple_chip          = self._read_apple_chip() if os_name == "darwin" else ""
        ram_gb              = self._read_ram_gb()

        # Classify platform kind
        if os_name == "darwin" and has_mlx:
            kind = PlatformKind.MACOS_APPLE_SILICON
        elif is_wsl:
            kind = PlatformKind.WINDOWS_WSL
        elif os_name.startswith("linux") and has_cuda:
            kind = PlatformKind.LINUX_CUDA
        elif os_name.startswith("linux") and has_rocm:
            kind = PlatformKind.LINUX_ROCM
        elif os_name.startswith("linux"):
            kind = PlatformKind.LINUX_CPU
        elif os_name == "win32":
            kind = PlatformKind.WINDOWS_NATIVE
        else:
            kind = PlatformKind.UNKNOWN

        return PlatformInfo(
            kind           = kind,
            os_name        = os_name,
            python_version = py_ver,
            arch           = arch,
            has_mlx        = has_mlx,
            has_cuda       = has_cuda,
            has_rocm       = has_rocm,
            is_wsl         = is_wsl,
            cuda_info      = cuda_info,
            apple_chip     = apple_chip,
            ram_gb         = ram_gb,
        )

    # ------------------------------------------------------------------
    # Library / hardware probes
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mlx() -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import mlx.core as mx  # type: ignore[import]
            mx.array([0], dtype=mx.int32)
            return True
        except Exception:
            return False

    @staticmethod
    def _check_cuda() -> Tuple[bool, Optional[CUDAInfo]]:
        try:
            import torch  # type: ignore[import]
            if not torch.cuda.is_available():
                return False, None
            n    = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if n > 0 else "unknown"
            props = torch.cuda.get_device_properties(0) if n > 0 else None
            mem_gb = round(props.total_memory / 1e9, 1) if props else 0.0
            cc     = f"{props.major}.{props.minor}" if props else "0.0"
            return True, CUDAInfo(
                device_count       = n,
                device_name        = name,
                total_memory_gb    = mem_gb,
                compute_capability = cc,
                is_available       = True,
            )
        except Exception:
            return False, None

    @staticmethod
    def _check_rocm() -> bool:
        try:
            import torch  # type: ignore[import]
            return (
                hasattr(torch, "version")
                and hasattr(torch.version, "hip")
                and torch.version.hip is not None
                and torch.cuda.is_available()
            )
        except Exception:
            return False

    @staticmethod
    def _check_wsl() -> bool:
        """Detect WSL2 via /proc/version content or environment variable."""
        try:
            with open("/proc/version") as fh:
                return "microsoft" in fh.read().lower()
        except Exception:
            import os
            return "WSL_DISTRO_NAME" in os.environ

    @staticmethod
    def _read_apple_chip() -> str:
        try:
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _read_ram_gb() -> float:
        try:
            if sys.platform == "darwin":
                import ctypes
                import ctypes.util
                libc = ctypes.CDLL(ctypes.util.find_library("c"))
                val  = ctypes.c_uint64(0)
                sz   = ctypes.c_size_t(8)
                libc.sysctlbyname(
                    b"hw.memsize",
                    ctypes.byref(val), ctypes.byref(sz),
                    None, 0,
                )
                return round(val.value / 1e9, 1)
            elif sys.platform.startswith("linux"):
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemTotal"):
                            return round(int(line.split()[1]) / 1e6, 1)
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._info is None:
            return "UnifiedPlatformDetector(not yet detected)"
        i = self._info
        return (
            f"UnifiedPlatformDetector("
            f"kind={i.kind.name!r}, arch={i.arch!r}, "
            f"ram_gb={i.ram_gb}, has_mlx={i.has_mlx}, "
            f"has_cuda={i.has_cuda}, has_rocm={i.has_rocm})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_detector = UnifiedPlatformDetector()


def detect_platform() -> PlatformInfo:
    """Return PlatformInfo for the current host (cached singleton).

    Convenience wrapper so callers don't need to instantiate
    ``UnifiedPlatformDetector`` directly::

        from squish.platform.detector import detect_platform
        info = detect_platform()
        print(info.is_apple_silicon)   # True on M-series Mac
        print(info.platform_name)      # "Apple Silicon (Apple M3 Pro)"
    """
    return _default_detector.detect()
