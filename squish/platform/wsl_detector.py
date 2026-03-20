"""squish/platform/wsl_detector.py — Windows Subsystem for Linux 2 detector.

Detects WSL2 by inspecting /proc/version, /proc/osrelease, and environment
variables.  Provides memory-limit and GPU-access queries for use in the
cross-platform server initialisation path.

Classes
───────
WSLConfig         — Configuration dataclass.
WSLInfo           — Detected WSL properties (frozen dataclass).
WSLDetectorStats  — Runtime statistics.
WSLDetector       — Main detector; call .detect() → WSLInfo.

Usage::

    det  = WSLDetector()
    info = det.detect()
    if info.is_wsl:
        print(info.wsl_version, info.distro_name, info.has_gpu)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WSLConfig:
    """Configuration for WSLDetector.

    Attributes
    ----------
    check_virtio_gpu:
        Verify virtio-gpu / D3D12 presence for GPU forwarding. Default True.
    check_memory_limit:
        Parse /proc/meminfo and cgroup limits to establish WSL2 memory cap.
    """
    check_virtio_gpu:    bool = True
    check_memory_limit:  bool = True


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WSLInfo:
    """Detected WSL2 environment properties."""
    is_wsl:          bool
    wsl_version:     int            # 1 or 2 (0 if not WSL)
    distro_name:     str            # e.g. "Ubuntu-22.04" or ""
    has_gpu:         bool           # virtio-gpu / D3D12 detected
    memory_limit_gb: float          # WSL cgroup memory limit (0.0 = unbounded)
    kernel_version:  str            # e.g. "5.15.153.1-microsoft-standard-WSL2"
    windows_version: str            # e.g. "Windows 11" or ""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class WSLDetectorStats:
    """Runtime statistics for WSLDetector."""
    detection_calls: int   = 0
    cache_hits:      int   = 0
    last_detect_ms:  float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_NOT_WSL = WSLInfo(
    is_wsl=False, wsl_version=0, distro_name="",
    has_gpu=False, memory_limit_gb=0.0,
    kernel_version="", windows_version="",
)


class WSLDetector:
    """Detect WSL2 and capability information.

    On bare-metal Linux and macOS this will return a WSLInfo with
    ``is_wsl=False`` immediately.  All probes are read-only and never
    write to the file system.

    Usage::

        det  = WSLDetector()
        info = det.detect()
        if info.is_wsl and info.has_gpu:
            # WSL2 + GPU forwarding available
            ...
    """

    def __init__(self, config: Optional[WSLConfig] = None) -> None:
        self._cfg  = config or WSLConfig()
        self.stats = WSLDetectorStats()
        self._info: Optional[WSLInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> WSLInfo:
        """Detect WSL2 environment; result is cached after first call."""
        self.stats.detection_calls += 1
        if self._info is not None:
            self.stats.cache_hits += 1
            return self._info

        t0 = time.perf_counter()
        self._info = self._run_detection()
        self.stats.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        return self._info

    def get_memory_limit_gb(self) -> float:
        """Return the WSL2 memory cap in GB, or 0.0 if unbounded / not WSL."""
        return self.detect().memory_limit_gb

    def has_gpu_access(self) -> bool:
        """Return True if WSL2 GPU forwarding (virtio-gpu / D3D12) is active."""
        return self.detect().has_gpu

    def reset(self) -> None:
        """Clear cached result, forcing re-detection."""
        self._info = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_detection(self) -> WSLInfo:
        if sys.platform != "linux":
            return _NOT_WSL

        is_wsl, wsl_ver, kernel = self._check_proc_version()
        if not is_wsl:
            # Check WSL environment variable as secondary signal
            if "WSL_DISTRO_NAME" not in os.environ and "WSLENV" not in os.environ:
                return _NOT_WSL
            is_wsl  = True
            wsl_ver = 2
            kernel  = self._read_kernel_version()

        distro     = os.environ.get("WSL_DISTRO_NAME", "")
        win_ver    = self._read_windows_version()
        has_gpu    = self._check_gpu() if self._cfg.check_virtio_gpu else False
        mem_limit  = self._read_memory_limit() if self._cfg.check_memory_limit else 0.0

        return WSLInfo(
            is_wsl=is_wsl,
            wsl_version=wsl_ver,
            distro_name=distro,
            has_gpu=has_gpu,
            memory_limit_gb=mem_limit,
            kernel_version=kernel,
            windows_version=win_ver,
        )

    @staticmethod
    def _check_proc_version() -> tuple[bool, int, str]:
        """Parse /proc/version for WSL markers.  Returns (is_wsl, version, kernel)."""
        try:
            with open("/proc/version") as f:
                text = f.read().lower()
        except Exception:
            return False, 0, ""

        kernel = ""
        try:
            import platform
            kernel = platform.release()
        except Exception:
            pass

        if "microsoft-standard" in text or "wsl2" in text:
            return True, 2, kernel
        if "microsoft" in text:
            return True, 1, kernel
        return False, 0, kernel

    @staticmethod
    def _read_kernel_version() -> str:
        try:
            import platform
            return platform.release()
        except Exception:
            return ""

    @staticmethod
    def _read_windows_version() -> str:
        """Try to read /proc/sys/kernel/osrelease or /etc/os-release."""
        try:
            for path in ("/proc/sys/kernel/osrelease",):
                try:
                    return open(path).read().strip()
                except Exception:
                    pass
        except Exception:
            pass
        return ""

    @staticmethod
    def _check_gpu() -> bool:
        """Check for /dev/dxg (WSL2 GPU forwarding device)."""
        # /dev/dxg is the D3D12 device interface exposed by WSL2
        if os.path.exists("/dev/dxg"):
            return True
        # Alternatively check for Mesa / virtio-gpu
        try:
            import subprocess
            r = subprocess.run(
                ["ls", "/dev/dri"],
                capture_output=True, text=True, timeout=1,
            )
            return r.returncode == 0 and "renderD" in r.stdout
        except Exception:
            return False

    @staticmethod
    def _read_memory_limit() -> float:
        """Read WSL2 cgroup memory limit in GB (0.0 = no limit / native)."""
        _HUGE = 9_223_372_036_854_771_712
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ):
            try:
                raw = open(path).read().strip()
                if raw == "max":
                    return 0.0
                val = int(raw)
                if val >= _HUGE:
                    return 0.0
                return round(val / 1e9, 2)
            except Exception:
                continue
        return 0.0

    def __repr__(self) -> str:
        info = self._info
        if info is None:
            return "WSLDetector(not yet detected)"
        return (
            f"WSLDetector("
            f"is_wsl={info.is_wsl}, "
            f"version={info.wsl_version}, "
            f"has_gpu={info.has_gpu}, "
            f"mem_limit={info.memory_limit_gb}GB)"
        )
