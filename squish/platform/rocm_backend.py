"""squish/platform/rocm_backend.py — AMD ROCm GPU backend.

Detects AMD ROCm presence and provides GCN/RDNA architecture info and
recommended serving configuration for ROCm-accelerated inference.

Classes
───────
ROCmConfig       — Configuration dataclass.
ROCmDeviceInfo   — Detected ROCm device properties.
ROCmStats        — Runtime statistics.
ROCmBackend      — Main backend class.

Usage::

    backend = ROCmBackend()
    if backend.is_available():
        info = backend.detect()
        cfg  = backend.get_recommended_config()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ROCmConfig:
    """Configuration for ROCmBackend.

    Attributes
    ----------
    device_index:
        HIP/CUDA-compat device index to interrogate. Default 0.
    memory_fraction:
        GPU memory fraction for serving. Range (0, 1]. Default 0.85.
    """
    device_index:     int   = 0
    memory_fraction:  float = 0.85

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
class ROCmDeviceInfo:
    """Properties of a detected ROCm GPU device."""
    device_name:   str
    vram_gb:       float
    rocm_version:  str    # e.g. "5.7.0"
    gcn_arch:      str    # e.g. "gfx90a" (MI250), "gfx1100" (RX 7xxx)
    is_available:  bool
    compute_units: int    # number of CUs / SMs


@dataclass
class ROCmStats:
    """Runtime statistics for ROCmBackend."""
    detection_calls: int   = 0
    cache_hits:      int   = 0
    last_detect_ms:  float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ROCmBackend:
    """Detect and describe AMD ROCm GPU for Squish inference.

    On non-ROCm machines (macOS, CPU Linux, NVIDIA CUDA) this class will
    report ``is_available() → False`` and return a zeroed-out DeviceInfo.

    Usage::

        backend = ROCmBackend(ROCmConfig(device_index=0))
        if backend.is_available():
            info = backend.detect()
            print(info.gcn_arch, info.vram_gb)
    """

    _NOT_AVAILABLE = ROCmDeviceInfo(
        device_name="N/A", vram_gb=0.0, rocm_version="0.0",
        gcn_arch="N/A", is_available=False, compute_units=0,
    )

    def __init__(self, config: Optional[ROCmConfig] = None) -> None:
        self._cfg   = config or ROCmConfig()
        self.stats  = ROCmStats()
        self._info: Optional[ROCmDeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if a ROCm device is present and torch.hip is active."""
        return self._check_rocm_present()

    def detect(self) -> ROCmDeviceInfo:
        """Detect and return ROCm device info (cached after first call)."""
        self.stats.detection_calls += 1
        if self._info is not None:
            self.stats.cache_hits += 1
            return self._info

        t0 = time.perf_counter()
        self._info = self._run_detection()
        self.stats.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        return self._info

    def get_recommended_config(self) -> dict:
        """Return a dict of recommended serving parameters for this device."""
        info = self.detect()
        if not info.is_available:
            return {"device": "cpu", "dtype": "float32", "batch_size": 1}

        # Architecture-based tuning
        is_cdna   = info.gcn_arch.startswith("gfx9")   # MI series (data centre)
        is_rdna3  = info.gcn_arch.startswith("gfx11")  # RX 7xxx consumer

        dtype = "float16"
        if is_cdna:
            dtype = "bfloat16"   # MI250/MI300 have high-perf bf16

        batch = 1
        if info.vram_gb >= 80:
            batch = 32
        elif info.vram_gb >= 40:
            batch = 16
        elif info.vram_gb >= 16:
            batch = 8
        elif info.vram_gb >= 8:
            batch = 4

        return {
            "device":              f"cuda:{self._cfg.device_index}",
            "dtype":               dtype,
            "batch_size":          batch,
            "memory_fraction":     self._cfg.memory_fraction,
            "gcn_arch":            info.gcn_arch,
            "flash_attn_support":  is_cdna or is_rdna3,
        }

    def reset(self) -> None:
        """Clear cached detection result."""
        self._info = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_rocm_present() -> bool:
        try:
            import torch
            return (
                hasattr(torch.version, "hip")
                and torch.version.hip is not None
                and torch.cuda.is_available()
            )
        except Exception:
            return False

    def _run_detection(self) -> ROCmDeviceInfo:
        if not self._check_rocm_present():
            return self._NOT_AVAILABLE
        try:
            import torch
            idx   = self._cfg.device_index
            props = torch.cuda.get_device_properties(idx)
            vram  = round(props.total_memory / 1e9, 1)
            name  = props.name

            # ROCm version from torch.version.hip
            hip_ver = getattr(torch.version, "hip", "0.0") or "0.0"
            hip_ver = str(hip_ver).split()[0]

            # GCN arch from gcnArchName property (ROCm >= 5.x)
            arch = getattr(props, "gcnArchName", "unknown")
            cus  = getattr(props, "multi_processor_count", 0)

            return ROCmDeviceInfo(
                device_name=name,
                vram_gb=vram,
                rocm_version=hip_ver,
                gcn_arch=arch,
                is_available=True,
                compute_units=cus,
            )
        except Exception as exc:
            return ROCmDeviceInfo(
                device_name=f"error:{exc}",
                vram_gb=0.0,
                rocm_version="0.0",
                gcn_arch="unknown",
                is_available=False,
                compute_units=0,
            )

    def __repr__(self) -> str:
        info = self._info
        if info is None:
            return f"ROCmBackend(detected=False)"
        return (
            f"ROCmBackend("
            f"arch={info.gcn_arch}, "
            f"vram={info.vram_gb}GB, "
            f"available={info.is_available})"
        )
