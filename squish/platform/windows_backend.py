"""squish/platform/windows_backend.py — Windows DirectML GPU backend.

Detects Windows DirectML presence (via ``torch_directml`` or the standalone
``directml`` package) and provides adapter enumeration and recommended serving
configuration for Windows 11 / Windows 10 DirectML-accelerated inference.

When running under WSL2, the backend delegates to the ``wsl_detector`` and
recommends forwarding to the CUDA backend instead of using DirectML directly.

Classes
───────
WindowsConfig      — Configuration dataclass.
DMLAdapterInfo     — Detected DirectML adapter properties.
WindowsBackendStats — Runtime statistics.
WindowsBackend      — Main backend class.

Usage::

    backend = WindowsBackend()
    if backend.is_available():
        adapters = backend.enumerate_adapters()
        best     = backend.best_adapter()
        cfg      = backend.get_recommended_config()
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowsConfig:
    """Configuration for WindowsBackend.

    Attributes
    ----------
    adapter_index:
        DirectML adapter index to use. -1 means auto-select. Default -1.
    memory_fraction:
        GPU memory fraction for serving. Range (0, 1]. Default 0.85.
    allow_wsl2_fallback:
        If running under WSL2, allow fallback to CUDA backend recommendation
        instead of DirectML. Default True.
    """
    adapter_index:       int   = -1
    memory_fraction:     float = 0.85
    allow_wsl2_fallback: bool  = True

    def __post_init__(self) -> None:
        if self.adapter_index < -1:
            raise ValueError(
                f"adapter_index must be >= -1, got {self.adapter_index}"
            )
        if not (0.0 < self.memory_fraction <= 1.0):
            raise ValueError(
                f"memory_fraction must be in (0, 1], "
                f"got {self.memory_fraction}"
            )


@dataclass(frozen=True)
class DMLAdapterInfo:
    """Properties of a detected DirectML GPU / NPU adapter."""
    adapter_index:    int
    adapter_name:     str
    vram_gb:          float   # Dedicated GPU memory (0.0 for integrated)
    is_discrete:      bool    # True for dedicated GPU, False for iGPU / NPU
    vendor_id:        int     # PCI vendor ID (0x10DE = NVIDIA, 0x1002 = AMD …)
    is_available:     bool


@dataclass
class WindowsBackendStats:
    """Runtime statistics for WindowsBackend."""
    detection_calls: int   = 0
    cache_hits:      int   = 0
    last_detect_ms:  float = 0.0
    adapters_found:  int   = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WindowsBackend:
    """Detect and describe DirectML GPU for Squish inference on Windows.

    On non-Windows machines (macOS, Linux bare-metal) ``is_available()``
    returns ``False``.  Under WSL2, availability depends on the
    ``allow_wsl2_fallback`` flag: if True, inference is routed through CUDA
    even when DirectML is importable.

    Usage::

        backend = WindowsBackend(WindowsConfig(adapter_index=-1))
        if backend.is_available():
            best = backend.best_adapter()
            print(best.adapter_name, best.vram_gb)
    """

    _NO_ADAPTER = DMLAdapterInfo(
        adapter_index=0,
        adapter_name="N/A",
        vram_gb=0.0,
        is_discrete=False,
        vendor_id=0,
        is_available=False,
    )

    def __init__(self, config: Optional[WindowsConfig] = None) -> None:
        self._cfg      = config or WindowsConfig()
        self.stats     = WindowsBackendStats()
        self._adapters: Optional[List[DMLAdapterInfo]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if DirectML serving is viable on this host.

        Conditions:
        - Running on Windows (sys.platform == "win32") OR DirectML package is
          importable (catches WSL2 with directml installed).
        - ``torch_directml`` or ``directml`` is importable.
        - If WSL2 context detected and ``allow_wsl2_fallback`` is True,
          returns False so callers use the CUDA backend instead.
        """
        if self._cfg.allow_wsl2_fallback and self._is_wsl2():
            return False
        return self._check_dml_present()

    def enumerate_adapters(self) -> List[DMLAdapterInfo]:
        """Enumerate all DirectML-visible adapters (GPU + NPU + iGPU)."""
        self.stats.detection_calls += 1
        if self._adapters is not None:
            self.stats.cache_hits += 1
            return self._adapters

        t0 = time.perf_counter()
        self._adapters = self._run_enumeration()
        self.stats.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.adapters_found = len(self._adapters)
        return self._adapters

    def best_adapter(self) -> DMLAdapterInfo:
        """Return the best available adapter (discrete > integrated, most VRAM)."""
        adapters = self.enumerate_adapters()
        if not adapters:
            return self._NO_ADAPTER

        # If a specific index was requested, honour it
        if self._cfg.adapter_index >= 0:
            for a in adapters:
                if a.adapter_index == self._cfg.adapter_index:
                    return a
            # index out of range — fall through to auto-select
            return adapters[0]

        # Auto-select: prefer discrete, then by VRAM descending
        discrete  = [a for a in adapters if a.is_discrete]
        candidate = discrete if discrete else adapters
        return max(candidate, key=lambda a: a.vram_gb)

    def get_device_map(self) -> dict:
        """Return a device map compatible with HuggingFace ``device_map`` arg."""
        adapter = self.best_adapter()
        if not adapter.is_available:
            return {"": "cpu"}
        idx = adapter.adapter_index
        return {"": f"privateuseone:{idx}"}

    def get_recommended_config(self) -> dict:
        """Return serving parameters recommended for the best DirectML adapter."""
        adapter = self.best_adapter()
        if not adapter.is_available:
            return {"device": "cpu", "dtype": "float32", "batch_size": 1}

        # DirectML currently supports FP16; BF16 depends on GPU driver version
        dtype = "float16"
        batch = self._recommend_batch_size(adapter.vram_gb)

        return {
            "device":           f"privateuseone:{adapter.adapter_index}",
            "dtype":            dtype,
            "batch_size":       batch,
            "memory_fraction":  self._cfg.memory_fraction,
            "adapter_name":     adapter.adapter_name,
            "is_discrete_gpu":  adapter.is_discrete,
        }

    def reset(self) -> None:
        """Clear cached enumeration (forces re-detection on next call)."""
        self._adapters = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dml_present() -> bool:
        """Return True if torch_directml or directml is importable."""
        for pkg in ("torch_directml", "directml"):
            try:
                __import__(pkg)
                return True
            except ImportError:
                continue
        # On Windows, also check if the OS is Windows directly
        return sys.platform == "win32"

    @staticmethod
    def _is_wsl2() -> bool:
        """Return True if running inside Windows Subsystem for Linux 2."""
        try:
            from squish.platform.wsl_detector import WSLDetector
            return WSLDetector().is_wsl2()
        except Exception:
            pass
        # Fallback: check /proc/version for "microsoft" marker
        try:
            proc_version = "/proc/version"
            if os.path.exists(proc_version):
                content = open(proc_version).read().lower()
                return "microsoft" in content and "wsl2" in content
        except OSError:
            pass
        return False

    def _run_enumeration(self) -> List[DMLAdapterInfo]:
        """Enumerate DirectML adapters, returning an empty list on failure."""
        adapters: List[DMLAdapterInfo] = []

        try:
            import torch_directml as dml  # type: ignore[import]
            count = dml.device_count()
            for i in range(count):
                adapter_name = dml.device_name(i)
                # torch_directml exposes dedicated memory in some versions
                vram = 0.0
                try:
                    vram = round(dml.dedicated_video_memory(i) / 1e9, 1)
                except AttributeError:
                    pass
                # Heuristic: discrete GPU has >0 dedicated VRAM
                is_discrete = vram > 0.1
                adapters.append(DMLAdapterInfo(
                    adapter_index=i,
                    adapter_name=adapter_name,
                    vram_gb=vram,
                    is_discrete=is_discrete,
                    vendor_id=0,   # torch_directml does not expose vendor ID
                    is_available=True,
                ))
            return adapters
        except ImportError:
            pass

        # Fallback: enumerate via WMI on native Windows
        if sys.platform == "win32":
            return self._enumerate_via_wmi()

        return []

    @staticmethod
    def _enumerate_via_wmi() -> List[DMLAdapterInfo]:
        """Enumerate GPU adapters on Windows via WMI when torch_directml is absent."""
        adapters: List[DMLAdapterInfo] = []
        try:
            import subprocess
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-WmiObject Win32_VideoController | "
                    "Select-Object Name,AdapterRAM,PNPDeviceID | "
                    "ConvertTo-Json"
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return adapters
            import json
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]   # single adapter returns object, not list
            for i, entry in enumerate(data):
                name     = entry.get("Name", f"GPU {i}")
                ram_b    = entry.get("AdapterRAM") or 0
                vram_gb  = round(int(ram_b) / 1e9, 1)
                pnp_id   = entry.get("PNPDeviceID", "")
                vendor   = 0
                if "VEN_10DE" in pnp_id:
                    vendor = 0x10DE   # NVIDIA
                elif "VEN_1002" in pnp_id:
                    vendor = 0x1002   # AMD
                elif "VEN_8086" in pnp_id:
                    vendor = 0x8086   # Intel
                adapters.append(DMLAdapterInfo(
                    adapter_index=i,
                    adapter_name=name,
                    vram_gb=vram_gb,
                    is_discrete=vram_gb >= 0.5,
                    vendor_id=vendor,
                    is_available=True,
                ))
        except Exception:
            pass
        return adapters

    @staticmethod
    def _recommend_batch_size(vram_gb: float) -> int:
        if vram_gb >= 16:
            return 8
        if vram_gb >= 8:
            return 4
        if vram_gb >= 4:
            return 2
        return 1

    def __repr__(self) -> str:
        adapters = self._adapters
        if adapters is None:
            return "WindowsBackend(detected=False)"
        best = self.best_adapter()
        return (
            f"WindowsBackend("
            f"adapters={len(adapters)}, "
            f"best={best.adapter_name!r}, "
            f"vram={best.vram_gb}GB)"
        )
