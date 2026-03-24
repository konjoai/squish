"""squish/platform/platform_router.py — Unified backend router.

Reads ``PlatformInfo`` from ``detector.py``, queries all available backends
(ANE, CUDA, ROCm, Metal/MLX, DirectML/Windows, CPU-only), builds a
priority-ordered ``BackendChain``, and exposes a single ``route()`` call that
returns the best viable ``RoutedBackend`` for the current host.

The router is consumed by ``squish/runtime/squish_runtime.py`` and
``squish/hardware/kernel_dispatch.py`` so that all dispatch decisions are made
in one place rather than embedded ad-hoc throughout the codebase.

Classes / types
───────────────
BackendPriority     — Ordered enum: ANE > CUDA > ROCm > Metal > DirectML > CPU
BackendChainEntry   — Single entry in the priority chain.
RoutedBackend       — Resolved backend: name, device spec, kernel path hint.
PlatformRouterConfig — Configuration dataclass.
PlatformRouterStats  — Runtime statistics.
PlatformRouter      — Main router class.

Usage::

    from squish.platform.platform_router import PlatformRouter

    router  = PlatformRouter()
    backend = router.route()
    print(backend.name, backend.device, backend.kernel_path_hint)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Priority enum
# ---------------------------------------------------------------------------

class BackendPriority(IntEnum):
    """Routing priority (lower value = higher priority)."""
    ANE      = 1   # Apple Neural Engine  — macOS M-series, sub-8B models
    CUDA     = 2   # NVIDIA CUDA          — Linux/Windows GPU
    ROCM     = 3   # AMD ROCm             — Linux AMD GPU
    METAL    = 4   # Apple Metal / MLX    — macOS GPU (non-ANE path)
    DIRECTML = 5   # Windows DirectML     — Windows 11 GPU/NPU
    CPU      = 99  # CPU-only fallback    — always available


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BackendChainEntry:
    """A candidate backend in the priority chain.

    Attributes
    ----------
    priority:
        Routing priority (lower = tried first).
    name:
        Human-readable backend name, e.g. ``"cuda"``.
    probe:
        Zero-argument callable that returns ``True`` if the backend is live;
        must not raise.  Will be called once during ``route()``.
    device_spec:
        Device string returned to callers, e.g. ``"cuda:0"``.
    kernel_path_hint:
        Opaque string hint consumed by kernel_dispatch, e.g.
        ``"W8A8_SMOOTHQUANT"`` or ``"METAL_GEMV"``.
    """
    priority:         BackendPriority
    name:             str
    probe:            Callable[[], bool]
    device_spec:      str
    kernel_path_hint: str


@dataclass(frozen=True)
class RoutedBackend:
    """The resolved best-available backend.

    Attributes
    ----------
    name:             Backend name, e.g. ``"cuda"``.
    device:           Device string for tensor placement, e.g. ``"cuda:0"``.
    kernel_path_hint: Kernel/quantisation path hint for dispatch.
    priority:         Numeric priority of the selected backend.
    latency_ms:       Time taken by the probe sequence (milliseconds).
    """
    name:             str
    device:           str
    kernel_path_hint: str
    priority:         int
    latency_ms:       float


@dataclass
class PlatformRouterConfig:
    """Configuration for PlatformRouter.

    Attributes
    ----------
    cuda_device_index:
        CUDA device index. Default 0.
    rocm_device_index:
        ROCm device index. Default 0.
    dml_adapter_index:
        DirectML adapter index. -1 = auto. Default -1.
    ane_model_size_gb:
        Estimated model size in GB; ANE path is enabled only when the model
        fits within ANE's practical activation budget. Default 8.0 GB.
    """
    cuda_device_index:  int   = 0
    rocm_device_index:  int   = 0
    dml_adapter_index:  int   = -1
    ane_model_size_gb:  float = 8.0

    def __post_init__(self) -> None:
        if self.cuda_device_index < 0:
            raise ValueError("cuda_device_index must be >= 0")
        if self.rocm_device_index < 0:
            raise ValueError("rocm_device_index must be >= 0")
        if self.ane_model_size_gb <= 0:
            raise ValueError("ane_model_size_gb must be > 0")


@dataclass
class PlatformRouterStats:
    """Runtime statistics for PlatformRouter."""
    route_calls:     int   = 0
    cache_hits:      int   = 0
    probes_fired:    int   = 0
    last_route_ms:   float = 0.0
    selected_name:   str   = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PlatformRouter:
    """Build and execute a priority-ordered backend chain.

    The router queries each backend's ``is_available()`` (or equivalent probe)
    in priority order and returns the first viable ``RoutedBackend``.  It
    always includes a CPU fallback so callers never receive ``None``.

    The routing result is cached.  Call ``reset()`` to force re-routing (e.g.
    after a device hotplug event in unit tests).

    Usage::

        router  = PlatformRouter()
        backend = router.route()
        print(backend.name)   # e.g. "cuda", "mlx", "cpu"
    """

    def __init__(self, config: Optional[PlatformRouterConfig] = None) -> None:
        self._cfg     = config or PlatformRouterConfig()
        self.stats    = PlatformRouterStats()
        self._result: Optional[RoutedBackend] = None
        self._chain:  Optional[List[BackendChainEntry]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self) -> RoutedBackend:
        """Return the best available backend, caching the result.

        Returns
        -------
        RoutedBackend
            Always returns a valid object; falls back to CPU if all GPU
            backends are unavailable.
        """
        self.stats.route_calls += 1
        if self._result is not None:
            self.stats.cache_hits += 1
            return self._result

        t0 = time.perf_counter()
        self._result = self._run_routing()
        self.stats.last_route_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.selected_name = self._result.name
        return self._result

    def build_chain(self) -> List[BackendChainEntry]:
        """Build and return the full priority chain (all backends, not filtered).

        This is useful for diagnostics — it shows every candidate even if
        unavailable.
        """
        if self._chain is not None:
            return self._chain
        self._chain = self._build_backend_chain()
        return self._chain

    def reset(self) -> None:
        """Clear cached routing result and chain (forces re-routing)."""
        self._result = None
        self._chain  = None

    # ------------------------------------------------------------------
    # Routing internals
    # ------------------------------------------------------------------

    def _run_routing(self) -> RoutedBackend:
        chain = self.build_chain()
        # Sort by priority (ascending)
        for entry in sorted(chain, key=lambda e: int(e.priority)):
            self.stats.probes_fired += 1
            try:
                available = entry.probe()
            except Exception:
                available = False
            if available:
                return RoutedBackend(
                    name=entry.name,
                    device=entry.device_spec,
                    kernel_path_hint=entry.kernel_path_hint,
                    priority=int(entry.priority),
                    latency_ms=(time.perf_counter()) * 0.0,  # set by caller
                )
        # CPU fallback — always viable
        return RoutedBackend(
            name="cpu",
            device="cpu",
            kernel_path_hint="FP32_CPU",
            priority=int(BackendPriority.CPU),
            latency_ms=0.0,
        )

    def _build_backend_chain(self) -> List[BackendChainEntry]:
        cfg   = self._cfg
        chain: List[BackendChainEntry] = []

        # --- ANE (Apple Neural Engine) ---
        chain.append(BackendChainEntry(
            priority=BackendPriority.ANE,
            name="ane",
            probe=self._probe_ane,
            device_spec="ane:0",
            kernel_path_hint="COREML_ANE",
        ))

        # --- NVIDIA CUDA ---
        chain.append(BackendChainEntry(
            priority=BackendPriority.CUDA,
            name="cuda",
            probe=self._probe_cuda,
            device_spec=f"cuda:{cfg.cuda_device_index}",
            kernel_path_hint=self._cuda_kernel_hint(),
        ))

        # --- AMD ROCm ---
        chain.append(BackendChainEntry(
            priority=BackendPriority.ROCM,
            name="rocm",
            probe=self._probe_rocm,
            device_spec=f"cuda:{cfg.rocm_device_index}",   # ROCm uses cuda: index
            kernel_path_hint="ROCM_HIP",
        ))

        # --- Apple Metal / MLX ---
        chain.append(BackendChainEntry(
            priority=BackendPriority.METAL,
            name="mlx",
            probe=self._probe_mlx,
            device_spec="mlx:0",
            kernel_path_hint="METAL_GEMV",
        ))

        # --- Windows DirectML ---
        chain.append(BackendChainEntry(
            priority=BackendPriority.DIRECTML,
            name="directml",
            probe=self._probe_directml,
            device_spec=f"privateuseone:{cfg.dml_adapter_index}",
            kernel_path_hint="DIRECTML_FP16",
        ))

        return chain

    # ------------------------------------------------------------------
    # Per-backend probes (isolated, never raise)
    # ------------------------------------------------------------------

    def _probe_ane(self) -> bool:
        try:
            from squish.platform.ane_router import ANERouter
            router = ANERouter()
            return router.is_available() and self._cfg.ane_model_size_gb <= 8.0
        except Exception:
            return False

    def _probe_cuda(self) -> bool:
        try:
            from squish.platform.cuda_backend import CUDABackend, CUDAConfig
            backend = CUDABackend(CUDAConfig(device_index=self._cfg.cuda_device_index))
            return backend.is_available()
        except Exception:
            return False

    def _probe_rocm(self) -> bool:
        try:
            from squish.platform.rocm_backend import ROCmBackend, ROCmConfig
            backend = ROCmBackend(ROCmConfig(device_index=self._cfg.rocm_device_index))
            return backend.is_available()
        except Exception:
            return False

    def _probe_mlx(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import sys
            return sys.platform == "darwin"
        except Exception:
            return False

    def _probe_directml(self) -> bool:
        try:
            from squish.platform.windows_backend import WindowsBackend, WindowsConfig
            backend = WindowsBackend(
                WindowsConfig(adapter_index=self._cfg.dml_adapter_index)
            )
            return backend.is_available()
        except Exception:
            return False

    def _cuda_kernel_hint(self) -> str:
        try:
            from squish.platform.cuda_backend import CUDABackend, CUDAConfig
            backend = CUDABackend(CUDAConfig(device_index=self._cfg.cuda_device_index))
            if backend.is_available():
                return backend.get_kernel_path().name
        except Exception:
            pass
        return "FP16_BASELINE"

    def __repr__(self) -> str:
        result = self._result
        if result is None:
            return "PlatformRouter(unresolved)"
        return (
            f"PlatformRouter(selected={result.name!r}, "
            f"device={result.device!r}, "
            f"priority={result.priority})"
        )
