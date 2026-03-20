"""squish/serving/linux_server_init.py — Linux inference server initialisation.

Configures the runtime environment for serving on Linux: CUDA device
selection, memory fraction pinning, thread pool sizing, and TF32 policy.

Falls back gracefully to CPU mode when CUDA is not available, so the
module can be imported and unit-tested on macOS.

Classes
───────
LinuxServerConfig   — Configuration dataclass.
LinuxInitResult     — Result of a successful initialisation.
LinuxServerStats    — Runtime statistics.
LinuxServerInit     — Main initialiser class.

Usage::

    cfg  = LinuxServerConfig(cuda_device="auto", memory_fraction=0.90)
    init = LinuxServerInit(cfg)
    res  = init.initialize()
    print(res.device, res.backend_name, res.memory_limit_gb)
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

_VALID_DEVICES = frozenset({"auto", "cpu"})


@dataclass
class LinuxServerConfig:
    """Configuration for LinuxServerInit.

    Attributes
    ----------
    cuda_device:
        'auto' selects cuda:0, 'cpu' forces CPU mode, or 'cuda:N' for index N.
    memory_fraction:
        Fraction of GPU memory to allow PyTorch to allocate. Range (0, 1].
    num_cpu_threads:
        Number of OMP/MKL threads. None → use os.cpu_count().
    enable_tf32:
        Allow TF32 matmuls on Ampere+ GPUs (improves throughput slightly).
    """
    cuda_device:       str            = "auto"
    memory_fraction:   float          = 0.90
    num_cpu_threads:   Optional[int]  = None
    enable_tf32:       bool           = True

    def __post_init__(self) -> None:
        if not (0.0 < self.memory_fraction <= 1.0):
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )
        if self.num_cpu_threads is not None and self.num_cpu_threads < 1:
            raise ValueError(
                f"num_cpu_threads must be >= 1, got {self.num_cpu_threads}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LinuxInitResult:
    """Result of a LinuxServerInit.initialize() call."""
    device:          str    # e.g. 'cuda:0' or 'cpu'
    backend_name:    str    # 'cuda' / 'rocm' / 'cpu'
    memory_limit_gb: float  # pinned GPU memory in GB (0.0 on CPU)
    num_threads:     int    # OMP/MKL threads configured
    init_ms:         float  # milliseconds taken


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class LinuxServerStats:
    """Runtime statistics for LinuxServerInit."""
    init_calls:  int = 0
    cuda_inits:  int = 0
    cpu_inits:   int = 0

    @property
    def total_inits(self) -> int:
        return self.cuda_inits + self.cpu_inits


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LinuxServerInit:
    """Initialise the Linux inference serving environment.

    Configures PyTorch CUDA settings, memory fraction, thread pool, and
    TF32 policy.  All configuration is idempotent — calling initialize()
    multiple times is safe.

    Usage::

        init = LinuxServerInit(LinuxServerConfig(memory_fraction=0.85))
        result = init.initialize()
        # result.device → 'cuda:0' or 'cpu'
    """

    def __init__(self, config: Optional[LinuxServerConfig] = None) -> None:
        self._cfg   = config or LinuxServerConfig()
        self.stats  = LinuxServerStats()
        self._last_result: Optional[LinuxInitResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> LinuxInitResult:
        """Run the full initialisation sequence and return a result object."""
        t0 = time.perf_counter()
        self.stats.init_calls += 1

        device, backend = self._resolve_device()
        mem_gb          = self._configure_cuda(device) if "cuda" in device else 0.0
        n_threads       = self._configure_cpu_threads()

        result = LinuxInitResult(
            device=device,
            backend_name=backend,
            memory_limit_gb=mem_gb,
            num_threads=n_threads,
            init_ms=(time.perf_counter() - t0) * 1000.0,
        )
        if "cuda" in device:
            self.stats.cuda_inits += 1
        else:
            self.stats.cpu_inits += 1

        self._last_result = result
        return result

    def get_recommended_batch_size(self) -> int:
        """Heuristic batch size based on available GPU memory."""
        if self._last_result is None:
            self.initialize()
        assert self._last_result is not None
        mem = self._last_result.memory_limit_gb
        if mem >= 80:
            return 64
        if mem >= 40:
            return 32
        if mem >= 20:
            return 16
        if mem >= 8:
            return 8
        return 1  # CPU or very small GPU

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        """Return (device_str, backend_name)."""
        requested = self._cfg.cuda_device.lower()
        if requested == "cpu":
            return "cpu", "cpu"

        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu", "cpu"
            # ROCm detection
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            backend = "rocm" if is_rocm else "cuda"
            if requested == "auto":
                return "cuda:0", backend
            # explicit 'cuda:N'
            idx = int(requested.split(":")[-1]) if ":" in requested else 0
            n   = torch.cuda.device_count()
            if idx >= n:
                raise ValueError(
                    f"CUDA device {idx} requested but only {n} device(s) found"
                )
            return f"cuda:{idx}", backend
        except ImportError:
            return "cpu", "cpu"

    def _configure_cuda(self, device: str) -> float:
        """Set memory fraction and TF32 policy; return memory limit in GB."""
        try:
            import torch
            idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_per_process_memory_fraction(
                self._cfg.memory_fraction, device=idx
            )
            if self._cfg.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32        = True
            props = torch.cuda.get_device_properties(idx)
            return round(props.total_memory * self._cfg.memory_fraction / 1e9, 1)
        except Exception:
            return 0.0

    def _configure_cpu_threads(self) -> int:
        """Set OMP threads and return count used."""
        n = self._cfg.num_cpu_threads or os.cpu_count() or 1
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
        try:
            import torch
            torch.set_num_threads(n)
        except Exception:
            pass
        return n

    def __repr__(self) -> str:
        last = self._last_result
        device = last.device if last else "not-initialized"
        return (
            f"LinuxServerInit("
            f"device={device}, "
            f"inits={self.stats.total_inits})"
        )
