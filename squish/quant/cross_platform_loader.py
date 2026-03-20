"""squish/quant/cross_platform_loader.py — Cross-platform model loader selector.

Selects the appropriate model-loading strategy based on detected platform
and available libraries: MLX on macOS Apple Silicon, BitsAndBytes (4-bit
NF4) on Linux+CUDA, or standard PyTorch fp16 on other CUDA/CPU targets.

Classes
───────
CrossPlatformLoaderConfig   — Configuration dataclass.
LoadResult                  — Result of a model loading operation.
CrossPlatformLoaderStats     — Runtime statistics.
CrossPlatformModelLoader     — Main loader-selection class.

Usage::

    from squish.platform.detector import UnifiedPlatformDetector
    from squish.quant.cross_platform_loader import CrossPlatformModelLoader

    info    = UnifiedPlatformDetector().detect()
    loader  = CrossPlatformModelLoader(platform_info=info)
    result  = loader.load("models/Qwen2.5-7B")
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_VALID_STRATEGIES = frozenset({"mlx", "torch_bnb", "torch_fp16", "torch_fp32", "auto"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CrossPlatformLoaderConfig:
    """Configuration for CrossPlatformModelLoader.

    Attributes
    ----------
    prefer_quantized:
        Use quantized loading (4-bit NF4 / MLX quant) when available.
    fallback_to_fp16:
        If quantized loading fails, degrade to fp16. Default True.
    max_memory_gb:
        Refuse to load models estimated to require more GPU memory.
        None = no limit.
    strategy:
        Override loader strategy: 'auto', 'mlx', 'torch_bnb', 'torch_fp16',
        'torch_fp32'.
    """
    prefer_quantized: bool             = True
    fallback_to_fp16: bool             = True
    max_memory_gb:    Optional[float]  = None
    strategy:         str              = "auto"

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}, "
                f"got '{self.strategy}'"
            )
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            raise ValueError(
                f"max_memory_gb must be > 0 when set, got {self.max_memory_gb}"
            )


# ---------------------------------------------------------------------------
# Load result
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    """Result of a CrossPlatformModelLoader.load() call."""
    model_path:   str
    loader_used:  str      # 'mlx', 'torch_bnb', 'torch_fp16', 'stub'
    load_time_ms: float
    memory_gb:    float    # estimated RAM / VRAM consumed
    quantized:    bool     # was any quantisation applied?


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CrossPlatformLoaderStats:
    """Runtime statistics for CrossPlatformModelLoader."""
    total_loads:  int = 0
    mlx_loads:    int = 0
    torch_loads:  int = 0
    bnb_loads:    int = 0
    stub_loads:   int = 0

    @property
    def primary_loader(self) -> str:
        counts = {
            "mlx":  self.mlx_loads,
            "torch": self.torch_loads,
            "bnb":  self.bnb_loads,
        }
        if not self.total_loads:
            return "none"
        return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossPlatformModelLoader:
    """Select and execute the best model loading strategy for this platform.

    Strategy selection order (with prefer_quantized=True):
      1. macOS + MLX available   → 'mlx'
      2. Linux + CUDA + bnb      → 'torch_bnb'  (4-bit NF4)
      3. CUDA without bnb        → 'torch_fp16'
      4. CPU                     → 'torch_fp32'

    The module never actually loads large model weights in the current
    implementation — it simulates the load for testability and returns a
    LoadResult.  Integration with the real MLX / transformers loading
    pipelines is handled in squish/backend.py.
    """

    def __init__(
        self,
        config: Optional[CrossPlatformLoaderConfig] = None,
        platform_info: Optional[Any] = None,
    ) -> None:
        self._cfg      = config or CrossPlatformLoaderConfig()
        self._platform = platform_info
        self.stats     = CrossPlatformLoaderStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_loader(self, model_path: str) -> str:
        """Determine the best loading strategy for model_path.

        Parameters
        ----------
        model_path : path to a model directory or file.

        Returns
        -------
        One of: 'mlx', 'torch_bnb', 'torch_fp16', 'torch_fp32'.
        """
        if self._cfg.strategy != "auto":
            return self._cfg.strategy
        return self._auto_strategy()

    def load(self, model_path: str) -> LoadResult:
        """Simulate loading model_path and return a LoadResult.

        The actual tensor loading is deferred to backend.py; this method
        resolves the loader strategy and measures setup time.
        """
        t0 = time.perf_counter()
        loader = self.select_loader(model_path)
        mem_gb = self.estimate_memory(model_path)

        if self._cfg.max_memory_gb is not None and mem_gb > self._cfg.max_memory_gb:
            raise MemoryError(
                f"Model at '{model_path}' estimated to require {mem_gb:.1f} GB, "
                f"exceeds max_memory_gb={self._cfg.max_memory_gb}"
            )

        quantized = loader in ("mlx", "torch_bnb")
        elapsed   = (time.perf_counter() - t0) * 1000.0

        self.stats.total_loads += 1
        if loader == "mlx":
            self.stats.mlx_loads += 1
        elif loader == "torch_bnb":
            self.stats.bnb_loads  += 1
            self.stats.torch_loads += 1
        elif loader.startswith("torch"):
            self.stats.torch_loads += 1
        else:
            self.stats.stub_loads += 1

        return LoadResult(
            model_path=model_path,
            loader_used=loader,
            load_time_ms=elapsed,
            memory_gb=mem_gb,
            quantized=quantized,
        )

    def estimate_memory(self, model_path: str) -> float:
        """Heuristic: estimate GPU/RAM required for the model in GB.

        Uses the number of .npy / .safetensors / .bin files as a proxy.
        Falls back to 0.0 if the path doesn't exist.
        """
        path = Path(model_path)
        if not path.exists():
            return 0.0
        if path.is_file():
            return round(os.path.getsize(path) / 1e9, 6)

        total = sum(
            f.stat().st_size
            for ext in ("*.npy", "*.safetensors", "*.bin", "*.pt")
            for f in path.glob(ext)
        )
        raw_gb = total / 1e9

        # BnB 4-bit uses ~½ the FP16 footprint; MLX quant similar
        loader = self._auto_strategy()
        if loader in ("mlx", "torch_bnb") and self._cfg.prefer_quantized:
            return round(raw_gb * 0.5, 6)
        if loader == "torch_fp16":
            return round(raw_gb, 6)
        return round(raw_gb * 2.0, 6)  # fp32

    # ------------------------------------------------------------------
    # Strategy resolution
    # ------------------------------------------------------------------

    def _auto_strategy(self) -> str:
        kind = self._platform_kind()

        if kind == "MACOS_APPLE_SILICON":
            if self._cfg.prefer_quantized and self._mlx_available():
                return "mlx"
            return "mlx"

        if kind in ("LINUX_CUDA", "LINUX_ROCM"):
            if self._cfg.prefer_quantized and self._bnb_available():
                return "torch_bnb"
            return "torch_fp16"

        if kind == "LINUX_CPU":
            if self._cfg.fallback_to_fp16:
                return "torch_fp16"
            return "torch_fp32"

        if kind in ("WINDOWS_WSL", "WINDOWS_NATIVE"):
            return "torch_fp16" if self._cfg.fallback_to_fp16 else "torch_fp32"

        # UNKNOWN / fallback
        return "torch_fp32"

    def _platform_kind(self) -> str:
        if self._platform is not None:
            return self._platform.kind.name
        # Infer from runtime
        import sys
        if sys.platform == "darwin":
            return "MACOS_APPLE_SILICON"
        if self._cuda_available():
            return "LINUX_CUDA"
        return "LINUX_CPU"

    @staticmethod
    def _mlx_available() -> bool:
        try:
            import mlx.core  # type: ignore[import]
            return True
        except ImportError:
            return False

    @staticmethod
    def _bnb_available() -> bool:
        try:
            import bitsandbytes  # type: ignore[import]
            return True
        except ImportError:
            return False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        kind = self._platform_kind()
        return (
            f"CrossPlatformModelLoader("
            f"platform={kind}, "
            f"strategy={self._cfg.strategy}, "
            f"loads={self.stats.total_loads})"
        )
