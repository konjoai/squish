"""squish/platform/feature_registry.py — Platform feature availability registry.

Maps each Squish optimization feature to its availability on the current
platform (NATIVE / EMULATED / UNSUPPORTED) and suggests fallback strategies.

Classes
───────
PlatformFeature      — Enum of all optimisation features.
FeatureSupport       — Support level: NATIVE / EMULATED / UNSUPPORTED.
FeatureRegistryStats — Runtime query statistics.
PlatformFeatureRegistry — Registry class; call is_supported(feature).

Usage::

    from squish.platform.detector import UnifiedPlatformDetector
    from squish.platform.feature_registry import (
        PlatformFeature, PlatformFeatureRegistry
    )

    info     = UnifiedPlatformDetector().detect()
    registry = PlatformFeatureRegistry(info)
    if registry.is_supported(PlatformFeature.FLASH_ATTENTION):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from squish.platform.detector import PlatformInfo, PlatformKind


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlatformFeature(Enum):
    """All optimisation features tracked by SquishAI."""
    FLASH_ATTENTION    = auto()   # hardware-accelerated attention
    METAL_DISPATCH     = auto()   # Metal GPU compute (macOS only)
    CUDA_GRAPHS        = auto()   # CUDA graph capture / replay
    INT4_QUANT         = auto()   # 4-bit NF4/FP4 quantization
    INT8_QUANT         = auto()   # 8-bit INT8 quantization
    SPECULATIVE_DECODE = auto()   # speculative decoding (draft model)
    LAYER_SKIP         = auto()   # adaptive layer skipping
    TOKEN_PIPELINE     = auto()   # concurrent token generation pipeline
    MMAP_WEIGHTS       = auto()   # memory-mapped weight loading
    BNB_QUANT          = auto()   # bitsandbytes quantization


class FeatureSupport(Enum):
    """Level of support for a given feature on the current platform."""
    NATIVE      = auto()   # full hardware acceleration available
    EMULATED    = auto()   # functional but not hardware-optimised
    UNSUPPORTED = auto()   # not available on this platform


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class FeatureRegistryStats:
    """Runtime statistics for PlatformFeatureRegistry."""
    query_count:      int = 0
    supported_count:  int = 0
    emulated_count:   int = 0
    unsupported_count: int = 0

    @property
    def native_rate(self) -> float:
        total = self.supported_count + self.emulated_count + self.unsupported_count
        return 0.0 if total == 0 else self.supported_count / total


# ---------------------------------------------------------------------------
# Feature → support table
# ---------------------------------------------------------------------------

# Lazily populated; keyed by PlatformKind
_SUPPORT_TABLES: Dict[str, Dict[PlatformFeature, FeatureSupport]] = {}

_FALLBACK_HINTS: Dict[PlatformFeature, str] = {
    PlatformFeature.FLASH_ATTENTION:    "numpy_attention",
    PlatformFeature.METAL_DISPATCH:     "numpy_kernels",
    PlatformFeature.CUDA_GRAPHS:        "eager_cuda",
    PlatformFeature.INT4_QUANT:         "fp16",
    PlatformFeature.INT8_QUANT:         "fp16",
    PlatformFeature.SPECULATIVE_DECODE: "standard_decode",
    PlatformFeature.LAYER_SKIP:         "full_layers",
    PlatformFeature.TOKEN_PIPELINE:     "sequential_tokens",
    PlatformFeature.MMAP_WEIGHTS:       "copy_load",
    PlatformFeature.BNB_QUANT:          "fp16",
}

N  = FeatureSupport.NATIVE
Em = FeatureSupport.EMULATED
Un = FeatureSupport.UNSUPPORTED

def _build_table(kind_name: str) -> Dict[PlatformFeature, FeatureSupport]:
    """Return feature→support table for the given PlatformKind.name."""
    F = PlatformFeature
    if kind_name == "MACOS_APPLE_SILICON":
        return {
            F.FLASH_ATTENTION:    N,
            F.METAL_DISPATCH:     N,
            F.CUDA_GRAPHS:        Un,
            F.INT4_QUANT:         N,
            F.INT8_QUANT:         N,
            F.SPECULATIVE_DECODE: N,
            F.LAYER_SKIP:         N,
            F.TOKEN_PIPELINE:     N,
            F.MMAP_WEIGHTS:       Em,   # Unified Memory, no POSIX mmap benefit
            F.BNB_QUANT:          Un,   # bitsandbytes is Linux/CUDA only
        }
    if kind_name == "LINUX_CUDA":
        return {
            F.FLASH_ATTENTION:    N,
            F.METAL_DISPATCH:     Un,
            F.CUDA_GRAPHS:        N,
            F.INT4_QUANT:         N,
            F.INT8_QUANT:         N,
            F.SPECULATIVE_DECODE: N,
            F.LAYER_SKIP:         N,
            F.TOKEN_PIPELINE:     N,
            F.MMAP_WEIGHTS:       N,
            F.BNB_QUANT:          N,
        }
    if kind_name == "LINUX_ROCM":
        return {
            F.FLASH_ATTENTION:    Em,   # composable-kernel fallback
            F.METAL_DISPATCH:     Un,
            F.CUDA_GRAPHS:        Un,   # ROCm has HIP graphs but separate code
            F.INT4_QUANT:         Em,
            F.INT8_QUANT:         N,
            F.SPECULATIVE_DECODE: N,
            F.LAYER_SKIP:         N,
            F.TOKEN_PIPELINE:     N,
            F.MMAP_WEIGHTS:       N,
            F.BNB_QUANT:          Em,
        }
    if kind_name == "LINUX_CPU":
        return {
            F.FLASH_ATTENTION:    Em,   # numpy fallback
            F.METAL_DISPATCH:     Un,
            F.CUDA_GRAPHS:        Un,
            F.INT4_QUANT:         Em,
            F.INT8_QUANT:         Em,
            F.SPECULATIVE_DECODE: Em,
            F.LAYER_SKIP:         N,
            F.TOKEN_PIPELINE:     Em,
            F.MMAP_WEIGHTS:       N,
            F.BNB_QUANT:          Un,
        }
    if kind_name in ("WINDOWS_WSL", "WINDOWS_NATIVE"):
        return {
            F.FLASH_ATTENTION:    Em,
            F.METAL_DISPATCH:     Un,
            F.CUDA_GRAPHS:        Un,
            F.INT4_QUANT:         Em,
            F.INT8_QUANT:         Em,
            F.SPECULATIVE_DECODE: Em,
            F.LAYER_SKIP:         N,
            F.TOKEN_PIPELINE:     Em,
            F.MMAP_WEIGHTS:       Em,
            F.BNB_QUANT:          Un,
        }
    # UNKNOWN / fallback — emulate everything
    return {f: Em for f in PlatformFeature}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PlatformFeatureRegistry:
    """Query which Squish optimisation features are available on this host.

    Parameters
    ----------
    platform_info : PlatformInfo
        Output of UnifiedPlatformDetector().detect().

    Usage::

        registry = PlatformFeatureRegistry(info)
        if registry.is_supported(PlatformFeature.BNB_QUANT):
            quant = BitsAndBytesQuantizer()
        else:
            quant = FallbackQuantizer()
    """

    def __init__(self, platform_info: "PlatformInfo") -> None:
        self._info  = platform_info
        self.stats  = FeatureRegistryStats()
        self._table = _build_table(platform_info.kind.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_supported(self, feature: PlatformFeature) -> bool:
        """Return True if the feature is NATIVE or EMULATED (i.e. usable)."""
        level = self.support_level(feature)
        return level in (FeatureSupport.NATIVE, FeatureSupport.EMULATED)

    def support_level(self, feature: PlatformFeature) -> FeatureSupport:
        """Return the exact FeatureSupport level for this feature."""
        self.stats.query_count += 1
        level = self._table.get(feature, FeatureSupport.EMULATED)
        if level == FeatureSupport.NATIVE:
            self.stats.supported_count += 1
        elif level == FeatureSupport.EMULATED:
            self.stats.emulated_count += 1
        else:
            self.stats.unsupported_count += 1
        return level

    def best_fallback(self, feature: PlatformFeature) -> Optional[str]:
        """Return a hint string describing the best available fallback."""
        return _FALLBACK_HINTS.get(feature)

    def supported_features(self) -> List[PlatformFeature]:
        """Return all features that are NATIVE or EMULATED on this platform."""
        return [f for f in PlatformFeature if self.is_supported(f)]

    def native_features(self) -> List[PlatformFeature]:
        """Return only NATIVE features."""
        return [f for f in PlatformFeature
                if self._table.get(f) == FeatureSupport.NATIVE]

    def summary(self) -> Dict[str, str]:
        """Return a human-readable dict of feature → support level."""
        return {
            f.name: self._table.get(f, FeatureSupport.EMULATED).name
            for f in PlatformFeature
        }

    def __repr__(self) -> str:
        native = len(self.native_features())
        total  = len(list(PlatformFeature))
        return (
            f"PlatformFeatureRegistry("
            f"platform={self._info.kind.name}, "
            f"native={native}/{total}, "
            f"queries={self.stats.query_count})"
        )
