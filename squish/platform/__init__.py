"""squish/platform — Cross-platform detection and adaptation.

This package provides unified platform detection (macOS Apple Silicon,
Linux CUDA/ROCm/CPU, WSL2), platform-aware feature flags, and platform-
specific memory governors for production cross-platform deployment.
"""

from squish.platform.detector import (
    CUDAInfo,
    PlatformDetectorStats,
    PlatformInfo,
    PlatformKind,
    UnifiedPlatformDetector,
)
from squish.platform.feature_registry import (
    FeatureSupport,
    PlatformFeature,
    PlatformFeatureRegistry,
)

__all__ = [
    "CUDAInfo",
    "PlatformDetectorStats",
    "PlatformInfo",
    "PlatformKind",
    "UnifiedPlatformDetector",
    "FeatureSupport",
    "PlatformFeature",
    "PlatformFeatureRegistry",
]
