"""Tests for PlatformFeatureRegistry — per-platform feature support tables.

The registry maps each optimization feature to NATIVE / EMULATED / UNSUPPORTED
for the detected platform. These tests pin the support contract for every
PlatformKind, the usability/stat-tracking query methods, and the summaries.
"""
from __future__ import annotations

import pytest

from squish.platform.detector import PlatformKind
from squish.platform.feature_registry import (
    FeatureRegistryStats,
    FeatureSupport,
    PlatformFeature,
    PlatformFeatureRegistry,
)


class _Info:
    """Minimal stand-in: the registry only reads platform_info.kind.name."""

    def __init__(self, kind: PlatformKind):
        self.kind = kind


def _registry(kind: PlatformKind) -> PlatformFeatureRegistry:
    return PlatformFeatureRegistry(_Info(kind))


ALL_KINDS = list(PlatformKind)


class TestSupportTable:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_table_covers_every_feature(self, kind):
        reg = _registry(kind)
        summary = reg.summary()
        # Every feature must have a defined level on every platform.
        assert set(summary) == {f.name for f in PlatformFeature}
        assert all(v in {"NATIVE", "EMULATED", "UNSUPPORTED"} for v in summary.values())

    def test_apple_silicon_known_mappings(self):
        reg = _registry(PlatformKind.MACOS_APPLE_SILICON)
        assert reg.support_level(PlatformFeature.METAL_DISPATCH) == FeatureSupport.NATIVE
        assert reg.support_level(PlatformFeature.CUDA_GRAPHS) == FeatureSupport.UNSUPPORTED
        assert reg.support_level(PlatformFeature.BNB_QUANT) == FeatureSupport.UNSUPPORTED
        assert reg.support_level(PlatformFeature.MMAP_WEIGHTS) == FeatureSupport.EMULATED

    def test_linux_cuda_known_mappings(self):
        reg = _registry(PlatformKind.LINUX_CUDA)
        assert reg.support_level(PlatformFeature.CUDA_GRAPHS) == FeatureSupport.NATIVE
        assert reg.support_level(PlatformFeature.METAL_DISPATCH) == FeatureSupport.UNSUPPORTED
        assert reg.support_level(PlatformFeature.BNB_QUANT) == FeatureSupport.NATIVE

    def test_linux_cpu_emulates_acceleration(self):
        reg = _registry(PlatformKind.LINUX_CPU)
        assert reg.support_level(PlatformFeature.FLASH_ATTENTION) == FeatureSupport.EMULATED
        assert reg.support_level(PlatformFeature.METAL_DISPATCH) == FeatureSupport.UNSUPPORTED
        assert reg.support_level(PlatformFeature.MMAP_WEIGHTS) == FeatureSupport.NATIVE

    def test_unknown_kind_emulates_everything(self):
        reg = _registry(PlatformKind.UNKNOWN)
        assert all(
            reg.support_level(f) == FeatureSupport.EMULATED for f in PlatformFeature
        )


class TestQueries:
    def test_is_supported_true_for_native_and_emulated(self):
        reg = _registry(PlatformKind.LINUX_CPU)
        # FLASH_ATTENTION is EMULATED on CPU → usable.
        assert reg.is_supported(PlatformFeature.FLASH_ATTENTION) is True
        # METAL_DISPATCH is UNSUPPORTED on CPU → not usable.
        assert reg.is_supported(PlatformFeature.METAL_DISPATCH) is False

    def test_supported_excludes_only_unsupported(self):
        reg = _registry(PlatformKind.MACOS_APPLE_SILICON)
        supported = set(reg.supported_features())
        assert PlatformFeature.CUDA_GRAPHS not in supported  # UNSUPPORTED
        assert PlatformFeature.METAL_DISPATCH in supported    # NATIVE

    def test_native_features_are_strictly_native(self):
        reg = _registry(PlatformKind.LINUX_CUDA)
        native = set(reg.native_features())
        assert PlatformFeature.CUDA_GRAPHS in native
        # EMULATED/UNSUPPORTED must be excluded; CPU-only emulation never native here.
        for f in native:
            assert reg.support_level(f) == FeatureSupport.NATIVE

    def test_best_fallback_hint(self):
        reg = _registry(PlatformKind.LINUX_CPU)
        assert reg.best_fallback(PlatformFeature.FLASH_ATTENTION) == "numpy_attention"
        assert reg.best_fallback(PlatformFeature.BNB_QUANT) == "fp16"


class TestStats:
    def test_query_count_and_breakdown(self):
        reg = _registry(PlatformKind.MACOS_APPLE_SILICON)
        reg.support_level(PlatformFeature.METAL_DISPATCH)   # NATIVE
        reg.support_level(PlatformFeature.MMAP_WEIGHTS)     # EMULATED
        reg.support_level(PlatformFeature.CUDA_GRAPHS)      # UNSUPPORTED
        s = reg.stats
        assert s.query_count == 3
        assert s.supported_count == 1
        assert s.emulated_count == 1
        assert s.unsupported_count == 1
        assert s.native_rate == pytest.approx(1 / 3)

    def test_native_rate_zero_when_no_queries(self):
        assert FeatureRegistryStats().native_rate == 0.0


def test_repr_mentions_platform_and_counts():
    reg = _registry(PlatformKind.LINUX_CUDA)
    reg.support_level(PlatformFeature.CUDA_GRAPHS)
    r = repr(reg)
    assert "LINUX_CUDA" in r
    assert "native=" in r and "queries=1" in r
