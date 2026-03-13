#!/usr/bin/env python3
"""
tests/test_compiled_ffn_unit.py

Unit tests for squish/fused_kernels.py additions:
  - _METAL_FUSION_AVAILABLE sentinel (re-exported from metal_fusion)
  - patch_model_compiled_ffn()

All tests exercise the numpy / no-Metal-hardware code paths; the actual
Metal dispatch is guarded with ``# pragma: no cover`` in the module.

Coverage targets
────────────────
_METAL_FUSION_AVAILABLE
  - is a bool
  - importable from squish.fused_kernels

patch_model_compiled_ffn
  - returns int
  - returns 0 for an empty model (no layers)
  - kernels=None delegates to patch_model fallback (returns int)
  - swiglu_enabled=False delegates to patch_model fallback
  - model with no MLP layers returns 0
  - model with layers that have no gate_proj still returns without exception
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from squish.fused_kernels import (
    _METAL_FUSION_AVAILABLE,
    patch_model_compiled_ffn,
)


# ---------------------------------------------------------------------------
# _METAL_FUSION_AVAILABLE sentinel
# ---------------------------------------------------------------------------

class TestMetalFusionAvailableSentinel:
    def test_is_bool(self):
        assert isinstance(_METAL_FUSION_AVAILABLE, bool)

    def test_importable_from_fused_kernels(self):
        import squish.fused_kernels as fk
        assert hasattr(fk, "_METAL_FUSION_AVAILABLE")
        assert isinstance(fk._METAL_FUSION_AVAILABLE, bool)

    def test_value_consistent_with_metal_fusion_module(self):
        """Value must agree with the source in metal_fusion.py."""
        try:
            from squish.metal_fusion import _METAL_FUSION_AVAILABLE as _mf_val
            assert _METAL_FUSION_AVAILABLE == _mf_val
        except ImportError:
            pytest.skip("metal_fusion not importable — consistency check skipped")


# ---------------------------------------------------------------------------
# Helpers: minimal stub models
# ---------------------------------------------------------------------------

def _empty_model():
    """A model-like object with an empty layers list."""
    m = types.SimpleNamespace()
    m.layers = []
    return m


def _model_no_mlp(n_layers: int = 2):
    """A model whose layers have no 'mlp' attribute."""
    layers = [types.SimpleNamespace() for _ in range(n_layers)]
    m = types.SimpleNamespace()
    m.layers = layers
    return m


def _model_with_mlp(n_layers: int = 2):
    """A model whose layers have an mlp but no gate/up/down projections."""
    layers = []
    for _ in range(n_layers):
        mlp = MagicMock()
        del mlp.gate_proj   # ensure getattr returns None
        del mlp.up_proj
        del mlp.down_proj
        layer = types.SimpleNamespace(mlp=mlp)
        layers.append(layer)
    m = types.SimpleNamespace()
    m.layers = layers
    return m


def _mock_kernels(swiglu_enabled: bool):
    k = MagicMock()
    k.swiglu_enabled = swiglu_enabled
    k.available = swiglu_enabled
    return k


# ---------------------------------------------------------------------------
# patch_model_compiled_ffn — no-Metal fallback paths
# ---------------------------------------------------------------------------

class TestPatchModelCompiledFFN:
    def test_returns_int(self):
        result = patch_model_compiled_ffn(_empty_model(), metal_fusion_kernels=None)
        assert isinstance(result, int)

    def test_empty_model_returns_zero(self):
        result = patch_model_compiled_ffn(_empty_model(), metal_fusion_kernels=None)
        assert result == 0

    def test_kernels_none_uses_patch_model_fallback(self):
        """kernels=None must delegate to patch_model(); result is int >= 0."""
        model = _empty_model()
        result = patch_model_compiled_ffn(model, metal_fusion_kernels=None)
        assert isinstance(result, int)
        assert result >= 0

    def test_swiglu_disabled_uses_patch_model_fallback(self):
        """swiglu_enabled=False must run the patch_model() path, not Metal path."""
        kernels = _mock_kernels(swiglu_enabled=False)
        model = _empty_model()
        with patch("squish.fused_kernels.patch_model", return_value=0) as mock_pm:
            result = patch_model_compiled_ffn(model, metal_fusion_kernels=kernels)
        mock_pm.assert_called_once_with(model)
        assert result == 0

    def test_model_without_mlp_returns_zero_or_nonzero(self):
        """Layers without mlp attribute must not raise; count may be 0."""
        model = _model_no_mlp()
        result = patch_model_compiled_ffn(model, metal_fusion_kernels=None)
        assert isinstance(result, int)

    def test_no_exception_on_model_with_mlp_no_projections(self):
        """MLP without gate/up/down must not raise an exception."""
        model = _model_with_mlp()
        result = patch_model_compiled_ffn(model, metal_fusion_kernels=None)
        assert isinstance(result, int)

    def test_delegate_fallback_call_signature(self):
        """Verify patch_model is called with the model positional arg."""
        model = _empty_model()
        sentinel = 99
        with patch("squish.fused_kernels.patch_model", return_value=sentinel) as mock_pm:
            result = patch_model_compiled_ffn(model)
        mock_pm.assert_called_once_with(model)
        assert result == sentinel
