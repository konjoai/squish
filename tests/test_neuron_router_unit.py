#!/usr/bin/env python3
"""
tests/test_neuron_router_unit.py

Unit tests for squish/neuron_router.py — Phase 10A inference-time routing.

Coverage targets
────────────────
NeuronRouterConfig
  - valid config constructs without error
  - invalid hot_device raises ValueError
  - invalid cold_device raises ValueError
  - non-NeuronProfile profile raises TypeError

NeuronRouter
  - forward output shape matches original dense pass output shape (2-D input)
  - forward output shape correct for 1-D input (squeezed)
  - forward numerically matches reference dense SwiGLU for small random weights
  - forward raises IndexError on out-of-range layer_idx
  - stats() returns correct n_hot, n_cold, n_total, hot_fraction
  - __repr__ contains expected fields

patch_model_neuron_routing
  - patches all layer mlp.__call__ attributes
  - returns dict mapping layer_idx → original callable
  - patched forward returns correct shape
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from squish.neuron_profile import NeuronProfile, NeuronProfileConfig, NeuronProfiler
from squish.neuron_router import NeuronRouter, NeuronRouterConfig, patch_model_neuron_routing

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(n_layers=2, ffn_dim=16, hot_fraction=0.25):
    counts = [RNG.random(ffn_dim).astype(np.float32) for _ in range(n_layers)]
    cfg = NeuronProfileConfig(hot_fraction=hot_fraction)
    return NeuronProfiler(cfg).calibrate(counts)


def _swiglu_dense(x, gate_w, up_w, down_w):
    """Reference dense SwiGLU forward: down_w @ (silu(gate_w @ x.T) * (up_w @ x.T))."""
    x_f = x.astype(np.float32)
    gate_act = x_f @ gate_w.T   # (batch, ffn)
    up_act   = x_f @ up_w.T     # (batch, ffn)
    mid = NeuronRouter._silu(gate_act) * up_act  # (batch, ffn)
    return mid @ down_w.T   # (batch, out)


def _random_weights(ffn_dim=16, hidden_dim=8, out_dim=8):
    gate_w = RNG.standard_normal((ffn_dim, hidden_dim)).astype(np.float32)
    up_w   = RNG.standard_normal((ffn_dim, hidden_dim)).astype(np.float32)
    down_w = RNG.standard_normal((out_dim, ffn_dim)).astype(np.float32)
    return gate_w, up_w, down_w


# ---------------------------------------------------------------------------
# NeuronRouterConfig
# ---------------------------------------------------------------------------

class TestNeuronRouterConfig:
    def test_valid_defaults(self):
        p = NeuronProfile()
        cfg = NeuronRouterConfig(profile=p)
        assert cfg.hot_device == "gpu"
        assert cfg.cold_device == "cpu"

    def test_invalid_hot_device_raises(self):
        with pytest.raises(ValueError, match="hot_device"):
            NeuronRouterConfig(profile=NeuronProfile(), hot_device="tpu")

    def test_invalid_cold_device_raises(self):
        with pytest.raises(ValueError, match="cold_device"):
            NeuronRouterConfig(profile=NeuronProfile(), cold_device="disk")

    def test_non_profile_raises_type_error(self):
        with pytest.raises(TypeError, match="NeuronProfile"):
            NeuronRouterConfig(profile="not-a-profile")


# ---------------------------------------------------------------------------
# NeuronRouter.forward
# ---------------------------------------------------------------------------

class TestNeuronRouterForward:
    @pytest.fixture()
    def router(self):
        profile = _make_profile(n_layers=2, ffn_dim=16, hot_fraction=0.25)
        return NeuronRouter(NeuronRouterConfig(profile=profile))

    @pytest.fixture()
    def weights(self):
        return _random_weights(ffn_dim=16, hidden_dim=8, out_dim=8)

    def test_output_shape_2d(self, router, weights):
        gate_w, up_w, down_w = weights
        x = RNG.standard_normal((3, 8)).astype(np.float32)
        out = router.forward(0, x, gate_w, up_w, down_w)
        assert out.shape == (3, 8)

    def test_output_shape_1d_squeezed(self, router, weights):
        gate_w, up_w, down_w = weights
        x = RNG.standard_normal(8).astype(np.float32)
        out = router.forward(0, x, gate_w, up_w, down_w)
        assert out.shape == (8,)

    def test_numerically_matches_dense_reference(self):
        """Hot/cold split must be numerically equivalent to the dense path."""
        ffn_dim, hidden_dim = 32, 16
        profile = _make_profile(n_layers=1, ffn_dim=ffn_dim, hot_fraction=0.50)
        router  = NeuronRouter(NeuronRouterConfig(profile=profile))
        gate_w, up_w, down_w = _random_weights(ffn_dim=ffn_dim, hidden_dim=hidden_dim,
                                               out_dim=hidden_dim)
        x = RNG.standard_normal((2, hidden_dim)).astype(np.float32)

        out_router = router.forward(0, x, gate_w, up_w, down_w)
        out_dense  = _swiglu_dense(x, gate_w, up_w, down_w)
        np.testing.assert_allclose(out_router, out_dense, rtol=1e-4, atol=1e-5)

    def test_out_of_range_layer_raises(self, router, weights):
        gate_w, up_w, down_w = weights
        x = RNG.standard_normal((1, 8)).astype(np.float32)
        with pytest.raises(IndexError):
            router.forward(99, x, gate_w, up_w, down_w)


# ---------------------------------------------------------------------------
# NeuronRouter.stats
# ---------------------------------------------------------------------------

class TestNeuronRouterStats:
    def test_stats_counts_correct(self):
        ffn_dim = 20
        profile = _make_profile(n_layers=1, ffn_dim=ffn_dim, hot_fraction=0.30)
        router  = NeuronRouter(NeuronRouterConfig(profile=profile))
        s = router.stats(0)
        assert s["n_hot"] + s["n_cold"] == ffn_dim
        assert s["n_total"] == ffn_dim
        assert 0.0 < s["hot_fraction"] < 1.0

    def test_stats_hot_fraction_close_to_config(self):
        profile = _make_profile(n_layers=1, ffn_dim=100, hot_fraction=0.25)
        router  = NeuronRouter(NeuronRouterConfig(profile=profile))
        s = router.stats(0)
        assert abs(s["hot_fraction"] - 0.25) < 0.05


# ---------------------------------------------------------------------------
# NeuronRouter.__repr__
# ---------------------------------------------------------------------------

class TestNeuronRouterRepr:
    def test_repr_contains_key_fields(self):
        router = NeuronRouter(NeuronRouterConfig(profile=NeuronProfile()))
        r = repr(router)
        assert "NeuronRouter" in r
        assert "hot=" in r
        assert "cold=" in r


# ---------------------------------------------------------------------------
# patch_model_neuron_routing
# ---------------------------------------------------------------------------

def _make_fake_model(n_layers=2, ffn_dim=16, hidden_dim=8):
    """Build a minimal MagicMock model that mimics a transformer for patching."""
    gate_w, up_w, down_w = _random_weights(ffn_dim=ffn_dim, hidden_dim=hidden_dim,
                                           out_dim=hidden_dim)

    class FakeMLP:
        def __init__(self):
            self.gate_proj = types.SimpleNamespace(weight=gate_w)
            self.up_proj   = types.SimpleNamespace(weight=up_w)
            self.down_proj = types.SimpleNamespace(weight=down_w)

        def __call__(self, x):
            return _swiglu_dense(x, gate_w, up_w, down_w)

    class FakeLayer:
        def __init__(self):
            self.mlp = FakeMLP()

    class FakeModel:
        def __init__(self):
            self.layers = [FakeLayer() for _ in range(n_layers)]

    return FakeModel()


class TestPatchModelNeuronRouting:
    def test_patches_all_layers(self):
        model = _make_fake_model(n_layers=3)
        profile = _make_profile(n_layers=3, ffn_dim=16)
        router = NeuronRouter(NeuronRouterConfig(profile=profile))
        originals = patch_model_neuron_routing(model, router)
        assert len(originals) == 3

    def test_returns_original_callables(self):
        model = _make_fake_model(n_layers=2)
        profile = _make_profile(n_layers=2, ffn_dim=16)
        router = NeuronRouter(NeuronRouterConfig(profile=profile))
        originals = patch_model_neuron_routing(model, router)
        for i, layer in enumerate(model.layers):
            assert originals[i] is layer.mlp._original_forward

    def test_patched_forward_correct_shape(self):
        n_layers = 2
        hidden_dim = 8
        model = _make_fake_model(n_layers=n_layers, hidden_dim=hidden_dim)
        profile = _make_profile(n_layers=n_layers, ffn_dim=16)
        router = NeuronRouter(NeuronRouterConfig(profile=profile))
        patch_model_neuron_routing(model, router)
        x = RNG.standard_normal((1, hidden_dim)).astype(np.float32)
        out = model.layers[0].mlp(x)
        assert out.shape == (1, hidden_dim)
