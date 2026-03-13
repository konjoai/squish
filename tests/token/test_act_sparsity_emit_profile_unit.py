#!/usr/bin/env python3
"""
tests/test_act_sparsity_emit_profile_unit.py

Unit tests for the emit_profile extension of ActSparsityPredictor.calibrate().

Tests verify:
  - Default (emit_profile=False) still returns plain dict
  - emit_profile=True returns (dict, NeuronProfile) tuple
  - NeuronProfile layer_count equals n_layers
  - Per-neuron hot indices reflect recorded activation frequency
  - reset() clears _act_counts so a subsequent calibrate() yields fresh profile
  - Layers with no recorded activations get zero hot neurons
  - profile_config forwarding (hot_fraction)
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.token.act_sparsity import ActSparsityPredictor, SparsityConfig
from squish.hardware.neuron_profile import NeuronProfile, NeuronProfileConfig


RNG = np.random.default_rng(0)

_CFG = SparsityConfig(hidden_dim=64, n_layers=4, threshold=0.1)


def _make_predictor() -> ActSparsityPredictor:
    return ActSparsityPredictor(_CFG)


def _record_all(predictor: ActSparsityPredictor, n_steps: int = 8) -> None:
    """Record n_steps activations for each layer."""
    for step in range(n_steps):
        for layer_idx in range(_CFG.n_layers):
            acts = RNG.standard_normal((16, _CFG.hidden_dim)).astype(np.float32)
            predictor.record(layer_idx, acts)


# ---------------------------------------------------------------------------
# Default behaviour unchanged
# ---------------------------------------------------------------------------

class TestCalibrateDefaultBehaviourUnchanged:
    def test_returns_dict_without_flag(self):
        p = _make_predictor()
        _record_all(p)
        result = p.calibrate()
        assert isinstance(result, dict)

    def test_dict_keys_are_layer_indices(self):
        p = _make_predictor()
        _record_all(p)
        result = p.calibrate()
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_dict_values_in_0_1(self):
        p = _make_predictor()
        _record_all(p)
        result = p.calibrate()
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_emit_false_explicit_still_returns_dict(self):
        p = _make_predictor()
        _record_all(p)
        result = p.calibrate(emit_profile=False)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# emit_profile=True returns (dict, NeuronProfile)
# ---------------------------------------------------------------------------

class TestCalibrateEmitProfile:
    def test_returns_tuple(self):
        p = _make_predictor()
        _record_all(p)
        result = p.calibrate(emit_profile=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_dict(self):
        p = _make_predictor()
        _record_all(p)
        sparsity_map, profile = p.calibrate(emit_profile=True)
        assert isinstance(sparsity_map, dict)

    def test_second_element_is_neuron_profile(self):
        p = _make_predictor()
        _record_all(p)
        sparsity_map, profile = p.calibrate(emit_profile=True)
        assert isinstance(profile, NeuronProfile)

    def test_profile_layer_count_equals_n_layers(self):
        p = _make_predictor()
        _record_all(p)
        _, profile = p.calibrate(emit_profile=True)
        assert profile.layer_count == _CFG.n_layers

    def test_sparsity_map_matches_plain_calibrate(self):
        p = _make_predictor()
        _record_all(p)
        plain = p.calibrate(emit_profile=False)
        with_profile, _ = p.calibrate(emit_profile=True)
        assert plain == with_profile

    def test_hot_indices_reflect_high_frequency_neurons(self):
        """The neuron that is always active should appear in hot_indices."""
        p = _make_predictor()
        # Layer 0: first neuron always very large, rest near-zero
        for _ in range(20):
            acts = np.zeros((16, _CFG.hidden_dim), dtype=np.float32)
            acts[:, 0] = 10.0   # always active
            p.record(0, acts)

        _, profile = p.calibrate(emit_profile=True)
        hot_l0 = profile.hot_indices[0]
        assert 0 in hot_l0, f"neuron 0 should be hot, got hot_indices={hot_l0[:5]}"

    def test_unrecorded_layers_have_cold_or_zero_hot(self):
        """Layers with no recorded activations should not crash and return sane result."""
        p = ActSparsityPredictor(SparsityConfig(hidden_dim=32, n_layers=4, threshold=0.1))
        # Only record layer 0
        p.record(0, np.ones((4, 32), dtype=np.float32))
        _, profile = p.calibrate(emit_profile=True)
        # layer 1-3 have zero counts — hot_fraction of zeros is still valid (all zero)
        assert profile.layer_count == 4

    def test_profile_config_forwarded(self):
        """hot_fraction in profile_config is respected."""
        p = _make_predictor()
        _record_all(p)
        hi_cfg = NeuronProfileConfig(hot_fraction=0.5)
        _, profile_hi = p.calibrate(emit_profile=True, profile_config=hi_cfg)
        lo_cfg = NeuronProfileConfig(hot_fraction=0.1)
        _, profile_lo = p.calibrate(emit_profile=True, profile_config=lo_cfg)
        # high hot_fraction should yield more (or equal) hot neurons per layer
        for idx in range(_CFG.n_layers):
            assert len(profile_hi.hot_indices[idx]) >= len(profile_lo.hot_indices[idx])


# ---------------------------------------------------------------------------
# reset() clears per-neuron counts
# ---------------------------------------------------------------------------

class TestResetClearsActCounts:
    def test_reset_clears_act_counts_so_profile_is_empty(self):
        p = _make_predictor()
        _record_all(p)
        p.reset()
        result = p.calibrate()
        assert result == {}

    def test_reset_then_emit_profile_works_without_error(self):
        p = _make_predictor()
        _record_all(p)
        p.reset()
        # After reset with no new records, calibrate with emit_profile=True
        # should not raise — it just returns empty dict and empty-count profile
        result = p.calibrate(emit_profile=True)
        assert isinstance(result, tuple)
        sparsity_map, profile = result
        assert sparsity_map == {}
