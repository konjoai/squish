"""tests/test_adaptive_budget_unit.py

Full-coverage unit tests for squish/adaptive_budget.py.

Covers:
  BudgetConfig          — all __post_init__ validation errors
  BudgetState           — all __post_init__ validation errors, skip_layers
  BudgetStats           — violation_rate zero and non-zero
  AdaptiveBudgetController — __init__, step (all quality modes, negative latency),
                             reset, current_budget (with/without history),
                             latency_history, n_steps, stats
"""
from __future__ import annotations

import pytest

from squish.adaptive_budget import (
    AdaptiveBudgetController,
    BudgetConfig,
    BudgetState,
    BudgetStats,
)


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_valid_defaults(self):
        cfg = BudgetConfig()
        assert cfg.target_latency_ms == 150.0
        assert cfg.kv_budget_min == 512
        assert cfg.kv_budget_max == 4096

    def test_target_latency_zero_raises(self):
        with pytest.raises(ValueError, match="target_latency_ms must be > 0"):
            BudgetConfig(target_latency_ms=0.0)

    def test_target_latency_negative_raises(self):
        with pytest.raises(ValueError, match="target_latency_ms must be > 0"):
            BudgetConfig(target_latency_ms=-1.0)

    def test_kv_min_gte_max_raises(self):
        with pytest.raises(ValueError, match="kv_budget_min .* must be strictly less"):
            BudgetConfig(kv_budget_min=100, kv_budget_max=100)

    def test_kv_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="kv_budget_min .* must be strictly less"):
            BudgetConfig(kv_budget_min=200, kv_budget_max=100)

    def test_kv_min_less_than_one_raises(self):
        with pytest.raises(ValueError, match="kv_budget_min must be >= 1"):
            BudgetConfig(kv_budget_min=0, kv_budget_max=100)

    def test_kp_zero_raises(self):
        with pytest.raises(ValueError, match="kp must be > 0"):
            BudgetConfig(kp=0.0)

    def test_kp_negative_raises(self):
        with pytest.raises(ValueError, match="kp must be > 0"):
            BudgetConfig(kp=-0.1)

    def test_ki_negative_raises(self):
        with pytest.raises(ValueError, match="ki must be >= 0"):
            BudgetConfig(ki=-0.001)

    def test_max_skip_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="max_skip_fraction must be in"):
            BudgetConfig(max_skip_fraction=-0.1)

    def test_max_skip_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="max_skip_fraction must be in"):
            BudgetConfig(max_skip_fraction=1.1)

    def test_max_skip_fraction_zero_valid(self):
        cfg = BudgetConfig(max_skip_fraction=0.0)
        assert cfg.max_skip_fraction == 0.0

    def test_max_skip_fraction_one_valid(self):
        cfg = BudgetConfig(max_skip_fraction=1.0)
        assert cfg.max_skip_fraction == 1.0


# ---------------------------------------------------------------------------
# BudgetState
# ---------------------------------------------------------------------------


class TestBudgetState:
    def test_valid_construction(self):
        s = BudgetState(kv_tokens=1024, skip_fraction=0.1)
        assert s.kv_tokens == 1024
        assert s.skip_fraction == 0.1
        assert s.quality_mode == "balanced"

    def test_kv_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="kv_tokens must be >= 1"):
            BudgetState(kv_tokens=0, skip_fraction=0.0)

    def test_kv_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="kv_tokens must be >= 1"):
            BudgetState(kv_tokens=-1, skip_fraction=0.0)

    def test_skip_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="skip_fraction must be in"):
            BudgetState(kv_tokens=100, skip_fraction=-0.1)

    def test_skip_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="skip_fraction must be in"):
            BudgetState(kv_tokens=100, skip_fraction=1.1)

    def test_invalid_quality_mode_raises(self):
        with pytest.raises(ValueError, match="quality_mode must be one of"):
            BudgetState(kv_tokens=100, skip_fraction=0.0, quality_mode="turbo")

    def test_all_valid_quality_modes(self):
        for mode in ("performance", "balanced", "quality"):
            s = BudgetState(kv_tokens=100, skip_fraction=0.0, quality_mode=mode)
            assert s.quality_mode == mode

    def test_skip_layers_basic(self):
        s = BudgetState(kv_tokens=512, skip_fraction=0.25)
        assert s.skip_layers(total_layers=32) == 8

    def test_skip_layers_zero_fraction(self):
        s = BudgetState(kv_tokens=512, skip_fraction=0.0)
        assert s.skip_layers() == 0

    def test_skip_layers_total_layers_zero_raises(self):
        s = BudgetState(kv_tokens=512, skip_fraction=0.5)
        with pytest.raises(ValueError, match="total_layers must be >= 1"):
            s.skip_layers(total_layers=0)

    def test_skip_layers_total_layers_negative_raises(self):
        s = BudgetState(kv_tokens=512, skip_fraction=0.5)
        with pytest.raises(ValueError, match="total_layers must be >= 1"):
            s.skip_layers(total_layers=-1)


# ---------------------------------------------------------------------------
# BudgetStats
# ---------------------------------------------------------------------------


class TestBudgetStats:
    def test_violation_rate_zero_steps(self):
        s = BudgetStats()
        assert s.violation_rate == 0.0

    def test_violation_rate_nonzero(self):
        s = BudgetStats(n_steps=10, slo_violations=3)
        assert s.violation_rate == pytest.approx(0.3)

    def test_violation_rate_all_violations(self):
        s = BudgetStats(n_steps=5, slo_violations=5)
        assert s.violation_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AdaptiveBudgetController
# ---------------------------------------------------------------------------


def _make_ctrl(**kwargs) -> AdaptiveBudgetController:
    cfg = BudgetConfig(
        target_latency_ms=100.0,
        kv_budget_min=256,
        kv_budget_max=2048,
        max_skip_fraction=0.5,
        kp=0.1,
        ki=0.01,
        **kwargs,
    )
    return AdaptiveBudgetController(cfg)


class TestAdaptiveBudgetControllerInit:
    def test_initial_kv_budget_is_midpoint(self):
        ctrl = _make_ctrl()
        # mid = (256 + 2048) // 2 = 1152
        state = ctrl.current_budget
        assert state.kv_tokens == 1152

    def test_initial_integral_zero(self):
        ctrl = _make_ctrl()
        assert ctrl._integral == 0.0

    def test_initial_n_steps_zero(self):
        ctrl = _make_ctrl()
        assert ctrl.n_steps == 0


class TestStep:
    def test_negative_latency_raises(self):
        ctrl = _make_ctrl()
        with pytest.raises(ValueError, match="observed_latency_ms must be >= 0"):
            ctrl.step(-1.0)

    def test_step_returns_budget_state(self):
        ctrl = _make_ctrl()
        state = ctrl.step(100.0)
        assert isinstance(state, BudgetState)

    def test_performance_mode_high_latency(self):
        """Latency > 1.2 * target → performance mode."""
        ctrl = _make_ctrl()
        state = ctrl.step(130.0)  # 130 > 1.2 * 100 = 120
        assert state.quality_mode == "performance"

    def test_quality_mode_low_latency(self):
        """Latency < 0.8 * target → quality mode."""
        ctrl = _make_ctrl()
        state = ctrl.step(70.0)  # 70 < 0.8 * 100 = 80
        assert state.quality_mode == "quality"

    def test_balanced_mode_on_target(self):
        """Latency ≈ target → balanced mode."""
        ctrl = _make_ctrl()
        state = ctrl.step(100.0)  # exactly on target
        assert state.quality_mode == "balanced"

    def test_balanced_mode_within_range(self):
        """0.8*target <= latency <= 1.2*target → balanced."""
        ctrl = _make_ctrl()
        state = ctrl.step(95.0)  # 80 <= 95 <= 120
        assert state.quality_mode == "balanced"

    def test_step_increments_n_steps(self):
        ctrl = _make_ctrl()
        ctrl.step(100.0)
        ctrl.step(100.0)
        assert ctrl.n_steps == 2

    def test_slo_violation_counted(self):
        ctrl = _make_ctrl()
        ctrl.step(200.0)  # well above target
        s = ctrl.stats()
        assert s.slo_violations == 1

    def test_no_slo_violation_at_target(self):
        ctrl = _make_ctrl()
        ctrl.step(100.0)  # exactly on target — not a violation (target is the boundary)
        s = ctrl.stats()
        assert s.slo_violations == 0

    def test_slo_violation_above_target(self):
        ctrl = _make_ctrl()
        ctrl.step(101.0)  # just above target
        s = ctrl.stats()
        assert s.slo_violations == 1

    def test_skip_fraction_on_high_latency(self):
        ctrl = _make_ctrl()
        state = ctrl.step(200.0)  # 2× target → max skip fraction
        assert state.skip_fraction == pytest.approx(ctrl._cfg.max_skip_fraction)

    def test_skip_fraction_zero_at_target(self):
        ctrl = _make_ctrl()
        state = ctrl.step(100.0)  # at target → no skipping
        assert state.skip_fraction == pytest.approx(0.0)

    def test_integral_windup_guard(self):
        """Many large errors should not cause integral to blow up."""
        ctrl = _make_ctrl()
        for _ in range(100):
            ctrl.step(1000.0)
        # Integral is clamped to ±10 * budget_range
        budget_range = float(ctrl._cfg.kv_budget_max - ctrl._cfg.kv_budget_min)
        assert abs(ctrl._integral) <= 10.0 * budget_range + 1e-6


class TestReset:
    def test_reset_restores_initial_state(self):
        ctrl = _make_ctrl()
        ctrl.step(200.0)
        ctrl.step(50.0)
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._latency_history == []
        assert ctrl._n_steps == 0
        assert ctrl._slo_violations == 0

    def test_reset_kv_budget_to_midpoint(self):
        ctrl = _make_ctrl()
        ctrl.step(1000.0)  # push budget to minimum
        ctrl.reset()
        mid = (ctrl._cfg.kv_budget_min + ctrl._cfg.kv_budget_max) // 2
        assert ctrl._kv_budget == float(mid)


class TestCurrentBudget:
    def test_current_budget_no_history(self):
        ctrl = _make_ctrl()
        state = ctrl.current_budget
        assert state.skip_fraction == 0.0
        assert state.quality_mode == "balanced"

    def test_current_budget_with_history(self):
        ctrl = _make_ctrl()
        ctrl.step(200.0)  # high latency pushes skip fraction up
        state = ctrl.current_budget
        assert state.skip_fraction > 0.0

    def test_current_budget_does_not_increment_n_steps(self):
        ctrl = _make_ctrl()
        _ = ctrl.current_budget
        _ = ctrl.current_budget
        assert ctrl.n_steps == 0


class TestProperties:
    def test_latency_history_is_copy(self):
        ctrl = _make_ctrl()
        ctrl.step(100.0)
        ctrl.step(150.0)
        hist = ctrl.latency_history
        assert hist == [100.0, 150.0]
        hist.clear()  # Modifying the copy should not affect internal state
        assert len(ctrl.latency_history) == 2

    def test_n_steps(self):
        ctrl = _make_ctrl()
        assert ctrl.n_steps == 0
        ctrl.step(100.0)
        assert ctrl.n_steps == 1

    def test_stats_mean_latency(self):
        ctrl = _make_ctrl()
        ctrl.step(100.0)
        ctrl.step(200.0)
        s = ctrl.stats()
        assert s.mean_latency_ms == pytest.approx(150.0)

    def test_stats_empty_history(self):
        ctrl = _make_ctrl()
        s = ctrl.stats()
        assert s.mean_latency_ms == 0.0
        assert s.n_steps == 0
        assert s.slo_violations == 0
