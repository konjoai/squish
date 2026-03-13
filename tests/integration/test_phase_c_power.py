#!/usr/bin/env python3
"""
tests/test_phase_c_power.py

Phase C tests: PowerMonitor + PowerModeConfig + apply_mode
(squish/power_monitor.py)

Coverage targets
────────────────
power_monitor.py — 100%

  PowerModeConfig
    - frozen dataclass construction and field access
    - POWER_CONFIGS contains all three profiles

  apply_mode
    - unknown mode key → no-op / no crash
    - "battery" → mutates _thinking_budget + _chunk_prefill_threshold
    - "balanced" → mutates correctly
    - "performance" → mutates correctly

  PowerMonitor.__init__
    - constructs, calls _refresh eagerly

  PowerMonitor._refresh
    - pmset succeeds → _raw updated
    - pmset fails (subprocess raises) → _raw unchanged

  PowerMonitor.get_power_source
    - "Battery Power" in raw → "battery"
    - "Battery Power" absent → "ac"

  PowerMonitor.get_battery_level
    - percentage present → float
    - no percentage → 1.0

  PowerMonitor.get_recommended_mode
    - AC → "performance"
    - battery ≥ 50% → "balanced"
    - battery < 50% → "battery"

  PowerMonitor.start_polling
    - starts timer on first call
    - second call is no-op (timer already set)

  PowerMonitor._tick
    - calls _refresh then _schedule_next

  PowerMonitor.stop_polling
    - cancels timer and sets _timer = None
    - safe to call when no timer running
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squish.serving.power_monitor import (
    POWER_CONFIGS,
    PowerMonitor,
    apply_mode,
)

# ══════════════════════════════════════════════════════════════════════════════
# POWER_CONFIGS + PowerModeConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestPowerModeConfig:
    def test_all_three_profiles_present(self):
        assert "performance" in POWER_CONFIGS
        assert "balanced" in POWER_CONFIGS
        assert "battery" in POWER_CONFIGS

    def test_performance_config_fields(self):
        cfg = POWER_CONFIGS["performance"]
        assert cfg.eagle_k_draft == 5
        assert cfg.batch_window_ms == 20.0
        assert cfg.metal_cache_fraction == 0.25
        assert cfg.thinking_budget == -1
        assert cfg.chunk_prefill_threshold == 512

    def test_balanced_config_fields(self):
        cfg = POWER_CONFIGS["balanced"]
        assert cfg.eagle_k_draft == 3
        assert cfg.batch_window_ms == 30.0
        assert cfg.metal_cache_fraction == 0.20
        assert cfg.thinking_budget == 256
        assert cfg.chunk_prefill_threshold == 512

    def test_battery_config_fields(self):
        cfg = POWER_CONFIGS["battery"]
        assert cfg.eagle_k_draft == 2
        assert cfg.batch_window_ms == 50.0
        assert cfg.metal_cache_fraction == 0.15
        assert cfg.thinking_budget == 64
        assert cfg.chunk_prefill_threshold == 256

    def test_dataclass_is_frozen(self):
        """PowerModeConfig instances must be immutable."""
        cfg = POWER_CONFIGS["performance"]
        with pytest.raises((TypeError, AttributeError)):
            cfg.eagle_k_draft = 99  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# apply_mode
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyMode:
    def _make_globals(self):
        return {
            "_thinking_budget": -1,
            "_chunk_prefill_threshold": 512,
        }

    def test_unknown_mode_is_noop(self):
        g = self._make_globals()
        apply_mode("turbo", g)
        assert g["_thinking_budget"] == -1
        assert g["_chunk_prefill_threshold"] == 512

    def test_battery_mode_applies_limits(self):
        g = self._make_globals()
        apply_mode("battery", g)
        assert g["_thinking_budget"] == 64
        assert g["_chunk_prefill_threshold"] == 256

    def test_balanced_mode_applies_limits(self):
        g = self._make_globals()
        apply_mode("balanced", g)
        assert g["_thinking_budget"] == 256
        assert g["_chunk_prefill_threshold"] == 512

    def test_performance_mode_applies_limits(self):
        g = self._make_globals()
        # Start with battery values; switch to performance
        g["_thinking_budget"] = 64
        g["_chunk_prefill_threshold"] = 256
        apply_mode("performance", g)
        assert g["_thinking_budget"] == -1
        assert g["_chunk_prefill_threshold"] == 512

    def test_apply_mode_does_not_require_all_keys(self):
        """apply_mode writes into globals; missing keys are just set."""
        g = {}
        apply_mode("battery", g)
        assert g["_thinking_budget"] == 64


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor helpers
# ══════════════════════════════════════════════════════════════════════════════

def _monitor_with_raw(raw: str) -> PowerMonitor:
    """Create a PowerMonitor with a fixed _raw value (no real subprocess)."""
    with patch("subprocess.run") as mock_run:
        r = MagicMock()
        r.stdout = raw
        mock_run.return_value = r
        mon = PowerMonitor(poll_interval_s=9999)  # don't auto-fire timer
    return mon


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor._refresh
# ══════════════════════════════════════════════════════════════════════════════

class TestPowerMonitorRefresh:
    def test_refresh_stores_pmset_output(self):
        raw = "Now drawing from 'Battery Power'; 73%"
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = raw
            mock_run.return_value = result
            mon = PowerMonitor(poll_interval_s=9999)
        assert mon._raw == raw

    def test_refresh_survives_subprocess_exception(self):
        """If pmset throws any exception, _raw is left unchanged (no crash)."""
        with patch("subprocess.run", side_effect=OSError("no pmset")):
            mon = PowerMonitor(poll_interval_s=9999)
        # _raw stays as "" (default) — no crash
        assert isinstance(mon._raw, str)

    def test_refresh_called_at_init(self):
        """__init__ calls _refresh exactly once."""
        with patch.object(PowerMonitor, "_refresh") as mock_refresh:
            PowerMonitor(poll_interval_s=9999)
        mock_refresh.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor.get_power_source
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPowerSource:
    def test_battery_power_detected(self):
        mon = _monitor_with_raw("Now drawing from 'Battery Power'\n73%")
        assert mon.get_power_source() == "battery"

    def test_ac_power_when_no_battery_power_string(self):
        mon = _monitor_with_raw("Now drawing from 'AC Power'\n100%")
        assert mon.get_power_source() == "ac"

    def test_empty_raw_returns_ac(self):
        mon = _monitor_with_raw("")
        assert mon.get_power_source() == "ac"


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor.get_battery_level
# ══════════════════════════════════════════════════════════════════════════════

class TestGetBatteryLevel:
    def test_parses_percentage_from_pmset_output(self):
        mon = _monitor_with_raw("InternalBattery-0 (id=0)\t; 73%; discharging")
        assert abs(mon.get_battery_level() - 0.73) < 0.001

    def test_returns_1_when_no_percentage(self):
        mon = _monitor_with_raw("Now drawing from 'AC Power'")
        assert mon.get_battery_level() == 1.0

    def test_100_percent_parses_correctly(self):
        mon = _monitor_with_raw("InternalBattery-0; 100%; charging")
        assert abs(mon.get_battery_level() - 1.0) < 0.001

    def test_zero_percent(self):
        mon = _monitor_with_raw("InternalBattery-0; 0%; discharging")
        assert mon.get_battery_level() == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor.get_recommended_mode
# ══════════════════════════════════════════════════════════════════════════════

class TestGetRecommendedMode:
    def test_ac_power_returns_performance(self):
        mon = _monitor_with_raw("Now drawing from 'AC Power'\n100%")
        assert mon.get_recommended_mode() == "performance"

    def test_battery_above_50_returns_balanced(self):
        mon = _monitor_with_raw("Now drawing from 'Battery Power'\n75%")
        assert mon.get_recommended_mode() == "balanced"

    def test_battery_at_50_returns_balanced(self):
        """At exactly 50%, level >= 0.5 → balanced (not battery)."""
        mon = _monitor_with_raw("Now drawing from 'Battery Power'\n50%")
        assert mon.get_recommended_mode() == "balanced"

    def test_battery_below_50_returns_battery(self):
        mon = _monitor_with_raw("Now drawing from 'Battery Power'\n35%")
        assert mon.get_recommended_mode() == "battery"

    def test_battery_at_0_returns_battery(self):
        mon = _monitor_with_raw("Now drawing from 'Battery Power'\n0%")
        assert mon.get_recommended_mode() == "battery"

    def test_no_raw_defaults_to_ac_performance(self):
        """Empty pmset output → source='ac' → 'performance'."""
        mon = _monitor_with_raw("")
        assert mon.get_recommended_mode() == "performance"


# ══════════════════════════════════════════════════════════════════════════════
# PowerMonitor.start_polling / stop_polling / _tick
# ══════════════════════════════════════════════════════════════════════════════

class TestPolling:
    def test_start_polling_creates_timer(self):
        mon = _monitor_with_raw("AC Power\n100%")
        assert mon._timer is None
        mon.start_polling()
        assert mon._timer is not None
        mon.stop_polling()

    def test_start_polling_is_noop_if_already_running(self):
        mon = _monitor_with_raw("AC Power\n100%")
        mon.start_polling()
        first_timer = mon._timer
        mon.start_polling()  # second call — must not replace timer
        assert mon._timer is first_timer
        mon.stop_polling()

    def test_stop_polling_cancels_timer(self):
        mon = _monitor_with_raw("AC Power\n100%")
        mon.start_polling()
        mon.stop_polling()
        assert mon._timer is None

    def test_stop_polling_safe_with_no_timer(self):
        """stop_polling when timer is None must not raise."""
        mon = _monitor_with_raw("AC Power\n100%")
        assert mon._timer is None
        mon.stop_polling()  # should not raise
        assert mon._timer is None

    def test_tick_calls_refresh_and_reschedules(self):
        """_tick() must call _refresh() then _schedule_next()."""
        mon = _monitor_with_raw("AC Power\n100%")
        with (
            patch.object(mon, "_refresh") as mock_refresh,
            patch.object(mon, "_schedule_next") as mock_schedule,
        ):
            mon._tick()
        mock_refresh.assert_called_once()
        mock_schedule.assert_called_once()

    def test_timer_is_daemon(self):
        """Background refresh timer must be a daemon thread (won't block shutdown)."""
        mon = _monitor_with_raw("AC Power\n100%")
        mon.start_polling()
        assert mon._timer.daemon is True
        mon.stop_polling()
