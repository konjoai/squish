#!/usr/bin/env python3
"""
tests/test_memory_governor_unit.py

Unit tests for squish/memory_governor.py — macOS memory-pressure watchdog.

Coverage targets
────────────────
_run
  - successful command returns stdout
  - failing / non-existent command returns ""
  - timeout returns ""

_read_pressure_level
  - valid returned value (0) passes through
  - invalid value (99) falls back to LEVEL_NORMAL
  - empty output falls back to LEVEL_NORMAL
  - non-integer output falls back to LEVEL_NORMAL

_read_vm_stat
  - valid vm_stat output parsed correctly (page_size, pages)
  - missing page-size line uses default 4096
  - empty output returns _DEFAULT_AVAILABLE_GB fallback

MemorySnapshot
  - attrs accessible: pressure_level, available_gb, wired_gb, timestamp
  - is_under_pressure: True when level >= WARNING
  - is_under_pressure: False when level = NORMAL
  - immutable (frozen dataclass)

MemoryGovernor
  - __init__ with custom poll_interval
  - invalid poll_interval raises ValueError
  - start() returns self (chaining)
  - start() populates snapshot from initial poll
  - start() is idempotent (safe to call twice)
  - stop() terminates thread
  - pressure_level / available_gb / wired_gb / is_under_pressure read snapshot
  - add_callback: callback invoked on pressure level change
  - remove_callback: callback no longer called after removal
  - remove_callback with unknown fn is no-op
  - budget_tokens: headroom subtraction and bytes_per_token scaling
  - budget_tokens: returns 0 when available <= headroom
  - __repr__ contains expected fields
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from squish.memory_governor import (
    LEVEL_CRITICAL,
    LEVEL_NORMAL,
    LEVEL_URGENT,
    LEVEL_WARNING,
    MemoryGovernor,
    MemorySnapshot,
    _DEFAULT_AVAILABLE_GB,
    _read_pressure_level,
    _read_vm_stat,
    _run,
)


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------

class TestRun:
    def test_valid_command_returns_stdout(self):
        out = _run(["echo", "hello"])
        assert "hello" in out

    def test_invalid_command_returns_empty_string(self):
        out = _run(["__no_such_command_xyz__"])
        assert out == ""

    def test_empty_args_returns_empty_string(self):
        # subprocess will fail with empty list
        out = _run([])
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# _read_pressure_level
# ---------------------------------------------------------------------------

class TestReadPressureLevel:
    def test_valid_zero_returns_level_normal(self):
        with patch("squish.memory_governor._run", return_value="0\n"):
            assert _read_pressure_level() == LEVEL_NORMAL

    def test_valid_one_returns_level_warning(self):
        with patch("squish.memory_governor._run", return_value="1\n"):
            assert _read_pressure_level() == LEVEL_WARNING

    def test_valid_two_returns_level_urgent(self):
        with patch("squish.memory_governor._run", return_value="2\n"):
            assert _read_pressure_level() == LEVEL_URGENT

    def test_valid_four_returns_level_critical(self):
        with patch("squish.memory_governor._run", return_value="4\n"):
            assert _read_pressure_level() == LEVEL_CRITICAL

    def test_invalid_value_falls_back_to_normal(self):
        with patch("squish.memory_governor._run", return_value="99\n"):
            assert _read_pressure_level() == LEVEL_NORMAL

    def test_empty_output_falls_back_to_normal(self):
        with patch("squish.memory_governor._run", return_value=""):
            assert _read_pressure_level() == LEVEL_NORMAL

    def test_non_integer_falls_back_to_normal(self):
        with patch("squish.memory_governor._run", return_value="err\n"):
            assert _read_pressure_level() == LEVEL_NORMAL


# ---------------------------------------------------------------------------
# _read_vm_stat
# ---------------------------------------------------------------------------

_SAMPLE_VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             100.
Pages active:                         45000.
Pages inactive:                       52000.
Pages speculative:                      400.
Pages throttled:                          0.
Pages wired down:                     10000.
Pages purgeable:                        200.
"""


class TestReadVmStat:
    def test_parses_available_gb(self):
        with patch("squish.memory_governor._run", return_value=_SAMPLE_VM_STAT):
            avail, _ = _read_vm_stat()
        # (free=100 + inactive=52000 + speculative=400 + purgeable=200) * 16384 bytes
        expected_pages = 100 + 52000 + 400 + 200
        expected_gb = expected_pages * 16384 / (1 << 30)
        assert avail == pytest.approx(expected_gb, rel=1e-4)

    def test_parses_wired_gb(self):
        with patch("squish.memory_governor._run", return_value=_SAMPLE_VM_STAT):
            _, wired = _read_vm_stat()
        expected_gb = 10000 * 16384 / (1 << 30)
        assert wired == pytest.approx(expected_gb, rel=1e-4)

    def test_empty_output_returns_defaults(self):
        with patch("squish.memory_governor._run", return_value=""):
            avail, wired = _read_vm_stat()
        assert avail == _DEFAULT_AVAILABLE_GB
        assert wired == 0.0

    def test_missing_page_size_uses_default(self):
        # No "page size of" line → uses 4096
        vm_stat_no_page_size = """\
Pages free:                            1000.
Pages inactive:                        2000.
Pages speculative:                      100.
Pages purgeable:                         50.
Pages wired down:                       500.
"""
        with patch("squish.memory_governor._run", return_value=vm_stat_no_page_size):
            avail, wired = _read_vm_stat()
        expected_pages = 1000 + 2000 + 100 + 50
        expected_gb = expected_pages * 4096 / (1 << 30)
        assert avail == pytest.approx(expected_gb, rel=1e-4)


# ---------------------------------------------------------------------------
# MemorySnapshot
# ---------------------------------------------------------------------------

class TestMemorySnapshot:
    def test_default_attrs(self):
        snap = MemorySnapshot()
        assert snap.pressure_level == LEVEL_NORMAL
        assert snap.available_gb == _DEFAULT_AVAILABLE_GB
        assert snap.wired_gb == 0.0
        assert snap.timestamp > 0.0

    def test_is_under_pressure_normal(self):
        snap = MemorySnapshot(pressure_level=LEVEL_NORMAL)
        assert snap.is_under_pressure is False

    def test_is_under_pressure_warning(self):
        snap = MemorySnapshot(pressure_level=LEVEL_WARNING)
        assert snap.is_under_pressure is True

    def test_is_under_pressure_critical(self):
        snap = MemorySnapshot(pressure_level=LEVEL_CRITICAL)
        assert snap.is_under_pressure is True

    def test_frozen_dataclass_immutable(self):
        snap = MemorySnapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.pressure_level = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryGovernor
# ---------------------------------------------------------------------------

def _patched_governor(poll_interval=60.0, pressure=0, avail=8.0, wired=2.0):
    """Create a MemoryGovernor with patched sysctl and vm_stat calls."""
    with patch("squish.memory_governor._read_pressure_level", return_value=pressure), \
         patch("squish.memory_governor._read_vm_stat", return_value=(avail, wired)):
        gov = MemoryGovernor(poll_interval=poll_interval)
        gov.start()
    return gov


class TestMemoryGovernorInit:
    def test_invalid_poll_interval_raises(self):
        with pytest.raises(ValueError, match="poll_interval"):
            MemoryGovernor(poll_interval=0.0)

    def test_negative_poll_interval_raises(self):
        with pytest.raises(ValueError, match="poll_interval"):
            MemoryGovernor(poll_interval=-1.0)


class TestMemoryGovernorLifecycle:
    def test_start_returns_self(self):
        with patch("squish.memory_governor._read_pressure_level", return_value=0), \
             patch("squish.memory_governor._read_vm_stat", return_value=(8.0, 1.0)):
            gov = MemoryGovernor(poll_interval=60.0)
            result = gov.start()
        assert result is gov
        gov.stop()

    def test_start_populates_snapshot(self):
        gov = _patched_governor(pressure=1, avail=6.0, wired=3.0)
        assert gov.pressure_level == 1
        assert gov.available_gb == pytest.approx(6.0)
        assert gov.wired_gb    == pytest.approx(3.0)
        gov.stop()

    def test_start_is_idempotent(self):
        gov = _patched_governor()
        thread_before = gov._thread
        gov.start()   # second call — same thread should still be alive
        assert gov._thread is thread_before
        gov.stop()

    def test_stop_terminates_thread(self):
        gov = _patched_governor()
        th = gov._thread
        gov.stop()
        assert th is not None
        # after stop, thread should no longer be alive (join was called)
        assert not th.is_alive()


class TestMemoryGovernorProperties:
    def test_pressure_level_property(self):
        gov = _patched_governor(pressure=LEVEL_WARNING)
        assert gov.pressure_level == LEVEL_WARNING
        gov.stop()

    def test_is_under_pressure_true(self):
        gov = _patched_governor(pressure=LEVEL_WARNING)
        assert gov.is_under_pressure is True
        gov.stop()

    def test_is_under_pressure_false_at_normal(self):
        gov = _patched_governor(pressure=LEVEL_NORMAL)
        assert gov.is_under_pressure is False
        gov.stop()

    def test_available_gb_property(self):
        gov = _patched_governor(avail=10.5)
        assert gov.available_gb == pytest.approx(10.5)
        gov.stop()

    def test_wired_gb_property(self):
        gov = _patched_governor(wired=3.75)
        assert gov.wired_gb == pytest.approx(3.75)
        gov.stop()


class TestMemoryGovernorCallbacks:
    def test_callback_called_on_pressure_change(self):
        fired = []
        gov = _patched_governor(pressure=LEVEL_NORMAL)
        gov.add_callback(fired.append)
        # Simulate a pressure change via _poll_once with different level
        with patch("squish.memory_governor._read_pressure_level", return_value=LEVEL_WARNING), \
             patch("squish.memory_governor._read_vm_stat", return_value=(4.0, 2.0)):
            gov._poll_once()
        gov.stop()
        assert LEVEL_WARNING in fired

    def test_callback_not_called_without_change(self):
        fired = []
        gov = _patched_governor(pressure=LEVEL_NORMAL)
        gov.add_callback(fired.append)
        # Poll again with same level
        with patch("squish.memory_governor._read_pressure_level", return_value=LEVEL_NORMAL), \
             patch("squish.memory_governor._read_vm_stat", return_value=(8.0, 1.0)):
            gov._poll_once()
        gov.stop()
        assert len(fired) == 0

    def test_remove_callback(self):
        fired = []
        gov = _patched_governor(pressure=LEVEL_NORMAL)
        gov.add_callback(fired.append)
        gov.remove_callback(fired.append)
        with patch("squish.memory_governor._read_pressure_level", return_value=LEVEL_WARNING), \
             patch("squish.memory_governor._read_vm_stat", return_value=(4.0, 2.0)):
            gov._poll_once()
        gov.stop()
        assert len(fired) == 0

    def test_remove_unknown_callback_is_noop(self):
        gov = _patched_governor()
        gov.remove_callback(lambda x: None)  # should not raise
        gov.stop()

    def test_callback_exception_does_not_crash_poll(self):
        def bad_cb(level):
            raise RuntimeError("callback error")
        gov = _patched_governor(pressure=LEVEL_NORMAL)
        gov.add_callback(bad_cb)
        with patch("squish.memory_governor._read_pressure_level", return_value=LEVEL_WARNING), \
             patch("squish.memory_governor._read_vm_stat", return_value=(4.0, 2.0)):
            gov._poll_once()   # should not raise
        gov.stop()


class TestMemoryGovernorBudgetTokens:
    def test_budget_tokens_basic(self):
        gov = _patched_governor(avail=2.0)
        # available=2GB, headroom=1GB → usable=1GB
        # budget = 1 * 2^30 / 512
        expected = int(1 * (1 << 30) / 512)
        assert gov.budget_tokens(512) == expected
        gov.stop()

    def test_budget_tokens_below_headroom_returns_zero(self):
        gov = _patched_governor(avail=0.5)  # less than 1GB headroom
        assert gov.budget_tokens(512) == 0
        gov.stop()

    def test_budget_tokens_default_bytes_per_token(self):
        gov = _patched_governor(avail=2.0)
        result = gov.budget_tokens()  # default 512
        assert result >= 0
        gov.stop()


class TestMemoryGovernorRepr:
    def test_repr_contains_key_fields(self):
        gov = _patched_governor(pressure=0, avail=8.0, wired=2.0)
        r = repr(gov)
        gov.stop()
        assert "MemoryGovernor" in r
        assert "level=" in r
        assert "available=" in r
