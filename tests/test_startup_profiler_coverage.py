"""Behavioral coverage for the disabled-report and import-timing paths of
``squish.serving.startup_profiler`` left untested by the baseline suite.
Pure-Python; no MLX.
"""
from __future__ import annotations

import sys

from squish.serving.startup_profiler import (
    StartupPhase,
    StartupReport,
    StartupTimer,
    _Entry,
    measure_import_ms,
)


def _entry(elapsed=5.0, phase="load", label="x"):
    e = _Entry(phase=phase, label=label, start_ms=0.0)
    e.end_ms = elapsed
    return e


# ── StartupReport ────────────────────────────────────────────────────────────


def test_report_disabled_drops_entries():
    report = StartupReport(enabled=False)
    report.add(_entry())  # disabled → not appended (89→exit)
    assert report.total_ms == 0.0
    assert report.slowest() == []


def test_report_enabled_records_and_sorts():
    report = StartupReport(enabled=True)
    report.add(_entry(elapsed=10.0, label="slow"))
    report.add(_entry(elapsed=2.0, label="fast"))
    assert report.total_ms == 12.0
    slowest = report.slowest(n=1)
    assert len(slowest) == 1 and slowest[0].label == "slow"
    d = report.to_dict()
    assert "total_ms" in d


# ── StartupTimer ─────────────────────────────────────────────────────────────


def test_timer_records_when_enabled():
    report = StartupReport(enabled=True)
    with StartupTimer(report, StartupPhase.MODEL_LOAD, "load"):
        pass
    assert len(report.slowest()) == 1


def test_timer_noop_when_disabled():
    report = StartupReport(enabled=False)
    with StartupTimer(report, "custom_phase"):
        pass
    assert report.total_ms == 0.0


# ── measure_import_ms ────────────────────────────────────────────────────────


def test_measure_import_already_cached_returns_zero():
    # 'sys' is always already imported → short-circuit to 0.0.
    assert measure_import_ms("sys") == 0.0


def test_measure_import_bad_module_returns_zero():
    assert measure_import_ms("definitely_not_a_real_module_xyz") == 0.0  # ImportError


def test_measure_import_fresh_module_returns_time(monkeypatch):
    monkeypatch.delitem(sys.modules, "colorsys", raising=False)  # force a real import
    ms = measure_import_ms("colorsys")  # success path (187)
    assert isinstance(ms, float) and ms >= 0.0
