"""Behavioral coverage for the edge/error paths of
``squish.serving.quality_monitor`` left untested by the baseline suite: the
empty-percentile guard, the record() error handler, the empty-stats branch,
the singleton double-checked lock, the record_completion_metric wrapper, and
quality_response_dict. Pure-Python — no MLX.
"""
from __future__ import annotations

import pytest

from squish.serving import quality_monitor as qm
from squish.serving.quality_monitor import (
    QualityMonitor,
    RequestMetric,
    _compute_stats,
    _percentile,
    quality_response_dict,
    record_completion_metric,
)


def _metric(model_id="m", latency_ms=100.0, ttft_ms=10.0, tps=20.0, success=True):
    import time
    return RequestMetric(
        timestamp=time.monotonic(), model_id=model_id, latency_ms=latency_ms,
        ttft_ms=ttft_ms, tokens_generated=50, tokens_per_sec=tps,
        success=success, error_type=None,
    )


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0  # line 52


def test_percentile_single_and_interpolation():
    assert _percentile([5.0], 99) == 5.0
    assert _percentile([0.0, 10.0], 50) == 5.0  # midpoint interpolation


def test_record_swallows_bad_metric():
    mon = QualityMonitor()
    mon.record(object())  # no .timestamp → AttributeError swallowed (238-240)
    assert mon.report().total_requests == 0


def test_compute_stats_empty_events():
    s = _compute_stats([], "m", 60, "2026-01-01T00:00:00Z")  # n == 0 branch (322-338)
    assert s.n_requests == 0 and s.error_rate == 0.0
    assert s.ttft_p50 is None and s.latency_p95 == 0.0


def test_compute_stats_with_events():
    events = [_metric(latency_ms=x, tps=x) for x in (10.0, 20.0, 30.0)]
    s = _compute_stats(events, "m", 60, "now")
    assert s.n_requests == 3 and s.latency_mean == 20.0


def test_report_aggregates_per_model():
    mon = QualityMonitor()
    mon.record(_metric(model_id="a"))
    mon.record(_metric(model_id="b", success=False))
    rep = mon.report()
    assert rep.total_requests == 2
    assert {m.model_id for m in rep.models} == {"a", "b"}


# ── singleton ───────────────────────────────────────────────────────────────


def test_get_quality_monitor_creates_and_reuses(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    first = qm.get_quality_monitor()
    assert isinstance(first, QualityMonitor)
    assert qm.get_quality_monitor() is first  # reused


def test_get_quality_monitor_double_checked_lock(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    existing = QualityMonitor()

    class _Lock:
        def __enter__(self):
            # Simulate another thread populating the singleton between the outer
            # and inner None-checks → the inner check sees non-None (376→378).
            qm._monitor_singleton = existing
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(qm, "_singleton_lock", _Lock())
    assert qm.get_quality_monitor() is existing


# ── record_completion_metric ────────────────────────────────────────────────


def test_record_completion_metric_records(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    record_completion_metric("model-x", duration_s=1.0, ttft_s=0.2, n_tokens=10, tps=25.0)
    rep = qm.get_quality_monitor().report()
    assert rep.total_requests == 1 and rep.models[0].model_id == "model-x"


def test_record_completion_metric_zero_ttft_is_none(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    record_completion_metric("m", 1.0, ttft_s=0.0, n_tokens=5, tps=10.0)
    stats = qm.get_quality_monitor().report().models[0]
    assert stats.ttft_p50 is None  # ttft_s == 0 → None


def test_record_completion_metric_swallows_errors(monkeypatch):
    class _BadMon:
        def record(self, m):
            raise ValueError("monitor boom")

    monkeypatch.setattr(qm, "get_quality_monitor", lambda: _BadMon())
    # Must never raise — monitoring can't interrupt generation (417-420).
    record_completion_metric("m", 1.0, 0.5, 10, 20.0)


# ── quality_response_dict ───────────────────────────────────────────────────


def test_quality_response_dict_clamps_window(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    qm.get_quality_monitor().record(_metric(model_id="m"))
    out = quality_response_dict(window=5)  # below floor → clamped to 60
    assert out["window_seconds"] == 60
    assert out["total_requests"] >= 1


def test_quality_response_dict_with_model_filter(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    mon = qm.get_quality_monitor()
    mon.record(_metric(model_id="keep"))
    mon.record(_metric(model_id="drop"))
    out = quality_response_dict(window=120, model_filter="keep")
    assert out["window_seconds"] == 120
    assert {m["model_id"] for m in out["models"]} == {"keep"}
    assert out["total_requests"] == 1


def test_quality_response_dict_clamps_max_window(monkeypatch):
    monkeypatch.setattr(qm, "_monitor_singleton", None)
    out = quality_response_dict(window=999_999)  # above ceiling → clamped to 86400
    assert out["window_seconds"] == 86400
