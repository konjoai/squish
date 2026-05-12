"""tests/test_quality_monitor.py — W111: Inference Quality Monitor.

25 tests covering:
- RequestMetric frozen dataclass
- Record single metric
- Report empty → zero requests
- Report n_requests matches recorded count
- Report per-model breakdown
- Error rate calculation
- Latency P50 in range
- Latency P95 >= P50
- Latency P99 >= P95
- Tokens-per-sec mean positive
- TTFT None when not provided
- TTFT stats populated when provided
- Window filters old events
- Thread-safe concurrent record (50 threads, no crash)
- Clear resets state
- stats_for known model returns QualityStats
- stats_for unknown model returns None
- QualityStats frozen
- to_dict returns dict
- _percentile single value
- _percentile sorted list
- _percentile p=0 is min
- _percentile p=100 is max
- get_quality_monitor singleton
- CLI build_parser() includes 'quality' subcommand
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squish.serving.quality_monitor import (  # noqa: E402
    QualityMonitor,
    QualityReport,
    QualityStats,
    RequestMetric,
    _percentile,
    get_quality_monitor,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _metric(
    model_id: str = "test-model",
    latency_ms: float = 100.0,
    ttft_ms: float | None = 20.0,
    tokens_generated: int = 50,
    tokens_per_sec: float = 500.0,
    success: bool = True,
    error_type: str | None = None,
    ts_offset: float = 0.0,
) -> RequestMetric:
    """Build a RequestMetric with a timestamp of now + ts_offset."""
    return RequestMetric(
        timestamp=time.monotonic() + ts_offset,
        model_id=model_id,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        tokens_generated=tokens_generated,
        tokens_per_sec=tokens_per_sec,
        success=success,
        error_type=error_type,
    )


# ── RequestMetric ─────────────────────────────────────────────────────────────


def test_request_metric_frozen():
    """RequestMetric must be frozen (mutation raises FrozenInstanceError)."""
    m = _metric()
    with pytest.raises(FrozenInstanceError):
        m.latency_ms = 999.0  # type: ignore[misc]


# ── QualityMonitor.record / report ───────────────────────────────────────────


def test_record_single_metric():
    """Recording one metric increments n_requests to 1."""
    mon = QualityMonitor()
    mon.record(_metric())
    report = mon.report()
    assert report.total_requests == 1


def test_report_empty_returns_zero_requests():
    """A fresh monitor returns total_requests=0."""
    mon = QualityMonitor()
    report = mon.report()
    assert report.total_requests == 0
    assert report.models == []


def test_report_n_requests_matches_recorded():
    """n_requests equals the number of recorded metrics."""
    mon = QualityMonitor()
    for _ in range(7):
        mon.record(_metric())
    report = mon.report()
    assert report.total_requests == 7


def test_report_per_model_breakdown():
    """Report splits metrics into one QualityStats per distinct model_id."""
    mon = QualityMonitor()
    mon.record(_metric(model_id="alpha"))
    mon.record(_metric(model_id="alpha"))
    mon.record(_metric(model_id="beta"))
    report = mon.report()
    model_ids = {s.model_id for s in report.models}
    assert model_ids == {"alpha", "beta"}
    alpha = next(s for s in report.models if s.model_id == "alpha")
    assert alpha.n_requests == 2
    beta = next(s for s in report.models if s.model_id == "beta")
    assert beta.n_requests == 1


def test_report_error_rate_calculation():
    """error_rate = n_errors / n_requests."""
    mon = QualityMonitor()
    mon.record(_metric(success=True))
    mon.record(_metric(success=False, error_type="RuntimeError"))
    mon.record(_metric(success=False, error_type="ValueError"))
    report = mon.report()
    stats = report.models[0]
    assert stats.n_errors == 2
    assert abs(stats.error_rate - 2 / 3) < 1e-9


# ── Latency percentiles ───────────────────────────────────────────────────────


def test_latency_p50_in_range():
    """P50 latency lies within the recorded range."""
    mon = QualityMonitor()
    for ms in [100.0, 200.0, 300.0]:
        mon.record(_metric(latency_ms=ms))
    stats = mon.report().models[0]
    assert 100.0 <= stats.latency_p50 <= 300.0


def test_latency_p95_ge_p50():
    """P95 latency must be >= P50 latency."""
    mon = QualityMonitor()
    for ms in range(10, 110, 10):
        mon.record(_metric(latency_ms=float(ms)))
    stats = mon.report().models[0]
    assert stats.latency_p95 >= stats.latency_p50


def test_latency_p99_ge_p95():
    """P99 latency must be >= P95 latency."""
    mon = QualityMonitor()
    for ms in range(10, 110, 10):
        mon.record(_metric(latency_ms=float(ms)))
    stats = mon.report().models[0]
    assert stats.latency_p99 >= stats.latency_p95


# ── Throughput ────────────────────────────────────────────────────────────────


def test_tokens_per_sec_mean_positive():
    """tokens_per_sec_mean must be > 0 when requests are recorded."""
    mon = QualityMonitor()
    mon.record(_metric(tokens_per_sec=300.0))
    mon.record(_metric(tokens_per_sec=600.0))
    stats = mon.report().models[0]
    assert stats.tokens_per_sec_mean > 0.0


# ── TTFT ──────────────────────────────────────────────────────────────────────


def test_ttft_none_when_not_provided():
    """ttft_p50/p95 are None when no metrics have TTFT data."""
    mon = QualityMonitor()
    mon.record(_metric(ttft_ms=None))
    stats = mon.report().models[0]
    assert stats.ttft_p50 is None
    assert stats.ttft_p95 is None


def test_ttft_stats_populated_when_provided():
    """ttft_p50/p95 are floats when at least one metric has TTFT."""
    mon = QualityMonitor()
    for ttft in [10.0, 20.0, 30.0]:
        mon.record(_metric(ttft_ms=ttft))
    stats = mon.report().models[0]
    assert stats.ttft_p50 is not None
    assert stats.ttft_p95 is not None
    assert 10.0 <= stats.ttft_p50 <= 30.0


# ── Window filtering ──────────────────────────────────────────────────────────


def test_window_filters_old_events():
    """Events older than window_seconds are excluded from the report."""
    mon = QualityMonitor(window_seconds=60)
    # Old event: 120 s in the past
    old = RequestMetric(
        timestamp=time.monotonic() - 120,
        model_id="model-x",
        latency_ms=100.0,
        ttft_ms=None,
        tokens_generated=10,
        tokens_per_sec=100.0,
        success=True,
        error_type=None,
    )
    mon.record(old)
    mon.record(_metric(model_id="model-x"))
    report = mon.report(window_seconds=60)
    # Only the recent event should survive
    stats = next((s for s in report.models if s.model_id == "model-x"), None)
    assert stats is not None
    assert stats.n_requests == 1


# ── Thread safety ─────────────────────────────────────────────────────────────


def test_thread_safe_concurrent_record():
    """50 threads recording simultaneously must not crash or corrupt state."""
    mon = QualityMonitor()
    errors: list[Exception] = []

    def _worker():
        try:
            for _ in range(10):
                mon.record(_metric())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    report = mon.report()
    assert report.total_requests == 500


# ── clear ─────────────────────────────────────────────────────────────────────


def test_clear_resets_state():
    """clear() empties the event store; subsequent report returns 0 requests."""
    mon = QualityMonitor()
    for _ in range(5):
        mon.record(_metric())
    mon.clear()
    assert mon.report().total_requests == 0


# ── stats_for ─────────────────────────────────────────────────────────────────


def test_stats_for_known_model():
    """stats_for returns a QualityStats for a model that has events."""
    mon = QualityMonitor()
    mon.record(_metric(model_id="known"))
    result = mon.stats_for("known")
    assert isinstance(result, QualityStats)
    assert result.model_id == "known"


def test_stats_for_unknown_model_returns_none():
    """stats_for returns None for a model_id with no events in window."""
    mon = QualityMonitor()
    assert mon.stats_for("does-not-exist") is None


# ── Frozen dataclasses ────────────────────────────────────────────────────────


def test_quality_stats_frozen():
    """QualityStats must be frozen (mutation raises FrozenInstanceError)."""
    mon = QualityMonitor()
    mon.record(_metric())
    stats = mon.report().models[0]
    with pytest.raises(FrozenInstanceError):
        stats.n_requests = 999  # type: ignore[misc]


# ── to_dict ───────────────────────────────────────────────────────────────────


def test_to_dict_returns_dict():
    """to_dict() on QualityStats and QualityReport returns plain dicts."""
    mon = QualityMonitor()
    mon.record(_metric())
    report = mon.report()
    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert "models" in report_dict
    assert isinstance(report_dict["models"], list)
    stats_dict = report.models[0].to_dict()
    assert isinstance(stats_dict, dict)
    assert "latency_p50_ms" in stats_dict


# ── _percentile ───────────────────────────────────────────────────────────────


def test_percentile_single_value():
    """_percentile([x], 50) == x for any scalar x."""
    assert _percentile([42.0], 50) == 42.0


def test_percentile_sorted_list():
    """_percentile returns the correct median for an odd-length list."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(vals, 50) == 30.0


def test_percentile_p0_is_min():
    """_percentile(vals, 0) returns the minimum value."""
    vals = [5.0, 10.0, 15.0, 20.0]
    assert _percentile(vals, 0) == 5.0


def test_percentile_p100_is_max():
    """_percentile(vals, 100) returns the maximum value."""
    vals = [5.0, 10.0, 15.0, 20.0]
    assert _percentile(vals, 100) == 20.0


# ── Singleton ─────────────────────────────────────────────────────────────────


def test_get_quality_monitor_singleton():
    """get_quality_monitor() returns the same instance on repeated calls."""
    a = get_quality_monitor()
    b = get_quality_monitor()
    assert a is b
    assert isinstance(a, QualityMonitor)


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_quality_subcommand_registered():
    """build_parser() must include a 'quality' subcommand."""
    from squish.cli import build_parser  # noqa: PLC0415

    ap = build_parser()
    # argparse stores subparser choices in the _subparsers action group
    subparsers_action = next(
        (a for a in ap._actions if hasattr(a, "choices") and a.choices),
        None,
    )
    assert subparsers_action is not None, "No subparsers found in build_parser()"
    assert "quality" in subparsers_action.choices, (
        "'quality' subcommand not registered in build_parser()"
    )
