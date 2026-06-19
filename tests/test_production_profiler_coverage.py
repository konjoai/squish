"""Behavioral coverage for ``squish.hardware.production_profiler`` — the
ProfilerWindow validation and the record/stats/report/reset API. Pure-Python;
no MLX.
"""
from __future__ import annotations

import pytest

from squish.hardware.production_profiler import ProductionProfiler, ProfilerWindow


def test_profiler_window_rejects_nonpositive():
    with pytest.raises(ValueError, match="window_size must be"):
        ProfilerWindow(window_size=0)  # line 66


def test_profiler_window_default_ok():
    assert ProfilerWindow().window_size == 1000


def test_record_negative_latency_raises():
    with pytest.raises(ValueError, match="latency_ms must be"):
        ProductionProfiler().record("op", -1.0)


def test_record_and_stats():
    p = ProductionProfiler()
    for v in (10.0, 20.0, 30.0):
        p.record("gen.prefill", v)
    s = p.stats("gen.prefill")
    assert s.n_samples == 3 and s.mean_ms == 20.0
    assert s.min_ms == 10.0 and s.max_ms == 30.0


def test_stats_unknown_operation_raises():
    with pytest.raises(KeyError, match="Unknown operation"):
        ProductionProfiler().stats("never_recorded")


def test_report_and_to_json_dict():
    p = ProductionProfiler()
    p.record("a", 5.0)
    p.record("b", 7.0)
    rep = p.report()
    assert set(rep) == {"a", "b"}
    js = p.to_json_dict()
    assert js["a"]["n_samples"] == 1 and "p99_ms" in js["a"]


def test_rolling_window_evicts_oldest():
    p = ProductionProfiler(ProfilerWindow(window_size=2))
    for v in (1.0, 2.0, 3.0):
        p.record("op", v)
    s = p.stats("op")
    assert s.n_samples == 2  # only the last two retained


def test_operations_sorted():
    p = ProductionProfiler()
    p.record("zeta", 1.0)
    p.record("alpha", 1.0)
    assert p.operations == ["alpha", "zeta"]


def test_reset_specific_and_all():
    p = ProductionProfiler()
    p.record("a", 1.0)
    p.record("b", 2.0)
    p.reset("a")
    assert p.stats("a").n_samples == 0 and p.stats("b").n_samples == 1
    p.reset()
    assert p.operations == []


def test_reset_unknown_operation_raises():
    with pytest.raises(KeyError, match="Unknown operation"):
        ProductionProfiler().reset("nope")
