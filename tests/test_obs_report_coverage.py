"""Behavioral coverage for ``squish.serving.obs_report`` — bottleneck detection
and observability-report generation. Pure-Python; profiler/tracer are fakes.
"""
from __future__ import annotations

import types

from squish.serving import obs_report as obs
from squish.serving.obs_report import detect_bottlenecks, generate_report


def _stats(p99, mean=1.0, n=5):
    return types.SimpleNamespace(p99_ms=p99, mean_ms=mean, n_samples=n)


class _Profiler:
    def __init__(self, report, json_dict=None, operations=None, raises=False):
        self._report = report
        self._json = json_dict or {}
        self._ops = operations or []
        self._raises = raises

    def report(self):
        return self._report

    def to_json_dict(self):
        if self._raises:
            raise ValueError("profiler boom")
        return self._json

    @property
    def operations(self):
        return self._ops


class _Span:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class _Tracer:
    def __init__(self, spans=None, raises=False):
        self._spans = spans or []
        self._raises = raises

    def slowest_spans(self, n=10):
        if self._raises:
            raise AttributeError("tracer boom")
        return self._spans[:n]


# ── _hint_for ───────────────────────────────────────────────────────────────


def test_hint_for_exact_prefix_and_none():
    assert "chunk-prefill" in obs._hint_for("gen.prefill")          # exact
    assert obs._hint_for("gen.prefill.inner") != ""                 # prefix match (57)
    assert obs._hint_for("completely.unknown.op") == ""             # no match


# ── detect_bottlenecks ──────────────────────────────────────────────────────


def test_detect_bottlenecks_none_profiler():
    assert detect_bottlenecks(None) == []


def test_detect_bottlenecks_filters_and_sorts():
    prof = _Profiler({
        "gen.prefill": _stats(300.0),
        "gen.tokenize": _stats(50.0),    # below threshold → excluded
        "server.model_load": _stats(500.0),
    })
    out = detect_bottlenecks(prof, threshold_ms=200.0)
    assert [b["op"] for b in out] == ["server.model_load", "gen.prefill"]  # desc by p99
    assert out[0]["hint"]  # remediation hint attached


# ── generate_report ─────────────────────────────────────────────────────────


def test_generate_report_all_none_is_ok():
    rep = generate_report(None, None)
    assert rep["status"] == "ok"
    assert rep["bottlenecks"] == [] and rep["profile"] == {}
    assert rep["profiler_ops"] == [] and rep["recent_spans"] == []


def test_generate_report_with_profiler_and_tracer():
    prof = _Profiler(
        {"gen.prefill": _stats(300.0)},
        json_dict={"gen.prefill": {"p99_ms": 300.0}},
        operations=["gen.prefill"],
    )
    tracer = _Tracer(spans=[_Span("s1"), _Span("s2")])
    rep = generate_report(prof, tracer, bottleneck_threshold_ms=200.0)
    assert rep["status"] == "degraded"  # a bottleneck exists
    assert rep["profile"] == {"gen.prefill": {"p99_ms": 300.0}}
    assert rep["profiler_ops"] == ["gen.prefill"]
    assert rep["recent_spans"] == [{"name": "s1"}, {"name": "s2"}]


def test_generate_report_profiler_error_swallowed():
    prof = _Profiler({}, raises=True)  # to_json_dict raises → 141-144
    rep = generate_report(prof, None)
    assert rep["profile"] == {} and rep["profiler_ops"] == []
    assert rep["status"] == "ok"


def test_generate_report_tracer_error_swallowed():
    rep = generate_report(None, _Tracer(raises=True))  # slowest_spans raises → 151-153
    assert rep["recent_spans"] == []
