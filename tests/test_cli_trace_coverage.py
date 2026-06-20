"""Coverage for squish.cli cmd_trace — span trace / observability viewer.
All HTTP (GET + DELETE) mocked via urllib.request.urlopen. Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from squish import cli


def _ns(**kw):
    kw.setdefault("host", "127.0.0.1")
    kw.setdefault("port", 11435)
    kw.setdefault("chrome", None)
    return argparse.Namespace(**kw)


class _FakeResp:
    def __init__(self, data):
        self._data = data if isinstance(data, bytes) else json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock(monkeypatch, responses, *, fail=False):
    def _open(req, timeout=None):
        if fail:
            raise urllib.error.URLError("refused")
        url = req.full_url if hasattr(req, "full_url") else req
        for path, data in responses.items():
            if path in url:
                if isinstance(data, Exception):
                    raise data
                return _FakeResp(data)
        raise urllib.error.URLError("404")

    monkeypatch.setattr(urllib.request, "urlopen", _open)


# ── reset ────────────────────────────────────────────────────────────────────


def test_trace_reset_success(monkeypatch, capsys):
    _mock(monkeypatch, {"/v1/trace": {"cleared": 5}})
    cli.cmd_trace(_ns(trace_action="reset"))
    assert "Trace cleared" in capsys.readouterr().out


def test_trace_reset_server_error(monkeypatch, capsys):
    _mock(monkeypatch, {}, fail=True)
    cli.cmd_trace(_ns(trace_action="reset"))
    assert "Could not connect" in capsys.readouterr().out


# ── view --chrome ────────────────────────────────────────────────────────────


def test_trace_chrome_export(monkeypatch, tmp_path, capsys):
    out_file = tmp_path / "trace.json"
    _mock(monkeypatch, {"/v1/trace?format=chrome": b'{"traceEvents":[]}'})
    cli.cmd_trace(_ns(trace_action="view", chrome=str(out_file)))
    assert out_file.read_bytes() == b'{"traceEvents":[]}'
    assert "Chrome trace written" in capsys.readouterr().out


def test_trace_chrome_export_error(monkeypatch, tmp_path, capsys):
    _mock(monkeypatch, {}, fail=True)
    cli.cmd_trace(_ns(trace_action="view", chrome=str(tmp_path / "t.json")))
    assert "Could not connect" in capsys.readouterr().out


# ── obs ──────────────────────────────────────────────────────────────────────


def test_trace_obs_ok_no_bottlenecks(monkeypatch, capsys):
    _mock(monkeypatch, {"/v1/obs-report": {"status": "ok", "bottlenecks": [], "profile": {}}})
    cli.cmd_trace(_ns(trace_action="obs"))
    assert "No bottlenecks detected" in capsys.readouterr().out


def test_trace_obs_with_bottlenecks_and_profile(monkeypatch, capsys):
    data = {
        "status": "degraded",
        "bottlenecks": [
            {"op": "attn", "p99_ms": 250.0, "n_samples": 10, "hint": "use flash"},
            {"op": "mlp", "p99_ms": 120.0, "n_samples": 5},  # no hint
        ],
        "profile": {
            "attn": {"n_samples": 10, "mean_ms": 50.0, "p50_ms": 40.0, "p99_ms": 250.0},
            "mlp": {"n_samples": 5, "mean_ms": 10.0, "p50_ms": 9.0, "p99_ms": 50.0},
        },
    }
    _mock(monkeypatch, {"/v1/obs-report": data})
    cli.cmd_trace(_ns(trace_action="obs"))
    out = capsys.readouterr().out
    assert "Bottlenecks" in out and "use flash" in out and "APM Profile" in out


def test_trace_obs_server_error(monkeypatch, capsys):
    _mock(monkeypatch, {}, fail=True)
    cli.cmd_trace(_ns(trace_action="obs"))
    assert "Could not connect" in capsys.readouterr().out


# ── default view ─────────────────────────────────────────────────────────────


def test_trace_view_with_spans_and_remediation(monkeypatch, capsys):
    trace = {
        "tracing_enabled": True,
        "total_spans": 3,
        "slowest_spans": [
            {"name": "fast", "duration_ms": 10.0, "status": "ok"},  # <50 green
            {"name": "mid", "duration_ms": 200.0, "status": "ok"},  # <500 yellow
            {"name": "slow", "duration_ms": 900.0, "status": "error"},  # >=500 red
        ],
    }
    obs = {"bottlenecks": [{"op": "attn", "hint": "use flash"}]}
    _mock(monkeypatch, {"/v1/trace": trace, "/v1/obs-report": obs})
    cli.cmd_trace(_ns(trace_action="view"))
    out = capsys.readouterr().out
    assert "Span Trace" in out and "Remediation Report" in out


def test_trace_view_disabled_no_spans(monkeypatch, capsys):
    trace = {"tracing_enabled": False, "total_spans": 0, "slowest_spans": [], "hint": "use --trace"}
    # obs-report fails → remediation block swallowed
    _mock(monkeypatch, {"/v1/trace": trace, "/v1/obs-report": urllib.error.URLError("x")})
    cli.cmd_trace(_ns(trace_action="view"))
    out = capsys.readouterr().out
    assert "No spans collected yet" in out and "OFF" in out


def test_trace_view_remediation_empty_bottlenecks(monkeypatch, capsys):
    trace = {"tracing_enabled": True, "total_spans": 0, "slowest_spans": []}
    _mock(monkeypatch, {"/v1/trace": trace, "/v1/obs-report": {"bottlenecks": []}})
    cli.cmd_trace(_ns(trace_action="view"))
    out = capsys.readouterr().out
    assert "No spans collected yet" in out and "Remediation Report" not in out


def test_trace_view_remediation_bottleneck_without_hint(monkeypatch, capsys):
    trace = {"tracing_enabled": True, "total_spans": 0, "slowest_spans": []}
    obs = {"bottlenecks": [{"op": "attn"}]}  # no hint → inner branch false
    _mock(monkeypatch, {"/v1/trace": trace, "/v1/obs-report": obs})
    cli.cmd_trace(_ns(trace_action="view"))
    assert "Remediation Report" in capsys.readouterr().out


def test_trace_view_server_error(monkeypatch, capsys):
    _mock(monkeypatch, {}, fail=True)
    cli.cmd_trace(_ns(trace_action="view"))
    assert "Could not connect" in capsys.readouterr().out
