"""Coverage for squish.cli observability commands cmd_route and cmd_quality.
cmd_route uses the real offline PromptRouter; cmd_quality is driven with a fake
quality-monitor report. Pure stdlib — host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import types

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


# ── cmd_route ────────────────────────────────────────────────────────────────


def test_route_table_output(capsys):
    cli.cmd_route(_ns(prompt="Write a Python function to sort a list", json=False))
    out = capsys.readouterr().out
    assert "Category" in out and "Confidence" in out


def test_route_json_output(capsys):
    cli.cmd_route(_ns(prompt="hello there", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert "category" in payload


# ── cmd_quality ──────────────────────────────────────────────────────────────


def _model_stats(model_id="m1", *, ttft=12.0):
    return types.SimpleNamespace(
        model_id=model_id,
        n_requests=10,
        n_errors=1,
        error_rate=0.1,
        latency_p50=5.0,
        latency_p95=9.0,
        latency_p99=11.0,
        latency_mean=6.0,
        tokens_per_sec_p50=40.0,
        tokens_per_sec_mean=42.0,
        ttft_p50=ttft,
        ttft_p95=ttft,
        to_dict=lambda: {"model_id": model_id, "n_requests": 10},
    )


def _fake_report(models, *, window=3600):
    return types.SimpleNamespace(
        models=models,
        window_seconds=window,
        total_requests=sum(m.n_requests for m in models),
        generated_at="2026-06-19T00:00:00Z",
    )


def _install_monitor(monkeypatch, report):
    import squish.serving.quality_monitor as qm

    monkeypatch.setattr(
        qm,
        "get_quality_monitor",
        lambda: types.SimpleNamespace(report=lambda window_seconds: report),
    )


def test_quality_table_with_models(monkeypatch, capsys):
    _install_monitor(monkeypatch, _fake_report([_model_stats("alpha")]))
    cli.cmd_quality(_ns(window=3600, model=None, json=False))
    out = capsys.readouterr().out
    assert "Model: alpha" in out and "TTFT P50" in out


def test_quality_ttft_none_renders_na(monkeypatch, capsys):
    _install_monitor(monkeypatch, _fake_report([_model_stats("beta", ttft=None)]))
    cli.cmd_quality(_ns(window=3600, model=None, json=False))
    assert "N/A" in capsys.readouterr().out


def test_quality_no_models(monkeypatch, capsys):
    _install_monitor(monkeypatch, _fake_report([]))
    cli.cmd_quality(_ns(window=60, model=None, json=False))
    assert "no requests recorded" in capsys.readouterr().out


def test_quality_model_filter(monkeypatch, capsys):
    report = _fake_report([_model_stats("alpha"), _model_stats("beta")])
    _install_monitor(monkeypatch, report)
    cli.cmd_quality(_ns(window=3600, model="beta", json=False))
    out = capsys.readouterr().out
    assert "Model: beta" in out and "Model: alpha" not in out


def test_quality_json_output(monkeypatch, capsys):
    _install_monitor(monkeypatch, _fake_report([_model_stats("alpha")]))
    cli.cmd_quality(_ns(window=3600, model=None, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_requests"] == 10 and payload["models"][0]["model_id"] == "alpha"
