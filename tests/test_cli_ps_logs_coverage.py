"""Coverage for squish.cli cmd_ps (HTTP /api/ps + startup profile) and cmd_logs
(daemon log tail). HTTP is mocked via urllib.request.urlopen; rich/plain via
squish.ui._RICH_AVAILABLE. Host-agnostic (no real server / network).
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

import pytest

from squish import cli


def _ns(**kw):
    kw.setdefault("host", "127.0.0.1")
    kw.setdefault("port", 11435)
    return argparse.Namespace(**kw)


def _set_rich(monkeypatch, value):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", value)


class _FakeResp:
    def __init__(self, data):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_http(monkeypatch, responses, *, fail=False):
    def _open(req, timeout=None):
        if fail:
            raise urllib.error.URLError("connection refused")
        url = req.full_url if hasattr(req, "full_url") else req
        for path, data in responses.items():
            if path in url:
                if isinstance(data, Exception):
                    raise data
                return _FakeResp(data)
        raise urllib.error.URLError("404")

    monkeypatch.setattr(urllib.request, "urlopen", _open)


# ── cmd_ps ───────────────────────────────────────────────────────────────────


def test_ps_connection_error_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _mock_http(monkeypatch, {}, fail=True)
    cli.cmd_ps(_ns(startup=False))
    assert "No server running" in capsys.readouterr().out


def test_ps_connection_error_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _mock_http(monkeypatch, {}, fail=True)
    cli.cmd_ps(_ns(startup=False))
    assert "No server running" in capsys.readouterr().out


def test_ps_no_models_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _mock_http(monkeypatch, {"/api/ps": {"models": []}})
    cli.cmd_ps(_ns(startup=False))
    assert "No model loaded" in capsys.readouterr().out


def test_ps_no_models_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _mock_http(monkeypatch, {"/api/ps": {"models": []}})
    cli.cmd_ps(_ns(startup=False))
    assert "No model loaded" in capsys.readouterr().out


def _model(size, *, details=None):
    return {"name": "qwen3-8b", "size": size, "details": details or {}}


def test_ps_models_rich_full_details(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    m = _model(
        int(2e9),
        details={
            "family": "qwen",
            "parameter_size": "8B",
            "quantization_level": "int4",
            "context_length": 32768,
        },
    )
    _mock_http(monkeypatch, {"/api/ps": {"models": [m]}})
    cli.cmd_ps(_ns(startup=False))
    out = capsys.readouterr().out
    assert "qwen3-8b" in out and "32,768" in out


def test_ps_models_rich_minimal_details(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    # no detail fields → rich detail-line branches all skipped (false arcs)
    _mock_http(monkeypatch, {"/api/ps": {"models": [_model(int(2e9))]}})
    cli.cmd_ps(_ns(startup=False))
    assert "qwen3-8b" in capsys.readouterr().out


def test_ps_models_plain_minimal_details(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    # missing detail fields → those lines skipped; MB-size branch
    _mock_http(monkeypatch, {"/api/ps": {"models": [_model(int(5e6))]}})
    cli.cmd_ps(_ns(startup=False))
    assert "qwen3-8b" in capsys.readouterr().out


def test_ps_models_plain_full_details(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    m = _model(
        int(2e9),
        details={
            "family": "qwen",
            "parameter_size": "8B",
            "quantization_level": "int4",
            "context_length": 32768,
        },
    )
    _mock_http(monkeypatch, {"/api/ps": {"models": [m]}})
    cli.cmd_ps(_ns(startup=False))
    out = capsys.readouterr().out
    assert "Family" in out and "Quant" in out and "32,768" in out


def test_ps_models_zero_size(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _mock_http(monkeypatch, {"/api/ps": {"models": [_model(0)]}})
    cli.cmd_ps(_ns(startup=False))
    assert "—" in capsys.readouterr().out


def test_ps_startup_profile_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _mock_http(
        monkeypatch,
        {
            "/api/ps": {"models": []},
            "/v1/startup-profile": {"total_ms": 1200, "phases": {"load": 800, "warm": 400}},
        },
    )
    cli.cmd_ps(_ns(startup=True))
    assert "Startup profile" in capsys.readouterr().out


def test_ps_startup_profile_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _mock_http(
        monkeypatch,
        {
            "/api/ps": {"models": []},
            "/v1/startup-profile": {"total_ms": 900, "phases": {"load": 900}},
        },
    )
    cli.cmd_ps(_ns(startup=True))
    assert "Startup profile" in capsys.readouterr().out


def test_ps_startup_profile_error_swallowed(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _mock_http(
        monkeypatch,
        {
            "/api/ps": {"models": []},
            "/v1/startup-profile": urllib.error.URLError("no profile"),
        },
    )
    cli.cmd_ps(_ns(startup=True))  # startup fetch fails → logged + swallowed
    assert "No model loaded" in capsys.readouterr().out


# ── cmd_logs ─────────────────────────────────────────────────────────────────


def test_logs_no_file(monkeypatch, tmp_path, capsys):
    cli.cmd_logs(_ns(log_file=str(tmp_path / "absent.log"), tail=50, follow=False))
    assert "No log file found" in capsys.readouterr().out


def test_logs_empty_file(tmp_path, capsys):
    f = tmp_path / "daemon.log"
    f.write_text("")
    cli.cmd_logs(_ns(log_file=str(f), tail=50, follow=False))
    assert "is empty" in capsys.readouterr().out


def test_logs_tail_lines(tmp_path, capsys):
    f = tmp_path / "daemon.log"
    f.write_text("".join(f"line{i}\n" for i in range(100)))
    cli.cmd_logs(_ns(log_file=str(f), tail=5, follow=False))
    out = capsys.readouterr().out
    assert "line99" in out and "line95" in out and "line94\n" not in out


def test_logs_default_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "squish_home", lambda: tmp_path)
    # no log_file attr → default squish_home()/daemon.log (absent)
    cli.cmd_logs(_ns(tail=50, follow=False))
    assert "No log file found" in capsys.readouterr().out
