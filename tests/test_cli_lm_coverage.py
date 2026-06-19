"""Coverage for squish.cli cmd_lm — LM Studio status + disk inventory.
probe_lm_studio and LocalModelScanner are faked; rich/plain via
squish.ui._RICH_AVAILABLE. Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


def _set_rich(monkeypatch, value):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", value)


def _install(monkeypatch, *, status, disk):
    import squish.experimental.lm_studio_bridge as bridge
    import squish.serving.local_model_scanner as lms

    monkeypatch.setattr(bridge, "probe_lm_studio", lambda timeout=1.0: status)
    monkeypatch.setattr(
        lms, "LocalModelScanner", lambda: types.SimpleNamespace(scan_lm_studio=lambda: disk)
    )


def _status(running=True, *, version="0.3.0", loaded=None):
    return types.SimpleNamespace(
        running=running,
        base_url="http://localhost:1234",
        loaded_models=loaded or [],
        server_version=version,
    )


def _disk(name, size):
    return types.SimpleNamespace(
        name=name, path=f"/models/{name}", size_bytes=size, family="qwen", params="8B"
    )


# ── status action ────────────────────────────────────────────────────────────


def test_status_rich_running_loaded(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _install(monkeypatch, status=_status(True, loaded=["m1"]), disk=[_disk("d", int(2e9))])
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    assert "running" in capsys.readouterr().out


def test_status_rich_running_no_version_no_loaded(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _install(monkeypatch, status=_status(True, version="", loaded=[]), disk=[])
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    assert "No model currently loaded" in capsys.readouterr().out


def test_status_plain_running_with_version_and_loaded(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _install(monkeypatch, status=_status(True, version="0.3.1", loaded=["m1"]), disk=[])
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    out = capsys.readouterr().out
    assert "Version : 0.3.1" in out and "Loaded" in out


def test_status_rich_not_running_empty_disk(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _install(monkeypatch, status=_status(False), disk=[])
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    assert "not running" in capsys.readouterr().out


def test_status_plain_running_no_version_no_loaded(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _install(
        monkeypatch,
        status=_status(True, version="", loaded=[]),
        disk=[_disk("mb", int(5e6)), _disk("zero", 0)],
    )
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    out = capsys.readouterr().out
    assert "running" in out and "No model currently loaded" in out


def test_status_plain_not_running_empty(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _install(monkeypatch, status=_status(False), disk=[])
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    assert "not running" in capsys.readouterr().out


def test_status_json(monkeypatch, capsys):
    _install(monkeypatch, status=_status(True, loaded=["m1"]), disk=[])
    cli.cmd_lm(_ns(lm_action="status", json_=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is True and payload["loaded_models"] == ["m1"]


# ── models action ────────────────────────────────────────────────────────────


def test_models_rich_with_disk(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _install(monkeypatch, status=_status(), disk=[_disk("big", int(2e9)), _disk("z", 0)])
    cli.cmd_lm(_ns(lm_action="models", json_=False))
    assert "model(s) found" in capsys.readouterr().out


def test_models_plain_with_disk(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _install(monkeypatch, status=_status(), disk=[_disk("big", int(2e9))])
    cli.cmd_lm(_ns(lm_action="models", json_=False))
    assert "model(s) found" in capsys.readouterr().out


def test_models_plain_empty(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    _install(monkeypatch, status=_status(), disk=[])
    cli.cmd_lm(_ns(lm_action="models", json_=False))
    assert "No LM Studio models found" in capsys.readouterr().out


def test_models_json(monkeypatch, capsys):
    _install(monkeypatch, status=_status(), disk=[_disk("a", int(1e9))])
    cli.cmd_lm(_ns(lm_action="models", json_=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "a" and payload[0]["params"] == "8B"


def test_models_empty_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _install(monkeypatch, status=_status(), disk=[])
    cli.cmd_lm(_ns(lm_action="models", json_=False))
    assert "No LM Studio models found" in capsys.readouterr().out


def test_lm_rich_import_failure_falls_back(monkeypatch, capsys):
    _install(monkeypatch, status=_status(False), disk=[])
    monkeypatch.setitem(sys.modules, "squish.ui", None)  # → ImportError → plain
    cli.cmd_lm(_ns(lm_action="status", json_=False))
    assert "not running" in capsys.readouterr().out
