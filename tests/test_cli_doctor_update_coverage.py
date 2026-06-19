"""Coverage for squish.cli system commands cmd_doctor and cmd_update.
Health checks and pip subprocess calls are stubbed; rich/non-rich branches are
driven via squish.ui._RICH_AVAILABLE. Host-agnostic (no real pip / network).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


def _check(label, passed, *, fix="", optional=False):
    return {"label": label, "passed": passed, "fix": fix, "optional": optional}


# ── cmd_doctor ───────────────────────────────────────────────────────────────


def _set_rich(monkeypatch, value):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", value)


def test_doctor_all_pass_with_optional_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    results = [
        _check("Python 3.10+", True),
        _check("mlx-vlm (optional)", False, fix="pip install mlx-vlm", optional=True),
        _check("extra (optional)", False, optional=True),  # no fix
    ]
    monkeypatch.setattr(cli, "run_health_checks", lambda: (True, results))
    cli.cmd_doctor(_ns(report=False))
    assert "optional feature" in capsys.readouterr().out


def test_doctor_all_pass_no_optional_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "run_health_checks", lambda: (True, [_check("core", True)]))
    cli.cmd_doctor(_ns(report=False))
    assert "All checks passed" in capsys.readouterr().out


def test_doctor_failure_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    results = [_check("mlx", False, fix="pip install mlx")]
    monkeypatch.setattr(cli, "run_health_checks", lambda: (False, results))
    cli.cmd_doctor(_ns(report=False))
    assert "Some checks failed" in capsys.readouterr().out


def test_doctor_failure_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    results = [_check("mlx", False, fix="pip install mlx")]
    monkeypatch.setattr(cli, "run_health_checks", lambda: (False, results))
    cli.cmd_doctor(_ns(report=False))
    assert "Some checks failed" in capsys.readouterr().out


def test_doctor_writes_report(monkeypatch, tmp_path, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "run_health_checks", lambda: (True, [_check("core", True)]))
    monkeypatch.setattr(cli, "squish_home", lambda: tmp_path)
    cli.cmd_doctor(_ns(report=True))
    out = capsys.readouterr().out
    assert "Report saved" in out
    assert list(tmp_path.glob("doctor-report-*.json"))


# ── cmd_update ───────────────────────────────────────────────────────────────


def _pip_ok(stdout="Successfully installed squish-9.34.2"):
    return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _pip_fail():
    return types.SimpleNamespace(returncode=1, stdout="", stderr="ERROR: boom\nline2")


def test_update_success_plain_no_version_change(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _pip_ok())
    monkeypatch.setattr("squish.dist_version", lambda: "9.34.2")
    cli.cmd_update(_ns(all=False))
    assert "up to date" in capsys.readouterr().out


def test_update_rich_with_all_and_version_bump(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _pip_ok())
    versions = iter(["9.34.0", "9.35.0"])  # dist_version() called start + end
    monkeypatch.setattr("squish.dist_version", lambda: next(versions, "9.35.0"))
    cli.cmd_update(_ns(all=True))  # --all → extra packages
    assert "upgraded" in capsys.readouterr().out


def test_update_pip_failure_branch(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _pip_fail())
    monkeypatch.setattr("squish.dist_version", lambda: "9.34.2")
    cli.cmd_update(_ns(all=False))
    assert "failed" in capsys.readouterr().out


def test_update_no_pip_summary_line(monkeypatch, capsys):
    # pip succeeds but stdout has no recognisable summary line → "done"
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _pip_ok(stdout="noise"))
    monkeypatch.setattr("squish.dist_version", lambda: "9.34.2")
    cli.cmd_update(_ns(all=False))
    assert "done" in capsys.readouterr().out


def test_update_rich_import_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "squish.ui", None)  # from squish.ui import → ImportError
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _pip_ok())
    monkeypatch.setattr("squish.dist_version", lambda: "9.34.2")
    cli.cmd_update(_ns(all=False))
    assert "squish update" in capsys.readouterr().out
