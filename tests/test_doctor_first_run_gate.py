"""
tests/test_doctor_first_run_gate.py

Unit tests for the version-keyed first-run health gate that folds
``squish doctor`` into ``squish run`` (squish/cli.py).

The real environment checks are mocked via ``run_health_checks`` so these tests
are deterministic on any platform. They assert the Reference Decision Table:

    | marker state           | SKIP | checks run | proceeds | marker written |
    |------------------------|------|------------|----------|----------------|
    | absent                 | no   | yes        | iff ok   | iff ok         |
    | == __version__         | no   | no         | yes      | unchanged      |
    | != __version__ (stale) | no   | yes        | iff ok   | iff ok         |
    | any                    | yes  | no         | yes      | unchanged      |
    | absent, required fails | no   | yes        | NO(_die) | no             |
    | absent, only opt unmet | no   | yes        | yes      | yes            |
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415

    return cli


def _version() -> str:
    from squish import __version__  # noqa: PLC0415

    return __version__


@pytest.fixture()
def marker(tmp_path, monkeypatch):
    """Redirect the doctor marker into a temp dir for the duration of a test."""
    cli = _import_cli()
    path = tmp_path / ".squish" / ".doctor_ok"
    monkeypatch.setattr(cli, "_DOCTOR_MARKER", path)
    return path


def _args(skip_doctor: bool = False) -> argparse.Namespace:
    return argparse.Namespace(skip_doctor=skip_doctor)


# All required checks pass; one optional feature unmet (never affects ok).
_RESULTS_OK = [
    {"label": "macOS / Apple Silicon", "passed": True, "fix": "", "optional": False},
    {"label": "squash-ai", "passed": False, "fix": "pip install squash-ai", "optional": True},
]
_RESULTS_REQUIRED_FAIL = [
    {"label": "macOS / Apple Silicon", "passed": True, "fix": "", "optional": False},
    {"label": "mlx", "passed": False, "fix": "pip install mlx", "optional": False},
]


# ── 1. marker absent → checks invoked, marker written with current version ─────
def test_marker_absent_runs_checks_and_writes_marker(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    assert not marker.exists()
    with patch.object(cli, "run_health_checks", return_value=(True, _RESULTS_OK)) as hc:
        cli._first_run_health_gate(_args())
    hc.assert_called_once()
    assert marker.read_text().strip() == _version()


# ── 2. marker == __version__ → checks NOT invoked ──────────────────────────────
def test_marker_current_version_skips_checks(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_version())
    with patch.object(cli, "run_health_checks") as hc:
        cli._first_run_health_gate(_args())
    hc.assert_not_called()
    # Marker is left untouched.
    assert marker.read_text().strip() == _version()


# ── 3. marker older version → checks invoked again (upgrade case) ──────────────
def test_marker_stale_version_reruns_checks(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("0.0.1-old")
    with patch.object(cli, "run_health_checks", return_value=(True, _RESULTS_OK)) as hc:
        cli._first_run_health_gate(_args())
    hc.assert_called_once()
    assert marker.read_text().strip() == _version()


# ── 4. SQUISH_SKIP_DOCTOR=1 → checks NOT invoked even when marker absent ────────
def test_env_skip_bypasses_checks(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.setenv("SQUISH_SKIP_DOCTOR", "1")
    assert not marker.exists()
    with patch.object(cli, "run_health_checks") as hc:
        cli._first_run_health_gate(_args())
    hc.assert_not_called()
    assert not marker.exists()


def test_flag_skip_bypasses_checks(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    with patch.object(cli, "run_health_checks") as hc:
        cli._first_run_health_gate(_args(skip_doctor=True))
    hc.assert_not_called()
    assert not marker.exists()


# ── 5. required check fails → abort (SystemExit) AND marker not written ─────────
def test_required_failure_aborts_and_no_marker(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    with patch.object(cli, "run_health_checks", return_value=(False, _RESULTS_REQUIRED_FAIL)) as hc:
        with pytest.raises(SystemExit):
            cli._first_run_health_gate(_args())
    hc.assert_called_once()
    assert not marker.exists()


# ── 6. only optional checks unmet → proceeds AND marker written ────────────────
def test_only_optional_unmet_proceeds_and_writes_marker(marker, monkeypatch):
    cli = _import_cli()
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    # _RESULTS_OK already has an unmet optional check; ok is True.
    with patch.object(cli, "run_health_checks", return_value=(True, _RESULTS_OK)) as hc:
        cli._first_run_health_gate(_args())
    hc.assert_called_once()
    assert marker.read_text().strip() == _version()
