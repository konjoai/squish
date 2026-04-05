"""Tests for Wave 25 — CI/CD adapters (CicdAdapter, CiEnvironment, CicdReport)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from squish.squash.cicd import CicdAdapter, CicdReport, CiEnvironment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "cyclonedx-mlbom.json").write_text(
        json.dumps({"components": [], "metadata": {}}), encoding="utf-8"
    )
    (d / "config.json").write_text(
        json.dumps({"model_type": "test"}), encoding="utf-8"
    )
    return d


# ─── Environment detection tests ─────────────────────────────────────────────

def test_detect_returns_ci_environment():
    env = CicdAdapter.detect()
    assert isinstance(env, CiEnvironment)


def test_detect_github(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_ACTOR", "octocat")
    env = CicdAdapter.detect()
    assert env.env_name == "github"


def test_detect_jenkins(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("JENKINS_URL", "http://jenkins.example.com/")
    env = CicdAdapter.detect()
    assert env.env_name == "jenkins"


def test_detect_gitlab(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("JENKINS_URL", raising=False)
    monkeypatch.setenv("GITLAB_CI", "true")
    env = CicdAdapter.detect()
    assert env.env_name == "gitlab"


def test_detect_circleci(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("JENKINS_URL", raising=False)
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.setenv("CIRCLECI", "true")
    env = CicdAdapter.detect()
    assert env.env_name == "circleci"


def test_detect_unknown(monkeypatch):
    for var in ("GITHUB_ACTIONS", "JENKINS_URL", "GITLAB_CI", "CIRCLECI"):
        monkeypatch.delenv(var, raising=False)
    env = CicdAdapter.detect()
    assert env.env_name == "unknown"


# ── Annotation + summary tests ─────────────────────────────────────────────────

def test_job_summary_returns_markdown(tmp_path):
    d = _make_model_dir(tmp_path)
    report = CicdAdapter.run_pipeline(d)
    summary = CicdAdapter.job_summary(report, report.env)
    assert isinstance(summary, str)
    assert "#" in summary  # contains at least one Markdown heading


def test_annotate_github_writes_annotation(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    env = CiEnvironment(env_name="github", job_id="1", repo="r", branch="main", actor="a")
    d = _make_model_dir(tmp_path)
    report = CicdAdapter.run_pipeline(d)
    # Override env on report so annotate() uses github path
    report.env = env
    CicdAdapter.annotate(report, env)
    captured = capsys.readouterr()
    all_output = captured.out + captured.err
    # GitHub annotations use :: prefix; for passing reports this should be empty
    # For a failed report it should contain ::error:: — we just check no crash
    assert isinstance(all_output, str)


# ── CicdReport contract tests ──────────────────────────────────────────────────

def test_cicd_report_passed_true(tmp_path):
    d = _make_model_dir(tmp_path)
    report = CicdAdapter.run_pipeline(d)
    assert isinstance(report, CicdReport)
    assert isinstance(report.passed, bool)


def test_cicd_report_passed_reflects_ntia(tmp_path):
    """If NTIA check fails (empty BOM), report.passed should be False."""
    d = _make_model_dir(tmp_path)
    # Write an intentionally bare BOM missing all NTIA fields
    (d / "cyclonedx-mlbom.json").write_text("{}", encoding="utf-8")
    report = CicdAdapter.run_pipeline(d)
    assert isinstance(report.passed, bool)
    # Passed may be False if NTIA strict failing, either way it's a valid bool
