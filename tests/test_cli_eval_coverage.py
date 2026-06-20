"""Coverage for squish.cli cmd_eval — lm_eval harness + score binding.
subprocess.run is mocked to emulate mlx_lm evaluate writing eval_* result
files; squash EvalBinder is faked for the bind path. Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from squish import cli


def _ns(model_dir, output_dir, **kw):
    kw.setdefault("tasks", "piqa")
    kw.setdefault("limit", None)
    kw.setdefault("baseline", None)
    kw.setdefault("no_bind", True)
    return argparse.Namespace(model_dir=str(model_dir), output_dir=str(output_dir), **kw)


def _model(tmp_path, name="piqa-model", *, config=True):
    d = tmp_path / name
    d.mkdir()
    if config:
        (d / "config.json").write_text("{}")
    return d


def _mock_run(
    monkeypatch,
    *,
    returncode=0,
    payload=None,
    write=True,
    raw_text=None,
    stdout="ran\n",
    stderr="trace\nboom",
):
    def _run(cmd, **kw):
        if write and returncode == 0:
            od = Path(cmd[cmd.index("--output-dir") + 1])
            task = cmd[cmd.index("--tasks") + 1]
            od.mkdir(parents=True, exist_ok=True)
            text = (
                raw_text
                if raw_text is not None
                else json.dumps(
                    payload if payload is not None else {"results": {task: {"acc_norm,none": 0.71}}}
                )
            )
            (od / f"eval_{task}.json").write_text(text)
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", _run)


# ── guard branches ───────────────────────────────────────────────────────────


def test_eval_model_dir_missing(tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_eval(_ns(tmp_path / "absent", tmp_path / "out"))


def test_eval_npy_dir_no_export(tmp_path):
    d = _model(tmp_path, config=False)  # no config.json, no 4bit sentinel
    with pytest.raises(SystemExit):
        cli.cmd_eval(_ns(d, tmp_path / "out"))


def test_eval_npy_dir_redirects_to_4bit(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path, config=False)
    (d / ".squish_4bit_ready").write_text("")
    fb = d / "squish_4bit"
    fb.mkdir()
    (fb / "config.json").write_text("{}")
    _mock_run(monkeypatch)
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "INT4 export found" in capsys.readouterr().out


# ── per-task result handling ─────────────────────────────────────────────────


def test_eval_success_results_wrapped(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, payload={"results": {"piqa": {"acc_norm,none": 0.71}}})
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    out = capsys.readouterr().out
    assert "71.00%" in out and "Average" in out


def test_eval_success_flat_format(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, payload={"piqa": {"acc_norm,none": 0.65}})  # no "results" wrapper
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "65.00%" in capsys.readouterr().out


def test_eval_metric_fallback_to_acc(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    # piqa primary is acc_norm,none; provide only acc,none → fallback loop
    _mock_run(monkeypatch, payload={"results": {"piqa": {"acc,none": 0.55}}})
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "55.00%" in capsys.readouterr().out


def test_eval_metric_not_found(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, payload={"results": {"piqa": {"unknown": 1}}})
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "metric not found" in capsys.readouterr().out


def test_eval_subprocess_failure(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, returncode=1)
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    err = capsys.readouterr().err
    assert "FAILED" in err and "task(s) failed" in err


def test_eval_success_empty_stdout(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, stdout="")  # empty stdout → print(proc.stdout) skipped
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "71.00%" in capsys.readouterr().out


def test_eval_failure_empty_stderr(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, returncode=2, stderr="")  # failure, no stderr tail
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "FAILED" in capsys.readouterr().err


def test_eval_no_output_file(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, write=False)  # returncode 0 but no eval_* file
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "no output file" in capsys.readouterr().out


def test_eval_json_parse_failure(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    _mock_run(monkeypatch, raw_text="{not json")
    cli.cmd_eval(_ns(d, tmp_path / "out"))
    assert "JSON parse failed" in capsys.readouterr().out


# ── thinking / limit / bind ──────────────────────────────────────────────────


def test_eval_qwen3_thinking_and_limit(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path, name="Qwen3-8b")
    _mock_run(monkeypatch)
    cli.cmd_eval(_ns(d, tmp_path / "out", limit=10))
    out = capsys.readouterr().out
    assert "thinking: disabled" in out and "limit" in out


def test_eval_bind_to_sidecar(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)
    (d / "cyclonedx-mlbom.json").write_text("{}")
    _mock_run(monkeypatch)
    bound = {}
    sb = types.ModuleType("squash.sbom_builder")
    sb.EvalBinder = types.SimpleNamespace(bind=lambda *a: bound.setdefault("done", True))
    monkeypatch.setitem(sys.modules, "squash", types.ModuleType("squash"))
    monkeypatch.setitem(sys.modules, "squash.sbom_builder", sb)
    cli.cmd_eval(_ns(d, tmp_path / "out", no_bind=False))
    assert bound.get("done") and "bound to sidecar" in capsys.readouterr().out


def test_eval_bind_no_sidecar_warns(tmp_path, monkeypatch, capsys):
    d = _model(tmp_path)  # no cyclonedx sidecar
    _mock_run(monkeypatch)
    cli.cmd_eval(_ns(d, tmp_path / "out", no_bind=False))
    assert "No sidecar found" in capsys.readouterr().err
