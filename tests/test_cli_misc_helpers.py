"""Coverage for cli.py pure-Python helpers: the already-quantized probe and the
first-run health gate. No MLX; the gate's environment probe is mocked."""

import types

import pytest

from squish import cli


# ── _model_is_already_quantized ────────────────────────────────────────────────


def test_already_quantized_no_config(tmp_path):
    assert cli._model_is_already_quantized(tmp_path) is False


def test_already_quantized_true(tmp_path):
    (tmp_path / "config.json").write_text('{"quantization": {"bits": 4}}')
    assert cli._model_is_already_quantized(tmp_path) is True


def test_already_quantized_false(tmp_path):
    (tmp_path / "config.json").write_text('{"hidden_size": 4096}')
    assert cli._model_is_already_quantized(tmp_path) is False


def test_already_quantized_bad_json(tmp_path):
    (tmp_path / "config.json").write_text("{ not valid json")
    assert cli._model_is_already_quantized(tmp_path) is False  # OSError/ValueError → False


# ── _first_run_health_gate ─────────────────────────────────────────────────────


def _args():
    return types.SimpleNamespace(skip_doctor=False)


def test_health_gate_skipped_by_env(monkeypatch):
    monkeypatch.setenv("SQUISH_SKIP_DOCTOR", "1")
    cli._first_run_health_gate(_args())  # early return, no checks run


def test_health_gate_skipped_by_flag(monkeypatch):
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    cli._first_run_health_gate(types.SimpleNamespace(skip_doctor=True))


def test_health_gate_marker_matches_version(monkeypatch, tmp_path):
    from squish import __version__

    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    marker = tmp_path / ".doctor_ok"
    marker.write_text(__version__)
    monkeypatch.setattr(cli, "_DOCTOR_MARKER", marker)
    cli._first_run_health_gate(_args())  # version matches → zero check work


def test_health_gate_failure_dies(monkeypatch, tmp_path):
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    monkeypatch.setattr(cli, "_DOCTOR_MARKER", tmp_path / ".doctor_ok")
    monkeypatch.setattr(
        cli,
        "run_health_checks",
        lambda: (False, [{"passed": False, "optional": False, "label": "MLX", "fix": "install"}]),
    )
    with pytest.raises(SystemExit):
        cli._first_run_health_gate(_args())


def test_health_gate_pass_writes_marker(monkeypatch, tmp_path):
    from squish import __version__

    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    marker = tmp_path / "sub" / ".doctor_ok"  # parent created by the gate
    monkeypatch.setattr(cli, "_DOCTOR_MARKER", marker)
    monkeypatch.setattr(cli, "run_health_checks", lambda: (True, []))
    cli._first_run_health_gate(_args())
    assert marker.read_text() == __version__  # marker persisted atomically


def test_health_gate_marker_read_and_write_errors(monkeypatch, tmp_path):
    # marker is a directory → read_text() raises OSError (treated as "no marker"),
    # and the final os.replace(tmp_file, <dir>) also raises OSError (logged, non-fatal).
    monkeypatch.delenv("SQUISH_SKIP_DOCTOR", raising=False)
    marker = tmp_path / ".doctor_ok"
    marker.mkdir()
    monkeypatch.setattr(cli, "_DOCTOR_MARKER", marker)
    monkeypatch.setattr(cli, "run_health_checks", lambda: (True, []))
    cli._first_run_health_gate(_args())  # must not raise
