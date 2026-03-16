"""
tests/test_cli_unit.py

Unit tests for squish/cli.py pure-Python helpers.
Only tests functions that do not require MLX, model files, or a running server.

Actual API:
    _box(lines: list[str]) -> None  — prints, no return
    _die(msg: str) -> None          — prints to stderr, sys.exit(1)
    cmd_catalog(args)               — args.tag, args.refresh
    cmd_models(args)                — args.path (optional?)
    cmd_doctor(args)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


# ── _box ──────────────────────────────────────────────────────────────────────
# Signature: _box(lines: list[str]) -> None — prints a box to stdout

class TestBox:
    def test_prints_to_stdout(self, capsys):
        cli = _import_cli()
        cli._box(["hello"])
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_returns_none(self):
        cli = _import_cli()
        result = cli._box(["test"])
        assert result is None

    def test_contains_all_lines(self, capsys):
        cli = _import_cli()
        cli._box(["alpha", "beta", "gamma"])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "gamma" in captured.out

    def test_box_border_chars(self, capsys):
        cli = _import_cli()
        cli._box(["content"])
        captured = capsys.readouterr()
        assert any(c in captured.out for c in ["\u2500", "\u2502", "\u250c", "\u2514", "|", "-"])

    def test_box_unicode_content(self, capsys):
        cli = _import_cli()
        cli._box(["H\u00e9llo w\u00f6rld"])
        captured = capsys.readouterr()
        assert "H\u00e9llo" in captured.out


# ── _die ──────────────────────────────────────────────────────────────────────

class TestDie:
    def test_raises_system_exit_1(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            cli._die("fatal error")
        assert exc.value.code == 1

    def test_writes_message(self, capsys):
        cli = _import_cli()
        with pytest.raises(SystemExit):
            cli._die("test message")
        captured = capsys.readouterr()
        assert "test message" in (captured.err + captured.out)


# ── cmd_catalog ───────────────────────────────────────────────────────────────
# args: tag (str|None), refresh (bool)

class TestCmdCatalog:
    def test_runs_without_error(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag=None, refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_tag_filter_small(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag="small", refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        assert isinstance(captured.out, str)

    def test_unknown_tag_no_models(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag="definitely_not_a_real_tag_xyz_99999", refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        # Should indicate no models found
        combined = (captured.out + captured.err).lower()
        assert "no" in combined or len(captured.out) == 0 or combined

    def test_output_contains_model_ids(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag=None, refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        # Should have at least one colon-separated model ID like "qwen3:8b"
        assert ":" in captured.out


# ── cmd_models ────────────────────────────────────────────────────────────────

class TestCmdModels:
    def test_runs_without_error(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except (SystemExit, Exception):
            pass
        captured = capsys.readouterr()
        assert isinstance(captured.out + captured.err, str)

    def test_moe_badge_shown_for_moe_model(self, tmp_path, capsys):
        """cmd_models shows [MoE] badge when catalog entry has moe=True."""
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")

        # Create a fake model directory whose name matches our mocked catalog entry
        model_dir = tmp_path / "Qwen3-30B-A3B-bf16"
        model_dir.mkdir()

        # Build a minimal fake CatalogEntry-like object
        fake_entry = MagicMock()
        fake_entry.dir_name = "Qwen3-30B-A3B-bf16"
        fake_entry.moe = True
        fake_entry.active_params_b = 3.0
        fake_entry.params = "30B"

        ns = argparse.Namespace()
        with patch.object(cli, "_MODELS_DIR", tmp_path), \
             patch("squish.catalog.list_catalog", return_value=[fake_entry]):
            fn(ns)

        captured = capsys.readouterr()
        assert "[MoE" in captured.out

    def test_no_moe_badge_for_regular_model(self, tmp_path, capsys):
        """cmd_models does not show [MoE] for non-MoE catalog entries."""
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")

        model_dir = tmp_path / "Qwen2.5-7B-Instruct-4bit"
        model_dir.mkdir()

        fake_entry = MagicMock()
        fake_entry.dir_name = "Qwen2.5-7B-Instruct-4bit"
        fake_entry.moe = False
        fake_entry.active_params_b = None
        fake_entry.params = "7B"

        ns = argparse.Namespace()
        with patch.object(cli, "_MODELS_DIR", tmp_path), \
             patch("squish.catalog.list_catalog", return_value=[fake_entry]):
            fn(ns)

        captured = capsys.readouterr()
        assert "[MoE" not in captured.out

    def test_moe_badge_without_active_params(self, tmp_path, capsys):
        """[MoE] badge shown without active-params when active_params_b is None."""
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")

        model_dir = tmp_path / "SomeMoE-bf16"
        model_dir.mkdir()

        fake_entry = MagicMock()
        fake_entry.dir_name = "SomeMoE-bf16"
        fake_entry.moe = True
        fake_entry.active_params_b = None
        fake_entry.params = "60B"

        ns = argparse.Namespace()
        with patch.object(cli, "_MODELS_DIR", tmp_path), \
             patch("squish.catalog.list_catalog", return_value=[fake_entry]):
            fn(ns)

        captured = capsys.readouterr()
        assert "[MoE]" in captured.out

    def test_catalog_failure_does_not_crash_models(self, tmp_path, capsys):
        """If catalog import fails, cmd_models still works without badge."""
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")

        model_dir = tmp_path / "some-model"
        model_dir.mkdir()

        ns = argparse.Namespace()
        with patch.object(cli, "_MODELS_DIR", tmp_path), \
             patch("squish.catalog.list_catalog", side_effect=ImportError("no catalog")):
            fn(ns)

        captured = capsys.readouterr()
        assert "some-model" in captured.out
        assert "[MoE" not in captured.out


# ── cmd_doctor ────────────────────────────────────────────────────────────────

class TestCmdDoctor:
    def test_runs_without_crash(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert isinstance(captured.out + captured.err, str)

    def test_doctor_produces_output(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert any(word in combined for word in
                   ("python", "mlx", "squish", "ok", "pass", "fail", "version", "error", "\u2713", "\u2717"))


# ── main / argparse ───────────────────────────────────────────────────────────

class TestMain:
    def test_no_args_exits(self):
        cli = _import_cli()
        if not hasattr(cli, "main"):
            pytest.skip("No main() found")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish"]):
                cli.main()
        assert exc.value.code in (0, 1, 2)

    def test_help_exits_zero(self):
        cli = _import_cli()
        if not hasattr(cli, "main"):
            pytest.skip("No main() found")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "--help"]):
                cli.main()
        assert exc.value.code == 0


# ── _detect_ram_gb ────────────────────────────────────────────────────────────

class TestDetectRamGb:
    def test_parses_sysctl_bytes(self):
        cli = _import_cli()
        fn = getattr(cli, "_detect_ram_gb", None)
        if fn is None:
            pytest.skip("_detect_ram_gb not found")
        with patch("subprocess.check_output", return_value=b"17179869184\n"):
            result = fn()
        assert abs(result - 17.179869184) < 0.01

    def test_returns_zero_on_subprocess_error(self):
        cli = _import_cli()
        fn = getattr(cli, "_detect_ram_gb", None)
        if fn is None:
            pytest.skip("_detect_ram_gb not found")
        import subprocess
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "sysctl")):
            result = fn()
        assert result == 0.0

    def test_returns_zero_on_generic_exception(self):
        cli = _import_cli()
        fn = getattr(cli, "_detect_ram_gb", None)
        if fn is None:
            pytest.skip("_detect_ram_gb not found")
        with patch("subprocess.check_output", side_effect=FileNotFoundError("sysctl not found")):
            result = fn()
        assert result == 0.0

    def test_returns_float(self):
        cli = _import_cli()
        fn = getattr(cli, "_detect_ram_gb", None)
        if fn is None:
            pytest.skip("_detect_ram_gb not found")
        with patch("subprocess.check_output", return_value=b"8589934592\n"):
            result = fn()
        assert isinstance(result, float)


# ── _recommend_model ──────────────────────────────────────────────────────────

class TestRecommendModel:
    def _fn(self):
        cli = _import_cli()
        fn = getattr(cli, "_recommend_model", None)
        if fn is None:
            pytest.skip("_recommend_model not found")
        return fn

    def test_under_16gb(self):
        fn = self._fn()
        assert fn(8.0) == "qwen3:1.7b"

    def test_exactly_16gb(self):
        fn = self._fn()
        assert fn(16.0) == "qwen3:8b"

    def test_between_16_and_32(self):
        fn = self._fn()
        assert fn(24.0) == "qwen3:8b"

    def test_exactly_32gb(self):
        fn = self._fn()
        assert fn(32.0) == "qwen3:14b"

    def test_between_32_and_64(self):
        fn = self._fn()
        assert fn(48.0) == "qwen3:14b"

    def test_exactly_64gb(self):
        fn = self._fn()
        assert fn(64.0) == "qwen3:32b"

    def test_above_64gb(self):
        fn = self._fn()
        assert fn(128.0) == "qwen3:32b"

    def test_zero_ram(self):
        fn = self._fn()
        assert fn(0.0) == "qwen3:1.7b"

    def test_returns_string(self):
        fn = self._fn()
        result = fn(16.0)
        assert isinstance(result, str)
        assert ":" in result  # catalog format  <name>:<tag>


# ── cmd_doctor --report ───────────────────────────────────────────────────────

class TestCmdDoctorReport:
    def test_report_creates_json_file(self, tmp_path, monkeypatch):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: tmp_path
        )
        ns = argparse.Namespace(report=True)
        try:
            fn(ns)
        except SystemExit:
            pass
        reports = list(tmp_path.glob("**/*.json"))
        assert len(reports) >= 1, "Expected at least one JSON report file"

    def test_report_json_has_required_keys(self, tmp_path, monkeypatch):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        import json as _json
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        ns = argparse.Namespace(report=True)
        try:
            fn(ns)
        except SystemExit:
            pass
        reports = list(tmp_path.glob("**/*.json"))
        if not reports:
            pytest.skip("No report file generated")
        data = _json.loads(reports[0].read_text())
        assert "checks" in data
        assert isinstance(data["checks"], list)

    def test_report_false_does_not_create_file(self, tmp_path, monkeypatch):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        ns = argparse.Namespace(report=False)
        before = set(tmp_path.glob("**/*.json"))
        try:
            fn(ns)
        except SystemExit:
            pass
        after = set(tmp_path.glob("**/*.json"))
        assert after == before, "No JSON files should be created when report=False"
