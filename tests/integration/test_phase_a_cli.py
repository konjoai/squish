"""
tests/test_phase_a_cli.py

Phase A4 tests: cmd_convert_model in squish/cli.py

Covers all branches of cmd_convert_model:
  - source_path not found (SystemExit 1)
  - dry_run=True (prints info, returns)
  - mlx_lm not installed (ImportError → SystemExit 1)
  - mlx_lm.convert raises on FFN pass (SystemExit 1)
  - embed_bits == ffn_bits → single mlx_lm.convert call (success)
  - embed_bits != ffn_bits, embed pass raises (SystemExit 1)
  - embed_bits != ffn_bits, both passes succeed (success)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


def _ns(source, output, ffn_bits=4, embed_bits=6, dry_run=False):
    return argparse.Namespace(
        source_path=str(source),
        output_path=str(output),
        ffn_bits=ffn_bits,
        embed_bits=embed_bits,
        dry_run=dry_run,
    )


class TestCmdConvertModel:
    @pytest.fixture(autouse=True)
    def _ensure_mlx_lm_importable(self, monkeypatch):
        """Install a MagicMock for mlx_lm when the real package is absent or
        broken (e.g. libmlx.so not found on Linux).

        Tests that need mlx_lm to be *missing* override via
        patch.dict(sys.modules, {'mlx_lm': None}) which takes precedence.
        """
        try:
            import mlx_lm  # noqa: F401
        except Exception:
            from unittest.mock import MagicMock
            monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock())

    def test_source_not_found_exits(self, tmp_path: Path):
        """Source path missing → _die() → SystemExit(1)."""
        cli = _import_cli()
        ns = _ns(tmp_path / "nonexistent", tmp_path / "out")
        with pytest.raises(SystemExit) as exc:
            cli.cmd_convert_model(ns)
        assert exc.value.code == 1

    def test_dry_run_prints_info_and_returns(self, tmp_path: Path, capsys):
        """dry_run=True: prints settings, does not call mlx_lm.convert."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        ns = _ns(source, tmp_path / "out", ffn_bits=4, embed_bits=6, dry_run=True)
        cli.cmd_convert_model(ns)
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "4" in out   # ffn_bits
        assert "6" in out   # embed_bits

    def test_mlx_lm_unavailable_exits(self, tmp_path: Path):
        """mlx_lm not importable → SystemExit(1)."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        ns = _ns(source, tmp_path / "out")
        with patch.dict(sys.modules, {"mlx_lm": None}):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_convert_model(ns)
        assert exc.value.code == 1

    def test_ffn_conversion_failure_exits(self, tmp_path: Path):
        """mlx_lm.convert raises on FFN pass → SystemExit(1)."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        ns = _ns(source, tmp_path / "out")
        with patch("mlx_lm.convert", side_effect=Exception("conv failed")):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_convert_model(ns)
        assert exc.value.code == 1

    def test_same_bits_single_pass_success(self, tmp_path: Path, capsys):
        """embed_bits == ffn_bits: one mlx_lm.convert call, prints saved message."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        output = tmp_path / "out"
        ns = _ns(source, output, ffn_bits=4, embed_bits=4)
        with patch("mlx_lm.convert") as mock_conv:
            cli.cmd_convert_model(ns)
        assert mock_conv.call_count == 1
        assert "saved" in capsys.readouterr().out.lower()

    def test_different_bits_embed_pass_failure_exits(self, tmp_path: Path):
        """Single-pass convert raises → SystemExit(1)."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        ns = _ns(source, tmp_path / "out", ffn_bits=4, embed_bits=6)
        with patch("mlx_lm.convert", side_effect=Exception("quantization failed")):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_convert_model(ns)
        assert exc.value.code == 1

    def test_different_bits_single_pass_success(self, tmp_path: Path, capsys):
        """embed_bits != ffn_bits: single mlx_lm.convert call with quant_predicate."""
        cli = _import_cli()
        source = tmp_path / "model"
        source.mkdir()
        output = tmp_path / "out"
        ns = _ns(source, output, ffn_bits=4, embed_bits=6)
        with patch("mlx_lm.convert") as mock_conv:
            cli.cmd_convert_model(ns)
        assert mock_conv.call_count == 1
        _, kwargs = mock_conv.call_args
        assert callable(kwargs.get("quant_predicate")), "quant_predicate must be a callable"
        assert "saved" in capsys.readouterr().out.lower()


class TestConvertModelSubparser:
    """Verify the convert-model subparser is wired in main()."""

    def test_convert_model_help_exits_zero(self):
        """squish convert-model --help exits 0 (verifies subparser registration)."""
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "convert-model", "--help"]):
                cli.main()
        assert exc.value.code == 0

    def test_convert_model_missing_required_args_exits(self):
        """squish convert-model with no args exits nonzero (argparse required)."""
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "convert-model"]):
                cli.main()
        assert exc.value.code != 0
