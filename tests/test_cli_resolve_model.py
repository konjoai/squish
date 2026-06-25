"""
tests/test_cli_resolve_model.py

Regression tests for `squish run` model-directory resolution.

`squish run gemma3:4b` failed because `squish pull` creates only the compressed
`<base>-int4` dir, while `_resolve_model` required the raw `<base>-bf16` dir to
exist first and died before computing the compressed dir. `_resolve_model` now
falls back to `_resolve_presquished_dir`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415

    return cli


def _entry(dir_name: str):
    e = MagicMock()
    e.dir_name = dir_name
    return e


def _make_quant_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"quantization": {"bits": 4}}')
    return d


# ── _resolve_presquished_dir (the extracted helper) ────────────────────────────


class TestResolvePresquishedDir:
    def test_returns_none_for_path_like_name(self, tmp_path):
        """A name containing '/' is treated as a path, not a catalog id."""
        cli = _import_cli()
        assert cli._resolve_presquished_dir("~/models/foo", tmp_path / "foo-bf16", "int4") is None

    def test_returns_none_when_no_compressed_dir(self, tmp_path):
        """No `<base>-<quant>` dir present → None."""
        cli = _import_cli()
        assert (
            cli._resolve_presquished_dir("gemma3:4b", tmp_path / "gemma-3-4b-it-bf16", "int4")
            is None
        )

    def test_finds_int4_dir(self, tmp_path):
        """Returns the compressed dir when it exists with a config.json."""
        cli = _import_cli()
        int4 = _make_quant_dir(tmp_path, "gemma-3-4b-it-int4")
        got = cli._resolve_presquished_dir("gemma3:4b", tmp_path / "gemma-3-4b-it-bf16", "int4")
        assert got == int4


# ── _resolve_model: pre-squished compressed-dir fallback ───────────────────────


class TestResolveModelCompressedFallback:
    def test_falls_back_to_int4_when_bf16_dir_absent(self, tmp_path):
        """Only `<base>-int4` on disk (no bf16) → resolves to the int4 dir."""
        cli = _import_cli()
        int4 = _make_quant_dir(tmp_path, "gemma-3-4b-it-int4")
        with (
            patch.object(cli, "_MODELS_DIR", tmp_path),
            patch.object(cli, "_CATALOG_AVAILABLE", True),
            patch.object(cli, "_catalog_resolve", return_value=_entry("gemma-3-4b-it-bf16")),
        ):
            model_dir, compressed_dir = cli._resolve_model("gemma3:4b")
        assert model_dir == int4
        assert compressed_dir == int4

    def test_honors_requested_quant_mode(self, tmp_path):
        """Requested quant (int3) is preferred over int4 when both exist."""
        cli = _import_cli()
        _make_quant_dir(tmp_path, "gemma-3-4b-it-int4")
        int3 = _make_quant_dir(tmp_path, "gemma-3-4b-it-int3")
        with (
            patch.object(cli, "_MODELS_DIR", tmp_path),
            patch.object(cli, "_CATALOG_AVAILABLE", True),
            patch.object(cli, "_catalog_resolve", return_value=_entry("gemma-3-4b-it-bf16")),
        ):
            model_dir, compressed_dir = cli._resolve_model("gemma3:4b", quant_mode="int3")
        assert model_dir == int3
        assert compressed_dir == int3

    def test_notes_when_falling_back_to_other_quant(self, tmp_path, capsys):
        """Requested int3 missing but int4 present → uses int4 and says so."""
        cli = _import_cli()
        int4 = _make_quant_dir(tmp_path, "gemma-3-4b-it-int4")
        with (
            patch.object(cli, "_MODELS_DIR", tmp_path),
            patch.object(cli, "_CATALOG_AVAILABLE", True),
            patch.object(cli, "_catalog_resolve", return_value=_entry("gemma-3-4b-it-bf16")),
        ):
            model_dir, _ = cli._resolve_model("gemma3:4b", quant_mode="int3")
        assert model_dir == int4
        out = capsys.readouterr().out
        assert "INT4" in out and "INT3" in out

    def test_still_dies_when_no_dir_exists(self, tmp_path):
        """No bf16 and no compressed dir → still a clean SystemExit (no regression)."""
        cli = _import_cli()
        with (
            patch.object(cli, "_MODELS_DIR", tmp_path),
            patch.object(cli, "_CATALOG_AVAILABLE", True),
            patch.object(cli, "_catalog_resolve", return_value=_entry("gemma-3-4b-it-bf16")),
        ):
            with pytest.raises(SystemExit):
                cli._resolve_model("gemma3:4b")
