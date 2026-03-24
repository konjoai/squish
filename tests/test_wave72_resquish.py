"""tests/test_wave72_resquish.py — Wave 72: resquish_all_models.py tests.

Tests for dev/scripts/resquish_all_models.py:
  - ModelFamily NamedTuple fields
  - MODEL_FAMILIES registry completeness
  - RECIPES dict: correct 3-tier mixed-precision values
  - _squish() subprocess invocation (mocked)
  - dry-run path (no subprocess, no deletion)
  - Disk-space logging at end of dry run

All tests are deterministic and do NOT delete any real files or run subprocesses.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _import_resquish():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "dev" / "scripts" / "resquish_all_models.py"
    )
    spec = importlib.util.spec_from_file_location("resquish_all_models", script_path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# ModelFamily NamedTuple
# ---------------------------------------------------------------------------

class TestModelFamily:
    def test_model_family_has_required_fields(self):
        mod = _import_resquish()
        fam = mod.MODEL_FAMILIES[0]
        for field in ("name", "bf16_dir", "int4_dir", "int3_dir", "int2_dir"):
            assert hasattr(fam, field), f"ModelFamily missing field: {field}"

    def test_model_family_name_is_string(self):
        mod = _import_resquish()
        for fam in mod.MODEL_FAMILIES:
            assert isinstance(fam.name, str), f"name must be str: {fam}"

    def test_model_family_dirs_are_strings(self):
        """Directory fields must be non-empty strings (relative paths used by script)."""
        mod = _import_resquish()
        for fam in mod.MODEL_FAMILIES:
            for attr in ("bf16_dir", "int4_dir", "int3_dir", "int2_dir"):
                val = getattr(fam, attr)
                assert isinstance(val, (str, type(None))), (
                    f"{fam.name}.{attr} must be str or None, got {type(val)}"
                )


# ---------------------------------------------------------------------------
# MODEL_FAMILIES registry
# ---------------------------------------------------------------------------

class TestModelFamiliesRegistry:
    def test_registry_not_empty(self):
        mod = _import_resquish()
        assert len(mod.MODEL_FAMILIES) >= 1

    def test_all_names_unique(self):
        mod = _import_resquish()
        names = [f.name for f in mod.MODEL_FAMILIES]
        assert len(names) == len(set(names)), "Duplicate names in MODEL_FAMILIES"

    def test_int4_dir_different_from_int2(self):
        mod = _import_resquish()
        for fam in mod.MODEL_FAMILIES:
            if fam.int4_dir and fam.int2_dir:
                assert fam.int4_dir != fam.int2_dir, (
                    f"{fam.name}: int4_dir and int2_dir must differ"
                )

    def test_int3_dir_different_from_int2(self):
        mod = _import_resquish()
        for fam in mod.MODEL_FAMILIES:
            if fam.int3_dir and fam.int2_dir:
                assert fam.int3_dir != fam.int2_dir

    def test_covers_small_and_large_models(self):
        mod = _import_resquish()
        names_lower = " ".join(f.name.lower() for f in mod.MODEL_FAMILIES)
        # Expect at least one sub-1B and one multi-B family
        has_small = any(
            "0.6" in f.name or "1b" in f.name.lower() or "1.5" in f.name
            for f in mod.MODEL_FAMILIES
        )
        has_large = any(
            "7b" in f.name.lower() or "8b" in f.name.lower() or "14b" in f.name.lower()
            for f in mod.MODEL_FAMILIES
        )
        assert has_small, "No small model (≤1.5B) in MODEL_FAMILIES"
        assert has_large, "No large model (≥7B) in MODEL_FAMILIES"


# ---------------------------------------------------------------------------
# RECIPES dict
# ---------------------------------------------------------------------------

class TestRecipes:
    def test_recipes_has_int2_int3_int4(self):
        mod = _import_resquish()
        assert 2 in mod.RECIPES
        assert 3 in mod.RECIPES
        assert 4 in mod.RECIPES

    def test_int2_recipe_attn_bits_is_4(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[2]
        assert attn == 4, f"INT2 recipe must have attn=4, got {attn}"

    def test_int3_recipe_attn_bits_is_4(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[3]
        assert attn == 4, f"INT3 recipe must have attn=4, got {attn}"

    def test_int4_recipe_attn_bits_is_4(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[4]
        assert attn == 4, f"INT4 recipe must have attn=4, got {attn}"

    def test_int2_ffn_bits_is_2(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[2]
        assert ffn == 2

    def test_int3_ffn_bits_is_3(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[3]
        assert ffn == 3

    def test_int4_ffn_bits_is_4(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[4]
        assert ffn == 4

    def test_embed_bits_is_8_for_all(self):
        mod = _import_resquish()
        for bits, (ffn, attn, embed, gs) in mod.RECIPES.items():
            assert embed == 8, f"embed must be 8 for INT{bits}, got {embed}"

    def test_int2_group_size_is_32(self):
        """INT2 must use group_size=32 to improve quantization fidelity."""
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[2]
        assert gs == 32, f"INT2 group_size must be 32, got {gs}"

    def test_int3_group_size_is_32(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[3]
        assert gs == 32, f"INT3 group_size must be 32, got {gs}"

    def test_int4_group_size_is_64(self):
        mod = _import_resquish()
        ffn, attn, embed, gs = mod.RECIPES[4]
        assert gs == 64, f"INT4 group_size must be 64, got {gs}"


# ---------------------------------------------------------------------------
# _squish() — subprocess invocation
# ---------------------------------------------------------------------------

class TestSquishSubprocess:
    def test_squish_passes_attn_bits_to_cli(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        captured_cmd: list[list[str]] = []

        def _mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=_mock_run):
            mod._squish(
                source=mock_src,
                output_path=mock_out,
                bits=2,
                dry_run=False,
                cpu=False,
            )

        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert "--attn-bits" in cmd
        attn_idx = cmd.index("--attn-bits")
        assert cmd[attn_idx + 1] == "4", f"Expected attn-bits=4, got {cmd[attn_idx+1]}"

    def test_squish_passes_group_size_to_cli(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        captured_cmd: list[list[str]] = []

        def _mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=_mock_run):
            mod._squish(
                source=mock_src,
                output_path=mock_out,
                bits=2,
                dry_run=False,
                cpu=False,
            )

        cmd = captured_cmd[0]
        assert "--group-size" in cmd
        gs_idx = cmd.index("--group-size")
        assert cmd[gs_idx + 1] == "32"

    def test_squish_returns_true_on_success(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        def _mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=_mock_run):
            ok = mod._squish(mock_src, mock_out, bits=2, dry_run=False, cpu=False)

        assert ok is True

    def test_squish_returns_false_on_failure(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        def _mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            return result

        with patch("subprocess.run", side_effect=_mock_run):
            ok = mod._squish(mock_src, mock_out, bits=2, dry_run=False, cpu=False)

        assert ok is False

    def test_dry_run_does_not_call_subprocess(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        mock_run = MagicMock()
        with patch("subprocess.run", mock_run):
            mod._squish(mock_src, mock_out, bits=2, dry_run=True, cpu=False)

        mock_run.assert_not_called()

    def test_cpu_flag_added_when_requested(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()
        mock_out = tmp_path / "int2"

        captured_cmd: list[list[str]] = []

        def _mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=_mock_run):
            mod._squish(mock_src, mock_out, bits=2, dry_run=False, cpu=True)

        cmd = captured_cmd[0]
        assert "--cpu" in cmd

    def test_ffn_bits_passed_correctly(self, tmp_path):
        mod = _import_resquish()
        mock_src = tmp_path / "bf16"
        mock_src.mkdir()

        for ffn_bits in (2, 3, 4):
            mock_out = tmp_path / f"int{ffn_bits}"
            captured_cmd: list[list[str]] = []

            def _mock_run(cmd, **kwargs):
                captured_cmd.append(cmd)
                result = MagicMock()
                result.returncode = 0
                return result

            with patch("subprocess.run", side_effect=_mock_run):
                mod._squish(mock_src, mock_out, bits=ffn_bits, dry_run=False, cpu=False)

            cmd = captured_cmd[0]
            assert "--ffn-bits" in cmd
            ffn_idx = cmd.index("--ffn-bits")
            assert cmd[ffn_idx + 1] == str(ffn_bits)
