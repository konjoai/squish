"""tests/test_wave95_release.py — Wave 95: Final public release audit tests.

Verifies:
- importlib.metadata.version("squish") returns a valid semver string
- squish version command runs and prints version + wave
- CHANGELOG contains Wave 85
- README has no "Coming soon" text
- All key CLI commands present in cli.py
- model count in catalog ≥ 34
- _CURRENT_WAVE = 95 in cli.py
- MODULES.md has wave 85 entries
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ── 1. Package version ─────────────────────────────────────────────────────────

class TestPackageVersion:
    def test_importlib_version_returns_string(self):
        import importlib.metadata as im
        v = im.version("squish")
        assert isinstance(v, str)
        assert len(v) > 0

    def test_version_is_semver_format(self):
        import importlib.metadata as im
        v = im.version("squish")
        # Accept X.Y.Z or X.Y format
        assert re.match(r"^\d+\.\d+(\.\d+)?", v), f"Not semver: {v!r}"

    def test_init_version_matches_metadata(self):
        import importlib.metadata as im
        from squish import __version__
        meta_version = im.version("squish")
        assert __version__ == meta_version

    def test_version_non_empty(self):
        from squish import __version__
        assert __version__
        assert __version__ != "unknown"


# ── 2. squish version command ──────────────────────────────────────────────────

class TestVersionCommand:
    def test_cmd_version_runs_without_error(self):
        import squish.cli as cli
        args = types.SimpleNamespace()
        # Should not raise
        cli.cmd_version(args)

    def test_cmd_version_prints_version(self, capsys):
        import squish.cli as cli
        args = types.SimpleNamespace()
        cli.cmd_version(args)
        captured = capsys.readouterr()
        assert "squish" in captured.out.lower()
        assert "9." in captured.out  # matches "9.1.0" etc.

    def test_cmd_version_prints_wave(self, capsys):
        import squish.cli as cli
        args = types.SimpleNamespace()
        cli.cmd_version(args)
        captured = capsys.readouterr()
        assert "Wave" in captured.out or "wave" in captured.out.lower()

    def test_current_wave_constant_defined(self):
        import squish.cli as cli
        assert hasattr(cli, "_CURRENT_WAVE"), "_CURRENT_WAVE not defined in cli.py"
        assert isinstance(cli._CURRENT_WAVE, int)
        assert cli._CURRENT_WAVE >= 85

    def test_current_wave_is_95(self):
        import squish.cli as cli
        assert cli._CURRENT_WAVE == 95

    def test_version_subcommand_registered(self):
        import squish.cli as cli
        # build_parser() should include 'version' subcommand
        parser = cli.build_parser()
        # Get all subcommand names
        subparsers_actions = [
            action for action in parser._actions
            if hasattr(action, '_name_parser_map')
        ]
        if subparsers_actions:
            subcommands = list(subparsers_actions[0]._name_parser_map.keys())
            assert "version" in subcommands, f"'version' not in subcommands: {subcommands}"


# ── 3. CHANGELOG completeness ──────────────────────────────────────────────────

class TestChangelog:
    CHANGELOG = ROOT / "CHANGELOG.md"

    def _content(self):
        return self.CHANGELOG.read_text(encoding="utf-8")

    def test_changelog_exists(self):
        assert self.CHANGELOG.exists()

    def test_changelog_has_wave_85(self):
        assert "Wave 85" in self._content(), "CHANGELOG missing Wave 85 entry"

    def test_changelog_has_wave_91(self):
        assert "Wave 91" in self._content(), "CHANGELOG missing Wave 91 entry"

    def test_changelog_has_wave_92(self):
        assert "Wave 92" in self._content(), "CHANGELOG missing Wave 92 entry"

    def test_changelog_has_wave_93(self):
        assert "Wave 93" in self._content(), "CHANGELOG missing Wave 93 entry"

    def test_changelog_has_wave_94(self):
        assert "Wave 94" in self._content(), "CHANGELOG missing Wave 94 entry"

    def test_changelog_has_wave_95(self):
        assert "Wave 95" in self._content(), "CHANGELOG missing Wave 95 entry"

    def test_changelog_entries_are_semantic_versions(self):
        content = self._content()
        # Find version lines like ## [65.0.0]
        matches = re.findall(r"##\s+\[(\d+\.\d+\.\d+)\]", content)
        assert len(matches) >= 5, f"Too few version entries: {matches}"
        for v in matches:
            assert re.match(r"\d+\.\d+\.\d+", v)


# ── 4. README accuracy ─────────────────────────────────────────────────────────

class TestReadmeAccuracy:
    README = ROOT / "README.md"

    def _content(self):
        return self.README.read_text(encoding="utf-8")

    def test_no_coming_soon(self):
        content = self._content()
        assert "coming soon" not in content.lower(), (
            "README still contains 'Coming soon' text"
        )

    def test_model_count_40(self):
        content = self._content()
        assert "40" in content, "README should mention 40 models"

    def test_no_apple_silicon_only_warning(self):
        content = self._content()
        assert "macOS + Apple Silicon (M1–M5) only" not in content

    def test_no_apple_silicon_in_title(self):
        first_line = self._content().split("\n")[0]
        assert "Apple Silicon" not in first_line

    def test_squishbar_is_not_coming_soon(self):
        content = self._content()
        # SquishBar section should not say "coming soon"
        assert "SquishBar" in content
        # Verify no "Coming soon" near SquishBar
        idx = content.find("SquishBar")
        nearby = content[max(0, idx-100):idx+200].lower()
        assert "coming soon" not in nearby

    def test_quick_start_commands_exist(self):
        content = self._content()
        assert "squish run" in content
        assert "squish pull" in content


# ── 5. Key CLI commands present ────────────────────────────────────────────────

class TestCliCommandsPresent:
    CLI_PATH = ROOT / "squish" / "cli.py"

    def _content(self):
        return self.CLI_PATH.read_text(encoding="utf-8")

    def test_cmd_run_present(self):
        assert "def cmd_run" in self._content()

    def test_cmd_pull_present(self):
        assert "def cmd_pull" in self._content()

    def test_cmd_setup_present(self):
        assert "def cmd_setup" in self._content()

    def test_cmd_catalog_present(self):
        assert "def cmd_catalog" in self._content()

    def test_cmd_trace_present(self):
        assert "def cmd_trace" in self._content()

    def test_cmd_compat_present(self):
        assert "def cmd_compat" in self._content()

    def test_cmd_import_present(self):
        assert "def cmd_import" in self._content()

    def test_cmd_version_present(self):
        assert "def cmd_version" in self._content()

    def test_cmd_models_present(self):
        assert "def cmd_models" in self._content()


# ── 6. Catalog completeness ────────────────────────────────────────────────────

class TestCatalogCompleteness:
    def test_catalog_has_at_least_40_models(self):
        from squish.catalog import list_catalog
        entries = list_catalog()
        assert len(entries) >= 40, f"Only {len(entries)} models in catalog"

    def test_key_models_present(self):
        from squish.catalog import list_catalog
        entries = {e.id for e in list_catalog()}
        for required in ("qwen3:8b", "qwen3:1.7b", "gemma3:4b", "llama3.2:3b"):
            assert required in entries, f"{required!r} not in catalog"

    def test_squish_repo_for_small_models(self):
        from squish.catalog import list_catalog
        entries = {e.id: e for e in list_catalog()}
        for model_id in ("qwen3:8b", "qwen3:1.7b", "qwen3:4b"):
            assert entries[model_id].squish_repo, f"{model_id} has no squish_repo"

    def test_llama33_70b_present(self):
        from squish.catalog import list_catalog
        entries = {e.id for e in list_catalog()}
        assert "llama3.3:70b" in entries


# ── 7. MODULES.md completeness ────────────────────────────────────────────────

class TestModulesMd:
    MODULES = ROOT / "MODULES.md"

    def test_modules_exists(self):
        assert self.MODULES.exists()

    def test_modules_has_wave_85(self):
        content = self.MODULES.read_text(encoding="utf-8")
        assert "Wave 85" in content or "85" in content

    def test_modules_has_wave_95(self):
        content = self.MODULES.read_text(encoding="utf-8")
        assert "Wave 95" in content or "95" in content

    def test_modules_has_wave_table(self):
        content = self.MODULES.read_text(encoding="utf-8")
        # Should have a table with wave entries
        assert "|" in content and "Wave" in content
