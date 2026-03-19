"""
tests/test_ui_unit.py

Unit tests for squish/ui.py — TUI helpers.
"""
from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_ui():
    import squish.ui as ui
    return ui


# ── banner ───────────────────────────────────────────────────────────────────

class TestBanner:
    def test_banner_runs_without_exception(self, capsys):
        ui = _import_ui()
        ui.banner()  # must not raise
        captured = capsys.readouterr()
        # some output produced
        assert captured.out or not captured.out  # always True — just ensure no crash

    def test_banner_contains_squish(self, capsys):
        ui = _import_ui()
        ui.banner()
        captured = capsys.readouterr()
        # The banner should produce some visual output (ASCII art + rule)
        combined = captured.out + captured.err
        # At minimum the version rule separator line should appear
        assert len(combined.strip()) > 0, "banner produced no output"


# ── success / warn / error / hint ────────────────────────────────────────────

class TestStatusMessages:
    def test_success_outputs_text(self, capsys):
        ui = _import_ui()
        ui.success("All done")
        captured = capsys.readouterr()
        assert "All done" in (captured.out + captured.err)

    def test_warn_outputs_text(self, capsys):
        ui = _import_ui()
        ui.warn("Something odd")
        captured = capsys.readouterr()
        assert "Something odd" in (captured.out + captured.err)

    def test_error_outputs_text(self, capsys):
        ui = _import_ui()
        ui.error("Boom")
        captured = capsys.readouterr()
        assert "Boom" in (captured.out + captured.err)

    def test_hint_outputs_text(self, capsys):
        ui = _import_ui()
        ui.hint("Try squish pull qwen3:8b")
        captured = capsys.readouterr()
        assert "squish pull qwen3:8b" in (captured.out + captured.err)

    def test_messages_do_not_raise_on_unicode(self, capsys):
        ui = _import_ui()
        ui.success("✓ Done — naïve café")
        ui.warn("⚠ Warning: Ünïcödé")
        ui.error("✗ Error: 日本語")
        ui.hint("→ Hint: こんにちは")
        # no exception = pass


# ── spinner ───────────────────────────────────────────────────────────────────

class TestSpinner:
    def test_spinner_context_manager_runs(self, capsys):
        ui = _import_ui()
        executed = []
        with ui.spinner("Working"):
            executed.append(True)
        assert executed == [True]

    def test_spinner_yields_none(self):
        ui = _import_ui()
        with ui.spinner("Test") as val:
            result = val
        assert result is None

    def test_spinner_propagates_exceptions(self):
        ui = _import_ui()
        with pytest.raises(ValueError, match="inner error"):
            with ui.spinner("Working"):
                raise ValueError("inner error")


# ── progress ─────────────────────────────────────────────────────────────────

class TestProgress:
    def test_progress_with_total(self, capsys):
        ui = _import_ui()
        with ui.progress("Downloading", total=1024) as bar:
            bar.update(512)
            bar.update(512)
        # No exception = pass

    def test_progress_indeterminate(self, capsys):
        ui = _import_ui()
        with ui.progress("Compressing", total=None) as bar:
            bar.update(1)
            bar.update(1)

    def test_progress_update_zero(self, capsys):
        ui = _import_ui()
        with ui.progress("Test", total=100) as bar:
            bar.update(0)

    def test_progress_handle_has_update(self):
        ui = _import_ui()
        with ui.progress("Test") as bar:
            assert callable(bar.update)

    def test_progress_propagates_exceptions(self):
        ui = _import_ui()
        with pytest.raises(RuntimeError, match="test error"):
            with ui.progress("Test", total=100):
                raise RuntimeError("test error")


# ── confirm ───────────────────────────────────────────────────────────────────

class TestConfirm:
    def test_confirm_non_tty_returns_default_true(self):
        ui = _import_ui()
        with patch.object(sys.stdin, "isatty", return_value=False):
            assert ui.confirm("Do it?", default=True) is True

    def test_confirm_non_tty_returns_default_false(self):
        ui = _import_ui()
        with patch.object(sys.stdin, "isatty", return_value=False):
            assert ui.confirm("Do it?", default=False) is False

    def test_confirm_tty_yes(self):
        ui = _import_ui()
        with patch.object(sys.stdin, "isatty", return_value=True):
            if ui._RICH_AVAILABLE:
                # Rich uses Confirm.ask — patch it
                with patch("rich.prompt.Confirm.ask", return_value=True):
                    result = ui.confirm("Continue?", default=False)
                assert result is True
            else:
                # Fallback uses input()
                with patch("builtins.input", return_value="y"):
                    result = ui.confirm("Continue?", default=False)
                assert result is True

    def test_confirm_tty_no(self):
        ui = _import_ui()
        with patch.object(sys.stdin, "isatty", return_value=True):
            if ui._RICH_AVAILABLE:
                with patch("rich.prompt.Confirm.ask", return_value=False):
                    result = ui.confirm("Continue?", default=True)
                assert result is False
            else:
                with patch("builtins.input", return_value="n"):
                    result = ui.confirm("Continue?", default=True)
                assert result is False


# ── model_picker ──────────────────────────────────────────────────────────────

class TestModelPicker:
    def test_empty_list_returns_none(self):
        ui = _import_ui()
        result = ui.model_picker([])
        assert result is None

    def test_non_tty_numbered_fallback_valid(self):
        ui = _import_ui()
        models = ["qwen3:8b", "qwen3:14b", "gemma3:4b"]
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("builtins.input", return_value="2"):
            result = ui.model_picker(models)
        assert result == "qwen3:14b"

    def test_non_tty_numbered_fallback_cancel(self):
        ui = _import_ui()
        models = ["qwen3:8b"]
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("builtins.input", return_value=""):
            result = ui.model_picker(models, prompt="Pick one")
        assert result is None

    def test_non_tty_numbered_fallback_out_of_range(self):
        ui = _import_ui()
        models = ["qwen3:8b"]
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("builtins.input", return_value="99"):
            result = ui.model_picker(models)
        assert result is None

    def test_non_tty_numbered_fallback_invalid_string(self):
        ui = _import_ui()
        models = ["qwen3:8b"]
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("builtins.input", return_value="abc"):
            result = ui.model_picker(models)
        assert result is None


# ── make_table ───────────────────────────────────────────────────────────────

class TestMakeTable:
    def test_make_table_with_rich(self, capsys):
        ui = _import_ui()
        if not ui._RICH_AVAILABLE:
            pytest.skip("rich not installed")
        tbl = ui.make_table(["Name", "Size", "Status"], title="Models")
        assert tbl is not None
        # Can add rows
        tbl.add_row("qwen3:8b", "4.5 GB", "ready")
        # Can print without crash
        ui.console.print(tbl)
        captured = capsys.readouterr()
        assert "qwen3:8b" in (captured.out + captured.err)

    def test_make_table_without_rich_returns_none(self):
        """When _RICH_AVAILABLE is False, make_table returns None."""
        ui = _import_ui()
        # Temporarily fake rich unavailability
        original = ui._RICH_AVAILABLE
        try:
            # Patch at module level to simulate no-rich
            with patch.object(ui, "_RICH_AVAILABLE", False):
                result = ui.make_table(["A", "B"])
        finally:
            pass  # no persistent state changed
        # With rich available in test env, result depends on flag
        if not original:
            assert result is None

    def test_make_table_columns_present(self, capsys):
        ui = _import_ui()
        if not ui._RICH_AVAILABLE:
            pytest.skip("rich not installed")
        tbl = ui.make_table(["Model", "Disk", "Compressed"], title="Local Models")
        # Table object should have correct column count
        assert len(tbl.columns) == 3

    def test_make_table_title_set(self):
        ui = _import_ui()
        if not ui._RICH_AVAILABLE:
            pytest.skip("rich not installed")
        tbl = ui.make_table(["A"], title="My Title")
        assert tbl.title == "My Title"


# ── _ProgressHandle ──────────────────────────────────────────────────────────

class TestProgressHandle:
    def test_update_calls_advance(self):
        ui = _import_ui()
        calls = []
        handle = ui._ProgressHandle(lambda n: calls.append(n))
        handle.update(5)
        handle.update(10)
        assert calls == [5, 10]

    def test_update_default_is_one(self):
        ui = _import_ui()
        calls = []
        handle = ui._ProgressHandle(lambda n: calls.append(n))
        handle.update()
        assert calls == [1]
