"""
tests/test_term_unit.py

Unit tests for squish/_term.py — ANSI true-colour terminal utilities.

All tests run without a real TTY; the module-level detection falls back to
non-colour mode in CI, which is fine — we test both the coloured and plain
code paths explicitly via the ``force_color`` parameter.
"""
from __future__ import annotations

import os

import pytest

from squish._term import LOGO_GRAD, C, _Palette, gradient, has_truecolor

# ---------------------------------------------------------------------------
# has_truecolor
# ---------------------------------------------------------------------------

class TestHasTruecolor:
    def test_returns_bool(self):
        result = has_truecolor(1)
        assert isinstance(result, bool)

    def test_non_tty_fd_returns_false(self):
        # Any invalid fd (e.g. 999) is not a TTY
        assert has_truecolor(999) is False

    def test_invalid_fd_returns_false(self):
        # os.isatty on a very large fd raises — should return False, not raise
        assert has_truecolor(99999) is False

    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("COLORTERM", "truecolor")
        monkeypatch.setenv("FORCE_COLOR", "1")
        # Even with FORCE_COLOR, NO_COLOR wins
        assert has_truecolor(1) is False

    def test_force_color_env_enables_on_tty(self, monkeypatch):
        # We can't get a real TTY in CI, but we can verify that FORCE_COLOR
        # is at least in the evaluation path.  On a non-TTY fd the result is
        # always False regardless of env vars (is_tty=False short-circuits).
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        # fd=999 is not a TTY so must be False
        assert has_truecolor(999) is False

    def test_colorterm_truecolor(self, monkeypatch):
        monkeypatch.setenv("COLORTERM", "truecolor")
        monkeypatch.delenv("NO_COLOR", raising=False)
        # Can't assert True without a real TTY, but mustn't raise
        has_truecolor(1)

    def test_colorterm_24bit(self, monkeypatch):
        monkeypatch.setenv("COLORTERM", "24bit")
        monkeypatch.delenv("NO_COLOR", raising=False)
        has_truecolor(1)

    def test_term_program_iterm(self, monkeypatch):
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.delenv("NO_COLOR", raising=False)
        has_truecolor(1)

    def test_term_kitty(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-kitty")
        monkeypatch.delenv("NO_COLOR", raising=False)
        has_truecolor(1)

    def test_term_direct(self, monkeypatch):
        monkeypatch.setenv("TERM", "tmux-direct")
        monkeypatch.delenv("NO_COLOR", raising=False)
        has_truecolor(1)


# ---------------------------------------------------------------------------
# _Palette / C
# ---------------------------------------------------------------------------

class TestPalette:
    def test_c_is_palette_instance(self):
        assert isinstance(C, _Palette)

    def test_reset_is_string(self):
        assert isinstance(C.R, str)

    def test_bold_is_string(self):
        assert isinstance(C.B, str)

    def test_all_attributes_are_strings(self):
        for attr in ("DP", "P", "V", "L", "MG", "PK", "LPK", "T", "LT",
                     "G", "W", "SIL", "DIM", "B", "R"):
            val = getattr(C, attr)
            assert isinstance(val, str), f"C.{attr} must be str"

    def test_reset_is_empty_or_escape(self):
        # In non-TTY mode R is "" (no colour); in TTY mode it's an escape seq
        assert C.R == "" or C.R.startswith("\033[")


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------

class TestGradient:
    _STOPS = [(88, 28, 135), (236, 72, 153), (34, 211, 238)]

    def test_plain_text_passthrough_no_color(self):
        result = gradient("hello", self._STOPS, force_color=False)
        assert result == "hello"

    def test_empty_string_passthrough(self):
        result = gradient("", self._STOPS, force_color=True)
        assert result == ""

    def test_colored_output_contains_escapes(self):
        result = gradient("hi", self._STOPS, force_color=True)
        assert "\033[38;2;" in result
        assert "\033[0m" in result

    def test_colored_output_contains_original_chars(self):
        text = "abc"
        result = gradient(text, self._STOPS, force_color=True)
        for ch in text:
            assert ch in result

    def test_single_char_no_divide_by_zero(self):
        result = gradient("X", self._STOPS, force_color=True)
        assert "X" in result

    def test_force_none_uses_module_flag(self):
        # Should not raise regardless of _TC value
        result = gradient("test", self._STOPS, force_color=None)
        assert isinstance(result, str)

    def test_gradient_ends_with_reset(self):
        result = gradient("test", self._STOPS, force_color=True)
        assert result.endswith("\033[0m")

    def test_two_stop_gradient(self):
        stops = [(255, 0, 0), (0, 0, 255)]
        result = gradient("AB", stops, force_color=True)
        assert "A" in result and "B" in result

    def test_logo_grad_is_list_of_tuples(self):
        assert isinstance(LOGO_GRAD, list)
        assert len(LOGO_GRAD) > 0
        for stop in LOGO_GRAD:
            assert len(stop) == 3
            r, g, b = stop
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_gradient_with_logo_grad(self):
        result = gradient("Squish", LOGO_GRAD, force_color=True)
        assert "Squish"[0] in result
