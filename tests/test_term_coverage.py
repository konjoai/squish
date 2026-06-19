"""Behavioral coverage for ``squish._term`` — truecolor / background detection,
the gradient renderer, and the import-time palette-selection + fd-probe guards.

The import-time branches are exercised by reloading the module under controlled
env / os.isatty / sys.stdout; a fixture always restores the real module state.
Pure-Python; no MLX.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

from squish import _term as term

_ENV_KEYS = ("COLORTERM", "TERM_PROGRAM", "TERM", "NO_COLOR", "FORCE_COLOR",
             "SQUISH_DARK_BG", "COLORFGBG")


# ── has_truecolor ───────────────────────────────────────────────────────────


def test_has_truecolor_isatty_error(monkeypatch):
    monkeypatch.setattr(os, "isatty", lambda fd: (_ for _ in ()).throw(OSError("no tty")))
    assert term.has_truecolor() is False  # 42-44


def test_has_truecolor_not_a_tty(monkeypatch):
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    assert term.has_truecolor() is False


def test_has_truecolor_colorterm(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert term.has_truecolor() is True


def test_has_truecolor_force_color(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert term.has_truecolor() is True


def test_has_truecolor_no_color_disables(monkeypatch):
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert term.has_truecolor() is False


# ── background detection ────────────────────────────────────────────────────


def test_detect_background_override(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SQUISH_DARK_BG", "1")
    assert term._detect_background_info() == (True, True)
    monkeypatch.setenv("SQUISH_DARK_BG", "no")
    assert term._detect_background_info() == (False, True)


def test_detect_background_colorfgbg(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("COLORFGBG", "15;0")  # bg 0 → dark
    assert term._detect_background_info() == (True, True)
    monkeypatch.setenv("COLORFGBG", "0;15")  # bg 15 → light
    assert term._detect_background_info() == (False, True)
    monkeypatch.setenv("COLORFGBG", "garbage")  # unparseable → fallback unconfirmed
    assert term._detect_background_info() == (True, False)


def test_detect_background_default(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    assert term._detect_background_info() == (True, False)
    assert term.detect_dark_background() is True


# ── gradient ────────────────────────────────────────────────────────────────


def test_gradient_force_color_emits_escapes():
    out = term.gradient("hello", term.LOGO_GRAD, force_color=True)
    assert "\033[38;2;" in out and out.endswith("\033[0m")


def test_gradient_force_off_is_plain():
    assert term.gradient("hello", term.LOGO_GRAD, force_color=False) == "hello"


def test_gradient_empty_text():
    assert term.gradient("", term.LOGO_GRAD, force_color=True) == ""


# ── import-time palette selection + fd guards (via reload) ──────────────────


@pytest.fixture
def reload_term(monkeypatch):
    """Exec a *fresh* copy of the module under a throwaway name so the
    import-time branches run without mutating the shared ``squish._term``
    (other modules hold references to its ``C`` object — see test_wave85)."""
    import importlib.util

    def _reload(env=None, isatty=None, stdout=None):
        for k in _ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        if isatty is not None:
            monkeypatch.setattr(os, "isatty", isatty)
        if stdout is not None:
            monkeypatch.setattr(sys, "stdout", stdout)
        spec = importlib.util.spec_from_file_location("squish._term_probe", term.__file__)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return _reload


def test_palette_dark(reload_term):
    m = reload_term({"COLORTERM": "truecolor", "SQUISH_DARK_BG": "1"}, isatty=lambda fd: True)
    assert m.C.__class__.__name__ == "_Palette"  # truecolor + confirmed dark (210)


def test_palette_light(reload_term):
    m = reload_term({"COLORTERM": "truecolor", "SQUISH_DARK_BG": "0"}, isatty=lambda fd: True)
    assert m.C.__class__.__name__ == "_PaletteLight"  # truecolor + confirmed light (210)


def test_palette_ansi(reload_term):
    # TTY but not truecolor and unconfirmed background → ANSI palette (212).
    m = reload_term({"TERM": "xterm"}, isatty=lambda fd: True)
    assert m.C.__class__.__name__ == "_PaletteANSI"


def test_palette_not_tty(reload_term):
    m = reload_term({}, isatty=lambda fd: False)  # not a TTY → empty _Palette (214)
    assert m.C.__class__.__name__ == "_Palette"
    assert m.C.P == ""  # _k() yields empty strings


def test_import_fileno_error(reload_term):
    class _BadStdout:
        def fileno(self):
            raise OSError("no fileno")

    m = reload_term({}, isatty=lambda fd: False, stdout=_BadStdout())
    assert m._stdout_fd == 1  # fileno() failure → fd 1 fallback (109-111)


def test_import_isatty_error(reload_term):
    m = reload_term({}, isatty=lambda fd: (_ for _ in ()).throw(OSError("boom")))
    assert m._IS_TTY is False  # isatty() failure → not a TTY (115-117)
