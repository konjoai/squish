"""tests/test_cli_main_guard_unit.py

Linux-runnable unit tests for the top-level error guard in ``squish.cli.main``.

The guard must convert *every* failure path into a clean, one-line stderr
message + non-zero exit — never a raw Python traceback.  These tests inject a
``func`` that raises and assert that contract without needing MLX or a model.
"""
from __future__ import annotations

import sys

import pytest

from squish import cli


def _run_main_with_func(monkeypatch, func, argv=("squish", "version")):
    """Drive ``cli.main`` with ``args.func`` replaced by *func*.

    We stub ``build_parser`` so parsing yields a namespace whose ``func`` is the
    injected callable and whose ``command`` is a non-inference command (so the
    Apple-Silicon early-exit guard does not fire).
    """
    import argparse

    ns = argparse.Namespace(command="version", func=func, log_level="warning")

    class _StubParser:
        def parse_args(self):
            return ns

    monkeypatch.setattr(cli, "build_parser", lambda: _StubParser())
    monkeypatch.setattr(sys, "argv", list(argv))
    return ns


def test_dunder_main_module_importable():
    """`python -m squish` entry point imports cleanly without executing main()."""
    import importlib

    mod = importlib.import_module("squish.__main__")
    assert hasattr(mod, "main")


def test_value_error_is_clean_nonzero(monkeypatch, capsys):
    def boom(_args):
        raise ValueError("bad value supplied")

    _run_main_with_func(monkeypatch, boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()

    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "bad value supplied" in err
    assert "Traceback (most recent call last)" not in err


def test_file_not_found_is_clean(monkeypatch, capsys):
    def boom(_args):
        raise FileNotFoundError("/no/such/model")

    _run_main_with_func(monkeypatch, boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()

    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "/no/such/model" in err
    assert "Traceback (most recent call last)" not in err


def test_os_error_is_clean(monkeypatch, capsys):
    def boom(_args):
        raise OSError("disk on fire")

    _run_main_with_func(monkeypatch, boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code != 0
    assert "Traceback (most recent call last)" not in capsys.readouterr().err


def test_unexpected_exception_logged_and_converted(monkeypatch, capsys):
    """A surprise exception is logged (not swallowed) and converted to a clean exit."""
    import logging

    def boom(_args):
        raise RuntimeError("totally unexpected")

    _run_main_with_func(monkeypatch, boom)

    # The CLI logger sets propagate=False, so attach a capture handler directly
    # to it to confirm the failure is recorded rather than silently swallowed.
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("squish.cli")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        with pytest.raises(SystemExit) as ei:
            cli.main()
    finally:
        logger.removeHandler(handler)

    assert ei.value.code != 0
    err = capsys.readouterr().err
    # User-facing stderr is clean — no raw traceback.
    assert "Traceback (most recent call last)" not in err
    assert "unexpected error" in err.lower()
    # But the failure was recorded, not silently swallowed.
    assert any("totally unexpected" in r.getMessage() for r in records)


def test_keyboard_interrupt_exits_130(monkeypatch, capsys):
    def boom(_args):
        raise KeyboardInterrupt

    _run_main_with_func(monkeypatch, boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()

    assert ei.value.code == 130
    assert "Traceback (most recent call last)" not in capsys.readouterr().err


def test_systemexit_passes_through(monkeypatch, capsys):
    """A SystemExit raised by the command (e.g. via _die) is not re-wrapped."""

    def boom(_args):
        raise SystemExit(7)

    _run_main_with_func(monkeypatch, boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()

    assert ei.value.code == 7


def test_backend_importerror_is_clean(monkeypatch, capsys):
    """A backend failure (e.g. MLX/torch unavailable) converts to a clean exit.

    squish runs on both Apple Silicon (MLX) and Linux (torch); rather than
    gating commands on platform up front (which would break supported Linux
    serving), an inference command that can't find its backend surfaces a clean
    message instead of a raw ImportError traceback.
    """

    def boom(_args):
        raise ImportError("No module named 'mlx'")

    _run_main_with_func(monkeypatch, boom, argv=("squish", "serve"))
    with pytest.raises(SystemExit) as ei:
        cli.main()

    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" not in err
    assert err.strip()
