"""tests/e2e/test_cli_clean_errors.py

Drive the real ``squish`` CLI as a subprocess for *every* subcommand and assert
none of them ever surface a raw Python traceback.

The command list is discovered dynamically by walking ``build_parser()`` — it is
never hard-coded, so a newly-added subcommand is covered automatically.

This module is Linux-runnable: the subprocess invocations exercise argparse and
the top-level error guard in ``squish.cli.main`` without needing MLX or a model
(``--help`` and unknown-flag paths never load a backend).  The inference-command
MLX-message assertion is meaningful only where MLX is *absent*, so it is skipped
on Apple Silicon.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from squish import cli

pytestmark = pytest.mark.e2e

_TRACEBACK_MARKER = "Traceback (most recent call last)"


def _discover_commands() -> list[str]:
    """Return every subcommand name registered on ``build_parser()``."""
    parser = cli.build_parser()
    names: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            names.extend(action.choices.keys())
    assert names, "no subcommands discovered — parser walk failed"
    return sorted(set(names))


_COMMANDS = _discover_commands()


def _run_cli(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 — fixed interpreter, no shell
        [sys.executable, "-m", "squish", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_command_list_is_substantial():
    # Sanity-check the dynamic walk found the full surface (≈37 commands).
    assert len(_COMMANDS) >= 20, _COMMANDS


@pytest.mark.parametrize("command", _COMMANDS)
def test_help_exits_clean(command):
    proc = _run_cli([command, "--help"])
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"`squish {command} --help` exited {proc.returncode}\n{combined[:600]}"
    )
    assert _TRACEBACK_MARKER not in combined, (
        f"`squish {command} --help` leaked a traceback:\n{combined[:800]}"
    )


@pytest.mark.parametrize("command", _COMMANDS)
def test_unknown_flag_is_clean_nonzero(command):
    proc = _run_cli([command, "--squish-no-such-flag-zzz"])
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        f"`squish {command} --bad-flag` unexpectedly succeeded\n{combined[:600]}"
    )
    assert _TRACEBACK_MARKER not in combined, (
        f"`squish {command} --bad-flag` leaked a traceback:\n{combined[:800]}"
    )
    assert combined.strip(), f"`squish {command} --bad-flag` produced no diagnostic output"


def test_top_level_help_clean():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0
    assert _TRACEBACK_MARKER not in (proc.stdout + proc.stderr)


def test_unknown_command_clean():
    proc = _run_cli(["definitely-not-a-command"])
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert _TRACEBACK_MARKER not in combined
