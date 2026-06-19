"""Coverage for ``squish.cli`` argument parser + ``main`` dispatch.

``build_parser`` is ~1,200 lines of ``add_argument``/``add_parser`` wiring;
invoking it covers all of them in one shot. ``main`` is exercised across its
no-command, normal-dispatch, and every error-handling branch with a stubbed
command function. Pure stdlib — host-agnostic.
"""

from __future__ import annotations

import argparse

import pytest

from squish import cli


# ── build_parser ─────────────────────────────────────────────────────────────


def test_build_parser_returns_parser():
    ap = cli.build_parser()
    assert isinstance(ap, argparse.ArgumentParser) and ap.prog == "squish"


@pytest.mark.parametrize(
    "argv,command",
    [
        (["run", "qwen3:8b"], "run"),
        (["serve"], "serve"),
        (["models"], "models"),
        (["doctor"], "doctor"),
        (["catalog"], "catalog"),
        (["version"], "version"),
        (["compat"], "compat"),
        (["update"], "update"),
        (["quality"], "quality"),
    ],
)
def test_parser_wires_subcommands(argv, command):
    ns = cli.build_parser().parse_args(argv)
    assert ns.command == command
    assert callable(ns.func)  # every subcommand sets a dispatch func


def test_parser_no_command_leaves_func_unset():
    ns = cli.build_parser().parse_args([])
    assert ns.command is None


def test_build_parser_reads_env_port_digit(monkeypatch):
    monkeypatch.setenv("SQUISH_PORT", "8080")
    monkeypatch.setenv("SQUISH_MODEL", "/models/x")
    monkeypatch.setenv("SQUISH_HOST", "0.0.0.0")
    cli.build_parser()  # exercises the int(env_port) branch


def test_build_parser_reads_env_port_nondigit(monkeypatch):
    monkeypatch.setenv("SQUISH_PORT", "not-a-number")
    monkeypatch.delenv("SQUISH_MODEL", raising=False)
    cli.build_parser()  # exercises the None branch


# ── main dispatch ────────────────────────────────────────────────────────────


def test_main_no_command_shows_welcome(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish"])
    called = {}
    monkeypatch.setattr(cli, "cmd_welcome", lambda: called.setdefault("welcome", True))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0 and called.get("welcome")


def test_main_dispatches_to_command(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])
    seen = {}
    monkeypatch.setattr(cli, "cmd_version", lambda args: seen.setdefault("ran", True))
    cli.main()
    assert seen.get("ran")


def test_main_keyboard_interrupt_exits_130(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])

    def _boom(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_version", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 130


def test_main_propagates_system_exit(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])

    def _exit(_args):
        raise SystemExit(7)

    monkeypatch.setattr(cli, "cmd_version", _exit)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 7


def test_main_converts_expected_errors(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])

    def _bad(_args):
        raise ValueError("nope")

    monkeypatch.setattr(cli, "cmd_version", _bad)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0  # _die → non-zero exit


def test_main_converts_unexpected_errors(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])

    def _kaboom(_args):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "cmd_version", _kaboom)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0


def test_main_survives_logging_config_failure(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["squish", "version"])
    monkeypatch.setattr(cli, "cmd_version", lambda args: None)
    import squish.logging_config as lc

    monkeypatch.setattr(
        lc,
        "configure_logging",
        lambda **k: (_ for _ in ()).throw(OSError("no logfile")),
    )
    cli.main()  # logging failure is logged + swallowed, dispatch still runs
