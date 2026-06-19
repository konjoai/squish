"""Coverage for squish.cli local-info commands: cmd_compat, cmd_config,
cmd_version, cmd_welcome. All offline (config + ui only). The rich/non-rich
branches are driven by monkeypatching ``squish.ui._RICH_AVAILABLE``; the
rich path is guarded with ``importorskip('rich')`` so it stays host-agnostic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


# ── cmd_compat ───────────────────────────────────────────────────────────────


def test_cmd_compat_prints_snippets(capsys):
    cli.cmd_compat(_ns(port=12345, host="localhost"))
    out = capsys.readouterr().out
    assert "http://localhost:12345/v1" in out and "OpenAI SDK" in out


def test_cmd_compat_defaults_when_attrs_missing(capsys):
    cli.cmd_compat(_ns())  # getattr fallbacks: port 11435, host localhost
    assert "11435" in capsys.readouterr().out


# ── cmd_config ───────────────────────────────────────────────────────────────


@pytest.fixture
def fake_config(monkeypatch, tmp_path):
    import squish.config as cfg

    store = {"default_model": "qwen3:8b"}
    monkeypatch.setattr(cfg, "load", lambda: dict(store))
    monkeypatch.setattr(cfg, "config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(cfg, "get", lambda k, default=None: store.get(k, default))
    monkeypatch.setattr(cfg, "set", lambda k, v: store.__setitem__(k, v))
    return store


def test_config_show_rich(fake_config, monkeypatch, capsys):
    pytest.importorskip("rich")
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", True)
    cli.cmd_config(_ns(config_action="show"))
    assert "Squish Configuration" in capsys.readouterr().out


def test_config_show_plain(fake_config, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    cli.cmd_config(_ns(config_action="show"))
    assert "default_model" in capsys.readouterr().out


def test_config_get_present(fake_config, capsys):
    cli.cmd_config(_ns(config_action="get", config_key="default_model"))
    assert "qwen3:8b" in capsys.readouterr().out


def test_config_get_missing_key_arg(fake_config):
    with pytest.raises(SystemExit):
        cli.cmd_config(_ns(config_action="get", config_key=""))


def test_config_get_unset_key(fake_config):
    with pytest.raises(SystemExit):
        cli.cmd_config(_ns(config_action="get", config_key="nope"))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("no", False),
        ("42", 42),
        ("3.14", 3.14),
        ("plainstr", "plainstr"),
    ],
)
def test_config_set_coercion(fake_config, value, expected):
    cli.cmd_config(_ns(config_action="set", config_key="k", config_value=value))
    assert fake_config["k"] == expected


def test_config_set_missing_args(fake_config):
    with pytest.raises(SystemExit):
        cli.cmd_config(_ns(config_action="set", config_key="k", config_value=None))


def test_config_unknown_action(fake_config):
    with pytest.raises(SystemExit):
        cli.cmd_config(_ns(config_action="bogus"))


def test_config_action_defaults_to_show(fake_config, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    cli.cmd_config(_ns(config_action=None))  # None → "show"
    assert "Squish Configuration" in capsys.readouterr().out


# ── cmd_version ──────────────────────────────────────────────────────────────


def test_version_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", True)
    cli.cmd_version(_ns())
    assert "squish" in capsys.readouterr().out.lower()


def test_version_plain(monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    cli.cmd_version(_ns())
    assert "squish" in capsys.readouterr().out.lower()


def test_version_platform_import_failure_plain(monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    monkeypatch.setitem(sys.modules, "platform", None)  # import platform → ImportError
    cli.cmd_version(_ns())  # non-rich except branch logged + swallowed
    assert "squish" in capsys.readouterr().out.lower()


def test_version_platform_import_failure_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "platform", None)  # rich except branch
    cli.cmd_version(_ns())
    assert "squish" in capsys.readouterr().out.lower()


# ── cmd_welcome ──────────────────────────────────────────────────────────────


def _mem_proc(text):
    return types.SimpleNamespace(stdout=text)


@pytest.fixture
def welcome_env(monkeypatch, tmp_path):
    import squish.config as cfg

    monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "get", lambda k, default=None: None)
    return tmp_path


@pytest.mark.parametrize(
    "mem_text,_tier",
    [
        ("Memory: 64 GB", "32b"),
        ("Memory: 32 GB", "14b"),
        ("Memory: 16 GB", "8b"),
        ("Memory: 65536 MB", "mb-branch"),  # unit != GB → // 1024
    ],
)
def test_welcome_ram_tiers_plain(welcome_env, monkeypatch, capsys, mem_text, _tier):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc(mem_text))
    cli.cmd_welcome()
    assert "squish" in capsys.readouterr().out.lower()


def test_welcome_rich_with_local_models(welcome_env, monkeypatch, capsys):
    pytest.importorskip("rich")
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", True)
    models = welcome_env / "models"
    (models / "qwen3-8b").mkdir(parents=True)
    (models / ".hidden").mkdir()  # skipped (dotfile)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc("Memory: 16 GB"))
    cli.cmd_welcome()
    assert capsys.readouterr().out  # rendered something


def test_welcome_rich_no_local_models(welcome_env, monkeypatch, capsys):
    pytest.importorskip("rich")
    import squish.ui as ui
    import squish.config as cfg

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", True)
    monkeypatch.setattr(cfg, "get", lambda k, default=None: "qwen3:14b")  # saved default
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc("Memory: 16 GB"))
    cli.cmd_welcome()
    assert capsys.readouterr().out


def test_welcome_plain_with_local_models(welcome_env, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    models = welcome_env / "models"
    (models / "qwen3-8b").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc("Memory: 16 GB"))
    cli.cmd_welcome()
    assert "model(s)" in capsys.readouterr().out


def test_welcome_no_memory_line(welcome_env, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    # stdout has no "Memory:" line → parse loop completes without a match
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc("Chip: Apple M3\n"))
    cli.cmd_welcome()
    assert "squish" in capsys.readouterr().out.lower()


def test_welcome_memory_line_without_digit(welcome_env, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    # "Memory:" present but no numeric token → inner loop finds no digit
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _mem_proc("Memory: unknown"))
    cli.cmd_welcome()
    assert "squish" in capsys.readouterr().out.lower()


def test_welcome_system_profiler_failure_plain(welcome_env, monkeypatch, capsys):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)

    def _boom(*a, **k):
        raise OSError("no system_profiler")

    monkeypatch.setattr(subprocess, "run", _boom)
    cli.cmd_welcome()  # ram_gb stays 0 → "?" + qwen3:4b suggestion
    assert "squish" in capsys.readouterr().out.lower()
