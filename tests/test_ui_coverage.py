"""Behavioral coverage for the Rich-render branches and interactive paths of
``squish.ui`` left untested by the baseline suite: the banner version-import
fallback, the model_picker questionary/numbered branches, the confirm plain
fallback, header/badges/panels, and the server-status / chat / generic panels.

Rich is a declared dependency so the ``_RICH_AVAILABLE`` branches execute; the
no-rich fallbacks are already ``# pragma: no cover`` except confirm's, which is
exercised by forcing ``_RICH_AVAILABLE = False``.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from squish import ui
from squish.ui import (
    banner,
    chat_header,
    confirm,
    header,
    model_picker,
    panel,
    quant_badge,
    server_status_panel,
    startup_panel,
    status_badge,
)

requires_rich = pytest.mark.skipif(not ui._RICH_AVAILABLE, reason="rich not installed")


# ── banner version-import fallback ──────────────────────────────────────────


def test_banner_uses_unknown_version_when_import_fails(capsys, monkeypatch):
    monkeypatch.delattr(sys.modules["squish"], "__version__", raising=False)
    banner()  # `from squish import __version__` raises → _ver = "unknown" (221-223)
    out = capsys.readouterr().out
    assert "squish" in out.lower()


# ── model_picker ────────────────────────────────────────────────────────────


@requires_rich
def test_model_picker_uses_questionary_when_available(monkeypatch):
    chosen = {}

    class _Sel:
        def ask(self):
            return "model-a"

    fake_q = types.ModuleType("questionary")

    def _select(prompt, choices):
        chosen["choices"] = choices
        return _Sel()

    fake_q.select = _select
    monkeypatch.setitem(sys.modules, "questionary", fake_q)
    with patch.object(sys.stdin, "isatty", return_value=True):
        assert model_picker(["model-a", "model-b"]) == "model-a"  # 329-332
    assert chosen["choices"] == ["model-a", "model-b"]


def test_model_picker_numbered_fallback_when_no_questionary(monkeypatch):
    monkeypatch.setitem(sys.modules, "questionary", None)  # import → ImportError (333)
    with patch.object(sys.stdin, "isatty", return_value=True), \
         patch("builtins.input", return_value="1"):
        assert model_picker(["alpha", "beta"]) == "alpha"


def test_model_picker_empty_returns_none():
    assert model_picker([]) is None


# ── confirm plain fallback (forced no-rich) ─────────────────────────────────


def test_confirm_plain_fallback_yes(monkeypatch):
    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    with patch.object(sys.stdin, "isatty", return_value=True), \
         patch("builtins.input", return_value="y"):
        assert confirm("proceed?", default=False) is True  # 369-373


def test_confirm_plain_fallback_empty_uses_default(monkeypatch):
    monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
    with patch.object(sys.stdin, "isatty", return_value=True), \
         patch("builtins.input", return_value=""):
        assert confirm("proceed?", default=True) is True  # empty → default (371-372)


# ── header ──────────────────────────────────────────────────────────────────


@requires_rich
def test_header_with_and_without_subtitle(capsys):
    header("Title only")
    header("Title", subtitle="and subtitle")  # 421-431 both branches
    out = capsys.readouterr().out
    assert "Title" in out and "subtitle" in out


# ── status / quant badges (pure strings) ────────────────────────────────────


def test_status_badge():
    assert "ONLINE" in status_badge(True)
    assert "OFFLINE" in status_badge(False)


def test_quant_badge_tiers():
    assert "INT4" in quant_badge("int4")
    assert "INT8" in quant_badge("8bit")
    assert "INT3" in quant_badge("INT3") and "⚠" in quant_badge("INT3")
    assert "INT2" in quant_badge("2BIT")
    # Unknown tier falls through to the lilac passthrough.
    assert "fp16" in quant_badge("fp16")


# ── startup / server-status / chat / generic panels ─────────────────────────


@requires_rich
def test_startup_panel_renders(capsys):
    startup_panel(
        model="qwen3:8b", endpoint="http://127.0.0.1:11435/v1",
        web_ui="http://127.0.0.1:11435", mode="INT4", api_key="sk-local",
    )
    out = capsys.readouterr().out
    assert "qwen3:8b" in out and "Endpoint" in out


@requires_rich
def test_server_status_panel_offline_when_empty(capsys):
    server_status_panel([], host="127.0.0.1", port=11435)
    out = capsys.readouterr().out
    assert "OFFLINE" in out and "No server running" in out


@requires_rich
def test_server_status_panel_lists_models(capsys):
    models = [
        {  # GB-sized with full details
            "name": "qwen3:8b", "size": 5_000_000_000,
            "details": {"parameter_size": "8B", "quantization_level": "int4",
                        "context_length": 32768},
        },
        {  # MB-sized, no quant/context → the "—" fallbacks
            "name": "tiny", "size": 5_000_000, "details": {},
        },
        {  # zero size → the "—" size branch
            "name": "zero", "size": 0, "details": {"parameter_size": "1B"},
        },
    ]
    server_status_panel(models, host="127.0.0.1", port=11435)
    out = capsys.readouterr().out
    assert "qwen3:8b" in out and "tiny" in out and "zero" in out


@requires_rich
def test_chat_header_renders(capsys):
    chat_header("qwen3:8b", "127.0.0.1", 11435)
    out = capsys.readouterr().out
    assert "Squish Chat" in out and "/quit" in out


@requires_rich
def test_panel_renders_with_title_and_blank_lines(capsys):
    panel(["first line", "", "third line"], title="Info")  # blank line → dim branch
    out = capsys.readouterr().out
    assert "first line" in out and "third line" in out
