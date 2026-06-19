"""Coverage for squish.cli model-browsing commands cmd_catalog and cmd_search.
Catalog entries are faked via monkeypatch; rich/plain branches driven through
squish.ui._RICH_AVAILABLE. Host-agnostic.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


def _set_rich(monkeypatch, value):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", value)


def _entry(
    id_,
    *,
    params="8B",
    prebuilt=False,
    notes="",
    tags=None,
    moe=False,
    active=None,
    size=4.0,
    squished=2.0,
):
    return types.SimpleNamespace(
        id=id_,
        params=params,
        has_prebuilt=prebuilt,
        notes=notes,
        tags=tags or [],
        moe=moe,
        active_params_b=active,
        size_gb=size,
        squished_size_gb=squished,
    )


def _fake_entries():
    return [
        _entry("qwen3:8b", prebuilt=True, notes="great", moe=True, active=3.0),
        _entry("moe:16b", prebuilt=False, notes="", tags=["moe", "fast"], moe=True, active=None),
        _entry("plain:4b", prebuilt=False, notes="", tags=[]),
    ]


# ── cmd_search ───────────────────────────────────────────────────────────────


def test_search_no_hits(monkeypatch, capsys):
    import squish.catalog as cat

    monkeypatch.setattr(cat, "search", lambda q: [])
    cli.cmd_search(_ns(query="zzz"))
    assert "No catalog entries match" in capsys.readouterr().out


def test_search_with_hits(monkeypatch, capsys):
    import squish.catalog as cat

    hits = [_entry("qwen3:8b", tags=["reasoning"]), _entry("plain:4b", tags=[])]
    monkeypatch.setattr(cat, "search", lambda q: hits)
    cli.cmd_search(_ns(query="qwen"))
    out = capsys.readouterr().out
    assert "qwen3:8b" in out and "reasoning" in out and "—" in out  # empty tags → —


# ── cmd_catalog ──────────────────────────────────────────────────────────────


def test_catalog_no_entries(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: [])
    cli.cmd_catalog(_ns(tag="reasoning", refresh=False))
    assert "No models found" in capsys.readouterr().out


def test_catalog_rich(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: _fake_entries())
    cli.cmd_catalog(_ns(tag=None, refresh=False))
    out = capsys.readouterr().out
    assert "Squish Catalog" in out and "MoE" in out


def test_catalog_rich_with_tag(monkeypatch, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: _fake_entries())
    cli.cmd_catalog(_ns(tag="moe", refresh=False))
    assert "Showing tag" in capsys.readouterr().out


def test_catalog_plain(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: _fake_entries())
    cli.cmd_catalog(_ns(tag=None, refresh=False))
    out = capsys.readouterr().out
    assert "Squish Model Catalog" in out and "MoE" in out


def test_catalog_plain_with_tag(monkeypatch, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: _fake_entries())
    cli.cmd_catalog(_ns(tag="fast", refresh=False))
    assert "Showing tag" in capsys.readouterr().out


def test_catalog_rich_import_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_catalog", lambda tag=None, refresh=False: _fake_entries())
    monkeypatch.setitem(sys.modules, "squish.ui", None)  # import → ImportError → plain
    cli.cmd_catalog(_ns(tag=None, refresh=False))
    assert "Squish Model Catalog" in capsys.readouterr().out
