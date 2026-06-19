"""Coverage for squish.cli cmd_models — the local + external model listing.
Filesystem fixtures + monkeypatched catalog / scanners drive every branch
(compressed status, MoE badge, SBOM sidecar parsing, external models, rich +
plain). Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


def _set_rich(monkeypatch, value):
    import squish.ui as ui

    monkeypatch.setattr(ui, "_RICH_AVAILABLE", value)


def _no_external(monkeypatch):
    import squish.serving.local_model_scanner as lms
    import squish.experimental.lm_studio_bridge as bridge

    class _Scn:
        def scan_ollama(self):
            return []

        def scan_lm_studio(self):
            return []

    monkeypatch.setattr(lms, "LocalModelScanner", _Scn)
    monkeypatch.setattr(
        bridge,
        "probe_lm_studio",
        lambda *a, **k: types.SimpleNamespace(running=False, loaded_models=[]),
    )


def _build_models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    # ready model w/ arc_easy SBOM
    ready = models / "qwen3-8b"
    ready.mkdir()
    (models / "qwen3-8b-compressed").mkdir()  # compressed sidecar → "✓ ready"
    (ready / "weights.bin").write_bytes(b"x" * 1024)
    (ready / "cyclonedx-mlbom.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "modelCard": {
                            "quantitativeAnalysis": {
                                "performanceMetrics": [{"slice": "arc_easy", "value": 71}]
                            }
                        }
                    }
                ]
            }
        )
    )
    # MoE model, no SBOM
    (models / "moe-model").mkdir()
    # MoE model without active params
    (models / "moe-noactive").mkdir()
    # raw model with malformed SBOM → parse-error → "✓ sidecar"
    raw = models / "rawmodel"
    raw.mkdir()
    (raw / "cyclonedx-mlbom.json").write_text("{not valid json")
    # model with valid SBOM but no arc metric → "✓ sidecar"
    noarc = models / "noarc"
    noarc.mkdir()
    (noarc / "cyclonedx-mlbom.json").write_text(json.dumps({"components": [{}]}))
    (models / ".hidden").mkdir()  # skipped
    (models / "loosefile.txt").write_text("x")  # non-dir, skipped
    monkeypatch.setattr(cli, "_MODELS_DIR", models)
    return models


def _fake_catalog(monkeypatch):
    import squish.catalog as cat

    entries = [
        types.SimpleNamespace(dir_name="moe-model", moe=True, params="30B", active_params_b=3.0),
        types.SimpleNamespace(
            dir_name="moe-noactive", moe=True, params="16B", active_params_b=None
        ),
    ]
    monkeypatch.setattr(cat, "list_catalog", lambda: entries)


# ── no models directory ──────────────────────────────────────────────────────


def test_models_dir_missing(monkeypatch, tmp_path, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path / "absent")
    _no_external(monkeypatch)
    cli.cmd_models(_ns())
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Download a model with" in out


# ── local models present ─────────────────────────────────────────────────────


def test_models_listing_rich(monkeypatch, tmp_path, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    _build_models_dir(tmp_path, monkeypatch)
    _fake_catalog(monkeypatch)
    _no_external(monkeypatch)
    cli.cmd_models(_ns())
    out = capsys.readouterr().out
    assert "Local Models" in out and "qwen3-8b" in out


def test_models_listing_plain(monkeypatch, tmp_path, capsys):
    _set_rich(monkeypatch, False)
    _build_models_dir(tmp_path, monkeypatch)
    _fake_catalog(monkeypatch)
    _no_external(monkeypatch)
    cli.cmd_models(_ns())
    out = capsys.readouterr().out
    assert "Local models in" in out and "MoE" in out


def test_models_disk_size_error_renders_question_mark(monkeypatch, tmp_path, capsys):
    import pathlib

    _set_rich(monkeypatch, False)
    models = tmp_path / "models"
    models.mkdir()
    (models / "m").mkdir()
    monkeypatch.setattr(cli, "_MODELS_DIR", models)
    _no_external(monkeypatch)

    def _raise_rglob(self, pattern):
        raise OSError("disk error")

    monkeypatch.setattr(pathlib.Path, "rglob", _raise_rglob)  # disk-size sum → OSError → "?"
    cli.cmd_models(_ns())
    assert "?" in capsys.readouterr().out


def test_models_catalog_import_failure(monkeypatch, tmp_path, capsys):
    # list_catalog import raises → annotated lookup skipped (badge "")
    _set_rich(monkeypatch, False)
    _build_models_dir(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "squish.catalog", None)
    _no_external(monkeypatch)
    cli.cmd_models(_ns())
    assert "qwen3-8b" in capsys.readouterr().out


# ── external models ──────────────────────────────────────────────────────────


def _external(monkeypatch, models, *, running=True, loaded=None):
    import squish.serving.local_model_scanner as lms
    import squish.experimental.lm_studio_bridge as bridge

    class _Scn:
        def scan_ollama(self):
            return models

        def scan_lm_studio(self):
            return []

    monkeypatch.setattr(lms, "LocalModelScanner", _Scn)
    monkeypatch.setattr(
        bridge,
        "probe_lm_studio",
        lambda *a, **k: types.SimpleNamespace(running=running, loaded_models=loaded or []),
    )


def _ext_model(source, name, size):
    return types.SimpleNamespace(source=source, name=name, size_bytes=size)


def test_external_models_rich(monkeypatch, tmp_path, capsys):
    pytest.importorskip("rich")
    _set_rich(monkeypatch, True)
    monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path / "absent")  # no local
    ext = [
        _ext_model("ollama", "llama3", int(2e9)),  # GB
        _ext_model("ollama", "tiny", int(5e6)),  # MB
        _ext_model("lmstudio", "zero", 0),  # —
    ]
    _external(monkeypatch, ext, running=True, loaded=["llama3"])
    cli.cmd_models(_ns())
    assert "External Models" in capsys.readouterr().out


def test_external_models_plain(monkeypatch, tmp_path, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path / "absent")
    ext = [_ext_model("ollama", "llama3", int(2e9)), _ext_model("lmstudio", "z", 0)]
    _external(monkeypatch, ext, running=True, loaded=["other"])
    cli.cmd_models(_ns())
    assert "External models detected" in capsys.readouterr().out


def test_external_scan_failure_is_swallowed(monkeypatch, tmp_path, capsys):
    _set_rich(monkeypatch, False)
    monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path / "absent")
    import squish.serving.local_model_scanner as lms

    def _boom():
        raise OSError("scan failed")

    monkeypatch.setattr(
        lms,
        "LocalModelScanner",
        lambda: types.SimpleNamespace(scan_ollama=_boom, scan_lm_studio=lambda: []),
    )
    cli.cmd_models(_ns())  # external scan exception logged + swallowed
    assert "Download a model with" in capsys.readouterr().out
