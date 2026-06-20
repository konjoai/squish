"""Coverage for squish.cli cmd_export — squish INT4 npy-dir → mlx safetensors.
Guard branches use filesystem fixtures; the heavy _build_squish_4bit_dir call
is mocked. Host-agnostic.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from squish import cli


def _ns(model_dir, **kw):
    kw.setdefault("force", False)
    kw.setdefault("source_model", None)
    kw.setdefault("group_size", None)
    return argparse.Namespace(model_dir=str(model_dir), **kw)


def _npy_dir(
    tmp_path, *, name="Qwen-compressed", manifest=True, tensors=True, q4a=True, shape_key=True
):
    d = tmp_path / name
    d.mkdir()
    if manifest:
        (d / "manifest.json").write_text('{"model.layers.0": "model"}')
    if tensors:
        td = d / "tensors"
        td.mkdir()
        if shape_key:
            np.save(td / "model__shape.npy", np.array([4, 16], dtype=np.int64))
        if q4a:
            np.save(td / "model__q4a.npy", np.zeros((4, 8), dtype=np.uint8))
            np.save(td / "model__s4a.npy", np.ones((4, 2), dtype=np.float32))
    return d


def _source(tmp_path, name="Qwen-bf16", *, config=True):
    s = tmp_path / name
    s.mkdir()
    if config:
        (s / "config.json").write_text("{}")
    return s


def _mock_build(monkeypatch):
    import squish.quant.compressed_loader as clmod

    def _fake_build(*, dir_path, **kw):
        out = dir_path / "squish_4bit"
        out.mkdir(exist_ok=True)
        (out / "model.safetensors").write_bytes(b"x" * 2048)

    monkeypatch.setattr(clmod, "_build_squish_4bit_dir", _fake_build)


# ── guard branches ───────────────────────────────────────────────────────────


def test_export_model_dir_missing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.cmd_export(_ns(tmp_path / "absent"))
    assert exc.value.code == 1


def test_export_no_manifest(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    with pytest.raises(SystemExit):
        cli.cmd_export(_ns(d))


def test_export_already_exported(tmp_path, capsys):
    d = _npy_dir(tmp_path)
    (d / ".squish_4bit_ready").write_text("")
    cli.cmd_export(_ns(d, force=False))  # sentinel + not force → return
    assert "Already exported" in capsys.readouterr().out


def test_export_source_autodetect_not_found(tmp_path):
    d = _npy_dir(tmp_path)  # no sibling -bf16 source
    with pytest.raises(SystemExit):
        cli.cmd_export(_ns(d))


def test_export_source_missing_config(tmp_path):
    d = _npy_dir(tmp_path)
    src = _source(tmp_path, "explicit-src", config=False)
    with pytest.raises(SystemExit):
        cli.cmd_export(_ns(d, source_model=str(src)))


def test_export_discover_failure(tmp_path):
    d = _npy_dir(tmp_path, tensors=False)  # manifest but no tensors/ → discover raises
    _source(tmp_path)
    with pytest.raises(SystemExit):
        cli.cmd_export(_ns(d))


def test_export_no_int4_tensors(tmp_path):
    d = _npy_dir(tmp_path, q4a=False)  # shape key present but no __q4a.npy
    _source(tmp_path)
    with pytest.raises(SystemExit):
        cli.cmd_export(_ns(d))


# ── success paths ────────────────────────────────────────────────────────────


def test_export_success_autodetect_group_size(tmp_path, monkeypatch, capsys):
    d = _npy_dir(tmp_path)
    _source(tmp_path)  # sibling Qwen-bf16
    _mock_build(monkeypatch)
    cli.cmd_export(_ns(d))
    out = capsys.readouterr().out
    assert "Exported to" in out and "Group size" in out


def test_export_success_explicit_group_size_and_source(tmp_path, monkeypatch, capsys):
    d = _npy_dir(tmp_path)
    src = _source(tmp_path, "my-source")
    _mock_build(monkeypatch)
    cli.cmd_export(_ns(d, source_model=str(src), group_size=32))
    assert "Group size   : 32" in capsys.readouterr().out


def test_export_group_size_autodetect_error_defaults_16(tmp_path, monkeypatch, capsys):
    d = _npy_dir(tmp_path)
    _source(tmp_path)
    _mock_build(monkeypatch)
    # make the s4a load fail → group-size detect except → default 16
    import squish.quant.compressed_loader as clmod

    real_load = clmod._load_npy_path

    def _flaky(path, *a, **k):
        if "__s4a" in str(path):
            raise OSError("bad read")
        return real_load(path, *a, **k)

    monkeypatch.setattr(clmod, "_load_npy_path", _flaky)
    cli.cmd_export(_ns(d))
    assert "Group size   : 16" in capsys.readouterr().out
