"""Coverage for squish.cli cmd_sparsity_trim — physical FFN neuron pruning.
Pure numpy + safetensors + filesystem (no model load). Real safetensors
fixtures drive both the float and INT4 (packed uint32) trim paths.
Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from squish import cli


def _ns(model, **kw):
    kw.setdefault("threshold", 0.5)
    kw.setdefault("group_size", 8)
    kw.setdefault("dry_run", False)
    kw.setdefault("output", "")
    return argparse.Namespace(model=str(model), **kw)


def _float_model(
    tmp_path, *, n_layers=1, inter=16, hidden=8, layers_with_up=None, extra_down_bias=False
):
    d = tmp_path / "m"
    d.mkdir()
    rng = np.random.default_rng(0)
    w = {"model.embed_tokens.weight": rng.standard_normal((10, hidden)).astype(np.float32)}
    for L in range(n_layers):
        if layers_with_up is not None and L not in layers_with_up:
            continue
        b = f"model.layers.{L}.mlp"
        w[f"{b}.up_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
        w[f"{b}.gate_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
        w[f"{b}.down_proj.weight"] = rng.standard_normal((hidden, inter)).astype(np.float32)
        if extra_down_bias:
            w[f"{b}.down_proj.bias"] = rng.standard_normal((hidden,)).astype(np.float32)
    save_file(w, str(d / "model.safetensors"))
    (d / "config.json").write_text(
        json.dumps({"num_hidden_layers": n_layers, "intermediate_size": inter})
    )
    (d / "tokenizer.json").write_text("{}")  # extra file copied on save
    return d


def _int4_model(tmp_path, *, inter=16, hidden=8, group_size=8, with_up=True):
    d = tmp_path / "m4"
    d.mkdir()
    rng = np.random.default_rng(1)
    n_groups = inter // group_size
    w = {"model.embed_tokens.weight": rng.standard_normal((10, hidden)).astype(np.float32)}
    b = "model.layers.0.mlp"
    if with_up:
        w[f"{b}.up_proj.weight"] = rng.integers(0, 2**31, (inter, hidden // 8), dtype=np.uint32)
        w[f"{b}.up_proj.scales"] = rng.standard_normal((inter, 1)).astype(np.float16)
        w[f"{b}.up_proj.biases"] = rng.standard_normal((inter, 1)).astype(np.float16)
        w[f"{b}.gate_proj.weight"] = rng.integers(0, 2**31, (inter, hidden // 8), dtype=np.uint32)
        w[f"{b}.gate_proj.scales"] = rng.standard_normal((inter, 1)).astype(np.float16)
        w[f"{b}.gate_proj.biases"] = rng.standard_normal((inter, 1)).astype(np.float16)
    w[f"{b}.down_proj.weight"] = rng.integers(0, 2**31, (hidden, inter // 8), dtype=np.uint32)
    w[f"{b}.down_proj.scales"] = rng.standard_normal((hidden, n_groups)).astype(np.float16)
    w[f"{b}.down_proj.biases"] = rng.standard_normal((hidden, n_groups)).astype(np.float16)
    save_file(w, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "intermediate_size": inter}))
    return d


# ── validation + resolution guards ───────────────────────────────────────────


def test_trim_safetensors_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(tmp_path))


@pytest.mark.parametrize(
    "kw", [{"threshold": 0.0}, {"threshold": 1.0}, {"group_size": 7}, {"group_size": 4}]
)
def test_trim_invalid_args(tmp_path, kw):
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(tmp_path, **kw))


def test_trim_alias_not_found(tmp_path, monkeypatch):
    import squish.serving.local_model_scanner as lms

    monkeypatch.setattr(
        lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=lambda: [])
    )
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns("ghost-alias"))


def test_trim_alias_scanner_error(monkeypatch):
    import squish.serving.local_model_scanner as lms

    def _boom():
        raise OSError("scan failed")

    monkeypatch.setattr(lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=_boom))
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns("ghost-alias"))


def test_trim_alias_candidate_not_dir(tmp_path, monkeypatch):
    import squish.serving.local_model_scanner as lms

    f = tmp_path / "afile"
    f.write_text("x")  # not a dir → comp_dir = parent (tmp_path) → no model.safetensors → _die
    cand = types.SimpleNamespace(name="x", path=f)
    monkeypatch.setattr(
        lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=lambda: [cand])
    )
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns("x"))


def test_trim_no_safetensors(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(d))


def test_trim_no_config(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    save_file({"x": np.ones(2, np.float32)}, str(d / "model.safetensors"))
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(d))


def test_trim_zero_config_values(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    save_file({"x": np.ones(2, np.float32)}, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 0, "intermediate_size": 0}))
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(d))


# ── float trim ───────────────────────────────────────────────────────────────


def test_trim_float_dry_run(tmp_path, capsys):
    d = _float_model(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sparsity_trim(_ns(d, dry_run=True))
    assert exc.value.code == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_trim_float_full(tmp_path, capsys):
    d = _float_model(tmp_path, extra_down_bias=True)
    cli.cmd_sparsity_trim(_ns(d))
    out_dir = d.parent / "m-trimmed"
    assert (out_dir / "model.safetensors").exists()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["intermediate_size"] == 8  # 50% of 16 pruned
    assert (out_dir / "tokenizer.json").exists()  # supporting file copied
    assert "reduction" in capsys.readouterr().out


def test_trim_float_layer_missing_up_proj(tmp_path):
    d = _float_model(tmp_path, n_layers=2, layers_with_up={0})  # layer 1 has no up_proj
    cli.cmd_sparsity_trim(_ns(d))
    assert (d.parent / "m-trimmed" / "model.safetensors").exists()


def test_trim_explicit_output_and_exists_guard(tmp_path):
    d = _float_model(tmp_path)
    out = tmp_path / "custom-out"
    cli.cmd_sparsity_trim(_ns(d, output=str(out)))
    assert (out / "model.safetensors").exists()
    # second run → output exists → _die
    with pytest.raises(SystemExit):
        cli.cmd_sparsity_trim(_ns(d, output=str(out)))


# ── INT4 trim ────────────────────────────────────────────────────────────────


def test_trim_int4_full(tmp_path, capsys):
    d = _int4_model(tmp_path)
    cli.cmd_sparsity_trim(_ns(d))
    out_dir = d.parent / "m4-trimmed"
    assert (out_dir / "model.safetensors").exists()
    assert "INT4" in capsys.readouterr().out


def test_trim_int4_layer_missing_up_proj(tmp_path):
    d = _int4_model(tmp_path, with_up=False)  # no up_proj → but probe also gone → float path
    # with no up_proj.weight the int4 probe is False → float path, down_proj has no .weight float
    cli.cmd_sparsity_trim(_ns(d))
    assert (d.parent / "m4-trimmed" / "model.safetensors").exists()


def _save(tmp_path, name, w, *, n_layers, inter):
    d = tmp_path / name
    d.mkdir()
    save_file(w, str(d / "model.safetensors"))
    (d / "config.json").write_text(
        json.dumps({"num_hidden_layers": n_layers, "intermediate_size": inter})
    )
    return d


def test_trim_alias_resolves_dir(tmp_path, monkeypatch):
    import squish.serving.local_model_scanner as lms

    md = _float_model(tmp_path)  # tmp_path/m is a valid dir
    cand = types.SimpleNamespace(name="qwen3:8b", path=md)  # path IS a dir → skip parent
    monkeypatch.setattr(
        lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=lambda: [cand])
    )
    cli.cmd_sparsity_trim(_ns("qwen3:8b"))
    assert (md.parent / "m-trimmed" / "model.safetensors").exists()


def test_trim_float_partial_projections(tmp_path):
    # up_proj present but gate_proj + down_proj absent → the _remove_*_float
    # "key in weights" guards take their false arcs.
    rng = np.random.default_rng(0)
    w = {
        "model.embed_tokens.weight": rng.standard_normal((4, 8)).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight": rng.standard_normal((16, 8)).astype(np.float32),
    }
    d = _save(tmp_path, "fp", w, n_layers=1, inter=16)
    cli.cmd_sparsity_trim(_ns(d))
    assert (d.parent / "fp-trimmed" / "model.safetensors").exists()


def test_trim_int4_partial_keys(tmp_path, capsys):
    # int4 detected (up_proj.weight uint32) but up has no biases and down_proj is
    # absent → exercises the false arcs of _remove_rows_int4 + _remove_cols_int4.
    rng = np.random.default_rng(1)
    b = "model.layers.0.mlp"
    w = {
        "model.embed_tokens.weight": rng.standard_normal((4, 8)).astype(np.float32),
        f"{b}.up_proj.weight": rng.integers(0, 2**31, (16, 1), dtype=np.uint32),
        f"{b}.up_proj.scales": rng.standard_normal((16, 1)).astype(np.float16),
        f"{b}.gate_proj.weight": rng.integers(0, 2**31, (16, 1), dtype=np.uint32),
        f"{b}.gate_proj.scales": rng.standard_normal((16, 1)).astype(np.float16),
        f"{b}.gate_proj.biases": rng.standard_normal((16, 1)).astype(np.float16),
        # down_proj omitted entirely
    }
    d = _save(tmp_path, "i4p", w, n_layers=1, inter=16)
    cli.cmd_sparsity_trim(_ns(d))
    assert "INT4" in capsys.readouterr().out


def test_trim_int4_second_layer_missing_up(tmp_path):
    # 2-layer int4: layer 0 full (→ is_int4 True), layer 1 has no up_proj → copy-continue
    rng = np.random.default_rng(2)
    w = {"model.embed_tokens.weight": rng.standard_normal((4, 8)).astype(np.float32)}
    b0 = "model.layers.0.mlp"
    w[f"{b0}.up_proj.weight"] = rng.integers(0, 2**31, (16, 1), dtype=np.uint32)
    w[f"{b0}.up_proj.scales"] = rng.standard_normal((16, 1)).astype(np.float16)
    w[f"{b0}.up_proj.biases"] = rng.standard_normal((16, 1)).astype(np.float16)
    w[f"{b0}.gate_proj.weight"] = rng.integers(0, 2**31, (16, 1), dtype=np.uint32)
    w[f"{b0}.gate_proj.scales"] = rng.standard_normal((16, 1)).astype(np.float16)
    w[f"{b0}.gate_proj.biases"] = rng.standard_normal((16, 1)).astype(np.float16)
    w[f"{b0}.down_proj.weight"] = rng.integers(0, 2**31, (8, 2), dtype=np.uint32)
    w[f"{b0}.down_proj.scales"] = rng.standard_normal((8, 2)).astype(np.float16)
    w[f"{b0}.down_proj.biases"] = rng.standard_normal((8, 2)).astype(np.float16)
    # layer 1: only a down_proj, no up_proj → int4 missing-up copy-continue
    w["model.layers.1.mlp.down_proj.weight"] = rng.integers(0, 2**31, (8, 2), dtype=np.uint32)
    d = _save(tmp_path, "i4l", w, n_layers=2, inter=16)
    cli.cmd_sparsity_trim(_ns(d))
    assert (d.parent / "i4l-trimmed" / "model.safetensors").exists()
