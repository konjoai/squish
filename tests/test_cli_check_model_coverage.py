"""Coverage for squish.cli cmd_check_model + _check_layer_config — quantized
model inspector with HQQ quality simulation on synthetic weights. Pure
numpy/filesystem; host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys

import pytest

from squish import cli


def _ns(model):
    return argparse.Namespace(model=str(model))


def _model(tmp_path, cfg, *, name="m", write_config=True):
    d = tmp_path / name
    d.mkdir()
    if write_config:
        (d / "config.json").write_text(json.dumps(cfg))
    return d


_BASE = {
    "model_type": "qwen",
    "hidden_size": 1024,
    "num_hidden_layers": 4,
    "vocab_size": 1000,
    "intermediate_size": 2048,
}


# ── guards ───────────────────────────────────────────────────────────────────


def test_check_path_missing(tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_check_model(_ns(tmp_path / "absent"))


def test_check_no_config(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    with pytest.raises(SystemExit):
        cli.cmd_check_model(_ns(d))


# ── config analysis ──────────────────────────────────────────────────────────


def test_check_bf16_no_quant(tmp_path, capsys):
    d = _model(tmp_path, dict(_BASE))  # no quantization config
    cli.cmd_check_model(_ns(d))
    out = capsys.readouterr().out
    assert "BF16" in out and "~" in out  # params estimated


def test_check_uniform_quant(tmp_path, capsys):
    cfg = dict(_BASE, quantization={"bits": 4, "group_size": 64})
    d = _model(tmp_path, cfg)
    cli.cmd_check_model(_ns(d))
    assert "uniform" in capsys.readouterr().out


def test_check_per_path_quant_with_warnings(tmp_path, capsys):
    cfg = dict(_BASE)
    cfg["hidden_size"] = 1024
    cfg["num_hidden_layers"] = 1
    cfg["intermediate_size"] = 1024
    cfg["vocab_size"] = 100  # keep param count < 1B for INT2 small-model warning
    cfg["quantization"] = {
        "model.layers.0.self_attn.q_proj": {"bits": 2, "group_size": 64},  # attention INT2
        "model.layers.0.mlp.gate_proj": {"group_size": 64},  # ffn, no bits → b None arc
        "lm_head": {"unused": 1},  # embed label, but no bits/gs → embed hits print continue
        "model.norm.weight": {"bits": 4},  # → "other" label, no group_size → gs None arc
        "scalar_path": 7,  # non-dict → skipped
    }
    d = _model(tmp_path, cfg)
    cli.cmd_check_model(_ns(d))
    out = capsys.readouterr().out
    assert "per layer type" in out
    assert "INT2" in out and "attention" in out  # _check_layer_config warnings


def test_check_hqq_3bit_caution(tmp_path, capsys):
    pytest.importorskip("squish.experimental.hqq_quant")
    cfg = dict(_BASE, quantization={"bits": 3, "group_size": 64})  # 3-bit → caution arc
    d = _model(tmp_path, cfg)
    cli.cmd_check_model(_ns(d))
    assert "SNR=" in capsys.readouterr().out


def test_check_zero_dims_no_params(tmp_path, capsys):
    cfg = {"model_type": "x", "hidden_size": 0, "num_hidden_layers": 0}
    d = _model(tmp_path, cfg)
    cli.cmd_check_model(_ns(d))
    assert "Params" not in capsys.readouterr().out  # param estimate skipped


# ── HQQ simulation ───────────────────────────────────────────────────────────


def test_check_hqq_import_failure(tmp_path, monkeypatch, capsys):
    cfg = dict(_BASE, quantization={"bits": 4, "group_size": 64})
    d = _model(tmp_path, cfg)
    monkeypatch.setitem(sys.modules, "squish.experimental.hqq_quant", None)
    cli.cmd_check_model(_ns(d))
    assert "HQQ simulation unavailable" in capsys.readouterr().out


def test_check_hqq_low_bit_warning(tmp_path, capsys):
    pytest.importorskip("squish.experimental.hqq_quant")
    cfg = dict(_BASE, quantization={"bits": 2, "group_size": 64})  # 2-bit → low SNR
    d = _model(tmp_path, cfg)
    cli.cmd_check_model(_ns(d))
    assert "SNR=" in capsys.readouterr().out


# ── _check_layer_config direct ───────────────────────────────────────────────


def test_check_layer_config_bits_none_noop(capsys):
    cli._check_layer_config("x", None, 64, 0, 1024)
    assert capsys.readouterr().out == ""


def test_check_layer_config_int2_large_group_default(capsys):
    cli._check_layer_config("ffn", 2, None, 0, 1024)  # group_size None → "default"
    assert "default" in capsys.readouterr().out
