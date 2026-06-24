"""Coverage for cli.cmd_convert_model non-dry-run branches (cli.py ~4705-4898):

  * --cpu device-forcing block
  * INT2 auto group-size tightening + small-model warning
  * mixed_2_6 recipe quant_predicate (and its per-layer bit assignment)
  * three-tier (FFN/attn/embed) quant_predicate
  * --hqq pre-optimisation hand-off + tmp-dir cleanup
  * conversion-failure → _die

``mlx_lm.convert`` is faked so no real quantisation runs; it captures the
``quant_predicate`` so each predicate branch can be invoked directly.
"""

import json
import sys
import types

import numpy as np
import pytest

from squish import cli


# ── shared fakes ────────────────────────────────────────────────────────────────


def _install_fake_mlx_lm(monkeypatch, *, convert=None):
    captured = {}

    def _default_convert(**kw):
        captured.update(kw)

    fake = types.ModuleType("mlx_lm")
    fake.convert = convert or _default_convert
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return captured


def _install_fake_mlx_core(monkeypatch):
    calls = {}
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.cpu = "CPU_DEVICE"
    fake_mx.set_default_device = lambda d: calls.__setitem__("device", d)
    fake_mx.float32 = np.float32
    pkg = types.ModuleType("mlx")
    pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return calls, fake_mx


def _src_with_config(tmp_path, **cfg):
    src = tmp_path / "model"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(cfg))
    return src


def _args(src, out, **over):
    base = dict(
        blazing_m3=False,
        source_path=str(src),
        output_path=str(out),
        dry_run=False,
        ffn_bits=4,
        embed_bits=4,
        attn_bits=None,
        group_size=64,
        _default_group_size=64,
        mixed_recipe=None,
        cpu=False,
        hqq=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# ── --cpu + mixed_2_6 recipe + INT2 tighten/warn ────────────────────────────────


def test_convert_cpu_mode_and_mixed_recipe(tmp_path, monkeypatch):
    src = _src_with_config(
        tmp_path,
        hidden_size=512,
        num_hidden_layers=16,
        intermediate_size=2048,
        vocab_size=32000,
    )
    captured = _install_fake_mlx_lm(monkeypatch)
    dev_calls, _ = _install_fake_mlx_core(monkeypatch)

    args = _args(
        src,
        tmp_path / "out",
        ffn_bits=2,
        embed_bits=8,
        attn_bits=4,
        mixed_recipe="mixed_2_6",
        cpu=True,
    )
    cli.cmd_convert_model(args)

    # --cpu forced the MLX default device
    assert dev_calls["device"] == "CPU_DEVICE"
    # INT2 auto-tightened group_size 64 → 32
    assert captured["q_group_size"] == 32
    pred = captured["quant_predicate"]
    # embed → embed_bits(8); attn → attn_bits(4); v_proj/down in critical layer → 6
    assert pred("model.embed_tokens.weight", None)["bits"] == 8
    assert pred("model.layers.5.self_attn.q_proj.weight", None)["bits"] == 4
    assert pred("model.layers.0.mlp.down_proj.weight", None)["bits"] == 6  # layer 0 critical
    assert pred("model.layers.5.mlp.gate_proj.weight", None)["bits"] == 2  # low/default


# ── three-tier predicate (no recipe, mismatched bits) ───────────────────────────


def test_convert_three_tier_predicate(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=8)
    captured = _install_fake_mlx_lm(monkeypatch)
    _install_fake_mlx_core(monkeypatch)

    args = _args(src, tmp_path / "out", ffn_bits=4, embed_bits=6, attn_bits=8)
    cli.cmd_convert_model(args)

    pred = captured["quant_predicate"]
    assert pred("model.embed_tokens.weight", None)["bits"] == 6
    assert pred("model.layers.0.self_attn.k_proj.weight", None)["bits"] == 8
    assert pred("model.layers.0.mlp.up_proj.weight", None)["bits"] == 4


# ── uniform path → no predicate ─────────────────────────────────────────────────


def test_convert_uniform_no_predicate(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=4)
    captured = _install_fake_mlx_lm(monkeypatch)
    _install_fake_mlx_core(monkeypatch)

    args = _args(src, tmp_path / "out", ffn_bits=4, embed_bits=4, attn_bits=4)
    cli.cmd_convert_model(args)
    assert captured["quant_predicate"] is None
    assert captured["q_bits"] == 4


# ── --hqq hand-off + tmp cleanup ────────────────────────────────────────────────


def test_convert_hqq_preoptimize_invoked(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=4)
    captured = _install_fake_mlx_lm(monkeypatch)
    _install_fake_mlx_core(monkeypatch)

    sentinel = tmp_path / "hqq_tmp"
    sentinel.mkdir()
    seen = {}

    def _fake_preopt(*, source_path, ffn_bits, group_size):
        seen["ffn_bits"] = ffn_bits
        return sentinel

    monkeypatch.setattr(cli, "_preoptimize_weights_with_hqq", _fake_preopt)

    args = _args(src, tmp_path / "out", ffn_bits=3, embed_bits=3, attn_bits=3, hqq=True)
    cli.cmd_convert_model(args)

    assert seen["ffn_bits"] == 3
    # convert was handed the HQQ temp dir as its source
    assert captured["hf_path"] == str(sentinel)
    # the temp dir is cleaned up in the finally block
    assert not sentinel.exists()


# ── conversion failure → _die ───────────────────────────────────────────────────


def test_convert_failure_calls_die(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=4)

    def _boom(**kw):
        raise RuntimeError("metal oom")

    _install_fake_mlx_lm(monkeypatch, convert=_boom)
    _install_fake_mlx_core(monkeypatch)

    args = _args(src, tmp_path / "out", ffn_bits=4, embed_bits=4, attn_bits=4)
    with pytest.raises(SystemExit):
        cli.cmd_convert_model(args)


# ── mlx_lm missing → _die ───────────────────────────────────────────────────────


def test_convert_requires_mlx_lm(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=4)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)  # force ImportError everywhere
    args = _args(src, tmp_path / "out")
    with pytest.raises(SystemExit):
        cli.cmd_convert_model(args)


# ── INT2 with no config.json → param-count unknown → NOTE branch ────────────────


def test_convert_int2_no_config_emits_note(tmp_path, monkeypatch):
    src = tmp_path / "model"
    src.mkdir()  # exists but no config.json → param count unknown
    captured = _install_fake_mlx_lm(monkeypatch)
    _install_fake_mlx_core(monkeypatch)
    args = _args(src, tmp_path / "out", ffn_bits=2, embed_bits=2, attn_bits=2)
    cli.cmd_convert_model(args)
    # uniform INT2 (all bits equal) → no predicate
    assert captured["quant_predicate"] is None
    assert captured["q_bits"] == 2


# ── --cpu where set_default_device fails → swallowed, non-fatal ──────────────────


def test_convert_cpu_device_failure_is_non_fatal(tmp_path, monkeypatch):
    src = _src_with_config(tmp_path, num_hidden_layers=4)
    captured = _install_fake_mlx_lm(monkeypatch)
    _, fake_mx = _install_fake_mlx_core(monkeypatch)

    def _raise(_d):
        raise RuntimeError("no metal")

    fake_mx.set_default_device = _raise
    args = _args(src, tmp_path / "out", ffn_bits=4, embed_bits=4, attn_bits=4, cpu=True)
    cli.cmd_convert_model(args)  # must not raise — failure is logged at debug
    assert captured["q_bits"] == 4
