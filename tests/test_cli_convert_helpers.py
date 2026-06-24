"""Coverage for cli.py convert-model helpers: the --blazing-m3 preset and the
dry-run path of cmd_convert_model. Pure-Python (the dry-run returns before any
mlx_lm import)."""

import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from squish import cli


# ── _apply_blazing_m3_preset ───────────────────────────────────────────────────


def test_blazing_preset_disabled_is_noop():
    args = types.SimpleNamespace(blazing_m3=False)
    assert cli._apply_blazing_m3_preset(args) is args


def test_blazing_preset_applies_all_defaults():
    args = types.SimpleNamespace(
        blazing_m3=True,
        ffn_bits=4,
        attn_bits=None,
        embed_bits=6,
        group_size=64,
        hqq=False,
        _default_group_size=64,
    )
    cli._apply_blazing_m3_preset(args)
    assert (args.ffn_bits, args.attn_bits, args.embed_bits, args.group_size, args.hqq) == (
        2,
        4,
        8,
        32,
        True,
    )


def test_blazing_preset_respects_user_overrides():
    args = types.SimpleNamespace(
        blazing_m3=True,
        ffn_bits=3,
        attn_bits=8,
        embed_bits=4,
        group_size=16,
        hqq=True,
        _default_group_size=64,
    )
    cli._apply_blazing_m3_preset(args)
    # every field was user-set → preset leaves them all untouched
    assert (args.ffn_bits, args.attn_bits, args.embed_bits, args.group_size, args.hqq) == (
        3,
        8,
        4,
        16,
        True,
    )


# ── cmd_convert_model dry-run ──────────────────────────────────────────────────


def _convert_args(**kw):
    base = dict(
        source_path="",
        output_path="",
        dry_run=True,
        ffn_bits=4,
        embed_bits=6,
        attn_bits=None,
        group_size=64,
        mixed_recipe=None,
        blazing_m3=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_convert_dry_run_local_model_with_size_estimate(tmp_path, capsys):
    src = tmp_path / "model"
    src.mkdir()
    save_file({"w": np.zeros((16, 16), np.float32)}, str(src / "m.safetensors"))
    out = tmp_path / "out"
    args = _convert_args(
        source_path=str(src), output_path=str(out), attn_bits=2, mixed_recipe="balanced"
    )  # attn!=ffn + mixed_recipe branches
    cli.cmd_convert_model(args)  # dry-run → returns before any mlx import
    printed = capsys.readouterr().out
    assert "[dry-run]" in printed and "source size" in printed


def test_convert_dry_run_hf_id(tmp_path):
    args = _convert_args(source_path="Qwen/Qwen3-14B", output_path=str(tmp_path / "out"))
    cli.cmd_convert_model(args)  # HF-id path: no local size estimate, returns cleanly


def test_convert_source_not_found_dies(tmp_path):
    args = _convert_args(source_path="/no/such/abs/path", output_path=str(tmp_path / "out"))
    with pytest.raises(SystemExit):
        cli.cmd_convert_model(args)
