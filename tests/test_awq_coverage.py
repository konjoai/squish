"""Behavioral coverage for the model-family detection and AWQ-grouping edge
paths of ``squish.quant.awq`` left untested by the baseline suite. Pure-Python
(config.json + numpy); the MLX collection path is already pragma'd. No MLX.
"""
from __future__ import annotations

import json

import numpy as np

from squish.quant.awq import detect_model_family, prepare_awq_application


# ── detect_model_family ──────────────────────────────────────────────────────


def test_detect_family_no_config(tmp_path):
    assert detect_model_family(tmp_path) is None  # no config.json


def test_detect_family_corrupt_config(tmp_path):
    (tmp_path / "config.json").write_text("{ not json")
    assert detect_model_family(tmp_path) is None


def test_detect_family_from_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert detect_model_family(tmp_path) == "qwen3"


def test_detect_family_from_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "", "architectures": ["LlamaForCausalLM"]}))
    assert detect_model_family(tmp_path) == "llama"


def test_detect_family_unknown(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "exotic", "architectures": ["ExoticForCausalLM"]}))
    assert detect_model_family(tmp_path) is None


# ── prepare_awq_application ──────────────────────────────────────────────────


def test_prepare_awq_skips_keys_without_dot():
    scales = {
        "noparent": np.ones(4, np.float32),  # no '.' → skipped (698)
        "model.layers.0.self_attn.q_proj": np.full(4, 2.0, np.float32),
        "model.layers.0.self_attn.k_proj": np.full(4, 2.0, np.float32),
    }
    proj_apply, ln_apply = prepare_awq_application(scales)
    # The dotless key contributed nothing; the q/k projections were grouped.
    assert "noparent" not in proj_apply
    assert any("q_proj" in k for k in proj_apply)


def test_prepare_awq_groups_mlp_and_attn():
    scales = {
        "model.layers.1.mlp.gate_proj": np.full(8, 1.5, np.float32),
        "model.layers.1.mlp.up_proj": np.full(8, 2.5, np.float32),
    }
    proj_apply, ln_apply = prepare_awq_application(scales)
    assert len(proj_apply) == 2  # both MLP projections mapped to a group scale
    assert all(isinstance(v, np.ndarray) for v in proj_apply.values())


def test_prepare_awq_empty():
    proj_apply, ln_apply = prepare_awq_application({})
    assert proj_apply == {} and ln_apply == {}
