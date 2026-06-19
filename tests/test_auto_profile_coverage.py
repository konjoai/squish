"""Behavioral coverage for the detection edge paths of
``squish.runtime.auto_profile`` left untested by the baseline suite: the
chip-attribute fallbacks in _detect_hardware, the corrupt/empty-config branches
in _detect_model_config, _model_slug_score, and the ``~/.squish/eagle-heads``
pass-2 search in _detect_eagle3.

All pure-Python — no MLX, no hardware required.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from squish.runtime import auto_profile as ap
from squish.runtime.auto_profile import ModelCapabilityDetector as Det
from squish.runtime.auto_profile import OptimizationProfile


# ── _detect_hardware fallbacks ──────────────────────────────────────────────


def test_detect_hardware_none_chip_is_noop():
    p = OptimizationProfile()
    Det._detect_hardware(p, None, 16.0)
    assert p.metal_cache_mb == 256  # unchanged default


def test_detect_hardware_missing_generation_returns():
    p = OptimizationProfile()
    # chip_profile lacks `.generation` → AttributeError → early return (218-219).
    Det._detect_hardware(p, types.SimpleNamespace(), 16.0)
    assert p.metal_cache_mb == 256


def test_detect_hardware_gen3_sets_metal_cache_scaled_to_ram():
    p = OptimizationProfile()
    # generation 3 but missing every optional attr → the AttributeError
    # fallbacks (224, 239, 250-251) all run; metal cache still scales (230).
    chip = types.SimpleNamespace(generation=3)
    Det._detect_hardware(p, chip, 24.0)
    assert p.metal_cache_mb == 96  # min(128, int(24*4)) == 96


def test_detect_hardware_older_gen_skips_metal_cache():
    p = OptimizationProfile()
    chip = types.SimpleNamespace(generation=2)  # int(gen) < 3 → skip (228→233)
    Det._detect_hardware(p, chip, 16.0)
    assert p.metal_cache_mb == 256  # untouched


def test_detect_hardware_full_chip_applies_all():
    p = OptimizationProfile()
    chip = types.SimpleNamespace(
        generation=3,
        recommended_chunk_prefill_ttft=128,
        recommended_kv_bits=4,
        recommended_model_bits=3,
    )
    Det._detect_hardware(p, chip, 16.0)
    assert p.chunk_prefill_size == 128
    assert p.kv_mode == "int4"
    assert p.kernel_path == "fused_int3"  # bits==3 → fused_int3


def test_detect_hardware_kv_fp16_and_int2_kernel():
    p = OptimizationProfile()
    chip = types.SimpleNamespace(
        generation=3, recommended_kv_bits=16, recommended_model_bits=2,
    )
    Det._detect_hardware(p, chip, 16.0)
    assert p.kv_mode == "fp16"
    assert p.kernel_path == "lut_int2"  # bits<=2


def test_detect_hardware_int4_kernel_for_high_bits():
    p = OptimizationProfile()
    chip = types.SimpleNamespace(generation=3, recommended_model_bits=4)
    Det._detect_hardware(p, chip, 16.0)
    assert p.kernel_path == "fused_int4"


# ── _detect_model_config ────────────────────────────────────────────────────


def test_detect_model_config_no_file_is_noop(tmp_path):
    p = OptimizationProfile()
    Det._detect_model_config(p, tmp_path)  # no config.json
    assert p.use_moe_lazy is False


def test_detect_model_config_corrupt_json_returns(tmp_path):
    (tmp_path / "config.json").write_text("{ not json")
    p = OptimizationProfile()
    Det._detect_model_config(p, tmp_path)  # JSONDecodeError → return (266-267)
    assert p.use_moe_lazy is False


def test_detect_model_config_moe_from_num_experts_without_arch(tmp_path):
    # No architectures key (271→275 skips arch) but num_experts > 0 → MoE.
    (tmp_path / "config.json").write_text(json.dumps({"num_experts": 8}))
    p = OptimizationProfile()
    Det._detect_model_config(p, tmp_path)
    assert p.use_moe_lazy is True


def test_detect_model_config_moe_from_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["MixtralForCausalLM"]}))
    p = OptimizationProfile()
    Det._detect_model_config(p, tmp_path)
    assert p.use_moe_lazy is True


def test_detect_model_config_dense_not_moe(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    p = OptimizationProfile()
    Det._detect_model_config(p, tmp_path)
    assert p.use_moe_lazy is False


# ── _model_slug_score ───────────────────────────────────────────────────────


def test_model_slug_score_overlap_ignoring_noise():
    # "eagle3"/"instruct" are noise; "qwen3" and "8b" overlap.
    score = Det._model_slug_score(Path("/models/qwen3-8b-instruct"), "eagle3-qwen3-8b")
    assert score == 2


def test_model_slug_score_no_overlap():
    assert Det._model_slug_score(Path("/models/llama"), "eagle3-qwen") == 0


# ── _detect_eagle3 pass 2 (~/.squish/eagle-heads) ───────────────────────────


def test_detect_eagle3_pass2_no_root_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "squish_home", lambda: tmp_path / "nohome")
    p = OptimizationProfile()
    # No adjacent head and no eagle-heads root → nothing detected (351-352).
    Det._detect_eagle3(p, tmp_path / "model", tmp_path / "comp")
    assert p.use_eagle3 is False


def test_detect_eagle3_pass2_picks_best_scoring_head(tmp_path, monkeypatch):
    home = tmp_path / "home"
    heads = home / "eagle-heads"
    # A non-dir entry (skipped), a head dir with no head file (skipped), and two
    # valid heads where the higher word-overlap slug wins.
    heads.mkdir(parents=True)
    (heads / "loose.txt").write_text("x")              # non-dir entry → skip (358-359)
    (heads / "empty-slug").mkdir()                     # dir w/o head → skip (361-362)
    (heads / "generic").mkdir()
    (heads / "generic" / "eagle3_head.safetensors").write_text("h")   # score 0
    (heads / "qwen3-8b").mkdir()
    (heads / "qwen3-8b" / "eagle3_head.safetensors").write_text("h")  # higher score
    (heads / "zzz-generic").mkdir()  # sorts last, score 0 → not > best (364→357)
    (heads / "zzz-generic" / "eagle3_head.safetensors").write_text("h")
    monkeypatch.setattr(ap, "squish_home", lambda: home)

    p = OptimizationProfile()
    model = tmp_path / "models" / "qwen3-8b-instruct"
    model.mkdir(parents=True)
    Det._detect_eagle3(p, model, tmp_path / "comp")  # comp/model have no adjacent head
    assert p.use_eagle3 is True
    assert p.eagle3_head_dir.endswith("qwen3-8b")  # best-scoring slug chosen


def test_detect_eagle3_pass2_no_valid_head_leaves_unset(tmp_path, monkeypatch):
    home = tmp_path / "home"
    heads = home / "eagle-heads"
    heads.mkdir(parents=True)
    (heads / "slug-without-head").mkdir()  # dir exists but has no head file
    monkeypatch.setattr(ap, "squish_home", lambda: home)
    p = OptimizationProfile()
    Det._detect_eagle3(p, tmp_path / "model", tmp_path / "comp")
    # Loop ran but found nothing → best_dir stays None (368→exit).
    assert p.use_eagle3 is False


def test_detect_eagle3_pass1_adjacent_head_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "squish_home", lambda: tmp_path / "unused-home")
    comp = tmp_path / "comp"
    comp.mkdir()
    (comp / "eagle3_head.safetensors").write_text("h")  # adjacent → pass 1 (342-347)
    p = OptimizationProfile()
    Det._detect_eagle3(p, tmp_path / "model", comp)
    assert p.use_eagle3 is True
    assert p.eagle3_head_dir.endswith("comp")
