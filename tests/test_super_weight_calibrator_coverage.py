"""Supplementary coverage for ``squish.quant.super_weight_calibrator`` — the
filesystem entry point, the shard loader (safetensors success + MLX BF16
fallback), and the scan guard branches the existing suite leaves uncovered.
Real safetensors files are used for the happy path; a fake ``mlx`` package
drives the fallback, so the suite is host-agnostic.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from squish.quant import super_weight_calibrator as swc
from squish.quant.super_weight_calibrator import (
    SuperWeightCalibrator,
    SuperWeightConfig,
    SuperWeightCoord,
    calibrate_from_dir,
)


# ── SuperWeightCoord ─────────────────────────────────────────────────────────


def test_coord_key():
    c = SuperWeightCoord("model.layers.3.mlp", 12, 34, 9.5, 250.0, (4096, 4096))
    assert c.coord_key == "model.layers.3.mlp[12,34]"


# ── scan_weights guards ──────────────────────────────────────────────────────


def test_scan_skips_patterns_and_scalars():
    cal = SuperWeightCalibrator()
    coords = cal.scan_weights(
        {
            "model.embed_tokens.weight": np.ones((4, 128), np.float32),  # skip pattern
            "scalar": np.array(5.0, np.float32),  # ndim 0 → skip
        }
    )
    assert coords == []


def test_scan_skips_narrow_2d_tensor():
    cal = SuperWeightCalibrator()  # min_2d_cols = 64
    row = np.ones((4, 16), np.float32)
    row[0, 0] = 1e6  # would be an outlier, but tensor is too narrow → skipped
    assert cal.scan_weights({"w": row}) == []


def test_scan_no_outliers_returns_empty():
    cal = SuperWeightCalibrator()
    assert cal.scan_weights({"w": np.ones((4, 128), np.float32)}) == []


def test_scan_1d_tensor_uses_threshold_1d_unlimited():
    cal = SuperWeightCalibrator()  # threshold_1d=5, max_per_tensor_1d=0 (no limit)
    vec = np.ones(64, np.float32)
    vec[7] = 100.0  # ratio ≈ 40 > 5
    coords = cal.scan_weights({"ln.weight": vec})
    assert len(coords) == 1 and coords[0].row == 0 and coords[0].col == 7


def test_scan_2d_applies_per_tensor_limit():
    cal = SuperWeightCalibrator()  # threshold=100, max_per_tensor=8
    mat = np.ones((10, 256), np.float32)
    for r in range(10):
        mat[r, r] = 1e6  # one extreme outlier per row → ratio ≈ 256 > 100
    coords = cal.scan_weights({"w": mat})
    assert len(coords) == 8  # 10 outliers capped to max_per_tensor
    # globally sorted by ratio descending
    assert coords == sorted(coords, key=lambda c: c.ratio, reverse=True)


# ── _load_shard_f32 ──────────────────────────────────────────────────────────


def test_load_shard_safetensors_success(tmp_path):
    p = tmp_path / "shard.safetensors"
    save_file({"a": np.ones((2, 2), np.float32)}, str(p))
    out = swc._load_shard_f32(p)
    assert out["a"].dtype == np.float32 and out["a"].shape == (2, 2)


def test_load_shard_falls_back_to_mlx(tmp_path, monkeypatch):
    # Force the safetensors path to fail → exercise the MLX CPU fallback.
    monkeypatch.setattr(
        "safetensors.numpy.load_file",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad dtype")),
    )

    state = {"dev": "gpu"}
    core = types.ModuleType("mlx.core")
    core.float32 = np.float32
    core.cpu = "cpu"
    core.default_device = lambda: state["dev"]
    core.set_default_device = lambda d: state.__setitem__("dev", d)
    core.load = lambda _p: {"w.bf16": np.array([1.0, 2.0], np.float32)}
    pkg = types.ModuleType("mlx")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    out = swc._load_shard_f32(tmp_path / "x.safetensors")
    assert "w.bf16" in out
    assert state["dev"] == "gpu"  # default device restored in finally


# ── calibrate_from_dir ───────────────────────────────────────────────────────


def test_calibrate_from_dir_no_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .safetensors"):
        calibrate_from_dir(tmp_path)


def test_calibrate_from_dir_finds_and_reports(tmp_path, capsys):
    mat = np.ones((4, 256), np.float32)
    mat[0, 0] = 1e6  # one super weight (ratio ≈ 256 > 100)
    save_file({"model.layers.0.mlp.down_proj": mat}, str(tmp_path / "s.safetensors"))

    verbose = calibrate_from_dir(tmp_path, verbose=True)
    assert len(verbose) == 1 and verbose[0].ratio > 100
    out = capsys.readouterr().out
    assert "Scanning" in out and "Total super weights found: 1" in out

    quiet = calibrate_from_dir(tmp_path, config=SuperWeightConfig(), verbose=False)
    assert len(quiet) == 1
    assert capsys.readouterr().out == ""  # nothing printed when not verbose


def test_calibrate_from_dir_verbose_with_no_super_weights(tmp_path, capsys):
    # verbose=True but no outliers → the `if all_coords:` summary branch is skipped.
    save_file({"w": np.ones((4, 256), np.float32)}, str(tmp_path / "s.safetensors"))
    coords = calibrate_from_dir(tmp_path, verbose=True)
    assert coords == []
    assert "Total super weights found: 0" in capsys.readouterr().out
