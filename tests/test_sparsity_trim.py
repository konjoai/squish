"""tests/test_sparsity_trim.py

Unit tests for ``squish sparsity-trim`` (Wave 112).

Tests are pure-unit: no Metal, no model loading, no subprocess.
A synthetic 2-layer BF16 model and a synthetic 2-layer INT4 model
(uint32-packed, with scales and biases) are written to tmp_path and
tested directly via ``cmd_sparsity_trim``.

Coverage:
  * BF16 model: dry-run (no files written), actual trim, output stats
  * INT4 model: dry-run, actual trim row/column removal
  * Config updated: intermediate_size matches kept neuron count
  * Group-alignment: pruning always a multiple of group_size
  * Error paths: bad threshold, bad group_size, missing files,
    output dir already exists
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    defaults = dict(
        model="",
        threshold=0.25,
        group_size=8,  # small group for tests
        dry_run=False,
        output="",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_bf16_model(tmp_path: Path, n_layers: int = 2, hidden: int = 32,
                      intermediate: int = 64) -> Path:
    """Write a minimal BF16 safetensors model to tmp_path."""
    from safetensors.numpy import save_file
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    weights = {
        "model.embed_tokens.weight": rng.standard_normal((16, hidden)).astype(np.float32),
    }
    for i in range(n_layers):
        base = f"model.layers.{i}.mlp"
        weights[f"{base}.up_proj.weight"]   = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        weights[f"{base}.gate_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        weights[f"{base}.down_proj.weight"] = rng.standard_normal((hidden, intermediate)).astype(np.float32)
    save_file(weights, str(tmp_path / "model.safetensors"))
    cfg = {
        "num_hidden_layers": n_layers,
        "intermediate_size": intermediate,
        "hidden_size": hidden,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def _pack_int4_rows(w_float: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fake-quantize float32 (n_rows, n_cols) to MLX uint32 INT4 format."""
    n_rows, n_cols = w_float.shape
    assert n_cols % 8 == 0, "n_cols must be multiple of 8 for INT4 packing"
    # Scales: one per column group of 8 for simplicity (group_size=8 in tests)
    group_size = 8
    n_groups = n_cols // group_size
    scales = np.ones((n_rows, n_groups), dtype=np.float16)
    biases = np.zeros((n_rows, n_groups), dtype=np.float16)
    # Pack: clamp to [0,15], store 8 values per uint32 (bits 0-3, 4-7, ..., 28-31)
    w_int4 = np.clip(((w_float + 8)).astype(np.int32), 0, 15).astype(np.uint32)
    n_cols_packed = n_cols // 8
    w_packed = np.zeros((n_rows, n_cols_packed), dtype=np.uint32)
    for i in range(8):
        w_packed |= (w_int4[:, i::8] & 0xF) << (i * 4)
    return w_packed, scales, biases


def _write_int4_model(tmp_path: Path, n_layers: int = 2, hidden: int = 32,
                      intermediate: int = 64) -> Path:
    """Write a minimal MLX INT4 safetensors model to tmp_path."""
    from safetensors.numpy import save_file
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    weights = {
        "model.embed_tokens.weight": rng.standard_normal((16, hidden)).astype(np.float32),
    }
    for i in range(n_layers):
        base = f"model.layers.{i}.mlp"
        # up_proj / gate_proj: (intermediate, hidden) → packed rows
        for proj in ("up_proj", "gate_proj"):
            w_f = rng.standard_normal((intermediate, hidden)).astype(np.float32)
            w_u32, scales, biases = _pack_int4_rows(w_f)
            weights[f"{base}.{proj}.weight"] = w_u32
            weights[f"{base}.{proj}.scales"] = scales
            weights[f"{base}.{proj}.biases"] = biases
        # down_proj: (hidden, intermediate) → packed cols (each col is intermediate neurons)
        # For simplicity, transpose the packing for down_proj:
        # stored as (hidden, intermediate//8) uint32
        w_f = rng.standard_normal((hidden, intermediate)).astype(np.float32)
        w_u32_d, scales_d, biases_d = _pack_int4_rows(w_f)
        weights[f"{base}.down_proj.weight"] = w_u32_d
        weights[f"{base}.down_proj.scales"] = scales_d
        weights[f"{base}.down_proj.biases"] = biases_d
    save_file(weights, str(tmp_path / "model.safetensors"))
    cfg = {
        "num_hidden_layers": n_layers,
        "intermediate_size": intermediate,
        "hidden_size": hidden,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def _import_cmd():
    from squish.cli import cmd_sparsity_trim  # noqa: PLC0415
    return cmd_sparsity_trim


# ---------------------------------------------------------------------------
# BF16 model tests
# ---------------------------------------------------------------------------

class TestSparsityTrimBF16(unittest.TestCase):

    def test_dry_run_exits_zero_no_files(self):
        """--dry-run must exit(0) without writing any files."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model")
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), dry_run=True, group_size=8)
            with pytest.raises(SystemExit) as exc:
                cmd(ns)
            assert exc.value.code == 0
            # No trimmed dir should exist
            trimmed = model_dir.parent / (model_dir.name + "-trimmed")
            assert not trimmed.exists()

    def test_bf16_trim_creates_output_dir(self):
        """Trimming creates output dir and model.safetensors + config.json."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model")
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            assert (out_dir / "model.safetensors").exists()
            assert (out_dir / "config.json").exists()

    def test_bf16_config_updated(self):
        """config.json intermediate_size should reflect kept neurons."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model",
                                          intermediate=64)
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            # threshold=0.25 → prune 25% of 8 groups (group_size=8) = 2 groups = 16 neurons
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            cfg = json.loads((out_dir / "config.json").read_text())
            # 64 neurons, 8 groups of 8, prune 2 → keep 48
            assert cfg["intermediate_size"] == 48

    def test_bf16_weight_shapes_updated(self):
        """Trimmed up_proj, gate_proj rows and down_proj cols match kept count."""
        import tempfile
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model",
                                          hidden=32, intermediate=64)
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            with safe_open(str(out_dir / "model.safetensors"), framework="numpy") as f:
                up_shape = f.get_tensor("model.layers.0.mlp.up_proj.weight").shape
                dp_shape = f.get_tensor("model.layers.0.mlp.down_proj.weight").shape
            # 48 neurons kept
            assert up_shape[0] == 48, f"up_proj rows should be 48, got {up_shape[0]}"
            assert dp_shape[1] == 48, f"down_proj cols should be 48, got {dp_shape[1]}"


# ---------------------------------------------------------------------------
# INT4 model tests
# ---------------------------------------------------------------------------

class TestSparsityTrimINT4(unittest.TestCase):

    def test_int4_dry_run_exits_zero(self):
        """--dry-run on INT4 model exits(0) without writing files."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_int4_model(Path(td) / "model")
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), dry_run=True, group_size=8)
            with pytest.raises(SystemExit) as exc:
                cmd(ns)
            assert exc.value.code == 0

    def test_int4_trim_preserves_dtype(self):
        """Trimmed INT4 weight stays uint32; scales and biases stay float16."""
        import tempfile
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_int4_model(Path(td) / "model",
                                          hidden=32, intermediate=64)
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            with safe_open(str(out_dir / "model.safetensors"), framework="numpy") as f:
                w = f.get_tensor("model.layers.0.mlp.up_proj.weight")
                s = f.get_tensor("model.layers.0.mlp.up_proj.scales")
                b = f.get_tensor("model.layers.0.mlp.up_proj.biases")
            assert w.dtype == np.uint32, f"weight dtype should be uint32, got {w.dtype}"
            assert s.dtype == np.float16, f"scales dtype should be float16, got {s.dtype}"
            assert b.dtype == np.float16, f"biases dtype should be float16, got {b.dtype}"

    def test_int4_row_count_reduced(self):
        """Trimmed up_proj row count matches kept neuron count."""
        import tempfile
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_int4_model(Path(td) / "model",
                                          hidden=32, intermediate=64)
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            # threshold=0.25 → prune 2/8 groups → keep 48 neurons
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            with safe_open(str(out_dir / "model.safetensors"), framework="numpy") as f:
                w = f.get_tensor("model.layers.0.mlp.up_proj.weight")
                s = f.get_tensor("model.layers.0.mlp.up_proj.scales")
            # 48 rows remaining
            assert w.shape[0] == 48, f"up_proj rows: expected 48, got {w.shape[0]}"
            # scales: (48, n_groups_in) — scales cover input hidden cols, not intermediate rows
            assert s.shape[0] == 48, f"up scales rows: expected 48, got {s.shape[0]}"

    def test_int4_down_proj_col_groups_reduced(self):
        """Trimmed down_proj uint32 column count reduced by pruned groups."""
        import tempfile
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as td:
            # intermediate=64, group_size=8: 8 groups, prune 2 → keep 6 → 48 cols
            model_dir = _write_int4_model(Path(td) / "model",
                                          hidden=32, intermediate=64)
            out_dir = Path(td) / "trimmed_out"
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), threshold=0.25, group_size=8,
                            output=str(out_dir))
            cmd(ns)
            with safe_open(str(out_dir / "model.safetensors"), framework="numpy") as f:
                dp_w = f.get_tensor("model.layers.0.mlp.down_proj.weight")
                dp_s = f.get_tensor("model.layers.0.mlp.down_proj.scales")
            # Each group of 8 neurons = 1 uint32 column (group_size=8, 8/8=1 u32 per group)
            # 6 groups kept → 6 uint32 columns
            assert dp_w.shape[1] == 6, f"down_proj u32 cols: expected 6, got {dp_w.shape[1]}"
            assert dp_s.shape[1] == 6, f"down scales cols: expected 6, got {dp_s.shape[1]}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestSparsityTrimErrors(unittest.TestCase):

    def test_bad_threshold_exits_one(self):
        """threshold <= 0 or >= 1 must exit(1)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model")
            cmd = _import_cmd()
            for bad in (0.0, 1.0, -0.1, 1.5):
                ns = _make_args(model=str(model_dir), threshold=bad)
                with pytest.raises(SystemExit) as exc:
                    cmd(ns)
                assert exc.value.code == 1, f"threshold={bad} should exit(1)"

    def test_bad_group_size_exits_one(self):
        """group_size not a multiple of 8 must exit(1)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model")
            cmd = _import_cmd()
            for bad in (3, 7, 9, 1):
                ns = _make_args(model=str(model_dir), group_size=bad)
                with pytest.raises(SystemExit) as exc:
                    cmd(ns)
                assert exc.value.code == 1, f"group_size={bad} should exit(1)"

    def test_missing_safetensors_exits_one(self):
        """Missing model.safetensors must exit(1)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "empty_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps(
                {"num_hidden_layers": 2, "intermediate_size": 64}))
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir))
            with pytest.raises(SystemExit) as exc:
                cmd(ns)
            assert exc.value.code == 1

    def test_output_dir_exists_exits_one(self):
        """Existing output directory must trigger exit(1)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_bf16_model(Path(td) / "model")
            out_dir = Path(td) / "already_exists"
            out_dir.mkdir()
            cmd = _import_cmd()
            ns = _make_args(model=str(model_dir), output=str(out_dir))
            with pytest.raises(SystemExit) as exc:
                cmd(ns)
            assert exc.value.code == 1


if __name__ == "__main__":
    unittest.main()
