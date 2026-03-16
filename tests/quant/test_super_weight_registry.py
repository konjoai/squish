"""tests/quant/test_super_weight_registry.py

Unit tests for squish/quant/super_weight_registry.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from squish.quant.super_weight_registry import (
    SuperWeightRegistry,
    load_registry,
    save_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(model_dir="/tmp/MyModel", threshold=100.0, entries=None):
    if entries is None:
        entries = [
            {
                "tensor_name": "model.layers.0.mlp.down_proj",
                "row": 5,
                "col": 3968,
                "value": 2.3,
                "ratio": 487.0,
                "original_shape": [4096, 11008],
            },
            {
                "tensor_name": "model.layers.0.mlp.down_proj",
                "row": 10,
                "col": 4021,
                "value": 1.8,
                "ratio": 210.0,
                "original_shape": [4096, 11008],
            },
            {
                "tensor_name": "model.layers.1.mlp.up_proj",
                "row": 0,
                "col": 512,
                "value": 3.1,
                "ratio": 301.0,
                "original_shape": [4096, 11008],
            },
        ]
    return SuperWeightRegistry(
        schema_version=1,
        model_name="MyModel",
        model_dir=model_dir,
        threshold=threshold,
        calibrated_at="2026-03-14T12:00:00+00:00",
        super_weights=entries,
    )


# ---------------------------------------------------------------------------
# from_coords
# ---------------------------------------------------------------------------

class TestFromCoords:
    def test_correct_model_name_from_dir(self, tmp_path):
        model_dir = tmp_path / "Qwen3-8B-bf16"
        model_dir.mkdir()
        from squish.quant.super_weight_calibrator import SuperWeightCoord
        coords = [
            SuperWeightCoord("t", 0, 5, 1.0, 200.0, (16, 128)),
        ]
        reg = SuperWeightRegistry.from_coords(coords, model_dir, threshold=100.0)
        assert reg.model_name == "Qwen3-8B-bf16"

    def test_threshold_stored(self, tmp_path):
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        from squish.quant.super_weight_calibrator import SuperWeightCoord
        coords = [SuperWeightCoord("t", 0, 0, 1.0, 150.0, (4, 64))]
        reg = SuperWeightRegistry.from_coords(coords, model_dir, threshold=75.5)
        assert reg.threshold == 75.5

    def test_super_weights_populated(self, tmp_path):
        model_dir = tmp_path / "M"
        model_dir.mkdir()
        from squish.quant.super_weight_calibrator import SuperWeightCoord
        coords = [
            SuperWeightCoord("a.b.c", 1, 7, 2.5, 300.0, (8, 128)),
            SuperWeightCoord("d.e.f", 2, 9, 1.1, 200.0, (8, 128)),
        ]
        reg = SuperWeightRegistry.from_coords(coords, model_dir)
        assert len(reg.super_weights) == 2
        assert reg.super_weights[0]["tensor_name"] == "a.b.c"
        assert reg.super_weights[0]["col"] == 7

    def test_schema_version_is_1(self, tmp_path):
        model_dir = tmp_path / "M"
        model_dir.mkdir()
        reg = SuperWeightRegistry.from_coords([], model_dir)
        assert reg.schema_version == 1

    def test_calibrated_at_is_iso_utc(self, tmp_path):
        model_dir = tmp_path / "M"
        model_dir.mkdir()
        reg = SuperWeightRegistry.from_coords([], model_dir)
        # Must be parseable as ISO datetime
        from datetime import datetime
        dt = datetime.fromisoformat(reg.calibrated_at)
        assert dt is not None


# ---------------------------------------------------------------------------
# protected_columns
# ---------------------------------------------------------------------------

class TestProtectedColumns:
    def test_returns_sorted_unique_cols(self):
        reg = _make_registry()
        cols = reg.protected_columns("model.layers.0.mlp.down_proj")
        assert cols == sorted(set(cols))
        assert 3968 in cols
        assert 4021 in cols

    def test_empty_for_unknown_tensor(self):
        reg = _make_registry()
        assert reg.protected_columns("nonexistent.tensor") == []

    def test_deduplicates(self):
        # Two entries with the same col
        entries = [
            {"tensor_name": "t", "row": 0, "col": 100, "value": 1.0, "ratio": 200.0, "original_shape": [4, 128]},
            {"tensor_name": "t", "row": 1, "col": 100, "value": 1.1, "ratio": 210.0, "original_shape": [4, 128]},
        ]
        reg = _make_registry(entries=entries)
        cols = reg.protected_columns("t")
        assert cols.count(100) == 1

    def test_returns_list(self):
        reg = _make_registry()
        result = reg.protected_columns("model.layers.0.mlp.down_proj")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# protected_mask
# ---------------------------------------------------------------------------

class TestProtectedMask:
    def test_shape_matches_input(self):
        reg = _make_registry()
        shape = (4096, 11008)
        mask = reg.protected_mask("model.layers.0.mlp.down_proj", shape)
        assert mask.shape == shape

    def test_protected_cols_are_true(self):
        reg = _make_registry()
        shape = (4096, 11008)
        mask = reg.protected_mask("model.layers.0.mlp.down_proj", shape)
        assert mask[:, 3968].all()
        assert mask[:, 4021].all()

    def test_unprotected_cols_are_false(self):
        reg = _make_registry()
        shape = (4096, 11008)
        mask = reg.protected_mask("model.layers.0.mlp.down_proj", shape)
        # Column 0 should not be protected
        assert not mask[:, 0].any()

    def test_empty_registry_returns_all_false(self):
        reg = _make_registry(entries=[])
        mask = reg.protected_mask("anything", (16, 64))
        assert not mask.any()

    def test_out_of_range_col_silently_ignored(self):
        entries = [
            {"tensor_name": "t", "row": 0, "col": 9999, "value": 1.0, "ratio": 200.0, "original_shape": [4, 64]},
        ]
        reg = _make_registry(entries=entries)
        mask = reg.protected_mask("t", (4, 64))   # col 9999 > 64
        assert not mask.any()

    def test_multidim_shape(self):
        entries = [
            {"tensor_name": "t", "row": 0, "col": 5, "value": 1.0, "ratio": 200.0, "original_shape": [2, 8, 64]},
        ]
        reg = _make_registry(entries=entries)
        mask = reg.protected_mask("t", (2, 8, 64))
        assert mask[..., 5].all()
        assert not mask[..., 0].any()


# ---------------------------------------------------------------------------
# Dunder methods and helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_len(self):
        reg = _make_registry()
        assert len(reg) == 3

    def test_len_empty(self):
        reg = _make_registry(entries=[])
        assert len(reg) == 0

    def test_has_tensor_true(self):
        reg = _make_registry()
        assert reg.has_tensor("model.layers.0.mlp.down_proj")

    def test_has_tensor_false(self):
        reg = _make_registry()
        assert not reg.has_tensor("nonexistent")

    def test_summary_contains_model_name(self):
        reg = _make_registry()
        s = reg.summary()
        assert "MyModel" in s

    def test_summary_contains_count(self):
        reg = _make_registry()
        s = reg.summary()
        assert "3" in s


# ---------------------------------------------------------------------------
# save_registry / load_registry round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_round_trip_identical(self, tmp_path):
        reg = _make_registry()
        path = tmp_path / "reg.json"
        save_registry(reg, path)
        loaded = load_registry(path)

        assert loaded.schema_version == reg.schema_version
        assert loaded.model_name == reg.model_name
        assert loaded.model_dir == reg.model_dir
        assert loaded.threshold == reg.threshold
        assert loaded.calibrated_at == reg.calibrated_at
        assert len(loaded.super_weights) == len(reg.super_weights)

    def test_round_trip_protected_cols(self, tmp_path):
        reg = _make_registry()
        path = tmp_path / "reg.json"
        save_registry(reg, path)
        loaded = load_registry(path)
        assert loaded.protected_columns("model.layers.0.mlp.down_proj") == [3968, 4021]

    def test_creates_parent_dirs(self, tmp_path):
        reg = _make_registry()
        path = tmp_path / "deep" / "nested" / "reg.json"
        save_registry(reg, path)
        assert path.exists()

    def test_valid_json_file(self, tmp_path):
        reg = _make_registry()
        path = tmp_path / "reg.json"
        save_registry(reg, path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        assert "super_weights" in data

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_registry(tmp_path / "missing.json")

    def test_load_wrong_schema_version_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": 999, "super_weights": []}))
        with pytest.raises(ValueError, match="schema"):
            load_registry(path)

    def test_empty_super_weights_round_trip(self, tmp_path):
        reg = _make_registry(entries=[])
        path = tmp_path / "empty.json"
        save_registry(reg, path)
        loaded = load_registry(path)
        assert len(loaded) == 0
        assert loaded.protected_columns("any.tensor") == []
