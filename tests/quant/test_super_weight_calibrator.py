"""tests/quant/test_super_weight_calibrator.py — unit + regression tests for
squish/quant/super_weight_calibrator.py.

Regression context (issue #37): the calibrator module was wrongly deleted by a
"lean purge" while three live callers kept referencing it — convert.py imports
it, the ``--super-weight`` CLI flag feeds it, and catalog.py appends
``--super-weight`` to *every* INT4 compress. The result: ``squish pull`` crashed
with ``ModuleNotFoundError`` on any model without pre-squished weights. These
tests pin both the calibrator's behaviour and the exact import convert.py does,
so the file cannot be silently removed again while its callers remain.
"""

import importlib
from pathlib import Path

import numpy as np

import squish

from squish.quant.super_weight_calibrator import (
    SuperWeightCalibrator,
    SuperWeightConfig,
    SuperWeightCoord,
    calibrate_from_dir,
)

RNG = np.random.default_rng(37)


# ---------------------------------------------------------------------------
# SuperWeightConfig
# ---------------------------------------------------------------------------

class TestSuperWeightConfig:
    def test_defaults(self):
        cfg = SuperWeightConfig()
        assert cfg.threshold == 100.0
        assert cfg.min_2d_cols == 64
        assert "lm_head" in cfg.skip_patterns

    def test_overrides_match_convert_call(self):
        # convert.py constructs exactly this — keep the kwargs valid.
        cfg = SuperWeightConfig(threshold=100.0, threshold_1d=1e9)
        assert cfg.threshold == 100.0
        assert cfg.threshold_1d == 1e9


# ---------------------------------------------------------------------------
# scan_weights
# ---------------------------------------------------------------------------

class TestScanWeights:
    def _matrix_with_outlier(self, row, col, value):
        m = (RNG.standard_normal((128, 128)) * 0.02).astype(np.float32)
        m[row, col] = value
        return {"model.layers.0.mlp.down_proj.weight": m}

    def test_detects_planted_super_weight(self):
        cal = SuperWeightCalibrator(SuperWeightConfig(threshold=100.0, threshold_1d=1e9))
        coords = cal.scan_weights(self._matrix_with_outlier(7, 11, 50.0))
        assert any(c.row == 7 and c.col == 11 for c in coords)
        assert all(isinstance(c, SuperWeightCoord) for c in coords)
        # convert.py reads c.tensor_name to build its passthrough set.
        assert coords[0].tensor_name == "model.layers.0.mlp.down_proj.weight"

    def test_threshold_suppresses_mild_outliers(self):
        # A 5× outlier is well under threshold=100 → not flagged.
        cal = SuperWeightCalibrator(SuperWeightConfig(threshold=100.0, threshold_1d=1e9))
        coords = cal.scan_weights(self._matrix_with_outlier(3, 4, 0.1))
        assert coords == []

    def test_small_tensors_skipped(self):
        # < min_2d_cols columns → skipped regardless of outlier magnitude.
        cal = SuperWeightCalibrator(SuperWeightConfig(threshold=1.0, threshold_1d=1e9))
        tiny = {"w": np.full((8, 8), 1.0, dtype=np.float32)}
        tiny["w"][0, 0] = 1e6
        assert cal.scan_weights(tiny) == []


# ---------------------------------------------------------------------------
# convert.py wiring — the actual regression guard
# ---------------------------------------------------------------------------

class TestConvertWiring:
    def test_convert_import_path_resolves(self):
        """The exact import convert.py performs must succeed. If the module is
        deleted again while convert.py still imports it, this fails loudly."""
        mod = importlib.import_module("squish.quant.super_weight_calibrator")
        assert hasattr(mod, "SuperWeightCalibrator")
        assert hasattr(mod, "SuperWeightConfig")

    def test_convert_still_references_module(self):
        """Tie the module to its caller: convert.py must import it, and
        catalog.py must keep wiring --super-weight onto INT4 pulls. Catches the
        inverse drift (caller removed but module kept, or vice versa). Reads the
        sources from disk to stay free of any import side-effects."""
        pkg = Path(squish.__file__).parent
        convert_src = (pkg / "convert.py").read_text(encoding="utf-8")
        assert "super_weight_calibrator import" in convert_src
        catalog_src = (pkg / "catalog.py").read_text(encoding="utf-8")
        assert "--super-weight" in catalog_src

    def test_calibrate_from_dir_is_exported(self):
        assert callable(calibrate_from_dir)
