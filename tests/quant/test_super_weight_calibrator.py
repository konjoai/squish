"""tests/quant/test_super_weight_calibrator.py

Unit tests for squish/quant/super_weight_calibrator.py.
"""
import numpy as np
import pytest

from squish.quant.super_weight_calibrator import (
    SuperWeightCalibrator,
    SuperWeightConfig,
    SuperWeightCoord,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# SuperWeightConfig
# ---------------------------------------------------------------------------

class TestSuperWeightConfig:
    def test_defaults(self):
        cfg = SuperWeightConfig()
        assert cfg.threshold == 100.0
        assert cfg.threshold_1d == 5.0
        assert cfg.max_per_tensor == 8
        assert cfg.max_per_tensor_1d == 0   # unlimited by default
        assert cfg.min_2d_cols == 64
        assert "embed_tokens" in cfg.skip_patterns
        assert "lm_head" in cfg.skip_patterns

    def test_custom_threshold(self):
        cfg = SuperWeightConfig(threshold=50.0)
        assert cfg.threshold == 50.0

    def test_custom_threshold_1d(self):
        cfg = SuperWeightConfig(threshold_1d=3.0)
        assert cfg.threshold_1d == 3.0

    def test_zero_max_per_tensor_means_no_limit(self):
        cfg = SuperWeightConfig(max_per_tensor=0)
        assert cfg.max_per_tensor == 0

    def test_max_per_tensor_1d_default_unlimited(self):
        cfg = SuperWeightConfig()
        assert cfg.max_per_tensor_1d == 0


# ---------------------------------------------------------------------------
# SuperWeightCoord
# ---------------------------------------------------------------------------

class TestSuperWeightCoord:
    def test_coord_key(self):
        c = SuperWeightCoord(
            tensor_name="model.layers.0.mlp.down_proj",
            row=5,
            col=123,
            value=3.14,
            ratio=250.0,
            original_shape=(256, 512),
        )
        assert c.coord_key == "model.layers.0.mlp.down_proj[5,123]"

    def test_frozen(self):
        c = SuperWeightCoord(
            tensor_name="t", row=0, col=0, value=1.0, ratio=1.0,
            original_shape=(4, 64),
        )
        with pytest.raises((AttributeError, TypeError)):
            c.row = 99  # frozen dataclass

    def test_ordering_by_ratio(self):
        # SuperWeightCoord uses dataclass lexicographic ordering.
        # Ensure identical (tensor_name, row, col, value) so only ratio differs.
        c1 = SuperWeightCoord("t", 0, 0, 1.0, 200.0, (4, 64))
        c2 = SuperWeightCoord("t", 0, 0, 1.0, 100.0, (4, 64))
        assert c1 > c2


# ---------------------------------------------------------------------------
# SuperWeightCalibrator._find_super_weights
# ---------------------------------------------------------------------------

class TestFindSuperWeights:
    """Tests for the internal per-tensor detection logic."""

    def _make_calibrator(self, threshold=100.0, max_per_tensor=8):
        return SuperWeightCalibrator(SuperWeightConfig(
            threshold=threshold, max_per_tensor=max_per_tensor
        ))

    def _inject_outlier(self, arr, row, col, ratio_target):
        """Set arr[row, col] so its outlier ratio exceeds ratio_target.

        Uses the analytic formula to account for how the injected value
        shifts the row mean post-injection:
            v = r * S_others / (n - r),  with n > r required.
        """
        arr = arr.copy()
        row_abs = np.abs(arr[row]).copy()
        row_abs[col] = 0.0
        S_others = float(row_abs.sum())
        n = arr.shape[1]
        # Clamp so n > effective_r (otherwise ratio is unachievable)
        effective_r = min(ratio_target, n * 0.9)
        v = effective_r * S_others / (n - effective_r)
        arr[row, col] = v * 1.5  # 50 % safety margin
        return arr

    def test_element_above_threshold_returned(self):
        arr = RNG.standard_normal((16, 128)).astype(np.float32) * 0.01
        arr = self._inject_outlier(arr, row=3, col=77, ratio_target=150.0)
        cal = self._make_calibrator(threshold=100.0)
        coords = cal._find_super_weights("test_tensor", arr, arr.shape)
        cols = [c.col for c in coords]
        assert 77 in cols

    def test_element_below_threshold_not_returned(self):
        arr = np.ones((16, 128), dtype=np.float32)
        # All elements identical → ratio = 1.0, well below 100
        cal = self._make_calibrator(threshold=100.0)
        coords = cal._find_super_weights("test_tensor", arr, arr.shape)
        assert coords == []

    def test_returned_coords_have_correct_tensor_name(self):
        arr = RNG.standard_normal((8, 128)).astype(np.float32) * 0.001
        arr = self._inject_outlier(arr, 0, 10, 200.0)
        cal = self._make_calibrator(threshold=100.0)
        coords = cal._find_super_weights("my.layer.weight", arr, arr.shape)
        assert all(c.tensor_name == "my.layer.weight" for c in coords)

    def test_max_per_tensor_respected(self):
        arr = RNG.standard_normal((32, 256)).astype(np.float32) * 0.001
        # Inject 20 outliers
        for col in range(20):
            arr = self._inject_outlier(arr, 0, col, 200.0)
        cal = self._make_calibrator(threshold=100.0, max_per_tensor=5)
        coords = cal._find_super_weights("t", arr, arr.shape)
        assert len(coords) <= 5

    def test_max_per_tensor_zero_returns_all(self):
        # Use separate rows so each injection is independent
        arr = RNG.standard_normal((32, 128)).astype(np.float32) * 0.001
        for row in range(10):
            arr = self._inject_outlier(arr, row, row, 200.0)
        cal = self._make_calibrator(threshold=100.0, max_per_tensor=0)
        coords = cal._find_super_weights("t", arr, arr.shape)
        assert len(coords) >= 10

    def test_coords_sorted_by_ratio_descending(self):
        arr = RNG.standard_normal((8, 128)).astype(np.float32) * 0.001
        arr = self._inject_outlier(arr, 0, 10, 500.0)
        arr = self._inject_outlier(arr, 1, 20, 200.0)
        cal = self._make_calibrator(threshold=100.0, max_per_tensor=0)
        coords = cal._find_super_weights("t", arr, arr.shape)
        ratios = [c.ratio for c in coords]
        assert ratios == sorted(ratios, reverse=True)

    def test_original_shape_preserved(self):
        arr = np.zeros((4, 128), dtype=np.float32)
        arr[0, 0] = 1_000_000.0   # enormous single outlier
        cal = self._make_calibrator(threshold=1.0)
        coords = cal._find_super_weights("t", arr, (2, 2, 128))
        assert all(c.original_shape == (2, 2, 128) for c in coords)


# ---------------------------------------------------------------------------
# SuperWeightCalibrator.scan_weights
# ---------------------------------------------------------------------------

class TestScanWeights:
    def _cal(self, **kwargs):
        return SuperWeightCalibrator(SuperWeightConfig(**kwargs))

    def _outlier_weight(self, shape, row, col, ratio=200.0):
        arr = RNG.standard_normal(shape).astype(np.float32) * 0.01
        # Analytic injection: v = r * S_others / (n - r)
        row_abs = np.abs(arr[row]).copy()
        row_abs[col] = 0.0
        S_others = float(row_abs.sum())
        n = arr.shape[1]
        effective_r = min(ratio, n * 0.9)
        v = effective_r * S_others / (n - effective_r)
        arr[row, col] = v * 1.5
        return arr

    def test_skip_patterns_respected(self):
        weights = {
            "embed_tokens.weight": self._outlier_weight((16, 128), 0, 10),
        }
        cal = self._cal(threshold=100.0)
        coords = cal.scan_weights(weights)
        assert coords == []

    def test_custom_skip_pattern(self):
        weights = {
            "my_custom_skip.weight": self._outlier_weight((8, 128), 0, 5),
        }
        cal = SuperWeightCalibrator(SuperWeightConfig(
            threshold=100.0, skip_patterns=["my_custom_skip"]
        ))
        coords = cal.scan_weights(weights)
        assert coords == []

    def test_min_2d_cols_respected(self):
        # Only 32 columns — below default min_2d_cols=64
        small = RNG.standard_normal((16, 32)).astype(np.float32)
        small[0, 0] = 1_000_000.0
        cal = self._cal(threshold=0.1, min_2d_cols=64)
        coords = cal.scan_weights({"layer.weight": small})
        assert coords == []

    def test_1d_tensor_handled(self):
        # 1-D tensors with sufficient length: not skipped, but ratios usually low
        arr = np.zeros(256, dtype=np.float32)
        arr[0] = 1_000_000.0
        cal = self._cal(threshold=0.1, min_2d_cols=64)
        # 1-D gets reshaped to (1, 256) which has 256 cols ≥ 64
        coords = cal.scan_weights({"bias": arr})
        assert isinstance(coords, list)

    def test_1d_tensor_uses_threshold_1d(self):
        # A 1-D tensor where the outlier ratio (≈7) is above threshold_1d=5
        # but below the 2-D threshold of 100.  It should be detected.
        base = np.ones(128, dtype=np.float32) * 0.5
        base[42] = 0.5 * 8.0  # ratio ≈ 8 × (1.0 + small_correction) > 5
        cal = self._cal(threshold=100.0, threshold_1d=5.0)
        coords = cal.scan_weights({"layernorm.weight": base})
        assert any(c.col == 42 for c in coords), (
            "1-D outlier at col=42 (ratio≈8) should be detected with threshold_1d=5"
        )

    def test_1d_tensor_below_threshold_1d_not_detected(self):
        # Ratio ≈ 2, below threshold_1d=5 — should not be detected
        base = np.ones(128, dtype=np.float32) * 0.5
        base[42] = 0.5 * 2.5  # ratio ≈ 2.5 < 5
        cal = self._cal(threshold=100.0, threshold_1d=5.0)
        coords = cal.scan_weights({"layernorm.weight": base})
        assert coords == []

    def test_1d_tensor_max_per_tensor_1d_unlimited_by_default(self):
        # Inject 20 outliers in a 1-D tensor; with max_per_tensor_1d=0 all are kept
        arr = np.ones(256, dtype=np.float32) * 0.1
        for i in range(20):
            arr[i] = 100.0  # ratio >> threshold_1d
        cal = self._cal(threshold=100.0, threshold_1d=5.0, max_per_tensor_1d=0)
        coords = cal.scan_weights({"bias": arr})
        assert len(coords) >= 20

    def test_1d_tensor_max_per_tensor_1d_respected(self):
        arr = np.ones(256, dtype=np.float32) * 0.1
        for i in range(20):
            arr[i] = 100.0
        cal = self._cal(threshold=100.0, threshold_1d=5.0, max_per_tensor_1d=3)
        coords = cal.scan_weights({"bias": arr})
        assert len(coords) <= 3

    def test_2d_tensor_not_affected_by_threshold_1d(self):
        # A 2-D tensor should use the regular threshold, NOT threshold_1d
        arr = RNG.standard_normal((8, 128)).astype(np.float32) * 0.01
        arr[0, 10] = 0.01 * 6.0  # ratio ≈ 6, above threshold_1d=5 but below threshold=100
        cal = self._cal(threshold=100.0, threshold_1d=5.0)
        coords = cal.scan_weights({"weight": arr})
        assert coords == [], "2-D tensor should not be detected by threshold_1d"

    def test_scalar_tensor_skipped(self):
        cal = self._cal(threshold=1.0)
        coords = cal.scan_weights({"scale": np.array(5.0, dtype=np.float32)})
        assert coords == []

    def test_multiple_tensors_all_scanned(self):
        w1 = self._outlier_weight((8, 128), 0, 10)
        w2 = self._outlier_weight((8, 128), 3, 50)
        cal = self._cal(threshold=100.0)
        coords = cal.scan_weights({"layer0": w1, "layer1": w2})
        names = {c.tensor_name for c in coords}
        assert "layer0" in names
        assert "layer1" in names

    def test_global_sort_by_ratio_descending(self):
        w1 = self._outlier_weight((8, 128), 0, 10, ratio=500.0)
        w2 = self._outlier_weight((8, 128), 0, 5,  ratio=200.0)
        cal = self._cal(threshold=100.0)
        coords = cal.scan_weights({"a": w1, "b": w2})
        ratios = [c.ratio for c in coords]
        assert ratios == sorted(ratios, reverse=True)

    def test_returns_list_of_super_weight_coord(self):
        arr = self._outlier_weight((8, 128), 0, 0, ratio=300.0)
        cal = self._cal(threshold=100.0)
        coords = cal.scan_weights({"t": arr})
        assert all(isinstance(c, SuperWeightCoord) for c in coords)


# ---------------------------------------------------------------------------
# is_1d path inside _find_super_weights
# ---------------------------------------------------------------------------

class TestFindSuperWeightsIs1D:
    """Tests for the is_1d=True path that uses threshold_1d."""

    def _cal(self, threshold_1d=5.0, threshold=1000.0, max_per_tensor_1d=0):
        return SuperWeightCalibrator(SuperWeightConfig(
            threshold=threshold,
            threshold_1d=threshold_1d,
            max_per_tensor=8,
            max_per_tensor_1d=max_per_tensor_1d,
        ))

    def _row_arr(self, n: int, outlier_col: int, approx_ratio: float) -> np.ndarray:
        """Build a (1, n) float32 array with one element at ~approx_ratio x row mean."""
        base = np.ones((1, n), dtype=np.float32) * 0.01
        base[0, outlier_col] = approx_ratio * 0.01 * 1.5   # 50 % margin
        return base

    def test_is_1d_true_detects_with_threshold_1d(self):
        """Outlier above threshold_1d (5.0) but far below threshold (1000) is found."""
        arr = self._row_arr(128, outlier_col=10, approx_ratio=8.0)
        cal = self._cal(threshold_1d=5.0, threshold=1000.0)
        coords = cal._find_super_weights("t", arr, (128,), is_1d=True)
        assert any(c.col == 10 for c in coords)

    def test_is_1d_false_misses_below_threshold(self):
        """Same array with is_1d=False should not be detected (threshold=1000 >> ratio)."""
        arr = self._row_arr(128, outlier_col=10, approx_ratio=8.0)
        cal = self._cal(threshold_1d=5.0, threshold=1000.0)
        coords = cal._find_super_weights("t", arr, (128,), is_1d=False)
        assert coords == []

    def test_max_per_tensor_1d_caps_results(self):
        arr = np.ones((1, 256), dtype=np.float32) * 0.01
        for col in range(20):
            arr[0, col] = 8.0 * 0.01 * 1.5
        cal = self._cal(threshold_1d=5.0, max_per_tensor_1d=4)
        coords = cal._find_super_weights("t", arr, (256,), is_1d=True)
        assert len(coords) <= 4

    def test_max_per_tensor_1d_zero_returns_all(self):
        arr = np.ones((1, 256), dtype=np.float32) * 0.01
        for col in range(20):
            arr[0, col] = 8.0 * 0.01 * 1.5
        cal = self._cal(threshold_1d=5.0, max_per_tensor_1d=0)
        coords = cal._find_super_weights("t", arr, (256,), is_1d=True)
        assert len(coords) >= 20


# ---------------------------------------------------------------------------
# 1D tensor handling through scan_weights
# ---------------------------------------------------------------------------

class TestScanWeights1D:
    """Tests that scan_weights correctly routes 1D tensors to threshold_1d."""

    def _cal(self, threshold_1d=5.0, threshold=1000.0, **kwargs):
        return SuperWeightCalibrator(SuperWeightConfig(
            threshold=threshold,
            threshold_1d=threshold_1d,
            **kwargs,
        ))

    def test_1d_tensor_uses_threshold_1d_not_threshold(self):
        """1D array with ratio above threshold_1d but below threshold is found."""
        arr = np.ones(128, dtype=np.float32) * 0.01
        arr[10] = 8.0 * 0.01 * 1.5   # ratio ~8 > threshold_1d=5 but << threshold=1000
        cal = self._cal(threshold_1d=5.0, threshold=1000.0)
        coords = cal.scan_weights({"layernorm.weight": arr})
        assert any(c.col == 10 for c in coords)

    def test_1d_tensor_below_threshold_1d_not_returned(self):
        """1D array with ratio < threshold_1d must not produce coordinates."""
        arr = np.ones(128, dtype=np.float32) * 0.01
        arr[10] = 3.0 * 0.01 * 1.5   # ratio ~3 < threshold_1d=5
        cal = self._cal(threshold_1d=5.0, threshold=1000.0)
        coords = cal.scan_weights({"t": arr})
        assert coords == []

    def test_1d_min_2d_cols_not_applied(self):
        """1D tensors below min_2d_cols should still be scanned."""
        arr = np.ones(32, dtype=np.float32) * 0.01    # 32 < min_2d_cols=64
        arr[5] = 100.0 * 0.01 * 1.5
        cal = self._cal(threshold_1d=5.0, threshold=1000.0, min_2d_cols=64)
        coords = cal.scan_weights({"small_bias": arr})
        assert any(c.col == 5 for c in coords)

    def test_1d_max_per_tensor_1d_caps_results(self):
        arr = np.ones(256, dtype=np.float32) * 0.01
        for col in range(20):
            arr[col] = 8.0 * 0.01 * 1.5
        cal = self._cal(threshold_1d=5.0, threshold=1000.0,
                        max_per_tensor_1d=3, max_per_tensor=8)
        coords = cal.scan_weights({"t": arr})
        assert len(coords) <= 3

    def test_2d_max_per_tensor_does_not_cap_1d(self):
        """max_per_tensor (for multi-row matrices) must NOT apply to 1D tensors."""
        arr = np.ones(256, dtype=np.float32) * 0.01
        for col in range(15):
            arr[col] = 8.0 * 0.01 * 1.5
        cal = self._cal(threshold_1d=5.0, threshold=1000.0,
                        max_per_tensor=2,   # would cap at 2 if wrongly applied
                        max_per_tensor_1d=0)  # unlimited for 1D
        coords = cal.scan_weights({"t": arr})
        assert len(coords) >= 15


# ---------------------------------------------------------------------------
# calibrate_from_dir — filesystem entry point
# ---------------------------------------------------------------------------

class TestCalibrateFromDir:
    def test_raises_file_not_found_on_empty_dir(self, tmp_path):
        from squish.quant.super_weight_calibrator import calibrate_from_dir
        with pytest.raises(FileNotFoundError):
            calibrate_from_dir(tmp_path)

    def test_raises_file_not_found_on_missing_dir(self, tmp_path):
        from squish.quant.super_weight_calibrator import calibrate_from_dir
        with pytest.raises(FileNotFoundError):
            calibrate_from_dir(tmp_path / "does_not_exist")

    def test_scans_safetensors_with_mock(self, tmp_path, monkeypatch):
        """Full calibrate_from_dir run against a mocked shard loader."""
        from squish.quant import super_weight_calibrator as mod

        # Fake shard: one tensor with a clear super weight
        arr = RNG.standard_normal((8, 128)).astype(np.float32) * 0.001
        row_mean = float(np.abs(arr[2]).mean()) + 1e-9
        arr[2, 77] = 300.0 * row_mean * 2  # big outlier

        def _fake_load(path):
            return {"model.layers.0.down_proj": arr}

        monkeypatch.setattr(mod, "_load_shard_f32", _fake_load)

        # Create a dummy .safetensors file so the glob finds something
        dummy = tmp_path / "model.safetensors"
        dummy.write_bytes(b"")

        from squish.quant.super_weight_calibrator import calibrate_from_dir
        coords = calibrate_from_dir(tmp_path, verbose=False)
        assert len(coords) >= 1
        assert coords[0].col == 77

    def test_verbose_does_not_raise(self, tmp_path, monkeypatch):
        from squish.quant import super_weight_calibrator as mod

        arr = np.zeros((8, 128), dtype=np.float32)
        arr[0, 0] = 1_000_000.0

        monkeypatch.setattr(mod, "_load_shard_f32", lambda p: {"t": arr})

        (tmp_path / "model.safetensors").write_bytes(b"")
        from squish.quant.super_weight_calibrator import calibrate_from_dir
        calibrate_from_dir(tmp_path, verbose=True)   # must not raise
