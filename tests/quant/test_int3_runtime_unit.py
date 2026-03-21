"""tests/quant/test_int3_runtime_unit.py

Dedicated unit tests for squish/quant/int3_runtime.py.

Covers:
  - INT3RuntimeConfig validation
  - INT3LayerWeights construction and properties
  - INT3RuntimeLoader.load_from_arrays
  - INT3RuntimeLoader.dequantize (full tensor)
  - INT3RuntimeLoader.dequantize_tiled (streaming)
  - INT3RuntimeLoader.load_layer (from on-disk .npy files)
  - Stats tracking
  - Round-trip accuracy (dequant SNR vs. FP32 original)
  - Edge cases: single group, all-zero weights, all-same weights
  - Error paths: bad shapes, missing files
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.quant.int3_runtime import (
    INT3LayerWeights,
    INT3LoaderStats,
    INT3RuntimeConfig,
    INT3RuntimeLoader,
)

RNG = np.random.default_rng(0xDEAD_BEEF)

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_int3_arrays(
    n_weights: int = 512,
    group_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_packed, scales, zeros) compatible with INT3RuntimeLoader.

    q_packed  — uint8, shape (n_groups, group_size), values in [0,7]
    scales    — float32, shape (n_groups,)
    zeros     — float32, shape (n_groups,)
    """
    rng = np.random.default_rng(seed)
    n_groups = n_weights // group_size
    q_packed = rng.integers(0, 8, size=(n_groups, group_size), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.05, size=(n_groups,)).astype(np.float32)
    zeros = rng.uniform(-0.1, 0.1, size=(n_groups,)).astype(np.float32)
    return q_packed, scales, zeros


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between two arrays."""
    signal_power = float(np.mean(original ** 2))
    noise_power = float(np.mean((original - reconstructed) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


# ── INT3RuntimeConfig ──────────────────────────────────────────────────────────

class TestINT3RuntimeConfig:
    def test_defaults(self):
        cfg = INT3RuntimeConfig()
        assert cfg.group_size == 64
        assert cfg.tile_size == 256
        assert cfg.dtype == np.dtype("float32")

    def test_custom_values(self):
        cfg = INT3RuntimeConfig(group_size=128, tile_size=64)
        assert cfg.group_size == 128
        assert cfg.tile_size == 64

    def test_invalid_group_size_too_small(self):
        with pytest.raises(ValueError, match="group_size"):
            INT3RuntimeConfig(group_size=4)

    def test_invalid_tile_size_zero(self):
        with pytest.raises(ValueError, match="tile_size"):
            INT3RuntimeConfig(tile_size=0)

    def test_group_size_minimum_8(self):
        cfg = INT3RuntimeConfig(group_size=8)
        assert cfg.group_size == 8


# ── INT3LayerWeights ───────────────────────────────────────────────────────────

class TestINT3LayerWeights:
    def test_n_groups(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = INT3LayerWeights(q_packed=q, scales=s, zeros=z,
                              original_shape=(512,), group_size=64)
        assert lw.n_groups == 8

    def test_compactness_positive(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = INT3LayerWeights(q_packed=q, scales=s, zeros=z,
                              original_shape=(512,), group_size=64)
        # About 3-bit: compactness ≈ fp32_bytes / packed_bytes ≈ > 1
        assert lw.compactness > 1.0

    def test_compactness_vs_fp32(self):
        """512 float32 weights = 2048 bytes; packed uint8 (512) + scales (32) + zeros (32)."""
        q, s, z = _make_int3_arrays(512, 64)
        lw = INT3LayerWeights(q_packed=q, scales=s, zeros=z,
                              original_shape=(512,), group_size=64)
        fp32_bytes = 512 * 4
        packed_bytes = q.nbytes + s.nbytes + z.nbytes
        expected_ratio = fp32_bytes / packed_bytes
        assert abs(lw.compactness - expected_ratio) < 1e-6


# ── INT3RuntimeLoader — load_from_arrays ──────────────────────────────────────

class TestINT3LoadFromArrays:
    def setup_method(self):
        self.loader = INT3RuntimeLoader(INT3RuntimeConfig())

    def test_returns_int3_layer_weights(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        assert isinstance(lw, INT3LayerWeights)

    def test_shape_round_trip(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(8, 64))
        assert lw.original_shape == (8, 64)

    def test_dtypes_normalised(self):
        q = np.ones((8, 64), dtype=np.uint16)  # non-uint8 input
        s = np.ones(8, dtype=np.float64)        # non-float32
        z = np.zeros(8, dtype=np.float64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        assert lw.q_packed.dtype == np.uint8
        assert lw.scales.dtype == np.float32
        assert lw.zeros.dtype == np.float32

    def test_stats_updated(self):
        q, s, z = _make_int3_arrays(512, 64)
        before = self.loader.stats.layers_loaded
        self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        assert self.loader.stats.layers_loaded == before + 1

    def test_mismatched_shapes_raises(self):
        q = np.zeros((8, 64), dtype=np.uint8)
        s = np.ones(7, dtype=np.float32)   # wrong length
        z = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="n_groups"):
            self.loader.load_from_arrays(q, s, z, original_shape=(512,))

    def test_non_2d_q_raises(self):
        q = np.zeros((512,), dtype=np.uint8)  # 1-D, not 2-D
        s = np.ones(8, dtype=np.float32)
        z = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            self.loader.load_from_arrays(q, s, z, original_shape=(512,))


# ── INT3RuntimeLoader — dequantize ────────────────────────────────────────────

class TestINT3Dequantize:
    def setup_method(self):
        self.loader = INT3RuntimeLoader(INT3RuntimeConfig())

    def test_output_shape_1d(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        out = self.loader.dequantize(lw)
        assert out.shape == (512,)

    def test_output_shape_2d(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(8, 64))
        out = self.loader.dequantize(lw)
        assert out.shape == (8, 64)

    def test_output_dtype_float32(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        out = self.loader.dequantize(lw)
        assert out.dtype == np.float32

    def test_stats_incremented(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        before = self.loader.stats.tensors_dequantized
        self.loader.dequantize(lw)
        assert self.loader.stats.tensors_dequantized == before + 1

    def test_weight_recovery_count(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        before_weights = self.loader.stats.total_weights_recovered
        self.loader.dequantize(lw)
        assert self.loader.stats.total_weights_recovered == before_weights + 512

    def test_all_zero_codes(self):
        """All codes = 0 → result should be scales[g]*0 + zeros[g] = zeros[g]."""
        n_groups, gs = 4, 64
        q = np.zeros((n_groups, gs), dtype=np.uint8)
        s = np.full(n_groups, 0.01, dtype=np.float32)
        z = np.full(n_groups, -0.5, dtype=np.float32)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(n_groups * gs,))
        out = self.loader.dequantize(lw)
        np.testing.assert_allclose(out, -0.5, atol=1e-5)

    def test_all_max_codes(self):
        """All codes = 7 → result should be scales[g]*7 + zeros[g]."""
        n_groups, gs = 4, 64
        q = np.full((n_groups, gs), 7, dtype=np.uint8)
        s = np.full(n_groups, 0.01, dtype=np.float32)
        z = np.zeros(n_groups, dtype=np.float32)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(n_groups * gs,))
        out = self.loader.dequantize(lw)
        np.testing.assert_allclose(out, 0.07, atol=1e-5)

    def test_dequant_formula(self):
        """Verify: w = code * scale + zero for each element."""
        q = np.array([[3, 5]], dtype=np.uint8)  # n_groups=1, gs=2
        s = np.array([0.02], dtype=np.float32)
        z = np.array([0.10], dtype=np.float32)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(2,))
        out = self.loader.dequantize(lw)
        expected = np.array([3 * 0.02 + 0.10, 5 * 0.02 + 0.10], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)


# ── INT3RuntimeLoader — dequantize_tiled ──────────────────────────────────────

class TestINT3DequantizeTiled:
    def setup_method(self):
        self.loader = INT3RuntimeLoader(INT3RuntimeConfig(group_size=64, tile_size=4))

    def test_tiles_concatenate_to_full(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        full = self.loader.dequantize(lw)
        # Reload to reset stats for tiled path
        lw2 = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        tiles = list(self.loader.dequantize_tiled(lw2))
        concatenated = np.concatenate(tiles)
        np.testing.assert_allclose(full, concatenated, atol=1e-6)

    def test_tile_count(self):
        """With tile_size=4 groups and 8 groups total → 2 tiles."""
        q, s, z = _make_int3_arrays(512, 64)  # 8 groups
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        tiles = list(self.loader.dequantize_tiled(lw))
        assert len(tiles) == 2

    def test_partial_last_tile(self):
        """With tile_size=4: 3 groups → 1 full tile (3 < 4 so one shot).
        With tile_size=2: 3 groups → 2 tiles (2 + 1)."""
        q, s, z = _make_int3_arrays(192, 64)  # 3 groups
        # Use tile_size=2 so we get partial last tile
        loader2 = INT3RuntimeLoader(INT3RuntimeConfig(group_size=64, tile_size=2))
        lw = loader2.load_from_arrays(q, s, z, original_shape=(192,))
        tiles = list(loader2.dequantize_tiled(lw))
        assert len(tiles) == 2
        assert tiles[0].shape[0] == 128  # 2 groups × 64
        assert tiles[1].shape[0] == 64   # 1 group × 64

    def test_all_tiles_float32(self):
        q, s, z = _make_int3_arrays(512, 64)
        lw = self.loader.load_from_arrays(q, s, z, original_shape=(512,))
        for tile in self.loader.dequantize_tiled(lw):
            assert tile.dtype == np.float32


# ── INT3RuntimeLoader — load_layer (on-disk) ──────────────────────────────────

class TestINT3LoadLayer:
    def setup_method(self):
        self.loader = INT3RuntimeLoader(INT3RuntimeConfig())

    def test_load_and_dequant_roundtrip(self, tmp_path):
        q, s, z = _make_int3_arrays(512, 64)
        layer_name = "model_layers_0_attention_q_proj"
        original_shape = (8, 64)
        np.save(tmp_path / f"{layer_name}__q3.npy", q)
        np.save(tmp_path / f"{layer_name}__s3.npy", s)
        np.save(tmp_path / f"{layer_name}__z3.npy", z)
        np.save(tmp_path / f"{layer_name}__shape.npy", np.array(original_shape))
        lw = self.loader.load_layer(str(tmp_path), layer_name)
        assert lw.original_shape == original_shape
        out = self.loader.dequantize(lw)
        assert out.shape == original_shape

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="q3"):
            self.loader.load_layer(str(tmp_path), "nonexistent_layer")

    def test_load_layer_group_size_inferred(self, tmp_path):
        q, s, z = _make_int3_arrays(512, 128)  # group_size=128
        layer_name = "layer_with_gs128"
        original_shape = (512,)
        np.save(tmp_path / f"{layer_name}__q3.npy", q)
        np.save(tmp_path / f"{layer_name}__s3.npy", s)
        np.save(tmp_path / f"{layer_name}__z3.npy", z)
        np.save(tmp_path / f"{layer_name}__shape.npy", np.array(original_shape))
        lw = self.loader.load_layer(str(tmp_path), layer_name)
        assert lw.group_size == 128


# ── INT3LoaderStats ────────────────────────────────────────────────────────────

class TestINT3LoaderStats:
    def test_initial_all_zero(self):
        stats = INT3LoaderStats()
        assert stats.layers_loaded == 0
        assert stats.tensors_dequantized == 0
        assert stats.total_weights_recovered == 0

    def test_repr_contains_values(self):
        stats = INT3LoaderStats(layers_loaded=3, tensors_dequantized=6, total_weights_recovered=1024)
        r = repr(stats)
        assert "3" in r
        assert "6" in r
        # The repr may format 1024 as '1024' or '1,024' depending on locale
        assert "1024" in r.replace(",", "")


# ── INT3RuntimeLoader.__repr__ ─────────────────────────────────────────────────

class TestINT3LoaderRepr:
    def test_repr_not_empty(self):
        loader = INT3RuntimeLoader(INT3RuntimeConfig())
        r = repr(loader)
        assert "INT3RuntimeLoader" in r
        assert "gs=" in r


# ── Round-trip accuracy ────────────────────────────────────────────────────────

class TestINT3RoundTripAccuracy:
    """End-to-end accuracy test: encode FP32 weights → INT3 codes → dequantize → SNR."""

    @pytest.mark.parametrize("n_weights,group_size", [
        (512,  64),
        (1024, 64),
        (4096, 128),
    ])
    def test_asymmetric_snr_acceptable(self, n_weights, group_size):
        """
        Simulate asymmetric INT3 quantization (like the loader's expected usage).
        The dequantize formula is w = code * scale + zero.
        After quantizing ideal Gaussian weights, SNR should be > 10 dB.
        (Theoretical max for asymmetric INT3 on Gaussian input is ~13–14 dB;
        the threshold is conservative to be robust to random variation.)
        """
        rng = np.random.default_rng(0)
        weight = rng.standard_normal(n_weights).astype(np.float32) * 0.02
        n_groups = n_weights // group_size
        wr = weight.reshape(n_groups, group_size)
        w_min = wr.min(axis=1)
        w_max = wr.max(axis=1)
        w_range = w_max - w_min
        w_range = np.where(w_range == 0.0, 1.0, w_range)
        scale = (w_range / 7.0).astype(np.float32)
        zero = w_min.astype(np.float32)
        # Quantize
        codes = np.round((wr - w_min[:, None]) / w_range[:, None] * 7.0)
        codes = np.clip(codes, 0, 7).astype(np.uint8)

        loader = INT3RuntimeLoader(INT3RuntimeConfig(group_size=group_size))
        lw = loader.load_from_arrays(codes, scale, zero, original_shape=(n_weights,))
        out = loader.dequantize(lw)
        snr = _snr_db(weight, out)
        assert snr > 10.0, f"SNR too low for INT3 round-trip: {snr:.1f} dB"

    def test_single_group_roundtrip(self):
        """Minimal edge case: exactly 1 group."""
        rng = np.random.default_rng(7)
        group_size = 64
        weight = rng.standard_normal(group_size).astype(np.float32) * 0.01
        w_min = weight.min()
        w_max = weight.max()
        w_range = w_max - w_min or 1.0
        scale = np.array([w_range / 7.0], dtype=np.float32)
        zero  = np.array([w_min], dtype=np.float32)
        codes = np.clip(np.round((weight - w_min) / w_range * 7.0), 0, 7).astype(np.uint8)
        codes_2d = codes.reshape(1, group_size)
        loader = INT3RuntimeLoader(INT3RuntimeConfig(group_size=group_size))
        lw = loader.load_from_arrays(codes_2d, scale, zero, original_shape=(group_size,))
        out = loader.dequantize(lw)
        assert out.shape == (group_size,)
        snr = _snr_db(weight, out)
        assert snr > 10.0

    def test_tiled_matches_full(self):
        """Tiled dequantization must produce the same output as full dequantization."""
        rng = np.random.default_rng(33)
        n_groups, gs = 16, 64
        q = rng.integers(0, 8, size=(n_groups, gs), dtype=np.uint8)
        s = rng.uniform(0.001, 0.05, size=(n_groups,)).astype(np.float32)
        z = rng.uniform(-0.05, 0.05, size=(n_groups,)).astype(np.float32)
        cfg = INT3RuntimeConfig(group_size=gs, tile_size=3)  # odd tile
        loader_full  = INT3RuntimeLoader(cfg)
        loader_tiled = INT3RuntimeLoader(cfg)
        lw_full  = loader_full.load_from_arrays(q, s, z, original_shape=(n_groups * gs,))
        lw_tiled = loader_tiled.load_from_arrays(q, s, z, original_shape=(n_groups * gs,))
        full = loader_full.dequantize(lw_full)
        tiled = np.concatenate(list(loader_tiled.dequantize_tiled(lw_tiled)))
        np.testing.assert_allclose(full, tiled, atol=1e-6)
