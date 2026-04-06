"""tests/test_squash_wave38.py — Wave 38: squish.quant.aqlm module.

Test taxonomy:
  Unit       — AQLMConfig / AQLMCodebook / AQLMLayer construction; no I/O,
               deterministic.
  Integration — aqlm_dequantize correctness against known-good reference
                values, shape/dtype contracts, regression snapshot, and
                compressed_loader end-to-end round-trip via a synthetic
                npy-dir fixture.

ML component test checklist (per CLAUDE.md):
  ✓ Shape/dtype contract test
  ✓ Numerical correctness test (known-good reference output)
  ✓ Regression snapshot test (stored tolerance vector)
  ✓ Failure case tests (bad inputs raise ValueError)

Covered:
  - AQLMConfig validates field constraints
  - AQLMCodebook validates vectors ndim
  - AQLMLayer validates group_size divisibility
  - aqlm_dequantize: single codebook, identity correctness
  - aqlm_dequantize: multi-codebook additive summation
  - aqlm_dequantize: scale multiplier applied correctly
  - aqlm_dequantize: output shape (out_features, in_features)
  - aqlm_dequantize: output dtype is float32
  - aqlm_dequantize: regression snapshot for a fixed seed
  - aqlm_dequantize: raises ValueError on bad indices ndim
  - aqlm_dequantize: raises ValueError on codebook size mismatch
  - aqlm_dequantize: raises ValueError on K mismatch in indices
  - compressed_loader round-trip: saves and loads AQLM npy-dir; reconstructed
    weights match reference within tolerance
"""

from __future__ import annotations

import struct
import numpy as np
import pytest

from squish.quant.aqlm import (
    AQLMCodebook,
    AQLMConfig,
    AQLMLayer,
    aqlm_dequantize,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_layer(
    out_features: int = 4,
    in_features: int = 8,
    n_codebooks: int = 2,
    codebook_size: int = 4,
    group_size: int = 4,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[AQLMLayer, np.ndarray]:
    """Create a synthetic AQLMLayer with known reference weights.

    Returns:
        (layer, reference_weights) where reference_weights is the float32
        matrix that aqlm_dequantize should return.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cfg = AQLMConfig(
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        group_size=group_size,
    )
    n_groups = in_features // group_size
    layer = AQLMLayer(out_features, in_features, cfg)
    layer.scale = scale

    # Assign random indices
    layer.indices = rng.integers(0, codebook_size, size=(out_features, n_groups, n_codebooks)).astype(np.int32)

    # Assign random codebook vectors and compute reference by hand
    reference = np.zeros((out_features, n_groups, group_size), dtype=np.float32)
    for k in range(n_codebooks):
        cb = rng.standard_normal((codebook_size, group_size)).astype(np.float32)
        layer.codebooks[k].vectors = cb
        reference += cb[layer.indices[:, :, k]]

    reference *= scale
    return layer, reference.reshape(out_features, in_features)


# ── AQLMConfig unit tests ──────────────────────────────────────────────────────


class TestAQLMConfig:
    def test_valid_construction(self):
        cfg = AQLMConfig(n_codebooks=2, codebook_size=256, group_size=8)
        assert cfg.n_codebooks == 2
        assert cfg.codebook_size == 256
        assert cfg.group_size == 8

    def test_n_codebooks_zero_raises(self):
        with pytest.raises(ValueError, match="n_codebooks"):
            AQLMConfig(n_codebooks=0, codebook_size=256, group_size=8)

    def test_codebook_size_one_raises(self):
        with pytest.raises(ValueError, match="codebook_size"):
            AQLMConfig(n_codebooks=1, codebook_size=1, group_size=8)

    def test_group_size_zero_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            AQLMConfig(n_codebooks=1, codebook_size=256, group_size=0)


# ── AQLMCodebook unit tests ────────────────────────────────────────────────────


class TestAQLMCodebook:
    def test_default_construction(self):
        cb = AQLMCodebook()
        assert cb.vectors.size == 0

    def test_valid_2d_vectors(self):
        v = np.ones((4, 8), dtype=np.float32)
        cb = AQLMCodebook(vectors=v)
        assert cb.vectors.shape == (4, 8)

    def test_1d_vectors_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            AQLMCodebook(vectors=np.ones(8, dtype=np.float32))


# ── AQLMLayer unit tests ───────────────────────────────────────────────────────


class TestAQLMLayer:
    def test_valid_construction(self):
        cfg = AQLMConfig(n_codebooks=2, codebook_size=64, group_size=4)
        layer = AQLMLayer(8, 16, cfg)
        assert layer.out_features == 8
        assert layer.in_features == 16
        assert layer.n_groups == 4
        assert len(layer.codebooks) == 2

    def test_default_scale_is_one(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        layer = AQLMLayer(4, 8, cfg)
        assert layer.scale == 1.0

    def test_default_indices_shape(self):
        cfg = AQLMConfig(n_codebooks=2, codebook_size=4, group_size=4)
        layer = AQLMLayer(4, 8, cfg)
        # (out_features, n_groups, n_codebooks)
        assert layer.indices.shape == (4, 2, 2)

    def test_in_features_not_divisible_raises(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        with pytest.raises(ValueError, match="divisible"):
            AQLMLayer(4, 9, cfg)  # 9 % 4 != 0


# ── aqlm_dequantize shape / dtype contracts ────────────────────────────────────


class TestAqlmDequantizeShapeDtype:
    """Shape and dtype contract — runs for several dimension combos."""

    @pytest.mark.parametrize(
        "out_feat,in_feat,n_cb,cb_size,gs",
        [
            (4, 8, 1, 4, 4),   # minimal single codebook
            (4, 8, 2, 4, 4),   # two codebooks
            (8, 16, 4, 8, 4),  # four codebooks, larger
            (1, 4, 1, 2, 4),   # single row
            (16, 32, 2, 256, 8),  # realistic small model
        ],
    )
    def test_output_shape(self, out_feat, in_feat, n_cb, cb_size, gs):
        layer, _ = _make_layer(
            out_features=out_feat,
            in_features=in_feat,
            n_codebooks=n_cb,
            codebook_size=cb_size,
            group_size=gs,
        )
        result = aqlm_dequantize(layer)
        assert result.shape == (out_feat, in_feat)

    def test_output_dtype_is_float32(self):
        layer, _ = _make_layer()
        result = aqlm_dequantize(layer)
        assert result.dtype == np.float32


# ── aqlm_dequantize numerical correctness ─────────────────────────────────────


class TestAqlmDequantizeNumerical:
    """Correctness against the hand-computed reference built in _make_layer."""

    def test_single_codebook_matches_reference(self):
        layer, ref = _make_layer(n_codebooks=1)
        result = aqlm_dequantize(layer)
        np.testing.assert_allclose(result, ref, rtol=0, atol=1e-6)

    def test_two_codebooks_matches_reference(self):
        layer, ref = _make_layer(n_codebooks=2)
        result = aqlm_dequantize(layer)
        np.testing.assert_allclose(result, ref, rtol=0, atol=1e-6)

    def test_four_codebooks_matches_reference(self):
        layer, ref = _make_layer(n_codebooks=4, out_features=8, in_features=16)
        result = aqlm_dequantize(layer)
        np.testing.assert_allclose(result, ref, rtol=0, atol=1e-6)

    def test_scale_applied(self):
        layer_unit, ref_unit = _make_layer(scale=1.0)
        layer_scaled, _ = _make_layer(scale=2.0)
        # Same indices, same codebooks — set identical
        layer_scaled.indices = layer_unit.indices.copy()
        for k in range(layer_unit.cfg.n_codebooks):
            layer_scaled.codebooks[k].vectors = layer_unit.codebooks[k].vectors.copy()
        result_scaled = aqlm_dequantize(layer_scaled)
        np.testing.assert_allclose(result_scaled, ref_unit * 2.0, rtol=0, atol=1e-6)

    def test_all_same_index_uses_same_codebook_vector(self):
        """When all indices point to row 0, result should be K * CB[0] * scale."""
        cfg = AQLMConfig(n_codebooks=2, codebook_size=4, group_size=4)
        layer = AQLMLayer(2, 8, cfg)
        layer.scale = 0.5
        layer.indices = np.zeros((2, 2, 2), dtype=np.int32)  # all zeros

        for k in range(2):
            # Codebook k: row 0 = [k+1, k+1, k+1, k+1]
            cb = np.zeros((4, 4), dtype=np.float32)
            cb[0, :] = float(k + 1)
            layer.codebooks[k].vectors = cb

        result = aqlm_dequantize(layer)
        # Each element should be (1 + 2) * 0.5 = 1.5
        expected = np.full((2, 8), 1.5, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_nan_not_present_in_valid_input(self):
        """Valid codebook vectors must not produce NaN/Inf in output."""
        layer, _ = _make_layer(n_codebooks=3, out_features=8, in_features=16)
        result = aqlm_dequantize(layer)
        assert not np.isnan(result).any(), "NaN in dequantized output"
        assert not np.isinf(result).any(), "Inf in dequantized output"


# ── aqlm_dequantize regression snapshot ───────────────────────────────────────


class TestAqlmDequantizeRegression:
    """Regression: exact numeric match for a fixed seed — catches silent changes."""

    # Reference values computed once from the implementation and frozen here.
    # Seed=42, out=4, in=8, K=2, cb_size=4, gs=4, scale=1.0
    # Row 0 of the expected result (first 4 elements):
    _EXPECTED_ROW0_FIRST4 = np.array(
        [-0.13074861, -1.69320035,  0.05491680,  1.42838478], dtype=np.float32
    )

    def test_regression_row0_first4(self):
        layer, ref = _make_layer(
            out_features=4, in_features=8, n_codebooks=2,
            codebook_size=4, group_size=4, scale=1.0,
            rng=np.random.default_rng(42),
        )
        result = aqlm_dequantize(layer)
        np.testing.assert_allclose(
            result[0, :4],
            self._EXPECTED_ROW0_FIRST4,
            rtol=0,
            atol=1e-5,
        )


# ── aqlm_dequantize failure cases ─────────────────────────────────────────────


class TestAqlmDequantizeFailures:
    """Failure case tests — bad inputs must raise ValueError."""

    def test_1d_indices_raises(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        layer = AQLMLayer(4, 8, cfg)
        layer.indices = np.zeros(8, dtype=np.int32)  # wrong ndim
        with pytest.raises(ValueError, match="3-D"):
            aqlm_dequantize(layer)

    def test_k_mismatch_between_cfg_and_indices_raises(self):
        cfg = AQLMConfig(n_codebooks=2, codebook_size=4, group_size=4)
        layer = AQLMLayer(4, 8, cfg)
        # Override indices with wrong K dimension
        layer.indices = np.zeros((4, 2, 3), dtype=np.int32)  # K=3 but cfg says 2
        # Also patch codebooks to avoid earlier raise
        with pytest.raises(ValueError, match="n_codebooks"):
            aqlm_dequantize(layer)

    def test_wrong_codebook_vector_shape_raises(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        layer = AQLMLayer(4, 8, cfg)
        # Wrong group_size in vectors (should be 4, set to 8)
        layer.codebooks[0].vectors = np.zeros((4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            aqlm_dequantize(layer)


# ── compressed_loader round-trip integration ──────────────────────────────────


class TestCompressedLoaderAqlmRoundTrip:
    """Integration: compressed_loader._dequantize_npy_dir reads AQLM npy-dir correctly.

    Saves the AQLM artifacts in the squish npy-dir layout and verifies that
    the loader reconstructs the expected float32 weight matrix.
    """

    def _write_aqlm_npy_dir(
        self,
        tmp_path,
        layer: AQLMLayer,
        stem: str = "weight",
    ) -> "Path":
        """Persist layer in squish AQLM npy-dir layout."""
        from pathlib import Path

        tensor_dir = tmp_path / "tensor"
        tensor_dir.mkdir()

        out_features = layer.out_features
        in_features = layer.in_features
        cfg = layer.cfg
        K = cfg.n_codebooks
        n_groups = layer.n_groups

        # __aqlm_idx.npy
        np.save(tensor_dir / f"{stem}__aqlm_idx.npy", layer.indices.astype(np.int16))

        # __aqlm_cb.npy — flat layout: [scale, cb_size, gs, *vectors]
        cb_vectors = np.stack(
            [layer.codebooks[k].vectors for k in range(K)], axis=0
        )  # (K, codebook_size, group_size)
        flat_cb = np.concatenate([
            np.array([layer.scale, float(cfg.codebook_size), float(cfg.group_size)],
                     dtype=np.float32),
            cb_vectors.astype(np.float32).ravel(),
        ])
        np.save(tensor_dir / f"{stem}__aqlm_cb.npy", flat_cb)

        # __shape.npy
        np.save(tensor_dir / f"{stem}__shape.npy",
                np.array([out_features, in_features], dtype=np.int64))

        return tensor_dir

    def test_round_trip_matches_dequantize(self, tmp_path):
        layer, reference = _make_layer(
            out_features=4, in_features=8, n_codebooks=2,
            codebook_size=4, group_size=4, scale=0.5,
            rng=np.random.default_rng(7),
        )
        tensor_dir = self._write_aqlm_npy_dir(tmp_path, layer, stem="weight")

        from squish.quant.compressed_loader import _dequantize_npy_dir

        result = _dequantize_npy_dir(tensor_dir, "weight")
        np.testing.assert_allclose(result, reference, rtol=0, atol=1e-5)

    def test_round_trip_output_shape(self, tmp_path):
        layer, _ = _make_layer(out_features=6, in_features=12, n_codebooks=1,
                               codebook_size=4, group_size=4,
                               rng=np.random.default_rng(17))
        tensor_dir = self._write_aqlm_npy_dir(tmp_path, layer, stem="fc")

        from squish.quant.compressed_loader import _dequantize_npy_dir

        result = _dequantize_npy_dir(tensor_dir, "fc")
        assert result.shape == (6, 12)
        assert result.dtype == np.float32

    def test_round_trip_no_nan(self, tmp_path):
        layer, _ = _make_layer(
            out_features=8, in_features=16, n_codebooks=2,
            codebook_size=8, group_size=4, scale=1.0,
            rng=np.random.default_rng(99),
        )
        tensor_dir = self._write_aqlm_npy_dir(tmp_path, layer)

        from squish.quant.compressed_loader import _dequantize_npy_dir

        result = _dequantize_npy_dir(tensor_dir, "weight")
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
