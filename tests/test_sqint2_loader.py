"""tests/test_sqint2_loader.py — W103.4a save/load round-trip + loader wiring.

Coverage:
  - save_sqint2_layer / load_sqint2_layer round-trip (Stage 1+2 only)
  - round-trip with W103.2 residual (rank-16 SVD + sparse 1%)
  - meta header layout, version handling
  - error paths: missing files, partial residual, partial sparse, bad shapes
  - SQINT2_SUFFIXES enumeration completeness
  - tensor-key discovery picks up SQINT2 layers via _collect_tensor_keys
  - compressed_loader._dequantize_npy_dir reconstructs SQINT2 to fp32 weight
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.quant.sqint2 import (
    SQINT2_FORMAT_VERSION,
    SQINT2_SUFFIXES,
    SQINT2Config,
    SQINT2Layer,
    compress_weight,
    decompress_weight,
    load_sqint2_layer,
    save_sqint2_layer,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _gaussian_weight(out_features=64, in_features=128, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((out_features, in_features)).astype(np.float32) * sigma


def _layers_close(a: SQINT2Layer, b: SQINT2Layer) -> None:
    """Strict structural+value equality between two SQINT2Layers."""
    assert a.in_features == b.in_features
    assert a.out_features == b.out_features
    assert a.cfg == b.cfg
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.scales, b.scales)
    np.testing.assert_array_equal(a.zero_points, b.zero_points)

    if a.residual_L is None:
        assert b.residual_L is None
    else:
        np.testing.assert_array_equal(a.residual_L, b.residual_L)
        np.testing.assert_array_equal(a.residual_R, b.residual_R)

    if a.sparse_rows is None:
        assert b.sparse_rows is None
    else:
        np.testing.assert_array_equal(a.sparse_rows, b.sparse_rows)
        np.testing.assert_array_equal(a.sparse_cols, b.sparse_cols)
        np.testing.assert_array_equal(a.sparse_vals, b.sparse_vals)


# ── round-trip: Stage 1+2 only ──────────────────────────────────────────────


class TestRoundTripStage12:
    def test_basic_round_trip(self, tmp_path):
        W = _gaussian_weight()
        layer = compress_weight(W, SQINT2Config(group_size=32))
        save_sqint2_layer(layer, tmp_path, "w")
        loaded = load_sqint2_layer(tmp_path, "w")
        _layers_close(layer, loaded)

    def test_decompress_matches_after_save_load(self, tmp_path):
        W = _gaussian_weight(out_features=96, in_features=160)
        layer = compress_weight(W, SQINT2Config(group_size=32, refine_iters=2))
        save_sqint2_layer(layer, tmp_path, "tensor.0")
        loaded = load_sqint2_layer(tmp_path, "tensor.0")
        np.testing.assert_array_equal(
            decompress_weight(layer), decompress_weight(loaded)
        )

    def test_residual_files_absent_when_rank_zero(self, tmp_path):
        layer = compress_weight(_gaussian_weight(), SQINT2Config(residual_rank=0))
        save_sqint2_layer(layer, tmp_path, "w")
        assert not (tmp_path / "w__sqint2_L.npy").exists()
        assert not (tmp_path / "w__sqint2_R.npy").exists()
        assert not (tmp_path / "w__sqint2_srows.npy").exists()

    def test_in_features_not_multiple_of_group_size(self, tmp_path):
        # in_features = 130 (not a multiple of 32) → padding to 160
        W = _gaussian_weight(out_features=64, in_features=130)
        layer = compress_weight(W, SQINT2Config(group_size=32))
        save_sqint2_layer(layer, tmp_path, "padded")
        loaded = load_sqint2_layer(tmp_path, "padded")
        assert loaded.in_features == 130
        np.testing.assert_array_equal(
            decompress_weight(layer), decompress_weight(loaded)
        )


# ── round-trip: full Stage 3 (residual + sparse) ────────────────────────────


class TestRoundTripStage3:
    def test_lowrank_residual_round_trip(self, tmp_path):
        W = _gaussian_weight(out_features=128, in_features=256)
        cfg = SQINT2Config(
            group_size=32,
            residual_rank=16,
            residual_factor_dtype="fp16",
            sparse_frac=0.0,
        )
        layer = compress_weight(W, cfg)
        assert layer.residual_L is not None and layer.residual_R is not None
        save_sqint2_layer(layer, tmp_path, "lowrank")
        loaded = load_sqint2_layer(tmp_path, "lowrank")
        _layers_close(layer, loaded)
        assert loaded.residual_L.dtype == np.float16
        assert loaded.residual_R.dtype == np.float16

    def test_lowrank_fp32_round_trip(self, tmp_path):
        cfg = SQINT2Config(
            residual_rank=8, residual_factor_dtype="fp32", sparse_frac=0.0
        )
        layer = compress_weight(_gaussian_weight(), cfg)
        save_sqint2_layer(layer, tmp_path, "lr32")
        loaded = load_sqint2_layer(tmp_path, "lr32")
        assert loaded.residual_L.dtype == np.float32
        assert loaded.cfg.residual_factor_dtype == "fp32"

    def test_full_stage3_round_trip(self, tmp_path):
        W = _gaussian_weight(out_features=128, in_features=256)
        cfg = SQINT2Config(
            group_size=32,
            residual_rank=16,
            residual_factor_dtype="fp16",
            sparse_frac=0.01,
        )
        layer = compress_weight(W, cfg)
        assert layer.sparse_rows is not None
        assert layer.sparse_rows.size > 0
        save_sqint2_layer(layer, tmp_path, "full")
        loaded = load_sqint2_layer(tmp_path, "full")
        _layers_close(layer, loaded)
        np.testing.assert_array_equal(
            decompress_weight(layer), decompress_weight(loaded)
        )


# ── meta header ─────────────────────────────────────────────────────────────


class TestMetaHeader:
    def test_meta_shape_and_version(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        meta = np.load(tmp_path / "w__sqint2_meta.npy")
        assert meta.shape == (16,)
        assert meta.dtype == np.float64
        assert meta[0] == SQINT2_FORMAT_VERSION

    def test_meta_records_all_cfg_fields(self, tmp_path):
        cfg = SQINT2Config(
            group_size=64,
            seed=1234,
            refine_iters=3,
            rotate_left=True,
            rotate_right=False,
            residual_rank=8,
            residual_factor_dtype="fp32",
            sparse_frac=0.05,
        )
        # with rotate_right=False, group_size must still divide in_features
        layer = compress_weight(_gaussian_weight(64, 128), cfg)
        save_sqint2_layer(layer, tmp_path, "cfg")
        loaded = load_sqint2_layer(tmp_path, "cfg")
        assert loaded.cfg == cfg

    def test_future_version_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        meta = np.load(tmp_path / "w__sqint2_meta.npy")
        meta[0] = SQINT2_FORMAT_VERSION + 1.0
        np.save(tmp_path / "w__sqint2_meta.npy", meta)
        with pytest.raises(ValueError, match="newer than this build"):
            load_sqint2_layer(tmp_path, "w")

    def test_invalid_version_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        meta = np.load(tmp_path / "w__sqint2_meta.npy")
        meta[0] = 0.5
        np.save(tmp_path / "w__sqint2_meta.npy", meta)
        with pytest.raises(ValueError, match="invalid"):
            load_sqint2_layer(tmp_path, "w")

    def test_unknown_dtype_code_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        meta = np.load(tmp_path / "w__sqint2_meta.npy")
        meta[9] = 99.0
        np.save(tmp_path / "w__sqint2_meta.npy", meta)
        with pytest.raises(ValueError, match="dtype_code"):
            load_sqint2_layer(tmp_path, "w")

    def test_meta_wrong_shape_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        np.save(tmp_path / "w__sqint2_meta.npy", np.zeros(8, dtype=np.float64))
        with pytest.raises(ValueError, match="meta has unexpected shape"):
            load_sqint2_layer(tmp_path, "w")


# ── error paths ─────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_missing_idx_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="idx"):
            load_sqint2_layer(tmp_path, "nope")

    def test_missing_meta_file(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        (tmp_path / "w__sqint2_meta.npy").unlink()
        with pytest.raises(FileNotFoundError, match="meta"):
            load_sqint2_layer(tmp_path, "w")

    def test_partial_residual_rejected(self, tmp_path):
        cfg = SQINT2Config(residual_rank=16, sparse_frac=0.0)
        layer = compress_weight(_gaussian_weight(), cfg)
        save_sqint2_layer(layer, tmp_path, "w")
        (tmp_path / "w__sqint2_R.npy").unlink()
        with pytest.raises(ValueError, match="partially present"):
            load_sqint2_layer(tmp_path, "w")

    def test_partial_sparse_rejected(self, tmp_path):
        cfg = SQINT2Config(residual_rank=16, sparse_frac=0.01)
        layer = compress_weight(_gaussian_weight(), cfg)
        save_sqint2_layer(layer, tmp_path, "w")
        (tmp_path / "w__sqint2_svals.npy").unlink()
        with pytest.raises(ValueError, match="sparse triplet is incomplete"):
            load_sqint2_layer(tmp_path, "w")

    def test_safe_key_with_path_separator_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        with pytest.raises(ValueError, match="path separators"):
            save_sqint2_layer(layer, tmp_path, "evil/key")

    def test_save_to_missing_dir(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        with pytest.raises(FileNotFoundError):
            save_sqint2_layer(layer, tmp_path / "does_not_exist", "w")

    def test_idx_shape_mismatch_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight(64, 128))
        save_sqint2_layer(layer, tmp_path, "w")
        bogus = np.zeros((64, 16), dtype=np.uint8)
        np.save(tmp_path / "w__sqint2_idx.npy", bogus)
        with pytest.raises(ValueError, match="indices shape"):
            load_sqint2_layer(tmp_path, "w")

    def test_idx_wrong_dtype_rejected(self, tmp_path):
        layer = compress_weight(_gaussian_weight())
        save_sqint2_layer(layer, tmp_path, "w")
        wrong = np.load(tmp_path / "w__sqint2_idx.npy").astype(np.int32)
        np.save(tmp_path / "w__sqint2_idx.npy", wrong)
        with pytest.raises(ValueError, match="indices must be uint8"):
            load_sqint2_layer(tmp_path, "w")


# ── suffix enumeration ─────────────────────────────────────────────────────


class TestSuffixes:
    def test_all_suffixes_unique(self):
        assert len(set(SQINT2_SUFFIXES)) == len(SQINT2_SUFFIXES)

    def test_all_suffixes_have_expected_prefix(self):
        for suf in SQINT2_SUFFIXES:
            assert suf.startswith("__sqint2_")
            assert suf.endswith(".npy")

    def test_save_writes_only_known_suffixes(self, tmp_path):
        cfg = SQINT2Config(residual_rank=16, sparse_frac=0.01)
        layer = compress_weight(_gaussian_weight(), cfg)
        save_sqint2_layer(layer, tmp_path, "w")
        for f in tmp_path.iterdir():
            stripped = f.name[len("w") :]
            assert stripped in SQINT2_SUFFIXES, f"unknown suffix: {stripped}"


# ── compressed_loader integration ───────────────────────────────────────────


class TestCompressedLoaderIntegration:
    def test_dequantize_npy_dir_reconstructs_sqint2(self, tmp_path):
        from squish.quant.compressed_loader import _dequantize_npy_dir

        W = _gaussian_weight(out_features=64, in_features=128)
        layer = compress_weight(W, SQINT2Config(group_size=32))
        save_sqint2_layer(layer, tmp_path, "model.layer.weight")
        # Also write the original-shape sidecar that compressed_loader reads
        np.save(
            tmp_path / "model.layer.weight__shape.npy",
            np.array(W.shape, dtype=np.int64),
        )

        recon = _dequantize_npy_dir(tmp_path, "model.layer.weight")
        assert recon.shape == W.shape
        assert recon.dtype == np.float32
        np.testing.assert_array_equal(recon, decompress_weight(layer))

    def test_dequantize_npy_dir_without_shape_sidecar(self, tmp_path):
        from squish.quant.compressed_loader import _dequantize_npy_dir

        layer = compress_weight(_gaussian_weight(48, 96), SQINT2Config(group_size=32))
        save_sqint2_layer(layer, tmp_path, "tensor")
        recon = _dequantize_npy_dir(tmp_path, "tensor")
        assert recon.shape == (48, 96)

    def test_collect_tensor_keys_finds_sqint2(self, tmp_path):
        from squish.quant.compressed_loader import _collect_tensor_keys

        layer = compress_weight(_gaussian_weight(), SQINT2Config())
        save_sqint2_layer(layer, tmp_path, "w0")
        save_sqint2_layer(layer, tmp_path, "w1")
        keys = _collect_tensor_keys(tmp_path)
        assert "w0" in keys
        assert "w1" in keys
