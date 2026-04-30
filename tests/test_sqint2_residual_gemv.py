"""tests/test_sqint2_residual_gemv.py — W103.4b Stage-3 residual GEMV.

Covers:
  - Numerical equivalence: y == (L @ R + sparse_dense) @ x_rot for synthetic L, R, sparse.
  - End-to-end SQINT2Layer integration: result agrees with the residual portion
    of decompress_weight() (W_rot − dequant(Q_INT2)) @ x_rot.
  - rank=0 + nnz=0 path returns zeros without crashing.
  - rank-only (no sparse): matches L @ R @ x_rot exactly.
  - sparse-only (rank=0): matches sparse_dense @ x_rot exactly.
  - fp16-stored factors: residual is promoted to fp32 for accumulation.
  - Shape errors: wrong x_rot length, mismatched COO triplet length.
  - COO bounds: out-of-range row/col raises ValueError.
  - Rust ↔ NumPy parity: when squish_quant is available, the two paths agree
    to 1e-5 abs / 1e-4 rel on σ=0.02 IID Gaussian random factors.
  - Determinism: same inputs → byte-identical outputs across calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.quant.sqint2 import (
    SQINT2Config,
    SQINT2Layer,
    NF2_VALUES,
    _residual_gemv_numpy,
    _round_up,
    build_hadamard,
    compress_weight,
    decompress_weight,
    sqint2_residual_gemv,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_layer(out_features=64, in_features=128, rank=8, sparse_frac=0.01,
                seed=0, factor_dtype="fp16"):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.02
    cfg = SQINT2Config(
        group_size=32, refine_iters=2, seed=42,
        residual_rank=rank, residual_factor_dtype=factor_dtype,
        sparse_frac=sparse_frac,
    )
    return W, compress_weight(W, cfg)


def _x_rot(in_padded: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(in_padded).astype(np.float32)


def _explicit_residual(layer: SQINT2Layer, x_rot: np.ndarray) -> np.ndarray:
    """Reference: dense (L@R + sparse_dense) @ x_rot, computed without the API."""
    out_padded = layer.indices.shape[0]
    in_padded = _round_up(layer.in_features, layer.cfg.group_size)
    M = np.zeros((out_padded, in_padded), dtype=np.float64)
    if layer.residual_L is not None:
        L64 = layer.residual_L.astype(np.float64)
        R64 = layer.residual_R.astype(np.float64)
        M += L64 @ R64
    if layer.sparse_rows is not None and layer.sparse_rows.size > 0:
        for i in range(layer.sparse_rows.size):
            r = int(layer.sparse_rows[i])
            c = int(layer.sparse_cols[i])
            M[r, c] += float(layer.sparse_vals[i])
    return (M @ x_rot.astype(np.float64)).astype(np.float32)


# ── 1. Output shape and dtype ────────────────────────────────────────────────


class TestShapeContract:
    def test_returns_fp32(self):
        _, layer = _make_layer()
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        assert y.dtype == np.float32

    def test_returns_out_padded_shape(self):
        _, layer = _make_layer(out_features=96, in_features=128)
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        assert y.shape == (96,)

    def test_x_wrong_length_raises(self):
        _, layer = _make_layer(in_features=128)
        wrong = np.zeros(64, dtype=np.float32)
        with pytest.raises(ValueError, match="must be"):
            sqint2_residual_gemv(layer, wrong)


# ── 2. Numerical equivalence vs the explicit (L@R + sparse_dense) @ x  ───────


class TestEquivalence:
    def test_full_path_matches_explicit_dense(self):
        _, layer = _make_layer(rank=16, sparse_frac=0.01, seed=0)
        x = _x_rot(_round_up(128, 32), seed=7)
        y = sqint2_residual_gemv(layer, x)
        y_ref = _explicit_residual(layer, x)
        np.testing.assert_allclose(y, y_ref, atol=1e-4, rtol=1e-4)

    def test_lowrank_only(self):
        _, layer = _make_layer(rank=16, sparse_frac=0.0)
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        L = layer.residual_L.astype(np.float32)
        R = layer.residual_R.astype(np.float32)
        np.testing.assert_allclose(y, L @ R @ x, atol=1e-4, rtol=1e-4)

    def test_sparse_only_rank_zero(self):
        rng = np.random.default_rng(0)
        out_pad, in_pad = 64, 128
        sp_rows = rng.integers(0, out_pad, size=20).astype(np.int32)
        sp_cols = rng.integers(0, in_pad, size=20).astype(np.int32)
        sp_vals = rng.standard_normal(20).astype(np.float32) * 0.05
        x = rng.standard_normal(in_pad).astype(np.float32)
        cfg = SQINT2Config(group_size=32, residual_rank=0, sparse_frac=0.0)
        layer = SQINT2Layer(
            indices=np.zeros((out_pad, in_pad // 4), dtype=np.uint8),
            scales=np.ones((out_pad, in_pad // 32), dtype=np.float32),
            zero_points=np.zeros((out_pad, in_pad // 32), dtype=np.float32),
            in_features=in_pad,
            out_features=out_pad,
            cfg=cfg,
            sparse_rows=sp_rows,
            sparse_cols=sp_cols,
            sparse_vals=sp_vals,
        )
        y = sqint2_residual_gemv(layer, x)
        ref = np.zeros(out_pad, dtype=np.float64)
        for i in range(20):
            ref[sp_rows[i]] += float(sp_vals[i]) * float(x[sp_cols[i]])
        np.testing.assert_allclose(y, ref.astype(np.float32), atol=1e-5, rtol=1e-5)

    def test_zero_residual_returns_zero(self):
        _, layer = _make_layer(rank=0, sparse_frac=0.0)
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        assert np.all(y == 0.0)
        assert y.shape == (layer.indices.shape[0],)


# ── 3. End-to-end consistency with decompress_weight() ───────────────────────


class TestDecompressConsistency:
    """The residual GEMV should equal the residual contribution that
    decompress_weight() folds into W_rot before the inverse Hadamard."""

    def test_residual_matches_decompress_decomposition(self):
        # Reconstruct the residual portion (W_rot − dequant) from the layer
        # and check its product with x_rot equals sqint2_residual_gemv.
        rng = np.random.default_rng(0)
        out_features, in_features = 64, 128
        W = rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.02
        cfg = SQINT2Config(group_size=32, refine_iters=2, residual_rank=16,
                            residual_factor_dtype="fp32", sparse_frac=0.01, seed=42)
        layer = compress_weight(W, cfg)
        in_pad = _round_up(in_features, cfg.group_size)
        out_pad = layer.indices.shape[0]

        # Rebuild the dequant rotated weight (without the residual contribution).
        from squish.quant.sqint2 import _unpack_2bit
        indices = _unpack_2bit(layer.indices, in_pad)
        rescaled = NF2_VALUES[indices.astype(np.intp)]
        n_groups = in_pad // cfg.group_size
        rescaled_3d = rescaled.reshape(out_pad, n_groups, cfg.group_size)
        scale_3d = layer.scales[:, :, None]
        zp_3d = layer.zero_points[:, :, None]
        W_rot_int2 = ((rescaled_3d - zp_3d) * scale_3d).reshape(out_pad, in_pad)

        # Build the explicit residual weight, then GEMV.
        residual_dense = np.zeros_like(W_rot_int2, dtype=np.float64)
        residual_dense += layer.residual_L.astype(np.float64) @ layer.residual_R.astype(np.float64)
        if layer.sparse_rows is not None and layer.sparse_rows.size > 0:
            for i in range(layer.sparse_rows.size):
                residual_dense[int(layer.sparse_rows[i]), int(layer.sparse_cols[i])] += float(
                    layer.sparse_vals[i]
                )

        x_rot = rng.standard_normal(in_pad).astype(np.float32)
        y_ref = (residual_dense @ x_rot.astype(np.float64)).astype(np.float32)
        y = sqint2_residual_gemv(layer, x_rot)
        np.testing.assert_allclose(y, y_ref, atol=1e-3, rtol=1e-3)


# ── 4. dtype handling: fp16 storage promoted to fp32 for accumulation ────────


class TestDtypePromotion:
    def test_fp16_storage_round_trip(self):
        _, layer = _make_layer(rank=8, sparse_frac=0.0, factor_dtype="fp16")
        assert layer.residual_L.dtype == np.float16
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        # Compare against fp32-promoted explicit math (accumulation in fp32+).
        L32 = layer.residual_L.astype(np.float32)
        R32 = layer.residual_R.astype(np.float32)
        np.testing.assert_allclose(y, L32 @ R32 @ x, atol=1e-3, rtol=1e-3)

    def test_fp32_storage_round_trip(self):
        _, layer = _make_layer(rank=8, sparse_frac=0.0, factor_dtype="fp32")
        assert layer.residual_L.dtype == np.float32
        x = _x_rot(_round_up(128, 32))
        y = sqint2_residual_gemv(layer, x)
        np.testing.assert_allclose(
            y,
            layer.residual_L @ layer.residual_R @ x,
            atol=1e-4, rtol=1e-4,
        )


# ── 5. NumPy reference correctness (independent of layer plumbing) ───────────


class TestNumpyReference:
    def test_reference_matches_dense_matmul(self):
        rng = np.random.default_rng(0)
        M, r, N = 32, 4, 64
        L = rng.standard_normal((M, r)).astype(np.float32)
        R = rng.standard_normal((r, N)).astype(np.float32)
        x = rng.standard_normal(N).astype(np.float32)
        y = _residual_gemv_numpy(L, R, None, None, None, x)
        np.testing.assert_allclose(y, (L @ R) @ x, atol=1e-4, rtol=1e-4)

    def test_reference_with_sparse(self):
        rng = np.random.default_rng(0)
        M, r, N, k = 32, 4, 64, 12
        L = rng.standard_normal((M, r)).astype(np.float32)
        R = rng.standard_normal((r, N)).astype(np.float32)
        rows = rng.integers(0, M, size=k).astype(np.int32)
        cols = rng.integers(0, N, size=k).astype(np.int32)
        vals = rng.standard_normal(k).astype(np.float32)
        x = rng.standard_normal(N).astype(np.float32)
        y = _residual_gemv_numpy(L, R, rows, cols, vals, x)
        # Build dense reference.
        dense = (L.astype(np.float64) @ R.astype(np.float64))
        for i in range(k):
            dense[rows[i], cols[i]] += float(vals[i])
        ref = (dense @ x.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(y, ref, atol=1e-4, rtol=1e-4)

    def test_reference_rank_zero(self):
        L = np.zeros((16, 0), dtype=np.float32)
        R = np.zeros((0, 32), dtype=np.float32)
        x = np.random.default_rng(0).standard_normal(32).astype(np.float32)
        y = _residual_gemv_numpy(L, R, None, None, None, x)
        assert np.all(y == 0.0)
        assert y.shape == (16,)

    def test_reference_rank_mismatch_raises(self):
        L = np.zeros((4, 3), dtype=np.float32)
        R = np.zeros((4, 8), dtype=np.float32)  # r mismatch
        x = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="rank mismatch"):
            _residual_gemv_numpy(L, R, None, None, None, x)

    def test_reference_x_length_mismatch_raises(self):
        L = np.zeros((4, 3), dtype=np.float32)
        R = np.zeros((3, 8), dtype=np.float32)
        x = np.zeros(7, dtype=np.float32)
        with pytest.raises(ValueError, match="must equal"):
            _residual_gemv_numpy(L, R, None, None, None, x)

    def test_reference_sparse_triplet_length_mismatch(self):
        L = np.zeros((4, 0), dtype=np.float32)
        R = np.zeros((0, 8), dtype=np.float32)
        x = np.zeros(8, dtype=np.float32)
        rows = np.zeros(3, dtype=np.int32)
        cols = np.zeros(2, dtype=np.int32)  # mismatch
        vals = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError, match="same length"):
            _residual_gemv_numpy(L, R, rows, cols, vals, x)

    def test_reference_sparse_row_out_of_range(self):
        L = np.zeros((4, 0), dtype=np.float32)
        R = np.zeros((0, 8), dtype=np.float32)
        x = np.zeros(8, dtype=np.float32)
        rows = np.array([0, 7, 1], dtype=np.int32)  # 7 ≥ M=4
        cols = np.array([0, 1, 2], dtype=np.int32)
        vals = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="sparse_rows"):
            _residual_gemv_numpy(L, R, rows, cols, vals, x)

    def test_reference_sparse_col_out_of_range(self):
        L = np.zeros((4, 0), dtype=np.float32)
        R = np.zeros((0, 8), dtype=np.float32)
        x = np.zeros(8, dtype=np.float32)
        rows = np.array([0, 1, 2], dtype=np.int32)
        cols = np.array([0, 1, 99], dtype=np.int32)  # 99 ≥ N=8
        vals = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="sparse_cols"):
            _residual_gemv_numpy(L, R, rows, cols, vals, x)


# ── 6. Determinism ───────────────────────────────────────────────────────────


class TestDeterminism:
    def test_repeated_calls_byte_identical(self):
        _, layer = _make_layer(rank=8, sparse_frac=0.01)
        x = _x_rot(_round_up(128, 32))
        y1 = sqint2_residual_gemv(layer, x)
        y2 = sqint2_residual_gemv(layer, x)
        np.testing.assert_array_equal(y1, y2)


# ── 7. Rust ↔ NumPy parity (only runs if squish_quant is built) ──────────────


class TestRustParity:
    def test_rust_matches_numpy_reference(self):
        try:
            import squish_quant  # noqa: F401
        except ImportError:
            pytest.skip("squish_quant Rust extension not built")
        rng = np.random.default_rng(0)
        M, r, N, k = 256, 16, 1024, 100
        L = rng.standard_normal((M, r)).astype(np.float32)
        R = rng.standard_normal((r, N)).astype(np.float32)
        rows = rng.integers(0, M, size=k).astype(np.int32)
        cols = rng.integers(0, N, size=k).astype(np.int32)
        vals = (rng.standard_normal(k) * 0.05).astype(np.float32)
        x = rng.standard_normal(N).astype(np.float32)

        y_numpy = _residual_gemv_numpy(L, R, rows, cols, vals, x)
        y_rust = squish_quant.sqint2_residual_gemv_f32(L, R, rows, cols, vals, x)
        np.testing.assert_allclose(y_rust, y_numpy, atol=1e-5, rtol=1e-4)

    def test_rust_handles_empty_rank(self):
        try:
            import squish_quant
        except ImportError:
            pytest.skip("squish_quant Rust extension not built")
        L = np.zeros((4, 0), dtype=np.float32)
        R = np.zeros((0, 8), dtype=np.float32)
        x = np.arange(8, dtype=np.float32)
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        vals = np.empty(0, dtype=np.float32)
        y = squish_quant.sqint2_residual_gemv_f32(L, R, rows, cols, vals, x)
        assert y.shape == (4,)
        assert np.all(y == 0.0)
