"""Supplementary coverage for ``squish.quant.sqint2`` — the numpy packing
helpers, INT3/INT4 numpy quant edge branches, the Stage-3 SVD failure
fallback, the mixed-precision router entry point, and the on-disk loader's
dtype-coercion / sparse-mismatch branches the existing suite leaves uncovered.
All pure-numpy + filesystem — host-agnostic (no MLX, runs on macOS + Linux CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.quant import sqint2 as sq
from squish.quant.sqint2 import (
    SQINT2Config,
    compress_weight,
    compress_weights_sqint2,
    load_sqint2_layer,
    save_sqint2_layer,
)


# ── 2-bit packing helpers (dtype coercion) ───────────────────────────────────


def test_pack_2bit_coerces_non_uint8():
    out = sq._pack_2bit(np.array([0, 1, 2, 3], dtype=np.int32))
    assert out.dtype == np.uint8 and out[0] == 0b11100100


def test_unpack_2bit_coerces_non_uint8():
    out = sq._unpack_2bit(np.array([0b11100100], dtype=np.int32), 4)
    assert out.tolist() == [0, 1, 2, 3]


# ── _nf2_quantise_groups guards ──────────────────────────────────────────────


def test_nf2_quantise_requires_2d():
    with pytest.raises(ValueError, match="groups must be 2-D"):
        sq._nf2_quantise_groups(np.zeros(8, dtype=np.float32), refine_iters=2)


# ── _compute_stage3_residual SVD failure fallback ────────────────────────────


def test_stage3_residual_svd_failure_returns_zero_factors():
    cfg = SQINT2Config(residual_rank=2, sparse_frac=0.0)
    E = np.full((4, 4), np.inf, dtype=np.float32)  # SVD raises LinAlgError
    resL, resR, *_ = sq._compute_stage3_residual(E, cfg)
    assert resL is not None and resR is not None  # zero factors, not propagated


def test_stage3_residual_disabled_returns_all_none():
    cfg = SQINT2Config(residual_rank=0, sparse_frac=0.0)
    out = sq._compute_stage3_residual(np.zeros((4, 4), dtype=np.float32), cfg)
    assert all(x is None for x in out)  # rank 0 + sparse 0 → no correction


# ── INT3 / INT4 numpy quant edge branches ────────────────────────────────────


def test_int3_dequantize_non_divisible_groups():
    codes = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    scales = np.ones((1, 2), dtype=np.float32)
    zeros = np.zeros((1, 2), dtype=np.float32)
    recon = sq._int3_dequantize_numpy(codes, scales, zeros)  # n_groups*gs != in
    assert recon.shape == (1, 5)


def test_int4_quantize_pads_odd_width():
    W = np.ones((2, 5), dtype=np.float32)  # odd N → nibble pad branch
    packed, scales, offsets = sq._int4_quantize_numpy(W, group_size=32)
    assert packed.dtype == np.uint8 and packed.shape[0] == 2


def test_int4_dequantize_non_divisible_groups():
    packed = np.array([[0x10, 0x32, 0x04]], dtype=np.uint8)  # 6 nibbles
    scales = np.ones((1, 2), dtype=np.float32)
    offsets = np.zeros((1, 2), dtype=np.float32)
    recon = sq._int4_dequantize_numpy(packed, scales, offsets, in_features=5)
    assert recon.shape == (1, 5)


# ── compress_weights_sqint2 (router entry point) ─────────────────────────────


def _gate_weight():
    rng = np.random.default_rng(0)
    return {
        "model.layers.2.mlp.gate_proj.weight": (
            rng.standard_normal((64, 64)).astype(np.float32) * 0.02
        )
    }


def test_compress_weights_default_cfg():
    # sqint2_cfg=None → default production config branch.
    arrays, manifest, counts = compress_weights_sqint2(_gate_weight(), n_layers=8)
    assert counts["sqint2"] == 1
    assert any(k.endswith("__sqint2_L") for k in arrays)  # default rank>0


def test_compress_weights_no_residual_no_sparse():
    cfg = SQINT2Config(group_size=32, refine_iters=1, residual_rank=0, sparse_frac=0.0)
    arrays, _manifest, counts = compress_weights_sqint2(_gate_weight(), n_layers=8, sqint2_cfg=cfg)
    assert counts["sqint2"] == 1
    # residual / sparse branches skipped → none of those arrays present
    assert not any("__sqint2_L" in k or "__sqint2_srows" in k for k in arrays)


# ── load_sqint2_layer dtype coercion + sparse-mismatch ───────────────────────


def _save_full_layer(tmp_path, key="w"):
    rng = np.random.default_rng(1)
    W = rng.standard_normal((64, 128)).astype(np.float32) * 0.02
    layer = compress_weight(W, SQINT2Config(group_size=32, residual_rank=2, sparse_frac=0.05))
    save_sqint2_layer(layer, tmp_path, key)
    return key


def test_load_coerces_non_float32_scales_and_zp(tmp_path):
    key = _save_full_layer(tmp_path)
    # Rewrite scales + zero-points as float16 → loader coerces back to float32.
    for suffix in ("scales", "zp"):
        p = tmp_path / f"{key}__sqint2_{suffix}.npy"
        np.save(p, np.load(p).astype(np.float16))
    loaded = load_sqint2_layer(tmp_path, key)
    assert loaded.scales.dtype == np.float32 and loaded.zero_points.dtype == np.float32


def test_load_rejects_sparse_triplet_shape_mismatch(tmp_path):
    key = _save_full_layer(tmp_path)
    svals = tmp_path / f"{key}__sqint2_svals.npy"
    # Truncate svals so its shape no longer matches srows/scols → mismatch raise.
    np.save(svals, np.load(svals)[:-1])
    with pytest.raises(ValueError, match="sparse triplet shape mismatch"):
        load_sqint2_layer(tmp_path, key)
