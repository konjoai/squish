"""tests/test_sqint2_compress.py — Integration tests for W103.3 compress pipeline.

Tests compress_weights_sqint2() with synthetic 6-layer model weight dicts.
No filesystem writes required for the core tests; npy-dir round-trip uses tmp_path.

Covers:
  - format_map routing: correct tier for every tensor class in a 6-layer model
  - SQINT2 array keys: __sqint2_idx / scales / zp / L / R / srows / scols / svals
  - INT3 array keys: __q3 / __s3 / __z3
  - INT4 array keys: __q4 / __s4 / __z4
  - Passthrough keys: __pt
  - Shape arrays: __shape present for all tensors
  - Dtype contracts: codes uint8, scales float32, L/R fp16, sparse int32/fp16
  - Round-trip fidelity: decompress_weight on SQINT2 tensors ≥ 9 dB SNR
  - INT3 round-trip: decode w = code*scale+zero produces finite, bounded values
  - INT4 round-trip: decode w = offset + code*scale produces finite, bounded values
  - fmt_counts: correct count of sqint2/int3/int4/skip
  - All-boundary model (n_layers=3): all weight tensors compressed at INT4
  - 1-D weights (layernorm): always passthrough regardless of router verdict
  - npy-dir write and reload: arrays survive np.save/np.load round-trip
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.quant.sqint2 import (
    SQINT2Config,
    _int3_dequantize_numpy,
    _int3_quantize_numpy,
    _int4_dequantize_numpy,
    _int4_quantize_numpy,
    compress_weights_sqint2,
    decompress_weight,
    snr_db,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_LAYERS = 6   # boundary = {0, 1, 4, 5}; non-boundary = {2, 3}

def _make_weights(n_layers: int = N_LAYERS, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic model weight dict matching the Llama/Qwen naming convention."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"model.layers.{i}.self_attn.{proj}.weight"] = (
                rng.standard_normal((32, 64)).astype(np.float32) * 0.02
            )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"model.layers.{i}.mlp.{proj}.weight"] = (
                rng.standard_normal((64, 32)).astype(np.float32) * 0.02
            )
        # 1-D layernorm (should always passthrough)
        weights[f"model.layers.{i}.input_layernorm.weight"] = (
            rng.standard_normal((32,)).astype(np.float32)
        )
    # Non-layer tensors (no layer index → router returns None)
    weights["model.embed_tokens.weight"] = rng.standard_normal((256, 32)).astype(np.float32)
    weights["lm_head.weight"] = rng.standard_normal((256, 32)).astype(np.float32)
    weights["model.norm.weight"] = rng.standard_normal((32,)).astype(np.float32)
    return weights


@pytest.fixture
def weights():
    return _make_weights()


@pytest.fixture
def result(weights):
    cfg = SQINT2Config(group_size=32, refine_iters=1, residual_rank=4, sparse_frac=0.01)
    return compress_weights_sqint2(weights, n_layers=N_LAYERS, sqint2_cfg=cfg)


# ── 1. Format routing ─────────────────────────────────────────────────────────


class TestFormatRouting:
    """Verify that fmt_counts and individual tensor formats match the W103.3 spec."""

    def test_non_boundary_gate_proj_is_sqint2(self, result):
        arrays, manifest, fmt_counts = result
        # Non-boundary layers are {2, 3}; gate_proj should be SQINT2
        sk = "model__layers__2__mlp__gate_proj__weight"
        assert f"{sk}__sqint2_idx" in arrays, "gate_proj missing __sqint2_idx"

    def test_non_boundary_up_proj_is_sqint2(self, result):
        arrays, manifest, fmt_counts = result
        sk = "model__layers__3__mlp__up_proj__weight"
        assert f"{sk}__sqint2_idx" in arrays

    def test_boundary_gate_proj_is_int4(self, result):
        arrays, manifest, fmt_counts = result
        # Layer 0 is boundary → INT4
        sk = "model__layers__0__mlp__gate_proj__weight"
        assert f"{sk}__q4" in arrays
        assert f"{sk}__sqint2_idx" not in arrays

    def test_non_boundary_attn_proj_is_int3(self, result):
        arrays, manifest, fmt_counts = result
        sk = "model__layers__2__self_attn__q_proj__weight"
        assert f"{sk}__q3" in arrays
        assert f"{sk}__sqint2_idx" not in arrays

    def test_boundary_attn_proj_is_int4(self, result):
        arrays, manifest, fmt_counts = result
        sk = "model__layers__1__self_attn__v_proj__weight"
        assert f"{sk}__q4" in arrays
        assert f"{sk}__q3" not in arrays

    def test_down_proj_is_int4(self, result):
        arrays, manifest, fmt_counts = result
        sk = "model__layers__2__mlp__down_proj__weight"
        assert f"{sk}__q4" in arrays

    def test_embed_tokens_is_passthrough(self, result):
        arrays, manifest, fmt_counts = result
        assert "model__embed_tokens__weight__pt" in arrays

    def test_lm_head_is_passthrough(self, result):
        arrays, manifest, fmt_counts = result
        assert "lm_head__weight__pt" in arrays

    def test_1d_layernorm_is_passthrough(self, result):
        arrays, manifest, fmt_counts = result
        # 1-D tensor: always passthrough even if inside a transformer block
        sk = "model__layers__2__input_layernorm__weight"
        assert f"{sk}__pt" in arrays
        assert f"{sk}__q3" not in arrays

    def test_fmt_counts_sqint2(self, result):
        _, _, fmt_counts = result
        # Non-boundary layers = {2,3}; 2 layers × (gate_proj + up_proj) = 4
        assert fmt_counts["sqint2"] == 4, fmt_counts

    def test_fmt_counts_int3(self, result):
        _, _, fmt_counts = result
        # Non-boundary {2,3} × (q+k+v+o) = 8
        assert fmt_counts["int3"] == 8, fmt_counts

    def test_fmt_counts_int4(self, result):
        _, _, fmt_counts = result
        # Boundary {0,1,4,5} × 7 projections = 28
        # Non-boundary {2,3} × down_proj = 2
        # Total: 30
        assert fmt_counts["int4"] == 30, fmt_counts

    def test_fmt_counts_skip(self, result):
        _, _, fmt_counts = result
        # embed_tokens (2D→pt), lm_head (2D→pt), model.norm (1D→pt),
        # 6 × input_layernorm (1D→pt) = 9
        assert fmt_counts["skip"] == 9, fmt_counts

    def test_all_boundary_model(self):
        """n_layers=3 → all layers are boundary → every projection is INT4."""
        w = _make_weights(n_layers=3)
        cfg = SQINT2Config(group_size=32, refine_iters=1, residual_rank=4, sparse_frac=0.01)
        arrays, _, fmt_counts = compress_weights_sqint2(w, n_layers=3, sqint2_cfg=cfg)
        assert fmt_counts["sqint2"] == 0
        assert fmt_counts["int3"] == 0
        assert fmt_counts["int4"] > 0


# ── 2. Array key and dtype contracts ─────────────────────────────────────────


class TestArrayContracts:
    def test_sqint2_idx_dtype_uint8(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__mlp__gate_proj__weight"
        assert arrays[f"{sk}__sqint2_idx"].dtype == np.uint8

    def test_sqint2_scales_dtype_float32(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__mlp__gate_proj__weight"
        assert arrays[f"{sk}__sqint2_scales"].dtype == np.float32

    def test_sqint2_L_dtype_fp16(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__mlp__gate_proj__weight"
        L_key = f"{sk}__sqint2_L"
        if L_key in arrays:
            assert arrays[L_key].dtype == np.float16

    def test_sqint2_srows_dtype_int32(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__mlp__gate_proj__weight"
        srows_key = f"{sk}__sqint2_srows"
        if srows_key in arrays:
            assert arrays[srows_key].dtype == np.int32

    def test_sqint2_svals_dtype_fp16(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__mlp__gate_proj__weight"
        svals_key = f"{sk}__sqint2_svals"
        if svals_key in arrays:
            assert arrays[svals_key].dtype == np.float16

    def test_int3_codes_dtype_uint8(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__self_attn__q_proj__weight"
        assert arrays[f"{sk}__q3"].dtype == np.uint8

    def test_int3_scales_dtype_float32(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__self_attn__q_proj__weight"
        assert arrays[f"{sk}__s3"].dtype == np.float32

    def test_int3_zeros_dtype_float32(self, result):
        arrays, _, _ = result
        sk = "model__layers__2__self_attn__q_proj__weight"
        assert arrays[f"{sk}__z3"].dtype == np.float32

    def test_int4_packed_dtype_uint8(self, result):
        arrays, _, _ = result
        sk = "model__layers__0__mlp__gate_proj__weight"
        assert arrays[f"{sk}__q4"].dtype == np.uint8

    def test_int4_scales_dtype_float32(self, result):
        arrays, _, _ = result
        sk = "model__layers__0__mlp__gate_proj__weight"
        assert arrays[f"{sk}__s4"].dtype == np.float32

    def test_passthrough_dtype_float16(self, result):
        arrays, _, _ = result
        assert arrays["model__embed_tokens__weight__pt"].dtype == np.float16

    def test_shape_array_present_for_all(self, weights, result):
        arrays, manifest, _ = result
        for name in weights:
            sk = manifest[name]
            assert f"{sk}__shape" in arrays, f"Missing __shape for {name}"

    def test_shape_array_dtype_int64(self, weights, result):
        arrays, manifest, _ = result
        for name in weights:
            sk = manifest[name]
            assert arrays[f"{sk}__shape"].dtype == np.int64


# ── 3. INT3 codec correctness ─────────────────────────────────────────────────


class TestINT3Codec:
    def test_codes_in_range(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        assert codes.min() >= 0
        assert codes.max() <= 7

    def test_round_trip_finite(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        W_rec = _int3_dequantize_numpy(codes, scales, zeros)
        assert np.isfinite(W_rec).all()

    def test_round_trip_shape(self):
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        W_rec = _int3_dequantize_numpy(codes, scales, zeros)
        assert W_rec.shape == W.shape

    def test_round_trip_snr_positive(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 128)).astype(np.float32) * 0.02
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        W_rec = _int3_dequantize_numpy(codes, scales, zeros)
        assert snr_db(W, W_rec) > 0.0

    def test_constant_group_safe(self):
        W = np.full((4, 32), 0.05, dtype=np.float32)
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        W_rec = _int3_dequantize_numpy(codes, scales, zeros)
        assert np.isfinite(W_rec).all()

    def test_non_divisible_width_padded(self):
        W = np.random.default_rng(0).standard_normal((8, 50)).astype(np.float32) * 0.02
        codes, scales, zeros = _int3_quantize_numpy(W, group_size=32)
        # codes shape should match W shape (padding stripped)
        assert codes.shape == W.shape


# ── 4. INT4 codec correctness ─────────────────────────────────────────────────


class TestINT4Codec:
    def test_packed_shape(self):
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        packed, scales, offsets = _int4_quantize_numpy(W, group_size=32)
        assert packed.shape == (32, 32)  # 64 values → 32 bytes nibble-packed
        assert packed.dtype == np.uint8

    def test_round_trip_finite(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        packed, scales, offsets = _int4_quantize_numpy(W, group_size=32)
        W_rec = _int4_dequantize_numpy(packed, scales, offsets, in_features=64)
        assert np.isfinite(W_rec).all()

    def test_round_trip_shape(self):
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        packed, scales, offsets = _int4_quantize_numpy(W, group_size=32)
        W_rec = _int4_dequantize_numpy(packed, scales, offsets, in_features=64)
        assert W_rec.shape == W.shape

    def test_round_trip_snr_positive(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 128)).astype(np.float32) * 0.02
        packed, scales, offsets = _int4_quantize_numpy(W, group_size=32)
        W_rec = _int4_dequantize_numpy(packed, scales, offsets, in_features=128)
        assert snr_db(W, W_rec) > 0.0


# ── 5. SQINT2 round-trip fidelity via compress_weights_sqint2 ─────────────────


class TestSQINT2RoundTrip:
    """Decompress SQINT2-tier tensors and check joint SNR ≥ 9 dB."""

    def _compress_and_decompress(self, tensor_name: str, W: np.ndarray, n_layers: int):
        cfg = SQINT2Config(group_size=32, refine_iters=2, residual_rank=4, sparse_frac=0.01)
        arrays, manifest, _ = compress_weights_sqint2(
            {tensor_name: W}, n_layers=n_layers, sqint2_cfg=cfg
        )
        from squish.quant.sqint2 import (
            SQINT2Layer, _unpack_2bit, _round_up,
            build_hadamard, inverse_hadamard, NF2_VALUES, _unpack_factor
        )
        sk = manifest[tensor_name]
        # Reconstruct SQINT2Layer from stored arrays
        indices_packed = arrays[f"{sk}__sqint2_idx"]
        scales = arrays[f"{sk}__sqint2_scales"]
        zero_points = arrays[f"{sk}__sqint2_zp"]
        res_L = arrays.get(f"{sk}__sqint2_L")
        res_R = arrays.get(f"{sk}__sqint2_R")
        sp_rows = arrays.get(f"{sk}__sqint2_srows")
        sp_cols = arrays.get(f"{sk}__sqint2_scols")
        sp_vals = arrays.get(f"{sk}__sqint2_svals")

        layer = SQINT2Layer(
            indices=indices_packed,
            scales=scales,
            zero_points=zero_points,
            in_features=W.shape[1],
            out_features=W.shape[0],
            cfg=cfg,
            residual_L=res_L,
            residual_R=res_R,
            sparse_rows=sp_rows,
            sparse_cols=sp_cols,
            sparse_vals=sp_vals,
        )
        return decompress_weight(layer)

    def test_gate_proj_snr_above_9db(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 128)).astype(np.float32) * 0.02
        W_rec = self._compress_and_decompress(
            "model.layers.5.mlp.gate_proj.weight", W, n_layers=8
        )
        assert snr_db(W, W_rec) >= 9.0

    def test_up_proj_snr_above_9db(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((64, 128)).astype(np.float32) * 0.02
        W_rec = self._compress_and_decompress(
            "model.layers.5.mlp.up_proj.weight", W, n_layers=8
        )
        assert snr_db(W, W_rec) >= 9.0

    def test_decompressed_shape_matches_original(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        W_rec = self._compress_and_decompress(
            "model.layers.3.mlp.gate_proj.weight", W, n_layers=6
        )
        assert W_rec.shape == W.shape

    def test_decompressed_is_finite(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        W_rec = self._compress_and_decompress(
            "model.layers.3.mlp.gate_proj.weight", W, n_layers=6
        )
        assert np.isfinite(W_rec).all()


# ── 6. Manifest correctness ───────────────────────────────────────────────────


class TestManifest:
    def test_manifest_has_all_original_names(self, weights, result):
        _, manifest, _ = result
        for name in weights:
            assert name in manifest, f"Missing manifest entry for {name}"

    def test_manifest_safe_key_format(self, weights, result):
        _, manifest, _ = result
        for orig, sk in manifest.items():
            assert "." not in sk, f"safe_key {sk!r} still contains dots"
            assert "__" in sk or sk == orig.replace(".", "__")


# ── 7. npy-dir round-trip (tmp_path) ─────────────────────────────────────────


class TestNpyDirRoundTrip:
    def test_write_and_reload(self, weights, result, tmp_path):
        from squish.convert import write_npy_dir
        arrays, manifest, _ = result
        write_npy_dir(tmp_path, arrays, manifest)

        assert (tmp_path / "manifest.json").exists()
        tensor_dir = tmp_path / "tensors"
        assert tensor_dir.exists()

        # Spot-check: SQINT2 idx for a non-boundary gate_proj
        sk = "model__layers__2__mlp__gate_proj__weight"
        idx_path = tensor_dir / f"{sk}__sqint2_idx.npy"
        assert idx_path.exists(), f"Expected {idx_path}"
        loaded = np.load(str(idx_path))
        assert loaded.dtype == np.uint8

    def test_manifest_json_readable(self, weights, result, tmp_path):
        from squish.convert import write_npy_dir
        arrays, manifest, _ = result
        write_npy_dir(tmp_path, arrays, manifest)
        with open(tmp_path / "manifest.json") as f:
            loaded_manifest = json.load(f)
        for name in weights:
            assert name in loaded_manifest

    def test_int3_arrays_survive_npy_roundtrip(self, weights, result, tmp_path):
        from squish.convert import write_npy_dir
        arrays, manifest, _ = result
        write_npy_dir(tmp_path, arrays, manifest)

        sk = "model__layers__2__self_attn__q_proj__weight"
        q3 = np.load(str(tmp_path / "tensors" / f"{sk}__q3.npy"))
        s3 = np.load(str(tmp_path / "tensors" / f"{sk}__s3.npy"))
        z3 = np.load(str(tmp_path / "tensors" / f"{sk}__z3.npy"))
        assert q3.dtype == np.uint8
        assert q3.min() >= 0 and q3.max() <= 7
        assert np.isfinite(s3).all()
        assert np.isfinite(z3).all()
