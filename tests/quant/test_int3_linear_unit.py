"""tests/quant/test_int3_linear_unit.py

Unit tests for squish/quant/int3_linear.py.

Covers:
  - INT3Linear construction (valid and invalid inputs)
  - Shape and dtype contracts (weight=uint8, scales/zeros=float16)
  - group_size / out_features / in_features properties
  - __call__ dequantization correctness (vs numpy reference)
  - __call__ with and without bias
  - Round-trip SNR vs FP32 reference quantization
  - repr / str coverage
  - _nav_and_set_module helper (flat attribute, nested, list index)
  - _build_squish_3bit_dir + _load_squish_3bit_cache via synthetic npy-dir
    (no Metal required — MLX falls back to CPU in CI)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Guard: skip entire module if MLX is not importable ────────────────────────
mlx_available = True
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mlx_available = False

pytestmark = pytest.mark.skipif(not mlx_available, reason="MLX not available")

if mlx_available:
    from squish.quant.int3_linear import INT3Linear
    from squish.quant.compressed_loader import _nav_and_set_module

RNG = np.random.default_rng(0xABCD_1234)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_int3_linear(
    n_out: int = 32,
    n_in: int = 64,
    gs: int = 16,
    seed: int = 0,
    bias: bool = False,
) -> "INT3Linear":
    """Build an INT3Linear with random uint8 codes, float16 scales/zeros."""
    rng = np.random.default_rng(seed)
    n_groups = n_in // gs
    codes = rng.integers(0, 8, size=(n_out, n_in), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.05, size=(n_out, n_groups)).astype(np.float16)
    zeros  = rng.uniform(-0.1, 0.1,  size=(n_out, n_groups)).astype(np.float16)
    b_arr  = None
    if bias:
        b_arr = mx.array(rng.standard_normal(n_out).astype(np.float32)).astype(mx.bfloat16)
    return INT3Linear(
        weight=mx.array(codes, dtype=mx.uint8),
        scales=mx.array(scales, dtype=mx.float16),
        zeros=mx.array(zeros,  dtype=mx.float16),
        bias=b_arr,
    )


def _numpy_dequant(
    codes: np.ndarray,   # (n_out, n_in) uint8
    scales: np.ndarray,  # (n_out, n_groups) float16 → float32
    zeros: np.ndarray,   # (n_out, n_groups) float16 → float32
) -> np.ndarray:
    """NumPy reference dequantization matching INT3Linear.__call__."""
    n_out, n_in = codes.shape
    n_groups = scales.shape[1]
    gs = n_in // n_groups
    s32 = scales.astype(np.float32)  # (n_out, n_groups)
    z32 = zeros.astype(np.float32)
    c_f = codes.reshape(n_out, n_groups, gs).astype(np.float32)
    w_dq = (c_f * s32[:, :, None] + z32[:, :, None]).reshape(n_out, n_in)
    return w_dq.astype(np.float32)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestINT3LinearConstruction:
    def test_basic_construction(self):
        lin = _make_int3_linear()
        assert lin.weight.dtype == mx.uint8
        assert lin.scales.dtype == mx.float16
        assert lin.zeros.dtype  == mx.float16

    def test_wrong_dtype_raises(self):
        codes = mx.array(np.zeros((4, 8), dtype=np.float32))
        scales = mx.array(np.ones((4, 1), dtype=np.float16))
        zeros  = mx.array(np.zeros((4, 1), dtype=np.float16))
        with pytest.raises(TypeError, match="uint8"):
            INT3Linear(weight=codes, scales=scales, zeros=zeros)

    def test_non_2d_weight_raises(self):
        codes  = mx.array(np.zeros((4,), dtype=np.uint8))
        scales = mx.array(np.ones((4, 1), dtype=np.float16))
        zeros  = mx.array(np.zeros((4, 1), dtype=np.float16))
        with pytest.raises(ValueError, match="2-D"):
            INT3Linear(weight=codes, scales=scales, zeros=zeros)

    def test_scales_zeros_shape_mismatch_raises(self):
        codes  = mx.array(np.zeros((4, 8), dtype=np.uint8))
        scales = mx.array(np.ones((4, 2), dtype=np.float16))
        zeros  = mx.array(np.zeros((4, 1), dtype=np.float16))
        with pytest.raises(ValueError, match="matching shapes"):
            INT3Linear(weight=codes, scales=scales, zeros=zeros)

    def test_scales_wrong_leading_dim_raises(self):
        codes  = mx.array(np.zeros((4, 8), dtype=np.uint8))
        scales = mx.array(np.ones((3, 2), dtype=np.float16))
        zeros  = mx.array(np.zeros((3, 2), dtype=np.float16))
        with pytest.raises(ValueError, match="out_features"):
            INT3Linear(weight=codes, scales=scales, zeros=zeros)

    def test_non_divisible_groups_raises(self):
        # n_in=9 with n_groups=4 → 9 % 4 = 1 → should raise
        codes  = mx.array(np.zeros((4, 9), dtype=np.uint8))
        scales = mx.array(np.ones((4, 4), dtype=np.float16))
        zeros  = mx.array(np.zeros((4, 4), dtype=np.float16))
        with pytest.raises(ValueError, match="divisible"):
            INT3Linear(weight=codes, scales=scales, zeros=zeros)

    def test_no_bias_by_default(self):
        lin = _make_int3_linear()
        assert not hasattr(lin, "bias")

    def test_with_bias(self):
        lin = _make_int3_linear(bias=True)
        assert hasattr(lin, "bias")
        assert lin.bias.shape == (32,)

    def test_repr(self):
        lin = _make_int3_linear()
        r = repr(lin)
        assert "INT3Linear" in r
        assert "in=64" in r
        assert "out=32" in r
        assert "gs=16" in r
        assert "no bias" in r


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestINT3LinearProperties:
    def test_out_features(self):
        lin = _make_int3_linear(n_out=16, n_in=64)
        assert lin.out_features == 16

    def test_in_features(self):
        lin = _make_int3_linear(n_out=8, n_in=128)
        assert lin.in_features == 128

    def test_group_size(self):
        lin = _make_int3_linear(n_out=4, n_in=64, gs=32)
        assert lin.group_size == 32

    def test_group_size_full_row(self):
        """Single group per row → group_size == n_in."""
        lin = _make_int3_linear(n_out=4, n_in=64, gs=64)
        assert lin.group_size == 64


# ---------------------------------------------------------------------------
# Forward pass (dequantisation correctness)
# ---------------------------------------------------------------------------


class TestINT3LinearForward:
    def test_output_shape_1d_input(self):
        lin = _make_int3_linear(n_out=16, n_in=32, gs=8)
        x = mx.zeros((32,))
        y = lin(x)
        mx.eval(y)
        assert y.shape == (16,)

    def test_output_shape_2d_batch(self):
        lin = _make_int3_linear(n_out=16, n_in=32, gs=8)
        x = mx.zeros((5, 32))
        y = lin(x)
        mx.eval(y)
        assert y.shape == (5, 16)

    def test_output_shape_3d_batch(self):
        lin = _make_int3_linear(n_out=16, n_in=32, gs=8)
        x = mx.zeros((2, 4, 32))
        y = lin(x)
        mx.eval(y)
        assert y.shape == (2, 4, 16)

    def test_dequant_matches_numpy_reference(self):
        """Verify INT3Linear dequantization equals NumPy reference."""
        n_out, n_in, gs = 8, 32, 8
        rng = np.random.default_rng(99)
        codes_np  = rng.integers(0, 8, (n_out, n_in), dtype=np.uint8)
        scales_np = rng.uniform(0.01, 0.1, (n_out, n_in // gs)).astype(np.float16)
        zeros_np  = rng.uniform(-0.2, 0.2, (n_out, n_in // gs)).astype(np.float16)
        x_np = rng.standard_normal((n_in,)).astype(np.float32)

        lin = INT3Linear(
            weight=mx.array(codes_np),
            scales=mx.array(scales_np),
            zeros=mx.array(zeros_np),
        )
        y_mlx = np.array(lin(mx.array(x_np))).astype(np.float32)
        mx.eval()

        # NumPy reference
        w_dq_ref = _numpy_dequant(codes_np, scales_np, zeros_np)  # (n_out, n_in)
        y_ref = (x_np @ w_dq_ref.T).astype(np.float32)

        # Tolerance: bfloat16 has ~7-bit mantissa; expect <1% relative error
        max_rel_err = float(np.max(np.abs(y_mlx - y_ref) / (np.abs(y_ref) + 1e-6)))
        assert max_rel_err < 0.02, f"Max relative error too high: {max_rel_err:.4f}"

    def test_zero_input_gives_zero(self):
        lin = _make_int3_linear()
        x = mx.zeros((lin.in_features,))
        y = lin(x)
        mx.eval(y)
        assert np.allclose(np.array(y), 0.0)

    def test_bias_added(self):
        n_out, n_in, gs = 4, 8, 4
        codes  = mx.array(np.zeros((n_out, n_in), dtype=np.uint8))
        scales = mx.array(np.ones((n_out, n_in // gs), dtype=np.float16) * 0.0)
        zeros  = mx.array(np.zeros((n_out, n_in // gs), dtype=np.float16))
        bias_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        lin = INT3Linear(
            weight=codes, scales=scales, zeros=zeros,
            bias=mx.array(bias_np).astype(mx.bfloat16),
        )
        x = mx.zeros((n_in,))
        y = np.array(lin(x)).astype(np.float32)
        mx.eval()
        np.testing.assert_allclose(y, bias_np, atol=1e-2)

    def test_all_zeros_scales_gives_zeros_output(self):
        n_out, n_in, gs = 4, 8, 4
        rng = np.random.default_rng(7)
        codes   = mx.array(rng.integers(0, 8, (n_out, n_in), dtype=np.uint8))
        scales  = mx.array(np.zeros((n_out, n_in // gs), dtype=np.float16))
        zeros   = mx.array(np.zeros((n_out, n_in // gs), dtype=np.float16))
        lin = INT3Linear(weight=codes, scales=scales, zeros=zeros)
        x = mx.array(rng.standard_normal((n_in,)).astype(np.float32))
        y = np.array(lin(x))
        mx.eval()
        np.testing.assert_allclose(y, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Round-trip SNR
# ---------------------------------------------------------------------------


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal_power = float(np.mean(original ** 2))
    noise_power  = float(np.mean((original - reconstructed) ** 2))
    if noise_power < 1e-15:
        return float("inf")
    return 10.0 * np.log10(max(signal_power, 1e-15) / noise_power)


class TestINT3LinearSNR:
    def test_round_trip_snr_acceptable(self):
        """INT3 asymmetric quantization should give ≥ 15 dB SNR for normal weights."""
        n_out, n_in, gs = 64, 128, 64
        rng = np.random.default_rng(42)
        w_fp32 = rng.standard_normal((n_out, n_in)).astype(np.float32)

        # Quantize: groups are per-row (n_out × n_groups_per_row groups)
        n_groups_per_row = n_in // gs
        w_reshaped = w_fp32.reshape(n_out, n_groups_per_row, gs)  # (n_out, ngpr, gs)

        w_min = w_reshaped.min(axis=2, keepdims=True)   # (n_out, ngpr, 1)
        w_max = w_reshaped.max(axis=2, keepdims=True)
        scales = ((w_max - w_min) / 7.0).astype(np.float16)      # (n_out, ngpr, 1)
        zeros  = w_min.astype(np.float16)                         # (n_out, ngpr, 1)

        safe_s = np.where(np.abs(scales) < 1e-6, 1.0, scales).astype(np.float32)
        codes = np.clip(
            np.round((w_reshaped - zeros) / safe_s), 0, 7
        ).astype(np.uint8)   # (n_out, ngpr, gs)

        codes_2d  = codes.reshape(n_out, n_in)
        scales_2d = scales.reshape(n_out, n_groups_per_row)
        zeros_2d  = zeros.reshape(n_out, n_groups_per_row)

        lin = INT3Linear(
            weight=mx.array(codes_2d),
            scales=mx.array(scales_2d),
            zeros=mx.array(zeros_2d),
        )
        x = mx.array(rng.standard_normal((n_in,)).astype(np.float32))
        y_mlx = np.array(lin(x)).astype(np.float32)
        mx.eval()

        w_ref = _numpy_dequant(codes_2d, scales_2d, zeros_2d)
        y_ref = (np.array(x).astype(np.float32) @ w_ref.T)

        snr = _snr_db(y_ref, y_mlx)
        assert snr > 15.0, f"Round-trip SNR too low: {snr:.1f} dB"


# ---------------------------------------------------------------------------
# _nav_and_set_module helper
# ---------------------------------------------------------------------------


class TestNavAndSetModule:
    def _make_nested(self):
        class Inner(nn.Module):
            def __init__(self):
                self.linear = nn.Linear(4, 8)
        class Outer(nn.Module):
            def __init__(self):
                self.inner = Inner()
                self.layers = [Inner(), Inner()]
        return Outer()

    def test_flat_attr(self):
        model = self._make_nested()
        replacement = nn.ReLU()
        _nav_and_set_module(model, ["inner"], replacement)
        assert model.inner is replacement

    def test_nested_attr(self):
        model = self._make_nested()
        replacement = nn.ReLU()
        _nav_and_set_module(model, ["inner", "linear"], replacement)
        assert model.inner.linear is replacement

    def test_list_index(self):
        model = self._make_nested()
        replacement = nn.ReLU()
        _nav_and_set_module(model, ["layers", "0"], replacement)
        assert model.layers[0] is replacement

    def test_nested_into_list(self):
        model = self._make_nested()
        replacement = nn.ReLU()
        _nav_and_set_module(model, ["layers", "1", "linear"], replacement)
        assert model.layers[1].linear is replacement

    def test_nonexistent_intermediate_attr_raises(self):
        """Navigating THROUGH a nonexistent intermediate attribute raises AttributeError."""
        model = self._make_nested()
        with pytest.raises(AttributeError):
            _nav_and_set_module(model, ["nonexistent_container", "linear"], nn.ReLU())


# ---------------------------------------------------------------------------
# Synthetic npy-dir → _build_squish_3bit_dir + _load_squish_3bit_cache
# ---------------------------------------------------------------------------


def _make_int3_npy_dir(root: Path, n_out: int = 8, n_in: int = 32, gs: int = 8) -> dict:
    """Create a minimal synthetic npy-dir with INT3 + passthrough tensors."""
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_groups_per_row = n_in // gs
    n_total_groups   = n_out * n_groups_per_row

    # INT3-quantised weight tensor
    codes   = rng.integers(0, 8, (n_total_groups, gs), dtype=np.uint8)
    scales  = rng.uniform(0.01, 0.1, n_total_groups).astype(np.float32)
    zeros_a = rng.uniform(-0.2, 0.2, n_total_groups).astype(np.float32)
    orig_shape = np.array([n_out, n_in])

    np.save(tensor_dir / "layer__q3.npy",     codes)
    np.save(tensor_dir / "layer__s3.npy",     scales)
    np.save(tensor_dir / "layer__z3.npy",     zeros_a)
    np.save(tensor_dir / "layer__shape.npy",  orig_shape)
    # Also write standard INT8 files so _collect_tensor_keys picks up "layer" as base key
    q_int8 = np.clip(np.round(rng.standard_normal((n_out, n_in)) * 127), -127, 127).astype(np.int8)
    s_int8 = np.ones(n_out, dtype=np.float32) * 0.01
    np.save(tensor_dir / "layer__q.npy",      q_int8)
    np.save(tensor_dir / "layer__s.npy",      s_int8)

    # Passthrough tensor (embedding-like)
    emb = rng.standard_normal((4, n_in)).astype(np.float16)
    np.save(tensor_dir / "embed__pt.npy",     emb)
    np.save(tensor_dir / "embed__shape.npy",  np.array([4, n_in]))

    # Manifest: safe_key → original_name
    manifest = {
        "model.layer.weight": "layer",
        "model.embed.weight": "embed",
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return {"n_out": n_out, "n_in": n_in, "gs": gs,
            "codes": codes, "scales": scales, "zeros": zeros_a}


class TestBuildSquish3bitDir:
    """End-to-end test for _build_squish_3bit_dir (no real model; uses synthetic data)."""

    def test_creates_sentinel(self, tmp_path):
        info = _make_int3_npy_dir(tmp_path)

        from squish.quant.compressed_loader import (
            _collect_tensor_keys,
            _build_squish_3bit_dir,
            _tensor_load_key,
        )

        tensor_dir = tmp_path / "tensors"
        base_keys  = sorted(_collect_tensor_keys(tensor_dir), key=_tensor_load_key)
        safe_to_original = {v: k for k, v in
                            json.loads((tmp_path / "manifest.json").read_text()).items()}

        _build_squish_3bit_dir(
            dir_path=tmp_path,
            tensor_dir=tensor_dir,
            base_keys=base_keys,
            safe_to_original=safe_to_original,
            model_dir=str(tmp_path),   # no real config.json — function handles gracefully
            verbose=False,
        )

        assert (tmp_path / ".squish_3bit_ready").exists()
        assert (tmp_path / "squish_3bit" / "model.safetensors").exists()

    def test_int3_layer_stored_as_uint8(self, tmp_path):
        _make_int3_npy_dir(tmp_path)

        from squish.quant.compressed_loader import (
            _collect_tensor_keys, _build_squish_3bit_dir, _tensor_load_key,
        )
        tensor_dir = tmp_path / "tensors"
        base_keys  = sorted(_collect_tensor_keys(tensor_dir), key=_tensor_load_key)
        safe_to_original = {v: k for k, v in
                            json.loads((tmp_path / "manifest.json").read_text()).items()}

        _build_squish_3bit_dir(
            dir_path=tmp_path, tensor_dir=tensor_dir, base_keys=base_keys,
            safe_to_original=safe_to_original, model_dir=str(tmp_path), verbose=False,
        )

        loaded = mx.load(str(tmp_path / "squish_3bit" / "model.safetensors"))
        # INT3 layer should have uint8 weight and float16 scales/zeros
        assert "model.layer.weight" in loaded
        assert loaded["model.layer.weight"].dtype == mx.uint8
        assert "model.layer.scales" in loaded
        assert loaded["model.layer.scales"].dtype == mx.float16
        assert "model.layer.zeros" in loaded
        assert loaded["model.layer.zeros"].dtype == mx.float16

    def test_non_int3_layer_stored_as_bfloat16(self, tmp_path):
        _make_int3_npy_dir(tmp_path)

        from squish.quant.compressed_loader import (
            _collect_tensor_keys, _build_squish_3bit_dir, _tensor_load_key,
        )
        tensor_dir = tmp_path / "tensors"
        base_keys  = sorted(_collect_tensor_keys(tensor_dir), key=_tensor_load_key)
        safe_to_original = {v: k for k, v in
                            json.loads((tmp_path / "manifest.json").read_text()).items()}

        _build_squish_3bit_dir(
            dir_path=tmp_path, tensor_dir=tensor_dir, base_keys=base_keys,
            safe_to_original=safe_to_original, model_dir=str(tmp_path), verbose=False,
        )

        loaded = mx.load(str(tmp_path / "squish_3bit" / "model.safetensors"))
        assert "model.embed.weight" in loaded
        assert loaded["model.embed.weight"].dtype == mx.bfloat16

    def test_uint8_codes_match_original(self, tmp_path):
        info = _make_int3_npy_dir(tmp_path)

        from squish.quant.compressed_loader import (
            _collect_tensor_keys, _build_squish_3bit_dir, _tensor_load_key,
        )
        tensor_dir = tmp_path / "tensors"
        base_keys  = sorted(_collect_tensor_keys(tensor_dir), key=_tensor_load_key)
        safe_to_original = {v: k for k, v in
                            json.loads((tmp_path / "manifest.json").read_text()).items()}

        _build_squish_3bit_dir(
            dir_path=tmp_path, tensor_dir=tensor_dir, base_keys=base_keys,
            safe_to_original=safe_to_original, model_dir=str(tmp_path), verbose=False,
        )

        loaded = mx.load(str(tmp_path / "squish_3bit" / "model.safetensors"))
        stored_codes = np.array(loaded["model.layer.weight"])
        mx.eval()

        n_out, n_in = info["n_out"], info["n_in"]
        expected_codes = info["codes"].ravel()[:n_out * n_in].reshape(n_out, n_in)
        np.testing.assert_array_equal(stored_codes, expected_codes)
