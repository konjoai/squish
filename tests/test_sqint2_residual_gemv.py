"""tests/test_sqint2_residual_gemv.py — W103.4b SQINT2 residual GEMV tests.

Covers:
  - Shape and dtype contracts for the public API
  - NumPy fallback correctness vs reference (x @ Rᵀ) @ Lᵀ
  - Rust kernel correctness vs NumPy fallback (when Rust extension built)
  - Numerical precision: fp32 accumulation, fp16 factor fidelity
  - Error paths: shape mismatches (L/R/x incompatible)
  - Edge cases: batch=1, rank=1, rank=16, non-square in_f ≠ out_f
  - Backend info reports sqint2_residual_gemv_rust correctly
  - Integration with SQINT2Layer residual fields (compress → gemv → reconstruct)
  - get_backend_info() key presence

W103.4b math:
    h = x @ Rᵀ     (batch, rank)
    y = h @ Lᵀ     (batch, out_f)
    out = y         = x @ (L @ R)ᵀ

L ∈ fp16 (out_f, rank), R ∈ fp16 (rank, in_f), x ∈ fp32 (batch, in_f).
All matmul accumulation in fp32 (CLAUDE.md mandate).
"""

from __future__ import annotations

import importlib
import numpy as np
import pytest

from squish.quant.quantizer import (
    sqint2_residual_gemv,
    _sqint2_residual_gemv_numpy,
    get_backend_info,
)

RNG = np.random.default_rng(0xC0FFEE42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_squish_quant = None
try:
    import squish_quant as _squish_quant
except ImportError:
    pass

RUST_AVAILABLE = (
    _squish_quant is not None
    and hasattr(_squish_quant, "sqint2_residual_gemv")
)


def _make_factors_and_x(out_f=64, in_f=128, rank=16, batch=4, seed=7):
    rng = np.random.default_rng(seed)
    # Scale σ=0.02 — representative of post-quantisation SQINT2 residuals
    L = (rng.standard_normal((out_f, rank)) * 0.02).astype(np.float16)
    R = (rng.standard_normal((rank, in_f)) * 0.02).astype(np.float16)
    x = rng.standard_normal((batch, in_f)).astype(np.float32)
    return L, R, x


def _reference_gemv(L: np.ndarray, R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact reference: x @ (L @ R)ᵀ in fp64 for numerical correctness checks."""
    L64 = L.astype(np.float64)
    R64 = R.astype(np.float64)
    x64 = x.astype(np.float64)
    return (x64 @ (L64 @ R64).T).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. NumPy fallback — correctness
# ---------------------------------------------------------------------------

class TestNumpyFallback:
    def test_output_shape_default(self):
        L, R, x = _make_factors_and_x(out_f=64, in_f=128, rank=16, batch=4)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.shape == (4, 64), f"expected (4, 64) got {out.shape}"

    def test_output_dtype_float32(self):
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=2)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.dtype == np.float32, f"expected float32, got {out.dtype}"

    def test_correctness_vs_reference_fp64(self):
        """NumPy fallback matches fp64 reference within fp16 factor round-trip error."""
        L, R, x = _make_factors_and_x(out_f=48, in_f=96, rank=12, batch=6, seed=1)
        out_np  = _sqint2_residual_gemv_numpy(L, R, x)
        out_ref = _reference_gemv(L, R, x)
        # fp16 factors → fp32 round-trip error. Max abs err ≪ 1e-4 on σ=0.02 data.
        max_err = float(np.abs(out_np - out_ref).max())
        assert max_err < 5e-4, f"NumPy fallback max abs error {max_err:.2e} > 5e-4"

    def test_batch_1(self):
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=1)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.shape == (1, 32)

    def test_rank_1(self):
        """Rank-1 residual: degenerate but valid — must not crash."""
        L, R, x = _make_factors_and_x(out_f=16, in_f=32, rank=1, batch=3)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.shape == (3, 16)
        assert out.dtype == np.float32

    def test_rank_16_production(self):
        """Production SQINT2 rank=16 — checks result is non-trivially non-zero."""
        L, R, x = _make_factors_and_x(out_f=128, in_f=256, rank=16, batch=4)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.shape == (4, 128)
        # Low-rank product of non-zero factors and non-zero x is non-zero.
        assert float(np.abs(out).max()) > 0.0, "expected non-zero output"

    def test_nonsquare_in_ne_out(self):
        """in_f ≠ out_f — transformer FFN dims are always rectangular."""
        L, R, x = _make_factors_and_x(out_f=200, in_f=50, rank=8, batch=2)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        assert out.shape == (2, 200)

    def test_fp32_accumulation_no_fp16_matmul(self):
        """Verify fp16 factors are upcast inside the fallback (not matmuled as fp16)."""
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=2)
        # Build expected output manually using explicit fp32 upcasting
        L32 = L.astype(np.float32)
        R32 = R.astype(np.float32)
        x32 = x.astype(np.float32)
        expected = x32 @ (L32 @ R32).T
        got = _sqint2_residual_gemv_numpy(L, R, x)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_zero_L_gives_zero_output(self):
        rng = np.random.default_rng(99)
        L = np.zeros((32, 8), dtype=np.float16)
        R = (rng.standard_normal((8, 64)) * 0.02).astype(np.float16)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_zero_R_gives_zero_output(self):
        rng = np.random.default_rng(100)
        L = (rng.standard_normal((32, 8)) * 0.02).astype(np.float16)
        R = np.zeros((8, 64), dtype=np.float16)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_zero_x_gives_zero_output(self):
        rng = np.random.default_rng(101)
        L = (rng.standard_normal((32, 8)) * 0.02).astype(np.float16)
        R = (rng.standard_normal((8, 64)) * 0.02).astype(np.float16)
        x = np.zeros((4, 64), dtype=np.float32)
        out = _sqint2_residual_gemv_numpy(L, R, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Public API — dtype coercion and shape contract
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_accepts_float16_L_R(self):
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=2)
        # Should not raise — float16 is the canonical dtype
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (2, 32)
        assert out.dtype == np.float32

    def test_accepts_float32_L_R_coerces_to_fp16(self):
        """float32 L/R passed in — must be auto-downcast to fp16 before GEMV."""
        rng = np.random.default_rng(42)
        L = (rng.standard_normal((32, 8)) * 0.02).astype(np.float32)  # fp32, not fp16
        R = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        # Internally the public API calls ascontiguousarray(..., dtype=np.float16)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (4, 32)
        assert out.dtype == np.float32

    def test_accepts_float64_x_coerces_to_fp32(self):
        """float64 x passed in — must be auto-downcast to fp32."""
        L, R, _ = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=2)
        rng = np.random.default_rng(9)
        x64 = rng.standard_normal((2, 64))  # float64
        out = sqint2_residual_gemv(L, R, x64)
        assert out.dtype == np.float32

    def test_non_contiguous_x_coerced(self):
        """Non-contiguous x slice must be coerced to contiguous before Rust call."""
        rng = np.random.default_rng(11)
        big = rng.standard_normal((8, 64)).astype(np.float32)
        x_nc = big[::2]  # non-contiguous stride
        L, R, _ = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=4)
        out = sqint2_residual_gemv(L, R, x_nc)
        assert out.shape == (4, 32)

    def test_large_batch(self):
        L, R, _ = _make_factors_and_x(out_f=64, in_f=128, rank=16, batch=1)
        rng = np.random.default_rng(12)
        x = rng.standard_normal((32, 128)).astype(np.float32)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (32, 64)

    def test_batch_1_vector(self):
        L, R, x = _make_factors_and_x(out_f=64, in_f=128, rank=16, batch=1)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# 3. Rust kernel (skip when not built)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUST_AVAILABLE, reason="squish_quant Rust extension not built")
class TestRustKernel:
    def test_rust_matches_numpy_fallback(self):
        """Rust kernel output must match NumPy fallback to within float round-trip."""
        L, R, x = _make_factors_and_x(out_f=64, in_f=128, rank=16, batch=4, seed=0xAB)
        out_np   = _sqint2_residual_gemv_numpy(L, R, x)
        out_rust = sqint2_residual_gemv(L, R, x)
        # uint16 bit-view round-trip is lossless so both paths see identical fp16 bits.
        # Small divergence from FP ordering is expected at <1e-5 relative tolerance.
        np.testing.assert_allclose(out_rust, out_np, rtol=1e-5, atol=1e-6,
                                   err_msg="Rust residual GEMV diverges from NumPy fallback")

    def test_rust_output_shape(self):
        L, R, x = _make_factors_and_x(out_f=48, in_f=96, rank=12, batch=8)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (8, 48)

    def test_rust_output_dtype_float32(self):
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=2)
        out = sqint2_residual_gemv(L, R, x)
        assert out.dtype == np.float32

    def test_rust_batch_1(self):
        L, R, x = _make_factors_and_x(out_f=32, in_f=64, rank=8, batch=1)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (1, 32)

    def test_rust_rank_1(self):
        L, R, x = _make_factors_and_x(out_f=16, in_f=32, rank=1, batch=3)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (3, 16)

    def test_rust_large_batch(self):
        L, R, _ = _make_factors_and_x(out_f=64, in_f=128, rank=16)
        rng = np.random.default_rng(13)
        x = rng.standard_normal((32, 128)).astype(np.float32)
        out = sqint2_residual_gemv(L, R, x)
        assert out.shape == (32, 64)

    def test_rust_zero_factors_give_zero(self):
        rng = np.random.default_rng(14)
        L = np.zeros((32, 8), dtype=np.float16)
        R = np.zeros((8, 64), dtype=np.float16)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        out = sqint2_residual_gemv(L, R, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. get_backend_info() — key presence
# ---------------------------------------------------------------------------

class TestBackendInfo:
    def test_sqint2_residual_gemv_rust_key_present(self):
        info = get_backend_info()
        assert "sqint2_residual_gemv_rust" in info, (
            "get_backend_info() must report 'sqint2_residual_gemv_rust' key"
        )

    def test_sqint2_residual_gemv_rust_matches_availability(self):
        info = get_backend_info()
        assert info["sqint2_residual_gemv_rust"] == RUST_AVAILABLE, (
            f"sqint2_residual_gemv_rust={info['sqint2_residual_gemv_rust']} "
            f"but RUST_AVAILABLE={RUST_AVAILABLE}"
        )

    def test_numpy_always_true(self):
        assert get_backend_info()["numpy"] is True

    def test_all_required_keys_present(self):
        info = get_backend_info()
        required = {"squish_quant_rust", "int4_matmul_rust",
                    "sqint2_residual_gemv_rust", "numpy"}
        missing = required - set(info.keys())
        assert not missing, f"get_backend_info() missing keys: {missing}"


# ---------------------------------------------------------------------------
# 5. Integration with SQINT2Layer residual fields
# ---------------------------------------------------------------------------

class TestIntegrationWithSQINT2Layer:
    """Verify sqint2_residual_gemv produces results consistent with the
    decompress_weight path when residual_rank > 0.

    We compress a small synthetic weight, inspect the stored L and R, then
    compare:
      a) decompress_weight (reference full-matrix path)
      b) sqint2_residual_gemv applied to the L·R residual only

    The test is NOT a drop-in inference test (we don't call the inverse
    Hadamard here); we verify that the GEMV output equals x @ (L@R)ᵀ as
    computed by the NumPy path — that confirms the residual GEMV has the
    correct numerical role in the inference pipeline.
    """

    def test_residual_gemv_matches_explicit_lowrank_product(self):
        from squish.quant.sqint2 import compress_weight, SQINT2Config

        rng = np.random.default_rng(0xBEEF)
        W = (rng.standard_normal((32, 64)) * 0.02).astype(np.float32)
        cfg = SQINT2Config(group_size=32, residual_rank=4, refine_iters=1)
        layer = compress_weight(W, cfg)

        assert layer.residual_L is not None, "expected non-None residual_L after compress"
        assert layer.residual_R is not None

        x = rng.standard_normal((3, 64)).astype(np.float32)

        # Reference: explicit (L @ R)ᵀ matmul in fp32
        L32 = layer.residual_L.astype(np.float32)
        R32 = layer.residual_R.astype(np.float32)
        expected = x @ (L32 @ R32).T   # (3, 32)

        # sqint2_residual_gemv path (NumPy fallback in this test context)
        got = _sqint2_residual_gemv_numpy(layer.residual_L, layer.residual_R, x)

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5,
                                   err_msg="residual GEMV diverges from explicit L@R product")

    def test_residual_gemv_output_shape_matches_weight_dims(self):
        from squish.quant.sqint2 import compress_weight, SQINT2Config

        rng = np.random.default_rng(0xCAFE)
        out_f, in_f = 48, 96
        W = (rng.standard_normal((out_f, in_f)) * 0.02).astype(np.float32)
        cfg = SQINT2Config(group_size=32, residual_rank=8, refine_iters=0)
        layer = compress_weight(W, cfg)

        x = rng.standard_normal((5, in_f)).astype(np.float32)
        out = sqint2_residual_gemv(layer.residual_L, layer.residual_R, x)
        assert out.shape == (5, out_f), (
            f"residual GEMV shape {out.shape} != (batch={5}, out_f={out_f})"
        )

    def test_residual_gemv_improves_snr_vs_base_int2(self):
        """residual GEMV contribution is non-negligible on real-distribution weights.

        Compress with residual_rank=8; compute the SNR lift of the residual term
        against the Stage-1+2 base. The residual correction should contribute
        a non-zero SNR lift (however small at rank=8 on IID Gaussian).
        """
        from squish.quant.sqint2 import compress_weight, decompress_weight, snr_db, SQINT2Config

        rng = np.random.default_rng(0x1234)
        W = (rng.standard_normal((64, 128)) * 0.02).astype(np.float32)

        # Stage 1+2 only (no residual)
        cfg_base = SQINT2Config(group_size=32, residual_rank=0, refine_iters=1)
        layer_base = compress_weight(W, cfg_base)
        W_base = decompress_weight(layer_base)
        snr_base = snr_db(W, W_base)

        # Stage 1+2+3 (with residual rank=8)
        cfg_res = SQINT2Config(group_size=32, residual_rank=8, refine_iters=1)
        layer_res = compress_weight(W, cfg_res)
        W_res = decompress_weight(layer_res)
        snr_res = snr_db(W, W_res)

        # The residual must improve SNR by at least 0.1 dB
        delta = snr_res - snr_base
        assert delta >= 0.1, (
            f"residual rank-8 failed to improve SNR: Δ={delta:.3f} dB "
            f"(base={snr_base:.2f} dB, residual={snr_res:.2f} dB)"
        )
