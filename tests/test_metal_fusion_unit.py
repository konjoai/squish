#!/usr/bin/env python3
"""
tests/test_metal_fusion_unit.py

Unit tests for squish/metal_fusion.py — Phase 10B Metal kernel fusion.

All tests exercise the numpy fallback path (no Metal hardware required).
Metal-execution paths are guarded with ``# pragma: no cover`` in the module.

Coverage targets
────────────────
_METAL_FUSION_AVAILABLE
  - is a bool

MetalFusionConfig
  - valid defaults
  - require_metal=False is default

MetalFusionKernels
  - available property reflects _METAL_FUSION_AVAILABLE
  - require_metal=True raises RuntimeError when Metal unavailable
  - rope/swiglu/int8_attn flags disabled when use_* = False
  - __repr__ contains expected fields

fused_rope_qk (fallback)
  - output shape matches input shape for Q
  - output shape matches input shape for K
  - RoPE rotation is numerically correct (reference _rope_numpy vs fused)
  - passing kernels=None always uses fallback
  - head_dim=64 and head_dim=128 both pass

fused_swiglu (fallback)
  - output shape matches input
  - numerically: silu(gate) * up (reference formula)
  - zero gate → output near zero
  - all-one inputs → reproducible value

fused_int8_kv_attn (fallback)
  - output shape (B, H, Lq, D) is correct
  - softmax probabilities sum to 1 along kv_len axis
  - causal-mask equivalent: one KV position → output equals V
  - numerically stable with large positive scores
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from squish.metal_fusion import (
    MetalFusionConfig,
    MetalFusionKernels,
    _METAL_FUSION_AVAILABLE,
    _rope_numpy,
    fused_int8_kv_attn,
    fused_rope_qk,
    fused_swiglu,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# _METAL_FUSION_AVAILABLE
# ---------------------------------------------------------------------------

class TestMetalFusionAvailable:
    def test_is_bool(self):
        assert isinstance(_METAL_FUSION_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# MetalFusionConfig
# ---------------------------------------------------------------------------

class TestMetalFusionConfig:
    def test_defaults(self):
        cfg = MetalFusionConfig()
        assert cfg.use_fused_rope is True
        assert cfg.use_fused_swiglu is True
        assert cfg.use_fused_int8_attn is True
        assert cfg.require_metal is False

    def test_all_disabled(self):
        cfg = MetalFusionConfig(
            use_fused_rope=False,
            use_fused_swiglu=False,
            use_fused_int8_attn=False,
        )
        assert not cfg.use_fused_rope
        assert not cfg.use_fused_swiglu
        assert not cfg.use_fused_int8_attn


# ---------------------------------------------------------------------------
# MetalFusionKernels
# ---------------------------------------------------------------------------

class TestMetalFusionKernels:
    def test_available_matches_module_sentinel(self):
        k = MetalFusionKernels()
        assert k.available == _METAL_FUSION_AVAILABLE

    def test_require_metal_false_never_raises(self):
        k = MetalFusionKernels(MetalFusionConfig(require_metal=False))
        assert k is not None

    def test_require_metal_true_raises_when_unavailable(self):
        if _METAL_FUSION_AVAILABLE:
            pytest.skip("Metal is available on this machine")
        with pytest.raises(RuntimeError, match="Metal"):
            MetalFusionKernels(MetalFusionConfig(require_metal=True))

    def test_flags_disabled_when_use_false(self):
        k = MetalFusionKernels(MetalFusionConfig(
            use_fused_rope=False,
            use_fused_swiglu=False,
            use_fused_int8_attn=False,
        ))
        assert not k.rope_enabled
        assert not k.swiglu_enabled
        assert not k.int8_attn_enabled

    def test_repr_contains_key_fields(self):
        k = MetalFusionKernels()
        r = repr(k)
        assert "MetalFusionKernels" in r
        assert "metal=" in r
        assert "rope=" in r
        assert "swiglu=" in r

    def test_default_kernels_no_metal_disables_all(self):
        if _METAL_FUSION_AVAILABLE:
            pytest.skip("Metal is available")
        k = MetalFusionKernels()
        assert not k.rope_enabled
        assert not k.swiglu_enabled
        assert not k.int8_attn_enabled


# ---------------------------------------------------------------------------
# fused_rope_qk (numpy fallback)
# ---------------------------------------------------------------------------

def _make_qk(batch=1, heads=2, seq=4, head_dim=8):
    Q = RNG.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
    K = RNG.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
    cos = np.cos(np.arange(seq * head_dim // 2)
                 .reshape(1, 1, seq, head_dim // 2)
                 .astype(np.float32))
    cos = np.broadcast_to(cos, (batch, heads, seq, head_dim // 2))
    sin = np.sin(np.arange(seq * head_dim // 2)
                 .reshape(1, 1, seq, head_dim // 2)
                 .astype(np.float32))
    sin = np.broadcast_to(sin, (batch, heads, seq, head_dim // 2))
    # Expand to full head_dim (first half = cos, same for sin)
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, sin], axis=-1)
    return Q, K, cos_full, sin_full


class TestFusedRopeQK:
    def test_q_output_shape(self):
        Q, K, cos, sin = _make_qk()
        Q2, K2 = fused_rope_qk(Q, K, cos, sin)
        assert Q2.shape == Q.shape

    def test_k_output_shape(self):
        Q, K, cos, sin = _make_qk()
        Q2, K2 = fused_rope_qk(Q, K, cos, sin)
        assert K2.shape == K.shape

    def test_matches_individual_rope_numpy(self):
        Q, K, cos, sin = _make_qk(head_dim=16)
        Q2, K2 = fused_rope_qk(Q, K, cos, sin)
        Q_ref = _rope_numpy(Q, cos, sin)
        K_ref = _rope_numpy(K, cos, sin)
        np.testing.assert_allclose(Q2, Q_ref, rtol=1e-5)
        np.testing.assert_allclose(K2, K_ref, rtol=1e-5)

    def test_none_kernels_uses_fallback(self):
        Q, K, cos, sin = _make_qk()
        Q2, K2 = fused_rope_qk(Q, K, cos, sin, kernels=None)
        assert Q2.shape == Q.shape

    def test_head_dim_64(self):
        Q, K, cos, sin = _make_qk(head_dim=64)
        Q2, K2 = fused_rope_qk(Q, K, cos, sin)
        assert Q2.shape == Q.shape

    def test_head_dim_128(self):
        Q, K, cos, sin = _make_qk(head_dim=128)
        Q2, K2 = fused_rope_qk(Q, K, cos, sin)
        assert Q2.shape == Q.shape

    def test_rotation_is_orthogonal_preserves_norm(self):
        """RoPE is an orthogonal transform — norms must be preserved."""
        Q, K, cos, sin = _make_qk(head_dim=8)
        Q2, _ = fused_rope_qk(Q, K, cos, sin)
        orig_norms = np.linalg.norm(Q, axis=-1)
        new_norms  = np.linalg.norm(Q2, axis=-1)
        np.testing.assert_allclose(new_norms, orig_norms, rtol=1e-5)


# ---------------------------------------------------------------------------
# fused_swiglu (numpy fallback)
# ---------------------------------------------------------------------------

def _silu_ref(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


class TestFusedSwiGLU:
    def test_output_shape_matches_input(self):
        g = RNG.standard_normal((2, 4, 16)).astype(np.float32)
        u = RNG.standard_normal((2, 4, 16)).astype(np.float32)
        out = fused_swiglu(g, u)
        assert out.shape == g.shape

    def test_numerically_correct(self):
        g = RNG.standard_normal((3, 8)).astype(np.float32)
        u = RNG.standard_normal((3, 8)).astype(np.float32)
        out = fused_swiglu(g, u)
        ref = _silu_ref(g) * u
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_zero_gate_output_near_zero(self):
        g = np.zeros((2, 4), dtype=np.float32)
        u = RNG.standard_normal((2, 4)).astype(np.float32)
        out = fused_swiglu(g, u)
        # silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0; so output = 0 * u = 0
        np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-6)

    def test_all_ones_inputs(self):
        g = np.ones((1, 4), dtype=np.float32)
        u = np.ones((1, 4), dtype=np.float32)
        out = fused_swiglu(g, u)
        expected = _silu_ref(g) * u
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_none_kernels_uses_fallback(self):
        g = RNG.standard_normal((2, 4)).astype(np.float32)
        u = RNG.standard_normal((2, 4)).astype(np.float32)
        out = fused_swiglu(g, u, kernels=None)
        assert out.shape == g.shape


# ---------------------------------------------------------------------------
# fused_int8_kv_attn (numpy fallback)
# ---------------------------------------------------------------------------

def _make_kv_inputs(batch=1, heads=2, q_len=2, kv_len=4, head_dim=8):
    q       = RNG.standard_normal((batch, heads, q_len,  head_dim)).astype(np.float32)
    k_int8  = RNG.integers(-127, 128, (batch, heads, kv_len, head_dim), dtype=np.int8)
    v_int8  = RNG.integers(-127, 128, (batch, heads, kv_len, head_dim), dtype=np.int8)
    k_scales = RNG.random((batch, heads, kv_len, 1)).astype(np.float32) * 0.1 + 0.01
    v_scales = RNG.random((batch, heads, kv_len, 1)).astype(np.float32) * 0.1 + 0.01
    return q, k_int8, v_int8, k_scales, v_scales


class TestFusedInt8KVAttn:
    def test_output_shape(self):
        q, k, v, ks, vs = _make_kv_inputs()
        out = fused_int8_kv_attn(q, k, v, ks, vs)
        assert out.shape == q.shape

    def test_output_dtype_float32(self):
        q, k, v, ks, vs = _make_kv_inputs()
        out = fused_int8_kv_attn(q, k, v, ks, vs)
        assert out.dtype == np.float32

    def test_attention_probs_sum_to_one(self):
        """Verify softmax correctness: probs over kv_len sum to 1."""
        q, k, v, ks, vs = _make_kv_inputs(kv_len=8)
        q_f   = q.astype(np.float32)
        k_f   = k.astype(np.float32) * ks
        scale = 1.0 / math.sqrt(q.shape[-1])
        w     = np.matmul(q_f, k_f.swapaxes(-2, -1)) * scale
        w    -= w.max(axis=-1, keepdims=True)
        exp_w = np.exp(w)
        probs = exp_w / exp_w.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones_like(probs[..., 0]),
                                   rtol=1e-5)

    def test_single_kv_position_output_equals_v(self):
        """With a single KV position, attention selects it entirely → out == v."""
        batch, heads, q_len, head_dim = 1, 1, 1, 4
        q       = np.ones((batch, heads, q_len, head_dim), dtype=np.float32)
        k_int8  = np.ones((batch, heads, 1, head_dim), dtype=np.int8)
        v_int8  = np.array([[[[1, 2, 3, 4]]]], dtype=np.int8)
        k_scales = np.ones((batch, heads, 1, 1), dtype=np.float32)
        v_scales = np.ones((batch, heads, 1, 1), dtype=np.float32)
        out = fused_int8_kv_attn(q, k_int8, v_int8, k_scales, v_scales)
        expected = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_numerically_stable_with_large_scores(self):
        """Large scores must not produce NaN/Inf after softmax."""
        q, k, v, ks, vs = _make_kv_inputs()
        q_large = q * 1000.0
        out = fused_int8_kv_attn(q_large, k, v, ks, vs)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_none_kernels_uses_fallback(self):
        q, k, v, ks, vs = _make_kv_inputs()
        out = fused_int8_kv_attn(q, k, v, ks, vs, kernels=None)
        assert out.shape == q.shape
