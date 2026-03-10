"""Unit tests for squish.sage_attention (SageAttention INT8 quantized attention)."""

import math

import numpy as np
import pytest

from squish.sage_attention import (
    KSmoother,
    SageAttentionConfig,
    SageAttentionKernel,
    SageAttentionStats,
    _block_split,
    _dequantize,
    _quantize_to_int8,
    simulate_sage_qk,
)


def _cfg(head_dim=64, n_heads=4, block_size=16, **kw):
    return SageAttentionConfig(head_dim=head_dim, n_heads=n_heads, block_size=block_size, **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# TestSageAttentionConfig
# ---------------------------------------------------------------------------


class TestSageAttentionConfig:
    def test_defaults(self):
        cfg = SageAttentionConfig()
        assert cfg.head_dim == 128
        assert cfg.n_heads == 32
        assert cfg.block_size == 64
        assert cfg.smooth_k is True
        assert cfg.qk_bits == 8

    def test_scale(self):
        cfg = _cfg(head_dim=64)
        assert math.isclose(cfg.scale, 1.0 / math.sqrt(64), rel_tol=1e-6)

    def test_qk_clamp_int8(self):
        cfg = _cfg(qk_bits=8)
        assert cfg.qk_clamp == 127.0

    def test_qk_clamp_int4(self):
        cfg = _cfg(qk_bits=4)
        assert cfg.qk_clamp == 7.0

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            SageAttentionConfig(head_dim=0)

    def test_invalid_qk_bits(self):
        with pytest.raises(ValueError, match="qk_bits"):
            SageAttentionConfig(qk_bits=16)

    def test_invalid_pv_bits(self):
        with pytest.raises(ValueError, match="pv_bits"):
            SageAttentionConfig(pv_bits=4)

    def test_invalid_smooth_alpha(self):
        with pytest.raises(ValueError, match="smooth_alpha"):
            SageAttentionConfig(smooth_alpha=0.0)


# ---------------------------------------------------------------------------
# TestQuantizeToInt8
# ---------------------------------------------------------------------------


class TestQuantizeToInt8:
    def test_output_dtype(self):
        rng = _rng()
        x = rng.normal(0, 1, (4, 8)).astype(np.float32)
        q, s = _quantize_to_int8(x)
        assert q.dtype == np.int8
        assert s.dtype == np.float32

    def test_range_clamped(self):
        x = np.array([[200.0, -300.0, 0.0]], dtype=np.float32)
        q, _ = _quantize_to_int8(x, clamp=127.0)
        assert q.max() <= 127 and q.min() >= -127

    def test_dequantize_roundtrip(self):
        rng = _rng(1)
        x = rng.normal(0, 1, (8, 16)).astype(np.float32)
        q, s = _quantize_to_int8(x)
        recon = _dequantize(q, s)
        # MAX relative error vs scale should be small
        err = np.abs(recon - x).max()
        assert err < 1.0  # coarse check; INT8 precision is 1/127 of range

    def test_scales_shape(self):
        x = np.ones((3, 10), dtype=np.float32)
        _, s = _quantize_to_int8(x)
        assert s.shape == (3, 1)


# ---------------------------------------------------------------------------
# TestBlockSplit
# ---------------------------------------------------------------------------


class TestBlockSplit:
    def test_exact_split(self):
        x = np.ones((8, 4), dtype=np.float32)
        blocks = _block_split(x, 4)
        assert len(blocks) == 2
        assert blocks[0].shape == (4, 4)

    def test_partial_last_block(self):
        x = np.ones((10, 4), dtype=np.float32)
        blocks = _block_split(x, 4)
        assert len(blocks) == 3
        assert blocks[-1].shape == (2, 4)

    def test_single_block(self):
        x = np.ones((5, 4))
        blocks = _block_split(x, 64)
        assert len(blocks) == 1


# ---------------------------------------------------------------------------
# TestKSmoother
# ---------------------------------------------------------------------------


class TestKSmoother:
    def test_first_update_sets_scale(self):
        cfg = _cfg()
        smoother = KSmoother(config=cfg)
        k = np.ones((32, 64), dtype=np.float32) * 2.0
        _, scales = smoother.update_and_smooth(k)
        assert scales is not None
        assert scales.shape == (64,)

    def test_ema_decreases_towards_lower(self):
        cfg = _cfg(smooth_alpha=0.5)
        smoother = KSmoother(config=cfg)
        k_big = np.ones((16, 64)) * 10.0
        smoother.update_and_smooth(k_big)
        k_small = np.ones((16, 64)) * 2.0
        _, scales = smoother.update_and_smooth(k_small)
        assert scales.max() < 10.0  # should have moved toward 2.0

    def test_reset_clears_state(self):
        cfg = _cfg()
        smoother = KSmoother(config=cfg)
        k = np.ones((16, 64))
        smoother.update_and_smooth(k)
        smoother.reset()
        assert smoother._scales is None


# ---------------------------------------------------------------------------
# TestSimulateSageQK
# ---------------------------------------------------------------------------


class TestSimulateSageQK:
    def _make_qk(self, n_heads=2, seq_q=16, seq_k=16, head_dim=32, seed=42):
        rng = np.random.default_rng(seed)
        q = rng.normal(0, 0.5, (n_heads, seq_q, head_dim)).astype(np.float32)
        k = rng.normal(0, 0.5, (n_heads, seq_k, head_dim)).astype(np.float32)
        return q, k

    def test_output_shape(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        q, k = self._make_qk()
        logits, stats = simulate_sage_qk(q, k, cfg)
        assert logits.shape == (2, 16, 16)

    def test_stats_total_blocks_positive(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        q, k = self._make_qk()
        _, stats = simulate_sage_qk(q, k, cfg)
        assert stats.total_blocks > 0

    def test_fallback_rate_in_range(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8, fallback_threshold=0.0)
        q, k = self._make_qk()
        _, stats = simulate_sage_qk(q, k, cfg)
        # With threshold=0, all blocks should fall back
        assert stats.fallback_rate == 1.0

    def test_no_fallback_normal_input(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8, fallback_threshold=1000.0)
        q, k = self._make_qk()
        _, stats = simulate_sage_qk(q, k, cfg)
        assert stats.fallback_rate == 0.0

    def test_logits_finite(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        q, k = self._make_qk()
        logits, _ = simulate_sage_qk(q, k, cfg)
        assert np.isfinite(logits).all()


# ---------------------------------------------------------------------------
# TestSageAttentionStats
# ---------------------------------------------------------------------------


class TestSageAttentionStats:
    def test_fallback_rate(self):
        s = SageAttentionStats(total_blocks=10, fallback_blocks=3)
        assert math.isclose(s.fallback_rate, 0.3)

    def test_int_compute_fraction(self):
        s = SageAttentionStats(total_blocks=10, fallback_blocks=3)
        assert math.isclose(s.int_compute_fraction, 0.7)

    def test_speedup_zero_blocks(self):
        s = SageAttentionStats(total_blocks=0)
        assert s.estimated_speedup_vs_fp16 >= 1.0

    def test_merge(self):
        s1 = SageAttentionStats(total_blocks=10, fallback_blocks=2)
        s2 = SageAttentionStats(total_blocks=5, fallback_blocks=1)
        merged = s1.merge(s2)
        assert merged.total_blocks == 15
        assert merged.fallback_blocks == 3

    def test_speedup_int8_gt1(self):
        s = SageAttentionStats(total_blocks=100, fallback_blocks=0, qk_bits=8)
        assert s.estimated_speedup_vs_fp16 > 1.0


# ---------------------------------------------------------------------------
# TestSageAttentionKernel
# ---------------------------------------------------------------------------


class TestSageAttentionKernel:
    def _inputs(self, n_heads=2, seq=16, head_dim=32, seed=7):
        rng = np.random.default_rng(seed)
        q = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        k = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        v = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        return q, k, v

    def test_forward_output_shape(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        kernel = SageAttentionKernel(cfg)
        q, k, v = self._inputs()
        out, _ = kernel.forward(q, k, v)
        assert out.shape == (2, 16, 32)

    def test_cumulative_stats_accumulate(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        kernel = SageAttentionKernel(cfg)
        q, k, v = self._inputs()
        kernel.forward(q, k, v)
        kernel.forward(q, k, v)
        assert kernel.cumulative_stats.total_blocks > 0

    def test_reset_stats(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        kernel = SageAttentionKernel(cfg)
        q, k, v = self._inputs()
        kernel.forward(q, k, v)
        kernel.reset_stats()
        assert kernel.cumulative_stats.total_blocks == 0

    def test_output_finite(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        kernel = SageAttentionKernel(cfg)
        q, k, v = self._inputs()
        out, _ = kernel.forward(q, k, v)
        assert np.isfinite(out).all()

    def test_reset_smoother_clears(self):
        cfg = _cfg(head_dim=32, n_heads=2, block_size=8)
        kernel = SageAttentionKernel(cfg)
        q, k, v = self._inputs()
        kernel.forward(q, k, v)
        kernel.reset_smoother()
        assert kernel._smoother._scales is None
