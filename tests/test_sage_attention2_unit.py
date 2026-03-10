"""Unit tests for squish.sage_attention2 (SageAttention2 INT4 + FP8)."""

import math

import numpy as np
import pytest

from squish.sage_attention2 import (
    SageAttention2Config,
    SageAttention2Kernel,
    SageAttention2Stats,
    WarpQuantResult,
    _fp8_simulate,
    _reconstruct,
    simulate_sage2_attention,
    warp_quantize_int4,
)


def _cfg(head_dim=32, n_heads=2, block_size=32, warp_size=8, **kw):
    return SageAttention2Config(
        head_dim=head_dim, n_heads=n_heads, block_size=block_size, warp_size=warp_size, **kw
    )


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# TestSageAttention2Config
# ---------------------------------------------------------------------------


class TestSageAttention2Config:
    def test_defaults(self):
        cfg = SageAttention2Config()
        assert cfg.head_dim == 128
        assert cfg.n_heads == 32
        assert cfg.use_int4 is True
        assert cfg.use_fp8_pv is True

    def test_active_qk_bits_int4(self):
        cfg = _cfg()
        assert cfg.active_qk_bits == 4

    def test_active_qk_bits_int8(self):
        cfg = _cfg(use_int4=False)
        assert cfg.active_qk_bits == 8

    def test_active_pv_bits_fp8(self):
        cfg = _cfg()
        assert cfg.active_pv_bits == 8

    def test_invalid_block_size_lt_warp(self):
        with pytest.raises(ValueError, match="block_size"):
            SageAttention2Config(block_size=4, warp_size=8)

    def test_invalid_smooth_alpha(self):
        with pytest.raises(ValueError):
            SageAttention2Config(smooth_alpha=0.0)


# ---------------------------------------------------------------------------
# TestWarpQuantizeInt4
# ---------------------------------------------------------------------------


class TestWarpQuantizeInt4:
    def test_int4_used_for_small_values(self):
        x = np.ones((32,), dtype=np.float32) * 3.0
        result = warp_quantize_int4(x, warp_size=8, fallback_threshold=6.5)
        assert result.used_int4 is True

    def test_int8_fallback_for_large_values(self):
        x = np.ones((32,), dtype=np.float32) * 50.0
        result = warp_quantize_int4(x, warp_size=8, fallback_threshold=6.5)
        assert result.used_int4 is False

    def test_scales_shape(self):
        x = np.ones((32,), dtype=np.float32)
        result = warp_quantize_int4(x, warp_size=8, fallback_threshold=6.5)
        assert result.warp_scales.shape == (4,)  # 32/8 = 4 warps

    def test_quantized_range_int4(self):
        x = np.ones((16,), dtype=np.float32) * 2.0
        result = warp_quantize_int4(x, warp_size=4, fallback_threshold=6.5)
        assert result.data_int.max() <= 7

    def test_reconstruction_approx(self):
        rng = _rng()
        x = rng.normal(0, 1, (32,)).astype(np.float32)
        result = warp_quantize_int4(x, warp_size=8, fallback_threshold=6.5)
        recon = _reconstruct(result, 8)
        err = np.abs(recon.ravel()[: x.size] - x).max()
        assert err < 2.0  # coarse: INT4 resolution


# ---------------------------------------------------------------------------
# TestFP8Simulate
# ---------------------------------------------------------------------------


class TestFP8Simulate:
    def test_output_shape(self):
        x = np.ones((4, 8), dtype=np.float32)
        y = _fp8_simulate(x)
        assert y.shape == x.shape

    def test_clamped_range(self):
        x = np.array([[1000.0, -1000.0]], dtype=np.float32)
        y = _fp8_simulate(x)
        assert np.abs(y).max() <= 448.0

    def test_near_identity_small_values(self):
        x = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        y = _fp8_simulate(x)
        # FP8 should not change small values too drastically
        assert np.abs(y - x).max() < 0.5


# ---------------------------------------------------------------------------
# TestSimulateSage2Attention
# ---------------------------------------------------------------------------


class TestSimulateSage2Attention:
    def _inputs(self, n_heads=2, seq=16, head_dim=32, seed=5):
        rng = _rng(seed)
        q = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        k = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        v = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        return q, k, v

    def test_output_shape(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        out, stats = simulate_sage2_attention(q, k, v, cfg)
        assert out.shape == (2, 16, 32)

    def test_output_finite(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        out, _ = simulate_sage2_attention(q, k, v, cfg)
        assert np.isfinite(out).all()

    def test_stats_total_blocks_positive(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        _, stats = simulate_sage2_attention(q, k, v, cfg)
        assert stats.total_blocks > 0

    def test_int4_rate_in_range(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        _, stats = simulate_sage2_attention(q, k, v, cfg)
        assert 0.0 <= stats.int4_rate <= 1.0

    def test_with_k_scales(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        k_scales = np.ones(32, dtype=np.float32)
        out, _ = simulate_sage2_attention(q, k, v, cfg, k_scales=k_scales)
        assert out.shape == (2, 16, 32)


# ---------------------------------------------------------------------------
# TestSageAttention2Stats
# ---------------------------------------------------------------------------


class TestSageAttention2Stats:
    def test_int4_rate(self):
        s = SageAttention2Stats(total_blocks=10, int4_blocks=7, int8_fallback_blocks=3)
        assert math.isclose(s.int4_rate, 0.7)

    def test_int8_rate(self):
        s = SageAttention2Stats(total_blocks=10, int4_blocks=7, int8_fallback_blocks=3)
        assert math.isclose(s.int8_rate, 0.3)

    def test_speedup_all_int4(self):
        s = SageAttention2Stats(total_blocks=100, int4_blocks=100, int8_fallback_blocks=0)
        assert math.isclose(s.estimated_speedup_vs_fa2, 3.1, rel_tol=0.01)

    def test_speedup_all_int8(self):
        s = SageAttention2Stats(total_blocks=100, int4_blocks=0, int8_fallback_blocks=100)
        assert math.isclose(s.estimated_speedup_vs_fa2, 2.1, rel_tol=0.01)

    def test_merge(self):
        s1 = SageAttention2Stats(total_blocks=5, int4_blocks=3, int8_fallback_blocks=2)
        s2 = SageAttention2Stats(total_blocks=5, int4_blocks=4, int8_fallback_blocks=1)
        merged = s1.merge(s2)
        assert merged.total_blocks == 10
        assert merged.int4_blocks == 7

    def test_zero_blocks_speedup(self):
        s = SageAttention2Stats(total_blocks=0)
        assert s.estimated_speedup_vs_fa2 >= 1.0


# ---------------------------------------------------------------------------
# TestSageAttention2Kernel
# ---------------------------------------------------------------------------


class TestSageAttention2Kernel:
    def _inputs(self, n_heads=2, seq=16, head_dim=32, seed=3):
        rng = _rng(seed)
        q = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        k = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        v = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        return q, k, v

    def test_forward_shape(self):
        cfg = _cfg()
        kernel = SageAttention2Kernel(cfg)
        out, _ = kernel.forward(*self._inputs())
        assert out.shape == (2, 16, 32)

    def test_cumulative_stats_grow(self):
        cfg = _cfg()
        kernel = SageAttention2Kernel(cfg)
        kernel.forward(*self._inputs())
        kernel.forward(*self._inputs())
        assert kernel.cumulative_stats.total_blocks > 0

    def test_reset_clears_scales_and_stats(self):
        cfg = _cfg()
        kernel = SageAttention2Kernel(cfg)
        kernel.forward(*self._inputs())
        kernel.reset()
        assert kernel._k_scales is None
        assert kernel.cumulative_stats.total_blocks == 0

    def test_output_finite(self):
        cfg = _cfg()
        kernel = SageAttention2Kernel(cfg)
        out, _ = kernel.forward(*self._inputs())
        assert np.isfinite(out).all()
