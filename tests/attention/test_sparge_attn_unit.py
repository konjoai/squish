"""Unit tests for squish.attention.sparge_attn (SpargeAttn sparse+quantized attention)."""

import numpy as np
import pytest

from squish.attention.sparge_attn import (
    BlockMask,
    SpargeAttnConfig,
    SpargeAttnEngine,
    SpargeAttnStats,
    _compress_k_block,
    _predict_block_importance,
    build_sparse_mask,
    sparge_attention_forward,
)


def _cfg(head_dim=32, n_heads=2, block_size=8, **kw):
    return SpargeAttnConfig(head_dim=head_dim, n_heads=n_heads, block_size=block_size, **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# TestSpargeAttnConfig
# ---------------------------------------------------------------------------


class TestSpargeAttnConfig:
    def test_defaults(self):
        cfg = SpargeAttnConfig()
        assert cfg.head_dim == 128
        assert cfg.n_heads == 32
        assert cfg.block_size == 64

    def test_k_repr_tokens(self):
        cfg = _cfg(block_size=8, k_compression_ratio=0.25)
        assert cfg.k_repr_tokens == 2

    def test_k_repr_tokens_min_one(self):
        cfg = _cfg(block_size=8, k_compression_ratio=0.01)
        assert cfg.k_repr_tokens >= 1

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            SpargeAttnConfig(head_dim=0)

    def test_invalid_k_compression_ratio(self):
        with pytest.raises(ValueError):
            SpargeAttnConfig(k_compression_ratio=0.0)

    def test_invalid_sparse_threshold(self):
        with pytest.raises(ValueError):
            SpargeAttnConfig(sparse_threshold=-0.1)


# ---------------------------------------------------------------------------
# TestCompressKBlock
# ---------------------------------------------------------------------------


class TestCompressKBlock:
    def test_no_compression_when_n_repr_ge_seq(self):
        k = np.ones((8, 32))
        result = _compress_k_block(k, n_repr=10)
        assert result.shape[0] == 8  # returned as-is

    def test_compression_reduces_seq(self):
        k = np.ones((16, 32))
        result = _compress_k_block(k, n_repr=4)
        assert result.shape == (4, 32)

    def test_shape_preserved(self):
        k = np.ones((16, 32))
        result = _compress_k_block(k, n_repr=4)
        assert result.shape[1] == 32

    def test_single_repr(self):
        k = np.ones((8, 32))
        result = _compress_k_block(k, n_repr=1)
        assert result.shape == (1, 32)


# ---------------------------------------------------------------------------
# TestPredictBlockImportance
# ---------------------------------------------------------------------------


class TestPredictBlockImportance:
    def test_returns_float(self):
        q = np.ones((4, 16), dtype=np.float32)
        k_repr = np.ones((2, 16), dtype=np.float32)
        val = _predict_block_importance(q, k_repr, scale=0.125)
        assert isinstance(val, float)

    def test_orthogonal_blocks_low_importance(self):
        q = np.zeros((4, 16), dtype=np.float32)
        k_repr = np.ones((2, 16), dtype=np.float32)
        val = _predict_block_importance(q, k_repr, scale=0.125)
        assert val == 0.0


# ---------------------------------------------------------------------------
# TestBlockMask
# ---------------------------------------------------------------------------


class TestBlockMask:
    def test_full_mask_density(self):
        mask = BlockMask.full(4, 4)
        assert mask.density == 1.0
        assert mask.sparsity == 0.0

    def test_partial_density(self):
        kept = np.array([[True, False], [True, True]], dtype=bool)
        mask = BlockMask(n_q_blocks=2, n_k_blocks=2, kept=kept)
        assert mask.density == 0.75


# ---------------------------------------------------------------------------
# TestBuildSparseMask
# ---------------------------------------------------------------------------


class TestBuildSparseMask:
    def _qk(self, seq_q=16, seq_k=16, head_dim=32, seed=0):
        rng = _rng(seed)
        q = rng.normal(0, 1, (seq_q, head_dim)).astype(np.float32)
        k = rng.normal(0, 1, (seq_k, head_dim)).astype(np.float32)
        return q, k

    def test_full_mask_with_zero_threshold(self):
        cfg = _cfg(sparse_threshold=0.0)
        q, k = self._qk()
        mask, skipped = build_sparse_mask(q, k, cfg)
        assert skipped == 0
        assert mask.density == 1.0

    def test_some_blocks_skipped_at_high_threshold(self):
        # With a very high threshold, most blocks will be skipped
        cfg = _cfg(sparse_threshold=1000.0)
        q, k = self._qk()
        mask, skipped = build_sparse_mask(q, k, cfg)
        assert skipped > 0

    def test_mask_shape_correct(self):
        cfg = _cfg(block_size=8)
        q = np.ones((16, 32))
        k = np.ones((16, 32))
        mask, _ = build_sparse_mask(q, k, cfg)
        assert mask.kept.shape == (mask.n_q_blocks, mask.n_k_blocks)


# ---------------------------------------------------------------------------
# TestSpargeAttnStats
# ---------------------------------------------------------------------------


class TestSpargeAttnStats:
    def test_total_skipped(self):
        s = SpargeAttnStats(total_blocks=10, stage1_skipped=3, stage2_skipped=2)
        assert s.total_skipped == 5

    def test_effective_sparsity(self):
        s = SpargeAttnStats(total_blocks=10, stage1_skipped=3, stage2_skipped=2)
        assert s.effective_sparsity == 0.5

    def test_estimated_speedup_no_sparsity(self):
        s = SpargeAttnStats(total_blocks=10, stage1_skipped=0, stage2_skipped=0)
        assert s.estimated_speedup == 1.0

    def test_estimated_speedup_half_sparse(self):
        s = SpargeAttnStats(total_blocks=10, stage1_skipped=5, stage2_skipped=0)
        assert s.estimated_speedup == pytest.approx(2.0)

    def test_merge(self):
        s1 = SpargeAttnStats(total_blocks=5, stage1_skipped=2, stage2_skipped=1)
        s2 = SpargeAttnStats(total_blocks=5, stage1_skipped=1, stage2_skipped=0)
        merged = s1.merge(s2)
        assert merged.total_blocks == 10
        assert merged.stage1_skipped == 3


# ---------------------------------------------------------------------------
# TestSpargeAttentionForward
# ---------------------------------------------------------------------------


class TestSpargeAttentionForward:
    def _inputs(self, n_heads=2, seq=16, head_dim=32, seed=7):
        rng = _rng(seed)
        q = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        k = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        v = rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32)
        return q, k, v

    def test_output_shape(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        out, stats = sparge_attention_forward(q, k, v, cfg)
        assert out.shape == (2, 16, 32)

    def test_output_finite(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        out, _ = sparge_attention_forward(q, k, v, cfg)
        assert np.isfinite(out).all()

    def test_stats_total_blocks_positive(self):
        cfg = _cfg()
        q, k, v = self._inputs()
        _, stats = sparge_attention_forward(q, k, v, cfg)
        assert stats.total_blocks > 0


# ---------------------------------------------------------------------------
# TestSpargeAttnEngine
# ---------------------------------------------------------------------------


class TestSpargeAttnEngine:
    def _inputs(self, n_heads=2, seq=16, head_dim=32):
        rng = _rng(42)
        return (
            rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32),
            rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32),
            rng.normal(0, 0.3, (n_heads, seq, head_dim)).astype(np.float32),
        )

    def test_forward_shape(self):
        engine = SpargeAttnEngine(_cfg())
        out, _ = engine.forward(*self._inputs())
        assert out.shape == (2, 16, 32)

    def test_cumulative_stats_accumulate(self):
        engine = SpargeAttnEngine(_cfg())
        q, k, v = self._inputs()
        engine.forward(q, k, v)
        engine.forward(q, k, v)
        assert engine.cumulative_stats.total_blocks > 0

    def test_reset_stats(self):
        engine = SpargeAttnEngine(_cfg())
        engine.forward(*self._inputs())
        engine.reset_stats()
        assert engine.cumulative_stats.total_blocks == 0
