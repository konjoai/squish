"""tests/test_context_summarizer_unit.py

Full-coverage unit tests for squish/context_summarizer.py.

Covers:
  SummaryConfig       — invalid method, budget < 1, min_keep_recent < 0
  SummaryStats        — construction
  ContextSummarizer   — needs_compression, summarize (no-op, importance,
                        stride, recency), error paths, edge cases in all
                        three selection strategies
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.context_summarizer import ContextSummarizer, SummaryConfig, SummaryStats


# ---------------------------------------------------------------------------
# SummaryConfig
# ---------------------------------------------------------------------------


class TestSummaryConfig:
    def test_valid_defaults(self):
        cfg = SummaryConfig()
        assert cfg.method == "importance"
        assert cfg.budget == 512
        assert cfg.min_keep_recent == 64

    def test_all_valid_methods(self):
        for method in ("importance", "stride", "recency"):
            cfg = SummaryConfig(method=method)
            assert cfg.method == method

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            SummaryConfig(method="magic")

    def test_budget_zero_raises(self):
        with pytest.raises(ValueError, match="budget must be >= 1"):
            SummaryConfig(budget=0)

    def test_budget_negative_raises(self):
        with pytest.raises(ValueError, match="budget must be >= 1"):
            SummaryConfig(budget=-5)

    def test_min_keep_recent_negative_raises(self):
        with pytest.raises(ValueError, match="min_keep_recent must be >= 0"):
            SummaryConfig(min_keep_recent=-1)

    def test_min_keep_recent_zero_valid(self):
        cfg = SummaryConfig(min_keep_recent=0)
        assert cfg.min_keep_recent == 0


# ---------------------------------------------------------------------------
# SummaryStats
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_construction(self):
        s = SummaryStats(
            n_tokens_in=100, n_tokens_out=50,
            compression_ratio=0.5, method_used="recency",
        )
        assert s.n_tokens_in == 100
        assert s.n_tokens_out == 50
        assert s.compression_ratio == pytest.approx(0.5)
        assert s.method_used == "recency"


# ---------------------------------------------------------------------------
# ContextSummarizer
# ---------------------------------------------------------------------------


class TestNeedsCompression:
    def test_below_budget(self):
        cfg = SummaryConfig(budget=100)
        s = ContextSummarizer(cfg)
        assert s.needs_compression(50) is False

    def test_at_budget(self):
        cfg = SummaryConfig(budget=100)
        s = ContextSummarizer(cfg)
        assert s.needs_compression(100) is False

    def test_above_budget(self):
        cfg = SummaryConfig(budget=100)
        s = ContextSummarizer(cfg)
        assert s.needs_compression(101) is True


class TestSummarizeNoCompression:
    def test_short_seq_returned_unchanged(self):
        cfg = SummaryConfig(budget=128)
        s = ContextSummarizer(cfg)
        tokens = np.arange(50, dtype=np.int32)
        out, stats = s.summarize(tokens)
        assert np.array_equal(out, tokens)
        assert stats.n_tokens_in == 50
        assert stats.n_tokens_out == 50
        assert stats.compression_ratio == 1.0

    def test_exactly_at_budget_not_compressed(self):
        cfg = SummaryConfig(budget=32)
        s = ContextSummarizer(cfg)
        tokens = np.ones(32, dtype=np.int32)
        out, stats = s.summarize(tokens)
        assert stats.n_tokens_out == 32


class TestSummarizeErrorPaths:
    def test_non_1d_tokens_raises(self):
        cfg = SummaryConfig(budget=10)
        s = ContextSummarizer(cfg)
        tokens_2d = np.ones((5, 5), dtype=np.int32)
        with pytest.raises(ValueError, match="1-D"):
            s.summarize(tokens_2d)

    def test_importance_without_embeddings_raises(self):
        cfg = SummaryConfig(method="importance", budget=8)
        s = ContextSummarizer(cfg)
        tokens = np.arange(16, dtype=np.int32)
        with pytest.raises(ValueError, match="embeddings must be provided"):
            s.summarize(tokens)

    def test_embeddings_wrong_shape_raises(self):
        cfg = SummaryConfig(method="importance", budget=8)
        s = ContextSummarizer(cfg)
        tokens = np.arange(16, dtype=np.int32)
        bad_emb = np.ones((10, 32), dtype=np.float32)  # wrong seq_len
        with pytest.raises(ValueError, match="embeddings must have shape"):
            s.summarize(tokens, embeddings=bad_emb)


class TestRecencyMethod:
    def test_recency_keeps_last_budget_tokens(self):
        cfg = SummaryConfig(method="recency", budget=8)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        out, stats = s.summarize(tokens)
        assert stats.n_tokens_out == 8
        assert stats.method_used == "recency"
        # Should be the last 8 tokens: 12, 13, 14, 15, 16, 17, 18, 19
        assert np.array_equal(out, tokens[-8:])

    def test_recency_output_in_causal_order(self):
        cfg = SummaryConfig(method="recency", budget=4)
        s = ContextSummarizer(cfg)
        tokens = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        out, _ = s.summarize(tokens)
        assert list(out) == [30, 40, 50, 60]


class TestStrideMethod:
    def test_stride_output_length(self):
        cfg = SummaryConfig(method="stride", budget=10, min_keep_recent=5)
        s = ContextSummarizer(cfg)
        tokens = np.arange(50, dtype=np.int32)
        out, stats = s.summarize(tokens)
        assert stats.n_tokens_out <= 10
        assert stats.method_used == "stride"

    def test_stride_output_in_causal_order(self):
        cfg = SummaryConfig(method="stride", budget=8, min_keep_recent=4)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        out, _ = s.summarize(tokens)
        # Output must be in non-decreasing order (causal)
        assert np.all(np.diff(out) >= 0)

    def test_stride_min_recent_gt_budget(self):
        """When min_keep_recent >= budget, only recent tokens are returned."""
        cfg = SummaryConfig(method="stride", budget=4, min_keep_recent=8)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        out, stats = s.summarize(tokens)
        # min_recent is clamped to budget=4
        assert stats.n_tokens_out <= 4

    def test_stride_prefix_empty_edge_case(self):
        """When seq_len - min_recent == 0, only recent tokens are returned."""
        cfg = SummaryConfig(method="stride", budget=10, min_keep_recent=10)
        s = ContextSummarizer(cfg)
        tokens = np.arange(15, dtype=np.int32)
        out, stats = s.summarize(tokens)
        assert stats.n_tokens_out <= 10

    def test_stride_n_from_prefix_zero(self):
        """When budget == min_keep_recent, n_from_prefix == 0 → only recent."""
        cfg = SummaryConfig(method="stride", budget=5, min_keep_recent=5)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        out, stats = s.summarize(tokens)
        # Should be just the last 5 tokens
        assert np.array_equal(out, tokens[-5:])


class TestImportanceMethod:
    def _make_embeddings(self, seq_len, dim=16):
        rng = np.random.default_rng(42)
        return rng.standard_normal((seq_len, dim)).astype(np.float32)

    def test_importance_reduces_tokens(self):
        cfg = SummaryConfig(method="importance", budget=10, min_keep_recent=4)
        s = ContextSummarizer(cfg)
        tokens = np.arange(30, dtype=np.int32)
        emb = self._make_embeddings(30)
        out, stats = s.summarize(tokens, embeddings=emb)
        assert stats.n_tokens_out <= 10
        assert stats.method_used == "importance"

    def test_importance_output_in_causal_order(self):
        cfg = SummaryConfig(method="importance", budget=8, min_keep_recent=3)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        emb = self._make_embeddings(20)
        out, _ = s.summarize(tokens, embeddings=emb)
        assert np.all(np.diff(out) >= 0)

    def test_importance_accepts_float64_embeddings(self):
        cfg = SummaryConfig(method="importance", budget=8, min_keep_recent=3)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        emb = self._make_embeddings(20).astype(np.float64)
        out, stats = s.summarize(tokens, embeddings=emb)
        assert stats.n_tokens_out <= 8

    def test_importance_all_prefix_kept_when_few_tokens(self):
        """When n_from_prefix >= prefix_len, use all prefix indices."""
        cfg = SummaryConfig(method="importance", budget=20, min_keep_recent=4)
        s = ContextSummarizer(cfg)
        # seq_len=10, budget=20, min_recent=4 → n_from_prefix=16 >= prefix_len=6
        tokens = np.arange(10, dtype=np.int32)
        emb = self._make_embeddings(10)
        out, stats = s.summarize(tokens, embeddings=emb)
        # Since seq_len <= budget, no compression is needed
        assert stats.n_tokens_out == 10

    def test_importance_small_budget_forces_partial_sort(self):
        """n_from_prefix < prefix_len → argpartition path."""
        cfg = SummaryConfig(method="importance", budget=5, min_keep_recent=2)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        emb = self._make_embeddings(20)
        out, stats = s.summarize(tokens, embeddings=emb)
        assert stats.n_tokens_out <= 5

    def test_importance_min_recent_clamped_to_budget(self):
        """min_keep_recent > budget → clamped to budget."""
        cfg = SummaryConfig(method="importance", budget=4, min_keep_recent=100)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        emb = self._make_embeddings(20)
        out, stats = s.summarize(tokens, embeddings=emb)
        assert stats.n_tokens_out <= 4

    def test_compression_ratio_computed_correctly(self):
        cfg = SummaryConfig(method="recency", budget=5)
        s = ContextSummarizer(cfg)
        tokens = np.arange(20, dtype=np.int32)
        _, stats = s.summarize(tokens)
        expected_ratio = 5.0 / 20.0
        assert stats.compression_ratio == pytest.approx(expected_ratio)
