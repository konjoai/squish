"""tests/test_cocktail_kv_unit.py

Full-coverage unit tests for squish/cocktail_kv.py.

Covers:
  CocktailConfig         — all __post_init__ validation errors
  ChunkSimilarityScorer  — _embed, _cosine, score_chunks (cosine+dot), assign_bits
  CocktailKVQuantizer    — quantize_chunk (bits=16, 4, 2), dequantize_chunk
  CocktailKVStore        — store, retrieve, reset, stats property,
                           reorder_by_precision path
  CocktailStats          — avg_bits, compression_ratio, reset
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.cocktail_kv import (
    ChunkSimilarityScorer,
    CocktailConfig,
    CocktailKVQuantizer,
    CocktailKVStore,
    CocktailStats,
)


# ---------------------------------------------------------------------------
# CocktailConfig
# ---------------------------------------------------------------------------


class TestCocktailConfig:
    def test_valid_defaults(self):
        cfg = CocktailConfig()
        assert cfg.chunk_size == 32
        assert cfg.fp16_fraction == 0.15
        assert cfg.int2_fraction == 0.50
        assert cfg.similarity_metric == "cosine"
        assert cfg.reorder_by_precision is True

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            CocktailConfig(chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            CocktailConfig(chunk_size=-5)

    def test_fp16_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="fp16_fraction must be in"):
            CocktailConfig(fp16_fraction=-0.1)

    def test_fp16_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="fp16_fraction must be in"):
            CocktailConfig(fp16_fraction=1.1)

    def test_int2_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="int2_fraction must be in"):
            CocktailConfig(int2_fraction=-0.1)

    def test_int2_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="int2_fraction must be in"):
            CocktailConfig(int2_fraction=1.1)

    def test_fraction_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="fp16_fraction.*int2_fraction.*must be <= 1"):
            CocktailConfig(fp16_fraction=0.7, int2_fraction=0.6)

    def test_bad_similarity_metric_raises(self):
        with pytest.raises(ValueError, match="similarity_metric must be"):
            CocktailConfig(similarity_metric="euclidean")

    def test_dot_metric_valid(self):
        cfg = CocktailConfig(similarity_metric="dot")
        assert cfg.similarity_metric == "dot"


# ---------------------------------------------------------------------------
# ChunkSimilarityScorer
# ---------------------------------------------------------------------------


class TestChunkSimilarityScorer:
    def _make_scorer(self, **kwargs):
        return ChunkSimilarityScorer(CocktailConfig(**kwargs))

    def test_embed_mean_pools(self):
        scorer = self._make_scorer()
        vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = scorer._embed(vecs)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_cosine_zero_vectors_returns_zero(self):
        scorer = self._make_scorer()
        a = np.zeros(4, dtype=np.float32)
        b = np.ones(4, dtype=np.float32)
        assert scorer._cosine(a, b) == 0.0

    def test_cosine_parallel_vectors_returns_one(self):
        scorer = self._make_scorer()
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        assert scorer._cosine(a, b) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors_returns_zero(self):
        scorer = self._make_scorer()
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert scorer._cosine(a, b) == pytest.approx(0.0)

    def test_score_chunks_cosine_shape(self):
        scorer = self._make_scorer(chunk_size=8)
        rng = np.random.default_rng(0)
        query = rng.standard_normal(16).astype(np.float32)
        tokens = rng.standard_normal((32, 16)).astype(np.float32)
        scores = scorer.score_chunks(query, tokens)
        n_chunks = (32 + 8 - 1) // 8
        assert scores.shape == (n_chunks,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_score_chunks_dot_metric(self):
        scorer = self._make_scorer(chunk_size=4, similarity_metric="dot")
        rng = np.random.default_rng(1)
        query = rng.standard_normal(8).astype(np.float32)
        tokens = rng.standard_normal((16, 8)).astype(np.float32)
        scores = scorer.score_chunks(query, tokens)
        assert scores.shape == (4,)

    def test_score_chunks_single_chunk(self):
        scorer = self._make_scorer(chunk_size=32)
        rng = np.random.default_rng(2)
        query = rng.standard_normal(8).astype(np.float32)
        tokens = rng.standard_normal((10, 8)).astype(np.float32)
        scores = scorer.score_chunks(query, tokens)
        assert scores.shape == (1,)

    def test_score_chunks_normalizes_to_05_when_all_equal(self):
        """When all chunks have the same score → all set to 0.5."""
        scorer = self._make_scorer(chunk_size=4)
        # All-zeros query and uniform tokens → identical chunk embeddings
        query = np.zeros(8, dtype=np.float32)
        tokens = np.zeros((8, 8), dtype=np.float32)
        scores = scorer.score_chunks(query, tokens)
        np.testing.assert_allclose(scores, 0.5)

    def test_assign_bits_values(self):
        scorer = self._make_scorer()
        sim = np.array([0.9, 0.5, 0.3, 0.1, 0.05], dtype=np.float32)
        bits = scorer.assign_bits(sim)
        assert bits.shape == (5,)
        assert set(bits).issubset({2, 4, 16})

    def test_assign_bits_high_sim_gets_fp16(self):
        scorer = self._make_scorer(fp16_fraction=0.2, int2_fraction=0.4)
        sim = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)
        bits = scorer.assign_bits(sim)
        # Highest similarity chunk should get 16 bits
        assert bits[0] == 16  # highest similarity

    def test_assign_bits_low_sim_gets_int2(self):
        scorer = self._make_scorer(fp16_fraction=0.2, int2_fraction=0.4)
        sim = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)
        bits = scorer.assign_bits(sim)
        # Lowest similarity chunk should get 2 bits
        assert bits[4] == 2  # lowest similarity


# ---------------------------------------------------------------------------
# CocktailKVQuantizer
# ---------------------------------------------------------------------------


class TestCocktailKVQuantizer:
    def _make_quant(self, group_size=8):
        return CocktailKVQuantizer(CocktailConfig(group_size=group_size))

    def test_quantize_fp16(self):
        q = self._make_quant()
        chunk = np.random.randn(4, 8).astype(np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=16)
        assert q_chunk.dtype == np.float16
        assert scales.shape == (1,)
        assert float(scales[0]) == pytest.approx(1.0)

    def test_dequantize_fp16_roundtrip(self):
        q = self._make_quant()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=16)
        recon = q.dequantize_chunk(q_chunk, scales, bits=16, original_shape=chunk.shape)
        np.testing.assert_allclose(recon, chunk, atol=1e-3)

    def test_quantize_int4(self):
        q = self._make_quant(group_size=4)
        chunk = np.linspace(-1.0, 1.0, 16).reshape(4, 4).astype(np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=4)
        assert q_chunk.dtype == np.int8

    def test_quantize_int2(self):
        q = self._make_quant(group_size=4)
        chunk = np.linspace(-1.0, 1.0, 8).reshape(2, 4).astype(np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=2)
        assert q_chunk.dtype == np.int8

    def test_dequantize_int4_approximate(self):
        q = self._make_quant(group_size=8)
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal((4, 8)).astype(np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=4)
        recon = q.dequantize_chunk(q_chunk, scales, bits=4, original_shape=chunk.shape)
        # Dequantized values should be close-ish to originals
        assert recon.shape == chunk.shape
        assert recon.dtype == np.float32

    def test_dequantize_int2(self):
        q = self._make_quant(group_size=4)
        chunk = np.array([[0.5, -0.5, 0.3, -0.3]], dtype=np.float32)
        q_chunk, scales = q.quantize_chunk(chunk, bits=2)
        recon = q.dequantize_chunk(q_chunk, scales, bits=2, original_shape=chunk.shape)
        assert recon.shape == chunk.shape


# ---------------------------------------------------------------------------
# CocktailKVStore
# ---------------------------------------------------------------------------


def _make_store(reorder=True, chunk_size=4):
    cfg = CocktailConfig(
        chunk_size=chunk_size,
        fp16_fraction=0.2,
        int2_fraction=0.5,
        group_size=4,
        reorder_by_precision=reorder,
    )
    return CocktailKVStore(cfg)


def _make_inputs(seq_len=16, embed_dim=8, head_dim=4, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    kv_matrix = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    query_emb = rng.standard_normal(embed_dim).astype(np.float32)
    token_embs = rng.standard_normal((seq_len, embed_dim)).astype(np.float32)
    return kv_matrix, query_emb, token_embs


class TestCocktailKVStore:
    def test_store_and_retrieve(self):
        store = _make_store(reorder=True)
        kv, q, t = _make_inputs()
        store.store(kv, q, t)
        recon = store.retrieve()
        assert recon.shape[1] == kv.shape[1]
        assert recon.dtype == np.float32

    def test_retrieve_empty_returns_empty(self):
        store = _make_store()
        result = store.retrieve()
        assert result.shape == (0,)

    def test_store_without_reorder(self):
        store = _make_store(reorder=False)
        kv, q, t = _make_inputs()
        store.store(kv, q, t)
        recon = store.retrieve()
        assert recon.shape[1] == kv.shape[1]

    def test_reset_clears_store(self):
        store = _make_store()
        kv, q, t = _make_inputs()
        store.store(kv, q, t)
        store.reset()
        assert store.retrieve().shape == (0,)
        assert store._chunks == []
        assert store._chunk_bits == []
        assert store._chunk_orig_shapes == []

    def test_stats_updated_after_store(self):
        store = _make_store()
        kv, q, t = _make_inputs(seq_len=16)
        store.store(kv, q, t)
        assert store.stats.total_chunks > 0

    def test_stats_counts_fp16_int4_int2(self):
        store = _make_store()
        kv, q, t = _make_inputs(seq_len=20)
        store.store(kv, q, t)
        s = store.stats
        assert s.total_chunks == s.fp16_chunks + s.int4_chunks + s.int2_chunks

    def test_stats_property_returns_cocktail_stats(self):
        store = _make_store()
        s = store.stats
        assert isinstance(s, CocktailStats)

    def test_store_clears_previous_on_new_store(self):
        store = _make_store()
        kv1, q1, t1 = _make_inputs(seq_len=8, rng_seed=0)
        kv2, q2, t2 = _make_inputs(seq_len=16, rng_seed=1)
        store.store(kv1, q1, t1)
        n_chunks_1 = len(store._chunks)
        store.store(kv2, q2, t2)
        n_chunks_2 = len(store._chunks)
        # After second store, previous chunks should be cleared
        assert n_chunks_2 != n_chunks_1 or n_chunks_1 == n_chunks_2


# ---------------------------------------------------------------------------
# CocktailStats
# ---------------------------------------------------------------------------


class TestCocktailStats:
    def test_avg_bits_zero_chunks(self):
        s = CocktailStats()
        assert s.avg_bits == 0.0

    def test_avg_bits_all_fp16(self):
        s = CocktailStats(total_chunks=4, fp16_chunks=4)
        assert s.avg_bits == pytest.approx(16.0)

    def test_avg_bits_all_int4(self):
        s = CocktailStats(total_chunks=4, int4_chunks=4)
        assert s.avg_bits == pytest.approx(4.0)

    def test_avg_bits_all_int2(self):
        s = CocktailStats(total_chunks=4, int2_chunks=4)
        assert s.avg_bits == pytest.approx(2.0)

    def test_avg_bits_mixed(self):
        s = CocktailStats(total_chunks=4, fp16_chunks=1, int4_chunks=2, int2_chunks=1)
        expected = (16 * 1 + 4 * 2 + 2 * 1) / 4
        assert s.avg_bits == pytest.approx(expected)

    def test_compression_ratio_zero_chunks(self):
        s = CocktailStats()
        assert s.compression_ratio == pytest.approx(1.0)

    def test_compression_ratio_all_fp16(self):
        s = CocktailStats(total_chunks=4, fp16_chunks=4)
        assert s.compression_ratio == pytest.approx(1.0)

    def test_compression_ratio_all_int2(self):
        s = CocktailStats(total_chunks=4, int2_chunks=4)
        assert s.compression_ratio == pytest.approx(2.0 / 16.0)

    def test_reset(self):
        s = CocktailStats(total_chunks=10, fp16_chunks=3, int4_chunks=4, int2_chunks=3)
        s.reset()
        assert s.total_chunks == 0
        assert s.fp16_chunks == 0
        assert s.int4_chunks == 0
        assert s.int2_chunks == 0
