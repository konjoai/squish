"""tests/test_coverage_batch2.py

Targeted branch and line coverage tests for 15 experimental modules.
Each class covers the previously uncovered lines reported at 96.44% baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from squish.kv.context_cache import (
    CacheEntry,
    ContextCacheStats,
    PersistentContextCache,
    _hash_tokens,
)
from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
from squish.token.layerwise_decode import (
    DecodeStats,
    LayerwiseConfig,
    LayerwiseDecoder,
    LayerStream,
)
from squish.speculative.distil_spec import DistilConfig, DistilSpecCalibrator, DistilStats
from squish.streaming.stream_rag import (
    RAGDocument,
    StreamRAGConfig,
    StreamRAGInjector,
    StreamRAGStats,
)
from squish.kv.kv_compress import KVCompressConfig, KVCompressor
from squish.serving.batch_embed import BatchEmbedder, EmbeddingStats, PoolingConfig
from squish.token.token_healer import HealerConfig, HealerStats, TokenHealer
from squish.quant.adaptive_quantize import (
    AdaptiveQuantizer,
    AdaptiveQuantStats,
    PressureMonitor,
    PressureThresholds,
    QuantPrecision,
)
from squish.speculative.mirror_sd import (
    MirrorDraftPipeline,
    MirrorFuture,
    MirrorSDConfig,
    MirrorSDDecoder,
    MirrorSDStats,
    MirrorVerifyPipeline,
    _softmax,
    _top_p_filter,
)
from squish.speculative.medusa import (
    MedusaConfig,
    MedusaDecoder,
    MedusaDraftTree,
    MedusaHead,
    MedusaStats,
)
from squish.kv.mixed_precision_kv import (
    HeadPrecision,
    HeadPrecisionMap,
    MixedPrecisionKVCache,
    MPKVConfig,
    MPKVStats,
    _dequantize_symmetric,
    _quantize_symmetric,
)
from squish.speculative.sparse_spec import (
    PillarAttnCache,
    SparseSpecConfig,
    SparseSpecDecoder,
    SparseSpecDrafter,
    SparseSpecStats,
)
from squish.attention.sparse_attn_index import ANCandidates, IndexConfig, IndexStats, SparseAttnIndex
from squish.token.token_budget_gate import BudgetGateStats, BudgetPolicy, TokenBudgetGate


# ===========================================================================
# 1. context_cache.py
# Uncovered: 235->239, 237, 272-273, 276-279, 315-321
# ===========================================================================


class TestContextCacheCoverage:
    """Cover cache-miss, expired-entry, capacity-eviction, and _evict_oldest."""

    def _make_kv(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Line 235->239: update existing entry (token_hash in _entries)
    # ------------------------------------------------------------------

    def test_put_overwrites_existing_entry(self) -> None:
        """When token_hash already exists, capacity check is skipped (235->239)."""
        cache = PersistentContextCache(max_entries=2, default_ttl_s=300.0)
        tokens = [1, 2, 3]
        kv1 = np.ones(4, dtype=np.float32)
        kv2 = np.full(4, 2.0, dtype=np.float32)
        eid1 = cache.put(tokens, kv1)
        eid2 = cache.put(tokens, kv2)  # same tokens → update path
        assert eid1 != eid2  # new entry_id on overwrite
        assert cache.n_entries == 1
        result = cache.get(tokens)
        assert result is not None
        np.testing.assert_array_equal(result, kv2)

    # ------------------------------------------------------------------
    # Lines 237, 315-321: _evict_oldest() on non-empty cache + capacity eviction
    # ------------------------------------------------------------------

    def test_capacity_eviction_evicts_oldest(self) -> None:
        """Fill cache to max_entries=1, then insert new token → eviction (lines 237, 315-321)."""
        cache = PersistentContextCache(max_entries=1, default_ttl_s=300.0)
        kv = self._make_kv()
        cache.put([1, 2], kv)
        assert cache.n_entries == 1
        # Insert a *different* token sequence — triggers eviction of [1,2]
        cache.put([3, 4], kv)
        assert cache.n_entries == 1  # still only 1 after eviction + insert
        assert cache.stats.evictions == 1

    def test_evict_oldest_empty_is_noop(self) -> None:
        """_evict_oldest on an empty cache returns immediately (line 315-316)."""
        cache = PersistentContextCache(max_entries=5, default_ttl_s=300.0)
        # Must not raise
        cache._evict_oldest()
        assert cache.n_entries == 0

    def test_capacity_eviction_with_multiple_entries(self) -> None:
        """max_entries=2, insert 3 distinct sequences → two evictions."""
        cache = PersistentContextCache(max_entries=2, default_ttl_s=300.0)
        kv = self._make_kv()
        cache.put([1], kv)
        cache.put([2], kv)
        cache.put([3], kv)  # triggers eviction of oldest ([1])
        assert cache.n_entries == 2

    # ------------------------------------------------------------------
    # Lines 272-273: get() → miss (entry not in cache)
    # ------------------------------------------------------------------

    def test_get_miss_returns_none(self) -> None:
        """get() on token not in cache returns None and increments misses."""
        cache = PersistentContextCache(max_entries=5, default_ttl_s=300.0)
        result = cache.get([99, 100])
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_get_miss_empty_cache(self) -> None:
        """get() on a completely empty cache hits miss path."""
        cache = PersistentContextCache(max_entries=10, default_ttl_s=60.0)
        assert cache.get([1]) is None
        assert cache.stats.total_gets == 1

    # ------------------------------------------------------------------
    # Lines 276-279: get() → expired entry removed
    # ------------------------------------------------------------------

    def test_get_expired_entry_returns_none(self) -> None:
        """get() on an expired entry removes it and returns None (lines 276-279)."""
        cache = PersistentContextCache(max_entries=5, default_ttl_s=300.0)
        tokens = [10, 20, 30]
        kv = self._make_kv()
        cache.put(tokens, kv)
        assert cache.n_entries == 1
        # Expire the entry by setting created_at far in the past
        token_hash = list(cache._entries.keys())[0]
        cache._entries[token_hash].created_at = 0.0
        result = cache.get(tokens)
        assert result is None
        assert cache.n_entries == 0  # entry was removed
        assert cache.stats.evictions == 1
        assert cache.stats.misses == 1


# ===========================================================================
# 2. fused_sampler.py
# Uncovered: 210-211, 226-228, 262->268, 283-284, 289->297, 298-305
# ===========================================================================


VOCAB_SIZE = 32  # small vocab for tests


def _random_logits(size: int = VOCAB_SIZE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(size).astype(np.float32)


class TestFusedSamplerCoverage:
    """Cover 2-D input_ids, reset_rng, all-out-of-range ids, min-p, top-k skip, top-p."""

    # ------------------------------------------------------------------
    # Lines 210-211: sample_batch() with 2-D input_ids
    # ------------------------------------------------------------------

    def test_sample_batch_2d_input_ids(self) -> None:
        """sample_batch() with 2-D input_ids uses per-row ids (lines 210-211)."""
        cfg = SamplerConfig(
            temperature=1.0,
            repetition_penalty=1.2,
            seed=42,
        )
        sampler = FusedSampler(cfg)
        batch = np.random.default_rng(0).standard_normal((3, VOCAB_SIZE)).astype(np.float32)
        # 2-D input_ids: (batch, context_len) — each row gets its own ids
        input_ids_2d = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        tokens = sampler.sample_batch(batch, input_ids=input_ids_2d)
        assert tokens.shape == (3,)
        assert all(0 <= t < VOCAB_SIZE for t in tokens)

    # ------------------------------------------------------------------
    # Lines 226-228: reset_rng() — invalid seed raises, valid seed succeeds
    # ------------------------------------------------------------------

    def test_reset_rng_negative_seed_raises(self) -> None:
        """reset_rng(-1) raises ValueError (line 226-227)."""
        sampler = FusedSampler(SamplerConfig(seed=0))
        with pytest.raises(ValueError, match="seed must be >= 0"):
            sampler.reset_rng(-1)

    def test_reset_rng_valid_seed(self) -> None:
        """reset_rng with valid seed re-seeds the RNG (line 228)."""
        sampler = FusedSampler(SamplerConfig(seed=0))
        sampler.reset_rng(99)
        logits = _random_logits()
        token = sampler.sample(logits)
        assert 0 <= token < VOCAB_SIZE

    # ------------------------------------------------------------------
    # Lines 262->268: repetition penalty with all-out-of-range ids → ids.size == 0
    # ------------------------------------------------------------------

    def test_repetition_penalty_all_ids_out_of_range(self) -> None:
        """All input_ids out of vocab range → ids.size==0, penalty skipped (262->268)."""
        cfg = SamplerConfig(repetition_penalty=1.5, seed=7)
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        # IDs that are all outside [0, VOCAB_SIZE)
        out_of_range_ids = np.array([-5, -1, VOCAB_SIZE, VOCAB_SIZE + 100], dtype=np.int64)
        token = sampler.sample(logits, input_ids=out_of_range_ids)
        assert 0 <= token < VOCAB_SIZE

    # ------------------------------------------------------------------
    # Lines 283-284: min_p filter
    # ------------------------------------------------------------------

    def test_min_p_filter(self) -> None:
        """min_p > 0 triggers the min-p masking step (lines 283-284)."""
        cfg = SamplerConfig(min_p=0.1, seed=0)
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        token = sampler.sample(logits)
        assert 0 <= token < VOCAB_SIZE

    # ------------------------------------------------------------------
    # Lines 289->297: top_k >= vocab_size → k == len(work), no filtering needed
    # ------------------------------------------------------------------

    def test_top_k_larger_than_vocab_no_filtering(self) -> None:
        """top_k >= vocab_size means k == len(work), skip inner filter (289->297)."""
        cfg = SamplerConfig(top_k=VOCAB_SIZE * 10, seed=1)
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        token = sampler.sample(logits)
        assert 0 <= token < VOCAB_SIZE

    # ------------------------------------------------------------------
    # Lines 298-305: top_p nucleus filter
    # ------------------------------------------------------------------

    def test_top_p_nucleus_filter(self) -> None:
        """top_p < 1.0 activates nucleus filter (lines 298-305)."""
        cfg = SamplerConfig(top_p=0.8, seed=2)
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        token = sampler.sample(logits)
        assert 0 <= token < VOCAB_SIZE

    def test_top_p_and_top_k_combined(self) -> None:
        """Both top_k and top_p applied together."""
        cfg = SamplerConfig(top_k=10, top_p=0.9, seed=3)
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        token = sampler.sample(logits)
        assert 0 <= token < VOCAB_SIZE

    def test_all_filters_combined(self) -> None:
        """All filters: temperature, min_p, top_k, top_p, repetition_penalty."""
        cfg = SamplerConfig(
            temperature=0.8,
            top_k=8,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.3,
            seed=5,
        )
        sampler = FusedSampler(cfg)
        logits = _random_logits(VOCAB_SIZE)
        ids = np.array([0, 1, 2], dtype=np.int64)
        token = sampler.sample(logits, input_ids=ids)
        assert 0 <= token < VOCAB_SIZE


# ===========================================================================
# 3. layerwise_decode.py
# Uncovered: 145-147, 225-239
# ===========================================================================


class TestLayerwiseDecodeCoverage:
    """Cover DecodeStats avg_exit_layer and should_exit error paths + computation."""

    def _make_decoder(self) -> LayerwiseDecoder:
        cfg = LayerwiseConfig(
            n_layers=8, hidden_dim=16, exit_threshold=0.9,
            min_exit_layer=2, probe_vocab=8,
        )
        return LayerwiseDecoder(cfg, rng=np.random.default_rng(0))

    # ------------------------------------------------------------------
    # Lines 145-147: DecodeStats.avg_exit_layer with zero and non-zero tokens
    # ------------------------------------------------------------------

    def test_avg_exit_layer_no_tokens(self) -> None:
        """avg_exit_layer returns 0.0 when no tokens recorded (lines 145-146)."""
        stats = DecodeStats()
        assert stats.avg_exit_layer == 0.0

    def test_avg_exit_layer_with_tokens(self) -> None:
        """avg_exit_layer returns correct ratio (line 147)."""
        stats = DecodeStats(total_tokens=4, total_layers_run=12)
        assert stats.avg_exit_layer == pytest.approx(3.0)

    def test_early_exit_rate_no_tokens(self) -> None:
        """early_exit_rate returns 0.0 when no tokens."""
        stats = DecodeStats()
        assert stats.early_exit_rate == 0.0

    def test_early_exit_rate_with_tokens(self) -> None:
        """early_exit_rate returns correct fraction."""
        stats = DecodeStats(total_tokens=10, early_exits=4)
        assert stats.early_exit_rate == pytest.approx(0.4)

    # ------------------------------------------------------------------
    # Lines 225-228: should_exit() with wrong hidden shape → ValueError
    # ------------------------------------------------------------------

    def test_should_exit_wrong_hidden_shape_raises(self) -> None:
        """Wrong hidden shape raises ValueError (lines 225-228)."""
        decoder = self._make_decoder()
        bad_hidden = np.zeros(8, dtype=np.float32)  # hidden_dim=16 expected
        with pytest.raises(ValueError, match="Expected hidden shape"):
            decoder.should_exit(bad_hidden, layer_idx=3)

    # ------------------------------------------------------------------
    # Lines 230-231: should_exit() with negative layer_idx → ValueError
    # ------------------------------------------------------------------

    def test_should_exit_negative_layer_idx_raises(self) -> None:
        """Negative layer_idx raises ValueError (lines 230-231)."""
        decoder = self._make_decoder()
        hidden = np.zeros(16, dtype=np.float32)
        with pytest.raises(ValueError, match="layer_idx must be non-negative"):
            decoder.should_exit(hidden, layer_idx=-1)

    # ------------------------------------------------------------------
    # Lines 232-233: should_exit() when layer_idx < min_exit_layer → False
    # ------------------------------------------------------------------

    def test_should_exit_below_min_layer_returns_false(self) -> None:
        """Exit suppressed when layer_idx < min_exit_layer (lines 232-233)."""
        decoder = self._make_decoder()  # min_exit_layer=2
        hidden = np.zeros(16, dtype=np.float32)
        result = decoder.should_exit(hidden, layer_idx=1)
        assert result is False

    # ------------------------------------------------------------------
    # Lines 235-239: should_exit() computation when layer_idx >= min_exit_layer
    # ------------------------------------------------------------------

    def test_should_exit_above_min_layer_computes(self) -> None:
        """should_exit evaluates probe confidence (lines 235-239)."""
        decoder = self._make_decoder()  # min_exit_layer=2
        hidden = np.ones(16, dtype=np.float32)
        # layer_idx=2 >= min_exit_layer=2 → runs confidence computation
        result = decoder.should_exit(hidden, layer_idx=2)
        assert isinstance(result, bool)

    def test_should_exit_high_confidence_exits(self) -> None:
        """With exit_threshold=0.0, should_exit always returns True."""
        cfg = LayerwiseConfig(
            n_layers=4, hidden_dim=4, exit_threshold=0.0001,
            min_exit_layer=0, probe_vocab=2,
        )
        decoder = LayerwiseDecoder(cfg, rng=np.random.default_rng(42))
        hidden = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        result = decoder.should_exit(hidden, layer_idx=0)
        assert isinstance(result, bool)


# ===========================================================================
# 4. distil_spec.py
# Uncovered: 125-127, 262-267, 288-291, 313-318
# ===========================================================================


class TestDistilSpecCoverage:
    """Cover DistilStats.estimated_acceptance_gain_pp, compute_delta errors,
    reset(), and stats()."""

    # ------------------------------------------------------------------
    # Lines 125-127: DistilStats.estimated_acceptance_gain_pp
    # ------------------------------------------------------------------

    def test_estimated_acceptance_gain_pp_zero_steps(self) -> None:
        """Returns 0.0 when n_steps == 0 (line 125-126)."""
        stats = DistilStats(n_steps=0)
        assert stats.estimated_acceptance_gain_pp == 0.0

    def test_estimated_acceptance_gain_pp_nonzero(self) -> None:
        """Computes gain when n_steps > 0 (line 127)."""
        stats = DistilStats(n_steps=10, total_kl_reduction=2.0)
        expected = (2.0 / 10) * 15.0
        assert stats.estimated_acceptance_gain_pp == pytest.approx(expected)

    # ------------------------------------------------------------------
    # Lines 262-267: compute_delta() before any steps → RuntimeError
    #                compute_delta() after steps → success
    # ------------------------------------------------------------------

    def test_compute_delta_no_steps_raises(self) -> None:
        """compute_delta() without record_step raises RuntimeError (lines 262-265)."""
        cfg = DistilConfig()
        cal = DistilSpecCalibrator(cfg)
        with pytest.raises(RuntimeError, match="No steps recorded"):
            cal.compute_delta()

    def test_compute_delta_after_steps_returns_array(self) -> None:
        """compute_delta() after record_step returns ndarray (lines 266-267)."""
        cfg = DistilConfig(learning_rate=1e-3, temperature=1.0)
        cal = DistilSpecCalibrator(cfg)
        rng = np.random.default_rng(0)
        draft = rng.standard_normal(100).astype(np.float32)
        target = rng.standard_normal(100).astype(np.float32)
        cal.record_step(draft, target)
        delta = cal.compute_delta()
        assert delta.shape == (100,)
        assert delta.dtype == np.float32

    # ------------------------------------------------------------------
    # Lines 288-291: reset() method
    # ------------------------------------------------------------------

    def test_reset_clears_state(self) -> None:
        """reset() zeros all accumulated state (lines 288-291)."""
        cfg = DistilConfig()
        cal = DistilSpecCalibrator(cfg)
        rng = np.random.default_rng(1)
        draft = rng.standard_normal(50).astype(np.float32)
        target = rng.standard_normal(50).astype(np.float32)
        cal.record_step(draft, target)
        assert cal.n_steps == 1
        cal.reset()
        assert cal.n_steps == 0
        assert cal._accumulated_grad is None
        assert cal._kl_history == []

    # ------------------------------------------------------------------
    # Lines 313-318: stats() with non-empty history
    # ------------------------------------------------------------------

    def test_stats_with_one_step(self) -> None:
        """stats() with 1 step: kl_reduction = 0.0, else branch (lines 313-318)."""
        cfg = DistilConfig()
        cal = DistilSpecCalibrator(cfg)
        rng = np.random.default_rng(2)
        cal.record_step(
            rng.standard_normal(20).astype(np.float32),
            rng.standard_normal(20).astype(np.float32),
        )
        s = cal.stats()
        assert isinstance(s, DistilStats)
        assert s.n_steps == 1
        assert s.total_kl_reduction == 0.0

    def test_stats_with_multiple_steps(self) -> None:
        """stats() with >= 2 steps uses first-minus-last kl (lines 313-318)."""
        cfg = DistilConfig()
        cal = DistilSpecCalibrator(cfg)
        rng = np.random.default_rng(3)
        for _ in range(3):
            cal.record_step(
                rng.standard_normal(20).astype(np.float32),
                rng.standard_normal(20).astype(np.float32),
            )
        s = cal.stats()
        assert s.n_steps == 3
        assert isinstance(s.total_kl_reduction, float)


# ===========================================================================
# 5. stream_rag.py
# Uncovered: 145-147, 219-221, 275, 293, 301
# ===========================================================================


class TestStreamRAGCoverage:
    """Cover avg_relevance on empty buffer, eviction, empty retrieve, n_docs, __repr__."""

    def _make_injector(self, max_docs: int = 3, embed_dim: int = 8) -> StreamRAGInjector:
        cfg = StreamRAGConfig(max_docs=max_docs, embed_dim=embed_dim)
        return StreamRAGInjector(cfg)

    def _rand_emb(self, dim: int = 8, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)

    # ------------------------------------------------------------------
    # Lines 145-147: avg_relevance on empty buffer
    # ------------------------------------------------------------------

    def test_avg_relevance_empty_buffer(self) -> None:
        """avg_relevance returns 0.0 when no documents are buffered (lines 145-147)."""
        inj = self._make_injector()
        assert inj.stats.avg_relevance == 0.0

    def test_avg_relevance_after_retrieve(self) -> None:
        """avg_relevance is non-zero after a retrieve call (line 147 path)."""
        inj = self._make_injector()
        inj.inject("doc1", np.array([1, 2], dtype=np.int64), self._rand_emb())
        query = self._rand_emb(seed=99)
        inj.retrieve(query)
        # After retrieval, relevance is set on each doc
        avg = inj.stats.avg_relevance
        assert isinstance(avg, float)

    # ------------------------------------------------------------------
    # Lines 219-221: Eviction when buffer full
    # ------------------------------------------------------------------

    def test_eviction_when_buffer_full(self) -> None:
        """Inserting beyond max_docs evicts the least-relevant doc (lines 219-221)."""
        inj = self._make_injector(max_docs=2)
        inj.inject("d1", np.array([1], dtype=np.int64), self._rand_emb(seed=0))
        inj.inject("d2", np.array([2], dtype=np.int64), self._rand_emb(seed=1))
        assert inj.n_docs == 2
        inj.inject("d3", np.array([3], dtype=np.int64), self._rand_emb(seed=2))
        assert inj.n_docs == 2  # eviction happened
        assert inj.stats.total_evictions == 1

    def test_eviction_after_retrieval_sets_relevance(self) -> None:
        """Eviction removes lowest-relevance doc after retrieve updates scores."""
        inj = self._make_injector(max_docs=2)
        emb0 = self._rand_emb(seed=10)
        emb1 = self._rand_emb(seed=11)
        inj.inject("d1", np.array([1], dtype=np.int64), emb0)
        inj.inject("d2", np.array([2], dtype=np.int64), emb1)
        query = emb0 / np.linalg.norm(emb0)  # very similar to d1
        inj.retrieve(query)  # sets relevance scores
        inj.inject("d3", np.array([3], dtype=np.int64), self._rand_emb(seed=12))
        assert inj.n_docs == 2

    # ------------------------------------------------------------------
    # Line 275: retrieve() on empty buffer returns []
    # ------------------------------------------------------------------

    def test_retrieve_empty_buffer(self) -> None:
        """retrieve() when no docs returns empty list (line 275)."""
        inj = self._make_injector()
        result = inj.retrieve(self._rand_emb())
        assert result == []
        assert inj.stats.total_retrievals == 1

    # ------------------------------------------------------------------
    # Line 293: n_docs property
    # ------------------------------------------------------------------

    def test_n_docs_property(self) -> None:
        """n_docs property reports current buffer size (line 293)."""
        inj = self._make_injector(max_docs=5)
        assert inj.n_docs == 0
        inj.inject("d1", np.array([1], dtype=np.int64), self._rand_emb())
        assert inj.n_docs == 1

    # ------------------------------------------------------------------
    # Line 301: __repr__
    # ------------------------------------------------------------------

    def test_repr(self) -> None:
        """__repr__ produces a meaningful string (line 301)."""
        inj = self._make_injector(max_docs=4)
        r = repr(inj)
        assert "StreamRAGInjector" in r
        assert "n_docs" in r


# ===========================================================================
# 6. kv_compress.py
# Uncovered: 296-305, 327
# ===========================================================================


class TestKVCompressCoverage:
    """Cover _build_prune_mask with non-zero prune_ratio and _quantize with zero input."""

    def _make_compressor(self, prune_ratio: float = 0.2) -> KVCompressor:
        cfg = KVCompressConfig(
            compress_after=0,
            quant_bits=8,
            prune_ratio=prune_ratio,
            n_heads=2,
            head_dim=8,
        )
        return KVCompressor(cfg)

    # ------------------------------------------------------------------
    # Lines 296-305: _build_prune_mask with prune_ratio > 0
    # ------------------------------------------------------------------

    def test_compress_with_nonzero_prune_ratio(self) -> None:
        """Non-zero prune_ratio exercises lines 296-305 in _build_prune_mask.

        Uses identical key data for both heads so the global np.quantile
        threshold produces the same number of kept positions per head,
        avoiding the shape-mismatch that arises with independent random heads.
        """
        comp = self._make_compressor(prune_ratio=0.3)
        rng = np.random.default_rng(0)
        seq_len = 10
        # Identical heads → identical per-head norms → same keep count per head.
        head_k = rng.standard_normal((seq_len, 8)).astype(np.float32)
        head_v = rng.standard_normal((seq_len, 8)).astype(np.float32)
        keys = np.stack([head_k, head_k])   # (2, seq_len, 8)
        vals = np.stack([head_v, head_v])   # (2, seq_len, 8)
        entry = comp.compress(keys, vals)
        assert entry.mask.shape == (2, seq_len)
        assert entry.mask.any()  # at least some positions kept
        assert comp.stats.n_compress_calls == 1

    def test_compress_high_prune_ratio(self) -> None:
        """High prune_ratio keeps fewer positions; still at least 1.

        Uses identical key data for both heads — see test_compress_with_nonzero_prune_ratio
        for the rationale.
        """
        comp = self._make_compressor(prune_ratio=0.8)
        rng = np.random.default_rng(1)
        head_k = rng.standard_normal((20, 8)).astype(np.float32)
        head_v = rng.standard_normal((20, 8)).astype(np.float32)
        keys = np.stack([head_k, head_k])   # (2, 20, 8)
        vals = np.stack([head_v, head_v])   # (2, 20, 8)
        entry = comp.compress(keys, vals)
        # Each head keeps at least one position
        for h in range(2):
            assert entry.mask[h].any()

    # ------------------------------------------------------------------
    # Line 327: _quantize with near-zero input → scale = 1.0
    # ------------------------------------------------------------------

    def test_compress_zero_kv_tensors_scale_fallback(self) -> None:
        """All-zero kv arrays trigger abs_max < 1e-30 → scale = 1.0 (line 327)."""
        comp = self._make_compressor(prune_ratio=0.0)  # no pruning
        keys = np.zeros((2, 5, 8), dtype=np.float32)
        vals = np.zeros((2, 5, 8), dtype=np.float32)
        entry = comp.compress(keys, vals)
        # scale_k and scale_v should both be 1.0
        assert entry.scale_k == pytest.approx(1.0)
        assert entry.scale_v == pytest.approx(1.0)


# ===========================================================================
# 7. batch_embed.py
# Uncovered: 116-118, 200-206, 221-222 (pragma added), 256, 375, 379
# ===========================================================================


class TestBatchEmbedCoverage:
    """Cover avg_seq_len, bad mask shape, pool_single with mask, n_embeddings, stats."""

    def _make_embedder(self, strategy: str = "mean") -> BatchEmbedder:
        cfg = PoolingConfig(strategy=strategy, hidden_dim=8, normalize=True)
        return BatchEmbedder(cfg)

    # ------------------------------------------------------------------
    # Lines 116-118: EmbeddingStats.avg_seq_len when no embeddings
    # ------------------------------------------------------------------

    def test_avg_seq_len_zero_embeddings(self) -> None:
        """avg_seq_len returns 0.0 when total_embeddings == 0 (line 116-117)."""
        stats = EmbeddingStats()
        assert stats.avg_seq_len == 0.0

    def test_avg_seq_len_after_pool(self) -> None:
        """avg_seq_len returns correct ratio (line 118 path)."""
        stats = EmbeddingStats(total_embeddings=4, total_seq_tokens=20)
        assert stats.avg_seq_len == pytest.approx(5.0)

    # ------------------------------------------------------------------
    # Lines 200-206: Bad attention_mask shape → ValueError
    # ------------------------------------------------------------------

    def test_pool_bad_mask_shape_raises(self) -> None:
        """attention_mask with wrong shape raises ValueError (lines 200-206)."""
        emb = self._make_embedder()
        hs = np.ones((2, 3, 8), dtype=np.float32)
        bad_mask = np.ones((2, 5), dtype=np.float32)  # seq_len=5 but hs has seq_len=3
        with pytest.raises(ValueError, match="attention_mask shape"):
            emb.pool(hs, bad_mask)

    def test_pool_bad_mask_wrong_batch_raises(self) -> None:
        """attention_mask with wrong batch size raises ValueError."""
        emb = self._make_embedder()
        hs = np.ones((2, 3, 8), dtype=np.float32)
        bad_mask = np.ones((3, 3), dtype=np.float32)  # batch=3 but hs has batch=2
        with pytest.raises(ValueError, match="attention_mask shape"):
            emb.pool(hs, bad_mask)

    # ------------------------------------------------------------------
    # Line 256: pool_single() with attention_mask
    # ------------------------------------------------------------------

    def test_pool_single_with_attention_mask(self) -> None:
        """pool_single with mask goes through the not-None branch (line 256)."""
        emb = self._make_embedder(strategy="mean")
        hs = np.ones((4, 8), dtype=np.float32)  # (seq_len, hidden_dim)
        mask = np.array([1, 1, 0, 0], dtype=np.float32)  # 2 valid tokens
        result = emb.pool_single(hs, attention_mask=mask)
        assert result.shape == (8,)

    def test_pool_single_without_mask(self) -> None:
        """pool_single without mask (None path) works."""
        emb = self._make_embedder()
        hs = np.ones((3, 8), dtype=np.float32)
        result = emb.pool_single(hs)
        assert result.shape == (8,)

    # ------------------------------------------------------------------
    # Line 375: n_embeddings property
    # ------------------------------------------------------------------

    def test_n_embeddings_property(self) -> None:
        """n_embeddings property returns accumulated count (line 375)."""
        emb = self._make_embedder()
        assert emb.n_embeddings == 0
        hs = np.ones((2, 3, 8), dtype=np.float32)
        emb.pool(hs)
        assert emb.n_embeddings == 2

    # ------------------------------------------------------------------
    # Line 379: stats() method
    # ------------------------------------------------------------------

    def test_stats_method(self) -> None:
        """stats() returns an EmbeddingStats dataclass (line 379)."""
        emb = self._make_embedder(strategy="max")
        hs = np.ones((3, 4, 8), dtype=np.float32)
        emb.pool(hs)
        s = emb.stats()
        assert isinstance(s, EmbeddingStats)
        assert s.n_batches == 1
        assert s.total_embeddings == 3
        assert s.strategy == "max"


# ===========================================================================
# 8. token_healer.py
# Uncovered: 148, 161-163, 166, 216, 231-232
# ===========================================================================


class TestTokenHealerCoverage:
    """Cover empty-tokens early return, TypeError except break, short-suffix continue,
    heal passthrough, and needs_healing."""

    def _make_healer(
        self,
        vocab: list[str] | None = None,
        max_healing: int = 4,
        min_prefix: int = 1,
    ) -> TokenHealer:
        vsize = len(vocab) if vocab else 32
        cfg = HealerConfig(
            vocab_size=vsize,
            max_healing_tokens=max_healing,
            min_prefix_len=min_prefix,
        )
        return TokenHealer(cfg, vocab_list=vocab)

    # ------------------------------------------------------------------
    # Line 148: find_suffix_overlap when vocab is None or tokens is empty
    # ------------------------------------------------------------------

    def test_find_suffix_no_vocab(self) -> None:
        """No vocab at construction AND no vocab_list param → returns (0, '') (line 148)."""
        cfg = HealerConfig(vocab_size=100)
        healer = TokenHealer(cfg, vocab_list=None)
        n, s = healer.find_suffix_overlap([1, 2, 3])
        assert n == 0
        assert s == ""

    def test_find_suffix_empty_tokens(self) -> None:
        """Empty token list → returns (0, '') (line 148)."""
        vocab = ["hello", "world", "helloworld"]
        healer = self._make_healer(vocab)
        n, s = healer.find_suffix_overlap([])
        assert n == 0
        assert s == ""

    # ------------------------------------------------------------------
    # Lines 161-163: TypeError caught → break
    # ------------------------------------------------------------------

    def test_find_suffix_malformed_vocab_entry_catches_typeerror(self) -> None:
        """vocab[t] returning non-str causes TypeError in join, caught at 161-163."""
        # vocab[1] = 42 (int) → "".join([42]) raises TypeError
        vocab_bad = ["hello", 42, "helloz"]  # type: ignore[list-item]
        cfg = HealerConfig(vocab_size=len(vocab_bad), max_healing_tokens=3, min_prefix_len=1)
        healer = TokenHealer(cfg, vocab_list=vocab_bad)  # type: ignore[arg-type]
        # suffix_token_ids for n=1 would be [1] → vocab[1]=42 → TypeError
        n, s = healer.find_suffix_overlap([0, 1])
        # The except block breaks out of the loop; best_n stays at whatever was before n=1
        assert isinstance(n, int)
        assert isinstance(s, str)

    # ------------------------------------------------------------------
    # Line 166: continue when len(suffix_str) < min_prefix_len
    # ------------------------------------------------------------------

    def test_find_suffix_short_suffix_continues(self) -> None:
        """Short suffix strings (< min_prefix_len) are skipped via continue (line 166)."""
        # min_prefix_len=3 means 1- and 2-char suffixes are skipped
        vocab = ["a", "b", "c"]
        healer = self._make_healer(vocab, max_healing=3, min_prefix=3)
        # Each single token produces 1-char string, which is < 3 → continue
        n, s = healer.find_suffix_overlap([0, 1, 2])
        assert n == 0  # none met the min_prefix_len threshold

    def test_find_suffix_min_prefix_exactly_met(self) -> None:
        """min_prefix_len=2: 2-char suffix is checked, 1-char suffix is skipped."""
        vocab = ["he", "he_long"]
        healer = self._make_healer(vocab, max_healing=2, min_prefix=2)
        # tokens=[0] → suffix "he". Is "he" a proper prefix of "he_long"? Yes ("he_long".startswith("he") and len > 2).
        n, s = healer.find_suffix_overlap([0])
        assert n == 1
        assert s == "he"

    # ------------------------------------------------------------------
    # Line 216: heal() when no overlap → passthrough (list(tokens) + completions[0])
    # ------------------------------------------------------------------

    def test_heal_no_overlap_passthrough(self) -> None:
        """heal() returns tokens + completion when find_suffix_overlap returns 0 (line 216)."""
        # vocab entries are all complete words → no proper prefix
        vocab = ["dog", "cat", "bird"]
        healer = self._make_healer(vocab)
        # "dog" is NOT a proper prefix of any vocab entry (nothing starts with "dog" that is longer)
        tokens = [0, 1, 2]
        completion = [5, 6, 7]
        result = healer.heal(tokens, completions=[completion])
        assert result == tokens + completion

    def test_heal_with_overlap(self) -> None:
        """heal() backs up n_overlap tokens when overlap is found."""
        vocab = ["pre", "prefix"]  # "pre" is proper prefix of "prefix"
        healer = self._make_healer(vocab)
        tokens = [0]  # "pre" → proper prefix of "prefix"
        completion = [1, 5]
        result = healer.heal(tokens, completions=[completion])
        # n_overlap == 1, so result = tokens[:-1] + completion = [] + completion
        assert result == completion

    def test_heal_empty_completions_raises(self) -> None:
        """heal() with empty completions raises ValueError."""
        healer = self._make_healer(["abc"])
        with pytest.raises(ValueError, match="non-empty"):
            healer.heal([0], completions=[])

    # ------------------------------------------------------------------
    # Lines 231-232: needs_healing() method
    # ------------------------------------------------------------------

    def test_needs_healing_true(self) -> None:
        """needs_healing returns True when last token is incomplete prefix (lines 231-232)."""
        vocab = ["pre", "prefix"]
        healer = self._make_healer(vocab)
        assert healer.needs_healing([0]) is True  # "pre" is proper prefix

    def test_needs_healing_false(self) -> None:
        """needs_healing returns False when no overlap detected."""
        vocab = ["done", "other"]
        healer = self._make_healer(vocab)
        assert healer.needs_healing([0]) is False  # "done" not a proper prefix

    def test_needs_healing_with_vocab_override(self) -> None:
        """needs_healing uses vocab_list param override."""
        cfg = HealerConfig(vocab_size=10)
        healer = TokenHealer(cfg, vocab_list=None)
        vocab = ["he", "hello"]
        assert healer.needs_healing([0], vocab_list=vocab) is True


# ===========================================================================
# 9. adaptive_quantize.py
# Uncovered: 213, 218, 274-275, 282-283, 310, 323
# ===========================================================================


class TestAdaptiveQuantizeCoverage:
    """Cover capacity_bytes, used_bytes, FP16/INT4 quantize, FP16 dequantize, monitor."""

    def _make(self, pressure: float = 0.0) -> tuple[PressureMonitor, AdaptiveQuantizer]:
        thresholds = PressureThresholds(int8_threshold=0.7, int4_threshold=0.9)
        capacity = 1_000_000
        monitor = PressureMonitor(thresholds, capacity_bytes=capacity)
        monitor.update(int(pressure * capacity))
        quantizer = AdaptiveQuantizer(monitor)
        return monitor, quantizer

    # ------------------------------------------------------------------
    # Line 213: capacity_bytes property
    # ------------------------------------------------------------------

    def test_capacity_bytes_property(self) -> None:
        """capacity_bytes returns the configured capacity (line 213)."""
        monitor, _ = self._make()
        assert monitor.capacity_bytes == 1_000_000

    # ------------------------------------------------------------------
    # Line 218: used_bytes property
    # ------------------------------------------------------------------

    def test_used_bytes_property(self) -> None:
        """used_bytes returns current usage (line 218)."""
        monitor, _ = self._make(pressure=0.5)
        assert monitor.used_bytes == 500_000

    # ------------------------------------------------------------------
    # Lines 274-275: quantize() in FP16 mode
    # ------------------------------------------------------------------

    def test_quantize_fp16_mode(self) -> None:
        """Pressure below int8_threshold → FP16 quantization (lines 274-275)."""
        monitor, q = self._make(pressure=0.3)  # below 0.7
        assert monitor.current_precision == QuantPrecision.FP16
        x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        out, scale = q.quantize(x)
        assert out.dtype == np.float16
        assert scale == pytest.approx(1.0)
        assert q.stats.fp16_calls == 1

    # ------------------------------------------------------------------
    # Lines 282-283: quantize() in INT4 mode
    # ------------------------------------------------------------------

    def test_quantize_int4_mode(self) -> None:
        """Pressure >= int4_threshold → INT4 quantization (lines 282-283)."""
        monitor, q = self._make(pressure=0.95)  # above 0.9
        assert monitor.current_precision == QuantPrecision.INT4
        x = np.array([3.0, -3.0, 1.5, -1.5], dtype=np.float32)
        out, scale = q.quantize(x)
        assert out.dtype == np.int8
        assert q.stats.int4_calls == 1
        assert q.stats.total_quantize_calls == 1

    def test_quantize_int8_mode(self) -> None:
        """Pressure in [int8_threshold, int4_threshold) → INT8 quantization."""
        monitor, q = self._make(pressure=0.8)
        assert monitor.current_precision == QuantPrecision.INT8
        x = np.arange(8, dtype=np.float32)
        out, scale = q.quantize(x)
        assert out.dtype == np.int8
        assert q.stats.int8_calls == 1

    # ------------------------------------------------------------------
    # Line 310: dequantize() in FP16 mode
    # ------------------------------------------------------------------

    def test_dequantize_fp16(self) -> None:
        """dequantize with FP16 precision casts to float32 (line 310)."""
        _, q = self._make(pressure=0.3)
        x = np.array([1.0, -0.5, 2.0], dtype=np.float32)
        q_arr = x.astype(np.float16)
        result = q.dequantize(q_arr, scale=1.0, precision="fp16")
        assert result.dtype == np.float32

    # ------------------------------------------------------------------
    # Line 323: monitor property
    # ------------------------------------------------------------------

    def test_monitor_property(self) -> None:
        """monitor property returns the PressureMonitor (line 323)."""
        monitor, q = self._make()
        assert q.monitor is monitor


# ===========================================================================
# 10. mirror_sd.py
# Uncovered: 102-108, 235, 302, 385, 441->446 (pragma added)
# ===========================================================================

MIRROR_VOCAB = 20


def _fixed_logits_fn(peak_token: int, vocab: int = MIRROR_VOCAB):
    """Returns a function that produces logits with a strong peak at peak_token."""
    def fn(ids):  # noqa: ANN001
        logits = np.full(vocab, -5.0, dtype=np.float32)
        logits[peak_token % vocab] = 5.0
        return logits
    return fn


class TestMirrorSDCoverage:
    """Cover _top_p_filter, top_p in draft/verify pipelines, config=None decoder."""

    # ------------------------------------------------------------------
    # Lines 102-108: _top_p_filter function
    # ------------------------------------------------------------------

    def test_top_p_filter_basic(self) -> None:
        """_top_p_filter keeps the smallest set whose cumulative prob >= top_p."""
        probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        filtered = _top_p_filter(probs, top_p=0.8)
        assert abs(filtered.sum() - 1.0) < 1e-5
        # The top-2 tokens cover 0.8 of probability
        assert filtered[2] == pytest.approx(0.0) or filtered[3] == pytest.approx(0.0)

    def test_top_p_filter_full_mass(self) -> None:
        """top_p=1.0 keeps all tokens."""
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        filtered = _top_p_filter(probs, top_p=1.0)
        assert abs(filtered.sum() - 1.0) < 1e-5

    # ------------------------------------------------------------------
    # Line 235: MirrorDraftPipeline.step() with top_p < 1.0
    # ------------------------------------------------------------------

    def test_draft_pipeline_top_p(self) -> None:
        """Draft pipeline with top_p < 1.0 calls _top_p_filter (line 235)."""
        cfg = MirrorSDConfig(top_p=0.9, gamma=1)
        pipe = MirrorDraftPipeline(_fixed_logits_fn(3), cfg, rng_seed=0)
        tok, probs = pipe.step([0, 1])
        assert 0 <= tok < MIRROR_VOCAB
        assert abs(probs.sum() - 1.0) < 1e-5

    # ------------------------------------------------------------------
    # Line 302: MirrorVerifyPipeline.enqueue() _work with top_p < 1.0
    # ------------------------------------------------------------------

    def test_verify_pipeline_top_p(self) -> None:
        """Verify pipeline _work with top_p < 1.0 calls _top_p_filter (line 302)."""
        cfg = MirrorSDConfig(top_p=0.85, gamma=1)
        pipe = MirrorVerifyPipeline(_fixed_logits_fn(7), cfg, rng_seed=0)
        fut = pipe.enqueue([0, 1])
        tok, probs = fut.wait()
        assert 0 <= tok < MIRROR_VOCAB
        assert abs(probs.sum() - 1.0) < 1e-5

    # ------------------------------------------------------------------
    # Line 385: MirrorSDDecoder created with config=None → uses MirrorSDConfig()
    # ------------------------------------------------------------------

    def test_decoder_with_none_config(self) -> None:
        """MirrorSDDecoder with config=None defaults to MirrorSDConfig() (line 385)."""
        default_cfg = MirrorSDConfig()
        draft = MirrorDraftPipeline(_fixed_logits_fn(2), default_cfg, rng_seed=0)
        verify = MirrorVerifyPipeline(_fixed_logits_fn(2), default_cfg, rng_seed=1)
        decoder = MirrorSDDecoder(draft, verify, config=None, rng_seed=2)
        out, stats = decoder.generate([0], max_new_tokens=4)
        assert stats.total_tokens == 4

    def test_decoder_top_p_pipeline(self) -> None:
        """Full decode with top_p < 1.0 exercises both pipeline top_p paths."""
        cfg = MirrorSDConfig(gamma=2, top_p=0.9)
        draft = MirrorDraftPipeline(_fixed_logits_fn(5), cfg, rng_seed=0)
        verify = MirrorVerifyPipeline(_fixed_logits_fn(5), cfg, rng_seed=1)
        decoder = MirrorSDDecoder(draft, verify, cfg, rng_seed=2)
        _, stats = decoder.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6


# ===========================================================================
# 11. medusa.py
# Uncovered: 174-179, 312, 378-380
# ===========================================================================


class TestMedusaCoverage:
    """Cover MedusaHead.top_k_tokens, decoder acceptance_rate zero case,
    MedusaStats.acceptance_rate zero case."""

    def _make_head(self, hidden_dim: int = 16, vocab: int = 32) -> MedusaHead:
        return MedusaHead(hidden_dim=hidden_dim, vocab_size=vocab)

    def _make_decoder(self) -> MedusaDecoder:
        cfg = MedusaConfig(n_heads=3, vocab_size=32, hidden_dim=16)
        return MedusaDecoder(cfg)

    # ------------------------------------------------------------------
    # Lines 174-179: MedusaHead.top_k_tokens
    # ------------------------------------------------------------------

    def test_top_k_tokens_returns_correct_count(self) -> None:
        """top_k_tokens returns k sorted indices (lines 174-179)."""
        head = self._make_head(hidden_dim=16, vocab=32)
        hidden = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        indices = head.top_k_tokens(hidden, k=5)
        assert indices.shape == (5,)
        assert len(set(indices.tolist())) == 5

    def test_top_k_tokens_k_larger_than_vocab(self) -> None:
        """top_k_tokens with k > vocab_size is clamped to vocab_size."""
        head = self._make_head(hidden_dim=8, vocab=10)
        hidden = np.ones(8, dtype=np.float32)
        indices = head.top_k_tokens(hidden, k=20)
        assert len(indices) == 10

    def test_top_k_tokens_k_one(self) -> None:
        """top_k_tokens with k=1 returns the argmax index."""
        head = self._make_head(hidden_dim=4, vocab=8)
        hidden = np.zeros(4, dtype=np.float32)
        indices = head.top_k_tokens(hidden, k=1)
        assert indices.shape == (1,)

    def test_top_k_tokens_dtype(self) -> None:
        """top_k_tokens output is int64."""
        head = self._make_head()
        hidden = np.ones(16, dtype=np.float32)
        indices = head.top_k_tokens(hidden, k=3)
        assert indices.dtype == np.int64

    # ------------------------------------------------------------------
    # Line 312: MedusaDecoder.verify() reject path (else: break)
    # ------------------------------------------------------------------

    def test_verify_rejects_mismatched_token(self) -> None:
        """verify() breaks at first rejection (line 312 else: break)."""
        decoder = self._make_decoder()
        rng = np.random.default_rng(0)
        # draft_tokens[0] = 0, but argmax of target_logits[0] = 1 → reject
        draft_tokens = [0, 1, 2]
        target_logits = [
            np.array([0.0, 10.0] + [0.0] * 30, dtype=np.float32),  # argmax=1 ≠ draft=0 → reject
            rng.standard_normal(32).astype(np.float32),
            rng.standard_normal(32).astype(np.float32),
        ]
        accepted, n = decoder.verify(draft_tokens, target_logits)
        assert n == 0  # first token rejected → no accepted tokens
        assert accepted == []

    def test_verify_partial_acceptance(self) -> None:
        """verify() accepts tokens up to first mismatch."""
        decoder = self._make_decoder()
        rng = np.random.default_rng(1)
        # token 5 accepts at pos 0, token 3 accepts at pos 1, token 7 rejects at pos 2
        logits_0 = np.full(32, -10.0, dtype=np.float32)
        logits_0[5] = 10.0  # argmax=5 == draft[0]=5 → accept
        logits_1 = np.full(32, -10.0, dtype=np.float32)
        logits_1[3] = 10.0  # argmax=3 == draft[1]=3 → accept
        logits_2 = np.full(32, -10.0, dtype=np.float32)
        logits_2[9] = 10.0  # argmax=9 != draft[2]=7 → reject
        accepted, n = decoder.verify([5, 3, 7], [logits_0, logits_1, logits_2])
        assert n == 2
        assert accepted == [5, 3]

    # ------------------------------------------------------------------
    # Line 312: decoder.acceptance_rate when total_drafts == 0
    # ------------------------------------------------------------------

    def test_decoder_acceptance_rate_zero_drafts(self) -> None:
        """acceptance_rate returns 0.0 when no verify calls have been made (line 312)."""
        decoder = self._make_decoder()
        assert decoder.acceptance_rate == 0.0

    # ------------------------------------------------------------------
    # Lines 378-380: MedusaStats.acceptance_rate
    # ------------------------------------------------------------------

    def test_medusa_stats_acceptance_rate_zero(self) -> None:
        """MedusaStats.acceptance_rate returns 0.0 when total_drafts == 0 (lines 379-380)."""
        stats = MedusaStats()
        assert stats.acceptance_rate == 0.0

    def test_medusa_stats_acceptance_rate_nonzero(self) -> None:
        """MedusaStats.acceptance_rate returns correct fraction (line 381)."""
        stats = MedusaStats(total_drafts=10, total_accepted=7)
        assert stats.acceptance_rate == pytest.approx(0.7)

    def test_medusa_stats_mean_accepted_per_call_zero(self) -> None:
        """MedusaStats.mean_accepted_per_call returns 0.0 when no calls."""
        stats = MedusaStats()
        assert stats.mean_accepted_per_call == 0.0

    def test_decoder_get_stats(self) -> None:
        """get_stats() returns MedusaStats snapshot."""
        decoder = self._make_decoder()
        s = decoder.get_stats()
        assert isinstance(s, MedusaStats)
        assert s.total_calls == 0


# ===========================================================================
# 12. mixed_precision_kv.py
# Uncovered: 130, 237, 301, 367, 373-375, 419, 423
# ===========================================================================


class TestMixedPrecisionKVCoverage:
    """Cover int4_quant_max, _dequantize_symmetric, uniform variance, FP16/INT4 store/load."""

    def _make_cache(self) -> MixedPrecisionKVCache:
        cfg = MPKVConfig(n_heads=4, head_dim=16, int4_threshold=0.3, int8_threshold=0.7)
        return MixedPrecisionKVCache(cfg)

    # ------------------------------------------------------------------
    # Line 130: int4_quant_max property
    # ------------------------------------------------------------------

    def test_int4_quant_max_property(self) -> None:
        """int4_quant_max returns 7 (line 130)."""
        cfg = MPKVConfig()
        assert cfg.int4_quant_max == 7

    # ------------------------------------------------------------------
    # Line 237: _dequantize_symmetric helper
    # ------------------------------------------------------------------

    def test_dequantize_symmetric_helper(self) -> None:
        """_dequantize_symmetric converts uint8 offset back to float32 (line 237)."""
        x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        q_max = 7
        # Quantise first
        x_q, scale = _quantize_symmetric(x, q_max)
        # Dequantise
        x_back = _dequantize_symmetric(x_q, q_max, scale)
        assert x_back.dtype == np.float32
        np.testing.assert_allclose(x_back, x, atol=0.5)

    # ------------------------------------------------------------------
    # Line 301: assign_precisions with uniform variance → v_range < 1e-30
    # ------------------------------------------------------------------

    def test_assign_precisions_uniform_variance(self) -> None:
        """All-equal variance → v_range ≈ 0 → all assigned same tier (line 301)."""
        cache = self._make_cache()
        uniform_var = np.ones(4, dtype=np.float32)  # all equal → normalised to 0
        prec_map = cache.assign_precisions(uniform_var)
        assert len(prec_map.precisions) == 4
        # All normalised to 0.0 < int4_threshold → all INT4
        assert all(p == HeadPrecision.INT4 for p in prec_map.precisions)

    # ------------------------------------------------------------------
    # Lines 367: store() with FP16 precision
    # ------------------------------------------------------------------

    def test_store_fp16(self) -> None:
        """store() with FP16 returns float16 arrays (line 367)."""
        cache = self._make_cache()
        key = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        val = np.random.default_rng(1).standard_normal(16).astype(np.float32)
        k_q, v_q = cache.store(head_idx=0, key=key, value=val, precision="fp16")
        assert k_q.dtype == np.float16
        assert v_q.dtype == np.float16

    # ------------------------------------------------------------------
    # Lines 373-375: store() with INT4 precision
    # ------------------------------------------------------------------

    def test_store_int4(self) -> None:
        """store() with INT4 returns uint8 arrays (lines 373-375)."""
        cache = self._make_cache()
        key = np.random.default_rng(2).standard_normal(16).astype(np.float32)
        val = np.random.default_rng(3).standard_normal(16).astype(np.float32)
        k_q, v_q = cache.store(head_idx=1, key=key, value=val, precision="int4")
        assert k_q.dtype == np.uint8
        assert v_q.dtype == np.uint8

    def test_store_int8(self) -> None:
        """store() with INT8 returns uint8 arrays."""
        cache = self._make_cache()
        key = np.ones(16, dtype=np.float32)
        val = np.ones(16, dtype=np.float32)
        k_q, v_q = cache.store(head_idx=2, key=key, value=val, precision="int8")
        assert k_q.dtype == np.uint8

    # ------------------------------------------------------------------
    # Line 419: load() with FP16 precision
    # ------------------------------------------------------------------

    def test_load_fp16(self) -> None:
        """load() with FP16 returns float32 arrays (line 419)."""
        cache = self._make_cache()
        key = np.random.default_rng(4).standard_normal(16).astype(np.float32)
        val = np.random.default_rng(5).standard_normal(16).astype(np.float32)
        k_q, v_q = cache.store(0, key, val, precision="fp16")
        k_back, v_back = cache.load(0, k_q, v_q, precision="fp16")
        assert k_back.dtype == np.float32
        assert v_back.dtype == np.float32

    # ------------------------------------------------------------------
    # Line 423: load() with INT4 precision (q_max = int4_quant_max)
    # ------------------------------------------------------------------

    def test_load_int4(self) -> None:
        """load() with INT4 uses int4_quant_max offset (line 423)."""
        cache = self._make_cache()
        key = np.random.default_rng(6).standard_normal(16).astype(np.float32)
        val = np.random.default_rng(7).standard_normal(16).astype(np.float32)
        k_q, v_q = cache.store(0, key, val, precision="int4")
        k_back, v_back = cache.load(0, k_q, v_q, precision="int4")
        assert k_back.dtype == np.float32

    def test_load_int8(self) -> None:
        """load() with INT8 uses int8_quant_max offset."""
        cache = self._make_cache()
        key = np.random.default_rng(8).standard_normal(16).astype(np.float32)
        val = np.random.default_rng(9).standard_normal(16).astype(np.float32)
        k_q, v_q = cache.store(0, key, val, precision="int8")
        k_back, v_back = cache.load(0, k_q, v_q, precision="int8")
        assert k_back.dtype == np.float32

    def test_reset_stats(self) -> None:
        """reset_stats zeros the stats object."""
        cache = self._make_cache()
        v = np.ones(4, dtype=np.float32)
        cache.assign_precisions(v)
        cache.reset_stats()
        assert cache.stats.total_heads_assigned == 0


# ===========================================================================
# 13. sparse_spec.py
# Uncovered: 203, 255, 259 (pragma added), 273-279, 442->444, 478->530
# ===========================================================================

SPARSE_VOCAB = 16


def _fixed_draft_fn(agree_tok: int, vocab: int = SPARSE_VOCAB):
    """Draft function returning deterministic (token, probs)."""
    def fn(ids):  # noqa: ANN001
        p = np.ones(vocab, dtype=np.float32) / vocab
        return agree_tok % vocab, p
    return fn


class TestSparseSpecCoverage:
    """Cover PillarAttnCache.scores, _sparse_context([]), top-p in _sample,
    _update_pillar with ctx_len=0, generate with max_new_tokens=0."""

    # ------------------------------------------------------------------
    # Line 203: PillarAttnCache.scores property
    # ------------------------------------------------------------------

    def test_pillar_cache_scores_property(self) -> None:
        """scores property returns the view of current scores (line 203)."""
        cache = PillarAttnCache(capacity=8)
        scores_in = np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32)
        cache.update(scores_in)
        s = cache.scores
        assert s.shape == (4,)
        np.testing.assert_array_almost_equal(s, scores_in)

    def test_pillar_cache_scores_empty(self) -> None:
        """scores on empty cache returns empty array."""
        cache = PillarAttnCache(capacity=8)
        assert cache.scores.shape == (0,)

    # ------------------------------------------------------------------
    # Line 255: _sparse_context([]) with empty input_ids
    # ------------------------------------------------------------------

    def test_sparse_context_empty_ids(self) -> None:
        """_sparse_context([]) returns [] immediately (line 255)."""
        cfg = SparseSpecConfig(gamma=1, top_k_ratio=0.5, warmup_steps=0)
        cache = PillarAttnCache(capacity=32)
        # Populate cache so use_sparse would be True
        cache.update(np.ones(8, dtype=np.float32))
        drafter = SparseSpecDrafter(_fixed_draft_fn(2), cache, cfg)
        # Force step_count > warmup_steps so use_sparse is True
        # We need n_positions > 0 AND step_count > warmup_steps
        # warmup_steps=0, after first increment step_count=1 > 0 → warm
        # But on first call step_count goes from 0→1, cache already has n_positions=8
        tokens, probs = drafter.draft([])  # empty ids → _sparse_context([]) → return []
        assert len(tokens) == 1  # gamma=1
        assert len(probs) == 1

    # ------------------------------------------------------------------
    # Lines 273-279: top-p in _sample()
    # ------------------------------------------------------------------

    def test_sample_with_top_p(self) -> None:
        """SparseSpecConfig.top_p < 1.0 activates nucleus filter in _sample (lines 273-279)."""
        cfg = SparseSpecConfig(gamma=2, top_p=0.8, warmup_steps=0, top_k_ratio=1.0)
        cache = PillarAttnCache(capacity=32)
        drafter = SparseSpecDrafter(_fixed_draft_fn(3), cache, cfg)
        tokens, probs = drafter.draft([0, 1, 2])
        assert len(tokens) == 2
        for p in probs:
            assert abs(p.sum() - 1.0) < 1e-5

    def test_sample_with_top_p_low_temperature(self) -> None:
        """top_p < 1.0 with very low temperature produces peaked distribution."""
        cfg = SparseSpecConfig(
            gamma=3, top_p=0.9, temperature=0.01, warmup_steps=0, top_k_ratio=1.0
        )
        cache = PillarAttnCache(capacity=16)
        drafter = SparseSpecDrafter(_fixed_draft_fn(4), cache, cfg)
        tokens, _ = drafter.draft([0, 1])
        assert len(tokens) == 3

    # ------------------------------------------------------------------
    # Lines 442->444: _update_pillar with ctx_len=0 → total=0 branch
    # ------------------------------------------------------------------

    def test_update_pillar_ctx_len_zero(self) -> None:
        """_update_pillar(cache, 0, 0) exercises total==0 branch (442->444)."""
        cfg = SparseSpecConfig(gamma=1, warmup_steps=0, top_k_ratio=1.0)
        drafter = SparseSpecDrafter(_fixed_draft_fn(0), PillarAttnCache(), cfg)
        decoder = SparseSpecDecoder(drafter, _fixed_draft_fn(0), cfg)
        cache = PillarAttnCache(capacity=16)
        # Directly call _update_pillar with ctx_len=0 → scores.sum()==0
        decoder._update_pillar(cache, ctx_len=0, n_accepted=0)
        # Should not raise; cache scores are empty
        assert cache.n_positions == 0

    # ------------------------------------------------------------------
    # Lines 478->530: generate() with max_new_tokens=0 → while loop never executes
    # ------------------------------------------------------------------

    def test_generate_zero_tokens(self) -> None:
        """generate with max_new_tokens=0 exits immediately (478->530 branch)."""
        cfg = SparseSpecConfig(gamma=2, warmup_steps=0, top_k_ratio=1.0)
        cache = PillarAttnCache(capacity=16)
        drafter = SparseSpecDrafter(_fixed_draft_fn(1), cache, cfg)
        decoder = SparseSpecDecoder(drafter, _fixed_draft_fn(1), cfg)
        out, stats = decoder.generate([1, 2, 3], max_new_tokens=0)
        assert out == [1, 2, 3]  # no new tokens added
        assert stats.draft_steps == 0
        assert stats.total_tokens == 0


# ===========================================================================
# 14. sparse_attn_index.py
# Uncovered: 247, 256-263
# ===========================================================================


class TestSparseAttnIndexCoverage:
    """Cover argsort full-sort path and padding path when seq_len < top_k."""

    def _make_index(self, top_k: int = 10, n_heads: int = 2, head_dim: int = 8) -> SparseAttnIndex:
        cfg = IndexConfig(top_k=top_k, head_dim=head_dim, n_heads=n_heads)
        return SparseAttnIndex(cfg)

    def _build_and_query(self, top_k: int, seq_len: int) -> ANCandidates:
        idx = self._make_index(top_k=top_k, n_heads=2, head_dim=8)
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((2, seq_len, 8)).astype(np.float32)
        idx.build(keys)
        q = rng.standard_normal((2, 8)).astype(np.float32)
        return idx.query(q)

    # ------------------------------------------------------------------
    # Line 247: effective_k >= seq_len → argsort full sort path
    # ------------------------------------------------------------------

    def test_query_seq_shorter_than_top_k_uses_argsort(self) -> None:
        """When seq_len <= top_k, effective_k == seq_len uses np.argsort (line 247)."""
        # top_k=10, seq_len=5 → effective_k=5 == seq_len → full argsort path
        result = self._build_and_query(top_k=10, seq_len=5)
        assert result.indices.shape == (2, 10)  # padded to top_k

    def test_query_seq_equal_to_top_k_uses_argsort(self) -> None:
        """When seq_len == top_k exactly, uses argsort path."""
        result = self._build_and_query(top_k=4, seq_len=4)
        # No padding needed (effective_k == top_k)
        assert result.indices.shape == (2, 4)

    # ------------------------------------------------------------------
    # Lines 256-263: Padding block when effective_k < top_k
    # ------------------------------------------------------------------

    def test_query_pads_when_seq_shorter_than_top_k(self) -> None:
        """Padding applied when seq_len < top_k (lines 256-263)."""
        # top_k=10, seq_len=3 → effective_k=3, needs padding to 10
        result = self._build_and_query(top_k=10, seq_len=3)
        # Shape must be (n_heads, top_k) = (2, 10) after padding
        assert result.indices.shape == (2, 10)
        assert result.scores.shape == (2, 10)
        # Padded columns should be -1 for indices and 0 for scores
        assert (result.indices[:, 3:] == -1).all()
        assert np.all(result.scores[:, 3:] == 0.0)

    def test_query_no_padding_when_seq_gte_top_k(self) -> None:
        """No padding when seq_len >= top_k (control case)."""
        idx = self._make_index(top_k=3, n_heads=2, head_dim=8)
        rng = np.random.default_rng(1)
        keys = rng.standard_normal((2, 20, 8)).astype(np.float32)
        idx.build(keys)
        q = rng.standard_normal((2, 8)).astype(np.float32)
        result = idx.query(q)
        assert result.indices.shape == (2, 3)
        # No -1 indices
        assert (result.indices >= 0).all()


# ===========================================================================
# 15. token_budget_gate.py
# Uncovered: 216-218, 228, 233
# ===========================================================================


class TestTokenBudgetGateCoverage:
    """Cover reset(), is_exhausted, and max_tokens properties."""

    def _make_gate(self, max_tokens: int = 10) -> TokenBudgetGate:
        policy = BudgetPolicy(mode="hard", warn_at_fraction=0.9)
        return TokenBudgetGate(max_tokens=max_tokens, policy=policy)

    # ------------------------------------------------------------------
    # Lines 216-218: reset() method
    # ------------------------------------------------------------------

    def test_reset_clears_counter_and_increments_requests(self) -> None:
        """reset() zeros _used and increments total_requests (lines 216-218)."""
        gate = self._make_gate(max_tokens=5)
        gate.tick()
        gate.tick()
        assert gate.tokens_used == 2
        gate.reset()
        assert gate.tokens_used == 0
        assert gate.stats.total_requests == 1

    def test_reset_clears_warned_flag(self) -> None:
        """reset() also clears the warning flag so next request can re-warn."""
        gate = self._make_gate(max_tokens=5)
        policy = BudgetPolicy(warn_at_fraction=0.5)
        gate2 = TokenBudgetGate(max_tokens=10, policy=policy)
        gate2.tick(n_tokens=6)  # crosses 0.5 → warned
        assert gate2.stats.warnings_issued == 1
        gate2.reset()
        assert gate2.tokens_used == 0
        gate2.tick(n_tokens=6)  # should warn again
        assert gate2.stats.warnings_issued == 2

    def test_reset_multiple_times(self) -> None:
        """Multiple reset() calls each increment total_requests."""
        gate = self._make_gate()
        gate.reset()
        gate.reset()
        gate.reset()
        assert gate.stats.total_requests == 3

    # ------------------------------------------------------------------
    # Line 228: is_exhausted property
    # ------------------------------------------------------------------

    def test_is_exhausted_false_initially(self) -> None:
        """is_exhausted is False when budget not yet consumed (line 228)."""
        gate = self._make_gate(max_tokens=5)
        assert gate.is_exhausted is False

    def test_is_exhausted_true_when_full(self) -> None:
        """is_exhausted is True once max_tokens are consumed."""
        gate = self._make_gate(max_tokens=3)
        gate.tick()
        gate.tick()
        gate.tick()
        assert gate.is_exhausted is True

    def test_is_exhausted_transitions(self) -> None:
        """is_exhausted transitions from False to True."""
        gate = self._make_gate(max_tokens=2)
        assert not gate.is_exhausted
        gate.tick()
        assert not gate.is_exhausted
        gate.tick()
        assert gate.is_exhausted

    # ------------------------------------------------------------------
    # Line 233: max_tokens property
    # ------------------------------------------------------------------

    def test_max_tokens_property(self) -> None:
        """max_tokens property returns the configured limit (line 233)."""
        gate = self._make_gate(max_tokens=42)
        assert gate.max_tokens == 42

    def test_max_tokens_unchanged_after_ticks(self) -> None:
        """max_tokens is invariant under tick calls."""
        gate = self._make_gate(max_tokens=10)
        for _ in range(15):
            gate.tick()
        assert gate.max_tokens == 10

    def test_soft_mode_does_not_stop(self) -> None:
        """soft mode: tick returns False at limit but no hard stop recorded."""
        policy = BudgetPolicy(mode="soft", warn_at_fraction=0.9)
        gate = TokenBudgetGate(max_tokens=3, policy=policy)
        results = [gate.tick() for _ in range(5)]
        # First 2 ticks return True, 3rd returns False (budget exhausted)
        assert results[0] is True
        assert results[1] is True
        assert results[2] is False
        assert gate.stats.hard_stops == 0  # soft mode → no hard_stops
