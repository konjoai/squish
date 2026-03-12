"""tests/test_wave23_server_wiring.py

Verifies that all Wave 23 module classes are importable and have the expected
public APIs. No GPU required — pure numpy smoke tests.

Wave 23 — Multi-Modal & Long Context Intelligence:
    vision_kv_fuse, image_token_prune, rag_prefetch, cot_compress,
    multimodal_batch, contextual_rerank, cross_modal_attn,
    hierarchical_kv, stream_rag, cross_doc_attn, video_frame_prune,
    embedding_gate, long_context_chunk, modality_router
"""
from __future__ import annotations

import numpy as np
import pytest

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 3. rag_prefetch
# ---------------------------------------------------------------------------

class TestRAGPrefetch:
    def test_imports(self):
        from squish.rag_prefetch import RAGConfig, RAGPrefetcher, RAGStats
        assert RAGConfig is not None
        assert RAGPrefetcher is not None
        assert RAGStats is not None

    def test_record_access(self):
        from squish.rag_prefetch import RAGConfig, RAGPrefetcher
        cfg = RAGConfig(max_docs=64, top_k=4)
        prefetcher = RAGPrefetcher(cfg)
        tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
        doc_id = prefetcher.record_access(tokens)
        assert isinstance(doc_id, str)
        assert prefetcher.n_tracked == 1

    def test_get_warmup_candidates(self):
        from squish.rag_prefetch import RAGConfig, RAGPrefetcher
        cfg = RAGConfig(max_docs=64, top_k=4, min_accesses=1)
        prefetcher = RAGPrefetcher(cfg)
        tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
        for _ in range(3):
            prefetcher.record_access(tokens)
        candidates = prefetcher.get_warmup_candidates()
        assert isinstance(candidates, list)

    def test_stats(self):
        from squish.rag_prefetch import RAGConfig, RAGPrefetcher
        cfg = RAGConfig(max_docs=64, top_k=4, min_accesses=1)
        prefetcher = RAGPrefetcher(cfg)
        tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
        prefetcher.record_access(tokens)
        stats = prefetcher.stats
        assert stats.total_accesses >= 1


# ---------------------------------------------------------------------------
# 4. cot_compress
# ---------------------------------------------------------------------------

class TestCoTCompress:
    def test_imports(self):
        from squish.cot_compress import CoTConfig, CoTCompressor, CoTStats
        assert CoTConfig is not None
        assert CoTCompressor is not None
        assert CoTStats is not None

    def test_compress_reduces_tokens(self):
        from squish.cot_compress import CoTConfig, CoTCompressor
        cfg = CoTConfig(compress_ratio=0.5, min_tokens=4)
        compressor = CoTCompressor(cfg)
        token_ids = RNG.integers(0, 1000, size=(100,)).astype(np.int32)
        compressed = compressor.compress(token_ids)
        assert len(compressed) < len(token_ids)

    def test_compress_respects_min_tokens(self):
        from squish.cot_compress import CoTConfig, CoTCompressor
        cfg = CoTConfig(compress_ratio=0.9, min_tokens=10)
        compressor = CoTCompressor(cfg)
        token_ids = RNG.integers(0, 1000, size=(12,)).astype(np.int32)
        compressed = compressor.compress(token_ids)
        assert len(compressed) >= 10

    def test_stats(self):
        from squish.cot_compress import CoTConfig, CoTCompressor
        cfg = CoTConfig(compress_ratio=0.5, min_tokens=4)
        compressor = CoTCompressor(cfg)
        token_ids = RNG.integers(0, 1000, size=(100,)).astype(np.int32)
        compressor.compress(token_ids)
        stats = compressor.stats
        assert stats.total_compress_calls == 1
        assert stats.total_tokens_in == 100


# ---------------------------------------------------------------------------
# 6. contextual_rerank
# ---------------------------------------------------------------------------

class TestContextualRerank:
    def test_imports(self):
        from squish.contextual_rerank import RerankConfig, ContextualReranker, RerankStats
        assert RerankConfig is not None
        assert ContextualReranker is not None
        assert RerankStats is not None

    def test_rerank_shape(self):
        from squish.contextual_rerank import RerankConfig, ContextualReranker
        cfg = RerankConfig(n_heads=2, head_dim=8, top_k=8)
        reranker = ContextualReranker(cfg)
        # keys shape: (n_heads, seq_len, head_dim)
        keys = RNG.random((2, 16, 8)).astype(np.float32)
        indices = reranker.rerank(keys)
        assert indices.shape[0] <= 8

    def test_rerank_with_query(self):
        from squish.contextual_rerank import RerankConfig, ContextualReranker
        cfg = RerankConfig(n_heads=2, head_dim=8, top_k=4)
        reranker = ContextualReranker(cfg)
        keys = RNG.random((2, 16, 8)).astype(np.float32)
        query = RNG.random((2, 8)).astype(np.float32)
        indices = reranker.rerank(keys, query=query)
        assert len(indices) <= 4

    def test_stats(self):
        from squish.contextual_rerank import RerankConfig, ContextualReranker
        cfg = RerankConfig(n_heads=2, head_dim=8, top_k=4)
        reranker = ContextualReranker(cfg)
        keys = RNG.random((2, 16, 8)).astype(np.float32)
        reranker.rerank(keys)
        stats = reranker.stats
        assert stats.total_rerank_calls == 1
        assert stats.total_positions_ranked == 16


# ---------------------------------------------------------------------------
# 8. hierarchical_kv
# ---------------------------------------------------------------------------

class TestHierarchicalKV:
    def test_imports(self):
        from squish.hierarchical_kv import TierConfig, HierarchicalKVStore, HierarchicalKVStats
        assert TierConfig is not None
        assert HierarchicalKVStore is not None
        assert HierarchicalKVStats is not None

    def test_put_and_get(self):
        from squish.hierarchical_kv import TierConfig, HierarchicalKVStore
        cfg = TierConfig(hot_capacity=4, warm_capacity=16, cold_capacity=64, n_heads=2, head_dim=8)
        store = HierarchicalKVStore(cfg)
        k = RNG.random((2, 8)).astype(np.float32)
        v = RNG.random((2, 8)).astype(np.float32)
        store.put(0, k, v)
        result = store.get(0)
        assert result is not None
        rk, rv = result
        assert rk.shape == k.shape

    def test_miss_returns_none(self):
        from squish.hierarchical_kv import TierConfig, HierarchicalKVStore
        cfg = TierConfig(hot_capacity=4, warm_capacity=16, cold_capacity=64, n_heads=2, head_dim=8)
        store = HierarchicalKVStore(cfg)
        assert store.get(999) is None

    def test_stats_hit_rate(self):
        from squish.hierarchical_kv import TierConfig, HierarchicalKVStore
        cfg = TierConfig(hot_capacity=4, warm_capacity=16, cold_capacity=64, n_heads=2, head_dim=8)
        store = HierarchicalKVStore(cfg)
        k = RNG.random((2, 8)).astype(np.float32)
        v = RNG.random((2, 8)).astype(np.float32)
        store.put(0, k, v)
        store.get(0)
        store.get(999)  # miss
        stats = store.stats
        assert 0.0 <= stats.hit_rate <= 1.0


# ---------------------------------------------------------------------------
# 9. stream_rag
# ---------------------------------------------------------------------------

class TestStreamRAG:
    def test_imports(self):
        from squish.stream_rag import StreamRAGConfig, RAGDocument, StreamRAGInjector, StreamRAGStats
        assert StreamRAGConfig is not None
        assert RAGDocument is not None
        assert StreamRAGInjector is not None
        assert StreamRAGStats is not None

    def test_inject_and_retrieve(self):
        from squish.stream_rag import StreamRAGConfig, StreamRAGInjector
        cfg = StreamRAGConfig(max_docs=8, embed_dim=16, top_k_retrieve=2)
        injector = StreamRAGInjector(cfg)
        tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
        emb = RNG.random((16,)).astype(np.float32)
        injector.inject("doc1", tokens, emb)
        query = RNG.random((16,)).astype(np.float32)
        docs = injector.retrieve(query)
        assert len(docs) >= 1

    def test_retrieve_top_k(self):
        from squish.stream_rag import StreamRAGConfig, StreamRAGInjector
        cfg = StreamRAGConfig(max_docs=8, embed_dim=16, top_k_retrieve=2)
        injector = StreamRAGInjector(cfg)
        for i in range(4):
            tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
            emb = RNG.random((16,)).astype(np.float32)
            injector.inject(f"doc{i}", tokens, emb)
        query = RNG.random((16,)).astype(np.float32)
        docs = injector.retrieve(query, top_k=2)
        assert len(docs) <= 2

    def test_stats(self):
        from squish.stream_rag import StreamRAGConfig, StreamRAGInjector
        cfg = StreamRAGConfig(max_docs=8, embed_dim=16, top_k_retrieve=2)
        injector = StreamRAGInjector(cfg)
        tokens = RNG.integers(0, 100, size=(32,)).astype(np.int32)
        emb = RNG.random((16,)).astype(np.float32)
        injector.inject("doc1", tokens, emb)
        query = RNG.random((16,)).astype(np.float32)
        injector.retrieve(query)
        stats = injector.stats
        assert stats.total_injections == 1
        assert stats.total_retrievals == 1


# ---------------------------------------------------------------------------
# 10. cross_doc_attn
# ---------------------------------------------------------------------------

class TestCrossDocAttn:
    def test_imports(self):
        from squish.cross_doc_attn import CrossDocConfig, CrossDocAttention, CrossDocStats
        assert CrossDocConfig is not None
        assert CrossDocAttention is not None
        assert CrossDocStats is not None

    def test_forward_shape(self):
        from squish.cross_doc_attn import CrossDocConfig, CrossDocAttention
        cfg = CrossDocConfig(n_heads=2, head_dim=8, max_docs=4)
        attn = CrossDocAttention(cfg)
        # query: (n_heads, seq_q, head_dim)
        query = RNG.random((2, 4, 8)).astype(np.float32)
        doc_keys = [RNG.random((2, 6, 8)).astype(np.float32) for _ in range(2)]
        doc_vals = [RNG.random((2, 6, 8)).astype(np.float32) for _ in range(2)]
        out = attn.forward(query, doc_keys, doc_vals)
        assert out.shape == query.shape

    def test_forward_dtype(self):
        from squish.cross_doc_attn import CrossDocConfig, CrossDocAttention
        cfg = CrossDocConfig(n_heads=2, head_dim=8, max_docs=4)
        attn = CrossDocAttention(cfg)
        query = RNG.random((2, 4, 8)).astype(np.float32)
        doc_keys = [RNG.random((2, 6, 8)).astype(np.float32)]
        doc_vals = [RNG.random((2, 6, 8)).astype(np.float32)]
        out = attn.forward(query, doc_keys, doc_vals)
        assert out.dtype == np.float32

    def test_stats(self):
        from squish.cross_doc_attn import CrossDocConfig, CrossDocAttention
        cfg = CrossDocConfig(n_heads=2, head_dim=8, max_docs=4)
        attn = CrossDocAttention(cfg)
        query = RNG.random((2, 4, 8)).astype(np.float32)
        doc_keys = [RNG.random((2, 6, 8)).astype(np.float32) for _ in range(2)]
        doc_vals = [RNG.random((2, 6, 8)).astype(np.float32) for _ in range(2)]
        attn.forward(query, doc_keys, doc_vals)
        stats = attn.stats
        assert stats.total_forward_calls == 1
        assert stats.total_docs_attended == 2


# ---------------------------------------------------------------------------
# 13. long_context_chunk
# ---------------------------------------------------------------------------

class TestLongContextChunk:
    def test_imports(self):
        from squish.long_context_chunk import ChunkConfig, LongContextChunker, ChunkStats
        assert ChunkConfig is not None
        assert LongContextChunker is not None
        assert ChunkStats is not None

    def test_chunk_covers_sequence(self):
        from squish.long_context_chunk import ChunkConfig, LongContextChunker
        cfg = ChunkConfig(max_chunk_size=64, min_chunk_size=8, embed_dim=16)
        chunker = LongContextChunker(cfg)
        embeddings = RNG.random((128, 16)).astype(np.float32)
        chunks = chunker.chunk(embeddings)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 128

    def test_chunk_respects_max_size(self):
        from squish.long_context_chunk import ChunkConfig, LongContextChunker
        cfg = ChunkConfig(max_chunk_size=32, min_chunk_size=4, embed_dim=16)
        chunker = LongContextChunker(cfg)
        embeddings = RNG.random((64, 16)).astype(np.float32)
        chunks = chunker.chunk(embeddings)
        for start, end in chunks:
            assert end - start <= 32

    def test_stats(self):
        from squish.long_context_chunk import ChunkConfig, LongContextChunker
        cfg = ChunkConfig(max_chunk_size=64, min_chunk_size=8, embed_dim=16)
        chunker = LongContextChunker(cfg)
        embeddings = RNG.random((128, 16)).astype(np.float32)
        chunker.chunk(embeddings)
        stats = chunker.stats
        assert stats.total_chunk_calls == 1
        assert stats.total_tokens_chunked == 128

