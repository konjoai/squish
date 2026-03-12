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
# 1. vision_kv_fuse
# ---------------------------------------------------------------------------

class TestVisionKVFuse:
    def test_imports(self):
        from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache, VisionKVFuseStats
        assert ModalityConfig is not None
        assert VisionKVFuseCache is not None
        assert VisionKVFuseStats is not None

    def test_append_and_get(self):
        from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache
        cfg = ModalityConfig(text_capacity=32, vision_capacity=16, n_heads=2, head_dim=8)
        cache = VisionKVFuseCache(cfg)
        k = RNG.random((2, 8)).astype(np.float32)
        v = RNG.random((2, 8)).astype(np.float32)
        cache.append("text", k, v)
        gk, gv = cache.get_kv("text")
        assert gk.ndim >= 1
        assert cache.text_len == 1

    def test_reset(self):
        from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache
        cfg = ModalityConfig(text_capacity=32, vision_capacity=16, n_heads=2, head_dim=8)
        cache = VisionKVFuseCache(cfg)
        k = RNG.random((2, 8)).astype(np.float32)
        v = RNG.random((2, 8)).astype(np.float32)
        cache.append("vision", k, v)
        assert cache.vision_len == 1
        cache.reset("vision")
        assert cache.vision_len == 0

    def test_stats(self):
        from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache
        cfg = ModalityConfig(text_capacity=32, vision_capacity=16, n_heads=2, head_dim=8)
        cache = VisionKVFuseCache(cfg)
        k = RNG.random((2, 8)).astype(np.float32)
        v = RNG.random((2, 8)).astype(np.float32)
        for _ in range(3):
            cache.append("text", k, v)
        stats = cache.stats
        assert stats.text_appends == 3
        assert stats.vision_appends == 0


# ---------------------------------------------------------------------------
# 2. image_token_prune
# ---------------------------------------------------------------------------

class TestImageTokenPrune:
    def test_imports(self):
        from squish.image_token_prune import PruneConfig, ImageTokenPruner, PruneStats
        assert PruneConfig is not None
        assert ImageTokenPruner is not None
        assert PruneStats is not None

    def test_prune_shape(self):
        from squish.image_token_prune import PruneConfig, ImageTokenPruner
        cfg = PruneConfig(n_tokens=16, prune_ratio=0.5, n_heads=2)
        pruner = ImageTokenPruner(cfg)
        # attn_weights shape: (n_heads, n_tokens, n_tokens)
        weights = np.abs(RNG.random((2, 16, 16))).astype(np.float32)
        weights /= weights.sum(axis=2, keepdims=True) + 1e-9
        kept_idx, pruned_idx = pruner.prune(weights)
        # kept + pruned must partition all 16 tokens
        assert len(kept_idx) + len(pruned_idx) == 16

    def test_prune_reduces_tokens(self):
        from squish.image_token_prune import PruneConfig, ImageTokenPruner
        cfg = PruneConfig(n_tokens=20, prune_ratio=0.5, n_heads=2)
        pruner = ImageTokenPruner(cfg)
        weights = np.abs(RNG.random((2, 20, 20))).astype(np.float32)
        weights /= weights.sum(axis=2, keepdims=True) + 1e-9
        kept_idx, pruned_idx = pruner.prune(weights)
        assert len(kept_idx) < 20

    def test_stats(self):
        from squish.image_token_prune import PruneConfig, ImageTokenPruner
        cfg = PruneConfig(n_tokens=16, prune_ratio=0.5, n_heads=2)
        pruner = ImageTokenPruner(cfg)
        weights = np.abs(RNG.random((2, 16, 16))).astype(np.float32)
        weights /= weights.sum(axis=2, keepdims=True) + 1e-9
        pruner.prune(weights)
        stats = pruner.stats
        assert stats.total_prune_calls == 1
        assert stats.total_tokens_pruned > 0


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
# 5. multimodal_batch
# ---------------------------------------------------------------------------

class TestMultiModalBatch:
    def test_imports(self):
        from squish.multimodal_batch import BatchConfig, BatchSlot, MultiModalBatcher, BatchStats
        assert BatchConfig is not None
        assert BatchSlot is not None
        assert MultiModalBatcher is not None
        assert BatchStats is not None

    def test_add_and_next_batch(self):
        from squish.multimodal_batch import BatchConfig, MultiModalBatcher
        cfg = BatchConfig(max_batch_size=4, max_text_len=512, max_vision_tokens=64)
        batcher = MultiModalBatcher(cfg)
        batcher.add_request(req_id=1, modality="text", text_len=64)
        batcher.add_request(req_id=2, modality="vision", text_len=32, vision_tokens=16)
        batch = batcher.next_batch()
        assert 1 <= len(batch) <= 4

    def test_pending_counts(self):
        from squish.multimodal_batch import BatchConfig, MultiModalBatcher
        cfg = BatchConfig(max_batch_size=4, max_text_len=512, max_vision_tokens=64)
        batcher = MultiModalBatcher(cfg)
        batcher.add_request(req_id=1, modality="text", text_len=64)
        batcher.add_request(req_id=2, modality="vision", text_len=32, vision_tokens=8)
        assert batcher.pending_text >= 1
        assert batcher.pending_vision >= 1

    def test_stats(self):
        from squish.multimodal_batch import BatchConfig, MultiModalBatcher
        cfg = BatchConfig(max_batch_size=4, max_text_len=512, max_vision_tokens=64)
        batcher = MultiModalBatcher(cfg)
        batcher.add_request(req_id=1, modality="text", text_len=64)
        batcher.next_batch()
        stats = batcher.stats
        assert stats.total_batches >= 1


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
# 7. cross_modal_attn
# ---------------------------------------------------------------------------

class TestCrossModalAttn:
    def test_imports(self):
        from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention, CrossModalStats
        assert CrossModalConfig is not None
        assert CrossModalAttention is not None
        assert CrossModalStats is not None

    def test_forward_shape(self):
        from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention
        cfg = CrossModalConfig(n_text_heads=2, n_vision_heads=2, head_dim=8)
        attn = CrossModalAttention(cfg)
        # text_q: (n_text_heads, seq_text, head_dim)
        text_q = RNG.random((2, 4, 8)).astype(np.float32)
        vis_k = RNG.random((2, 8, 8)).astype(np.float32)
        vis_v = RNG.random((2, 8, 8)).astype(np.float32)
        out = attn.forward(text_q, vis_k, vis_v)
        assert out.shape == text_q.shape

    def test_forward_dtype(self):
        from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention
        cfg = CrossModalConfig(n_text_heads=2, n_vision_heads=2, head_dim=8)
        attn = CrossModalAttention(cfg)
        text_q = RNG.random((2, 4, 8)).astype(np.float32)
        vis_k = RNG.random((2, 8, 8)).astype(np.float32)
        vis_v = RNG.random((2, 8, 8)).astype(np.float32)
        out = attn.forward(text_q, vis_k, vis_v)
        assert out.dtype == np.float32

    def test_stats(self):
        from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention
        cfg = CrossModalConfig(n_text_heads=2, n_vision_heads=2, head_dim=8)
        attn = CrossModalAttention(cfg)
        text_q = RNG.random((2, 4, 8)).astype(np.float32)
        vis_k = RNG.random((2, 8, 8)).astype(np.float32)
        vis_v = RNG.random((2, 8, 8)).astype(np.float32)
        attn.forward(text_q, vis_k, vis_v)
        stats = attn.stats
        assert stats.total_forward_calls == 1
        assert stats.total_text_tokens == 4


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
# 11. video_frame_prune
# ---------------------------------------------------------------------------

class TestVideoFramePrune:
    def test_imports(self):
        from squish.video_frame_prune import FrameConfig, VideoFramePruner, FrameStats
        assert FrameConfig is not None
        assert VideoFramePruner is not None
        assert FrameStats is not None

    def test_prune_temporal(self):
        from squish.video_frame_prune import FrameConfig, VideoFramePruner
        cfg = FrameConfig(n_frames=8, tokens_per_frame=16, similarity_threshold=0.92, embed_dim=16)
        pruner = VideoFramePruner(cfg)
        frames = RNG.random((8, 16)).astype(np.float32)
        # prune_temporal returns 1D int64 array of kept frame indices
        kept_indices = pruner.prune_temporal(frames)
        assert kept_indices.ndim == 1
        assert len(kept_indices) <= 8

    def test_prune_spatial(self):
        from squish.video_frame_prune import FrameConfig, VideoFramePruner
        cfg = FrameConfig(n_frames=8, tokens_per_frame=16, spatial_prune_ratio=0.3, embed_dim=16)
        pruner = VideoFramePruner(cfg)
        patches = RNG.random((16, 16)).astype(np.float32)
        kept = pruner.prune_spatial(patches)
        assert kept.shape[0] <= 16

    def test_stats(self):
        from squish.video_frame_prune import FrameConfig, VideoFramePruner
        cfg = FrameConfig(n_frames=8, tokens_per_frame=16, embed_dim=16)
        pruner = VideoFramePruner(cfg)
        frames = RNG.random((8, 16)).astype(np.float32)
        pruner.prune_temporal(frames)
        stats = pruner.stats
        assert stats.total_temporal_prune_calls == 1
        assert stats.total_frames_in == 8


# ---------------------------------------------------------------------------
# 12. embedding_gate
# ---------------------------------------------------------------------------

class TestEmbeddingGate:
    def test_imports(self):
        from squish.embedding_gate import GateConfig, EmbeddingGate, GateStats
        assert GateConfig is not None
        assert EmbeddingGate is not None
        assert GateStats is not None

    def test_gate_returns_tuple(self):
        from squish.embedding_gate import GateConfig, EmbeddingGate
        cfg = GateConfig(embed_dim=16, threshold=0.5, n_routes=2)
        gate = EmbeddingGate(cfg)
        embeddings = RNG.random((8, 16)).astype(np.float32)
        routes, masked = gate.gate(embeddings)
        assert routes.shape[0] == 8
        assert masked.shape == embeddings.shape

    def test_routes_are_binary(self):
        from squish.embedding_gate import GateConfig, EmbeddingGate
        cfg = GateConfig(embed_dim=16, threshold=0.5, n_routes=2)
        gate = EmbeddingGate(cfg)
        embeddings = RNG.random((8, 16)).astype(np.float32)
        routes, _ = gate.gate(embeddings)
        unique = set(int(x) for x in routes.flat)
        assert unique.issubset({0, 1})

    def test_stats(self):
        from squish.embedding_gate import GateConfig, EmbeddingGate
        cfg = GateConfig(embed_dim=16, threshold=0.5, n_routes=2)
        gate = EmbeddingGate(cfg)
        embeddings = RNG.random((8, 16)).astype(np.float32)
        gate.gate(embeddings)
        stats = gate.stats
        assert stats.total_gate_calls == 1


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


# ---------------------------------------------------------------------------
# 14. modality_router
# ---------------------------------------------------------------------------

class TestModalityRouter:
    def test_imports(self):
        from squish.modality_router import ModalityPolicy, ModalityRouter, RouterStats
        assert ModalityPolicy is not None
        assert ModalityRouter is not None
        assert RouterStats is not None

    def test_route_returns_bool(self):
        from squish.modality_router import ModalityPolicy, ModalityRouter
        policies = [
            ModalityPolicy(modality="text", max_concurrent=4),
            ModalityPolicy(modality="vision", max_concurrent=2),
        ]
        router = ModalityRouter(policies)
        result = router.route(req_id=1, modality="text")
        assert isinstance(result, bool)

    def test_route_respects_capacity(self):
        from squish.modality_router import ModalityPolicy, ModalityRouter
        policies = [ModalityPolicy(modality="text", max_concurrent=2)]
        router = ModalityRouter(policies)
        router.route(req_id=1, modality="text")
        router.route(req_id=2, modality="text")
        # Third route at capacity should return False
        result = router.route(req_id=3, modality="text")
        assert result is False

    def test_stats(self):
        from squish.modality_router import ModalityPolicy, ModalityRouter
        policies = [ModalityPolicy(modality="text", max_concurrent=4)]
        router = ModalityRouter(policies)
        router.route(req_id=1, modality="text")
        stats = router.stats
        assert stats.total_routes >= 1
