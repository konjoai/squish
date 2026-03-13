#!/usr/bin/env python3
"""
bench_wave23_24.py — Micro-benchmark suite for Squish Wave 23+24 modules.

Measures in-process CPU/numpy performance of all 28 Wave 23 and Wave 24 modules
and produces a structured JSON results file + human-readable summary table.

Wave 23 modules benchmarked (Multi-Modal & Long Context Intelligence)
────────────────────────────────────────────────────────────────────────────────
  VisionKVFuseCache   Dual-modal KV cache fusion                  (append+get)
  ImageTokenPruner    Attention-saliency image token pruning       (prune lat)
  RAGPrefetcher       Predictive RAG doc prefetch advisor          (record+cands)
  CoTCompressor       Chain-of-thought token distillation          (compress lat)
  MultiModalBatcher   Vision+text joint batch scheduler            (add+next)
  ContextualReranker  Recency-weighted KV context reranking        (rerank lat)
  CrossModalAttention Text↔vision cross-attention                  (forward lat)
  HierarchicalKVStore Three-tier hot/warm/cold KV routing          (put+get lat)
  StreamRAGInjector   Streaming RAG document injection             (inject+retrieve)
  CrossDocAttention   Multi-document cross-attention fusion        (forward lat)
  VideoFramePruner    Temporal + spatial video frame pruning       (prune lat)
  EmbeddingGate       Learned modality routing gate                (gate lat)
  LongContextChunker  Semantic-boundary long context chunking      (chunk lat)
  ModalityRouter      SLO-aware modality request router            (route lat)

Wave 24 modules benchmarked (Quantisation Evolution & Model Surgery)
────────────────────────────────────────────────────────────────────────────────
  TernaryQuantizer    ±1/0 ternary weight quantisation             (quantize lat)
  BinaryAttention     1-bit binarised attention kernel             (forward lat)
  StructuredPruner    N:M structured sparsity pruner               (prune lat)
  LayerFuser          Cosine-similarity layer fusion               (fuse lat)
  WeightSharer        Cross-layer delta-residual weight sharing    (get_weight lat)
  QuantCalibrator     Unified MinMax/Percentile/MSE calibration    (calibrate lat)
  SparseWeightStore   CSR sparse weight compression store          (compress+decomp)
  DeltaCompressor     SVD delta-weight compression                 (compress lat)
  ModelSurgeon        In-place model layer/head surgery            (plan+estimate)
  ZeroQuantV2         ZeroQuant-V2 groupwise + outlier quant       (quantize lat)
  GPTQCalibrator      GPTQ Hessian-weighted column quantisation    (calibrate lat)
  SparseMoERouter     Sparse MoE top-k expert routing              (route lat)
  AWQv2Calibrator     AWQ v2 activation-aware scale search         (calibrate lat)
  IterativePruner     Cubic iterative magnitude pruning schedule   (prune_step lat)

Usage
─────
    python3 dev/benchmarks/bench_wave23_24.py
    python3 dev/benchmarks/bench_wave23_24.py --output dev/results/wave23_24_bench.json
    python3 dev/benchmarks/bench_wave23_24.py --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Colour helpers
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

RNG = np.random.default_rng(42)


def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 64}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 64}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<44} {G}{val:>14}{NC}  {D}{extra}{NC}")


def _skip(label: str, reason: str = "") -> None:
    print(f"  {Y}~ SKIP{NC}  {label:<44} {D}{reason}{NC}")


def _timeit(fn, n: int = 200, warmup: int = 10):
    """Return (mean_us, min_us, max_us) over *n* calls after *warmup* discards."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(times)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# ─────────────────────────────────────────────────────────────────────────────
# Wave 23 benchmarks — Multi-Modal & Long Context Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def bench_vision_kv_fuse(results: dict) -> None:
    _hdr("VisionKVFuseCache — Dual-Modal KV Cache Fusion")
    try:
        from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache

        n_heads, head_dim = 8, 64
        cfg   = ModalityConfig(text_capacity=512, vision_capacity=256,
                               n_heads=n_heads, head_dim=head_dim)
        cache = VisionKVFuseCache(cfg)

        key = RNG.random((n_heads, head_dim)).astype(np.float32)
        val = RNG.random((n_heads, head_dim)).astype(np.float32)

        mean_a, lo_a, hi_a = _timeit(
            lambda: cache.append("text", key, val), n=500
        )
        _row(f"append() text h={n_heads} d={head_dim} (single token)",
             f"{mean_a:.2f} µs", f"min={lo_a:.2f} max={hi_a:.2f} µs")

        # Reset so get_kv doesn't exceed capacity
        cache.reset()
        cache.append("text", key, val)
        mean_g, lo_g, hi_g = _timeit(
            lambda: cache.get_kv("text"), n=2000
        )
        _row(f"get_kv() text modality len=1",
             f"{mean_g:.2f} µs", f"min={lo_g:.2f} max={hi_g:.2f} µs")

        results["vision_kv_fuse"] = dict(
            append_mean_us=mean_a,
            get_kv_mean_us=mean_g,
        )
    except Exception as e:
        _skip("VisionKVFuseCache", str(e))


def bench_image_token_prune(results: dict) -> None:
    _hdr("ImageTokenPruner — Attention-Saliency Image Token Pruning")
    try:
        from squish.image_token_prune import PruneConfig, ImageTokenPruner

        n_heads, n_tokens = 8, 196
        cfg    = PruneConfig(n_tokens=n_tokens, prune_ratio=0.5, n_heads=n_heads)
        pruner = ImageTokenPruner(cfg)

        # Full (n_heads, n_tokens, n_tokens) softmax attention weight matrix
        raw = np.abs(RNG.random((n_heads, n_tokens, n_tokens))).astype(np.float32)
        attn = raw / raw.sum(axis=2, keepdims=True)

        mean_p, lo_p, hi_p = _timeit(
            lambda: pruner.prune(attn), n=200
        )
        _row(f"prune() h={n_heads} n_tokens={n_tokens} ratio=0.5",
             f"{mean_p:.1f} µs", f"min={lo_p:.1f} max={hi_p:.1f} µs")

        results["image_token_prune"] = dict(
            prune_mean_us=mean_p,
            prune_min_us=lo_p,
            prune_max_us=hi_p,
        )
    except Exception as e:
        _skip("ImageTokenPruner", str(e))


def bench_rag_prefetch(results: dict) -> None:
    _hdr("RAGPrefetcher — Predictive RAG Document Prefetch Advisor")
    try:
        from squish.rag_prefetch import RAGConfig, RAGPrefetcher

        cfg       = RAGConfig(max_docs=1024, top_k=16, recency_decay=0.95,
                              min_accesses=2)
        prefetcher = RAGPrefetcher(cfg)

        # Pre-populate with docs accessed multiple times
        for i in range(50):
            tokens = RNG.integers(0, 1000, size=(32,)).astype(np.int32)
            for _ in range(3):
                prefetcher.record_access(tokens)

        new_tokens = RNG.integers(0, 1000, size=(32,)).astype(np.int32)

        mean_r, lo_r, hi_r = _timeit(
            lambda: prefetcher.record_access(new_tokens), n=2000
        )
        _row("record_access() 32-token doc fingerprint",
             f"{mean_r:.2f} µs", f"min={lo_r:.2f} max={hi_r:.2f} µs")

        mean_c, lo_c, hi_c = _timeit(
            lambda: prefetcher.get_warmup_candidates(), n=2000
        )
        _row(f"get_warmup_candidates() top_k=16 tracked={prefetcher.n_tracked}",
             f"{mean_c:.2f} µs", f"min={lo_c:.2f} max={hi_c:.2f} µs")

        results["rag_prefetch"] = dict(
            record_access_mean_us=mean_r,
            get_candidates_mean_us=mean_c,
        )
    except Exception as e:
        _skip("RAGPrefetcher", str(e))


def bench_cot_compress(results: dict) -> None:
    _hdr("CoTCompressor — Chain-of-Thought Token Distillation")
    try:
        from squish.cot_compress import CoTConfig, CoTCompressor

        cfg        = CoTConfig(compress_ratio=0.4, min_tokens=16)
        compressor = CoTCompressor(cfg)

        tokens_256 = RNG.integers(0, 32000, size=(256,)).astype(np.int32)
        tokens_64  = RNG.integers(0, 32000, size=(64,)).astype(np.int32)

        mean_256, lo_256, hi_256 = _timeit(
            lambda: compressor.compress(tokens_256), n=500
        )
        _row("compress() 256-token CoT chain ratio=0.4",
             f"{mean_256:.1f} µs", f"min={lo_256:.1f} max={hi_256:.1f} µs")

        mean_64, lo_64, hi_64 = _timeit(
            lambda: compressor.compress(tokens_64), n=1000
        )
        _row("compress() 64-token CoT chain ratio=0.4",
             f"{mean_64:.1f} µs", f"min={lo_64:.1f} max={hi_64:.1f} µs")

        results["cot_compress"] = dict(
            compress_256_mean_us=mean_256,
            compress_64_mean_us=mean_64,
        )
    except Exception as e:
        _skip("CoTCompressor", str(e))


def bench_multimodal_batch(results: dict) -> None:
    _hdr("MultiModalBatcher — Vision+Text Joint Batch Scheduler")
    try:
        from squish.multimodal_batch import BatchConfig, MultiModalBatcher

        cfg     = BatchConfig(max_batch_size=8, max_text_len=2048,
                              max_vision_tokens=256)
        batcher = MultiModalBatcher(cfg)

        counter = [0]

        def _add():
            counter[0] += 1
            batcher.add_request(req_id=counter[0], modality="text",
                                 text_len=128)

        # Pre-fill a few requests
        for i in range(4):
            batcher.add_request(req_id=i, modality="text", text_len=64)

        mean_a, lo_a, hi_a = _timeit(_add, n=1000)
        _row("add_request() text len=128",
             f"{mean_a:.2f} µs", f"min={lo_a:.2f} max={hi_a:.2f} µs")

        mean_n, lo_n, hi_n = _timeit(
            lambda: batcher.next_batch(), n=1000
        )
        _row("next_batch() max_batch=8 mixed queue",
             f"{mean_n:.2f} µs", f"min={lo_n:.2f} max={hi_n:.2f} µs")

        results["multimodal_batch"] = dict(
            add_request_mean_us=mean_a,
            next_batch_mean_us=mean_n,
        )
    except Exception as e:
        _skip("MultiModalBatcher", str(e))


def bench_contextual_rerank(results: dict) -> None:
    _hdr("ContextualReranker — Recency-Weighted KV Context Reranking")
    try:
        from squish.contextual_rerank import RerankConfig, ContextualReranker

        n_heads, head_dim, seq_len = 8, 64, 256
        cfg      = RerankConfig(n_heads=n_heads, head_dim=head_dim,
                                recency_weight=0.5, top_k=64)
        reranker = ContextualReranker(cfg)

        # keys: (n_heads, seq_len, head_dim)
        keys  = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)
        query = RNG.random((n_heads, head_dim)).astype(np.float32)

        mean_r, lo_r, hi_r = _timeit(
            lambda: reranker.rerank(keys), n=200
        )
        _row(f"rerank() h={n_heads} seq={seq_len} d={head_dim} (no query)",
             f"{mean_r:.1f} µs", f"min={lo_r:.1f} max={hi_r:.1f} µs")

        mean_q, lo_q, hi_q = _timeit(
            lambda: reranker.rerank(keys, query=query), n=200
        )
        _row(f"rerank() with query h={n_heads} seq={seq_len} d={head_dim}",
             f"{mean_q:.1f} µs", f"min={lo_q:.1f} max={hi_q:.1f} µs")

        results["contextual_rerank"] = dict(
            rerank_mean_us=mean_r,
            rerank_with_query_mean_us=mean_q,
        )
    except Exception as e:
        _skip("ContextualReranker", str(e))


def bench_cross_modal_attn(results: dict) -> None:
    _hdr("CrossModalAttention — Text↔Vision Cross-Modal Attention")
    try:
        from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention

        n_heads, seq_text, seq_vis, head_dim = 8, 32, 64, 64
        cfg  = CrossModalConfig(n_text_heads=n_heads, n_vision_heads=n_heads,
                                head_dim=head_dim)
        attn = CrossModalAttention(cfg)

        # All: (n_heads, seq, head_dim)
        text_q = RNG.random((n_heads, seq_text, head_dim)).astype(np.float32)
        vis_k  = RNG.random((n_heads, seq_vis, head_dim)).astype(np.float32)
        vis_v  = RNG.random((n_heads, seq_vis, head_dim)).astype(np.float32)

        mean_f, lo_f, hi_f = _timeit(
            lambda: attn.forward(text_q, vis_k, vis_v), n=200
        )
        _row(f"forward() h={n_heads} text={seq_text} vis={seq_vis} d={head_dim}",
             f"{mean_f:.1f} µs", f"min={lo_f:.1f} max={hi_f:.1f} µs")

        results["cross_modal_attn"] = dict(
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
            forward_max_us=hi_f,
        )
    except Exception as e:
        _skip("CrossModalAttention", str(e))


def bench_hierarchical_kv(results: dict) -> None:
    _hdr("HierarchicalKVStore — Three-Tier Hot/Warm/Cold KV Tiering")
    try:
        from squish.hierarchical_kv import TierConfig, HierarchicalKVStore

        n_heads, head_dim = 8, 64
        cfg   = TierConfig(hot_capacity=64, warm_capacity=256, cold_capacity=1024,
                           n_heads=n_heads, head_dim=head_dim)
        store = HierarchicalKVStore(cfg)

        key = RNG.random((n_heads, head_dim)).astype(np.float32)
        val = RNG.random((n_heads, head_dim)).astype(np.float32)

        # Pre-fill hot tier
        for i in range(32):
            k_ = RNG.random((n_heads, head_dim)).astype(np.float32)
            v_ = RNG.random((n_heads, head_dim)).astype(np.float32)
            store.put(i, k_, v_)

        pos_counter = [100]

        def _put():
            pos_counter[0] += 1
            store.put(pos_counter[0], key, val)

        mean_p, lo_p, hi_p = _timeit(_put, n=200)
        _row(f"put() h={n_heads} d={head_dim} (hot tier)",
             f"{mean_p:.2f} µs", f"min={lo_p:.2f} max={hi_p:.2f} µs")

        mean_g_hit, lo_g, hi_g = _timeit(
            lambda: store.get(1), n=1000
        )
        _row(f"get() hot hit h={n_heads} d={head_dim}",
             f"{mean_g_hit:.2f} µs", f"min={lo_g:.2f} max={hi_g:.2f} µs")

        mean_miss, lo_m, hi_m = _timeit(
            lambda: store.get(999999), n=2000
        )
        _row("get() cold miss",
             f"{mean_miss:.2f} µs", f"min={lo_m:.2f} max={hi_m:.2f} µs")

        stats = store.stats
        results["hierarchical_kv"] = dict(
            put_mean_us=mean_p,
            get_hit_mean_us=mean_g_hit,
            get_miss_mean_us=mean_miss,
            hit_rate=stats.hit_rate,
        )
    except Exception as e:
        _skip("HierarchicalKVStore", str(e))


def bench_stream_rag(results: dict) -> None:
    _hdr("StreamRAGInjector — Streaming RAG Document Injection")
    try:
        from squish.stream_rag import StreamRAGConfig, StreamRAGInjector

        embed_dim = 256
        cfg      = StreamRAGConfig(max_docs=8, max_doc_tokens=512,
                                   embed_dim=embed_dim, top_k_retrieve=3)
        injector = StreamRAGInjector(cfg)

        tokens = RNG.integers(0, 1000, size=(128,)).astype(np.int32)
        emb    = RNG.random(embed_dim).astype(np.float32)
        query  = RNG.random(embed_dim).astype(np.float32)

        # Pre-fill docs
        for i in range(4):
            e_ = RNG.random(embed_dim).astype(np.float32)
            injector.inject(f"doc{i}", tokens, e_)

        counter = [10]

        def _inject():
            counter[0] += 1
            injector.inject(f"doc{counter[0]}", tokens, emb)

        mean_i, lo_i, hi_i = _timeit(_inject, n=500)
        _row(f"inject() doc embed_dim={embed_dim} tokens=128",
             f"{mean_i:.2f} µs", f"min={lo_i:.2f} max={hi_i:.2f} µs")

        mean_r, lo_r, hi_r = _timeit(
            lambda: injector.retrieve(query, top_k=3), n=1000
        )
        _row(f"retrieve() top_k=3 embed_dim={embed_dim}",
             f"{mean_r:.2f} µs", f"min={lo_r:.2f} max={hi_r:.2f} µs")

        results["stream_rag"] = dict(
            inject_mean_us=mean_i,
            retrieve_mean_us=mean_r,
        )
    except Exception as e:
        _skip("StreamRAGInjector", str(e))


def bench_cross_doc_attn(results: dict) -> None:
    _hdr("CrossDocAttention — Multi-Document Cross-Attention Fusion")
    try:
        from squish.cross_doc_attn import CrossDocConfig, CrossDocAttention

        n_heads, seq_q, seq_k, head_dim = 8, 16, 32, 64
        cfg  = CrossDocConfig(n_heads=n_heads, head_dim=head_dim, max_docs=4)
        attn = CrossDocAttention(cfg)

        # All: (n_heads, seq, head_dim)
        query    = RNG.random((n_heads, seq_q, head_dim)).astype(np.float32)
        doc_keys = [RNG.random((n_heads, seq_k, head_dim)).astype(np.float32)
                    for _ in range(4)]
        doc_vals = [RNG.random((n_heads, seq_k, head_dim)).astype(np.float32)
                    for _ in range(4)]

        mean_f, lo_f, hi_f = _timeit(
            lambda: attn.forward(query, doc_keys, doc_vals), n=200
        )
        _row(f"forward() h={n_heads} q={seq_q} k={seq_k} n_docs=4 d={head_dim}",
             f"{mean_f:.1f} µs", f"min={lo_f:.1f} max={hi_f:.1f} µs")

        results["cross_doc_attn"] = dict(
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
            forward_max_us=hi_f,
        )
    except Exception as e:
        _skip("CrossDocAttention", str(e))


def bench_video_frame_prune(results: dict) -> None:
    _hdr("VideoFramePruner — Temporal + Spatial Video Frame Pruning")
    try:
        from squish.video_frame_prune import FrameConfig, VideoFramePruner

        n_frames, n_patches, embed_dim = 32, 196, 256
        cfg    = FrameConfig(n_frames=n_frames, tokens_per_frame=n_patches,
                             similarity_threshold=0.92, spatial_prune_ratio=0.3,
                             embed_dim=embed_dim)
        pruner = VideoFramePruner(cfg)

        frames  = RNG.random((n_frames, embed_dim)).astype(np.float32)
        patches = RNG.random((n_patches, embed_dim)).astype(np.float32)

        mean_t, lo_t, hi_t = _timeit(
            lambda: pruner.prune_temporal(frames), n=500
        )
        _row(f"prune_temporal() n_frames={n_frames} embed_dim={embed_dim}",
             f"{mean_t:.1f} µs", f"min={lo_t:.1f} max={hi_t:.1f} µs")

        mean_s, lo_s, hi_s = _timeit(
            lambda: pruner.prune_spatial(patches), n=500
        )
        _row(f"prune_spatial() n_patches={n_patches} ratio=0.3",
             f"{mean_s:.1f} µs", f"min={lo_s:.1f} max={hi_s:.1f} µs")

        results["video_frame_prune"] = dict(
            prune_temporal_mean_us=mean_t,
            prune_spatial_mean_us=mean_s,
        )
    except Exception as e:
        _skip("VideoFramePruner", str(e))


def bench_embedding_gate(results: dict) -> None:
    _hdr("EmbeddingGate — Learned Modality Routing Gate")
    try:
        from squish.embedding_gate import GateConfig, EmbeddingGate

        embed_dim, n_tokens = 512, 32
        cfg  = GateConfig(embed_dim=embed_dim, threshold=0.5, n_routes=2)
        gate = EmbeddingGate(cfg)

        emb = RNG.random((n_tokens, embed_dim)).astype(np.float32)

        mean_g, lo_g, hi_g = _timeit(
            lambda: gate.gate(emb), n=500
        )
        _row(f"gate() n_tokens={n_tokens} embed_dim={embed_dim}",
             f"{mean_g:.2f} µs", f"min={lo_g:.2f} max={hi_g:.2f} µs")

        results["embedding_gate"] = dict(
            gate_mean_us=mean_g,
            gate_min_us=lo_g,
            gate_max_us=hi_g,
        )
    except Exception as e:
        _skip("EmbeddingGate", str(e))


def bench_long_context_chunk(results: dict) -> None:
    _hdr("LongContextChunker — Semantic-Boundary Long Context Chunking")
    try:
        from squish.long_context_chunk import ChunkConfig, LongContextChunker

        seq_len, embed_dim = 2048, 128
        cfg     = ChunkConfig(max_chunk_size=512, min_chunk_size=64,
                              boundary_sensitivity=2.0, embed_dim=embed_dim)
        chunker = LongContextChunker(cfg)

        embeddings = RNG.random((seq_len, embed_dim)).astype(np.float32)
        short_emb  = RNG.random((256, embed_dim)).astype(np.float32)

        mean_l, lo_l, hi_l = _timeit(
            lambda: chunker.chunk(embeddings), n=200
        )
        _row(f"chunk() seq={seq_len} embed_dim={embed_dim} max_chunk=512",
             f"{mean_l:.1f} µs", f"min={lo_l:.1f} max={hi_l:.1f} µs")

        mean_s, lo_s, hi_s = _timeit(
            lambda: chunker.chunk(short_emb), n=500
        )
        _row(f"chunk() seq=256 embed_dim={embed_dim}",
             f"{mean_s:.1f} µs", f"min={lo_s:.1f} max={hi_s:.1f} µs")

        results["long_context_chunk"] = dict(
            chunk_2048_mean_us=mean_l,
            chunk_256_mean_us=mean_s,
        )
    except Exception as e:
        _skip("LongContextChunker", str(e))


def bench_modality_router(results: dict) -> None:
    _hdr("ModalityRouter — SLO-Aware Modality Request Router")
    try:
        from squish.modality_router import ModalityPolicy, ModalityRouter

        policies = [
            ModalityPolicy(modality="text",   max_concurrent=32, priority=1),
            ModalityPolicy(modality="vision", max_concurrent=16, priority=2),
            ModalityPolicy(modality="audio",  max_concurrent=8,  priority=3),
        ]
        router = ModalityRouter(policies)

        counter = [0]

        def _route_complete():
            counter[0] += 1
            req_id = counter[0]
            routed = router.route(req_id=req_id, modality="text")
            if routed:
                router.complete(req_id=req_id, modality="text", latency_ms=50.0)

        mean_r, lo_r, hi_r = _timeit(_route_complete, n=2000)
        _row("route()+complete() text request",
             f"{mean_r:.2f} µs", f"min={lo_r:.2f} max={hi_r:.2f} µs")

        results["modality_router"] = dict(
            route_complete_mean_us=mean_r,
            route_complete_min_us=lo_r,
            route_complete_max_us=hi_r,
        )
    except Exception as e:
        _skip("ModalityRouter", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Wave 24 benchmarks — Quantisation Evolution & Model Surgery
# ─────────────────────────────────────────────────────────────────────────────

def bench_ternary_quant(results: dict) -> None:
    _hdr("TernaryQuantizer — ±1/0 Ternary Weight Quantisation")
    try:
        from squish.ternary_quant import TernaryConfig, TernaryQuantizer

        rows, cols = 256, 256
        cfg       = TernaryConfig(zero_threshold=0.05)
        quantizer = TernaryQuantizer(cfg)

        weights = RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_q, lo_q, hi_q = _timeit(
            lambda: quantizer.quantize(weights), n=200
        )
        _row(f"quantize() {rows}×{cols} float32 → int8 + scale",
             f"{mean_q:.1f} µs", f"min={lo_q:.1f} max={hi_q:.1f} µs")

        tern, scale = quantizer.quantize(weights)
        mean_d, lo_d, hi_d = _timeit(
            lambda: quantizer.dequantize(tern, scale), n=500
        )
        _row(f"dequantize() {rows}×{cols} int8 → float32",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        stats = quantizer.stats
        results["ternary_quant"] = dict(
            quantize_mean_us=mean_q,
            dequantize_mean_us=mean_d,
            sparsity=stats.sparsity,
        )
    except Exception as e:
        _skip("TernaryQuantizer", str(e))


def bench_binary_attn(results: dict) -> None:
    _hdr("BinaryAttention — 1-Bit Binarised Attention Kernel")
    try:
        from squish.binary_attn import BinaryConfig, BinaryAttention

        n_heads, seq_len, head_dim = 8, 64, 64
        cfg  = BinaryConfig(n_heads=n_heads, head_dim=head_dim)
        attn = BinaryAttention(cfg)

        q = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)
        k = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)
        v = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)

        mean_f, lo_f, hi_f = _timeit(
            lambda: attn.forward(q, k, v), n=200
        )
        _row(f"forward() h={n_heads} seq={seq_len} d={head_dim} binary-Q/K",
             f"{mean_f:.1f} µs", f"min={lo_f:.1f} max={hi_f:.1f} µs")

        results["binary_attn"] = dict(
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
            forward_max_us=hi_f,
        )
    except Exception as e:
        _skip("BinaryAttention", str(e))


def bench_structured_prune(results: dict) -> None:
    _hdr("StructuredPruner — N:M Structured Sparsity Pruner")
    try:
        from squish.structured_prune import PruneConfig, StructuredPruner

        rows, cols = 256, 256  # cols must be divisible by M=4
        cfg    = PruneConfig(N=2, M=4)
        pruner = StructuredPruner(cfg)

        weights = RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_p, lo_p, hi_p = _timeit(
            lambda: pruner.prune(weights), n=200
        )
        _row(f"prune() 2:4 sparsity {rows}×{cols} float32",
             f"{mean_p:.1f} µs", f"min={lo_p:.1f} max={hi_p:.1f} µs")

        pruned, _ = pruner.prune(weights)
        frac = pruner.sparsity_fraction(pruned)
        _row("sparsity_fraction()", f"{frac:.3f}", "(measured post-prune)")

        results["structured_prune"] = dict(
            prune_mean_us=mean_p,
            prune_min_us=lo_p,
            prune_max_us=hi_p,
            sparsity_fraction=frac,
        )
    except Exception as e:
        _skip("StructuredPruner", str(e))


def bench_layer_fuse(results: dict) -> None:
    _hdr("LayerFuser — Cosine-Similarity Layer Fusion")
    try:
        from squish.layer_fuse import FusionConfig, LayerFuser

        hidden_dim = 512
        cfg   = FusionConfig(hidden_dim=hidden_dim, similarity_threshold=0.97)
        fuser = LayerFuser(cfg)

        a = RNG.standard_normal((hidden_dim, hidden_dim)).astype(np.float32)
        b = RNG.standard_normal((hidden_dim, hidden_dim)).astype(np.float32)

        mean_sim, lo_sim, hi_sim = _timeit(
            lambda: fuser.cosine_similarity(a, b), n=500
        )
        _row(f"cosine_similarity() {hidden_dim}×{hidden_dim}",
             f"{mean_sim:.2f} µs", f"min={lo_sim:.2f} max={hi_sim:.2f} µs")

        mean_f, lo_f, hi_f = _timeit(
            lambda: fuser.fuse(a, b), n=500
        )
        _row(f"fuse() {hidden_dim}×{hidden_dim} (weighted mean)",
             f"{mean_f:.2f} µs", f"min={lo_f:.2f} max={hi_f:.2f} µs")

        results["layer_fuse"] = dict(
            cosine_similarity_mean_us=mean_sim,
            fuse_mean_us=mean_f,
        )
    except Exception as e:
        _skip("LayerFuser", str(e))


def bench_weight_sharing(results: dict) -> None:
    _hdr("WeightSharer — Cross-Layer Delta-Residual Weight Sharing")
    try:
        from squish.weight_sharing import SharingConfig, WeightSharer

        hidden_dim, n_shared, rank = 256, 8, 16
        cfg    = SharingConfig(hidden_dim=hidden_dim, n_shared_layers=n_shared,
                               rank=rank)
        sharer = WeightSharer(cfg)

        mean_w, lo_w, hi_w = _timeit(
            lambda: sharer.get_effective_weight(0), n=500
        )
        _row(f"get_effective_weight() hidden={hidden_dim} rank={rank}",
             f"{mean_w:.2f} µs", f"min={lo_w:.2f} max={hi_w:.2f} µs")

        ratio = sharer.stats.memory_ratio
        _row("memory_ratio (sparse / dense)", f"{ratio:.4f}", "(< 1.0 = savings)")

        results["weight_sharing"] = dict(
            get_effective_weight_mean_us=mean_w,
            memory_ratio=ratio,
        )
    except Exception as e:
        _skip("WeightSharer", str(e))


def bench_quant_calib(results: dict) -> None:
    _hdr("QuantCalibrator — Unified MinMax/Percentile/MSE Calibration")
    try:
        from squish.quant_calib import CalibConfig, QuantCalibrator

        n_samples, n_features = 128, 256

        for method in ("minmax", "percentile", "mse"):
            cfg        = CalibConfig(method=method, n_bits=8, per_channel=True)
            calibrator = QuantCalibrator(cfg)
            acts       = RNG.standard_normal((n_samples, n_features)).astype(np.float32)

            mean_m, lo_m, hi_m = _timeit(
                lambda a=acts, c=calibrator: c.calibrate(a), n=200
            )
            _row(f"calibrate() method={method} {n_samples}×{n_features}",
                 f"{mean_m:.1f} µs", f"min={lo_m:.1f} max={hi_m:.1f} µs")

        cfg_last   = CalibConfig(method="minmax", n_bits=8, per_channel=True)
        calibrator = QuantCalibrator(cfg_last)
        acts       = RNG.standard_normal((n_samples, n_features)).astype(np.float32)
        mean_f, lo_f, hi_f = _timeit(
            lambda: calibrator.calibrate(acts), n=200
        )
        results["quant_calib"] = dict(
            calibrate_minmax_mean_us=mean_f,
        )
    except Exception as e:
        _skip("QuantCalibrator", str(e))


def bench_sparse_weight(results: dict) -> None:
    _hdr("SparseWeightStore — CSR Sparse Weight Compression Store")
    try:
        from squish.sparse_weight import SparsityConfig, SparseWeightStore

        rows, cols = 256, 256
        cfg   = SparsityConfig(N=2, M=4)
        store = SparseWeightStore(cfg)

        dense = RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_c, lo_c, hi_c = _timeit(
            lambda: store.compress(dense), n=200
        )
        _row(f"compress() 2:4 dense {rows}×{cols} → CSR",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        store.compress(dense)
        mean_d, lo_d, hi_d = _timeit(
            lambda: store.decompress(), n=500
        )
        _row(f"decompress() CSR → dense {rows}×{cols}",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        ratio = store.compression_ratio
        _row("compression_ratio (dense_bytes / sparse_bytes)",
             f"{ratio:.2f}×", "(>1 = memory saved)")

        results["sparse_weight"] = dict(
            compress_mean_us=mean_c,
            decompress_mean_us=mean_d,
            compression_ratio=ratio,
        )
    except Exception as e:
        _skip("SparseWeightStore", str(e))


def bench_delta_compress(results: dict) -> None:
    _hdr("DeltaCompressor — SVD Delta-Weight Compression")
    try:
        from squish.delta_compress import DeltaConfig, DeltaCompressor

        rows, cols, rank = 256, 256, 16
        cfg        = DeltaConfig(rank=rank)
        compressor = DeltaCompressor(cfg)

        base      = RNG.standard_normal((rows, cols)).astype(np.float32)
        finetuned = base + 0.01 * RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_c, lo_c, hi_c = _timeit(
            lambda: compressor.compress(base, finetuned), n=100
        )
        _row(f"compress() SVD rank={rank} {rows}×{cols}",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        U_k, S_k, Vt_k = compressor.compress(base, finetuned)
        mean_d, lo_d, hi_d = _timeit(
            lambda: compressor.decompress(U_k, S_k, Vt_k), n=500
        )
        _row(f"decompress() rank={rank} → {rows}×{cols}",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        ratio = DeltaCompressor.compression_ratio(rows, cols, rank)
        _row(f"compression_ratio({rows}, {cols}, rank={rank})",
             f"{ratio:.2f}×")

        results["delta_compress"] = dict(
            compress_mean_us=mean_c,
            decompress_mean_us=mean_d,
            compression_ratio=ratio,
        )
    except Exception as e:
        _skip("DeltaCompressor", str(e))


def bench_model_surgery(results: dict) -> None:
    _hdr("ModelSurgeon — In-Place Model Layer/Head Surgery")
    try:
        from squish.model_surgery import ModelSurgeon

        surgeon = ModelSurgeon()

        mean_p, lo_p, hi_p = _timeit(
            lambda: surgeon.plan(n_layers=32, n_heads=32, head_dim=128), n=2000
        )
        _row("plan() 32-layer 32-head model",
             f"{mean_p:.2f} µs", f"min={lo_p:.2f} max={hi_p:.2f} µs")

        plan = surgeon.plan(n_layers=32, n_heads=32, head_dim=128)
        mean_e, lo_e, hi_e = _timeit(
            lambda: ModelSurgeon.estimate_reduction(plan), n=5000
        )
        _row("estimate_reduction() static method",
             f"{mean_e:.2f} µs", f"min={lo_e:.2f} max={hi_e:.2f} µs")

        reduction = ModelSurgeon.estimate_reduction(plan)
        results["model_surgery"] = dict(
            plan_mean_us=mean_p,
            estimate_reduction_mean_us=mean_e,
            estimated_reduction=reduction,
        )
    except Exception as e:
        _skip("ModelSurgeon", str(e))


def bench_zero_quant_v2(results: dict) -> None:
    _hdr("ZeroQuantV2 — Groupwise + Outlier Quantisation")
    try:
        from squish.zero_quant_v2 import ZQConfig, ZeroQuantV2

        rows, cols, group_size = 128, 256, 128
        cfg = ZQConfig(n_bits=8, group_size=group_size, outlier_threshold=0.95)
        zq  = ZeroQuantV2(cfg)

        weights = RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_q, lo_q, hi_q = _timeit(
            lambda: zq.quantize(weights), n=200
        )
        _row(f"quantize() {rows}×{cols} group_size={group_size} 8-bit",
             f"{mean_q:.1f} µs", f"min={lo_q:.1f} max={hi_q:.1f} µs")

        quant, scales, residual = zq.quantize(weights)
        mean_d, lo_d, hi_d = _timeit(
            lambda: zq.dequantize(quant, scales, residual), n=500
        )
        _row(f"dequantize() {rows}×{cols}",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        stats = zq.stats
        results["zero_quant_v2"] = dict(
            quantize_mean_us=mean_q,
            dequantize_mean_us=mean_d,
            outlier_rate=stats.outlier_rate,
        )
    except Exception as e:
        _skip("ZeroQuantV2", str(e))


def bench_gptq_layer(results: dict) -> None:
    _hdr("GPTQCalibrator — Hessian-Weighted Column Quantisation")
    try:
        from squish.gptq_layer import GPTQConfig, GPTQCalibrator

        rows, cols, n_samples = 64, 64, 128
        cfg        = GPTQConfig(n_bits=4, block_size=64, damp_percent=0.01)
        calibrator = GPTQCalibrator(cfg)

        W = RNG.standard_normal((rows, cols)).astype(np.float32)
        # X shape: (n_samples, rows) — activations
        X = RNG.standard_normal((n_samples, rows)).astype(np.float32)

        mean_c, lo_c, hi_c = _timeit(
            lambda: calibrator.calibrate(W, X), n=100
        )
        _row(f"calibrate() GPTQ 4-bit {rows}×{cols} X={n_samples}×{rows}",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        stats = calibrator.stats
        results["gptq_layer"] = dict(
            calibrate_mean_us=mean_c,
            calibrate_min_us=lo_c,
            calibrate_max_us=hi_c,
            total_columns=stats.total_columns,
        )
    except Exception as e:
        _skip("GPTQCalibrator", str(e))


def bench_sparse_moe(results: dict) -> None:
    _hdr("SparseMoERouter — Sparse MoE Top-K Expert Routing")
    try:
        from squish.sparse_moe import MoEConfig, SparseMoERouter

        n_experts, top_k, hidden_dim, n_tokens = 8, 2, 256, 32
        cfg    = MoEConfig(n_experts=n_experts, top_k=top_k,
                           hidden_dim=hidden_dim, load_balance_weight=0.01)
        router = SparseMoERouter(cfg)

        hidden = RNG.random((n_tokens, hidden_dim)).astype(np.float32)

        mean_r, lo_r, hi_r = _timeit(
            lambda: router.route(hidden), n=500
        )
        _row(f"route() n_tokens={n_tokens} n_experts={n_experts} top_k={top_k}",
             f"{mean_r:.2f} µs", f"min={lo_r:.2f} max={hi_r:.2f} µs")

        results["sparse_moe"] = dict(
            route_mean_us=mean_r,
            route_min_us=lo_r,
            route_max_us=hi_r,
        )
    except Exception as e:
        _skip("SparseMoERouter", str(e))


def bench_awq_v2(results: dict) -> None:
    _hdr("AWQv2Calibrator — Activation-Aware Scale & Shift Search")
    try:
        from squish.quant.awq_v2 import AWQv2Config, AWQv2Calibrator

        # W: (out_features, in_features); act_scales: (in_features,) = W.shape[1]
        out_features, in_features = 128, 256
        cfg        = AWQv2Config(n_bits=4, group_size=128, n_search_steps=20)
        calibrator = AWQv2Calibrator(cfg)

        W          = RNG.standard_normal((out_features, in_features)).astype(np.float32)
        act_scales = np.abs(RNG.standard_normal((in_features,))).astype(np.float32)

        mean_c, lo_c, hi_c = _timeit(
            lambda: calibrator.calibrate(W, act_scales), n=100
        )
        _row(f"calibrate() AWQ v2 {out_features}×{in_features} steps=20",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        scales, shifts = calibrator.calibrate(W, act_scales)
        mean_q, lo_q, hi_q = _timeit(
            lambda: calibrator.quantize(W, scales, shifts), n=500
        )
        _row(f"quantize() {out_features}×{in_features} 4-bit",
             f"{mean_q:.1f} µs", f"min={lo_q:.1f} max={hi_q:.1f} µs")

        results["awq_v2"] = dict(
            calibrate_mean_us=mean_c,
            quantize_mean_us=mean_q,
        )
    except Exception as e:
        _skip("AWQv2Calibrator", str(e))


def bench_iter_prune(results: dict) -> None:
    _hdr("IterativePruner — Cubic Iterative Magnitude Pruning Schedule")
    try:
        from squish.iter_prune import PruneSchedule, IterativePruner

        rows, cols = 256, 256
        schedule = PruneSchedule(initial_sparsity=0.0, target_sparsity=0.7,
                                 n_steps=10, start_step=0)
        pruner   = IterativePruner(schedule)

        weights = RNG.standard_normal((rows, cols)).astype(np.float32)

        mean_p5, lo_5, hi_5 = _timeit(
            lambda: pruner.prune_step(weights, step=5), n=200
        )
        _row(f"prune_step() step=5 {rows}×{cols} sparsity≈{pruner.current_sparsity(5):.0%}",
             f"{mean_p5:.1f} µs", f"min={lo_5:.1f} max={hi_5:.1f} µs")

        mean_p10, lo_10, hi_10 = _timeit(
            lambda: pruner.prune_step(weights, step=10), n=200
        )
        _row(f"prune_step() step=10 {rows}×{cols} sparsity≈{pruner.current_sparsity(10):.0%}",
             f"{mean_p10:.1f} µs", f"min={lo_10:.1f} max={hi_10:.1f} µs")

        results["iter_prune"] = dict(
            prune_step5_mean_us=mean_p5,
            prune_step10_mean_us=mean_p10,
            sparsity_at_step10=pruner.current_sparsity(10),
        )
    except Exception as e:
        _skip("IterativePruner", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary table + Markdown export
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    _hdr("Summary — Wave 23+24 Kernel Latencies")

    rows_wave23 = [
        ("vision_kv_fuse",     "VisionKVFuseCache append()",    "append_mean_us"),
        ("image_token_prune",  "ImageTokenPruner prune()",       "prune_mean_us"),
        ("rag_prefetch",       "RAGPrefetcher record_access()",  "record_access_mean_us"),
        ("cot_compress",       "CoTCompressor compress() 256t",  "compress_256_mean_us"),
        ("multimodal_batch",   "MultiModalBatcher next_batch()", "next_batch_mean_us"),
        ("contextual_rerank",  "ContextualReranker rerank()",    "rerank_mean_us"),
        ("cross_modal_attn",   "CrossModalAttn forward()",       "forward_mean_us"),
        ("hierarchical_kv",    "HierarchicalKV put() hot",       "put_mean_us"),
        ("stream_rag",         "StreamRAGInjector inject()",     "inject_mean_us"),
        ("cross_doc_attn",     "CrossDocAttn forward() 4-docs",  "forward_mean_us"),
        ("video_frame_prune",  "VideoFramePruner temporal()",    "prune_temporal_mean_us"),
        ("embedding_gate",     "EmbeddingGate gate() 32-tok",    "gate_mean_us"),
        ("long_context_chunk", "LongContextChunker chunk() 2k",  "chunk_2048_mean_us"),
        ("modality_router",    "ModalityRouter route()+complete","route_complete_mean_us"),
    ]

    rows_wave24 = [
        ("ternary_quant",   "TernaryQuantizer quantize()",     "quantize_mean_us"),
        ("binary_attn",     "BinaryAttention forward()",       "forward_mean_us"),
        ("structured_prune","StructuredPruner prune() 2:4",    "prune_mean_us"),
        ("layer_fuse",      "LayerFuser fuse()",               "fuse_mean_us"),
        ("weight_sharing",  "WeightSharer get_weight()",       "get_effective_weight_mean_us"),
        ("quant_calib",     "QuantCalibrator calibrate()",     "calibrate_minmax_mean_us"),
        ("sparse_weight",   "SparseWeightStore compress()",    "compress_mean_us"),
        ("delta_compress",  "DeltaCompressor compress()",      "compress_mean_us"),
        ("model_surgery",   "ModelSurgeon plan()",             "plan_mean_us"),
        ("zero_quant_v2",   "ZeroQuantV2 quantize()",          "quantize_mean_us"),
        ("gptq_layer",      "GPTQCalibrator calibrate()",      "calibrate_mean_us"),
        ("sparse_moe",      "SparseMoERouter route()",         "route_mean_us"),
        ("awq_v2",          "AWQv2Calibrator calibrate()",     "calibrate_mean_us"),
        ("iter_prune",      "IterativePruner prune_step()",    "prune_step5_mean_us"),
    ]

    print(f"\n{B}Wave 23 — Multi-Modal & Long Context Intelligence{NC}")
    for key, label, field in rows_wave23:
        if key in results and field in results[key]:
            _row(label, f"{results[key][field]:.2f} µs")

    print(f"\n{B}Wave 24 — Quantisation Evolution & Model Surgery{NC}")
    for key, label, field in rows_wave24:
        if key in results and field in results[key]:
            _row(label, f"{results[key][field]:.2f} µs")


def to_markdown(results: dict) -> str:
    lines = [
        "# Squish — Wave 23+24 Benchmark Results",
        "",
        "> CPU/numpy micro-benchmarks — pure Python, no GPU required.",
        "> Measured on Apple Silicon M-series (or equivalent CPU).",
        "",
        "---",
        "",
        "## Wave 23 — Multi-Modal & Long Context Intelligence",
        "",
        "| Module | Operation | Latency (µs) | Notes |",
        "|--------|-----------|:------------:|-------|",
    ]

    if "vision_kv_fuse" in results:
        r = results["vision_kv_fuse"]
        lines += [f"| VisionKVFuseCache | `append()` h=8 d=64 single token | {r['append_mean_us']:.2f} | Per-token KV insertion |"]
    if "image_token_prune" in results:
        r = results["image_token_prune"]
        lines += [f"| ImageTokenPruner | `prune()` h=8 n_tokens=196 ratio=0.5 | {r['prune_mean_us']:.1f} | Entropy-saliency pruning |"]
    if "rag_prefetch" in results:
        r = results["rag_prefetch"]
        lines += [
            f"| RAGPrefetcher | `record_access()` 32-token doc | {r['record_access_mean_us']:.2f} | Access frequency tracking |",
            f"| RAGPrefetcher | `get_warmup_candidates()` top_k=16 | {r['get_candidates_mean_us']:.2f} | Recency-weighted candidate ranking |",
        ]
    if "cot_compress" in results:
        r = results["cot_compress"]
        lines += [
            f"| CoTCompressor | `compress()` 256-token chain | {r['compress_256_mean_us']:.1f} | Entropy-based token distillation |",
            f"| CoTCompressor | `compress()` 64-token chain | {r['compress_64_mean_us']:.1f} | Short chain compression |",
        ]
    if "multimodal_batch" in results:
        r = results["multimodal_batch"]
        lines += [
            f"| MultiModalBatcher | `add_request()` text len=128 | {r['add_request_mean_us']:.2f} | Queue insertion |",
            f"| MultiModalBatcher | `next_batch()` max=8 | {r['next_batch_mean_us']:.2f} | Batch assembly |",
        ]
    if "contextual_rerank" in results:
        r = results["contextual_rerank"]
        lines += [f"| ContextualReranker | `rerank()` h=8 seq=256 d=64 | {r['rerank_mean_us']:.1f} | Recency+content blend |"]
    if "cross_modal_attn" in results:
        r = results["cross_modal_attn"]
        lines += [f"| CrossModalAttention | `forward()` h=8 text=32 vis=64 d=64 | {r['forward_mean_us']:.1f} | Cross-modal attention |"]
    if "hierarchical_kv" in results:
        r = results["hierarchical_kv"]
        lines += [
            f"| HierarchicalKVStore | `put()` hot tier h=8 d=64 | {r['put_mean_us']:.2f} | LRU tier insert |",
            f"| HierarchicalKVStore | `get()` hot hit | {r['get_hit_mean_us']:.2f} | Cache hit retrieval |",
        ]
    if "stream_rag" in results:
        r = results["stream_rag"]
        lines += [
            f"| StreamRAGInjector | `inject()` embed_dim=256 | {r['inject_mean_us']:.2f} | Document index update |",
            f"| StreamRAGInjector | `retrieve()` top_k=3 | {r['retrieve_mean_us']:.2f} | Cosine similarity retrieval |",
        ]
    if "cross_doc_attn" in results:
        r = results["cross_doc_attn"]
        lines += [f"| CrossDocAttention | `forward()` h=8 q=16 k=32 docs=4 | {r['forward_mean_us']:.1f} | Multi-doc fusion |"]
    if "video_frame_prune" in results:
        r = results["video_frame_prune"]
        lines += [
            f"| VideoFramePruner | `prune_temporal()` n=32 embed=256 | {r['prune_temporal_mean_us']:.1f} | Cosine similarity dedup |",
            f"| VideoFramePruner | `prune_spatial()` patches=196 | {r['prune_spatial_mean_us']:.1f} | Spatial attention pruning |",
        ]
    if "embedding_gate" in results:
        r = results["embedding_gate"]
        lines += [f"| EmbeddingGate | `gate()` n_tokens=32 embed=512 | {r['gate_mean_us']:.2f} | Dot-product routing |"]
    if "long_context_chunk" in results:
        r = results["long_context_chunk"]
        lines += [
            f"| LongContextChunker | `chunk()` seq=2048 embed=128 | {r['chunk_2048_mean_us']:.1f} | Semantic boundary detection |",
            f"| LongContextChunker | `chunk()` seq=256 | {r['chunk_256_mean_us']:.1f} | Short context chunking |",
        ]
    if "modality_router" in results:
        r = results["modality_router"]
        lines += [f"| ModalityRouter | `route()+complete()` text | {r['route_complete_mean_us']:.2f} | SLO-aware request dispatch |"]

    lines += [
        "",
        "## Wave 24 — Quantisation Evolution & Model Surgery",
        "",
        "| Module | Operation | Latency (µs) | Notes |",
        "|--------|-----------|:------------:|-------|",
    ]

    if "ternary_quant" in results:
        r = results["ternary_quant"]
        lines += [
            f"| TernaryQuantizer | `quantize()` 256×256 float32 | {r['quantize_mean_us']:.1f} | ±1/0 ternary + scale |",
            f"| TernaryQuantizer | `dequantize()` int8 → float32 | {r['dequantize_mean_us']:.1f} | Scale broadcast |",
        ]
    if "binary_attn" in results:
        r = results["binary_attn"]
        lines += [f"| BinaryAttention | `forward()` h=8 seq=64 d=64 | {r['forward_mean_us']:.1f} | 1-bit Q/K attention |"]
    if "structured_prune" in results:
        r = results["structured_prune"]
        lines += [f"| StructuredPruner | `prune()` 2:4 256×256 | {r['prune_mean_us']:.1f} | N:M magnitude zeroing |"]
    if "layer_fuse" in results:
        r = results["layer_fuse"]
        lines += [
            f"| LayerFuser | `cosine_similarity()` 512×512 | {r['cosine_similarity_mean_us']:.2f} | Frobenius-norm cosine sim |",
            f"| LayerFuser | `fuse()` 512×512 | {r['fuse_mean_us']:.2f} | Weighted mean fusion |",
        ]
    if "weight_sharing" in results:
        r = results["weight_sharing"]
        lines += [f"| WeightSharer | `get_effective_weight()` d=256 rank=16 | {r['get_effective_weight_mean_us']:.2f} | Base + SVD delta reconstruction |"]
    if "quant_calib" in results:
        r = results["quant_calib"]
        lines += [f"| QuantCalibrator | `calibrate()` minmax 128×256 | {r['calibrate_minmax_mean_us']:.1f} | Per-channel scale computation |"]
    if "sparse_weight" in results:
        r = results["sparse_weight"]
        lines += [
            f"| SparseWeightStore | `compress()` 2:4 256×256 → CSR | {r['compress_mean_us']:.1f} | N:M sparse encoding |",
            f"| SparseWeightStore | `decompress()` CSR → dense | {r['decompress_mean_us']:.1f} | CSR scatter |",
        ]
    if "delta_compress" in results:
        r = results["delta_compress"]
        lines += [
            f"| DeltaCompressor | `compress()` SVD rank=16 256×256 | {r['compress_mean_us']:.1f} | Truncated SVD delta |",
            f"| DeltaCompressor | `decompress()` rank=16 → 256×256 | {r['decompress_mean_us']:.1f} | Low-rank reconstruction |",
        ]
    if "model_surgery" in results:
        r = results["model_surgery"]
        lines += [
            f"| ModelSurgeon | `plan()` 32L 32H | {r['plan_mean_us']:.2f} | Pruning plan generation |",
            f"| ModelSurgeon | `estimate_reduction()` static | {r['estimate_reduction_mean_us']:.2f} | Parameter reduction estimate |",
        ]
    if "zero_quant_v2" in results:
        r = results["zero_quant_v2"]
        lines += [
            f"| ZeroQuantV2 | `quantize()` 8-bit group=128 128×256 | {r['quantize_mean_us']:.1f} | Groupwise + outlier |",
            f"| ZeroQuantV2 | `dequantize()` | {r['dequantize_mean_us']:.1f} | Scale + residual add |",
        ]
    if "gptq_layer" in results:
        r = results["gptq_layer"]
        lines += [f"| GPTQCalibrator | `calibrate()` 4-bit 64×64 | {r['calibrate_mean_us']:.1f} | Hessian-weighted column rounding |"]
    if "sparse_moe" in results:
        r = results["sparse_moe"]
        lines += [f"| SparseMoERouter | `route()` n_tokens=32 experts=8 top_k=2 | {r['route_mean_us']:.2f} | Top-k gating + load balance |"]
    if "awq_v2" in results:
        r = results["awq_v2"]
        lines += [
            f"| AWQv2Calibrator | `calibrate()` 128×256 steps=20 | {r['calibrate_mean_us']:.1f} | Scale+shift grid search |",
            f"| AWQv2Calibrator | `quantize()` 4-bit 128×256 | {r['quantize_mean_us']:.1f} | Activation-aware quantisation |",
        ]
    if "iter_prune" in results:
        r = results["iter_prune"]
        lines += [
            f"| IterativePruner | `prune_step()` step=5 256×256 | {r['prune_step5_mean_us']:.1f} | Cubic schedule magnitude prune |",
            f"| IterativePruner | `prune_step()` step=10 (target) | {r['prune_step10_mean_us']:.1f} | Full-sparsity step |",
        ]

    lines += [
        "",
        "---",
        "",
        "*Generated by `dev/benchmarks/bench_wave23_24.py`*",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave 23+24 Squish micro-benchmark suite"
    )
    parser.add_argument(
        "--output", "-o",
        default="dev/results/wave23_24_bench.json",
        help="Path for JSON results (default: dev/results/wave23_24_bench.json)",
    )
    parser.add_argument(
        "--markdown", "-m",
        action="store_true",
        help="Also write Markdown results to docs/benchmark_wave23_24.md",
    )
    args = parser.parse_args()

    print(f"\n{W}{B}Squish Wave 23+24 Benchmark Suite{NC}")
    print(f"{D}CPU/numpy micro-benchmarks  ·  no GPU required{NC}\n")

    results: dict = {}

    # Wave 23 — Multi-Modal & Long Context Intelligence
    bench_vision_kv_fuse(results)
    bench_image_token_prune(results)
    bench_rag_prefetch(results)
    bench_cot_compress(results)
    bench_multimodal_batch(results)
    bench_contextual_rerank(results)
    bench_cross_modal_attn(results)
    bench_hierarchical_kv(results)
    bench_stream_rag(results)
    bench_cross_doc_attn(results)
    bench_video_frame_prune(results)
    bench_embedding_gate(results)
    bench_long_context_chunk(results)
    bench_modality_router(results)

    # Wave 24 — Quantisation Evolution & Model Surgery
    bench_ternary_quant(results)
    bench_binary_attn(results)
    bench_structured_prune(results)
    bench_layer_fuse(results)
    bench_weight_sharing(results)
    bench_quant_calib(results)
    bench_sparse_weight(results)
    bench_delta_compress(results)
    bench_model_surgery(results)
    bench_zero_quant_v2(results)
    bench_gptq_layer(results)
    bench_sparse_moe(results)
    bench_awq_v2(results)
    bench_iter_prune(results)

    # Print summary
    print_summary(results)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n{G}✓{NC} Results saved to {out_path}")

    # Optional Markdown
    if args.markdown:
        md_path = Path("docs/benchmark_wave23_24.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w") as fh:
            fh.write(to_markdown(results))
        print(f"{G}✓{NC} Markdown saved to {md_path}")

    n_ok = len(results)
    n_total = 28
    print(f"\n{B}Benchmarked {n_ok}/{n_total} modules{NC}")
    if n_ok < n_total:
        print(f"{Y}  {n_total - n_ok} module(s) were skipped (see SKIP lines above){NC}")


if __name__ == "__main__":
    main()
