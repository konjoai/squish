# Squish — Wave 23+24 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 23 — Multi-Modal & Long Context Intelligence

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| VisionKVFuse | `append()` fused image+text KV | 1.43 | Cross-modal KV slot append |
| VisionKVFuse | `get_kv()` hit | 1.37 | Fused retrieval |
| ImageTokenPrune | `prune()` 256 tokens → key 64 | 1070 | Importance-score pruning |
| ImageTokenPrune | `prune()` best case | 980 | Min latency |
| RAGPrefetch | `record_access()` 8-doc window | 4.89 | Rolling LRU access record |
| RAGPrefetch | `get_candidates()` top_k=4 | 4.87 | Recency-weighted ranking |
| CoTCompress | `compress()` 256-token input | 75.8 | BM25 + entropy filtering |
| CoTCompress | `compress()` 64-token input | 26.0 | Short-context path |
| MultiModalBatch | `add_request()` text+image | 0.67 | Priority-queue enqueue |
| MultiModalBatch | `next_batch()` | 0.28 | O(1) dequeue |
| ContextualRerank | `rerank()` 16 docs | 87.9 | BM25 relevance re-scoring |
| ContextualRerank | `rerank()` with query embedding | 42.7 | Query-conditioned rerank |
| CrossModalAttn | `forward()` seq=64 h=4 d=32 | 455 | Cross-attention over modalities |
| CrossModalAttn | `forward()` best case | 409 | Min latency |
| HierarchicalKV | `put()` 8 tokens kv=(8,4,32) | 1.74 | L1 → L2 tier write |
| HierarchicalKV | `get()` hit (33% hit rate) | 0.72 | L1 cache hit path |
| HierarchicalKV | `get()` miss | 0.25 | L2 fallback lookup |
| StreamRAG | `inject()` 4 passages | 3.47 | Passage token injection |
| StreamRAG | `retrieve()` 4 passages | 21.4 | BM25 similarity retrieval |
| CrossDocAttn | `forward()` seq=64 docs=2 | 548 | Cross-document attention |
| CrossDocAttn | `forward()` best case | 496 | Min latency |
| VideoFramePrune | `prune_temporal()` 16 frames | 32.2 | Temporal motion difference |
| VideoFramePrune | `prune_spatial()` 16 frames | 28.1 | Spatial entropy score |
| EmbeddingGate | `gate()` 8 tokens hidden=64 | 37.3 | Cosine-threshold gating |
| EmbeddingGate | `gate()` best case | 26.4 | Min latency |
| LongContextChunk | `chunk()` seq=2048 chunk=256 | 207 | 8-chunk split |
| LongContextChunk | `chunk()` seq=256 chunk=256 | 0.65 | Single-chunk fast path |
| ModalityRouter | `route()+complete()` text | 0.65 | Type-dispatch routing |
| ModalityRouter | `route()` best case | 0.54 | Min latency |

---

## Wave 24 — Quantisation Evolution & Model Surgery

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| TernaryQuant | `quantize()` 256×256 weight | 719 | 1.58-bit {−1,0,+1} · 3.1% sparsity |
| TernaryQuant | `dequantize()` 256×256 | 38.5 | Scale multiply |
| BinaryAttn | `forward()` seq=64 h=4 d=32 | 224 | XNOR/popcount attention |
| BinaryAttn | `forward()` best case | 199 | Min latency |
| StructuredPrune | `prune()` 128×256 → 50% sparse | 1255 | Group L2-norm row pruning |
| StructuredPrune | `prune()` best case | 1156 | Min latency |
| LayerFusion | `cosine_similarity()` h=256 | 20.1 | Layer redundancy probe |
| LayerFusion | `fuse()` 2-layer merge 256-dim | 109 | Weighted average merge |
| WeightSharing | `get_effective_weight()` shared | 25.3 | Offset-based shared tensor |
| WeightSharing | memory_ratio | 0.25 | 4 layers → 1 copy |
| QuantCalib | `calibrate_minmax()` 128×256 | 606 | Per-channel min/max pass |
| SparseWeight | `compress()` 128×256 30% sparse | 1316 | CSR format conversion |
| SparseWeight | `decompress()` CSR → dense | 152 | Dense reconstruction |
| SparseWeight | compression_ratio | 1.33× | vs dense storage |
| DeltaCompress | `compress()` 256×256 SVD-delta | 9087 | SVD rank-16 delta fit |
| DeltaCompress | `decompress()` | 23.8 | rank-16 matmul |
| DeltaCompress | compression_ratio | **7.98×** | SVD delta vs full weight |
| ModelSurgery | `plan()` 3-layer graph | 0.59 | Surgery plan generation |
| ModelSurgery | `estimate_reduction()` | 0.45 | Perf delta estimation |
| ZeroQuantV2 | `quantize()` 256×256 INT8 | 233 | Outlier-separated quant |
| ZeroQuantV2 | `dequantize()` INT8 → float | 66.0 | Per-group dequant |
| ZeroQuantV2 | outlier_rate | 1.17% | Weight outlier fraction |
| GPTQLayer | `calibrate()` 32×220 inputs | 1053 | Hessian-gated GPTQ update |
| SparseMoE | `route()` 4 experts · 2 top-k | 58.3 | Load-balanced expert gate |
| AWQv2 | `calibrate()` 128×256 | 73402 | One-time activation calibration |
| AWQv2 | `quantize()` post-calibration | 64.4 | Per-row INT4 scale |
| IterPrune | `prune_step()` 5 iterations | 956 | Magnitude prune + regrow |
| IterPrune | `prune_step()` 10 iterations | 784 | 70% sparsity achieved |
| IterPrune | sparsity_at_step10 | 70% | Target convergence |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.

| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| Image token reduction | **50–70%** fewer tokens | ImageTokenPrune importance scoring |
| Video token reduction | **60–80%** fewer tokens | VideoFramePrune temporal+spatial |
| CoT length reduction | **30–50%** fewer tokens | CoTCompress BM25+entropy |
| KV memory (hierarchical) | **hot-tier hit rate** driven | HierarchicalKV L1/L2 |
| RAG prefetch hit rate | **access-pattern** driven | RAGPrefetch predictive fetch |
| Cross-modal attention FLOPs | **seq × modality_len** | CrossModalAttn |
| Weight size (1.58-bit) | **~5× vs FP16** | TernaryQuant {−1,0,+1} |
| Weight size (binary) | **~16× vs FP16** | BinaryAttn 1-bit |
| Structured sparsity | **2:4 or custom ratio** | StructuredPrune group L2 |
| Delta compression | **7.98× ratio** measured | DeltaCompress SVD rank-16 |
| Iterative pruning sparsity | **70% sparsity** converges | IterPrune magnitude prune |
| Model size (AWQ INT4) | **4× vs FP16** | AWQv2 activation-aware |
| Expert routing overhead | **<100 µs** / dispatch | SparseMoE load-balanced |
| Layer fusion memory | **N_fused × layer_size** | LayerFusion cosine-redundancy |

---

## Accuracy Baseline

> Wave 24 quantisation modules trade off compression ratio against generation quality.
> The baseline below is from the Squish INT8 compressed Qwen2.5-1.5B reference run.
> Per-module quality impact (perplexity delta, accuracy delta) is measured separately
> via `dev/benchmarks/bench_wave23_24.py` with a loaded model.

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
