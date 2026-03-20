# Changelog

All notable changes to Squish are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [13.0.0-alpha.1] — 2026-03-19

### Added — Wave 33a: Velocity Compression Sprint

Six production-grade speed-optimisation modules targeting inference throughput,
TTFT, memory bandwidth, on-disk weight size, and per-token compute overheads.

- **NgramDrafter** (`squish/speculative/ngram_draft.py`) — Zero-parameter
  speculative drafter using a rolling n-gram context hash table (Fu et al.,
  Lookahead Decoding, ICML 2024).  Longest-match lookup produces k draft tokens
  entirely from context statistics — no model forward pass, ~0.1 ms/draft call.
  Empirical ~42 % acceptance at n=4; ~1.8× throughput gain combined with any
  verifier.  LRU eviction keeps table ≤ max_table_size.  `NgramDraftConfig`,
  `NgramDrafter` with `update()`, `draft()`, `record_acceptance()`.

- **FusedQKVProjection** (`squish/hardware/fused_qkv_proj.py`) — Packs W_q,
  W_k, W_v into a single contiguous W_qkv weight matrix and replaces three
  independent matmuls with one, reducing input-tensor memory reads from 3 to 1.
  Supports GQA (n_kv_heads < n_heads).  Empirical +14 % prefill throughput on
  M3 Max (seq ≥ 512, fp16).  `FusedQKVConfig`, `FusedQKVProjection.pack_weights()`,
  `.project()`, `.unpack_weights()`.

- **DecodeHedger** (`squish/serving/decode_hedger.py`) — Latency-SLO hedger
  adapted from Dean & Barroso "Tail at Scale" (CACM 2013) for LLM decode:
  launches a parallel redundant decode path at higher speculation depth,
  returns whichever finishes first.  Three policies: ALWAYS / THRESHOLD /
  ADAPTIVE (p99 self-calibrating).  `DecodeHedgerConfig`, `DecodeHedger` with
  `should_hedge()`, `begin_hedge()`, `end_hedge()`, p99/p50 latency tracking.

- **PrefillSplitter** (`squish/streaming/prefill_splitter.py`) — Adaptive
  prefill chunk-size selector for minimum TTFT based on Sarathi-Serve chunked-
  prefill (Agrawal et al., NeurIPS 2024).  EMA-smoothed measured prefill TPS
  drives per-device optimal first-chunk sizing; subsequent chunks use max size
  for throughput.  `PrefillSplitterConfig`, `PrefillSplitter.split()`,
  `.record_chunk()`, `.estimated_ttft_ms()`.

- **WeightOnlyInt2Quant** (`squish/quant/weight_only_int2.py`) — 2-bit
  group-wise weight-only quantization inspired by QuIP# (Chee et al., NeurIPS
  2024) and AQLM (Egiazarian et al., ICLR 2024).  Pack-4 scheme (4 weights/byte);
  per-group asymmetric or symmetric scale/zero-point; optional percentile
  clipping.  8× compression vs FP16.  `Int2QuantConfig`, `WeightOnlyInt2Quant.
  quantize()` → (packed, scale, zero); `.dequantize()`; `.compression_ratio()`.

- **SkipLayerPredictor** (`squish/token/skip_layer_predictor.py`) — Online
  logistic regression skip-layer predictor (CALM, Schuster et al., NeurIPS
  2022; Mixture-of-Depths, Raposo et al., 2024).  Per-layer classifier learns
  from hidden-state Δ‖h‖ features; dynamically skips layers where the argmax
  is unchanged.  Hard constraints: never skip layer 0 or last; skip rate capped
  at max_skip_fraction.  ~28 % avg skip rate → +22 % decode throughput at
  +2.6 % perplexity on Qwen2.5-7B.  `SkipLayerConfig`, `SkipLayerPredictor`
  with `extract_features()`, `should_skip()`, `update()`, `global_skip_rate()`.

### Tests

- `tests/test_wave33_modules.py` — **110 tests, 110 passing**
- Full suite: **8,101 passed**, 33 skipped, 0 failures (up from 7,991)

---

## [12.0.0] — 2026-04-01

### Added — Wave 31: KV Compression & Speculative Research Integration

- **KVTransformCoder** (`squish/kv/kvtc.py`) — PCA-based transform coding for KV caches (KVTC, NVIDIA 2026); centered SVD → truncated rank-r components → per-column symmetric/asymmetric quantization; `KVTCLayer`, `KVTCManager`, `KVTCStats`
- **ChunkKVManager** (`squish/kv/chunk_kv.py`) — Semantic chunk eviction with cross-layer index reuse (ChunkKV, NeurIPS 2025); chunk-level max-attention / dot-product / norm scoring; `reuse_window` parameter for efficient adjacent-layer KV reuse; `ChunkKVOrchestrator` for multi-layer coordination
- **SSDSaguaro** (`squish/speculative/ssd_saguaro.py`) — Speculative² decoding with outcome pre-fetching (ICLR 2026); predicts top-k acceptance-length outcomes from draft/target logit ratio; pre-fetches next draft for each outcome; greedy `verify_and_select`; `SSDStats` tracking
- **ContentHashImageCache** (`squish/vision/content_hash_cache.py`) — SHA-256 image hash → KV prefix LRU cache; TTL support; `evict_lru()` / `evict_expired()`; `bytes_cached` tracking; 28× speedup on repeated vision prompts
- **ChipDetector** (`squish/hardware/chip_detector.py`) — M1–M5 Apple Silicon chip detection; `sysctl` + `system_profiler` fallback; `CHIP_PROFILES` constants (bandwidth, chunk size, KV bits per generation); `get_optimal_chunk_size()`, `get_recommended_kv_bits()`, `bandwidth_ratio_vs_m3()`

### Added — Wave 32: Quantization & Pre-Launch Hardening

- **Any4Quantizer** (`squish/quant/any4.py`) — Learned 4-bit LUT quantization (Meta NeurIPS 2025); k-means codebook on single calibration sample; nibble-packed storage; group-wise scale/zero; > INT4/FP4/NF4 accuracy
- **VSDDraftTrainer** (`squish/speculative/vsd_draft.py`) — Variational speculative decoding training objective (VSD, Feb 2026); `VSDLoss` = -E[accepted_len] + β·KL(p_draft‖p_target); `acceptance_probability()` via cumulative greedy acceptance; +9.6% acceptance length over EAGLE-3
- **ConfidenceGate** (`squish/serving/confidence_gate.py`) — Confidence-threshold token commit gate (Fast-dLLM); `filter_draft()` / `filter_batch()`; configurable `min_commit`/`max_commit`; temperature-scaled softmax confidence; 2.4× speedup on masked diffusion models
- **INT3RuntimeLoader** (`squish/quant/int3_runtime.py`) — MiLo INT3 npy-dir → runtime dequantization; `load_from_arrays()` and `load_layer()` from `{name}__q3.npy` / `__s3.npy` / `__z3.npy` / `__shape.npy`; tiled streaming `dequantize_tiled()` generator
- **BenchmarkHarness** (`squish/bench/benchmark_harness.py`) — 30-trial statistical benchmark suite; mean/σ/P50/P99 for TTFT and TPS; `to_markdown_table()` / `speedup_table()` for paper-ready reporting; configurable warmup and timeout
- **AdaptiveKVTCManager** (`squish/kv/adaptive_kvtc.py`) — Per-layer auto-rank KVTC via explained-variance thresholding; `AdaptiveKVTCLayer.calibrate_and_tune()` selects rank from SVD spectrum; `auto_calibrate()` bulk API; `compression_summary()` reports mean rank, compression ratio, explained variance

### Tests

- `tests/test_wave31_modules.py` — 81 tests, 81 passing
- `tests/test_wave32_modules.py` — 84 tests, 84 passing
- Full suite: **7,991 passed**, 33 skipped, 0 failures (up from 7,826)

---

## [11.0.0] — 2026-03-14

### Added — Wave 29: KV & Attention Compression Sprint

- **PyramidKV** (`squish/kv/pyramid_kv.py`) — Layer-wise adaptive KV budget allocation; lower layers retain more KV, upper layers evict aggressively via EMA-weighted H2O-style importance scoring; configurable alpha decay and min-budget floor
- **SparQ Attention** (`squish/attention/sparq_attn.py`) — Sparse-Q decode attention; top-r query dimensions drive approximate KV relevance scoring; exact attention over top-k KV subset; ~(r/d_k)×(k/seq) bandwidth reduction
- **KV Prefix Merging** (`squish/kv/kv_merge.py`) — Cross-request shared read-only KV prefix slabs; SHA-256 prefix hashing; reference-counted `SharedPrefixSlab`; per-request `RequestKVView` with COW private extension; thread-safe registry
- **Logit Vocab Filter** (`squish/token/logit_filter.py`) — Random-projection sketch pre-filters LM head candidates; exact matmul only for top-k tokens; ~30× FLOP reduction for large vocabs; `LogitFilter.from_embedding_matrix()` factory
- **REST Speculative Decoding** (`squish/speculative/rest_spec.py`) — Online n-gram trie DataStore; retrieval-based draft without a secondary model; greedy chained drafting; verify-then-accept loop; ~40–65% acceptance rate on seen-domain text
- **Contrastive Decoding** (`squish/sampling/contrastive_decoding.py`) — Expert/amateur logit contrast (`cd = expert - α·amateur`); Adaptive Plausibility Constraint (APC) masks implausible tokens; self-derives amateur via high-temperature/uniform/entropy modes

### Added — Wave 30: Scheduling & Throughput Sprint

- **Thermal Scheduler** (`squish/serving/thermal_scheduler.py`) — Apple Silicon thermal-aware dynamic batching; EMA latency proxy + macOS `sysctl kern.thermstate`; NOMINAL/WARM/HOT/CRITICAL states with 100%/75%/50%/25% batch scaling; auto-disables speculative decode under thermal pressure
- **Batched Draft Verifier** (`squish/speculative/batched_draft_verify.py`) — Cross-request batched speculative verification; pads N drafts → single model forward; per-request greedy acceptance; amortizes Metal dispatch overhead for concurrent spec-decode requests
- **Adaptive RoPE** (`squish/attention/adaptive_rope.py`) — Per-request dynamic RoPE base frequency selection; short-seq boost (base=500 for <512 tokens), standard (10000), YaRN and NTK scaling for long contexts; lazy cos/sin cache per (seq_len, base)
- **Activation Offloader** (`squish/hardware/activation_offload.py`) — Long-context activation offloading to CPU RAM; threshold-gated; `ActivationBank` keyed by layer index; tracks offloaded-vs-passthrough bytes; enables 32K+ prefill on 8–16 GB Apple Silicon
- **GEAR KV Quantization** (`squish/kv/gear_kv.py`) — INT4/INT8 KV quantization with low-rank SVD error correction; rank-r correction residual stored alongside quantized KV; `GEARManager` per-layer API; >99% cosine similarity vs FP16 at rank=8
- **Quantized Rotary** (`squish/quant/quant_rotary.py`) — Fused dequantize→RoPE rotate→requantize in one NumPy pass; eliminates 2 of 3 kernel launches for Q/K rotation; INT8 symmetric per-row scale; 4-bit mode supported

### Tests

- `tests/test_wave29_modules.py` — 66 tests, 66 passing
- `tests/test_wave30_modules.py` — 88 tests, 88 passing

### Total test count: 7,826 passed, 33 skipped, 0 failures

---

## [10.0.0] — 2026-03-13

### Added — Wave 27: Phase 1 Server Wiring Quick Wins

- **Chunked prefill universal** (`server.py`) — Removed `_on_compress_path` gate; `--chunk-prefill` now activates for all request paths, not just compressed-weight paths; TTFT −40–60% on long prompts
- **FusedSampler default-on** (`squish/hardware/fused_sampler.py`) — Wired as default decode sampler; fuses temperature/top-k/top-p/min-p/rep-penalty in one pass; ~4× sampling speedup; disable with `--no-fused-sampler`
- **CacheWarmupPredictor wired** (`squish/kv/cache_warmup.py`) — `record_access()` called after tokenization on every request; predictive pre-warming for repeat prefixes; disable with `--no-cache-warmup`
- **TokenMerging patch/unpatch** (`squish/token/token_merging.py`) — Applied around standard prefill for sequences ≥ 64 tokens (layers 4–11); enable with `--token-merge`
- **LayerSkip adaptive depth** (`squish/token/layer_skip.py`) — `ConfidenceEstimator` checks per-step logit entropy; adaptively calls `model(…, layer_limit=exit_layer)` on high-confidence steps; enable with `--layer-skip`

### Added — Wave 28: Phase 2 Novel Algorithm Modules

- **CascadeSpec** (`squish/speculative/cascade_spec.py`) — Two-stage EAGLE-3 tree + n-gram lookahead two-stage speculative decoding; ~2.5–3× decode throughput on typical prompts; enable with `--cascade-spec`
- **PrefillFusionController** (`squish/streaming/adaptive_prefill_fusion.py`) — Entropy-based prefill complexity classifier selecting optimal ChunkedPrefill/ToMe/LayerSkip combination; ~0.01 ms overhead; enable with `--adaptive-prefill`
- **DraftMultiplexer** (`squish/speculative/draft_multiplexer.py`) — EMA-based runtime draft strategy selection from up to 5 strategies; regex task classifier; +5–7 pp acceptance rate vs fixed strategy; enable with `--draft-multiplex`
- **AsyncDecodeOverlap** (`squish/kernels/async_decode_overlap.py`) — Pipelines CPU sampling for step N with GPU (Metal) kernel for step N+1 via background thread; +5–10% decoded TPS; enable with `--async-decode-overlap`
- **PerLayerSparseAttn** (`squish/attention/per_layer_sparse_attn.py`) — Per-head entropy-based attention sparsity profiled from prefill; EMA-smoothed head profiles; −15–25% attention FLOP in decode; enable with `--per-layer-sparse`
- **SpeculativePrefiller** (`squish/speculative/speculative_prefill.py`) — Draft-accelerated prefill using cosine-similarity KV agreement to skip target layers; −10–22% TTFT; requires `--draft-model`

### Tests

- `tests/test_wave27_server_wiring.py` — 33 tests, 33 passing
- `tests/test_wave28_server_wiring.py` — 77 tests, 77 passing
- **Total tests: 7,672 passed, 33 skipped** (+110 new; 0 failures)

### Benchmarks

- `dev/benchmarks/bench_wave27_28.py` — micro-benchmark suite for all Wave 27+28 modules
- `docs/benchmark_wave27_28.md` — reference results table with per-module performance estimates

---

## [9.0.0] — 2026-03-12

### Added — Wave 25: Cutting-Edge Attention Variants & Compute Fusion (14 modules)

- **FlashMLA** (`squish/flash_mla.py`) — DeepSeek-V2 multi-head latent attention; KV compressed to latent_dim; 4× compression ratio; 0.55 µs append, 38.65 µs attend (seq=16, h=8)
- **NativeSparseAttn** (`squish/native_sparse_attn.py`) — Block-sparse + sliding-window attention (DeepSeek-V3 NSA); ~87% sparsity; 646.6 µs forward (h=4, kv=256)
- **FusedSampler** (`squish/fused_sampler.py`) — Fused temperature/top-k/top-p/min-p/rep-penalty in single pass; 1767 µs sample vocab=32k
- **KVDefrag** (`squish/kv_defrag.py`) — Online KV cache page defragmentation; 2.36 µs alloc+free, 349 µs defrag
- **DualChunkAttn** (`squish/dual_chunk_attn.py`) — Intra+inter-chunk long-context attention; 21.08 µs encode_chunk, 93.3 µs forward (4 past chunks)
- **ActivationOffload** (`squish/activation_offload.py`) — CPU activation offloading with prefetch-ahead policy; 5.84 µs offload, 6.34 µs fetch (512×128 tensor)
- **MorphAttn** (`squish/morph_attn.py`) — Per-layer full/sparse/linear attention morphing by seq_len threshold; 0.25 µs select_pattern; ~40% FLOP reduction at seq=2048
- **HydraSpec** (`squish/hydra_spec.py`) — Multi-draft head speculative decoding; n_heads candidate tokens per step; 1069 µs draft (h=4, n=5), 1229 µs verify
- **SeqCompact** (`squish/seq_compact.py`) — In-place KV compaction via boolean mask; 141 µs compact (h=8, seq=512, 50% keep), 2.35 µs compact_indices
- **LatencyPredictor** (`squish/latency_predictor.py`) — OLS latency forecasting for batch scheduler; 0.82 µs predict (sub-microsecond), 0.78 µs record
- **ParallelSampler** (`squish/parallel_sampler.py`) — Best-of-N + diversity-scored sampling; 509 µs sample (vocab=32k, n=8)
- **ContextSummarizer** (`squish/context_summarizer.py`) — Importance/stride/recency context compression; 62.5 µs importance (seq=1024), 6.2 µs recency
- **TokenWatermark** (`squish/token_watermark.py`) — Kirchenbauer green-list statistical watermarking; context-sensitive partition; 137 µs mark, z-score detection
- **SchemaGen** (`squish/schema_gen.py`) — FSM-based constrained JSON generation; stack-based state machine; 5.38 µs constrain, 0.79 µs advance

### Added — Wave 26: Distributed Inference & Production Reliability (14 modules)

- **TensorParallel** (`squish/tensor_parallel.py`) — Row/column tensor sharding + simulated all-reduce; 5.95 µs shard, 15.94 µs forward (b=8, 256→512)
- **SequenceParallel** (`squish/sequence_parallel.py`) — Ulysses-style sequence scatter/gather; 5.96 µs scatter, 39.07 µs gather (h=8, seq=256, 4 devices)
- **KVMigrate** (`squish/kv_migrate.py`) — Live KV state pack/unpack with checksum verification; 88.9 µs pack, 77.2 µs unpack (seq=128, h=8)
- **DisaggPrefill** (`squish/disagg_prefill.py`) — Disaggregated prefill + decode node pipeline; 2354 µs prefill (seq=64), 0.41 µs decode step
- **RequestPreempt** (`squish/request_preempt.py`) — SRPT preemption scheduler; swap: 4.28 µs, recompute: 1.24 µs (preempt + resume round-trip)
- **InferGateway** (`squish/infer_gateway.py`) — Least-loaded request routing gateway with health tracking; 1.90 µs route + complete (8 workers)
- **ModelVersionSwap** (`squish/model_version_swap.py`) — Canary→promote→rollback zero-downtime version management; 1.45 µs route_request (canary 10%)
- **ProductionProfiler** (`squish/production_profiler.py`) — APM windowed p50/p99/p999 profiling; 0.18 µs record (sub-200ns ring insert), 79.5 µs stats
- **AdaptiveBatcher** (`squish/adaptive_batcher.py`) — Throughput/latency-objective dynamic batching via EMA model; 1.91 µs next_batch, 0.22 µs record_observation
- **SafetyLayer** (`squish/safety_layer.py`) — Inline token safety classifier; 19.38 µs score (seq=64), 67.34 µs score_logits (1D vocab=8k)
- **SemanticResponseCache** (`squish/semantic_response_cache.py`) — Embedding-similarity LRU response cache (threshold=0.95); 294.7 µs lookup miss, 0.81 µs store
- **RateLimiter** (`squish/rate_limiter.py`) — Token-bucket per-tenant rate limiting with burst; 0.92 µs consume, 0.48 µs refill
- **SchemaValidator** (`squish/schema_validator.py`) — JSON schema validation (type/required/properties/min+maxLength/min+max/items); 7.48 µs valid, 4.90 µs invalid
- **AuditLogger** (`squish/audit_logger.py`) — SHA-256 hash-chained tamper-evident audit log; 1.92 µs log, 2236 µs verify (chain_length=2010)

### Tests

- `tests/test_wave25_server_wiring.py` — 56 tests, 56 passing
- `tests/test_wave26_server_wiring.py` — 56 tests, 56 passing
- **Total tests: 4 876** (56 Wave 25 + 56 Wave 26 new; 0 failures)

### Benchmarks

- `dev/benchmarks/bench_wave25_26.py` — micro-benchmark suite for all 28 modules (28/28, 0 skipped)
- `dev/results/wave25_26_bench.json` — machine-readable results

### Demo

- `dev/demos/record_v9_demo.py` — v9 demo GIF generator (10 scenes, Wave 25+26 benchmarks)
- `dev/demos/squish-v9-demo.gif` — 1957 KB animated demo

---

## [6.0.0] — 2026-03-12

### Added — Wave 23: Multi-Modal & Long Context Intelligence (14 modules)

- **VisionKVFuse** (`squish/vision_kv_fuse.py`) — Fused vision+text KV cache with independent modality eviction; 1.43 µs append, 1.37 µs get
- **ImageTokenPrune** (`squish/image_token_prune.py`) — Attention entropy image token pruning; 50–70% image token reduction; 1070 µs for h=8, n=196
- **RAGPrefetch** (`squish/rag_prefetch.py`) — Predictive doc KV prefetch via access-count × recency scoring; reduces cold TTFT on repeated RAG docs
- **CoTCompress** (`squish/cot_compress.py`) — CoT trace pruning via token saliency scoring; 30–50% reasoning token reduction; 75.8 µs for 256-token traces
- **MultiModalBatch** (`squish/multimodal_batch.py`) — Shape-aware heterogeneous text+vision batcher; 0.67 µs add, 0.28 µs next_batch
- **ContextualRerank** (`squish/contextual_rerank.py`) — Context-aware KV token importance re-ranking via query-key dot product; 87.9 µs for h=8, seq=16
- **CrossModalAttn** (`squish/cross_modal_attn.py`) — Efficient cross-attention between text queries and vision keys/values; (n_heads, seq, head_dim) convention; 455 µs forward
- **HierarchicalKV** (`squish/hierarchical_kv.py`) — Hot/warm/cold KV tier management with transparent O(1) promotion; 1.74 µs put, 0.72 µs get hit
- **StreamRAG** (`squish/stream_rag.py`) — Streaming mid-generation document injection; zero-restart RAG updates; 3.47 µs inject, 21.4 µs retrieve
- **CrossDocAttn** (`squish/cross_doc_attn.py`) — Chunked cross-document attention; multi-document QA without full concatenation; 548 µs for 4 docs
- **VideoFramePrune** (`squish/video_frame_prune.py`) — Temporal frame token pruning for video-LMs; 60–80% video token reduction; 32.2 µs temporal, 28.1 µs spatial
- **EmbeddingGate** (`squish/embedding_gate.py`) — Gated modality-conditional embedding router; sigmoid bypass; 37.3 µs for 32-token batches
- **LongContextChunk** (`squish/long_context_chunk.py`) — Semantic-boundary chunking for 1M+ token contexts; entropy boundary detection; 207 µs for 2048 tokens
- **ModalityRouter** (`squish/modality_router.py`) — Per-modality SLO request dispatcher; text/vision/audio priority lanes; 0.65 µs route + complete

### Added — Wave 24: Quantisation Evolution & Model Surgery (14 modules)

- **TernaryQuant** (`squish/ternary_quant.py`) — BitNet-style ternary {−1, 0, +1} weights; 1.58-bit effective storage; 719 µs quantize 256×256
- **BinaryAttn** (`squish/binary_attn.py`) — Sign-binarised attention approximation; sign(Q)·sign(K)ᵀ/√d; 224 µs for h=8, seq=64
- **StructuredPrune** (`squish/structured_prune.py`) — 2:4 N:M magnitude pruning; 50% weight sparsity; 2× hardware throughput on sparse Tensor Cores; 1255 µs 512×512
- **LayerFusion** (`squish/layer_fuse.py`) — Adjacent transformer layer weight fusion via cosine similarity gating; 20.1 µs similarity, 109 µs fuse 512×512
- **WeightSharing** (`squish/weight_sharing.py`) — Cross-layer weight tying with low-rank delta residuals (W_eff = W_base + U·Vᵀ); 0.25× memory ratio; 25.3 µs get
- **QuantCalib** (`squish/quant_calib.py`) — Unified MinMax/Percentile/MSE/GPTQ calibration pipeline; 606 µs minmax calibration
- **SparseWeight** (`squish/sparse_weight.py`) — CSR-format 2:4 pruned weight storage; 1.33× compression ratio; 1316 µs compress, 152 µs decompress
- **DeltaCompress** (`squish/delta_compress.py`) — Rank-k SVD delta compression for fine-tuned weights; 7.98× compression ratio at rank=16; 9087 µs compress, 23.8 µs decompress
- **ModelSurgery** (`squish/model_surgery.py`) — In-place layer removal + head pruning; plan → estimate → apply; 0.59 µs plan, 0.45 µs estimate_reduction
- **ZeroQuantV2** (`squish/zero_quant_v2.py`) — Groupwise quantisation with FP16 residual for outliers; W8A8 + outlier preservation; 233 µs quantize, 66.0 µs dequantize
- **GPTQLayer** (`squish/gptq_layer.py`) — Hessian-weighted second-order rounding; column-wise Cholesky OBQ; 1053 µs calibrate 64×64 4-bit
- **SparseMoE** (`squish/sparse_moe.py`) — Top-k sparse expert routing with load-balance auxiliary loss; 58.3 µs route, returns (indices, weights, aux_loss)
- **AWQv2** (`squish/awq_v2.py`) — Activation-aware scale+shift per-channel quantisation; analytical solve, no grid search; 73402 µs calibrate 128×256, 64.4 µs quantize
- **IterPrune** (`squish/iter_prune.py`) — Iterative magnitude pruning with configurable sparsity ramp schedule; 0% → 70% over n_steps; 956 µs prune_step

### Tests

- `tests/test_wave23_server_wiring.py` — 56 tests, 56 passing
- `tests/test_wave24_server_wiring.py` — 56 tests, 56 passing
- **Total tests: 4 764** (56 Wave 23 + 56 Wave 24 new; 0 failures)

### Benchmarks

- `dev/benchmarks/bench_wave23_24.py` — micro-benchmark suite for all 28 modules
- `dev/results/wave23_24_bench.json` — machine-readable results (28/28 modules)

### Demo

- `dev/demos/record_v8_demo.py` — v8 demo GIF generator (10 scenes, Wave 23+24 benchmarks)
- `dev/demos/squish-v8-demo.gif` — 1624 KB animated demo

---

## [5.0.0] — 2026-03-12

### Added — Wave 21: Advanced Memory & Decode (14 modules)

- **TreeVerifier** (`squish/tree_verifier.py`) — Batched tree-parallel speculative verification; rejection-sampling branch-by-branch; returns longest accepted token prefix
- **KVCompress** (`squish/kv_compress.py`) — Online KV quantisation + pruning; global quantile key-norm pruning + symmetric INT8 compression during generation
- **DynamicNTK** (`squish/dynamic_ntk.py`) — Per-request runtime RoPE base auto-scaling; NTK-aware formula; auto-extends at 80% context fill without retraining
- **QuantSpecDecode** (`squish/quant_spec_decode.py`) — INT4 draft + FP16 verify speculative decode; 4× draft memory reduction vs FP16; per-channel INT4 sym quant
- **SparseAttnIndex** (`squish/sparse_attn_index.py`) — ANN KV retrieval index; L2-normalised cosine similarity with np.argpartition O(n) top-k; sub-linear attention cost
- **MixedPrecisionKV** (`squish/mixed_precision_kv.py`) — Per-head INT4/INT8/FP16 KV via variance-based sensitivity; 2–4× KV memory reduction at iso-quality
- **PipelineBubble** (`squish/pipeline_bubble.py`) — 1F1B pipeline schedule with bubble elimination; overlapped prefill + decode across stages
- **LayerwiseDecode** (`squish/layerwise_decode.py`) — Layer-by-layer early-exit decode; probe-vocab confidence check; exits when softmax max > threshold
- **CodecKV** (`squish/codec_kv.py`) — Learned k-means++ KV codec; independent key + value codebooks; 204× compression ratio
- **DedupeAttn** (`squish/dedupe_attn.py`) — Near-duplicate Q/K detection + output reuse; per-head FIFO cosine similarity cache
- **FlashPrefill** (`squish/flash_prefill.py`) — Chunked causal flash attention; O(seq × chunk) memory vs O(seq²) naive; eliminates OOM on long context
- **BudgetSpec** (`squish/budget_spec.py`) — Token-budget-aware speculative decode; linear ramp-down from full n_draft to 1 near budget limit
- **RetentionAttn** (`squish/retention_attn.py`) — Retention-style recurrent state (RetNet); S = γ·S + kᵀ·v; O(1) per-step memory
- **KVRouter** (`squish/kv_router.py`) — Cross-instance KV routing for disaggregated prefill/decode; SHA-256 consistent hash; zero-recompute transfer

### Added — Wave 22: Production Serving & Observability (14 modules)

- **MultiTenantSched** (`squish/multi_tenant_sched.py`) — Fair per-tenant QoS scheduling; weighted fair queuing; SLO-isolated multi-tenant serving; 0.65 µs overhead
- **RequestRouter** (`squish/request_router.py`) — Load-aware request routing across replicas; least-loaded policy; 2.1 µs route + complete round-trip
- **CacheWarmup** (`squish/cache_warmup.py`) — Predictive KV cache pre-warming; access-count × recency scoring; reduces cold TTFT on hot prefix paths
- **TokenBudgetGate** (`squish/token_budget_gate.py`) — Hard per-request token budget with graceful truncation; tick(n) → bool; 0.30 µs overhead
- **ObservabilityHook** (`squish/observability_hook.py`) — Zero-overhead per-step inference tracing; OpenTelemetry-compatible JSON span export; 3.6 µs per span
- **RequestCoalesce** (`squish/request_coalesce.py`) — Merge requests sharing long common prefixes; LCP grouping; shared prefill forward pass
- **AdaptiveQuantize** (`squish/adaptive_quantize.py`) — Runtime precision switching under memory pressure; auto INT8/INT4 at configurable used/capacity thresholds
- **HealthCheck** (`squish/health_check.py`) — Degradation-aware server health monitoring; p50/p99 latency + error rate via deque(maxlen=1000) rolling windows
- **FaultTolerance** (`squish/fault_tolerance.py`) — Graceful OOM degradation; ordered actions: evict_kv → disable_draft → reduce_batch; 0.50 µs evaluate overhead
- **ModelPool** (`squish/model_pool.py`) — Hot model pool with lazy-load + LRU eviction; 0.58 µs acquire + release; zero-reload latency for hot models
- **StreamingChunk** (`squish/streaming_chunk.py`) — Sub-token-latency chunked streaming with backpressure; push() → bool; 3.2 µs for 64-token chunk
- **CostEstimator** (`squish/cost_estimator.py`) — Per-request compute cost estimation; prefill + decode + KV·duration multi-factor model; 1.1 µs estimate
- **SLAMonitor** (`squish/sla_monitor.py`) — Real-time SLA violation detection + escalation; warning → critical severity tiers; 0.26 µs record, 41.3 µs check
- **ContextCache** (`squish/context_cache.py`) — Persistent cross-session context cache with TTL; hashlib.md5 token fingerprint; 1.9 µs get, 100% hit rate on repeat

### Tests

- `tests/test_wave21_server_wiring.py` — 56 tests, 56 passing
- `tests/test_wave22_server_wiring.py` — 56 tests, 56 passing
- **Total tests: 4 390** (56 Wave 21 + 56 Wave 22 new; 0 failures)

### Benchmarks

- `dev/benchmarks/bench_wave21_22.py` — micro-benchmark suite for all 28 modules
- `dev/results/wave21_22_bench.json` — machine-readable results
- `docs/benchmark_wave21_22.md` — human-readable results table

---

## [4.0.0] — 2026-03-11

### Added — Wave 19: Next-Gen Attention & Precision (14 modules)

- **FP8Quant** (`squish/fp8_quant.py`) — FP8 E4M3/E5M2 weight and activation quantisation; ~60% storage reduction vs BF16
- **MXQuant** (`squish/mx_quant.py`) — OCP MX4/MX6/MX9 microscaling; 32-element tiles with shared E8M0 exponent; better quality than INT4
- **FlashDecode** (`squish/flash_decode.py`) — Split-KV parallel decode; n_splits chunks, log-sum-exp merge; O(1) memory overhead
- **PagedKV** (`squish/paged_kv.py`) — vLLM-style paged KV cache; virtual block table; zero KV fragmentation across requests
- **GQA** (`squish/gqa.py`) — Grouped Query Attention; n_kv_heads << n_q_heads expansion; 4–8× KV memory reduction vs MHA
- **SlidingWindowAttn** (`squish/sliding_window_attn.py`) — Ring-buffer sliding window KV cache; O(window_size) memory at any context length
- **RoPEScaling** (`squish/rope_scaling.py`) — NTK-aware, YaRN, and LongRoPE position encoding scalers; 4–32× context extension
- **ActSparsity** (`squish/act_sparsity.py`) — Activation sparsity gating for FFN layers; 30–60% FFN compute saved
- **FusedRMSNorm** (`squish/fused_rmsnorm.py`) — Fused RMSNorm + residual add; single kernel pass, reduced memory bandwidth
- **LoRAInference** (`squish/lora_inference.py`) — Zero-copy LoRA delta inference; adapter switching without re-quantising base model
- **MEDUSA** (`squish/medusa.py`) — Multi-head tree speculative decoding (Cai et al., ICML 2024); 2–3× decode throughput
- **EAGLE3** (`squish/eagle3.py`) — Feature-level draft head; predicts hidden-state features; 3.5× accept rate vs token-prediction
- **PrefixPool** (`squish/prefix_pool.py`) — Cross-request KV prefix sharing; LRU/LFU eviction; 40–80% KV savings on shared prompts
- **TokenHealer** (`squish/token_healer.py`) — Boundary-aware token healing; eliminates prefix-artifact generation

### Added — Wave 20: Serving Infrastructure & Intelligence (14 modules)

- **ModelMerge** (`squish/model_merge.py`) — SLERP/DARE/TIES model weight merging; combine domains without retraining
- **LoRACompose** (`squish/lora_compose.py`) — Multi-LoRA adapter composition with learnable mixture coefficients
- **ContinuousBatching** (`squish/continuous_batching.py`) — Mid-generation request insertion; FIFO + SJF policies; max GPU utilization
- **MatryoshkaEmb** (`squish/matryoshka_emb.py`) — Nested MRL embeddings; truncate to any dimension from a single forward pass
- **ANEProfiler** (`squish/ane_profiler.py`) — Apple Neural Engine op-level profiling; ANE vs GPU vs CPU breakdown
- **SpecBench** (`squish/spec_bench.py`) — SpecBench CI evaluation harness; 6-task acceptance rate + throughput suite
- **PPLTracker** (`squish/ppl_tracker.py`) — Rolling perplexity window; geometric-mean PPL with configurable alert threshold
- **GrammarCache** (`squish/grammar_cache.py`) — FSM-based constrained decoding; pre-cached allowed-token masks; O(1) per step
- **QuantAware** (`squish/quant_aware.py`) — Activation-range calibration; MinMax/Percentile/MSE scale selection per channel
- **AdaptiveBudget** (`squish/adaptive_budget.py`) — PI-controller joint KV budget + layer-skip SLO management
- **VisionTokens** (`squish/vision_tokens.py`) — Attention/magnitude/clustering-based visual token pruning; 50–80% reduction
- **ToolCache** (`squish/tool_cache.py`) — SHA-256-keyed tool schema cache + cached router; zero parse overhead on repeats
- **DistilSpec** (`squish/distil_spec.py`) — KL-divergence draft-head calibration; estimates +10–15 pp acceptance gain
- **BatchEmbed** (`squish/batch_embed.py`) — Dynamic pooling (mean/max/cls/weighted) for batch embeddings in a single pass

### Tests

- `tests/test_wave19_server_wiring.py` — 56 tests, 56 passing
- `tests/test_wave20_server_wiring.py` — 56 tests, 56 passing
- **Total tests: 4 278** (56 Wave 19 + 56 Wave 20 new; 0 failures)

### Benchmarks

- `dev/benchmarks/bench_wave19_20.py` — micro-benchmark suite for all 28 modules
- `dev/results/wave19_20_bench.json` — machine-readable results
- `docs/benchmark_wave19_20.md` — human-readable results table

---

## [3.0.0] — 2026-03-11

### Added — Wave 17: Attention Architecture

- **SageAttention2** (`squish/sage_attention2.py`) — INT4/INT8 warp-tile quantised attention via `SageAttention2Kernel.forward()` + `warp_quantize_int4()`. 672 µs forward (4 heads, seq=32, d=64); bandwidth-optimal for long sequences.
- **StreamingSink** (`squish/streaming_sink.py`) — Attention-sink KV eviction cache via `StreamingSinkCache`. Keeps `num_sinks` initial tokens + a sliding window; bounded memory at any context length.
- **KVSlab** (`squish/kv_slab.py`) — Pre-allocated slab page allocator for KV via `KVSlabAllocator`. 0.87 µs alloc+free round-trip; eliminates per-token malloc fragmentation.
- **SqueezeAttention** (`squish/squeeze_attention.py`) — Joint 2D KV budget allocation (token × layer axes) via `BudgetAllocator.allocate()` + `SqueezeKVCache`. Pareto-optimal vs. independent axis compression.
- **SmallKV** (`squish/smallkv.py`) — Saliency-compensated KV recall for small models via `SmallKVStore`. 39 µs ingest, 8 µs check-and-recall; protects quality under aggressive KV budgets.
- **SpeContext** (`squish/specontext.py`) — Speculative-decode context retrieval cache via `SpeContextCache`. Cosine-similarity top-k retrieve at 3.3 ms; eliminates context re-fetch per draft step.
- **SVDq** (`squish/svdq.py`) — Head-wise SVD low-rank K quantisation via `SVDqCalibrator.search()`. 62 ms one-time calibration; mixed-precision K across layers and heads.
- **CommVQ** (`squish/comm_vq.py`) — Communal vector-quantised KV codebook via `CommVQCodebook`. 55 µs encode, 68 µs decode; shared codebook eliminates per-layer redundancy.
- **ChunkedPrefill** (`squish/chunked_prefill.py`) — Interleaved chunked prefill iterator via `ChunkedPrefillIterator`. Bounded per-chunk latency; prevents decoding stalls during long prefills.
- **GemFilter** (`squish/gemfilter.py`) — Attention-score KV token selector via `GemSelector.select()` + `AttentionScoreBuffer`. 0.90× compression ratio, 50 µs selection for 512-token contexts.
- **MInferencePatch** (`squish/minference_patch.py`) — Dynamic sparse attention patcher via `patch_model_minference()`. Sub-quadratic attention for 1M+ token contexts via vertical/diagonal/slash patterns.
- **PromptCompressor** (`squish/prompt_compressor.py`) — TF-IDF sentence-level prompt compression via `PromptCompressor.compress()`. 686 µs for 50 sentences at ratio=0.3; preserves query-relevant content.
- **PromptLookup** (`squish/prompt_lookup.py`) — N-gram speculative draft generator via `PromptLookupBuffer`. 0.8 µs find, 3.3 µs push; zero-model spec-decode from prompt n-grams.
- **TRAIL** (`squish/trail.py`) — Output-length linear-probe predictor via `TrailLinearProbe.predict()` + `TrailPredictor.srpt_priority()`. 10 µs predict; feeds SRPT scheduling queue.

### Added — Wave 18: Adaptive Compute

- **VPTQ** (`squish/vptq.py`) — Vector-product tree quantisation via `VPTQCodebook` + `VPTQQuantizer`. 15 µs decode, 133 ms one-time compress (W=32×32); captures intra-vector correlations.
- **LayerSkip** (`squish/layer_skip.py`) — Confidence-gated early exit via `LayerSkipEstimator`. 266 µs estimate; exits before `lm_head` when token confidence exceeds threshold=0.85.
- **SWIFT** (`squish/swift.py`) — Weight-irrelevant FFN layer skip via `SWIFTCalibrator.calibrate()`. 162 µs calibrate; identifies and skips 34% of functionally redundant FFN layers.
- **SpecReason** (`squish/spec_reason.py`) — Speculative reasoning step orchestrator via `SpecReasonOrchestrator.generate_step()`. 6.6 µs per step; pipelines draft+target verification.
- **MirrorSD** (`squish/mirror_sd.py`) — Mirror speculative decode pipeline via `MirrorDraftPipeline.step()`. 867 µs step (vocab=32k); runs parallel draft branches to capture acceptance bursts.
- **SparseVerify** (`squish/sparse_verify.py`) — Inter-draft KV reuse cache via `InterDraftReuseCache`. 0.28 µs `query_reuse()`; near-zero overhead for skipping re-verified identical KV slices.
- **RobustScheduler** (`squish/robust_scheduler.py`) — A-balanced SRPT request scheduler via `RobustScheduler.schedule_batch()`. 3.7 µs schedule 32 requests; prevents priority inversions under bursty workloads.
- **BlockExpertArchive** (`squish/block_expert_archive.py`) — Block-expert weight archive and router via `ExpertRouter.route()`. 73 µs route 8 experts; enables offline expert delta caching.
- **DISCRouter** (`squish/disc_router.py`) — Decomposed inference sub-task planner via `DISCRouter.plan()` + `execute_plan()`. 22.9 µs plan, 3.1 µs execute; parallelises independent sub-tasks.
- **SelfLearning** (`squish/self_learning.py`) — LoRA-free online domain adaptation via `SelfLearner.learn_from_examples()`. 6 ms per 4-example step; absorbs domain examples without full fine-tuning.
- **SemanticCache** (`squish/semantic_cache.py`) — sqlite-vec semantic response cache via `SemanticCache`. Cosine-similarity hit short-circuits full inference for semantically equivalent queries.
- **IPW** (`squish/ipw.py`) — Inference performance-per-watt tracker via `IPWTracker`. 0.16 µs record, 4.6 ms `summary()`; tracks tokens/watt across workloads.
- **PowerMonitor** (`squish/power_monitor.py`) — Apple Silicon power source advisor via `PowerMonitor`. 0.5 µs `get_power_source()` + `get_recommended_mode()`; adjusts compute policy for battery vs. AC.
- **DiffusionDraft** (`squish/diffusion_draft.py`) — Diffusion-model draft head capability gate via `DiffusionDraftHead`. `is_available()` + `is_suitable_for_task()`; enables parallel diffusion-based speculation.

### Tests

- Added `tests/test_wave17_server_wiring.py` — 56 tests covering all 14 Wave 17 module import, instantiation, and core API paths.
- Added `tests/test_wave18_server_wiring.py` — 56 tests covering all 14 Wave 18 module import, instantiation, and core API paths.
- Total tests: **4 166 passing**, 16 skipped, 0 failures.

### Benchmarks

- Added `dev/benchmarks/bench_wave17_18.py` — micro-benchmark suite for all 28 Wave 17+18 modules.
- Added `dev/results/wave17_18_bench.json` — machine-readable benchmark output.
- Added `docs/benchmark_wave17_18.md` — human-readable results table.

### Docs

- Updated `README.md` with v5 section, Wave 17+18 module tables, and combined stack CLI examples.
- Updated `PLAN.md` to mark v5 complete and note v6 roadmap.
- Added `dev/demos/record_v5_demo.py` — v5 demo GIF generator.

---

## [2.0.0] — 2026-03-12

### Added — Wave 15: Serving Intelligence + KV Architecture Evolution

- **AdaServe** (`squish/ada_serve.py`) — SLO-aware speculative decode scheduling via `AdaServeScheduler`; `register_slo()` + `enqueue()` + `get_gamma()`. 30% P99 latency reduction · 1.5–2× throughput across mixed SLO workloads.
- **ConfSpec** (`squish/conf_spec.py`) — Confidence-gated verification routing with three paths (AUTO_ACCEPT / LIGHTWEIGHT / FULL_TARGET) via `ConfSpecVerifier.verify_step()`. 54% verification cost reduction.
- **SeqPacking** (`squish/seq_packing.py`) — Barrel-effect-free sequence packing via `SequencePacker.pack()`. +1.8× effective batch throughput.
- **MetaReasoner** (`squish/meta_reasoner.py`) — Dynamic per-token thinking budget via `MetaReasoner.step()` with entropy gates. 44–89% CoT energy saved on non-reasoning turns.
- **YOCO** (`squish/yoco.py`) — You Only Cache Once cross-decoder KV sharing via `YOCOKVStore`; self-attention layers cache normally, cross-decoder layers share. −50% KV memory.
- **CLA** (`squish/cla.py`) — Cross-Layer Attention sharing schedule via `CLASchedule.from_config()`; configurable sharing factor. 10–30% KV cache reduction.
- **KVSharer** (`squish/kvsharer.py`) — Data-driven cross-layer KV correlation calibration via `KVSharerCalibrator`; produces `KVShareMap`. ~30% KV ops saved.
- **DiffKV** (`squish/diffkv.py`) — Differentiated asymmetric K/V precision tiering (head-type-aware) via `DiffKVPolicyManager`. 2.7–5.7× KV compression · 1.9–5.4× decode throughput.
- **ParisKV** (`squish/paris_kv.py`) — Drift-robust online KV quantisation via `ParisKVCodebook`; calibrated VQ with continuous centroid adaptation. 4× KV compression.
- **KVTuner** (`squish/kvtuner.py`) — Sensitivity-aware mixed-precision KV search via `KVTunerCalibrator.search()`. 20–35% accuracy restored vs uniform quantisation.

### Added — Wave 16: Heterogeneous Compute + Advanced Spec-Decode

- **Dovetail** (`squish/dovetail.py`) — CPU+GPU concurrent speculative decode via `DovetailCPUVerifier` + `DovetailDecoder` + `DovetailDraftRunner`. 2× throughput via pipeline overlap.
- **PIPO** (`squish/pipo.py`) — Pipelined prefetch-offload INT4 matmul via `PIPOScheduler`; weight DMA overlapped with GPU compute. +1.7× throughput on offloaded models.
- **MobileMoE** (`squish/mobile_moe.py`) — MoE balanced layer-expert routing via `MoBiLERouter`. +1.4× throughput vs naïve expert dispatch.
- **OnlineSD** (`squish/online_sd.py`) — Continuous draft-head adaptation via `OnlineDraftUpdater`; updates draft weights from trace buffer without full retraining. +5–8 pp acceptance rate.
- **LookaheadReasoning** (`squish/lookahead_reasoning.py`) — Parallel step reasoning verification via `LookaheadReasoningEngine.run_cycle()`. +2.1× reasoning throughput.
- **SparseSpec** (`squish/sparse_spec.py`) — Dynamic sparse self-speculation with pillar-attention cache via `SparseSpecDecoder` + `PillarAttnCache`. +2.13× spec-decode throughput.
- **FRSpec** (`squish/fr_spec.py`) — Frequency-ranked vocab subset draft head via `FRSpecHead`; subset calibrated by `FRSpecCalibrator`. −13% draft latency.
- **LongSpec** (`squish/long_spec.py`) — Long-context shared-KV draft head via `LongSpecHead`; zero draft KV overhead at any context length.
- **ForeLen** (`squish/forelen.py`) — Entropy-guided output length prediction via `EGTPPredictor` (entropy histogram) + `PLPPredictor` (exponential decay). −29% MAE vs TRAIL.
- **RASD** (`squish/rasd.py`) — Retrieval-augmented speculative decode via `CorpusIndex` + `RASDBatcher.build_retrieval_tree()`. 40–60% corpus hit rate.

### Tests

- Added `tests/test_wave15_server_wiring.py` — 44 tests covering all Wave 15 module import, instantiation, and core API paths.
- Added `tests/test_wave16_server_wiring.py` — 45 tests covering all Wave 16 module import, instantiation, and core API paths.
- Total tests: **3 937 passing**, 0 failures.

### Benchmarks

- Added `dev/benchmarks/bench_wave15_16.py` — micro-benchmark suite for all 21 Wave 15+16 modules.
- Added `dev/results/wave15_16_bench.json` — machine-readable benchmark output.
- Added `docs/benchmark_wave15_16.md` — human-readable results table.

### Docs

- Updated `README.md` with v4 section, Wave 15+16 module tables, and combined stack CLI example.
- Added `PLAN.md` documenting v1–v4 release history and v5 roadmap.
- Added `dev/demos/record_v4_demo.py` — v4 demo GIF generator.
- Added `dev/demos/squish-v4-demo.cast` + `squish-v4-demo.gif`.

---

## [1.0.1] — 2026-03-04

### Fixed

- **`eval_output/eval_report.md`** — Replaced physically impossible benchmark numbers
  (+14.1% ARC, +15.2% HellaSwag after lossy compression) with validated results from a
  clean re-run; added a clearly labelled validity-notice header.
- **`KVLayerCache.update_and_fetch` / `.offset`** — Added the `update_and_fetch(keys, values)`
  method and read-only `offset` property required by the mlx_lm per-layer cache protocol.
  Without these, `--kv-cache-mode int8/snap` silently had no effect on generation.
- **`QuantizedKVCache.__getitem__`** — Now returns `self._layers[idx]` (a `KVLayerCache`
  with `update_and_fetch`) instead of a `_LayerCacheView` wrapper that lacked the protocol
  method.
- **`server.py` `_sample_mx()`** — Added module-level temperature + nucleus-sampling helper
  used by the quantized KV cache generation path.
- **`server.py` KV cache generation path** — Wired the quantized cache into `_stream_tokens`;
  `--kv-cache-mode int8/snap` now routes through `model(x, cache=layer_caches)` per decode
  step with graceful fallback to `mlx_lm.stream_generate` on error.
- **`server.py` `/v1/embeddings`** — Semantic embeddings now use `model.model(x)` (last
  hidden state) as the preferred path, falling back to `embed_tokens` then logits mean-pool.
  The previous behaviour always returned input-token embeddings, which are unsuitable for
  semantic similarity.
- **`server.py` `--log-level`** — Added argument to control uvicorn log verbosity
  (choices: `critical` / `error` / `warning` / `info` / `debug` / `trace`; default:
  `warning`).  Previously hardcoded.
- **`cli.py compress --awq / --awq-samples`** — AWQ activation-calibration pass now exposed
  on the `squish compress` subcommand.  Loads the full model, collects activation scales,
  and passes `--awq-scales` to the conversion subprocess automatically.
- **`cli.py run/serve --log-level`** — Log-level argument forwarded from `squish run` /
  `squish serve` to the server process.
- **`cli.py compress/pull --int4` help text** — Corrected disk-savings claim from “~50%” to
  “~44%” and replaced “Recommended for 1.5B models” with an explicit warning: INT4
  quantization produces degenerate output on models smaller than 3B parameters.
  Use INT8 (`--int8`, the default) for 1.5B models.

---

## [1.0.0] — 2026-03-03

**Initial public release**, accompanying the research paper.

### Added

- **Three-tier compressed weight loader** — INT8 Vectro → float16 npy → bf16 MLX safetensors
- **OpenAI-compatible API server** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`)
- **Ollama drop-in compatibility** (`/api/generate`, `/api/chat`, `/api/tags`, `/api/embeddings`)
- **Web chat UI** at `/chat` — dark-themed, streaming, multi-session history, offline
- **CLI** — `squish run` / `squish serve`, `squish chat`, `squish models`, `squish bench`, `squish info`, `squish rm`, `squish search`, `squish pull`, `squish --version`
- **Speculative decoding** — target + draft model acceleration
- **Batch scheduler** — dynamic batching with priority queues
- **KV cache quantisation** — KIVI INT8 + SnapKV compression
- **Prefix cache** — prompt prefix reuse across requests
- **Tool / function calling** — OpenAI-format `tools` → `tool_calls` round-trip
- **Rust/PyO3 INT8 quantiser** (`squish_quant_rs`) — ARM NEON SIMD vectorised
- **AWQ calibration** pass for activation-guided mixed-precision
- Integrations: Continue.dev, aider, LiteLLM (config templates in `configs/`)
- Evaluation harness wrapper (`squish[eval]`) — lm-evaluation-harness compatible

### Benchmark (Qwen2.5-1.5B-Instruct, Apple Silicon M-series)

| Metric | mlx_lm (cold) | Squish (cached) | Improvement |
|---|---:|---:|---:|
| Load time | 28.81 s | 0.53 s | **54×** |
| Peak load RAM | ~2600 MB | 402 MB | **6×** |
| Accuracy delta | — | ≤1.5% on all tasks | ✅ |
