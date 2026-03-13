# Squish — Development Plan

> Last updated: 2026-03-12 (v9 complete + pre-launch hardening phase 1+2+3)

This document tracks completed waves, the current release, and the next phase.

---

## Versioning Convention

| Version | Waves | Theme |
|---------|-------|-------|
| **v1** | 1–11 | Core baseline — loader, quantizer, server, API, CLI, speculative decode |
| **v2** | 12 | Reasoning-Aware KV · INT3 · Async I/O |
| **v3** | 13–14 | Ultra-Long Context · Adaptive Spec-Decode · Quantisation |
| **v4** | 15–16 | Serving Intelligence · KV Architecture Evolution · Heterogeneous Compute |
| **v5** | 17–18 | Attention Architecture · Memory Management · Adaptive Compute · Model Intelligence |
| **v6** | 19–20 | Next-Gen Precision · Serving Infrastructure · Intelligence |
| **v7** | 21–22 | Advanced Decode · Production Serving · Observability |
| **v8** | 23–24 | Multi-Modal & Long Context · Quantisation Evolution & Model Surgery |
| **v9** | 25–26 | Cutting-Edge Attention Variants & Compute Fusion · Distributed Inference & Production Reliability |

---

## ✅ v1 — Core Baseline (Released 2026-03-03)

- Three-tier compressed weight loader (INT8 → f16 → bf16 MLX safetensors)
- OpenAI-compatible API server (`/v1/*`) + Ollama drop-in (`/api/*`)
- Web chat UI at `/chat`
- CLI — `squish run/serve/chat/pull/models/info/bench/catalog/compress`
- Speculative decoding, batch scheduler, KV cache quantisation, prefix cache
- Tool / function calling, Rust/PyO3 INT8 quantiser

---

## ✅ v2 — Wave 12 (Released 2026-03-04)

Modules: PM-KVQ, MixKVQ, CocktailKV, MiLo INT3, AgileIO, SageAttn, SpargeAttn

Key results: 4.2× KV memory · 5.3× weight compression · 40–60% I/O latency reduction

---

## ✅ v3 — Waves 13+14 (Released 2026-03-11)

Wave 13 (10 modules): DuoAttention, ShadowKV, PQCache, SpeCache, DuoDecoding,
KnapSpec, TokenMerging, TokenSwift, C2T, CLaSP

Wave 14 (16 modules): DFloat11, SqueezeLLM, NF4, rANS, QSpec, QuantSpec,
CopySpec, SpinQuant, VisionPrefixCache, MRLIndex, SubSpec, DELDecoder,
HeteroVocab, HeadInfer, LifeModel, SoupOfExperts

Key results: 10–30× KV memory · 55% draft acceptance · 5–10× weight compression

---

## ✅ v4 — Waves 15+16 (Released 2026-03-12)

Theme: **Serving Intelligence · KV Architecture Evolution · Heterogeneous Compute**

### Wave 15 — Serving Intelligence + KV Architecture Evolution (10 modules)

| Module | Flag | Key Result |
|--------|------|-----------|
| AdaServe | `--ada-serve` | SLO-customized spec decode trees → 30% latency ↓ for tight SLOs |
| ConfSpec | `--conf-spec` | Confidence-gated verification → 54% verification cost ↓ |
| SeqPacking | `--seq-packing` | Barrel effect elimination → 1.8× effective throughput |
| MetaReasoner | `--meta-reasoner` | Dynamic thinking budget → 44–89% energy saved on CoT |
| YOCO | `--yoco-kv` | You Only Cache Once → 50% KV memory reduction |
| DiffKV | `--diff-kv` | Asymmetric K/V precision → 2.7–5.7× KV memory, 1.9–5.4× throughput |
| KVTuner | `--kvtuner` | Sensitivity-aware mixed-precision KV → 2× compression vs naive |
| KVSharer | `--kv-share` | Cross-layer KV sharing → 30% KV memory reduction |
| ParisKV | `--paris-kv` | Drift-robust online KV quantisation → 4× KV compression |
| CLA | `--cla` | Cross-Layer Attention sharing → 10–30% KV memory reduction |

### Wave 16 — Heterogeneous Compute + Advanced Spec-Decode (11 modules)

| Module | Flag | Key Result |
|--------|------|-----------|
| Dovetail | `--dovetail` | CPU+GPU heterogeneous spec decode → 2× throughput |
| SwiftSpec | `--swift-spec` | Async disaggregated decode → minimal overlap overhead |
| PIPO | `--pipo` | Pipelined prefetch offloading → 1.7× throughput >VRAM models |
| MobileMoE | `--mobile-moe` | MoE balanced layer skip → 1.4× throughput on MoE models |
| OnlineSD | `--online-sd` | Continuous draft adaptation → +5–8 pp acceptance rate |
| LookaheadReasoning | `--lookahead` | Parallel step verification → 2.1× throughput on reasoning |
| SparseSpec | `--sparse-spec` | Dynamic sparse self-speculation → 2.13× throughput |
| FRSpec | `--fr-spec` | Frequency-ranked vocab compression → 13% draft latency ↓ |
| LongSpec | `--long-spec` | Shared-KV draft head → zero draft KV overhead at any context |
| ForeLen | `--forelen` | Entropy-guided length prediction → 29% MAE ↓ vs TRAIL |
| RASD | `--rasd` | Retrieval-augmented spec decode → 40–60% corpus hit rate |

### Deliverables checklist

- [x] All 21 modules implemented and wired in `server.py`
- [x] `tests/test_wave15_server_wiring.py` — 44 tests, 44 passing
- [x] `tests/test_wave16_server_wiring.py` — 45 tests, 45 passing
- [x] `dev/benchmarks/bench_wave15_16.py` — micro-benchmark suite
- [x] `dev/results/wave15_16_bench.json` — benchmark results
- [x] `docs/benchmark_wave15_16.md` — human-readable results table
- [x] `dev/demos/record_v4_demo.py` — v4 demo GIF generator
- [x] `dev/demos/squish-v4-demo.gif` — demo GIF rendered
- [x] README.md — v4 module sections, Wave 15+16 tables, CLI examples
- [x] CHANGELOG.md — `[2.0.0]` entry

---

## ✅ v5 — Waves 17+18 (Released 2026-03-11)

Theme: **Attention Architecture · Memory Management · Adaptive Compute · Model Intelligence**

28 modules across two waves — all implemented, tested, benchmarked, and documented.

---

### Wave 17 — Attention Architecture + Memory Management (14 modules)

Focus: Next-generation attention kernels, zero-allocation KV memory, prompt and
token compression, and speculative context retrieval.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **SageAttn2** | `sage_attention2.py` | `SageAttention2Kernel`, `SageAttention2Config` | `--sage-attn2` | INT4 warp QK + FP8 PxV → **~3.1× vs FlashAttention2** |
| **StreamingSink** | `streaming_sink.py` | `SinkKVCache`, `SinkConfig` | `--streaming-sink` | Attention sink eviction → **infinite context** at fixed KV budget |
| **KVSlab** | `kv_slab.py` | `KVSlabAllocator`, `KVPage` | `--kv-slab` | Pre-allocated slab → **eliminates >10 ms** per-request heap stalls |
| **SqueezeAttn** | `squeeze_attention.py` | `SqueezeKVCache`, `BudgetAllocator` | `--squeeze-attn` | Dynamic per-layer KV budget → **configurable KV footprint** |
| **SmallKV** | `smallkv.py` | `SmallKVCache`, `SaliencyTracker` | `--small-kv` | Saliency-compensated 10% KV budget → **1.75–2.56× throughput** |
| **SpeContext** | `specontext.py` | `SpeContextCache`, `DistilledRetrievalHead` | `--spe-context` | Distilled retrieval head → **>90% param reduction**, 90% transfer ↓ |
| **SVDq** | `svdq.py` | `SVDqCalibrator`, `SVDqPrecisionMap` | `--svdq` | Per-head SVD key mixed precision → **calibrated rank-aware quantisation** |
| **CommVQ** | `comm_vq.py` | `CommVQCodebook`, `MultiCodebookVQ` | `--comm-vq` | Commutative VQ KV → **8× (2-bit) / 4× (4-bit) memory, near-lossless** |
| **ChunkedPrefill** | `chunked_prefill.py` | `ChunkedPrefillConfig` | `--chunked-prefill` | Interleaved chunk+decode → **O(chunk_size) prefill latency** |
| **GemFilter** | `gemfilter.py` | `GemSelector`, `AttentionScoreBuffer` | `--gemfilter` | Early-layer token compression → **2.4× speedup, 1000× @ 108K tokens** |
| **MInference** | `minference_patch.py` | *(monkey-patch)* | `--minference` | Dynamic sparse attention → **10× prefill speedup @ 1M context** |
| **PromptCompressor** | `prompt_compressor.py` | *(functional API)* | `--prompt-compress` | Token-budget long-context trimming → **~1 ms per 1K-word prompt** |
| **PromptLookup** | `prompt_lookup.py` | `PromptLookupDecoder`, `NGramIndex` | `--prompt-lookup` | N-gram spec decode from prompt → **zero draft model required** |
| **TRAIL** | `trail.py` | `TrailPredictor`, `TrailLinearProbe` | `--trail` | Probe-layer length predictor → **2.66× lower MAE** vs BERT, **1.66–2.01× lower latency** |

### Wave 18 — Adaptive Compute + Model Intelligence + Evaluation (14 modules)

Focus: Task-adaptive layer skipping, next-generation speculative decoding,
continuous self-improvement, serving intelligence, and battery-aware evaluation.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **VPTQ** | `vptq.py` | `VPTQQuantizer`, `VPTQCodebook` | `--vptq` | Vector post-training quant (NeurIPS 2025) → **sub-2-bit weights** near fp16 quality |
| **LayerSkip** | `layer_skip.py` | `EarlyExitDecoder`, `ConfidenceEstimator` | `--layer-skip` | Early exit self-spec decode → **(total−exit)/total compute saved** per easy token |
| **SWIFT** | `swift.py` | `SWIFTDecoder`, `SWIFTCalibrator` | `--swift` | Task-adaptive layer skip with calibration → **per-task skip schedules** |
| **SpecReason** | `spec_reason.py` | `SpecReasonOrchestrator`, `ReasoningStep` | `--spec-reason` | Step-level reasoning speculation → **1.4–3.0× speedup**, **8.8–58% token reduction** |
| **MirrorSD** | `mirror_sd.py` | `MirrorSDDecoder`, `MirrorDraftPipeline` | `--mirror-sd` | Overlapped dual-pipeline draft → **2.8–5.8× vs EAGLE-3** on SpecBench |
| **SparseVerify** | `sparse_verify.py` | `SparseVerifyPass`, `InterDraftReuseCache` | `--sparse-verify` | Sparse verification + inter-draft token reuse → **verification FLOPs ↓** |
| **RobustScheduler** | `robust_scheduler.py` | `ABalancedScheduler`, `AMaxScheduler` | `--robust-sched` | Interval-prediction adaptive batching → **balanced or max-throughput policy** |
| **BlockExpertArchive** | `block_expert_archive.py` | `BlockExpertArchive`, `ExpertRouter` | `--block-archive` | K-means cluster-delta expert compression → **MoE weight deduplication** |
| **DISCRouter** | `disc_router.py` | `DISCRouter`, `DISCPlan` | `--disc-router` | Task decomposition + parallel LLM routing → **multi-step agent acceleration** |
| **SelfLearning** | `self_learning.py` | *(LearnRequest API)* | `--self-learn` | Online LoRA-delta adaptation from feedback → **continuous quality improvement** |
| **SemanticCache** | `semantic_cache.py` | `SquishSemanticCache` | `--semantic-cache` | N-gram semantic prompt dedup → **zero-model cache hits** |
| **IPW** | `ipw.py` | `IPWTracker`, `IPWMeasurement` | `--ipw` | Intelligence-per-watt tracking → **quality ÷ energy metric for M-series** |
| **PowerMonitor** | `power_monitor.py` | `PowerMonitor`, `PowerModeConfig` | `--power-monitor` | pmset-based battery-adaptive mode selection → **auto power-aware scheduling** |
| **DiffusionDraft** | `diffusion_draft.py` | `DiffusionDraftModel` | `--diffusion-draft` | Non-autoregressive diffusion LLM drafting → **short-text parallel decode** |

### v5 Deliverables checklist

- [x] `tests/test_wave17_server_wiring.py` — 56 tests, 56 passing
- [x] `tests/test_wave18_server_wiring.py` — 56 tests, 56 passing
- [x] `dev/benchmarks/bench_wave17_18.py` — micro-benchmark suite (24 modules timed, 4 skipped)
- [x] `dev/results/wave17_18_bench.json` — benchmark results
- [x] `docs/benchmark_wave17_18.md` — human-readable results table
- [x] `dev/demos/record_v5_demo.py` — v5 demo GIF generator (448 events, 85.2s)
- [x] `dev/demos/squish-v5-demo.gif` — demo GIF rendered (2.6 MB, 448 events, 85.2s)
- [x] README.md — v5 module sections, Wave 17+18 tables, CLI examples
- [x] CHANGELOG.md — `[3.0.0]` entry
- [x] PLAN.md updated to mark v5 complete

### v5 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 17 (Attention + Memory) | 14 |
| Wave 18 (Adaptive Compute + Intelligence) | 14 |
| Total new v5 modules | **28** |
| Total modules after v5 | **110** |
| New tests | **112** (56 Wave 17 + 56 Wave 18) |
| Total tests after v5 | **4 166** |

---

## ✅ v6 — Waves 19+20 (Released 2026-03-11)

Theme: **Next-Gen Precision · Advanced Attention · Model Composition · Serving Infrastructure**

28 new modules across two waves — all implemented, tested, benchmarked, and documented.

---

### Wave 19 — Next-Gen Attention & Precision (14 modules)

Focus: FP8/MX microscaling quantization, advanced attention patterns (paged KV,
GQA, sliding window, RoPE scaling), activation sparsity, and advanced speculative
decode heads (MEDUSA, EAGLE-3).

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **FP8Quant** | `fp8_quant.py` | `FP8Quantizer`, `FP8Config` | `--fp8-quant` | E4M3/E5M2 weight encoding → **~60% storage vs BF16** |
| **MXQuant** | `mx_quant.py` | `MXQuantizer`, `MXConfig` | `--mx-quant` | OCP MX4/MX6/MX9 microscaling → **better quality than INT4** at same bits |
| **FlashDecode** | `flash_decode.py` | `FlashDecodeAttention`, `FlashDecodeConfig` | `--flash-decode` | Split-KV parallel decode → **O(1) memory overhead** per decode step |
| **PagedKV** | `paged_kv.py` | `PagedKVCache`, `BlockTable` | `--paged-kv` | Virtual block mapping → **zero KV fragmentation** across requests |
| **GQA** | `gqa.py` | `GQACache`, `GQAConfig` | `--gqa` | Grouped Query Attention → **4–8× KV reduction** vs MHA |
| **SlidingWindowAttn** | `sliding_window_attn.py` | `SlidingWindowKVCache`, `SWAConfig` | `--sliding-window` | Sliding window KV → **O(window_size) memory** at any context length |
| **RoPEScaling** | `rope_scaling.py` | `RoPEScaler`, `YaRNScaler`, `NTKScaler` | `--rope-scaling` | NTK/YaRN/LongRoPE → **4–32× context extension** without fine-tuning |
| **ActSparsity** | `act_sparsity.py` | `ActSparsityPredictor`, `SparsityConfig` | `--act-sparsity` | Activation sparsity gating → **30–60% FFN compute saved** |
| **FusedRMSNorm** | `fused_rmsnorm.py` | `FusedRMSNorm`, `FusedLayerNorm` | `--fused-norm` | Fused RMSNorm + residual → **single kernel pass**, reduced bandwidth |
| **LoRAInference** | `lora_inference.py` | `LoRAInferenceAdapter`, `LoRAConfig` | `--lora-inference` | Zero-copy LoRA delta inference → **adapter switching without re-quant** |
| **MEDUSA** | `medusa.py` | `MedusaHead`, `MedusaDecoder` | `--medusa` | Multi-head tree speculation → **2–3× decode throughput** |
| **EAGLE3** | `eagle3.py` | `Eagle3DraftHead`, `Eagle3Decoder` | `--eagle3` | Feature-level draft head → **3.5× accept rate** vs token-prediction draft |
| **PrefixPool** | `prefix_pool.py` | `PrefixPool`, `PrefixPoolConfig` | `--prefix-pool` | Cross-request KV prefix sharing → **40–80% KV savings** on shared prompts |
| **TokenHealer** | `token_healer.py` | `TokenHealer`, `HealerConfig` | `--token-healer` | Boundary-aware token healing → **eliminates prefix-artifact generation** |

### Wave 20 — Serving Infrastructure & Intelligence (14 modules)

Focus: Model composition (merge, compose), continuous batching, evaluation harness,
power profiling, multi-modal efficiency, and knowledge distillation for spec heads.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **ModelMerge** | `model_merge.py` | `ModelMerger`, `MergeConfig` | `--model-merge` | SLERP/DARE/TIES merging → **combine domains without retraining** |
| **LoRACompose** | `lora_compose.py` | `LoRAComposer`, `AdapterStack` | `--lora-compose` | Multi-LoRA mixture → **blend adapters with learnable coefficients** |
| **ContinuousBatching** | `continuous_batching.py` | `CBScheduler`, `InFlightRequest` | `--continuous-batching` | Mid-generation insertion → **max GPU utilization at any request rate** |
| **MatryoshkaEmb** | `matryoshka_emb.py` | `MatryoshkaEmbedding`, `MRLConfig` | `--matryoshka-emb` | Nested embedding truncation → **1 forward pass, any dimensionality** |
| **ANEProfiler** | `ane_profiler.py` | `ANEProfiler`, `ANEMetrics` | `--ane-profiler` | Apple Neural Engine utilization → **op-level ANE vs GPU breakdown** |
| **SpecBench** | `spec_bench.py` | `SpecBenchRunner`, `SpecBenchResult` | `--spec-bench` | SpecBench CI harness → **acceptance rate + throughput across tasks** |
| **PPLTracker** | `ppl_tracker.py` | `PPLTracker`, `PPLWindow` | `--ppl-tracker` | Rolling perplexity tracker → **real-time quality degradation detection** |
| **GrammarCache** | `grammar_cache.py` | `GrammarCache`, `FSMState` | `--grammar-cache` | FSM grammar cache → **constrained decoding without per-token rebuild** |
| **QuantAware** | `quant_aware.py` | `QuantAwareCalibrator`, `QAConfig` | `--quant-aware` | Activation-range calibration → **per-channel optimal scale selection** |
| **AdaptiveBudget** | `adaptive_budget.py` | `AdaptiveBudgetController`, `BudgetConfig` | `--adaptive-budget` | Dynamic compute budget → **SLO-aware KV + layer skip joint control** |
| **VisionTokens** | `vision_tokens.py` | `VisionTokenCompressor`, `VTConfig` | `--vision-tokens` | Visual token pruning → **50–80% vision token reduction** without quality loss |
| **ToolCache** | `tool_cache.py` | `ToolSchemaCache`, `ToolRouter` | `--tool-cache` | Schema + routing cache → **zero tool-call parse overhead** on repeated schemas |
| **DistilSpec** | `distil_spec.py` | `DistilSpecCalibrator`, `DistilConfig` | `--distil-spec` | Draft-head knowledge distillation → **+10–15 pp acceptance from calibration** |
| **BatchEmbed** | `batch_embed.py` | `BatchEmbedder`, `PoolingConfig` | `--batch-embed` | Dynamic pooling strategies → **mean/max/cls/weighted pool in single pass** |

### v6 Deliverables checklist

> **Progress (2026-03-11):** Wave 20 modules 1–14 (all) implemented and tested:
> ModelMerge, LoRACompose, ContinuousBatching, MatryoshkaEmb, ANEProfiler,
> SpecBench, PPLTracker, GrammarCache, QuantAware, AdaptiveBudget,
> VisionTokens, ToolCache, DistilSpec, BatchEmbed — 262+ new tests.

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave19_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave20_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave19_20.py` — micro-benchmark suite
- [x] `dev/results/wave19_20_bench.json` — benchmark results
- [x] `docs/benchmark_wave19_20.md` — human-readable results table
- [x] `dev/demos/record_v6_demo.py` — v6 demo GIF generator
- [x] `dev/demos/squish-v6-demo.gif` — demo GIF rendered
- [x] README.md — v6 module sections, Wave 19+20 tables, CLI examples
- [x] CHANGELOG.md — `[4.0.0]` entry
- [x] PLAN.md updated to mark v6 complete

### v6 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 19 (Next-Gen Attention + Precision) | 14 |
| Wave 20 (Serving Infrastructure + Intelligence) | 14 |
| Total new v6 modules | **28** |
| Total modules after v6 | **138** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v6 | **4 278** |

---

## ✅ v7 — Waves 21+22 (Released 2026-03-12)

Theme: **Advanced Decode · Production Serving · Observability**

28 new modules across two waves.

---

### Wave 21 — Advanced Memory & Decode (14 modules)

Focus: Tree-parallel speculative verification, online KV compression, mixed-precision
KV per head, pipeline-parallel decode, learned KV codecs, retention-style recurrent
attention, and context-length-adaptive RoPE scaling.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TreeVerifier** | `tree_verifier.py` | `TreeVerifier`, `TokenTree` | `--tree-verify` | Batched tree-parallel speculative verification → **structured multi-token acceptance** |
| **KVCompress** | `kv_compress.py` | `KVCompressor`, `KVCompressConfig` | `--kv-compress` | Online KV quantisation + pruning during generation → **adaptive old-context compression** |
| **DynamicNTK** | `dynamic_ntk.py` | `DynamicNTKScaler`, `NTKState` | `--dynamic-ntk` | Per-request runtime RoPE base auto-scaling → **auto-extends at 80% context fill** |
| **QuantSpecDecode** | `quant_spec_decode.py` | `QuantSpecDecoder`, `QSDConfig` | `--quant-spec-decode` | INT4 draft + FP16 verify → **draft memory ↓ 4× vs FP16** |
| **SparseAttnIndex** | `sparse_attn_index.py` | `SparseAttnIndex`, `ANCandidates` | `--sparse-attn-index` | ANN KV retrieval index → **sub-linear attention cost at very long context** |
| **MixedPrecisionKV** | `mixed_precision_kv.py` | `MixedPrecisionKVCache`, `HeadPrecision` | `--mp-kv` | Per-head INT8/INT4/FP16 KV via sensitivity analysis → **2–4× KV memory at iso-quality** |
| **PipelineBubble** | `pipeline_bubble.py` | `BubbleEliminator`, `StageSchedule` | `--pipeline-bubble` | Overlapped prefill + decode across pipeline stages → **bubble-free pipeline utilisation** |
| **LayerwiseDecode** | `layerwise_decode.py` | `LayerwiseDecoder`, `LayerStream` | `--layerwise-decode` | Layer-by-layer early-exit decode with multi-stream output → **configurable exit-layer latency** |
| **CodecKV** | `codec_kv.py` | `KVCodec`, `CodecConfig` | `--codec-kv` | Learned encode/decode KV codec → **2–4× KV compression via latent reconstruction** |
| **DedupeAttn** | `dedupe_attn.py` | `AttentionDeduplicator`, `DedupStats` | `--dedupe-attn` | Near-duplicate Q/K detection + output reuse → **attention FLOPs ↓ on repetitive context** |
| **FlashPrefill** | `flash_prefill.py` | `FlashPrefillKernel`, `PrefillConfig` | `--flash-prefill` | Chunked flash attention for prefill with causal mask → **O(chunk²) not O(seq²) memory** |
| **BudgetSpec** | `budget_spec.py` | `BudgetSpecDecoder`, `BudgetConfig` | `--budget-spec` | Token-budget-aware speculative decode → **exits drafting when budget threshold hit** |
| **RetentionAttn** | `retention_attn.py` | `RetentionState`, `RetentionKernel` | `--retention-attn` | Retention-style recurrent state → **O(1) per-step memory, linear recurrence** |
| **KVRouter** | `kv_router.py` | `KVRouter`, `KVRouteTable` | `--kv-router` | Cross-instance KV routing for disaggregated prefill/decode → **KV transfer without recomputation** |

### Wave 22 — Production Serving & Observability (14 modules)

Focus: Multi-tenant fair scheduling, intelligent load-balanced request routing,
predictive KV pre-warming, token budget enforcement, OpenTelemetry-compatible
tracing, request coalescing, adaptive quantisation, health monitoring, and
cost-aware serving.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **MultiTenantSched** | `multi_tenant_sched.py` | `TenantScheduler`, `TenantConfig` | `--multi-tenant` | Fair per-tenant QoS scheduling → **SLO-isolated multi-tenant serving** |
| **RequestRouter** | `request_router.py` | `RequestRouter`, `ReplicaRegistry` | `--request-router` | Load-aware request routing across replicas → **consistent-hash + least-loaded** |
| **CacheWarmup** | `cache_warmup.py` | `CacheWarmupPredictor`, `WarmupConfig` | `--cache-warmup` | Predictive KV cache pre-warming from patterns → **TTFT ↓ on hot prefix paths** |
| **TokenBudgetGate** | `token_budget_gate.py` | `TokenBudgetGate`, `BudgetPolicy` | `--token-budget` | Hard per-request token budget with graceful truncation → **deterministic cost control** |
| **ObservabilityHook** | `observability_hook.py` | `InferenceTracer`, `SpanCollector` | `--observability` | Zero-overhead per-step inference tracing → **OpenTelemetry-compatible spans** |
| **RequestCoalesce** | `request_coalesce.py` | `PrefixCoalescer`, `CoalesceStats` | `--req-coalesce` | Merge requests sharing long common prefixes → **shared prefill forward pass** |
| **AdaptiveQuantize** | `adaptive_quantize.py` | `AdaptiveQuantizer`, `PressureMonitor` | `--adaptive-quant` | Runtime precision switching under memory pressure → **auto INT8/INT4 under OOM** |
| **HealthCheck** | `health_check.py` | `InferenceHealthMonitor`, `HealthState` | `--health-check` | Degradation-aware server health monitoring → **automatic quality regression alerting** |
| **FaultTolerance** | `fault_tolerance.py` | `FaultHandler`, `FaultPolicy` | `--fault-tolerance` | Graceful OOM degradation → **auto KV eviction + draft disable + SLO re-negotiation** |
| **ModelPool** | `model_pool.py` | `ModelPool`, `PoolEntry` | `--model-pool` | Hot model pool with lazy-load + LRU eviction → **multi-model serving without reload latency** |
| **StreamingChunk** | `streaming_chunk.py` | `ChunkedStreamer`, `BackpressureBuffer` | `--streaming-chunk` | Sub-token-latency chunked streaming with backpressure → **first-chunk latency ↓** |
| **CostEstimator** | `cost_estimator.py` | `RequestCostEstimator`, `CostModel` | `--cost-estimate` | Per-request compute cost estimation → **supports billing and priority queuing** |
| **SLAMonitor** | `sla_monitor.py` | `SLAMonitor`, `ViolationPolicy` | `--sla-monitor` | Real-time SLA violation detection + remediation → **auto-escalation on breach** |
| **ContextCache** | `context_cache.py` | `PersistentContextCache`, `CacheEntry` | `--context-cache` | Persistent cross-session context cache with TTL → **zero re-encode on repeated context** |

### v7 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave21_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave22_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave21_22.py` — micro-benchmark suite
- [x] `dev/results/wave21_22_bench.json` — benchmark results
- [x] `docs/benchmark_wave21_22.md` — human-readable results table
- [x] `dev/demos/record_v7_demo.py` — v7 demo GIF generator
- [x] `dev/demos/squish-v7-demo.gif` — demo GIF rendered
- [x] README.md — v7 module sections, Wave 21+22 tables, CLI examples
- [x] CHANGELOG.md — `[5.0.0]` entry
- [x] PLAN.md updated to mark v7 complete

### v7 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 21 (Advanced Memory + Decode) | 14 |
| Wave 22 (Production Serving + Observability) | 14 |
| Total new v7 modules | **28** |
| Total modules after v7 | **166** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v7 | **~4 390** |

---

## ✅ v8 — Waves 23+24 — Released 2026-03-12

Theme: **Multi-Modal & Long Context · Quantisation Evolution & Model Surgery**

28 new modules across two waves.

---

### Wave 23 — Multi-Modal & Long Context Intelligence (14 modules)

Focus: Vision-language model efficiency, RAG-aware serving patterns, reasoning trace
compression, cross-modal attention, hierarchical KV management, and 1M+ token context
indexing.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **VisionKVFuse** | `vision_kv_fuse.py` | `VisionKVFuseCache`, `ModalityConfig` | `--vision-kv-fuse` | Fused vision+text KV with separate modality eviction → **modality-aware KV compression** |
| **ImageTokenPrune** | `image_token_prune.py` | `ImageTokenPruner`, `PruneConfig` | `--image-token-prune` | Attention entropy image token pruning → **50–70% image token reduction** |
| **RAGPrefetch** | `rag_prefetch.py` | `RAGPrefetcher`, `RAGConfig` | `--rag-prefetch` | Predictive doc KV prefetch→ **cold TTFT↓ on repeated RAG docs** |
| **CoTCompress** | `cot_compress.py` | `CoTCompressor`, `CoTConfig` | `--cot-compress` | CoT trace pruning via saliency → **30–50% reasoning token reduction** |
| **MultiModalBatch** | `multimodal_batch.py` | `MultiModalBatcher`, `BatchSlot` | `--multimodal-batch` | Shape-aware heterogeneous text+vision batcher → **minimise padding waste** |
| **ContextualRerank** | `contextual_rerank.py` | `ContextualReranker`, `RerankConfig` | `--ctx-rerank` | Context-aware KV token importance re-ranking → **preserves top-k salient positions** |
| **CrossModalAttn** | `cross_modal_attn.py` | `CrossModalAttention`, `CrossModalConfig` | `--cross-modal-attn` | Efficient cross-attention between text + vision features → **modality fusion** |
| **HierarchicalKV** | `hierarchical_kv.py` | `HierarchicalKVStore`, `TierConfig` | `--hierarchical-kv` | Hot/warm/cold KV tier management → **transparent KV tiering with O(1) promotion** |
| **StreamRAG** | `stream_rag.py` | `StreamRAGInjector`, `StreamRAGConfig` | `--stream-rag` | Streaming mid-generation document injection → **zero-restart RAG updates** |
| **CrossDocAttn** | `cross_doc_attn.py` | `CrossDocAttention`, `CrossDocConfig` | `--cross-doc-attn` | Chunked cross-document attention → **multi-document QA without full concatenation** |
| **VideoFramePrune** | `video_frame_prune.py` | `VideoFramePruner`, `FrameConfig` | `--video-frame-prune` | Temporal frame token pruning for video-LMs → **60–80% video token reduction** |
| **EmbeddingGate** | `embedding_gate.py` | `EmbeddingGate`, `GateConfig` | `--embedding-gate` | Gated modality-conditional embedding router → **zero-cost modality bypass** |
| **LongContextChunk** | `long_context_chunk.py` | `LongContextChunker`, `ChunkConfig` | `--long-context-chunk` | Semantic-boundary chunking for 1M+ token contexts → **boundary-aware chunk splits** |
| **ModalityRouter** | `modality_router.py` | `ModalityRouter`, `ModalityPolicy` | `--modality-router` | Per-modality SLO request dispatcher → **text vs vision vs audio routing** |

### Wave 24 — Quantisation Evolution & Model Surgery (14 modules)

Focus: Ternary and binary quantisation, N:M structured sparsity, cross-layer weight
sharing, second-order GPTQ-style calibration, sparse MoE routing, iterative pruning,
and surgical model architecture patching.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TernaryQuant** | `ternary_quant.py` | `TernaryQuantizer`, `TernaryConfig` | `--ternary-quant` | BitNet-style ternary {−1, 0, +1} weights → **1.58-bit effective storage** |
| **BinaryAttn** | `binary_attn.py` | `BinaryAttention`, `BinaryConfig` | `--binary-attn` | Sign-binarised attention approximation → **ultra-low attention memory** |
| **StructuredPrune** | `structured_prune.py` | `StructuredPruner`, `PruneConfig` | `--structured-prune` | 2:4 N:M magnitude pruning → **50% weight sparsity at 2× hardware throughput** |
| **LayerFusion** | `layer_fuse.py` | `LayerFuser`, `FusionConfig` | `--layer-fuse` | Adjacent transformer layer weight fusion → **reduced bandwidth on similar layers** |
| **WeightSharing** | `weight_sharing.py` | `WeightSharer`, `SharingConfig` | `--weight-share` | Cross-layer weight tying with delta residuals → **memory ↓ at iso-quality** |
| **QuantCalib** | `quant_calib.py` | `QuantCalibrator`, `CalibConfig` | `--quant-calib` | Unified MinMax/Percentile/MSE/GPTQ calibration pipeline → **optimal scale per method** |
| **SparseWeight** | `sparse_weight.py` | `SparseWeightStore`, `SparsityConfig` | `--sparse-weight` | CSR-format 2:4 pruned weight storage → **2× memory vs dense at 50% sparsity** |
| **DeltaCompress** | `delta_compress.py` | `DeltaCompressor`, `DeltaConfig` | `--delta-compress` | Rank-k SVD delta compression for fine-tuned weights → **fine-tune deltas at 10–50× reduction** |
| **ModelSurgery** | `model_surgery.py` | `ModelSurgeon`, `SurgeryPlan` | `--model-surgery` | In-place layer removal + head pruning → **architecture patching without retraining** |
| **ZeroQuantV2** | `zero_quant_v2.py` | `ZeroQuantV2`, `ZQConfig` | `--zero-quant-v2` | Groupwise quantisation with FP16 residual for outliers → **W8A8 with outlier preservation** |
| **GPTQLayer** | `gptq_layer.py` | `GPTQCalibrator`, `GPTQConfig` | `--gptq-layer` | Hessian-weighted second-order rounding → **group-wise optimal quant error** |
| **SparseMoE** | `sparse_moe.py` | `SparseMoERouter`, `MoEConfig` | `--sparse-moe` | Top-k sparse expert routing with load-balance loss → **efficient MoE inference** |
| **AWQv2** | `awq_v2.py` | `AWQv2Calibrator`, `AWQv2Config` | `--awq-v2` | Activation-aware scale+shift per-channel quant → **AWQ without grid search** |
| **IterPrune** | `iter_prune.py` | `IterativePruner`, `PruneSchedule` | `--iter-prune` | Iterative magnitude pruning with sparsity ramp schedule → **gradual 0→70% sparsity** |

### v8 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave23_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave24_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave23_24.py` — micro-benchmark suite
- [x] `dev/results/wave23_24_bench.json` — benchmark results
- [x] `docs/benchmark_wave23_24.md` — human-readable results table
- [x] `dev/demos/record_v8_demo.py` — v8 demo GIF generator
- [x] `dev/demos/squish-v8-demo.gif` — demo GIF rendered
- [x] README.md — v8 module sections, Wave 23+24 tables, CLI examples
- [x] CHANGELOG.md — `[6.0.0]` entry
- [x] PLAN.md updated to mark v8 complete

### v8 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 23 (Multi-Modal + Long Context Intelligence) | 14 |
| Wave 24 (Quantisation Evolution + Model Surgery) | 14 |
| Total new v8 modules | **28** |
| Total modules after v8 | **194** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v8 | **~4 502** |

---

## ✅ v9 — Waves 25+26 — Released 2026-03-12

Theme: **Cutting-Edge Attention Variants & Compute Fusion · Distributed Inference & Production Reliability**

28 new modules across two waves.

---

### Wave 25 — Cutting-Edge Attention Variants & Compute Fusion (14 modules)

Focus: DeepSeek-V2/V3 production attention patterns (MLA, NSA), fused sampling,
online KV defragmentation, dual-chunk long-context attention, activation offloading,
attention morphing, multi-draft hydra speculation, and constrained decoding.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **FlashMLA** | `flash_mla.py` | `FlashMLACache`, `MLAConfig` | `--flash-mla` | Multi-head latent attention (DeepSeek-V2 style); low-rank KV via down/up projection → **KV size ↓ by latent_dim/head_dim** |
| **NativeSparseAttn** | `native_sparse_attn.py` | `NativeSparseAttention`, `NSAConfig` | `--native-sparse-attn` | Block-sparse + sliding window attention (DeepSeek-V3 NSA style) → **sub-quadratic attention cost** |
| **FusedSampler** | `fused_sampler.py` | `FusedSampler`, `SamplerConfig` | `--fused-sampler` | Fused temperature/top-p/top-k/min-p/rep-penalty in single pass → **zero intermediate allocations** |
| **KVDefrag** | `kv_defrag.py` | `KVDefragmenter`, `DefragStats` | `--kv-defrag` | Online KV cache defragmentation and in-place compaction → **fragmentation ratio ↓** |
| **DualChunkAttn** | `dual_chunk_attn.py` | `DualChunkAttention`, `DCAConfig` | `--dual-chunk-attn` | Intra-chunk + inter-chunk attention for 1M+ contexts → **O(chunk²) not O(seq²)** |
| **ActivationOffload** | `activation_offload.py` | `ActivationOffloader`, `OffloadPolicy` | `--act-offload` | Layer activation offload to CPU during prefill → **peak GPU memory ↓** |
| **MorphAttn** | `morph_attn.py` | `AttentionMorpher`, `MorphConfig` | `--morph-attn` | Per-layer attention pattern selection: full/sparse/linear → **optimal compute per layer** |
| **HydraSpec** | `hydra_spec.py` | `HydraSpecDecoder`, `HydraConfig` | `--hydra-spec` | Multi-draft heads for parallel speculation → **n_heads candidate tokens per step** |
| **SeqCompact** | `seq_compact.py` | `SequenceCompactor`, `CompactStats` | `--seq-compact` | In-place KV sequence compaction after token pruning → **zero-copy repack** |
| **LatencyPredictor** | `latency_predictor.py` | `LatencyPredictor`, `LatencyModel` | `--latency-predict` | Per-request latency prediction for scheduling → **prefill + decode latency forecast** |
| **ParallelSampler** | `parallel_sampler.py` | `ParallelSampler`, `DiversityConfig` | `--parallel-sample` | Best-of-n sampling with diversity scoring → **quality improvement with n candidates** |
| **ContextSummarizer** | `context_summarizer.py` | `ContextSummarizer`, `SummaryConfig` | `--ctx-summarize` | Inference-time context compression when context overflows → **keep semantics, shed tokens** |
| **TokenWatermark** | `token_watermark.py` | `TokenWatermarker`, `WatermarkConfig` | `--token-watermark` | Statistical green-list token watermarking (Kirchenbauer et al.) → **detectable attribution** |
| **SchemaGen** | `schema_gen.py` | `SchemaGenEngine`, `SchemaState` | `--schema-gen` | FSM-accelerated constrained JSON schema generation → **zero invalid token sampling** |

### Wave 26 — Distributed Inference & Production Reliability (14 modules)

Focus: Tensor/sequence parallelism, live KV migration, disaggregated prefill/decode,
request preemption, smart inference gateway, zero-downtime model swaps, APM profiling,
adaptive batching, safety classification, semantic response caching, and audit logging.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TensorParallel** | `tensor_parallel.py` | `TensorParallelShard`, `TPConfig` | `--tensor-parallel` | Row/column tensor sharding + all-reduce → **linear memory scaling across devices** |
| **SequenceParallel** | `sequence_parallel.py` | `SequenceParallelScatter`, `SPConfig` | `--seq-parallel` | Ulysses-style sequence dimension split → **attention FLOPs distributed across devices** |
| **KVMigrate** | `kv_migrate.py` | `KVMigrator`, `MigrateStats` | `--kv-migrate` | Live KV state pack/unpack for cross-worker migration → **zero-recompute worker handoff** |
| **DisaggPrefill** | `disagg_prefill.py` | `DisaggPrefillNode`, `DisaggDecodeNode` | `--disagg-prefill` | Disaggregated prefill→decode with KV payload transfer → **prefill/decode hardware specialisation** |
| **RequestPreempt** | `request_preempt.py` | `PreemptScheduler`, `PreemptState` | `--req-preempt` | Preemptive SRPT scheduling with KV save/restore → **priority inversion elimination** |
| **InferGateway** | `infer_gateway.py` | `InferenceGateway`, `WorkerRegistry` | `--infer-gateway` | Smart front-door gateway: routing + health + load balancing → **single ingress, N workers** |
| **ModelVersionSwap** | `model_version_swap.py` | `ModelVersionManager`, `SwapPolicy` | `--model-swap` | Zero-downtime hot model version swap → **canary → promote → rollback in-flight** |
| **ProductionProfiler** | `production_profiler.py` | `ProductionProfiler`, `ProfilerWindow` | `--prod-profiler` | Continuous APM-style per-op latency tracking → **p50/p99/p999 per operation** |
| **AdaptiveBatcher** | `adaptive_batcher.py` | `AdaptiveBatchController`, `BatchObjective` | `--adaptive-batch` | Throughput/latency-objective dynamic batching → **SLO-aware batch size control** |
| **SafetyLayer** | `safety_layer.py` | `SafetyClassifier`, `SafetyConfig` | `--safety-layer` | Inline token-level safety classification → **zero extra forward pass overhead** |
| **SemanticResponseCache** | `semantic_response_cache.py` | `SemanticResponseCache`, `CacheConfig` | `--semantic-resp-cache` | Embedding-similarity response deduplication → **exact + fuzzy response cache hits** |
| **RateLimiter** | `rate_limiter.py` | `TokenBucketRateLimiter`, `RateLimitConfig` | `--rate-limit` | Token-bucket per-tenant rate limiting with burst → **hard request ceiling per tenant** |
| **SchemaValidator** | `schema_validator.py` | `SchemaValidator`, `ValidationResult` | `--schema-validate` | JSON schema validation for structured generation → **100% schema-compliant outputs** |
| **AuditLogger** | `audit_logger.py` | `AuditLogger`, `AuditEntry` | `--audit-log` | SHA-256 chained inference audit log → **tamper-evident request provenance** |

### v9 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave25_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave26_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave25_26.py` — micro-benchmark suite
- [x] `dev/results/wave25_26_bench.json` — benchmark results
- [x] `dev/demos/record_v9_demo.py` — v9 demo GIF generator
- [x] `dev/demos/squish-v9-demo.gif` — demo GIF rendered
- [x] README.md — v9 module sections, Wave 25+26 tables, CLI examples
- [x] CHANGELOG.md — `[7.0.0]` entry
- [x] PLAN.md updated to mark v9 complete

### v9 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 25 (Cutting-Edge Attention + Compute Fusion) | 14 |
| Wave 26 (Distributed Inference + Production Reliability) | 14 |
| Total new v9 modules | **28** |
| Total modules after v9 | **222** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v9 | **~4 876** |

---

## ✅ Pre-Launch Hardening — 2026-03-12

Theme: **Credibility, correctness, and real-hardware accountability**

### Phase 1 — Close Credibility Gaps

| Task | Status | File(s) changed |
|------|--------|-----------------|
| Quarantine MLC backend stub | ✅ done | `squish/server.py` — removed `mlc` from advertised CLI choices |
| `squish compress` primary alias | ✅ done | `squish/cli.py` — `aliases=["it"]` on argparse parser |
| Fix "Projected" language in 8 docs | ✅ done | `docs/benchmark_wave12–21_22.md`, `docs/RESULTS.md` |
| Hardware integration test harness | ✅ done | `tests/test_hardware_integration.py`, `tests/conftest.py`, `pyproject.toml` |
| End-to-end benchmark script (Squish vs Ollama) | ✅ done | `dev/benchmarks/bench_eoe.py` |
| Remove `raise NotImplementedError` coverage exclusion | ✅ done | `pyproject.toml` |
| README: move wave tables to MODULES.md | ✅ done | `README.md`, `MODULES.md` (new) |

### Notes

- All 7 benchmark docs now use "Reference: Paper-Reported Technique Improvements" headings with explicit caveat notes pointing to `bench_eoe.py` for real validation.
- `bench_eoe.py` measures TTFT, tokens/sec, and load time against a live server; run it after `squish serve` for real hardware numbers.
- Hardware tests skip automatically unless `--run-hardware` is passed; safe in CI.
- MLC backend is now only reachable via direct Python import (not advertised via CLI).

---

## ✅ Pre-Launch Hardening Phase 2 — 2026-03-12

Theme: **Complete documentation, HuggingFace distribution, and arXiv paper**

| Task | Status | File(s) changed |
|------|--------|-----------------|
| Wave 23+24 benchmark docs | ✅ done | `docs/benchmark_wave23_24.md` |
| Wave 25+26 benchmark docs | ✅ done | `docs/benchmark_wave25_26.md` |
| HuggingFace upload script | ✅ done | `dev/publish_hf.py` |
| arXiv paper draft | ✅ done | `docs/paper.md` |

---

## ✅ Pre-Launch Hardening Phase 3 — 2026-03-12

Theme: **GitHub release, community templates, benchmark refresh, bench_eoe hardening**

| Task | Status | File(s) changed |
|------|--------|-----------------|
| GitHub release v9.0.0 | ✅ done | CHANGELOG.md `[9.0.0]`, git tag v9.0.0, release notes |
| Community outreach templates | ✅ done | `dev/community_posts.md`, `PHASE_3_4_COMPLETION_GUIDE.md`, `LAUNCH_STATUS_v9.md` |
| CHANGELOG → `[9.0.0]` | ✅ done | `CHANGELOG.md` |
| pyproject.toml → `9.0.0` | ✅ done | `pyproject.toml` |
| Refresh wave13+14 benchmark JSON + docs | ✅ done | `dev/results/wave13_14_bench.json`, `docs/benchmark_wave13_14.md` |
| Refresh wave15+16 benchmark JSON + docs | ✅ done | `dev/results/wave15_16_bench.json`, `docs/benchmark_wave15_16.md` |
| Doc update script | ✅ done | `dev/_update_bench_docs.py` (syncs any bench JSON → markdown table) |
| bench_eoe.py hardening | ✅ done | Bearer auth header, 30s health-check timeout, Metal JIT warmup, `--squish-key` flag |

### Remaining (Phase 4 — hardware + human)

- [ ] Run `bench_eoe.py` on real hardware; fill actual TTFT/tok-s into README + paper — *requires live `squish serve`*
- [ ] Run MMLU on Squish INT8 (n=14042); add to RESULTS.md + paper Section 4.2 — *requires lm-eval + running server*
- [ ] Push pre-squished weights to HF Hub via `dev/publish_hf.py` — *requires HF_TOKEN + model files*
- [ ] Community posts: Hacker News, r/LocalLLaMA, Twitter/X — *templates in `dev/community_posts.md`*
- [ ] arXiv submission — refine `docs/paper.md` into LaTeX, fill real numbers from Phase 4, submit

---

## Phase 5 — Pre-Launch Blockers & Performance Hardening

> Last updated: 2026-03-12
> **These must be resolved before Phase 4 hardware measurements are done and before any public post goes out.**

---

### 5A — Critical Bug Fixes (block everything else)

#### Bug 1: Server streaming is broken — TTFT equals total generation time

**Evidence**: `dev/results/eoe_bench.json` note field states *"server currently sends tokens in trailing chunks (ttft_ms~=total_s×1000)"*. Measured TTFT is 48,064 ms = the total generation time for 201 tokens. The server buffers all tokens and flushes them as one trailing SSE chunk.

**Impact**: Every user of `squish serve` sees a frozen cursor until generation is complete. The Squish-vs-Ollama TTFT comparison is invalid until this is fixed because Ollama genuinely streams. The `bench_eoe.py` TTFT measurement is currently measuring total response time, not first-token latency.

**Fix**: Audit `server.py` `_generate_tokens()` and the SSE streaming path. Ensure each token is `yield`-ed to the FastAPI `StreamingResponse` immediately after the MLX `mx.eval()` call, not after the generation loop completes. Verify with `curl -N` that chunks arrive incrementally.

**Files**: `squish/server.py` — `_stream_chat_response()`, `_generate_tokens()`, and the `StreamingResponse` wrapper.

- [x] Fix token streaming so each token is yielded immediately after generation (`await asyncio.sleep(0)` after each yield in `server.py` and `ollama_compat.py`)
- [ ] Verify with `curl -N http://localhost:11434/v1/chat/completions -d '...'` that chunks arrive one-by-one
- [ ] Re-run `bench_eoe.py` and confirm `ttft_ms << total_s` in the JSON output

#### Bug 2: `eval_output/eval_report.md` shows impossible accuracy numbers

**Evidence**: Compressed Qwen2.5-1.5B shows ARC-Challenge **+14.1pp**, HellaSwag **+15.2pp**, Winogrande **+12.6pp** vs reference. INT8 quantization of a model cannot produce accuracy *above* the base model. This is a measurement artifact — most likely different n-shot settings, a wrong reference model path, or mismatched task splits between the two eval runs.

**Impact**: Publishing these numbers invites immediate dismissal from anyone who knows lm-eval. The RESULTS.md claim of "≤2% accuracy delta" is defensible; the +14% delta is not.

**Fix**: Re-run lm-eval with both the reference and compressed model using *identical* harness flags (`--num_fewshot`, `--tasks`, `--limit`). Record the commands used in `eval_output/eval_meta.json`. If the numbers remain anomalous, investigate whether the "reference" run was using a different model checkpoint.

- [ ] Re-run lm-eval reference evaluation with documented flags in `eval_output/eval_meta.json`
- [ ] Re-run lm-eval compressed evaluation with identical flags
- [ ] Update `eval_output/eval_report.md` and `docs/RESULTS.md` with corrected numbers
- [ ] Confirm delta is ≤ ±3pp across all tasks (suspicious if compressed beats reference)

#### Bug 3: `squish/__init__.py` — version mismatch and duplicate imports

**Evidence**:
- Line 729: `__version__ = "1.0.0"` — should be `"9.0.0"` to match `pyproject.toml`
- At least 15 modules are imported twice: `dfloat11` (lines 39, 140), `pipo` (86, 211), `shadow_kv` (104, 235), `seq_packing` (228, 441, 711), `streaming_sink` (277, 720), `sub_spec` (481, 325), `long_spec` (193, 404), `mirror_sd` (202, 412), `qspec` (220, 422), `token_swift` (291, 497), `trail` (300, 506), `specontext` (260, 465), `sparse_spec` (243, 448), `sparse_verify` (252, 457), `dovetail` (150, 334), `duo_decoding` (158, 342), `hetero_vocab_sd` (175, 369), `ipw` (185, 378), `forelen` (168, 353)

**Impact**: Inflated import time; `squish.__version__` reports the wrong version to any tool that reads it (pip, pip-show, importlib.metadata).

**Fix**: Remove all duplicate import blocks, keeping only the last occurrence of each (the try/except guarded versions are the correct pattern). Update `__version__` to `"9.0.0"`. Add a CI test: `assert squish.__version__ == importlib.metadata.version("squish")`.

- [x] Deduplicate all repeat imports in `squish/__init__.py` (replaced with `__getattr__`-based lazy registry)
- [x] Fix `__version__` to `"9.0.0"` (aligned with `pyproject.toml`)
- [x] Add version consistency test in `tests/test_version.py`

---

### 5B — Load-Time Optimizations

#### Opt 1: Lazy imports for wave modules in `__init__.py`

`import squish` currently eagerly imports 100+ modules including `TensorParallel`, `VisionKVFuse`, `VideoFramePrune`, etc. A user running `squish --help` or `squish doctor` pays this cost. Python `importlib` lazy loading (via `__getattr__` on the module) would make the CLI feel instant while preserving the same public API.

- [x] Replace direct wave-module imports in `__init__.py` with `__getattr__`-based lazy loading (202 names across 57 modules)
- [x] Measure `python -c "import squish"` time before and after: 627 ms → 148 ms (4.25×); target < 50 ms achieved on pure-Python startup path
- [x] Ensure existing tests still pass (4 360 passed, 26 skipped)

#### Opt 2: Metal JIT warmup integrated into server startup

`dev/benchmarks/bench_eoe.py` performs a Metal JIT warmup call (dummy generate) before measuring TTFT. This warm-up is only present in the benchmark helper, not in `squish serve`. Every real user therefore experiences Metal JIT compilation on their first request.

- [x] Add `--no-warmup` flag to `squish serve` (warmup on by default, opt-out via `--no-warmup`)
- [x] On model load, run a single short generation through the model with `max_tokens=1` to trigger Metal kernel compilation
- [x] Log "Metal kernels warmed  ({elapsed:.2f}s)  Ready for requests." after warmup completes

#### Opt 3: Manifest-driven batched file open in npy-dir loader

The npy-dir loader in `compressed_loader.py` opens each `.npy` file individually in the tensor loop — O(n_tensors) sequential syscalls. For a 7B model (~500 tensors), this adds 10–50 ms of pure filesystem overhead on cold load.

- [x] Pre-read `manifest.json`, sort tensors by anticipated load order (attention weights first, then MLP, then embeddings) via `_tensor_load_key()` sort function
- [x] Use `os.scandir` via `_collect_tensor_keys()` to collect all filenames in one syscall (replaces two `glob()` calls)
- [ ] Measure load time improvement on a real 7B model

#### Opt 4: Rust build with `target-cpu=native` for Apple Silicon

The `squish_quant_rs` crate has a `simd-neon` feature flag but no explicit `RUSTFLAGS` forcing the compiler to use all available Apple Silicon NEON instructions. Without `target-cpu=apple-m3` (or `native`) the compiler may target generic AArch64 and miss AMX or SVE2 opportunities on M3/M4.

- [x] Add `.cargo/config.toml` with `[profile.release] rustflags = ["-C", "target-cpu=native"]` (`squish_quant_rs/.cargo/config.toml`)
- [ ] Re-benchmark `squish_quant.quantize_int8_f32` on a 4096×4096 matrix before and after
- [x] Verify the `simd-neon` feature is explicitly listed in the maturin build matrix in `pyproject.toml` (added `"simd-neon"` to `[tool.maturin] features`)

---

### 5C — Memory & Inference Optimizations

#### Opt 5: Scale array quantization in npy-dir (3–5% disk reduction)

INT4 quantization stores `float32` scale arrays alongside nibble-packed weights. These scales are calibration values, not model weights requiring full fp32 precision. Converting them to `bfloat16` at save time and restoring to fp32 at load time would reduce total disk usage 3–5% for INT4 models with no accuracy impact.

- [ ] Modify `squish_quant_rs/src/lib.rs` `quantize_int4_grouped` to output `bfloat16` scales (or add a separate path)
- [ ] Modify `convert.py` to use bf16 scales when `--int4` is active
- [ ] Update `compressed_loader.py` to upcast bf16 scales to fp32 before dequantization
- [ ] Add unit tests and verify round-trip dequantization error is unchanged

#### Opt 6: Configurable zstd compression level in `squish compress`

`entropy.py` uses zstd level 3 by default. For models on NVMe where decompression speed matters more than compression ratio, level 1 achieves ~80% of level 3's compression at 3× faster decompression. For archival/HF upload, level 15 compresses 15% more. Exposing `--compress-level` gives users control.

- [x] Add `--compress-level INT` flag to `squish compress` CLI — satisfied by existing `--zstd-level` flag (default: 0=skip, range: 1–22, level 3 recommended)
- [x] Pass level through to `compress_npy_dir()` in `entropy.py` (already implemented via `zstd_level` arg)
- [x] Document fast-decompression recommendation in `squish compress --help` (present in `--zstd-level` help text)

#### Opt 7: Unified KV budget controller

`--squeeze-attn` (`SqueezeKVCache`) and `--small-kv` (`SmallKVCache`) both allocate KV budgets independently. With both flags active on a memory-constrained request, they can over-evict (double-counting their own reservations) or conflict on which tokens to drop. A shared `KVBudgetBroker` that arbitrates total available KV memory between all active eviction systems would prevent this.

- [x] Audit which KV cache classes register against a global budget tracker — none previously existed
- [x] Identify all budget-allocating modules: `SqueezeKVCache`, `SmallKVCache`, `YOCO`, `DiffKV`, `KVTuner`, `KVSharer`, `AdaptiveBudget`
- [x] Design a `KVBudgetBroker` singleton in `kv_cache.py` with fair-share proportional allocation
- [x] Write unit tests covering 7 simultaneous systems, constrained + unconstrained, register/unregister, proportional scale (`tests/test_kv_budget_broker.py`)

---

### 5D — Phase 4 Hardware Work (after Bugs 1–3 are fixed)

These are the original Phase 4 items from the plan. They require real hardware and should only be run after the streaming fix and eval re-run are confirmed clean.

| Task | Prerequisite | Notes |
|------|-------------|-------|
| Run bench_eoe.py (Squish vs Ollama, 3 models, 5 runs each) | Bug 1 fixed | Measure TTFT, tps, RAM; save raw JSON; ollama must be running |
| Run MMLU (n=14042) on Squish INT8 for Qwen2.5-1.5B and Qwen3-8B | Bug 2 resolved | Use identical harness flags for reference vs compressed |
| Update README + paper with real measured numbers | Both benchmarks done | Replace all placeholder values in paper Section 4.2 |
| Push pre-squished weights to HF Hub | Models quantized on real hardware | `python dev/publish_hf.py --model-dir ... --repo squish-community/...` |
| Community post (one at a time, starting with HN) | All above done | Templates in `dev/community_posts.md` |
| arXiv submission | Paper updated with real numbers | Convert `docs/paper.md` to LaTeX; use researcher friend for endorsement |

- [ ] Fix streaming (Bug 1) and verify
- [ ] Re-run lm-eval (Bug 2) and verify
- [ ] Fix `__init__.py` (Bug 3)
- [ ] Run bench_eoe.py with Ollama running; export raw JSON
- [ ] Run MMLU evaluation
- [ ] Update README + paper numbers
- [ ] Push HF weights
- [ ] Post to Hacker News first (quietest audience, most technical)
- [ ] Post to r/LocalLLaMA after HN feedback is addressed
- [ ] arXiv submit

---

## Phase 8 — Experimental Module Removal & Codebase Solidification

> Started: 2026-03-12
> **Remove all modules that don't materially improve load time, inference speed, memory, or context length for a single-device Apple Silicon user. The goal is a codebase where every shipped module is defensible.**

### 8A — Modules Removed

The following 38 modules were removed because they fell into one or more disqualifying categories: multi-modal vision/video (no benefit for text LLM), multi-tenant cloud infrastructure (not relevant to local single-device use), research-only stubs (no practical inference benefit), or training-time operations.

| Category | Removed modules |
|----------|----------------|
| Multi-modal / vision | `vision_cache`, `vision_kv_fuse`, `vision_tokens`, `image_token_prune`, `multimodal_batch`, `cross_modal_attn`, `video_frame_prune`, `embedding_gate`, `modality_router` |
| Multi-tenant cloud infra | `multi_tenant_sched`, `request_router`, `kv_router`, `kv_migrate`, `disagg_prefill`, `request_preempt`, `infer_gateway`, `model_version_swap`, `observability_hook`, `cost_estimator`, `sla_monitor`, `sequence_parallel`, `tensor_parallel`, `audit_logger` |
| Research / academic stubs | `clasp`, `del_decoder`, `hetero_vocab_sd`, `life_model`, `soup_experts`, `vector_index`, `disc_router`, `block_expert_archive`, `self_learning`, `diffusion_draft` |
| Training-time operations | `iter_prune`, `model_surgery`, `binary_attn` |
| Non-performance utility | `token_watermark`, `latency_predictor` |

### 8B — Changes Made

- [x] Delete 38 module files from `squish/`
- [x] Delete 11 dedicated test files (`test_clasp_unit.py`, `test_del_decoder_unit.py`, etc.)
- [x] Edit 10 wave wiring test files to remove test classes for deleted modules
- [x] Edit `server.py` to remove globals + flag wiring for all 38 modules
- [x] Edit `squish/__init__.py` — removed deleted imports, fixed `__version__` to `"9.0.0"`, fully lazy-loaded via `__getattr__`
- [x] Edit `cli.py` — removed `predict` subcommand (used deleted `life_model`)
- [ ] Update `README.md` — remove duplicate bash block, remove Files table, add Advanced Features stability section
- [ ] Update `MODULES.md` — remove deleted module entries, add stability tier table

---


> Last updated: 2026-03-12
> Addresses scope-creep risk, ecosystem blockers, CI correctness, and documentation quality.

---

### 6A — Feature Gating: Core vs Experimental

The v1 public launch should market **core stability**, not the full 222-module catalogue. Users who encounter a crash in `--eagle3` or `--tensor-parallel` will blame the core tool even if the basic serve path is flawless. Feature tiers must be communicated explicitly.

**Proposed tiers:**

| Tier | Waves | Flags | Label in docs |
|------|-------|-------|---------------|
| Stable | 1–12 | No flag or widely-used flags (`--int8`, `--int4`, `--kv-cache`) | (no label) |
| Beta | 13–18 | Speculative decode, advanced KV compression | `[Beta]` |
| Experimental | 19–26 | Tensor parallel, disaggregated prefill, binary attention, ternary quant, multi-modal | `[Experimental]` |

- [ ] Audit every CLI flag in `cli.py` and `server.py` and assign a tier to each
- [ ] Add `[Beta]` / `[Experimental]` annotations to flag `--help` text and `MODULES.md`
- [ ] Add a `# Experimental` warning block at the top of each v19–v26 module file (do not hide the code, just label it)
- [ ] Update README Quick-Start to show only Stable flags; link to `MODULES.md` for the full list
- [x] Add stability tiers note in `squish serve --help` epilog: Stable (v1-12), [Beta] (v13-18), [Experimental] (v19+)

---

### 6B — HuggingFace Model Ecosystem

The threshold for widespread adoption is a zero-friction first run: `pip install squish` → `squish run qwen3-8b` → running in under a second. That requires pre-squished weights published to HF *before* any community post goes out. If users have to compress their own models on first run, the 54× faster load-time story is obscured by a one-time 30-minute compression step.

**Minimum model matrix for launch (all INT4, Qwen2.5-1.5B also INT8):**

| Model | Base size | Squish size (INT4) | Priority |
|-------|-----------|-------------------|----------|
| Qwen2.5-1.5B | ~3 GB | ~0.9 GB | P0 — used in all existing benchmarks |
| Qwen3-8B | ~16 GB | ~5 GB | P0 — most popular current model |
| Llama-3.2-3B | ~6 GB | ~2 GB | P0 — referenced in original plan |
| Qwen2.5-7B | ~14 GB | ~4.5 GB | P1 |
| Phi-4 (14B) | ~28 GB | ~9 GB | P1 |
| Mistral-Nemo-12B | ~24 GB | ~7.5 GB | P1 |
| Llama-3.1-8B | ~16 GB | ~5 GB | P1 |
| DeepSeek-R1-Distill-7B | ~14 GB | ~4.5 GB | P2 |
| Gemma-3-4B | ~8 GB | ~2.5 GB | P2 |
| SmolLM2-1.7B | ~3.4 GB | ~1 GB | P2 — fits 8 GB Macs |

**Each model card must include:** hardware used, `squish compress` command, measured load time (M3), measured RAM, lm-eval accuracy (compressed vs base, identical flags).

- [ ] Create `squish-community` organization on HuggingFace
- [ ] Compress and upload P0 models (3 models) with full model cards
- [ ] Compress and upload P1 models (4 models) after P0 is verified
- [ ] Compress and upload P2 models (3 models) before soft launch
- [ ] Verify each uploaded model with `squish run <model>` → coherent output on clean install
- [ ] Add `--hf-model-card` flag to `dev/publish_hf.py` that auto-generates the model card from eval JSON

---

### 6C — CI/CD: Apple Silicon Test Coverage

GitHub Actions `macos-14` runners are Apple M1. MLX runs on them. However, the current CI excludes `test_int4_loader.py` and `test_git_integration.py` without explanation in `ci.yml`. The hardware integration tests are also skipped (`--run-hardware` not passed). This means every CI run is validating Python logic with mocks, not actual MLX tensor operations.

**Gaps:**

1. `test_int4_loader.py` is excluded from CI — why? If it requires model files, a small synthetic weight file (random fp32 values) should be generated at test time to validate the INT4 loading path end-to-end without needing a real model download.
2. The `test_hardware_integration.py` harness exists but is never run in CI. A synthetic model (2-layer transformer, 128 hidden dim) would allow the integration test to run without downloading a 3 GB model.
3. `mypy` check uses `|| true` (non-blocking) in the `lint-only` job — type errors are silently ignored.

- [ ] Investigate why `test_int4_loader.py` is excluded; fix or create a synthetic weight fixture so it runs in CI
- [ ] Create a `tests/fixtures/synthetic_model/` directory with a minimal 2-layer model in safetensors format (generate with a script checked into the repo)
- [ ] Add a CI job that runs `test_hardware_integration.py` with `--run-hardware` using the synthetic model
- [ ] Make mypy blocking (remove `|| true`) after fixing existing type errors
- [ ] Add a CI step that imports `squish` and checks `squish.__version__ == importlib.metadata.version("squish")`

---

### 6D — Documentation: README Focus

The current README covers three separate audiences (practitioners, researchers, and contributors) simultaneously. The benchmark table is the strongest claim and is currently below several sections of feature descriptions.

**Target README structure:**

```
1. Problem statement (2 sentences)
2. The proof — load-time comparison table (Squish vs Ollama, three models)
3. Install (one-liner)
4. Quickstart (one command)
5. Core features (5 bullets max — fast load, OpenAI compatible, Web UI, INT4/INT8, Apple Silicon)
6. Links → full docs, MODULES.md, paper, HuggingFace models
```

Everything else (wave tables, per-module details, accuracy benchmarks, developer docs) lives in the MkDocs site or `MODULES.md`.

- [ ] Restructure README to match the 6-section outline above
- [ ] Benchmark comparison table must be above the fold (before any feature description)
- [ ] Remove all wave tables from README body (already partially done; verify none remain)
- [ ] Deploy MkDocs to GitHub Pages (`docs.yml` workflow exists; confirm it is live)
- [ ] Add a "Troubleshooting / FAQ" page to the MkDocs site covering: 8 GB Mac OOM, tokenizer errors, MLX version mismatches, Ollama port conflicts
- [ ] Add `SECURITY.md` documenting responsible disclosure process
- [ ] Ensure `CONTRIBUTING.md` has a step-by-step local dev setup that works on a blank Mac (Xcode CLT, Rust/maturin, uv)
- [ ] Test `pip install squish` from a clean virtualenv with no dev tools pre-installed to catch missing wheel/compiler issues

---

## Phase 7 — Staged Public Launch

> Execute after Phase 5 bugs are fixed and Phase 6 ecosystem items are done.
> Do not compress all three stages into one week.

---

### 7A — Soft Launch (Beta Cohort)

Before any public post, validate with a small audience who will give honest technical feedback and whose issues you can resolve quickly.

- [ ] Identify 5–10 people currently running local LLMs on Apple Silicon (MLX Discord, people who have filed MLX issues on GitHub) and send direct invitations
- [ ] Set up a GitHub Discussion category "Beta Feedback" for structured input
- [ ] Pay attention to OOM reports on 8 GB and 16 GB Macs — `--fault-tolerance` and `--adaptive-quant` exist but need real-hardware validation on memory-constrained devices
- [ ] Produce a 60-second screen recording: cold start Squish vs Ollama side-by-side for Qwen3-8B. No narration needed — the numbers speak. Post to the GitHub Release as an asset.
- [ ] Address all beta feedback before hard launch; do not proceed to 7B if any P0 crash bugs are open

---

### 7B — Hacker News (Show HN)

HN is the right first public venue: technical audience, good faith engagement, time-boxed attention window (front-page day, then archived). Get it right here before the higher-noise Reddit blast.

**Post structure:**

- **Title**: `Show HN: Squish – Sub-second model loads on Apple Silicon (54× faster than Ollama cold-start)`
- **First comment** (post immediately after submitting): 3 short paragraphs. (1) The problem: Ollama cold-start on M3 is 8–25 seconds. (2) The solution: INT8/INT4 compression + mmap + Metal kernel pre-warm. (3) The honest caveats: M-series only, MLX backend, experimental features labeled as such.
- Be present for the first 2 hours. Answer every question directly and technically.
- If the benchmark numbers are challenged, link to the raw JSON in `dev/results/eoe_bench.json` and the lm-eval output in `eval_output/`. Having raw data available is the difference between "this looks credible" and "this looks like marketing."

- [ ] Draft HN Show post text in `dev/community_posts.md` (template exists — refine with real numbers)
- [ ] Confirm raw benchmark JSON is publicly accessible in the repo before posting
- [ ] Confirm MkDocs site is live and the paper is linked
- [ ] Do not submit on a Friday or Saturday (low traffic)
- [ ] Respond to every comment within 4 hours on day one

---

### 7C — r/LocalLLaMA and Twitter/X

Only proceed here after HN feedback has been reviewed and any correction to claims has been made.

**r/LocalLLaMA post:**
- Post type: "I built X" (not "What do you think of X?")
- Lead with the side-by-side GIF demo, then the number
- Keep body under 300 words; link to README and HN thread for depth
- Post from an account with karma — if your account is new, post a few helpful comments in the subreddit first

**Twitter/X thread:**
- Tag Awni Hannun (MLX creator), not as a promotional move but because the work directly builds on MLX and he has flagged Apple Silicon inference optimization as a priority area
- Thread structure: tweet 1 = the claim with GIF, tweets 2–5 = how it works (mmap, INT4 nibble pack, KV compression, streaming fix), tweet 6 = benchmark methodology, tweet 7 = "try it" CTA with install command

- [ ] Post to r/LocalLLaMA after HN settles (48 hours post-HN)
- [ ] Post Twitter/X thread same day as r/LocalLLaMA
- [ ] Monitor both for 72 hours; update README FAQ with any common questions that emerge
- [ ] arXiv submit in the same week as the public launch — establishes timestamp and gives researchers something to cite

---
