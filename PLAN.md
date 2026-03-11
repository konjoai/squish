# Squish — Development Plan

> Last updated: 2026-03-11

This document tracks completed waves, the current release, and the next phase.

---

## Versioning Convention

| Version | Waves | Theme |
|---------|-------|-------|
| **v1** | 1–11 | Core baseline — loader, quantizer, server, API, CLI, speculative decode |
| **v2** | 12 | Reasoning-Aware KV · INT3 · Async I/O |
| **v3** | 13–14 | Ultra-Long Context · Adaptive Spec-Decode · Quantisation |
| **v4** | 15–16 | Serving Intelligence · KV Architecture Evolution · Heterogeneous Compute |

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

## 🚧 v4 — Waves 15+16 (Current Phase)

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

### Wave 16 — Heterogeneous Compute + Advanced Spec Decode (11 modules)

| Module | Flag | Key Result |
|--------|------|-----------|
| Dovetail | `--dovetail` | CPU+GPU heterogeneous spec decode → 2× throughput (ANE-constrained) |
| SwiftSpec | `--swift-spec` | Async disaggregated decode → minimal overlap overhead |
| PIPO | `--pipo` | Pipelined prefetch offloading → 1.7× throughput >VRAM models |
| MobileMoE | `--mobile-moe` | MoE balanced layer skip → 1.4× throughput on MoE models |
| OnlineSD | `--online-sd` | Continuous draft adaptation → +5–8% acceptance rate |
| LookaheadReasoning | `--lookahead` | Parallel step verification → 2.1× throughput on reasoning |
| SparseSpec | `--sparse-spec` | Dynamic sparse self-speculation → 2.13× throughput |
| FRSpec | `--fr-spec` | Frequency-ranked vocab compression → 13% draft latency ↓ |
| LongSpec | `--long-spec` | Shared-KV draft head → zero draft KV overhead at any context |
| ForeLen | `--forelen` | Entropy-guided length prediction → 29% MAE ↓ vs TRAIL |
| RASD | `--rasd` | Retrieval-augmented spec decode → 40–60% corpus hit rate |

### Deliverables checklist

- [x] All 21 modules implemented and wired in `server.py`
- [ ] `tests/test_wave15_server_wiring.py` — import + instantiation tests
- [ ] `tests/test_wave16_server_wiring.py` — import + instantiation tests
- [ ] `dev/benchmarks/bench_wave15_16.py` — micro-benchmark suite
- [ ] `docs/benchmark_wave15_16.md` — benchmark results
- [ ] `dev/demos/record_v4_demo.py` — v4 demo GIF generator
- [ ] README.md — v4 module sections
- [ ] CHANGELOG.md — v4 entry
- [ ] Demo GIF rendered (`dev/demos/squish-v4-demo.gif`)

---

## 🔮 v5 — Waves 17+ (Future)

Candidate themes (not yet scoped):
- **Multi-GPU / distributed inference** — tensor parallelism, pipeline parallelism
- **Flash attention kernel** — custom Metal kernel integration
- **VPTQ** — vector post-training quantisation (NeurIPS 2025 Spotlight)
- **CommVQ** — commutative vector quantization for KV (ICML 2025)
- **SVDq** — SVD per-head key mixed precision
- **StreamingSink** — attention sink for infinite context
- **SmallKV** — small-model saliency compensation
- **SparseVerify** — sparse verification framework
- **DiffusionDraft** — diffusion-LLM draft slot
- **GemFilter** — early-layer input token compression (ICLR 2025)
- **LayerSkip** — early exit and self-speculative decoding
- **IPW** — intelligence per watt evaluation framework
- **KVSlab** — pre-allocated KV slab allocator
- **RobustScheduler** — adaptive interval-prediction scheduling
