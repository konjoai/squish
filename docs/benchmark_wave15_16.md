# Squish v4 — Wave 15+16 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 15 — Serving Intelligence + KV Architecture Evolution

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| AdaServe | `get_gamma()` tight SLO | 1.99 | SLO-customized gamma selection |
| AdaServe | `get_gamma()` relaxed SLO | 1.82 | |
| ConfSpec | `verify_step()` flat logits | 100.21 | Full verification path |
| ConfSpec | `verify_step()` peaked logits | 78.46 | Auto-accept path (high confidence) |
| SeqPacking | `pack()` 32 short seqs | 2521.5 | 8–64 token sequences |
| SeqPacking | `pack()` 8 long seqs | 43959.5 | 128–512 token sequences |
| MetaReasoner | `compute_entropy()` 32k | 500.74 | Static method |
| MetaReasoner | `step()` 32k vocab | 0.23 | Per-token thinking budget decision |
| YOCO | `append()` seq=64 dim=128 | 1.11 | KV append to shared store |
| YOCO | `get_shared_kv()` | 6473.63 | Retrieve cached KV for cross-decoder layers |
| DiffKV | `get_policy()` | 1.59 | Per-head precision policy lookup |
| DiffKV | `record_attention()` 4×4 | 6.33 | Attention pattern accumulation |
| ParisKV | `encode()` batch=32 dim=128 | 34.4 | Online codebook assignment |
| ParisKV | `decode()` batch=32 | 4.2 | Codebook reconstruction |
| ParisKV | `online_update()` batch=8 | 129.4 | Drift-corrected centroid update |
| KVTuner | `search()` 32 layers | 3815.4 | Sensitivity-aware bit assignment |
| CLA | `CLASchedule.from_config()` | 27.56 | Cross-layer attention schedule gen |

---

## Wave 16 — Heterogeneous Compute + Advanced Spec-Decode

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| Dovetail | `verify_one()` vocab=32k | 384.8 | CPU target verification |
| PIPO | `run_layer()` in=out=4096 | 1785.8 | INT4 dequant + matmul w/ prefetch |
| MobileMoE | `route()` single 128 experts | 27.19 | Expert selection |
| MobileMoE | `route_batch()` 32 tokens | 490.2 | |
| OnlineSD | `record()` hidden=4096 | 2.30 | Trace buffer append |
| LookaheadReasoning | `run_cycle()` k=4 | 15.5 | Parallel step verification cycle |
| SparseSpec | `PillarAttnCache.update()` cap=4096 | 1.3 | Attention pillar accumulation |
| SparseSpec | `top_k_indices()` k=205 | 13.9 | Sparse position selection |
| FRSpec | `head.forward()` top-25% vocab | 3881.7 | Compressed draft logits |
| FRSpec | `compress_logits()` 32k→subset | 13.8 | Vocab projection |
| FRSpec | `expand_logits()` subset→32k | 25.3 | Full-vocab restore |
| LongSpec | `LongSpecHead.forward()` h=4096 | 19966.0 | Shared-KV draft head |
| ForeLen | `EGTPPredictor.predict()` | 109.92 | Entropy histogram → length |
| ForeLen | `PLPPredictor.update()` | 0.89 | Exponential decay estimate |
| RASD | `CorpusIndex.search()` 1k seqs | 0.6 | Prefix-tree lookup |
| RASD | `build_retrieval_tree()` | 2.0 | Draft tree construction |

---

## Projected End-to-End Improvements (Apple Silicon + Qwen3-8B)

| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| KV memory (YOCO) | **50%** reduction | YOCO — only cross-decoder layers use KV |
| KV memory (DiffKV) | **2.7–5.7×** compression | DiffKV asymmetric K/V precision |
| KV memory (KVTuner) | **2×** vs naive quant | KVTuner mixed-precision calibration |
| CoT decode energy | **44–89%** saving | MetaReasoner dynamic thinking budget |
| Batch throughput | **1.8×** effective | SeqPacking barrel-effect elimination |
| Spec decode throughput | **2.13×** | SparseSpec dynamic sparse self-speculation |
| Reasoning throughput | **2.1×** | LookaheadReasoning parallel step verification |
| Offloaded model throughput | **1.7×** | PIPO pipelined prefetch offloading |
| Heterogeneous throughput | **2×** | Dovetail CPU+GPU spec decode |
| Draft acceptance | **+5–8 pp** | OnlineSD continuous adaptation |
| Length prediction (MAE) | **29% ↓** vs TRAIL | ForeLen entropy-guided prediction |
| Corpus hit rate | **40–60%** | RASD retrieval-augmented spec decode |

---

## Accuracy Baseline (unchanged — v4 operates on KV / serving paths)

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
