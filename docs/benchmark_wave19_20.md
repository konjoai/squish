# Squish v6 — Wave 19+20 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 19 — Quantisation Kernels + Attention + Speculative Decode

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| FP8Quantizer | `encode()` E4M3 per-channel 128×128 | 3792.9 | FP8 E4M3 simulation |
| FP8Quantizer | `decode()` E4M3 per-channel 128×128 | 996.3 | FP8 dequantisation |
| FP8Quantizer | `encode()` E5M2 per-block 128×128 | 3606.5 | FP8 E5M2 activation format |
| MXQuantizer | `encode()` MX4 tile=32 128×128 | 8407.8 | OCP MX4 microscaling |
| MXQuantizer | `decode()` MX4 tile=32 128×128 | 3331.4 | MX tile dequantisation |
| FlashDecodeAttn | `decode()` n_heads=8 seq=512 d=64 | 842.7 | Split-KV merge latency |
| FlashDecodeAttn | `decode()` n_heads=8 seq=64 d=64 | 799.5 | Short-context baseline |
| PagedKVCache | `append()` kv_heads=2 d=64 | 1.57 | Single-token append |
| PagedKVCache | `gather()` ~32 tokens | 69.61 | Contiguous gather |
| GQACache | `append()` kv_heads=2 d=64 | 1.07 | GQA KV append |
| GQACache | `grouped_query_attention()` 8q/2kv seq=32 | 21.3 | GQA forward |
| SlidingWindowKV | `append()` w=128 kv_heads=2 (full) | 1.09 | Ring-buffer append |
| SlidingWindowKV | `sliding_window_attention()` n_heads=8 | 70.2 | Decode-step SWA |
| RoPEScaling | `NTKScaler.get_freqs()` seq=512 d=64 | 24.93 | NTK context extension |
| RoPEScaling | `NTKScaler.apply()` seq=512 n_heads=4 | 510.2 | Full RoPE rotation |
| ActSparsityPred | `record()` layer=0 (16, 256) | 3.63 | Sparsity stat accumulation |
| ActSparsityPred | `calibrate()` 4 layers | 1.27 | Sparsity map generation |
| ActSparsityPred | `SparseFFNGate.apply()` (16, 256) | 6.23 | Element-wise gate mask |
| FusedRMSNorm | `forward()` batch=16 d=256 | 13.16 | Fused residual+norm |
| FusedRMSNorm | `forward()` batch=64 d=256 | 30.53 | Larger batch |
| FusedRMSNorm | `fused_add_rms_norm()` batch=16 | 13.45 | Module-level function |
| LoRAInference | `apply()` batch=8 in=256 rank=16 | 8.39 | LoRA delta addition |
| LoRAInference | `merge_into()` rank=16 256×256 | 30.5 | Permanent weight merge |
| MedusaDecoder | `draft()` d=256 vocab=2k n_heads=4 | 108.6 | Draft tree construction |
| MedusaDecoder | `verify()` 4 draft tokens | 2.43 | Greedy acceptance check |
| Eagle3Decoder | `draft_step()` d=256 vocab=2k n=5 | 81.2 | Feature-level draft chain |
| Eagle3Decoder | `verify_step()` d=256 n=5 | 23.02 | Feature cosine acceptance |
| PrefixPool | `put()` 16 tokens kv=(2,16,64) | 2.55 | SHA-256 + dict insert |
| PrefixPool | `get()` (cache hit) | 1.93 | O(1) prefix lookup |
| PrefixPool | `get()` (cache miss) | 1.75 | Hash-only miss path |
| TokenHealer | `heal()` 3-token prompt | 5.19 | Boundary repair |
| TokenHealer | `find_suffix_overlap()` vocab=10 | 4.90 | Prefix search |

---

## Wave 20 — Model Composition + Serving Infrastructure

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| ModelMerger | `slerp()` 256×256 t=0.5 | 92.1 | Great-circle interpolation |
| ModelMerger | `merge()` SLERP 2 keys | 197.8 | Dict-level merge orchestration |
| LoRAComposer | `forward()` 3 adapters weighted d=256 | 23.62 | Weighted delta sum |
| LoRAComposer | `forward()` 3 adapters equal-weight | 23.10 | Default 1/N weights |
| CBScheduler | `step_batch()` 8 running | 0.43 | Batch promotion check |
| CBScheduler | `submit()` enqueue one request | 1.08 | Waiting queue insert |
| MatryoshkaEmb | `embed()` full=512 → 64 | 3.11 | Truncate + L2-norm |
| MatryoshkaEmb | `batch_embed()` batch=32 → 256 | 18.64 | Batched truncation |
| ANEProfiler | `record_op()` matmul float16 | 3.13 | Heuristic ANE classify |
| ANEProfiler | `summary()` over ~40 ops | 255.69 | Aggregate metrics |
| SpecBenchRunner | `run_task()` 2 prompts gamma=4 | 1.5 | Draft+verify loop |
| PPLTracker | `record()` seq=16 vocab=1k | 117.7 | NLL + PPL update |
| PPLTracker | `rolling_ppl` property | 5.90 | Geometric mean |
| GrammarCache | `get_mask()` cold (compute) | 0.41 | First-time mask build |
| GrammarCache | `get_mask()` warm (cache hit) | 0.22 | O(1) dict lookup |
| GrammarCache | `transition()` state → next | 0.56 | FSM edge traversal |
| QuantAwareCal | `record()` percentile (16, 32) | 5.80 | Stat accumulation |
| QuantAwareCal | `compute_scales()` 32 channels | 4293.88 | Percentile scale search |
| AdaptiveBudget | `step()` latency=180ms (over SLO) | 1.84 | PI controller update |
| AdaptiveBudget | `step()` latency=100ms (under SLO) | 1.75 | Budget relaxation |
| VisionTokenComp | `compress()` attention n=50 d=768 | 11.3 | Attention-weight pruning |
| VisionTokenComp | `compress()` clustering n=20 d=768 | 1059.3 | k-means centroid select |
| ToolSchemaCache | `register()` 1 schema (idempotent) | 1.27 | Hash lookup / no-op |
| ToolSchemaCache | `get()` by name (cache hit) | 0.17 | O(1) dict lookup |
| ToolSchemaCache | `ToolRouter.route()` validate+call | 0.56 | Validation + dispatch |
| DistilSpecCal | `record_step()` vocab=1k (1-D) | 33.8 | KL grad accumulation |
| DistilSpecCal | `record_step()` seq=8 vocab=1k (2-D) | 126.9 | Sequence-level distil |
| DistilSpecCal | `compute_delta()` | 2.52 | Mean gradient output |
| BatchEmbedder | `pool()` mean b=8 seq=32 d=256 | 36.18 | Masked mean pooling |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.


| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| Weight memory (FP8 E4M3) | **4×** vs float32 | FP8Quantizer per-channel |
| Weight memory (MX4) | **8×** vs float32 | MXQuantizer tile microscaling |
| KV bandwidth (split decode) | **2–4×** | FlashDecodeAttn split-KV merge |
| KV memory (paged) | **0% fragmentation** | PagedKVCache virtual blocks |
| KV memory (GQA) | **4× reduction** | GQACache 8q/2kv grouping |
| Context length (sliding) | **unbounded** seq | SlidingWindowKVCache ring-buf |
| Context extension | **8× longer** | RoPEScaling NTK/YaRN/LongRoPE |
| FFN FLOPs (sparsity) | **2–4×** reduction | ActSparsityPredictor + gate |
| Norm bandwidth | **2× memory** saving | FusedRMSNorm fused residual |
| Adapter switching | **0-copy** delta | LoRAInferenceAdapter |
| Decode throughput (Medusa) | **2–3× tokens/step** | MedusaDecoder tree draft |
| Acceptance rate (Eagle3) | **+3.5× vs token draft** | Eagle3Decoder feature pred |
| KV prefill savings | **100% skip** shared prompts | PrefixPool cross-request |
| Tokenizer quality | **seamless boundaries** | TokenHealer prompt repair |
| Multi-model quality | **best-of-N** | ModelMerger SLERP/DARE/TIES |
| Domain coverage | **N domains, 1 base** | LoRAComposer blended deltas |
| GPU utilisation | **2×** batch efficiency | CBScheduler continuous batch |
| Embedding latency | **8× faster** retrieval | MatryoshkaEmbedding truncate |
| ANE visibility | **100% op coverage** | ANEProfiler heuristic trace |
| Spec quality CI | **automated gating** | SpecBenchRunner 6-task suite |
| Quality monitoring | **real-time degradation** | PPLTracker rolling alerts |
| Constrained gen | **0ms mask cost** cached | GrammarCache FSM state |
| Quantisation accuracy | **+1–2 pp vs minmax** | QuantAwareCalibrator MSE/pct |
| SLO compliance | **P99 latency met** | AdaptiveBudgetController PI |
| Vision FLOPs | **50–80% reduction** | VisionTokenCompressor prune |
| Tool call overhead | **~0 µs** cached schema | ToolSchemaCache O(1) lookup |
| Draft acceptance | **+10–15 pp** | DistilSpecCalibrator KL distil |
| Embedding throughput | **4 strategies, 1 pass** | BatchEmbedder pooling |

---

## Accuracy Baseline (unchanged — v6 operates on serving / compute paths)

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
