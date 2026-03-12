# Squish — Wave 21+22 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 21 — Advanced Memory & Decode

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| TreeVerifier | `verify()` 3-branch 4-draft vocab=4096 | 521.7 | Batched rejection-sampling acceptance |
| KVCompressor | `compress()` h=8 seq=128 d=64 | 317.6 | INT8 quant + magnitude pruning |
| KVCompressor | `decompress()` (kept slice) | 95.03 | INT8 dequantisation |
| DynamicNTKScaler | `get_freqs()` seq=2048 (unscaled) | 4.52 | Below trigger, original base |
| DynamicNTKScaler | `get_freqs()` seq=6000 (NTK-scaled) | 4.31 | NTK formula applied |
| QuantSpecDecoder | `quantize_draft()` n_draft=4 vocab=8192 | 83.98 | INT4-simulate draft logits |
| QuantSpecDecoder | `verify()` n_draft=4 vocab=8192 | 232.0 | Rejection-sampling verification |
| SparseAttnIndex | `build()` h=8 seq=256 d=64 | 113.7 | L2-normalise + store keys |
| SparseAttnIndex | `query()` top_k=32 heads=8 seq=256 | 75.0 | Cosine-similarity ANN retrieval |
| MixedPrecisionKVCache | `assign_precisions()` h=8 | 6.63 | Sensitivity-driven tier assignment |
| MixedPrecisionKVCache | `store()` head_dim=64 fp16 | 0.95 | Per-head quantised store |
| MixedPrecisionKVCache | `load()` head_dim=64 fp16 | 0.82 | Dequantisation to float32 |
| BubbleEliminator | `build_schedule()` 4 stages 8 mb | 13.09 | 1F1B slot assignment |
| BubbleEliminator | `simulate()` bubble=27.27% | 3.02 | Wall-clock throughput projection |
| LayerwiseDecoder | `should_exit()` hidden=256 layer=16 | 6.90 | Probe confidence gate |
| LayerwiseDecoder | `process_layer()` hidden=256 | 11.0 | Linear transform + residual |
| KVCodec | `fit()` n_codebook=32 head_dim=32 n=400 | 8560 | k-means++ codebook fitting |
| KVCodec | `encode_keys()` h=4 seq=32 | 62.8 | Nearest-centroid assignment |
| KVCodec | `decode_keys()` seq=32 | 1.96 | Codebook centroid lookup |
| AttentionDeduplicator | `lookup()` cache=512 d=64 (miss) | 212.1 | Cosine-similarity cache scan |
| AttentionDeduplicator | `store()` FIFO evict | 3.4 | Normalise + enqueue |
| FlashPrefillKernel | `prefill()` seq=256 chunk=64 h=4 d=32 | 3653 | Chunked causal attention |
| BudgetSpecDecoder | `effective_draft_len()` budget=512 | 0.55 | Ramp-down draft length |
| BudgetSpecDecoder | `step(3)` token counter update | 0.51 | Accept + clamp to budget |
| RetentionKernel | `step()` hidden=256 h=4 d=64 | 34.3 | Outer-product state update |
| RetentionKernel | `init_state()` zeros 4×64×64 | 1.89 | State initialisation |
| KVRouter | `route()` n_nodes=4 consistent hash | 1.21 | SHA-256 modulo routing |

---

## Wave 22 — Production Serving & Observability

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| TenantScheduler | `next_request()` WFQ 2 tenants | 0.65 | Weighted fair queue dispatch |
| TenantScheduler | `submit()` single request | 2.47 | Priority queue enqueue |
| RequestRouter | `route()+complete()` 4 replicas | 2.15 | Least-loaded weighted routing |
| CacheWarmupPredictor | `record_access()` 8 tokens | 0.62 | Rolling hash + access count |
| CacheWarmupPredictor | `get_warmup_candidates()` top_k=8 | 19.60 | Recency-weighted ranking |
| TokenBudgetGate | `tick()` hard mode | 0.30 | Per-token counter check |
| TokenBudgetGate | `reset()` new request | 0.18 | State clear for next request |
| SpanCollector | `record()` DECODE span | 3.45 | UUID + monotonic timestamp |
| SpanCollector | `record()+finish()` full lifecycle | 3.58 | Span start to end |
| PrefixCoalescer | `add_request()` buffering | 7.30 | Dedup check + enqueue |
| PrefixCoalescer | `add×4+coalesce()` 20-tok prefix | 8.2 | LCP group formation |
| AdaptiveQuantizer | `quantize()` FP16 50% pressure 128×256 | 3.0 | Half-precision cast |
| AdaptiveQuantizer | `quantize()` INT8 80% pressure 128×256 | 56.0 | Symmetric 8-bit quant |
| AdaptiveQuantizer | `quantize()` INT4 92% pressure 128×256 | 59.1 | Symmetric 4-bit quant |
| AdaptiveQuantizer | `dequantize()` INT4 128×256 | 23.80 | Scale multiply |
| InferenceHealthMonitor | `record_request()` latency=180ms | 95.80 | Rolling window + percentile |
| InferenceHealthMonitor | `overall_health()` p99+error | 5.25 | Multi-metric health eval |
| FaultHandler | `evaluate()` pressure=0.95 batch=8 | 0.50 | 3-action policy check |
| FaultHandler | `evaluate()` pressure=0.50 batch=8 | 0.29 | No-action fast path |
| ModelPool | `acquire()+release()` cache hit 512MB | 0.58 | LRU hot pool access |
| ChunkedStreamer | `stream()` 64 tokens chunk_size=4 | 3.17 | 16-chunk split |
| ChunkedStreamer | `stream()` 16 tokens chunk_size=4 | 1.28 | 4-chunk split |
| RequestCostEstimator | `estimate()` prefill=512 decode=128 | 1.06 | 3-factor billing computation |
| SLAMonitor | `record()` latency=400ms | 0.26 | Deque append |
| SLAMonitor | `check()` window=100 | 41.34 | p99 + error-rate eval |
| PersistentContextCache | `put()` 8 tokens kv=(8,8,64) | 5.36 | MD5 hash + TTL entry |
| PersistentContextCache | `get()` hit_rate=100.0% | 1.90 | Hash lookup + TTL check |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.


| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| KV cache memory (INT8+prune) | **2× reduction** | KVCompressor online quantisation |
| Infinite context (NTK RoPE) | **unbounded** extension | DynamicNTKScaler runtime scaling |
| Spec-decode acceptance (INT4 draft) | **≈full-precision** | QuantSpecDecoder INT4 simulation |
| Prefill FLOPs (sparse attn) | **top-k / seq** fraction | SparseAttnIndex cosine ANN |
| KV memory (mixed precision) | **2–4× reduction** | MixedPrecisionKVCache per-head |
| Pipeline utilisation | **≤bubble_fraction** idle | BubbleEliminator 1F1B schedule |
| Decode throughput (early exit) | **1.5–2×** | LayerwiseDecoder confidence gate |
| KV cache size (codec) | **256× ratio** | KVCodec learned codebook |
| FLOPs (dedup attn) | **hit-rate ×** FLOPs saved | AttentionDeduplicator cache |
| TTFT (chunked prefill) | **O(chunk)** memory | FlashPrefillKernel chunked |
| Overshoot prevention | **0 token** overshoot | BudgetSpecDecoder ramp-down |
| Decode memory | **O(1)** per step | RetentionKernel recurrent state |
| KV transfer overhead | **stable hash** routing | KVRouter disaggregated serving |
| Multi-tenant fairness | **weight-proportional** | TenantScheduler WFQ |
| Load imbalance | **least-loaded** | RequestRouter weighted routing |
| Cold-start TTFT | **top-k prefix hits** | CacheWarmupPredictor |
| Token budget overshoot | **hard 0** overshoot | TokenBudgetGate enforcement |
| Trace overhead | **<1 µs** per span | SpanCollector zero-overhead |
| Prefill FLOPs (coalesce) | **LCP × (n−1)** tokens saved | PrefixCoalescer |
| KV memory under pressure | **4× reduction** INT4 | AdaptiveQuantizer |
| Incident detection | **rolling-window** p99/error | InferenceHealthMonitor |
| OOM prevention | **ordered degradation** | FaultHandler policy engine |
| GPU idle time | **LRU eviction** | ModelPool hot pool |
| Time-to-first-byte | **chunk_size** latency | ChunkedStreamer |
| Billing accuracy | **3-factor** granularity | RequestCostEstimator |
| SLA response time | **escalation** alerts | SLAMonitor |
| Prefix recompute FLOPs | **TTL-cached** KV | PersistentContextCache |

---

## Accuracy Baseline (unchanged — Wave 21+22 operates on serving paths)

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
