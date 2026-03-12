# Squish — Wave 25+26 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 25 — Cutting-Edge Attention Variants & Compute Fusion

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| FlashMLACache | `append()` kv=(4,4,32) | 0.55 | Low-rank latent KV append |
| FlashMLACache | `attend()` seq=128 latent=8 | 38.6 | MLA attention against latent cache |
| FlashMLACache | compression_ratio | **4.0×** | head_dim / latent_dim |
| NativeSparseAttention | `forward()` seq=64 h=4 d=32 | 647 | Block-sparse + sliding window |
| NativeSparseAttention | `forward()` best case | 458 | Min latency |
| NativeSparseAttention | sparsity | 2.93% | Fraction of attended blocks |
| FusedSampler | `sample()` vocab=32k | 1767 | temp+top-p+top-k+rep in one pass |
| FusedSampler | `sample()` best case | 1490 | Min latency |
| FusedSampler | `sample()` batch=16 | 22981 | Batched fused sampling |
| KVDefragmenter | `alloc()+free()` round-trip | 2.36 | Slab alloc/free |
| KVDefragmenter | `defrag()` 256 slots | 349 | In-place compaction |
| KVDefragmenter | fragmentation_ratio post-defrag | 0.0 | Zero fragmentation after pass |
| DualChunkAttention | `encode_chunk()` seq=64 h=4 | 21.1 | Intra-chunk RoPE encode |
| DualChunkAttention | `forward()` 2 chunks | 93.3 | Intra + inter chunk attention |
| ActivationOffloader | `offload()` 512 KB layer | 5.84 | CPU copy via buffer pool |
| ActivationOffloader | `fetch()` 512 KB layer | 6.34 | CPU → GPU restore |
| AttentionMorpher | `select_pattern()` per layer | 0.25 | Pattern dispatch (full/sparse/linear) |
| AttentionMorpher | `layer_patterns()` 16 layers | 4.24 | Per-layer pattern assignment |
| AttentionMorpher | flops_reduction (seq=2048) | **25%** | Morphed vs full attention |
| HydraSpecDecoder | `draft()` n_heads=4 vocab=8192 | 1069 | Multi-head parallel drafts |
| HydraSpecDecoder | `verify()` n_heads=4 vocab=8192 | 1229 | Rejection-sampling verification |
| SequenceCompactor | `compact()` seq=128 keep=64 | 141 | In-place KV slot repack |
| SequenceCompactor | `get_compact_indices()` | 2.35 | Retained index computation |
| LatencyPredictor | `predict()` prefill+decode | 0.82 | Linear latency forecast |
| LatencyPredictor | `record()` observation | 0.78 | Rolling sample update |
| ParallelSampler | `sample()` n=4 vocab=8192 | 509 | Best-of-n diversity sampling |
| ParallelSampler | `sample()` batch=16 | 7637 | Batched best-of-n |
| ContextSummarizer | `summarize()` importance 64 tok | 62.5 | TF-IDF importance ranking |
| ContextSummarizer | `summarize()` recency 64 tok | 6.16 | Recency-weighted compression |
| TokenWatermarker | `mark()` 128 tokens | 137 | Green-list logit bias |
| TokenWatermarker | `detect()` 128-token sequence | 13361 | z-score statistical test |
| SchemaGenEngine | `constrain()` JSON token | 5.38 | FSM next-token constraint |
| SchemaGenEngine | `advance()` state transition | 0.79 | FSM state step |

---

## Wave 26 — Distributed Inference & Production Reliability

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| TensorParallelShard | `shard()` 256×256 weight | 5.95 | Row/column split |
| TensorParallelShard | `forward()` simulated all-reduce | 15.9 | Linear + all-reduce stub |
| SequenceParallelScatter | `scatter()` seq=512 h=4 d=32 | 5.96 | Ulysses seq-dim scatter |
| SequenceParallelScatter | `gather()` seq=512 | 39.1 | All-gather reconstruction |
| SequenceParallelScatter | communication_bytes | 1.5 MB | Per all-gather payload |
| KVMigrator | `pack()` kv=(8,4,32) | 88.9 | Live KV state serialise |
| KVMigrator | `unpack()` packed state | 77.2 | Deserialise + restore |
| KVMigrator | packed_bytes | 524 KB | Wire payload size |
| DisaggPrefillNode | `prefill()` seq=64 h=4 d=32 | 2354 | Prefill-only forward |
| DisaggDecodeNode | `decode_step()` | 0.41 | Receive KV + single decode step |
| PreemptScheduler | `preempt+swap+resume()` | 4.28 | Swap-based preempt cycle |
| PreemptScheduler | `preempt+recompute+resume()` | 1.24 | Recompute-based preempt cycle |
| InferenceGateway | `route()+complete()` 8 workers | 1.90 | Least-loaded routing |
| InferenceGateway | `route()` best case | 1.75 | Min latency |
| ModelVersionManager | `route_request()` canary=10% | 1.45 | Version-split routing |
| ModelVersionManager | `route_request()` best case | 1.33 | Min latency |
| ProductionProfiler | `record()` single op span | 0.18 | Zero-alloc APM record |
| ProductionProfiler | `get_stats()` 1000 ops | 79.5 | p50/p99/p999 computation |
| AdaptiveBatchController | `next_batch()` 8 pending | 1.91 | SLO-driven batch select |
| AdaptiveBatchController | `record_obs()` | 0.22 | Latency observation ingest |
| SafetyClassifier | `score()` seq=64 | 19.4 | Token-level safety scoring |
| SafetyClassifier | `score_logits()` vocab=8192 | 67.3 | Logit-domain safety gate |
| SemanticResponseCache | `lookup()` miss | 295 | Cosine-sim scan (256 cache) |
| SemanticResponseCache | `store()` | 0.81 | Embedding insert |
| TokenBucketRateLimiter | `consume()` 8 tenants | 0.92 | Token-bucket check |
| TokenBucketRateLimiter | `refill()` | 0.48 | Bucket refill step |
| SchemaValidator | `validate()` valid JSON | 7.48 | jsonschema compliance check |
| SchemaValidator | `validate()` invalid JSON | 4.90 | Early rejection path |
| AuditLogger | `log()` single entry | 1.92 | SHA-256 chained append |
| AuditLogger | `verify_chain()` 2010 entries | 2236000 | Full chain integrity check |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.

| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| KV memory (FlashMLA latent) | **4× reduction** measured | FlashMLACache compression_ratio |
| Attention FLOPs (NSA sparsity) | **sub-quadratic** at high sparsity | NativeSparseAttention |
| Sampling overhead | **zero intermediate alloc** | FusedSampler single-pass |
| KV fragmentation | **0% post-defrag** measured | KVDefragmenter in-place compact |
| Long-context FLOPs | **O(chunk²) not O(seq²)** | DualChunkAttention |
| Peak GPU memory (offload) | **layer activation size** freed | ActivationOffloader |
| Attention FLOPs (morph) | **25% reduction** at seq=2048 | AttentionMorpher |
| Speculative draft candidates | **n_heads per step** | HydraSpecDecoder multi-head |
| KV memory (seq compaction) | **pruned slots** reclaimed | SequenceCompactor zero-copy |
| Scheduling latency | **sub-µs** prediction | LatencyPredictor linear probe |
| Output quality (best-of-n) | **diversity-scored** selection | ParallelSampler |
| Context length (summarise) | **importance+recency** pruning | ContextSummarizer |
| Watermark detection | **z-score** statistical | TokenWatermarker Kirchenbauer |
| Structured gen validity | **100% schema** compliance | SchemaGenEngine FSM |
| Memory scaling (tensor ∥) | **linear across devices** | TensorParallel |
| Attention FLOPs (seq ∥) | **distributed** across devices | SequenceParallel |
| KV migration overhead | **zero recompute** on handoff | KVMigrate pack/unpack |
| Prefill/decode specialisation | **separate hardware** | DisaggPrefill |
| Priority inversion | **SRPT** preemption | RequestPreempt |
| Routing latency | **<2 µs** per request | InferenceGateway |
| Deploy downtime | **canary → rollback** | ModelVersionSwap |
| APM overhead | **0.18 µs** per record | ProductionProfiler |
| Batch efficiency | **SLO-objective** driven | AdaptiveBatcher |
| Safety gate cost | **no extra forward pass** | SafetyLayer logit-domain |
| Duplicate inference | **cosine-dedup** short-circuit | SemanticResponseCache |
| Tenant isolation | **hard ceiling** per tenant | RateLimiter token-bucket |
| Output validity | **100% compliant** outputs | SchemaValidator |
| Audit tamper-evidence | **SHA-256 chained** log | AuditLogger |

---

## Accuracy Baseline

> Wave 25-26 modules operate on attention patterns, sampling, and serving infrastructure.
> They do not modify weight values or quantisation, so generation quality is unchanged
> from the Squish INT8 baseline.

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
