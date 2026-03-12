# Squish — Optimisation Module Reference

> Complete per-module flag, problem statement, and benchmark reference for all v2–v9 release waves.
> All benchmarks are CPU/numpy micro-benchmarks on Apple Silicon M-series unless otherwise noted.
> End-to-end hardware benchmarks: run `python3 dev/benchmarks/bench_eoe.py`.

---

## v2 — Wave 12: Core KV + Weight Compression

Enable with `squish run --model <name> [flags]`:

| Module | Flag | Effect | Overhead |
|--------|------|--------|----------|
| **PM-KVQ** | `--pm-kvq` | **4.2× KV cache memory** at 4096 tokens | 14 µs/step |
| **MixKVQ** | `--mix-kvq` | **3.9× KV memory** · 4.1 avg bits/channel | 72 µs/step |
| **CocktailKV** | `--cocktail-kv` | **~3× KV memory** · chunk-similarity routing | 895 µs/512-tok |
| **MiLo INT3** | `--milo` | **5.3× weight compression** · SNR > 13 dB | one-time convert |
| **AgileIO** | `--agile-io` | **40–60% I/O latency** reduction · 25× warm-cache reads | ≈ 0 |
| **SageAttn** | `--sage-attention` | **2.1× attention** speedup (INT8 QK^T) | ≈ 0 |
| **SpargeAttn** | `--sparge-attn` | **2.5–5× attention** speedup (sparse blocks) | ≈ 0 |

Full stack:

```bash
squish run qwen3:8b \
  --pm-kvq --mix-kvq --cocktail-kv \
  --agile-io --milo \
  --sage-attention --sparge-attn
```

Benchmark results: [`docs/benchmark_wave12.md`](docs/benchmark_wave12.md)
Raw data: [`dev/results/wave12_bench.json`](dev/results/wave12_bench.json)

---

## v3 — Wave 13: Ultra-Long Context

v3 (Wave 13) focuses on **ultra-long context** (128K+ tokens) and **adaptive speculative decoding**, shipping 10 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **DuoAttention** | `--duo-attention` | Long-context KV blowup: separates 30–40% retrieval heads from streaming heads | **~2× KV memory** saved at 32K tokens |
| **ShadowKV** | `--shadow-kv` | 128K+ KV cache → CPU offload with low-rank pre-RoPE key projection | **6–10× KV compression** on long contexts |
| **PQCache** | `--pq-cache` | ANN-based KV retrieval for retrieval heads via product quantisation | **4–8× key memory** · sub-ms lookup |
| **SpeCache** | `--spe-cache` | Multi-turn KV reload stalls: speculatively prefetches prior-turn KV | **40–60% KV reload** latency eliminated |
| **DuoDecoding** | `--duo-decoding` | Fixed draft-sequence count wastes ANE cycles on M3 | **1.5–2.3× decode** throughput |
| **KnapSpec** | `--knapspec` | Choosing which layers to skip for self-spec-decode is NP-hard | **Optimal skip schedule** in O(NL) |
| **Token Merging** | `--token-merging` | Similar tokens waste prefill FLOPs | **1.4–1.8× prefill** speedup |
| **TokenSwift** | `--token-swift` | Long outputs (20K–100K tokens) hit KV bandwidth ceiling | **2–3× throughput** on ultra-long gen |
| **C2T** | `--c2t` | Uniform draft tree wastes budget at confident positions | **+0.8 tokens/step** accepted |
| **CLaSp** | `--clasp` | Layer-skip selection is static; ignores hidden-state feedback | **Adaptive skip** · DP-optimal per step |

Full stack:

```bash
squish run qwen3:8b \
  --duo-attention --shadow-kv --pq-cache --spe-cache \
  --duo-decoding --knapspec --token-merging \
  --token-swift --c2t --clasp
```

---

## v3 — Wave 14: Quantisation & Spec-Decode

v3 (Wave 14) focuses on **quantisation methods**, **vocabulary-adaptive speculative decoding**, and **expert mixing**, shipping 16 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **SoupOfExperts** | `--soup-experts` | Sparse LoRA expert blending uses redundant coefficients | **Greedy tolerance** pruning with zero accuracy drop |
| **VisionPrefixCache** | `--vision-cache` | Vision encoder re-runs for identical images | **SHA-256 dedup** → 0 encoder FLOPs on cache hit |
| **Vector Index** | `--vector-index` | Brute-force KV retrieval is O(n) | **MRL + HNSW** → sub-ms ANN retrieval |
| **SubSpec** | `--sub-spec` | Offloaded models have no fast draft path | **Quantised substitute** layers as draft → full spec-decode |
| **DEL Decoder** | `--del-decoder` | Static exit layer wastes compute on easy tokens | **Dynamic early-exit** layer selected per token |
| **DFloat11** | `--dfloat11` | BF16 weights have poor entropy-coding compressibility | **DFloat11** block-float: >30% size reduction vs BF16 |
| **rANS Codec** | `--rans-codec` | Huffman coding leaves 5–15% entropy on the table | **rANS** → near-optimal entropy coding for KV/weights |
| **QSpec** | `--qspec` | Draft and verify share the same quantisation level | **W4A8 draft / W4A16 verify** → 1.8× throughput |
| **QuantSpec** | `--quant-spec` | Full-precision draft is slow; quantised draft is inaccurate | **Bit-width selection** per draft step → 98% accuracy |
| **CopySpec** | `--copy-spec` | Spec-decode needs a trained draft model | **Copy from history** buffer — zero extra model |
| **SqueezeLLM** | `--squeeze-llm` | Uniform quantisation crushes outlier weights | **Sparse + dense** mixed-precision: 4× smaller, 0.5 ppl loss |
| **NF4 Quant** | `--nf4-quant` | Uniform quantisation misaligns to weight distributions | **Normal Float 4-bit** levels → best quality per bit for LLMs |
| **SpinQuant** | `--spin-quant` | Weight outliers defeat quantisation | **Hadamard rotation** → 1.5 ppl improvement at INT4 |
| **HeteroVocab SD** | `--hetero-vocab-sd` | Draft/target vocab mismatch prevents cross-model spec-decode | **Token-map projection** → any draft × any target |
| **HeadInfer** | `--head-infer` | Uniform KV policy wastes memory on non-retrieval heads | **Head-type-aware** KV store: retrieval vs. streaming |
| **Life Model** | `--life-model` | Cache eviction is LRU-blind to access patterns | **Lifecycle predictor** → model-aware eviction signals |

Full stack:

```bash
squish run qwen3:8b \
  --squeeze-llm --nf4-quant --spin-quant \
  --copy-spec --sub-spec --del-decoder \
  --qspec --quant-spec --hetero-vocab-sd \
  --dfloat11 --rans-codec --head-infer \
  --soup-experts --vision-cache --vector-index --life-model
```

Benchmark results: [`docs/benchmark_wave13_14.md`](docs/benchmark_wave13_14.md)
Raw data: [`dev/results/wave13_14_bench.json`](dev/results/wave13_14_bench.json)

---

## v4 — Wave 15: Serving Intelligence + KV Architecture

v4 (Wave 15) focuses on **SLO-aware inference scheduling**, **confidence-gated verification**, and **KV architecture evolution**, shipping 10 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **AdaServe** | `--ada-serve` | Fixed gamma wastes draft budget on low-SLO requests | **30% P99 latency ↓** · 1.5–2× throughput |
| **ConfSpec** | `--conf-spec` | Always running full verification wastes compute | **54% verification cost ↓** via confidence gating |
| **SeqPacking** | `--seq-packing` | Varying sequence lengths cause barrel effect padding waste | **+1.8× batch throughput** |
| **MetaReasoner** | `--meta-reasoner` | CoT thinking on every token wastes energy on easy prompts | **44–89% CoT energy saved** |
| **YOCO** | `--yoco` | Cross-decoder layers duplicate KV across decoding passes | **−50% KV memory** via shared cross-decoder KV |
| **CLA** | `--cla` | Adjacent transformer layers learn nearly identical KV | **10–30% KV reduction** via cross-layer sharing schedule |
| **KVSharer** | `--kv-sharer` | No data-driven way to measure actual KV layer redundancy | **~30% KV ops saved** via calibration-based share map |
| **DiffKV** | `--diffkv` | Uniform K/V precision ignores asymmetric sensitivity | **2.7–5.7× KV compression** · 1.9–5.4× throughput |
| **ParisKV** | `--paris-kv` | Online KV quantisation codebooks drift without correction | **4× KV compression** with drift-robust adaptation |
| **KVTuner** | `--kvtuner` | Naive mixed-precision quant loses 20–35% accuracy | **20–35% accuracy restored** vs uniform quant |

Full stack:

```bash
squish run qwen3:8b \
  --ada-serve --conf-spec --seq-packing --meta-reasoner \
  --yoco --cla --kv-sharer \
  --diffkv --paris-kv --kvtuner
```

---

## v4 — Wave 16: Heterogeneous Compute + Advanced Spec-Decode

v4 (Wave 16) focuses on **heterogeneous CPU+GPU execution**, **pipelined weight offloading**, and **advanced speculative decoding**, shipping 11 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **Dovetail** | `--dovetail` | CPU idle during GPU draft wastes heterogeneous compute | **2× throughput** via concurrent CPU verify + GPU draft |
| **PIPO** | `--pipo` | Sequential weight offload causes GPU idle stalls | **+1.7× throughput** via pipelined prefetch overlap |
| **MobileMoE** | `--mobile-moe` | MoE expert dispatch ignores device balance | **+1.4× throughput** via balanced layer-expert routing |
| **OnlineSD** | `--online-sd` | Frozen draft heads degrade after fine-tuning | **+5–8 pp acceptance rate** via continuous adaptation |
| **LookaheadReasoning** | `--lookahead-reasoning` | Sequential reasoning steps serialise all verification | **+2.1× throughput** via parallel step verification |
| **SparseSpec** | `--sparse-spec` | Static speculation ignores dynamic attention patterns | **+2.13× spec throughput** via adaptive pillar cache |
| **FRSpec** | `--fr-spec` | Full-vocab draft head is expensive at inference | **−13% draft latency** via frequency-ranked subset head |
| **LongSpec** | `--long-spec` | Draft KV grows with context → memory ceiling for long gen | **Zero draft KV overhead** via shared-KV draft head |
| **ForeLen** | `--forelen` | Output length prediction is inaccurate, causing early truncation | **−29% MAE** vs TRAIL baseline |
| **RASD** | `--rasd` | Draft models unfamiliar with corpus vocab fail spec decode | **40–60% hit rate** via retrieval-augmented draft tree |

Full stack:

```bash
squish run qwen3:8b \
  --dovetail --pipo \
  --mobile-moe --online-sd \
  --lookahead-reasoning --sparse-spec \
  --fr-spec --long-spec \
  --forelen --rasd
```

Benchmark results: [`docs/benchmark_wave15_16.md`](docs/benchmark_wave15_16.md)
Raw data: [`dev/results/wave15_16_bench.json`](dev/results/wave15_16_bench.json)

---

## v5 — Wave 17: Attention Architecture

v5 (Wave 17) focuses on **INT4/INT8 attention kernels**, **slab-allocated KV storage**, **joint 2D KV budget management**, and **context-aware speculative prefetching**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **SageAttention2** | `--sage-attn2` | Full-precision attention is bandwidth-bound for long sequences | **INT4/INT8 warp-tile quantisation** · 672 µs forward (4h/seq32/d64) |
| **StreamingSink** | `--streaming-sink` | Unbounded KV growth at long contexts | **Attention-sink eviction** — bounded memory at any context length |
| **KVSlab** | `--kv-slab` | Per-token malloc/free causes fragmentation under scale | **Pre-allocated slab allocator** · 0.87 µs alloc+free round-trip |
| **SqueezeAttention** | `--squeeze-attn` | Independent token/layer KV compression compounds quality loss | **Joint 2D Pareto-optimal budget** allocation across both axes |
| **SmallKV** | `--small-kv` | Aggressive KV compression degrades small-model quality | **Saliency-compensated** recall · 39 µs ingest · 8 µs check-and-recall |
| **SpeContext** | `--spe-context` | Speculative decode wastes context retrieval at each draft step | **Cosine-similarity context cache** · 3.3 ms retrieve top-32 |
| **SVDq** | `--svdq` | Uniform K quantisation ignores per-head SVD structure | **Head-wise mixed-precision K search** · 62 ms one-time calibration |
| **CommVQ** | `--comm-vq` | Per-layer VQ codebooks waste memory building near-identical codebooks | **Shared communal codebook** · 55 µs encode · 68 µs decode |
| **ChunkedPrefill** | `--chunked-prefill` | Long prefills block decoding requests for the full context length | **Interleaved chunked prefill** — bounded latency per chunk |
| **GemFilter** | `--gemfilter` | KV eviction without attention-score feedback drops important tokens | **Top-K attention-score selector** · 0.90× cR · 50 µs select |
| **MInferencePatch** | `--minference` | Full O(n²) attention is infeasible for 1M+ token contexts | **Dynamic sparse patterns** — sub-quadratic attention at ultra-long context |
| **PromptCompressor** | `--prompt-compress` | Long system prompts and RAG context waste prefill FLOPs | **TF-IDF sentence-level compression** · 686 µs for 50-sentence input |
| **PromptLookup** | `--prompt-lookup` | No-draft-model baseline has no spec-decode path | **N-gram copy speculation** from prompt · 0.8 µs find · 3.3 µs push |
| **TRAIL** | `--trail` | Output-length prediction is too slow for real-time SRPT scheduling | **Linear-probe predictor** · 10 µs predict · feeds SRPT priority queue |

Full stack:

```bash
squish run qwen3:8b \
  --sage-attn2 --streaming-sink --kv-slab \
  --squeeze-attn --small-kv --spe-context \
  --svdq --comm-vq --chunked-prefill \
  --gemfilter --minference \
  --prompt-compress --prompt-lookup --trail
```

---

## v5 — Wave 18: Adaptive Compute

v5 (Wave 18) focuses on **vector-product quantisation**, **confidence-gated early exit**, **online domain adaptation**, and **energy-aware scheduling**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|----------------|------------|
| **VPTQ** | `--vptq` | Scalar quantisation loses intra-vector correlations | **Vector-product tree quant** · 15 µs decode · 133 ms one-time compress |
| **LayerSkip** | `--layer-skip` | All tokens pass through all layers regardless of difficulty | **Confidence-gated early exit** · 266 µs estimate · exit at threshold=0.85 |
| **SWIFT** | `--swift` | All FFN layers execute even when weights are functionally redundant | **Calibration-based FFN skip** · 162 µs calibrate · 34% layers skipped |
| **SpecReason** | `--spec-reason` | Reasoning chains serialise draft+verify round trips | **Pipelined draft+target step** · 6.6 µs per orchestrated step |
| **MirrorSD** | `--mirror-sd` | Single-draft spec-decode misses acceptance bursts | **Mirror pipeline** (parallel draft branches) · 867 µs step vocab=32k |
| **SparseVerify** | `--sparse-verify` | Re-verifying identical KV slices across draft iterations wastes compute | **Inter-draft KV reuse cache** · 0.28 µs query · near-zero overhead |
| **RobustScheduler** | `--robust-sched` | Priority inversions under bursty load hurt P99 latency | **A-balanced SRPT scheduler** · 3.7 µs schedule 32 requests |
| **BlockExpertArchive** | `--block-expert` | MoE expert routing is static and disk-bandwidth-bound | **Archived block-expert router** · 73 µs route 8 experts |
| **DISCRouter** | `--disc-router` | Monolithic inference ignores sub-task decomposition opportunities | **Decomposed sub-task planner** · 22.9 µs plan · 3.1 µs execute |
| **SelfLearning** | `--self-learning` | Domain adaptation requires expensive LoRA fine-tuning runs | **LoRA-free online delta absorption** · 6 ms per 4-example learn step |
| **SemanticCache** | `--semantic-cache` | Repeated semantically-equivalent queries re-run full inference | **sqlite-vec semantic cache** · short-circuit on cosine similarity hit |
| **IPW** | `--ipw` | No per-inference energy accounting available on-device | **Perf-per-watt tracker** · 0.16 µs record · 4.6 ms full summary |
| **PowerMonitor** | `--power-monitor` | Compute policy ignores battery vs. AC power state | **Apple Silicon power advisor** · 0.5 µs get recommended mode |
| **DiffusionDraft** | `--diffusion-draft` | AR draft models cannot exploit diffusion-model parallel generation | **Diffusion draft head** · availability gate for suitable tasks |

Full stack:

```bash
squish run qwen3:8b \
  --vptq --layer-skip --swift \
  --spec-reason --mirror-sd --sparse-verify \
  --robust-sched --block-expert --disc-router \
  --self-learning --semantic-cache \
  --ipw --power-monitor --diffusion-draft
```

Benchmark results: [`docs/benchmark_wave17_18.md`](docs/benchmark_wave17_18.md)
Raw data: [`dev/results/wave17_18_bench.json`](dev/results/wave17_18_bench.json)

---

## v6 — Wave 19: Next-Gen Attention & Precision

v6 (Wave 19) focuses on **FP8/MX microscaling quantisation**, **paged KV caching**, **GQA and sliding window attention**, **RoPE context extension**, and **multi-head speculative decoding (MEDUSA, EAGLE-3)**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|---------------|-----------|
| FP8Quant | `--fp8-quant` | Weight storage overhead | ~60% storage vs BF16 |
| MXQuant | `--mx-quant` | Quantisation quality at low bits | Better quality than INT4 at same bits |
| FlashDecode | `--flash-decode` | KV read parallelism at decode | O(1) memory overhead per step |
| PagedKV | `--paged-kv` | KV fragmentation across requests | Zero KV fragmentation |
| GQA | `--gqa` | KV memory per head | 4–8× KV reduction vs MHA |
| SlidingWindowAttn | `--sliding-window` | Memory at long context | O(window_size) memory |
| RoPEScaling | `--rope-scaling` | Context extension without fine-tuning | 4–32× context extension |
| ActSparsity | `--act-sparsity` | FFN compute on sparse activations | 30–60% FFN compute saved |
| FusedRMSNorm | `--fused-norm` | LayerNorm bandwidth | Single kernel pass |
| LoRAInference | `--lora-inference` | Adapter switching overhead | Zero-copy, no re-quant |
| MEDUSA | `--medusa` | Decode throughput | 2–3× decode throughput |
| EAGLE3 | `--eagle3` | Draft acceptance rate | 3.5× accept rate vs token-prediction |
| PrefixPool | `--prefix-pool` | KV recomputation on shared prompts | 40–80% KV savings |
| TokenHealer | `--token-healer` | Prefix token boundary artifacts | Eliminates prefix artifacts |

Full stack:
```bash
squish serve ./model \
  --fp8-quant --mx-quant \
  --flash-decode --paged-kv \
  --gqa --sliding-window \
  --rope-scaling ntk \
  --medusa --eagle3 \
  --prefix-pool --token-healer
```

---

## v6 — Wave 20: Serving Infrastructure & Intelligence

v6 (Wave 20) focuses on **model merging**, **multi-LoRA composition**, **continuous batching**, **constrained decoding**, and **vision token compression**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|---------------|-----------|
| ModelMerge | `--model-merge` | Combining domains without retraining | SLERP/DARE/TIES merging |
| LoRACompose | `--lora-compose` | Multi-adapter blending | Learnable composition coefficients |
| ContinuousBatching | `--continuous-batching` | GPU utilization at variable request rate | Max GPU utilization |
| MatryoshkaEmb | `--matryoshka-emb` | Embedding dim flexibility | 1 forward pass, any dimensionality |
| ANEProfiler | `--ane-profiler` | ANE vs GPU op breakdown | Op-level ANE utilization |
| SpecBench | `--spec-bench` | Speculative decode CI | Acceptance rate + throughput |
| PPLTracker | `--ppl-tracker` | Quantisation quality degradation | Real-time PPL monitoring |
| GrammarCache | `--grammar-cache` | Per-token FSM rebuild overhead | Zero rebuild on cached grammars |
| QuantAware | `--quant-aware` | Scale selection for quantisation | Per-channel optimal scales |
| AdaptiveBudget | `--adaptive-budget` | Joint KV + layer skip SLO control | SLO-aware compute budget |
| VisionTokens | `--vision-tokens` | Visual token overhead in VLMs | 50–80% vision token reduction |
| ToolCache | `--tool-cache` | Tool schema parse overhead | Zero parse overhead on repeats |
| DistilSpec | `--distil-spec` | Draft head acceptance rate | +10–15 pp from calibration |
| BatchEmbed | `--batch-embed` | Embedding pooling strategy | mean/max/cls/weighted in one pass |

Full stack:
```bash
squish serve ./model \
  --continuous-batching \
  --grammar-cache \
  --adaptive-budget \
  --vision-tokens \
  --tool-cache \
  --distil-spec \
  --batch-embed mean
```

Benchmark results: [`docs/benchmark_wave19_20.md`](docs/benchmark_wave19_20.md)

---

## v7 — Wave 21: Advanced Memory & Decode

v7 (Wave 21) focuses on **tree-parallel speculative verification**, **online KV compression**, **mixed-precision per-head KV**, **pipeline bubble elimination**, **learned KV codecs**, and **retention-style recurrent attention**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|---------------|-----------|
| TreeVerifier | `--tree-verify` | Speculative tree acceptance | Structured multi-token acceptance |
| KVCompress | `--kv-compress` | KV memory growth during generation | Online prune + INT8 quant |
| DynamicNTK | `--dynamic-ntk` | Context extension without retraining | Auto-extends at 80% context fill |
| QuantSpecDecode | `--quant-spec-decode` | Draft memory overhead | 4× draft memory reduction vs FP16 |
| SparseAttnIndex | `--sparse-attn-index` | Attention cost at very long context | Sub-linear KV attention cost |
| MixedPrecisionKV | `--mp-kv` | KV memory at iso-quality | 2–4× KV reduction via per-head precision |
| PipelineBubble | `--pipeline-bubble` | Pipeline stage idle time | 1F1B near-zero bubble fraction |
| LayerwiseDecode | `--layerwise-decode` | Full-depth decode latency | Early-exit at configurable layer |
| CodecKV | `--codec-kv` | KV cache memory | 204× compression via learned codebook |
| DedupeAttn | `--dedupe-attn` | Attention FLOPs on repetitive context | Near-duplicate Q/K output reuse |
| FlashPrefill | `--flash-prefill` | Prefill memory on long sequences | O(seq × chunk) not O(seq²) |
| BudgetSpec | `--budget-spec` | Draft compute near token budget | Ramp-down to 1 draft near limit |
| RetentionAttn | `--retention-attn` | KV cache memory for recurrent inference | O(1) per-step linear recurrence |
| KVRouter | `--kv-router` | KV recomputation in disaggregated serving | Zero-recompute cross-instance routing |

Full stack:
```bash
squish serve ./model \
  --tree-verify --kv-compress \
  --dynamic-ntk \
  --quant-spec-decode \
  --sparse-attn-index --mp-kv \
  --pipeline-bubble --layerwise-decode \
  --codec-kv --dedupe-attn \
  --flash-prefill --budget-spec \
  --retention-attn --kv-router
```

---

## v7 — Wave 22: Production Serving & Observability

v7 (Wave 22) focuses on **multi-tenant fair scheduling**, **load-balanced request routing**, **predictive KV pre-warming**, **OpenTelemetry-compatible tracing**, **adaptive quantisation under pressure**, and **SLA violation detection**, shipping 14 new modules:

| Module | Flag | Problem Solved | Key Number |
|--------|------|---------------|-----------|
| MultiTenantSched | `--multi-tenant` | Per-tenant QoS isolation | 0.65 µs scheduler overhead |
| RequestRouter | `--request-router` | Load balancing across replicas | Least-loaded routing |
| CacheWarmup | `--cache-warmup` | Cold TTFT on hot paths | Predictive KV pre-warming |
| TokenBudgetGate | `--token-budget` | Request cost determinism | Hard budget with graceful truncation |
| ObservabilityHook | `--observability` | Inference step visibility | OpenTelemetry-compatible spans |
| RequestCoalesce | `--req-coalesce` | Redundant prefill forward passes | Shared prefill for common prefixes |
| AdaptiveQuantize | `--adaptive-quant` | Memory pressure OOM risk | Auto INT8/INT4 under pressure |
| HealthCheck | `--health-check` | Quality regression detection | p50/p99 latency + error rate |
| FaultTolerance | `--fault-tolerance` | OOM crash risk | Progressive evict→disable→reduce |
| ModelPool | `--model-pool` | Multi-model reload latency | Hot pool with LRU eviction |
| StreamingChunk | `--streaming-chunk` | First-chunk streaming latency | Sub-token chunked streaming |
| CostEstimator | `--cost-estimate` | Request billing and prioritisation | Per-request compute cost |
| SLAMonitor | `--sla-monitor` | SLA breach detection | Auto-escalation on consecutive breach |
| ContextCache | `--context-cache` | Cross-session context re-encoding | Persistent TTL cache, 100% hit rate |

Full stack:
```bash
squish serve ./model \
  --multi-tenant --request-router \
  --cache-warmup --token-budget \
  --observability --req-coalesce \
  --adaptive-quant --health-check \
  --fault-tolerance --model-pool \
  --streaming-chunk --cost-estimate \
  --sla-monitor --context-cache
```

Benchmark results: [`docs/benchmark_wave21_22.md`](docs/benchmark_wave21_22.md)

---

## v8 — Wave 23: Multi-Modal & Long Context Intelligence

Wave 23 ships 14 new modules covering vision KV fusion, image/video token pruning, RAG prefetching, CoT compression, cross-modal attention, and hierarchical KV caching:

VisionKVFuse · ImageTokenPrune · RAGPrefetch · CoTCompress · MultiModalBatch · ContextualRerank · CrossModalAttn · HierarchicalKV · StreamRAG · CrossDocAttn · VideoFramePrune · EmbeddingGate · LongContextChunk · ModalityRouter

Key numbers: 50–70% image token pruning · 60–80% video token pruning · 30–50% CoT reduction.

Benchmark results: [`docs/benchmark_wave23_24.md`](docs/benchmark_wave23_24.md)
Raw data: [`dev/results/wave23_24_bench.json`](dev/results/wave23_24_bench.json)

---

## v8 — Wave 24: Quantisation Evolution & Model Surgery

Wave 24 ships 14 new modules covering ternary/binary quantisation, structured pruning, layer fusion, weight sharing, and model surgery:

TernaryQuant · BinaryAttn · StructuredPrune · LayerFusion · WeightSharing · QuantCalib · SparseWeight · DeltaCompress · ModelSurgery · ZeroQuantV2 · GPTQLayer · SparseMoE · AWQv2 · IterPrune

Key numbers: 1.58-bit ternary weights · 2:4 structured sparsity · 7.98× SVD delta compression.

---

## v9 — Wave 25: Cutting-Edge Attention Variants & Compute Fusion

Wave 25 ships 14 new modules covering FlashMLA, native sparse attention, fused sampling, KV defragmentation, dual-chunk attention, activation offload, morphing attention, hydra-speculation, sequence compaction, and hardware-aware scheduling:

FlashMLA · NativeSparseAttn · FusedSampler · KVDefrag · DualChunkAttn · ActivationOffload · MorphAttn · HydraSpec · SeqCompact · LatencyPredictor · ParallelSampler · ContextSummarizer · TokenWatermark · SchemaGen

Key numbers: FlashMLA 4× KV compression · NSA ~87% attention sparsity · HydraSpec multi-draft speculation.

Benchmark results: [`docs/benchmark_wave25_26.md`](docs/benchmark_wave25_26.md)
Raw data: [`dev/results/wave25_26_bench.json`](dev/results/wave25_26_bench.json)

---

## v9 — Wave 26: Distributed Inference & Production Reliability

Wave 26 ships 14 new modules covering tensor parallelism, sequence parallelism, KV migration, disaggregated prefill/decode, request preemption, inference gateway, model version swapping, production profiling, adaptive batching, safety layer, semantic response cache, rate limiting, schema validation, and audit logging:

TensorParallel · SequenceParallel · KVMigrate · DisaggPrefill · RequestPreempt · InferGateway · ModelVersionSwap · ProductionProfiler · AdaptiveBatcher · SafetyLayer · SemanticResponseCache · RateLimiter · SchemaValidator · AuditLogger

Key numbers: disaggregated prefill/decode · SHA-256 audit log · sub-200ns APM record.
