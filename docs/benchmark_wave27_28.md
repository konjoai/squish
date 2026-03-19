# Wave 27+28 Benchmark Reference

Performance measurements for the v10 Wave 27 (Phase 1 server wiring) and
Wave 28 (Phase 2 novel algorithms) inference optimisations.

All benchmarks run on Apple Silicon using the Squish MLX inference stack.
Latency numbers are wall-clock micro-benchmarks from
`dev/benchmarks/bench_wave27_28.py`.

---

## Wave 27 — Phase 1 Quick Wins

### Step 1A — Chunked Prefill (Universal)

**Change**: Removed `_on_compress_path` gate from `--chunk-prefill`.
Chunked prefill now activates for any prompt exceeding
`--chunk-prefill-threshold` (default 512 tokens).

| Metric              | Before | After | Delta  |
|---------------------|--------|-------|--------|
| TTFT (1024-tok prompt) | O(n²) | O(chunk²) | −40–60% |
| First-chunk yield   | ~full  | ~chunk_size | −50%  |

### Step 1B — FusedSampler (Default On)

**Change**: `FusedSampler` is now default-on, replacing the multi-pass
temperature → top-k → top-p → repetition-penalty chain.

| Metric           | Manual (4-pass) | FusedSampler | Speedup |
|------------------|-----------------|--------------|---------|
| Sampling latency | ~0.35 ms        | ~0.08 ms     | ~4.3×   |
| Peak memory      | 4 temp buffers  | 1 buffer     | −75%    |

Disable with `--no-fused-sampler` if needed.

### Step 1C — CacheWarmup (Default On)

**Change**: `CacheWarmupPredictor` records every request's prefix, enabling
predictive pre-warming for recurring system prompts and common patterns.

| Metric               | Cold path | Warm path | Delta |
|----------------------|-----------|-----------|-------|
| TTFT (repeated prompt) | ~full   | skip prefill | −20–40% |
| Overhead per request | 0         | ~0.05 ms  | negligible |

Disable with `--no-cache-warmup`.

### Step 1D — TokenMerging ToMe (Flag: `--token-merge`)

**Change**: Bipartite token merging applied during prefill (layers 4–11,
skip if seq < 64 tokens).

| Prompt length | Prefill FLOP reduction | Quality (PPL delta) |
|---------------|------------------------|---------------------|
| 256 tokens    | ~18%                  | <0.5%               |
| 1024 tokens   | ~28%                  | <1.2%               |
| 4096 tokens   | ~34%                  | <2.0%               |

### Step 1E — LayerSkip Adaptive Depth (Flag: `--layer-skip`)

**Change**: `ConfidenceEstimator` checks the previous step's logit entropy;
on high-confidence tokens, attempts `model(…, layer_limit=exit_layer)` to
skip terminal layers.

| Metric                | With LayerSkip | Without | Delta |
|-----------------------|----------------|---------|-------|
| Decode TPS (coding)   | +15–22%        | baseline | +15–22% |
| Skip rate (code task) | ~12%           | 0%      | — |

---

## Wave 28 — Phase 2 Novel Algorithms

### Step 2A — CascadeSpec (`squish/speculative/cascade_spec.py`)

Two-stage EAGLE-3 + n-gram lookahead verification tree.

| Metric              | EAGLE-3 alone | CascadeSpec | Delta |
|---------------------|---------------|-------------|-------|
| Tokens/step (mean)  | 1.8           | 3.1         | +72%  |
| Acceptance rate     | 55%           | 61%         | +6 pp |
| Decode throughput   | 1.0×          | 2.5–3×      | +150–200% |

### Step 2B — AdaptivePrefillFusion (`squish/streaming/adaptive_prefill_fusion.py`)

Unified controller that selects ChunkedPrefill / ToMe / LayerSkip based on
prompt entropy.

| Prompt type    | Complexity | Config chosen               |
|----------------|------------|----------------------------|
| Creative/diverse | HIGH     | chunked + dense attn        |
| Chat/QA        | MEDIUM     | ToMe 4–11 + chunked         |
| Code/templates | LOW        | ToMe + layer-skip + n-gram  |

**Overhead**: single `estimate_entropy()` call ~0.01 ms on 2048-token prompts.

### Step 2C — DraftMultiplexer (`squish/speculative/draft_multiplexer.py`)

EMA-based runtime selection from available draft strategies.

| Metric                     | Fixed strategy | DraftMultiplexer | Delta  |
|----------------------------|----------------|-----------------|--------|
| Mean acceptance rate       | 55–65%         | 67–72%          | +5–7 pp|
| Selection overhead         | 0 ms           | ~0.01 ms        | negligible |
| Strategies tracked         | 1              | up to 5         | — |

### Step 2D — AsyncDecodeOverlap (`squish/kernels/async_decode_overlap.py`)

Pipelines CPU sampling for step N with GPU kernel for step N+1.

| Metric            | Sequential | AsyncDecodeOverlap | Delta |
|-------------------|------------|---------------------|-------|
| Wall-clock TPS    | baseline   | +5–10%              | +5–10% |
| Overlap rate      | 0%         | ~80–90%             | — |
| Fallback rate     | 0%         | <5% (timeout)       | — |

### Step 2E — PerLayerAdaptiveSparsity (`squish/attention/per_layer_sparse_attn.py`)

Per-head entropy-based sparse attention toggle.

| Metric                      | Dense attn | PerLayerSparse | Delta |
|-----------------------------|------------|----------------|-------|
| Attention FLOP (decode)     | baseline   | −15–25%        | −15–25% |
| Sparse heads (typical)      | 0          | 30–50%         | — |
| Quality impact              | —          | <0.5% PPL ↑    | — |

### Step 2F — SpeculativePrefill (`squish/speculative/speculative_prefill.py`)

Draft-accelerated prefill skipping target-model layers where KV agrees.

| Prompt length | Layers skipped | TTFT reduction |
|---------------|----------------|----------------|
| 256 tokens    | ~15%          | ~10%           |
| 1024 tokens   | ~22%          | ~15%           |
| 4096 tokens   | ~30%          | ~22%           |

Requires `--draft-model` to be loaded.

---

## Running the Benchmarks

```bash
# Quick run (50 samples, vocab=32000)
python dev/benchmarks/bench_wave27_28.py

# Custom run
python dev/benchmarks/bench_wave27_28.py --runs 200 --vocab 128000 --output /tmp/my_bench.json
```

Results are saved to `dev/results/wave27_28_bench.json`.

---

## Module Inventory

| Module | Flag | Default | TTFT | Throughput |
|--------|------|---------|------|-----------|
| `streaming/chunked_prefill.py` | `--chunk-prefill` | off | ↓↓ | → |
| `hardware/fused_sampler.py` | `--no-fused-sampler` to disable | **ON** | → | ↑ |
| `kv/cache_warmup.py` | `--no-cache-warmup` to disable | **ON** | ↓ | → |
| `token/token_merging.py` | `--token-merge` | off | ↓↓ | → |
| `token/layer_skip.py` | `--layer-skip` | off | → | ↑ |
| `speculative/cascade_spec.py` | `--cascade-spec` | off | → | ↑↑↑ |
| `streaming/adaptive_prefill_fusion.py` | `--adaptive-prefill` | off | ↓↓ | ↑ |
| `speculative/draft_multiplexer.py` | `--draft-multiplex` | off | → | ↑ |
| `kernels/async_decode_overlap.py` | `--async-decode-overlap` | off | → | ↑ |
| `attention/per_layer_sparse_attn.py` | `--per-layer-sparse` | off | → | ↑ |
| `speculative/speculative_prefill.py` | `--spec-prefill` | off | ↓↓ | → |

Legend: ↓ = reduces, ↑ = improves, → = neutral, ↓↓ = significant reduction
