# Squish Module Reference — Wave 27+28 (v10)

This document describes the six new algorithm modules added in v10 (Wave 28)
and the five server-wiring changes from Wave 27.

For the full module inventory across all waves, see the versioning table in
[docs/planning/PLAN.md](docs/planning/PLAN.md).

---

## Wave 27 — Server Wiring Quick Wins

All five changes are in `squish/server.py`. They wire pre-existing modules into
the live request path with minimal overhead.

### 1A — Chunked Prefill (Universal)
**File**: `squish/streaming/chunked_prefill.py`
**Flag**: `--chunk-prefill` (off by default; `--chunk-prefill-threshold N`)
**Change**: Removed the `_on_compress_path` gate so chunked prefill works on
every request path, not just compressed-weight paths.
**Impact**: TTFT −40–60% on prompts > threshold (default 512 tokens).

### 1B — FusedSampler Default-On
**File**: `squish/hardware/fused_sampler.py`
**Flag**: enabled by default; disable with `--no-fused-sampler`
**Change**: FusedSampler (fused temperature/top-k/top-p/min-p/rep-penalty) is
now the default decode-step sampler, replacing the 4-pass manual chain.
**Impact**: Sampling latency ~0.35 ms → ~0.08 ms (~4× faster).

### 1C — CacheWarmupPredictor Wired
**File**: `squish/kv/cache_warmup.py`
**Flag**: enabled by default; disable with `--no-cache-warmup`
**Change**: `record_access(input_ids[:256], timestamp)` is called after
tokenization on every request, enabling predictive pre-warming for repeat
system prompts and frequent prefixes.
**Impact**: TTFT −20–40% on repeated prefixes (system prompt reuse, chat turns).

### 1D — TokenMerging Patch/Unpatch
**File**: `squish/token/token_merging.py`
**Flag**: `--token-merge` (off by default)
**Change**: `patch_model_tome()` / `unpatch_model_tome()` are called around the
standard prefill model call for sequences ≥ 64 tokens (layers 4–11).
**Impact**: Prefill FLOP −18–34% depending on sequence length; PPL delta < 2%.

### 1E — LayerSkip Adaptive Depth
**File**: `squish/token/layer_skip.py`
**Flag**: `--layer-skip` (off by default)
**Change**: `ConfidenceEstimator` is initialised once per request; each decode
step estimates logit entropy and attempts `model(x, layer_limit=exit_layer)`
when confidence exceeds threshold. Fallback to full model on `TypeError`.
**Impact**: Decode TPS +15–22% on high-confidence generation tasks.

---

## Wave 28 — Novel Algorithm Modules

### cascade_spec.py
**Path**: `squish/speculative/cascade_spec.py`
**Flag**: `--cascade-spec`
**Purpose**: Two-stage speculative decoding combining an EAGLE-3 tree draft
with n-gram lookahead extension.

**Key classes**:
| Class | Role |
|-------|------|
| `CascadeSpecConfig` | Dataclass holding `eagle_depth`, `ngram_extend`, `ngram_order`, `temperature` |
| `CascadeSpecDecoder` | Main decoder; `.generate(prompt_ids, max_new_tokens, eos_id)` |
| `CascadeSpecStats` | Latency / acceptance-rate counters |

**Algorithm**:
1. EAGLE-3 tree draft builds candidate tokens from a heuristic head (or loaded
   EAGLE-3 head via `set_eagle_head()`).
2. N-gram lookahead extends each tree leaf by `ngram_extend` positions.
3. Full model verifies the tree; greedy-accept prefix up to first mismatch.
4. Stats track `mean_accept_len` and `draft_calls` per generation.

**Expected throughput**: 2.5–3× vs greedy decode on typical prompts.

---

### adaptive_prefill_fusion.py
**Path**: `squish/streaming/adaptive_prefill_fusion.py`
**Flag**: `--adaptive-prefill`
**Purpose**: Classifies prompt complexity from token-frequency entropy and
returns a `PrefillPlan` describing which prefill optimisations to enable.

**Key classes**:
| Class | Role |
|-------|------|
| `PrefillComplexity` | `HIGH` / `MEDIUM` / `LOW` enum |
| `PrefillFusionConfig` | Entropy thresholds + per-complexity settings |
| `PrefillPlan` | Output: `use_chunked`, `use_tome`, `use_layer_skip`, `use_ngram` |
| `PrefillFusionController` | `.plan(token_ids) → PrefillPlan` |

**Complexity routing**:
- **HIGH** (diverse/creative): chunked prefill only; no ToMe (entropy too high)
- **MEDIUM** (chat/QA): ToMe (layers 4–11) + chunked prefill
- **LOW** (code/templates): ToMe + LayerSkip + n-gram lookahead

**Overhead**: single entropy estimation pass ~0.01 ms on 2048-token prompts.

---

### draft_multiplexer.py
**Path**: `squish/speculative/draft_multiplexer.py`
**Flag**: `--draft-multiplex`
**Purpose**: Selects the best available draft strategy at runtime using
per-task EMA acceptance rates and throughput scores.

**Key classes**:
| Class | Role |
|-------|------|
| `DraftStrategy` | `NGRAM` / `EAGLE` / `MEDUSA` / `HYDRA` / `CASCADE` enum |
| `DraftTaskType` | `CODING` / `MATH` / `RAG` / `CONVERSATION` / `UNKNOWN` |
| `DraftMultiplexerConfig` | EMA alpha, cost weight, min samples before EMA |
| `StrategyStats` | Per-strategy `acceptance_rate`, `tps`, `n_samples` |
| `DraftMultiplexer` | `.select(prompt) → DraftStrategy`; `.update(strategy, task_type, rate, tps)` |

**Selection logic**:
- Round-robin during init phase (< `min_samples` per strategy)
- Regex task classifier: coding/math/RAG/conversation patterns
- EMA score = `acceptance_rate + cost_weight × normalised_tps`
- Highest score among available strategies wins

**Expected gain**: +5–7 pp acceptance rate vs fixed strategy selection.

---

### async_decode_overlap.py
**Path**: `squish/kernels/async_decode_overlap.py`
**Flag**: `--async-decode-overlap`
**Purpose**: Pipelines CPU sampling computation for step N with the GPU
(Metal) kernel for step N+1 using a background thread and queue.

**Key classes**:
| Class | Role |
|-------|------|
| `OverlapConfig` | `timeout_ms`, `max_queue_depth`, `fallback_sync` |
| `AsyncDecodeOverlap` | `.decode_loop(model_forward, first_token_id, max_tokens, eos_id) → Generator[int]` |
| `OverlapStats` | `overlap_steps`, `fallback_steps`, `timeout_steps` |

**Algorithm**:
- Step N logits sent to background thread for `_sample_np` (numpy argmax/top-k)
- GPU launches step N+1 kernel while background thread samples step N
- `queue.SimpleQueue` passes sampled tokens back; timeout forces sync fallback
- Overlap rate typically 80–90%; throughput gain +5–10% decoded TPS

---

### per_layer_sparse_attn.py
**Path**: `squish/attention/per_layer_sparse_attn.py`
**Flag**: `--per-layer-sparse`
**Purpose**: Profiles attention head entropy during prefill, then applies a
per-head sparse attention mask during decode for low-entropy (predictable) heads.

**Key classes**:
| Class | Role |
|-------|------|
| `PerLayerSparseConfig` | `entropy_threshold`, `warmup_steps`, `ema_alpha`, `n_layers`, `n_heads` |
| `HeadProfile` | Per-head EMA entropy + `is_sparse` flag |
| `PerLayerSparseAttn` | `.profile_prefill(attn_weights_4d)` → `.sparse_mask(layer) → bool[n_heads]` |

**Algorithm**:
- During prefill: compute entropy of `mean_over_queries(attn_weights)` per head
- EMA-smooth across requests: `ema = alpha * new + (1-alpha) * old`
- After `warmup_steps`: heads with `ema_entropy < entropy_threshold` → `is_sparse = True`
- Decode: `sparse_mask(layer)` returns bitmask for caller to skip compute

**Expected reduction**: 15–25% attention FLOP in decode on typical prompts;
quality impact < 0.5% PPL increase.

---

### speculative_prefill.py
**Path**: `squish/speculative/speculative_prefill.py`
**Flag**: `--spec-prefill` (requires `--draft-model`)
**Purpose**: Reduces TTFT by running a draft model over the full prompt to
produce KV states, then having the target model only recompute layers where
the KV diverges (cosine similarity below threshold).

**Key classes**:
| Class | Role |
|-------|------|
| `SpecPrefillConfig` | `similarity_threshold`, `max_skip_rate`, `chunk_size` |
| `SpecPrefillStats` | `skip_rate`, `speedup_estimate`, `recompute_layers` |
| `SpeculativePrefiller` | `.prefill(token_ids) → (kv_states, stats)` |

**Algorithm**:
1. Draft model forward pass produces KV for all layers
2. Consecutive-layer cosine similarity of K matrices used as KV-agreement proxy
3. Layers with similarity ≥ threshold are marked for skipping
4. `recompute_mask` passed to target forward; target only runs unmasked layers
5. `speedup_estimate = 1 / (1 − skip_rate)`

**Expected TTFT reduction**: 10% (256 tok) → 22% (4096 tok) when draft and
target share architecture.

---

## Testing

| Test file | Tests | Status |
|-----------|------:|-------|
| `tests/test_wave27_server_wiring.py` | 33 | ✅ passing |
| `tests/test_wave28_server_wiring.py` | 77 | ✅ passing |
| Full suite | 7,672 | ✅ passing |

## Benchmarking

```bash
python dev/benchmarks/bench_wave27_28.py [--runs N] [--vocab N] [--output path]
```

Results saved to `dev/results/wave27_28_bench.json`.
Reference table: [docs/benchmark_wave27_28.md](docs/benchmark_wave27_28.md).
