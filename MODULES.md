# Squish Module Reference

> Historical per-wave module log. "Wave" numbers and the old `vNN.0.0` scheme are
> the internal development cadence; the current public release is **v9.34.2** (see
> [CHANGELOG.md](CHANGELOG.md)). For waves 1–28, see the historical record below.

---

## Waves 85–95 Summary (v58.0.0–v68.0.0)

| Wave | Version | Theme | Key Files |
|------|---------|-------|-----------|
| 85 | 58.0.0 | CLI color dedup + README accuracy | `cli.py`, `server.py`, `api/v1_router.py` |
| 86 | 59.0.0 | Observability: ProductionProfiler + `squish trace` | `hardware/production_profiler.py`, `serving/obs_report.py`, `cli.py` |
| 87 | 60.0.0 | VS Code/Web UI agent tool execution fix | `serving/tool_calling.py`, `agent/tool_name_map.py`, `squishClient.ts` |
| 88 | 61.0.0 | Ollama gaps + LocalAI + `squish compat` | `serving/ollama_compat.py`, `serving/localai_compat.py`, `serving/backend_router.py` |
| 89 | 62.0.0 | Local model scanner + `squish pull` URI schemes | `serving/local_model_scanner.py`, `cli.py` |
| 90 | 63.0.0 | Lean startup profiler + server.py decomposition | `serving/startup_profiler.py`, `serving/feature_state.py`, `serving/blazing.py` |
| 91 | 64.0.0 | Sub-3s TTFT (blazing default) + 70B loader | `server.py`, `cli.py`, `catalog.py`, `serving/blazing.py` |
| 92 | 65.0.0 | Pre-compress pipeline + HF batch upload | `catalog.py`, `dev/scripts/upload_to_hub.py`, `.github/workflows/model_upload.yml` |
| 93 | 66.0.0 | macOS SquishBar: model picker, progress, hotkey | `apps/macos/SquishBar/Sources/SquishBar/SquishEngine.swift`, `SquishMenuView.swift`, `Makefile` |
| 94 | 67.0.0 | Cross-platform support review | `platform/detector.py`, `platform/platform_router.py`, `cli.py`, `README.md` |
| 95 | 68.0.0 | README final audit + public release | `README.md`, `MODULES.md`, `cli.py`, `squish/__init__.py` |

---

## Waves 57–83 (v9.0.0–v9.14.0: Compliance Layer, now squash-ai)

These waves implemented the EU AI Act, NIST AI RMF, and enterprise compliance features. That code now lives in the standalone **[konjoai/squash](https://github.com/konjoai/squash)** repository (`pip install squash-ai`). It is no longer part of the squish package.

---

## Wave 85: CLI Color Dedup + README Accuracy (v58.0.0)

Consolidated three duplicate terminal palette implementations into a single
`squish/_term.py` source of truth.  `cli.py` and `server.py` now import from
`_term` instead of carrying their own copies.  Fixed hardcoded `localhost:11434`
port in `api/v1_router.py` default URL.

**Key changes:**
- `squish/cli.py`: removed local `_C`/`_CTerminal` classes; import from `squish._term`
- `squish/server.py`: removed duplicate `_gradient()`, `_LOGO_GRAD`, local `_C`
- `squish/api/v1_router.py`: default `server_url` reads `SQUISH_SERVER_URL` env or `localhost:11435`

---

## Wave 86: Observability: Profiler + `squish trace` (v59.0.0)

Wired `trace_span` into hot paths and instantiated `ProductionProfiler` at
server start.  Added `GET /v1/obs-report` endpoint and `squish trace` CLI
command with remediation hints.

**New file:** `squish/serving/obs_report.py`: `detect_bottlenecks()`,
`generate_report()`, `_REMEDIATION_HINTS` dict.

---

## Wave 87: Agent Tool Execution Fix (v60.0.0)

Fixed truncated `<tool_call>` tag parsing (Strategy 0.5 added before existing
strategies), normalized VS Code tool names via `agent/tool_name_map.py`, fixed
30-second timeout in `_toolRunTerminal`, and added agent mode toggle to Web UI.

**New file:** `squish/agent/tool_name_map.py`: `VSCODE_TO_BACKEND` dict,
`normalize_for_backend()`, `normalize_for_client()`.

---

## Wave 88: Drop-in Compat: Ollama + LocalAI (v61.0.0)

Implemented `/api/pull` streaming, `/api/ps`, `/api/version` (dynamic), and
other previously-stubbed Ollama endpoints.  Added LocalAI compatibility routes
(`GET /`, `GET /v1/version`, `GET /readyz`).  Added `squish compat` command
printing client configuration snippets.

**New files:** `squish/serving/localai_compat.py`, `squish/serving/backend_router.py`

---

## Wave 89: Local Model Scanner + URI Schemes (v62.0.0)

`LocalModelScanner` scans Squish, Ollama, and LM Studio model directories.
`squish models` shows an "External models detected" section.  `squish pull`
accepts `ollama:` and `hf:` URI prefixes.  `squish import` added as new command.

**New file:** `squish/serving/local_model_scanner.py`: `LocalModel` dataclass,
`scan_squish()`, `scan_ollama()`, `scan_lm_studio()`, `find_all()`.

---

## Wave 90: Lean Startup Profiler (v63.0.0)

`StartupTimer` context manager + `StartupReport` with `slowest()` / `to_dict()`,
enabled by `SQUISH_TRACE_STARTUP=1`.  `FeatureState` dataclass centralises all
`_xxx = None` server globals.  `BlazingPreset` / `auto_blazing_eligible` moved
to `serving/blazing.py`.

**New files:** `squish/serving/startup_profiler.py`, `squish/serving/feature_state.py`,
`squish/serving/blazing.py`

---

## Wave 91: Sub-3s TTFT + 70B Loader (v64.0.0)

Blazing mode auto-activates on M3/M4/M5 with ≥16 GB RAM (pass `--no-blazing`
to disable).  `cmd_run` auto-selects INT2/INT3 based on available RAM vs
model size.  `_recommend_model()` priority order fixed (was recommending llama3.3:70b
on 64+ GB machines).  `llama3.3:70b` catalog entry added with `squish_repo`.

---

## Wave 92: Pre-Compress Pipeline + HF Batch Upload (v65.0.0)

`dev/scripts/upload_to_hub.py` gained `--all-missing`, `--batch-file`, `--int2`,
`--force`, `--org` flags.  `catalog.py` `squish_repo` backfilled for 5 models.
GitHub Actions `model_upload.yml` workflow added for CI-triggered uploads.

---

## Wave 93: macOS SquishBar Polish (v66.0.0)

SquishBar gained: model picker with active-model checkmark (`switchModel()`),
"Pull Model…" button with live compression progress bar, global hotkey (`⌘⌥S`
default, configurable in Settings…), and `Makefile` `release` + `dmg` targets.
New `docs/squishbar.md` reference page.

---

## Wave 94: Cross-Platform Support (v67.0.0)

README title, badge, and Requirements section updated for multi-platform.
`cmd_setup()` no longer calls `sys.exit(1)` on non-Apple platforms; instead
detects backend via `get_inference_backend(detect_platform())` and prints
guidance.  `platform/` module verified with `is_apple_silicon`, `is_cuda`,
`name`, `platform_name`, and `get_inference_backend()` all confirmed present.

---

## Wave 95: Final Public Release Audit (v68.0.0)

`_CURRENT_WAVE = 95` constant added to `cli.py`.  `cmd_version` / `squish version`
subcommand prints version + wave from `importlib.metadata`.  README model count
updated to 40.  MODULES.md backfilled with Waves 85–95 summary.
CHANGELOG fully populated through v68.0.0.

---

---

# Historical Reference: Wave 27+28 (v10)

## Wave 27: Server Wiring Quick Wins

All five changes are in `squish/server.py`. They wire pre-existing modules into
the live request path with minimal overhead.

### 1A: Chunked Prefill (Universal)
**File**: `squish/streaming/chunked_prefill.py`
**Flag**: `--chunk-prefill` (off by default; `--chunk-prefill-threshold N`)
**Change**: Removed the `_on_compress_path` gate so chunked prefill works on
every request path, not just compressed-weight paths.
**Impact**: TTFT −40–60% on prompts > threshold (default 512 tokens).

### 1B: FusedSampler Default-On
**File**: `squish/hardware/fused_sampler.py`
**Flag**: enabled by default; disable with `--no-fused-sampler`
**Change**: FusedSampler (fused temperature/top-k/top-p/min-p/rep-penalty) is
now the default decode-step sampler, replacing the 4-pass manual chain.
**Impact**: Sampling latency ~0.35 ms → ~0.08 ms (~4× faster).

### 1C: CacheWarmupPredictor Wired
**File**: `squish/kv/cache_warmup.py`
**Flag**: enabled by default; disable with `--no-cache-warmup`
**Change**: `record_access(input_ids[:256], timestamp)` is called after
tokenization on every request, enabling predictive pre-warming for repeat
system prompts and frequent prefixes.
**Impact**: TTFT −20–40% on repeated prefixes (system prompt reuse, chat turns).

### 1D: TokenMerging Patch/Unpatch
**File**: `squish/token/token_merging.py`
**Flag**: `--token-merge` (off by default)
**Change**: `patch_model_tome()` / `unpatch_model_tome()` are called around the
standard prefill model call for sequences ≥ 64 tokens (layers 4–11).
**Impact**: Prefill FLOP −18–34% depending on sequence length; PPL delta < 2%.

### 1E: LayerSkip Adaptive Depth
**File**: `squish/token/layer_skip.py`
**Flag**: `--layer-skip` (off by default)
**Change**: `ConfidenceEstimator` is initialised once per request; each decode
step estimates logit entropy and attempts `model(x, layer_limit=exit_layer)`
when confidence exceeds threshold. Fallback to full model on `TypeError`.
**Impact**: Decode TPS +15–22% on high-confidence generation tasks.

---

## Wave 28: Novel Algorithm Modules

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
Reference table: see the per-wave entries below.

---

## Waves 85–95: Tooling + Platform Maturity (v58–v68)

| Wave | Version | Theme | New Files |
|------|---------|-------|-----------|
| 85 | 58.0.0 | CLI color dedup + README accuracy | — |
| 86 | 59.0.0 | Observability: profiler wiring + `squish trace` | `squish/serving/obs_report.py` |
| 87 | 60.0.0 | Agent tool execution fix | `squish/agent/tool_name_map.py` |
| 88 | 61.0.0 | Ollama/LocalAI compat gaps | `squish/serving/localai_compat.py`, `squish/serving/backend_router.py` |
| 89 | 62.0.0 | Local model scanner + `squish pull` URI schemes | `squish/serving/local_model_scanner.py` |
| 90 | 63.0.0 | Startup profiler + core module extraction | `squish/serving/startup_profiler.py`, `squish/serving/feature_state.py`, `squish/serving/blazing.py`, `dev/scripts/import_scan.py` |
| 91 | 64.0.0 | Sub-3s TTFT + 70B INT2 loader | — |
| 92 | 65.0.0 | Pre-compress pipeline + HF batch upload | — |
| 93 | 66.0.0 | macOS SquishBar polish | `docs/squishbar.md` |
| 94 | 67.0.0 | Cross-platform support review | — |
| 95 | 68.0.0 | README final audit + public release | — |

### Wave 90: Key New Modules

#### `squish/serving/startup_profiler.py`
Phase-level startup timing via `StartupTimer` context manager and `StartupReport`.
`SQUISH_TRACE_STARTUP=1` enables tracing; result accessible at `GET /v1/startup-profile`.

#### `squish/serving/feature_state.py`
`FeatureState` dataclass centralises ~90 previously scattered `_xxx = None` globals
from `server.py` into a typed, importable structure.

#### `squish/serving/blazing.py`
M3/M4/M5 auto-blazing eligibility (`auto_blazing_eligible`), `BlazingPreset` dataclass,
and `get_preset(chip, ram_gb)` which selects INT4 for ≥ 24 GB RAM configs.

#### `squish/serving/local_model_scanner.py` (Wave 89)
`LocalModelScanner` discovers Squish, Ollama, and LM Studio models from standard
local directories and exposes them through `/api/tags` for OpenWebUI compatibility.

### Wave 90: Import Audit Script

#### `dev/scripts/import_scan.py`
AST-based import dependency analyzer. Report A: orphan modules (zero inbound imports).
Report B: server.py globals assigned only `None` (dead feature flags).

### Wave 91: Performance

- `--no-blazing` flag disables auto-activation on M3+ for users preferring
  full context window over sub-3s TTFT.
- RAM-aware quant auto-selection: INT2 when model > 75% RAM, INT3 when > 55%.
- `llama3.3:70b` wired with INT2 catalog entry and `"impossible"` tag.

### Wave 94: Platform Properties

`PlatformInfo` (frozen dataclass in `squish/platform/detector.py`) now exposes:
- `.is_apple_silicon`: True when `kind == MACOS_APPLE_SILICON`
- `.is_cuda`: alias for `has_cuda`
- `.name`: lower-case kind string (e.g. `"macos_apple_silicon"`)
- `.platform_name`: human-readable (e.g. `"Apple Silicon (M3 Pro)"`)

`detect_platform()` module-level convenience function added.

`get_inference_backend(platform)` in `platform_router.py` returns
`"mlx" | "torch_cuda" | "torch_rocm" | "torch_cpu"`.

### Test Coverage: Waves 85–95

| Test file | Tests |
|-----------|------:|
| `tests/test_wave89_local_model_scan.py` | 36 |
| `tests/test_wave90_startup_lean.py` | 33 |
| `tests/test_wave91_performance.py` | 32 |
| `tests/test_wave92_presquish.py` | 25 |
| `tests/test_wave93_squishbar.py` | 37 |
| `tests/test_wave94_cross_platform.py` | 29 |
| `tests/test_wave95_release.py` | TBD |

### Wave 130: Multimodal Backend — Architecture Resolver + Text-Only Dispatch

`mlx_lm` only implements text-only decoder architectures. `mlx-community` now
hosts checkpoints declaring `model_type`s it doesn't know (Gemma 4's
`gemma4_unified`, plus the VLM/omni long tail). Wave 130 adds `mlx_vlm` as a
second, optional runtime backend and teaches Squish's existing SQUIZD load
path to dispatch to it — reusing the quantization pipeline unchanged.

- `squish/runtime/arch_resolver.py` (new): `resolve_runtime(model_dir)` reads
  `config.json`'s `model_type` and probes `mlx_lm`'s model registry via a
  cheap `importlib` import (no weights loaded). Falls back to `mlx_vlm` when
  the `multimodal` extra is installed; raises `UnsupportedArchitectureError`
  with an install hint when neither backend can load it. Caches the
  resolution in a `.squish_runtime.json` sidecar next to the model dir.
- `squish/backend.py`: `_AppleBackend.load_model`/`.stream_generate` dispatch
  on the resolver. mlx_vlm's `load()` returns `(model, processor)` — a valid
  drop-in for the `(model, tokenizer)` contract. Models loaded via mlx_vlm
  are tagged `model.__squish_runtime__ = "mlx_vlm"` so `stream_generate` can
  route to the right generator (kwarg name differs: mlx_lm's `temp` vs
  mlx_vlm's `temperature`; both yield the same `GenerationResult`-shaped
  objects otherwise).
- `squish/quant/compressed_loader.py`: `_instantiate_model` (the real
  integration point for `squish run` on Apple Silicon — `server.py`'s
  `load_model()` calls this directly, not `BE.load_model`) now dispatches
  through the resolver. `_instantiate_model_mlx_vlm` wraps mlx_vlm's own
  `get_model_and_args` / `update_module_configs` /
  `apply_generation_config_defaults` helpers rather than reimplementing
  mlx_vlm's config normalization, matching mlx_vlm's own `load_model()`
  skeleton-building steps.
- `pyproject.toml`: new `multimodal` extra (`mlx-vlm>=0.5`, Darwin/arm64
  only, lazy-imported everywhere — never a hard dependency).
- `squish/catalog.py`: `gemma4:12b` entry (`mlx-community/gemma-4-12B-bf16`,
  `model_type=gemma4_unified`, 11.96B params, 47.9 GB raw BF16 — live HF
  metadata, not an estimate). `squish_repo` is `None` and `squished_size_gb`
  is a Gemma-3-ratio placeholder pending Wave 131's real INT4 measurement.
- Fixed a latent bug surfaced by the mlx-vlm install bumping `mlx` past
  0.20: `mx.metal.set_memory_limit()` dropped its `relaxed=` kwarg
  (`TypeError`, previously uncaught in `backend.py`). Both call sites
  (`backend.py`, `compressed_loader.py`) now try the new signature first
  and fall back to the old one.
- Phase 1 explicit non-goals (deferred to Phase 2 / Wave 134+): no
  image/audio HTTP surface, no `VisionFeatureCache` integration, no
  multimodal-aware quant calibration, no new SQUIZD `arch_id` flag bit.

| Test file | Tests |
|-----------|------:|
| `tests/test_wave130_vlm_backend_resolver.py` | 16 |

### Wave 131: Delete-As-You-Go Raw Shard Cleanup

`squish pull`/`squish compress` require raw + compressed model on disk
simultaneously — for a 12B model that's ~24 GB raw + ~12–14 GB compressed
≈ 36–38 GB peak, and the raw copy is never auto-deleted afterward. Wave 131
closes the easy 80% of that gap with an opt-in `--delete-source` flag:
delete each raw `.safetensors` shard the moment its tensors have been fully
quantized and written, instead of leaving the whole raw model sitting on
disk until the user manually runs `squish rm`.

(Numbering note: this wave claims slot 131 rather than 130 — Wave 130 in
this repo is the multimodal-backend wave, already shipped before this brief
landed.)

- `squish/convert.py::process_weights_streaming()` — already shard-streamed
  (loads one shard, quantizes, writes `.npy`, frees RAM); this wave adds a
  `delete_source: bool = False` parameter. After each shard's per-tensor
  loop and its existing per-shard disk-space check, the shard's manifest
  entries are committed to `manifest.json` on disk and a `completed_shards`
  list is written to `_compress_progress.json` — both incrementally, per
  shard, not once at the end of the run — *before* the raw shard is
  unlinked (when `delete_source=True`). Manifest/progress being
  incremental (not just the deletion) matters for diagnosability: a crash
  at shard 6 of 10 still leaves an accurate on-disk record of shards 1–5.
  The pre-flight disk estimate switches to
  `compressed_estimate + largest_shard_size + min_free_gb` when
  `delete_source=True` (vs. `compressed_estimate + min_free_gb` otherwise,
  unchanged).
- `squish/convert.py::main()` — new `--delete-source` flag (off by
  default). The existing `except (Exception, KeyboardInterrupt)` handler's
  "clean up partial output" behavior is skipped when `delete_source` was
  active; instead it reads `_compress_progress.json` and prints exactly
  which raw shards were deleted and are unrecoverable, points the user at
  `squish pull <model> --force` to retry, and exits non-zero — deleting
  the partial output in this mode would destroy the only local record of
  how much work would need to be redone.
- `squish/cli.py::cmd_compress` / `_cmd_compress_inner` — `--delete-source`
  threaded through to the `squish.convert` subprocess invocation.
- `squish/cli.py::cmd_pull` — `--delete-source` (alias `--reclaim-space`)
  threaded through `_catalog_pull` to `squish/catalog.py::pull()`'s
  subprocess invocation. The pre-flight box (Raw size / Compressed /
  Context / Dest) gains a `Peak disk` line for *both* the flag-on and
  flag-off paths — previously there was no peak-disk figure shown at all.
- `squish/catalog.py::pull()` — new `delete_source: bool = False`
  parameter, passed through to the `squish.convert` subprocess the same
  way as the other quantization flags.
- Default is opt-in (`False`) for both `squish compress` and `squish pull`
  in this wave: `--delete-source` changes resumability — a raw shard
  deleted mid-run cannot be recovered without re-downloading the model, so
  it's a real tradeoff the user opts into, not a free win. Flipping the
  default for `squish pull` (download + compress is one atomic user
  intent there) is a plausible follow-up once real usage exists, but that
  needs its own decision, not this wave's default.
- Backlog (not this wave, logged for future scheduling): Option B
  (two-pass true streaming pull — discard raw shards during AWQ
  calibration too, ~2× network transfer) and Option C (single-pass
  GPTQ-style sequential streaming — one network pass, requires a rewrite
  of AWQ calibration to run layer-sequentially). Both build on this wave's
  incremental progress tracking.

| Test file | Tests |
|-----------|------:|
| `tests/test_wave131_streaming_delete_source.py` | 11 |

---

### Waves 139–147b: `squish quantize-remote` — Quantizing Models Beyond Local RAM/Disk (v9.34.14)

Wave 131 bounded peak disk during the *quantization* pass to one raw shard
in flight, but AWQ calibration (`squish/quant/awq.py::collect_activation_scales`)
still called `mlx_lm.load(model_dir)` — loading the entire bf16 model into
unified memory to run real forward passes — so a model too large for local
RAM couldn't be AWQ-calibrated at all, and the raw model still had to be
downloaded in full before quantization could start. This wave sequence adds
`squish quantize-remote <hf-repo>`, an end-to-end download → quantize →
(optionally) push command that decides its strategy from HF metadata alone,
before any bytes are downloaded, and — outside full-load mode, which
genuinely needs the whole model resident for `mlx_lm.load` — never requires
the model to fit in local RAM or on local disk beyond one shard at a time.

#### Wave 139/141: Verifying the Non-AWQ Path Already Scales, Documenting the Real Wall

Confirmed (not assumed) that `process_weights_streaming`'s existing
shard-at-a-time loop already quantizes models far larger than local RAM
today, with zero AWQ scales — real bf16 shards fail `numpy.load_file`
(no native bfloat16 dtype) and fall through to the documented MLX-CPU
fallback, and every quantization backend downstream operates on plain
numpy with zero MLX/Metal involvement. `docs/ARCHITECTURE.md` section 11
documents this finding plus the actual constraint: **AWQ calibration**,
not the quantization pass itself, is the RAM wall. `--no-awq` documents
the choice explicitly for a caller who wants that path on a big model
today, ahead of the streaming-calibration rework below.

#### Wave 140: `squish push`

`squish push <local-dir> <repo-id>` — uploads a local squish-compressed
model directory to a Hugging Face repo (`--private`, `--commit-message`,
`--dry-run`), generating a model card from the directory's compression
stats. `quantize-remote --push` (Wave 146) calls this directly as its
final step.

#### Wave 142: Shard-Aware Bookkeeping + Standalone Block Reconstruction

Two building blocks the sequential-calibration rework (Wave 143) depends on:

- **`squish/quant/shard_index.py`** (new): `ShardIndex`/`load_shard_index()`
  parse `model.safetensors.index.json`'s `weight_map` into a layer-indexed
  lookup — `tensors_for_layer()`, `shards_for_layer()`, `non_layer_tensors()`
  (embed_tokens/lm_head/final norm), and `shard_to_layers()` (the reverse
  map that drives delete-as-you-go eviction timing: a shard is safe to
  release only once every layer index in its set has been processed).
  Returns `None` for an unsharded single-file checkpoint rather than
  erroring — callers fall back to treating the whole model as one unit.
- A spike proving a standalone `mlx_lm` `TransformerBlock(args)` can be
  constructed and run in isolation given just its own weights — the
  foundation `awq_streaming.py` builds on to avoid `mlx_lm.load()`.

#### Wave 143: `squish/quant/block_adapters.py` — Block-Kind Adapter Registry

Resolves a model architecture to the right calibration adapter by decoder
block *kind* (structural shape), not by architecture family name or class
name — verified directly against installed `mlx_lm` 0.31.3 model files
that class-name matching is unsafe in both directions (`olmoe.py` names its
MoE block `TransformerBlock` too; `phi.py`'s dense block is named
`PhiDecoderLayer`, not `TransformerBlock`).

- `resolve_dense_architecture(model_type)` reuses `mlx_lm.utils.MODEL_REMAPPING`
  — the same table `mlx_lm.utils._get_classes` itself uses — to resolve a
  model_type to its `ModelArgs`/`TransformerBlock` classes, so any
  architecture `mlx_lm` supports is covered automatically, no hand-maintained
  list. Returns `None` for genuinely different block-shape families
  (Mamba, etc.) — the caller falls back to plain non-AWQ quantization.
- `is_standard_dense_block(block)` structurally verifies a *constructed
  instance* has no MoE/SSM submodules, checking submodule *class* names
  against marker lists (`Moe`/`MoE`/`Sparse`/`Expert`, `Mamba`/`SSM`/
  `RGLRU`/`Recurrent`) — never attribute names, so a dense SwiGLU MLP's own
  `gate_proj` linear doesn't false-positive against a naive "gate" check.
- One "standard dense block" adapter covers Llama, Qwen2, Qwen3, Mistral,
  Phi3, Starcoder2, and any future family sharing the same attention+MLP
  shape — zero new code needed here for a new family name of the same kind.

#### Wave 143/144: `squish/quant/awq_streaming.py` — Sequential AWQ Calibration

`collect_activation_scales_streaming()` — the layer-at-a-time counterpart
to `awq.py::collect_activation_scales`, never instantiating the full
model. Processes one decoder layer at a time (construct block from
`block_adapters`, load only that layer's raw weights via `ShardIndex`,
run the forward pass, capture activations with the existing
`_attach_activation_hooks`/`_scales_from_hooks` machinery, carry the
hidden state forward), returning the same `layer_name -> np.ndarray` scale
dict contract as the full-load function. Returns `None` — not an error —
when the architecture doesn't resolve to the standard dense block adapter;
the caller falls back to plain non-AWQ quantization.

Wave 144 integrates delete-as-you-go (Wave 139's convention) directly into
this loop: with `delete_source=True`, each raw shard is deleted once every
layer that needs it (per `ShardIndex.shard_to_layers()`) has finished
calibrating — handling the case where a shard is shared between
embed_tokens and layer 0 (embed_tokens' one-time read happens before the
per-layer loop, so the shard has no remaining consumers by the time layer 0
completes). Deletion failures are logged as warnings, never raised.

#### Wave 145: Accuracy Validation Gate — Finds and Fixes a Real Causal-Mask Bug

`tests/test_wave145_calibration_accuracy_gate.py` compares
`collect_activation_scales_streaming`'s scale vectors directly against
`collect_activation_scales`'s full-load ground truth on a real ~2.5 GB
model (`mlx-community/Llama-3.2-1B-Instruct-bf16`) — deliberately not a
synthetic-weights test, since only a real forward pass through a real
checkpoint has independent ground truth to diverge from. It caught a real
bug: streaming calibration was passing `mask=None` unconditionally to each
standalone block (no causal masking — every token could attend to every
position, including future ones), which measured as low as 0.75
correlation on some layers against ground truth. Fixed with
`create_attention_mask(h)`, matching exactly what the full-load model
applies per layer — brought all 112 compared layers to ≥0.9999
correlation. Opt-in via `SQUISH_RUN_ACCURACY_GATE=1` (downloads a real
model, ~30–60s per pass), so it doesn't run by default.

#### Wave 146: `squish quantize-remote` — End-to-End Command

`squish quantize-remote <hf-repo>` (`squish/cli.py`) — scans HF repo
metadata (size, shard-index presence) before downloading anything, then
picks between full-load AWQ, streaming AWQ, or no AWQ:

- **full-load** when `model_size_gb * 2.0 + 2.0` (the full-load AWQ peak
  estimate) fits under total local RAM — downloads the whole raw model
  (required for `mlx_lm.load` regardless of any streaming path), then
  quantizes with `process_weights_streaming(..., delete_source=True)`.
- **streaming** when it doesn't fit but the repo has a shard index.
- **none** (`--no-awq`, or automatically for an unsharded checkpoint too
  big for full-load) — falls straight to Wave 147a's per-shard streaming pull.

`--push <repo-id>` chains directly into `cmd_push` (Wave 140) after
quantization. Runs the same pre/post-download security scans (Wave 100) as
`squish pull`.

#### Wave 147a: `squish/quant/streaming_pull.py` — True Per-Shard Streaming Pull

`pull_and_quantize_shard_by_shard()` — fetch one raw shard from HF,
quantize it, delete it, fetch the next. Never more than one raw weight
shard resident on local disk at once, regardless of total model size —
the local model directory ends up holding only non-weight files (config,
tokenizer, shard index), so it still works as the config source for the
compressed model. Reuses `convert.py`'s `quantize_tensor`/`safe_key`/
`load_mlx_weights_shard` and `awq.py`'s AWQ-scale application unchanged, so
output matches `process_weights_streaming`'s regardless of which path
produced it. `fetch_repo_metadata()`/`ensure_shard_local()` run the same
pre/post-download security scans (Wave 100) as every other pull path.

#### Wave 147b: Fetch-on-Demand Folded Into Streaming AWQ Calibration

`awq_streaming.py::collect_activation_scales_streaming` gains optional
`hf_repo`/`token` params — when given, each shard is fetched on demand via
`streaming_pull.ensure_shard_local()` the first time a layer needs it,
instead of assuming it's already local. Combined with `delete_source=True`,
streaming AWQ calibration alone never requires the full raw model
downloaded up front, matching Wave 147a's non-AWQ path. The trade:
streaming-AWQ mode downloads every shard twice — once for calibration,
once for the quantization pass — in exchange for peak disk staying bounded
to the compressed output plus one raw shard in flight across both passes,
never the full raw model. When `hf_repo` is `None` (the default), behavior
is unchanged: shards must already exist locally.

| Test file | Tests |
|-----------|------:|
| `tests/test_wave139_delete_source_shards.py` | 8 |
| `tests/test_wave140_push_command.py` | 15 |
| `tests/test_wave141_no_awq_large_model_path.py` | 6 |
| `tests/test_wave142_shard_index.py` | 13 |
| `tests/test_wave142_single_layer_reconstruction.py` | 6 |
| `tests/test_wave143_block_adapters.py` | 12 |
| `tests/test_wave143_streaming_calibration.py` | 7 |
| `tests/test_wave144_calibration_delete_source.py` | 7 |
| `tests/test_wave145_calibration_accuracy_gate.py` | 2 (opt-in, `SQUISH_RUN_ACCURACY_GATE=1`) |
| `tests/test_wave146_quantize_remote_e2e.py` | 9 |
| `tests/test_wave147a_streaming_pull.py` | 6 |
| `tests/test_wave147b_awq_streaming_pull.py` | 5 |

