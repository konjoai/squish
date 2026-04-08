# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-08

## Last commits
- **`(w42 — TBD hash)`** — fix(wave42): bias formula + norm exclusion + g=32 default; INT4 AWQ arc_easy 70.8% PASS
- `98885d0 (w41)` — feat(wave41): squish-native lm_eval harness — dev/benchmarks/squish_lm_eval.py (52 tests, 4642 suite)
- `d20b0ea` — feat(wave40): GCP Vertex AI integration — VertexAISquash platform adapter (24 tests)
- `ebbb56b` — fix(ci): resolve 4 test failures
- `3c3a0d5` — feat(wave38): AQLM dequantization module (AQLMConfig, AQLMCodebook, AQLMLayer)
- `de5d598` — docs(wave37): post-ship docs update (NEXT_SESSION/SESSION)
- `60c2bf1` — chore(bench): add Qwen2.5-7B-int3 full lm_eval run

---

## Quantization status (as of 2026-03-27, overnight bench results)

**⚠️ CORRECTION (2026-03-27 follow-up session):**
The overnight bench (`run_overnight_bench.py`) does NOT use `squish compress`. It calls
`mlx_lm.convert` directly with `q_group_size = {4: 64, 3: 32, 2: 64}`. Verified via:
- `config.json` in `~/models/Qwen2.5-1.5B-Instruct-int4`: `bits=4, group_size=64`
- `config.json` in `~/models/Qwen2.5-1.5B-Instruct-int3`: `bits=3, group_size=32`
- Source: `run_overnight_bench.py` line 259: `q_group_size = {4: 64, 3: 32, 2: 64}`
The "g=16 AWQ throughout" description in the previous session note was WRONG.

**⚠️ FORMAT DISCOVERY:**
`squish compress --format int4` / `--format mixed_attn` outputs squish `.npy-dir` format
(via `squish.convert --format npy-dir`) which `mlx_lm.load()` CANNOT load. Only
`squish compress --format int3` uses `mlx_lm.convert` internally and produces standard
MLX safetensors loadable by the bench harness. This means:
- INT3 g=16 squish compress results: **measurable** via bench_lmeval_all_models.py ✅
- INT4 AWQ g=16 squish results: **CODE-COMPLETE harness built** — `dev/benchmarks/squish_lm_eval.py` ✅ (Wave 41)
  Run: `python3 dev/benchmarks/squish_lm_eval.py --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq --model-dir ~/models/Qwen2.5-1.5B-Instruct --limit 500`
  Acceptance: arc_easy within ±2pp of 70.6% (lm_eval-waiver in commit — validation run queued)
- mixed_attn squish results: **harness built** — same command with mixed-attn npy-dir ✅

Overnight bench: `results/overnight_20260326T232055/` — M3 16GB, limit=500,
mlx_lm.convert (g=64 INT4, g=32 INT3, g=64 INT2) — NOT squish compress AWQ.

**Qwen2.5-1.5B (key decision model) — mlx_lm.convert OLD baselines:**

| Format | arc_easy | arc_challenge | hellaswag | winogrande | piqa | openbookqa | Delta vs INT4 |
|---|---|---|---|---|---|---|---|
| INT4 g=64 (mlx baseline) | 70.6% | 41.2% | 54.2% | 61.0% | 72.2% | 38.6% | baseline |
| INT3 g=32 (mlx baseline) | 67.2% | 41.6% | 50.6% | 57.2% | 71.6% | 37.6% | **-3.4pp arc_easy** |
| INT2 g=64 (naive) | 29.8% | 24.4% | 24.6% | 51.0% | 51.6% | 29.8% | incoherent |

**Squish compress g=32 baselines (MEASURED 2026-03-28, mlx_lm 0.31.1, limit=500):**

| Format | arc_easy | arc_challenge | hellaswag | winogrande | piqa | openbookqa | Notes |
|---|---|---|---|---|---|---|---|
| INT3 g=32 (squish compress) | **67.20% ±2.1%** | 41.60% | 50.60% | 59.40% | 70.80% | 37.20% | Below 72% gate — INT4 stays default |
| INT4 AWQ g=32 (squish) | **70.8% ✅** | **44.4%** | TBD | TBD | TBD | TBD | W42 PASS (+0.2pp). 4 tasks still running. |
| mixed_attn (squish) | n/a — broken compress | n/a | n/a | n/a | n/a | n/a | AWQ failed silently → effectively BF16. Fix in W43. |

**Q1 DECISION: INT4 is default. INT3 = "efficient" memory-saving option (-3.4pp arc_easy, -3.8x disk).**

**Qwen3-0.6B:**

| Format | arc_easy | hellaswag | Notes |
|---|---|---|---|
| INT4 | 34.0% | 33.0% | |
| INT3 | 36.4% | 32.0% | arc_easy delta within noise at limit=500 |
| INT2 | 27.0% | 26.4% | incoherent |

**Llama-3.2-1B:**

| Format | arc_easy | hellaswag |
|---|---|---|
| INT4 | 40.0% | 44.0% |
| INT3 | 37.2% | 41.6% |
| INT2 | 27.2% | 29.6% |

**gemma-3-1b:**

| Format | arc_easy | hellaswag | Notes |
|---|---|---|---|
| INT4 | 53.2% | 39.4% | |
| INT3 | 38.0% | 36.4% | **-15.2pp arc_easy — very sensitive** |
| INT2 | 26.2% | 28.2% | incoherent |

**⚠️ DATA QUALITY WARNING (2026-03-28):** All lm_eval results from 2026-03-22/23/24 are suspect.
Root cause: multiple model dirs existed in squish npy-dir format at that time (not loadable by
mlx_lm evaluate), AND a "bad INT3 model" existed (commit c1f0982 "remove bad INT3 model" confirms
this). The March 23 Qwen2.5-1.5B-int3 result (42.6%) vs this session's result (67.2%) shows the
discrepancy. **All 1B/3B/4B/7B/8B results prior to 2026-03-28 should be treated as unreliable.**
AFRESH BENCH IS RUNNING (PID 42230, started 2026-03-28, log: /tmp/squish_bench_overnight.log).

**Qwen3-4B:** All dirs exist (int2/int3/int4 all safetensors). Fresh bench running.

**Global summary table:**

| Format | Code | lm_eval | arc_easy (Qwen2.5-1.5B) | Notes |
|---|---|---|---|---|
| INT4 mlx g=64 | ✅ | ✅ | **70.6%** | Production default (validated) |
| INT3 g=32 (squish) | ✅ | ✅ | **67.2% ±2.1%** | Q1 ANSWERED. -3.4pp. Memory-efficiency option. |
| INT4 AWQ g=32 (squish) | ✅ | ✅ (partial) | **70.8% ✅** | W42 PASS. arc_easy+arc_challenge confirmed. 4 tasks pending. |
| mixed_attn | ✅ | ❌ broken compress | — | AWQ calibration failed → ~BF16 quality. Fix W43. |
| Qwen3 alpha=0.07 | ✅ | ✅ | confirmed fix | hellaswag inversion resolved |
| INT2 naive | ✅ | ❌ broken | ~28% | Coherence collapse confirmed. Never expose as production. |
| INT2 AQLM | stub | ⚠️ unrun | — | Begin after mixed_attn compress is fixed |

---

## Open questions

1. **INT4 AWQ g=32 arc_easy = 70.8% PASS** ✅ (W42 — lm_eval-waiver CLOSED)
   - arc_challenge = 44.4% (+0.8pp vs 43.6% baseline)
   - hellaswag / winogrande / piqa / openbookqa: benchmark still running (terminal `3af20b5b`)
   - When complete, update CLAUDE.md accuracy table TBD cells and SESSION.md

2. **INT4 AWQ arc_easy open question:** ANSWERED — 70.8% (within ±2pp of 70.6% baseline) ✅

3. **Does mixed_attn improve piqa/winogrande vs INT4?** ❌ **BLOCKED — compress broken (W42 discovery).**
   mixed_attn npy-dir has 59 `__q4a.npy` (57 are 1D norms; only 2 actual matrices INT4).
   AWQ calibration failed silently during compression. gate_proj outlier ratio=28.5 > threshold=20.0
   without AWQ pre-scaling → almost all MLP tensors stored as BF16 passthrough.
   Fix for W43: re-run `squish compress --format mixed_attn` with visible output; verify
   `__q4a.npy` count ≈ 245 before launching benchmark.

4. **Qwen3 alpha=0.07 hellaswag inversion resolved?** → ✅ **ANSWERED: YES.** (unchanged)

5. **INT2 AQLM path** → Begin after mixed_attn compress is fixed (W43+).

---

## Known test issues

- `test_int4_conversion_and_round_trip` — requires Rust `squish_quant` extension
  (maturin build). Skip in CI without Rust toolchain. Not a regression.

---

## Immediate next task

### COMPLETED (2026-03-31 / 2026-04-01 sessions)

**Tier 2/3 bench complete — all 7 target models validated and committed.**

**⚠️ THINKING MODE DISCOVERY (2026-03-31):**
Qwen3 models emit `<think>...</think>` reasoning tokens before answers. `mlx_lm evaluate`
greedy extraction scores the CoT prefix → near-random results (50% arc_easy on 8B).
Fix: `--apply-chat-template --chat-template-args '{"enable_thinking": false}'`.
Implemented in `bench_lmeval_all_models.py` via `_is_thinking_model()` (commit `eb0684b`).
All `Qwen3-*` models auto-detected. Qwen2.5 unaffected.

**⚠️ INVALID RESULTS — needs re-run:**
- `Qwen3-4B-int4` in commit `4a2ff5c` — thinking mode ON → arc_easy=41% (invalid)
- All prior Qwen3-8B results (2026-03-23/28) — thinking mode ON → always invalid
- **Re-run Qwen3-4B-int4 with thinking disabled is the immediate next bench task**

**Tier 2/3 validated results (limit=500, mlx-lm 0.30.7, M3 16GB, thinking disabled for Qwen3):**

| Model | arc_easy | arc_challenge | hellaswag | winogrande | piqa | openbookqa | Avg | Commit |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-int4 | 46.8% | 36.8% | 55.4% | 63.2% | 76.6% | 40.4% | 53.2% | `4a2ff5c` |
| gemma-3-4b-int4 | 80.8% | 55.0% | 43.8% | 64.8% | 68.4% | 42.8% | 59.3% | `4a2ff5c` |
| Qwen3-4B-int4 ⚠️ | 41.0% | 36.0% | 40.2% | 62.0% | 71.6% | 32.8% | 47.3% | `4a2ff5c` — **INVALID (thinking on)** |
| Qwen2.5-7B-int4 | 83.0% | 58.8% | 59.4% | 67.4% | 73.8% | 43.0% | 64.2% | `9634949` |
| Qwen2.5-7B-int3 | 79.0% | 56.0% | 53.6% | 63.4% | 76.0% | 41.4% | 61.6% | `60c2bf1` |
| Qwen3-8B-int4 ✅ | 79.2% | 58.6% | 49.8% | 61.8% | 74.8% | 39.8% | 60.7% | `61502cd` |
| Qwen3-8B-int3 ✅ | 71.4% | 52.0% | 45.2% | 59.8% | 73.4% | 36.4% | 56.4% | `61502cd` |

**INT3 delta summary (Tier 2/3):**
- Qwen2.5-7B: −4.0pp arc_easy (79.0% vs 83.0%) — consistent with 1.5B pattern
- Qwen3-8B: −7.8pp arc_easy (71.4% vs 79.2%) — larger delta; 8B Qwen3 more INT3-sensitive

### Wave 39 — COMPLETE ✅

**CLAUDE.md per-model validated results table completed (2026-04-06).**
- Expanded from 11 rows (Tier 1 only) to 25 rows (all Tiers 1–3: 0.6B–8B).
- Added `winogrande` and `openbookqa` columns.
- Source: `results/lmeval_*_2026040[12]*.json` (thinking disabled for all Qwen3 runs).
- Key new UNSAFE findings:
  - `gemma-3-4b INT3`: −16.4pp → confirms gemma family UNSAFE at ≤4B
  - `Qwen3-4B INT3`: −14.8pp → UNSAFE (same risk class as gemma family)
  - `Qwen3-8B INT3`: −7.8pp → coherent but large delta
  - `Llama-3.2-3B INT3`: −4.6pp → coherent
- Stale Qwen3-4B-int4 score corrected (73.2% thinking-disabled, was 41% invalid).
- 4562 tests passing. No code changes. No regressions.

### Immediate next task (Wave 43)

**Priority 1: Complete INT4 AWQ benchmark results**
When terminal `3af20b5b` finishes (hellaswag, winogrande, piqa, openbookqa), read scores:
```bash
for task in hellaswag winogrande piqa openbookqa; do
  f=$(ls /Users/wscholl/squish/results/_tmp_Qwen2.5-1.5B-Instruct-int4-awq/$task/eval__* 2>/dev/null | head -1)
  [ -n "$f" ] && python3 -c "import json; d=json.load(open('$f')); t=d.get('$task',{}); print('$task:', t.get('acc_norm,none', t.get('acc,none', 'n/a')))"
done
```
Update CLAUDE.md table TBD cells and this SESSION.md.

**Priority 2: Fix mixed_attn compression (AWQ failure)**
Root cause: AWQ calibration ran but produced negligible scale reduction (or was silently skipped).
Steps:
1. Delete existing broken npy-dir: `rm -rf ~/models/Qwen2.5-1.5B-Instruct-mixed-attn`
2. Re-run with verbose output — do NOT pipe to `tail`:
   ```bash
   squish compress --format mixed_attn ~/models/Qwen2.5-1.5B-Instruct-bf16 ~/models/Qwen2.5-1.5B-Instruct-mixed-attn 2>&1
   ```
3. Verify AWQ ran: check `__q4a.npy` count ≈ 245  
   `find ~/models/Qwen2.5-1.5B-Instruct-mixed-attn -name '__q4a.npy' | wc -l`
4. Only launch benchmark AFTER verifying count.

**Priority 3: Azure DevOps integration** (`squish/squash/integrations/azure_devops.py`)

### COMPLETED (Wave 42 — 2026-04-08 session)

**Bug fixes: 6 bugs in `compressed_loader.py` + 1 in `cli.py`.**
- CRITICAL: bias formula corrected (`biases = zeros`, not `-zeros/scales`)
- 1D norm weights stored as BF16 (ndim check)
- Attention bias vectors excluded from QuantizedLinear
- group_size default 16→32 in cli.py (3 locations)
- group_size validation ∈ {32,64,128} in loader
- squish_4bit build inside else: block
- Docstring corrected

**INT4 AWQ g=32 arc_easy = 70.8% PASS** (lm_eval-waiver CLOSED).  
Tests: 4642 passing (all green). No regressions.

### COMPLETED (2026-03-31 / 2026-04-01 sessions)

---

## Model catalog decision tree (FINAL — Q1 answered 2026-03-28)

```
Q1 ANSWERED: INT3 g=32 squish compress arc_easy = 67.2% < 72% gate.
INT4 is the default. INT3 is the memory-efficiency ("efficient") tier.

Catalog labels (confirmed):
  "balanced"   → INT4 (mlx safetensors g=64 for lm_eval; squish npy-dir for serve)
  "efficient"  → INT3 g=32 (mlx safetensors, squish compress --format int3)
  "ultra"      → INT2 AQLM (pending; naive INT2 is incoherent — never expose)

Accuracy by model size (INT3 g=32 arc_easy delta vs INT4 — PRELIMINARY, needs Tier 2/3 data):
  0.6B: delta unknown (fresh bench running)
  1B:   Llama-3.2-1B  — fresh bench running (old results suspect)
  1.5B: Qwen2.5-1.5B  — -3.4pp (67.2% vs 70.6%) CONFIRMED ✅
  3B+:  PENDING (bench running overnight)

Key insight: gemma-3-1b-int3 at g=32 shows -15.2pp vs INT4. Do not recommend INT3 for
1b-class gemma models. Qwen3-0.6B, Llama-3.2-1B: fresh data pending this run.
```

---

## INT2 paths (priority order after INT3 confirmed)

1. **AQLM** — additive codebook, already stubbed in codebase. Encodes outlier channels
   without uniform grid collapse. Published: ~4–6pp arc_easy delta vs INT4 on Llama-class.
2. **SpQR/SqueezeLLM mixed** — keep top 1–5% outlier weights in FP16, 2-bit everything else.
   Fixes coherence collapse at the weight level.
3. **Mixed-layer** — first/last 2 transformer layers FP16, all attn projections FP16,
   FFN down to 2-bit AQLM. Effective ~2.8–3.0 bpw without collapse.

None of these have been run. Zero INT2 AQLM data exists in results/.
