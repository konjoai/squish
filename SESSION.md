# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-03-28

## Last commits
- `7f492a9` — bench(results): overnight lm_eval Tier 0-1 partial (8 of 19 models) — Qwen3-0.6B, Llama-3.2-1B, gemma-3-1b INT4/3/2 pushed
- `c2c1f0c` — docs(session): overnight bench running (19 models Tier 0-3); data quality findings
- `9fce455` — fix(compress): INT3 group_size 16→32 (MLX only supports 32/64/128) + Tier 1 lm_eval results

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
- INT4 AWQ g=16 squish results: **NOT measurable** via standard lm_eval harness ❌
- mixed_attn squish results: **NOT measurable** via standard lm_eval harness ❌
  (Q2 blocked until a squish-native lm_eval harness is built, or npy→safetensors converter)

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
| INT4 AWQ g=16 (squish) | n/a | n/a | n/a | n/a | n/a | n/a | npy-dir — not lm_evaluable |
| mixed_attn (squish) | n/a | n/a | n/a | n/a | n/a | n/a | npy-dir — not lm_evaluable |

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
| mixed_attn | ✅ | ❌ BLOCKED | — | npy-dir format — not lm_evaluable until harness built |
| Qwen3 alpha=0.07 | ✅ | ✅ | confirmed fix | hellaswag inversion resolved |
| INT2 naive | ✅ | ❌ broken | ~28% | Coherence collapse confirmed. Never expose as production. |
| INT2 AQLM | stub | ⚠️ unrun | — | Begin after mixed_attn harness built |

---

## Open questions

1. **INT3 g=32 squish compress ≥72% on Qwen2.5-1.5B?** → ✅ **ANSWERED: NO — 67.20% ±2.1%.**  
   squish compress --format int3 uses g=32 (MLX only supports 32/64/128; g=16 was a bug — fixed).  
   Result matches overnight mlx_lm.convert g=32 baseline exactly.  
   **Decision: INT4 stays default. INT3 = memory-efficiency option ("efficient" tier).**  
   gemma-3-1b INT3-sensitive (-15.2pp at g=32). Do not recommend INT3 for 1b class.  
   Note: `_INT3_GROUP_SIZE` bug fixed in squish/cli.py (16→32). 3562 tests pass.

2. **Does mixed_attn improve piqa/winogrande vs INT4 g=16?** → ❌ **BLOCKED (format issue).**  
   `squish compress --format mixed_attn` writes squish npy-dir format that `mlx_lm.load()` cannot load.  
   Standard bench harness (`mlx_lm evaluate`) cannot evaluate this model.  
   To unblock: build a squish npy-dir → lm_eval adapter (squish_lm_eval.py for MLX), OR  
   convert npy-dir → mlx safetensors via a passthrough export step.

3. **Qwen3 alpha=0.07 hellaswag inversion resolved?** → ✅ **ANSWERED: YES.**  
   Pre-fix (2026-03-22, alpha=0.10): INT3 hellaswag=36.2 > INT4=31.2 (anomalous inversion).  
   Post-fix overnight: INT3 hellaswag=32.0 < INT4=33.0 (correct ordering, inversion gone).

4. **INT2 AQLM path** → Begin after Q2 (mixed_attn) is answered.

---

## Known test issues

- `test_int4_conversion_and_round_trip` — requires Rust `squish_quant` extension
  (maturin build). Skip in CI without Rust toolchain. Not a regression.

---

## Immediate next task

### RUNNING NOW (2026-03-28 session 2)

**Overnight bench still running** — PID 42230. Currently on Qwen2.5-1.5B-int2.

**Completed (10 of 19 models, all fresh 2026-03-28 data):**
```
Qwen3-0.6B-int4/3/2      ✅ committed in 7f492a9
Llama-3.2-1B-int4/3/2    ✅ committed in 7f492a9
gemma-3-1b-int4/3/2      ✅ int4+int3 committed in 7f492a9; int2 pending commit
Qwen2.5-1.5B-int3        ✅ committed in 9fce455
```

**Still pending (10 models):**
```
Qwen2.5-1.5B-int2  ← currently running
Llama-3.2-3B-int3/2
Qwen3-4B-int3/2
gemma-3-4b-int3/2  (compressed this session; safetensors format ✅)
Qwen2.5-7B-int3
Qwen3-8B-int3/2
```

**gemma-3-4b format:** ls ~/models/gemma-3-4b-it-int3/ showed `model.safetensors` — confirmed ✅. NOT npy-dir. Safe to bench.

Models NOT included (excluded intentionally — OOM or npy-dir format):
- INT4 dirs: Qwen3-4B-int4, Qwen3-8B-int4, Qwen2.5-7B-int4, Llama-3.2-3B-int4, gemma-3-4b-int4 (all squish npy-dir, 12-14 GB)
- Qwen2.5-1.5B-int4 (70.6% confirmed, 2026-03-23)

### After bench completes:
1. Commit remaining result JSONs (gemma-3-1b-int2 + all Tier 2/3)
2. Update CLAUDE.md per-model table with Tier 2/3 data (3B/4B/7B/8B)
3. Answer: does INT3 safety hold at 3B+? (−3 to −4pp expected for non-gemma)
4. Final SESSION.md update + commit push
4. Answer: does INT3 accuracy hold at 3B/4B/7B/8B? (expected: yes, unlike 1B class)

### Blocked:
- Q2 mixed_attn: npy-dir format. Needs squish-native lm_eval harness.
- Qwen2.5-7B-int2: source is 14 GB BF16 → OOM on 16GB M3 during conversion. Not attempting.
- INT2 AQLM: begin after mixed_attn blocked issue is resolved.

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
