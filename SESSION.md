# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-03-27

## Last commits
- `0e67a61` — AWQ alpha=0.1 + g=16 INT4 default + INT3 g=16 + max-model-gb OOM guard + mixed_attn format
- `0d2eb81` — Architecture-aware AWQ calibration: detect_model_family(), Qwen3 alpha=0.07 + 25 CoT texts, _MODEL_FAMILY_DEFAULTS, _DEFAULT_AWQ_ALPHA
- `c755b20` — Task 1 (MODEL_PLAN clean, 42 tests pass), lm_eval waivers for Tasks 2–4, dev/ scan scripts (bundled accidentally)

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

**Qwen3-4B:** ❌ Bench FAILED — model not at `/Users/wscholl/models/Qwen3-4B-int4/config.json`
(OOM guard skipped INT4 compress; lm_eval tried anyway; Qwen3-4B-int3 and int2 were
compressed successfully per squish_log.txt but lm_eval not run against them).

**Global summary table:**

| Format | Code | lm_eval | arc_easy (Qwen2.5-1.5B) | Notes |
|---|---|---|---|---|
| INT4 + AWQ g=16 | ✅ | ✅ | 70.6% | Production default |
| INT3 g=16 | ✅ | ✅ | 67.2% | Confirmed unstable for ≤1.5B. Memory-efficiency option. |
| mixed_attn | ✅ | ⚠️ PENDING | — | FP16 attn projections + INT4 g=16 MLP. Not in bench. |
| Qwen3 alpha=0.07 | ✅ | ✅ | confirmed fix | hellaswag inversion resolved (see below) |
| INT2 naive | ✅ | ❌ broken | ~27–30% | Coherence collapse confirmed. Never ship. |
| INT2 AQLM | stub | ⚠️ unrun | — | Begin after mixed_attn confirmed |

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

1. ✅ MODEL_PLAN verified clean (Qwen3-4B correct, 42 tests pass) — `c755b20`
2. ⚠️ INT3 g=16 decision gate: **RE-OPENED** — was g=32 mlx_lm.convert, not squish compress g=16. Run pending.
3. ✅ Qwen3 alpha=0.07 hellaswag inversion confirmed resolved
4. **NEXT: Run squish compress --format int3 on Qwen2.5-1.5B-bf16 + bench (Q1 answer):**
   ```bash
   squish compress ~/models/Qwen2.5-1.5B-Instruct-bf16 --format int3
   # output defaults to ~/models/Qwen2.5-1.5B-Instruct-int3 (mlx safetensors, g=16)
   python3 dev/benchmarks/bench_lmeval_all_models.py \
     --models Qwen2.5-1.5B-int3 \
     --limit 500 \
     --tasks arc_easy arc_challenge hellaswag winogrande piqa openbookqa \
     --output-dir results
   # Decision gate: arc_easy ≥ 72% → INT3 becomes default
   ```
5. **MIXED_ATTN BLOCKED:** Q2 cannot be answered with current harness.  
   `squish compress --format mixed_attn` writes npy-dir format (not loadable by mlx_lm evaluate).  
   Future work: implement squish_lm_eval.py MLX harness OR add a npy-dir → safetensors export step.
6. **NEXT (Tier 2): 3B/4B models** — Use squish compress --format int3 for INT3, mlx_lm.convert for INT4:
   ```bash
   # For each 3B/4B BF16 model, run:
   squish compress ~/models/<model>-bf16 --format int3   # → mlx safetensors g=16
   # then bench_lmeval_all_models.py
   ```
7. **NEXT (Tier 3): 7B/8B models** — same pattern, --max-model-gb 12

---

## Model catalog decision tree (UPDATED — awaiting squish compress g=16 results)

```
INT3 g=32 mlx_lm.convert arc_easy on Qwen2.5-1.5B: 67.2% — OLD BASELINE (not squish compress).
INT3 g=16 squish compress arc_easy: PENDING (run this session).

DECISION (tentative, pending g=16 result):
  If INT3 g=16 ≥ 72%: INT3 becomes default.
  If INT3 g=16 < 72%: INT4 stays default. INT3 = memory-efficiency option ("efficient" tier).

Catalog labels (provisional — update after Q1 is answered):
  "balanced"   → INT4 (squish npy-dir + AWQ, or mlx safetensors g=64 for mlx_lm compat)
  "efficient"  → INT3 g=16 (mlx safetensors, squish compress --format int3)
  "ultra"      → INT2 AQLM (pending; naive INT2 is incoherent — never expose)

For ≤1B models: INT3 degradation varies (gemma-3-1b drops -15.2pp at g=32 — may differ at g=16).
For 1.5B models: INT3 g=32 = -3.4pp; g=16 result pending.
For 7B+: INT3 likely safe (not yet measured with current squish).
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
