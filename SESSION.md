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

Overnight bench: `results/overnight_20260326T232055/` — M3 16GB, limit=500, g=16 AWQ throughout.

**Qwen2.5-1.5B (key decision model):**

| Format | arc_easy | arc_challenge | hellaswag | winogrande | piqa | openbookqa | Delta vs INT4 |
|---|---|---|---|---|---|---|---|
| INT4 (g=16) | 70.6% | 41.2% | 54.2% | 61.0% | 72.2% | 38.6% | baseline |
| INT3 (g=16) | 67.2% | 41.6% | 50.6% | 57.2% | 71.6% | 37.6% | **-3.4pp arc_easy** |
| INT2 (naive) | 29.8% | 24.4% | 24.6% | 51.0% | 51.6% | 29.8% | incoherent |

Note: Previous "~74.2%" figure in SESSION.md was from a different measurement context
(different limit or model source). Ground truth with current squish g=16 AWQ: 70.6%.

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

1. **INT3 g=16 ≥72% on Qwen2.5-1.5B?** → ✅ **ANSWERED: NO — 67.2%.**  
   Decision: **INT4 stays default.** INT3 = memory-efficiency option ("efficient" tier).  
   gemma-3-1b is particularly INT3-sensitive (-15.2pp). Do not recommend INT3 for 1b class.

2. **Does mixed_attn improve piqa/winogrande vs INT4 g=16?** → ⚠️ **STILL UNANSWERED.**  
   Not included in overnight bench. Needs a dedicated run: `squish compress --format mixed_attn`  
   on Qwen2.5-1.5B-bf16 → lm_eval on piqa, winogrande, arc_easy.

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
2. ✅ INT3 g=16 decision gate answered: 67.2% — INT4 stays default
3. ✅ Qwen3 alpha=0.07 hellaswag inversion confirmed resolved
4. **NEXT: Run mixed_attn benchmark** (Q2 still open):
   ```bash
   squish compress --format mixed_attn Qwen2.5-1.5B-bf16 ~/models/Qwen2.5-1.5B-mixed
   # then lm_eval arc_easy piqa winogrande --limit 500
   ```
5. **NEXT: Get Qwen3-4B model** (bench failed — not downloaded):
   ```bash
   # Download Qwen3-4B-bf16 to ~/models/ before running bench again
   ```

---

## Model catalog decision tree (UPDATED — decision made)

```
INT3 g=16 arc_easy on Qwen2.5-1.5B: 67.2% — BELOW 72% gate.

DECISION: INT4 is default. INT3 is the "efficient" memory option.

Catalog labels:
  "balanced"   → INT4 g=16 AWQ  (default: 70.6% arc_easy, 2.1x compress)
  "efficient"  → INT3 g=16      (67.2% arc_easy, 2.8x compress; -3.4pp tradeoff)
  "ultra"      → INT2 AQLM      (pending; naive INT2 is incoherent — never expose)

For ≤1B models: INT3 is not recommended (gemma-3-1b drops -15.2pp).
For 1.5B models: INT3 is available but marginal (-3.4pp).
For 7B+: INT3 likely safe at similar delta (not yet measured with current squish).
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
