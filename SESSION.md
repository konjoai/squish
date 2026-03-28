# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-03-27

## Last commits
- `0e67a61` — AWQ alpha=0.1 + g=16 INT4 default + INT3 g=16 + max-model-gb OOM guard + mixed_attn format
- `0d2eb81` — Architecture-aware AWQ calibration: detect_model_family(), Qwen3 alpha=0.07 + 25 CoT texts, _MODEL_FAMILY_DEFAULTS, _DEFAULT_AWQ_ALPHA

---

## Quantization status (as of 2026-03-27)

| Format | Code | lm_eval | arc_easy | Notes |
|---|---|---|---|---|
| INT4 + AWQ g=16 | ✅ | ✅ | ~74.2% Qwen2.5-1.5B | Production default |
| INT3 g=16 | ✅ | ⚠️ PENDING | — | Hypothesis: +3–5pp over old g=32 |
| mixed_attn | ✅ | ⚠️ PENDING | — | FP16 attn projections + INT4 g=16 MLP |
| Qwen3 alpha=0.07 | ✅ | ⚠️ PENDING | — | Verify hellaswag inversion resolves |
| INT2 naive | ✅ | ❌ broken | ~27–32% | Coherence collapse. Never ship. |
| INT2 AQLM | stub | ⚠️ unrun | — | Run after INT3 confirmed |

---

## Open questions (must answer with lm_eval)

1. **Does INT3 g=16 hit >72% arc_easy on Qwen2.5-1.5B?**  
   Decision gate: ≥72% → parity with INT4 within noise → INT3 becomes default.  
   < 72% → keep INT4 as default; INT3 as memory-efficiency option.

2. **Does mixed_attn improve piqa/winogrande vs INT4 g=16?**  
   Theory: keeping q/k/v in FP16 preserves attention sink stability.

3. **Does Qwen3 alpha=0.07 resolve the hellaswag inversion?**  
   Symptom: INT3 > INT4 on hellaswag before the AWQ fix (anomalous, indicates
   oversmoothing at alpha=0.10 for Qwen3's GQA structure).

4. **INT2 AQLM path** — only begin after Q1 and Q2 are answered.

---

## Known test issues

- `test_int4_conversion_and_round_trip` — requires Rust `squish_quant` extension
  (maturin build). Skip in CI without Rust toolchain. Not a regression.

---

## Immediate next task

See `NEXT_SESSION_PROMPT.md` for the full agent prompt.

Short version:
1. Fix `run_overnight_bench.py` MODEL_PLAN — ensure Qwen3-4B is `[4, 3, 2] True`, tests clean
2. Run INT3 g=16 ablation: `squish compress --format int3` on Qwen2.5-1.5B-bf16 → lm_eval arc_easy
3. Run mixed_attn benchmark: same model → lm_eval arc_easy / piqa / winogrande
4. Run Qwen3 alpha=0.07 validation: Qwen3-0.6B before/after architecture-aware AWQ

---

## Model catalog decision tree

```
INT3 g=16 arc_easy ≥72%?
  YES → INT3 is default. INT4 is quality flag (--int4 / --quality).
  NO  → INT4 is default. INT3 is memory-efficiency option (--int3 / --efficient).

Either way: ship all pre-squished variants (INT4 + INT3 + INT2-AQLM once ready).
Catalog labels: "balanced" (INT4), "efficient" (INT3), "ultra" (INT2-AQLM).
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
