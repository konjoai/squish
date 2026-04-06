# NEXT_SESSION_PROMPT.md — Wave 40: squish-native lm_eval harness

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Opening prompt

```
Code only. Single-wave prompt. State the wave purpose before writing any code.

Repo: /Users/wscholl/squish
Read SESSION.md and CLAUDE.md before writing anything.

--- Context ---

Wave 39 is COMPLETE and committed.
- CLAUDE.md per-model validated results table expanded from 11 → 25 rows (all Tiers 1–3)
- All Tier 2/3 bench data now in table (0.6B–8B, with winogrande + openbookqa columns)
- Key UNSAFE findings added: gemma-3-4b INT3 (−16.4pp), Qwen3-4B INT3 (−14.8pp)
- No code changes. 4562 tests passing, 4 pre-existing wave12x failures, 25 skipped.

Wave 38 AQLM status:
- squish/quant/aqlm.py: AQLMConfig, AQLMCodebook, AQLMLayer, aqlm_dequantize (decode side only)
- Encode side (compress → AQLM weights) NOT yet implemented

--- Wave 40 priority (single-wave scope) ---

PRIMARY: squish-native lm_eval harness
Build dev/benchmarks/squish_lm_eval.py that evaluates squish .npy-dir format models.

Problem: squish compress --format int4 / --format mixed_attn writes squish npy-dir format
(not mlx safetensors). The existing bench_lmeval_all_models.py uses mlx_lm.load() which
CANNOT load npy-dir. So mixed_attn and INT4 AWQ accuracy are completely unmeasured.

Solution: Write a harness using squish's own compressed_loader.py to load the model,
then shell into mlx_lm.evaluate with the loaded model.

Acceptance criteria:
1. dev/benchmarks/squish_lm_eval.py exists and is runnable
2. Runs arc_easy on Qwen2.5-1.5B-Instruct-int4 (squish npy-dir format) and produces a score
3. Score lands within 2pp of the mlx_lm.convert INT4 baseline (70.6% arc_easy)
4. At least one integration test in tests/ covers the harness entrypoint
5. No regressions in full suite (4562 passing, 4 pre-existing wave12x failures expected)

IF the lm_eval harness is blocked (API mismatch, mlx_lm evaluate internals changed):
FALLBACK: INT2 AQLM encode path — write the compress-side AQLM quantizer in squish/quant/aqlm.py
using the existing AQLMCodebook/AQLMLayer decode targets.

--- Module count note ---
squish/ is at 107 modules (justified: closes existing stubs). Any new file must either
delete an existing file or carry written justification in the commit message.

--- Done when ---
1. 0 failing tests in full suite (4 pre-existing wave12x failures are expected)
2. Harness produces reproducible arc_easy score within 2pp of baseline (or lm_eval-waiver filed)
3. CHANGELOG entry written
4. SESSION.md + NEXT_SESSION_PROMPT updated
5. Module count checked
```
