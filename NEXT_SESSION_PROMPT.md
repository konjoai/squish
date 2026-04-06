# NEXT_SESSION_PROMPT.md — Wave 39: CLAUDE.md table update + mixed_attn lm_eval harness

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Opening prompt

```
Code only. Single-wave prompt. State the wave purpose before writing any code.

Repo: /Users/wscholl/squish
Read SESSION.md and CLAUDE.md before writing anything.

--- Context ---

Wave 38 is COMPLETE and committed.
- squish/quant/aqlm.py implemented (AQLMConfig, AQLMCodebook, AQLMLayer, aqlm_dequantize)
- closes stub import at compressed_loader.py:664
- 30 new tests, 4562 total passing, 4 pre-existing wave12x failures, 25 skipped

Qwen3-4B-int4 re-run is COMPLETE:
- results/lmeval_Qwen3-4B-int4_20260401T103031.json
- 73.2% arc_easy (thinking disabled, limit=500) ✅

All Tier 2/3 bench data is valid and present in results/.

--- Wave 39 options (priority order) ---

1. CLAUDE.md per-model accuracy table update — update the per-format validated
   results table in CLAUDE.md with all current Tier 2/3 data. No code changes.
   Highest value, zero code risk.

   Key updates:
   - Qwen3-4B-int4: 73.2% arc_easy ✅ (was 41% invalid at thinking-on)
   - Qwen3-4B-int3: 58.4% (−14.8pp — INT3-unsafe, same risk class as gemma-3-1b)
   - Qwen3-8B-int4: 79.2% ✅
   - Qwen3-8B-int3: 71.4% (−7.8pp)
   - Qwen2.5-7B-int4: 83.0% ✅
   - Qwen2.5-7B-int3: 79.0% (−4.0pp)
   - Qwen3-0.6B: int4=34.0, int3=36.4 (within noise), int2=27.0 (incoherent)
   - Llama-3.2-1B: int4=40.0, int3=37.2, int2=27.2

2. squish-native lm_eval harness — build squish_lm_eval.py in dev/benchmarks/
   that runs mlx_lm.evaluate on squish npy-dir format models (needed to measure
   mixed_attn and INT4 AWQ g=16).
   
   Acceptance: harness runs on Qwen2.5-1.5B-Instruct npy-dir, produces arc_easy score.

3. INT2 AQLM encode path (if Wave 39 scope allows) — write AQLM quantizer
   (compress side) using the aqlm.py decode target.

--- Done when ---
1. 0 failing tests in full suite (4 pre-existing wave12x failures are expected)
2. CLAUDE.md table is complete, accurate, and citable
3. CHANGELOG entry written
4. SESSION.md + NEXT_SESSION_PROMPT updated
5. No new files unless justified (module count check)
```
