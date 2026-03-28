# NEXT_SESSION_PROMPT.md — Agent Prompt for Next Session

> Copy the block below verbatim as your opening prompt.
> This is a **code session** with one research confirmation at the end.

---

## Prompt

**Code session. Minimum viable. Read SESSION.md first.**

Read `/Users/wscholl/squish/SESSION.md` and `/Users/wscholl/squish/CLAUDE.md` before
writing a single line of code. Tell me the current open questions and what's
code-complete but unvalidated before you proceed.

---

### Task 1 — Fix overnight bench MODEL_PLAN (code-only, 5 minutes)

Read `dev/benchmarks/run_overnight_bench.py` lines 85–120. Verify that:
- `Qwen3-4B` appears as `("Qwen3-4B", "Qwen3-4B-bf16", [4, 3, 2], True)`
- No `Qwen3-8B` entry with `[3, 2] False` that conflicts with test expectations

Then run `python3 -m pytest tests/test_overnight_bench_unit.py -v` and confirm
all tests pass. Fix any MODEL_PLAN mismatch if found. Do NOT add any new tests
or new files — fix the source data only.

**Done when:** `pytest tests/test_overnight_bench_unit.py` exits 0, 0 failures.

---

### Task 2 — INT3 g=16 ablation (hardware required)

Run the INT3 ablation on Qwen2.5-1.5B-bf16:

```bash
squish compress Qwen2.5-1.5B-bf16 --format int3
lm_eval --model mlx_lm --model_args pretrained=<output_dir> \
        --tasks arc_easy --num_fewshot 0 --limit 200
```

Record the arc_easy result. Compare to the last committed INT4 baseline (~74.2%).

**Decision gate:**
- ≥72% → INT3 g=16 hits parity within noise. Make INT3 the default format.
- <72%  → Keep INT4 as default. INT3 is the `--efficient` option.

Write the result to `results/int3_g16_ablation_<date>.json` in the standard
benchmark result format (hardware metadata included).

**Waiver format (if not running on M3 this session):**
```
# lm_eval-waiver: M3 hardware not available this session
# expected-delta: +3–5pp over old g=32 baseline (72–74% arc_easy expected)
# validation-run: queued for next bench session
```

---

### Task 3 — mixed_attn benchmark (hardware required)

Run the mixed_attn format on Qwen2.5-1.5B-bf16:

```bash
squish compress Qwen2.5-1.5B-bf16 --format mixed_attn
lm_eval --model mlx_lm --model_args pretrained=<output_dir> \
        --tasks arc_easy,piqa,winogrande --num_fewshot 0 --limit 200
```

Record arc_easy, piqa, and winogrande. Compare to INT4 g=16 on the same tasks.
Check peak RSS during inference — mixed_attn should match or beat INT4 g=16.

Write result to `results/mixed_attn_ablation_<date>.json`.

**Waiver format (if not running this session):**
```
# lm_eval-waiver: M3 hardware not available this session
# expected-delta: +1–2pp piqa/winogrande vs INT4 g=16 (FP16 attn hypothesis)
# validation-run: queued for next bench session
```

---

### Task 4 — Qwen3 alpha=0.07 validation (hardware required)

Squish Qwen3-0.6B-bf16 twice:
1. With explicit `--awq-alpha 0.10` (old behaviour)
2. With no `--awq-alpha` flag (auto-detect → should print `arch=qwen3, alpha=0.07`)

Run lm_eval arc_easy + hellaswag on both (limit=200):
```bash
lm_eval --tasks arc_easy,hellaswag --num_fewshot 0 --limit 200
```

**What we're watching for:** The hellaswag inversion — at alpha=0.10, INT3 scored
higher than INT4 on hellaswag (anomalous; sign of oversmoothing). If alpha=0.07
resolves this, the gap disappears or reverses. Document the before/after delta.

Write result to `results/qwen3_awq_alpha_ablation_<date>.json`.

---

### Task 5 — Update SESSION.md with results

After any tasks above, update SESSION.md:
- Fill in the lm_eval results in the quantization status table
- Answer the open questions with real numbers
- Update the model catalog decision tree with the confirmed default

---

### Task 6 — Commit

For each task that completes with an lm_eval result:
```
feat(quant): <task> — arc_easy <X>% (<+/-Npp vs baseline>)
```

For each task that completes code-only (no lm_eval):
```
fix(<scope>): <task>
# lm_eval-waiver: ...
# expected-delta: ...
# validation-run: queued
```

---

### Research question (text only, no code, no tool calls)

Once the ablation results are in: given INT3 g=16 arc_easy result, piqa/winogrande
from mixed_attn, and the Qwen3 alpha delta — what should the default quantization
format be for Squish v10? INT3 default + INT4 quality flag? Or something else?
Output text only. Give me the decision, not just the options.
