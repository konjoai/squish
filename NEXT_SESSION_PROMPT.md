# NEXT_SESSION_PROMPT.md — Wave 42: validate INT4 AWQ lm_eval score + next integration

> Paste the content below verbatim as your opening prompt.
> Start with hardware validation (run the harness), then pick the next code wave.

---

## Opening prompt

```
Research only, no code. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 41 is COMPLETE and committed.
- dev/benchmarks/squish_lm_eval.py built (385 lines) — squish-native lm_eval harness
- 52 unit tests in tests/test_squash_wave41.py — all passing
- Full suite: 4642 passed, 0 failures, 25 skipped
- Wave 41 is CODE-COMPLETE but lm_eval-waiver filed (no hardware run yet)

Wave 41 harness strategy:
  squish.quant.compressed_loader.load_from_npy_dir() → builds squish_4bit/ cache (one time)
  → mlx_lm evaluate subprocess on squish_4bit/ standard safetensors
  → same JSON output format as bench_lmeval_all_models.py

--- Wave 42 priorities ---

PRIORITY 1 — Hardware validation run (if hardware available):
  python3 dev/benchmarks/squish_lm_eval.py \
      --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \
      --model-dir ~/models/Qwen2.5-1.5B-Instruct \
      --limit 500 --baseline 70.6

  Target: arc_easy within ±2pp of 70.6% (68.6–72.6%).
  If passes → update CLAUDE.md accuracy table (fill in INT4 AWQ n/a rows).
  If fails → file regression report, investigate squish_4bit/ construction.

PRIORITY 2 — mixed_attn harness run (if ~/models/Qwen2.5-1.5B-Instruct-mixed-attn exists):
  python3 dev/benchmarks/squish_lm_eval.py \
      --npy-dir ~/models/Qwen2.5-1.5B-Instruct-mixed-attn \
      --model-dir ~/models/Qwen2.5-1.5B-Instruct \
      --limit 200 --tasks arc_easy

PRIORITY 3 (code wave) — Azure DevOps Extension:
  squish/squash/integrations/azure_devops.py
  Same pattern as vertex_ai.py (Wave 40).

PRIORITY 4 (code wave) — INT2 AQLM encode path:
  squish/quant/aqlm.py already has the decode side (Wave 38).
  Encode side needed: compress-side AQLM quantizer.

--- Done when (for any code wave) ---
1. 0 failing tests in full suite
2. Memory + latency contracts measured or lm_eval-waiver filed
3. CHANGELOG entry written
4. SESSION.md + NEXT_SESSION_PROMPT updated
5. Module count checked (new module requires deletion or justification)
```
