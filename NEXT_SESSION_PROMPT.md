# NEXT_SESSION_PROMPT.md — Wave 43: finish INT4 AWQ benchmark + fix mixed_attn compress

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 42 is COMPLETE and committed.
- 6 bugs fixed in squish/quant/compressed_loader.py + 1 in squish/cli.py
- CRITICAL FIX: bias formula corrected (biases = zeros, not -zeros/scales)
- INT4 AWQ g=32 arc_easy = 70.8% PASS (lm_eval-waiver CLOSED)
- arc_challenge = 44.4% (+0.8pp vs baseline)
- 4 remaining tasks (hellaswag/winogrande/piqa/openbookqa) may STILL BE RUNNING in background terminal 3af20b5b
- Tests: 4642 passed, 0 failures

--- Wave 43 priorities ---

PRIORITY 1 — Check if INT4 AWQ benchmark finished:
  ls /Users/wscholl/squish/results/_tmp_Qwen2.5-1.5B-Instruct-int4-awq/
  # Should show hellaswag/ winogrande/ piqa/ openbookqa/ if done
  # Read each: python3 -c "import json; d=json.load(open('<path>/eval__*')); print(d.get('<task>',{}).get('acc_norm,none'))"
  If done: update CLAUDE.md accuracy table TBD cells + SESSION.md.

PRIORITY 2 — Fix mixed_attn compression (AWQ failure):
  Root cause: raw Qwen2.5 MLP weights have outlier ratio ~28.5 > has_outliers() threshold 20.0
  WITHOUT AWQ pre-scaling → most MLP weights stored as BF16 passthrough (~59 q4a vs expected 245).
  Steps:
  1. rm -rf ~/models/Qwen2.5-1.5B-Instruct-mixed-attn
  2. squish compress --format mixed_attn ~/models/Qwen2.5-1.5B-Instruct-bf16 ~/models/Qwen2.5-1.5B-Instruct-mixed-attn 2>&1
     (do NOT pipe to tail — must see full output to detect AWQ failure)
  3. Verify: find ~/models/Qwen2.5-1.5B-Instruct-mixed-attn -name '__q4a.npy' | wc -l
     Target: ≈ 245 (same as INT4 AWQ). If < 100, AWQ still failing — investigate convert.py.
  4. Run benchmark (only after count confirmed):
     python3 dev/benchmarks/squish_lm_eval.py \
         --npy-dir ~/models/Qwen2.5-1.5B-Instruct-mixed-attn \
         --model-dir ~/models/Qwen2.5-1.5B-Instruct-bf16 \
         --limit 500 --baseline 70.6

PRIORITY 3 (code wave) — Azure DevOps integration:
  squish/squash/integrations/azure_devops.py (same pattern as vertex_ai.py, Wave 40)

--- Done when ---
1. INT4 AWQ all 6 tasks recorded in CLAUDE.md + SESSION.md
2. mixed_attn compress verified (245 q4a) OR bug documented with fix plan
3. mixed_attn benchmark run or lm_eval-waiver filed
4. CHANGELOG + SESSION.md + NEXT_SESSION_PROMPT updated
5. Commit + push
```
