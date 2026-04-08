# NEXT_SESSION_PROMPT.md — Wave 44: fix mixed_attn compress + Azure DevOps integration

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 43 is COMPLETE and committed.
- .github/workflows/publish-orb.yml — CircleCI Orb publish (dev + prod)
- .github/workflows/publish-helm.yml — Helm chart push to GHCR OCI (Artifact Hub)
- artifacthub-repo.yml — repo metadata (fill in repositoryID when registering on artifacthub.io)
- tests/test_squash_wave43.py — 23 passing YAML-validation tests
- No Python module changes. Module count = 121.

INT4 AWQ benchmark: arc_easy=70.8% + arc_challenge=44.4% confirmed (Wave 42).
hellaswag/winogrande/piqa/openbookqa may have finished in background terminal 3af20b5b.
Check: ls /Users/wscholl/squish/results/_tmp_Qwen2.5-1.5B-Instruct-int4-awq/

--- Wave 44 priorities ---

PRIORITY 1 — Check if INT4 AWQ remaining 4 tasks finished (terminal 3af20b5b):
  If done: update CLAUDE.md accuracy table TBD cells + SESSION.md.

PRIORITY 2 — Fix mixed_attn compression (AWQ silent failure):
  Root cause: MLP outlier ratio ~28.5 > has_outliers() threshold 20.0 without AWQ pre-scaling
  → most MLP weights stored as BF16 passthrough (~59 q4a files vs expected 245).
  Steps:
  1. rm -rf ~/models/Qwen2.5-1.5B-Instruct-mixed-attn
  2. squish compress --format mixed_attn ~/models/Qwen2.5-1.5B-Instruct-bf16 ~/models/Qwen2.5-1.5B-Instruct-mixed-attn 2>&1
     (do NOT pipe to tail — must see full output to detect AWQ failure)
  3. Verify: find ~/models/Qwen2.5-1.5B-Instruct-mixed-attn -name '__q4a.npy' | wc -l
     Target ≈ 245. If < 100, AWQ still failing — investigate squish/convert.py outlier threshold.

PRIORITY 3 — Azure DevOps integration (squish/squash/integrations/azure_devops.py)

--- Required secrets (for Wave 43 workflows to run) ---
CIRCLECI_TOKEN — CircleCI personal API token with orb:write scope
  Add at: https://github.com/squishai/squish/settings/secrets/actions

--- Done-when ---
1. 0 failing tests in full suite
2. CHANGELOG.md entry written
3. SESSION.md + NEXT_SESSION_PROMPT.md updated
4. All tests passing: git add / commit / push
```
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
