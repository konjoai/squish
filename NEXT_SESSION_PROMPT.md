# NEXT_SESSION_PROMPT.md — Wave 45: Post-W44 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 44 is COMPLETE and committed.
- integrations/azure-devops/vss-extension.json — ADO Marketplace manifest
- integrations/azure-devops/SquashAttestTask/task.json — SquashAttest@1 task definition
- integrations/azure-devops/SquashAttestTask/run_squash.ps1 — cross-platform PS runner
- integrations/azure-devops/SquashAttestTask/run_squash.sh — bash companion runner
- squish/squash/integrations/azure_devops.py — Python adapter (module #122)
- tests/test_squash_wave44.py — N tests passing
- CI/CD matrix: GitHub Actions ✅ GitLab ✅ Jenkins ✅ Argo ✅ CircleCI ✅ Azure DevOps ✅

Module count: 122 (1 above the 100-file ceiling — see W44 CHANGELOG justification).

--- ADO publishing (action required by user) ---

To publish the extension to ADO Marketplace:
  1. Register publisher "squishai" at marketplace.visualstudio.com
  2. cd integrations/azure-devops
  3. npm install -g tfx-cli
  4. AZURE_DEVOPS_EXT_PAT=<your-pat> tfx extension publish --manifest-globs vss-extension.json

--- Wave 45 priorities ---

PRIORITY 1 — Check INT4 AWQ remaining 4 tasks (from W42):
  ls /Users/wscholl/squish/results/_tmp_Qwen2.5-1.5B-Instruct-int4-awq/
  If done: update CLAUDE.md accuracy table + SESSION.md.

PRIORITY 2 — Fix mixed_attn compression (if not already done):
  See SESSION.md for steps.

PRIORITY 3 — Prometheus/Grafana metrics export from squash attest runs
  OR Datadog integration (follows same pattern as azure_devops.py).

--- Done-when ---

All W45 tests pass; no regressions in full suite; CHANGELOG.md entry; SESSION.md updated;
NEXT_SESSION_PROMPT.md updated; module count checked.
```
