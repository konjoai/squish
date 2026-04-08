# NEXT_SESSION_PROMPT.md — Wave 46: Post-W45 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 45 is COMPLETE and committed.
- squish/squash/mcp.py — McpScanner (6 threat classes) + McpSigner (Sigstore keyless)
- squash attest-mcp CLI subcommand (squish/squash/cli.py)
- POST /attest/mcp REST endpoint (squish/squash/api.py)
- mcp-strict policy added to AVAILABLE_POLICIES (squish/squash/policy.py)
- squish/squash/eval_binder.py DELETED (12-line shim; EvalBinder canonical: sbom_builder.py)
- All 8 eval_binder callers redirected to squish.squash.sbom_builder
- tests/test_squash_wave45.py — 110 tests, 0 failures
- Module count: 122 (net zero: mcp.py +1, eval_binder.py −1)

--- Open questions ---

- mixed_attn lm_eval validation still pending (code-complete since W41)
- INT2 AQLM / SpQR experimental stubs — begin only after mixed_attn harness confirmed
- McpSigner.sign() requires sigstore OIDC flow — needs integration test on hardware

--- Next wave candidates (in priority order) ---

1. McpScanner server-side integration: `squish serve --mcp-gate`
   — reject requests whose tool catalog fails mcp-strict scan at serve time.
2. mixed_attn lm_eval harness gate: run arc_easy on mixed_attn model, close the
   accuracy-validation debt from W41.
3. INT2 AQLM codebook implementation (experimental — promote only after mixed_attn confirmed).

--- Done-when for next session ---

State the wave purpose before writing code.
All tests pass. Module count ≤ 122. CHANGELOG entry written. lm_eval-waiver if needed.
```

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
