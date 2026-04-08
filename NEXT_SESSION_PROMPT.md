# NEXT_SESSION_PROMPT.md — Wave 47: Post-W46 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 46 is COMPLETE and committed.
- squish/squash/governor.py — AuditEntry dataclass + AgentAuditLogger (append-only JSONL,
  SHA-256 forward hash chain, EU AI Act Art. 12). _hash_text(). get_audit_logger() singleton.
  governor.py: 599 lines (under 600 constraint met).
- squish/squash/integrations/langchain.py — SquashAuditCallback extends SquashCallback.
  Logs llm_start/llm_end with hashed prompt/response + latency. Never raises on logger error.
- squish/squash/cli.py — squash audit show (--n, --log, --json) + squash audit verify (exit 0/2).
- squish/squash/api.py — GET /audit/trail (limit, log params; count + log_path + entries).
- tests/test_squash_wave46.py — 66 tests, 0 failures. Full suite: 4907 passed.
- Module count: 122 (net zero — no new modules).

Wave 45 is COMPLETE and committed.
- squish/squash/mcp.py — McpScanner (6 threat classes) + McpSigner (Sigstore keyless)
- squash attest-mcp CLI subcommand (squish/squash/cli.py)
- POST /attest/mcp REST endpoint (squish/squash/api.py)
- mcp-strict policy added to AVAILABLE_POLICIES (squish/squash/policy.py)
- squish/squash/eval_binder.py DELETED; EvalBinder canonical: sbom_builder.py
- Module count: 122 (net zero: mcp.py +1, eval_binder.py −1)

--- Open questions ---

- mixed_attn lm_eval validation still pending (code-complete since W41)
- INT2 AQLM / SpQR experimental stubs — begin only after mixed_attn harness confirmed
- McpSigner.sign() requires sigstore OIDC flow — needs integration test on hardware

--- Next wave candidates (in priority order) ---

1. `squish serve --mcp-gate` (W47): reject inference requests whose tool catalog fails
   mcp-strict scan at serve time. Integrates McpScanner into the hot path.
2. Prometheus metrics export from the audit trail: expose audit entry counts + chain
   health as /metrics endpoint for SOC/SIEM dashboards.
3. mixed_attn lm_eval harness gate: run arc_easy on mixed_attn model, close the
   accuracy-validation debt from W41.
4. INT2 AQLM codebook implementation (experimental — promote only after mixed_attn confirmed).

--- Done-when for next session ---

State the wave purpose before writing code.
All tests pass. Module count ≤ 122. CHANGELOG entry written. lm_eval-waiver if needed.
```


--- Done-when ---

All W45 tests pass; no regressions in full suite; CHANGELOG.md entry; SESSION.md updated;
NEXT_SESSION_PROMPT.md updated; module count checked.
```
