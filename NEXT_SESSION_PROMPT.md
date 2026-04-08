# NEXT_SESSION_PROMPT.md — Wave 48: Post-W47 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 47 is COMPLETE and committed.
- squish/squash/rag.py — RagScanner (index/verify), RagManifest, RagFileEntry,
  RagDriftItem, RagVerifyResult. MANIFEST_FILENAME=".rag_manifest.json".
  index() walks corpus with configurable glob, SHA-256 hashes every file, writes
  tamper-evident manifest with manifest_hash (sha256 of sorted JSON file list).
  verify() catches added/removed/modified drift; returns ok=False + missing_manifest
  item when no manifest. Stdlib only.
- squish/squash/cli.py — squash scan-rag index <corpus_dir> [--glob] [--quiet]
  + squash scan-rag verify <corpus_dir> [--json] [--quiet]. Exit 0=ok, 1=user error, 2=drift.
- squish/squash/api.py — POST /rag/index + POST /rag/verify.
  RagIndexRequest + RagVerifyRequest Pydantic models.
- tests/test_squash_wave47.py — 57 tests (all pass). Full suite: 4964 passed, 0 failed.
- Module count: 123 (+1 rag.py, pre-authorised W45 slot).

Wave 46 is COMPLETE and committed (ed27727).
- squish/squash/governor.py — AgentAuditLogger JSONL hash chain (EU AI Act Art. 12).
- squash audit show / audit verify CLI + GET /audit/trail REST.
- Module count: 122.

Wave 45 is COMPLETE and committed.
- squish/squash/mcp.py — McpScanner (6 threat classes) + McpSigner (Sigstore keyless).
- squash attest-mcp CLI + POST /attest/mcp REST.
- Module count: 122 (net zero: mcp.py +1, eval_binder.py −1).

--- Open questions ---

- mixed_attn lm_eval validation still pending (code-complete since W41)
- INT2 AQLM / SpQR experimental stubs — begin only after mixed_attn harness confirmed
- McpSigner.sign() requires sigstore OIDC flow — needs integration test on hardware

--- Next wave candidates (in priority order) ---

1. `squash scan-rag sign` (W48): sign the manifest with Sigstore (mirrors McpSigner pattern).
   Adds cryptographic non-repudiation to the corpus integrity system.
2. `squish serve --rag-gate` (W48 alt): reject inference requests when the RAG corpus fails
   verify() at serve time. Integrates RagScanner into the hot path.
3. Prometheus metrics export from the audit trail: expose audit entry counts + chain health
   as /metrics endpoint for SOC/SIEM dashboards.
4. mixed_attn lm_eval harness gate: run arc_easy on mixed_attn model, close the
   accuracy-validation debt from W41.
5. INT2 AQLM codebook implementation (experimental — promote only after mixed_attn confirmed).
```

--- Done-when for next session ---

State the wave purpose before writing code.
All tests pass. Module count ≤ 122. CHANGELOG entry written. lm_eval-waiver if needed.
```


--- Done-when ---

All W45 tests pass; no regressions in full suite; CHANGELOG.md entry; SESSION.md updated;
NEXT_SESSION_PROMPT.md updated; module count checked.
```
