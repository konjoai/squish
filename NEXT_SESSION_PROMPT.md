# NEXT_SESSION_PROMPT.md — Wave 49: Post-W48 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 48 is COMPLETE and committed.
- squish/squash/lineage.py — TransformationEvent + LineageVerifyResult dataclasses.
  LineageChain (Merkle-chained audit ledger): create_event(), record(), load(), verify().
  Chain file: ".lineage_chain.json" per model directory. SHA-256 Merkle chain.
  Regulatory drivers: EU AI Act Annex IV (Art. 11), NIST AI RMF GOVERN 1.7,
  M&A model transfer provenance. Stdlib only.
- squish/squash/cli.py — squash lineage record/show/verify subcommands.
  lineage record: --operation (required), --model-id, --input-dir, --params KEY=VALUE.
  lineage show: [--json]. lineage verify. Exit 0=ok/intact, 1=user error, 2=tampered/missing.
- squish/squash/api.py — POST /lineage/record, GET /lineage/show, POST /lineage/verify.
  LineageRecordRequest (model_dir, operation, model_id, input_dir, params).
  LineageVerifyRequest (model_dir).
- squish/cli.py — non-fatal auto-hook in _cmd_compress_inner: every
  "squish compress" run automatically records a lineage event (except ImportError silently).
- tests/test_squash_wave48.py — 69 tests (all pass). Full suite: 5033 passed, 0 failed.
- Module count: 124 (+1 lineage.py, justified: EU AI Act Annex IV).

Wave 47 is COMPLETE.
- squish/squash/rag.py — RagScanner (index/verify), RagManifest, RagFileEntry,
  RagDriftItem, RagVerifyResult (57 tests). Module count: 123.

Wave 46 is COMPLETE and committed (ed27727).
- squish/squash/governor.py — AgentAuditLogger JSONL hash chain.
  squash audit show/verify CLI + GET /audit/trail REST. Module count: 122.

Wave 45 is COMPLETE and committed.
- squish/squash/mcp.py — McpScanner + McpSigner (Sigstore keyless). Module count: 122.

--- Open questions ---

- mixed_attn lm_eval validation still pending (code-complete since W41)
- INT2 AQLM / SpQR experimental stubs — begin only after mixed_attn harness confirmed
- McpSigner.sign() requires sigstore OIDC flow — needs integration test on hardware
- lineage records persist but are never signed — a natural W49 follow-on

--- Next wave candidates (in priority order) ---

1. `squash lineage sign` (W49): sign the lineage chain file with Sigstore (mirrors
   McpSigner pattern). Adds cryptographic non-repudiation to the lineage chain.
2. `squash scan-rag sign` (W49 alt): sign the RAG manifest (same Sigstore pattern).
3. `squish serve --lineage-gate`: reject inference requests when the model's lineage
   verify() returns ok=False (auto-tamper detection at serve time).
4. Prometheus metrics export from audit trail + lineage: /metrics endpoint.
5. mixed_attn lm_eval harness gate: close the accuracy-validation debt from W41.
```

--- Done-when for next session ---


State the wave purpose before writing code.
All tests pass. Module count ≤ 122. CHANGELOG entry written. lm_eval-waiver if needed.
```


--- Done-when ---

All W45 tests pass; no regressions in full suite; CHANGELOG.md entry; SESSION.md updated;
NEXT_SESSION_PROMPT.md updated; module count checked.
```
