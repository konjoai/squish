# NEXT_SESSION_PROMPT.md — Wave 50: Post-W49 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 49 is COMPLETE and committed.
- squish/squash/oms_signer.py — _is_offline() helper (SQUASH_OFFLINE=1).
  OmsSigner.sign() returns None immediately when offline (no OIDC calls).
  OmsSigner.keygen(key_name, key_dir) — Ed25519 keypair → .priv.pem / .pub.pem.
  OmsSigner.sign_local(bom_path, priv_key_path) — Ed25519 sig → <bom>.sig (128-char hex).
  OmsSigner.pack_offline(model_dir, output_path) — .squash-bundle.tar.gz.
  OmsVerifier.verify_local(bom_path, pub_key_path, sig_path) → bool.
  Requires cryptography>=42.0 (added to pyproject.toml [squash] optional-deps).
- squish/squash/attest.py — AttestConfig: offline: bool, local_signing_key: Path|None.
  Step 7: offline+key→sign_local, offline+no-key→skip warning, online→sigstore.
- squish/squash/cli.py — squash keygen / squash verify-local / squash pack-offline.
  squash attest --offline --offline-key PATH.
- squish/squash/api.py — POST /keygen, POST /attest/verify-local, POST /pack/offline.
  AttestRequest: offline: bool, local_signing_key: str|None.
- tests/test_squash_wave49.py — 68 tests (all pass). Module count: 124 (unchanged).
- Unblocks: DoD CMMC, EU sovereign AI, healthcare networks, IL4/IL5.

Wave 48 is COMPLETE.
- squish/squash/lineage.py — TransformationEvent + LineageVerifyResult dataclasses.
  LineageChain (Merkle-chained audit ledger): create_event(), record(), load(), verify().
  Chain file: ".lineage_chain.json" per model directory. SHA-256 Merkle chain.
  Regulatory drivers: EU AI Act Annex IV (Art. 11), NIST AI RMF GOVERN 1.7,
  M&A model transfer provenance. Stdlib only.
- squish/squash/cli.py — squash lineage record/show/verify subcommands.
- squish/squash/api.py — POST /lineage/record, GET /lineage/show, POST /lineage/verify.
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
