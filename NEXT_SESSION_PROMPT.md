# NEXT_SESSION_PROMPT.md — Wave 51: Post-W50 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 50 is COMPLETE and committed.
- squish/squash/integrations/kubernetes.py — Shadow AI detection layer:
  SHADOW_AI_MODEL_EXTENSIONS frozenset (.gguf, .safetensors, .bin, .pt,
  .ckpt, .pkl, .pth, .onnx, .tflite, .mlmodel).
  ShadowAiConfig dataclass — controls which pod locations are scanned.
  ShadowAiHit / ShadowAiScanResult dataclasses.
  scan_pod_for_model_files(pod_spec, config) — scans host_path volumes,
  volume mounts, env vars, args (and initContainers). Stdlib only, no K8s SDK.
  ShadowAiScanner.scan_pod_list() / scan_namespace() — batch scanner.
  WebhookConfig: shadow_ai_scan_mode: bool = False added.
  ANNOTATION_SHADOW_AI = "squash.ai/shadow-ai-detected" added.
- squish/squash/cli.py — squash shadow-ai scan command.
  Supports stdin ('-'), --namespace, --extensions, --output-json, --fail-on-hits.
  Exit 0 = clean, 1 = error, 2 = hits+fail.
- tests/test_squash_wave50.py — 65 tests (all pass).
  Full suite: 5166 passed, 0 failed. Module count: 124 (unchanged).

--- W51 task ---

Wave 51: SBOM drift detection — MEDIUM, 2w · P5

Purpose: detect when a running model's actual files diverge from its attested BOM.
Allows CISO teams to detect post-deployment tampering or silent model swaps.

Scope:
- squish/squash/drift.py — new module (module count: 124→125, justified: new security domain)
  DriftConfig(bom_path, model_dir, tolerance: float=0.0)
  DriftHit(path, expected_digest, actual_digest)
  DriftResult(hits, files_checked, ok, summary)
  check_drift(config: DriftConfig) -> DriftResult — compares SHA-256 digests
    from the BOM against the model directory on disk.
- squish/squash/cli.py — squash drift-check <model_dir> --bom <path> [--fail-on-drift]
  [--output-json PATH] [--quiet]. Exit 0 = clean, 1 = error, 2 = drift+fail.
- tests/test_squash_wave51.py — unit + integration:
  clean model passes, tampered file fails, missing file fails, extra files ignored,
  --fail-on-drift exit 2, --output-json written, CLI help flags.

Hard constraints:
- No external dependencies (use hashlib, stdlib only).
- BOM format: CycloneDX JSON (components[].hashes[].content for SHA-256).
- Module count: 125 after this wave (one new module: squash/drift.py).

Done-when:
1. All W51 tests pass, no regressions in full suite.
2. CHANGELOG.md entry written.
3. SESSION.md + NEXT_SESSION_PROMPT updated.
4. Module count checked (should be 125).
5. git add -A && git commit -m "feat(squash): W51 SBOM drift detection — check_drift + squash drift-check CLI" && git push
```
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
