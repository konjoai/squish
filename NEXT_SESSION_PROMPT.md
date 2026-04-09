# NEXT_SESSION_PROMPT.md — Wave 52: Post-W51 context

> Paste the content below verbatim as your opening prompt.

---

## Opening prompt

```
Code session. Read SESSION.md and CLAUDE.md first.

Repo: /Users/wscholl/squish

--- Context ---

Wave 51 is COMPLETE and committed.
- squish/squash/drift.py — NEW module (module count: 125).
  DriftConfig(bom_path, model_dir, tolerance: float=0.0) dataclass.
  DriftHit(path, expected_digest, actual_digest) with .missing/.tampered props.
  DriftResult(hits, files_checked, ok, summary) with auto-built summary.
  _parse_bom_hashes(bom) — reads squish:weight_hash:* component properties.
  check_drift(config) — SHA-256 comparison of BOM against disk files.
  Stdlib only: hashlib, json, pathlib. No external deps.
  BOM format: squish CycloneDX sidecar (cyclonedx-mlbom.json).
- squish/squash/cli.py — squash drift-check <model_dir> --bom <path>.
  Supports --fail-on-drift (exit 2), --output-json, --quiet.
  Exit 0 = clean, 1 = error, 2 = drift found + --fail-on-drift.
- tests/test_squash_wave51.py — 54 tests (all pass).
  Full suite: 5220 passed, 0 failed. Module count: 125.

Wave 50 is COMPLETE and committed.
- squish/squash/integrations/kubernetes.py — Shadow AI detection layer:
  SHADOW_AI_MODEL_EXTENSIONS frozenset (.gguf, .safetensors, .bin, .pt,
  .ckpt, .pkl, .pth, .onnx, .tflite, .mlmodel).
  ShadowAiConfig / ShadowAiHit / ShadowAiScanResult / ShadowAiScanner.
  scan_pod_for_model_files(pod_spec, config) — stdlib only, no K8s SDK.
  WebhookConfig: shadow_ai_scan_mode: bool = False added.
- squish/squash/cli.py — squash shadow-ai scan command.
- tests/test_squash_wave50.py — 65 tests (all pass).

--- W52 task ---

Wave 52: Drift-check REST endpoint — SMALL, 1w · P4

Purpose: expose drift-check over HTTP so CI/CD pipelines and policy agents
can call it without spawning a subprocess.

Scope:
- squish/squash/api.py — POST /drift-check — accepts JSON body:
    { "model_dir": "/abs/path", "bom_path": "/abs/path" }
  Returns 200 + DriftResult as JSON on success (ok=true or ok=false).
  Returns 422 + { "error": "..." } on missing dirs or bad BOM.
  Returns 500 + { "error": "..." } on unexpected errors.
  Drift found does NOT cause a non-2xx status — caller decides based on
  "ok": false in response body. --fail-on-drift is a CLI-only concept.
- tests/test_squash_wave52.py — unit + integration:
  clean model → 200 + ok=true, tampered → 200 + ok=false,
  missing bom_path → 422, invalid JSON BOM → 422, missing model_dir → 422.

Hard constraints:
- No new module (add endpoint to existing api.py). Module count stays 125.
- api.py already imports from squish.squash.drift — no circular imports.
- Endpoint must be mockable in CI (DriftConfig/check_drift fully injectable).

Done-when:
1. All W52 tests pass, no regressions in full suite.
2. CHANGELOG.md entry written.
3. SESSION.md + NEXT_SESSION_PROMPT updated.
4. Module count checked (should be 125 — no new files).
5. git add -A && git commit && git push

--- Open questions ---

- mixed_attn lm_eval validation still pending (code-complete since W41)
- INT2 AQLM / SpQR experimental stubs — begin only after mixed_attn harness confirmed
- McpSigner.sign() requires sigstore OIDC flow — needs integration test on hardware
- lineage records persist but are never signed — a natural W52/W53 follow-on

--- Next wave candidates (after W52) ---

1. `squash lineage sign` (W53): sign the lineage chain file with Sigstore
   (mirrors McpSigner pattern). Adds cryptographic non-repudiation.
2. `squash drift-check --continuous` (W53 alt): poll model_dir at interval,
   emit events on drift (pairs with squash monitor).
```
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
