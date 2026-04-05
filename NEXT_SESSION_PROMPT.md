# NEXT_SESSION_PROMPT.md — Squash Wave 32+: VEX Feed Hosting + Quantization Validation

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Wave 31 is complete (VEX cache management REST endpoints: GET /vex/status, POST /vex/update).
Next priority: Wave 32 — the squash REST API surface is now feature-complete at 28 endpoints.
Focus: (a) VEX feed hosting — commit a static community VEX feed JSON to squishai/vex-feed, and
(b) INT2 AQLM / SpQR experimental path (begin only after mixed_attn lm_eval result is in).
One commit per wave. Minimum viable implementation — no stubs left in shipped code.**

---

## Waves 1–31 complete (commit HEAD on `main`)

### Delivery summary

| Waves | What | Status |
|-------|------|--------|
| 1–13  | CycloneDX SBOM, SPDX, scanner, policy engine, VEX, provenance, Sigstore, eval binder, governor, CLI, REST API, SARIF | ✅ |
| 14–19 | HTML report, VEX cache, policy webhooks, composite attestation, SBOM registry push, advanced policy templates | ✅ |
| 20    | NTIA minimum elements validator (`NtiaValidator`, `ntia-check` CLI, `POST /ntia/validate`) | ✅ |
| 21    | SLSA 1.0 provenance (`SlsaProvenanceBuilder`, L1/L2/L3, `slsa-attest` CLI, `POST /slsa/attest`) | ✅ |
| 22    | BOM merge & composition (`BomMerger`, `merge` CLI, `POST /sbom/merge`) | ✅ |
| 23    | AI risk assessment — EU AI Act + NIST AI RMF (`AiRiskAssessor`, `risk-assess` CLI, `POST /risk/assess`) | ✅ |
| 24    | Drift detection & continuous monitoring (`DriftMonitor`, `monitor` CLI, `POST /monitor/snapshot+compare`) | ✅ |
| 25    | CI/CD runtime adapter — GitHub/Jenkins/GitLab/CircleCI (`CicdAdapter`, `ci-run` CLI, `POST /cicd/report`) | ✅ |
| 26    | SageMaker Pipeline Step, ORAS OCI registry push, VEX feed MVP (`SageMakerSquash`, `OrasAdapter`, `VexFeedManifest`) | ✅ |
| 27    | Kubernetes Admission Webhook (`KubernetesWebhookHandler`, `WebhookConfig`, Helm chart, `squash webhook` CLI) | ✅ |
| 28    | CircleCI Orb (`orb.yml`) + Ray Serve (`squash_serve` decorator, `SquashServeDeployment`) | ✅ |
| 29    | VEX publish CLI (`squash vex-publish`) + integration CLI shims (`attest-mlflow`, `attest-wandb`, `attest-huggingface`, `attest-langchain`) | ✅ |
| 30    | REST API endpoints for Wave 29 CLI additions (`POST /vex/publish`, `/attest/mlflow`, `/attest/wandb`, `/attest/huggingface`, `/attest/langchain`) | ✅ |
| 31    | VEX cache management REST endpoints (`GET /vex/status`, `POST /vex/update`) — closes the last CLI/REST gap | ✅ |

### Test state
- **4326 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over limit — all justified in CHANGELOG, unchanged from wave 30)
  Waves 29–31: 0 new modules (all additions inside cli.py / api.py)
```

### Key files added/changed in wave 31
- `squish/squash/api.py` (extended) — 2 new REST endpoints + 1 request model + 2 counters:
  - `VexUpdateRequest` + `GET /vex/status` + `POST /vex/update`
- `tests/test_squash_wave31.py` — 28 new tests (mock pattern: `patch("squish.squash.vex.VexCache")`)

### Complete REST API surface (28 endpoints, all CLI commands covered)
```
GET  /health, /metrics, /policies, /report, /vex/status
POST /attest, /scan, /policy/evaluate, /vex/evaluate, /vex/update
POST /attest/verify, /webhooks/test, /attest/composed, /sbom/push
POST /ntia/validate, /slsa/attest, /sbom/merge, /risk/assess
POST /monitor/snapshot, /monitor/compare, /cicd/report
POST /vex/publish, /attest/mlflow, /attest/wandb, /attest/huggingface, /attest/langchain
GET  /scan/{job_id}, /scan/{job_id}/sarif
```
Every `squash` CLI subcommand now has a REST equivalent.

---

## Remaining gaps (post wave 31)

### 1. lm_eval validation for mixed_attn (still blocked)
`mixed_attn` (FP16 attn + INT4 MLP) is code-complete but **unvalidated**.
lm_eval result or lm_eval-waiver required before any accuracy claims.
Baseline: INT4 = **70.6% arc_easy** (Qwen2.5-1.5B, limit=500).
Blocked: squish npy-dir format is not loadable by `mlx_lm evaluate`.
To unblock: build a squish npy-dir → mlx safetensors export step (no large models needed — test with synthetic tensors).

### 2. VEX feed hosting
`VexFeedManifest.generate()` is complete; no hosted feed yet.
First step: commit a static JSON community feed to `squishai/vex-feed` on GitHub.
This makes `VexCache.DEFAULT_URL` point to something real, enabling end-to-end VEX tests.

### 3. INT2 AQLM / SpQR experimental path
Begin only after mixed_attn lm_eval result is in. See CLAUDE.md quantization table.

---

## Hard stops

- **Module count is at 106.** Any new file requires deleting one or writing justification in CHANGELOG.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing (4326 passing, 4 pre-existing wave12x failures acceptable).
- **For any REST API additions: integration tests must call the real endpoint (no mocking the handler).**
- **For quantization path changes: lm_eval result or lm_eval-waiver in commit message.**

---
