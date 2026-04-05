# NEXT_SESSION_PROMPT.md — Squash Wave 31+: mixed_attn lm_eval Validation + Next Gaps

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Wave 30 is complete (REST API integration endpoints).
Next priority: Wave 31 — validate `mixed_attn` (FP16 attn + INT4 MLP) with lm_eval,
identify and close the highest-value remaining REST/CLI gaps.
One commit per wave. Minimum viable implementation — no stubs left in shipped code.**

---

## Waves 1–30 complete (commit HEAD on `main`)

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

### Test state
- **4298 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over limit — all justified in CHANGELOG)
  - slsa.py, risk.py, cicd.py: +3 (waves 20-25)
  - integrations/sagemaker.py: +1 (wave 26)
  - integrations/kubernetes.py: +1 (wave 27)
  - integrations/ray.py: +1 (wave 28)
  Waves 29–30: 0 new modules (all additions inside cli.py / api.py)
```

### Key files added/changed in wave 30
- `squish/squash/api.py` (extended) — 5 new REST endpoints + 3 request models:
  - `VexPublishRequest` + `POST /vex/publish`
  - `AttestIntegrationRequest` + `POST /attest/mlflow`, `POST /attest/wandb`, `POST /attest/langchain`
  - `AttestHuggingFaceRequest` + `POST /attest/huggingface`
- `tests/test_squash_wave30.py` — 56 new tests (rate-limiter reset fix: `_rate_window.clear()`)

### Critical bug fixes in wave 30 (relevant for future attest handlers)
- `fail_on_violation` pattern: always pass `fail_on_violation=False` to `AttestConfig`;
  check `req.fail_on_violation` post-run to return 422. Never pass it to AttestConfig directly.
- Policy fallback: `req.policies if req.policies is not None else ["enterprise-strict"]`
  (NOT `req.policies or [...]` — empty list is falsy).
- Rate-limiter test isolation: import and clear `_rate_window` in test fixtures.

---

## Remaining gaps (post wave 30)

### 1. lm_eval validation for mixed_attn (Wave 31 priority)
`mixed_attn` (FP16 attn + INT4 MLP) is code-complete but **unvalidated**.
lm_eval result or lm_eval-waiver required before any accuracy claims.
Baseline: INT4 = **70.6% arc_easy** (Qwen2.5-1.5B, limit=500).

### 2. INT2 AQLM / SpQR experimental path
Begin only after mixed_attn lm_eval result is in. See CLAUDE.md quantization table.

### 3. VEX feed hosting
`VexFeedManifest.generate()` is complete; no hosted feed yet.
First step: static JSON commit to `squishai/vex-feed` GitHub repo.

---

## Hard stops

- **Module count is at 106.** Any new file requires deleting one or writing justification in CHANGELOG.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing (4298 passing, 4 pre-existing wave12x failures acceptable).
- **For any REST API additions: integration tests must call the real endpoint (no mocking the handler).**
- **For quantization path changes: lm_eval result or lm_eval-waiver in commit message.**

---
