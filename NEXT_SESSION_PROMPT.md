# NEXT_SESSION_PROMPT.md — Squash Wave 29+: VEX Feed Static Hosting + Remaining Gaps

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Implement Wave 29: VEX feed static JSON commit + any outstanding gaps.
One commit per wave. Minimum viable implementation — no stubs left in shipped code.**

---

## Waves 1–28 complete (commit HEAD on `main`)

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
| 28    | CircleCI Orb (`orb.yml` with `squash/attest`, `squash/check`, `squash/policy-gate`) + Ray Serve (`squash_serve` decorator, `SquashServeDeployment`, `SquashServeConfig`) | ✅ |

### Test state
- **4184 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over nominal limit — all justified in CHANGELOG)
  - slsa.py, risk.py, cicd.py: +3 (waves 20-25)
  - integrations/sagemaker.py: +1 (wave 26 — MLOps integration suite)
  - integrations/kubernetes.py: +1 (wave 27 — K8s enforcement plane)
  - integrations/ray.py: +1 (wave 28 — Ray Serve deployment lifecycle)
```

### Key files added in wave 28
- `squish/squash/integrations/circleci/orb.yml` — CircleCI Orb YAML (data file; no module count impact)
- `squish/squash/integrations/ray.py` — `squash_serve` decorator, `SquashServeDeployment`, `SquashServeConfig`
- `squish/squash/__init__.py` (extended) — `squash_serve`, `SquashServeConfig`, `SquashServeDeployment` exported
- `tests/test_squash_wave28.py` — 68 new tests

---

## Remaining gaps (post wave 28)

### 1. VEX feed static hosting (Wave 29)
**Status: Infrastructure only (client + manifest generator in Wave 26).**

`VexFeedManifest.generate()` and `VexCache.fetch_squash_feed()` are complete,
but no hosted feed exists yet.
- First step: a static JSON file committed to `squishai/vex-feed` GitHub repo
  (separate repo action, not squish/ itself).
- Second step: a `squash vex-publish` CLI subcommand that invokes
  `VexFeedManifest.generate()` and writes the output to a configurable path.

### 2. CLI completeness audit (Wave 29 extension)
Verify every public integration has a corresponding CLI subcommand or
`squash ci-run` flag:
- `mlflow` / `wandb` / `huggingface` / `langchain` — do they have pipe-friendly
  CLI access? If not, add `squash attest-mlflow`, etc.

### 3. lm-eval-validated quantization results (ongoing)
mixed_attn (FP16 attn + INT4 MLP) is code-complete but unvalidated.
Requires running `lm_eval` on M3 hardware before merging accuracy claims.

---

## Hard stops

- **Module count is at 106.** Any new file requires deleting one or writing justification in CHANGELOG.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing.

---

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

### Test state
- **4028+ tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 104/100 (+4 over nominal limit — justified)
  - slsa.py, risk.py, cicd.py: +3 (waves 20-25, written in CHANGELOG)
  - integrations/sagemaker.py: +1 (wave 26, completes MLOps integration suite)
```

### Key files added in wave 26
- `squish/squash/integrations/sagemaker.py` — `SageMakerSquash.attach_attestation()`, `SageMakerSquash.tag_model_package()`
- `squish/squash/sbom_builder.py` (extended) — `OrasAdapter.push()` + `OrasAdapter.build_manifest()`
- `squish/squash/vex.py` (extended) — `VexFeedManifest.generate()`, `VexFeedManifest.validate()`, `VexCache.fetch_squash_feed()`
- `squish/squash/__init__.py` (extended) — new exports

---

## Remaining gaps (post wave 26)

### 1. Kubernetes Admission Webhook (Wave 27)
**Status: Not started.**

A Mutating/Validating Webhook controller that verifies Squash signature before pod admission.
This is the hardest enforcement point — operates at cluster admission, not CI/CD pipeline.

Deliverables:
- `squish/squash/integrations/kubernetes.py` — `KubernetesWebhookHandler.handle(request)` (FastAPI endpoint)
- `squish/squash/integrations/kubernetes_helm/` — Helm chart with `MutatingWebhookConfiguration`
- CLI: `squash webhook serve [--port 8443] [--tls-cert TLS_CERT] [--tls-key TLS_KEY]`

Acceptance:
- `handle()` returns 200 `{allowed: true}` if Squash signature is valid, 403 `{allowed: false, status: {...}}` if not
- Integration test: fake admission review JSON → handler → allowed/denied response

### 2. CircleCI Orb (Wave 28)
**Status: Partially delivered (CircleCI detection in `cicd.py`).**

Need a formal CircleCI Orb YAML (`integrations/circleci/orb.yml`).

### 3. VEX feed hosting
**Status: Infrastructure only (client + manifest generator).**

`VexFeedManifest.generate()` and `VexCache.fetch_squash_feed()` are complete, but no hosted feed exists yet.
First step: a static JSON file commit to `squishai/vex-feed` GitHub repo (separate repo action).

### 4. Ray Serve decorator
**Status: Not started.** Lower priority.

---

## Hard stops

- **Module count is at 104.** Any new file requires deleting one or writing justification in CHANGELOG.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing.

---
