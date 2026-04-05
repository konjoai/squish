# NEXT_SESSION_PROMPT.md — Squash Wave 30+: REST API Integration Endpoints + Accuracy Gate

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Implement Wave 30: REST API endpoints for the Wave 29 CLI additions
(`POST /vex/publish`, `POST /attest/mlflow`, etc.) + any outstanding gaps.
One commit per wave. Minimum viable implementation — no stubs left in shipped code.**

---

## Waves 1–29 complete (commit HEAD on `main`)

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
| 29    | VEX publish CLI (`squash vex-publish`) + integration CLI shims (`attest-mlflow`, `attest-wandb`, `attest-huggingface`, `attest-langchain`) | ✅ |

### Test state
- **4242 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over nominal limit — all justified in CHANGELOG)
  - slsa.py, risk.py, cicd.py: +3 (waves 20-25)
  - integrations/sagemaker.py: +1 (wave 26 — MLOps integration suite)
  - integrations/kubernetes.py: +1 (wave 27 — K8s enforcement plane)
  - integrations/ray.py: +1 (wave 28 — Ray Serve deployment lifecycle)
  Wave 29: 0 new modules (all additions inside cli.py)
```

### Key files added/changed in wave 29
- `squish/squash/cli.py` (extended) — 5 new subcommands + handler functions:
  - `vex-publish`: `_cmd_vex_publish()` — VexFeedManifest.generate() → JSON file
  - `attest-mlflow`: `_cmd_attest_mlflow()` — offline AttestPipeline shim, JSON to stdout
  - `attest-wandb`: `_cmd_attest_wandb()` — offline AttestPipeline shim, JSON to stdout
  - `attest-huggingface`: `_cmd_attest_huggingface()` — AttestPipeline + optional HFSquash push
  - `attest-langchain`: `_cmd_attest_langchain()` — one-shot pre-deployment attestation
- `tests/test_squash_wave29.py` — 58 new tests

---

## Remaining gaps (post wave 29)

### 1. REST API endpoints for Wave 29 CLI additions (Wave 30 priority)
The CLI completeness audit (Wave 29) added 5 new subcommands. Each should have
a corresponding REST endpoint in `squish/squash/server.py`:
- `POST /vex/publish` — body: `{entries, author?, doc_id?}` → returns OpenVEX doc JSON
- `POST /attest/mlflow` — body: `{model_path, policies?, sign?, fail_on_violation?}` → AttestResult
- `POST /attest/wandb` — same interface as mlflow
- `POST /attest/huggingface` — body: `{model_path, repo_id?, hf_token?, policies?}` → AttestResult
- `POST /attest/langchain` — same as mlflow/wandb

### 2. lm-eval-validated quantization results (ongoing — squish inference)
`mixed_attn` (FP16 attn + INT4 MLP) is code-complete but unvalidated.
Requires running `lm_eval` on M3 hardware before merging accuracy claims.

---

## Hard stops

- **Module count is at 106.** Any new file requires deleting one or writing justification in CHANGELOG.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing.
- **For any REST API additions: integration tests must call the real endpoint (no mocking the handler).**



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
