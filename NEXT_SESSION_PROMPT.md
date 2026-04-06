# NEXT_SESSION_PROMPT.md — Wave 36+: mixed_attn lm_eval

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Wave 35 is complete (CLI help text: eu-cra, fedramp, cmmc now visible
in `squash attest --policy`, `squash attest-composed --policy`, and all four
integration-shim `--policies` help strings; 30 new tests, 4481 passing).
Next priority: Wave 36 — lm_eval validation for `mixed_attn` format.
Workflow: `squish compress --format mixed_attn <model_dir>` → `squish export` →
`squish eval` (or direct lm_eval harness). Hardware required; add lm_eval-waiver to
commit if hardware unavailable. Target: arc_easy vs INT4 baseline (70.6%, Qwen2.5-1.5B,
limit=500). One commit per wave. Minimum viable — no stubs.**

---

## Waves 1–34 complete (commit HEAD on `main`)

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
| 32    | `squish export` (npy-dir → mlx safetensors), `discover_npy_dir_metadata()`, `squish eval` redirect | ✅ |
| 33    | VEX feed hosting: `squishai/vex-feed` repo, `feed.openvex.json` seed, `DEFAULT_URL` fix, `VexCache.load_bundled()` | ✅ |
| 34    | EU CRA + FedRAMP/CMMC policy templates (`eu-cra`, `fedramp`, `cmmc` added to `_POLICIES` and `AVAILABLE_POLICIES`) | ✅ |
| 35    | CLI help text: eu-cra/fedramp/cmmc surfaced in `squash attest --policy`, `attest-composed`, and 4 integration shims | ✅ |

### Test state
- **4481 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over limit — all justified in CHANGELOG, unchanged from wave 30)
  Waves 29–34: 0 new Python modules
  Wave 33 added: squish/squash/data/community_vex_feed.openvex.json (JSON, not Python)
```

### Key files added/changed in wave 35
- `squish/squash/cli.py` — 6 policy help strings updated (attest, attest-composed, 4 shims)
- `tests/test_squash_wave35.py` — 30 new tests (5 test classes)

### VEX feed URL reference
```
VexCache.DEFAULT_URL  = https://raw.githubusercontent.com/squishai/vex-feed/main/feed.openvex.json
SQUASH_VEX_FEED_URL  = https://vex.squish.ai/ml-models/feed.openvex.json  (custom CDN)
SQUASH_VEX_FEED_FALLBACK_URL = https://raw.githubusercontent.com/squishai/vex-feed/main/feed.openvex.json
```
All three filenames now agree on `.openvex.json`.

---

## Remaining gaps (post wave 34)

### 1. CLI policy help text (Wave 35 — trivial, ~5 min)
The `squash attest --policy` choices list in `cli.py` needs to show `eu-cra`,
`fedramp`, `cmmc` in the help string and any argparse choices tuple.
Search for `eu-ai-act` in `cli.py` to find the exact location.

### 2. lm_eval validation for mixed_attn (hardware required)
`mixed_attn` (FP16 attn + INT4 MLP) is code-complete but **unvalidated**.
lm_eval result or lm_eval-waiver required before any accuracy claims.
- Run: `squish compress --format mixed_attn <model>` → `squish export <dir>` → `squish eval`
- Baseline: INT4 = **70.6% arc_easy** (Qwen2.5-1.5B, limit=500)
- Expected: within ~1pp of INT4 (FP16 attn preserves attention quality)

### 3. INT2 AQLM / SpQR experimental path
Begin only after mixed_attn lm_eval result is in. See CLAUDE.md quantization table.

---

## Hard stops

- **Module count is at 106.** Any new Python file requires deleting one or writing justification.
- **Do not add sidecar or model files to git.**
- Tests must pass before committing (4386 passing, 4 pre-existing wave12x failures acceptable).
- **For any REST API additions: integration tests must call the real endpoint.**
- **For quantization path changes: lm_eval result or lm_eval-waiver in commit message.**

---

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — implement the remaining plan gaps.

---

## Prompt

**Code session. Wave 32 is complete (`squish export` — INT4 npy-dir → mlx safetensors exporter,
`squish eval` npy-dir redirect, `discover_npy_dir_metadata()`, 29 new tests, 4355 passing).
Next priority: Wave 33 — (a) VEX feed hosting: commit a static community VEX feed JSON to
squishai/vex-feed so VexCache.DEFAULT_URL points to something real, and (b) run lm_eval on
the exported mixed_attn model now that the export path is unblocked.
One commit per wave. Minimum viable implementation — no stubs left in shipped code.**

---

## Waves 1–32 complete (commit HEAD on `main`)

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
| 32    | `squish export` (npy-dir → mlx safetensors), `discover_npy_dir_metadata()`, `squish eval` redirect | ✅ |

### Test state
- **4355 tests passing** (4 pre-existing line-count failures — wave12x, unchanged)
- 25 skipped

### Module count
```
squish/ non-experimental: 106/100 (+6 over limit — all justified in CHANGELOG, unchanged from wave 30)
  Waves 29–32: 0 new Python modules (all additions inside cli.py / api.py / compressed_loader.py)
```

### Key files added/changed in wave 32
- `squish/quant/compressed_loader.py` (extended) — new public function `discover_npy_dir_metadata()`
- `squish/cli.py` (extended) — `cmd_export()` function + `p_export` subparser + `cmd_eval` redirect
- `tests/test_squash_wave32.py` — 29 new tests (4 test classes, all passing)

### squish export format summary
- **Input**: npy-dir with `manifest.json` + `tensors/` (INT4 weights as `__q4a.npy`, `__s4a.npy`, etc.)
- **Output**: `<npy-dir>/squish_4bit/model.safetensors` + `config.json` + `.squish_4bit_ready` sentinel
- **Source model**: auto-detected by stripping `-compressed`/`-squished-*` suffix, scanning for siblings
- **Unlock**: `squish eval <npy-dir>` now redirects to `squish_4bit/` when exported

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

## Remaining gaps (post wave 32)

### 1. lm_eval validation for mixed_attn (NOW UNBLOCKED by wave 32)
`mixed_attn` (FP16 attn + INT4 MLP) is code-complete but **unvalidated**.
lm_eval result or lm_eval-waiver required before any accuracy claims.
Baseline: INT4 = **70.6% arc_easy** (Qwen2.5-1.5B, limit=500).
**Wave 32 unblocks this**: run `squish export <mixed_attn-npy-dir>` then `squish eval <npy-dir>`.
Expected result: within ~1pp of INT4 (FP16 attn, INT4 MLP should preserve quality).

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
- Tests must pass before committing (4355 passing, 4 pre-existing wave12x failures acceptable).
- **For any REST API additions: integration tests must call the real endpoint (no mocking the handler).**
- **For quantization path changes: lm_eval result or lm_eval-waiver in commit message.**

---
