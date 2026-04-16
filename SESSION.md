# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-16

## Last commits
- **`f999f23 (w80)`** — feat(squash): W80 per-tenant EU AI Act risk profile — `_compute_model_risk_tier()`, `GET /cloud/tenants/{id}/risk-profile`, `GET /cloud/risk-overview`, `cloud-risk` CLI — 24 new tests, 4303 suite, 112 modules
- **`f812412 (w52-55)`** — feat(squash): W52-55 Squash Cloud dashboard API — 10 /cloud/* endpoints, JWT multi-tenant auth, model inventory, VEX alerts, drift events, policy dashboard, audit — 61 new tests, 5333 suite, 125 modules
- **`80492ee (w52)`** — feat(squash): W52 VEX feed subscription — api_key support + subscribe CLI + 25-statement community feed — 52 new tests, 5272 suite, 125 modules

---

## Module count
- **112** Python files in `squish/` (non-experimental). W80: no new files added (cloud-risk extends api.py + cli.py in-place).

---

## Open accuracy-validation items
- **W56 AQLM**: lm_eval on Qwen2.5-1.5B never run. `results/` has only Llama-3.2-1B results (W28–W43 era). Not blocking squash waves.
- **mixed_attn lm_eval**: Code-complete (W41), still unvalidated. Not blocking squash.

---

## Next wave: W81 — Remediation Plan Generator

**Goal:** Given a tenant's risk tier, generate a prioritised remediation plan.

**Changes:**
- `risk.py`: add `RemediationStep` dataclass + `generate_remediation_plan()`
- `api.py`: add `GET /cloud/tenants/{tenant_id}/remediation-plan`
- `cli.py`: add `cloud-remediate <tenant_id>` subcommand
- `tests/test_squash_w81.py`: ~22 tests → 4325 total, 112 modules (no new file)

**W82+ roadmap:**
- W82: enforcement action log (cloud_db.py extension + 3 endpoints + CLI)
- W83: `/cloud/dashboard` CISO aggregation endpoint
- W84: GitHub Actions integration (`github_actions.py` + `action.yml`)

