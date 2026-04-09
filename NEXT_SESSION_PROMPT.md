# Next Session Prompt — W56

## Context
W52-55 is complete and committed. Squash Cloud dashboard API is live:
- 10 `/cloud/*` REST endpoints (tenant CRUD, inventory, VEX alerts, drift events, policy dashboard, audit)
- HS256 JWT multi-tenant auth (`SQUASH_JWT_SECRET`) + `X-Tenant-ID` fallback
- `AttestRequest.tenant_id` auto-registration wires existing attestation into cloud dashboard
- 5 in-memory per-tenant deques with configurable env-var caps
- 5333 tests pass, 125 modules

## W56 Candidate A: Sigstore cosign lineage signing

**Goal:** `squash lineage sign <model_dir>` — sign the CycloneDX BOM with
Sigstore keyless signing (cosign). Produce `.sig` and `.cert` alongside the BOM.

**Acceptance criteria:**
1. `squash lineage sign <model_dir> [--bom BOM_PATH] [--quiet]` exits 0.
2. Writes `<bom_path>.sig` and `<bom_path>.cert`.
3. `squash lineage verify <model_dir> [--bom BOM_PATH]` exits 0/1.
4. No new Python module (inline into `squash/lineage.py`).
5. Graceful: if `cosign` not found → exit 2.
6. All tests pass, module count 125.

## W56 Candidate B: Cloud API persistence (SQLite backend)

**Goal:** Replace in-memory deques with SQLite-backed stores so the cloud
dashboard survives server restart. `SQUASH_CLOUD_DB` env var selects the path;
default `:memory:` preserves existing test behavior.

**Acceptance criteria:**
1. All 61 W52-55 tests still pass with `:memory:` backend.
2. `SQUASH_CLOUD_DB=/tmp/squash.db squish serve` survives a restart and retains data.
3. No new Python module (inline into `squish/squash/api.py` or a new `squish/squash/cloud_db.py` with written justification).

## W56 Candidate C: drift-check REST endpoint

`POST /drift-check` — REST wrapper around the existing `check_drift(bom_a, bom_b)` logic.

## Open questions
- Which candidate is highest value? B (persistence) unblocks production deployment.
  A (Sigstore) delivers supply chain signing compliance. C is the simplest.
- Dashboard Next.js repo: do we scaffold it this session or stay API-only?

## lm_eval status
- INT4 AWQ g=32 (squish): 70.8% arc_easy (Qwen2.5-1.5B, limit=500, partial — 2 tasks pending)
- INT3: 67.2% arc_easy — below 72% gate; "efficient" tier only
- gemma INT3 ≤4B: UNSAFE (−15–16pp)
- Qwen3-4B INT3: UNSAFE (−14.8pp)
