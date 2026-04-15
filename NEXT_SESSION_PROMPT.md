# Wave 60 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 59 complete. 3977 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

## W59 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.delete_tenant(tenant_id)` — cascade DELETE tenants + all data tables | ✅ |
| `TenantUpdateRequest` Pydantic model (optional name / plan / contact_email) | ✅ |
| `_db_delete_tenant()` helper — in-memory pop × 5 stores + CloudDB cascade | ✅ |
| `PATCH /cloud/tenant/{tenant_id}` — delta-merge, 404 for unknown, updates `updated_at` | ✅ |
| `DELETE /cloud/tenant/{tenant_id}` — 204 No Content, 404 for unknown, cascade-clears all data | ✅ |
| `tests/test_squash_w59.py` — 15/15 passing (CloudDB×5, PATCH×5, DELETE×5) | ✅ |

## W58 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.read_inventory(tenant_id)` | ✅ |
| `CloudDB.read_vex_alerts(tenant_id)` | ✅ |
| `CloudDB.read_policy_stats()` (cross-tenant aggregate) | ✅ |
| `_db_read_inventory/vex_alerts/policy_stats()` helpers in api.py | ✅ |
| `GET /cloud/tenants/{id}/inventory` endpoint | ✅ |
| `GET /cloud/tenants/{id}/vex-alerts` endpoint | ✅ |
| `GET /cloud/policy-stats` endpoint | ✅ |
| `tests/test_squash_w58.py` — 16/16 passing | ✅ |
| AQLM lm_eval validation | ⚠️ PENDING (lm_eval-waiver filed) |

## W57 COMPLETE ✅

| Task | Status |
|---|---|
| `squish/cli.py` mixed_attn calibration fix (`outlier_threshold=100.0`) | ✅ |
| AQLM loader wired (`compressed_loader.py` lines 660-691, W56) | ✅ |
| `POST /drift-check` REST endpoint in `squish/squash/api.py` | ✅ |
| `squish/squash/cloud_db.py` — SQLite write-through backend | ✅ |
| All 5 api.py CloudDB write points wired | ✅ |
| `tests/test_squash_w57.py` — 20/20 passing | ✅ |
| AQLM lm_eval validation | ⚠️ PENDING (lm_eval-waiver filed) |

---

## PRE-WORK: AQLM lm_eval gate (carries forward from W58)

Still pending. Run before any AQLM-dependent work. Waiver format documented in prior waves.

---

## W60 — Tenant-scoped drift-events + policy-stats reads

**Purpose:** Complete the per-tenant read surface. Post-W59 the only missing tenant-scoped reads are `drift_events` and per-tenant `policy_stats`. The aggregate `GET /cloud/policy-stats` (W58) exists; per-tenant and drift-events do not.

| Existing | W60 adds |
|---|---|
| `GET /cloud/tenants/{id}/inventory` (W58) | `GET /cloud/tenants/{id}/drift-events` |
| `GET /cloud/tenants/{id}/vex-alerts` (W58) | `GET /cloud/tenants/{id}/policy-stats` |
| `GET /cloud/policy-stats` (W58, aggregate) | — |

---

### Methods to add in `squish/squash/cloud_db.py`

```python
def read_drift_events(self, tenant_id: str) -> list[dict]:
    """Return all drift_events rows for *tenant_id*.  Returns [] on fresh DB."""

def read_tenant_policy_stats(self, tenant_id: str) -> dict:
    """Return policy evaluation counts for *tenant_id* keyed by policy_id.
    Returns {} on fresh DB or unknown tenant."""
```

Pattern: match `read_inventory` / `read_vex_alerts` (W58). Handle fresh-DB (missing table) gracefully — return empty container, no raise.

---

### Endpoints to add in `squish/squash/api.py`

```
GET /cloud/tenants/{tenant_id}/drift-events
GET /cloud/tenants/{tenant_id}/policy-stats
```

- Both require the tenant to exist — raise `HTTPException(404)` for unknown `tenant_id`.
- Both return JSON (list and dict respectively) with HTTP 200.
- Both backed by the new CloudDB read methods + in-memory fallback pattern from W58.

Helper functions to add (pattern: `_db_read_inventory` / `_db_read_vex_alerts`):

```python
def _db_read_drift_events(tenant_id: str) -> list[dict]: ...
def _db_read_tenant_policy_stats(tenant_id: str) -> dict: ...
```

---

### Tests — `tests/test_squash_w60.py` (new file)

**`TestCloudDBDriftEvents`** (4 tests):
1. `test_read_drift_events_returns_empty_on_fresh_db` — fresh `:memory:` DB → `[]`
2. `test_read_drift_events_returns_data_after_write` — write via `POST /drift-check`, read back, assert content matches
3. `test_read_drift_events_unknown_tenant_returns_empty` — unknown tenant_id → `[]` (CloudDB level, no 404 here)
4. `test_read_drift_events_isolates_by_tenant` — two tenants, write to one, other returns `[]`

**`TestCloudDBTenantPolicyStats`** (4 tests):
1. `test_read_tenant_policy_stats_returns_empty_on_fresh_db` — fresh DB → `{}`
2. `test_read_tenant_policy_stats_returns_data_after_write` — write via `POST /cloud/policy-eval`, read back
3. `test_read_tenant_policy_stats_unknown_tenant_returns_empty` — unknown tenant → `{}`
4. `test_read_tenant_policy_stats_isolates_by_tenant` — two tenants, assert no cross-contamination

**`TestCloudAPIDriftEventsEndpoint`** (4 tests):
1. `test_returns_empty_list_for_known_tenant` — create tenant, GET drift-events → `[]`, HTTP 200
2. `test_returns_data_after_drift_check` — create tenant, POST drift-check, GET drift-events → data present
3. `test_unknown_tenant_returns_404` — no create, GET drift-events → 404
4. `test_isolates_by_tenant` — two tenants, drift-check on one, other returns `[]`

**`TestCloudAPITenantPolicyStatsEndpoint`** (4 tests):
1. `test_returns_empty_dict_for_known_tenant` — create tenant, GET policy-stats → `{}`, HTTP 200
2. `test_returns_data_after_policy_eval` — create tenant, POST policy-eval, GET policy-stats → data present
3. `test_unknown_tenant_returns_404` — no create, GET policy-stats → 404
4. `test_isolates_by_tenant` — two tenants, policy-eval on one, other returns `{}`

**Total: 16 new tests**. Suite target: **~3993 passing** after W60.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w60.py` included, 16 tests passing.
2. **Memory**: No new in-memory structures introduced — no RSS impact.
3. **CLI**: No new CLI flags. No `--help` update needed.
4. **CHANGELOG**: Wave 60 entry prepended in `CHANGELOG.md`.
5. **Module count**: `find squish -name "*.py" | grep -v __pycache__ | grep -v experimental | wc -l` ≤ 125. W60 adds no new production modules (test file only).

---

## Key Files

| File | W60 Action |
|---|---|
| `squish/squash/cloud_db.py` | Add `read_drift_events()` + `read_tenant_policy_stats()` (pattern: W58 read methods) |
| `squish/squash/api.py` | Add `_db_read_drift_events/policy_stats` helpers + 2 GET endpoints |
| `tests/test_squash_w60.py` | New file — 16 tests (CloudDB×8, API×8) |
| `CHANGELOG.md` | Prepend Wave 60 entry |

---

## lm_eval Status (last validated, 2026-03-28–2026-04-02)

| Model | Format | arc_easy | Notes |
|---|---|---|---|
| Qwen2.5-1.5B | INT4 AWQ g=32 (squish) | **70.8%** | W42 canonical baseline |
| Qwen2.5-1.5B | INT3 g=32 | 67.2% | −3.4pp; "efficient" tier; below 72% gate |
| Qwen2.5-1.5B | AQLM | ❓ PENDING | Pre-work gate, carries forward |
| Qwen2.5-1.5B | INT2 naive | ~29% | Incoherent — never ship |
| gemma-3-1b/4b | INT3 | −15–16pp | **UNSAFE** — do not recommend |
| Qwen3-4B | INT3 | −14.8pp | **UNSAFE** |
| Qwen3-8B | INT3 | −7.8pp | Coherent but large delta |

---

## Context Markers

- **squash module path:** `squish/squash/`
- **server.py ceiling:** 4743 lines — W60 routes live in `squash/api.py`, no server.py changes needed
- **SQUASH_CLOUD_DB:** default `:memory:` — all existing 3977 tests pass with in-memory behavior
- **drift_events table:** written by `POST /drift-check` (W57); no read endpoint yet
- **policy_stats table:** written by `POST /cloud/policy-eval`; aggregate read exists (W58); per-tenant read missing
- **Commit scope:** `feat(squash): W60 tenant drift-events + policy-stats reads`

