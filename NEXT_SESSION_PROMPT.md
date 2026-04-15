# Wave 61 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 60 complete. ~3993 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

## W60 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.read_drift_events(tenant_id)` | ✅ |
| `CloudDB.read_tenant_policy_stats(tenant_id)` | ✅ |
| `_db_read_drift_events/policy_stats()` helpers in api.py | ✅ |
| `GET /cloud/tenants/{id}/drift-events` endpoint | ✅ |
| `GET /cloud/tenants/{id}/policy-stats` endpoint | ✅ |
| `tests/test_squash_w60.py` — 16/16 passing | ✅ |
| Fix: `_C` NameError in server.py (hoisted import) | ✅ |
| Fix: server.py line count gate (4743 ≤ ceiling) | ✅ |

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

## W61 — Tenant summary endpoint

**Purpose:** Add `GET /cloud/tenants/{tenant_id}/summary` — a single aggregate read that collects inventory count, vex-alert count, drift-event count, and policy pass/fail stats for a tenant in one call. This is the "boardroom at a glance" endpoint: clients no longer need to make four separate round-trips for a full compliance posture view.

Per-tenant reads are now complete through W60. W61 closes the read surface by providing a convenience aggregate on top of the existing four CloudDB read methods.

| Existing per-tenant reads | W61 adds |
|---|---|
| `GET /cloud/tenants/{id}/inventory` (W58) | `GET /cloud/tenants/{id}/summary` |
| `GET /cloud/tenants/{id}/vex-alerts` (W58) | — |
| `GET /cloud/tenants/{id}/drift-events` (W60) | — |
| `GET /cloud/tenants/{id}/policy-stats` (W60) | — |

---

### Method to add in `squish/squash/cloud_db.py`

```python
def read_tenant_summary(self, tenant_id: str) -> dict:
    """Return aggregated stats for *tenant_id* across all data tables.

    Keys: inventory_count, vex_alert_count, drift_event_count, policy_stats.
    Returns zero-counts for an unknown or empty tenant — no raise.
    Composes the four existing per-tenant read methods.
    """
```

Pattern: call `read_inventory`, `read_vex_alerts`, `read_drift_events`, `read_tenant_policy_stats` and
aggregate. Handle fresh-DB / unknown tenant gracefully — return `{"inventory_count": 0, "vex_alert_count": 0, "drift_event_count": 0, "policy_stats": {}}`.

---

### Endpoint to add in `squish/squash/api.py`

```
GET /cloud/tenants/{tenant_id}/summary
```

- Requires the tenant to exist — raise `HTTPException(404)` for unknown `tenant_id`.
- Returns HTTP 200 JSON with fields: `tenant_id`, `tenant` (full tenant record), `inventory_count`, `vex_alert_count`, `drift_event_count`, `policy_stats`.
- Backed by new `_db_read_tenant_summary(tenant_id)` helper + in-memory fallback pattern from W58/W60.

Helper function to add (pattern: `_db_read_drift_events` / `_db_read_tenant_policy_stats`):

```python
def _db_read_tenant_summary(tenant_id: str) -> dict: ...
```

In-memory fallback (when `_db is None`):
```python
{
    "inventory_count": len(_inventory[tenant_id]),
    "vex_alert_count": len(_vex_alerts[tenant_id]),
    "drift_event_count": len(_drift_events[tenant_id]),
    "policy_stats": dict(_policy_stats[tenant_id]),
}
```

---

### Tests — `tests/test_squash_w61.py` (new file)

**`TestCloudDBTenantSummary`** (8 tests):
1. `test_summary_returns_zero_counts_on_fresh_db` — fresh `:memory:` DB → all counts 0, policy_stats `{}`
2. `test_summary_inventory_count_after_write` — append 2 inventory records, inventory_count=2
3. `test_summary_vex_alert_count_after_write` — append 3 vex alerts, vex_alert_count=3
4. `test_summary_drift_event_count_after_write` — append 2 drift events, drift_event_count=2
5. `test_summary_policy_stats_included` — write policy stats, verify non-empty dict in summary
6. `test_summary_isolates_by_tenant` — two tenants, writes to tenant A only; tenant B still returns zeros
7. `test_summary_has_required_keys` — assert summary dict has all four required keys
8. `test_summary_unknown_tenant_returns_zeros` — unknown tenant_id → zeros, no raise (DB-layer only)

**`TestCloudAPITenantSummaryEndpoint`** (8 tests):
1. `test_returns_200_for_known_tenant` — create tenant, GET summary → HTTP 200
2. `test_response_contains_tenant_record` — verify `tenant.id` present in response `tenant` field
3. `test_all_counts_zero_on_new_tenant` — new tenant (no data) → all counts 0
4. `test_inventory_count_after_register` — POST `/cloud/inventory/register`, GET summary → inventory_count=1
5. `test_vex_alert_count_after_post` — POST `/cloud/vex/alert`, GET summary → vex_alert_count=1
6. `test_drift_event_count_after_post` — POST `/cloud/drift/event`, GET summary → drift_event_count=1
7. `test_unknown_tenant_returns_404` — no tenant created, GET summary → 404
8. `test_isolates_by_tenant` — two tenants, data only on tenant A; tenant B counts remain 0

**Total: 16 new tests.** Suite target: **~4009 passing** after W61.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w61.py` included, 16 tests passing.
2. **Memory**: No new in-memory structures introduced — no RSS impact.
3. **CLI**: No new CLI flags. No `--help` update needed.
4. **CHANGELOG**: Wave 61 entry prepended in `CHANGELOG.md`.
5. **Module count**: `find squish -name "*.py" | grep -v __pycache__ | grep -v experimental | wc -l` ≤ 125. W61 adds no new production modules (test file only).

---

## Key Files

| File | W61 Action |
|---|---|
| `squish/squash/cloud_db.py` | Add `read_tenant_summary()` (composes W58+W60 read methods) |
| `squish/squash/api.py` | Add `_db_read_tenant_summary()` helper + `GET /cloud/tenants/{id}/summary` endpoint |
| `tests/test_squash_w61.py` | New file — 16 tests (CloudDB×8, API×8) |
| `CHANGELOG.md` | Prepend Wave 61 entry |

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
- **server.py ceiling:** 4743 lines — W61 routes live in `squash/api.py`, no server.py changes needed
- **SQUASH_CLOUD_DB:** default `:memory:` — all existing ~3993 tests pass with in-memory behavior
- **CloudDB current method count:** 16 methods across 5 data types; W61 adds `read_tenant_summary()`
- **Per-tenant endpoint surface post-W60:** inventory, vex-alerts, drift-events, policy-stats; W61 adds summary
- **`_rate_window`:** import + `.clear()` required in every API test class fixture to avoid 429s in full-suite runs
- **Commit scope:** `feat(squash): W61 tenant summary endpoint`

