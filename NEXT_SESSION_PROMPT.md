# Wave 63 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 62 complete. ~4029 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

## W62 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.read_tenant_compliance_score(tenant_id)` — score 0–100 + grade A/B/C/D/F | ✅ |
| `_db_read_tenant_compliance_score()` helper in api.py (SQLite + in-memory fallback) | ✅ |
| `GET /cloud/tenants/{tenant_id}/compliance-score` endpoint | ✅ |
| `tests/test_squash_w62.py` — 16 tests (20 collected), all passing | ✅ |

## W61 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.read_tenant_summary(tenant_id)` — composes 4 per-tenant reads | ✅ |
| `_db_read_tenant_summary()` helper in api.py (SQLite + in-memory fallback) | ✅ |
| `GET /cloud/tenants/{tenant_id}/summary` endpoint | ✅ |
| `tests/test_squash_w61.py` — 16/16 passing | ✅ |

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

## W63 — Tenant compliance history (time-series)

**Purpose:** Add `GET /cloud/tenants/{tenant_id}/compliance-history` — a time-bucketed view of compliance scores showing how posture has changed over time. Each drift-event carries a timestamp; W63 groups them by calendar day and returns a sorted list of `{date, score, grade}` entries.

This is the natural successor to W62’s point-in-time score: W62 tells you *where you are*, W63 tells you *how you got here*.

History formula per bucket: apply the same score/grade logic to the cumulative pass/fail counts of policy events recorded **on or before** that day. This gives a "running compliance score" so a dashboard can render a trend line.

---

### Method to add in `squish/squash/cloud_db.py`

```python
def read_tenant_compliance_history(self, tenant_id: str) -> list[dict]:
    """Return day-bucketed compliance scores for *tenant_id* in ascending date order.

    Each entry: {date: str (ISO YYYY-MM-DD), score: float, grade: str}.
    Returns [] for unknown tenant or one with no drift events.
    Derived from drift_events.timestamp field using per-day grouping.
    """
```

Pattern: query `drift_events` rows for `tenant_id`, group by `date(timestamp)`, compute cumulative pass/fail per day using policy-stats data, return sorted list.

**Simplest viable approach:** Group `drift_events` by day. For each day, reuse `read_tenant_compliance_score()` snapshot approach but filtered to events on/before that day. A single-pass approach collecting daily aggregates is acceptable since history is bounded by the number of days with events.

---

### Endpoint to add in `squish/squash/api.py`

```
GET /cloud/tenants/{tenant_id}/compliance-history
```

- Requires the tenant to exist — raise `HTTPException(404)` for unknown `tenant_id`.
- Returns HTTP 200 JSON: `{tenant_id, history: [{date, score, grade}, ...]}`.
- `history` is sorted ascending by date. Empty list for new tenants with no events.
- Backed by `_db_read_tenant_compliance_history(tenant_id)` helper + in-memory fallback.

In-memory fallback: since `_drift_events` stores raw event dicts, iterate them, group by `event.get("date")` or `event.get("timestamp", "")[:10]`, compute per-day cumulative score using the same grade formula.

---

### Tests — `tests/test_squash_w63.py` (new file, 16 tests)

**`TestCloudDBTenantComplianceHistory`** (8 tests):
1. `test_returns_list` — result is a list
2. `test_empty_for_no_events` — tenant with no drift_events → []
3. `test_single_day_entry` — events on one day → list length 1
4. `test_two_days_sorted_ascending` — events on two days → sorted by date asc
5. `test_entry_has_required_keys` — each entry has date, score, grade
6. `test_score_is_float` — score field is float
7. `test_grade_is_string` — grade field is one of A/B/C/D/F
8. `test_history_scoped_to_tenant` — two tenants, independent histories

**`TestCloudAPIComplianceHistoryEndpoint`** (8 tests):
1. `test_404_for_unknown_tenant`
2. `test_200_for_known_tenant`
3. `test_response_has_tenant_id_and_history`
4. `test_history_empty_for_new_tenant`
5. `test_history_sorted_ascending` — inject two-day data, verify order
6. `test_history_entry_has_date_score_grade`
7. `test_tenant_id_echoed`
8. `test_history_type_is_list`

**Total: 16 new tests.** Suite target: **~4045 passing** after W63.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w63.py` included, 16 tests passing.
2. **Memory**: No new in-memory structures introduced.
3. **CLI**: No new CLI flags.
4. **CHANGELOG**: Wave 63 entry prepended in `CHANGELOG.md`.
5. **Module count**: ≤ 125 (no new production module, test file only).

---

## Key Files

| File | W63 Action |
|---|---|
| `squish/squash/cloud_db.py` | Add `read_tenant_compliance_history()` (day-bucketed scores from drift_events + policy_stats) |
| `squish/squash/api.py` | Add `_db_read_tenant_compliance_history()` helper + `GET /cloud/tenants/{id}/compliance-history` endpoint |
| `tests/test_squash_w63.py` | New file — 16 tests (CloudDB×8, API×8) |
| `CHANGELOG.md` | Prepend Wave 63 entry |

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

