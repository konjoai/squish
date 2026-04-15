# Wave 62 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 61 complete. ~4009 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

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

## W62 — Tenant compliance score endpoint

**Purpose:** Add `GET /cloud/tenants/{tenant_id}/compliance-score` — a computed metric that distills the tenant's per-policy pass/fail stats into a single score (0–100 float) and letter grade (A/B/C/D/F). This is the actionable "executive dashboard" number that follows naturally from the W61 summary endpoint.

Score formula: `sum(passed) / sum(passed + failed) * 100` across all policies. An empty policy_stats dict (no policy checks run yet) returns `100.0` with grade `A` — no violations recorded.

Grade thresholds: `A ≥ 90 | B ≥ 75 | C ≥ 60 | D ≥ 45 | F < 45`.

| Prior aggregates | W62 adds |
|---|---|
| `GET /cloud/tenants/{id}/summary` — counts (W61) | `GET /cloud/tenants/{id}/compliance-score` — derived score |
| `GET /cloud/tenants/{id}/policy-stats` — raw pass/fail per policy (W60) | — |

---

### Method to add in `squish/squash/cloud_db.py`

```python
def read_tenant_compliance_score(self, tenant_id: str) -> dict:
    """Return a compliance score derived from per-policy pass/fail stats.

    Keys: score (float 0-100), grade (str A/B/C/D/F),
          policy_breakdown (dict[str, {passed, failed, rate}]).
    Returns score=100.0, grade='A' for unknown tenant or no policy checks.
    """
```

Pattern: call `read_tenant_policy_stats(tenant_id)`, compute totals, derive score and grade.

Grade thresholds (inclusive lower bound):
- A: ≥ 90.0
- B: ≥ 75.0
- C: ≥ 60.0
- D: ≥ 45.0
- F: < 45.0

Each policy entry in `policy_breakdown`:
```python
{"passed": int, "failed": int, "rate": float}  # rate = passed/(passed+failed)*100
```

---

### Endpoint to add in `squish/squash/api.py`

```
GET /cloud/tenants/{tenant_id}/compliance-score
```

- Requires the tenant to exist — raise `HTTPException(404)` for unknown `tenant_id`.
- Returns HTTP 200 JSON with fields: `tenant_id`, `score`, `grade`, `policy_breakdown`.
- Backed by new `_db_read_tenant_compliance_score(tenant_id)` helper + in-memory fallback.

Helper (pattern: `_db_read_tenant_summary`):
```python
def _db_read_tenant_compliance_score(tenant_id: str) -> dict: ...
```

In-memory fallback computes the same logic from `_policy_stats[tenant_id]` directly.

---

### Tests — `tests/test_squash_w62.py` (new file)

**`TestCloudDBTenantComplianceScore`** (8 tests):
1. `test_returns_dict_with_required_keys` — result has keys: score, grade, policy_breakdown
2. `test_no_policies_returns_perfect_score` — empty/unknown tenant → score 100.0, grade 'A'
3. `test_all_passed_returns_100` — all policies 100% pass → score 100.0, grade 'A'
4. `test_all_failed_returns_0` — all policies 100% fail → score 0.0, grade 'F'
5. `test_mixed_score_computed_correctly` — 3 passed 1 failed → 75.0
6. `test_grade_thresholds` — parameterize boundary scores: 90→A, 75→B, 60→C, 45→D, 44→F
7. `test_policy_breakdown_contains_rate` — each policy entry has `rate` float
8. `test_score_scoped_to_tenant` — two tenants, different stats → different scores

**`TestCloudAPIComplianceScoreEndpoint`** (8 tests):
1. `test_404_for_unknown_tenant` — no tenant created → 404
2. `test_200_for_known_tenant` — create tenant → 200
3. `test_response_has_required_keys` — check score, grade, policy_breakdown, tenant_id
4. `test_perfect_score_no_policies` — new tenant (no policy checks) → score 100.0, grade 'A'
5. `test_score_reflects_in_memory_stats` — set _policy_stats directly, verify score
6. `test_grade_a_on_high_pass_rate` — 10 passed 0 failed → grade 'A'
7. `test_grade_f_on_low_pass_rate` — 0 passed 10 failed → grade 'F'
8. `test_tenant_id_echoed_in_response` — verify tenant_id field in response body

**Total: 16 new tests.** Suite target: **~4025 passing** after W62.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w62.py` included, 16 tests passing.
2. **Memory**: No new in-memory structures introduced — no RSS impact.
3. **CLI**: No new CLI flags. No `--help` update needed.
4. **CHANGELOG**: Wave 62 entry prepended in `CHANGELOG.md`.
5. **Module count**: `find squish -name "*.py" | grep -v __pycache__ | grep -v experimental | wc -l` ≤ 125. W62 adds no new production modules (test file only).

---

## Key Files

| File | W62 Action |
|---|---|
| `squish/squash/cloud_db.py` | Add `read_tenant_compliance_score()` (computes score+grade from policy_stats) |
| `squish/squash/api.py` | Add `_db_read_tenant_compliance_score()` helper + `GET /cloud/tenants/{id}/compliance-score` endpoint |
| `tests/test_squash_w62.py` | New file — 16 tests (CloudDB×8, API×8) |
| `CHANGELOG.md` | Prepend Wave 62 entry |

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

