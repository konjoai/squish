# Session Update — 2026-04-22 (W100 hotfix run)

## Completed in this session

- Fixed Hugging Face URI dispatch robustness in `squish/cli.py`:
  - Added `_normalize_hf_repo_ref()` to canonicalize supported inputs to `owner/repo`.
  - Wired normalization into both `cmd_pull` and `cmd_import` Hugging Face paths.
  - Added invalid-reference guards with clear user-facing messages.
- Hardened `cmd_doctor` dependency version checks:
  - Added `_resolve_pkg_version()` with safe fallback order:
    module `__version__` → root module `__version__` → `importlib.metadata.version(...)`.
  - Prevents crashes when modules like `mlx.core` / `transformers` omit `__version__`.
- Added/updated regression coverage in `tests/test_wave89_local_model_scan.py`:
  - Assert normalized repo IDs for HF URL and `/tree/...` URL dispatch in `cmd_pull`.
  - Added `cmd_import` URL dispatch normalization test.

## Verification snapshot

- Targeted tests: passing
  - `tests/test_wave89_local_model_scan.py`
  - `tests/test_cli_extras.py::TestCmdDoctorFailing::test_failing_check_prints_some_checks_failed`
  - `tests/test_cli_unit.py::TestCmdDoctor`
  - `tests/test_cli_unit.py::TestCmdDoctorReport`
- Full suite: `4403 passed, 12 skipped`.

## Known environment caveat (tooling)

- VS Code sandboxed terminal wrapper intermittently injects malformed startup commands in this
  environment (`zsh: command not found: Studio`). Use unsandboxed shell mode for reliable runs.

---

# Wave 64 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 63 complete. ~4045 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

## W63 COMPLETE ✅

| Task | Status |
|---|---|
| `CloudDB.read_tenant_compliance_history(tenant_id)` — day-bucketed `[{date, score, grade}]` sorted ASC | ✅ |
| `_db_read_tenant_compliance_history()` helper in api.py (SQLite + in-memory fallback) | ✅ |
| `GET /cloud/tenants/{tenant_id}/compliance-history` endpoint | ✅ |
| `tests/test_squash_w63.py` — 16 tests (CloudDB×8, API×8), all passing | ✅ |

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

## W64 — Cross-tenant compliance overview

**Purpose:** Add `GET /cloud/compliance-overview` — an aggregate view across **all** registered tenants showing platform-wide compliance health. This closes the read-only reporting layer started in W58 before moving to write/mutation endpoints.

W62 answers “what is *this* tenant's current posture?”
W63 answers “how has *this* tenant's posture evolved?”
W64 answers “how is the **entire platform** doing right now?”

Response shape:
```json
{
  "total_tenants": 12,
  "compliant_tenants": 9,
  "non_compliant_tenants": 3,
  "average_score": 82.4,
  "top_at_risk": [
    {"tenant_id": "acme", "score": 41.0, "grade": "D"},
    {"tenant_id": "globex", "score": 53.5, "grade": "C"},
    {"tenant_id": "initech", "score": 61.0, "grade": "C"}
  ]
}
```

- `compliant_tenants` = count where score ≥ 80.0 (grade A or B).
- `non_compliant_tenants` = count where score < 80.0.
- `average_score` = mean of all per-tenant scores; `0.0` when no tenants exist.
- `top_at_risk` = up to 3 tenants sorted ascending by score (worst first).
- Empty platform (no tenants) → `{total_tenants: 0, compliant_tenants: 0, non_compliant_tenants: 0, average_score: 0.0, top_at_risk: []}`.

---

### Method to add in `squish/squash/cloud_db.py`

```python
def read_compliance_overview(self) -> dict:
    """Return platform-wide compliance aggregate across all tenants.

    Returns: {total_tenants, compliant_tenants, non_compliant_tenants,
              average_score, top_at_risk: [{tenant_id, score, grade}, ...]}.
    compliant = score >= 80.0 (grade A or B).
    top_at_risk = up to 3 lowest-scoring tenants, sorted ascending.
    """
```

Pattern: fetch all tenant IDs from the `tenants` table, call `read_tenant_compliance_score()` for each, aggregate. For SQLite this is a small loop (bounded by tenant count, not event count).

**Insertion point:** after `read_tenant_compliance_history()` and before `delete_tenant()`.

---

### Endpoint to add in `squish/squash/api.py`

```
GET /cloud/compliance-overview
```

- No path parameter — cross-tenant aggregate.
- Returns HTTP 200 always (empty response for no tenants).
- Backed by `_db_read_compliance_overview()` helper + in-memory fallback.

In-memory fallback: iterate `_tenants.keys()`, call `_db_read_tenant_compliance_score()` for each, aggregate counts + scores, sort by score for at_risk.

**Insertion point (helper):** after `_db_read_tenant_compliance_history()` and before `# ── Cloud auth helpers`.
**Insertion point (endpoint):** after `cloud_get_tenant_compliance_history` and before `def _result_to_dict`.

---

### Tests — `tests/test_squash_w64.py` (new file, 16 tests)

**`TestCloudDBComplianceOverview`** (8 tests):
1. `test_returns_dict` — result is a dict
2. `test_empty_platform_all_zeros` — no tenants → zeros, empty top_at_risk
3. `test_single_tenant_compliant` — freshly-upserted tenant (100.0 score) → compliant_tenants=1
4. `test_single_tenant_non_compliant` — inject policy failures → non_compliant_tenants=1
5. `test_total_count_correct` — 3 tenants → total_tenants=3
6. `test_average_score_is_float` — average_score is a float
7. `test_top_at_risk_sorted_ascending` — 3 tenants, different scores → worst first
8. `test_top_at_risk_capped_at_three` — 5 tenants → len(top_at_risk) ≤ 3

**`TestCloudAPIComplianceOverviewEndpoint`** (8 tests):
1. `test_200_response` — GET returns 200
2. `test_response_has_required_keys` — all 5 keys present
3. `test_empty_platform` — no tenants → zero counts, empty top_at_risk
4. `test_total_tenants_count` — inject 2 tenants → total_tenants=2
5. `test_compliant_count` — 2 tenants, no failures → compliant_tenants=2
6. `test_average_score_nonzero_with_tenants` — 2 tenants → average_score > 0
7. `test_top_at_risk_is_list` — top_at_risk is a list
8. `test_no_path_parameter` — endpoint accessible at `/cloud/compliance-overview`

**Total: 16 new tests.** Suite target: **~4061 passing** after W64.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w64.py` included, 16 tests passing.
2. **Memory**: No new in-memory structures introduced.
3. **CLI**: No new CLI flags.
4. **CHANGELOG**: Wave 64 entry prepended in `CHANGELOG.md`.
5. **Module count**: ≤ 125 (no new production module, test file only).

---

## Key Files

| File | W64 Action |
|---|---|
| `squish/squash/cloud_db.py` | Add `read_compliance_overview()` (aggregate loop over all tenants) |
| `squish/squash/api.py` | Add `_db_read_compliance_overview()` helper + `GET /cloud/compliance-overview` endpoint |
| `tests/test_squash_w64.py` | New file — 16 tests (CloudDB×8, API×8) |
| `CHANGELOG.md` | Prepend Wave 64 entry |

---

## Implementation Notes

**SQLite path:** `CloudDB.read_compliance_overview()` fetch all IDs:
```python
with self._lock:
    rows = self._conn.execute("SELECT tenant_id FROM tenants").fetchall()
```
Loop calling `self.read_tenant_compliance_score(tid)` for each, then aggregate.

**Compliant threshold:** `score >= 80.0`. Define as module-level constant `_COMPLIANCE_THRESHOLD = 80.0` in `cloud_db.py` if not already present.

**`test_single_tenant_non_compliant`** (CloudDB): inject rows into `policy_stats` with explicit `pass_count < total_count` via `db._conn.execute()` — `append_policy_stat` auto-derives from the payload, so direct SQL is required.

**`_rate_window.clear()`** must appear in `setup_method` for all API tests to prevent 429 bleed.

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
- **server.py ceiling:** 4743 lines — W64 routes live in `squash/api.py`, no server.py changes needed