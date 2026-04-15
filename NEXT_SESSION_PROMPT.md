# Wave 59 Session Prompt

**Session type:** Code session. Single wave, one commit.
**State when written:** Wave 58 complete. 3962 tests pass (0 failed, 2 skipped). 120 modules (ceiling: 125 ✅).

---

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

## PRE-WORK: AQLM lm_eval gate (gates Option A)

Run this **before** starting Option A. If hardware unavailable, use the waiver format below.

```bash
# 1. Compress (~5-10 min on M3):
squish compress --format aqlm ~/.cache/huggingface/hub/Qwen2.5-1.5B-Instruct --output /tmp/qwen2.5-1.5b-aqlm

# 2. Run lm_eval (~20 min):
python3 scripts/squish_lm_eval.py --model-dir /tmp/qwen2.5-1.5b-aqlm --tasks arc_easy --limit 200

# 3. Compare vs baseline
# INT4 AWQ baseline:  70.8% arc_easy (Qwen2.5-1.5B, W42 canonical)
# INT3 baseline:      67.2% arc_easy (−3.4pp — "efficient" tier)
# AQLM gate:          ≥ 64.6% arc_easy (< 6pp delta vs INT4)
# Naive INT2 floor:   ~27-30% (incoherent — AQLM must crush this)
```

**Gate pass**: ≥ 64.6% arc_easy → Option A proceeds. Record result in CLAUDE.md quantization table. Promote AQLM to "ultra" catalog tier.

**Gate fail**: Document result in CLAUDE.md. Keep AQLM experimental. Skip Option A this wave.

**Waiver format** (if hardware unavailable — add to commit message):
```
# lm_eval-waiver: <reason hardware not available>
# expected-delta: ~-4 to -6pp vs INT4 (extrapolated from INT3 curve)
# validation-run: queued for next session with hardware
```

---

## W58 Option A — AQLM loader wiring (conditional on lm_eval gate ≥ 64.6%)

**Purpose:** Wire `squish/loader.py` to detect `__aqlm_idx.npy` → route through `aqlm_dequantize` at load time. This unblocks `squish serve` with AQLM models.

### Files to change

**`squish/loader.py`** — AQLM detection
```python
# In the model-dir detection logic, add:
aqlm_idx = os.path.join(model_dir, "__aqlm_idx.npy")
if os.path.exists(aqlm_idx):
    from squish.quant.aqlm import aqlm_dequantize
    return aqlm_dequantize(model_dir)
```
Pattern: match existing INT3/INT4 detection blocks already in this file.

**`squish/quant/aqlm.py`** — `aqlm_dequantize` function (may already exist; verify first)
- Loads `__aqlm_idx.npy` + `__aqlm_cb.npy`
- Reconstructs weight matrices
- Returns model in a form compatible with `squish serve`

**`tests/test_squash_w58.py`** (new file — merge A+B tests here)
- `test_aqlm_loader_detects_npy_dir()` — mock model dir with `__aqlm_idx.npy`, assert loader routes correctly (unit, no real weights)
- `test_aqlm_loader_fallback_if_no_idx()` — dir without AQLM markers takes normal path

### Done-when for Option A
- `squish serve --model /tmp/qwen2.5-1.5b-aqlm` starts without error
- AQLM entry added to CLAUDE.md lm_eval table with validated arc_easy result

---

## W58 Option B — CloudDB REST read endpoints (always runs, no hardware gate)

**Purpose:** Add 3 GET endpoints backed by `cloud_db.py` SQLite reads. Completes the CRUD story for the CloudDB audit layer introduced in W57.

### Endpoints to add in `squish/squash/api.py`

| Endpoint | Returns | Fallback |
|---|---|---|
| `GET /cloud/tenants/{tenant_id}/inventory` | List of inventory records for tenant | Empty list `[]` |
| `GET /cloud/tenants/{tenant_id}/vex-alerts` | List of VEX alerts for tenant | Empty list `[]` |
| `GET /cloud/policy-stats` | Aggregate policy evaluation counts | Empty dict `{}` |

Pattern: mirror the W57 write endpoints (all 5 CloudDB write points in api.py). Use `CloudDB` instance via `get_cloud_db()` dependency.

### Methods to add in `squish/squash/cloud_db.py`

```python
def read_inventory(self, tenant_id: str) -> list[dict]: ...
def read_vex_alerts(self, tenant_id: str) -> list[dict]: ...
def read_policy_stats(self) -> dict: ...
```

- Each method queries the SQLite table written by the corresponding W57 write method.
- If table does not exist (e.g., fresh `:memory:` DB): return `[]` or `{}` respectively — no error, no raise.
- Pattern: match the W57 write helpers already in this file.

### Tests to add in `tests/test_squash_w58.py`

For each GET endpoint (3 × 3 = 9 unit tests minimum):
1. `test_{endpoint}_returns_empty_when_no_data()` — fresh DB, expect empty list/dict, HTTP 200
2. `test_{endpoint}_returns_data_after_write()` — write via W57 POST endpoint, read back via W58 GET, assert content matches
3. `test_{endpoint}_missing_tenant_returns_404()` — for tenant-scoped endpoints only

Additional:
- `test_cloud_db_read_does_not_raise_on_fresh_db()` — calls all 3 read methods on `:memory:` DB, no exception

Total W58 tests: ~15 unit + 2 AQLM (if Option A runs) = **~17 new tests**.
Suite should reach **~3963 passing** after W58.

---

## Ship Gate — Done When (all 5 required)

1. **Tests**: `python3 -m pytest tests/ --tb=no -q` → 0 failures. `tests/test_squash_w58.py` included.
2. **Memory**: `squish serve` with `SQUASH_CLOUD_DB=:memory:` — no RSS regression vs W57 baseline.
3. **CLI**: No new CLI flags added this wave (REST-only). No `--help` update needed.
4. **CHANGELOG**: Wave 58 entry prepended in `CHANGELOG.md`.
5. **Module count**: `python3 scripts/check_module_count.py` ≤ 125 before commit.
   - Currently 120. Adding `tests/test_squash_w58.py` is a test file — excluded from count.
   - No new production modules expected. If adding one: delete or justify.

**Accuracy gate (must include in commit message):**
- If lm_eval ran and AQLM passed: record result `# lm_eval: AQLM Qwen2.5-1.5B {result}% arc_easy`
- If lm_eval waiver: use waiver format above

---

## Key Files

| File | W58 Action |
|---|---|
| `squish/squash/api.py` | Add 3 GET endpoints (pattern: W57 write endpoints) |
| `squish/squash/cloud_db.py` | Add 3 read query methods (pattern: W57 helpers) |
| `squish/loader.py` | Detect `__aqlm_idx.npy` → `aqlm_dequantize` (Option A, conditional) |
| `squish/quant/aqlm.py` | Verify/add `aqlm_dequantize` function (Option A) |
| `tests/test_squash_w58.py` | New file — all W58 tests (CloudDB reads + AQLM loader) |
| `CHANGELOG.md` | Prepend Wave 58 entry |

---

## lm_eval Status (last validated, 2026-03-28–2026-04-02)

| Model | Format | arc_easy | Notes |
|---|---|---|---|
| Qwen2.5-1.5B | INT4 AWQ g=32 (squish) | **70.8%** | W42 canonical baseline |
| Qwen2.5-1.5B | INT3 g=32 | 67.2% | −3.4pp; "efficient" tier; below 72% gate |
| Qwen2.5-1.5B | AQLM | ❓ PENDING | W58 pre-work gate |
| Qwen2.5-1.5B | INT2 naive | ~29% | Incoherent — never ship |
| gemma-3-1b/4b | INT3 | −15–16pp | **UNSAFE** — do not recommend |
| Qwen3-4B | INT3 | −14.8pp | **UNSAFE** |
| Qwen3-8B | INT3 | −7.8pp | Coherent but large delta |

---

## Context Markers

- **squash module path:** `squish/squash/`
- **server.py ceiling:** 4743 lines — W58 routes live in `squash/api.py`, no server.py changes needed
- **SQUASH_CLOUD_DB:** default `:memory:` — all existing 3946 tests pass with in-memory behavior unchanged
- **AQLM npy-dir format:** `__aqlm_idx.npy` + `__aqlm_cb.npy` + passthrough tensors + `squish.json`
- **Commit scope:** `feat(squash): W58 CloudDB reads + AQLM loader` — squash scope for api.py, squash for cloud_db.py, loader scope if Option A runs
