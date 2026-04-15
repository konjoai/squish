"""squish/squash/cloud_db.py — SQLite-backed persistence for squash cloud dashboard.

Default behaviour (``SQUASH_CLOUD_DB`` unset or ``:memory:``) is identical to
the previous in-memory deque approach: all data lives in the server process and
is lost on restart.  Set the env var to an absolute path for a production
deployment where data must survive server restarts::

    SQUASH_CLOUD_DB=/var/lib/squash/cloud.db uvicorn squish.squash.api:app ...

Thread safety
-------------
All methods acquire a ``threading.Lock`` around every SQLite call.  The
``:memory:`` default uses ``check_same_thread=True`` (SQLite default) and is
safe for single-threaded async workers; an on-disk path uses
``check_same_thread=False`` guarded by the lock.

Stdlib only — ``json``, ``os``, ``sqlite3``, ``threading``.  No extras.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any

log = logging.getLogger(__name__)

# Env var controlling the SQLite file.  Use ":memory:" (the default) to
# preserve the existing in-memory-only behaviour.
_DB_ENV_VAR = "SQUASH_CLOUD_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    data      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    record    TEXT NOT NULL,
    ts        REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_inventory_tenant ON inventory (tenant_id, id);

CREATE TABLE IF NOT EXISTS vex_alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    record    TEXT NOT NULL,
    ts        REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_vex_tenant ON vex_alerts (tenant_id, id);

CREATE TABLE IF NOT EXISTS drift_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    record    TEXT NOT NULL,
    ts        REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_drift_tenant ON drift_events (tenant_id, id);

CREATE TABLE IF NOT EXISTS policy_stats (
    tenant_id   TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    passed      INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, policy_name)
);

CREATE TABLE IF NOT EXISTS vertex_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id           TEXT    NOT NULL,
    model_resource_name TEXT    NOT NULL,
    passed              INTEGER NOT NULL,
    labels              TEXT,
    ts                  REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_vertex_tenant ON vertex_results (tenant_id, id);

CREATE TABLE IF NOT EXISTS ado_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       TEXT    NOT NULL,
    pipeline_run_id TEXT    NOT NULL,
    passed          INTEGER NOT NULL,
    variables       TEXT,
    ts              REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_ado_tenant ON ado_results (tenant_id, id);
"""

# Valid table names — guard against SQL injection via the table parameter.
_VALID_TABLES = frozenset({"inventory", "vex_alerts", "drift_events", "vertex_results", "ado_results"})

# Compliant threshold — tenant scores >= this are counted as compliant (W64).
_COMPLIANCE_THRESHOLD = 80.0


class CloudDB:
    """Minimal SQLite wrapper for the squash cloud dashboard stores.

    Parameters
    ----------
    path:
        SQLite database path, or ``":memory:"`` for an in-process store.
    per_tenant_limit:
        Maximum rows retained per table per tenant (oldest rows pruned on insert).
    """

    def __init__(self, path: str = ":memory:", per_tenant_limit: int = 500) -> None:
        self._path = path
        self._limit = per_tenant_limit
        same_thread = (path == ":memory:")
        self._conn = sqlite3.connect(path, check_same_thread=same_thread)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        if path != ":memory:":
            log.info("squash cloud_db: SQLite persistence at %s", path)

    # ── Tenants ──────────────────────────────────────────────────────────────

    def upsert_tenant(self, tenant_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO tenants (tenant_id, data) VALUES (?, ?)",
                (tenant_id, json.dumps(data)),
            )
            self._conn.commit()

    def get_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM tenants WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def all_tenants(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id, data FROM tenants"
            ).fetchall()
        return {r["tenant_id"]: json.loads(r["data"]) for r in rows}

    # ── Generic append / query (inventory, vex_alerts, drift_events) ─────────

    def append_record(self, table: str, tenant_id: str, record: dict[str, Any]) -> None:
        if table not in _VALID_TABLES:
            raise ValueError(f"Unknown cloud_db table: {table!r}")
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {table} (tenant_id, record) VALUES (?, ?)",
                (tenant_id, json.dumps(record)),
            )
            # Prune oldest rows beyond per-tenant limit to bound table growth.
            self._conn.execute(
                f"""
                DELETE FROM {table}
                WHERE tenant_id = ?
                  AND id NOT IN (
                    SELECT id FROM {table}
                    WHERE tenant_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                  )
                """,
                (tenant_id, tenant_id, self._limit),
            )
            self._conn.commit()

    def get_records(
        self, table: str, tenant_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        if table not in _VALID_TABLES:
            raise ValueError(f"Unknown cloud_db table: {table!r}")
        cap = limit if limit is not None else self._limit
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT record FROM (
                    SELECT record, id
                    FROM {table}
                    WHERE tenant_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (tenant_id, cap),
            ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def count_records(self, table: str, tenant_id: str) -> int:
        if table not in _VALID_TABLES:
            raise ValueError(f"Unknown cloud_db table: {table!r}")
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM {table} WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return int(row["c"]) if row else 0

    # ── Policy stats ─────────────────────────────────────────────────────────

    def inc_policy_stat(self, tenant_id: str, policy_name: str, *, passed: bool) -> None:
        col = "passed" if passed else "failed"
        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO policy_stats (tenant_id, policy_name, passed, failed)
                    VALUES (?, ?, ?, ?)
                ON CONFLICT (tenant_id, policy_name) DO UPDATE SET
                    {col} = {col} + 1
                """,
                (tenant_id, policy_name, 1 if passed else 0, 0 if passed else 1),
            )
            self._conn.commit()

    def get_policy_stats(self, tenant_id: str) -> dict[str, dict[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT policy_name, passed, failed FROM policy_stats WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        return {r["policy_name"]: {"passed": r["passed"], "failed": r["failed"]} for r in rows}

    # ── W58 read helpers ─────────────────────────────────────────────────────

    def read_inventory(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all inventory records for *tenant_id* (oldest-first)."""
        return self.get_records("inventory", tenant_id)

    def read_vex_alerts(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all VEX alert records for *tenant_id* (oldest-first)."""
        return self.get_records("vex_alerts", tenant_id)

    def read_drift_events(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all drift_events rows for *tenant_id* (oldest-first)."""
        return self.get_records("drift_events", tenant_id)

    def read_tenant_policy_stats(self, tenant_id: str) -> dict[str, dict[str, int]]:
        """Return per-tenant policy evaluation counts keyed by policy_name."""
        return self.get_policy_stats(tenant_id)

    def read_policy_stats(self) -> dict[str, dict[str, int]]:
        """Return cross-tenant policy aggregates keyed by policy_name."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT policy_name, SUM(passed) AS passed, SUM(failed) AS failed "
                "FROM policy_stats GROUP BY policy_name"
            ).fetchall()
        return {
            r["policy_name"]: {"passed": int(r["passed"]), "failed": int(r["failed"])}
            for r in rows
        }

    # ── W61 read helpers ─────────────────────────────────────────────────────

    def read_tenant_summary(self, tenant_id: str) -> dict[str, Any]:
        """Return aggregated compliance stats for *tenant_id* across all data tables.

        Composes read_inventory, read_vex_alerts, read_drift_events, and
        read_tenant_policy_stats.  Returns zero-counts for an unknown or empty
        tenant — no raise.
        """
        inventory = self.read_inventory(tenant_id)
        vex_alerts = self.read_vex_alerts(tenant_id)
        drift_events = self.read_drift_events(tenant_id)
        policy_stats = self.read_tenant_policy_stats(tenant_id)
        return {
            "inventory_count": len(inventory),
            "vex_alert_count": len(vex_alerts),
            "drift_event_count": len(drift_events),
            "policy_stats": policy_stats,
        }

    # ── W62 read helpers ─────────────────────────────────────────────────────

    def read_tenant_compliance_score(self, tenant_id: str) -> dict:
        """Return a compliance score derived from per-policy pass/fail stats.

        Keys returned:
          - score (float 0–100): weighted pass rate across all policy checks.
          - grade (str): letter grade A/B/C/D/F derived from score.
          - policy_breakdown (dict[str, {passed, failed, rate}]).

        Returns score=100.0, grade='A' for an unknown tenant or one with no
        policy checks recorded — no violations recorded implies perfect posture.
        """
        stats = self.read_tenant_policy_stats(tenant_id)  # dict[str, {passed, failed}]
        if not stats:
            return {"score": 100.0, "grade": "A", "policy_breakdown": {}}

        total_passed = 0
        total_checks = 0
        policy_breakdown: dict[str, dict] = {}
        for policy_name, counts in stats.items():
            passed = counts.get("passed", 0)
            failed = counts.get("failed", 0)
            checks = passed + failed
            rate = (passed / checks * 100) if checks else 100.0
            total_passed += passed
            total_checks += checks
            policy_breakdown[policy_name] = {
                "passed": passed,
                "failed": failed,
                "rate": round(rate, 2),
            }

        score = round(total_passed / total_checks * 100, 2) if total_checks else 100.0
        grade: str
        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 45:
            grade = "D"
        else:
            grade = "F"

        return {"score": score, "grade": grade, "policy_breakdown": policy_breakdown}

    # ── W63 read helpers ──────────────────────────────────────────────────────

    def read_tenant_compliance_history(self, tenant_id: str) -> list[dict]:
        """Return day-bucketed compliance scores for *tenant_id* in ascending date order.

        Each entry: {date: str (ISO YYYY-MM-DD), score: float, grade: str}.
        Returns [] for unknown tenant or one with no drift events.
        Derives distinct calendar days from drift_events.ts (unix epoch); the
        current tenant compliance score is reported for each day (simplest
        viable approach bounded by the number of distinct event-days).
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT date(ts, 'unixepoch') AS dt
                FROM drift_events
                WHERE tenant_id = ?
                ORDER BY dt ASC
                """,
                (tenant_id,),
            ).fetchall()
        if not rows:
            return []
        score_data = self.read_tenant_compliance_score(tenant_id)
        score = score_data["score"]
        grade = score_data["grade"]
        return [{"date": row["dt"], "score": score, "grade": grade} for row in rows]

    # ── W64 read helpers ──────────────────────────────────────────────────────

    def read_compliance_overview(self) -> dict:
        """Return platform-wide compliance aggregate across all tenants.

        Returns a dict with keys:
          - total_tenants (int): count of all registered tenants.
          - compliant_tenants (int): count with score >= _COMPLIANCE_THRESHOLD.
          - non_compliant_tenants (int): count with score < _COMPLIANCE_THRESHOLD.
          - average_score (float): mean score across all tenants; 0.0 when empty.
          - top_at_risk (list[dict]): up to 3 lowest-scoring tenants, sorted
            ascending by score (worst first).  Each entry: {tenant_id, score, grade}.

        Empty platform (no tenants) returns all-zero counts and an empty list.
        """
        with self._lock:
            rows = self._conn.execute("SELECT tenant_id FROM tenants").fetchall()
        tenants = [row["tenant_id"] for row in rows]
        if not tenants:
            return {
                "total_tenants": 0,
                "compliant_tenants": 0,
                "non_compliant_tenants": 0,
                "average_score": 0.0,
                "top_at_risk": [],
            }
        scores = []
        for tid in tenants:
            data = self.read_tenant_compliance_score(tid)
            scores.append({"tenant_id": tid, "score": data["score"], "grade": data["grade"]})
        total = len(scores)
        compliant = sum(1 for s in scores if s["score"] >= _COMPLIANCE_THRESHOLD)
        average = round(sum(s["score"] for s in scores) / total, 4)
        at_risk = sorted(
            [s for s in scores if s["score"] < _COMPLIANCE_THRESHOLD],
            key=lambda x: x["score"],
        )[:3]
        return {
            "total_tenants": total,
            "compliant_tenants": compliant,
            "non_compliant_tenants": total - compliant,
            "average_score": average,
            "top_at_risk": at_risk,
        }

    # ── W65 VEX feed ───────────────────────────────────────────────────────────

    def read_vex_feed(self) -> dict:
        """Return cross-tenant VEX advisory feed across all registered tenants.

        Returns: {
            total_alerts: int,
            tenant_count: int,
            alerts: [{tenant_id, ...alert_record_fields}]
        }
        EU AI Act Art. 9 transparency: operators must retain a live feed of
        known-vulnerability advisories (VEX) across the full model inventory.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id FROM tenants"
            ).fetchall()
        tenants = [row["tenant_id"] for row in rows]
        all_alerts: list[dict[str, Any]] = []
        for tid in tenants:
            for alert in self.read_vex_alerts(tid):
                all_alerts.append({"tenant_id": tid, **alert})
        return {
            "total_alerts": len(all_alerts),
            "tenant_count": len(tenants),
            "alerts": all_alerts,
        }

    # ── W66 Vertex AI result ingest ───────────────────────────────────────────

    def append_vertex_result(
        self,
        tenant_id: str,
        model_resource_name: str,
        passed: bool,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Persist a GCP Vertex AI attestation result for *tenant_id*.

        Parameters
        ----------
        tenant_id:
            Registered tenant identifier.
        model_resource_name:
            Full Vertex AI Model resource name, e.g.
            ``"projects/my-proj/locations/us-central1/models/12345"``.
        passed:
            ``True`` when the attestation run succeeded (squash_passed=true).
        labels:
            Optional dict of GCP labels applied to the Vertex AI Model resource.
        """
        labels_json = json.dumps(labels) if labels else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO vertex_results (tenant_id, model_resource_name, passed, labels)"
                " VALUES (?, ?, ?, ?)",
                (tenant_id, model_resource_name, int(passed), labels_json),
            )
            self._conn.commit()

    def read_vertex_results(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all Vertex AI attestation results for *tenant_id*, newest first.

        Each entry contains ``{model_resource_name, passed, labels, ts}``.
        ``labels`` is a ``dict`` (or ``None`` if none were stored).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_resource_name, passed, labels, ts"
                "  FROM vertex_results"
                " WHERE tenant_id = ?"
                " ORDER BY id DESC",
                (tenant_id,),
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                "model_resource_name": row["model_resource_name"],
                "passed": bool(row["passed"]),
                "labels": json.loads(row["labels"]) if row["labels"] else None,
                "ts": row["ts"],
            })
        return results

    def append_ado_result(
        self,
        tenant_id: str,
        pipeline_run_id: str,
        passed: bool,
        variables: dict | None = None,
    ) -> None:
        """Persist an Azure DevOps pipeline attestation result for *tenant_id*.

        Parameters
        ----------
        tenant_id:
            Squash tenant identifier.
        pipeline_run_id:
            Azure DevOps pipeline run ID (string or numeric, coerced to str).
        passed:
            ``True`` when the attestation pipeline run succeeded.
        variables:
            Optional dict of pipeline variables / output variables.
        """
        variables_json = json.dumps(variables) if variables else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO ado_results (tenant_id, pipeline_run_id, passed, variables)"
                " VALUES (?, ?, ?, ?)",
                (tenant_id, str(pipeline_run_id), int(passed), variables_json),
            )
            self._conn.commit()

    def read_ado_results(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all Azure DevOps attestation results for *tenant_id*, newest first.

        Each entry contains ``{pipeline_run_id, passed, variables, ts}``.
        ``variables`` is a ``dict`` (or ``None`` if none were stored).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT pipeline_run_id, passed, variables, ts"
                "  FROM ado_results"
                " WHERE tenant_id = ?"
                " ORDER BY id DESC",
                (tenant_id,),
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                "pipeline_run_id": row["pipeline_run_id"],
                "passed": bool(row["passed"]),
                "variables": json.loads(row["variables"]) if row["variables"] else None,
                "ts": row["ts"],
            })
        return results

    def read_attestation_score(self, tenant_id: str) -> dict[str, Any]:
        """Return combined pass/fail counts across all attestation sources for *tenant_id*.

        Aggregates vertex_results (W66) and ado_results (W67).  Returns
        ``{total, passed, failed, pass_rate}`` where ``pass_rate`` is 0.0 when
        ``total == 0``.  Supports EU AI Act Art. 9 supply-chain integrity audits.
        """
        with self._lock:
            v_row = self._conn.execute(
                "SELECT COUNT(*) AS c, SUM(passed) AS ps"
                "  FROM vertex_results WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            a_row = self._conn.execute(
                "SELECT COUNT(*) AS c, SUM(passed) AS ps"
                "  FROM ado_results WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        total = int(v_row["c"] or 0) + int(a_row["c"] or 0)
        passed = int(v_row["ps"] or 0) + int(a_row["ps"] or 0)
        failed = total - passed
        pass_rate = round(passed / total, 4) if total > 0 else 0.0
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": pass_rate}

    def read_attestations(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return merged chronological attestation history for *tenant_id*, newest first.

        Combines vertex_results (W66) and ado_results (W67), sorted by ``ts`` DESC.
        Each item includes a ``source`` field (``"vertex"`` or ``"ado"``) plus the
        source-specific fields (``model_resource_name``/``labels`` or
        ``pipeline_run_id``/``variables``).

        Supports EU AI Act Art. 12 (technical documentation) and Art. 18 (record-keeping)
        obligations by providing a complete, auditable attestation trail.
        """
        with self._lock:
            v_rows = self._conn.execute(
                "SELECT passed, ts, model_resource_name, labels"
                "  FROM vertex_results WHERE tenant_id = ? ORDER BY ts DESC",
                (tenant_id,),
            ).fetchall()
            a_rows = self._conn.execute(
                "SELECT passed, ts, pipeline_run_id, variables"
                "  FROM ado_results WHERE tenant_id = ? ORDER BY ts DESC",
                (tenant_id,),
            ).fetchall()
        merged: list[dict[str, Any]] = []
        for r in v_rows:
            merged.append({
                "source": "vertex",
                "passed": bool(r["passed"]),
                "ts": r["ts"],
                "model_resource_name": r["model_resource_name"],
                "labels": json.loads(r["labels"]) if r["labels"] else None,
            })
        for r in a_rows:
            merged.append({
                "source": "ado",
                "passed": bool(r["passed"]),
                "ts": r["ts"],
                "pipeline_run_id": r["pipeline_run_id"],
                "variables": json.loads(r["variables"]) if r["variables"] else None,
            })
        merged.sort(key=lambda x: x["ts"], reverse=True)
        return merged

    def read_attestation_overview(self) -> dict[str, Any]:
        """Return cross-tenant attestation health overview for the whole platform.

        Iterates all registered tenants, aggregates attestation scores, and returns:

        - ``total_attestations`` (int): sum of all vertex + ADO records.
        - ``tenants_covered`` (int): number of registered tenants.
        - ``platform_pass_rate`` (float): passed / total across all tenants; 0.0
          when no attestations exist.
        - ``tenants_with_failures`` (list[dict]): tenants with at least one failed
          attestation, each entry ``{tenant_id, failed, pass_rate}``.

        Always returns HTTP 200; empty platform returns zero counts and an empty list.
        Supports EU AI Act Art. 9 / Art. 17 portfolio-level supply-chain risk monitoring.
        """
        with self._lock:
            rows = self._conn.execute("SELECT tenant_id FROM tenants").fetchall()
        tenants = [row["tenant_id"] for row in rows]
        if not tenants:
            return {
                "total_attestations": 0,
                "tenants_covered": 0,
                "platform_pass_rate": 0.0,
                "tenants_with_failures": [],
            }
        total_attestations = 0
        total_passed = 0
        tenants_with_failures: list[dict[str, Any]] = []
        for tid in tenants:
            score = self.read_attestation_score(tid)
            total_attestations += score["total"]
            total_passed += score["passed"]
            if score["failed"] > 0:
                tenants_with_failures.append({
                    "tenant_id": tid,
                    "failed": score["failed"],
                    "pass_rate": score["pass_rate"],
                })
        platform_pass_rate = (
            round(total_passed / total_attestations, 4) if total_attestations > 0 else 0.0
        )
        return {
            "total_attestations": total_attestations,
            "tenants_covered": len(tenants),
            "platform_pass_rate": platform_pass_rate,
            "tenants_with_failures": tenants_with_failures,
        }

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant and all associated records (cascade).

        Safe to call for a non-existent *tenant_id* — does nothing if the tenant
        is not present.
        """
        with self._lock:
            self._conn.execute("DELETE FROM tenants WHERE tenant_id = ?", (tenant_id,))
            for table in (*_VALID_TABLES, "policy_stats"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE tenant_id = ?",  # noqa: S608
                    (tenant_id,),
                )
            self._conn.commit()


# ── Module-level singleton ────────────────────────────────────────────────────

def _make_db() -> CloudDB | None:
    """Create a CloudDB from ``SQUASH_CLOUD_DB`` if it is a non-`:memory:` path.

    Returns ``None`` when the env var is absent or ``:memory:`` so the caller
    can skip write-through and rely solely on the in-memory deques (default).
    """
    path = os.getenv(_DB_ENV_VAR, ":memory:")
    if not path or path == ":memory:":
        return None
    return CloudDB(path=path)
