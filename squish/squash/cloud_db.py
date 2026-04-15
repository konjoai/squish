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
"""

# Valid table names — guard against SQL injection via the table parameter.
_VALID_TABLES = frozenset({"inventory", "vex_alerts", "drift_events"})


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
