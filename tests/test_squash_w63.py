"""W63 — Tenant compliance-history endpoint tests.

Covers:
  - CloudDB.read_tenant_compliance_history() (8 unit tests)
  - GET /cloud/tenants/{tenant_id}/compliance-history (8 API integration tests)
"""
from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from squish.squash.cloud_db import CloudDB
from squish.squash.api import (
    app,
    _tenants,
    _policy_stats,
    _inventory,
    _vex_alerts,
    _drift_events,
    _rate_window,
)


# ── CloudDB unit tests ────────────────────────────────────────────────────────


class TestCloudDBTenantComplianceHistory:
    def setup_method(self):
        self.db = CloudDB(path=":memory:")

    # 1. Result is a list
    def test_returns_list(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_tenant_compliance_history("t1")
        assert isinstance(result, list)

    # 2. Tenant with no drift events → []
    def test_empty_for_no_events(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_tenant_compliance_history("t1")
        assert result == []

    # 3. Events on one day → list length 1 (multiple events same day deduplicated)
    def test_single_day_entry(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.append_record("drift_events", "t1", {"model_id": "m1"})
        self.db.append_record("drift_events", "t1", {"model_id": "m2"})  # same day
        result = self.db.read_tenant_compliance_history("t1")
        assert len(result) == 1

    # 4. Events on two distinct days → 2 entries sorted ascending by date
    def test_two_days_sorted_ascending(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        ts_yesterday = time.time() - 86400
        ts_today = time.time()
        # Insert directly with controlled ts values to produce two distinct dates.
        self.db._conn.execute(
            "INSERT INTO drift_events (tenant_id, record, ts) VALUES (?, ?, ?)",
            ("t1", '{"event":"past"}', ts_yesterday),
        )
        self.db._conn.execute(
            "INSERT INTO drift_events (tenant_id, record, ts) VALUES (?, ?, ?)",
            ("t1", '{"event":"present"}', ts_today),
        )
        self.db._conn.commit()
        result = self.db.read_tenant_compliance_history("t1")
        assert len(result) == 2
        assert result[0]["date"] < result[1]["date"]

    # 5. Each entry has the required keys: date, score, grade
    def test_entry_has_required_keys(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.append_record("drift_events", "t1", {"model_id": "m1"})
        result = self.db.read_tenant_compliance_history("t1")
        entry = result[0]
        assert "date" in entry
        assert "score" in entry
        assert "grade" in entry

    # 6. score field is a float
    def test_score_is_float(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.append_record("drift_events", "t1", {"model_id": "m1"})
        result = self.db.read_tenant_compliance_history("t1")
        assert isinstance(result[0]["score"], float)

    # 7. grade field is one of the valid letter grades
    def test_grade_is_string(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.append_record("drift_events", "t1", {"model_id": "m1"})
        result = self.db.read_tenant_compliance_history("t1")
        assert result[0]["grade"] in {"A", "B", "C", "D", "F"}

    # 8. History scoped to tenant — two tenants get independent histories
    def test_history_scoped_to_tenant(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.upsert_tenant("t2", {"name": "T2"})
        self.db.append_record("drift_events", "t1", {"model_id": "m1"})
        # t2 has no drift events
        r1 = self.db.read_tenant_compliance_history("t1")
        r2 = self.db.read_tenant_compliance_history("t2")
        assert len(r1) == 1
        assert r2 == []


# ── API integration tests ─────────────────────────────────────────────────────


class TestCloudAPIComplianceHistoryEndpoint:
    def setup_method(self):
        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # prevent 429s in full-suite runs
        self.client = TestClient(app, raise_server_exceptions=True)

    # 1. 404 for unknown tenant
    def test_404_for_unknown_tenant(self):
        resp = self.client.get("/cloud/tenants/no-such-tenant/compliance-history")
        assert resp.status_code == 404

    # 2. 200 for known tenant
    def test_200_for_known_tenant(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        assert resp.status_code == 200

    # 3. Response has tenant_id and history fields
    def test_response_has_tenant_id_and_history(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        body = resp.json()
        assert "tenant_id" in body
        assert "history" in body

    # 4. New tenant with no drift events → history is an empty list
    def test_history_empty_for_new_tenant(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        body = resp.json()
        assert body["history"] == []

    # 5. History sorted ascending when two-day events are injected
    def test_history_sorted_ascending(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _drift_events["acme"].append({"timestamp": "2026-04-14T00:00:00", "model_id": "m1"})
        _drift_events["acme"].append({"timestamp": "2026-04-15T00:00:00", "model_id": "m2"})
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        body = resp.json()
        history = body["history"]
        assert len(history) == 2
        assert history[0]["date"] < history[1]["date"]

    # 6. Each history entry contains date, score, and grade
    def test_history_entry_has_date_score_grade(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _drift_events["acme"].append({"timestamp": "2026-04-15T10:00:00", "model_id": "m1"})
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        body = resp.json()
        entry = body["history"][0]
        assert "date" in entry
        assert "score" in entry
        assert "grade" in entry

    # 7. tenant_id echoed in response body
    def test_tenant_id_echoed(self):
        _tenants["widget-co"] = {"name": "Widget Co"}
        resp = self.client.get("/cloud/tenants/widget-co/compliance-history")
        body = resp.json()
        assert body["tenant_id"] == "widget-co"

    # 8. history field is a list
    def test_history_type_is_list(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-history")
        body = resp.json()
        assert isinstance(body["history"], list)
