"""W65 — Hosted VEX feed: cross-tenant advisory aggregation.

Tests for CloudDB.read_vex_feed() and GET /cloud/vex-feed.
16 tests total: TestCloudDBVexFeed (8) + TestCloudAPIVexFeedEndpoint (8).
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest
from starlette.testclient import TestClient

from squish.squash.cloud_db import CloudDB


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_db() -> CloudDB:
    """Return a fresh in-memory CloudDB."""
    db = CloudDB(path=":memory:")
    return db


def _register(db: CloudDB, tenant_id: str) -> None:
    db.upsert_tenant(tenant_id, {"name": tenant_id, "plan": "pro"})


def _add_vex(db: CloudDB, tenant_id: str, record: dict[str, Any]) -> None:
    db.append_record("vex_alerts", tenant_id, record)


# ── CloudDB unit tests ─────────────────────────────────────────────────────────

class TestCloudDBVexFeed:
    """8 unit tests for CloudDB.read_vex_feed()."""

    def test_returns_dict(self) -> None:
        db = _make_db()
        result = db.read_vex_feed()
        assert isinstance(result, dict)

    def test_empty_platform_no_alerts(self) -> None:
        db = _make_db()
        result = db.read_vex_feed()
        assert result["total_alerts"] == 0
        assert result["tenant_count"] == 0
        assert result["alerts"] == []

    def test_single_tenant_no_alerts(self) -> None:
        db = _make_db()
        _register(db, "alpha")
        result = db.read_vex_feed()
        assert result["tenant_count"] == 1
        assert result["total_alerts"] == 0

    def test_single_tenant_with_alerts(self) -> None:
        db = _make_db()
        _register(db, "alpha")
        _add_vex(db, "alpha", {"cve": "CVE-2024-0001", "severity": "HIGH"})
        _add_vex(db, "alpha", {"cve": "CVE-2024-0002", "severity": "MEDIUM"})
        result = db.read_vex_feed()
        assert result["total_alerts"] == 2
        assert result["tenant_count"] == 1

    def test_multi_tenant_aggregates_all_alerts(self) -> None:
        db = _make_db()
        for tid in ("alpha", "beta", "gamma"):
            _register(db, tid)
            _add_vex(db, tid, {"cve": f"CVE-{tid}", "severity": "LOW"})
        result = db.read_vex_feed()
        assert result["total_alerts"] == 3
        assert result["tenant_count"] == 3

    def test_alerts_contain_tenant_id(self) -> None:
        db = _make_db()
        _register(db, "alpha")
        _add_vex(db, "alpha", {"cve": "CVE-2024-0001"})
        result = db.read_vex_feed()
        assert result["alerts"][0]["tenant_id"] == "alpha"

    def test_alert_preserves_original_fields(self) -> None:
        db = _make_db()
        _register(db, "alpha")
        _add_vex(db, "alpha", {"cve": "CVE-2024-9999", "severity": "CRITICAL"})
        alert = db.read_vex_feed()["alerts"][0]
        assert alert["cve"] == "CVE-2024-9999"
        assert alert["severity"] == "CRITICAL"

    def test_required_keys_present(self) -> None:
        db = _make_db()
        result = db.read_vex_feed()
        assert set(result.keys()) == {"total_alerts", "tenant_count", "alerts"}


# ── API endpoint tests ─────────────────────────────────────────────────────────

class TestCloudAPIVexFeedEndpoint:
    """8 integration tests for GET /cloud/vex-feed."""

    def setup_method(self) -> None:
        from squish.squash import api

        api._tenants.clear()
        api._inventory.clear()
        api._vex_alerts.clear()
        api._drift_events.clear()
        api._policy_stats.clear()
        api._rate_window.clear()
        api._db = None
        self.client = TestClient(api.app, raise_server_exceptions=True)

    def test_200_response(self) -> None:
        resp = self.client.get("/cloud/vex-feed")
        assert resp.status_code == 200

    def test_response_has_required_keys(self) -> None:
        resp = self.client.get("/cloud/vex-feed")
        body = resp.json()
        assert "total_alerts" in body
        assert "tenant_count" in body
        assert "alerts" in body

    def test_empty_platform_zeros(self) -> None:
        resp = self.client.get("/cloud/vex-feed")
        body = resp.json()
        assert body["total_alerts"] == 0
        assert body["tenant_count"] == 0
        assert body["alerts"] == []

    def _register_tenant(self, tenant_id: str) -> None:
        from squish.squash import api

        api._tenants[tenant_id] = {
            "tenant_id": tenant_id,
            "name": tenant_id,
            "plan": "pro",
            "contact_email": f"{tenant_id}@example.com",
        }

    def test_tenant_count_with_tenants(self) -> None:
        self._register_tenant("acme")
        self._register_tenant("globex")
        resp = self.client.get("/cloud/vex-feed")
        body = resp.json()
        assert body["tenant_count"] == 2

    def test_total_alerts_zero_when_no_vex_posted(self) -> None:
        self._register_tenant("acme")
        resp = self.client.get("/cloud/vex-feed")
        body = resp.json()
        assert body["total_alerts"] == 0

    def test_total_alerts_after_drift_check(self) -> None:
        from squish.squash import api

        self._register_tenant("acme")
        api._vex_alerts["acme"].append({"cve": "CVE-2024-001", "severity": "HIGH"})
        api._vex_alerts["acme"].append({"cve": "CVE-2024-002", "severity": "LOW"})
        resp = self.client.get("/cloud/vex-feed")
        body = resp.json()
        assert body["total_alerts"] == 2

    def test_alerts_include_tenant_id(self) -> None:
        from squish.squash import api

        self._register_tenant("acme")
        api._vex_alerts["acme"].append({"cve": "CVE-2024-001"})
        resp = self.client.get("/cloud/vex-feed")
        alerts = resp.json()["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["tenant_id"] == "acme"

    def test_no_path_parameter(self) -> None:
        # Endpoint must be accessible at exactly /cloud/vex-feed (no path param).
        resp = self.client.get("/cloud/vex-feed")
        assert resp.status_code == 200
