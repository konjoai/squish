"""W74 — EU AI Act enforcement deadline signal.

Tests for:
- ``CloudDB.read_tenant_conformance()`` enforcement fields
- ``CloudDB.read_conformance_report()`` enforcement fields
- ``_db_read_tenant_conformance()`` in-memory path enforcement fields
- ``_db_read_conformance_report()`` in-memory path enforcement fields

Enforcement date: 2026-08-02
Mock date:        2026-04-15 → days=109, risk_level=MODERATE

Threshold breakdown (time-only):
  CRITICAL  <  30 days  (< 2026-07-03)
  HIGH      < 90  days  (< 2026-05-04,  but ≥ 30 days)  wait, better to say:
      days < 30  → CRITICAL
      days < 90  → HIGH (29 >= days >= 30)
      days < 180 → MODERATE (89 >= days >= 90)
      else       → LOW
"""

from __future__ import annotations

import datetime
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import squish.squash.api as _api_module
from squish.squash.api import app
from squish.squash.cloud_db import CloudDB

# ── constants ─────────────────────────────────────────────────────────────────

_SIMULATED_TODAY = datetime.date(2026, 4, 15)
_EXPECTED_DAYS = 109   # (2026-08-02 - 2026-04-15).days
_EXPECTED_RISK = "MODERATE"
_EXPECTED_DEADLINE = "2026-08-02"

# All required enforcement keys on conformance responses.
_ENF_KEYS = ("enforcement_deadline", "days_until_enforcement", "enforcement_risk_level")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_db(tmp_path) -> CloudDB:
    return CloudDB(tmp_path / "w74.db")


# ── CloudDB: per-tenant conformance enforcement fields ────────────────────────

class TestCloudDBTenantEnforcement:
    def test_enforcement_keys_present(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_tenant_conformance("t1")
        for key in _ENF_KEYS:
            assert key in result, f"missing enforcement key: {key}"

    def test_enforcement_deadline_value(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_tenant_conformance("t1")
        assert result["enforcement_deadline"] == _EXPECTED_DEADLINE
        assert isinstance(result["enforcement_deadline"], str)

    def test_days_until_enforcement_type_and_value(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_tenant_conformance("t1")
        assert isinstance(result["days_until_enforcement"], int)
        assert result["days_until_enforcement"] == _EXPECTED_DAYS

    def test_risk_level_moderate_on_sim_date(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_tenant_conformance("t1")
        assert result["enforcement_risk_level"] == _EXPECTED_RISK

    def test_risk_level_critical_at_29_days(self, tmp_path):
        """29 days until enforcement → CRITICAL."""
        db = _make_db(tmp_path)
        # 2026-08-02 - 29 days = 2026-07-04
        simulated = datetime.date(2026, 7, 4)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = simulated
            result = db.read_tenant_conformance("t1")
        assert result["days_until_enforcement"] == 29
        assert result["enforcement_risk_level"] == "CRITICAL"

    def test_risk_level_high_at_30_days_boundary(self, tmp_path):
        """30 days until enforcement → HIGH (boundary: not CRITICAL, still < 90)."""
        db = _make_db(tmp_path)
        # 2026-08-02 - 30 days = 2026-07-03
        simulated = datetime.date(2026, 7, 3)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = simulated
            result = db.read_tenant_conformance("t1")
        assert result["days_until_enforcement"] == 30
        assert result["enforcement_risk_level"] == "HIGH"

    def test_existing_fields_preserved(self, tmp_path):
        """Enforcement fields are additive; all original conformance fields intact."""
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_tenant_conformance("t1")
        for key in ("conformant", "compliance_score", "attestation_pass_rate",
                    "open_vex_alerts", "reasons"):
            assert key in result, f"original key missing: {key}"


# ── CloudDB: platform conformance report enforcement fields ───────────────────

class TestCloudDBReportEnforcement:
    def test_report_enforcement_keys_present(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_conformance_report()
        for key in _ENF_KEYS:
            assert key in result, f"missing from report: {key}"

    def test_report_enforcement_deadline(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_conformance_report()
        assert result["enforcement_deadline"] == _EXPECTED_DEADLINE

    def test_report_days_and_risk_on_sim_date(self, tmp_path):
        db = _make_db(tmp_path)
        with mock.patch("squish.squash.cloud_db.datetime") as m:
            m.date.today.return_value = _SIMULATED_TODAY
            result = db.read_conformance_report()
        assert result["days_until_enforcement"] == _EXPECTED_DAYS
        assert result["enforcement_risk_level"] == _EXPECTED_RISK


# ── API HTTP responses carry enforcement fields ───────────────────────────────

class TestAPIResponseEnforcement:
    @pytest.fixture(autouse=True)
    def _client(self):
        _api_module._rate_window.pop("testclient", None)
        self.client = TestClient(app)
        yield
        _api_module._rate_window.pop("testclient", None)

    def test_tenant_conformance_response_has_enforcement_keys(self):
        r = self.client.get("/cloud/tenants/w74-tenant/conformance")
        assert r.status_code == 200
        body = r.json()
        for key in _ENF_KEYS:
            assert key in body, f"enforcement key missing from HTTP response: {key}"

    def test_conformance_report_response_has_enforcement_keys(self):
        r = self.client.get("/cloud/conformance-report")
        assert r.status_code == 200
        body = r.json()
        for key in _ENF_KEYS:
            assert key in body, f"enforcement key missing from report HTTP response: {key}"
