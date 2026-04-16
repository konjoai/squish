"""W76: Tenant Audit Export Bundle — unit + API integration tests.

Tests:
    TestTenantExportHelper  (8 unit tests): _db_build_tenant_export helper
    TestCloudAPITenantExport (10 tests):   GET /cloud/tenants/{id}/export
    TestCloudAPIPlatformExport (6 tests):  GET /cloud/export
"""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Minimal stubs so api.py loads without optional heavy deps
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "mlx", "mlx.core", "mlx.nn", "mlx.utils",
    "mlx_lm", "mlx_lm.utils", "mlx_lm.tuner", "mlx_lm.tuner.utils",
    "transformers", "huggingface_hub",
    "numpy",
    "cyclonedx", "cyclonedx.model", "cyclonedx.model.bom",
    "cyclonedx.model.component", "cyclonedx.output", "cyclonedx.output.json",
    "cyclonedx.schema",
    "spdx_tools", "spdx_tools.spdx", "spdx_tools.spdx.model",
    "spdx_tools.spdx.writer", "spdx_tools.spdx.writer.write_anything",
    "jose", "jose.jwt",
    "passlib", "passlib.context",
    "boto3", "botocore",
    "google", "google.cloud", "google.cloud.storage",
    "azure", "azure.storage", "azure.storage.blob",
]

for _m in _STUB_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# Ensure squish package path is importable
_SQUISH_ROOT = os.path.join(os.path.dirname(__file__), "..", "squish")
if _SQUISH_ROOT not in sys.path:
    sys.path.insert(0, _SQUISH_ROOT)

import squash.api as _api  # noqa: E402 — must come after stubs
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(_api.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_tenant(tenant_id: str = "t-w76") -> dict:
    resp = client.post("/cloud/tenant", json={"tenant_id": tenant_id, "name": "W76 Corp"})
    assert resp.status_code in (200, 201, 409)
    return _api._tenants.get(tenant_id, {})


def _clear_tenant(tenant_id: str) -> None:
    _api._tenants.pop(tenant_id, None)


# ---------------------------------------------------------------------------
# 1. Unit tests: _db_build_tenant_export
# ---------------------------------------------------------------------------

class TestTenantExportHelper(unittest.TestCase):
    """Tests for the _db_build_tenant_export() in-memory helper."""

    TENANT = "t-w76-unit"

    def setUp(self) -> None:
        _api._tenants[self.TENANT] = {"tenant_id": self.TENANT, "name": "Unit Tenant"}
        _api.SQUASH_PLAN = "community"

    def tearDown(self) -> None:
        _clear_tenant(self.TENANT)
        _api.SQUASH_PLAN = "community"

    def test_returns_dict(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIsInstance(result, dict)

    def test_tenant_id_present(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertEqual(result["tenant_id"], self.TENANT)

    def test_compliance_fields_present(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIn("compliance", result)
        self.assertIn("score", result["compliance"])
        self.assertIn("grade", result["compliance"])

    def test_conformance_fields_present(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIn("conformance", result)
        self.assertIsInstance(result["conformance"], dict)

    def test_enforcement_signal_present(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIn("enforcement_deadline", result)
        self.assertIn("days_until_enforcement", result)
        self.assertIn("enforcement_risk_level", result)

    def test_attestation_score_present(self) -> None:
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIn("attestation_score", result)

    def test_community_plan_excludes_policy_stats(self) -> None:
        """Community (summary) scope must NOT include detailed sections."""
        _api.SQUASH_PLAN = "community"
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertNotIn("policy_stats", result)
        self.assertNotIn("inventory", result)

    def test_enterprise_plan_includes_full_sections(self) -> None:
        """Enterprise (full) scope must include all extended sections."""
        _api.SQUASH_PLAN = "enterprise"
        result = _api._db_build_tenant_export(self.TENANT)
        self.assertIn("policy_stats", result)
        self.assertIn("inventory", result)
        self.assertIn("attestations", result)
        self.assertIn("drift_events", result)


# ---------------------------------------------------------------------------
# 2. API integration: GET /cloud/tenants/{tenant_id}/export
# ---------------------------------------------------------------------------

class TestCloudAPITenantExport(unittest.TestCase):
    """Integration tests for the per-tenant export endpoint."""

    TENANT = "t-w76-api"

    def setUp(self) -> None:
        _register_tenant(self.TENANT)
        _api.SQUASH_PLAN = "community"

    def tearDown(self) -> None:
        _clear_tenant(self.TENANT)
        _api.SQUASH_PLAN = "community"

    def test_unknown_tenant_returns_404(self) -> None:
        resp = client.get("/cloud/tenants/NO-SUCH-TENANT-W76/export")
        self.assertEqual(resp.status_code, 404)

    def test_known_tenant_returns_200(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        self.assertEqual(resp.status_code, 200)

    def test_response_has_tenant_id(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        self.assertEqual(resp.json()["tenant_id"], self.TENANT)

    def test_response_has_exported_at(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        self.assertIn("exported_at", resp.json())

    def test_response_has_squash_plan(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        self.assertIn("squash_plan", resp.json())

    def test_x_squash_plan_header(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        self.assertIn("x-squash-plan", resp.headers)

    def test_compliance_fields_present(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        body = resp.json()
        self.assertIn("compliance", body)

    def test_conformance_fields_present(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        body = resp.json()
        self.assertIn("conformance", body)

    def test_enforcement_signal_fields_present(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        body = resp.json()
        self.assertIn("enforcement_deadline", body)
        self.assertIn("days_until_enforcement", body)

    def test_export_scope_in_response(self) -> None:
        resp = client.get(f"/cloud/tenants/{self.TENANT}/export")
        body = resp.json()
        self.assertIn("export_scope", body)
        self.assertEqual(body["export_scope"], "summary")


# ---------------------------------------------------------------------------
# 3. API integration: GET /cloud/export
# ---------------------------------------------------------------------------

class TestCloudAPIPlatformExport(unittest.TestCase):
    """Integration tests for the platform-wide export endpoint."""

    TENANT_A = "t-w76-plat-a"
    TENANT_B = "t-w76-plat-b"

    def setUp(self) -> None:
        _api.SQUASH_PLAN = "community"

    def tearDown(self) -> None:
        _clear_tenant(self.TENANT_A)
        _clear_tenant(self.TENANT_B)
        _api.SQUASH_PLAN = "community"

    def test_empty_platform_returns_200(self) -> None:
        # Remove our test tenants to get a clean slate
        _clear_tenant(self.TENANT_A)
        _clear_tenant(self.TENANT_B)
        resp = client.get("/cloud/export")
        self.assertEqual(resp.status_code, 200)

    def test_response_has_exported_at(self) -> None:
        resp = client.get("/cloud/export")
        self.assertIn("exported_at", resp.json())

    def test_response_has_squash_plan(self) -> None:
        resp = client.get("/cloud/export")
        self.assertIn("squash_plan", resp.json())

    def test_x_squash_plan_header(self) -> None:
        resp = client.get("/cloud/export")
        self.assertIn("x-squash-plan", resp.headers)

    def test_response_has_tenant_count(self) -> None:
        _register_tenant(self.TENANT_A)
        _register_tenant(self.TENANT_B)
        resp = client.get("/cloud/export")
        body = resp.json()
        self.assertIn("tenant_count", body)
        self.assertGreaterEqual(body["tenant_count"], 2)

    def test_tenants_list_populated(self) -> None:
        _register_tenant(self.TENANT_A)
        resp = client.get("/cloud/export")
        body = resp.json()
        self.assertIsInstance(body["tenants"], list)
        tenant_ids = [t["tenant_id"] for t in body["tenants"]]
        self.assertIn(self.TENANT_A, tenant_ids)


if __name__ == "__main__":
    unittest.main()
