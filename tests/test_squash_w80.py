"""W80: Per-Tenant Risk Profile API + CLI — unit tests.

Tests:
    TestComputeModelRiskTier   (6 tests): pure-function tier logic
    TestCloudRiskProfileApi    (6 tests): GET /cloud/tenants/{id}/risk-profile
    TestCloudRiskOverviewApi   (3 tests): GET /cloud/risk-overview
    TestCloudRiskCmd           (6 tests): CLI cloud-risk happy/sad paths
    TestCloudRiskJsonFlag      (3 tests): --json output validation
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

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

_SQUISH_ROOT = os.path.join(os.path.dirname(__file__), "..", "squish")
if _SQUISH_ROOT not in sys.path:
    sys.path.insert(0, _SQUISH_ROOT)

import squash.api as _api  # noqa: E402

_squish_pkg = sys.modules.setdefault("squish", types.ModuleType("squish"))
_squish_squash_pkg = sys.modules.setdefault("squish.squash", types.ModuleType("squish.squash"))

from squash.cli import _cmd_cloud_risk  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TENANT_RISK = "t-w80-risk"
_TENANT_OVERVIEW = "t-w80-overview"
_TENANT_OVERVIEW_B = "t-w80-overview-b"


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "tenant_id": _TENANT_RISK,
        "overview": False,
        "output_json": False,
        "quiet": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


def _register_tenant(tid: str) -> None:
    _api._tenants[tid] = {"tenant_id": tid, "name": "W80 Test Corp"}


def _clear_tenant(*tids: str) -> None:
    for tid in tids:
        _api._tenants.pop(tid, None)
        _api._policy_stats.pop(tid, None)
        _api._vertex_results.pop(tid, None)
        _api._vex_alerts.pop(tid, None)
        _api._inventory.pop(tid, None)


def _seed_inventory(tid: str, attestation_passed: bool = True,
                    policy_passed: bool = True, error_count: int = 0,
                    model_id: str = "test-model") -> dict:
    record = {
        "model_id": model_id,
        "model_path": f"/models/{model_id}",
        "bom_path": "",
        "attestation_passed": attestation_passed,
        "policy_results": {
            "enterprise-strict": {
                "passed": policy_passed,
                "error_count": error_count,
                "warning_count": 0,
            }
        },
        "vex_cves": [],
        "timestamp": "2026-04-16T00:00:00Z",
        "record_id": "rec-w80",
    }
    _api._inventory[tid].append(record)
    return record


def _seed_vex(tid: str, count: int = 2) -> None:
    for i in range(count):
        _api._vex_alerts[tid].append({
            "alert_id": f"va-w80-{i}",
            "cve_id": f"CVE-2026-8{i:03d}",
            "severity": "high",
            "status": "open",
            "model_id": "test-model",
            "tenant_id": tid,
            "timestamp": "2026-04-16T00:00:00Z",
        })


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------

_orig_squash_api = None


def setUpModule() -> None:  # noqa: N802
    global _orig_squash_api
    _orig_squash_api = sys.modules.get("squish.squash.api")
    sys.modules["squish.squash.api"] = _api
    _squish_squash_pkg.api = _api  # type: ignore[attr-defined]


def tearDownModule() -> None:  # noqa: N802
    if _orig_squash_api is not None:
        sys.modules["squish.squash.api"] = _orig_squash_api
        _squish_squash_pkg.api = _orig_squash_api  # type: ignore[attr-defined]
    else:
        sys.modules.pop("squish.squash.api", None)


# ---------------------------------------------------------------------------
# 1. _compute_model_risk_tier — pure-function tests
# ---------------------------------------------------------------------------

class TestComputeModelRiskTier(unittest.TestCase):
    """Unit tests for the W80 risk tier pure function."""

    def _rec(self, attestation_passed: bool = True, passed: bool = True,
             error_count: int = 0) -> dict:
        return {
            "attestation_passed": attestation_passed,
            "policy_results": {
                "p1": {"passed": passed, "error_count": error_count, "warning_count": 0},
            },
        }

    def test_minimal_attested_no_vex(self):
        """Attested + no VEX → MINIMAL."""
        tier = _api._compute_model_risk_tier(self._rec(True, True, 0), open_vex=0)
        self.assertEqual(tier, "MINIMAL")

    def test_limited_attested_with_vex(self):
        """Attested + open VEX → LIMITED."""
        tier = _api._compute_model_risk_tier(self._rec(True, True, 0), open_vex=3)
        self.assertEqual(tier, "LIMITED")

    def test_high_not_attested_with_errors(self):
        """Not attested + errors → HIGH."""
        tier = _api._compute_model_risk_tier(self._rec(False, False, 2), open_vex=0)
        self.assertEqual(tier, "HIGH")

    def test_unacceptable_majority_failure_plus_vex(self):
        """Majority policy failures AND VEX present → UNACCEPTABLE."""
        rec = {
            "attestation_passed": False,
            "policy_results": {
                "p1": {"passed": False, "error_count": 2, "warning_count": 0},
                "p2": {"passed": False, "error_count": 1, "warning_count": 0},
                "p3": {"passed": True, "error_count": 0, "warning_count": 0},
            },
        }
        # 2/3 failed > 0.5
        tier = _api._compute_model_risk_tier(rec, open_vex=5)
        self.assertEqual(tier, "UNACCEPTABLE")

    def test_not_unacceptable_without_vex(self):
        """Majority policy failures but NO VEX → should not be UNACCEPTABLE."""
        rec = {
            "attestation_passed": False,
            "policy_results": {
                "p1": {"passed": False, "error_count": 2, "warning_count": 0},
                "p2": {"passed": False, "error_count": 1, "warning_count": 0},
            },
        }
        tier = _api._compute_model_risk_tier(rec, open_vex=0)
        self.assertNotEqual(tier, "UNACCEPTABLE")

    def test_empty_inventory_record_minimal(self):
        """Empty record (no policies, attested=False implicitly) → MINIMAL when no VEX."""
        tier = _api._compute_model_risk_tier({}, open_vex=0)
        # no policies, no attestation_passed key → defaults: no errors, no vex → MINIMAL
        self.assertEqual(tier, "MINIMAL")


# ---------------------------------------------------------------------------
# 2. GET /cloud/tenants/{tenant_id}/risk-profile
# ---------------------------------------------------------------------------

class TestCloudRiskProfileApi(unittest.TestCase):
    """Tests for the W80 per-tenant risk-profile endpoint."""

    def setUp(self):
        _register_tenant(_TENANT_RISK)

    def tearDown(self):
        _clear_tenant(_TENANT_RISK)

    def _call(self, tid: str = _TENANT_RISK):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            _api.cloud_get_tenant_risk_profile(tid)
        )

    def _body(self, resp) -> dict:
        import json as _json
        return _json.loads(resp.body)

    def test_unknown_tenant_returns_404(self):
        resp = self._call("no-such-tenant")
        self.assertEqual(resp.status_code, 404)
        body = self._body(resp)
        self.assertIn("not found", body["detail"])

    def test_empty_inventory_returns_minimal(self):
        resp = self._call()
        self.assertEqual(resp.status_code, 200)
        body = self._body(resp)
        self.assertEqual(body["tenant_id"], _TENANT_RISK)
        self.assertEqual(body["overall_risk_tier"], "MINIMAL")
        self.assertEqual(body["model_count"], 0)

    def test_attested_model_no_vex_is_minimal(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        resp = self._call()
        body = self._body(resp)
        self.assertEqual(body["overall_risk_tier"], "MINIMAL")
        self.assertEqual(body["model_count"], 1)
        self.assertEqual(body["models"][0]["risk_tier"], "MINIMAL")

    def test_attested_with_vex_is_limited(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        _seed_vex(_TENANT_RISK, count=2)
        resp = self._call()
        body = self._body(resp)
        self.assertEqual(body["overall_risk_tier"], "LIMITED")

    def test_not_attested_with_errors_is_high(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=False,
                        policy_passed=False, error_count=3)
        resp = self._call()
        body = self._body(resp)
        self.assertEqual(body["overall_risk_tier"], "HIGH")

    def test_response_contains_enforcement_signal(self):
        resp = self._call()
        body = self._body(resp)
        self.assertIn("enforcement_deadline", body)
        self.assertIn("days_until_enforcement", body)
        self.assertIn("enforcement_risk_level", body)


# ---------------------------------------------------------------------------
# 3. GET /cloud/risk-overview
# ---------------------------------------------------------------------------

class TestCloudRiskOverviewApi(unittest.TestCase):
    """Tests for the W80 platform-wide risk-overview endpoint."""

    def setUp(self):
        _register_tenant(_TENANT_OVERVIEW)
        _register_tenant(_TENANT_OVERVIEW_B)

    def tearDown(self):
        _clear_tenant(_TENANT_OVERVIEW, _TENANT_OVERVIEW_B)

    def _call(self):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            _api.cloud_get_risk_overview()
        )

    def _body(self, resp) -> dict:
        import json as _json
        return _json.loads(resp.body)

    def test_returns_200_with_risk_summary_keys(self):
        resp = self._call()
        self.assertEqual(resp.status_code, 200)
        body = self._body(resp)
        self.assertIn("risk_summary", body)
        for tier in ("UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"):
            self.assertIn(tier, body["risk_summary"])

    def test_empty_tenants_all_minimal(self):
        resp = self._call()
        body = self._body(resp)
        # Both tenants registered, both empty → both MINIMAL
        self.assertGreaterEqual(body["total_tenants"], 2)
        self.assertEqual(body["risk_summary"]["UNACCEPTABLE"], 0)
        self.assertEqual(body["risk_summary"]["HIGH"], 0)

    def test_high_risk_tenant_reflected_in_overview(self):
        _seed_inventory(_TENANT_OVERVIEW, attestation_passed=False,
                        policy_passed=False, error_count=3)
        resp = self._call()
        body = self._body(resp)
        self.assertGreater(body["risk_summary"]["HIGH"], 0)


# ---------------------------------------------------------------------------
# 4. CLI cloud-risk command
# ---------------------------------------------------------------------------

class TestCloudRiskCmd(unittest.TestCase):
    """Integration tests for _cmd_cloud_risk."""

    def setUp(self):
        _register_tenant(_TENANT_RISK)

    def tearDown(self):
        _clear_tenant(_TENANT_RISK)

    def test_unknown_tenant_returns_rc1(self):
        args = _make_args(tenant_id="no-such-tenant")
        rc, _ = _capture_stdout(_cmd_cloud_risk, args, False)
        self.assertEqual(rc, 1)

    def test_no_tenant_and_no_overview_returns_rc1(self):
        args = _make_args(tenant_id=None)
        rc, _ = _capture_stdout(_cmd_cloud_risk, args, False)
        self.assertEqual(rc, 1)

    def test_minimal_tier_returns_rc0(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        args = _make_args()
        rc, _ = _capture_stdout(_cmd_cloud_risk, args, False)
        self.assertEqual(rc, 0)

    def test_limited_tier_returns_rc0(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        _seed_vex(_TENANT_RISK, count=1)
        args = _make_args()
        rc, _ = _capture_stdout(_cmd_cloud_risk, args, False)
        self.assertEqual(rc, 0)

    def test_high_tier_returns_rc2(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=False,
                        policy_passed=False, error_count=2)
        args = _make_args()
        rc, _ = _capture_stdout(_cmd_cloud_risk, args, False)
        self.assertEqual(rc, 2)

    def test_quiet_suppresses_output(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        args = _make_args(quiet=True)
        rc, out = _capture_stdout(_cmd_cloud_risk, args, True)
        self.assertEqual(out, "")
        self.assertEqual(rc, 0)


# ---------------------------------------------------------------------------
# 5. CLI cloud-risk --json flag
# ---------------------------------------------------------------------------

class TestCloudRiskJsonFlag(unittest.TestCase):
    """Validate --json output shape for cloud-risk."""

    def setUp(self):
        _register_tenant(_TENANT_RISK)

    def tearDown(self):
        _clear_tenant(_TENANT_RISK)

    def test_json_output_is_parseable(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        args = _make_args(output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_risk, args, False)
        parsed = json.loads(out)
        self.assertIn("tenant_id", parsed)
        self.assertIn("overall_risk_tier", parsed)

    def test_json_output_contains_model_list(self):
        _seed_inventory(_TENANT_RISK, attestation_passed=True, policy_passed=True)
        args = _make_args(output_json=True)
        _, out = _capture_stdout(_cmd_cloud_risk, args, False)
        parsed = json.loads(out)
        self.assertIn("models", parsed)
        self.assertIsInstance(parsed["models"], list)

    def test_overview_json_output_contains_risk_summary(self):
        args = _make_args(tenant_id=None, overview=True, output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_risk, args, False)
        parsed = json.loads(out)
        self.assertIn("risk_summary", parsed)
        self.assertIn("total_tenants", parsed)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
