"""W81: Remediation Plan Generator — unit tests.

Tests:
    TestRemediationStep          (4 tests): dataclass field contracts
    TestGenerateRemediationPlan  (8 tests): pure-function logic coverage
    TestRemediationPlanApi       (6 tests): GET /cloud/tenants/{id}/remediation-plan
    TestCloudRemediateCmd        (6 tests): CLI cloud-remediate happy/sad paths
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

from squash.risk import RemediationStep, generate_remediation_plan  # noqa: E402
from squash.cli import _cmd_cloud_remediate  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TENANT_A = "t-w81-a"
_TENANT_B = "t-w81-b"
_TENANT_C = "t-w81-c"


def _make_args(tenant_id: str = _TENANT_A, output_json: bool = False, quiet: bool = False) -> argparse.Namespace:
    return argparse.Namespace(tenant_id=tenant_id, output_json=output_json, quiet=quiet)


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


def _register_tenant(tid: str) -> None:
    _api._tenants[tid] = {"tenant_id": tid, "name": "W81 Test Corp"}


def _clear_tenant(*tids: str) -> None:
    for tid in tids:
        _api._tenants.pop(tid, None)
        _api._policy_stats.pop(tid, None)
        _api._vertex_results.pop(tid, None)
        _api._vex_alerts.pop(tid, None)
        _api._inventory.pop(tid, None)


def _seed_inventory(
    tid: str,
    attestation_passed: bool = True,
    policy_passed: bool = True,
    error_count: int = 0,
    model_id: str = "test-model",
) -> dict:
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
        "record_id": "rec-w81",
    }
    _api._inventory[tid].append(record)
    return record


def _seed_vex(tid: str, count: int = 2) -> None:
    for i in range(count):
        _api._vex_alerts[tid].append({
            "alert_id": f"va-w81-{i}",
            "cve_id": f"CVE-2026-9{i:03d}",
            "severity": "high",
            "status": "open",
            "model_id": "test-model",
            "tenant_id": tid,
            "timestamp": "2026-04-16T00:00:00Z",
        })


def _make_multi_policy_record(passed_count: int, failed_count: int) -> dict:
    """Return a record with a mix of passing and failing policy results."""
    policy_results = {}
    for i in range(passed_count):
        policy_results[f"policy-pass-{i}"] = {"passed": True, "error_count": 0, "warning_count": 0}
    for i in range(failed_count):
        policy_results[f"policy-fail-{i}"] = {"passed": False, "error_count": 1, "warning_count": 0}
    return {
        "model_id": "multi-model",
        "model_path": "/models/multi-model",
        "bom_path": "",
        "attestation_passed": failed_count == 0,
        "policy_results": policy_results,
        "vex_cves": [],
        "timestamp": "2026-04-16T00:00:00Z",
        "record_id": "rec-w81-multi",
    }


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
# 1. RemediationStep — dataclass field contracts
# ---------------------------------------------------------------------------

class TestRemediationStep(unittest.TestCase):
    """Contracts for the RemediationStep dataclass."""

    def _make(self, priority: int = 1) -> RemediationStep:
        return RemediationStep(
            id="test_step",
            priority=priority,
            action="Test action",
            description="Do the thing.",
            evidence_required="Proof of thing",
            estimated_effort="1d",
        )

    def test_fields_present(self):
        """All six required fields are accessible on a RemediationStep."""
        s = self._make()
        self.assertEqual(s.id, "test_step")
        self.assertEqual(s.priority, 1)
        self.assertEqual(s.action, "Test action")
        self.assertIn("thing", s.description)
        self.assertIn("thing", s.evidence_required)
        self.assertEqual(s.estimated_effort, "1d")

    def test_priority_ordering(self):
        """Steps with lower priority int sort before higher."""
        steps = [self._make(3), self._make(1), self._make(2)]
        sorted_steps = sorted(steps, key=lambda s: s.priority)
        self.assertEqual([s.priority for s in sorted_steps], [1, 2, 3])

    def test_evidence_required_non_empty(self):
        """evidence_required should not be an empty string for a real step."""
        s = RemediationStep(
            id="sign_model_artifact",
            priority=2,
            action="Sign model artefact",
            description="Add a cosign signature.",
            evidence_required="Cosign signature verifiable",
            estimated_effort="1d",
        )
        self.assertTrue(len(s.evidence_required) > 0)

    def test_estimated_effort_format(self):
        """estimated_effort values should match expected short format."""
        for effort in ("1d", "1w", "2w"):
            s = self._make()
            s = RemediationStep(
                id="x", priority=1, action="x", description="x",
                evidence_required="x", estimated_effort=effort,
            )
            self.assertEqual(s.estimated_effort, effort)


# ---------------------------------------------------------------------------
# 2. generate_remediation_plan — pure-function logic
# ---------------------------------------------------------------------------

class TestGenerateRemediationPlan(unittest.TestCase):
    """Unit tests for generate_remediation_plan()."""

    def _policy(self, passed: bool = True, error_count: int = 0) -> dict:
        return {"passed": passed, "error_count": error_count, "warning_count": 0}

    def test_minimal_no_issues_returns_empty(self):
        """MINIMAL tier, all clear → empty plan."""
        steps = generate_remediation_plan(
            risk_tier="MINIMAL",
            policy_results={"p1": self._policy(True, 0)},
            open_vex=0,
            attestation_passed=True,
        )
        self.assertEqual(steps, [])

    def test_attestation_failure_adds_priority1_step(self):
        """Not attested → priority-1 'obtain_attestation' step."""
        steps = generate_remediation_plan(
            risk_tier="HIGH",
            policy_results={"p1": self._policy(True, 0)},
            open_vex=0,
            attestation_passed=False,
        )
        ids = [s.id for s in steps]
        self.assertIn("obtain_attestation", ids)
        attest_step = next(s for s in steps if s.id == "obtain_attestation")
        self.assertEqual(attest_step.priority, 1)

    def test_high_policy_failure_rate_adds_priority1_step(self):
        """Majority policy failures → priority-1 'remediate_policy_failures' step."""
        policy = {
            "p1": self._policy(False, 2),
            "p2": self._policy(False, 1),
            "p3": self._policy(True, 0),
        }
        steps = generate_remediation_plan(
            risk_tier="UNACCEPTABLE",
            policy_results=policy,
            open_vex=3,
            attestation_passed=False,
        )
        ids = [s.id for s in steps]
        self.assertIn("remediate_policy_failures", ids)
        step = next(s for s in steps if s.id == "remediate_policy_failures")
        self.assertEqual(step.priority, 1)

    def test_open_vex_adds_priority2_step(self):
        """open_vex > 0 → priority-2 'close_vex_alerts' step."""
        steps = generate_remediation_plan(
            risk_tier="LIMITED",
            policy_results={"p1": self._policy(True, 0)},
            open_vex=5,
            attestation_passed=True,
        )
        ids = [s.id for s in steps]
        self.assertIn("close_vex_alerts", ids)
        vex_step = next(s for s in steps if s.id == "close_vex_alerts")
        self.assertEqual(vex_step.priority, 2)

    def test_steps_sorted_by_priority(self):
        """Returned steps are sorted ascending by priority."""
        steps = generate_remediation_plan(
            risk_tier="UNACCEPTABLE",
            policy_results={
                "p1": self._policy(False, 2),
                "p2": self._policy(False, 1),
            },
            open_vex=3,
            attestation_passed=False,
        )
        priorities = [s.priority for s in steps]
        self.assertEqual(priorities, sorted(priorities))

    def test_empty_policy_results_no_failure_step(self):
        """Empty policy_results → no policy-failure step (division by zero guard)."""
        steps = generate_remediation_plan(
            risk_tier="HIGH",
            policy_results={},
            open_vex=0,
            attestation_passed=False,
        )
        ids = [s.id for s in steps]
        self.assertNotIn("remediate_policy_failures", ids)
        # But attestation step IS present
        self.assertIn("obtain_attestation", ids)

    def test_no_vex_no_vex_step(self):
        """open_vex == 0 → no close_vex_alerts step."""
        steps = generate_remediation_plan(
            risk_tier="HIGH",
            policy_results={"p1": self._policy(False, 1)},
            open_vex=0,
            attestation_passed=False,
        )
        ids = [s.id for s in steps]
        self.assertNotIn("close_vex_alerts", ids)

    def test_duplicate_step_ids_not_emitted(self):
        """Each step id appears at most once even when multiple triggers fire."""
        policy = {
            "p1": self._policy(False, 2),
            "p2": self._policy(False, 1),
        }
        steps = generate_remediation_plan(
            risk_tier="UNACCEPTABLE",
            policy_results=policy,
            open_vex=5,
            attestation_passed=False,
        )
        ids = [s.id for s in steps]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate step IDs emitted")


# ---------------------------------------------------------------------------
# 3. GET /cloud/tenants/{tenant_id}/remediation-plan — API tests
# ---------------------------------------------------------------------------

class TestRemediationPlanApi(unittest.TestCase):
    """Tests for the W81 remediation-plan REST endpoint."""

    def setUp(self):
        _clear_tenant(_TENANT_B)
        _register_tenant(_TENANT_B)

    def tearDown(self):
        _clear_tenant(_TENANT_B)

    def _call(self, tenant_id: str = _TENANT_B):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            _api.cloud_get_remediation_plan(tenant_id)
        )

    def test_unknown_tenant_404(self):
        """Non-existent tenant → HTTPException 404."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self._call("no-such-tenant-w81")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_response_has_required_keys(self):
        """200 response includes all required top-level keys."""
        _seed_inventory(_TENANT_B, attestation_passed=True, policy_passed=True)
        resp = self._call()
        data = json.loads(resp.body)
        for key in ("tenant_id", "risk_tier", "total_steps", "critical_count", "steps",
                    "enforcement_deadline", "days_until_enforcement", "enforcement_risk_level"):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_minimal_tenant_zero_steps(self):
        """Fully compliant tenant → 0 steps, 0 critical."""
        _seed_inventory(_TENANT_B, attestation_passed=True, policy_passed=True, error_count=0)
        resp = self._call()
        data = json.loads(resp.body)
        self.assertEqual(data["total_steps"], 0)
        self.assertEqual(data["critical_count"], 0)
        self.assertEqual(data["steps"], [])

    def test_non_attested_generates_steps(self):
        """Non-attested model → at least one step returned."""
        _seed_inventory(_TENANT_B, attestation_passed=False, policy_passed=False, error_count=2)
        resp = self._call()
        data = json.loads(resp.body)
        self.assertGreater(data["total_steps"], 0)

    def test_critical_count_matches_priority1_steps(self):
        """critical_count equals the number of priority-1 steps."""
        _seed_inventory(_TENANT_B, attestation_passed=False, policy_passed=False, error_count=2)
        resp = self._call()
        data = json.loads(resp.body)
        computed = sum(1 for s in data["steps"] if s["priority"] == 1)
        self.assertEqual(data["critical_count"], computed)

    def test_enforcement_signal_present(self):
        """enforcement_deadline and enforcement_risk_level are non-empty strings."""
        _seed_inventory(_TENANT_B, attestation_passed=True, policy_passed=True)
        resp = self._call()
        data = json.loads(resp.body)
        self.assertTrue(data["enforcement_deadline"])
        self.assertTrue(data["enforcement_risk_level"])


# ---------------------------------------------------------------------------
# 4. CLI cloud-remediate — command tests
# ---------------------------------------------------------------------------

class TestCloudRemediateCmd(unittest.TestCase):
    """Tests for the W81 CLI cloud-remediate subcommand."""

    def setUp(self):
        _clear_tenant(_TENANT_C)
        _register_tenant(_TENANT_C)

    def tearDown(self):
        _clear_tenant(_TENANT_C)

    def test_unknown_tenant_rc1(self):
        """Unknown tenant → rc 1."""
        args = _make_args(tenant_id="no-such-w81")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = _cmd_cloud_remediate(args, quiet=False)
        self.assertEqual(rc, 1)
        self.assertIn("not found", buf.getvalue())

    def test_compliant_tenant_rc0(self):
        """Fully compliant tenant → rc 0."""
        _seed_inventory(_TENANT_C, attestation_passed=True, policy_passed=True, error_count=0)
        args = _make_args(tenant_id=_TENANT_C)
        rc, _ = _capture_stdout(_cmd_cloud_remediate, args, quiet=False)
        self.assertEqual(rc, 0)

    def test_critical_steps_rc2(self):
        """Tenant with critical steps → rc 2."""
        _seed_inventory(_TENANT_C, attestation_passed=False, policy_passed=False, error_count=3)
        args = _make_args(tenant_id=_TENANT_C)
        rc, _ = _capture_stdout(_cmd_cloud_remediate, args, quiet=False)
        self.assertEqual(rc, 2)

    def test_quiet_mode_no_stdout(self):
        """--quiet suppresses stdout output."""
        _seed_inventory(_TENANT_C, attestation_passed=False, policy_passed=False, error_count=2)
        args = _make_args(tenant_id=_TENANT_C, quiet=True)
        rc, output = _capture_stdout(_cmd_cloud_remediate, args, quiet=True)
        self.assertEqual(output, "")
        # rc is still meaningful
        self.assertIn(rc, (0, 2))

    def test_json_flag_valid_json(self):
        """--json outputs valid JSON with required keys."""
        _seed_inventory(_TENANT_C, attestation_passed=False, policy_passed=False, error_count=2)
        args = _make_args(tenant_id=_TENANT_C, output_json=True)
        rc, output = _capture_stdout(_cmd_cloud_remediate, args, quiet=False)
        data = json.loads(output)
        for key in ("tenant_id", "risk_tier", "total_steps", "critical_count", "steps"):
            self.assertIn(key, data)

    def test_json_rc_reflects_critical(self):
        """--json rc is 2 when critical steps exist, 0 when none."""
        _seed_inventory(_TENANT_C, attestation_passed=True, policy_passed=True, error_count=0)
        args = _make_args(tenant_id=_TENANT_C, output_json=True)
        rc, _ = _capture_stdout(_cmd_cloud_remediate, args, quiet=False)
        self.assertEqual(rc, 0)

        _clear_tenant(_TENANT_C)
        _register_tenant(_TENANT_C)
        _seed_inventory(_TENANT_C, attestation_passed=False, policy_passed=False, error_count=2)
        args2 = _make_args(tenant_id=_TENANT_C, output_json=True)
        rc2, _ = _capture_stdout(_cmd_cloud_remediate, args2, quiet=False)
        self.assertEqual(rc2, 2)


if __name__ == "__main__":
    unittest.main()
