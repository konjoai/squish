"""W79: Cloud Attestation CLI + Tenant VEX CLI — unit tests.

Tests:
    TestCloudAttestCmd  (6 tests): cloud-attest happy/sad paths with mocked pipeline
    TestCloudVexCmd     (6 tests): cloud-vex happy/sad paths with seeded VEX alerts
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
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# Ensure squish package root is importable
_SQUISH_ROOT = os.path.join(os.path.dirname(__file__), "..", "squish")
if _SQUISH_ROOT not in sys.path:
    sys.path.insert(0, _SQUISH_ROOT)

import squash.api as _api  # noqa: E402 — after stubs

# Build minimal package stubs so `from squish.squash import api` resolves to
# _api at test execution time. Aliasing is deferred to setUpModule/tearDownModule
# to avoid permanently replacing squish.squash.api during pytest collection.
_squish_pkg = sys.modules.setdefault("squish", types.ModuleType("squish"))
_squish_squash_pkg = sys.modules.setdefault("squish.squash", types.ModuleType("squish.squash"))

# Import CLI handlers directly after stubs are in place
from squash.cli import (  # noqa: E402
    _cmd_cloud_attest,
    _cmd_cloud_vex,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TENANT_A = "t-w79-attest"
_TENANT_VEX = "t-w79-vex"


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "output_json": False,
        "quiet": False,
        "output_path": None,
        "policy": "enterprise-strict",
        "limit": 50,
        "vex_status": None,
        "severity": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


def _register_tenant(tid: str) -> None:
    _api._tenants[tid] = {"tenant_id": tid, "name": "W79 Test Corp"}


def _clear_tenant(*tids: str) -> None:
    for tid in tids:
        _api._tenants.pop(tid, None)
        _api._policy_stats.pop(tid, None)
        _api._vertex_results.pop(tid, None)
        _api._vex_alerts.pop(tid, None)
        _api._inventory.pop(tid, None)


def _seed_vex_alerts(tid: str, count: int = 3) -> None:
    """Seed synthetic VEX alert records directly into the in-memory store."""
    for i in range(count):
        record = {
            "alert_id": f"alert-{i}",
            "cve_id": f"CVE-2025-{1000 + i}",
            "severity": "high" if i % 2 == 0 else "medium",
            "status": "open",
            "model_id": "test-model",
            "tenant_id": tid,
            "timestamp": f"2025-06-{10 + i:02d}T00:00:00Z",
        }
        _api._vex_alerts[tid].append(record)


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------

_orig_squash_api = None


def setUpModule() -> None:  # noqa: N802 — unittest naming convention
    global _orig_squash_api
    _orig_squash_api = sys.modules.get("squish.squash.api")
    sys.modules["squish.squash.api"] = _api
    _squish_squash_pkg.api = _api  # type: ignore[attr-defined]


def tearDownModule() -> None:  # noqa: N802 — unittest naming convention
    if _orig_squash_api is not None:
        sys.modules["squish.squash.api"] = _orig_squash_api
        _squish_squash_pkg.api = _orig_squash_api  # type: ignore[attr-defined]
    else:
        sys.modules.pop("squish.squash.api", None)


# ---------------------------------------------------------------------------
# 1. cloud-attest
# ---------------------------------------------------------------------------

def _make_passing_attest_result(model_id: str = "test-model", out_dir: Path | None = None):
    """Build a minimal AttestResult-like mock that reports a passing attestation."""
    from dataclasses import dataclass, field

    @dataclass
    class _FakePolicyResult:
        passed: bool = True
        error_count: int = 0
        warning_count: int = 0

        def summary(self) -> str:
            return "PASS (0 errors)"

    result = MagicMock()
    result.passed = True
    result.model_id = model_id
    result.cyclonedx_path = (out_dir / "bom.json") if out_dir else None
    result.policy_results = {"enterprise-strict": _FakePolicyResult()}
    return result


def _make_failing_attest_result(model_id: str = "test-model"):
    """Build a minimal AttestResult-like mock that reports a failing attestation."""
    result = MagicMock()
    result.passed = False
    result.model_id = model_id
    result.cyclonedx_path = None
    result.policy_results = {}
    return result


class TestCloudAttestCmd(unittest.TestCase):
    """Tests for _cmd_cloud_attest."""

    def setUp(self) -> None:
        _register_tenant(_TENANT_A)

    def tearDown(self) -> None:
        _clear_tenant(_TENANT_A)

    # ------------------------------------------------------------------
    def test_passing_attest_exits_zero_and_registers(self) -> None:
        """A passing attestation should return exit 0 and write to inventory."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-model"
            model_dir.mkdir()

            passing_result = _make_passing_attest_result("test-model", model_dir)

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.return_value = passing_result
                args = _make_args(tenant_id=_TENANT_A, model_path=str(model_dir))
                rc, _ = _capture_stdout(_cmd_cloud_attest, args, quiet=False)

            self.assertEqual(rc, 0)
            # Inventory record should be written
            records = _api._db_read_inventory(_TENANT_A)
            self.assertGreater(len(records), 0)
            self.assertTrue(records[-1]["attestation_passed"])

    def test_passing_attest_output_contains_pass(self) -> None:
        """stdout should contain PASS when attestation passes."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-model"
            model_dir.mkdir()

            passing_result = _make_passing_attest_result("test-model", model_dir)

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.return_value = passing_result
                args = _make_args(tenant_id=_TENANT_A, model_path=str(model_dir))
                rc, out = _capture_stdout(_cmd_cloud_attest, args, quiet=False)

            self.assertIn("PASS", out)

    def test_failing_attest_exits_two(self) -> None:
        """A failing attestation should return exit 2."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-model"
            model_dir.mkdir()

            failing_result = _make_failing_attest_result("test-model")

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.return_value = failing_result
                args = _make_args(tenant_id=_TENANT_A, model_path=str(model_dir))
                rc, _ = _capture_stdout(_cmd_cloud_attest, args, quiet=False)

            self.assertEqual(rc, 2)

    def test_unknown_tenant_exits_one(self) -> None:
        """An unknown tenant_id should return exit 1 without calling AttestPipeline."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-model"
            model_dir.mkdir()
            args = _make_args(tenant_id="no-such-tenant-xyz", model_path=str(model_dir))
            rc, _ = _capture_stdout(_cmd_cloud_attest, args, quiet=False)
            self.assertEqual(rc, 1)

    def test_nonexistent_model_path_exits_one(self) -> None:
        """A model_path that does not exist should return exit 1."""
        args = _make_args(tenant_id=_TENANT_A, model_path="/nonexistent/path/to/model")
        rc, _ = _capture_stdout(_cmd_cloud_attest, args, quiet=False)
        self.assertEqual(rc, 1)

    def test_json_flag_outputs_parseable_json(self) -> None:
        """--json flag should produce parseable JSON containing 'model_id' and 'attestation_passed'."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-model"
            model_dir.mkdir()

            passing_result = _make_passing_attest_result("test-model", model_dir)

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.return_value = passing_result
                args = _make_args(tenant_id=_TENANT_A, model_path=str(model_dir), output_json=True)
                rc, out = _capture_stdout(_cmd_cloud_attest, args, quiet=False)

            self.assertEqual(rc, 0)
            # Find JSON block in output
            lines = out.strip().splitlines()
            json_start = next(
                (i for i, line in enumerate(lines) if line.strip().startswith("{")),
                None,
            )
            self.assertIsNotNone(json_start, "No JSON block found in output")
            parsed = json.loads("\n".join(lines[json_start:]))
            self.assertIn("model_id", parsed)
            self.assertIn("attestation_passed", parsed)
            self.assertTrue(parsed["attestation_passed"])


# ---------------------------------------------------------------------------
# 2. cloud-vex
# ---------------------------------------------------------------------------

class TestCloudVexCmd(unittest.TestCase):
    """Tests for _cmd_cloud_vex."""

    def setUp(self) -> None:
        _register_tenant(_TENANT_VEX)

    def tearDown(self) -> None:
        _clear_tenant(_TENANT_VEX)

    # ------------------------------------------------------------------
    def test_no_alerts_exits_zero(self) -> None:
        """An empty VEX store should return exit 0 (no alerts is not an error)."""
        args = _make_args(tenant_id=_TENANT_VEX)
        rc, _ = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertEqual(rc, 0)

    def test_no_alerts_output_says_no_alerts(self) -> None:
        """stdout should mention no alerts when store is empty."""
        args = _make_args(tenant_id=_TENANT_VEX)
        rc, out = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertIn("No VEX alerts", out)

    def test_seeded_alerts_exits_zero_and_shows_cve(self) -> None:
        """Seeded alerts should be listed and exit 0."""
        _seed_vex_alerts(_TENANT_VEX, count=3)
        args = _make_args(tenant_id=_TENANT_VEX)
        rc, out = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertEqual(rc, 0)
        self.assertIn("CVE-2025-1000", out)

    def test_unknown_tenant_exits_one(self) -> None:
        """An unknown tenant_id should return exit 1."""
        args = _make_args(tenant_id="no-such-tenant-xyz")
        rc, _ = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertEqual(rc, 1)

    def test_json_flag_outputs_parseable_json(self) -> None:
        """--json flag should dump parseable JSON with 'tenant_id', 'count', 'alerts'."""
        _seed_vex_alerts(_TENANT_VEX, count=2)
        args = _make_args(tenant_id=_TENANT_VEX, output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertEqual(rc, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["tenant_id"], _TENANT_VEX)
        self.assertIn("count", parsed)
        self.assertIn("alerts", parsed)
        self.assertEqual(parsed["count"], 2)

    def test_limit_flag_caps_results(self) -> None:
        """--limit N should return at most N alerts."""
        _seed_vex_alerts(_TENANT_VEX, count=10)
        args = _make_args(tenant_id=_TENANT_VEX, limit=3, output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_vex, args, quiet=False)
        self.assertEqual(rc, 0)
        parsed = json.loads(out)
        self.assertLessEqual(parsed["count"], 3)
        self.assertLessEqual(len(parsed["alerts"]), 3)


if __name__ == "__main__":
    unittest.main()
