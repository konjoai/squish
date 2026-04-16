"""W77: Cloud CLI Commands — unit tests for cloud-status, cloud-report, cloud-export.

Tests:
    TestCloudStatusCmd  (5 tests): cloud-status happy/sad paths
    TestCloudReportCmd  (4 tests): cloud-report happy/sad paths
    TestCloudExportCmd  (3 tests): cloud-export happy/sad paths
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
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

# Ensure squish package root is importable
_SQUISH_ROOT = os.path.join(os.path.dirname(__file__), "..", "squish")
if _SQUISH_ROOT not in sys.path:
    sys.path.insert(0, _SQUISH_ROOT)

import squash.api as _api  # noqa: E402 — after stubs

# Build minimal package stubs so `from squish.squash import api` resolves to
# _api at test execution time. The actual sys.modules aliasing is deferred to
# setUpModule/tearDownModule so pytest collection does NOT permanently replace
# squish.squash.api in sys.modules (which would break other test modules that
# also import squish.squash.api at collection time).
_squish_pkg = sys.modules.setdefault("squish", types.ModuleType("squish"))
_squish_squash_pkg = sys.modules.setdefault("squish.squash", types.ModuleType("squish.squash"))

# Import CLI handlers directly so we can call them in-process
from squash.cli import (  # noqa: E402 — after stubs
    _cmd_cloud_export,
    _cmd_cloud_report,
    _cmd_cloud_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFORMANT_TENANT = "t-w77-ok"
_NON_CONFORMANT_TENANT = "t-w77-fail"


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for CLI commands."""
    defaults = {
        "output_json": False,
        "quiet": False,
        "output_path": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _register_conformant(tid: str = _CONFORMANT_TENANT) -> None:
    """Seed a passing tenant directly into the in-memory store."""
    _api._tenants[tid] = {"tenant_id": tid, "name": "W77 Conformant Corp"}
    # No _policy_stats → score defaults to 100.0 (Art. 9/17 passes)
    # Add one passing attestation so pass_rate=1.0 >= 0.8 (Art. 12/18 passes)
    _api._vertex_results[tid].appendleft({"passed": True, "model_id": "test-model", "ts": "2025-01-01"})


def _register_non_conformant(tid: str = _NON_CONFORMANT_TENANT) -> None:
    """Seed a failing tenant with a policy_stats entry that forces score < 80."""
    _api._tenants[tid] = {"tenant_id": tid, "name": "W77 Non-Conformant Corp"}
    # Force compliance_score to 0 — all checks failed
    _api._policy_stats[tid]["test_policy"] = {"passed": 0, "failed": 100}


def _clear_tenant(*tids: str) -> None:
    for tid in tids:
        _api._tenants.pop(tid, None)
        _api._policy_stats.pop(tid, None)
        _api._vertex_results.pop(tid, None)
        _api._ado_results.pop(tid, None)


def _capture_stdout(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (return_code, captured_stdout_str)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# Module-level setup / teardown — alias squish.squash.api only while running
# so collection-time imports in other test modules are not affected.
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
# 1. cloud-status
# ---------------------------------------------------------------------------

class TestCloudStatusCmd(unittest.TestCase):
    """Tests for _cmd_cloud_status."""

    def setUp(self) -> None:
        _register_conformant(_CONFORMANT_TENANT)

    def tearDown(self) -> None:
        _clear_tenant(_CONFORMANT_TENANT, _NON_CONFORMANT_TENANT)

    # ------------------------------------------------------------------
    def test_known_conformant_exits_zero(self) -> None:
        """A conformant tenant should return exit code 0."""
        args = _make_args(tenant_id=_CONFORMANT_TENANT)
        rc, _ = _capture_stdout(_cmd_cloud_status, args, quiet=False)
        self.assertEqual(rc, 0)

    def test_output_contains_score_label(self) -> None:
        """stdout should contain 'score:' when not quiet."""
        args = _make_args(tenant_id=_CONFORMANT_TENANT)
        rc, out = _capture_stdout(_cmd_cloud_status, args, quiet=False)
        self.assertIn("score:", out)

    def test_output_contains_enforcement_days(self) -> None:
        """stdout should mention '...days until enforcement'."""
        args = _make_args(tenant_id=_CONFORMANT_TENANT)
        rc, out = _capture_stdout(_cmd_cloud_status, args, quiet=False)
        self.assertIn("days until enforcement", out)

    def test_unknown_tenant_exits_one(self) -> None:
        """An unknown tenant_id should return exit code 1."""
        args = _make_args(tenant_id="no-such-tenant-xyz")
        rc, _ = _capture_stdout(_cmd_cloud_status, args, quiet=False)
        self.assertEqual(rc, 1)

    def test_json_flag_outputs_parseable_json(self) -> None:
        """--json flag should append parseable JSON containing 'conformant' key."""
        args = _make_args(tenant_id=_CONFORMANT_TENANT, output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_status, args, quiet=False)
        # Find the JSON block (may be preceded by human-readable line)
        lines = out.strip().splitlines()
        json_start = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("{")),
            None,
        )
        self.assertIsNotNone(json_start, "No JSON block found in output")
        parsed = json.loads("\n".join(lines[json_start:]))
        self.assertIn("conformant", parsed)


# ---------------------------------------------------------------------------
# 2. cloud-report
# ---------------------------------------------------------------------------

class TestCloudReportCmd(unittest.TestCase):
    """Tests for _cmd_cloud_report."""

    def tearDown(self) -> None:
        _clear_tenant(_CONFORMANT_TENANT, _NON_CONFORMANT_TENANT, "t-w77-a", "t-w77-b")

    # ------------------------------------------------------------------
    def test_empty_platform_exits_zero(self) -> None:
        """When no tenants are registered, report exits 0 and says so."""
        # Ensure no tenants
        _clear_tenant(*list(_api._tenants.keys()))
        args = _make_args()
        rc, out = _capture_stdout(_cmd_cloud_report, args, quiet=False)
        self.assertEqual(rc, 0)
        self.assertIn("no tenants", out.lower())

    def test_two_conformant_tenants_exit_zero(self) -> None:
        """All-conformant platform should return exit code 0."""
        _register_conformant("t-w77-a")
        _register_conformant("t-w77-b")
        args = _make_args()
        rc, _ = _capture_stdout(_cmd_cloud_report, args, quiet=False)
        self.assertEqual(rc, 0)

    def test_any_non_conformant_exits_two(self) -> None:
        """Platform with at least one non-conformant tenant should return exit code 2."""
        _register_conformant("t-w77-a")
        _register_non_conformant(_NON_CONFORMANT_TENANT)
        args = _make_args()
        rc, _ = _capture_stdout(_cmd_cloud_report, args, quiet=False)
        self.assertEqual(rc, 2)

    def test_json_flag_outputs_total_tenants_key(self) -> None:
        """--json flag should include 'total_tenants' key."""
        _register_conformant("t-w77-a")
        args = _make_args(output_json=True)
        rc, out = _capture_stdout(_cmd_cloud_report, args, quiet=False)
        lines = out.strip().splitlines()
        json_start = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("{")),
            None,
        )
        self.assertIsNotNone(json_start, "No JSON block found in output")
        parsed = json.loads("\n".join(lines[json_start:]))
        self.assertIn("total_tenants", parsed)


# ---------------------------------------------------------------------------
# 3. cloud-export
# ---------------------------------------------------------------------------

class TestCloudExportCmd(unittest.TestCase):
    """Tests for _cmd_cloud_export."""

    def setUp(self) -> None:
        _register_conformant(_CONFORMANT_TENANT)

    def tearDown(self) -> None:
        _clear_tenant(_CONFORMANT_TENANT)

    # ------------------------------------------------------------------
    def test_known_tenant_exits_zero(self) -> None:
        """Export for a known tenant should return exit code 0."""
        args = _make_args(tenant_id=_CONFORMANT_TENANT)
        rc, _ = _capture_stdout(_cmd_cloud_export, args, quiet=False)
        self.assertEqual(rc, 0)

    def test_output_path_writes_json_file(self) -> None:
        """--output PATH should write a JSON file containing 'tenant_id'."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            args = _make_args(tenant_id=_CONFORMANT_TENANT, output_path=tmp_path)
            rc, _ = _capture_stdout(_cmd_cloud_export, args, quiet=False)
            self.assertEqual(rc, 0)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("tenant_id", data)
            self.assertEqual(data["tenant_id"], _CONFORMANT_TENANT)
        finally:
            os.unlink(tmp_path)

    def test_unknown_tenant_exits_one(self) -> None:
        """Export for an unknown tenant should return exit code 1."""
        args = _make_args(tenant_id="no-such-tenant-export-xyz")
        rc, _ = _capture_stdout(_cmd_cloud_export, args, quiet=False)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
