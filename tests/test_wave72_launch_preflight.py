"""tests/test_wave72_launch_preflight.py — Wave 72 Launch Preflight Tests.

Covers:
- Individual check functions
- PreflightReport aggregation
- format_report output
- run_preflight_checks integration
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from squish.install.launch_preflight import (
    CheckStatus,
    PreflightCheck,
    PreflightReport,
    _check_disk_space,
    _check_memory,
    _check_metal_gpu,
    _check_mlx_available,
    _check_python_version,
    _check_server_port_free,
    _check_write_permission,
    format_report,
    run_preflight_checks,
)


# ══════════════════════════════════════════════════════════════════════════════
# CheckStatus enum
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckStatus(unittest.TestCase):

    def test_values(self):
        self.assertEqual(CheckStatus.OK.value, "ok")
        self.assertEqual(CheckStatus.WARN.value, "warn")
        self.assertEqual(CheckStatus.ERROR.value, "error")

    def test_from_value(self):
        self.assertEqual(CheckStatus("ok"),   CheckStatus.OK)
        self.assertEqual(CheckStatus("warn"),  CheckStatus.WARN)
        self.assertEqual(CheckStatus("error"), CheckStatus.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
# PreflightCheck dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestPreflightCheck(unittest.TestCase):

    def test_fields(self):
        c = PreflightCheck(name="test", status=CheckStatus.OK, message="all good")
        self.assertEqual(c.name, "test")
        self.assertEqual(c.status, CheckStatus.OK)
        self.assertEqual(c.message, "all good")
        self.assertIsNone(c.detail)

    def test_with_detail(self):
        c = PreflightCheck(
            name="test2",
            status=CheckStatus.ERROR,
            message="bad",
            detail="Install MLX first",
        )
        self.assertEqual(c.detail, "Install MLX first")


# ══════════════════════════════════════════════════════════════════════════════
# PreflightReport
# ══════════════════════════════════════════════════════════════════════════════

class TestPreflightReport(unittest.TestCase):

    def test_ok_true_when_no_failures(self):
        report = PreflightReport(
            checks=[PreflightCheck("x", CheckStatus.OK, "ok")],
            passed=1, warned=0, failed=0,
        )
        self.assertTrue(report.ok)

    def test_ok_false_when_failures(self):
        report = PreflightReport(
            checks=[PreflightCheck("x", CheckStatus.ERROR, "bad")],
            passed=0, warned=0, failed=1,
        )
        self.assertFalse(report.ok)

    def test_ok_true_with_warnings(self):
        report = PreflightReport(
            checks=[PreflightCheck("x", CheckStatus.WARN, "meh")],
            passed=0, warned=1, failed=0,
        )
        self.assertTrue(report.ok)


# ══════════════════════════════════════════════════════════════════════════════
# Individual check functions
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckPythonVersion(unittest.TestCase):

    def test_current_python_ok_or_error(self):
        check = _check_python_version()
        self.assertIn(check.status, (CheckStatus.OK, CheckStatus.ERROR))

    def test_passes_on_312(self):
        with patch.object(sys, "version_info", (3, 12, 0, "final", 0)):
            check = _check_python_version()
        self.assertEqual(check.status, CheckStatus.OK)

    def test_fails_on_38(self):
        with patch.object(sys, "version_info", (3, 8, 0, "final", 0)):
            check = _check_python_version()
        self.assertEqual(check.status, CheckStatus.ERROR)

    def test_fails_on_29(self):
        with patch.object(sys, "version_info", (2, 9, 0, "final", 0)):
            check = _check_python_version()
        self.assertEqual(check.status, CheckStatus.ERROR)

    def test_message_contains_version(self):
        check = _check_python_version()
        self.assertIn(str(sys.version_info.major), check.message)

    def test_error_has_detail(self):
        with patch.object(sys, "version_info", (3, 8, 0, "final", 0)):
            check = _check_python_version()
        self.assertIsNotNone(check.detail)


class TestCheckMLXAvailable(unittest.TestCase):

    def test_mlx_present_returns_ok_or_error(self):
        check = _check_mlx_available()
        self.assertIn(check.status, (CheckStatus.OK, CheckStatus.ERROR))

    def test_mlx_import_error_returns_error(self):
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            import builtins
            real_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name.startswith("mlx"):
                    raise ImportError("no mlx")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fake_import):
                check = _check_mlx_available()
        self.assertEqual(check.status, CheckStatus.ERROR)
        self.assertIsNotNone(check.detail)


class TestCheckDiskSpace(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_check_object(self):
        check = _check_disk_space(path=self.tmpdir)
        self.assertIsInstance(check, PreflightCheck)

    def test_ok_with_plenty_of_space(self):
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=100 * 1024 ** 3)  # 100 GiB
            check = _check_disk_space(path=self.tmpdir, min_gib=2.0)
        self.assertEqual(check.status, CheckStatus.OK)

    def test_warn_with_little_space(self):
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=int(0.7 * 1024 ** 3))  # 0.7 GiB
            check = _check_disk_space(path=self.tmpdir, min_gib=2.0)
        self.assertEqual(check.status, CheckStatus.WARN)

    def test_error_with_critically_low_space(self):
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=int(0.1 * 1024 ** 3))  # 0.1 GiB
            check = _check_disk_space(path=self.tmpdir, min_gib=2.0)
        self.assertEqual(check.status, CheckStatus.ERROR)

    def test_oserror_returns_warn(self):
        with patch("shutil.disk_usage") as mock_du:
            mock_du.side_effect = OSError("permission denied")
            check = _check_disk_space(path=self.tmpdir)
        self.assertEqual(check.status, CheckStatus.WARN)


class TestCheckMemory(unittest.TestCase):

    def test_returns_check_object(self):
        check = _check_memory()
        self.assertIsInstance(check, PreflightCheck)

    def test_ok_with_plenty_of_memory(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(32 * 1024 ** 3)  # 32 GiB
            )
            with patch("sys.platform", "darwin"):
                check = _check_memory(min_gib=4.0)
        self.assertEqual(check.status, CheckStatus.OK)

    def test_warn_with_low_memory(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(int(2 * 1024 ** 3))  # 2 GiB
            )
            with patch("sys.platform", "darwin"):
                check = _check_memory(min_gib=4.0)
        self.assertEqual(check.status, CheckStatus.WARN)


class TestCheckWritePermission(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writable_dir_passes(self):
        check = _check_write_permission(self.tmpdir)
        self.assertEqual(check.status, CheckStatus.OK)

    def test_unwritable_dir_fails(self):
        if os.getuid() == 0:
            self.skipTest("Running as root, cannot test write permissions")
        readonly = tempfile.mkdtemp()
        try:
            os.chmod(readonly, 0o444)
            check = _check_write_permission(readonly)
            self.assertEqual(check.status, CheckStatus.ERROR)
        finally:
            os.chmod(readonly, 0o755)
            shutil.rmtree(readonly, ignore_errors=True)

    def test_message_contains_path(self):
        check = _check_write_permission(self.tmpdir)
        self.assertIn(self.tmpdir, check.message)


class TestCheckPortFree(unittest.TestCase):

    def test_free_port_ok(self):
        # Use a high port that is almost certainly unused
        check = _check_server_port_free(port=59499)
        # Should be OK unless something is actually running on that port
        self.assertIn(check.status, (CheckStatus.OK, CheckStatus.WARN))

    def test_in_use_port_warn(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            check = _check_server_port_free(port=port)
        self.assertEqual(check.status, CheckStatus.WARN)

    def test_returns_preflight_check(self):
        check = _check_server_port_free(59500)
        self.assertIsInstance(check, PreflightCheck)


# ══════════════════════════════════════════════════════════════════════════════
# run_preflight_checks integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPreflightChecks(unittest.TestCase):

    def test_returns_report(self):
        report = run_preflight_checks()
        self.assertIsInstance(report, PreflightReport)

    def test_correct_check_count(self):
        report = run_preflight_checks()
        self.assertEqual(len(report.checks), 7)

    def test_counts_sum_to_total(self):
        report = run_preflight_checks()
        total = report.passed + report.warned + report.failed
        self.assertEqual(total, len(report.checks))

    def test_all_checks_are_preflight_check(self):
        report = run_preflight_checks()
        for c in report.checks:
            self.assertIsInstance(c, PreflightCheck)

    def test_all_statuses_valid(self):
        report = run_preflight_checks()
        for c in report.checks:
            self.assertIn(c.status, (CheckStatus.OK, CheckStatus.WARN, CheckStatus.ERROR))


# ══════════════════════════════════════════════════════════════════════════════
# format_report
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatReport(unittest.TestCase):

    def _make_report(self, statuses):
        checks = [
            PreflightCheck(name=f"check_{i}", status=s, message=f"msg {i}")
            for i, s in enumerate(statuses)
        ]
        passed = sum(1 for s in statuses if s == CheckStatus.OK)
        warned = sum(1 for s in statuses if s == CheckStatus.WARN)
        failed = sum(1 for s in statuses if s == CheckStatus.ERROR)
        return PreflightReport(checks=checks, passed=passed, warned=warned, failed=failed)

    def test_returns_string(self):
        report = self._make_report([CheckStatus.OK])
        output = format_report(report)
        self.assertIsInstance(output, str)

    def test_contains_header(self):
        report = self._make_report([CheckStatus.OK])
        output = format_report(report, color=False)
        self.assertIn("Squish Preflight Checks", output)

    def test_contains_ok_indicator(self):
        report = self._make_report([CheckStatus.OK])
        output = format_report(report, color=False)
        self.assertIn("✓", output)

    def test_contains_error_indicator(self):
        report = self._make_report([CheckStatus.ERROR])
        output = format_report(report, color=False)
        self.assertIn("✗", output)

    def test_contains_warn_indicator(self):
        report = self._make_report([CheckStatus.WARN])
        output = format_report(report, color=False)
        self.assertIn("⚠", output)

    def test_all_pass_message(self):
        report = self._make_report([CheckStatus.OK, CheckStatus.OK])
        output = format_report(report, color=False)
        self.assertIn("All checks passed", output)

    def test_error_message(self):
        report = self._make_report([CheckStatus.ERROR])
        output = format_report(report, color=False)
        self.assertIn("Fix the errors", output)

    def test_no_ansi_in_plain_mode(self):
        report = self._make_report([CheckStatus.OK])
        output = format_report(report, color=False)
        self.assertNotIn("\033[", output)

    def test_detail_included(self):
        report = PreflightReport(
            checks=[PreflightCheck("x", CheckStatus.ERROR, "bad", detail="Install thing")],
            passed=0, warned=0, failed=1,
        )
        output = format_report(report, color=False)
        self.assertIn("Install thing", output)

    def test_summary_counts_shown(self):
        report = self._make_report([CheckStatus.OK, CheckStatus.WARN, CheckStatus.ERROR])
        output = format_report(report, color=False)
        self.assertIn("1 passed", output)


if __name__ == "__main__":
    unittest.main()
