"""
tests/serving/test_new_endpoints.py

Unit tests for the two new server endpoints added in the monitoring dashboard:
  - GET /sys-stats
  - GET /debug-info
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import squish.server as _srv


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_srv.app, raise_server_exceptions=False)


# ── /sys-stats ────────────────────────────────────────────────────────────────


class TestSysStats:
    """GET /sys-stats — stdlib-only system resource metrics."""

    def test_status_200(self, client: TestClient) -> None:
        r = client.get("/sys-stats")
        assert r.status_code == 200

    def test_response_is_json(self, client: TestClient) -> None:
        r = client.get("/sys-stats")
        body = r.json()
        assert isinstance(body, dict)

    def test_contains_load_avg(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert "load_avg" in body
        assert isinstance(body["load_avg"], list)
        assert len(body["load_avg"]) == 3

    def test_load_avg_values_are_floats(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        for v in body["load_avg"]:
            assert isinstance(v, (int, float))

    def test_contains_process_rss_mb(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert "process_rss_mb" in body
        assert isinstance(body["process_rss_mb"], (int, float))

    def test_rss_mb_non_negative(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert body["process_rss_mb"] >= 0

    def test_contains_disk_fields(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert "disk_used_pct" in body
        assert "disk_free_gb" in body
        assert "disk_total_gb" in body

    def test_disk_used_pct_range(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert 0 <= body["disk_used_pct"] <= 100

    def test_disk_free_gb_non_negative(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert body["disk_free_gb"] >= 0

    def test_disk_total_ge_free(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert body["disk_total_gb"] >= body["disk_free_gb"]

    def test_contains_pid(self, client: TestClient) -> None:
        body = client.get("/sys-stats").json()
        assert "pid" in body
        assert isinstance(body["pid"], int)
        assert body["pid"] > 0

    def test_load_avg_fallback_on_oserror(self, client: TestClient) -> None:
        """If os.getloadavg raises OSError, load_avg should default to [0.0, 0.0, 0.0]."""
        import os
        with patch.object(os, "getloadavg", side_effect=OSError("not supported")):
            body = client.get("/sys-stats").json()
        assert body["load_avg"] == [0.0, 0.0, 0.0]

    def test_rss_fallback_on_exception(self, client: TestClient) -> None:
        """If resource.getrusage raises, process_rss_mb should default to 0.0."""
        import resource
        with patch.object(resource, "getrusage", side_effect=RuntimeError("fail")):
            body = client.get("/sys-stats").json()
        assert body["process_rss_mb"] == 0.0

    def test_disk_fallback_on_exception(self, client: TestClient) -> None:
        """If shutil.disk_usage raises, disk fields should default to 0.0."""
        import shutil
        with patch.object(shutil, "disk_usage", side_effect=OSError("fail")):
            body = client.get("/sys-stats").json()
        assert body["disk_used_pct"] == 0.0
        assert body["disk_free_gb"] == 0.0
        assert body["disk_total_gb"] == 0.0

    def test_macos_rss_conversion(self, client: TestClient) -> None:
        """On macOS, ru_maxrss is in bytes; verify the /1024/1024 path is covered."""
        import resource
        import os
        mock_rusage = resource.struct_rusage((0,) * 16)
        # Set ru_maxrss (index 2) to 100 MB in bytes (macOS style)
        rss_bytes = 100 * 1024 * 1024
        mock_rusage_vals = list(mock_rusage)
        mock_rusage_vals[2] = rss_bytes
        fake_ru = resource.struct_rusage(mock_rusage_vals)

        with patch.object(resource, "getrusage", return_value=fake_ru):
            with patch.object(sys, "platform", "darwin"):
                body = client.get("/sys-stats").json()
        # 100 MB → 100.0 (bytes / 1024 / 1024)
        assert body["process_rss_mb"] == pytest.approx(100.0, abs=1.0)

    def test_linux_rss_conversion(self, client: TestClient) -> None:
        """On Linux, ru_maxrss is in KB; verify the /1024 path is covered."""
        import resource
        mock_rusage = resource.struct_rusage((0,) * 16)
        rss_kb = 50 * 1024  # 50 MB in KB
        mock_rusage_vals = list(mock_rusage)
        mock_rusage_vals[2] = rss_kb
        fake_ru = resource.struct_rusage(mock_rusage_vals)

        with patch.object(resource, "getrusage", return_value=fake_ru):
            with patch.object(sys, "platform", "linux"):
                body = client.get("/sys-stats").json()
        # 50 MB → 50.0 (KB / 1024)
        assert body["process_rss_mb"] == pytest.approx(50.0, abs=1.0)


# ── /debug-info ───────────────────────────────────────────────────────────────


class TestDebugInfo:
    """GET /debug-info — server config & CLI flags for observability."""

    def test_status_200(self, client: TestClient) -> None:
        r = client.get("/debug-info")
        assert r.status_code == 200

    def test_response_is_json(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert isinstance(body, dict)

    def test_contains_cli_flags(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert "cli_flags" in body
        assert isinstance(body["cli_flags"], dict)

    def test_contains_python_version(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert "python_version" in body
        assert isinstance(body["python_version"], str)
        assert len(body["python_version"]) > 0

    def test_python_version_matches_runtime(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert sys.version in body["python_version"]

    def test_contains_platform(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert "platform" in body
        assert isinstance(body["platform"], str)

    def test_platform_matches_runtime(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert body["platform"] == sys.platform

    def test_contains_pid(self, client: TestClient) -> None:
        body = client.get("/debug-info").json()
        assert "pid" in body
        assert isinstance(body["pid"], int)
        assert body["pid"] > 0

    def test_cli_flags_reflect_server_args(self, client: TestClient) -> None:
        """Populated _server_args dict should appear in the response."""
        original = dict(_srv._server_args)
        try:
            _srv._server_args["test_key"] = "test_value"
            body = client.get("/debug-info").json()
            assert body["cli_flags"].get("test_key") == "test_value"
        finally:
            _srv._server_args.clear()
            _srv._server_args.update(original)

    def test_empty_server_args_returns_empty_dict(self, client: TestClient) -> None:
        original = dict(_srv._server_args)
        try:
            _srv._server_args.clear()
            body = client.get("/debug-info").json()
            assert body["cli_flags"] == {}
        finally:
            _srv._server_args.clear()
            _srv._server_args.update(original)


# ── _server_args module-level variable ───────────────────────────────────────


class TestServerArgsGlobal:
    """Verify the _server_args module-level dict is accessible and mutable."""

    def test_server_args_exists(self) -> None:
        assert hasattr(_srv, "_server_args")

    def test_server_args_is_dict(self) -> None:
        assert isinstance(_srv._server_args, dict)

    def test_server_args_mutable(self) -> None:
        original = dict(_srv._server_args)
        try:
            _srv._server_args["_test"] = "val"
            assert _srv._server_args["_test"] == "val"
        finally:
            _srv._server_args.clear()
            _srv._server_args.update(original)
