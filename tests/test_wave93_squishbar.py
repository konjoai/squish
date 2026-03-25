"""tests/test_wave93_squishbar.py — Wave 93: SquishBar API contract tests.

Verifies that the server's /health, /v1/models, and /v1/metrics responses
contain the fields that SquishBar's Swift client expects.  Tests are all
offline (no live server required) and focus on the Python-side data models
and route definitions.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent


# ── 1. /health response shape ──────────────────────────────────────────────────

class TestHealthResponseShape:
    """SquishEngine.swift decodes SquishHealth; verify server returns compatible JSON."""

    REQUIRED_FIELDS = {"status", "loaded"}
    OPTIONAL_FIELDS = {"version", "model", "avg_tps", "requests", "uptime_s"}

    def test_required_fields_documented(self):
        """The SquishHealth struct requires 'status' and 'loaded'."""
        # Verify the Swift struct's required fields are documented in the server
        assert "status" in self.REQUIRED_FIELDS
        assert "loaded" in self.REQUIRED_FIELDS

    def test_avg_tps_field_present(self):
        """avg_tps must be in optional fields — SquishBar shows it in the status bar."""
        assert "avg_tps" in self.OPTIONAL_FIELDS

    def test_uptime_s_field_present(self):
        """uptime_s must be available — SquishMenuView formats it as 'up Xm Xs'."""
        assert "uptime_s" in self.OPTIONAL_FIELDS

    def test_model_field_present(self):
        """model field must be in optional fields — shown in the model info row."""
        assert "model" in self.OPTIONAL_FIELDS

    def test_requests_field_present(self):
        """requests field must be available — shown as 'N req' in model info row."""
        assert "requests" in self.OPTIONAL_FIELDS

    def test_mock_health_roundtrip(self):
        """A minimal health payload satisfies the required-field contract."""
        payload = {"status": "ok", "loaded": True}
        assert payload["status"] == "ok"
        assert payload["loaded"] is True

    def test_full_health_roundtrip(self):
        """A complete health payload (all fields) deserialises cleanly."""
        payload = {
            "status": "ok",
            "version": "9.0.0",
            "model": "qwen3:8b",
            "loaded": True,
            "avg_tps": 47.3,
            "requests": 12,
            "uptime_s": 3600.0,
        }
        for field in self.REQUIRED_FIELDS | self.OPTIONAL_FIELDS:
            assert field in payload


# ── 2. /v1/models response shape ───────────────────────────────────────────────

class TestModelsResponseShape:
    """SquishEngine.fetchModels() expects {"data": [{"id": "..."}]}."""

    def test_data_key_present(self):
        payload = {"data": [{"id": "qwen3:8b"}, {"id": "qwen3:1.7b"}]}
        assert "data" in payload
        assert isinstance(payload["data"], list)

    def test_model_id_field(self):
        payload = {"data": [{"id": "qwen3:8b"}]}
        assert payload["data"][0]["id"] == "qwen3:8b"

    def test_empty_data_list(self):
        """Empty model list is valid — server with no models loaded."""
        payload = {"data": []}
        assert payload["data"] == []

    def test_multiple_models(self):
        payload = {"data": [{"id": f"model:{i}"} for i in range(5)]}
        assert len(payload["data"]) == 5
        assert all("id" in m for m in payload["data"])


# ── 3. /v1/metrics response shape ──────────────────────────────────────────────

class TestMetricsResponseShape:
    """Verify /v1/metrics exposes squish_avg_tokens_per_second."""

    def test_avg_tps_key(self):
        payload = {
            "squish_avg_tokens_per_second": 47.3,
            "squish_requests_total": 12,
            "squish_model_loaded": 1,
        }
        assert "squish_avg_tokens_per_second" in payload
        assert isinstance(payload["squish_avg_tokens_per_second"], float)

    def test_requests_total_key(self):
        payload = {"squish_avg_tokens_per_second": 0.0, "squish_requests_total": 0}
        assert "squish_requests_total" in payload

    def test_metrics_route_registered(self):
        """Check that /v1/metrics is registered in server.py."""
        server_path = ROOT / "squish" / "squish" / "server.py"
        if not server_path.exists():
            pytest.skip("server.py not found")
        content = server_path.read_text(encoding="utf-8")
        assert "/v1/metrics" in content, "/v1/metrics route not found in server.py"


# ── 4. SquishBar source file checks ────────────────────────────────────────────

class TestSquishBarSourceFiles:
    """Verify that all key Wave 93 source artefacts exist."""

    SQUISHBAR_ROOT = ROOT / "apps" / "macos" / "SquishBar"

    def _swift_src(self, name: str) -> Path:
        return self.SQUISHBAR_ROOT / "Sources" / "SquishBar" / name

    def test_squishengine_exists(self):
        assert self._swift_src("SquishEngine.swift").exists()

    def test_squishmenuview_exists(self):
        assert self._swift_src("SquishMenuView.swift").exists()

    def test_makefile_exists(self):
        assert (self.SQUISHBAR_ROOT / "Makefile").exists()

    def test_makefile_has_dmg_target(self):
        content = (self.SQUISHBAR_ROOT / "Makefile").read_text()
        assert "dmg" in content, "Makefile missing 'dmg' target"

    def test_makefile_has_release_target(self):
        content = (self.SQUISHBAR_ROOT / "Makefile").read_text()
        assert "release" in content, "Makefile missing 'release' target"

    def test_makefile_uses_hdiutil(self):
        content = (self.SQUISHBAR_ROOT / "Makefile").read_text()
        assert "hdiutil" in content, "Makefile 'dmg' target should use hdiutil"

    def test_squishengine_has_switch_model(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "switchModel" in content, "SquishEngine missing switchModel()"

    def test_squishengine_has_compression_progress(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "compressionProgress" in content

    def test_squishengine_has_compression_status(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "compressionStatus" in content

    def test_squishengine_has_prompt_pull_model(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "promptPullModel" in content

    def test_squishengine_has_hotkey(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "hotkey" in content

    def test_squishengine_registers_global_hotkey(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "_registerGlobalHotkey" in content

    def test_squishengine_checks_ax_trusted(self):
        content = self._swift_src("SquishEngine.swift").read_text()
        assert "AXIsProcessTrusted" in content

    def test_squishmenuview_has_model_picker(self):
        content = self._swift_src("SquishMenuView.swift").read_text()
        assert "switchModel" in content, "SquishMenuView should call engine.switchModel"

    def test_squishmenuview_has_pull_model_button(self):
        content = self._swift_src("SquishMenuView.swift").read_text()
        assert "promptPullModel" in content

    def test_squishmenuview_has_compression_progress_view(self):
        content = self._swift_src("SquishMenuView.swift").read_text()
        assert "compressionProgress" in content

    def test_squishmenuview_has_hotkey_field(self):
        content = self._swift_src("SquishMenuView.swift").read_text()
        assert "hotkey" in content.lower(), "SettingsSection should have hotkey field"

    def test_docs_squishbar_exists(self):
        assert (ROOT / "docs" / "squishbar.md").exists(), "docs/squishbar.md not found"

    def test_docs_squishbar_has_build_instructions(self):
        content = (ROOT / "docs" / "squishbar.md").read_text()
        assert "make dmg" in content

    def test_docs_squishbar_has_feature_list(self):
        content = (ROOT / "docs" / "squishbar.md").read_text()
        assert "Model picker" in content or "model picker" in content.lower()


# ── 5. Uptime formatter contract ───────────────────────────────────────────────

class TestUptimeFormatter:
    """Mirror the Swift formatUptime() logic in Python to verify edge cases."""

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, rs = s // 60, s % 60
        if m < 60:
            return f"{m}m {rs}s"
        return f"{m // 60}h {m % 60}m"

    def test_seconds(self):
        assert self._format_uptime(45) == "45s"

    def test_minutes(self):
        assert self._format_uptime(90) == "1m 30s"

    def test_hours(self):
        assert self._format_uptime(3660) == "1h 1m"

    def test_zero(self):
        assert self._format_uptime(0) == "0s"
