"""tests/test_wave64a_trace_endpoint.py

Unit tests for the /v1/trace GET and DELETE endpoints added in wave 64.

These tests run entirely without a live model — they exercise the telemetry
infrastructure (span recording, Chrome export format, clear) through the
FastAPI TestClient.

Coverage targets
────────────────
 • GET  /v1/trace            — default JSON summary (no spans, with spans)
 • GET  /v1/trace?format=chrome — Chrome DevTools JSON structure
 • DELETE /v1/trace          — clears tracer; subsequent GET returns 0 spans
 • Auth enforcement          — 401 when API key is set and missing/wrong
 • Telemetry unavailable     — 503 when _TELEMETRY_AVAILABLE is False
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import squish.server as _srv
import squish.telemetry as _tel


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """Ensure each test starts with a clean tracer and known tracing state."""
    orig_enabled = _tel.TRACING_ENABLED
    _tel.reset_tracer()
    _tel.configure_tracing(True)
    yield
    _tel.reset_tracer()
    _tel.configure_tracing(orig_enabled)


@pytest.fixture()
def client():
    """Return a TestClient with no API-key auth configured."""
    from fastapi.testclient import TestClient

    orig_key = _srv._API_KEY
    _srv._API_KEY = None
    c = TestClient(_srv.app, raise_server_exceptions=False)
    yield c
    _srv._API_KEY = orig_key


@pytest.fixture()
def client_with_key():
    """Return a TestClient with API key 'test-secret' configured in server."""
    from fastapi.testclient import TestClient

    orig_key = _srv._API_KEY
    _srv._API_KEY = "test-secret"
    c = TestClient(_srv.app, raise_server_exceptions=False)
    yield c
    _srv._API_KEY = orig_key


# ── Helpers ───────────────────────────────────────────────────────────────────


def _record_span(name: str = "test.op", duration_s: float = 0.01) -> None:
    """Record a finished span in the global tracer."""
    tracer = _tel.get_tracer()
    span = tracer.start_span(name)
    import time
    time.sleep(duration_s)
    span.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# GET /v1/trace  — default JSON summary
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetTraceDefault:
    def test_returns_200(self, client):
        r = client.get("/v1/trace")
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.get("/v1/trace")
        body = r.json()
        assert "total_spans" in body
        assert "slowest_spans" in body
        assert "tracing_enabled" in body
        assert "hint" in body

    def test_no_spans_returns_empty_list(self, client):
        r = client.get("/v1/trace")
        body = r.json()
        assert body["total_spans"] == 0
        assert body["slowest_spans"] == []

    def test_records_spans_are_returned(self, client):
        _record_span("gen.prefill", duration_s=0.005)
        _record_span("gen.compress", duration_s=0.002)
        r = client.get("/v1/trace")
        body = r.json()
        assert body["total_spans"] == 2
        names = {s["name"] for s in body["slowest_spans"]}
        assert "gen.prefill" in names
        assert "gen.compress" in names

    def test_slowest_spans_ordered_by_duration(self, client):
        """The slowest span should appear first in the list."""
        _record_span("fast.op", duration_s=0.001)
        _record_span("slow.op", duration_s=0.020)
        r = client.get("/v1/trace")
        spans = r.json()["slowest_spans"]
        assert len(spans) >= 2
        durations = [s["duration_ms"] for s in spans if s["duration_ms"] is not None]
        assert durations == sorted(durations, reverse=True)

    def test_tracing_enabled_reflects_configure_tracing(self, client):
        _tel.configure_tracing(True)
        r = client.get("/v1/trace")
        assert r.json()["tracing_enabled"] is True

    def test_hint_is_non_empty_string(self, client):
        r = client.get("/v1/trace")
        hint = r.json()["hint"]
        assert isinstance(hint, str) and len(hint) > 0

    def test_span_dict_has_standard_keys(self, client):
        _record_span("test.span")
        body = client.get("/v1/trace").json()
        span = body["slowest_spans"][0]
        for key in ("id", "name", "duration_ms", "status"):
            assert key in span, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# GET /v1/trace?format=chrome
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetTraceChrome:
    def test_returns_200(self, client):
        r = client.get("/v1/trace?format=chrome")
        assert r.status_code == 200

    def test_chrome_format_has_trace_events_key(self, client):
        r = client.get("/v1/trace?format=chrome")
        body = r.json()
        assert "traceEvents" in body

    def test_chrome_format_has_metadata(self, client):
        r = client.get("/v1/trace?format=chrome")
        body = r.json()
        assert "metadata" in body

    def test_chrome_format_has_display_time_unit(self, client):
        r = client.get("/v1/trace?format=chrome")
        body = r.json()
        assert "displayTimeUnit" in body
        assert body["displayTimeUnit"] == "ms"

    def test_chrome_events_empty_when_no_spans(self, client):
        r = client.get("/v1/trace?format=chrome")
        assert r.json()["traceEvents"] == []

    def test_chrome_events_populated_after_span(self, client):
        _record_span("gen.prefill")
        r = client.get("/v1/trace?format=chrome")
        events = r.json()["traceEvents"]
        assert len(events) == 1

    def test_chrome_event_has_required_fields(self, client):
        _record_span("gen.prefill")
        event = client.get("/v1/trace?format=chrome").json()["traceEvents"][0]
        for key in ("name", "ph", "ts", "dur", "pid", "tid"):
            assert key in event, f"Chrome event missing key: {key}"

    def test_chrome_event_ph_is_x(self, client):
        """Complete events must use ph='X' per Chrome trace format spec."""
        _record_span("foo")
        event = client.get("/v1/trace?format=chrome").json()["traceEvents"][0]
        assert event["ph"] == "X"

    def test_chrome_multiple_spans(self, client):
        for name in ("gen.compress", "gen.prefill", "gen.prefix_cache"):
            _record_span(name, duration_s=0.001)
        events = client.get("/v1/trace?format=chrome").json()["traceEvents"]
        names = {e["name"] for e in events}
        assert names == {"gen.compress", "gen.prefill", "gen.prefix_cache"}


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /v1/trace
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeleteTrace:
    def test_returns_200(self, client):
        r = client.delete("/v1/trace")
        assert r.status_code == 200

    def test_ok_field_true(self, client):
        r = client.delete("/v1/trace")
        assert r.json()["ok"] is True

    def test_clears_existing_spans(self, client):
        _record_span("gen.prefill")
        assert client.get("/v1/trace").json()["total_spans"] == 1
        client.delete("/v1/trace")
        assert client.get("/v1/trace").json()["total_spans"] == 0

    def test_delete_idempotent(self, client):
        """Deleting an already-empty tracer should not error."""
        r1 = client.delete("/v1/trace")
        r2 = client.delete("/v1/trace")
        assert r1.json()["ok"] is True
        assert r2.json()["ok"] is True

    def test_chrome_events_empty_after_delete(self, client):
        _record_span("gen.decode")
        client.delete("/v1/trace")
        events = client.get("/v1/trace?format=chrome").json()["traceEvents"]
        assert events == []


# ═══════════════════════════════════════════════════════════════════════════════
# Auth enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceAuth:
    def test_get_no_key_required_passes(self, client):
        r = client.get("/v1/trace")
        assert r.status_code == 200

    def test_get_wrong_key_returns_401(self, client_with_key):
        r = client_with_key.get(
            "/v1/trace",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_get_correct_key_passes(self, client_with_key):
        r = client_with_key.get(
            "/v1/trace",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert r.status_code == 200

    def test_delete_wrong_key_returns_401(self, client_with_key):
        r = client_with_key.delete(
            "/v1/trace",
            headers={"Authorization": "Bearer bad"},
        )
        assert r.status_code == 401

    def test_delete_correct_key_passes(self, client_with_key):
        r = client_with_key.delete(
            "/v1/trace",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert r.status_code == 200

    def test_get_no_auth_header_returns_401_when_key_required(self, client_with_key):
        r = client_with_key.get("/v1/trace")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry unavailable path (_TELEMETRY_AVAILABLE = False)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceWhenTelemetryUnavailable:
    def test_get_returns_503(self, client):
        with patch.object(_srv, "_TELEMETRY_AVAILABLE", False):
            r = client.get("/v1/trace")
        assert r.status_code == 503

    def test_get_error_message(self, client):
        with patch.object(_srv, "_TELEMETRY_AVAILABLE", False):
            body = client.get("/v1/trace").json()
        assert "error" in body

    def test_delete_returns_ok_false(self, client):
        with patch.object(_srv, "_TELEMETRY_AVAILABLE", False):
            body = client.delete("/v1/trace").json()
        assert body["ok"] is False
        assert "error" in body
