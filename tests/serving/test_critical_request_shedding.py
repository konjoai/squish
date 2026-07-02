"""
tests/serving/test_critical_request_shedding.py

Kill-test evidence for Phase 4 of the memory-governor eviction sprint:
proves ``_MemoryPressureShedMiddleware`` rejects NEW requests with HTTP 503
while the memory governor reports LEVEL_CRITICAL, exempts pure-observability
endpoints, leaves WARNING/URGENT/NORMAL traffic untouched, and never aborts
a request that was already in flight when pressure became CRITICAL.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import squish.server as _srv
from squish.serving.memory_governor import LEVEL_CRITICAL, LEVEL_NORMAL, LEVEL_URGENT, LEVEL_WARNING


@pytest.fixture(autouse=True)
def _reset_governor():
    orig = _srv._memory_governor
    _srv._memory_governor = None
    yield
    _srv._memory_governor = orig


@pytest.fixture
def client():
    return TestClient(_srv.app, raise_server_exceptions=False)


def _mock_governor(level):
    return MagicMock(pressure_level=level)


# ── Shedding behavior on the real app ────────────────────────────────────────


class TestCriticalSheds:
    def test_critical_rejects_chat_completions_with_503(self, client):
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert r.status_code == 503
        assert "critical memory pressure" in r.json()["detail"].lower()

    def test_critical_rejects_completions_with_503(self, client):
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.post("/v1/completions", json={"model": "x", "prompt": "hi"})

        assert r.status_code == 503
        assert "critical memory pressure" in r.json()["detail"].lower()

    def test_critical_rejects_embeddings_with_503(self, client):
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.post("/v1/embeddings", json={"model": "x", "input": "hi"})

        assert r.status_code == 503
        assert "critical memory pressure" in r.json()["detail"].lower()

    def test_shed_response_is_distinct_from_no_model_503(self, client):
        """The shed message must be identifiable/distinct from the pre-existing
        'model not loaded' 503 — callers need to tell "try again shortly" apart
        from "this server was never configured with a model"."""
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)
        shed = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        ).json()["detail"]

        _srv._memory_governor = None
        no_model = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        ).json()["detail"]

        assert shed != no_model


class TestExemptEndpoints:
    def test_health_stays_reachable_under_critical(self, client):
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.get("/health")

        assert r.status_code == 200

    def test_metrics_stays_reachable_under_critical(self, client):
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.get("/v1/metrics")

        assert r.status_code == 200


class TestNonCriticalPressureNotShed:
    @pytest.mark.parametrize("level", [LEVEL_NORMAL, LEVEL_WARNING, LEVEL_URGENT])
    def test_only_critical_sheds_requests(self, client, level):
        """WARNING/URGENT degrade cache budgets and context size (Phases 2-3)
        but must NOT reject requests outright — only CRITICAL does."""
        _srv._memory_governor = _mock_governor(level)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        # Falls through to the ordinary (no-model) 503, not the pressure-shed one.
        assert r.status_code == 503
        assert "critical memory pressure" not in r.json()["detail"].lower()

    def test_no_governor_configured_is_never_shed(self, client):
        _srv._memory_governor = None

        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert "critical memory pressure" not in r.json()["detail"].lower()


class TestCorsHeadersOnShedResponse:
    def test_shed_503_still_carries_cors_headers(self, client):
        """Registered before CORSMiddleware so CORS still wraps (and adds
        headers to) the 503 — otherwise browser clients would see an opaque
        network error instead of a readable 503 body during an incident."""
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Origin": "https://example.com"},
        )

        assert r.status_code == 503
        # allow_credentials=True makes CORSMiddleware echo the specific origin
        # rather than "*" — presence of the header at all is what proves CORS
        # still wrapped this response despite the middleware short-circuiting.
        assert r.headers.get("access-control-allow-origin") == "https://example.com"


# ── In-flight requests are never aborted ─────────────────────────────────────


class TestInFlightRequestsSurviveCriticalTransition:
    def test_request_already_past_middleware_completes_normally(self):
        """Build an isolated app with the same middleware and a slow route.
        Flip pressure to CRITICAL *after* the request has already been
        admitted (i.e. mid-generation) and confirm it still completes with
        its normal 200 — CRITICAL sheds NEW work, not in-flight work."""
        governor = _mock_governor(LEVEL_NORMAL)
        _srv._memory_governor = governor

        app = FastAPI()
        app.add_middleware(_srv._MemoryPressureShedMiddleware)
        app.add_middleware(CORSMiddleware, allow_origins=["*"])

        admitted = threading.Event()

        @app.get("/slow")
        def slow():
            admitted.set()
            time.sleep(0.2)
            return {"ok": True}

        def _flip_to_critical_once_admitted():
            admitted.wait(timeout=2)
            governor.pressure_level = LEVEL_CRITICAL

        flipper = threading.Thread(target=_flip_to_critical_once_admitted)
        flipper.start()

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/slow")
        flipper.join()

        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_next_new_request_after_flip_is_shed(self):
        """Sanity check for the above: once pressure IS critical, a brand-new
        request (not already in flight) really does get rejected."""
        governor = _mock_governor(LEVEL_CRITICAL)
        _srv._memory_governor = governor

        app = FastAPI()
        app.add_middleware(_srv._MemoryPressureShedMiddleware)

        @app.get("/slow")
        def slow():
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/slow")

        assert r.status_code == 503
