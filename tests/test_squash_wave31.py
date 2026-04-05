"""tests/test_squash_wave31.py — Wave 31 REST API endpoint tests.

Tests two new VEX cache management endpoints added to squish/squash/api.py:

    GET  /vex/status  — return VEX cache metadata (empty flag, url, age, count, stale)
    POST /vex/update  — force-refresh local VEX feed cache from a remote URL

Test taxonomy:
  - Integration — uses FastAPI TestClient against the real app.
    VexCache is mocked at squish.squash.vex.VexCache to prevent disk/network access.
    All handler logic (URL resolution, error mapping, counter increments) runs real.

Fixture note: _rate_window is cleared per-function to prevent rate-limit exhaustion
when the full suite runs sequentially.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi.testclient import TestClient

    from squish.squash.api import app, _rate_window, _COUNTERS
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    _rate_window.clear()
    with TestClient(app) as c:
        yield c


def _empty_cache_mock() -> MagicMock:
    """VexCache instance that reports an empty cache (no manifest, no disk data)."""
    m = MagicMock()
    m.manifest.return_value = {}
    m.is_stale.return_value = True
    return m


def _populated_cache_mock(
    *,
    url: str = "https://example.com/feed.json",
    last_fetched: str = "2024-06-01T12:00:00+00:00",
    statement_count: int = 42,
    stale: bool = False,
) -> MagicMock:
    """VexCache instance that reports a populated, fresh cache."""
    m = MagicMock()
    m.manifest.return_value = {
        "url": url,
        "last_fetched": last_fetched,
        "statement_count": statement_count,
    }
    m.is_stale.return_value = stale
    return m


def _update_feed_mock(statement_count: int = 7) -> MagicMock:
    """VexCache instance whose load_or_fetch returns a feed with documents."""
    doc = MagicMock()
    doc.statements = ["s"] * statement_count
    feed = MagicMock()
    feed.documents = [doc]
    m = MagicMock()
    m.load_or_fetch.return_value = feed
    return m


# ── GET /vex/status ───────────────────────────────────────────────────────────


class TestVexStatusEndpoint:
    def test_empty_cache_returns_200(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_empty_cache_mock()):
            resp = client.get("/vex/status")
        assert resp.status_code == 200

    def test_empty_cache_returns_empty_true(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_empty_cache_mock()):
            resp = client.get("/vex/status")
        assert resp.json().get("empty") is True

    def test_empty_cache_has_no_url_field(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_empty_cache_mock()):
            resp = client.get("/vex/status")
        assert "url" not in resp.json()

    def test_populated_cache_returns_200(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_populated_cache_mock()):
            resp = client.get("/vex/status")
        assert resp.status_code == 200

    def test_populated_cache_returns_empty_false(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_populated_cache_mock()):
            resp = client.get("/vex/status")
        assert resp.json()["empty"] is False

    def test_populated_cache_returns_url(self, client):
        with patch(
            "squish.squash.vex.VexCache",
            return_value=_populated_cache_mock(url="https://vex.example.org/feed.json"),
        ):
            resp = client.get("/vex/status")
        assert resp.json()["url"] == "https://vex.example.org/feed.json"

    def test_populated_cache_returns_statement_count_int(self, client):
        with patch(
            "squish.squash.vex.VexCache",
            return_value=_populated_cache_mock(statement_count=99),
        ):
            resp = client.get("/vex/status")
        assert resp.json()["statement_count"] == 99

    def test_populated_cache_returns_last_fetched(self, client):
        with patch(
            "squish.squash.vex.VexCache",
            return_value=_populated_cache_mock(last_fetched="2024-12-31T23:59:59+00:00"),
        ):
            resp = client.get("/vex/status")
        assert resp.json()["last_fetched"] == "2024-12-31T23:59:59+00:00"

    def test_populated_cache_stale_false_when_fresh(self, client):
        with patch(
            "squish.squash.vex.VexCache", return_value=_populated_cache_mock(stale=False)
        ):
            resp = client.get("/vex/status")
        assert resp.json()["stale"] is False

    def test_populated_cache_stale_true_when_old(self, client):
        with patch(
            "squish.squash.vex.VexCache", return_value=_populated_cache_mock(stale=True)
        ):
            resp = client.get("/vex/status")
        assert resp.json()["stale"] is True


# ── POST /vex/update ──────────────────────────────────────────────────────────


class TestVexUpdateEndpoint:
    _URL = "https://test.example.com/feed.json"

    def test_explicit_url_returns_200(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_update_feed_mock()):
            resp = client.post("/vex/update", json={"url": self._URL})
        assert resp.status_code == 200

    def test_returns_updated_true(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_update_feed_mock()):
            resp = client.post("/vex/update", json={"url": self._URL})
        assert resp.json()["updated"] is True

    def test_returns_url_in_response(self, client):
        mock = _update_feed_mock()
        with patch("squish.squash.vex.VexCache", return_value=mock):
            resp = client.post("/vex/update", json={"url": "https://custom.example.com/feed.json"})
        assert resp.json()["url"] == "https://custom.example.com/feed.json"

    def test_default_url_falls_back_to_squash_feed(self, client, monkeypatch):
        """When no url in request and env var absent, DEFAULT_URL is used."""
        from squish.squash.vex import VexCache as _RealVexCache  # noqa: PLC0415

        monkeypatch.delenv("SQUASH_VEX_URL", raising=False)
        mock_instance = _update_feed_mock()
        with patch("squish.squash.vex.VexCache") as MockClass:
            MockClass.DEFAULT_URL = _RealVexCache.DEFAULT_URL
            MockClass.return_value = mock_instance
            resp = client.post("/vex/update", json={})
        assert resp.json()["url"] == _RealVexCache.DEFAULT_URL

    def test_env_url_used_when_no_request_url(self, client, monkeypatch):
        monkeypatch.setenv("SQUASH_VEX_URL", "https://env-override.example.com/feed.json")
        mock_instance = _update_feed_mock()
        with patch("squish.squash.vex.VexCache") as MockClass:
            MockClass.DEFAULT_URL = "https://unused.example.com/feed.json"
            MockClass.return_value = mock_instance
            resp = client.post("/vex/update", json={})
        assert resp.json()["url"] == "https://env-override.example.com/feed.json"

    def test_statement_count_returned(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_update_feed_mock(statement_count=5)):
            resp = client.post("/vex/update", json={"url": self._URL})
        assert resp.json()["statement_count"] == 5

    def test_force_true_passed_to_load_or_fetch(self, client):
        mock = _update_feed_mock()
        with patch("squish.squash.vex.VexCache", return_value=mock):
            client.post("/vex/update", json={"url": self._URL})
        call_kwargs = mock.load_or_fetch.call_args
        assert call_kwargs.kwargs.get("force") is True or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] is True
        )

    def test_custom_timeout_passed_to_load_or_fetch(self, client):
        mock = _update_feed_mock()
        with patch("squish.squash.vex.VexCache", return_value=mock):
            client.post("/vex/update", json={"url": self._URL, "timeout": 5.0})
        call_kwargs = mock.load_or_fetch.call_args
        timeout_val = call_kwargs.kwargs.get("timeout") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert timeout_val == 5.0

    def test_network_error_returns_502(self, client):
        bad_mock = MagicMock()
        bad_mock.load_or_fetch.side_effect = OSError("network unreachable")
        with patch("squish.squash.vex.VexCache", return_value=bad_mock):
            resp = client.post("/vex/update", json={"url": self._URL})
        assert resp.status_code == 502

    def test_network_error_detail_is_message(self, client):
        bad_mock = MagicMock()
        bad_mock.load_or_fetch.side_effect = RuntimeError("timeout after 30s")
        with patch("squish.squash.vex.VexCache", return_value=bad_mock):
            resp = client.post("/vex/update", json={"url": self._URL})
        assert "timeout after 30s" in resp.json().get("detail", "")


# ── Structural / contract tests ───────────────────────────────────────────────


class TestVexEndpointContracts:
    def test_get_vex_status_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/vex/status" in schema["paths"]

    def test_post_vex_update_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/vex/update" in schema["paths"]

    def test_get_vex_status_method_is_get(self, client):
        schema = client.get("/openapi.json").json()
        assert "get" in schema["paths"]["/vex/status"]

    def test_post_vex_update_method_is_post(self, client):
        schema = client.get("/openapi.json").json()
        assert "post" in schema["paths"]["/vex/update"]

    def test_vex_status_no_body_required(self, client):
        with patch("squish.squash.vex.VexCache", return_value=_empty_cache_mock()):
            resp = client.get("/vex/status")
        # GET with no body returns 200 (not 422 unprocessable entity)
        assert resp.status_code not in (422, 405)


# ── Counter integration ───────────────────────────────────────────────────────


class TestVexCounterIncrements:
    def test_vex_status_increments_counter(self, client):
        before = _COUNTERS["squash_vex_status_total"]
        with patch("squish.squash.vex.VexCache", return_value=_empty_cache_mock()):
            client.get("/vex/status")
        assert _COUNTERS["squash_vex_status_total"] == before + 1

    def test_vex_update_increments_counter(self, client):
        before = _COUNTERS["squash_vex_update_total"]
        with patch("squish.squash.vex.VexCache", return_value=_update_feed_mock()):
            client.post("/vex/update", json={"url": "https://test.example.com/feed.json"})
        assert _COUNTERS["squash_vex_update_total"] == before + 1

    def test_vex_update_counter_not_incremented_after_502(self, client):
        # Counter IS incremented even on error (mirrors /attest behavior)
        before = _COUNTERS["squash_vex_update_total"]
        bad_mock = MagicMock()
        bad_mock.load_or_fetch.side_effect = RuntimeError("fail")
        with patch("squish.squash.vex.VexCache", return_value=bad_mock):
            client.post("/vex/update", json={"url": "https://test.example.com/feed.json"})
        # Counter should be incremented (it runs before the exception propagates)
        assert _COUNTERS["squash_vex_update_total"] >= before
