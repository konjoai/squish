"""tests/serving/test_server_misc_endpoints_unit.py

Linux-runnable coverage for several small ``squish.server`` HTTP handlers that
were previously dark:

  * ``web_chat_ui``  — the API-key injection path + both no-inject branches
  * ``get_sbom``     — no-model-dir 404, no-sidecar 404, and the success read
  * ``get_obs_report`` and ``quality`` — the happy 200 responses

Each handler is an async ``app`` route; we call the coroutine directly with a
fake request (the established server-test pattern).
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

import squish.server as server


def _req(host):
    """Fake Starlette Request exposing only ``.client.host``."""
    client = types.SimpleNamespace(host=host) if host is not None else None
    return types.SimpleNamespace(client=client)


@pytest.fixture(autouse=True)
def _reset_api_key(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    yield


# ── web_chat_ui ───────────────────────────────────────────────────────────────────


def test_web_chat_ui_injects_key_for_local_client(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", "secret-key-123", raising=False)
    resp = asyncio.run(server.web_chat_ui(_req("127.0.0.1")))
    body = bytes(resp.body).decode()
    # key injected as a JS global, JSON-escaped, with no-cache headers
    assert "window.SQUISH_DEFAULT_API_KEY" in body
    assert "secret-key-123" in body
    assert resp.headers["Cache-Control"].startswith("no-cache")


def test_web_chat_ui_no_inject_when_no_key(monkeypatch):
    # local client but no API key configured → plain FileResponse, no injection
    resp = asyncio.run(server.web_chat_ui(_req("127.0.0.1")))
    assert resp.__class__.__name__ == "FileResponse"


def test_web_chat_ui_no_inject_for_remote_client(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", "secret-key-123", raising=False)
    # non-local client → never inject the key, even when one is set
    resp = asyncio.run(server.web_chat_ui(_req("8.8.8.8")))
    assert resp.__class__.__name__ == "FileResponse"


# ── get_sbom ──────────────────────────────────────────────────────────────────────


def test_get_sbom_404_without_model_dir(monkeypatch):
    monkeypatch.setattr(server._state, "model_dir", None, raising=False)
    resp = asyncio.run(server.get_sbom())
    assert resp.status_code == 404


def test_get_sbom_404_when_sidecar_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server._state, "model_dir", str(tmp_path), raising=False)
    resp = asyncio.run(server.get_sbom())  # dir exists, sidecar does not
    assert resp.status_code == 404


def test_get_sbom_returns_sidecar(monkeypatch, tmp_path):
    sidecar = tmp_path / "cyclonedx-mlbom.json"
    sidecar.write_text(json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.5"}))
    monkeypatch.setattr(server._state, "model_dir", str(tmp_path), raising=False)
    resp = asyncio.run(server.get_sbom())
    payload = json.loads(bytes(resp.body))
    assert payload["bomFormat"] == "CycloneDX"


# ── obs-report + quality ──────────────────────────────────────────────────────────


def test_get_obs_report_ok():
    resp = asyncio.run(server.get_obs_report(creds=None))
    assert resp.status_code == 200
    payload = json.loads(bytes(resp.body))
    assert "status" in payload


def test_quality_endpoint_ok():
    resp = asyncio.run(server.quality(creds=None))
    assert resp.status_code == 200
    payload = json.loads(bytes(resp.body))
    assert "window_seconds" in payload
