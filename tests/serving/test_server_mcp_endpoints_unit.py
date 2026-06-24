"""tests/serving/test_server_mcp_endpoints_unit.py

Linux-runnable coverage for the agent/MCP HTTP endpoints in ``squish.server``
(server.py ~3920-3991): list, connect (every guard + success + failure) and
disconnect (not-found + success + swallowed-error).  The endpoints are async
``app`` handlers; we call the coroutines directly with a fake request and patch
the ``squish.serving.mcp_client`` symbols so no real MCP process is launched.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import squish.serving.mcp_client as mcp_client_mod
import squish.server as server


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _FakeClient:
    def __init__(self, src, transport=None, server_id="mcp"):
        self.src = src
        self.transport = transport
        self.server_id = server_id
        self.disconnected = False

    async def connect(self):
        return None

    async def disconnect(self):
        self.disconnected = True


class _FakeAdapter:
    def __init__(self, client):
        self.client = client

    async def load(self, registry):
        return 3  # pretend three tools registered


@pytest.fixture(autouse=True)
def _no_auth(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    # Start every test from an empty connected-server map.
    monkeypatch.setattr(server, "_mcp_servers", {}, raising=False)
    yield


# ── list ────────────────────────────────────────────────────────────────────────


def test_agent_list_mcp_returns_connected(monkeypatch):
    monkeypatch.setattr(server, "_mcp_servers", {"alpha": object()}, raising=False)
    out = asyncio.run(server.agent_list_mcp(creds=None))
    assert out == {"servers": [{"id": "alpha", "status": "connected"}]}


# ── connect guards ────────────────────────────────────────────────────────────────


def test_connect_registry_uninitialised_503(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", None, raising=False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.agent_connect_mcp(_FakeRequest({"command": "x"}), creds=None))
    assert ei.value.status_code == 503


def test_connect_requires_command_or_url_400(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", object(), raising=False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.agent_connect_mcp(_FakeRequest({}), creds=None))
    assert ei.value.status_code == 400


def test_connect_duplicate_server_id_409(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", object(), raising=False)
    monkeypatch.setattr(server, "_mcp_servers", {"dup": object()}, raising=False)
    body = {"server_id": "dup", "command": "run-me"}
    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.agent_connect_mcp(_FakeRequest(body), creds=None))
    assert ei.value.status_code == 409


# ── connect success + failure ─────────────────────────────────────────────────────


def test_connect_success_stdio(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", object(), raising=False)
    monkeypatch.setattr(mcp_client_mod, "MCPClient", _FakeClient, raising=False)
    monkeypatch.setattr(mcp_client_mod, "MCPToolAdapter", _FakeAdapter, raising=False)

    body = {"server_id": "fs", "command": "python -m mcp_fs"}
    out = asyncio.run(server.agent_connect_mcp(_FakeRequest(body), creds=None))
    assert out == {"server_id": "fs", "transport": "stdio", "tools_registered": 3}
    assert "fs" in server._mcp_servers


def test_connect_success_sse(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", object(), raising=False)
    monkeypatch.setattr(mcp_client_mod, "MCPClient", _FakeClient, raising=False)
    monkeypatch.setattr(mcp_client_mod, "MCPToolAdapter", _FakeAdapter, raising=False)

    body = {"server_id": "remote", "url": "https://mcp.example", "transport": "sse"}
    out = asyncio.run(server.agent_connect_mcp(_FakeRequest(body), creds=None))
    assert out["transport"] == "sse"
    assert server._mcp_servers["remote"].src == "https://mcp.example"


def test_connect_failure_returns_500(monkeypatch):
    monkeypatch.setattr(server, "_agent_registry", object(), raising=False)

    class _BoomClient(_FakeClient):
        async def connect(self):
            raise RuntimeError("spawn failed")

    monkeypatch.setattr(mcp_client_mod, "MCPClient", _BoomClient, raising=False)
    monkeypatch.setattr(mcp_client_mod, "MCPToolAdapter", _FakeAdapter, raising=False)
    body = {"server_id": "bad", "command": "nope"}
    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.agent_connect_mcp(_FakeRequest(body), creds=None))
    assert ei.value.status_code == 500
    assert "MCP connect failed" in ei.value.detail


# ── disconnect ────────────────────────────────────────────────────────────────────


def test_disconnect_not_found_404(monkeypatch):
    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.agent_disconnect_mcp("ghost", creds=None))
    assert ei.value.status_code == 404


def test_disconnect_success(monkeypatch):
    client = _FakeClient("cmd")
    monkeypatch.setattr(server, "_mcp_servers", {"fs": client}, raising=False)
    out = asyncio.run(server.agent_disconnect_mcp("fs", creds=None))
    assert out == {"disconnected": "fs"}
    assert client.disconnected is True
    assert "fs" not in server._mcp_servers


def test_disconnect_swallows_client_error(monkeypatch):
    class _BoomClient(_FakeClient):
        async def disconnect(self):
            raise RuntimeError("already dead")

    monkeypatch.setattr(server, "_mcp_servers", {"fs": _BoomClient("cmd")}, raising=False)
    out = asyncio.run(server.agent_disconnect_mcp("fs", creds=None))
    assert out == {"disconnected": "fs"}  # error logged at debug, not surfaced
