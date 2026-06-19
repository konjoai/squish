"""Behavioral coverage for ``squish.serving.mcp_client`` — the async STDIO and
SSE JSON-RPC paths, the connect/disconnect lifecycle, list_tools / call_tool,
and the MCPToolAdapter registry bridge.

The subprocess and HTTP layers are faked (in-memory reader/writer streams,
mocked ``create_subprocess_shell`` / ``urlopen``) so no real MCP server is
needed. Tests drive coroutines with ``asyncio.run`` (matching the existing
suite) — host-agnostic, no MLX.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
from unittest.mock import patch

import pytest

from squish.serving import mcp_client as mc
from squish.serving.mcp_client import (
    MCPClient,
    MCPToolAdapter,
    MCPToolDef,
    MCPTransport,
)


class _FakeWriter:
    def __init__(self) -> None:
        self.buf: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buf.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _stdio_client(response_objs: list[dict]) -> MCPClient:
    """A STDIO client pre-wired with fake streams returning *response_objs*."""
    client = MCPClient("dummy-cmd", transport=MCPTransport.STDIO)
    client._writer_pipe = _FakeWriter()
    client._reader = _FakeReader([_line(o) for o in response_objs])
    client._connected = True
    return client


# ── _next_id / _send_stdio / _recv_stdio ────────────────────────────────────


def test_next_id_increments():
    client = MCPClient("c")
    assert client._next_id() == 1
    assert client._next_id() == 2


def test_send_stdio_writes_framed_line():
    client = _stdio_client([])
    asyncio.run(client._send_stdio({"hello": "world"}))
    assert client._writer_pipe.buf == [b'{"hello":"world"}\n']


def test_recv_stdio_skips_blank_lines():
    client = MCPClient("c")
    client._reader = _FakeReader([b"\n", b"   \n", _line({"ok": 1})])
    assert asyncio.run(client._recv_stdio()) == {"ok": 1}


def test_recv_stdio_raises_on_eof():
    client = MCPClient("c")
    client._reader = _FakeReader([b""])  # immediate EOF
    with pytest.raises(ConnectionError, match="closed connection"):
        asyncio.run(client._recv_stdio())


# ── _rpc (STDIO) ────────────────────────────────────────────────────────────


def test_rpc_stdio_returns_result():
    client = _stdio_client([{"jsonrpc": "2.0", "id": 1, "result": {"v": 42}}])
    assert asyncio.run(client._rpc("ping", {})) == {"v": 42}


def test_rpc_stdio_skips_mismatched_id_then_matches():
    client = _stdio_client([
        {"jsonrpc": "2.0", "id": 999, "result": "other"},  # not ours → loop again
        {"jsonrpc": "2.0", "id": 1, "result": "mine"},
    ])
    assert asyncio.run(client._rpc("ping", {})) == "mine"


def test_rpc_stdio_raises_on_error():
    client = _stdio_client([
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "no method"}},
    ])
    with pytest.raises(RuntimeError, match="MCP RPC error -32601: no method"):
        asyncio.run(client._rpc("bad", {}))


# ── _rpc (SSE) ──────────────────────────────────────────────────────────────


class _FakeHTTPResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_rpc_sse_returns_result():
    client = MCPClient("http://host:9/", transport=MCPTransport.SSE)
    with patch("urllib.request.urlopen", return_value=_FakeHTTPResp({"result": {"ok": 1}})):
        assert asyncio.run(client._rpc("ping", {})) == {"ok": 1}


def test_rpc_sse_http_error():
    client = MCPClient("http://host:9", transport=MCPTransport.SSE)
    exc = urllib.error.HTTPError("u", 500, "Server Error", {}, None)
    with patch("urllib.request.urlopen", side_effect=exc):
        with pytest.raises(RuntimeError, match="MCP SSE HTTP error 500"):
            asyncio.run(client._rpc("ping", {}))


def test_rpc_sse_rpc_error_in_body():
    client = MCPClient("http://host:9", transport=MCPTransport.SSE)
    body = {"error": {"code": -1, "message": "nope"}}
    with patch("urllib.request.urlopen", return_value=_FakeHTTPResp(body)):
        with pytest.raises(RuntimeError, match="MCP RPC error -1: nope"):
            asyncio.run(client._rpc("ping", {}))


# ── _send_notification ──────────────────────────────────────────────────────


def test_send_notification_stdio_writes():
    client = _stdio_client([])
    asyncio.run(client._send_notification("notifications/x", {"a": 1}))
    assert client._writer_pipe.buf  # a frame was written


def test_send_notification_sse_is_noop():
    client = MCPClient("http://host:9", transport=MCPTransport.SSE)
    # SSE notifications are dropped (no transport write) — must not raise.
    asyncio.run(client._send_notification("notifications/x", {}))


# ── connect / disconnect lifecycle ──────────────────────────────────────────


def test_connect_sse_sets_connected():
    client = MCPClient("http://host:9", transport=MCPTransport.SSE)
    asyncio.run(client.connect())
    assert client._connected is True


def test_connect_stdio_runs_handshake(monkeypatch):
    writer = _FakeWriter()
    reader = _FakeReader([_line({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}})])

    class _FakeProc:
        stdout = reader
        stdin = writer

    async def fake_create(*a, **k):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create)
    client = MCPClient("serve-cmd", transport=MCPTransport.STDIO)
    asyncio.run(client.connect())
    assert client._connected is True
    assert client._process is not None
    # initialize request + initialized notification were both written.
    assert len(writer.buf) == 2


def test_disconnect_swallows_writer_and_process_errors():
    client = MCPClient("c", transport=MCPTransport.STDIO)

    class _BadPipe:
        def close(self):
            raise OSError("close boom")

        async def wait_closed(self):
            return None

    class _BadProc:
        def terminate(self):
            raise OSError("term boom")

        def wait(self, timeout):
            return None

    client._writer_pipe = _BadPipe()   # 139-140
    client._process = _BadProc()       # 145-146
    asyncio.run(client.disconnect())   # both errors swallowed
    assert client._connected is False


def test_disconnect_terminates_process_cleanly():
    client = MCPClient("c", transport=MCPTransport.STDIO)
    events = {}

    class _GoodPipe:
        def close(self):
            events["closed"] = True

        async def wait_closed(self):
            return None

    class _GoodProc:
        def terminate(self):
            events["terminated"] = True

        def wait(self, timeout):
            events["waited"] = timeout

    client._writer_pipe = _GoodPipe()
    client._process = _GoodProc()
    asyncio.run(client.disconnect())  # terminate() then wait() both run (142-144)
    assert events == {"closed": True, "terminated": True, "waited": 3}
    assert client._connected is False


def test_async_context_manager(monkeypatch):
    client = MCPClient("http://host:9", transport=MCPTransport.SSE)
    entered = {}

    async def run():
        async with client as c:
            entered["connected"] = c._connected
        return client._connected

    after = asyncio.run(run())
    assert entered["connected"] is True   # __aenter__ → connect
    assert after is False                 # __aexit__ → disconnect


# ── list_tools ──────────────────────────────────────────────────────────────


def test_list_tools_parses_tool_defs():
    client = _stdio_client([{
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [
            {"name": "search", "description": "web search",
             "inputSchema": {"type": "object"}},
            {"name": "calc"},  # missing fields → defaults applied
        ]},
    }])
    tools = asyncio.run(client.list_tools())
    assert [t.name for t in tools] == ["search", "calc"]
    assert tools[0].server_id == "mcp"
    assert tools[1].input_schema == {"type": "object", "properties": {}}


def test_list_tools_handles_bare_list_result():
    # result is a bare list (not wrapped in {"tools": ...}).
    client = _stdio_client([{"jsonrpc": "2.0", "id": 1,
                             "result": [{"name": "a"}]}])
    tools = asyncio.run(client.list_tools())
    assert tools[0].name == "a"


# ── call_tool ───────────────────────────────────────────────────────────────


def test_call_tool_joins_text_content():
    client = _stdio_client([{
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"text": "line1"}, {"text": "line2"}, "ignored"]},
    }])
    assert asyncio.run(client.call_tool("t", {})) == "line1\nline2"


def test_call_tool_none_result_returns_empty():
    client = _stdio_client([{"jsonrpc": "2.0", "id": 1, "result": None}])
    assert asyncio.run(client.call_tool("t", {})) == ""


def test_call_tool_error_raises():
    client = _stdio_client([{
        "jsonrpc": "2.0", "id": 1,
        "result": {"isError": True, "content": [{"text": "boom"}]},
    }])
    with pytest.raises(RuntimeError, match="MCP tool error: boom"):
        asyncio.run(client.call_tool("t", {}))


def test_call_tool_non_list_content_returned_as_is():
    client = _stdio_client([{"jsonrpc": "2.0", "id": 1,
                             "result": {"content": "plain string"}}])
    assert asyncio.run(client.call_tool("t", {})) == "plain string"


# ── MCPToolAdapter ──────────────────────────────────────────────────────────


class _FakeRegistry:
    def __init__(self, existing=()):
        self._names = set(existing)
        self.registered: list[str] = []

    def __contains__(self, name):
        return name in self._names

    def register(self, defn):
        self._names.add(defn.name)
        self.registered.append(defn.name)


def test_adapter_registers_new_tools_and_skips_existing(monkeypatch):
    client = _stdio_client([])

    async def fake_list():
        return [
            MCPToolDef(name="search", description="d", input_schema={}, server_id="mcp"),
            MCPToolDef(name="dup", description="d", input_schema={}, server_id="mcp"),
        ]

    monkeypatch.setattr(client, "list_tools", fake_list)
    registry = _FakeRegistry(existing={"dup"})  # "dup" already present → skipped
    adapter = MCPToolAdapter(client)
    registered = asyncio.run(adapter.load(registry))
    assert registered == ["search"]
    assert "search" in registry._names


def test_adapter_registered_fn_calls_through_to_client(monkeypatch):
    client = _stdio_client([])

    async def fake_list():
        return [MCPToolDef(name="echo", description="d", input_schema={}, server_id="mcp")]

    calls = {}

    async def fake_call(name, arguments):
        calls["name"] = name
        calls["args"] = arguments
        return "result-text"

    monkeypatch.setattr(client, "list_tools", fake_list)
    monkeypatch.setattr(client, "call_tool", fake_call)

    captured = {}

    class _CapRegistry(_FakeRegistry):
        def register(self, defn):
            super().register(defn)
            captured["fn"] = defn.fn

    registry = _CapRegistry()
    asyncio.run(MCPToolAdapter(client).load(registry))
    # The closure forwards kwargs to client.call_tool with the captured name.
    out = asyncio.run(captured["fn"](query="hi"))
    assert out == "result-text"
    assert calls == {"name": "echo", "args": {"query": "hi"}}
