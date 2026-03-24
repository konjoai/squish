"""squish/serving/mcp_client.py — Model Context Protocol (MCP) Client.

Wave 72: lightweight MCP client that connects to external tool servers over
either stdio (subprocess JSON-RPC) or HTTP SSE, discovers their tools, and
adapts them as :class:`~squish.agent.tool_registry.ToolDefinition` entries.

Spec reference: https://modelcontextprotocol.io/specification

Only the JSON-RPC 2.0 ``initialize`` → ``tools/list`` → ``tools/call``
sequence is implemented here — sufficient for agentic tool dispatch.

Usage::

    from squish.serving.mcp_client import MCPClient, MCPTransport

    async with MCPClient("uvx mcp-server-filesystem --root /tmp",
                         transport=MCPTransport.STDIO) as client:
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/hello.txt"})
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "MCPTransport",
    "MCPToolDef",
    "MCPClient",
    "MCPToolAdapter",
]

# JSON-RPC 2.0 protocol version used by MCP
_JSONRPC_VERSION = "2.0"
_MCP_PROTOCOL_VERSION = "2024-11-05"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class MCPTransport(enum.Enum):
    """Supported MCP connection transports."""

    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPToolDef:
    """A tool discovered from an MCP server.

    Attributes:
        name: Tool identifier (globally unique within a server).
        description: Human-readable description.
        input_schema: JSON Schema for the ``arguments`` object.
        server_id: Tag identifying which MCP server this came from.
    """

    name: str
    description: str
    input_schema: dict
    server_id: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MCPClient:
    """Async context manager that connects to an MCP server.

    Args:
        server_cmd_or_url: For ``STDIO``, the shell command string to launch
            the MCP server process. For ``SSE``, the HTTP/HTTPS base URL.
        transport: :class:`MCPTransport` variant.
        server_id: Human-readable tag for logging and tool name-spacing.
        connect_timeout: Seconds to wait for the server to become ready.
    """

    def __init__(
        self,
        server_cmd_or_url: str,
        transport: MCPTransport = MCPTransport.STDIO,
        *,
        server_id: str = "mcp",
        connect_timeout: float = 10.0,
    ) -> None:
        self.server_cmd_or_url = server_cmd_or_url
        self.transport = transport
        self.server_id = server_id
        self.connect_timeout = connect_timeout

        self._process: Optional[subprocess.Popen] = None  # STDIO only
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._msg_id = 0
        self._connected = False

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the transport connection and run the MCP handshake."""
        if self.transport == MCPTransport.STDIO:
            await self._connect_stdio()
        else:
            self._connected = True  # SSE is stateless per request
        logger.debug("MCPClient(%s) connected via %s", self.server_id, self.transport.value)

    async def disconnect(self) -> None:
        """Close the connection and terminate any subprocess."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
        logger.debug("MCPClient(%s) disconnected", self.server_id)

    async def _connect_stdio(self) -> None:
        """Launch the subprocess and perform the JSON-RPC ``initialize`` handshake."""
        cmd = self.server_cmd_or_url
        self._process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader = self._process.stdout
        self._writer_pipe = self._process.stdin

        # MCP initialize handshake
        init_resp = await self._rpc(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "squish-mcp-client", "version": "72.0"},
            },
        )
        logger.debug("MCP initialize response: %s", init_resp)

        # Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized", {})
        self._connected = True

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send_stdio(self, payload: dict) -> None:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        self._writer_pipe.write(line.encode())
        await self._writer_pipe.drain()

    async def _recv_stdio(self) -> dict:
        while True:
            raw = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self.connect_timeout,
            )
            if not raw:
                raise ConnectionError("MCP server closed connection")
            line = raw.decode().strip()
            if not line:
                continue
            return json.loads(line)

    async def _rpc(self, method: str, params: dict) -> Any:
        """Send a JSON-RPC request and await the response."""
        msg_id = self._next_id()
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": msg_id,
            "method": method,
            "params": params,
        }
        if self.transport == MCPTransport.STDIO:
            await self._send_stdio(payload)
            while True:
                resp = await self._recv_stdio()
                if resp.get("id") == msg_id:
                    if "error" in resp:
                        raise RuntimeError(
                            f"MCP RPC error {resp['error'].get('code')}: "
                            f"{resp['error'].get('message')}"
                        )
                    return resp.get("result")
        else:
            # SSE transport — POST to <base>/message
            url = self.server_cmd_or_url.rstrip("/") + "/message"
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.connect_timeout) as resp_http:  # noqa: S310
                    data = json.loads(resp_http.read())
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"MCP SSE HTTP error {exc.code}: {exc.reason}") from exc
            if "error" in data:
                raise RuntimeError(
                    f"MCP RPC error {data['error'].get('code')}: "
                    f"{data['error'].get('message')}"
                )
            return data.get("result")

    async def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no ``id``, no response expected)."""
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "method": method,
            "params": params,
        }
        if self.transport == MCPTransport.STDIO:
            await self._send_stdio(payload)

    # ------------------------------------------------------------------
    # MCP API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolDef]:
        """Discover available tools from the MCP server.

        Returns:
            List of :class:`MCPToolDef` instances.
        """
        result = await self._rpc("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else result or []
        return [
            MCPToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                server_id=self.server_id,
            )
            for t in tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name as returned by :meth:`list_tools`.
            arguments: Argument dict matching the tool's ``inputSchema``.

        Returns:
            The ``content`` value from the MCP ``tools/call`` response.
        """
        result = await self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        # MCP tools/call response: {"content": [...], "isError": bool}
        if result is None:
            return ""
        if result.get("isError"):
            content = result.get("content", [])
            error_text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
            raise RuntimeError(f"MCP tool error: {error_text}")
        content = result.get("content", [])
        if isinstance(content, list):
            return "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        return content


# ---------------------------------------------------------------------------
# Adapter: bridge MCP tools → ToolRegistry
# ---------------------------------------------------------------------------

class MCPToolAdapter:
    """Wraps a connected :class:`MCPClient` so its tools can be added to a
    :class:`~squish.agent.tool_registry.ToolRegistry`.

    Usage::

        adapter = MCPToolAdapter(client)
        await adapter.load(registry)
    """

    def __init__(self, client: MCPClient) -> None:
        self.client = client

    async def load(self, registry: Any) -> list[str]:
        """Fetch tool list and register them in *registry*.

        Args:
            registry: A :class:`~squish.agent.tool_registry.ToolRegistry`
                instance.

        Returns:
            List of tool names that were registered.
        """
        from squish.agent.tool_registry import ToolDefinition

        mcp_tools = await self.client.list_tools()
        registered: list[str] = []
        for mt in mcp_tools:
            tool_name = mt.name
            if tool_name in registry:
                logger.warning(
                    "MCPToolAdapter: skipping tool %r (already registered)", tool_name
                )
                continue

            # Capture `mt` in closure properly
            def _make_fn(captured_name: str, captured_client: MCPClient):
                async def _call(**kwargs: Any) -> str:  # noqa: ANN202
                    return await captured_client.call_tool(captured_name, kwargs)
                _call.__name__ = captured_name
                return _call

            defn = ToolDefinition(
                name=tool_name,
                description=mt.description,
                parameters=mt.input_schema,
                fn=_make_fn(mt.name, self.client),
                source=f"mcp:{self.client.server_id}",
            )
            registry.register(defn)
            registered.append(tool_name)

        return registered
