"""Regression: disconnect() must close the real stdin pipe (_writer_pipe).

Previously disconnect() closed self._writer, an attribute that was declared but
never assigned (always None for the stdio transport), so the subprocess stdin
pipe (_writer_pipe) was leaked on every disconnect.
"""
from __future__ import annotations

import asyncio

from squish.serving.mcp_client import MCPClient, MCPTransport


class _FakePipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:  # mirrors asyncio.StreamWriter
        return None


def test_disconnect_closes_writer_pipe():
    client = MCPClient("dummy-cmd", transport=MCPTransport.STDIO)
    pipe = _FakePipe()
    client._writer_pipe = pipe  # simulate a connected stdio transport
    client._process = None

    asyncio.run(client.disconnect())

    assert pipe.closed is True
    assert client._connected is False


def test_disconnect_without_pipe_is_noop():
    client = MCPClient("dummy-cmd", transport=MCPTransport.STDIO)
    # Never connected → _writer_pipe is None; disconnect must not raise.
    asyncio.run(client.disconnect())
    assert client._connected is False
