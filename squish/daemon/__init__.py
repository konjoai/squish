"""squish.daemon — persistent background inference daemon (squishd).

Public API
----------
    from squish.daemon import DaemonClient, DaemonServer, is_running, SOCK_PATH

The daemon binds a Unix domain socket at SOCK_PATH, keeps N models in memory,
and routes requests by model name.  The thin client connects to the socket and
speaks the same JSON wire protocol as the OpenAI /v1/chat/completions endpoint.

Entry points
------------
    squishd start|stop|status|reload     (via pyproject.toml [project.scripts])
    squish daemon install|uninstall      (installs/removes macOS LaunchAgent plist)
"""
from __future__ import annotations

from squish.daemon.squishd import (
    SOCK_PATH,
    DaemonServer,
    is_running,
    send_request,
)
from squish.daemon.client import DaemonClient

__all__ = [
    "SOCK_PATH",
    "DaemonClient",
    "DaemonServer",
    "is_running",
    "send_request",
]
