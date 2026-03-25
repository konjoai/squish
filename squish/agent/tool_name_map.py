"""squish/agent/tool_name_map.py — Tool name normalization between VSCode and backend.

VSCode extension tools use unprefixed names (e.g. ``create_file``) while the
Squish Python backend uses ``squish_`` prefixed names (e.g. ``squish_create_file``).
This module provides bidirectional mapping so both clients work without change.

Public API
──────────
VSCODE_TO_BACKEND   dict[str, str] — VSCode name → backend name
BACKEND_TO_VSCODE   dict[str, str] — backend name → VSCode name (auto-generated inverse)
normalize_for_backend(name) → str
normalize_for_client(name)  → str
"""
from __future__ import annotations

__all__ = [
    "VSCODE_TO_BACKEND",
    "BACKEND_TO_VSCODE",
    "normalize_for_backend",
    "normalize_for_client",
]

# ---------------------------------------------------------------------------
# Name mapping tables
# ---------------------------------------------------------------------------

VSCODE_TO_BACKEND: dict[str, str] = {
    # File operations
    "create_file":        "squish_create_file",
    "write_file":         "squish_write_file",
    "read_file":          "squish_read_file",
    "delete_file":        "squish_delete_file",
    "rename_file":        "squish_rename_file",
    "apply_edit":         "squish_apply_edit",
    # Directory operations
    "list_directory":     "squish_list_directory",
    "create_directory":   "squish_create_directory",
    # Terminal / shell
    "run_terminal":       "squish_run_terminal",
    "run_command":        "squish_run_terminal",   # alias
    # Web / search
    "web_search":         "squish_web_search",
    # Workspace
    "get_workspace_info": "squish_get_workspace_info",
}

# Build the inverse automatically so normalize_for_client() stays in sync
BACKEND_TO_VSCODE: dict[str, str] = {v: k for k, v in VSCODE_TO_BACKEND.items()
                                       if not k.endswith("_command")}  # skip alias overlap


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_for_backend(name: str) -> str:
    """Translate a client-side (VSCode) tool name to the backend canonical name.

    If the name is already a backend name (has ``squish_`` prefix) or is
    unknown, it is returned unchanged.

    Parameters
    ----------
    name:
        Tool name received from the client.

    Returns
    -------
    str
        Backend tool name.
    """
    return VSCODE_TO_BACKEND.get(name, name)


def normalize_for_client(name: str) -> str:
    """Translate a backend tool name to the VSCode client-compatible name.

    If the name is already a client name or is unknown, it is returned unchanged.

    Parameters
    ----------
    name:
        Tool name used internally (may have ``squish_`` prefix).

    Returns
    -------
    str
        Client-friendly tool name (no ``squish_`` prefix for known tools).
    """
    return BACKEND_TO_VSCODE.get(name, name)
