"""squish/agent/builtin_tools.py — Built-in Tools for the Squish Agent.

Wave 72: six sandboxed tools provided out-of-the-box for filesystem access,
shell execution, Python REPL, and HTTP fetching.

The tools require no external dependencies beyond the Python standard library.
Security notes:
- ``squish_run_shell`` executes arbitrary commands — only enable in trusted
  environments, or set ``auto_approve=False`` and confirm per-call.
- ``squish_python_repl`` restricts the global namespace but is not a sandbox.
- ``squish_fetch_url`` validates the scheme and blocks ``file://`` URLs.

Call :func:`register_builtin_tools` to add all six tools to a registry::

    registry = ToolRegistry()
    register_builtin_tools(registry)
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
import urllib.request
import urllib.parse
import urllib.error
from contextlib import redirect_stdout
from typing import Any

from squish.agent.tool_registry import ToolDefinition, ToolRegistry


__all__ = ["register_builtin_tools"]

# Maximum number of bytes returned by squish_fetch_url
_FETCH_MAX_BYTES = 131_072  # 128 KiB
# Maximum number of lines returned by squish_read_file per call
_READ_MAX_LINES = 400


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_path(path: str) -> str:
    """Normalise and verify *path* does not contain null bytes."""
    if "\x00" in path:
        raise ValueError("Path contains null bytes")
    return os.path.normpath(path)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def squish_read_file(
    path: str,
    start_line: int = 1,
    end_line: int = _READ_MAX_LINES,
) -> str:
    """Read lines from a text file and return them as a string.

    Args:
        path: Absolute path to the file.
        start_line: First 1-based line to include (default: 1).
        end_line: Last 1-based line to include (default: 400).

    Returns:
        File content with a ``# Lines X–Y of Z total`` header.
    """
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        raise FileNotFoundError(f"File not found: {safe}")

    start = max(1, int(start_line))
    end = max(start, int(end_line))

    with open(safe, encoding="utf-8", errors="replace") as fh:
        all_lines = fh.readlines()

    total = len(all_lines)
    window = all_lines[start - 1 : end]
    header = f"# Lines {start}–{min(end, total)} of {total} total — {safe}\n"
    return header + "".join(window)


def squish_write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed.

    Args:
        path: Absolute path to write.
        content: Full file contents (UTF-8 string).

    Returns:
        Confirmation message with byte count.
    """
    safe = _safe_path(path)
    os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
    data = content.encode("utf-8")
    with open(safe, "wb") as fh:
        fh.write(data)
    return f"Wrote {len(data):,} bytes to {safe}"


def squish_list_dir(path: str) -> str:
    """List the contents of a directory.

    Args:
        path: Absolute path to the directory.

    Returns:
        Newline-separated entries, each prefixed with ``[DIR]`` or ``[FILE]``
        and suffixed with file sizes.
    """
    safe = _safe_path(path)
    if not os.path.isdir(safe):
        raise NotADirectoryError(f"Not a directory: {safe}")

    entries = sorted(os.listdir(safe))
    lines: list[str] = [f"# Contents of {safe}"]
    for entry in entries:
        full = os.path.join(safe, entry)
        if os.path.isdir(full):
            lines.append(f"[DIR]  {entry}/")
        else:
            try:
                size = os.path.getsize(full)
                lines.append(f"[FILE] {entry}  ({size:,} bytes)")
            except OSError:
                lines.append(f"[FILE] {entry}  (unknown size)")
    lines.append(f"\n{len(entries)} items total")
    return "\n".join(lines)


def squish_run_shell(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return its output.

    Args:
        command: Shell command string (executed by ``/bin/sh -c``).
        timeout: Maximum seconds to wait (default: 30).

    Returns:
        Combined stdout + stderr with exit code footer.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    try:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=int(timeout),
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Command exceeded {timeout}s limit"

    stdout = result.stdout.rstrip("\n")
    stderr = result.stderr.rstrip("\n")
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    parts.append(f"\n[exit {result.returncode}]")
    return "\n".join(parts)


def squish_python_repl(code: str, timeout: int = 10) -> str:
    """Execute Python *code* in an isolated namespace and capture stdout.

    Args:
        code: Python source string to execute.
        timeout: Maximum seconds (enforced via ``signal.alarm`` on POSIX;
            best-effort on Windows).

    Returns:
        Captured stdout, or an error traceback.
    """
    import builtins as _builtins
    import signal
    import traceback

    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")

    _ALLOWED = [
        "print", "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "list", "dict", "set", "tuple", "int",
        "float", "str", "bool", "bytes", "None", "True", "False",
        "abs", "min", "max", "sum", "round", "type", "isinstance",
        "issubclass", "repr", "hash", "id", "dir", "vars",
        "getattr", "setattr", "hasattr", "callable",
        "open", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "RuntimeError",
    ]
    restricted_builtins = {
        name: getattr(_builtins, name) for name in _ALLOWED
        if hasattr(_builtins, name)
    }

    namespace: dict[str, Any] = {"__builtins__": restricted_builtins}

    buf = io.StringIO()

    def _run() -> None:
        exec(textwrap.dedent(code), namespace)  # noqa: S102

    # POSIX timeout via SIGALRM
    if hasattr(signal, "SIGALRM"):
        def _handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Execution exceeded {timeout}s")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
    try:
        with redirect_stdout(buf):
            _run()
    except TimeoutError as exc:
        return f"[TIMEOUT] {exc}"
    except Exception:  # noqa: BLE001
        return f"[ERROR]\n{traceback.format_exc()}"
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    result = buf.getvalue()
    return result if result else "[no output]"


def squish_fetch_url(url: str, max_bytes: int = _FETCH_MAX_BYTES) -> str:
    """Fetch a URL and return its text content.

    Args:
        url: HTTPS or HTTP URL to fetch. ``file://`` URLs are blocked.
        max_bytes: Maximum bytes to read (default: 131 072).

    Returns:
        Decoded response text (truncated to *max_bytes* with a notice).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"squish_fetch_url only supports http/https URLs, "
            f"got scheme: {parsed.scheme!r}"
        )
    if parsed.netloc == "":
        raise ValueError("URL must include a host")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "squish-agent/72 (Python)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        return f"[HTTP {exc.code}] {exc.reason} — {url}"
    except urllib.error.URLError as exc:
        return f"[URLError] {exc.reason} — {url}"

    truncated = len(raw) > max_bytes
    content = raw[:max_bytes].decode("utf-8", errors="replace")
    notice = (
        f"\n\n[TRUNCATED at {max_bytes:,} bytes — full response larger]"
        if truncated
        else ""
    )
    return content + notice


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all six built-in tools into *registry*.

    Call this once during server or agent initialisation::

        registry = ToolRegistry()
        register_builtin_tools(registry)
    """
    _tools = [
        ToolDefinition(
            name="squish_read_file",
            description=(
                "Read lines from a text file on disk and return them as a string. "
                "Use start_line and end_line to paginate large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First 1-based line to include (default 1).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last 1-based line to include (default 400).",
                    },
                },
                "required": ["path"],
            },
            fn=squish_read_file,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_write_file",
            description=(
                "Write content to a file, creating parent directories as needed. "
                "WARNING: this overwrites existing files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full UTF-8 file contents.",
                    },
                },
                "required": ["path", "content"],
            },
            fn=squish_write_file,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_list_dir",
            description=(
                "List the contents of a directory, showing file sizes and "
                "distinguishing files from subdirectories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory.",
                    },
                },
                "required": ["path"],
            },
            fn=squish_list_dir,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_run_shell",
            description=(
                "Execute a shell command and return stdout + stderr. "
                "Only available in trusted environments."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run via /bin/sh -c.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to wait (default 30).",
                    },
                },
                "required": ["command"],
            },
            fn=squish_run_shell,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_python_repl",
            description=(
                "Execute Python code in an isolated namespace. "
                "Returns captured stdout or an error traceback."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds (default 10).",
                    },
                },
                "required": ["code"],
            },
            fn=squish_python_repl,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_fetch_url",
            description=(
                "Fetch an HTTP or HTTPS URL and return its text content. "
                "Truncated to 128 KiB. file:// URLs are blocked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP/HTTPS URL to fetch.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Max bytes to read (default 131072).",
                    },
                },
                "required": ["url"],
            },
            fn=squish_fetch_url,
            source="builtin",
        ),
    ]
    for defn in _tools:
        registry.register(defn)
