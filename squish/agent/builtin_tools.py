"""squish/agent/builtin_tools.py — Built-in Tools for the Squish Agent.

Wave 72: six sandboxed tools provided out-of-the-box for filesystem access,
shell execution, Python REPL, and HTTP fetching.

Wave 76: added five more tools — web search, create_file, delete_file,
apply_edit, and move_file — giving the agent full filesystem CRUD and search
capability.

The tools require no external dependencies beyond the Python standard library.
Security notes:
- ``squish_run_shell`` executes arbitrary commands — only enable in trusted
  environments, or set ``auto_approve=False`` and confirm per-call.
- ``squish_python_repl`` restricts the global namespace and runs in a spawned
  subprocess with memory (RLIMIT_AS) and CPU (RLIMIT_CPU) limits; it is a
  resource sandbox, not a security sandbox (``open`` is still exposed).
- ``squish_fetch_url`` validates the scheme and blocks ``file://`` URLs.
- ``squish_delete_file`` permanently removes a file — use with care.

Wave 99+: added ``create_directory`` so the VSCode-side ``create_directory``
tool resolves to a real backend implementation (previously mapped to an
unregistered name and failed at dispatch).

Call :func:`register_builtin_tools` to add all thirteen tools to a registry::

    registry = ToolRegistry()
    register_builtin_tools(registry)
"""

from __future__ import annotations

import html
import io
import logging
import os
import re
import subprocess
import sys
import textwrap
import urllib.request
import urllib.parse
import urllib.error
from contextlib import redirect_stdout
from typing import Any

from squish.agent.tool_registry import ToolDefinition, ToolRegistry

_LOG = logging.getLogger("squish.agent.builtin_tools")


__all__ = [
    "register_builtin_tools",
    "squish_apply_edit",
    "squish_create_directory",
    "squish_create_file",
    "squish_delete_file",
    "squish_fetch_url",
    "squish_list_dir",
    "squish_move_file",
    "squish_python_repl",
    "squish_read_document",
    "squish_read_file",
    "squish_run_shell",
    "squish_web_search",
    "squish_write_file",
]

# Maximum number of bytes returned by squish_fetch_url / squish_web_search
_FETCH_MAX_BYTES = 131_072  # 128 KiB
# Maximum number of lines returned by squish_read_file per call
_READ_MAX_LINES = 400
# Maximum web-search results returned
_SEARCH_MAX_RESULTS = 10


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


# Builtin names exposed inside the REPL's restricted namespace.
_REPL_ALLOWED_BUILTINS: tuple[str, ...] = (
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "list", "dict", "set", "tuple", "int", "float", "str", "bool",
    "bytes", "None", "True", "False", "abs", "min", "max", "sum", "round",
    "type", "isinstance", "issubclass", "repr", "hash", "id", "dir", "vars",
    "getattr", "setattr", "hasattr", "callable", "open", "Exception",
    "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "RuntimeError",
)

# Default address-space cap for the isolated REPL child (MiB). A freshly
# spawned child baselines at ~30 MiB, so this leaves ample headroom for normal
# code while stopping an allocation bomb from exhausting the host process.
# NOTE: RLIMIT_AS is enforced on Linux; macOS/Darwin accepts the call but does
# not enforce it, so on macOS the memory cap is best-effort and protection
# falls back to the CPU limit + wall-clock timeout + process isolation.
_REPL_DEFAULT_MAX_MEMORY_MB = 512


def _repl_namespace() -> "dict[str, Any]":
    """Build the restricted global namespace shared by both REPL execution paths."""
    import builtins as _builtins
    restricted = {
        name: getattr(_builtins, name)
        for name in _REPL_ALLOWED_BUILTINS
        if hasattr(_builtins, name)
    }
    return {"__builtins__": restricted}


def _repl_isolated_worker(code: str, mem_mb: int, cpu_s: int, conn) -> None:  # pragma: no cover - runs in child process
    """Execute *code* in a spawned child with hard memory/CPU rlimits.

    Sends ``(status, output)`` back over *conn* where status is ``"ok"`` /
    ``"mem"`` (memory limit hit) / ``"err"`` (any other failure). Runs in a
    fresh interpreter, so the parent's loaded model is never inherited and the
    RLIMIT_AS cap is meaningful.
    """
    import io as _io
    import textwrap as _tw
    import traceback as _tb
    from contextlib import redirect_stdout as _rso

    try:
        import resource as _res
        if mem_mb > 0:
            _b = mem_mb * 1024 * 1024
            _res.setrlimit(_res.RLIMIT_AS, (_b, _b))
        if cpu_s > 0:
            _res.setrlimit(_res.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ImportError, ValueError, OSError):
        pass  # limits are best-effort; execution still proceeds

    buf = _io.StringIO()
    try:
        with _rso(buf):
            exec(_tw.dedent(code), _repl_namespace())  # noqa: S102
        conn.send(("ok", buf.getvalue()))
    except MemoryError:
        conn.send(("mem", buf.getvalue()))
    except Exception:  # noqa: BLE001 — sandbox boundary: report any exec failure to the parent
        conn.send(("err", buf.getvalue() + _tb.format_exc()))
    finally:
        conn.close()


def _repl_run_isolated(code: str, timeout: int, max_memory_mb: int) -> "str | None":
    """Run *code* in a spawned, rlimit-capped child process.

    Returns the result string, or ``None`` if process isolation is unavailable
    (caller then falls back to in-process execution).
    """
    try:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
    except (ImportError, ValueError) as exc:
        _LOG.debug("REPL isolation unavailable (%s)", exc)
        return None

    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_repl_isolated_worker,
        # CPU limit backstops the wall-clock join; give it 1s of headroom.
        args=(code, max_memory_mb, int(timeout) + 1, send_conn),
        daemon=True,
    )
    try:
        proc.start()
    except (OSError, ValueError, RuntimeError) as exc:
        _LOG.debug("REPL subprocess spawn failed (%s); falling back in-process", exc)
        return None
    send_conn.close()  # parent keeps only the receive end → EOF on child exit

    proc.join(timeout + 2)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return f"[TIMEOUT] Execution exceeded {timeout}s"

    terminated = "[ERROR] sandboxed process terminated (memory or CPU limit exceeded)"
    if recv_conn.poll():
        try:
            status, out = recv_conn.recv()
        except EOFError:
            # poll() reports readable on EOF too — child was killed before sending.
            return terminated
        if status == "mem":
            return (out + "\n" if out else "") + "[MEMORY LIMIT EXCEEDED]"
        if status == "err":
            return f"[ERROR]\n{out}"
        return out if out else "[no output]"

    # Child exited without a message — killed by RLIMIT_CPU, the OOM killer, or sys.exit().
    return terminated


def _repl_run_in_process(code: str, timeout: int) -> str:
    """In-process REPL execution (SIGALRM timeout, no memory cap). Fallback path."""
    import signal
    import traceback

    namespace = _repl_namespace()
    buf = io.StringIO()
    if hasattr(signal, "SIGALRM"):

        def _handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Execution exceeded {timeout}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
    try:
        with redirect_stdout(buf):
            exec(textwrap.dedent(code), namespace)  # noqa: S102
    except TimeoutError as exc:
        return f"[TIMEOUT] {exc}"
    except Exception as exc:  # noqa: BLE001 — sandbox boundary: exec of arbitrary code
        _LOG.debug("sandboxed exec raised: %s", exc)
        return f"[ERROR]\n{traceback.format_exc()}"
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    result = buf.getvalue()
    return result if result else "[no output]"


def squish_python_repl(
    code: str, timeout: int = 10, max_memory_mb: int = _REPL_DEFAULT_MAX_MEMORY_MB
) -> str:
    """Execute Python *code* in a restricted namespace and capture stdout.

    Runs in a spawned child process with hard memory (``RLIMIT_AS``) and CPU
    (``RLIMIT_CPU``) limits so an allocation bomb or runaway loop cannot exhaust
    the host serving process. Falls back to in-process execution (timeout only,
    no memory cap) when process isolation is unavailable.

    Note: ``RLIMIT_AS`` is enforced on Linux but not on macOS/Darwin (which
    accepts the call yet ignores it); on macOS the runaway-loop and timeout
    guards still apply, but the memory cap is best-effort.

    Args:
        code: Python source string to execute.
        timeout: Maximum wall-clock seconds before the child is terminated.
        max_memory_mb: Child address-space cap in MiB (default 512; Linux-enforced).

    Returns:
        Captured stdout, or an error / timeout / memory-limit message.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")

    isolated = _repl_run_isolated(code, timeout, max_memory_mb)
    if isolated is not None:
        return isolated

    _LOG.warning("Isolated REPL unavailable; running in-process without a memory cap")
    return _repl_run_in_process(code, timeout)


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
            f"squish_fetch_url only supports http/https URLs, got scheme: {parsed.scheme!r}"
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
    notice = f"\n\n[TRUNCATED at {max_bytes:,} bytes — full response larger]" if truncated else ""
    return content + notice


def squish_web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return result snippets.

    Uses DuckDuckGo Lite (no API key required, no tracking).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5, max: 10).

    Returns:
        Formatted string with search result titles, URLs, and snippets.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    n = min(int(max_results), _SEARCH_MAX_RESULTS)
    encoded = urllib.parse.urlencode({"q": query.strip()})
    url = f"https://lite.duckduckgo.com/lite/?{encoded}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "squish-agent/76 (Python; web-search)",
            "Accept": "text/html",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            raw = resp.read(_FETCH_MAX_BYTES)
    except urllib.error.HTTPError as exc:
        return f"[HTTP {exc.code}] {exc.reason}"
    except urllib.error.URLError as exc:
        return f"[URLError] {exc.reason}"

    page = raw.decode("utf-8", errors="replace")

    # DuckDuckGo Lite result rows are structured as:
    # <a class="result-link" href="...">Title</a>
    # <td class="result-snippet">...snippet...</td>
    results: list[str] = []
    # Extract result links and titles
    link_pattern = re.compile(
        r'<a[^>]+class="result-link"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>',
        re.DOTALL | re.IGNORECASE,
    )

    links = link_pattern.findall(page)
    snippets_raw = snippet_pattern.findall(page)
    snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets_raw]
    # Unescape HTML entities
    snippets = [html.unescape(s) for s in snippets]

    for i, (href, title) in enumerate(links):
        if i >= n:
            break
        clean_title = html.unescape(re.sub(r"<[^>]+>", "", title).strip())
        snippet = snippets[i] if i < len(snippets) else ""
        # DDG Lite redirects through its own tracking URL — decode the real URL
        qs = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(qs)
        real_url = params.get("uddg", [href])[0] or href
        results.append(f"[{i + 1}] {clean_title}\n    {real_url}\n    {snippet}")

    if not results:
        # Fallback: try to extract any meaningful links from the page
        any_links = re.findall(r'href="(https?://[^"]+)"', page)
        for link in any_links[:n]:
            if "duckduckgo.com" not in link:
                results.append(f"• {link}")
        if not results:
            return f"No results found for: {query}"

    header = f"Search results for: {query}\n" + "=" * 40
    return header + "\n\n" + "\n\n".join(results)


def squish_create_file(path: str, content: str) -> str:
    """Create a new file with the given content. Fails if the file already exists.

    Use :func:`squish_write_file` to overwrite existing files.

    Args:
        path: Absolute path for the new file.
        content: Full UTF-8 content to write.

    Returns:
        Confirmation message with byte count.
    """
    safe = _safe_path(path)
    if os.path.exists(safe):
        raise FileExistsError(f"File already exists: {safe}. Use squish_write_file to overwrite.")
    os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
    data = content.encode("utf-8")
    with open(safe, "wb") as fh:
        fh.write(data)
    return f"Created {len(data):,} bytes at {safe}"


def squish_create_directory(path: str) -> str:
    """Create a directory, including any missing parent directories.

    Idempotent: succeeds if the directory already exists.

    Args:
        path: Absolute path of the directory to create.

    Returns:
        Confirmation message.
    """
    safe = _safe_path(path)
    if os.path.isfile(safe):
        raise FileExistsError(f"Path exists and is a file, not a directory: {safe}")
    os.makedirs(safe, exist_ok=True)
    return f"Created directory: {safe}"


def squish_delete_file(path: str) -> str:
    """Delete a file from disk. Use with caution — this is irreversible.

    Args:
        path: Absolute path of the file to delete.

    Returns:
        Confirmation message.
    """
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        raise FileNotFoundError(f"File not found: {safe}")
    os.remove(safe)
    return f"Deleted: {safe}"


def squish_move_file(src: str, dst: str) -> str:
    """Move (rename) a file or directory.

    Both *src* and *dst* must be absolute paths.  Parent directories of *dst*
    are created automatically.  If *dst* already exists the call raises
    ``FileExistsError``.

    Args:
        src: Absolute path of the source file or directory.
        dst: Absolute destination path.

    Returns:
        Confirmation message.
    """
    safe_src = _safe_path(src)
    safe_dst = _safe_path(dst)
    if not os.path.exists(safe_src):
        raise FileNotFoundError(f"Source not found: {safe_src}")
    if os.path.exists(safe_dst):
        raise FileExistsError(f"Destination already exists: {safe_dst}")
    os.makedirs(os.path.dirname(safe_dst) or ".", exist_ok=True)
    os.rename(safe_src, safe_dst)
    return f"Moved {safe_src} -> {safe_dst}"


def squish_apply_edit(path: str, old_text: str, new_text: str) -> str:
    """Apply a surgical text replacement in a file.

    Replaces the first (and only) occurrence of *old_text* with *new_text*.
    The replacement is rejected if *old_text* appears zero or more than once
    (ambiguous edit).

    Args:
        path: Absolute path of the file to edit.
        old_text: Exact text to find (must appear exactly once).
        new_text: Replacement text.

    Returns:
        Confirmation message.
    """
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        raise FileNotFoundError(f"File not found: {safe}")

    with open(safe, encoding="utf-8", errors="replace") as fh:
        src = fh.read()

    count = src.count(old_text)
    if count == 0:
        raise ValueError(f"old_text not found in {safe}")
    if count > 1:
        raise ValueError(f"old_text is ambiguous — found {count} occurrences in {safe}")

    result = src.replace(old_text, new_text, 1)
    data = result.encode("utf-8")
    with open(safe, "wb") as fh:
        fh.write(data)
    return f"Applied edit to {safe}"


# Maximum characters of extracted document text returned per call.
_DOC_MAX_CHARS = 200_000


def _extract_docx(path: str) -> str:
    """Extract text from a .docx using only the standard library.

    A .docx is a zip archive; the body lives in ``word/document.xml``. We strip
    the XML tags and turn paragraph/break markers into newlines — no
    ``python-docx`` dependency required.
    """
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    # Paragraphs and line breaks → newlines, then drop the remaining tags.
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(text)


def _extract_pdf(path: str) -> str:
    """Extract text from a PDF when an optional parser is installed.

    Tries ``pypdf`` then ``pdfplumber``; returns an actionable message when
    neither is available rather than failing hard.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except ImportError:
        pass
    try:
        import pdfplumber  # noqa: PLC0415

        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except ImportError:
        return (
            "[PDF text extraction requires `pypdf` or `pdfplumber` — install "
            "one with `pip install pypdf`, then retry.]"
        )


def squish_read_document(path: str, max_chars: int = _DOC_MAX_CHARS) -> str:
    """Extract readable text from a document so the agent can analyse it.

    Handles PDF, DOCX, and every text-like format (code, Markdown, JSON, CSV,
    plain text). Use this for uploaded files / attachments the user references.

    Args:
        path: Absolute path to the document.
        max_chars: Cap on returned characters (default 200,000).
    """
    safe = _safe_path(path)
    if not os.path.exists(safe):
        raise FileNotFoundError(f"No such file: {path}")
    if not os.path.isfile(safe):
        raise ValueError(f"Not a file: {path}")

    ext = os.path.splitext(safe)[1].lower()
    size = os.path.getsize(safe)
    cap = max(1, int(max_chars))

    if ext == ".pdf":
        body = _extract_pdf(safe)
    elif ext == ".docx":
        body = _extract_docx(safe)
    else:
        with open(safe, "rb") as fh:
            raw = fh.read(cap * 4 + 8)
        body = raw.decode("utf-8", errors="replace")

    body = body[:cap]
    header = f"# Document: {os.path.basename(safe)} ({ext or 'no-ext'}, {size:,} bytes)\n\n"
    return header + (body if body.strip() else "[no extractable text found]")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all twelve built-in tools into *registry*.

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
            name="squish_read_document",
            description=(
                "Extract readable text from a document (PDF, DOCX, CSV, "
                "Markdown, JSON, code, plain text) so you can analyse it. Use "
                "this for uploaded files and attachments referenced by path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the document.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 200000).",
                    },
                },
                "required": ["path"],
            },
            fn=squish_read_document,
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
            name="squish_create_file",
            description=(
                "Create a new file with the given content. "
                "Fails if the file already exists — use squish_write_file to overwrite."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path for the new file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full UTF-8 content.",
                    },
                },
                "required": ["path", "content"],
            },
            fn=squish_create_file,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_create_directory",
            description=(
                "Create a directory, including any missing parents. "
                "Succeeds if the directory already exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path of the directory to create.",
                    },
                },
                "required": ["path"],
            },
            fn=squish_create_directory,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_delete_file",
            description=("Delete a file from disk. Irreversible — use with caution."),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path of the file to delete.",
                    },
                },
                "required": ["path"],
            },
            fn=squish_delete_file,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_move_file",
            description=(
                "Move or rename a file or directory. "
                "Parent directories of the destination are created automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "Absolute path of the source.",
                    },
                    "dst": {
                        "type": "string",
                        "description": "Absolute destination path.",
                    },
                },
                "required": ["src", "dst"],
            },
            fn=squish_move_file,
            source="builtin",
        ),
        ToolDefinition(
            name="squish_apply_edit",
            description=(
                "Apply a surgical text replacement in a file. "
                "Replaces old_text with new_text (must appear exactly once)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path of the file to edit.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (must be unique in the file).",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
            fn=squish_apply_edit,
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
                "Execute Python code in a restricted namespace, isolated in a "
                "memory- and CPU-limited subprocess. Returns captured stdout or "
                "an error / timeout / memory-limit message."
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
                        "description": "Maximum wall-clock seconds (default 10).",
                    },
                    "max_memory_mb": {
                        "type": "integer",
                        "description": "Child address-space cap in MiB (default 512).",
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
        ToolDefinition(
            name="squish_web_search",
            description=(
                "Search the web using DuckDuckGo and return result titles, URLs, "
                "and snippets. No API key required."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
            fn=squish_web_search,
            source="builtin",
        ),
    ]
    for defn in _tools:
        registry.register(defn)
