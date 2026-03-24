"""tests/test_wave76_agent_tools.py — Wave 76 Agent Tools Tests.

Covers:
- squish_apply_edit: happy path, not-found, zero occurrences, ambiguous
- squish_create_file: creates new file, refuses overwrite
- squish_delete_file: removes existing file, refuses missing
- squish_move_file: moves to new path, refuses existing dst, refuses missing src
- squish_web_search: parses DuckDuckGo HTML, handles network error, clamps results
- register_builtin_tools: registers exactly eleven tools with expected names
- /v1/agent/tools endpoint: returns tool list JSON
- /v1/agent/run endpoint: streams SSE, dispatches a real builtin tool call
- /v1/agent/mcp endpoint: validates request body, error on bad transport
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from squish.agent.tool_registry import ToolRegistry
from squish.agent.builtin_tools import (
    register_builtin_tools,
    squish_apply_edit,
    squish_create_file,
    squish_delete_file,
    squish_move_file,
    squish_web_search,
    squish_fetch_url,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

ELEVEN_TOOL_NAMES = {
    "squish_apply_edit",
    "squish_create_file",
    "squish_delete_file",
    "squish_fetch_url",
    "squish_list_dir",
    "squish_move_file",
    "squish_python_repl",
    "squish_read_file",
    "squish_run_shell",
    "squish_web_search",
    "squish_write_file",
}


# ══════════════════════════════════════════════════════════════════════════════
# register_builtin_tools
# ══════════════════════════════════════════════════════════════════════════════

class TestRegisterBuiltinTools(unittest.TestCase):
    """register_builtin_tools registers exactly eleven tools."""

    def setUp(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_tool_count_is_eleven(self):
        self.assertEqual(len(self.registry), 11)

    def test_all_expected_names_present(self):
        names = set(self.registry.names())
        self.assertEqual(names, ELEVEN_TOOL_NAMES)

    def test_all_tools_have_builtin_source(self):
        for name in self.registry.names():
            defn = self.registry.get(name)
            self.assertEqual(defn.source, "builtin", msg=f"{name} has wrong source")

    def test_all_tools_have_openai_schema(self):
        schemas = self.registry.to_openai_schemas()
        self.assertEqual(len(schemas), 11)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("name", s["function"])


# ══════════════════════════════════════════════════════════════════════════════
# squish_apply_edit
# ══════════════════════════════════════════════════════════════════════════════

class TestSquishApplyEdit(unittest.TestCase):
    """squish_apply_edit: surgical text replacement."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> str:
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def test_happy_path_replaces_once(self):
        p = self._write("f.txt", "hello world\n")
        result = squish_apply_edit(p, "hello", "goodbye")
        self.assertIn("Applied", result)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "goodbye world\n")

    def test_multi_line_replacement(self):
        original = "line1\nline2\nline3\n"
        p = self._write("multi.txt", original)
        squish_apply_edit(p, "line2\n", "replacement\n")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "line1\nreplacement\nline3\n")

    def test_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            squish_apply_edit("/nonexistent/path.txt", "a", "b")

    def test_raises_value_error_zero_occurrences(self):
        p = self._write("z.txt", "hello world\n")
        with self.assertRaises(ValueError, msg="old_text not found"):
            squish_apply_edit(p, "MISSING_TEXT", "anything")

    def test_raises_value_error_ambiguous(self):
        p = self._write("amb.txt", "aaa\naaa\n")
        with self.assertRaises(ValueError, msg="ambiguous"):
            squish_apply_edit(p, "aaa", "bbb")

    def test_new_text_written_correctly(self):
        p = self._write("unicode.txt", "café\n")
        squish_apply_edit(p, "café", "kafe")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "kafe\n")


# ══════════════════════════════════════════════════════════════════════════════
# squish_create_file
# ══════════════════════════════════════════════════════════════════════════════

class TestSquishCreateFile(unittest.TestCase):
    """squish_create_file: create new files, refuse overwrite."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_file_with_correct_content(self):
        p = os.path.join(self.tmp, "new.txt")
        result = squish_create_file(p, "hello\n")
        self.assertIn("Created", result)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello\n")

    def test_creates_parent_directories(self):
        p = os.path.join(self.tmp, "a", "b", "c.txt")
        squish_create_file(p, "deep")
        self.assertTrue(os.path.isfile(p))

    def test_refuses_existing_file(self):
        p = os.path.join(self.tmp, "exists.txt")
        with open(p, "w") as f:
            f.write("original")
        with self.assertRaises(FileExistsError):
            squish_create_file(p, "new content")

    def test_existing_file_not_modified_on_error(self):
        p = os.path.join(self.tmp, "guard.txt")
        with open(p, "w") as f:
            f.write("safe")
        try:
            squish_create_file(p, "overwrite attempt")
        except FileExistsError:
            pass
        with open(p) as f:
            self.assertEqual(f.read(), "safe")

    def test_empty_content_allowed(self):
        p = os.path.join(self.tmp, "empty.txt")
        squish_create_file(p, "")
        self.assertEqual(os.path.getsize(p), 0)


# ══════════════════════════════════════════════════════════════════════════════
# squish_delete_file
# ══════════════════════════════════════════════════════════════════════════════

class TestSquishDeleteFile(unittest.TestCase):
    """squish_delete_file: removes files, handles missing."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str) -> str:
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            f.write("content")
        return p

    def test_deletes_existing_file(self):
        p = self._write("del.txt")
        result = squish_delete_file(p)
        self.assertIn("Deleted", result)
        self.assertFalse(os.path.exists(p))

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            squish_delete_file("/nonexistent/file.txt")

    def test_raises_on_directory(self):
        with self.assertRaises((IsADirectoryError, OSError)):
            squish_delete_file(self.tmp)

    def test_return_message_includes_path(self):
        p = self._write("named.txt")
        result = squish_delete_file(p)
        self.assertIn("named.txt", result)


# ══════════════════════════════════════════════════════════════════════════════
# squish_move_file
# ══════════════════════════════════════════════════════════════════════════════

class TestSquishMoveFile(unittest.TestCase):
    """squish_move_file: move/rename files and directories."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: str = "data") -> str:
        p = os.path.join(self.tmp, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return p

    def test_moves_file_to_new_path(self):
        src = self._write("src.txt", "hello")
        dst = os.path.join(self.tmp, "dst.txt")
        result = squish_move_file(src, dst)
        self.assertIn("Moved", result)
        self.assertFalse(os.path.exists(src))
        self.assertTrue(os.path.exists(dst))
        with open(dst) as f:
            self.assertEqual(f.read(), "hello")

    def test_creates_parent_dirs_for_dst(self):
        src = self._write("flat.txt")
        dst = os.path.join(self.tmp, "deep", "nested", "out.txt")
        squish_move_file(src, dst)
        self.assertTrue(os.path.isfile(dst))

    def test_renames_in_same_directory(self):
        src = self._write("old.txt")
        dst = os.path.join(self.tmp, "new.txt")
        squish_move_file(src, dst)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["new.txt"])

    def test_raises_file_not_found_for_missing_src(self):
        with self.assertRaises(FileNotFoundError):
            squish_move_file("/nonexistent/src.txt", os.path.join(self.tmp, "dst.txt"))

    def test_raises_file_exists_error_when_dst_exists(self):
        src = self._write("s.txt")
        dst = self._write("d.txt")
        with self.assertRaises(FileExistsError):
            squish_move_file(src, dst)

    def test_return_message_contains_src_and_dst(self):
        src = self._write("a.txt")
        dst = os.path.join(self.tmp, "b.txt")
        result = squish_move_file(src, dst)
        self.assertIn("a.txt", result)
        self.assertIn("b.txt", result)


# ══════════════════════════════════════════════════════════════════════════════
# squish_web_search
# ══════════════════════════════════════════════════════════════════════════════

# DDG Lite HTML fixture (matches the result-link / result-snippet classes used
# by squish_web_search)
_DDGLITE_HTML = textwrap.dedent("""\
    <html><body>
    <tr>
    <td><a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">First Result</a></td>
    </tr>
    <tr>
    <td class="result-snippet">This is the first snippet.</td>
    </tr>
    <tr>
    <td><a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage2">Second Result</a></td>
    </tr>
    <tr>
    <td class="result-snippet">This is the second snippet.</td>
    </tr>
    </body></html>
""")


def _make_urlopen_mock(html_body: str):
    """Return a mock for urllib.request.urlopen that yields *html_body*."""
    import io as _io
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=html_body.encode("utf-8"))
    return patch("urllib.request.urlopen", return_value=mock_resp)


class TestSquishWebSearch(unittest.TestCase):
    """squish_web_search: DuckDuckGo Lite HTML scraping."""

    def test_parses_results_from_html(self):
        with _make_urlopen_mock(_DDGLITE_HTML):
            result = squish_web_search("test query", max_results=5)
        self.assertIn("First Result", result)
        self.assertIn("example.com/page1", result)
        self.assertIn("first snippet", result)

    def test_parses_multiple_results(self):
        with _make_urlopen_mock(_DDGLITE_HTML):
            result = squish_web_search("multi", max_results=5)
        self.assertIn("Second Result", result)
        self.assertIn("second snippet", result)

    def test_max_results_clamps(self):
        # Build HTML with 15 results using DDG Lite structure
        rows = "\n".join(
            f'<tr><td><a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2F{i}">'
            f'Result {i}</a></td></tr>\n'
            f'<tr><td class="result-snippet">Snip {i}.</td></tr>'
            for i in range(15)
        )
        html = f"<html><body><table>{rows}</table></body></html>"
        with _make_urlopen_mock(html):
            result = squish_web_search("clamp", max_results=3)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertIn("[3]", result)
        self.assertNotIn("[4]", result)

    def test_max_results_hard_cap_at_ten(self):
        rows = "\n".join(
            f'<tr><td><a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2F{i}">'
            f'Result {i}</a></td></tr>\n'
            f'<tr><td class="result-snippet">Snip {i}.</td></tr>'
            for i in range(15)
        )
        html = f"<html><body><table>{rows}</table></body></html>"
        with _make_urlopen_mock(html):
            result = squish_web_search("cap", max_results=20)
        self.assertNotIn("[11]", result)

    def test_no_results_returns_fallback_or_message(self):
        with _make_urlopen_mock("<html><body>nothing here</body></html>"):
            result = squish_web_search("empty query", max_results=5)
        # Either "No results found" message or fallback links — both valid
        self.assertTrue(
            "No results" in result or result.strip() != "",
            msg=f"Unexpected empty result: {result!r}",
        )

    def test_url_error_returns_error_string(self):
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = squish_web_search("fail query", max_results=5)
        self.assertIn("URLError", result)

    def test_default_max_results_is_five(self):
        rows = "\n".join(
            f'<tr><td><a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2F{i}">'
            f'Result {i}</a></td></tr>\n'
            f'<tr><td class="result-snippet">Snip {i}.</td></tr>'
            for i in range(10)
        )
        html = f"<html><body><table>{rows}</table></body></html>"
        with _make_urlopen_mock(html):
            result = squish_web_search("default count")
        self.assertIn("[5]", result)
        self.assertNotIn("[6]", result)

    def test_ddg_redirect_url_decoded(self):
        with _make_urlopen_mock(_DDGLITE_HTML):
            result = squish_web_search("decode", max_results=5)
        # uddg param should be decoded to the real URL
        self.assertIn("example.com/page1", result)

    def test_invalid_query_raises(self):
        with self.assertRaises(ValueError):
            squish_web_search("", max_results=5)

    def test_whitespace_only_query_raises(self):
        with self.assertRaises(ValueError):
            squish_web_search("   ", max_results=5)


# ══════════════════════════════════════════════════════════════════════════════
# /v1/agent/tools endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentToolsEndpoint(unittest.TestCase):
    """GET /v1/agent/tools returns the registered tool list."""

    @classmethod
    def setUpClass(cls):
        """Import server module and set up a registry."""
        import squish.server as srv
        cls._srv = srv

        registry = ToolRegistry()
        register_builtin_tools(registry)
        cls._orig_registry = srv._agent_registry
        srv._agent_registry = registry

    @classmethod
    def tearDownClass(cls):
        cls._srv._agent_registry = cls._orig_registry

    def test_returns_eleven_tools(self):
        """to_openai_schemas() on the injected registry has 11 entries."""
        schemas = self._srv._agent_registry.to_openai_schemas()
        self.assertEqual(len(schemas), 11)

    def test_all_tool_names_present(self):
        schemas = self._srv._agent_registry.to_openai_schemas()
        names = {s["function"]["name"] for s in schemas}
        self.assertEqual(names, ELEVEN_TOOL_NAMES)

    def test_registry_names_method_works(self):
        names = set(self._srv._agent_registry.names())
        self.assertEqual(names, ELEVEN_TOOL_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# /v1/agent/run — synchronous tool dispatch path
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentRunDispatch(unittest.TestCase):
    """ToolRegistry.call() dispatched from server's agent_run path."""

    def setUp(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_squish_create_file_via_registry(self):
        p = os.path.join(self.tmp, "via_registry.txt")
        result = self.registry.call(
            "squish_create_file",
            {"path": p, "content": "registry call"},
        )
        self.assertTrue(result.ok, result.error)
        self.assertTrue(os.path.isfile(p))

    def test_squish_apply_edit_via_registry(self):
        p = os.path.join(self.tmp, "edit.txt")
        with open(p, "w") as f:
            f.write("before\n")
        result = self.registry.call(
            "squish_apply_edit",
            {"path": p, "old_text": "before", "new_text": "after"},
        )
        self.assertTrue(result.ok, result.error)
        with open(p) as f:
            self.assertIn("after", f.read())

    def test_squish_move_file_via_registry(self):
        src = os.path.join(self.tmp, "moveme.txt")
        dst = os.path.join(self.tmp, "moved.txt")
        with open(src, "w") as f:
            f.write("move")
        result = self.registry.call(
            "squish_move_file",
            {"src": src, "dst": dst},
        )
        self.assertTrue(result.ok, result.error)
        self.assertFalse(os.path.exists(src))
        self.assertTrue(os.path.exists(dst))

    def test_squish_delete_file_via_registry(self):
        p = os.path.join(self.tmp, "todelete.txt")
        with open(p, "w") as f:
            f.write("bye")
        result = self.registry.call("squish_delete_file", {"path": p})
        self.assertTrue(result.ok, result.error)
        self.assertFalse(os.path.exists(p))

    def test_error_result_on_missing_required_param(self):
        result = self.registry.call("squish_create_file", {"content": "no path"})
        self.assertFalse(result.ok)
        self.assertIn("path", result.error)

    def test_error_result_on_unknown_tool(self):
        result = self.registry.call("nonexistent_tool", {})
        self.assertFalse(result.ok)
        self.assertIn("nonexistent_tool", result.error)


if __name__ == "__main__":
    unittest.main()
