"""tests/test_wave72_agent_engine.py — Wave 72 Agent Engine Tests.

Covers:
- ToolRegistry: registration, decorator, validation, dispatch, edge cases
- ToolResult: ok property, to_message serialisation
- ToolDefinition: to_openai_schema
- builtin_tools: squish_read_file, squish_write_file, squish_list_dir,
  squish_run_shell, squish_python_repl, squish_fetch_url
- register_builtin_tools: correct count and names
- CORSConfig + apply_cors_headers + is_origin_allowed
- MCPClient: unit-level (no external process required)
- MCPToolDef: dataclass fields
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure repo root is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from squish.agent.tool_registry import (
    ToolCallError,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from squish.agent.builtin_tools import (
    register_builtin_tools,
    squish_fetch_url,
    squish_list_dir,
    squish_python_repl,
    squish_read_file,
    squish_run_shell,
    squish_write_file,
)
from squish.serving.cors_config import (
    DEFAULT_CORS,
    CORSConfig,
    apply_cors_headers,
    is_origin_allowed,
)
from squish.serving.mcp_client import MCPToolDef, MCPTransport


# ══════════════════════════════════════════════════════════════════════════════
# ToolDefinition
# ══════════════════════════════════════════════════════════════════════════════

class TestToolDefinition(unittest.TestCase):
    """ToolDefinition dataclass and to_openai_schema()."""

    def _make(self, name: str = "echo", description: str = "Echo tool"):
        return ToolDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            fn=lambda text: text,
        )

    def test_fields_set_correctly(self):
        d = self._make("my_tool", "does things")
        self.assertEqual(d.name, "my_tool")
        self.assertEqual(d.description, "does things")
        self.assertEqual(d.source, "user")

    def test_custom_source(self):
        d = ToolDefinition(
            name="t", description="d",
            parameters={"type": "object", "properties": {}},
            fn=lambda: None,
            source="builtin",
        )
        self.assertEqual(d.source, "builtin")

    def test_to_openai_schema_structure(self):
        d = self._make()
        schema = d.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        fn = schema["function"]
        self.assertEqual(fn["name"], "echo")
        self.assertEqual(fn["description"], "Echo tool")
        self.assertIn("parameters", fn)
        self.assertEqual(fn["parameters"]["type"], "object")

    def test_to_openai_schema_parameters_pass_through(self):
        d = self._make()
        fn = d.to_openai_schema()["function"]
        self.assertIn("text", fn["parameters"]["properties"])


# ══════════════════════════════════════════════════════════════════════════════
# ToolResult
# ══════════════════════════════════════════════════════════════════════════════

class TestToolResult(unittest.TestCase):
    """ToolResult dataclass, ok property, to_message."""

    def test_ok_true_when_no_error(self):
        r = ToolResult(tool_name="t", call_id="c1", output="hello")
        self.assertTrue(r.ok)
        self.assertIsNone(r.error)

    def test_ok_false_when_error(self):
        r = ToolResult(tool_name="t", call_id="c1", output=None, error="bad")
        self.assertFalse(r.ok)

    def test_to_message_success(self):
        r = ToolResult(tool_name="t", call_id="c1", output="result text")
        msg = r.to_message()
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "c1")
        self.assertEqual(msg["content"], "result text")

    def test_to_message_error(self):
        r = ToolResult(tool_name="t", call_id="c2", output=None, error="something went wrong")
        msg = r.to_message()
        self.assertIn("[ERROR]", msg["content"])
        self.assertIn("something went wrong", msg["content"])

    def test_to_message_dict_output_serialised(self):
        r = ToolResult(tool_name="t", call_id="c3", output={"key": "val"})
        msg = r.to_message()
        self.assertIn('"key"', msg["content"])

    def test_elapsed_defaults_to_zero(self):
        r = ToolResult(tool_name="t", call_id="c4", output="x")
        self.assertEqual(r.elapsed_ms, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    """ToolRegistry registration, validation, and dispatch."""

    def setUp(self):
        self.registry = ToolRegistry()

    def _add_echo(self, name: str = "echo"):
        defn = ToolDefinition(
            name=name,
            description="Echo input",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            fn=lambda text: f"echo:{text}",
        )
        self.registry.register(defn)
        return defn

    # ── Registration ──────────────────────────────────────────────────────────

    def test_register_adds_tool(self):
        self._add_echo()
        self.assertIn("echo", self.registry)

    def test_register_duplicate_raises(self):
        self._add_echo()
        with self.assertRaises(ValueError):
            self._add_echo()

    def test_len(self):
        self.assertEqual(len(self.registry), 0)
        self._add_echo("t1")
        self._add_echo("t2")
        self.assertEqual(len(self.registry), 2)

    def test_names_sorted(self):
        self._add_echo("z_tool")
        self._add_echo("a_tool")
        names = self.registry.names()
        self.assertEqual(names, sorted(names))

    def test_get_existing(self):
        self._add_echo()
        self.assertIsNotNone(self.registry.get("echo"))

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_unregister(self):
        self._add_echo()
        self.registry.unregister("echo")
        self.assertNotIn("echo", self.registry)

    def test_unregister_noop_if_missing(self):
        self.registry.unregister("ghost")  # should not raise

    def test_clear(self):
        self._add_echo("a")
        self._add_echo("b")
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)

    # ── Decorator ────────────────────────────────────────────────────────────

    def test_decorator_registers(self):
        @self.registry.tool(description="Add nums", parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        })
        def add(a: int, b: int) -> int:
            return a + b
        self.assertIn("add", self.registry)

    def test_decorator_name_override(self):
        @self.registry.tool(name="custom_name", description="x", parameters={
            "type": "object", "properties": {}
        })
        def my_fn():
            return "ok"
        self.assertIn("custom_name", self.registry)
        self.assertNotIn("my_fn", self.registry)

    def test_decorator_returns_original_fn(self):
        @self.registry.tool(description="test", parameters={"type":"object","properties":{}})
        def noop():
            return 42
        self.assertEqual(noop(), 42)

    # ── Validation ───────────────────────────────────────────────────────────

    def test_validate_missing_required(self):
        self._add_echo()
        with self.assertRaises(ToolCallError):
            self.registry.validate_call("echo", {})  # "text" is required

    def test_validate_wrong_type(self):
        self._add_echo()
        with self.assertRaises(ToolCallError):
            self.registry.validate_call("echo", {"text": 123})  # should be string

    def test_validate_unknown_tool(self):
        with self.assertRaises(ToolCallError):
            self.registry.validate_call("ghost", {})

    def test_validate_enum_pass(self):
        defn = ToolDefinition(
            name="mode_tool", description="d",
            parameters={
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
                "required": ["mode"],
            },
            fn=lambda mode: mode,
        )
        self.registry.register(defn)
        self.registry.validate_call("mode_tool", {"mode": "a"})  # should not raise

    def test_validate_enum_fail(self):
        defn = ToolDefinition(
            name="mode_tool2", description="d",
            parameters={
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
                "required": ["mode"],
            },
            fn=lambda mode: mode,
        )
        self.registry.register(defn)
        with self.assertRaises(ToolCallError):
            self.registry.validate_call("mode_tool2", {"mode": "z"})

    # ── Dispatch ────────────────────────────────────────────────────────────

    def test_call_success(self):
        self._add_echo()
        result = self.registry.call("echo", {"text": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "echo:hello")

    def test_call_unknown_tool_returns_error_result(self):
        result = self.registry.call("ghost", {})
        self.assertFalse(result.ok)
        self.assertIn("Unknown", result.error)

    def test_call_exception_captured(self):
        def boom(**_):
            raise RuntimeError("explode")
        defn = ToolDefinition(
            name="bomb", description="d",
            parameters={"type": "object", "properties": {}},
            fn=boom,
        )
        self.registry.register(defn)
        result = self.registry.call("bomb", {}, validate=False)
        self.assertFalse(result.ok)
        self.assertIn("explode", result.error)

    def test_call_elapsed_ms_positive(self):
        self._add_echo()
        result = self.registry.call("echo", {"text": "x"})
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_call_custom_call_id(self):
        self._add_echo()
        result = self.registry.call("echo", {"text": "x"}, call_id="my-id")
        self.assertEqual(result.call_id, "my-id")

    # ── OpenAI schemas ───────────────────────────────────────────────────────

    def test_to_openai_schemas_empty(self):
        schemas = self.registry.to_openai_schemas()
        self.assertIsInstance(schemas, list)
        self.assertEqual(len(schemas), 0)

    def test_to_openai_schemas_populated(self):
        self._add_echo("t1")
        self._add_echo("t2")
        schemas = self.registry.to_openai_schemas()
        self.assertEqual(len(schemas), 2)
        names = {s["function"]["name"] for s in schemas}
        self.assertEqual(names, {"t1", "t2"})


# ══════════════════════════════════════════════════════════════════════════════
# Built-in tools
# ══════════════════════════════════════════════════════════════════════════════

class TestBuiltinToolReadFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test.txt")
        with open(self.path, "w") as f:
            for i in range(20):
                f.write(f"line {i+1}\n")

    def test_read_all_lines(self):
        result = squish_read_file(self.path, start_line=1, end_line=20)
        self.assertIn("line 1", result)
        self.assertIn("line 20", result)

    def test_read_window(self):
        result = squish_read_file(self.path, start_line=5, end_line=10)
        self.assertIn("line 5", result)
        self.assertNotIn("line 4", result)
        self.assertIn("line 10", result)

    def test_header_present(self):
        result = squish_read_file(self.path)
        self.assertIn("# Lines", result)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            squish_read_file("/nonexistent/path/file.txt")

    def test_null_byte_in_path_raises(self):
        with self.assertRaises(ValueError):
            squish_read_file("/tmp/\x00bad")


class TestBuiltinToolWriteFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_write_creates_file(self):
        path = os.path.join(self.tmpdir, "out.txt")
        squish_write_file(path, "hello world")
        with open(path) as f:
            self.assertEqual(f.read(), "hello world")

    def test_write_returns_confirmation(self):
        path = os.path.join(self.tmpdir, "out2.txt")
        result = squish_write_file(path, "data")
        self.assertIn("bytes", result)

    def test_write_creates_parent_dirs(self):
        path = os.path.join(self.tmpdir, "sub", "deep", "file.txt")
        squish_write_file(path, "test")
        self.assertTrue(os.path.exists(path))

    def test_write_overwrites_existing(self):
        path = os.path.join(self.tmpdir, "over.txt")
        squish_write_file(path, "first")
        squish_write_file(path, "second")
        with open(path) as f:
            self.assertEqual(f.read(), "second")


class TestBuiltinToolListDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        open(os.path.join(self.tmpdir, "a.txt"), "w").close()
        os.makedirs(os.path.join(self.tmpdir, "sub"))

    def test_lists_files_and_dirs(self):
        result = squish_list_dir(self.tmpdir)
        self.assertIn("[FILE]", result)
        self.assertIn("[DIR]", result)
        self.assertIn("a.txt", result)
        self.assertIn("sub/", result)

    def test_shows_item_count(self):
        result = squish_list_dir(self.tmpdir)
        self.assertIn("items total", result)

    def test_not_a_directory_raises(self):
        path = os.path.join(self.tmpdir, "a.txt")
        with self.assertRaises(NotADirectoryError):
            squish_list_dir(path)

    def test_nonexistent_raises(self):
        with self.assertRaises(NotADirectoryError):
            squish_list_dir("/nonexistent/xyz")


class TestBuiltinToolRunShell(unittest.TestCase):

    def test_simple_command(self):
        result = squish_run_shell("echo hello_wave72")
        self.assertIn("hello_wave72", result)

    def test_exit_code_shown(self):
        result = squish_run_shell("exit 0")
        self.assertIn("[exit 0]", result)

    def test_nonzero_exit(self):
        result = squish_run_shell("exit 1")
        self.assertIn("[exit 1]", result)

    def test_stderr_captured(self):
        result = squish_run_shell("echo err >&2")
        self.assertIn("err", result)

    def test_empty_command_raises(self):
        with self.assertRaises(ValueError):
            squish_run_shell("")

    def test_timeout_returns_message(self):
        result = squish_run_shell("sleep 10", timeout=1)
        self.assertIn("TIMEOUT", result)


class TestBuiltinToolPythonRepl(unittest.TestCase):

    def test_print_captured(self):
        result = squish_python_repl("print('wave72')")
        self.assertIn("wave72", result)

    def test_expression_result_not_captured_without_print(self):
        result = squish_python_repl("1 + 1")
        # No output — Python exec doesn't print expressions
        self.assertEqual(result, "[no output]")

    def test_error_traceback(self):
        result = squish_python_repl("raise ValueError('oops')")
        self.assertIn("[ERROR]", result)
        self.assertIn("oops", result)

    def test_multiline_code(self):
        code = "total = 0\nfor i in range(5):\n  total += i\nprint(total)"
        result = squish_python_repl(code)
        self.assertIn("10", result)

    def test_empty_code_raises(self):
        with self.assertRaises(ValueError):
            squish_python_repl("")

    def test_no_output_returns_placeholder(self):
        result = squish_python_repl("x = 1")
        self.assertEqual(result, "[no output]")


class TestBuiltinToolFetchUrl(unittest.TestCase):

    def test_file_scheme_blocked(self):
        with self.assertRaises(ValueError):
            squish_fetch_url("file:///etc/passwd")

    def test_empty_host_blocked(self):
        with self.assertRaises(ValueError):
            squish_fetch_url("http://")

    def test_non_http_scheme_blocked(self):
        with self.assertRaises(ValueError):
            squish_fetch_url("ftp://example.com/file")

    def test_http_error_returns_message(self):
        import urllib.error
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.HTTPError(
                url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
            )
            result = squish_fetch_url("http://example.com")
            self.assertIn("404", result)

    def test_url_error_returns_message(self):
        import urllib.error
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("network unreachable")
            result = squish_fetch_url("http://fake-host.invalid")
            self.assertIn("URLError", result)

    def test_truncation_notice(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"x" * (131073)
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = squish_fetch_url("http://example.com", max_bytes=131072)
        self.assertIn("TRUNCATED", result)


class TestRegisterBuiltinTools(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_six_tools_registered(self):
        # Wave 76 extended the tool set from 6 to 11
        self.assertEqual(len(self.registry), 11)

    def test_all_names_present(self):
        expected = {
            "squish_read_file", "squish_write_file", "squish_list_dir",
            "squish_run_shell", "squish_python_repl", "squish_fetch_url",
            # Wave 76 additions
            "squish_apply_edit", "squish_create_file", "squish_delete_file",
            "squish_move_file", "squish_web_search",
        }
        self.assertEqual(set(self.registry.names()), expected)

    def test_all_tools_are_builtin(self):
        for name in self.registry.names():
            self.assertEqual(self.registry.get(name).source, "builtin")

    def test_double_register_raises(self):
        with self.assertRaises(ValueError):
            register_builtin_tools(self.registry)

    def test_openai_schemas_valid(self):
        schemas = self.registry.to_openai_schemas()
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("function", schema)
            fn = schema["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)


# ══════════════════════════════════════════════════════════════════════════════
# CORSConfig + CORS helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestCORSConfig(unittest.TestCase):

    def test_wildcard_default(self):
        cfg = CORSConfig()
        self.assertIn("*", cfg.allowed_origins)

    def test_credentials_with_wildcard_raises(self):
        with self.assertRaises(ValueError):
            CORSConfig(allowed_origins=["*"], allow_credentials=True)

    def test_credentials_with_explicit_origin_ok(self):
        cfg = CORSConfig(allowed_origins=["https://app.example.com"], allow_credentials=True)
        self.assertTrue(cfg.allow_credentials)

    def test_default_cors_singleton(self):
        self.assertIsInstance(DEFAULT_CORS, CORSConfig)
        self.assertEqual(DEFAULT_CORS.allowed_origins, ["*"])


class TestIsOriginAllowed(unittest.TestCase):

    def test_wildcard_allows_any(self):
        self.assertTrue(is_origin_allowed("https://anything.com", ["*"]))

    def test_exact_match(self):
        self.assertTrue(is_origin_allowed("https://app.example.com", ["https://app.example.com"]))

    def test_exact_mismatch(self):
        self.assertFalse(is_origin_allowed("https://evil.com", ["https://good.com"]))

    def test_empty_origin_denied(self):
        self.assertFalse(is_origin_allowed("", ["*"]))

    def test_multiple_origins(self):
        allowed = ["https://a.com", "https://b.com"]
        self.assertTrue(is_origin_allowed("https://b.com", allowed))
        self.assertFalse(is_origin_allowed("https://c.com", allowed))


class TestApplyCORSHeaders(unittest.TestCase):

    def test_wildcard_origin_header(self):
        headers = {}
        apply_cors_headers(headers, "https://any.com", CORSConfig(allowed_origins=["*"]))
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

    def test_specific_origin_echoed(self):
        headers = {}
        cfg = CORSConfig(allowed_origins=["https://app.example.com"])
        apply_cors_headers(headers, "https://app.example.com", cfg)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://app.example.com")

    def test_disallowed_origin_no_headers(self):
        headers = {}
        cfg = CORSConfig(allowed_origins=["https://safe.com"])
        apply_cors_headers(headers, "https://evil.com", cfg)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_none_origin_no_headers(self):
        headers = {}
        apply_cors_headers(headers, None, CORSConfig())
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_preflight_headers(self):
        headers = {}
        cfg = CORSConfig()
        apply_cors_headers(headers, "https://any.com", cfg, is_preflight=True)
        self.assertIn("Access-Control-Allow-Methods", headers)
        self.assertIn("Access-Control-Allow-Headers", headers)
        self.assertIn("Access-Control-Max-Age", headers)

    def test_no_preflight_missing_preflight_headers(self):
        headers = {}
        apply_cors_headers(headers, "https://any.com", CORSConfig())
        self.assertNotIn("Access-Control-Allow-Methods", headers)

    def test_vary_header_added_for_specific_origins(self):
        headers = {}
        cfg = CORSConfig(allowed_origins=["https://app.com"])
        apply_cors_headers(headers, "https://app.com", cfg)
        self.assertIn("Origin", headers.get("Vary", ""))

    def test_credentials_header(self):
        headers = {}
        cfg = CORSConfig(allowed_origins=["https://app.com"], allow_credentials=True)
        apply_cors_headers(headers, "https://app.com", cfg)
        self.assertEqual(headers.get("Access-Control-Allow-Credentials"), "true")

    def test_expose_headers(self):
        headers = {}
        cfg = CORSConfig(expose_headers=["X-Request-ID"])
        apply_cors_headers(headers, "https://any.com", cfg)
        self.assertIn("X-Request-ID", headers.get("Access-Control-Expose-Headers", ""))


# ══════════════════════════════════════════════════════════════════════════════
# MCPToolDef + MCPTransport
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPTypes(unittest.TestCase):

    def test_mcp_tool_def_fields(self):
        td = MCPToolDef(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_id="fs_server",
        )
        self.assertEqual(td.name, "read_file")
        self.assertEqual(td.server_id, "fs_server")

    def test_mcp_tool_def_defaults(self):
        td = MCPToolDef(name="x", description="", input_schema={})
        self.assertEqual(td.server_id, "")

    def test_transport_enum_values(self):
        self.assertEqual(MCPTransport.STDIO.value, "stdio")
        self.assertEqual(MCPTransport.SSE.value, "sse")

    def test_transport_from_value(self):
        self.assertEqual(MCPTransport("stdio"), MCPTransport.STDIO)
        self.assertEqual(MCPTransport("sse"), MCPTransport.SSE)


if __name__ == "__main__":
    unittest.main()
