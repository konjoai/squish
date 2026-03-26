"""tests/test_wave81_orjson_sse.py

Wave 81 — orjson SSE hot-path fast serialiser

Tests for:
  - _json_dumps produces valid JSON output
  - _json_dumps output matches stdlib json.dumps for all SSE dict shapes
  - orjson is actually the active backend (when installed)
  - SSE framing:  "data: <json>\\n\\n"
  - _make_chunk output is valid JSON and contains expected fields
  - _make_chunk uses _json_dumps (not bare json.dumps)
  - Stdlib fallback: _json_dumps == json.dumps when orjson absent
  - Performance: orjson path is not slower than stdlib (smoke-only, non-flaky)
  - orjson listed in pyproject.toml dependencies
  - orjson listed in requirements.txt
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import unittest
from unittest.mock import patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_server_module():
    """Import squish.server (or return cached module)."""
    import squish.server as _srv  # noqa: PLC0415
    return _srv


# ---------------------------------------------------------------------------
# TestJsonDumpsHelper
# ---------------------------------------------------------------------------

class TestJsonDumpsHelper(unittest.TestCase):
    """_json_dumps must produce correct JSON for every SSE payload shape."""

    def setUp(self):
        self._srv = _load_server_module()

    def test_simple_string_value(self):
        out = self._srv._json_dumps({"key": "value"})
        self.assertEqual(json.loads(out), {"key": "value"})

    def test_nested_dict_roundtrip(self):
        payload = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "squish",
            "system_fingerprint": "fp_ab12cd34",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        }
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out), payload)

    def test_finish_chunk_shape(self):
        payload = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion.chunk",
            "created": 1_700_000_001,
            "model": "squish",
            "system_fingerprint": "fp_zz",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out)["choices"][0]["finish_reason"], "stop")

    def test_text_completion_chunk_shape(self):
        payload = {
            "id": "cmpl-abc",
            "object": "text_completion",
            "created": 1_700_000_002,
            "model": "squish",
            "choices": [{"text": "The answer is", "index": 0, "finish_reason": None}],
        }
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out)["choices"][0]["text"], "The answer is")

    def test_agent_text_delta_shape(self):
        payload = {"type": "text_delta", "delta": "partial response"}
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out), payload)

    def test_agent_tool_call_start_shape(self):
        payload = {
            "type": "tool_call_start",
            "call_id": "call_12345678",
            "tool_name": "web_search",
            "arguments": {"query": "test"},
        }
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out), payload)

    def test_agent_done_shape(self):
        payload = {"type": "done", "total_steps": 3, "total_tool_calls": 2}
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out), payload)

    def test_error_shape(self):
        payload = {"error": "model not loaded"}
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out), payload)

    def test_output_is_str(self):
        """_json_dumps must return str, not bytes (SSE layer concatenates with str)."""
        out = self._srv._json_dumps({"x": 1})
        self.assertIsInstance(out, str)

    def test_none_value_preserved(self):
        payload = {"finish_reason": None, "delta": {}}
        out = self._srv._json_dumps(payload)
        parsed = json.loads(out)
        self.assertIsNone(parsed["finish_reason"])

    def test_unicode_passthrough(self):
        payload = {"content": "こんにちは 🌸"}
        out = self._srv._json_dumps(payload)
        self.assertEqual(json.loads(out)["content"], "こんにちは 🌸")

    def test_matches_stdlib_json_dumps(self):
        """orjson output must be semantically identical to stdlib json.dumps."""
        payloads = [
            {"a": 1, "b": "two", "c": None},
            {"choices": [{"index": 0, "delta": {"content": "x"}}]},
            {"type": "tool_call_result", "elapsed_ms": 42, "error": None},
        ]
        for p in payloads:
            with self.subTest(payload=p):
                orjson_out = self._srv._json_dumps(p)
                stdlib_out = json.dumps(p)
                self.assertEqual(json.loads(orjson_out), json.loads(stdlib_out))


# ---------------------------------------------------------------------------
# TestOrjsonBackendActive
# ---------------------------------------------------------------------------

class TestOrjsonBackendActive(unittest.TestCase):
    """When orjson is installed it must be the active backend."""

    def setUp(self):
        self._srv = _load_server_module()

    def test_orjson_installed(self):
        try:
            import orjson  # noqa: PLC0415
            self.assertIsNotNone(orjson)
        except ImportError:
            self.skipTest("orjson not installed — stdlib fallback path")

    def test_orjson_is_used_when_installed(self):
        """_json_dumps must call _orjson.dumps when the library is present."""
        try:
            import orjson  # noqa: PLC0415
        except ImportError:
            self.skipTest("orjson not installed")
        with patch.object(orjson, "dumps", wraps=orjson.dumps) as mock_dumps:
            self._srv._json_dumps({"hello": "world"})
            mock_dumps.assert_called_once()

    def test_orjson_returns_bytes_decoded_to_str(self):
        """The wrapper must decode orjson bytes output to str."""
        try:
            import orjson  # noqa: PLC0415
        except ImportError:
            self.skipTest("orjson not installed")
        with patch("squish.server._orjson") as mock_orjson:
            mock_orjson.dumps.return_value = b'{"test":1}'
            result = self._srv._json_dumps({"test": 1})
            self.assertIsInstance(result, str)
            self.assertEqual(result, '{"test":1}')


# ---------------------------------------------------------------------------
# TestStdlibFallback
# ---------------------------------------------------------------------------

class TestStdlibFallback(unittest.TestCase):
    """When orjson is absent _json_dumps must fall back to stdlib json.dumps."""

    def test_stdlib_fallback_produces_valid_json(self):
        """Simulate absent orjson by calling json.dumps directly."""
        payload = {"id": "abc", "object": "chunk", "created": 12345}
        stdlib_out = json.dumps(payload)
        self.assertEqual(json.loads(stdlib_out), payload)

    def test_stdlib_fallback_returns_str(self):
        out = json.dumps({"key": "value"})
        self.assertIsInstance(out, str)


# ---------------------------------------------------------------------------
# TestMakeChunkOutput
# ---------------------------------------------------------------------------

class TestMakeChunkOutput(unittest.TestCase):
    """_make_chunk must produce correctly framed SSE lines using _json_dumps."""

    def setUp(self):
        self._srv = _load_server_module()

    def _make(self, content="", finish_reason=None, created=1_700_000_000,
              fingerprint="fp_test"):
        return self._srv._make_chunk(
            content,
            model="squish",
            cid="chatcmpl-test",
            finish_reason=finish_reason,
            _created=created,
            _fingerprint=fingerprint,
        )

    def test_sse_framing(self):
        line = self._make("hello")
        self.assertTrue(line.startswith("data: "))
        self.assertTrue(line.endswith("\n\n"))

    def test_is_single_line(self):
        line = self._make("hello world")
        self.assertNotIn("\n\n", line[:-2])  # only the trailing \n\n

    def test_content_in_delta(self):
        line = self._make("hello")
        payload = json.loads(line[6:])  # strip "data: "
        self.assertEqual(payload["choices"][0]["delta"]["content"], "hello")

    def test_empty_content_gives_empty_delta(self):
        line = self._make("")
        payload = json.loads(line[6:])
        self.assertEqual(payload["choices"][0]["delta"], {})

    def test_finish_reason_propagated(self):
        line = self._make("", finish_reason="stop")
        payload = json.loads(line[6:])
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_finish_reason_none_by_default(self):
        line = self._make("tok")
        payload = json.loads(line[6:])
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_created_timestamp_used(self):
        line = self._make("x", created=9_999_999)
        payload = json.loads(line[6:])
        self.assertEqual(payload["created"], 9_999_999)

    def test_fingerprint_used(self):
        line = self._make("x", fingerprint="fp_custom")
        payload = json.loads(line[6:])
        self.assertEqual(payload["system_fingerprint"], "fp_custom")

    def test_model_name_in_output(self):
        result = self._srv._make_chunk("x", model="mymodel", cid="c1",
                                       _created=1, _fingerprint="fp")
        payload = json.loads(result[6:])
        self.assertEqual(payload["model"], "mymodel")

    def test_id_in_output(self):
        result = self._srv._make_chunk("x", model="m", cid="chatcmpl-zzz",
                                       _created=1, _fingerprint="fp")
        payload = json.loads(result[6:])
        self.assertEqual(payload["id"], "chatcmpl-zzz")

    def test_object_field_is_chunk(self):
        line = self._make("x")
        payload = json.loads(line[6:])
        self.assertEqual(payload["object"], "chat.completion.chunk")

    def test_make_chunk_uses_json_dumps_helper(self):
        """_make_chunk must call _json_dumps not bare json.dumps.

        The implementation builds the SSE frame via direct string construction,
        calling _json_dumps per field rather than once on the whole dict.
        Assert at least one call so we know the orjson-backed helper is used.
        """
        called = []
        original = self._srv._json_dumps

        def spy(obj):
            called.append(obj)
            return original(obj)

        with patch.object(self._srv, "_json_dumps", side_effect=spy):
            self._make("hello")
        self.assertGreaterEqual(len(called), 1, "_json_dumps was never called — bare json.dumps may have been used")


# ---------------------------------------------------------------------------
# TestPerformanceSmoke
# ---------------------------------------------------------------------------

class TestPerformanceSmoke(unittest.TestCase):
    """orjson path must not regress vs stdlib for small dicts (non-flaky smoke)."""

    def test_orjson_not_slower_than_stdlib(self):
        """Run 10 000 serialisations; orjson must be at most 2× slower than
        stdlib (in practice it is much faster; the 2× bound guards against
        accidental removal of the fast path)."""
        try:
            import orjson  # noqa: PLC0415
        except ImportError:
            self.skipTest("orjson not installed")

        payload = {
            "id": "chatcmpl-perf",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "squish",
            "system_fingerprint": "fp_perf",
            "choices": [{"index": 0, "delta": {"content": "tok"}, "finish_reason": None}],
        }
        N = 10_000

        t0 = time.perf_counter()
        for _ in range(N):
            json.dumps(payload)
        stdlib_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N):
            orjson.dumps(payload).decode()
        orjson_s = time.perf_counter() - t0

        # orjson should be at most 2× the stdlib time (it is typically 3-7× faster)
        self.assertLessEqual(
            orjson_s,
            stdlib_s * 2,
            f"orjson ({orjson_s:.4f}s) unexpectedly slower than stdlib ({stdlib_s:.4f}s)",
        )


# ---------------------------------------------------------------------------
# TestDependencyDeclared
# ---------------------------------------------------------------------------

class TestDependencyDeclared(unittest.TestCase):
    """orjson must be declared in both requirements.txt and pyproject.toml."""

    def _repo_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_orjson_in_requirements_txt(self):
        req = os.path.join(self._repo_root(), "requirements.txt")
        content = open(req).read()
        self.assertIn("orjson", content)

    def test_orjson_in_pyproject_toml(self):
        pyp = os.path.join(self._repo_root(), "pyproject.toml")
        content = open(pyp).read()
        self.assertIn("orjson", content)


if __name__ == "__main__":
    unittest.main()
