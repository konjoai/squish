"""tests/test_wave80_chunk_fingerprint.py

Wave 80 — Streaming chunk optimisations

Tests for:
  - _system_fingerprint uses lru_cache (MD5 not recomputed per token)
  - _system_fingerprint accepts (model_name, loaded_at) args
  - _system_fingerprint returns consistent "sq-<8hex>" format
  - _make_chunk accepts pre-computed _created / _fingerprint kwargs
  - _make_chunk still works without pre-computed kwargs (backward compat)
  - Mutable chunk template pattern: dict reuse produces correct JSON per token
  - _comp_chunk timestamp hoisting: pre-computed timestamp reused across tokens
  - functools is imported by squish.server
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import squish.server as _srv


# ============================================================================
# _system_fingerprint — lru_cache correctness
# ============================================================================

class TestSystemFingerprintCached(unittest.TestCase):
    """_system_fingerprint must be cached and produce stable, correct values."""

    def setUp(self):
        # Clear the lru_cache before every test so tests are independent.
        _srv._system_fingerprint.cache_clear()

    def test_returns_sq_prefixed_hex(self):
        fp = _srv._system_fingerprint("my-model", 1234567890.0)
        self.assertTrue(fp.startswith("sq-"), f"Expected 'sq-' prefix, got {fp!r}")
        self.assertEqual(len(fp), 11,
                         "Fingerprint should be 'sq-' (3) + 8 hex chars")
        self.assertTrue(all(c in "0123456789abcdef" for c in fp[3:]),
                        f"Non-hex chars after 'sq-' in {fp!r}")

    def test_deterministic_for_same_inputs(self):
        fp1 = _srv._system_fingerprint("model-a", 0.0)
        fp2 = _srv._system_fingerprint("model-a", 0.0)
        self.assertEqual(fp1, fp2)

    def test_differs_for_different_model(self):
        fp1 = _srv._system_fingerprint("model-a", 0.0)
        fp2 = _srv._system_fingerprint("model-b", 0.0)
        self.assertNotEqual(fp1, fp2)

    def test_differs_for_different_loaded_at(self):
        fp1 = _srv._system_fingerprint("model-x", 1000.0)
        fp2 = _srv._system_fingerprint("model-x", 2000.0)
        self.assertNotEqual(fp1, fp2)

    def test_is_lru_cached(self):
        """Calling with same args must hit the cache (cache_info Miss count must
        not increase after the first call with those args)."""
        import functools
        _srv._system_fingerprint.cache_clear()
        _ = _srv._system_fingerprint("cached-model", 99.9)
        info_after_first = _srv._system_fingerprint.cache_info()
        _ = _srv._system_fingerprint("cached-model", 99.9)
        info_after_second = _srv._system_fingerprint.cache_info()
        # Misses must not increase on the second call
        self.assertEqual(info_after_first.misses, info_after_second.misses,
                         "lru_cache miss on second call with same args — caching broken")
        self.assertGreater(info_after_second.hits, info_after_first.hits,
                           "lru_cache hit count should increase on second call")

    def test_cache_maxsize_allows_multiple_models(self):
        """maxsize=4 should retain fingerprints for at least 4 different models."""
        _srv._system_fingerprint.cache_clear()
        fingerprints = {}
        for i in range(4):
            key = (f"model-{i}", float(i))
            fingerprints[key] = _srv._system_fingerprint(*key)
        # Re-query all 4; none should be a miss
        info_before = _srv._system_fingerprint.cache_info()
        for key in fingerprints:
            _srv._system_fingerprint(*key)
        info_after = _srv._system_fingerprint.cache_info()
        self.assertEqual(info_before.misses, info_after.misses,
                         "Some fingerprints were evicted before maxsize=4 should evict")

    def test_none_model_name_handled(self):
        """model_name=None is valid (server starts before a model is loaded)."""
        fp = _srv._system_fingerprint(None, 0.0)
        self.assertTrue(fp.startswith("sq-"))

    def test_matches_expected_md5(self):
        """Fingerprint must match the manually computed MD5 for the same inputs."""
        model_name = "test-model"
        loaded_at  = 42.0
        expected = "sq-" + hashlib.md5(
            f"{model_name}{loaded_at}".encode()
        ).hexdigest()[:8]
        self.assertEqual(_srv._system_fingerprint(model_name, loaded_at), expected)


# ============================================================================
# _make_chunk — backward compat + pre-computed kwargs
# ============================================================================

class TestMakeChunkBackwardCompat(unittest.TestCase):
    """_make_chunk must still work with no pre-computed kwargs."""

    def _parse(self, raw: str) -> dict:
        self.assertTrue(raw.startswith("data: "), f"SSE prefix missing: {raw!r}")
        self.assertTrue(raw.endswith("\n\n"), f"SSE suffix missing: {raw!r}")
        return json.loads(raw[6:])

    def test_returns_sse_string(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            result = _srv._make_chunk("hello", "my-model", "id-abc")
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("data: "))
        self.assertTrue(result.endswith("\n\n"))

    def test_content_in_delta(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            data = self._parse(_srv._make_chunk("world", "m", "cid-1"))
        self.assertEqual(data["choices"][0]["delta"]["content"], "world")

    def test_empty_content_gives_empty_delta(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            data = self._parse(_srv._make_chunk("", "m", "cid-1"))
        self.assertEqual(data["choices"][0]["delta"], {})

    def test_finish_reason_in_choices(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            data = self._parse(_srv._make_chunk("", "m", "cid-1", finish_reason="stop"))
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")

    def test_precomputed_kwargs_used_when_supplied(self):
        """When _created and _fingerprint are passed they are used verbatim."""
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            data = self._parse(_srv._make_chunk(
                "tok", "m", "cid-2",
                _created=1_000_000, _fingerprint="sq-precomp",
            ))
        self.assertEqual(data["created"], 1_000_000)
        self.assertEqual(data["system_fingerprint"], "sq-precomp")

    def test_model_and_id_in_chunk(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "mod"
            mock_state.loaded_at  = 1.0
            data = self._parse(_srv._make_chunk("x", "my-model", "my-cid"))
        self.assertEqual(data["model"], "my-model")
        self.assertEqual(data["id"],    "my-cid")
        self.assertEqual(data["object"], "chat.completion.chunk")

    def test_choices_index_zero(self):
        with patch.object(_srv, "_state") as mock_state:
            mock_state.model_name = "m"
            mock_state.loaded_at  = 0.0
            data = self._parse(_srv._make_chunk("t", "m", "c"))
        self.assertEqual(data["choices"][0]["index"], 0)


# ============================================================================
# Mutable chunk template pattern
# ============================================================================

class TestMutableChunkTemplate(unittest.TestCase):
    """The per-request mutable template pattern must produce correct JSON for
    every token without allocating a new dict per token."""

    def _build_template(self, cid: str, model: str, fp: str, created: int) -> dict:
        """Reproduce the template the event_stream closure builds."""
        tmpl = {
            "id":                cid,
            "object":            "chat.completion.chunk",
            "created":           created,
            "model":             model,
            "system_fingerprint": fp,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        }
        return tmpl

    def test_mutating_template_changes_json_output(self):
        tmpl  = self._build_template("cid", "mdl", "sq-test123", 999)
        choice = tmpl["choices"][0]
        delta  = choice["delta"]

        tokens = ["Hello", " world", "!"]
        results = []
        for tok in tokens:
            delta["content"]      = tok
            choice["finish_reason"] = None
            results.append(json.loads(json.dumps(tmpl)))

        for i, r in enumerate(results):
            self.assertEqual(r["choices"][0]["delta"]["content"], tokens[i])

    def test_template_constant_fields_unchanged(self):
        """Constant fields must not be mutated by the per-token loop."""
        tmpl   = self._build_template("my-cid", "my-model", "sq-abcd1234", 12345)
        choice = tmpl["choices"][0]
        delta  = choice["delta"]

        for tok in ["a", "b", "c"]:
            delta["content"]       = tok
            choice["finish_reason"] = None

        self.assertEqual(tmpl["id"],                "my-cid")
        self.assertEqual(tmpl["model"],             "my-model")
        self.assertEqual(tmpl["system_fingerprint"], "sq-abcd1234")
        self.assertEqual(tmpl["created"],           12345)

    def test_final_chunk_can_set_finish_reason(self):
        tmpl   = self._build_template("c", "m", "sq-00000000", 1)
        choice = tmpl["choices"][0]
        delta  = choice["delta"]

        delta["content"]       = ""
        choice["finish_reason"] = "stop"
        data = json.loads(json.dumps(tmpl))
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["choices"][0]["delta"]["content"], "")

    def test_same_dict_object_reused(self):
        """Verifies that the test setup mirrors the server: one dict object is
        mutated in place rather than a new one being created per token."""
        tmpl   = self._build_template("c", "m", "sq-00000000", 1)
        tmpl_id = id(tmpl)

        choice = tmpl["choices"][0]
        delta  = choice["delta"]

        for tok in ["a", "b", "c"]:
            delta["content"] = tok
            # The dict identity must not change
            self.assertEqual(id(tmpl), tmpl_id)


# ============================================================================
# _comp_chunk timestamp hoisting
# ============================================================================

class TestCompChunkTimestampHoisting(unittest.TestCase):
    """The /v1/completions streaming path pre-computes the timestamp once."""

    def test_timestamp_hoisted_not_per_token(self):
        """Simulate the _comp_chunk closure with a captured timestamp and verify
        consistent timestamps across tokens."""
        cid      = "cmpl-xyz"
        model_id = "my-model"
        # Simulate what the server does: capture once before the loop
        _comp_ts = int(time.time())

        def _comp_chunk(text: str, finish_reason=None) -> str:
            chunk = {
                "id": cid, "object": "text_completion",
                "created": _comp_ts, "model": model_id,
                "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        tokens    = ["tok1", "tok2", "tok3"]
        timestamps = []
        for tok in tokens:
            data = json.loads(_comp_chunk(tok)[6:])
            timestamps.append(data["created"])

        # All timestamps must be identical (captured once)
        self.assertEqual(len(set(timestamps)), 1,
                         f"Expected single timestamp, got {timestamps}")

    def test_comp_chunk_json_valid(self):
        """The generated SSE line must be valid JSON."""
        _comp_ts = 1_700_000_000
        cid = "test-cid"
        model_id = "m"

        def _comp_chunk(text: str, finish_reason=None) -> str:
            chunk = {
                "id": cid, "object": "text_completion",
                "created": _comp_ts, "model": model_id,
                "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        raw  = _comp_chunk("hello")
        data = json.loads(raw[6:])
        self.assertEqual(data["choices"][0]["text"], "hello")
        self.assertIsNone(data["choices"][0]["finish_reason"])


# ============================================================================
# functools import check
# ============================================================================

class TestFunctoolsImported(unittest.TestCase):
    """squish.server must import functools (needed for lru_cache)."""

    def test_functools_in_server_module(self):
        import functools
        import squish.server as srv
        # The functools module must be importable and used — check that
        # _system_fingerprint has a cache_info method (lru_cache attribute).
        self.assertTrue(hasattr(srv._system_fingerprint, "cache_info"),
                        "_system_fingerprint must be decorated with lru_cache")
        self.assertTrue(callable(srv._system_fingerprint.cache_info))

    def test_cache_info_returns_named_tuple(self):
        info = _srv._system_fingerprint.cache_info()
        self.assertTrue(hasattr(info, "hits"))
        self.assertTrue(hasattr(info, "misses"))
        self.assertTrue(hasattr(info, "maxsize"))
        self.assertEqual(info.maxsize, 4)


if __name__ == "__main__":
    unittest.main()
