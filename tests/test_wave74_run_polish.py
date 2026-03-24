"""
tests/test_wave74_run_polish.py

Wave 74 — squish run polish:
  - _detect_local_ai_services: probe local AI ports, parse JSON responses
  - _open_browser_when_ready: fork-based browser opener contract
  - _recommend_model: RAM-band to model mapping
"""
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Ensure squish package is importable from the repo root
# ---------------------------------------------------------------------------
import importlib
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from squish.cli import (
    _detect_local_ai_services,
    _open_browser_when_ready,
    _recommend_model,
)


# ===========================================================================
# _detect_local_ai_services
# ===========================================================================

class TestDetectLocalAIServices(unittest.TestCase):
    """Tests for _detect_local_ai_services()."""

    def _stub_urlopen(self, response_body: bytes, status: int = 200):
        """Return a context-manager mock that yields a response object."""
        resp = MagicMock()
        resp.status = status
        resp.read.return_value = response_body
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return MagicMock(return_value=resp)

    def test_no_services_returns_empty_list(self):
        """When all ports are closed, result is an empty list."""
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = _detect_local_ai_services()
        self.assertEqual(result, [])

    def test_ollama_detected_via_api_tags(self):
        """Ollama's /api/tags endpoint ({"models": [...]}) is parsed correctly."""
        payload = json.dumps({"models": [{"name": "llama3:8b"}, {"name": "qwen3:4b"}]}).encode()
        with patch("urllib.request.urlopen", self._stub_urlopen(payload)):
            result = _detect_local_ai_services()
        # At least one service detected
        self.assertTrue(len(result) >= 1)
        found = next((s for s in result if s["name"] == "Ollama"), None)
        self.assertIsNotNone(found, "Ollama not found in result")
        self.assertEqual(found["models"], ["llama3:8b", "qwen3:4b"])
        self.assertEqual(found["model_count"], 2)
        self.assertEqual(found["base_url"], "http://127.0.0.1:11434")

    def test_openai_compat_detected_via_v1_models(self):
        """OpenAI-compat /v1/models endpoint ({"data": [...]}) is parsed correctly."""
        payload = json.dumps({"data": [{"id": "my-model"}, {"id": "other-model"}]}).encode()
        # Patch only calls for LM Studio's port to succeed
        call_count = {"n": 0}

        def selective_urlopen(req, timeout=0.5):
            call_count["n"] += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "1234" in url:
                resp = MagicMock()
                resp.status = 200
                resp.read.return_value = payload
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            raise OSError("refused")

        with patch("urllib.request.urlopen", side_effect=selective_urlopen):
            result = _detect_local_ai_services()

        found = next((s for s in result if s["name"] == "LM Studio"), None)
        self.assertIsNotNone(found, "LM Studio not found in result")
        self.assertEqual(found["models"], ["my-model", "other-model"])
        self.assertEqual(found["model_count"], 2)

    def test_jan_detected(self):
        """Jan.ai /v1/models endpoint is detected on port 1337."""
        payload = json.dumps({"data": [{"id": "jan-model"}]}).encode()

        def selective_urlopen(req, timeout=0.5):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "1337" in url:
                resp = MagicMock()
                resp.status = 200
                resp.read.return_value = payload
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            raise OSError("refused")

        with patch("urllib.request.urlopen", side_effect=selective_urlopen):
            result = _detect_local_ai_services()

        found = next((s for s in result if s["name"] == "Jan"), None)
        self.assertIsNotNone(found, "Jan not found in result")
        self.assertEqual(found["model_count"], 1)

    def test_multiple_services_detected(self):
        """When two services are running, both appear in the result."""
        ollama_payload = json.dumps({"models": [{"name": "llama3:8b"}]}).encode()
        lmstudio_payload = json.dumps({"data": [{"id": "phi4:14b"}]}).encode()

        def selective_urlopen(req, timeout=0.5):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "11434" in url:
                resp = MagicMock()
                resp.status = 200
                resp.read.return_value = ollama_payload
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            if "1234" in url:
                resp = MagicMock()
                resp.status = 200
                resp.read.return_value = lmstudio_payload
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            raise OSError("refused")

        with patch("urllib.request.urlopen", side_effect=selective_urlopen):
            result = _detect_local_ai_services()

        names = [s["name"] for s in result]
        self.assertIn("Ollama", names)
        self.assertIn("LM Studio", names)
        self.assertEqual(len(result), 2)

    def test_timeout_error_is_ignored(self):
        """TimeoutError on a probe is silently swallowed; no crash."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = _detect_local_ai_services()
        self.assertEqual(result, [])

    def test_malformed_json_is_ignored(self):
        """A response with invalid JSON is silently skipped."""
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not-json{{{"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            result = _detect_local_ai_services()
        self.assertEqual(result, [])

    def test_empty_models_list_gives_model_count_zero(self):
        """A service that reports zero models has model_count == 0."""
        payload = json.dumps({"models": []}).encode()
        with patch("urllib.request.urlopen", self._stub_urlopen(payload)):
            result = _detect_local_ai_services()
        # We may get entries for every probed port that succeeded
        for entry in result:
            self.assertEqual(entry["model_count"], 0)


# ===========================================================================
# _open_browser_when_ready
# ===========================================================================

class TestOpenBrowserWhenReady(unittest.TestCase):
    """Tests for _open_browser_when_ready()."""

    def test_returns_immediately_without_blocking(self):
        """The function must return immediately — the polling runs in a daemon thread."""
        import threading

        threads_before = set(t.ident for t in threading.enumerate())
        _open_browser_when_ready("http://localhost:11435/chat", 11435, timeout_s=0)
        # Function should return without raising; new thread may have started and
        # already finished (timeout_s=0 → no-op poll loop).
        # The key assertion is just that we return at all.

    def test_spawns_daemon_thread(self):
        """A daemon thread must be spawned, not os.fork()."""
        import threading

        spawned: list = []
        original_init = threading.Thread.__init__

        def capturing_init(self_t, *a, **kw):
            spawned.append(kw.get("daemon") or a[4] if len(a) > 4 else kw.get("daemon"))
            original_init(self_t, *a, **kw)

        with patch.object(threading.Thread, "__init__", capturing_init), \
             patch.object(threading.Thread, "start", lambda s: None):
            _open_browser_when_ready("http://localhost:11435/chat", 11435)

        self.assertTrue(len(spawned) >= 1, "Expected a Thread to be constructed")

    def test_thread_opens_browser_on_200(self):
        """The polling function inside the thread should open the browser on HTTP 200."""
        import threading

        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        threads_started: list = []

        def capture_thread(self_t):
            threads_started.append(self_t)
            # Run the thread target synchronously so we can observe side-effects.
            self_t._target()

        with patch("urllib.request.urlopen", return_value=resp), \
             patch("webbrowser.open") as mock_wb, \
             patch.object(threading.Thread, "start", capture_thread):
            _open_browser_when_ready("http://localhost:11435/chat", 11435, timeout_s=5)

        mock_wb.assert_called_once_with("http://localhost:11435/chat")


# ===========================================================================
# _recommend_model
# ===========================================================================

class TestRecommendModel(unittest.TestCase):
    """Parametric tests for _recommend_model()."""

    def test_64gb_recommends_32b(self):
        self.assertEqual(_recommend_model(64.0), "qwen3:32b")

    def test_96gb_recommends_32b(self):
        self.assertEqual(_recommend_model(96.0), "qwen3:32b")

    def test_32gb_recommends_14b(self):
        self.assertEqual(_recommend_model(32.0), "qwen3:14b")

    def test_36gb_recommends_14b(self):
        self.assertEqual(_recommend_model(36.0), "qwen3:14b")

    def test_16gb_recommends_8b(self):
        self.assertEqual(_recommend_model(16.0), "qwen3:8b")

    def test_24gb_recommends_8b(self):
        self.assertEqual(_recommend_model(24.0), "qwen3:8b")

    def test_8gb_recommends_1p7b(self):
        self.assertEqual(_recommend_model(8.0), "qwen3:1.7b")

    def test_0gb_recommends_1p7b(self):
        """Edge case: 0 GB RAM still returns the smallest model (never crashes)."""
        self.assertEqual(_recommend_model(0.0), "qwen3:1.7b")


if __name__ == "__main__":
    unittest.main()
