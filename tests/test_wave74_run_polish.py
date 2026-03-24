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

    def test_parent_returns_immediately_after_fork(self):
        """The calling (parent) process returns without blocking."""
        with patch("os.fork", return_value=1) as mock_fork:
            # return value 1 == parent side
            _open_browser_when_ready("http://localhost:11435/chat", 11435)
        mock_fork.assert_called_once()

    def test_no_fork_called_when_service_probed_manually(self):
        """Verify fork is called exactly once per call to open_browser_when_ready."""
        with patch("os.fork", return_value=42) as mock_fork:
            _open_browser_when_ready("http://localhost:11435/chat", 11435)
        self.assertEqual(mock_fork.call_count, 1)

    def test_child_side_opens_browser_on_200_then_exits(self):
        """Child (fork returns 0) should open browser on HTTP 200 then os._exit(0)."""
        import urllib.error

        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("os.fork", return_value=0), \
             patch("urllib.request.urlopen", return_value=resp), \
             patch("webbrowser.open") as mock_wb, \
             patch("os._exit") as mock_exit:
            # _exit terminates the child; we raise SystemExit to escape the loop
            mock_exit.side_effect = SystemExit(0)
            try:
                _open_browser_when_ready("http://localhost:11435/chat", 11435, timeout_s=5)
            except SystemExit:
                pass

        mock_wb.assert_called_once_with("http://localhost:11435/chat")
        mock_exit.assert_called_once_with(0)


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
