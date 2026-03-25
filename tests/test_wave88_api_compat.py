"""tests/test_wave88_api_compat.py

Wave 88 — Drop-in Compat: Ollama Gaps + LocalAI + squish compat

Tests for:
  - /api/ps shape and model loaded / not loaded variants
  - /api/version is not hardcoded "0.3.0"
  - HEAD /api/blobs raises 404
  - LocalAI GET / returns {message, version}
  - LocalAI GET /readyz returns {status: "ok"|"loading"}
  - LocalAI GET /healthz returns {status: "ok"}
  - BackendConfig defaults + env override
  - BackendRouter.proxy_url() builds correct URLs
  - cmd_compat exists in cli
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestOllamaPs — /api/ps endpoint shape
# ============================================================================

class TestOllamaPs(unittest.TestCase):
    """mount_ollama must expose GET /api/ps with correct shape."""

    def _make_app(self, model_loaded: bool = True):
        """Create a minimal FastAPI test app with Ollama routes mounted."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.ollama_compat import mount_ollama

        app = FastAPI()

        state = MagicMock()
        state.model = MagicMock() if model_loaded else None
        state.model_name = "qwen3:8b" if model_loaded else ""

        def get_state():
            return state

        mount_ollama(
            app,
            get_state=get_state,
            get_generate=lambda: None,
            get_tokenizer=lambda: None,
        )

        return TestClient(app)

    def test_ps_when_model_loaded_contains_models(self):
        client = self._make_app(model_loaded=True)
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 1

    def test_ps_when_no_model_returns_empty_list(self):
        client = self._make_app(model_loaded=False)
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert data["models"] == []

    def test_ps_model_card_has_required_fields(self):
        client = self._make_app(model_loaded=True)
        resp = client.get("/api/ps")
        card = resp.json()["models"][0]
        for field in ("name", "model", "size", "digest"):
            assert field in card, f"Model card missing field: {field}"


class TestOllamaVersion(unittest.TestCase):
    """GET /api/version must not return hardcoded '0.3.0'."""

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.ollama_compat import mount_ollama

        app = FastAPI()
        state = MagicMock()
        state.model = None
        state.model_name = ""
        mount_ollama(app, get_state=lambda: state, get_generate=lambda: None, get_tokenizer=lambda: None)
        return TestClient(app)

    def test_version_not_hardcoded(self):
        client = self._make_app()
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        # Must NOT be the hardcoded "0.3.0" Ollama version
        assert data["version"] != "0.3.0", (
            "/api/version must return the squish package version, not '0.3.0'"
        )

    def test_version_is_string(self):
        client = self._make_app()
        resp = client.get("/api/version")
        data = resp.json()
        assert isinstance(data["version"], str)


class TestOllamaBlobsHead(unittest.TestCase):
    """HEAD /api/blobs/{digest} must return 404."""

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.ollama_compat import mount_ollama

        app = FastAPI()
        state = MagicMock()
        state.model = None
        state.model_name = ""
        mount_ollama(app, get_state=lambda: state, get_generate=lambda: None, get_tokenizer=lambda: None)
        return TestClient(app)

    def test_blobs_head_returns_404(self):
        client = self._make_app()
        resp = client.head("/api/blobs/sha256:abc123def456")
        assert resp.status_code == 404, (
            f"HEAD /api/blobs/{{digest}} must return 404, got {resp.status_code}"
        )


# ============================================================================
# TestLocalAICompat — LocalAI routes
# ============================================================================

class TestLocalAICompat(unittest.TestCase):
    """mount_localai must expose / , /v1/version, /readyz, /healthz."""

    def _make_app(self, model_loaded=False):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.localai_compat import mount_localai

        app = FastAPI()
        state = MagicMock()
        state.model = MagicMock() if model_loaded else None

        mount_localai(app, get_state=lambda: state)
        return TestClient(app)

    def test_root_returns_200_with_message(self):
        client = self._make_app()
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data or "version" in data, (
            "GET / must return JSON with 'message' or 'version' field"
        )

    def test_v1_version_has_build_squish(self):
        client = self._make_app()
        resp = client.get("/v1/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("build") == "squish", (
            "/v1/version must contain build='squish'"
        )

    def test_readyz_loading_when_no_model(self):
        client = self._make_app(model_loaded=False)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("status") == "loading"

    def test_readyz_ok_when_model_loaded(self):
        client = self._make_app(model_loaded=True)
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"

    def test_healthz_always_ok(self):
        client = self._make_app()
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"


# ============================================================================
# TestBackendConfig — BackendConfig defaults + env override
# ============================================================================

class TestBackendConfig(unittest.TestCase):
    """BackendConfig must read SQUISH_BACKEND and SQUISH_BACKEND_URL env vars."""

    def test_defaults_to_squish_backend(self):
        from squish.serving.backend_router import BackendConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("SQUISH_BACKEND", "SQUISH_BACKEND_URL")}
        with patch.dict(os.environ, env, clear=True):
            cfg = BackendConfig()
        assert cfg.backend == "squish"

    def test_env_override_backend(self):
        from squish.serving.backend_router import BackendConfig
        with patch.dict(os.environ, {"SQUISH_BACKEND": "ollama"}, clear=False):
            cfg = BackendConfig()
        assert cfg.backend == "ollama"

    def test_env_override_url(self):
        from squish.serving.backend_router import BackendConfig
        with patch.dict(os.environ, {
            "SQUISH_BACKEND":     "squish",
            "SQUISH_BACKEND_URL": "http://custom:9999",
        }, clear=False):
            cfg = BackendConfig()
        assert cfg.base_url == "http://custom:9999"

    def test_default_squish_url_is_11435(self):
        from squish.serving.backend_router import BackendConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("SQUISH_BACKEND", "SQUISH_BACKEND_URL")}
        with patch.dict(os.environ, env, clear=True):
            cfg = BackendConfig()
        assert "11435" in cfg.base_url, (
            f"Default squish backend URL must contain port 11435, got {cfg.base_url!r}"
        )

    def test_is_squish_property(self):
        from squish.serving.backend_router import BackendConfig
        cfg = BackendConfig(backend="squish")
        assert cfg.is_squish is True
        assert cfg.is_ollama is False

    def test_is_ollama_property(self):
        from squish.serving.backend_router import BackendConfig
        cfg = BackendConfig(backend="ollama")
        assert cfg.is_ollama is True
        assert cfg.is_squish is False


class TestBackendRouter(unittest.TestCase):
    """BackendRouter.proxy_url() must build correct URLs."""

    def test_proxy_url_squish(self):
        from squish.serving.backend_router import BackendConfig, BackendRouter
        cfg = BackendConfig(backend="squish", base_url="http://localhost:11435")
        router = BackendRouter(cfg)
        url = router.proxy_url("/v1/chat/completions")
        assert url == "http://localhost:11435/v1/chat/completions"

    def test_proxy_url_ollama(self):
        from squish.serving.backend_router import BackendConfig, BackendRouter
        cfg = BackendConfig(backend="ollama", base_url="http://localhost:11434")
        router = BackendRouter(cfg)
        url = router.proxy_url("/api/generate")
        assert url == "http://localhost:11434/api/generate"

    def test_proxy_url_no_double_slash(self):
        from squish.serving.backend_router import BackendConfig, BackendRouter
        cfg = BackendConfig(backend="squish", base_url="http://localhost:11435/")
        router = BackendRouter(cfg)
        url = router.proxy_url("/v1/models")
        assert "//" not in url.split("://", 1)[1], (
            f"proxy_url must not produce double slashes, got {url!r}"
        )

    def test_health_check_returns_bool(self):
        from squish.serving.backend_router import BackendConfig, BackendRouter
        cfg = BackendConfig(backend="squish", base_url="http://localhost:39999")
        router = BackendRouter(cfg)
        result = router.health_check(timeout=0.1)
        assert isinstance(result, bool), "health_check() must return bool"
        assert result is False, "health_check() on non-existent server must return False"


# ============================================================================
# TestCmdCompat — cmd_compat in cli
# ============================================================================

class TestCmdCompat(unittest.TestCase):
    """cmd_compat must be callable and print without error."""

    def test_cmd_compat_callable(self):
        import squish.cli as cli
        assert callable(cli.cmd_compat)

    def test_cmd_compat_runs_without_error(self):
        import argparse
        import squish.cli as cli
        args = argparse.Namespace(host="localhost", port=11435)
        # Should not raise
        try:
            cli.cmd_compat(args)
        except SystemExit:
            pass  # acceptable
        except Exception as exc:
            raise AssertionError(f"cmd_compat raised {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    unittest.main()
