"""Behavioral coverage for ``squish.serving.backend_router`` — backend config
resolution and the proxy-URL / health-check router. Pure-Python; no MLX.
"""
from __future__ import annotations

import urllib.error

import pytest

from squish.serving import backend_router as br
from squish.serving.backend_router import BackendConfig, BackendRouter


def _clear_env(monkeypatch):
    for k in ("SQUISH_BACKEND", "SQUISH_BACKEND_URL", "SQUISH_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


# ── BackendConfig ───────────────────────────────────────────────────────────


def test_config_defaults_to_squish(monkeypatch):
    _clear_env(monkeypatch)
    cfg = BackendConfig()
    assert cfg.backend == "squish"
    assert cfg.base_url == "http://localhost:11435"
    assert cfg.api_key == ""


def test_config_backend_and_url_from_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SQUISH_BACKEND", "Ollama")  # upper → lowercased
    cfg = BackendConfig()
    assert cfg.backend == "ollama"
    assert cfg.base_url == "http://localhost:11434"  # ollama default


def test_config_url_override(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SQUISH_BACKEND_URL", "http://example:9999")
    assert BackendConfig().base_url == "http://example:9999"


def test_config_unknown_backend_falls_back_to_squish_url(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SQUISH_BACKEND", "weirdbackend")
    cfg = BackendConfig()
    assert cfg.default_url == "http://localhost:11435"  # unknown → squish default
    assert cfg.base_url == "http://localhost:11435"


def test_config_api_key_sources(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert BackendConfig().api_key == "sk-openai"
    monkeypatch.setenv("SQUISH_API_KEY", "sk-squish")
    assert BackendConfig().api_key == "sk-squish"  # SQUISH takes precedence


def test_config_backend_flags():
    assert BackendConfig(backend="squish").is_squish is True
    assert BackendConfig(backend="ollama").is_ollama is True
    assert BackendConfig(backend="openai").is_openai is True
    assert BackendConfig(backend="localai").is_localai is True
    assert BackendConfig(backend="squish").is_ollama is False


# ── BackendRouter ───────────────────────────────────────────────────────────


def test_router_uses_default_config_when_none(monkeypatch):
    _clear_env(monkeypatch)
    r = BackendRouter()
    assert r.config.backend == "squish"


def test_proxy_url_joins_path():
    r = BackendRouter(BackendConfig(backend="squish", base_url="http://h:1/"))
    assert r.proxy_url("/v1/chat") == "http://h:1/v1/chat"
    assert r.proxy_url("v1/models") == "http://h:1/v1/models"  # leading slash added


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_health_check_success_with_api_key(monkeypatch):
    captured = {}

    def _urlopen(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")
        return _Resp(200)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    r = BackendRouter(BackendConfig(backend="openai", base_url="http://h", api_key="sk-x"))
    assert r.health_check() is True
    assert captured["auth"] == "Bearer sk-x"


def test_health_check_non_2xx_is_false(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp(503))
    assert BackendRouter(BackendConfig(backend="squish")).health_check() is False


def test_health_check_handles_connection_error(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")))
    assert BackendRouter(BackendConfig(backend="ollama")).health_check() is False


def test_health_check_unknown_backend_uses_default_probe(monkeypatch):
    seen = {}

    def _urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp(204)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    r = BackendRouter(BackendConfig(backend="mystery", base_url="http://h"))
    assert r.health_check() is True
    assert seen["url"].endswith("/health")  # default probe path


def test_repr():
    text = repr(BackendRouter(BackendConfig(backend="ollama", base_url="http://h:2")))
    assert "backend='ollama'" in text and "http://h:2" in text
