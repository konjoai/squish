"""squish/serving/backend_router.py

Backend routing configuration for Squish.

Allows easy switching between Squish, Ollama, OpenAI, and LocalAI backends
via environment variables or configuration, without changing client code.

Environment variables:
    SQUISH_BACKEND       One of: ``squish`` (default), ``ollama``, ``openai``, ``localai``
    SQUISH_BACKEND_URL   Base URL for the selected backend (overrides default)

Usage::

    from squish.serving.backend_router import BackendConfig, BackendRouter

    cfg    = BackendConfig()
    router = BackendRouter(cfg)
    url    = router.proxy_url("/v1/chat/completions")
    up     = router.health_check()
"""
from __future__ import annotations

__all__ = ["BackendConfig", "BackendRouter"]

import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Default base URLs per backend
_DEFAULT_URLS: dict[str, str] = {
    "squish":  "http://localhost:11435",
    "ollama":  "http://localhost:11434",
    "openai":  "https://api.openai.com",
    "localai": "http://localhost:8080",
}


@dataclass
class BackendConfig:
    """Configuration for the active inference backend.

    Reads ``SQUISH_BACKEND`` (default ``"squish"``) and
    ``SQUISH_BACKEND_URL`` (overrides :attr:`default_url`).

    Attributes
    ----------
    backend:
        One of ``"squish"``, ``"ollama"``, ``"openai"``, ``"localai"``.
    base_url:
        Resolved base URL for the selected backend.
    api_key:
        API key if required (read from ``SQUISH_API_KEY`` or ``OPENAI_API_KEY``).
    """

    backend:  str = field(default_factory=lambda: os.environ.get("SQUISH_BACKEND", "squish").lower())
    base_url: str = field(default="")
    api_key:  str = field(default_factory=lambda: (
        os.environ.get("SQUISH_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ))

    def __post_init__(self) -> None:
        if not self.base_url:
            env_url = os.environ.get("SQUISH_BACKEND_URL", "")
            self.base_url = env_url or _DEFAULT_URLS.get(self.backend, _DEFAULT_URLS["squish"])

    @property
    def default_url(self) -> str:
        """Return the default URL for the configured backend (ignoring env override)."""
        return _DEFAULT_URLS.get(self.backend, _DEFAULT_URLS["squish"])

    @property
    def is_squish(self) -> bool:
        return self.backend == "squish"

    @property
    def is_ollama(self) -> bool:
        return self.backend == "ollama"

    @property
    def is_openai(self) -> bool:
        return self.backend == "openai"

    @property
    def is_localai(self) -> bool:
        return self.backend == "localai"


class BackendRouter:
    """Routes proxy requests to the configured backend.

    Parameters
    ----------
    config:
        A :class:`BackendConfig` instance.  When ``None``, a default
        config is built from environment variables.
    """

    def __init__(self, config: BackendConfig | None = None) -> None:
        self._cfg = config if config is not None else BackendConfig()

    @property
    def config(self) -> BackendConfig:
        return self._cfg

    def proxy_url(self, path: str) -> str:
        """Build the full proxy URL for *path* relative to the backend base URL.

        Parameters
        ----------
        path:
            API path, e.g. ``"/v1/chat/completions"``.

        Returns
        -------
        str
            Full URL: ``<base_url><path>``
        """
        base = self._cfg.base_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def health_check(self, timeout: float = 2.0) -> bool:
        """Probe the configured backend for liveness.

        Returns ``True`` if a health endpoint responds with a 2xx status,
        ``False`` otherwise.

        Parameters
        ----------
        timeout:
            Socket timeout in seconds (default 2.0).
        """
        # Choose a lightweight probe endpoint per backend
        probe_paths: dict[str, str] = {
            "squish":  "/health",
            "ollama":  "/api/version",
            "openai":  "/v1/models",
            "localai": "/healthz",
        }
        probe = probe_paths.get(self._cfg.backend, "/health")
        url = self.proxy_url(probe)
        try:
            req = urllib.request.Request(url)
            if self._cfg.api_key:
                req.add_header("Authorization", f"Bearer {self._cfg.api_key}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def __repr__(self) -> str:
        return f"BackendRouter(backend={self._cfg.backend!r}, url={self._cfg.base_url!r})"
