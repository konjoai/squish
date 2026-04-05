"""squish/serving/lm_studio_bridge.py — LM Studio runtime detection and client.

Probes a locally-running LM Studio instance and optionally forwards chat
completions through its OpenAI-compatible API.

Public API
──────────
LMStudioStatus      — frozen dataclass describing a probe result
probe_lm_studio()   — probe LM Studio at localhost:1234 (or env override)
LMStudioClient      — thin HTTP client for LM Studio's /v1 API
"""
from __future__ import annotations

__all__ = ["LMStudioStatus", "probe_lm_studio", "LMStudioClient"]

import json
import os
from dataclasses import dataclass, field
from typing import Iterator


# Default LM Studio OpenAI-compat endpoint
_DEFAULT_BASE_URL = "http://127.0.0.1:1234"


# ---------------------------------------------------------------------------
# LMStudioStatus
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMStudioStatus:
    """Result of a probe against a running LM Studio instance.

    Attributes
    ----------
    running:        True if LM Studio responded within the timeout.
    base_url:       URL that was probed (may differ via LMSTUDIO_BASE_URL).
    loaded_models:  Model IDs currently loaded in LM Studio (may be empty even
                    when running — LM Studio only shows a model once it is loaded
                    into GPU/unified memory).
    server_version: LM Studio version string if returned by the API, else ``""``.
    """
    running:        bool
    base_url:       str
    loaded_models:  list[str] = field(default_factory=list, compare=False)
    server_version: str = ""

    @property
    def model_count(self) -> int:
        """Number of models currently loaded in LM Studio."""
        return len(self.loaded_models)

    def __str__(self) -> str:
        if not self.running:
            return f"LM Studio  not running  ({self.base_url})"
        n = self.model_count
        model_str = (
            self.loaded_models[0] if n == 1
            else f"{n} models" if n > 1
            else "no model loaded"
        )
        ver = f"  v{self.server_version}" if self.server_version else ""
        return f"LM Studio  running{ver}  —  {model_str}  ({self.base_url})"


# ---------------------------------------------------------------------------
# probe_lm_studio
# ---------------------------------------------------------------------------

def probe_lm_studio(timeout: float = 0.8) -> LMStudioStatus:
    """Probe a running LM Studio instance and return its status.

    Uses ``LMSTUDIO_BASE_URL`` environment variable to override the default
    ``http://127.0.0.1:1234``.  Never raises — all errors map to a
    ``LMStudioStatus(running=False, ...)``.

    Parameters
    ----------
    timeout:
        Socket timeout in seconds.  Keep this short — it runs at CLI startup.
    """
    import urllib.error
    import urllib.request

    base_url = os.environ.get("LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    models_url = f"{base_url}/v1/models"

    try:
        req = urllib.request.Request(
            models_url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw: dict = json.loads(resp.read())
    except Exception:
        return LMStudioStatus(running=False, base_url=base_url)

    # LM Studio returns {"data": [{"id": "...", ...}, ...]}
    data = raw.get("data") or raw.get("models") or []
    loaded_models = [
        m.get("id") or m.get("name") or ""
        for m in data
        if isinstance(m, dict)
    ]
    loaded_models = [m for m in loaded_models if m]

    # LM Studio may include a version in the response root (varies by version)
    server_version: str = str(raw.get("version") or raw.get("server_version") or "")

    return LMStudioStatus(
        running=True,
        base_url=base_url,
        loaded_models=loaded_models,
        server_version=server_version,
    )


# ---------------------------------------------------------------------------
# LMStudioClient
# ---------------------------------------------------------------------------

class LMStudioClient:
    """Thin HTTP client for LM Studio's OpenAI-compatible /v1 API.

    Parameters
    ----------
    base_url:
        Base URL of the LM Studio server.  Defaults to ``http://127.0.0.1:1234``
        or ``LMSTUDIO_BASE_URL`` env var.
    timeout:
        Request timeout in seconds for non-streaming calls.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (
            (base_url or os.environ.get("LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL))
            .rstrip("/")
        )
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def models(self) -> list[dict]:
        """Return the list of models from ``GET /v1/models``.

        Returns an empty list if LM Studio is not running.
        """
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/models",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read())
            return raw.get("data") or raw.get("models") or []
        except Exception:
            return []

    def chat_completions(
        self,
        messages: list[dict],
        model: str = "",
        stream: bool = True,
        **kwargs,
    ) -> Iterator[str] | dict:
        """Forward a chat completion request to LM Studio.

        Parameters
        ----------
        messages:
            OpenAI-format message list: ``[{"role": "user", "content": "..."}]``.
        model:
            Model ID to use.  If empty, LM Studio uses the currently loaded model.
        stream:
            If ``True`` — yields SSE delta strings (content tokens only).
            If ``False`` — returns the full response dict.
        **kwargs:
            Extra parameters forwarded in the request body (e.g. ``temperature``,
            ``max_tokens``).

        Raises
        ------
        ConnectionError
            When LM Studio is not reachable.
        RuntimeError
            When LM Studio returns a non-200 status.
        """
        import urllib.error
        import urllib.request

        body: dict = {"messages": messages, "stream": stream, **kwargs}
        if model:
            body["model"] = model

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json" if not stream else "text/event-stream",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"LM Studio not reachable at {self.base_url}: {exc}"
            ) from exc

        if resp.status not in (200, 201):  # type: ignore[attr-defined]
            raise RuntimeError(
                f"LM Studio returned HTTP {resp.status}"  # type: ignore[attr-defined]
            )

        if not stream:
            return json.loads(resp.read())

        return self._iter_sse(resp)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_sse(resp) -> Iterator[str]:  # type: ignore[type-arg]
        """Yield content token strings from an SSE stream."""
        for raw_line in resp:
            line: str = (
                raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            ).rstrip("\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                if content:
                    yield content
