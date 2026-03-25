"""squish/serving/localai_compat.py

LocalAI-compatible HTTP API layer for Squish.

Mounts lightweight LocalAI-specific routes onto the existing FastAPI app.
Core AI endpoints (``/v1/chat/completions``, ``/v1/models``, ``/v1/embeddings``)
are already handled by Squish's OpenAI-compatible layer and do not need
duplication here.

New routes added:
  GET  /                  → LocalAI welcome JSON (version info)
  GET  /v1/version        → {"version": "2.0.0", "build": "squish"}
  GET  /readyz            → readiness probe ("ok" once model is loaded)
  GET  /healthz           → liveness probe (always 200 if server is up)

Usage::

    from squish.serving.localai_compat import mount_localai
    mount_localai(app, get_state=lambda: _state)
"""
from __future__ import annotations

import importlib.metadata

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def mount_localai(app: FastAPI, get_state) -> None:
    """
    Register LocalAI-compatible routes on *app*.

    Parameters
    ----------
    app:
        The FastAPI application from server.py.
    get_state:
        Zero-arg callable returning the ``_ModelState`` global.
    """

    def _squish_version() -> str:
        try:
            return importlib.metadata.version("squish")
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0-dev"

    @app.get("/")
    async def localai_root():
        """LocalAI root — version info for discovery."""
        return JSONResponse({
            "message": "LocalAI-compatible API (Squish backend)",
            "version": "2.0.0",
            "squish_version": _squish_version(),
            "build": "squish",
        })

    @app.get("/v1/version")
    async def localai_version():
        """GET /v1/version — LocalAI version endpoint."""
        return JSONResponse({
            "version": "2.0.0",
            "build":   "squish",
            "squish_version": _squish_version(),
        })

    @app.get("/readyz")
    async def localai_readyz():
        """GET /readyz — readiness probe; returns 200/ok once the model is loaded."""
        state = get_state()
        if state.model is not None:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.get("/healthz")
    async def localai_healthz():
        """GET /healthz — liveness probe; always returns 200 when server is up."""
        return JSONResponse({"status": "ok"})
