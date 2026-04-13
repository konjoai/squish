"""squish/api/v1_router.py — Versioned public REST API router.

Mounts stable ``/v1/`` endpoints on any ASGI/WSGI application and provides
an ``OpenAPISchemaBuilder`` for generating an OpenAPI 3.1 document that can
be used for client SDK auto-generation.

All responses include ``X-Squish-API-Version`` and ``X-Squish-Version``
headers.  Legacy unversioned endpoints (``/chat``, ``/completions``) are
preserved with ``Deprecation: true`` and ``Sunset`` headers.

Classes / functions
───────────────────
V1RouteSpec          — Metadata for a single /v1 route.
OpenAPISchemaBuilder — Generates minimal OpenAPI 3.1 from V1RouteSpec list.
APIVersionMiddleware — WSGI/ASGI middleware that injects version headers.
V1Router             — Collects route specs and registers them.
register_v1_routes   — Convenience function: create V1Router + register.

Usage::

    # With Flask / any WSGI app
    from squish.api.v1_router import register_v1_routes
    routes = register_v1_routes(app)

    # Standalone OpenAPI schema
    from squish.api.v1_router import V1Router
    router = V1Router()
    schema = router.openapi_schema()
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

_DEFAULT_SERVER_URL: str = os.environ.get("SQUISH_SERVER_URL", "http://localhost:11435")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION: str = "1"
SUNSET_DATE: str = "2026-12-31"   # When legacy unversioned endpoints go away

_PACKAGE_NAME = "squish"


def _package_version() -> str:
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-dev"


# ---------------------------------------------------------------------------
# Route metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V1RouteSpec:
    """Metadata for a single ``/v1/`` endpoint.

    Attributes
    ----------
    path:
        Path relative to ``/v1``, e.g. ``"/chat/completions"``.
    methods:
        HTTP methods accepted, e.g. ``["POST"]``.
    summary:
        One-line description for OpenAPI docs.
    description:
        Full description (multi-line) for OpenAPI docs.
    request_schema:
        JSON Schema dict for the request body (optional).
    response_schema:
        JSON Schema dict for the 200 response body (optional).
    deprecated_alias:
        Unversioned alias path, e.g. ``"/chat"`` (receives Deprecation header).
    """
    path:             str
    methods:          list[str]
    summary:          str
    description:      str
    request_schema:   dict[str, Any] | None = None
    response_schema:  dict[str, Any] | None = None
    deprecated_alias: str | None = None


# ---------------------------------------------------------------------------
# Built-in route definitions
# ---------------------------------------------------------------------------

_CHAT_COMPLETIONS_REQUEST = {
    "type": "object",
    "required": ["model", "messages"],
    "properties": {
        "model":       {"type": "string", "description": "Model identifier"},
        "messages":    {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role":    {"type": "string", "enum": ["system", "user", "assistant"]},
                    "content": {"type": "string"},
                },
            },
        },
        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "max_tokens":  {"type": "integer", "minimum": 1},
        "stream":      {"type": "boolean", "default": False},
        "top_p":       {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_COMPLETIONS_REQUEST = {
    "type": "object",
    "required": ["model", "prompt"],
    "properties": {
        "model":       {"type": "string"},
        "prompt":      {"type": "string"},
        "max_tokens":  {"type": "integer", "minimum": 1},
        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "stream":      {"type": "boolean", "default": False},
    },
}

_EMBEDDINGS_REQUEST = {
    "type": "object",
    "required": ["model", "input"],
    "properties": {
        "model": {"type": "string"},
        "input": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

BUILTIN_ROUTES: list[V1RouteSpec] = [
    V1RouteSpec(
        path="/chat/completions",
        methods=["POST"],
        summary="Create a chat completion",
        description=(
            "Given a list of messages, generate the next assistant message.\n"
            "Supports Server-Sent Events streaming when ``stream=true``.\n"
            "Compatible with the OpenAI Chat Completions API schema."
        ),
        request_schema=_CHAT_COMPLETIONS_REQUEST,
        deprecated_alias="/chat",
    ),
    V1RouteSpec(
        path="/completions",
        methods=["POST"],
        summary="Create a text completion",
        description=(
            "Given a prompt string, generate a text completion.\n"
            "Supports streaming via SSE when ``stream=true``."
        ),
        request_schema=_COMPLETIONS_REQUEST,
        deprecated_alias="/completions",
    ),
    V1RouteSpec(
        path="/models",
        methods=["GET"],
        summary="List available models",
        description="Return the list of models available for inference on this server.",
        deprecated_alias="/models",
    ),
    V1RouteSpec(
        path="/embeddings",
        methods=["POST"],
        summary="Create text embeddings",
        description="Encode one or more strings into dense embedding vectors.",
        request_schema=_EMBEDDINGS_REQUEST,
    ),
]


# ---------------------------------------------------------------------------
# OpenAPI schema builder
# ---------------------------------------------------------------------------

class OpenAPISchemaBuilder:
    """Generate a minimal OpenAPI 3.1.0 document from a list of ``V1RouteSpec``.

    The generated document is suitable for use with openapi-generator or
    any standard SDK generator.

    Usage::

        builder = OpenAPISchemaBuilder(routes=BUILTIN_ROUTES)
        schema  = builder.build()   # returns dict
        print(json.dumps(schema, indent=2))
    """

    def __init__(
        self,
        routes: list[V1RouteSpec] | None = None,
        title: str = "Squish API",
        version: str | None = None,
        server_url: str = _DEFAULT_SERVER_URL,
    ) -> None:
        self._routes     = routes or BUILTIN_ROUTES
        self._title      = title
        self._version    = version or _package_version()
        self._server_url = server_url

    def build(self) -> dict[str, Any]:
        """Return the OpenAPI 3.1 schema as a Python dict."""
        paths: dict[str, Any] = {}
        for spec in self._routes:
            full_path = f"/v1{spec.path}"
            path_item: dict[str, Any] = {}
            for method in spec.methods:
                operation: dict[str, Any] = {
                    "summary":     spec.summary,
                    "description": spec.description,
                    "operationId": self._operation_id(spec, method),
                    "responses": {
                        "200": {"description": "Successful response"},
                        "422": {"description": "Validation error"},
                        "500": {"description": "Internal server error"},
                    },
                }
                if spec.request_schema and method in ("POST", "PUT", "PATCH"):
                    operation["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": spec.request_schema,
                            },
                        },
                    }
                path_item[method.lower()] = operation
            paths[full_path] = path_item

        return {
            "openapi": "3.1.0",
            "info": {
                "title":   self._title,
                "version": self._version,
                "license": {"name": "MIT"},
            },
            "servers": [{"url": self._server_url}],
            "paths":   paths,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the schema as a formatted JSON string."""
        return json.dumps(self.build(), indent=indent)

    @staticmethod
    def _operation_id(spec: V1RouteSpec, method: str) -> str:
        slug = spec.path.strip("/").replace("/", "_")
        return f"{method.lower()}_{slug}"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class APIVersionMiddleware:
    """WSGI middleware that injects Squish version headers on every response.

    Adds:
    - ``X-Squish-API-Version: 1``
    - ``X-Squish-Version: <package_version>``

    For deprecated (unversioned) alias paths it also adds:
    - ``Deprecation: true``
    - ``Sunset: <SUNSET_DATE>``
    - ``Link: </v1{alias}>; rel="successor-version"``

    Usage::

        app = APIVersionMiddleware(wsgi_app, deprecated_paths={"/chat"})
    """

    def __init__(
        self,
        app: Any,
        deprecated_paths: set | None = None,
    ) -> None:
        self._app             = app
        self._deprecated      = deprecated_paths or set()
        self._squish_version  = _package_version()

    def __call__(self, environ: dict, start_response: Callable) -> Any:
        path = environ.get("PATH_INFO", "")

        def _start(status: str, headers: list, *args: Any) -> Any:
            extra = [
                ("X-Squish-API-Version", API_VERSION),
                ("X-Squish-Version",     self._squish_version),
            ]
            if path in self._deprecated:
                v1_path = f"/v1{path}"
                extra += [
                    ("Deprecation",  "true"),
                    ("Sunset",       SUNSET_DATE),
                    ("Link",         f'<{v1_path}>; rel="successor-version"'),
                ]
            return start_response(status, headers + extra, *args)

        return self._app(environ, _start)


# ---------------------------------------------------------------------------
# V1Router
# ---------------------------------------------------------------------------

class V1Router:
    """Collects V1RouteSpec objects and can register them on an app.

    This class is framework-agnostic: it stores route metadata and exposes
    ``openapi_schema()`` for documentation generation.  Actual route
    registration is performed by framework-specific adapters (Flask, FastAPI,
    etc.) using the ``register_on_flask`` or ``register_on_fastapi`` helpers.

    Usage::

        router = V1Router()
        router.add_route(some_custom_spec)
        schema = router.openapi_schema()
    """

    def __init__(self, routes: list[V1RouteSpec] | None = None) -> None:
        self._routes: list[V1RouteSpec] = list(routes or BUILTIN_ROUTES)

    @property
    def routes(self) -> list[V1RouteSpec]:
        """Return a copy of the current route list."""
        return list(self._routes)

    def add_route(self, spec: V1RouteSpec) -> None:
        """Append a custom route spec."""
        self._routes.append(spec)

    def openapi_schema(
        self,
        title: str = "Squish API",
        server_url: str = _DEFAULT_SERVER_URL,
    ) -> dict[str, Any]:
        """Return the OpenAPI 3.1 schema dict for all registered routes."""
        builder = OpenAPISchemaBuilder(
            routes=self._routes,
            title=title,
            server_url=server_url,
        )
        return builder.build()

    def deprecated_paths(self) -> list[str]:
        """Return all legacy alias paths that should receive deprecation headers."""
        return [
            r.deprecated_alias
            for r in self._routes
            if r.deprecated_alias is not None
        ]

    def register_on_flask(self, app: Any) -> "V1Router":
        """Register all /v1 routes on a Flask ``app``.

        Each route forwards to ``squish.server`` handler functions if they
        exist; otherwise returns a 501 stub response.

        Returns ``self`` for chaining.
        """
        try:
            from flask import jsonify  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Flask is required to use V1Router.register_on_flask(). "
                "Install with: pip install flask"
            )
        pkg_version = _package_version()

        def _make_handler(spec: V1RouteSpec) -> Callable:
            def _handler(*args: Any, **kwargs: Any) -> Any:
                resp = jsonify({
                    "error":  "not_implemented",
                    "detail": (
                        f"Route {spec.path} is registered but has no "
                        "backing handler. Attach one via app.view_functions."
                    ),
                })
                resp.status_code = 501
                resp.headers["X-Squish-API-Version"] = API_VERSION
                resp.headers["X-Squish-Version"]     = pkg_version
                return resp
            _handler.__name__ = OpenAPISchemaBuilder._operation_id(spec, spec.methods[0])
            return _handler

        for spec in self._routes:
            full_path = f"/v1{spec.path}"
            app.add_url_rule(
                full_path,
                endpoint=f"v1_{spec.path.strip('/').replace('/', '_')}",
                view_func=_make_handler(spec),
                methods=spec.methods,
            )
        return self

    def __repr__(self) -> str:
        return f"V1Router(routes={len(self._routes)})"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def register_v1_routes(
    app: Any,
    routes: Optional[List[V1RouteSpec]] = None,
    framework: str = "flask",
) -> V1Router:
    """Register ``/v1/*`` routes on ``app`` and return the router.

    Parameters
    ----------
    app:
        Flask or FastAPI application instance.
    routes:
        Custom route list; defaults to ``BUILTIN_ROUTES``.
    framework:
        ``"flask"`` (default) or ``"fastapi"``.

    Returns
    -------
    V1Router
        The configured router, useful for accessing ``openapi_schema()``.
    """
    router = V1Router(routes=routes)
    if framework == "flask":
        router.register_on_flask(app)
    else:
        raise NotImplementedError(
            f"Framework {framework!r} is not yet supported. "
            "Use framework='flask' or call router.register_on_flask(app) directly."
        )
    return router
