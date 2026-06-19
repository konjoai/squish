"""Behavioral coverage for ``squish.api.v1_router`` — the OpenAPI JSON
serializer, the WSGI version/deprecation middleware, the V1Router metadata
API, and the Flask registration + convenience entrypoint.

Flask is not a dependency, so the registration body uses an injected fake
``flask`` module (and the missing-Flask path is asserted too). Pure-Python —
no MLX.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from squish.api.v1_router import (
    BUILTIN_ROUTES,
    OpenAPISchemaBuilder,
    APIVersionMiddleware,
    V1Router,
    V1RouteSpec,
    register_v1_routes,
)


# ── OpenAPISchemaBuilder.to_json ────────────────────────────────────────────


def test_to_json_serializes_schema():
    builder = OpenAPISchemaBuilder(routes=BUILTIN_ROUTES, title="T", server_url="http://x")
    text = builder.to_json(indent=2)
    doc = json.loads(text)
    assert doc["openapi"] == "3.1.0" and doc["info"]["title"] == "T"
    assert "/v1/chat/completions" in doc["paths"]


# ── APIVersionMiddleware ────────────────────────────────────────────────────


def _run_middleware(mw, path):
    captured = {}

    def downstream_app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    def start_response(status, headers, *args):
        captured["status"] = status
        captured["headers"] = headers

    body = mw({"PATH_INFO": path}, start_response)
    captured["body"] = body
    return captured


def test_middleware_adds_version_headers_on_normal_path():
    mw = APIVersionMiddleware(lambda e, s: s("200 OK", []) or [b""],
                             deprecated_paths={"/chat"})
    cap = _run_middleware(mw, "/v1/chat/completions")
    names = [h[0] for h in cap["headers"]]
    assert "X-Squish-API-Version" in names
    assert "X-Squish-Version" in names
    assert "Deprecation" not in names  # not a deprecated path


def test_middleware_adds_deprecation_headers_on_alias_path():
    mw = APIVersionMiddleware(lambda e, s: s("200 OK", []) or [b""],
                             deprecated_paths={"/chat"})
    cap = _run_middleware(mw, "/chat")
    names = [h[0] for h in cap["headers"]]
    assert "Deprecation" in names and "Sunset" in names
    link = dict(cap["headers"])["Link"]
    assert link == '</v1/chat>; rel="successor-version"'


def test_middleware_default_deprecated_paths_empty():
    mw = APIVersionMiddleware(lambda e, s: s("200 OK", []) or [b""])  # no set passed
    cap = _run_middleware(mw, "/anything")
    names = [h[0] for h in cap["headers"]]
    assert "Deprecation" not in names


# ── V1Router metadata API ───────────────────────────────────────────────────


def _spec(path="/custom"):
    return V1RouteSpec(path=path, methods=["GET"], summary="s", description="d")


def test_routes_returns_copy():
    r = V1Router()
    routes = r.routes
    routes.append(_spec())  # mutating the returned list must not affect the router
    assert len(r.routes) == len(BUILTIN_ROUTES)


def test_add_route_appends():
    # Note: V1Router([]) falls back to BUILTIN_ROUTES (`routes or BUILTIN_ROUTES`),
    # so pass an explicit non-empty list to start from a known base.
    r = V1Router(routes=[_spec("/base")])
    r.add_route(_spec("/x"))
    assert [s.path for s in r.routes] == ["/base", "/x"]


def test_deprecated_paths_collects_aliases():
    r = V1Router()
    aliases = r.deprecated_paths()
    # BUILTIN_ROUTES includes several deprecated aliases (e.g. /chat).
    assert "/chat" in aliases
    # A route with no alias contributes nothing.
    r2 = V1Router(routes=[_spec("/no-alias")])
    assert r2.deprecated_paths() == []


def test_openapi_schema_builds():
    schema = V1Router().openapi_schema(title="Squish", server_url="http://h")
    assert schema["info"]["title"] == "Squish"


def test_repr_reports_route_count():
    assert repr(V1Router(routes=[_spec()])) == "V1Router(routes=1)"


# ── Flask registration ──────────────────────────────────────────────────────


def _install_fake_flask(monkeypatch):
    fake_flask = types.ModuleType("flask")

    class _Resp:
        def __init__(self, data):
            self.data = data
            self.status_code = 200
            self.headers: dict = {}

    fake_flask.jsonify = lambda d: _Resp(d)
    monkeypatch.setitem(sys.modules, "flask", fake_flask)


class _FakeApp:
    def __init__(self):
        self.rules = []

    def add_url_rule(self, path, endpoint, view_func, methods):
        self.rules.append((path, endpoint, view_func, methods))


def test_register_on_flask_registers_routes_and_stub_handler(monkeypatch):
    _install_fake_flask(monkeypatch)
    app = _FakeApp()
    router = V1Router()
    result = router.register_on_flask(app)
    assert result is router
    assert len(app.rules) == len(BUILTIN_ROUTES)
    # Every rule path is /v1-prefixed.
    assert all(path.startswith("/v1") for path, *_ in app.rules)
    # Invoking a stub handler returns the 501 not-implemented response + headers.
    handler = app.rules[0][2]
    resp = handler()
    assert resp.status_code == 501
    assert resp.headers["X-Squish-API-Version"] == "1"
    assert "X-Squish-Version" in resp.headers


def test_register_on_flask_without_flask_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "flask", None)  # import → ImportError
    with pytest.raises(ImportError, match="Flask is required"):
        V1Router().register_on_flask(object())


# ── register_v1_routes convenience ──────────────────────────────────────────


def test_register_v1_routes_flask(monkeypatch):
    _install_fake_flask(monkeypatch)
    app = _FakeApp()
    router = register_v1_routes(app, framework="flask")
    assert isinstance(router, V1Router)
    assert len(app.rules) == len(BUILTIN_ROUTES)


def test_register_v1_routes_unsupported_framework_raises():
    with pytest.raises(NotImplementedError, match="not yet supported"):
        register_v1_routes(object(), framework="fastapi")
