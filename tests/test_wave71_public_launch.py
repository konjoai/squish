"""tests/test_wave71_public_launch.py — Wave 71 public launch tests.

Tests for:
  - squish/api/v1_router.py         (V1Router, OpenAPISchemaBuilder, APIVersionMiddleware)
  - squish/packaging/release_validator.py  (ReleaseValidator)
  - squish/packaging/pypi_manifest.py      (PyPIManifest)

All tests are deterministic and do NOT require a live server, running
pytest suite, or built wheel.  File-system-dependent logic is tested via
temporary directories and mocking.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
from dataclasses import FrozenInstanceError

import pytest


# ===========================================================================
# V1Router / OpenAPISchemaBuilder / APIVersionMiddleware
# ===========================================================================
from squish.api.v1_router import (
    API_VERSION,
    BUILTIN_ROUTES,
    SUNSET_DATE,
    APIVersionMiddleware,
    OpenAPISchemaBuilder,
    V1RouteSpec,
    V1Router,
    register_v1_routes,
)


# -- V1RouteSpec -------------------------------------------------------------

class TestV1RouteSpec:
    def test_frozen(self):
        spec = V1RouteSpec(
            path="/chat/completions",
            methods=["POST"],
            summary="test",
            description="test desc",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            spec.path = "/v2/chat"  # type: ignore[misc]

    def test_optional_fields_default_none(self):
        spec = V1RouteSpec(path="/models", methods=["GET"], summary="s", description="d")
        assert spec.request_schema is None
        assert spec.response_schema is None
        assert spec.deprecated_alias is None

    def test_with_all_fields(self):
        spec = V1RouteSpec(
            path="/chat/completions",
            methods=["POST"],
            summary="Chat",
            description="Chat endpoint",
            request_schema={"type": "object"},
            response_schema={"type": "object"},
            deprecated_alias="/chat",
        )
        assert spec.deprecated_alias == "/chat"
        assert spec.request_schema == {"type": "object"}


# -- BUILTIN_ROUTES ----------------------------------------------------------

class TestBuiltinRoutes:
    def test_builtin_routes_not_empty(self):
        assert len(BUILTIN_ROUTES) >= 4

    def test_chat_completions_route_exists(self):
        paths = {r.path for r in BUILTIN_ROUTES}
        assert "/chat/completions" in paths

    def test_completions_route_exists(self):
        paths = {r.path for r in BUILTIN_ROUTES}
        assert "/completions" in paths

    def test_models_route_exists(self):
        paths = {r.path for r in BUILTIN_ROUTES}
        assert "/models" in paths

    def test_embeddings_route_exists(self):
        paths = {r.path for r in BUILTIN_ROUTES}
        assert "/embeddings" in paths

    def test_chat_completions_has_post(self):
        chat = next(r for r in BUILTIN_ROUTES if r.path == "/chat/completions")
        assert "POST" in chat.methods

    def test_models_has_get(self):
        models = next(r for r in BUILTIN_ROUTES if r.path == "/models")
        assert "GET" in models.methods

    def test_chat_completions_has_request_schema(self):
        chat = next(r for r in BUILTIN_ROUTES if r.path == "/chat/completions")
        assert chat.request_schema is not None
        assert "messages" in chat.request_schema.get("properties", {})

    def test_deprecated_alias_present(self):
        chat = next(r for r in BUILTIN_ROUTES if r.path == "/chat/completions")
        assert chat.deprecated_alias is not None


# -- OpenAPISchemaBuilder ----------------------------------------------------

class TestOpenAPISchemaBuilder:
    def test_build_returns_dict(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        assert isinstance(schema, dict)

    def test_openapi_version_3_1(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        assert schema["openapi"] == "3.1.0"

    def test_info_title(self):
        builder = OpenAPISchemaBuilder(title="My API")
        schema  = builder.build()
        assert schema["info"]["title"] == "My API"

    def test_paths_contain_v1_prefix(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        for path in schema["paths"]:
            assert path.startswith("/v1/"), f"Path {path!r} missing /v1/ prefix"

    def test_post_includes_request_body(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        chat = schema["paths"]["/v1/chat/completions"]
        assert "post" in chat
        assert "requestBody" in chat["post"]

    def test_get_no_request_body(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        models = schema["paths"]["/v1/models"]
        assert "get" in models
        assert "requestBody" not in models["get"]

    def test_operation_id_format(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        op = schema["paths"]["/v1/chat/completions"]["post"]
        assert "operationId" in op
        assert "chat_completions" in op["operationId"]

    def test_to_json_produces_valid_json(self):
        builder = OpenAPISchemaBuilder()
        raw     = builder.to_json()
        parsed  = json.loads(raw)
        assert parsed["openapi"] == "3.1.0"

    def test_server_url_in_schema(self):
        builder = OpenAPISchemaBuilder(server_url="http://example.com:9999")
        schema  = builder.build()
        urls = [s["url"] for s in schema["servers"]]
        assert "http://example.com:9999" in urls

    def test_responses_include_200(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        for path_item in schema["paths"].values():
            for method_obj in path_item.values():
                responses = method_obj.get("responses", {})
                assert "200" in responses, f"No 200 in {path_item}"

    def test_license_mit(self):
        builder = OpenAPISchemaBuilder()
        schema  = builder.build()
        assert schema["info"]["license"]["name"] == "MIT"


# -- APIVersionMiddleware ----------------------------------------------------

class TestAPIVersionMiddleware:
    def _make_app(self):
        def wsgi_app(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"OK"]
        return wsgi_app

    def _call_middleware(self, app, path="/v1/models", deprecated: set = None):
        mw      = APIVersionMiddleware(app, deprecated_paths=deprecated or set())
        captured_headers = []

        def capture_start(status, headers, *args):
            captured_headers.extend(headers)

        environ = {"PATH_INFO": path, "REQUEST_METHOD": "GET"}
        mw(environ, capture_start)
        return {k: v for k, v in captured_headers}

    def test_api_version_header_present(self):
        headers = self._call_middleware(self._make_app())
        assert "X-Squish-API-Version" in headers
        assert headers["X-Squish-API-Version"] == API_VERSION

    def test_squish_version_header_present(self):
        headers = self._call_middleware(self._make_app())
        assert "X-Squish-Version" in headers

    def test_no_deprecation_on_v1_path(self):
        headers = self._call_middleware(self._make_app(), path="/v1/models")
        assert "Deprecation" not in headers

    def test_deprecation_header_on_legacy_path(self):
        headers = self._call_middleware(
            self._make_app(),
            path="/chat",
            deprecated={"/chat"},
        )
        assert headers.get("Deprecation") == "true"

    def test_sunset_header_on_legacy_path(self):
        headers = self._call_middleware(
            self._make_app(),
            path="/chat",
            deprecated={"/chat"},
        )
        assert headers.get("Sunset") == SUNSET_DATE

    def test_link_header_on_legacy_path(self):
        headers = self._call_middleware(
            self._make_app(),
            path="/chat",
            deprecated={"/chat"},
        )
        assert "Link" in headers
        assert "successor-version" in headers["Link"]

    def test_no_link_header_on_v1_path(self):
        headers = self._call_middleware(self._make_app(), path="/v1/chat/completions")
        assert "Link" not in headers


# -- V1Router ----------------------------------------------------------------

class TestV1Router:
    def test_default_routes_equal_builtin(self):
        router = V1Router()
        assert len(router.routes) == len(BUILTIN_ROUTES)

    def test_add_route_increases_count(self):
        router = V1Router()
        orig   = len(router.routes)
        router.add_route(V1RouteSpec("/health", ["GET"], "Health", "Health check"))
        assert len(router.routes) == orig + 1

    def test_routes_property_returns_copy(self):
        router  = V1Router()
        routes1 = router.routes
        routes2 = router.routes
        assert routes1 is not routes2   # new list each time

    def test_openapi_schema_returns_dict(self):
        router = V1Router()
        schema = router.openapi_schema()
        assert isinstance(schema, dict)
        assert schema["openapi"] == "3.1.0"

    def test_deprecated_paths_list(self):
        router = V1Router()
        dpaths = router.deprecated_paths()
        # chat/completions has a deprecated alias
        assert any("/chat" in p for p in dpaths if p)

    def test_repr(self):
        router = V1Router()
        r = repr(router)
        assert "V1Router(routes=" in r

    def test_register_flask_requires_flask(self):
        """register_on_flask should raise ImportError if flask missing."""
        router = V1Router()
        mock_app = MagicMock()
        with patch.dict("sys.modules", {"flask": None}):
            with pytest.raises((ImportError, TypeError)):
                router.register_on_flask(mock_app)

    def test_custom_routes_passed_to_schema(self):
        custom = [V1RouteSpec("/ping", ["GET"], "Ping", "Ping endpoint")]
        router = V1Router(routes=custom)
        schema = router.openapi_schema()
        assert "/v1/ping" in schema["paths"]


# -- register_v1_routes function ---------------------------------------------

class TestRegisterV1Routes:
    def test_unsupported_framework_raises(self):
        mock_app = MagicMock()
        with pytest.raises(NotImplementedError, match="fastapi"):
            register_v1_routes(mock_app, framework="fastapi")

    def test_flask_framework_returns_router(self):
        mock_app = MagicMock()
        mock_app.add_url_rule = MagicMock()
        with patch.dict("sys.modules", {"flask": MagicMock()}):
            router = register_v1_routes(mock_app, framework="flask")
        assert isinstance(router, V1Router)


# ===========================================================================
# ReleaseValidator tests
# ===========================================================================
from squish.packaging.release_validator import (
    CheckResult,
    ReleaseConfig,
    ReleaseReport,
    ReleaseValidator,
)


# -- CheckResult / ReleaseReport ---------------------------------------------

class TestCheckResult:
    def test_passed(self):
        r = CheckResult("test", True, "ok")
        assert r.passed is True
        assert r.mandatory is True   # default

    def test_advisory(self):
        r = CheckResult("test", False, "warn", mandatory=False)
        assert r.mandatory is False


class TestReleaseReport:
    def test_passed_all_mandatory(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [
            CheckResult("a", True, "ok", mandatory=True),
            CheckResult("b", True, "ok", mandatory=True),
        ]
        assert rpt.passed is True

    def test_failed_one_mandatory(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [
            CheckResult("a", True, "ok",   mandatory=True),
            CheckResult("b", False, "fail", mandatory=True),
        ]
        assert rpt.passed is False

    def test_advisory_failure_does_not_block(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [
            CheckResult("a", True, "ok", mandatory=True),
            CheckResult("warn", False, "warn", mandatory=False),
        ]
        assert rpt.passed is True

    def test_failures_property(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [
            CheckResult("pass", True,  "ok",  mandatory=True),
            CheckResult("fail", False, "err", mandatory=True),
        ]
        assert len(rpt.failures) == 1
        assert rpt.failures[0].name == "fail"

    def test_warnings_property(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [
            CheckResult("advisory", False, "warn", mandatory=False),
        ]
        assert len(rpt.warnings) == 1

    def test_summary_contains_version(self):
        rpt = ReleaseReport("45.0.0")
        s = rpt.summary()
        assert "45.0.0" in s

    def test_summary_contains_pass_fail(self):
        rpt = ReleaseReport("45.0.0")
        rpt.results = [CheckResult("x", True, "ok")]
        assert "PASS" in rpt.summary() or "FAIL" in rpt.summary()


# -- ReleaseConfig -----------------------------------------------------------

class TestReleaseConfig:
    def test_defaults(self):
        cfg = ReleaseConfig()
        assert cfg.pytest_pass_threshold == 0.99
        assert "name" in cfg.required_pyproject_fields

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="pytest_pass_threshold"):
            ReleaseConfig(pytest_pass_threshold=0.0)

    def test_threshold_one_valid(self):
        cfg = ReleaseConfig(pytest_pass_threshold=1.0)
        assert cfg.pytest_pass_threshold == 1.0


# -- ReleaseValidator with temp directories ----------------------------------

class TestReleaseValidatorChecks:
    def _make_repo(self, tmp_path: Path, version: str = "45") -> Path:
        """Set up a minimal repository structure for validation tests."""
        # pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\n'
            'name = "squish"\n'
            'version = "45.0.0"\n'
            'description = "A test project"\n'
            'requires-python = ">=3.9"\n'
            'license = {text = "MIT"}\n'
            'authors = [{name = "Test"}]\n'
            'classifiers = ["License :: OSI Approved :: MIT License"]\n',
            encoding="utf-8",
        )
        # CHANGELOG.md
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            f"# Changelog\n\n## [{version}.0.0] - 2025-01-01\n\nSome changes.\n",
            encoding="utf-8",
        )
        # README.md
        readme = tmp_path / "README.md"
        readme.write_text("# Squish\nSee https://arxiv.org/abs/1234.56789\n", encoding="utf-8")
        # squish source dir with one .py file
        src = tmp_path / "squish"
        src.mkdir()
        (src / "__init__.py").write_text(
            "# SPDX-License-Identifier: MIT\n\"\"\"squish package.\"\"\"\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_changelog_check_passes(self, tmp_path):
        root = self._make_repo(tmp_path, version="45")
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_changelog()
        assert result.passed is True

    def test_changelog_check_fails_wrong_version(self, tmp_path):
        root = self._make_repo(tmp_path, version="44")
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_changelog()
        assert result.passed is False

    def test_changelog_check_fails_missing_file(self, tmp_path):
        root = self._make_repo(tmp_path)
        (root / "CHANGELOG.md").unlink()
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_changelog()
        assert result.passed is False

    def test_spdx_check_passes(self, tmp_path):
        root = self._make_repo(tmp_path)
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_spdx_headers()
        assert result.passed is True

    def test_spdx_check_fails_missing_identifier(self, tmp_path):
        root = self._make_repo(tmp_path)
        (root / "squish" / "__init__.py").write_text(
            "\"\"\"No SPDX here.\"\"\"\n", encoding="utf-8"
        )
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_spdx_headers()
        assert result.passed is False

    def test_spdx_check_fails_missing_src_dir(self, tmp_path):
        root = tmp_path  # no squish/ subdir
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_spdx_headers()
        assert result.passed is False

    def test_pyproject_check_passes(self, tmp_path):
        root = self._make_repo(tmp_path)
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_pyproject()
        assert result.passed is True

    def test_pyproject_check_fails_missing_file(self, tmp_path):
        root = self._make_repo(tmp_path)
        (root / "pyproject.toml").unlink()
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_pyproject()
        assert result.passed is False

    def test_arxiv_advisory_passes(self, tmp_path):
        root = self._make_repo(tmp_path)
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_arxiv_reference()
        assert result.passed is True
        assert result.mandatory is False

    def test_arxiv_advisory_fails_no_readme(self, tmp_path):
        root = self._make_repo(tmp_path)
        (root / "README.md").unlink()
        validator = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        result = validator._check_arxiv_reference()
        assert result.mandatory is False   # still advisory

    def test_empty_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            ReleaseValidator("", ReleaseConfig())

    def test_repr(self, tmp_path):
        root = self._make_repo(tmp_path)
        v = ReleaseValidator("45.0.0", ReleaseConfig(repo_root=root))
        assert "45.0.0" in repr(v)

    def test_parse_pytest_counts_basic(self):
        output = "1234 passed, 5 failed, 12 warnings in 45.32s"
        passed, failed = ReleaseValidator._parse_pytest_counts(output)
        assert passed == 1234
        assert failed == 5

    def test_parse_pytest_counts_all_passed(self):
        output = "500 passed in 10.0s"
        passed, failed = ReleaseValidator._parse_pytest_counts(output)
        assert passed == 500
        assert failed == 0


# ===========================================================================
# PyPIManifest tests
# ===========================================================================
from squish.packaging.pypi_manifest import (
    ManifestConfig,
    ManifestRule,
    PyPIManifest,
    PyPIManifestReport,
    WheelEntry,
)


class TestManifestRule:
    def test_str_representation(self):
        rule = ManifestRule("include", "squish/**/*.py")
        assert str(rule) == "include squish/**/*.py"

    def test_global_exclude(self):
        rule = ManifestRule("global-exclude", "*.pyc")
        assert "global-exclude" in str(rule)

    def test_recursive_include(self):
        rule = ManifestRule("recursive-include", "squish *.metal")
        assert "recursive-include" in str(rule)


class TestWheelEntry:
    def test_path_field(self):
        entry = WheelEntry(path="squish/__init__.py", size_kb=1.2)
        assert entry.path == "squish/__init__.py"
        assert entry.size_kb == 1.2


class TestPyPIManifestReport:
    def test_passed_no_flagged(self):
        rpt = PyPIManifestReport()
        assert rpt.passed is True

    def test_failed_with_flagged(self):
        rpt = PyPIManifestReport(flagged=["dev/script.py"])
        assert rpt.passed is False

    def test_summary_contains_pass_fail(self):
        rpt = PyPIManifestReport()
        s = rpt.summary()
        assert "PASS" in s or "FAIL" in s

    def test_summary_with_violation(self):
        rpt = PyPIManifestReport(flagged=["dev/bad_file.py"])
        s = rpt.summary()
        assert "dev/bad_file.py" in s or "Violation" in s

    def test_total_size_computed(self):
        rpt = PyPIManifestReport(
            wheel_entries=[
                WheelEntry("squish/__init__.py", 2.0),
                WheelEntry("squish/server.py", 10.0),
            ],
            total_size_kb=12.0,
        )
        assert rpt.total_size_kb == 12.0


class TestManifestConfig:
    def test_defaults(self):
        cfg = ManifestConfig()
        assert "squish/**/*.py" in cfg.include_patterns
        assert "dev" in cfg.exclude_dirs
        assert cfg.max_wheel_size_kb > 0

    def test_allowlist_starts_with_squish(self):
        cfg = ManifestConfig()
        assert any(p.startswith("squish") for p in cfg.allowlist_prefixes)


class TestPyPIManifestRules:
    def test_generate_rules_not_empty(self):
        m = PyPIManifest()
        rules = m.generate_rules()
        assert len(rules) > 0

    def test_rules_have_excludes_for_dev(self):
        m = PyPIManifest()
        rules = m.generate_rules()
        excl_patterns = [str(r) for r in rules if "exclude" in r.directive]
        assert any("dev" in p for p in excl_patterns)

    def test_rules_have_pyc_global_exclude(self):
        m = PyPIManifest()
        rules = m.generate_rules()
        assert any("*.pyc" in str(r) for r in rules)

    def test_rules_include_squish_py(self):
        m = PyPIManifest()
        rules = m.generate_rules()
        include_patterns = [str(r) for r in rules if "include" in r.directive]
        assert any("squish" in p and ".py" in p for p in include_patterns)

    def test_rules_include_metal(self):
        m = PyPIManifest()
        rules = m.generate_rules()
        include_patterns = [str(r) for r in rules]
        assert any(".metal" in p for p in include_patterns)


class TestPyPIManifestWrite:
    def test_write_manifest_in(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        path = m.write_manifest_in()
        assert path.exists()
        content = path.read_text()
        assert "Auto-generated" in content

    def test_written_manifest_has_excludes(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        m.write_manifest_in()
        content = (tmp_path / "MANIFEST.in").read_text()
        assert "global-exclude" in content or "recursive-exclude" in content

    def test_written_manifest_has_includes(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        m.write_manifest_in()
        content = (tmp_path / "MANIFEST.in").read_text()
        assert "include" in content


class TestPyPIManifestWheelValidation:
    def _make_wheel(self, tmp_path: Path, files: list) -> Path:
        wheel_path = tmp_path / "squish-45.0.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel_path, "w") as zf:
            for f in files:
                zf.writestr(f, "# content")
        return wheel_path

    def test_valid_wheel_no_flagged(self, tmp_path):
        wheel = self._make_wheel(tmp_path, [
            "squish/__init__.py",
            "squish/server.py",
            "squish-45.0.0.dist-info/METADATA",
        ])
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        flagged = m.validate_wheel(wheel)
        assert flagged == []

    def test_invalid_wheel_flags_dev_files(self, tmp_path):
        wheel = self._make_wheel(tmp_path, [
            "squish/__init__.py",
            "dev/secret_script.py",   # should be flagged
        ])
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        flagged = m.validate_wheel(wheel)
        assert any("dev/" in f for f in flagged)

    def test_wheel_entries_listing(self, tmp_path):
        wheel = self._make_wheel(tmp_path, [
            "squish/__init__.py",
            "squish/server.py",
        ])
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        entries = m.wheel_entries(wheel)
        paths = [e.path for e in entries]
        assert "squish/__init__.py" in paths
        assert "squish/server.py" in paths

    def test_wheel_not_found_raises(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        with pytest.raises(FileNotFoundError):
            m.validate_wheel(tmp_path / "nonexistent.whl")

    def test_wheel_entries_empty_when_not_found(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        entries = m.wheel_entries(tmp_path / "missing.whl")
        assert entries == []

    def test_build_and_validate_no_wheel(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        rpt = m.build_and_validate()
        assert isinstance(rpt, PyPIManifestReport)
        assert rpt.passed is True   # no wheel to flag

    def test_build_and_validate_with_clean_wheel(self, tmp_path):
        wheel = self._make_wheel(tmp_path, ["squish/__init__.py"])
        cfg   = ManifestConfig(repo_root=tmp_path)
        m     = PyPIManifest(cfg)
        rpt   = m.build_and_validate(wheel_path=wheel)
        assert rpt.passed is True
        assert len(rpt.wheel_entries) == 1

    def test_build_and_validate_with_flagged_wheel(self, tmp_path):
        wheel = self._make_wheel(tmp_path, [
            "squish/__init__.py",
            "tests/test_something.py",   # not in allowlist
        ])
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        rpt = m.build_and_validate(wheel_path=wheel)
        assert rpt.passed is False


class TestPyPIManifestExcludedDirs:
    def test_find_excluded_dirs_empty(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        # No excluded dirs present → empty list
        found = m.find_excluded_files_in_tree()
        assert found == []

    def test_find_excluded_dirs_present(self, tmp_path):
        (tmp_path / "dev").mkdir()
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        found = m.find_excluded_files_in_tree()
        assert "dev" in found

    def test_repr(self, tmp_path):
        cfg = ManifestConfig(repo_root=tmp_path)
        m   = PyPIManifest(cfg)
        assert "PyPIManifest(" in repr(m)
