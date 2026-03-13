"""tests/test_tool_cache_unit.py

Full-coverage unit tests for squish/tool_cache.py.

Covers:
  ToolSchema        — valid construction, invalid name/parameters, properties
  ToolCacheStats    — success_rate zero and non-zero
  ToolSchemaCache   — init validation, register (valid/duplicate/missing keys/
                      empty name/full cache), get/get_by_hash (hit/miss),
                      validate_call (unknown tool/missing params/valid),
                      n_cached, cache_hit_rate, stats()
  ToolRouter        — route (valid/validation-failure/missing-handler),
                      n_routes, n_validation_failures, stats()
"""
from __future__ import annotations

import pytest

from squish.tool_cache import (
    ToolCacheStats,
    ToolRouter,
    ToolSchema,
    ToolSchemaCache,
)


# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_valid_construction(self):
        ts = ToolSchema(name="get_weather", parameters={"city": "string"})
        assert ts.name == "get_weather"
        assert ts.parameters == {"city": "string"}
        assert ts.description == ""
        assert ts.handler_id == ""
        assert ts.schema_hash == ""

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolSchema(name="", parameters={})

    def test_non_dict_parameters_raises(self):
        with pytest.raises(ValueError, match="parameters must be a dict"):
            ToolSchema(name="tool", parameters=["city"])  # type: ignore[arg-type]

    def test_n_params_empty(self):
        ts = ToolSchema(name="tool", parameters={})
        assert ts.n_params == 0

    def test_n_params_nonzero(self):
        ts = ToolSchema(name="tool", parameters={"a": "int", "b": "str"})
        assert ts.n_params == 2

    def test_required_params(self):
        ts = ToolSchema(name="tool", parameters={"x": "float", "y": "float"})
        assert set(ts.required_params) == {"x", "y"}

    def test_required_params_empty(self):
        ts = ToolSchema(name="tool", parameters={})
        assert ts.required_params == []


# ---------------------------------------------------------------------------
# ToolCacheStats
# ---------------------------------------------------------------------------


class TestToolCacheStats:
    def test_success_rate_zero_validations(self):
        s = ToolCacheStats()
        assert s.success_rate == 0.0

    def test_success_rate_all_pass(self):
        s = ToolCacheStats(n_validations=5, n_failures=0)
        assert s.success_rate == pytest.approx(1.0)

    def test_success_rate_some_failures(self):
        s = ToolCacheStats(n_validations=10, n_failures=3)
        assert s.success_rate == pytest.approx(0.7)

    def test_success_rate_all_failures(self):
        s = ToolCacheStats(n_validations=4, n_failures=4)
        assert s.success_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ToolSchemaCache
# ---------------------------------------------------------------------------


class TestToolSchemaCacheInit:
    def test_default_max_entries(self):
        c = ToolSchemaCache()
        assert c._max_entries == 512

    def test_custom_max_entries(self):
        c = ToolSchemaCache(max_entries=10)
        assert c._max_entries == 10

    def test_max_entries_zero_raises(self):
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            ToolSchemaCache(max_entries=0)

    def test_max_entries_negative_raises(self):
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            ToolSchemaCache(max_entries=-5)


class TestRegister:
    def _cache(self, max_entries=10):
        return ToolSchemaCache(max_entries=max_entries)

    def _schema(self, name="tool", params=None):
        return {
            "name": name,
            "parameters": params if params is not None else {"x": "int"},
        }

    def test_register_valid(self):
        c = self._cache()
        h = c.register(self._schema())
        assert isinstance(h, str)
        assert len(h) == 16

    def test_register_missing_name_raises(self):
        c = self._cache()
        with pytest.raises(ValueError, match="must contain a 'name' key"):
            c.register({"parameters": {}})

    def test_register_missing_parameters_raises(self):
        c = self._cache()
        with pytest.raises(ValueError, match="must contain a 'parameters' key"):
            c.register({"name": "tool"})

    def test_register_empty_name_raises(self):
        c = self._cache()
        with pytest.raises(ValueError, match="Tool name must not be empty"):
            c.register({"name": "", "parameters": {}})

    def test_register_duplicate_is_idempotent(self):
        c = self._cache()
        h1 = c.register(self._schema())
        h2 = c.register(self._schema())
        assert h1 == h2
        assert c.n_cached == 1

    def test_register_cache_full_raises(self):
        c = self._cache(max_entries=2)
        c.register({"name": "a", "parameters": {}})
        c.register({"name": "b", "parameters": {}})
        with pytest.raises(RuntimeError, match="is full"):
            c.register({"name": "c", "parameters": {}})

    def test_register_with_description_and_handler_id(self):
        c = self._cache()
        c.register({
            "name": "do_thing",
            "parameters": {"arg": "str"},
            "description": "Does a thing",
            "handler_id": "h1",
        })
        schema = c.get("do_thing")
        assert schema is not None
        assert schema.description == "Does a thing"
        assert schema.handler_id == "h1"

    def test_register_increments_n_cached(self):
        c = self._cache()
        assert c.n_cached == 0
        c.register(self._schema("a"))
        assert c.n_cached == 1
        c.register(self._schema("b"))
        assert c.n_cached == 2


class TestGetLookup:
    def _populated_cache(self):
        c = ToolSchemaCache()
        h = c.register({"name": "greet", "parameters": {"name": "str"}})
        return c, h

    def test_get_hit(self):
        c, _ = self._populated_cache()
        schema = c.get("greet")
        assert schema is not None
        assert schema.name == "greet"

    def test_get_miss(self):
        c, _ = self._populated_cache()
        schema = c.get("nonexistent")
        assert schema is None

    def test_get_by_hash_hit(self):
        c, h = self._populated_cache()
        schema = c.get_by_hash(h)
        assert schema is not None
        assert schema.schema_hash == h

    def test_get_by_hash_miss(self):
        c, _ = self._populated_cache()
        schema = c.get_by_hash("deadbeefdeadbeef")
        assert schema is None

    def test_cache_hit_rate_zero(self):
        c = ToolSchemaCache()
        assert c.cache_hit_rate == 0.0

    def test_cache_hit_rate_after_hits_and_misses(self):
        c, h = self._populated_cache()
        c.get("greet")      # hit
        c.get("greet")      # hit
        c.get("missing")    # miss
        # 2 hits, 1 miss → 2/3
        assert c.cache_hit_rate == pytest.approx(2 / 3)


class TestValidateCall:
    def _populated_cache(self):
        c = ToolSchemaCache()
        c.register({"name": "add", "parameters": {"a": "int", "b": "int"}})
        return c

    def test_valid_call(self):
        c = self._populated_cache()
        ok, msg = c.validate_call("add", {"a": 1, "b": 2})
        assert ok is True
        assert msg == ""

    def test_extra_args_allowed(self):
        c = self._populated_cache()
        ok, msg = c.validate_call("add", {"a": 1, "b": 2, "extra": "ignored"})
        assert ok is True

    def test_missing_param(self):
        c = self._populated_cache()
        ok, msg = c.validate_call("add", {"a": 1})  # missing "b"
        assert ok is False
        assert "b" in msg

    def test_unknown_tool(self):
        c = self._populated_cache()
        ok, msg = c.validate_call("fly", {"altitude": 100})
        assert ok is False
        assert "fly" in msg

    def test_validation_failure_increments_counter(self):
        c = self._populated_cache()
        c.validate_call("unknown_tool", {})
        s = c.stats()
        assert s.n_failures == 1

    def test_successful_validation_does_not_increment_failure_counter(self):
        c = self._populated_cache()
        c.validate_call("add", {"a": 1, "b": 2})
        s = c.stats()
        assert s.n_failures == 0


class TestSchemaCacheStats:
    def test_stats_after_registration(self):
        c = ToolSchemaCache()
        c.register({"name": "f", "parameters": {"x": "int"}})
        s = c.stats()
        assert s.n_registrations == 1
        assert s.n_routes == 0

    def test_stats_validation_counts(self):
        c = ToolSchemaCache()
        c.register({"name": "f", "parameters": {"x": "int"}})
        c.validate_call("f", {"x": 1})
        c.validate_call("f", {})  # missing x → failure
        s = c.stats()
        assert s.n_validations == 2
        assert s.n_failures == 1


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


class TestToolRouter:
    def _setup(self):
        cache = ToolSchemaCache()
        cache.register({"name": "greet", "parameters": {"name": "str"}})
        router = ToolRouter(cache)
        return cache, router

    def test_route_valid(self):
        cache, router = self._setup()

        def handler(args):
            return f"Hello, {args['name']}!"

        result = router.route("greet", {"name": "World"}, handlers={"greet": handler})
        assert result == "Hello, World!"

    def test_route_increments_n_routes(self):
        cache, router = self._setup()
        router.route("greet", {"name": "X"}, handlers={"greet": lambda a: None})
        assert router.n_routes == 1

    def test_route_validation_failure_raises_value_error(self):
        cache, router = self._setup()
        with pytest.raises(ValueError, match="validation failed"):
            router.route("greet", {}, handlers={"greet": lambda a: None})  # missing "name"

    def test_route_validation_failure_increments_failure_counter(self):
        cache, router = self._setup()
        try:
            router.route("greet", {}, handlers={"greet": lambda a: None})
        except ValueError:
            pass
        assert router.n_validation_failures == 1

    def test_route_missing_handler_raises_key_error(self):
        cache, router = self._setup()
        with pytest.raises(KeyError, match="No handler registered"):
            router.route("greet", {"name": "Y"}, handlers={})

    def test_route_unknown_tool_raises_value_error(self):
        cache, router = self._setup()
        with pytest.raises(ValueError, match="validation failed"):
            router.route("fly", {}, handlers={"fly": lambda a: None})

    def test_n_routes_increments_on_success(self):
        cache, router = self._setup()
        router.route("greet", {"name": "A"}, handlers={"greet": lambda a: "ok"})
        router.route("greet", {"name": "B"}, handlers={"greet": lambda a: "ok"})
        assert router.n_routes == 2

    def test_n_validation_failures_initial(self):
        _, router = self._setup()
        assert router.n_validation_failures == 0

    def test_router_stats(self):
        cache, router = self._setup()
        cache.register({"name": "other", "parameters": {}})
        router.route("greet", {"name": "Z"}, handlers={"greet": lambda a: None})
        try:
            router.route("greet", {}, handlers={"greet": lambda a: None})
        except ValueError:
            pass
        s = router.stats()
        assert s.n_routes == 1
        assert s.n_registrations == 2  # greet + other
