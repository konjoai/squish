"""tests/test_api_validation_unit.py

Branch-complete unit tests for ``squish.api.validation``.

These are pure-Python and import no MLX — they run on Linux CI and therefore
protect the konjo coverage gate for the new validation source (the e2e suites
that also exercise this code are skipped off Apple Silicon).
"""
from __future__ import annotations

import asyncio
import math

import pytest
from fastapi import HTTPException

from squish.api import validation as v


# ── parse_max_tokens ────────────────────────────────────────────────────────
class TestParseMaxTokens:
    def test_none_returns_default(self):
        assert v.parse_max_tokens(None, 4096) == 4096

    def test_valid_int(self):
        assert v.parse_max_tokens(128, 4096) == 128

    def test_zero_allowed(self):
        assert v.parse_max_tokens(0, 4096) == 0

    def test_numeric_string_coerced(self):
        assert v.parse_max_tokens("256", 4096) == 256

    def test_float_truncated(self):
        assert v.parse_max_tokens(12.9, 4096) == 12

    def test_negative_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_max_tokens(-5, 4096)
        assert ei.value.status_code == 400
        assert "max_tokens" in ei.value.detail

    def test_non_numeric_string_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_max_tokens("abc", 4096)
        assert ei.value.status_code == 400

    def test_astronomically_large_rejected(self):
        # int() parses this fine, but it would trigger a runaway generation.
        with pytest.raises(HTTPException) as ei:
            v.parse_max_tokens("99999999999999999999999999", 4096)
        assert ei.value.status_code == 400
        assert "max_tokens" in ei.value.detail

    def test_ceiling_is_inclusive(self):
        assert v.parse_max_tokens(v._MAX_TOKENS_CEILING, 4096) == v._MAX_TOKENS_CEILING

    def test_just_above_ceiling_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_max_tokens(v._MAX_TOKENS_CEILING + 1, 4096)

    def test_bool_rejected(self):
        # bool is an int subclass — must not be silently accepted.
        with pytest.raises(HTTPException):
            v.parse_max_tokens(True, 4096)

    def test_list_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_max_tokens([1, 2], 4096)


# ── parse_max_steps ─────────────────────────────────────────────────────────
class TestParseMaxSteps:
    def test_none_returns_default(self):
        assert v.parse_max_steps(None, 10) == 10

    def test_valid(self):
        assert v.parse_max_steps(3, 10) == 3

    def test_zero_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_max_steps(0, 10)
        assert ei.value.status_code == 400

    def test_negative_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_max_steps(-1, 10)

    def test_garbage_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_max_steps("xyz", 10)

    def test_above_ceiling_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_max_steps(v._MAX_STEPS_CEILING + 1, 10)
        assert ei.value.status_code == 400

    def test_ceiling_inclusive(self):
        assert v.parse_max_steps(v._MAX_STEPS_CEILING, 10) == v._MAX_STEPS_CEILING

    def test_bool_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_max_steps(False, 10)


# ── parse_temperature ───────────────────────────────────────────────────────
class TestParseTemperature:
    def test_none_returns_default(self):
        assert v.parse_temperature(None) == pytest.approx(0.7)

    def test_valid(self):
        assert v.parse_temperature(1.5) == pytest.approx(1.5)

    def test_bounds_inclusive(self):
        assert v.parse_temperature(0.0) == 0.0
        assert v.parse_temperature(2.0) == 2.0

    def test_numeric_string(self):
        assert v.parse_temperature("0.3") == pytest.approx(0.3)

    def test_too_high_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_temperature(1e9)
        assert ei.value.status_code == 400
        assert "temperature" in ei.value.detail

    def test_negative_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_temperature(-0.1)

    def test_nan_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_temperature(float("nan"))
        assert "finite" in ei.value.detail

    def test_inf_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_temperature(math.inf)

    def test_garbage_string_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_temperature("hot")

    def test_bool_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_temperature(True)


# ── parse_top_p ─────────────────────────────────────────────────────────────
class TestParseTopP:
    def test_none_returns_default(self):
        assert v.parse_top_p(None) == pytest.approx(0.9)

    def test_valid(self):
        assert v.parse_top_p(0.5) == pytest.approx(0.5)

    def test_upper_bound_inclusive(self):
        assert v.parse_top_p(1.0) == 1.0

    def test_zero_rejected(self):
        # lower bound is exclusive
        with pytest.raises(HTTPException):
            v.parse_top_p(0.0)

    def test_above_one_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_top_p(5)
        assert ei.value.status_code == 400

    def test_negative_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_top_p(-1)

    def test_nan_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_top_p(float("nan"))


# ── parse_embedding_input ───────────────────────────────────────────────────
class TestParseEmbeddingInput:
    def test_single_string(self):
        assert v.parse_embedding_input("hello") == ["hello"]

    def test_list_of_strings(self):
        assert v.parse_embedding_input(["a", "b"]) == ["a", "b"]

    def test_empty_string_rejected(self):
        with pytest.raises(HTTPException) as ei:
            v.parse_embedding_input("")
        assert ei.value.status_code == 400
        assert "input" in ei.value.detail

    def test_whitespace_only_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input("   ")

    def test_none_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input(None)

    def test_empty_list_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input([])

    def test_list_with_non_string_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input(["ok", 5])

    def test_list_with_blank_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input(["ok", "  "])

    def test_int_rejected(self):
        with pytest.raises(HTTPException):
            v.parse_embedding_input(42)


# ── parse_json_body ─────────────────────────────────────────────────────────
class _FakeRequest:
    """Minimal stand-in exposing the awaitable ``json()`` used by the parser."""

    def __init__(self, payload=None, raise_exc: Exception | None = None):
        self._payload = payload
        self._raise = raise_exc

    async def json(self):
        if self._raise is not None:
            raise self._raise
        return self._payload


class TestParseJsonBody:
    def test_valid_object(self):
        req = _FakeRequest(payload={"a": 1})
        body = asyncio.run(v.parse_json_body(req))
        assert body == {"a": 1}

    def test_malformed_json_raises_400(self):
        import json

        req = _FakeRequest(raise_exc=json.JSONDecodeError("boom", "doc", 0))
        with pytest.raises(HTTPException) as ei:
            asyncio.run(v.parse_json_body(req))
        assert ei.value.status_code == 400

    def test_value_error_raises_400(self):
        req = _FakeRequest(raise_exc=ValueError("bad"))
        with pytest.raises(HTTPException) as ei:
            asyncio.run(v.parse_json_body(req))
        assert ei.value.status_code == 400

    def test_non_object_body_raises_400(self):
        req = _FakeRequest(payload=[1, 2, 3])
        with pytest.raises(HTTPException) as ei:
            asyncio.run(v.parse_json_body(req))
        assert ei.value.status_code == 400
        assert "object" in ei.value.detail


# ── _bad helper ─────────────────────────────────────────────────────────────
def test_bad_builds_400():
    exc = v._bad("foo", "must be nice")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail == "'foo' must be nice"
