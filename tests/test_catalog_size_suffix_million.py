"""Regression: dash-form resolution must work for million-param (m) sizes.

_SIZE_SUFFIX_RE only recognised the 'b' (billions) unit, so dash-form names of
million-param models (e.g. "smollm2-135m") never got converted to the canonical
"family:size" form and failed to resolve, even though the colon form resolved.
"""
from __future__ import annotations

from squish.catalog import resolve


def test_million_param_dash_form_resolves():
    r = resolve("smollm2-135m")
    assert r is not None and r.id == "smollm2:135m"


def test_million_param_dash_form_matches_colon_form():
    assert resolve("smollm2-360m") is not None
    assert resolve("smollm2-360m").id == resolve("smollm2:360m").id


def test_billion_param_dash_form_still_resolves():
    # Guard against regressing the original 'b' behaviour.
    r = resolve("qwen3-8b")
    assert r is not None and r.id == "qwen3:8b"
