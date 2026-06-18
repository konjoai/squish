"""Regression: _dir_to_canonical must strip decorators only as suffixes.

Using str.replace deleted tokens like "-instruct" anywhere in the name, so a
directory with an interior decorator segment produced a wrong canonical id.
"""
from __future__ import annotations

from squish.serving.local_model_scanner import _dir_to_canonical


def test_interior_decorator_token_is_not_stripped():
    # "-instruct" appears mid-name and must be preserved; only true suffixes go.
    assert _dir_to_canonical("llama-3-instruct-merge-8b") == "llama-3-instruct-merge:8b"


def test_trailing_decorators_are_stripped():
    assert _dir_to_canonical("Qwen3-8B-bf16") == "qwen3:8b"


def test_chained_trailing_decorators_all_stripped():
    # "-instruct-bf16" is a chain of two trailing decorators.
    assert _dir_to_canonical("Llama-3.1-8B-Instruct-bf16") == "llama-3.1:8b"


def test_quant_suffix_stripped():
    assert _dir_to_canonical("Mistral-7B-int4") == "mistral:7b"
