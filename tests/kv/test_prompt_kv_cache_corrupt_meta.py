"""Regression: incomplete-but-valid meta.json must be a clean miss, not a crash.

PromptKVStore.get() read n_layers/offset by subscript while every other
corruption path returned None. A valid-JSON meta missing those keys raised an
uncaught KeyError, crashing the lookup (and the request).
"""
from __future__ import annotations

import json

from squish.kv.prompt_kv_cache import _CACHE_VERSION, PromptKVStore


def test_meta_missing_n_layers_is_clean_miss(tmp_path):
    store = PromptKVStore(cache_dir=tmp_path)
    prompt = "hello world"
    h = store.hash_prompt(prompt)
    entry = store._entry_dir(h)
    entry.mkdir(parents=True, exist_ok=True)
    # Valid JSON, correct version, but missing the required n_layers/offset keys.
    (entry / "meta.json").write_text(json.dumps({"version": _CACHE_VERSION, "offset": 5}))

    # Must return None (clean miss) rather than raising KeyError.
    assert store.get(prompt) is None
    # And the corrupt entry should have been evicted.
    assert not (entry / "meta.json").exists()


def test_meta_missing_offset_is_clean_miss(tmp_path):
    store = PromptKVStore(cache_dir=tmp_path)
    prompt = "another prompt"
    h = store.hash_prompt(prompt)
    entry = store._entry_dir(h)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "meta.json").write_text(json.dumps({"version": _CACHE_VERSION, "n_layers": 4}))

    assert store.get(prompt) is None
