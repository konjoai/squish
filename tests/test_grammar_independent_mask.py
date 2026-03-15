"""tests/test_grammar_independent_mask.py

Unit tests for GrammarCache compiled-grammar store and GrammarEngine
context-independent token bitmask (Phase 15D / 15F).

Coverage map
────────────
GrammarCache (squish.grammar.grammar_cache)
  1.  get_compiled returns None on a cache miss
  2.  put_compiled stores the object in the cache
  3.  get_compiled retrieves exactly the stored object
  4.  put_compiled raises ValueError for an empty schema_hash
  5.  LRU eviction: compiled_maxsize=2 evicts the oldest entry on the 3rd insert
  6.  put_compiled return value is None (implicit, no explicit return)

GrammarEngine (squish.grammar.grammar_engine)
  7.  _precompute_independent_mask does not run when xgrammar is unavailable
  8.  _independent_mask is None before xgrammar initialises / when unavailable

Stability
  9.  Same schema_hash always returns the identical stored object on repeated
      get_compiled calls.

All tests are self-contained.  xgrammar is not required; the no-xgrammar path
is exercised by blocking the import via sys.modules patching so the suite
passes whether or not xgrammar is installed in the current environment.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from squish.grammar.grammar_cache import GrammarCache
from squish.grammar.grammar_engine import GrammarEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub accepted by GrammarEngine.__init__."""

    def decode(self, ids: list) -> str:  # noqa: D401
        return chr(0x41 + (ids[0] % 26))

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        return [ord(c) % 100 for c in text]


def _engine_without_xgrammar() -> GrammarEngine:
    """Return a GrammarEngine constructed with xgrammar forcibly blocked.

    Setting ``sys.modules["xgrammar"] = None`` causes Python to raise
    ``ImportError`` the moment ``import xgrammar`` is attempted inside
    ``__init__``, regardless of whether the package is actually installed.
    """
    with patch.dict(sys.modules, {"xgrammar": None}):
        engine = GrammarEngine(_StubTokenizer())
    return engine


# ---------------------------------------------------------------------------
# Test 1 — get_compiled returns None on a cache miss
# ---------------------------------------------------------------------------


class TestGetCompiledMiss:
    """get_compiled must return None for any key that has not been stored."""

    def test_fresh_cache_misses_on_any_key(self) -> None:
        cache = GrammarCache()
        assert cache.get_compiled("deadbeef01234567") is None

    def test_different_key_misses_when_only_other_key_stored(self) -> None:
        cache = GrammarCache()
        cache.put_compiled("hash_stored", object())
        assert cache.get_compiled("hash_not_stored") is None


# ---------------------------------------------------------------------------
# Tests 2 & 3 — put_compiled stores; get_compiled retrieves the same object
# ---------------------------------------------------------------------------


class TestPutAndGetCompiled:
    """put_compiled stores an object and get_compiled retrieves the identical object."""

    def test_put_then_get_returns_stored_object(self) -> None:
        cache = GrammarCache()
        sentinel = object()
        cache.put_compiled("cafebabe12345678", sentinel)
        assert cache.get_compiled("cafebabe12345678") is sentinel

    def test_identity_not_just_equality(self) -> None:
        """get_compiled must return the exact same object reference, not a copy."""
        cache = GrammarCache()
        compiled = {"type": "object", "properties": {}}  # mutable dict sentinel
        cache.put_compiled("identity_hash_001", compiled)
        assert cache.get_compiled("identity_hash_001") is compiled

    def test_multiple_distinct_keys_stored_independently(self) -> None:
        cache = GrammarCache(compiled_maxsize=8)
        objects = {f"key_{i:04d}": object() for i in range(5)}
        for k, v in objects.items():
            cache.put_compiled(k, v)
        for k, v in objects.items():
            assert cache.get_compiled(k) is v


# ---------------------------------------------------------------------------
# Test 4 — put_compiled raises ValueError for an empty schema_hash
# ---------------------------------------------------------------------------


class TestPutCompiledEmptyHashRaises:
    """put_compiled must raise ValueError when schema_hash is an empty string."""

    def test_empty_string_raises_value_error(self) -> None:
        cache = GrammarCache()
        with pytest.raises(ValueError, match="schema_hash must not be empty"):
            cache.put_compiled("", object())

    def test_single_character_key_does_not_raise(self) -> None:
        """Any non-empty key is valid; a single char must succeed."""
        cache = GrammarCache()
        cache.put_compiled("x", object())  # must not raise


# ---------------------------------------------------------------------------
# Test 5 — LRU eviction with compiled_maxsize=2
# ---------------------------------------------------------------------------


class TestLRUEviction:
    """Oldest entry is evicted when the store grows beyond compiled_maxsize."""

    def test_third_insert_evicts_oldest_with_maxsize_two(self) -> None:
        cache = GrammarCache(compiled_maxsize=2)
        obj_a, obj_b, obj_c = object(), object(), object()

        cache.put_compiled("hash_a", obj_a)
        cache.put_compiled("hash_b", obj_b)
        # Insertion order: a (oldest) … b … c (newest).
        # After inserting c the store has 3 entries which exceeds maxsize=2;
        # the oldest entry (hash_a) must be popped.
        cache.put_compiled("hash_c", obj_c)

        assert cache.get_compiled("hash_a") is None, "hash_a (oldest) should be evicted"
        assert cache.get_compiled("hash_b") is obj_b
        assert cache.get_compiled("hash_c") is obj_c

    def test_store_never_exceeds_maxsize_under_sustained_insertions(self) -> None:
        maxsize = 3
        cache = GrammarCache(compiled_maxsize=maxsize)
        for i in range(12):
            cache.put_compiled(f"h{i:04d}", object())
        assert len(cache._compiled_grammars) <= maxsize

    def test_re_inserting_key_promotes_it_to_mru_preventing_its_eviction(self) -> None:
        """Re-inserting an existing key must move it to the most-recently-used end.

        Sequence with maxsize=2:
          put a, put b   → order: a (oldest), b
          put a again    → moves a to MRU end → order: b (oldest), a
          put c          → evicts oldest = b, leaving a and c
        """
        cache = GrammarCache(compiled_maxsize=2)
        cache.put_compiled("hash_a", "v1")
        cache.put_compiled("hash_b", "v2")
        cache.put_compiled("hash_a", "v1_refreshed")   # promote a to MRU
        cache.put_compiled("hash_c", "v3")             # evicts b (now oldest)

        assert cache.get_compiled("hash_b") is None, "hash_b should be evicted"
        assert cache.get_compiled("hash_a") == "v1_refreshed"
        assert cache.get_compiled("hash_c") == "v3"


# ---------------------------------------------------------------------------
# Test 6 — put_compiled return value is None
# ---------------------------------------------------------------------------


class TestPutCompiledReturnValue:
    """put_compiled has no explicit return; the implicit return value is None."""

    def test_put_compiled_returns_none_on_new_key(self) -> None:
        cache = GrammarCache()
        result = cache.put_compiled("return_test_hash_a", object())
        assert result is None

    def test_put_compiled_returns_none_on_duplicate_key(self) -> None:
        cache = GrammarCache()
        cache.put_compiled("return_test_hash_b", object())
        result = cache.put_compiled("return_test_hash_b", object())  # duplicate
        assert result is None


# ---------------------------------------------------------------------------
# Tests 7 & 8 — GrammarEngine._independent_mask when xgrammar is unavailable
# ---------------------------------------------------------------------------


class TestGrammarEngineNoXgrammar:
    """GrammarEngine falls back gracefully when xgrammar cannot be imported."""

    def test_independent_mask_is_none_when_xgrammar_import_fails(self) -> None:
        """Test 8: _independent_mask stays None when the xgrammar import raises.

        In __init__ the attribute is assigned None unconditionally; it is only
        overwritten by _precompute_independent_mask(), which is only called when
        _available is True.  Blocking xgrammar keeps _available False, so the
        attribute remains None.
        """
        engine = _engine_without_xgrammar()
        assert engine._independent_mask is None

    def test_available_is_false_when_xgrammar_import_fails(self) -> None:
        """_available must be False when the try-block in __init__ raises."""
        engine = _engine_without_xgrammar()
        assert engine._available is False

    def test_independent_mask_attribute_always_exists(self) -> None:
        """_independent_mask must be defined on every GrammarEngine instance.

        The attribute is set to None unconditionally before the import attempt,
        so it is always accessible without raising AttributeError.
        """
        engine = _engine_without_xgrammar()
        assert hasattr(engine, "_independent_mask")

    def test_precompute_independent_mask_not_called_when_xgrammar_unavailable(
        self,
    ) -> None:
        """Test 7: _precompute_independent_mask must not be invoked when xgrammar
        is unavailable.

        __init__ guards the call with ``if self._available:``.  When the import
        fails _available is False, so the precompute method must never execute.
        We verify this with a spy that records every invocation.
        """
        call_log: list[str] = []
        original = GrammarEngine._precompute_independent_mask

        def spy(self_arg: GrammarEngine) -> None:  # type: ignore[override]
            call_log.append("called")
            original(self_arg)

        with patch.dict(sys.modules, {"xgrammar": None}):
            with patch.object(GrammarEngine, "_precompute_independent_mask", spy):
                GrammarEngine(_StubTokenizer())

        assert call_log == [], (
            "_precompute_independent_mask must not be called when xgrammar "
            f"is absent; call_log={call_log}"
        )


# ---------------------------------------------------------------------------
# Test 9 — Stability: same schema_hash always returns the same stored object
# ---------------------------------------------------------------------------


class TestMaskStability:
    """Repeated get_compiled calls for the same key must always return the
    identical stored object (i.e., no spontaneous mutation or eviction)."""

    def test_same_key_returns_same_object_across_many_repeated_calls(self) -> None:
        cache = GrammarCache(compiled_maxsize=64)
        sentinel = object()
        cache.put_compiled("stable_schema_hash_abc123", sentinel)

        for iteration in range(50):
            result = cache.get_compiled("stable_schema_hash_abc123")
            assert result is sentinel, (
                f"get_compiled returned a different object on iteration {iteration}"
            )

    def test_stability_unaffected_by_other_key_insertions(self) -> None:
        """Inserting other keys (below the eviction threshold) must not disturb
        an already-stored entry."""
        cache = GrammarCache(compiled_maxsize=16)
        anchor = object()
        cache.put_compiled("anchor_hash", anchor)

        # Insert several other entries (well below maxsize=16)
        for i in range(8):
            cache.put_compiled(f"other_{i:04d}", object())

        assert cache.get_compiled("anchor_hash") is anchor

    def test_two_different_hashes_remain_independent_across_repeated_calls(
        self,
    ) -> None:
        cache = GrammarCache(compiled_maxsize=8)
        obj1, obj2 = object(), object()
        cache.put_compiled("schema_aaa", obj1)
        cache.put_compiled("schema_bbb", obj2)

        for _ in range(20):
            assert cache.get_compiled("schema_aaa") is obj1
            assert cache.get_compiled("schema_bbb") is obj2
