"""tests/grammar/test_grammar_independent_mask.py

Unit tests for Phase 15F — context-independent token bitmask and compiled
grammar cache in the squish grammar subsystem.

Coverage targets
────────────────
GrammarEngine (fallback mode — xgrammar NOT installed)
  1. _precompute_independent_mask result is None when _available is False
  2. _independent_mask attribute exists on a freshly constructed GrammarEngine

GrammarCache compiled-grammar store (get_compiled / put_compiled)
  3. get_compiled returns None on a cache miss
  4. put_compiled followed by get_compiled returns the stored value
  5. put_compiled triggers LRU eviction when the store exceeds compiled_maxsize
  6. put_compiled with a duplicate key moves that entry to the MRU end
  7. put_compiled raises ValueError when schema_hash is an empty string
  8. compiled_maxsize constructor parameter is persisted on the instance
  9. _compiled_grammars OrderedDict starts empty after construction
 10. Sequential put / get round-trip over multiple distinct keys

All tests are deterministic and require no xgrammar installation.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

from squish.grammar.grammar_cache import FSMState, GrammarCache
from squish.grammar.grammar_engine import GrammarEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer stub sufficient for GrammarEngine construction."""

    def decode(self, ids: list) -> str:
        return chr(0x41 + (ids[0] % 26))  # 'A' … 'Z'

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        return [ord(c) % 100 for c in text]


def _fallback_engine() -> GrammarEngine:
    """Return a GrammarEngine in fallback mode (xgrammar absent)."""
    return GrammarEngine(_MockTokenizer())


# ---------------------------------------------------------------------------
# GrammarEngine — Phase 15F independent mask, fallback mode
# ---------------------------------------------------------------------------


class TestGrammarEngineFallback:
    """GrammarEngine behaviour when xgrammar is not installed."""

    def test_precompute_independent_mask_none_when_unavailable(self) -> None:
        """_independent_mask must be None when _available is False.

        Without xgrammar the __init__ try-block raises ImportError, so
        _precompute_independent_mask is never invoked and the attribute stays
        at its default value of None.
        """
        engine = _fallback_engine()

        assert engine._available is False
        assert engine._independent_mask is None

    def test_independent_mask_attr_exists_on_fresh_engine(self) -> None:
        """_independent_mask must be defined on every GrammarEngine instance.

        The attribute is unconditionally set to None in __init__ before the
        xgrammar import attempt, so callers can always access it without
        risking an AttributeError regardless of the library's availability.
        """
        engine = _fallback_engine()

        assert hasattr(engine, "_independent_mask")


# ---------------------------------------------------------------------------
# GrammarCache — compiled-grammar store (Phase 15D / 15F)
# ---------------------------------------------------------------------------


class TestGrammarCacheCompiledStore:
    """get_compiled / put_compiled LRU store on GrammarCache."""

    # ------------------------------------------------------------------
    # 3 — cache miss
    # ------------------------------------------------------------------

    def test_get_compiled_returns_none_on_miss(self) -> None:
        """get_compiled must return None for a key that has never been stored."""
        cache = GrammarCache()

        result = cache.get_compiled("nonexistent_hash_abc123")

        assert result is None

    # ------------------------------------------------------------------
    # 4 — basic round-trip
    # ------------------------------------------------------------------

    def test_put_then_get_returns_stored_value(self) -> None:
        """A value stored with put_compiled must be retrievable with get_compiled."""
        cache = GrammarCache()
        schema_hash = "deadbeef01234567"
        compiled = object()  # sentinel — any value is accepted

        cache.put_compiled(schema_hash, compiled)

        assert cache.get_compiled(schema_hash) is compiled

    # ------------------------------------------------------------------
    # 5 — LRU eviction
    # ------------------------------------------------------------------

    def test_lru_eviction_when_over_compiled_maxsize(self) -> None:
        """Oldest entry must be evicted when the store grows past compiled_maxsize.

        With maxsize=2 the insertion order is a → b → c.  After c is
        inserted the store holds {a, b, c} (len=3 > maxsize=2) and the
        oldest key, a, is immediately popped.
        """
        cache = GrammarCache(compiled_maxsize=2)

        cache.put_compiled("hash_a", "compiled_a")
        cache.put_compiled("hash_b", "compiled_b")
        cache.put_compiled("hash_c", "compiled_c")  # triggers eviction of hash_a

        assert cache.get_compiled("hash_a") is None, "hash_a should have been evicted"
        assert cache.get_compiled("hash_b") == "compiled_b"
        assert cache.get_compiled("hash_c") == "compiled_c"

    # ------------------------------------------------------------------
    # 6 — duplicate key moves to MRU end
    # ------------------------------------------------------------------

    def test_duplicate_key_updates_lru_order(self) -> None:
        """Re-inserting an existing key must move it to the most-recently-used end.

        Sequence with maxsize=2:
          put a → put b  (order: a, b)
          put a again   (moves a to end → order: b, a)
          put c         (evicts oldest = b, leaving a and c)
        """
        cache = GrammarCache(compiled_maxsize=2)

        cache.put_compiled("hash_a", "v1")
        cache.put_compiled("hash_b", "v2")
        cache.put_compiled("hash_a", "v1_updated")  # re-insert: moves a to MRU end
        cache.put_compiled("hash_c", "v3")           # evicts oldest = hash_b

        assert cache.get_compiled("hash_b") is None, "hash_b should have been evicted"
        assert cache.get_compiled("hash_a") == "v1_updated"
        assert cache.get_compiled("hash_c") == "v3"

    # ------------------------------------------------------------------
    # 7 — ValueError on empty schema_hash
    # ------------------------------------------------------------------

    def test_put_compiled_raises_on_empty_schema_hash(self) -> None:
        """put_compiled must raise ValueError when schema_hash is an empty string."""
        cache = GrammarCache()

        with pytest.raises(ValueError, match="schema_hash must not be empty"):
            cache.put_compiled("", "some_compiled_grammar")

    # ------------------------------------------------------------------
    # 8 — compiled_maxsize param stored correctly
    # ------------------------------------------------------------------

    def test_compiled_maxsize_param_stored_correctly(self) -> None:
        """The compiled_maxsize argument must be persisted as _compiled_maxsize.

        A value of 0 is clamped to 1 by max(1, compiled_maxsize) in __init__.
        Positive values are stored verbatim.
        """
        cache_default = GrammarCache()
        cache_custom = GrammarCache(compiled_maxsize=16)

        assert cache_default._compiled_maxsize == 64  # default documented value
        assert cache_custom._compiled_maxsize == 16

    # ------------------------------------------------------------------
    # 9 — _compiled_grammars starts empty
    # ------------------------------------------------------------------

    def test_compiled_grammars_starts_empty(self) -> None:
        """_compiled_grammars must be an empty OrderedDict after construction."""
        cache = GrammarCache()

        assert isinstance(cache._compiled_grammars, OrderedDict)
        assert len(cache._compiled_grammars) == 0

    # ------------------------------------------------------------------
    # 10 — sequential put/get round-trip over multiple items
    # ------------------------------------------------------------------

    def test_sequential_roundtrip_multiple_items(self) -> None:
        """All items inserted sequentially must be retrievable without collision.

        Uses a maxsize larger than the number of items so no eviction occurs.
        Verifies that each key maps to its own distinct compiled value and that
        there is no cross-contamination between entries.
        """
        n_items = 6
        cache = GrammarCache(compiled_maxsize=n_items + 4)

        stored: dict = {f"sha256hash{i:08x}": f"compiled_grammar_{i}" for i in range(n_items)}

        for schema_hash, compiled in stored.items():
            cache.put_compiled(schema_hash, compiled)

        for schema_hash, expected in stored.items():
            assert cache.get_compiled(schema_hash) == expected, (
                f"Round-trip failed for key {schema_hash!r}"
            )

        assert len(cache._compiled_grammars) == n_items
