"""Unit + integration tests for the in-memory prompt-prefix KV reuse slot.

The pure borrow/store/prefix-match logic is covered without a model; the
byte-identity of reuse vs cold prefill is covered by a model-gated test.
"""
import os

import pytest

from squish.kv.prompt_prefix_cache import PromptPrefixCache, _common_prefix_len

_MODEL = os.environ.get("PL_TEST_MODEL", "") or os.path.expanduser(
    "~/models/Qwen2.5-1.5B-Instruct-int4")
_needs_model = pytest.mark.skipif(
    not os.path.isdir(_MODEL), reason=f"model not present: {_MODEL}")


def test_common_prefix_len():
    assert _common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3   # one is a prefix
    assert _common_prefix_len([5, 6], [7, 8]) == 0
    assert _common_prefix_len([], [1]) == 0


def test_borrow_requires_min_prefix():
    pc = PromptPrefixCache(min_prefix=4)
    pc.store([1, 2, 3, 4, 5], object())
    # only 3 shared tokens (< min_prefix=4) → no reuse
    assert pc.borrow([1, 2, 3, 9, 9]) == (None, 0)


def test_borrow_returns_cache_and_shared_len_then_empties_slot():
    pc = PromptPrefixCache(min_prefix=2)
    sentinel = object()
    pc.store([1, 2, 3, 4], sentinel)
    cache, shared = pc.borrow([1, 2, 3, 9])
    assert cache is sentinel and shared == 3
    # slot was handed off — a second borrow finds nothing
    assert pc.borrow([1, 2, 3, 9]) == (None, 0)


def test_borrow_empty_slot_is_none():
    assert PromptPrefixCache().borrow([1, 2, 3]) == (None, 0)


def test_store_then_borrow_roundtrip():
    pc = PromptPrefixCache(min_prefix=1)
    a = object()
    pc.store([10, 11], a)
    assert pc.borrow([10, 11, 12]) == (a, 2)


def test_reuse_safe_only_for_plain_kvcache():
    """Reuse must be allowed for plain KVCache and disabled for windowed/hybrid
    caches (RotatingKVCache) whose rolling window would drop prefix tokens."""
    pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from squish.kv.prompt_prefix_cache import _caches_reuse_safe
    assert _caches_reuse_safe([KVCache(), KVCache()]) is True
    assert _caches_reuse_safe([KVCache(), RotatingKVCache(max_size=8, keep=2)]) is False
    assert _caches_reuse_safe([]) is False


@_needs_model
def test_reuse_is_byte_identical_to_cold():
    """A prompt that extends a recent one reuses its prefix and yields the exact
    same tokens as a cold (full-prefill) run — reuse changes only how much
    prefill runs, never the logits."""
    import mlx.core as mx
    mx.set_default_device(mx.gpu)  # warm/cold share forward math → identical here
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

    model, tok = load(_MODEL)
    eos = {tok.eos_token_id}

    def gen(ids, n, reuse):
        return [t for t, _ in prompt_lookup_generate(
            model, ids, n, eos_ids=eos, reuse_prefix=reuse)]

    # >128 tokens so the prefix-reuse threshold engages on turn 2.
    p1 = tok.encode("Background: " + "the river flows past the old stone bridge. " * 14)
    g1 = gen(p1, 16, reuse=True)                       # turn 1 populates the slot
    p2 = p1 + g1 + tok.encode(" Then answer briefly:")  # turn 2 extends turn 1

    warm = gen(p2, 16, reuse=True)                     # reuses the shared prefix
    cold = gen(p2, 16, reuse=False)                    # full cold prefill (truth)
    assert warm == cold
