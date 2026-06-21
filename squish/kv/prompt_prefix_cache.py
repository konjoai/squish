"""In-memory prompt-prefix KV reuse for the default (prompt-lookup) decode path.

By default squish re-prefills every request from scratch — for multi-turn chat,
agent loops, and RAG (where each request *extends* a recent prompt) that repays
the full O(prompt) prefill every turn (≈1.2 s at 1 K tokens, ≈5.7 s at 4 K on an
M3). This module keeps the most recent request's KV cache in memory and, when a
new prompt shares a long token prefix with it, restores that prefix and prefills
only the new suffix — turning TTFT from O(prompt) into O(new tokens).

Correctness: KV state is positional and causal, so the cache for ``ids[:L]`` is
byte-identical whether it was built in one prefill or reused from a prior request
that shared those first ``L`` tokens. Reuse therefore changes only *how much*
prefill runs, never the logits — output is identical to a cold prefill.

Concurrency: a single slot guarded by short, non-blocking lock holds around the
borrow and the store (never across generation). Concurrent requests that can't
borrow simply run a normal cold prefill — never a corrupted cache. squish is a
single-user local server, so in practice requests are sequential and every turn
reuses the previous one.
"""
from __future__ import annotations

import threading

# NOTE: mlx / mlx_lm are imported lazily inside `prefill_with_reuse` only — the
# slot logic (borrow/store/prefix-match) is pure Python so this module stays
# importable on non-Apple-Silicon (Linux) paths per the MLX-gating rule.


def _common_prefix_len(a: "list[int]", b: "list[int]") -> int:
    """Length of the longest shared leading token run of ``a`` and ``b``."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class PromptPrefixCache:
    """One-slot in-memory KV cache keyed by the token sequence it covers.

    ``min_prefix`` gates reuse: below it a cold prefill is cheap enough that the
    trim+suffix bookkeeping isn't worth it.
    """

    def __init__(self, min_prefix: int = 128) -> None:
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._cache: list | None = None
        self._min = min_prefix

    def borrow(self, prompt_ids: "list[int]") -> "tuple[list | None, int]":
        """Hand the stored cache to the caller if it shares a long prefix.

        Returns ``(cache, reuse_len)``; ``cache`` is removed from the slot so the
        caller owns it exclusively until :meth:`store`. Returns ``(None, 0)`` when
        there is nothing worth reusing (or another request holds the slot).
        """
        if not self._lock.acquire(blocking=False):
            return None, 0
        try:
            if self._cache is None:
                return None, 0
            shared = _common_prefix_len(prompt_ids, self._ids)
            if shared < self._min:
                return None, 0
            cache, self._cache = self._cache, None
            return cache, shared
        finally:
            self._lock.release()

    def store(self, ids: "list[int]", cache: list) -> None:
        """Publish ``cache`` (covering ``ids``) as the slot for the next request."""
        if not self._lock.acquire(blocking=False):
            return  # another request owns the slot; drop ours rather than block
        try:
            self._ids, self._cache = list(ids), cache
        finally:
            self._lock.release()


_DEFAULT = PromptPrefixCache()


def default_prefix_cache() -> PromptPrefixCache:
    """The process-wide singleton used by the default decode path."""
    return _DEFAULT


def prefill_with_reuse(
    model: "object", prompt_ids: "list[int]", prefix_cache: "PromptPrefixCache | None"
) -> list:
    """Return a KV cache covering ``prompt_ids[:-1]`` (last token held out as the
    first decode input), reusing a shared prefix from ``prefix_cache`` when one is
    available and prefilling only the remaining suffix.

    With ``prefix_cache`` None this is a plain cold prefill.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as _cache

    cache, shared = (prefix_cache.borrow(prompt_ids) if prefix_cache else (None, 0))
    if cache is None:
        cache = _cache.make_prompt_cache(model)
        start = 0
    else:
        # Reuse the first `keep` tokens; cap below the held-out last token and the
        # cache's true coverage so the trim count is never negative.
        keep = min(shared, cache[0].offset, len(prompt_ids) - 1)
        drop = cache[0].offset - keep
        if drop > 0:
            _cache.trim_prompt_cache(cache, drop)
        start = keep

    suffix = prompt_ids[start:-1]
    if suffix:
        model(mx.array(suffix, mx.uint32)[None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache
