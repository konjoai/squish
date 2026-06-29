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

import logging
import threading

# NOTE: mlx / mlx_lm are imported lazily inside the functions that need them — the
# slot logic (borrow/store/prefix-match) is pure Python so this module stays
# importable on non-Apple-Silicon (Linux) paths per the MLX-gating rule.

_LOG = logging.getLogger("squish.kv.prompt_prefix_cache")
_reuse_safe_by_model: "dict[int, bool]" = {}

# Prefill is processed in fixed-size, absolute-position-aligned chunks so that a
# token at position p is always computed in the chunk [floor(p/C)*C, ...) with the
# same matmul shape whether or not its prefix was cached. That makes reuse byte-
# identical to a cold prefill (bf16 rounding is shape-stable per fixed shape) — no
# fp32 needed. Correctness holds for ANY C; the value is purely a perf tradeoff
# (smaller C = less alignment-boundary waste but more, smaller forwards). Measured
# sweet spot is 64-128 on M3; below ~48 the cold prefill slows (GPU underutilised).
# See benchmarks/prefix_reuse_chunking.md for the data. Must be process-stable.
_PREFILL_CHUNK = 64


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

    def __init__(self, min_prefix: int = _PREFILL_CHUNK) -> None:
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._cache: list | None = None
        # Only chunk-prefilled *prompt* positions may be reused; the stored cache
        # also covers decode/spec tokens written off the chunk grid (see ``store``).
        self._prompt_len = 0
        # Default gates reuse on at least one full chunk so a borrow reuses whole,
        # grid-aligned chunks; an explicit smaller value falls back cleanly (keep
        # aligns to 0).
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
            # R1: never reuse past the prior request's prompt — decode/spec tokens
            # beyond it were written off the chunk grid and would not be bit-exact.
            shared = min(shared, self._prompt_len)
            if shared < self._min:
                return None, 0
            cache, self._cache = self._cache, None
            return cache, shared
        finally:
            self._lock.release()

    def store(self, ids: "list[int]", cache: list, prompt_len: "int | None" = None) -> None:
        """Publish ``cache`` (covering ``ids``) as the slot for the next request.

        ``prompt_len`` is how many leading ``ids`` were chunk-prefilled as a prompt
        (the rest are decode/spec tokens written off the chunk grid); reuse is capped
        to it in :meth:`borrow`. Defaults to ``len(ids)``.
        """
        if not self._lock.acquire(blocking=False):
            return  # another request owns the slot; drop ours rather than block
        try:
            self._ids, self._cache = list(ids), cache
            self._prompt_len = len(self._ids) if prompt_len is None else prompt_len
        finally:
            self._lock.release()


_DEFAULT = PromptPrefixCache()


def default_prefix_cache() -> PromptPrefixCache:
    """The process-wide singleton used by the default decode path."""
    return _DEFAULT


def _caches_reuse_safe(caches: list) -> bool:
    """True only if every layer cache is a plain ``KVCache`` (no windowed/hybrid
    cache whose rolling window would drop prefix tokens). Pure — unit-testable
    without a model."""
    from mlx_lm.models.cache import KVCache
    return bool(caches) and all(isinstance(c, KVCache) for c in caches)


def reuse_safe(model: "object") -> bool:
    """Whether prompt-prefix KV reuse is *correct* for ``model``.

    Reuse trims a stored cache to a shared token prefix and prefills only the
    suffix — valid only for plain :class:`KVCache`. Sliding-window / hybrid models
    use ``RotatingKVCache`` / ``ChunkedKVCache``, whose window discards old
    positions, so a "restored prefix" would be missing tokens and silently
    produce wrong output (and re-prefill would be O(prompt) anyway). Detect this
    once per model, skip reuse for those, and warn so the behaviour is observable
    rather than a silent correctness/latency surprise.
    """
    key = id(model)
    cached = _reuse_safe_by_model.get(key)
    if cached is not None:
        return cached
    from mlx_lm.models import cache as _cache
    probe = _cache.make_prompt_cache(model)
    safe = _caches_reuse_safe(probe)
    _reuse_safe_by_model[key] = safe
    if not safe:
        kinds = ", ".join(sorted({type(c).__name__ for c in probe}))
        _LOG.warning(
            "prompt-prefix KV reuse disabled: model uses %s (windowed/hybrid "
            "attention); shared-prefix reuse would be incorrect, so requests "
            "re-prefill in full. Use --no-prefix-reuse to silence.", kinds)
    return safe


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
        start = cache = None  # cold prefill — build a fresh cache below
    else:
        # Reuse the first `keep` tokens, aligned DOWN to the chunk grid so every
        # reused position is a whole chunk: the suffix then re-prefills from a
        # multiple of C, giving the same per-chunk forward shape as a cold prefill.
        raw_keep = min(shared, cache[0].offset, len(prompt_ids) - 1)
        keep = (raw_keep // _PREFILL_CHUNK) * _PREFILL_CHUNK
        if keep <= 0:
            cache, keep = None, 0  # nothing whole to reuse → clean cold prefill
        else:
            drop = cache[0].offset - keep
            if drop > 0:
                _cache.trim_prompt_cache(cache, drop)
        start = keep
    if cache is None:
        cache = _cache.make_prompt_cache(model)
        start = 0

    # Prefill the suffix in fixed-size, absolute-position-aligned chunks. `start`
    # is a multiple of C, so every slice boundary is a multiple of C — identical
    # forward shapes whether or not the prefix was reused.
    end = len(prompt_ids) - 1  # hold out the last token as the first decode input
    for s in range(start, end, _PREFILL_CHUNK):
        chunk = prompt_ids[s:min(s + _PREFILL_CHUNK, end)]
        model(mx.array(chunk, mx.uint32)[None], cache=cache)
    if end > start:
        mx.eval([c.state for c in cache])
    return cache
