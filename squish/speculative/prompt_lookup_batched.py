"""Batched prompt-lookup speculative decoding — greedy-lossless.

The original :mod:`squish.speculative.prompt_lookup` decoder verifies draft
tokens one full forward at a time, so it costs the same as plain greedy (no
speedup) — and it was never wired into the server.  This module does it
properly: it verifies a whole n-gram draft in a SINGLE batched forward and
rewinds the KV cache on partial acceptance, exactly like ``mlx_lm``'s draft-model
speculative loop — but the "draft" is a free n-gram lookup from the context, so
there is no draft model and no second cache.

Correctness contract: every emitted token is ``argmax`` of the target model's
logits given the accepted prefix, so the output is **token-for-token identical
to greedy decoding**.  Speculation only changes *how many forwards* it takes to
produce those tokens, never *which* tokens — which is the bar the previous
temp-0 speculative attempt failed (it produced non-greedy output and was
reverted).  The speedup comes on repetitive content (code, JSON, repeated
paths) where the n-gram draft is accepted.

Only valid for greedy decoding (temperature == 0 / deterministic).
"""
from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
from mlx_lm.models import cache as _cache

from squish.speculative.prompt_lookup import NGramIndex


def prompt_lookup_generate(
    model: "object",
    prompt_ids: list[int],
    max_tokens: int = 256,
    *,
    ngram_min: int = 2,
    ngram_max: int = 3,
    num_draft: int = 4,
    eos_ids: "set[int] | None" = None,
    prompt_cache: "list | None" = None,
) -> Iterator[tuple[int, int]]:
    """Yield ``(token_id, n_accepted)`` greedily, using batched n-gram lookup.

    ``n_accepted`` is how many draft tokens this step confirmed (0 = the token
    came from a plain forward) — useful for measuring acceptance.  Output is
    identical to greedy decoding of ``model`` from ``prompt_ids``.
    """
    eos_ids = eos_ids or set()
    model_cache = prompt_cache if prompt_cache is not None else _cache.make_prompt_cache(model)
    idx = NGramIndex(ngram_min=ngram_min, ngram_max=ngram_max, max_continuations=num_draft)
    ctx: list[int] = list(prompt_ids)
    idx.build(ctx)

    def _greedy(y_list: list[int]) -> list[int]:
        """One batched forward over ``y_list``; return per-position argmax."""
        y = mx.array(y_list, mx.uint32)[None]
        logits = model(y, cache=model_cache)        # (1, len(y), vocab)
        toks = mx.argmax(logits[0], axis=-1)         # (len(y),)
        # Force the cache writes to materialize before any trim — trimming a
        # lazily-built cache corrupts it (mlx_lm evals cache state every step).
        mx.eval(toks, [c.state for c in model_cache])
        return toks.tolist()

    # ── Prefill: process all but the last prompt token; hold the last as `cur`.
    if len(prompt_ids) > 1:
        _pf = mx.array(prompt_ids[:-1], mx.uint32)[None]
        model(_pf, cache=model_cache)
        mx.eval([c.state for c in model_cache])
    cur = prompt_ids[-1]                              # not yet in the cache
    produced = 0

    while produced < max_tokens:
        # Draft = longest n-gram continuation of the current context suffix.
        cands = idx.find(ctx)
        draft: list[int] = next((c for c in cands if c), [])[:num_draft]
        draft = draft[: max_tokens - produced - 1] if draft else draft

        # One forward verifies `cur` + all draft tokens at once.
        toks = _greedy([cur, *draft])                # len = len(draft)+1 predictions

        # Accept the longest prefix where the model agrees with the draft.
        n = 0
        while n < len(draft) and toks[n] == draft[n]:
            n += 1
        accepted = toks[: n + 1]                      # n matched + 1 correction/bonus

        # The forward grew the cache by len(draft)+1; we keep cur + n drafts
        # (the bonus token toks[n] is fed next), so trim the surplus.
        _cache.trim_prompt_cache(model_cache, len(draft) - n)

        for t in accepted:
            ctx.append(t)
            idx.push(t)
            yield t, n
            produced += 1
            cur = t
            if t in eos_ids or produced >= max_tokens:
                return
