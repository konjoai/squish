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

from collections.abc import Iterable, Iterator

import mlx.core as mx
from mlx_lm.models import cache as _cache

from squish.kv.prompt_prefix_cache import default_prefix_cache, prefill_with_reuse, reuse_safe
from squish.speculative.prompt_lookup import NGramIndex


def _advance_guard(
    cooldown: int,
    miss_streak: int,
    drafted: bool,
    n_accepted: int,
    miss_limit: int,
    cooldown_steps: int,
) -> "tuple[int, int]":
    """Return the next ``(cooldown, miss_streak)`` for the adaptive accept guard.

    Pure decision logic, separated from the model loop so it is unit-testable
    without a model.  ``drafted`` is whether a non-empty draft was verified this
    step; ``n_accepted`` is how many of its tokens the model confirmed (0 == a
    wasted draft that cost a wider forward + cache trim for nothing).

    Rules:
    * cooling down → tick the counter down, leave the streak untouched;
    * not cooling down and no draft (free single-token forward) → no change;
    * a wasted draft → grow the streak; once it reaches ``miss_limit`` enter a
      ``cooldown_steps`` cooldown and reset the streak;
    * a draft that accepted anything → reset the streak.

    ``cooldown_steps <= 0`` disables cooldown entirely (the guard never trips).
    """
    if cooldown > 0:
        return cooldown - 1, miss_streak
    if not drafted:
        return 0, miss_streak
    if n_accepted == 0:
        miss_streak += 1
        if cooldown_steps > 0 and miss_streak >= miss_limit:
            return cooldown_steps, 0
        return 0, miss_streak
    return 0, 0


def _emit(
    tokens: "Iterable[int]",
    ctx: list,
    idx: "object",
    eos_ids: set,
    max_tokens: int,
    produced: int,
) -> "Iterator[tuple[int, int, bool]]":
    """Yield ``(token, produced, stop)`` for each token, updating ``ctx`` + the
    n-gram ``idx``.  ``stop`` is True once EOS or ``max_tokens`` is reached; the
    caller terminates on it.  Shared by the drafted-accept and cooldown paths so
    the emit/terminate logic lives in one place."""
    for t in tokens:
        ctx.append(t)
        idx.push(t)
        produced += 1
        stop = t in eos_ids or produced >= max_tokens
        yield t, produced, stop
        if stop:
            return


def _lookup_draft(idx: "object", ctx: list, num_draft: int, budget: int) -> list[int]:
    """Longest in-context n-gram continuation of ``ctx``, capped to ``num_draft``
    tokens and the ``budget`` still owed (a slice of [] is [])."""
    cands = idx.find(ctx)
    return next((c for c in cands if c), [])[:num_draft][:budget]


def _accept(toks: list[int], draft: list[int]) -> "tuple[list[int], int]":
    """Return ``(accepted, n)`` — the longest draft prefix the model agreed with
    (``n`` tokens) plus the one bonus/correction token at ``toks[n]``."""
    n = 0
    while n < len(draft) and toks[n] == draft[n]:
        n += 1
    return toks[: n + 1], n


def _setup_cache(
    model: "object", prompt_ids: list[int], prompt_cache: "list | None", reuse_prefix: bool
) -> "tuple[list, object | None]":
    """Return ``(model_cache, store_to)`` — a cache prefilled over
    ``prompt_ids[:-1]`` plus the prefix slot to publish to on completion (None
    when the caller supplied a cache or reuse is off).  When we own the cache we
    reuse a recent request's shared prefix so only the new suffix is prefilled."""
    if prompt_cache is not None:
        if len(prompt_ids) > 1:
            model(mx.array(prompt_ids[:-1], mx.uint32)[None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
        return prompt_cache, None
    # Reuse only for plain-KVCache models — windowed/hybrid caches make it
    # incorrect (see `reuse_safe`); store_to=None falls back to a cold prefill.
    store_to = default_prefix_cache() if (reuse_prefix and reuse_safe(model)) else None
    return prefill_with_reuse(model, prompt_ids, store_to), store_to


def _pipelined_greedy_block(
    model: "object", model_cache: "list", start: int, n_steps: int
) -> "Iterator[int]":
    """Yield up to ``n_steps`` pure-greedy token ids from ``start``, pipelined.

    Each step's argmax is kept lazy on the GPU and dispatched with
    ``mx.async_eval`` so the forward of token *i+1* overlaps the host-side
    ``.item()`` read of token *i* (the mlx-lm ``generate_step`` pattern). squish's
    synchronous loop instead materialises the full logits and blocks on every
    token; on an M3 this pipelining alone is ~1.38x on plain greedy.

    Safe to chain because NO cache trim happens inside the block (unlike a
    drafted verify step), so the lazily-grown cache is never rewound mid-flight;
    the cache state is materialised before the generator returns.  Output is
    token-for-token identical to single-token greedy.
    """
    if n_steps <= 0:
        return

    def _argmax_after(x: "object") -> "object":
        return mx.argmax(model(x, cache=model_cache)[0, -1])

    y = _argmax_after(mx.array([[start]], mx.uint32))
    mx.async_eval(y)
    for i in range(n_steps):
        nxt = _argmax_after(y[None][None]) if i + 1 < n_steps else None
        if nxt is not None:
            mx.async_eval(nxt)        # queue forward(i+1) before reading token i
        yield int(y.item())           # forces token i; overlaps with forward(i+1)
        if nxt is None:
            break
        y = nxt
    mx.eval([c.state for c in model_cache])  # settle cache before drafting resumes


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
    miss_limit: int = 3,
    cooldown_steps: int = 16,
    pipeline_cooldown: bool = True,
    reuse_prefix: bool = True,
) -> Iterator[tuple[int, int]]:
    """Yield ``(token_id, n_accepted)`` greedily, using batched n-gram lookup.

    ``n_accepted`` is how many draft tokens this step confirmed (0 = the token
    came from a plain forward) — useful for measuring acceptance.  Output is
    identical to greedy decoding of ``model`` from ``prompt_ids``.

    Adaptive guard: a verified-but-rejected draft is *not* free — it costs a
    wider ``(1, 1+k)`` forward plus a KV-cache trim.  On low-reuse output (open
    chat) common-bigram n-grams keep matching but mispredict, so unconditional
    speculation regresses ~5% vs plain greedy.  To stay non-regressing we count
    consecutive wasted drafts; after ``miss_limit`` of them we stop drafting for
    ``cooldown_steps`` (plain single-token forwards, == greedy) and then
    re-probe.  This only changes *how many* forwards run, never *which* tokens
    are emitted — output remains token-for-token greedy.  Set ``cooldown_steps``
    to 0 to disable the guard (always draft).
    """
    eos_ids = eos_ids or set()
    idx = NGramIndex(ngram_min=ngram_min, ngram_max=ngram_max, max_continuations=num_draft)
    ctx: list[int] = list(prompt_ids)
    idx.build(ctx)

    # ── Prefill (process all but the last prompt token; hold the last as `cur`).
    # When we own the cache, reuse a shared prefix from a recent request so a
    # multi-turn / agent / RAG prompt prefills only its new suffix (see
    # `prompt_prefix_cache`); a caller-supplied cache opts out of reuse.
    model_cache, _store_to = _setup_cache(model, prompt_ids, prompt_cache, reuse_prefix)

    def _greedy(y_list: list[int]) -> list[int]:
        """One batched forward over ``y_list``; return per-position argmax."""
        y = mx.array(y_list, mx.uint32)[None]
        logits = model(y, cache=model_cache)        # (1, len(y), vocab)
        toks = mx.argmax(logits[0], axis=-1)         # (len(y),)
        # Force the cache writes to materialize before any trim — trimming a
        # lazily-built cache corrupts it (mlx_lm evals cache state every step).
        mx.eval(toks, [c.state for c in model_cache])
        return toks.tolist()

    cur = prompt_ids[-1]                              # not yet in the cache
    produced = 0
    miss_streak = 0                                   # consecutive wasted drafts
    cooldown = 0                                      # steps left running plain greedy

    try:
      while produced < max_tokens:
        # Draft = longest n-gram continuation of the current context suffix —
        # skipped while cooling down so a low-acceptance workload pays only a
        # single-token forward (== plain greedy), never a wasted wide forward.
        draft = [] if cooldown else _lookup_draft(idx, ctx, num_draft, max_tokens - produced - 1)

        # One forward verifies `cur` + all draft tokens at once.
        toks = _greedy([cur, *draft])                # len = len(draft)+1 predictions
        accepted, n = _accept(toks, draft)

        # The forward grew the cache by len(draft)+1; we keep cur + n drafts
        # (the bonus token toks[n] is fed next), so trim the surplus.
        _cache.trim_prompt_cache(model_cache, len(draft) - n)

        # Adaptive guard bookkeeping: a draft that accepted nothing (n == 0) was
        # pure overhead; after `miss_limit` such in a row, cool down (see
        # `_advance_guard`). Only changes forward width, never emitted tokens.
        cooldown, miss_streak = _advance_guard(
            cooldown, miss_streak, bool(draft), n, miss_limit, cooldown_steps)

        for t, pc, stop in _emit(accepted, ctx, idx, eos_ids, max_tokens, produced):
            yield t, n
            cur, produced = t, pc
            if stop:
                return

        # The guard just tripped: the next `cooldown` tokens are guaranteed plain
        # greedy (no draft, no trim), so run them as one async-pipelined block —
        # this is the open-chat / low-reuse stretch where speculation doesn't help
        # but pipelining does. Falls back to the per-token sync path when disabled.
        if pipeline_cooldown and cooldown:
            block = _pipelined_greedy_block(model, model_cache, cur, cooldown)
            cooldown = 0
            for t, pc, stop in _emit(block, ctx, idx, eos_ids, max_tokens, produced):
                yield t, 0
                cur, produced = t, pc
                if stop:
                    return
    finally:
        # Publish the (now-extended) cache for the next request to reuse its
        # prefix. Runs on every exit — EOS, max_tokens, or client disconnect.
        if _store_to is not None:
            _store_to.store(ctx, model_cache)


def stream_prompt_lookup(
    model: "object",
    tokenizer: "object",
    prompt: str,
    max_tokens: int,
    stop: "list[str] | str | None",
    eos_id: int,
    cfg: "object",
) -> Iterator[tuple[str, "str | None"]]:
    """Server-facing adapter: stream ``(text_delta, finish_reason)`` tuples.

    Wraps :func:`prompt_lookup_generate` with detokenization, EOS handling, text
    stop-sequence truncation, and a terminal ``length`` finish — matching the
    server's ``_generate_tokens`` contract.  Output text is identical to greedy.
    """
    eos_ids = {eos_id}
    _extra = getattr(tokenizer, "eos_token_ids", None)
    if _extra:
        eos_ids.update(_extra)
    stops = [stop] if isinstance(stop, str) else list(stop) if stop else []
    ids = list(tokenizer.encode(prompt))

    out: list[int] = []
    buf = ""
    for tid, _ in prompt_lookup_generate(
        model, ids, max_tokens,
        ngram_min=getattr(cfg, "ngram_min", 2),
        ngram_max=getattr(cfg, "ngram_max", 3),
        num_draft=getattr(cfg, "max_speculative", 4),
        eos_ids=eos_ids,
        reuse_prefix=getattr(cfg, "reuse_prefix", True),
    ):
        if tid in eos_ids:
            yield "", "stop"
            return
        out.append(tid)
        full = tokenizer.decode(out)
        cut = None
        for s in stops:
            i = full.find(s)
            if i != -1 and (cut is None or i < cut):
                cut = i
        if cut is not None:
            delta = full[len(buf):cut]
            if delta:
                yield delta, None
            yield "", "stop"
            return
        delta = full[len(buf):]
        buf = full
        if delta:
            yield delta, None
    yield "", "length"
