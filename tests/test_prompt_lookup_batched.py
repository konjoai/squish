"""Greedy-identity contract for batched prompt-lookup speculative decoding.

The decoder must produce output token-for-token identical to plain greedy
decoding — speculation may only change how many forwards it takes, never which
tokens come out. (The prior temp-0 speculative attempt was reverted precisely
because it produced non-greedy output.) These tests pin that contract.

mlx + a small local model are required, so they skip cleanly in the CI sandbox.
"""
import os

import pytest

mx = pytest.importorskip("mlx.core")

# Run against PL_TEST_MODEL if set, else a local M3 model (skips cleanly in CI).
_MODEL = os.environ.get("PL_TEST_MODEL", "") or os.path.expanduser(
    "~/models/Qwen2.5-1.5B-Instruct-int4")
_pytestmark_model = pytest.mark.skipif(
    not os.path.isdir(_MODEL), reason=f"model not present: {_MODEL}"
)


def _greedy_reference(model, prompt_ids, max_tokens, eos_ids):
    from mlx_lm.models import cache as _cache
    c = _cache.make_prompt_cache(model)
    if len(prompt_ids) > 1:
        model(mx.array(prompt_ids[:-1], mx.uint32)[None], cache=c)
    cur, out = prompt_ids[-1], []
    for _ in range(max_tokens):
        logits = model(mx.array([cur], mx.uint32)[None], cache=c)
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
        if cur in eos_ids:
            break
    return out


@_pytestmark_model
@pytest.mark.parametrize("num_draft", [0, 1, 3, 6])
def test_greedy_identical(num_draft):
    """Prompt-lookup output == plain greedy, for any draft length."""
    mx.set_default_device(mx.cpu)
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

    model, tok = load(_MODEL)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    # Repetitive prompt so n-gram drafts actually fire (exercises the accept path).
    prompt = "Repeat exactly: red green blue red green blue red green blue red"
    pids = tok.encode(prompt)

    ref = _greedy_reference(model, pids, 48, eos)
    got = [t for t, _ in prompt_lookup_generate(
        model, pids, 48, ngram_min=2, ngram_max=3, num_draft=num_draft, eos_ids=eos)]

    assert got == ref, f"num_draft={num_draft} diverged from greedy"


# ── Adaptive accept-rate guard: pure decision logic (no model, fast) ──────────


def test_guard_grows_streak_then_trips_cooldown():
    """`miss_limit` consecutive wasted drafts enters cooldown and resets streak."""
    from squish.speculative.prompt_lookup_batched import _advance_guard

    cd, ms = 0, 0
    cd, ms = _advance_guard(cd, ms, drafted=True, n_accepted=0, miss_limit=3, cooldown_steps=16)
    assert (cd, ms) == (0, 1)
    cd, ms = _advance_guard(cd, ms, drafted=True, n_accepted=0, miss_limit=3, cooldown_steps=16)
    assert (cd, ms) == (0, 2)
    cd, ms = _advance_guard(cd, ms, drafted=True, n_accepted=0, miss_limit=3, cooldown_steps=16)
    assert (cd, ms) == (16, 0)  # tripped


def test_guard_ticks_down_during_cooldown():
    """While cooling down the counter decrements and drafting stays suppressed."""
    from squish.speculative.prompt_lookup_batched import _advance_guard

    cd, ms = 3, 0
    for expect in (2, 1, 0):
        cd, ms = _advance_guard(cd, ms, drafted=False, n_accepted=0, miss_limit=3, cooldown_steps=16)
        assert cd == expect


def test_guard_resets_streak_on_acceptance():
    """A draft that accepts >=1 token clears the miss streak."""
    from squish.speculative.prompt_lookup_batched import _advance_guard

    cd, ms = _advance_guard(0, 2, drafted=True, n_accepted=2, miss_limit=3, cooldown_steps=16)
    assert (cd, ms) == (0, 0)


def test_guard_no_draft_is_noop():
    """A free single-token step (no draft) leaves guard state unchanged."""
    from squish.speculative.prompt_lookup_batched import _advance_guard

    assert _advance_guard(0, 2, drafted=False, n_accepted=0, miss_limit=3, cooldown_steps=16) == (0, 2)


def test_guard_disabled_never_trips():
    """`cooldown_steps=0` disables the guard: misses accrue but never cool down."""
    from squish.speculative.prompt_lookup_batched import _advance_guard

    cd, ms = 0, 0
    for i in range(10):
        cd, ms = _advance_guard(cd, ms, drafted=True, n_accepted=0, miss_limit=3, cooldown_steps=0)
        assert cd == 0 and ms == i + 1


def test_accept_matches_draft_prefix():
    """_accept keeps the agreed draft prefix plus one bonus/correction token."""
    from squish.speculative.prompt_lookup_batched import _accept

    assert _accept([6, 7, 9], [6, 7]) == ([6, 7, 9], 2)   # full draft accepted
    assert _accept([6, 8, 9], [6, 7]) == ([6, 8], 1)      # first ok, second rejected
    assert _accept([5, 6, 7], [6, 7]) == ([5], 0)         # immediate rejection
    assert _accept([1], []) == ([1], 0)                   # no draft → plain token


def test_lookup_draft_caps_and_skips_when_no_match():
    """_lookup_draft returns the longest continuation, capped to num_draft+budget."""
    from squish.speculative.prompt_lookup import NGramIndex
    from squish.speculative.prompt_lookup_batched import _lookup_draft

    idx = NGramIndex(ngram_min=2, ngram_max=3, max_continuations=8)
    idx.build([1, 2, 3, 4, 1, 2, 3, 4])
    # ctx ending in "1,2" → continuation "3,4,..."; num_draft caps to 1.
    assert _lookup_draft(idx, [9, 1, 2], num_draft=1, budget=8) == [3]
    # budget caps below num_draft.
    assert _lookup_draft(idx, [9, 1, 2], num_draft=4, budget=0) == []
    # no matching n-gram → empty draft.
    assert _lookup_draft(idx, [98, 99], num_draft=4, budget=8) == []


class _StubIdx:
    def __init__(self):
        self.pushed = []

    def push(self, t):
        self.pushed.append(t)


def test_emit_updates_state_and_stops_on_eos():
    from squish.speculative.prompt_lookup_batched import _emit

    ctx, idx = [], _StubIdx()
    out = list(_emit([10, 11, 12], ctx, idx, {12}, max_tokens=100, produced=0))
    assert out == [(10, 1, False), (11, 2, False), (12, 3, True)]
    assert ctx == [10, 11, 12] and idx.pushed == [10, 11, 12]


def test_emit_stops_on_max_tokens():
    from squish.speculative.prompt_lookup_batched import _emit

    out = list(_emit([10, 11, 12], [], _StubIdx(), set(), max_tokens=2, produced=0))
    assert out == [(10, 1, False), (11, 2, True)]  # capped before the 3rd token


@_pytestmark_model
def test_adaptive_guard_is_greedy_identical():
    """The guard never changes which tokens are emitted — only forward width.

    Uses the same near-tie-free repetitive prompt as the contract test above so
    the assertion is robust (open-ended prompts can hit batched-vs-single argmax
    near-ties that flip independently of the guard). Aggressive guard params
    drive the cooldown branch during generation.
    """
    mx.set_default_device(mx.cpu)
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

    model, tok = load(_MODEL)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    prompt = "Repeat exactly: red green blue red green blue red green blue red"
    pids = tok.encode(prompt)

    ref = _greedy_reference(model, pids, 24, eos)
    guarded = [t for t, _ in prompt_lookup_generate(
        model, pids, 24, ngram_min=2, ngram_max=3, num_draft=4, eos_ids=eos,
        miss_limit=1, cooldown_steps=4)]

    assert guarded == ref


# ── stream adapter (now the default decode path for deterministic requests) ────

import types  # noqa: E402


def _cfg(**kw):
    base = {"ngram_min": 2, "ngram_max": 3, "max_speculative": 4}
    base.update(kw)
    return types.SimpleNamespace(**base)


@_pytestmark_model
def test_stream_text_matches_greedy_cpu():
    """Streamed text == detokenized plain greedy (CPU → exact, no GPU near-ties)."""
    mx.set_default_device(mx.cpu)
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import stream_prompt_lookup

    model, tok = load(_MODEL)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    prompt = "List three primary colors, comma separated: red, green, blue. Repeat: "
    ref_ids = _greedy_reference(model, tok.encode(prompt), 32, eos)
    ref_text = tok.decode([t for t in ref_ids if t not in eos])

    streamed = "".join(
        d for d, _ in stream_prompt_lookup(model, tok, prompt, 32, None,
                                           tok.eos_token_id, _cfg()))
    assert streamed == ref_text


@_pytestmark_model
def test_stream_respects_stop_sequence():
    """A stop string present in the greedy output truncates the stream before it."""
    mx.set_default_device(mx.cpu)
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import stream_prompt_lookup

    model, tok = load(_MODEL)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    prompt = "Count up: one two three four five six seven eight nine ten. Again: one two "
    full = tok.decode([t for t in _greedy_reference(model, tok.encode(prompt), 24, eos)
                       if t not in eos])
    # pick a stop token that actually occurs partway through
    stop = "five"
    if stop not in full:
        pytest.skip("greedy output did not contain the stop token")
    deltas = []
    reason = None
    for d, r in stream_prompt_lookup(model, tok, prompt, 24, [stop],
                                     tok.eos_token_id, _cfg()):
        if d:
            deltas.append(d)
        if r == "stop":
            reason = r
            break
    assert reason == "stop"
    assert stop not in "".join(deltas)


@_pytestmark_model
def test_stream_max_tokens_one_terminates():
    """max_tokens=1 yields at most one delta then a terminal finish reason."""
    mx.set_default_device(mx.cpu)
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import stream_prompt_lookup

    model, tok = load(_MODEL)
    chunks = list(stream_prompt_lookup(model, tok, "Hello", 1, None,
                                       tok.eos_token_id, _cfg()))
    assert chunks, "expected at least a terminal chunk"
    assert chunks[-1][1] in ("stop", "length")
