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

_MODEL = "/Users/wscholl/models/Qwen3-0.6B-int4"
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
