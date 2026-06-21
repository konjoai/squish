"""Live-model tests for the per-request KV-cached batch decode loop.

These exercise the MLX-only `BatchScheduler._decode_loop` path (previously
`# pragma: no cover`, only reachable with a real model). They assert the core
correctness guarantee of the rewrite: batched greedy output is BIT-IDENTICAL to
the single-stream generation path, because both forward only the new token
through a per-request KV cache.

Gated on Apple Silicon + a local Qwen2.5-1.5B-int4 model; skipped elsewhere.
"""
from __future__ import annotations

import os
import platform

import numpy as np
import pytest

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not os.path.isdir(_MODEL),
    reason="requires Apple Silicon MLX + local Qwen2.5-1.5B-int4 model",
)


@pytest.fixture(scope="module")
def model_tok():
    from mlx_lm import load
    return load(_MODEL)


def _single_stream_greedy(model, ids: list[int], n: int) -> list[int]:
    """Reference: greedy decode of one prompt via its own incremental cache."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    nid = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
    out: list[int] = []
    for _ in range(n):
        out.append(nid)
        logits = model(mx.array([[nid]]), cache=cache)
        mx.eval(logits)
        nid = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
    return out


def _make_reqs(tok, prompts: list[str], n: int):
    from squish.serving.scheduler import _Request
    return [
        _Request(request_id=str(i), input_ids=tok.encode(p), max_tokens=n,
                 temperature=0.0, top_p=1.0, stop_ids=[], seed=None)
        for i, p in enumerate(prompts)
    ]


def test_batch_greedy_bit_identical_to_single_stream(model_tok):
    """The whole point of the rewrite: batched == single-stream, token for token."""
    import mlx.core as mx
    from squish.serving.scheduler import BatchScheduler

    model, tok = model_tok
    # Deliberately different prompt lengths — the case the old padded re-forward
    # got numerically wrong.
    prompts = [
        "The capital of France is",
        "In a distant galaxy, a lone engineer discovered that the quantum",
        "2 + 2 =",
    ]
    n = 20
    sched = BatchScheduler(model, tok, max_batch_size=4)
    reqs = _make_reqs(tok, prompts, n)
    sched._decode_loop(reqs, mx, nested=False)

    for i, p in enumerate(prompts):
        ref = _single_stream_greedy(model, tok.encode(p), n)
        gen = reqs[i].generated_ids
        # Generation may stop early on EOS; require it to be an exact prefix.
        assert gen == ref[:len(gen)], (
            f"prompt {i!r} diverged from single-stream:\n"
            f"  batch: {gen}\n  ref:   {ref[:len(gen)]}"
        )


def test_batch_streams_to_out_queue_and_finishes(model_tok):
    """Each request streams tokens then a terminal _DONE sentinel."""
    import mlx.core as mx
    from squish.serving.scheduler import _DONE, BatchScheduler

    model, tok = model_tok
    sched = BatchScheduler(model, tok, max_batch_size=4)
    reqs = _make_reqs(tok, ["Hello", "The sky is"], n=8)
    sched._decode_loop(reqs, mx, nested=False)

    for req in reqs:
        assert req.done
        drained = []
        while not req.out_queue.empty():
            drained.append(req.out_queue.get_nowait())
        assert drained[-1] is _DONE
        # finish reason rides on the final (text, reason) tuple before _DONE
        text_items = [d for d in drained if d is not _DONE]
        assert text_items, "no tokens streamed"
        assert text_items[-1][1] in ("stop", "length")


def test_single_request_batch_matches_single_stream(model_tok):
    """B=1 through the batch loop must equal single-stream exactly."""
    import mlx.core as mx
    from squish.serving.scheduler import BatchScheduler

    model, tok = model_tok
    sched = BatchScheduler(model, tok, max_batch_size=4)
    prompt = "List three colors:"
    n = 12
    reqs = _make_reqs(tok, [prompt], n)
    sched._decode_loop(reqs, mx, nested=False)
    ref = _single_stream_greedy(model, tok.encode(prompt), n)
    gen = reqs[0].generated_ids
    assert gen == ref[:len(gen)]
