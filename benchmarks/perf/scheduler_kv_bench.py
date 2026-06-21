#!/usr/bin/env python3
"""Before/after for PR-C — KV-cached batch decode vs the old cacheless re-forward.

Isolated, reproducible (no server): builds a real `BatchScheduler`, then runs a
batch of requests two ways and reports aggregate decode tok/s:

  OLD: rebuild the right-padded (B, max_len) sequence and re-forward the WHOLE
       batch every step (the pre-PR-C `_generate_batch` — inlined here verbatim).
  NEW: `scheduler._decode_loop` — per-request KV cache, one new-token forward
       per request per step.

The gap grows with prompt length because OLD is O(seq) per token (O(seq²) per
request) while NEW is O(1) per token.

Run:
  PYTHONPATH=. ~/squish/.venv/bin/python benchmarks/perf/scheduler_kv_bench.py
"""
from __future__ import annotations

import os
import time

import mlx.core as mx
import numpy as np

from squish.serving.scheduler import (
    BatchScheduler,
    _Request,
    _sample_token,
)

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
GEN_TOKENS = 64


def _old_cacheless_loop(sched: BatchScheduler, batch: list[_Request]) -> None:
    """The pre-PR-C batch loop: full padded re-forward every step (verbatim)."""
    active = list(batch)
    step = 0
    max_steps = max(r.max_tokens for r in active)
    while active and step < max_steps:
        seqs = [r.input_ids + r.generated_ids for r in active]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded = np.full((len(active), max_len), sched._pad_id, dtype=np.int32)
        for i, seq in enumerate(seqs):
            padded[i, : len(seq)] = seq
        logits_all = sched._model(mx.array(padded, dtype=mx.int32))
        mx.eval(logits_all)
        logits_np = np.array(logits_all.astype(mx.float32))
        still_active: list[_Request] = []
        for i, req in enumerate(active):
            next_id = _sample_token(logits_np[i, lengths[i] - 1, :],
                                    req.temperature, req.top_p, sched._rng)
            req.generated_ids.append(next_id)
            if (next_id == sched._eos_id
                    or len(req.generated_ids) >= req.max_tokens):
                req.done = True
            else:
                still_active.append(req)
        active = still_active
        step += 1


def _mk(tok, prompts: list[str], n: int) -> list[_Request]:
    return [
        _Request(request_id=str(i), input_ids=tok.encode(p), max_tokens=n,
                 temperature=0.0, top_p=1.0, stop_ids=[], seed=None)
        for i, p in enumerate(prompts)
    ]


def _bench(sched, tok, prompt: str, batch_size: int, run) -> tuple[float, int]:
    prompts = [prompt + f" (variant {i})" for i in range(batch_size)]
    reqs = _mk(tok, prompts, GEN_TOKENS)
    # drain out_queues isn't needed; we only time generation
    t0 = time.perf_counter()
    run(reqs)
    dt = time.perf_counter() - t0
    toks = sum(len(r.generated_ids) for r in reqs)
    return dt, toks


def main() -> int:
    from mlx_lm import load
    model, tok = load(_MODEL)
    sched = BatchScheduler(model, tok, max_batch_size=8)

    # NOTE: the OLD path materialises the full (B, max_len, vocab) float32 logits
    # every step (~3 GB/step at B=4 / 1k tokens / 152k vocab) and OOMs at longer
    # context — itself a limitation of the cacheless design that the per-request
    # cache removes. We bench a short prompt at two batch sizes (both safe for
    # the OLD path); the re-forward gap only widens with context length.
    base = "You are reviewing a production incident. "
    cases = [
        ("short prompt (~10 tok)", base, 4),
        ("short prompt (~10 tok)", base, 8),
    ]
    print(f"model=Qwen2.5-1.5B-int4  gen_tokens={GEN_TOKENS}  M3 16GB\n")
    print(f"{'case':<28}{'B':>3}{'OLD tok/s':>12}{'NEW tok/s':>12}{'speedup':>10}")
    print("-" * 65)
    # One short warmup to JIT both kernels (cheap), then time each path once.
    _bench(sched, tok, base, 2, lambda r: _old_cacheless_loop(sched, r))
    _bench(sched, tok, base, 2, lambda r: sched._decode_loop(r, mx, nested=False))
    for label, prompt, bs in cases:
        new_dt, new_tk = _bench(sched, tok, prompt, bs,
                                lambda r: sched._decode_loop(r, mx, nested=False))
        old_dt, old_tk = _bench(sched, tok, prompt, bs,
                                lambda r: _old_cacheless_loop(sched, r))
        old_tps = old_tk / old_dt
        new_tps = new_tk / new_dt
        print(f"{label:<28}{bs:>3}{old_tps:>12.1f}{new_tps:>12.1f}"
              f"{new_tps / old_tps:>9.2f}×")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
