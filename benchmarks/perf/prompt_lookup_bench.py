#!/usr/bin/env python3
"""Prompt-lookup speculative decoding — measured win vs plain greedy (M3, batch=1).

Prompt-lookup drafts the next tokens from an in-context n-gram match (zero model
cost) and verifies them in ONE batched forward, rewinding the KV cache on
rejection. On weight-bandwidth-bound Apple-Silicon decode a (1,K) verify forward
costs far less than K separate forwards, so accepted drafts amortize the forward
across multiple tokens. A draft *miss* costs one dict lookup, so it never slows
non-copy work.

This reproduces the numbers behind making `--prompt-lookup` the default for
deterministic requests: ~2.2x on copy-heavy (RAG/quote/extract), ~1.6x on code,
~1.2x on novel prose — all wins, no regression.

Run:
  ~/squish/.venv/bin/python benchmarks/perf/prompt_lookup_bench.py [model_dir]
"""
from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np

from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
GEN_TOKENS = 120


def _greedy(model, ids, n, eos):
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(model)
    lg = model(mx.array([ids]), cache=c)
    mx.eval(lg)
    nid = int(np.argmax(np.array(lg[0, -1].astype(mx.float32))))
    out = []
    for _ in range(n):
        out.append(nid)
        if nid in eos:
            break
        lg = model(mx.array([[nid]]), cache=c)
        mx.eval(lg)
        nid = int(np.argmax(np.array(lg[0, -1].astype(mx.float32))))
    return out


_TASKS = {
    "copy-heavy (RAG quote)": (
        "CONTEXT: The Falcon-9 first stage landed on the droneship at 08:42 UTC after a "
        "nominal entry burn. The payload was a 4.2-tonne communications satellite bound "
        "for geostationary transfer orbit. Telemetry showed nominal stage separation. " * 3
        + "\nTASK: Quote verbatim the full sentence about the droneship landing, then the "
          "payload mass.\nANSWER:"),
    "code continuation": (
        "def fibonacci(n):\n    if n < 2: return n\n    a, b = 0, 1\n"
        "    for _ in range(n - 1): a, b = b, a + b\n    return b\n\n"
        "# Write fib_list(n) returning the first n fibonacci numbers as a list\n"
        "def fib_list(n):\n"),
    "novel prose": (
        "Write an original short story about a lighthouse keeper who finds a message in "
        "a bottle:\n"),
}


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    from mlx_lm import load
    model, tok = load(model_path)
    eos = {tok.eos_token_id}

    print(f"model={os.path.basename(model_path)}  N={GEN_TOKENS}  M3 batch=1\n")
    print(f"{'task':28}{'greedy t/s':>12}{'plookup t/s':>13}{'speedup':>9}")
    for name, prompt in _TASKS.items():
        ids = tok.encode(prompt)
        list(prompt_lookup_generate(model, ids, 4, eos_ids=eos))  # warm

        t0 = time.perf_counter()
        g = _greedy(model, ids, GEN_TOKENS, eos)
        g_tps = len(g) / (time.perf_counter() - t0)

        t0 = time.perf_counter()
        p = [t for t, _ in prompt_lookup_generate(model, ids, GEN_TOKENS, eos_ids=eos)]
        p_tps = len(p) / (time.perf_counter() - t0)

        print(f"{name:28}{g_tps:>11.1f}{p_tps:>13.1f}{p_tps / g_tps:>8.2f}x")
    print("\n(plookup output is greedy-equivalent; bit-identical on CPU, may differ on "
          "GPU near-ties due to batched-vs-single-token matmul numerics.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
