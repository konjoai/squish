#!/usr/bin/env python3
"""Rigorous repeated A/B: plain greedy vs prompt-lookup, per workload.

Benchmarking rules: >=5 reps, report median/min/max + speedup. Determines
whether any workload REGRESSES under prompt-lookup (the gate for making it
default-on). Greedy is deterministic so each rep is the same tokens; variance
is pure timing.

Run:
  ~/squish/.venv/bin/python benchmarks/perf/prompt_lookup_rigorous.py [model_dir]
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

import mlx.core as mx
import numpy as np

from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
GEN_TOKENS = 120
REPS = 7
WARMUP = 2


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
    "copy-heavy (RAG)": (
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
    "chat Q&A (no context reuse)": (
        "What are three benefits of regular exercise? Answer concisely.\n"),
}


def _bench(fn):
    times = []
    for _ in range(WARMUP):
        fn()
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) / len(out))  # sec/token
    return times  # list of sec/token


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    from mlx_lm import load
    model, tok = load(model_path)
    eos = {tok.eos_token_id}

    print(f"model={os.path.basename(model_path)}  N={GEN_TOKENS}  reps={REPS}  M3 batch=1\n")
    hdr = (f"{'task':30}{'greedy':>9}{'unguard':>9}{'x':>7}{'guarded':>9}{'x':>7}"
           f"{'guard verdict':>15}")
    print(hdr)
    any_regress = False
    for name, prompt in _TASKS.items():
        ids = tok.encode(prompt)
        g_t = _bench(lambda ids=ids: _greedy(model, ids, GEN_TOKENS, eos))
        u_t = _bench(lambda ids=ids: [t for t, _ in prompt_lookup_generate(
            model, ids, GEN_TOKENS, eos_ids=eos, cooldown_steps=0)])  # unguarded
        a_t = _bench(lambda ids=ids: [t for t, _ in prompt_lookup_generate(
            model, ids, GEN_TOKENS, eos_ids=eos)])  # guarded (default)
        g_tps = 1.0 / st.median(g_t)
        u_tps = 1.0 / st.median(u_t)
        a_tps = 1.0 / st.median(a_t)
        u_spd = u_tps / g_tps
        a_spd = a_tps / g_tps
        # regression gate on guarded: p95 per-token latency vs greedy
        a_regressed = np.percentile(a_t, 95) > np.percentile(g_t, 95) * 1.05
        verdict = "REGRESS" if a_regressed else ("win" if a_spd > 1.02 else "neutral")
        if a_regressed:
            any_regress = True
        print(f"{name:30}{g_tps:>9.1f}{u_tps:>9.1f}{u_spd:>6.2f}x"
              f"{a_tps:>9.1f}{a_spd:>6.2f}x{verdict:>15}")
    print(f"\nany guarded p95 regression >5%: {any_regress}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
