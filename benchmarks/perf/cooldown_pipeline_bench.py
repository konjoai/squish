#!/usr/bin/env python3
"""A/B: prompt-lookup cooldown block — sync vs async_eval-pipelined.

On low-reuse workloads (open chat) the adaptive guard spends most of its time in
cooldown running plain greedy. This measures the speedup from pipelining those
cooldown forwards with mx.async_eval vs the per-token sync path. Output is
identical either way (verified by the greedy-identity probe/tests).

Run:
  ~/squish/.venv/bin/python benchmarks/perf/cooldown_pipeline_bench.py [model_dir]
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
N = 120
REPS = 7
WARMUP = 2

_TASKS = {
    "chat Q&A": "What are three benefits of regular exercise? Answer concisely.\n",
    "open prose": "Write an original short paragraph about a city at dawn.\n",
    "explain": "Explain how a bicycle stays upright while moving, in plain language.\n",
}


def _bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) / len(out))
    return ts


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    from mlx_lm import load
    model, tok = load(model_path)
    eos = {tok.eos_token_id}

    print(f"model={os.path.basename(model_path)}  N={N}  reps={REPS}  M3 batch=1\n")
    print(f"{'task':24}{'sync t/s':>10}{'pipelined t/s':>15}{'speedup':>9}")
    for name, prompt in _TASKS.items():
        ids = tok.encode(prompt)
        sync = _bench(lambda ids=ids: [t for t, _ in prompt_lookup_generate(
            model, ids, N, eos_ids=eos, pipeline_cooldown=False)])
        pipe = _bench(lambda ids=ids: [t for t, _ in prompt_lookup_generate(
            model, ids, N, eos_ids=eos, pipeline_cooldown=True)])
        s_tps, p_tps = 1.0 / st.median(sync), 1.0 / st.median(pipe)
        print(f"{name:24}{s_tps:>10.1f}{p_tps:>15.1f}{p_tps / s_tps:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
