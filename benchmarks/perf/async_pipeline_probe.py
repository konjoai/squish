#!/usr/bin/env python3
"""Microbench: sync greedy decode vs mx.async_eval-pipelined greedy decode.

Tests the headline "decode pipelining" opportunity — overlap the GPU forward of
token N+1 with the CPU-side read/detok of token N by keeping argmax lazy on the
GPU and using mx.async_eval (mlx-lm's generate_step pattern), which squish does
NOT currently use on its manual decode loop.

Both variants are plain greedy (argmax), so output is identical; only the
host<->device scheduling differs. Reports median tok/s over repeated runs.

Run:
  ~/squish/.venv/bin/python benchmarks/perf/async_pipeline_probe.py [model_dir]
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
N = 128
REPS = 7
WARMUP = 2


def sync_greedy(model, prompt_ids, n):
    """Current squish-style: eval logits, read argmax via .item() each step."""
    c = make_prompt_cache(model)
    x = mx.array(prompt_ids, mx.uint32)[None]
    logits = model(x, cache=c)
    mx.eval(logits)
    y = int(mx.argmax(logits[0, -1]).item())
    out = [y]
    for _ in range(n - 1):
        logits = model(mx.array([[y]], mx.uint32), cache=c)
        mx.eval(logits)
        y = int(mx.argmax(logits[0, -1]).item())
        out.append(y)
    return out


def async_greedy(model, prompt_ids, n):
    """Pipelined: argmax stays on GPU, async_eval overlaps next forward with the
    .item() read of the previous token (mlx-lm generate_step pattern)."""
    c = make_prompt_cache(model)

    def step(x):
        logits = model(x, cache=c)
        return mx.argmax(logits[0, -1])  # lazy, stays on GPU

    x = mx.array(prompt_ids, mx.uint32)[None]
    y = step(x)
    mx.async_eval(y)
    out = []
    for _ in range(n - 1):
        next_y = step(y[None][None])  # queue next forward (graph builds now)
        mx.async_eval(next_y)
        out.append(int(y.item()))     # forces prev y; overlaps with next forward
        y = next_y
    out.append(int(y.item()))
    return out


def bench(fn, model, pids):
    for _ in range(WARMUP):
        fn(model, pids, N)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn(model, pids, N)
        ts.append((time.perf_counter() - t0) / len(out))
    return ts


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    from mlx_lm import load
    model, tok = load(model_path)
    pids = tok.encode("Write a detailed paragraph about ocean currents and climate.\n")

    s = bench(sync_greedy, model, pids)
    a = bench(async_greedy, model, pids)
    # correctness: identical tokens
    ok = sync_greedy(model, pids, 16) == async_greedy(model, pids, 16)
    s_tps, a_tps = 1.0 / st.median(s), 1.0 / st.median(a)
    print(f"model={os.path.basename(model_path)}  N={N}  reps={REPS}  M3 batch=1")
    print(f"  sync   greedy: {s_tps:6.1f} tok/s  (median itl {st.median(s)*1e3:.2f} ms)")
    print(f"  async  greedy: {a_tps:6.1f} tok/s  (median itl {st.median(a)*1e3:.2f} ms)")
    print(f"  speedup: {a_tps / s_tps:.3f}x   tokens identical: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
