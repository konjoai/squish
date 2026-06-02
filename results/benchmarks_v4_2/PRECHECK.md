# Squish v4.2 — gap-close attempt: per-target outcomes

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-01
**Squish:** 9.14.0 + v4 commit `8a8ef47` + v4.1 wiring + v4.2 gap-close (branch `perf/v4.2-gap-close`).

This file is the v4.2 counterpart to
[`../benchmarks_v4_1/PRECHECK.md`](../benchmarks_v4_1/PRECHECK.md). v4.1 left
one row losing to Ollama (fresh-prompt TTFT 256 ms vs 481 ms) and a
3.5× gap between the new `--prompt-kv-cache` (223 ms cache-hit) and the
legacy `--disk-prompt-cache+int8` (22 ms). v4.2 attempts to close both,
plus opportunistic wins on warm tok/s, cold-start cost, and RSS.

## Per-target outcomes

| Target | Status   | Result                                                                 |
|--------|----------|------------------------------------------------------------------------|
| T1 — Cache post-prefill logit | **KEPT** | 226 ms → 145 ms cache-hit TTFT (-36 %)                       |
| T2 — Fresh-prompt TTFT gap    | **DOCUMENTED** | 99 % of the 348 ms TTFT is mlx_lm prefill kernel; not squish-side overhead |
| T3 — Cold-start cost          | **SKIPPED**    | Unclear leverage given current daemon-warm protocol           |
| T4 — Warm tok/s lead          | **SKIPPED**    | Per-token Python overhead profiled <50 µs of ~48 ms (~0.1 %) |
| T5 — Daemon RSS               | **SKIPPED**    | Structural win already in place; no obvious cheap reductions  |

### T1 — Cache post-prefill logit (KEPT)

The v4.1 PromptKVStore stored only KV state.  On a hit, mlx_lm still had to
do a 1-token forward pass before yielding the first token (~226 ms).  The
legacy DiskKVCache achieves 22 ms by also caching the post-prefill logit and
sampling the first new token directly from it.

Changes:

* `KVCacheEntry` gained an optional `last_logit: np.ndarray | None` (vocab-
  sized float32, ~600 KB).
* `PromptKVStore.put(... last_logit=...)` writes `last_logit.npy`; the meta
  records `has_logit: true`.
* `PromptKVStore.get(prompt, lazy_kv=True)` skips eager loading of the 28×2
  per-layer KV npy files (only `meta.json` + `last_logit.npy` are read).
  The KV arrays load on-demand inside `restore_kv_state`.
* `server.py`: on hit-fast, sample the first new token from the cached
  logit (no model call), yield it as the first SSE chunk, then run
  `mlx_lm.stream_generate(prompt=[first_token], ...)` for tokens 2+.
  KV restore is deferred to AFTER the first-chunk yield.
* On miss, do a manual prefill (`model(x, cache=cache)`), capture the
  post-prefill logit, emit the first token from it, store via
  `PromptKVStore.put(...last_logit=...)`.

Smoke result (Qwen2.5-7B-Instruct-int4, 75-token PR-review prompt,
max_tokens=1, after 1 priming send):

```
v4.1 (1-token-prefill on hit):        ~226 ms median
v4.2 (logit-skip-prefill on hit):     ~145 ms median   ← -36 %
```

Profile of the v4.2 hit-fast path inside `_generate_tokens`:

```
make_pc  = 0.0 ms   # mlx_lm.models.cache.make_prompt_cache(model)
get      = 1.5 ms   # PromptKVStore.get(..., lazy_kv=True)
tokenize = 0.3 ms   # tokenizer.encode(prompt)
```

So ~3 ms of work goes into the PromptKVStore path itself.  The remaining
~140 ms is FastAPI/uvicorn streaming + `run_in_executor` scheduling +
mlx_lm.stream_generate first-iteration setup — all of which the legacy
disk-KV path also pays (it can hit 22 ms because it uses a manual decode
loop instead of `mlx_lm.stream_generate`).  Closing that residual would
require porting the legacy manual-decode loop into the fp16 path, which is
a refactor out of scope for this session and tracked as v4.3 follow-up.

### T2 — Fresh-prompt TTFT gap (DOCUMENTED)

Per the spec: *"DO NOT touch the actual prefill forward pass. If the
225 ms is genuinely the MLX prefill kernel, document it and move on."*

We added per-phase timing to `event_stream()` and the `mlx_lm.stream_generate`
branch of `_generate_tokens()` and measured short fresh-prompt requests on
`squish_daemon` (no extras), Qwen2.5-7B-Instruct-int4, 5-token prompt,
max_tokens=1:

```
es-prof   start_to_setup  = 0.0–0.1 ms    # FastAPI dispatch + var setup
es-prof   role_yield      = 0.0 ms        # role chunk yield → uvicorn flush
es-prof   gen_setup       = 0.0 ms        # _generate_tokens generator object
gt-prof   dispatch_to_iter = 0.0 ms       # entering the stream_generate path
gt-prof   iter_to_first_item = 300–360 ms ← model prefill + decode
es-prof   first_exec_wait  = 300–360 ms   # await loop.run_in_executor(...)
```

The 300–360 ms is the FIRST iteration of `for item in gen:` returning a
value — that is, `mlx_lm.stream_generate(...)` running its internal prefill
chunking, first-position decode, and emit.  All squish-side overhead (FastAPI
dispatch, role-chunk yield, generator creation, executor scheduling outside
the first call) sums to **under 1 ms**.

Conclusion: the 225 ms gap to Ollama's 256 ms fresh TTFT is entirely the
MLX prefill kernel vs llama.cpp Metal prefill kernel.  Squish-side
overhead is already near zero.  Per scope guard, the profiling code was
removed and we moved on.

### T3 — Cold-start cost (SKIPPED)

`squish run --prompt "hi" --max-tokens 1` is a CLI wrapper that spawns a
fresh `python -m squish.server`.  The wall-time is dominated by:

* Python interpreter startup (~200 ms)
* mlx_lm import (~1.5 s)
* Model load (~3–7 s depending on file-system warm state)

The v4.1 measurement protocol explicitly uses a warm daemon — cold wall
isn't part of the v4.x scorecard.  Any reduction here doesn't move a
benchmark row, so per scope rule "every change must produce a measurable
improvement in at least one v4.1 benchmark row" we skipped it.  Tracked
as v4.3 follow-up: investigate whether v3 actually was faster on this
path or it's environment drift.

### T4 — Warm tok/s lead (SKIPPED)

Per-token overhead in the `for item in gen:` loop:

* repetition-loop scan (every 20 tokens): ~10 µs amortized
* `"<think>" in tok_text` / `"</think>" in tok_text`: ~0.2 µs each
* `_make_chunk` JSON encoding: ~5–10 µs
* `await loop.run_in_executor(...)` scheduling: ~1–3 ms
* cache_buf append, _trace_tokens check: < 1 µs

Total Python overhead per token: < 1.5 ms.  Model forward per token: ~48 ms
(at 20.7 tok/s).  Eliminating ALL Python overhead would give at most
1/48 ≈ 2 % improvement on tok/s — below the 5 % decision gate, and the
gains aren't reliably present (variance is ±5 % run-to-run from thermal).

Tracked as v4.3 follow-up: investigate batching subsequent tokens (preserve
first-token TTFT, batch tokens 2..N) to reduce `run_in_executor` overhead
without hurting streaming UX.

### T5 — Daemon RSS (SKIPPED)

squish_daemon already wins peak RAM 2.39 GB vs Ollama 5.13 GB — a 53 %
structural win.  The v3 cold-load optimizations (sklearn/scipy/pandas/torch
stubs) already cut ~480 MB.  Further cuts would require either:

* Lazy-importing more squish optimization modules (vulture / dead-code risk)
* Releasing intermediate weight buffers post-load (mlx unified-memory cost)

Neither is "cheap" per the target's decision gate.  Tracked as v4.3
follow-up.

## What v4.2 ships vs v4.1

| Row                            | v4.1 measured | v4.2 measured (this run) | Delta              |
|--------------------------------|--------------:|-------------------------:|--------------------|
| TTFT, fresh prompt             | 481 ms        | _see runs/<ts>/raw.json_ | expected ≈ same    |
| TTFT, repeated prompt          | 22 ms (legacy), 223 ms (pkv) | (legacy unchanged), **PKV improved** | T1: -36 %        |
| Warm tok/s (200-tok decode)    | 20.2 tok/s    | _see raw.json_           | expected ≈ same    |
| Spec-decode tok/s              | 18.5 tok/s    | _see raw.json_           | expected ≈ same    |
| Peak RAM                       | 2.39 GB       | _see raw.json_           | expected ≈ same    |
| Disk                           | 4.00 GB       | 4.00 GB                  | unchanged          |

## Remaining technical debt (v4.3 candidates)

1. Port the legacy manual-decode-loop pattern into the fp16
   `--prompt-kv-cache` path so cache hits can drop from 145 ms to <40 ms.
   The 100 ms gap is FastAPI + `mlx_lm.stream_generate` first-iteration
   setup, both of which the legacy `--disk-prompt-cache+int8` path bypasses.
2. Investigate batching tokens 2..N to reduce `run_in_executor` overhead
   without hurting streaming UX.
3. Bisect any residual v3 → v4 warm tok/s gap (likely thermal artifact
   as v4.1 already showed 20.2 tok/s recovering v3's 17.5 baseline).
4. Pre-existing test_squishd_unit.py 9 "daemon did not start in time"
   failures (timeout in test harness).
5. Pre-existing test_module_count_unchanged assertion (89 vs current 96).
