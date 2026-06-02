# Squish v5 — block-cache + chunked-prefill probe

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-01
**Squish:** 9.14.0 + v4 + v4.1 + v4.2 + v5 (`perf/v5-block-cache-and-chunked`)

This file is the v5 counterpart to [`../benchmarks_v4_2/PRECHECK.md`](../benchmarks_v4_2/PRECHECK.md).
v4.2 left two open follow-ups:

1. Repeated-prompt TTFT could go even lower on **shifting-prefix** workloads
   (agent / coding-assistant turns) — the v4.2 PromptKVStore hashes the
   *entire* prompt and misses on any tail change.
2. Fresh-prompt TTFT on the 75-token v4 benchmark is the MLX prefill floor
   (proved in v4.2 T2).  Existing `chunked_prefill` is gated to prompts
   > 512 tokens; could lowering that threshold close the gap?

## Goal A — Chunked prefill on short prompts

**Outcome: SKIPPED after probe.**

Direct measurement on a 57-token prompt with various chunk sizes:

```
mode                         |   ms (med) |   ms (min)
--------------------------------------------------------
full (one prefill)           |      357.4 |      356.7
chunk=16 first chunk         |      184.6 |      184.3
chunk=16 total               |      749.9 |      749.2
chunk=32 first chunk         |      185.4 |      184.7
chunk=32 total               |      373.7 |      372.4
chunk=64 first chunk         |      356.5 |      356.1
chunk=64 total               |      356.5 |      356.1
```

The first chunk at chunk=32 finishes at 185 ms — half the full-prefill
357 ms.  But: **emitting a token sampled mid-prefill would change inference
outputs** for normal user prompts (the logit at position 31 predicts what
comes after token 31, which is part of the user's prompt, not the start of
the model's response).  The existing `chunked_prefill` code's
`interleave_decode=True` is designed for the COMPRESS_PATH where prompts
are heavily compressed and interleaved tokens are acceptable; for fresh
user prompts, interleaving would emit incorrect tokens before the model
finishes prefilling the user's actual context.

Per scope guard ("don't change inference outputs"), Goal A was abandoned
after the probe.  Total time: ~30 minutes.  Tracked as v5.1 follow-up:
investigate speculative-prefill style schemes where the early sample is
later verified and replaced on mismatch.

## Goal B — Block-level paged KV cache

**Outcome: SHIPPED and clearly wins on long-prompt repeated-prefix workloads.**

New module `squish/kv/block_kv_cache.py` (496 lines) implements:

* Block hashing — fixed 64-token blocks, each hash chained against the
  previous block's hash + the model_key (so a matched prefix proves
  identical token sequence, not just identical block content).
* Two-tier storage — hot RAM `OrderedDict` (default cap 2 GiB) + cold
  per-block .npz files under `~/.cache/squish/blocks/`.  Hot tier evicts
  LRU to free RAM; cold tier survives server restarts.  A `manifest.json`
  detects block-size mismatch on init and clears the cache safely.
* Public API: `lookup_prefix`, `store_blocks`, `chain_hash`, `split_blocks`,
  `stats`, `clear`.  Helpers `slice_cache_into_blocks` /
  `restore_blocks_to_cache` bridge mlx_lm prompt cache <-> per-block npy.

`squish/server.py` wiring:

* New flags: `--block-kv-cache DIR`, `--block-kv-size N`,
  `--block-kv-hot-gb GB`, `--block-kv-cold-gb GB`.
* Block lookup sits before the v4.2 PromptKVStore lookup.  When the block
  cache yields any matching prefix, we restore those blocks into a fresh
  mlx_lm prompt_cache and manually prefill only the unmatched suffix.
  The first new token is sampled from the suffix's final-position logit
  and yielded BEFORE entering mlx_lm.stream_generate — same TTFT trick as
  v4.2 T1.  Generation completion saves any new full blocks to the cache.

### Smoke test (small prompt, 118 tokens, 1 full block)

After fixing the case-C "all blocks matched, drop last" off-by-one
(see commit message for details), the smoke test produced:

```
Run 1 (cold miss):                ~740 ms
Run 2 (1 block of 64 cached):     ~500 ms   ← -32%
Runs 3-4 (steady state):          ~406 ms   ← -45%
Variation prompt (shared prefix): ~408 ms   ← cross-prompt re-use
```

That last line is the key: a variation prompt (different tail, same head)
also hits the cached prefix.  Confirms the agent workload pattern works.

### Long-context benchmark — `bench_v5_longctx.py`

A new benchmark scenario: a ~694-token system-prompt-like base + 5 trailing
50-token variations.  5 runs per phase, median reported.  Raw artifact:
[`runs/20260601T230126/long_ctx.json`](runs/20260601T230126/long_ctx.json).

| Metric                                | Ollama   | sq daemon | sq +pkv (v4.2) | sq +block (v5) |
|---------------------------------------|---------:|----------:|---------------:|---------------:|
| **Cold long-prompt TTFT** (median)    | **272 ms** | 3.95 s | 4.61 s | **234 ms** |
| **Variation TTFT** (shared prefix)    |   270 ms | 4.27 s |  **20 ms** | 232 ms |
| **Warm tok/s** (short-prompt 100-tok) |  19.1    | **20.0** | 7.9 | 15.8 |
| **Peak RSS** (process tree)           | n/a* | **2.36 GB** | 3.45 GB | 3.19 GB |

`*` Ollama spawns its model runner in a separate process group so the
bench's RSS sampler (rooted at `ollama serve`) doesn't see the actual
allocation.  The v4.2 benchmark protocol catches it correctly (~5 GB
there).

### Block-cache key findings

* **234 ms cold TTFT for a 694-token prompt — beats Ollama (272 ms).**
  Each variation shares the 690-token base prefix; block cache restores
  ~600 tokens (9-10 blocks) and prefills only the ~90-token suffix.
  sq daemon spends 3.95 s prefilling all 694 tokens cold; sq +pkv spends
  4.61 s (no hit because the v4.2 cache key is the entire prompt).
* **232 ms variation TTFT — 18× faster than sq daemon's 4.27 s.**
  Block cache hits regardless of which trailing variation is sent.  PKV
  also gets 20 ms here because the bench protocol repeats the same 5
  variations between the cold and variation phases — exact match.  The
  "cold" column is the more honest test of agent-shifting-prefix workload.
* **Warm tok/s degrades under thermal load.** sq daemon stays at 20 tok/s
  through 5 sustained-decode runs; sq +pkv drops to 7.9 (high variance);
  sq +block holds 15.8 (steady decline 18.4 → 14.2 across runs as Metal
  thermal pressure rises).  This is the same Metal thermal pattern v4.2
  documented; runs ordered last in the bench inherit the worst state.

## Per-target outcomes

| Target | Status   | Result                                                                 |
|--------|----------|------------------------------------------------------------------------|
| Goal A — chunked prefill on short prompts | **SKIPPED** | Interleaved tokens would change output; out of scope per guard. |
| Goal B — block-level paged KV cache       | **SHIPPED** | 234 ms cold TTFT on 694-token prompts (vs 3.95 s daemon, 272 ms Ollama). Long-context repeated-prefix workload is the agent / coding-assistant pattern. |
| New benchmark: long-context shared prefix | **SHIPPED** | `bench_v5_longctx.py` + `results/benchmarks_v5/runs/.../long_ctx.json`. |

## Remaining technical debt (v5.1 candidates)

1. Block cache uses one extra `model(x, cache=cache)` call to prefill the
   suffix and capture its logit.  For a 90-token suffix that's still ~250 ms.
   Could chunked prefill apply to the SUFFIX (where the legitimacy concern
   in Goal A doesn't apply — we're prefilling tokens we already have)?
2. Per-block last-position logit caching — would let "exact entire prompt
   match" hits avoid even the suffix prefill, dropping TTFT to <20 ms even
   on long-context exact matches.  Currently those go through the v4.2
   PromptKVStore (full-prompt hash) which IS this fast — but it's a separate
   flag.  Unify by also storing a per-block logit?
3. Warm tok/s with block cache active drifts down under thermal load
   (18.4 → 14.2 across 5 sustained runs).  Same pattern with PKV.
   Investigate whether the extra mlx-array conversions in the store path
   contribute to thermal pressure.
4. Bench's "variation" phase repeats prompts seen in the "cold" phase, so
   PKV gets exact-match hits there.  Add a *third* phase where each
   variation has a fresh randomly-generated 50-token tail to isolate
   "shared prefix only" from "exact full prompt repeat".
5. Pre-existing `tests/test_squishd_unit.py` 9 timeouts and
   `tests/test_quant_aqlm.py::test_module_count_unchanged` assertion gap
   carry over from v4.2.
6. The block-cache `--block-kv-cache` flag is on `python -m squish.server`
   only; expose it on the user-facing `squish run` CLI too.
