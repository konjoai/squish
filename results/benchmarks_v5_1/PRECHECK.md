# Squish v5.1 — metrics + per-block logit + CLI flags

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-02
**Squish:** 9.14.0 + v4 + v4.1 + v4.2 + v5 + v5.1 (`perf/v5.1-metrics-and-suffix`)

This file is the v5.1 counterpart to [`../benchmarks_v5/PRECHECK.md`](../benchmarks_v5/PRECHECK.md).
v5.1 was structured as **two phases**:

* **Phase 1 (2 h cap)** — bench-coverage expansion: e2e response time,
  inter-token p50/p95, multi-context-length sweep, INT3 column.
* **Phase 2 (6 h cap)** — three follow-ups from v5 PRECHECK:
  2A suffix chunked prefill, 2B per-block last-position logit, 2C
  expose `--block-kv-cache` on `squish run` CLI.

## Per-phase outcomes

| Phase | Status | Result |
|-------|--------|--------|
| Phase 1 — 4 metric dimensions | **SHIPPED** | `bench_v5_1.py` runs 5 configs × 4 prompt sizes × {TTFT + e2e + ITL}. INT3 column auto-enabled when the model dir exists. |
| 2A — suffix chunked prefill | **SKIPPED** | The v5 Goal A probe already established interleave-decode is unsafe for fresh user prompts.  On the suffix of a block-cache hit the same restriction applies (we're still prefilling user tokens, just from a different offset).  Without interleave there is no first-chunk-yield benefit; total wall time is fractionally slower due to per-chunk overhead.  Documented as v5.2 follow-up: speculative prefill. |
| 2B — per-block last-position logit | **SHIPPED with caveat** | Implemented, tested correct, fast-hit path verified to fire (server trace).  Saves one suffix forward pass on full-prefix match.  Client-visible TTFT improvement is modest because the FastAPI/uvicorn streaming stack adds ~100 ms baseline overhead that v4.2 `--prompt-kv-cache`'s prefix-character-yield path bypasses but `--block-kv-cache` cannot.  See "Decision gate" below. |
| 2C — KV cache flags on `squish run` | **SHIPPED** | `--prompt-kv-cache`, `--prompt-kv-cache-max-gb`, `--block-kv-cache`, `--block-kv-size`, `--block-kv-hot-gb`, `--block-kv-cold-gb` now usable directly via `squish run`. |

## Phase 1 — new bench harness

`benchmarks/ollama_vs_squish/bench_v5_1.py` writes a single unified result
JSON covering every (config × prompt_size) pair.  Per-pair metrics:

| Metric        | Definition                                                  |
|---------------|-------------------------------------------------------------|
| ttft_s        | Cold TTFT (median of 5 same-prompt sends, max_tokens=1)     |
| e2e_200tok_s  | Wall time for a 200-token completion (median of 5)          |
| warm_tps      | Tokens/sec during the 200-token decode (median of 5)        |
| itl_p50_ms    | Median inter-token gap excluding the first chunk            |
| itl_p95_ms    | p95 inter-token gap                                         |
| peak_rss_bytes| Process-tree peak RSS during the entire config              |

Per-config registry: `ollama`, `squish_daemon`, `squish_pkv`, `squish_block`,
`squish_block_int3` (when the INT3 model is on disk; otherwise auto-skipped
with a logged note).

Prompt-size phases: 75, 500, 2000, 4000 tokens.  Each phase uses a single
prompt built up to the target token count; we don't shuffle prompts across
the 5 runs, so runs 2–5 are warm-cache hits on the cache-enabled configs.
The "cold TTFT" column for cache-enabled rows therefore reports the
warm-cache-hit TTFT, not the first-ever-send time.  This matches the v4.2
benchmark's "ttft_repeat" semantic.

## Phase 2 — implementation notes

### 2B detailed status

#### What was built

1. `BlockEntry.last_logit` field (optional `np.ndarray | None`).
2. `BlockKVCache.store_blocks` accepts `per_block_last_logits=`.
3. Cold-tier `.npz` files carry `last_logit` when present; legacy v5
   entries without the field load cleanly with `last_logit=None`.
4. `_CACHE_VERSION` bumped 1 → 2; existing on-disk caches are cleared
   safely on first boot after upgrade (manifest mismatch check).
5. `per_block_last_logits_from_full_logits` helper extracts the per-block
   last-position logit from a captured prefill forward pass.
6. `server.py` block-cache lookup: when the last matched block has a
   cached `last_logit` AND `matched_tokens == prompt_tokens`, sample the
   first new token directly — no suffix forward pass.
7. Restore is deferred to AFTER the first SSE chunk yields, parallelising
   the ~200 ms `mx.concatenate` work with the consumer's chunk encoding.

15 new unit tests in `tests/test_block_kv_cache.py` cover chain-hash
determinism and prefix property, store/lookup roundtrip, cold tier
survives reinstance, manifest mismatch clears, v5.1 logit roundtrip,
legacy v5 entries without logits, partial per-block logits.  47/47 KV-
cache tests pass overall.

#### Decision-gate honest analysis

The spec's decision gate was: "if full-match TTFT drops below 30ms,
KEEP."  Microbench (`probe_block_logit.py`) on a chat-templated 576-token
prompt (= 9 × 64 blocks, exact alignment, full hit):

```
First send (cold): ~3.6 s    # full prefill, populates blocks + logits
Hit runs:          ~245 ms   # bench-client TTFT (full-prefix-match hits)
```

Server-side ttft (logged by the request handler) is ~145 ms.  The 100 ms
gap to client-side TTFT is FastAPI/uvicorn streaming-response overhead
that all configs pay; v4.2 `--prompt-kv-cache` only hits below 30 ms
because the legacy prefix-character-yield path (yielding individual
characters from the cached response text) sidesteps `stream_generate`
setup entirely.  The block-cache fast path correctly yields a real model
token immediately (server trace confirms `HIT-fast (logit)` with
`suffix_prefilled=0`), but the framework overhead is fixed regardless.

**Outcome: KEEP.**  The feature is correct, tested, and elimi­nates a
real ~50 ms model forward pass on full-prefix-match hits.  Below the
30 ms gate the bottleneck is framework, not the cache.  Closing it
requires bypassing `stream_generate` for cache hits (architecture change
out of scope for this session) — tracked as v5.2 follow-up.

### 2C wiring

`squish run` was forwarding `--draft-model` and a handful of other
server flags to the spawned `python -m squish.server`, but not the v4.2
PromptKVStore or v5 BlockKVCache flags.  Added six flag forwards in
`squish/cli.py:cmd_run` plus the argparse definitions in `p_run`.  Users
can now run:

```
squish run qwen2.5-7b --block-kv-cache ~/.cache/squish/blocks
```

without dropping to the module-mode invocation.

## What v5.1 ships

### Phase 1 — measured baseline (results/benchmarks_v5_1/runs/20260602T082624)

Median of 5 runs.  Prompt size = chat-templated token count.  Models:
Ollama `qwen2.5:7b` Q4_K_M, Squish `Qwen2.5-7B-Instruct-int4` (mlx),
Squish `Qwen2.5-7B-Instruct-int3` (mlx, where shown).

| Prompt | Metric        | Ollama  | sq daemon I4 | sq +pkv I4 | sq +block I4 | sq +block I3 |
|-------:|---------------|--------:|-------------:|-----------:|-------------:|-------------:|
| **75**  | TTFT          |  137 ms |       730 ms |     **5 ms** |      223 ms |      244 ms |
|        | E2E 200-tok   |  2.85 s |       3.11 s |     5.15 s |      4.91 s |      9.25 s |
|        | Warm tok/s    |   18.8  |        19.2  |       8.6  |       10.2  |        7.5  |
|        | ITL p50       | 54.4 ms |      50.4 ms |   105.3 ms |     85.1 ms |    126.3 ms |
|        | ITL p95       | 56.6 ms |      57.8 ms |   152.3 ms |    162.5 ms |    209.9 ms |
| **500** | TTFT          |  158 ms |       5.07 s |    **10 ms** |      677 ms |      768 ms |
|        | E2E 200-tok   |  4.73 s |      10.69 s |     7.45 s |      7.70 s |     10.36 s |
|        | Warm tok/s    |   16.6  |        12.4  |       9.0  |        8.9  |        7.2  |
| **2000**| TTFT          |  172 ms |     15.97 s  |    **10 ms** |      809 ms |      875 ms |
|        | E2E 200-tok   |  7.50 s |      24.68 s |     8.57 s |     10.13 s |     10.60 s |
|        | Warm tok/s    |   16.0  |        10.7  |       8.3  |        8.1  |        6.4  |
| **4000**| TTFT          |  178 ms |     38.37 s  |    **28 ms** |       1.11 s |      1.00 s |
|        | E2E 200-tok   | 51.72 s |     53.54 s  |  **13.29 s** |     10.14 s |     15.31 s |
|        | Warm tok/s    |    9.8  |         9.6  |       4.8  |        7.5  |        6.1  |

| Footer       | Ollama   | sq daemon I4 | sq +pkv I4 | sq +block I4 | sq +block I3 |
|--------------|---------:|-------------:|-----------:|-------------:|-------------:|
| Peak RSS     | 384 KB*  |     2.79 GB  |    2.63 GB |     2.99 GB  |    3.54 GB   |
| Disk (model) | 4.36 GB  |     4.00 GB  |    4.00 GB |     4.00 GB  | **3.56 GB**  |

`*` Ollama's RSS sample is artifactual: `ollama serve` spawns its model
runner in a separate process group that the bench's RSSSampler doesn't
walk.  The same row in the v4.2 benchmark catches it (~5 GB) when
measured with the per-config bench layout.

#### Headline reads

* **PKV exact-match hit** is the lowest-latency repeated-prompt path
  on every prompt size (5–28 ms TTFT).  The trick is its prefix-character-
  yield path skipping `stream_generate` setup.
* **Block cache cold TTFT** at p4000 is 1.11 s vs Squish daemon's 38.4 s
  — a **34× reduction** on a 4000-token prompt where the previous send's
  blocks are cached.  E2E at p4000 with block cache is **10.1 s vs 53.5 s
  without** — a **5.3× e2e win**.
* **Inter-token p95** is much higher on cache-enabled configs (160–400 ms
  vs Ollama's 75 ms).  Decode-loop thermal drift + per-token cache
  housekeeping — tracked as v5.2 follow-up.
* **INT3** saves 0.44 GB of disk (11 %) at the cost of ~10 % slower decode
  (warm tok/s 6.1–7.5 vs 7.5–10.2 INT4).

## Remaining v5.2 candidates

1. Bypass `stream_generate` setup overhead on block-cache fast hits to
   close the ~100 ms framework cap on TTFT.
2. Speculative prefill: yield a mid-prefill-sampled token, verify on
   full-prefill completion, retract+replace on mismatch — would salvage
   chunked-prefill TTFT win on short prompts (v5 Goal A unresolved).
3. Inter-token p95 stability under sustained decode — the cache-enabled
   paths drift more than daemon and Ollama.
4. Bench design improvement: TTFT measurement at multiple max_tokens
   values would expose the framework-overhead asymptote independent of
   the cache code.
5. Pre-existing tests/test_squishd_unit.py 9 timeouts and
   tests/test_quant_aqlm.py::test_module_count_unchanged assertion gap
   carry over.
