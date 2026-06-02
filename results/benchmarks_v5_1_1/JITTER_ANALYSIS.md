# v5.1.1 — Inter-token p95 jitter analysis

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-02
**Source:** `results/benchmarks_v5_1_1/runs/20260602T095112/raw.json`
**Method:** per-token inter-token gaps from the median-e2e run's
`chunk_timestamps` (read-only; `benchmarks/ollama_vs_squish/analyze_jitter_v5_1_1.py`).
No new inference was run for this analysis. **Investigation only — no fix
attempted (out of scope per the session brief).**

## What v5.1 claimed

The v5.1 PRECHECK flagged: "Inter-token p95 is much higher on cache-enabled
configs (160–400 ms vs Ollama's 75 ms). Decode-loop thermal drift + per-token
cache housekeeping." This analysis tests those two attributed causes —
**thermal drift** and **periodic per-token housekeeping** — against the
measured per-token series.

## Measured per-token gaps at p2000 (median-e2e run, decode-only)

| Config              | p50 ms | p95 ms | p99/max ms | drift (Q4−Q1) | spikes >2×p50 |
|---------------------|-------:|-------:|-----------:|--------------:|--------------:|
| ollama              |  110.9 |  160.0 |   632.9    |  **−27.9**    | 4 |
| squish_daemon       |   90.5 |  351.2 |   369.5    |  **+46.9**    | 7 |
| squish_pkv          |  131.5 |  210.9 |   250.6    |  **−26.0**    | 0 |
| squish_block        |  134.6 |  246.2 |   480.8    |  **−11.2**    | 3 |
| squish_recommended  |  116.0 |  243.9 |   522.5    |  **−2.3**     | 5 |

(drift = mean of last decode quartile − mean of first decode quartile;
positive = gaps growing over the run.)

## Finding 1 — It is NOT thermal drift

Thermal drift would show gaps **increasing monotonically** across the decode
(positive Q4−Q1). The measured drift on every **cache-enabled** path is flat
or **negative**: pkv −26.0, block −11.2, recommended −2.3 ms. Gaps actually
*shrink* slightly as the run proceeds.

The only config with strong positive drift is `squish_daemon` (+46.9 ms) — and
that is the **uncached** full-prefill path, where each decode step attends over
the full 2000-token context with no cache reuse; that is a context-length cost,
not a cache artifact. So thermal drift does not explain the cache-path p95.

At p4000 the same holds: `squish_recommended` Q1 mean 184.8 ms → Q4 mean
138.0 ms (drift −46.8). The cost is **front-loaded**, not accumulating.

## Finding 2 — It is NOT periodic block-boundary housekeeping

If the cause were cache housekeeping every N tokens (e.g. an eviction or a
re-hash at each 64-token block boundary), spikes would land at regular,
64-aligned token indices. They do not. The spikes cluster in the **first
~10–30 decode tokens** and are not 64-aligned:

```
recommended p2000 spikes:  token#16(446) #17(523) #18(372) #19(237) #40(244)
recommended p4000 spikes:  token#11(418) #12(482) #13(367) #18(317) #28(545)
block       p4000 spikes:  token#17(340) ...
```

None fall on a decode-time block boundary. The pattern is a **one-time early
settling burst**, not a periodic signal.

## Finding 3 — The tail is common to all engines, including Ollama

Ollama's own p99/max at p2000 is **632.9 / 686.5 ms** — *higher* than every
squish config's tail. Ollama wins the **p95** (160 ms, its decode is tighter
in the body) but it suffers the same kind of sporadic single-token stall. This
points to a runtime/OS-level cause shared by both engines (Metal kernel
dispatch, GPU scheduling, unified-memory pressure on a 16 GB machine), not a
squish-cache-specific defect.

## Most likely cause (front-loaded early-decode stalls)

The cache-enabled squish paths concentrate their worst gaps in the first
handful of decode tokens. The plausible contributors, in order:

1. **v5.1 deferred KV restore lands on an early decode token.** v5.1 moved the
   ~200 ms `mx.concatenate` KV-restore *off* the TTFT critical path and runs it
   after the first chunk yields. That work has to happen somewhere — it now
   surfaces as an early-decode inter-token spike. This is a **TTFT↔ITL
   trade**, not free latency. The combined config restores both a block-cache
   prefix and (potentially) a PKV state, which is consistent with its early
   cluster (tokens 11–19) being the densest of any config.
2. **Metal kernel JIT/warmup** for the decode-shaped kernels on the first few
   steps, before the GPU command stream is hot.
3. **OS scheduling / unified-memory paging** on a 16 GB machine — the same
   effect that gives Ollama a 632 ms p99.

Steady-state (last-quartile) inter-token latency is well-behaved on every
cache path (≈ 120–140 ms), so there is no sustained runaway.

## Cause classification & decision

**Is it fixable with a config change** (bigger block size, deferred evictions)?
**No.** The spikes are not eviction-driven and not block-boundary-aligned, so
`--block-kv-size` / hot/cold-GB tuning would not move them. Block size is
already 64; raising it reduces the number of partial-block prefills (a TTFT
concern) but does nothing for the early-decode ITL burst.

**Is it architectural?** **Yes.** The dominant lever is the v5.1 deferred-restore
design: it correctly protects TTFT but parks the restore on an early decode
token. Smoothing it (chunked/async restore spread across several tokens, or a
restore that overlaps GPU compute) is a core change — **out of scope for this
session**. Tracked as a v5.2 follow-up (this matches v5.1's existing candidate
#3, now with a concrete mechanism rather than the "thermal drift" guess).

**No quick config fix exists, so per the session scope this is documented and
skipped — not fixed here.** The headline numbers in `RESULTS.md` report the
measured ITL p95 honestly rather than papering over it.
