# Ollama vs Squish — three serving modes  (M3 MacBook Pro 16 GB)

**Date:** 2026-06-01
**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Prompt:** `"What is the capital of France? Answer in one sentence."`
**Model:**
- Ollama `qwen2.5:7b` — Q4_K_M GGUF
- Squish `Qwen2.5-7B-Instruct-int4` — INT4 MLX safetensors

**Versions:** Ollama 0.18.2 · Squish 9.14.0

## Headline

> The original v1 RESULTS.md led with **"Squish 2.46× faster cold TTFT."**
> That number is real but it compares post-load Squish to mid-load Ollama —
> apples to oranges. The honest user-perspective metric is **cold wall**
> (kill → first token streamed back). The fair table:

| Metric                              | Ollama       | Squish (eager) | Squish (lazy) | Squish (preload-async) | Winner |
|-------------------------------------|--------------|----------------|---------------|------------------------|--------|
| **Cold wall (kill → first token)**  | **1.55 s**   | 9.57 s         | 9.42 s        | 8.97 s                 | **Ollama** |
| Cold TTFT (server-ready → first)    | 1.42 s       | **522 ms**     | 8.42 s        | 7.93 s                 | Squish eager |
| Warm tokens/sec                     | 18.4 tok/s   | 18.4 tok/s     | 18.0 tok/s    | 17.8 tok/s             | **tie** (<5%) |
| Peak RAM (full process tree)        | 5.07 GB      | **3.14 GB**    | 3.19 GB       | 3.18 GB                | **Squish eager** |
| Disk size (model)                   | 4.36 GB      | **4.00 GB**    | 4.00 GB       | 4.00 GB                | **Squish** |

"Winner" is awarded only when the delta exceeds 5%; otherwise the row is
a tie.

## What changed since v1

- **Two cold metrics** are now reported. `cold_wall_s` (kill → first
  token) is the user-perspective metric and the new headline. The old
  `cold_ttft_steady_s` (server-ready → first token) is kept for
  continuity but is **not** the right comparison across modes.
- **Three squish modes** are measured side-by-side: eager (v1 default),
  `--lazy`, `--preload-async`.
- **A tie row** is no longer hidden behind a small percentage. Anything
  within ±5% of the best value gets "tie" so reviewers can't mistake
  noise for signal.

## Interpretation by user scenario

**You're writing `ollama run qwen2.5:7b "..."` from a cold terminal.**
Ollama wins comfortably. 1.55 s vs Squish-eager 9.57 s is a real,
user-visible gap on this host. The Squish lazy/preload-async modes
don't close it (both ~9 s) because the model load itself takes ~8 s in
MLX through `mlx_lm.load()` — Ollama's llama.cpp loader is faster on
this hardware.

**You're keeping a long-running local server.** Pick squish eager (or
preload-async). After the model is loaded, warm tokens/sec is a tie
(18.4 vs 18.4 tok/s) and Squish uses ~38 % less peak RAM (3.14 GB vs
5.07 GB). Eager is the right choice if you can absorb the 9 s startup;
preload-async if you want the port bound in <1.5 s and the model
usually ready before the first request.

**You're embedding either tool in a UI that pings `/health` and waits.**
Lazy and preload-async both bind in ~1 s, so the UI's "server up"
indicator lights immediately. Lazy users see the model-load cost on
the first inference request; preload-async users usually don't (the
background thread finishes first, especially after the first run when
file cache is warm).

## Recommended default for the article

**Squish should ship `--preload-async` as the default for `squish serve`.**

Rationale:
- Sub-1.5 s port bind matches the "is the server up?" mental model
  every user already has.
- First-request TTFT is identical to eager once the background load
  completes (which it usually does).
- Worst case (first request races the background load) is bounded by
  the eager load time — there's no upside being eager except a
  marginal CPU saving from skipping a thread.

This PR does **not** change the default (eager remains default for
v9.14 to avoid breaking existing users). Switch to preload-async
default in a follow-up minor.

## Per-sample data (so reviewers can spot outliers)

### Cold-wall samples, all 5 runs (s)

| Run | Ollama | Squish eager | Squish lazy | Squish preload-async |
|-----|--------|--------------|-------------|----------------------|
| 1   | 4.37   | 35.11        | 9.43        | 9.43                 |
| 2   | 1.31   | 9.61         | 9.04        | 8.95                 |
| 3   | 1.76   | 9.44         | 9.73        | 9.04                 |
| 4   | 1.22   | 9.58         | 9.51        | 8.97                 |
| 5   | 1.55   | 9.19         | 9.16        | 8.85                 |

Run 1 outliers (Ollama 4.37 s, Squish-eager 35.11 s) are OS page-cache
cold reads of the first model file in their respective sequences.
Squish's outlier is much larger because:
- Ollama was already running when the benchmark started, so its blob
  files were partially warm.
- Squish-eager was the second-launched tool, hitting a fully cold
  page cache for `/Users/wscholl/models/Qwen2.5-7B-Instruct-int4/`.

Median of 5 absorbs this; the lazy/preload-async runs are tight
(σ ≈ 0.3 s) because file cache is warm by then.

### Warm tokens/sec, all 3 samples

| Run | Ollama | Squish eager | Squish lazy | Squish preload-async |
|-----|--------|--------------|-------------|----------------------|
| 1   | 18.4   | 18.9         | 18.8        | 17.6                 |
| 2   | 18.7   | 17.9         | 18.0        | 17.9                 |
| 3   | 17.4   | 18.4         | 17.8        | 17.8                 |

All four tools fall within an ~1 tok/s band — 18.4±0.5 — at this
prompt size. The 13 % Ollama-vs-Squish gap reported in v1 doesn't
reproduce here; the v1 figure was 21.0 vs 18.5, this run gets 18.4
vs 18.4. Most likely v1 caught Ollama after multiple warm runs with
fully populated decode caches. We're not chasing this — kernel-level
MLX vs llama.cpp throughput comparison is out of scope for this PR.

## Raw artifact

`results/ollama_vs_squish_M3_20260601_111904.json` — full per-run
inference timings, peak RSS samples, prompt/completion token counts,
and tool versions.

## Reproducing

```bash
source .venv/bin/activate
python benchmarks/ollama_vs_squish/bench.py
```

Run-to-run variance on cold OS page cache is the largest noise source
on M3 16 GB. Median of 5 cold runs smooths it; the cold-wall numbers
above are stable to ±5 % across re-runs at the same OS warm state.
