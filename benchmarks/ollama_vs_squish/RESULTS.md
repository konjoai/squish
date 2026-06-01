# Ollama vs Squish — M3 MacBook Pro 16 GB

**Date:** 2026-06-01
**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Prompt:** `"What is the capital of France? Answer in one sentence."`
**Model:**
- Ollama `qwen2.5:7b` — Q4_K_M GGUF
- Squish `Qwen2.5-7B-Instruct-int4` — INT4 MLX safetensors

**Versions:** Ollama 0.18.2 · Squish 9.14.0

## Headline

| Metric              | Ollama          | Squish          | Delta   |
|---------------------|-----------------|-----------------|---------|
| Cold TTFT (median)  | 1.26 s          | 511 ms          | 2.46× faster (Squish) |
| Cold peak RAM       | 5.25 GB         | 3.51 GB         | 1.50× lower (Squish)  |
| Warm tokens/sec     | 21.0 tok/s      | 18.5 tok/s      | 1.13× faster (Ollama) |
| Cold total wall     | 1.94 s          | 9.34 s          | 4.82× faster (Ollama) |
| Disk size (model)   | 4.36 GB         | 4.00 GB         | 1.09× smaller (Squish)|

## What "cold" means here

- **Cold phase, 5 runs, median reported:** kill all `ollama` / `squish` processes, wait 3 s, spawn the
  server, send a streaming request, kill the server.
- **Warm phase, 3 runs, median reported:** start the server once, prime it with one throwaway
  request, then issue 3 more identical requests against the loaded model.
- macOS does not expose page-cache flushing, so "cold" here means "process restarted" — file-cache
  warmth still varies between runs. We mitigate by taking the median of 5 cold runs.
- Peak RSS is sampled at 50 ms intervals across the **entire process tree** (server + any spawned
  runner child) so Ollama's `ollama runner` subprocess is included.

## What the numbers say

The two tools make opposite startup tradeoffs and that shows up in the headline:

- **Squish loads the model at server startup.** Once the server binds the port, the next request
  is already warm — so cold-phase TTFT is ~511 ms (essentially just prefill + 1 token). The
  price is a ~8.5 s server-ready time; the model has to be resident before the port is open.
- **Ollama defers model load until the first request.** `ollama serve` binds in ~280 ms, but the
  first request pays the load cost — TTFT median is 1.26 s. Once loaded the model stays resident
  for subsequent requests.

If you measure "kill → first answer," Ollama wins by ~5× because it doesn't preload. If you
measure "server-ready → first answer," Squish wins by ~2.5× because the model is already in
memory. Both stories are true; pick the one that matches your use case.

In steady state (warm tokens/sec), Ollama edges out Squish 21.0 vs 18.5 tok/s on this single
prompt — within ~13%, both well within usable range on an M3 16 GB. Peak RAM is the more
striking difference: Squish's full process tree peaks at 3.5 GB vs Ollama's 5.25 GB, a 1.5×
gap on the same 7B model at the same quant level.

## Cold-run TTFT, all 5 samples (s)

| Run | Ollama | Squish |
|-----|--------|--------|
| 1   | 4.52   | 0.538  |
| 2   | 1.39   | 0.496  |
| 3   | 1.22   | 0.511  |
| 4   | 1.26   | 0.517  |
| 5   | 1.25   | 0.503  |

Squish's TTFT is tight (std ~16 ms) because the model is preloaded — there's no first-request
cliff. Ollama's run 1 outlier (4.5 s) is the model load from cold OS page cache; runs 2–5
hit warm cache and converge around 1.25 s.

## Warm-run tokens/sec, all 3 samples

| Run | Ollama  | Squish  |
|-----|---------|---------|
| 1   | 21.05   | 18.52   |
| 2   | 20.88   | 18.86   |
| 3   | 21.00   | 18.54   |

## Raw artifact

`results/ollama_vs_squish_M3_20260601_104831.json` — full per-run inference timings, peak RSS
samples, prompt / completion token counts, and tool versions.

## Reproducing

```bash
source .venv/bin/activate
python benchmarks/ollama_vs_squish/bench.py
```

Run-to-run variance on cold OS page cache is the largest noise source on M3 16 GB. For a
single-prompt comparison this is acceptable; for larger studies, drop the cache and pin
frequency before each run.
