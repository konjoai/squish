# Cold-load phase analysis — what's reducible

Based on the baseline profile in `PROFILE_REPORT.md`, each phase is
classified as **Fundamental** (cannot be reduced without changing upstream
MLX, the tokenizer library, or the model format), **Avoidable** (work being
done that shouldn't be), **Cacheable** (one-time at install), or
**Parallelizable** (could overlap with another phase).

## Classification

| Phase             | Warm Δ (s) | Class              | Reasoning |
|-------------------|-----------:|--------------------|-----------|
| squish_imports    |      0.36  | Parallelizable     | Could overlap with mlx_lm import in a background thread. Savings ≤ 0.36 s. |
| mlx_core_import   |      0.03  | Fundamental        | `mlx.core` is the smallest dependency — Metal context init. Cannot remove. |
| **mlx_lm_import** |    **3.84** | **Avoidable (mostly)** | `mlx_lm/__init__.py` eagerly imports `from .convert import convert` which transitively loads `transformers → sklearn → torch`. Squish never calls `convert` at runtime. Stub `sys.modules["mlx_lm.convert"]` before importing mlx_lm and the chain collapses. Estimated savings: ~3 s. The remaining ~0.8 s is `mlx_lm.generate` + `mlx_lm.utils` + the necessary tokenizer setup, which IS used at runtime. |
| weights_loaded    |      1.32  | Fundamental        | Real file I/O of ~4 GB INT4 weights + Metal eval. The MLX safetensors path already mmaps; we're bottlenecked on SSD read + Metal layout. Not reducible without changing the weight format. |
| tokenizer_loaded  |      0.63  | Parallelizable     | Independent of weight load; both can run concurrently. The HF tokenizer loader does pure Python work (parse merge tables, build trie) while weights stream from disk. Savings ≤ 0.63 s (in practice the smaller of the two phases). |
| warmup_done       |      1.15  | Avoidable (tradeoff) | Pre-compiles Metal JIT kernels via a 1-token forward pass. If we skip warmup at startup, first-request TTFT grows by ~1 s (the JIT compile happens on demand). Net savings on the *cold-wall* metric depends on whether the user's first request happens immediately or later. For the Ollama benchmark (first request hits a second after server-ready), it's roughly a wash. Skipping warmup is the right call for lazy/preload-async mode (already done); for eager mode it's a tradeoff. |
| server_bound      |      0.13  | Fundamental        | uvicorn binds the socket and starts its event loop. Tiny. |

## Targets for Phase 3 (ordered by expected impact)

1. **Stub `mlx_lm.convert`** — Avoidable. Expected savings: **~3 s** of the
   4 s `mlx_lm_import` phase. The fix is purely import-time: before
   `import mlx_lm` runs, we register a stub `mlx_lm.convert` in `sys.modules`
   so `mlx_lm/__init__.py`'s `from .convert import convert` finds the stub
   and skips the real loader. squish doesn't call `convert`, and any
   external caller (e.g. `python -m mlx_lm.convert`) bypasses
   `mlx_lm/__init__.py` entirely.

2. **Parallel tokenizer + weight load** — Parallelizable. Expected
   savings: **~0.6 s**. Spawn a thread for `load_tokenizer()` immediately
   after configs are read; weights load on the main thread. Join the
   thread before returning. Outputs are independent — bit-for-bit
   identical to the serial version.

3. **Skip eager-mode startup warmup; defer to first request** — Avoidable
   (with tradeoff). Expected savings on cold-wall: **~0–1 s** (depends on
   first-request timing). Move `_warmup_model()` into a background thread
   that runs in parallel with uvicorn binding. First-request latency
   unchanged in the common case (warmup completes before request arrives);
   worst case is no worse than the current --lazy mode.

4. **Parallel squish_imports with mlx_lm_import** — Parallelizable.
   Expected savings: **~0.3 s**. Spawn an import thread for the heavy
   `mlx_lm` early in `cmd_run` / before banner printing. Squish's own
   package imports + argparse run on the main thread in parallel.

5. *(Optional)* **Cache tokenizer as pickle** — Cacheable. Expected
   savings: ~0.4 s of the 0.6 s tokenizer load. The HF tokenizer is
   deterministic given a model dir; pickling the wrapped tokenizer once
   at install and reloading via `pickle.load` skips the merge-table /
   trie reconstruction. Requires a cache-invalidation hook on tokenizer
   file mtime. Not implemented in this PR — listed for completeness.

## Hard floor

Even after every fix that doesn't have a quality tradeoff:

  Fundamental floor ≈ 0.03 + 1.32 + 0.13 = **1.48 s**
                      mlx_core + weights + uvicorn

Plus whatever squish-side imports we can't parallelize away (~0.1 s
worth that aren't already covered by the mlx_lm-import overlap).

So the realistic best case is **~1.6–2.0 s cold load**, plus first-token
inference latency (~0.5 s) for a cold wall of **~2.1–2.5 s**.

That gets us competitive with Ollama's 1.55 s cold wall on this
hardware — Ollama's daemon binds the port and lazy-loads the model via
llama.cpp's mmap-on-demand path, which on M3 hits unified memory faster
than `mlx_lm.load()` + Metal eval. We won't beat Ollama on cold wall
without either (a) matching their lazy-load architecture (which we
already offer via --preload-async) or (b) a faster MLX weights path
upstream.

## What we will NOT fix in this PR

* `weights_loaded` (1.4 s) — fundamental, fixing requires changing
  the safetensors file format or shipping a squish-native format
  that mmaps directly with no Metal eval. Out of scope.
* The 0.8 s of mlx_lm imports that remain after stubbing convert —
  also fundamental for runtime (we genuinely need `generate` and
  `utils.load`).
* HF tokenizer pickle cache — Cacheable but adds an invalidation
  story and a cache directory. Saved for follow-up.

## Article-ready summary (for after Phase 5)

Before: 7.45 s warm / 8.0 s cold load (median of 5)
After:  X s
Improvement: Y % reduction
Headline: mlx_lm.convert eager import accounts for ~3 s of that — stub
it out and the rest is fundamental file-I/O floor.
