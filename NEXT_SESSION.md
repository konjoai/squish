# NEXT_SESSION

## KV-cache restore dtype fix (v9.33.9) — follow-ups

The fp16→bf16 KV-restore fix landed and is verified end-to-end (see CHANGELOG
9.33.9, gates G1–G3 / V1–V5). Outstanding items:

### 1. Optional step-boundary padding (sprint step 5) — STILL PENDING
Even with matched bf16 dtype, a restored buffer still forces **one** realloc on
the first decode step:
- **Block path**: restore yields `offset == capacity` → realloc every time
  (now a same-dtype bf16 concat, so no fp32 promotion — but still one alloc +
  copy of the whole cache on token 0).
- **Prompt-kv path**: the server already saves the *full padded* buffer
  (`cap` = next multiple of 256), so there is usually headroom and no
  first-step realloc — padding would only help block restore.

Optional optimization: pad the restored block buffer up to the next
`KVCache.step` (256) boundary so the first decode step writes in place. This is
**not** the 2× cause (that was the dtype, now fixed) — it is a smaller
first-token-latency item. Measure first-token latency on the block path before
deciding it's worth the extra code.

### 2. V4 — 30-run paired Wilcoxon (house standard) — NOT RUN
The effect is unambiguous (unit: 1.38×→0.95×; E2E: 0.5×→1.15×), so the paired
stat test was skipped to save M3 time. If house discipline requires it for the
record, run `bench_v5_1.py` 30× paired E2E-200 @p2000 for `+pkv` before-vs-after
and report the Wilcoxon p-value.

### 3. Benchmark article / version numbers — NEEDS REFRESH
The published benchmark numbers (and any article) were taken with the
regression present (pkv/block ~0.5× daemon decode). They now under-report
squish: with the fix, `+pkv` and `recommended` decode at parity-or-better with
daemon **and** keep the single-digit-ms TTFT advantage (measured 11.5 ms vs
daemon 1170 ms). Re-run `benchmarks/ollama_vs_squish/bench_v5_1.py` for all
configs and refresh the article / README tok/s tables.

> Note: absolute tok/s on the current M3 16 GB measured lower than the sprint's
> reference table (daemon ~10 vs ~19.6) — likely memory pressure / thermal at
> measurement time. Re-run on a clean machine state before publishing numbers;
> the **ratios** (pkv ≈ daemon, both ≫ pre-fix) are the durable result.
