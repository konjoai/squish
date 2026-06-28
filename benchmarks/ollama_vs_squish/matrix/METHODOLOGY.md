# Methodology — reuse × context-length benchmark matrix

This harness rebuilds the Ollama-vs-Squish head-to-head so every number is
labelled for exactly what it measures. It is designed to survive hostile review
with no asterisks. This document is the contract: controls, exact configs,
prompt construction, seeds, hardware, and the cache-posture decisions per
condition.

> **Execution scope.** The harness is authored and unit-tested on any platform,
> but the *numbers* must be produced on Apple Silicon with MLX, a local Ollama,
> and the on-disk models. No results in this repository were produced on the
> Linux CI box — the orchestration modules only run on the bench host.

## The matrix

| Axis | Values |
|------|--------|
| Reuse | 0% (unique), 25%, 50%, 75% shared prefix, 100% (exact repeat) |
| Context | 4k, 8k, 16k, 32k tokens (Qwen2.5 tokenizer) |
| Systems | Squish INT4, Squish INT3 (capability), Ollama Q4_K_M 0.30.7; Ollama 0.18.2 cross-check on the 0% and 50% @ 8k cells |
| Metrics/cell | cold-start (load+first token), cold prefill TTFT, decode tok/s, end-to-end (fixed 200-token gen), peak RSS, KV-cache memory, **measured** cache-hit % for both systems |

## Fairness rules (encoded in `systems.py`)

1. **Like-for-like quant.** The head-to-head speed number is **Squish INT4 vs
   Ollama Q4_K_M** (both ~4-bit; `role="head_to_head"`). Squish INT3 is a
   separate `role="capability"` row, never the head-to-head number, and is
   reported beside the accuracy gate so "faster but dumber" is dead on arrival.
2. **Ollama caching is measured, not assumed.** `cache_probe.ollama_hit_fraction`
   derives Ollama's reuse from the engine's own `prompt_eval_count` in the
   terminal `/api/generate` chunk: `hit = 1 − prompt_eval_count / prompt_tokens`.
   Both systems are configured to the SAME cache posture per condition (see
   below). Where Ollama cannot reuse but Squish can, that is reported as a
   measured *capability* difference, never assumed.
3. **No credit for Ollama unloading.** Ollama runs with `keep_alive=-1` (model
   stays resident) and `num_ctx` sized to `prompt + 200 + 256` rounded to 1024,
   for every length. Exact flags and the binary version are logged per run.
4. **Prefill separated from decode.** Cold prefill TTFT (`max_tokens=1`) and
   decode tok/s (steady-state of the 200-token generation, first gap excluded)
   are measured independently, plus end-to-end. Each win is attributable to a
   phase.
5. **Determinism.** Temperature 0 (greedy), fixed 200-token generation, fixed
   seed (`FIXED_SEED=1234`). Speculative prompt-lookup runs at its DEFAULT (on)
   for the matrix; `squish_int4_nospec` repeats the head-to-head with it OFF so
   its contribution is explicit.

## Cache posture per condition

| Reuse | Posture | Intended hit % | Pass band |
|-------|---------|----------------|-----------|
| 0% | both cold; unique full document each run; no shared prefix | ~0% | ≤ 5% |
| 25/50/75% | both allowed to reuse; fixed shared prefix primed once, then measured | = reuse % | ±max(10pp, 15%·reuse) |
| 100% | both allowed to reuse; identical prompt resent | ~100% | ≥ 95% |

`cell.py` primes the shared prefix once for partial/exact cells (so reuse is
measurable from run 1), then runs the paired set. `cache_probe.classify` FAILS
any cell whose measured hit % falls outside the pass band, and the cell status
becomes `cache_mismatch` rather than reporting a number we cannot explain.

## Prompt construction (`corpus.py`)

- **Real, varied content — no repetition padding.** Prompts are synthesised from
  large lexical pools (subsystems, components, risks, actions, metrics) so every
  sentence is distinct; there is no single-paragraph repetition that would
  artificially inflate prefix-cache hit rate. Real corpus files dropped into
  `corpus_files/*.txt` are used verbatim when present.
- **Reuse is a construction, not an accident.** A reuse-X prompt is
  `[fixed shared prefix sized to X% of target] + [unique tail sized to (100−X)%]`.
  The shared prefix is a fixed realistic context block (system/document
  preamble); the tail varies every run.
- **Exact token lengths.** Text is sliced by encoding to token ids and decoding
  the exact slice; the measured token count is recorded per prompt.
- **Auditable.** Every prompt is saved to `prompts/<cell_id>/run_NNN.txt` with a
  `manifest.json` carrying the seed, token counts, and SHA-256. The base seed is
  `20260628`; each prompt's seed derives deterministically from
  `(base_seed, reuse, ctx, run_index)`.

## Controls

- **Thermal.** Each system is measured from its OWN cold baseline: a 120 s
  cooldown (servers down) then a wait until die temperature ≤ 50 °C
  (`thermal.wait_for_baseline`), with a 25 s settle between phases. Live
  die-temperature is logged (`TemperatureSampler`). A first-vs-last drift probe
  re-measures the first system at the end; drift must be ≤ 1.7%
  (`thermal.drift_check`). If no temperature sensor is available the gate is
  time-based and the writeup must say so.
- **Order effects.** Per-system cold-baseline isolation (the `bench_p4000_iso`
  pattern) removes order entirely; `counterbalanced_order` additionally rotates
  first-position across cells. Never "favoured tool first/cool, other last/hot".
- **Statistics.** ≥30 paired runs per cell (the same prompt set through both
  systems). Each headline ships median, IQR, the full distribution, a paired
  Wilcoxon signed-rank p-value, and Cliff's delta (+ rank-biserial). Implemented
  in pure stdlib (`stats_ext.py`) so it is reproducible without scipy.
- **Memory / OOM.** Peak RSS (whole process tree, 50 ms cadence) and KV-cache
  memory (Squish `/metrics` gauge) are recorded per cell.
  `memory.classify_memory_status` records each system as `fit` /
  `degraded_via_governor` / `oom`; request failures and OOM/governor log markers
  are caught so the harness never crashes — a non-fit is a recorded result.

## Hardware / software to record per run

`systems.System.read_version` logs the Ollama and Squish versions; the runner
records host RAM (`host.detect_ram_bytes`), the model paths/quant, the exact
flags, the seed, the prompt token counts, and the per-cell temperatures. Fill in
the bench host's chip / RAM / macOS / MLX / PyTorch versions in the result
header before publishing.

## Reproduce

```bash
# 1) kill-test (one cell, then STOP for review)
python -m benchmarks.ollama_vs_squish.matrix.run_killtest
# 2) after human approval only:
python -m benchmarks.ollama_vs_squish.matrix.run_matrix --i-have-approved
```

Raw per-run JSON, saved prompts, seeds, and the commit hash make every cell
reproducible from one command.

## Non-goals (explicit scope)

- Multi-user / concurrent serving (single-developer scope).
- Non-Apple-Silicon platforms.
- Models beyond the tested set, unless the second-model (3B) rows are included;
  if scoped to 7B only, that is a stated limitation.
