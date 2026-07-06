# Results — cold/unique head-to-head (0% reuse)

Companion to `COLD_UNIQUE_H2H_METHODOLOGY.md`. Produced by
`bench_cold_unique_h2h.py` on Apple M3, 16 GB unified memory.

Raw per-run JSON lives under `results/cold_unique_h2h/<UTC>/` (gitignored —
not committed; reproduce locally with the commands in the methodology doc).

## Primary result: Qwen2.5-7B-Instruct, INT4 vs Q4_K_M

`results/cold_unique_h2h/20260703T194847/raw.json` — full `--full` sweep,
512/1024/2048/4096 tokens. Ollama 0.30.7, squish (INT4). Thermal: baseline
82 °C reached before every one of the 3 passes (ollama / squish /
ollama-recheck), drift **+1.99%** (ceiling 8%) — **passed**. Cache-hit
verification: **all 45 requests** (4 contexts × 2 systems × 5 runs, + a
5-run ollama drift-recheck) measured **0% cache hit** on both engines, **0
discards, 0 retries needed**.

| Context | Ollama TTFT | Squish TTFT | Ollama decode | Squish decode | Ollama E2E | Squish E2E | E2E speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512  | 3.90 s  | 3.33 s  | 16.7 tok/s | 19.1 tok/s | 16.27 s | 13.74 s | **1.18×** |
| 1024 | 10.01 s | 8.59 s  | 10.8 tok/s | 14.8 tok/s | 28.93 s | 21.98 s | **1.32×** |
| 2048 | 19.79 s | 17.31 s | 10.5 tok/s | 14.2 tok/s | 38.84 s | 32.24 s | **1.20×** |
| 4096 | 41.50 s | 36.14 s | 9.4 tok/s  | 11.9 tok/s | 62.80 s | 52.93 s | **1.19×** |

(All values are medians of 5 clean — cache-verified, cache-hit ≤5% — runs
per cell.) **Squish wins TTFT, decode tok/s, and end-to-end response time at
every context length, with nothing to reuse on either side.** This is a
reversal from the article's other TTFT row ("Ollama wins TTFT at every
prompt size" — that number is measured on an *already-loaded, warm* model
answering a single fixed prompt; this number is measured on a cold, unique
prompt every time, which is the more representative case for real usage
where consecutive prompts rarely repeat verbatim).

One caveat worth carrying forward: the squish/4096 TTFT cell was the
noisiest in the run (stddev ≈ 5.8s, range 25.4–43.6s across the 5 clean
runs) — still every run passed the 0%-cache-hit check independently, so this
is ordinary prefill-time variance at long context, not a methodology
concern, but treat the 4096 row as the least precise of the four.

## Floor vs. ceiling — how this differs from the existing 9.8× number

`BENCHMARKS.md`'s headline "E2E @ 4000-token prompt: 37.5s vs 3.8s (9.8×)"
sends the **same prompt 5×** within its measurement loop
(`bench_thermal_h2h.py`'s `run_one_phase`). Since squish's in-memory
prompt-prefix KV reuse is wired into the default prefill path
(`squish/server.py`, `_prefix_reuse_state`, default-on, no flag required),
repeats 2-5 of an identical prompt are exact-match cache hits on squish's
side. **9.8× is a reuse ceiling** — the best case, when everything an
application sends repeats a recent prompt (multi-turn chat, agent loops,
RAG over a stable context). **The ~1.18-1.32× number above is the floor** —
every request unique, nothing to reuse on either side, verified request by
request. Real workloads land somewhere between these two numbers depending
on how much prompt content repeats; the honest range to quote is
**"1.2-1.3× cold, up to 9.8× when prompts repeat,"** not a single number.

## Supporting single-context kill-test data (other model families)

Same harness, `BENCH_OLLAMA_MODEL`/`BENCH_SQUISH_INT4` env overrides, ctx=1024
only (kill-test scope, not a full 4-length sweep). Included as directional
support, not as rigorously as the primary Qwen2.5-7B result above.

| Model | Ollama TTFT | Squish TTFT | Ollama decode | Squish decode | Ollama E2E | Squish E2E | E2E speedup | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gemma-3-4B (`gemma3:4b`)        | 3.87 s | 3.10 s | 28.8 tok/s | 38.7 tok/s | 10.96 s | 8.34 s | **1.31×** | -0.94% (ceiling 8%) OK |
| Llama-3.2-3B (`llama3.2:3b`)    | 3.03 s | 2.74 s | 36.2 tok/s | 41.0 tok/s | 8.61 s  | 7.62 s | **1.13×** | -0.89% (ceiling 8%) OK |

Both passed cache-hit verification cleanly (0 discards) and their drift
checks passed comfortably inside the 8% ceiling.

**Not included — did not pass, or unverified:**
- **Qwen3-8B** (`qwen3:8b`) ran cleanly on cache-hit verification but its
  drift check **failed** (+12.4%, ceiling 8%) — not reported as a result.
  (Getting this far needed two harness fixes: the native Ollama.app
  port-11434 re-fight, and `_parse_ollama_line`'s reasoning-model streaming
  gap — see `COLD_UNIQUE_H2H_METHODOLOGY.md`.) Worth a clean re-run later;
  not blocking this PR.
- Two additional result directories exist on the bench host
  (`20260704T095157`, `20260704T153057`) that are not referenced in this
  sprint's own working notes and were not run against a checked-in config —
  one fails its drift check badly (-56.8%), the other has unusually high
  intra-cell variance despite a passing drift check. Excluded pending a
  confirmed re-run; not part of this PR's claims.

## Reproduce

```bash
cd benchmarks/ollama_vs_squish && ~/squish/.venv/bin/python bench_cold_unique_h2h.py            # kill-test, ctx=1024
cd benchmarks/ollama_vs_squish && ~/squish/.venv/bin/python bench_cold_unique_h2h.py --full --i-have-approved   # full sweep
```

Override model/engine via env for other families (no code changes needed):
`BENCH_OLLAMA_MODEL=<ollama tag> BENCH_SQUISH_INT4=<path to squish INT4 dir>`.
