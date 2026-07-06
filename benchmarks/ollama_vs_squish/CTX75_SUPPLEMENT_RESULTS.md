# ctx=75 supplement — settling the short-prompt TTFT question

Follow-up to `COLD_UNIQUE_H2H_RESULTS.md`'s 512/1024/2048/4096 sweep. This
adds one more context length, 75 tokens, to directly answer a question that
sweep didn't cover: `bench_thermal_h2h.py` (same fixed 75-token prompt resent
5×) reports Ollama winning TTFT at short prompts (167 ms vs 192 ms) — but
that run measures squish's prefix-KV cache hit on repeats 2-5, not a cold
comparison. This point settles what happens at 75 tokens under the same
cold/unique, 0%-cache-hit-verified methodology as the rest of the sweep.

Script: `bench_cold_unique_h2h.py --ctx75` (new flag, additive only — `--full`
still reproduces exactly the settled 512/1024/2048/4096 sweep, unchanged).

## Result: Qwen2.5-7B-Instruct, squish INT4 vs Ollama 0.30.7

Raw: `results/cold_unique_h2h/20260706T105559/raw.json` (gitignored, not
committed — reproduce with the command below). Same host (Apple M3, 16 GB),
same thermal protocol (82 °C baseline, 8% drift ceiling) as the primary
sweep.

Cache-hit verification: **all 15 requests** (5 runs × ollama / squish /
ollama-recheck) measured **0% cache hit** on both engines' own counters
(Ollama `prompt_eval_count`, squish `/metrics` deltas) — **0 discards, 0
retries needed**. Baseline reached before every pass (54.3 °C / 60.1 °C /
56.7 °C). Drift (ollama decode tok/s, first pass vs. recheck): 16.4 → 16.3,
**-0.46%** (ceiling 8%) — **passed**.

| Context | Ollama TTFT | Squish TTFT | Ollama decode | Squish decode | Ollama E2E | Squish E2E | E2E speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 75 | 812 ms | 800 ms | 16.4 tok/s | 17.5 tok/s | 14.12 s | 12.30 s | **1.15×** |

(Medians of 5 clean, cache-verified runs per cell — same statistic as the
primary sweep.)

## Verdict

**Squish wins TTFT at 75 tokens too** — by a thin margin (12 ms, ~1.5%, the
closest of any length tested so far), but a win, not a loss. The pattern
from 512-4096 holds all the way down to the shortest prompt length measured:
squish leads TTFT, decode tok/s, and E2E at every context length under
cold/unique conditions.

**The old 167 ms/192 ms number does not hold up.** Under genuine cold/unique
conditions both engines' TTFT are roughly 4-5× higher (812 ms / 800 ms, not
167/192 ms) because there is no cache to hit, and the winner flips: Ollama's
reported "win" in `bench_thermal_h2h.py` was measuring its own cache hit on
runs 2-5 of a resent prompt, not a real cold-prefill advantage. That number
is superseded, not merely refined, by this result.

## Reproduce

```bash
cd benchmarks/ollama_vs_squish && ~/squish/.venv/bin/python bench_cold_unique_h2h.py --ctx75
```
