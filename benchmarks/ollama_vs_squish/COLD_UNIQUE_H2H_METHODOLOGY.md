# Methodology — cold/unique head-to-head (0% reuse)

Companion to `bench_cold_unique_h2h.py`. Fills the one point missing from the
article: Squish INT4 vs Ollama Q4_K_M when neither system has anything to
reuse — the floor, as opposed to the 9.8× number's ceiling.

## Why this exists

BENCHMARKS.md's headline "E2E @ 4000-token prompt: 37.5s vs 3.8s (9.8×)" comes
from `bench_thermal_h2h.py`'s `run_one_phase`, which sends the **same** prompt
5 times for the TTFT loop and 5 times for the E2E loop. As of the in-memory
prompt-prefix KV reuse wired into `squish/server.py`'s standard prefill path
(default-on, no flag required), repeat sends of an identical prompt within a
phase are exact-match cache hits on Squish's side from the second send onward.
That number is real, but it measures a **reuse ceiling** — the best case, when
everything repeats — not cold inference. This sprint measures the opposite
end: every request, on both systems, is a prompt nothing has seen before.

## What is reused, unchanged, from the matrix harness

This script is a thin, single-purpose combination of three already-trusted
`benchmarks/ollama_vs_squish/matrix/` modules, plus the thermal-control
pattern from `bench_thermal_h2h.py` (cooldown → baseline gate → settle →
drift-check-at-the-end), not the heavier 30-run matrix orchestration
(`cell.py` / `run_matrix.py` — not built or run for this sprint):

- **`thermal.py`** — baseline gate (`wait_for_baseline`), cooldown (`cooldown`),
  live die-temperature logging (`TemperatureSampler`), and first-vs-last drift
  check (`drift_check`). The gate/cooldown/drift *mechanisms* are untouched;
  `read_die_temp_c`'s sensor probing gained a real fix this session — see
  "Thermal recalibration on this machine" below — and this script overrides
  the target/ceiling constants locally (not in `thermal.py`) for the same
  reason.
- **`corpus.py`** — `Corpus.build_prompt(reuse=0.0, ctx_tokens, run_index)`
  generates a genuinely unique full document per run (no fixed preamble, no
  repetition padding — varied technical prose from large lexical pools).
  Untouched.
- **`cache_probe.py`** — `ollama_hit_fraction`, `squish_hit_fraction`, and
  `classify` for measured (not assumed) per-run cache-hit verification.
  Untouched functions, but called differently — see below.
- **`systems.py`** — `stream_ollama` / `stream_squish` (capture
  `done_chunk` / `usage` for cache probing), `num_ctx_for` (prompt + 200 gen +
  256 headroom, rounded to 1024), `squish_metrics`, `start_ollama`,
  `kill_all_serving`, `wait_ready`, `stop_server`. Untouched. Squish itself is
  **not** started via `systems.py`'s launcher (which enables
  `--block-kv-cache` + `--prompt-kv-cache`, the "recommended" production
  config used by the reuse-curve sweep) — this sprint starts the plain
  default daemon (`start_squish_plain`, no cache flags) per the brief's
  "shipped default quant" instruction. Nothing on disk to reuse from either
  way; only the in-memory, process-lifetime prefix slot is live, and it never
  fires on a prompt it has never seen.

## Deliberate deviations, and why

**1. One request per run, not five.** `bench_thermal_h2h.py`'s
`run_one_phase` sends one prompt 5× for TTFT (`max_tokens=1`) and 5× for E2E
(`max_tokens=200`) — the repeat-prompt pattern this sprint exists to
eliminate. Here every one of the 5 runs per cell is a single
`max_tokens=200` request on its own unique prompt; TTFT, decode tok/s, and
E2E are all read from that one request's stream.

**2. The cache-probe timing-ratio fallback is disabled.** `cache_probe`'s
`ollama_hit_fraction` and `squish_hit_fraction` both accept optional
`prefill_cold_s` / `prefill_warm_s` arguments that, when supplied, estimate a
reuse fraction from prefill-time collapse (`1 - warm/cold`) whenever the
engine's own counter is inconclusive. This script never passes them.

That fallback exists to *quantify* partial reuse (25/50/75%) when a counter
can't. At 0% intent it is pure noise: ordinary run-to-run prefill jitter
(Metal kernel warmup, thermal micro-drift) routinely exceeds the 5% mismatch
band on its own, with no caching involved. This is not a hypothetical —
it already happened: the full matrix's own `r000_c4000` cell (0% reuse,
4000-token context) was marked `cache_mismatch` on **30/30 runs on both
systems** (`results/benchmark_matrix/matrix/20260629T021221/r000_c4000.json`).
Inspecting that file: Squish's runs used method
`squish:prefill_ratio(no_counter)` with measured "hits" of 13–21% on every
single run, and no `/metrics` counter ever incremented — i.e. no reuse event
occurred; the ratio-vs-single-cold-reference comparison was just noisy. This
script instead trusts only each engine's direct, authoritative signal:

- **Ollama** — `1 - prompt_eval_count / prompt_tokens` from the terminal
  `/api/generate` chunk. Direct token accounting, not inferred from timing.
- **Squish** — `usage.prompt_tokens_details.cached_tokens` if the server
  emits it, else a hard check on `/metrics` counter deltas
  (`squish_prefix_cache_hits_total`, `squish_radix_prefix_hits_total`)
  around the single request. Verified against `squish/server.py`: the
  in-memory prefix-reuse path increments `_prefix_cache.prefix_hits` (→
  `squish_radix_prefix_hits_total`) whenever it restores a shared prefix, so
  a zero delta is a hard guarantee that path didn't fire, not an absence of
  evidence.

**3. Any nonzero measured hit discards and retries the run.** Per
`cache_probe.classify`, a 0%-intent run passes only if measured ≤ 5%.
A run that fails is discarded (logged, not dropped silently) and retried
with a fresh, never-before-used prompt (own seed slot, never reused even on
retry), up to 3 retries. If all retries still show a hit, the script raises
rather than keeping a contaminated run — matching the brief's "don't let a
stale cache quietly turn a unique run into a partial-reuse run."

**4. Disjoint seed blocks per system pass.** `Corpus`'s prompt seed derives
from `(base_seed, reuse, ctx_tokens, run_index)` — it has no notion of which
system or which pass (first-measurement vs. drift recheck) is asking. Left
alone, the ollama-first pass and the ollama-recheck pass at the same context
length would draw the *same* `PromptSpec`s (harmless for caching — they're
different process instances with empty caches — but it would violate the
brief's literal "nothing shares a prefix with anything else in this sprint").
This script assigns each pass (`ollama` / `squish` / `ollama_recheck`) a
disjoint block of the seed space (`SEED_BLOCK`, spaced 100,000 apart, with a
further `ctx_tokens * 10` sub-offset so different context lengths within one
pass never collide either) so no two requests anywhere in the sprint, on
either system, at any context length, in the first pass or the recheck, ever
draw the same prompt. A base seed (`20260703`) distinct from the matrix
sweep's `20260628` additionally rules out any cross-sprint collision.
Verified offline: 180 generated `PromptSpec`s across all (pass, context, run,
retry-slot) combinations, zero SHA-256 collisions.

Squish and Ollama at the *same* context length within the *same* pass
(ollama's run 0 at 512 tokens vs. squish's run 0 at 512 tokens) intentionally
share `run_index` — they draw from different `SEED_BLOCK`s so the text
differs, but this is not required for correctness (they are separate server
processes with no shared cache) and was preserved only because per-pass
blocking already makes every prompt in the run unique regardless.

## Thermal recalibration on this machine

The matrix harness's thermal defaults (50 °C baseline, 1.7% drift ceiling)
turned out not to fit this specific machine, discovered by running the
kill-test repeatedly and reading the results honestly rather than assuming
the defaults were right:

- **`osx-cpu-temp`/`istats` are non-functional on Apple Silicon.** They read
  Intel-only SMC keys and return a fixed `0.0 °C` rather than failing —
  `thermal.wait_for_baseline` would then pass *instantly* every time,
  silently turning the 50 °C gate into a no-op. Fixed in `matrix/thermal.py`
  (shared module, benefits every harness that imports it): `read_die_temp_c`
  now tries `macmon pipe -s 1` first (sudoless Apple Silicon sensor CLI,
  https://github.com/vladkens/macmon — the same tool `bench_thermal_h2h.py`
  validated cooldowns against previously, per commit `b9b5d8e`, but only as
  an offline sidecar trace, never wired as a live gate before now), and a new
  `_plausible_die_temp` helper rejects any reading outside 5–110 °C from any
  probe so a bogus `0.0` can never be mistaken for "baseline reached."
- **50 °C is unreachable on this machine at any cooldown length.** Measured
  true idle (nothing running, ~0.3% CPU, ~13W system power) five times over
  ~20s with the real sensor: 79.8/80.3/80.8/81.1/81.5 °C — a tight,
  reproducible cluster, not residual heat from a prior pass. `BASELINE_TARGET_C`
  is overridden to **82 °C** locally in `bench_cold_unique_h2h.py` (not in
  `thermal.py` — this is a property of this machine, not a fact about the
  method).
- **The 1.7% drift ceiling traces to one specific historical run** (commit
  `b9b5d8e`'s observed value), not a designed statistical bound. With the real
  sensor confirming the baseline gate was genuinely reached before every pass
  (not skipped or timed out), two kill-test runs on this machine measured
  +3.87% and -5.65% ollama decode-tps drift — normal run-to-run variance here
  (peak inference temp was ~102-103 °C in every pass regardless of starting
  point), not a thermal-control failure. `DRIFT_CEILING_PCT` is overridden to
  **8%** locally, for the same reason as the baseline target.

Any future run of this harness on different hardware should re-derive these
three numbers rather than trust 82 °C / 8% as universal — they are this
machine's measured floor and noise band, not a spec.

## Other bugs found and fixed while running this sprint

- **Orphaned `squishd` daemon (`ai.konjo.squishd` LaunchAgent, `KeepAlive=true`)
  contended for the GPU/model slot** during the very first kill-test attempt,
  auto-relaunching every time `kill_all_serving()`'s `pkill` killed it — this
  (plus the sensor no-op above) is the likely cause of that run's ~1.5 hour
  stall and its uniformly 2-4× slower-than-expected numbers on *both* systems.
  Traced to a test leak (`tests/test_launchagent_coverage.py`, not mocking a
  `subprocess.run` call) — fixed as a separate, unrelated change, not part of
  this sprint's diff. Always check `launchctl list | grep -i squish` (not just
  `ps aux`) for orphaned agents before trusting a run.
- **Native Ollama.app auto-relaunches `ollama serve` after a `pkill`,
  re-fighting port 11434** for the next `start_ollama()` call within the same
  ~4-minute cooldown+baseline window. A clean quit
  (`osascript -e 'tell application "Ollama" to quit'`) stayed down through a
  full run; a `pkill` did not. Not fully root-caused beyond "SIGTERM reads to
  the app as a crash, triggering auto-relaunch, where a clean quit doesn't."
- **`matrix/systems.py`'s `_parse_ollama_line` only read the `"response"`
  field**, dropping every token for reasoning models (Qwen3 streams
  chain-of-thought via a separate `"thinking"` field with `"response": ""`
  for the whole reasoning phase). With a fixed generation budget, a reasoning
  model can spend it entirely on `"thinking"`, so `t_first` never got set and
  `ttft_s` came back `None` — this is a shared-module fix (any future harness
  reusing `stream_ollama` for a reasoning model benefits), not scoped to this
  script. Squish's own OpenAI-compatible stream does not have this problem —
  it already unifies `<think>...</think>` into normal `delta.content` chunks.

## Cache-clearing between runs

- **Squish**: `thermal.cooldown` calls `kill_all_serving`, which `pkill`s the
  squish server process before every pass. The in-memory prefix slot
  (`_prefix_reuse_state`) is a module global in that process — killing the
  process discards it. No `--prompt-kv-cache` / `--block-kv-cache` flags are
  passed, so there is no on-disk cache directory to clear.
- **Ollama**: same `kill_all_serving` call tears down `ollama serve` before
  every pass. `keep_alive=-1` keeps the model resident *within* a pass (as
  required — this is a load-time fairness control, not a cache-reuse
  vector), but each pass starts from a freshly spawned server process with an
  empty KV cache, and every prompt sent within a pass is unique by
  construction, so nothing is ever available to reuse.

## Output

`results/cold_unique_h2h/<UTC>/raw.json` — per-run TTFT / decode-tok/s / E2E,
the measured cache-hit fraction and method for every kept **and** discarded
run, thermal die-temperature max per pass, the drift-check result, and the
full prompt corpus under `results/cold_unique_h2h/<UTC>/prompts/`
(`<pass>_c<ctx>/run_NNN.txt` + `manifest.json` with seed/token-count/SHA-256
per prompt — same audit format as the matrix harness's `save_cell_prompts`).

## Non-goals (explicit scope, per the sprint brief)

- Not the full reuse × context-length matrix — `matrix/run_matrix.py` was not
  built or run for this sprint.
- No 30-run Wilcoxon statistical rigor — 5 clean runs per cell, matching
  `bench_thermal_h2h.py`'s protocol weight, not the matrix's.
- No 25/50/75/95% overlap levels — that data already exists
  (`results/prefix_reuse_curve/`, `results/benchmark_matrix/`).
- No INT3 row — INT4 vs Q4_K_M only, matching the rest of the article's
  head-to-head convention.

## Reproduce

```bash
# kill-test gate: ctx=1024 only, then STOP for review
cd benchmarks/ollama_vs_squish && python3 bench_cold_unique_h2h.py

# after human approval only — all 4 context lengths
cd benchmarks/ollama_vs_squish && python3 bench_cold_unique_h2h.py --full --i-have-approved
```
