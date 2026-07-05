# Optimization Triage — Handoff Prompt (run locally on Apple Silicon, e.g. M3)

> **This is a self-contained handoff document.** Paste this whole file as the first
> message to a fresh Claude Code session running **on your own Mac** (the environment
> that produced this document had no Apple Silicon/MLX and could not run any of the
> measurements below — see [Environment Prerequisites](#environment-prerequisites)).
> Everything the session needs — the module list, the protocol, the decision
> thresholds, and the exact recipes for keeping or killing a module — is inline. No
> URL fetch, no external artifact lookup required.

## Mission

squish has accumulated ~80 optimization modules that are off by default, partially
wired, or outright broken. A prior audit (see [Provenance](#provenance)) catalogued
*what exists and whether it's reachable* but did not measure *whether any of it
actually helps*. Most claims below are a docstring, a paper citation, or an arithmetic
estimate — not a real number from real hardware.

Your job, working through the [Module Ledger](#module-ledger) one row at a time:

1. **Wire it** if it isn't reachable yet (many aren't — see the `Wiring tier` column).
2. **Measure it** — baseline vs. optimization-enabled, using the real tool named for
   its category, on real hardware.
3. **Decide** — KEEP-AND-PROMOTE (make it a new default), KEEP-AS-OPT-IN (real benefit,
   keep the flag, document the tradeoff), or KILL (no measurable benefit, or it
   regresses something that matters) — using the numeric thresholds below, not vibes.
4. **Act** on the decision using the exact [Removal Recipe](#removal-recipe) or
   [Wiring Recipe](#wiring-recipe) — both replicate conventions this repo has already
   used twice (Waves 118/119/124), so don't invent a new pattern.
5. **Report** each module in the [Output Format](#output-format-per-module) so results
   can be reconciled back into the audit artifact.

Work category by category (KV cache → speculative decoding → quantization → hardware →
context/grammar/streaming → serving → experimental). Within a category, wire everything
first (Pass 1, no hardware dependency), then measure everything serially (Pass 2, one
model load at a time — MLX ties up unified memory, so don't try to run two model-loaded
benchmarks concurrently on one Mac). See [Execution Strategy](#execution-strategy).

## Environment Prerequisites

- **Apple Silicon required.** None of `squish bench`/`squish benchmark`/`squish eval`/
  `squish quality` can produce a real number without MLX, which only works on
  `darwin`/`arm64`. If you're not on one, stop here and hand this file to a session
  that is.
- **Models needed**, download once per category batch and reuse:
  - `Qwen2.5-1.5B` — this repo's own accuracy-gate model (CLAUDE.md, BENCHMARKS.md §3).
    Used for every quantization, KV-cache, and serving/latency measurement below.
  - `Qwen2.5-0.5B` — draft model for the speculative-decoding rows.
- **Existing tools — do not build new ones, these already exist:**
  | Tool | Command | What it measures |
  |---|---|---|
  | `squish eval` | `squish eval MODEL_DIR --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa --limit 500` (`cli.py:2394`) | Accuracy (lm-evaluation-harness via `mlx_lm evaluate`). Writes `results/lmeval_<model>_<ts>.json`. Use `--baseline PATH` for delta-from-baseline. |
  | `squish benchmark` | `squish benchmark --format {int8,int4,int2} --save baseline.json` then `--compare baseline.json --threshold 0.05` (`cli.py:6477`) | KV-cache compression SNR (dB), compression ratio, memory, throughput. |
  | `squish quality` | `squish quality [--json]` against a running `squish serve` (`cli.py:2478`, `/v1/quality`) | Live P50/P95/P99 latency, tok/s, TTFT. Run ≥50 requests before reading — noise swamps small deltas below that. |
  | `squish bench` | `squish bench --format int4 --in-features 4096 --out-features 4096 --iters 100` (`cli.py:6301`) | Synthetic GEMV kernel throughput/latency (p50/p95/p99 ms, GOPS, GB/s) — for kernel-level modules only (fused_kernels, astc). |
- **Reporting convention** — match `BENCHMARKS.md`'s existing style (dB SNR for KV
  fidelity, tok/s + ms for throughput/latency, acc_norm % for accuracy) so any new
  numbers slot into that doc without translation.

## Inviolable Gates

These override any measured "improvement" — never promote or ship past them, full stop:

- **INT4 AWQ g=32 must be ≥ 70.6% arc_easy on Qwen2.5-1.5B** (`CLAUDE.md:25`,
  `BENCHMARKS.md:106`).
- **INT3 g=32 must be ≥ 67.2% arc_easy on Qwen2.5-1.5B** (`BENCHMARKS.md:107`).
- **INT2-naive quantization is never shippable, period** — no measured result excuses
  this (`CLAUDE.md`: "INT2 naive is NEVER SHIP"). `squish/quant/sqint2.py` exists
  specifically because naive INT2 measures ~7dB SNR / ~26–30% arc_easy (≈ random) —
  don't re-litigate that; SQINT2 (already in-scope below) is the sanctioned attempt to
  clear INT2 safely, not a reason to relax the gate on anything else.

## Known Pre-Existing Drift (context, not a ledger row)

`squish/cli.py`'s `cmd_run` (line 1958-1959) forwards `--system-prompt` to
`squish/server.py`, but **server.py has no `--system-prompt` argparse argument at
all** — confirmed absent from its full flag set. This is a real, pre-existing bug,
unrelated to this triage effort. Don't assume the rest of the forwarding list is
100% trustworthy just because most of it checks out — verify each row's forwarding
claim independently rather than trusting this document blindly if something looks off.

## Out of Scope (no benchmark — one-line decisions instead)

**Already-shipped defaults** (re-testing these is out of scope — they're already
exercised by every existing eval/bench run): `chunked_prefill` (streaming/), memory
governor + loop guard (serving/), `fused_sampler` (hardware/), `chip_detector`
(hardware/), `blazing` preset (serving/), `SplitLayerLoader.auto_split` (io/),
`arch_resolver` (runtime/), `auto_profile` (runtime/), `obs_report` + `quality_monitor`
+ `production_profiler` (serving/), `PromptPrefixCache` + `RadixTree` text-cache (kv/).

**Pure correctness/dispatch utilities, no perf claim to test**: `runtime/format_validator.py`,
`platform/feature_registry.py`, `platform/detector.py`, `platform/platform_router.py`.
Judgment call only: keep as forward-looking infra, or delete if truly unused — does not
need the measurement protocol below.

**Phantom — registry cleanup only, nothing to benchmark** (confirmed: none of these
files exist anywhere in the repo or its git history, despite being registered with
claimed speedups in `squish/__init__.py`'s `_LAZY_IMPORTS` dict, line 88+):
`squish.attention.{cla,duo_decoding,flash_attention,sage_attention,sage_attention2,
sparge_attn,squeeze_attention,yoco}`, `squish.kv.{diffkv,kv_slab,kvsharer,kvtuner,
paris_kv,shadow_kv,smallkv}`, `squish.speculative.{conf_spec,dovetail,fr_spec,long_spec,
mirror_sd,qspec,sparse_spec,sparse_verify,spec_reason,specontext,sub_spec,trail}`,
`squish.token.*` (all 6), `squish.moe.pipo`, `squish.quant.{vptq,svdq,squeeze_llm,
dfloat11}`, `squish.serving.{ada_serve,robust_scheduler}`, `squish.streaming.seq_packing`,
`squish.hardware.layerwise_loader`, `squish.lora.lora_manager`. Delete every one of
these dict entries following the [Removal Recipe](#removal-recipe) (skip steps
2/wiring-side of it — there's no source file to purge strings from, just the
registry entry, plus a CHANGELOG note and a regression test asserting the entries
are gone from `_LAZY_IMPORTS`).

**Pre-triaged KILL, no testing needed** (fixing these just to confirm what's already
documented wastes real-hardware time better spent elsewhere):
- `squish/quant/vptq.py`, `squish/quant/dfloat11.py` — both target nonexistent modules
  from the separate `squish-convert` binary's `--vptq`/`--dfloat11` flags; VPTQ's own
  benchmark doc already states "not in the shipped tree."
- `squish/reasoning/coconut.py`, `squish/reasoning/self_consistency.py` — `server.py`
  still has `if getattr(args, "coconut"/"self_consistency", False): ...` construction
  branches, but the `--coconut`/`--self-consistency` argparse flags that would set
  those True were already deleted in Wave 118 (CHANGELOG.md:4122) — these are dead
  `getattr` branches with no way to ever become `True`. Finish deleting them; don't
  resurrect the flags to test something already abandoned twice.

## Module Ledger

Columns: **Module** (file) · **Status** (opt/dead/inert/broken from the source audit)
· **Wiring tier** (what must happen before it's testable) · **Tool** · **What "works"
would look like**.

### A. KV Cache

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `--block-kv-cache` (`kv/block_kv_cache.py`) | opt | none — forwarded (`cli.py:1931-1935`, argparse `cli.py:6753`) | `squish benchmark` | long-context + repeat-prompt sets |
| `--prompt-kv-cache` (`kv/prompt_kv_cache.py`) | opt | none — forwarded (`cli.py:1927-1930`) | `squish benchmark` + `squish quality` (TTFT on repeat) | |
| `--kv-cache-mode {int8,int4,int2,snap}` (`kv/kv_cache.py`) | opt | none — forwarded (`cli.py:1938`) | `squish benchmark --compare` | |
| `k8v4_codec.py` (via `--prompt-kv-cache-quant k8v4`) | opt | none | `squish benchmark` | on-disk size + greedy-decode exactness |
| `--disk-prompt-cache` (`server.py:5040`) | opt, unreachable via CLI | **add flag+forward** (defined in server.py argparse only; no `p_run`/`p_serve` entry, no forward line) | `squish quality` (TTFT on hit) | |
| `--session-cache-dir` (`server.py:5093`) | opt, unreachable via CLI | **add flag+forward** | `squish quality` (TTFT across sessions) | |
| `HadamardKVCache` (`kv/kv_cache.py`) | dead (only used by a separate HF-transformers adapter, never squish's own path) | **add flag+forward** (net-new flag) | `squish benchmark` (quant MSE) | |
| `H2OEvictionPolicy` (`kv/kv_cache.py`) | dead | **add flag+forward** | `squish benchmark` | |
| `KVBudgetBroker` (`kv/kv_cache.py`) | dead | **add flag+forward** | `squish benchmark` | coordinates several *other* modules that don't exist — scope down to what's real |
| `delta.py` (`KVCacheDelta`) | dead | **add flag+forward** | custom: measure diff-encode cost vs. full snapshot on an iterative/agent-loop workload | |
| `head_importance.py` | dead | **add flag+forward** | custom: measure downstream accuracy after pruning flagged heads | |
| `mmap_cache.py` | dead | **add flag+forward** | custom: long-context (32K+) run that would OOM without it, measure it doesn't crash + latency cost | |
| `RadixTree` prefix-trie (`kv/radix_cache.py`, needs `--paged-attention`) | broken — `squish.kv.paged_attention` doesn't exist, ImportError caught silently | **finish stub / fix ImportError** (biggest lift in this category — the whole `PagedKVCache` module needs writing, not just a flag) | `squish benchmark` once fixed | consider this a kill candidate if the implementation cost is out of proportion to the benefit |

### B. Speculative Decoding

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `--draft-model` (`speculative/speculative.py`) | opt | none — confirmed forwarded (`cli.py:1924-1925`) | `squish quality` (tok/s, acceptance rate) | pair: Qwen2.5-1.5B target + Qwen2.5-0.5B draft |
| `--eagle-head-dir` (EAGLE-3 path inline in `speculative.py`) | opt | none | `squish quality` (tok/s, acceptance rate) | the *production* EAGLE-3 path — not `speculative/eagle3.py` |
| `--jacobi` (`experimental/jacobi_decode.py`) | opt, unreachable via CLI | **add flag+forward** (server.py:5241, no `p_run`/`p_serve` entry) | `squish quality` | explicitly excluded from `--all-optimizations` for its O(n²) cost — measure that cost directly |
| `--prompt-lookup` (`speculative/prompt_lookup_batched.py`) | on by default already, but verify it's still forwarded correctly | none, just verify | `squish quality` | sanity-check only, not a promote/kill decision — this one already ships |

### C. Quantization (alternatives to default INT4-AWQ)

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `squish compress --format sqint2` (`quant/sqint2.py`) | opt | none | `squish eval` | the sanctioned INT2 path — compare against the inviolable gates directly |
| `squish compress --format aqlm` (`quant/aqlm.py`, `AQLMEncoder`) | opt | none (this path works) | `squish eval` | ~2 bpw, no established accuracy number yet |
| `squish convert --aqlm` (same module, `AQLMQuantizer`) | broken — confirmed: `AQLMQuantizer` does not exist in `quant/aqlm.py` (only `AQLMConfig`/`AQLMEncoder` are defined), so this path `ImportError`s and silently falls back to INT8 | **fix ImportError** (rename the call site to use `AQLMEncoder`, or decide this path is redundant with `squish compress --format aqlm` and just delete it) | `squish eval` once fixed | low effort — likely just a wrong class name, worth a quick fix-and-test rather than an immediate kill |
| `squish convert --int3` (MiLo, `quant/milo_quant.py`) | opt | none | `squish eval` | CHANGELOG already shows per-model results as bad as −16.4pp arc_easy (gemma-3-4b) — re-verify against the gate on Qwen2.5-1.5B specifically, not just trust the existing per-model table |
| `squish compress --format int3` (native mlx_lm 3-bit, distinct from MiLo) | opt | none | `squish eval` | confirm this is the one meeting the ≥67.2% gate, not MiLo |
| `--hqq` (`quant/hqq.py` via `experimental/hqq_quant.py` shim) | opt | none | `squish eval` | fractional bit-widths — test at least one non-integer bit-width in addition to the integer ones |
| `--nf4` / `--ultra` (`quant/nf4_quant.py`) | opt | none | `squish eval` | |
| `--super-weight` (`quant/super_weight_calibrator.py`) | on by default already (catalog.py auto-appends to every INT4 compress) | none — verify it's actually helping | `squish eval` (with vs. without, isolate its effect) | this is the one "on" item worth re-testing since it was never isolated — everything ships with it on, so there's no existing A/B |
| `--zstd-level N` (`io/entropy.py`) | opt | none | disk-size delta + `squish eval` (confirm lossless) | convert-time only, not a serving optimization |
| `squish rotate` (SpinQuant, `experimental/spin_quant.py`) | opt | none | `squish eval` before/after rotation, same target bit-width | |
| `int3_linear.py` (MLX Metal INT3 runtime, via `--int3` on `squish run`/`serve`) | opt | none | `squish quality` (inference speed of the *runtime* path, separate from the accuracy of the *quantization*) | |

### D. Hardware / Runtime / Platform / IO

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `astc_loader.py` | inert — probe wired, but Metal texture creation is an admitted stub | **finish stub implementation** (real `MTLTextureDescriptor`, not a placeholder buffer) | `squish bench` once real | high effort — triage whether this is worth finishing before investing |
| `layer_overlap_loader.py` (`--layer-overlap`) | inert — flag wired, `load_fn` is a stub, no real prefetch occurs | **finish stub implementation** (wire the real Metal weight dispatch the module's own comment says is missing) | cold-start load time (custom timer or `squish quality` load phase) | |
| `--lazy-llm` (`context/lazy_llm.py`) | opt, unreachable via CLI | **add flag+forward** (server.py:5162, no `p_run`/`p_serve` entry) | `squish quality` (TTFT on long prompts) | |
| `fused_kernels.py` | dead — zero call sites anywhere | **add flag+forward**, or judgment-call kill given it duplicates ground `fused_sampler` already covers for sampling | `squish bench` (GEMV) once wired | |
| `kernel_cache.py` | inert — admitted no-op (MLX has no public kernel-cache API yet) | **kill candidate, skip benchmarking** — nothing to measure until MLX exposes the API this depends on | n/a | revisit when upstream MLX ships the API, not now |
| `startup_profiler.py` | inert — read side wired (`/v1/startup-profile`), nothing ever writes | **finish stub implementation** — small: add real `StartupTimer` calls at existing phase boundaries in `main()` | n/a (instrumentation, not a perf claim) | low effort, worth finishing even without a speedup number since it's a debugging tool |

### E. Context / Grammar / Streaming

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `--compress-prompt` (`context/prompt_compressor.py`) | opt, unreachable via CLI | **add flag+forward** (server.py:5101, no `p_run`/`p_serve` entry) | `squish quality` (TTFT, token-count reduction %) | |
| `--structured-output {json,json-schema}` (`grammar/grammar_engine.py`) | opt, unreachable via CLI | **add flag+forward** (server.py:4958, no `p_run`/`p_serve` entry) | `squish quality` + a schema-validity check on outputs | |
| `grammar/schema_gen.py` | dead — `grammar_engine.py` (xgrammar) is used instead in production | **add flag+forward**, or kill given it's a fully redundant reimplementation of what `grammar_engine.py` already does | `squish quality` once wired | low priority — likely a straightforward kill unless xgrammar itself becomes a liability |
| `grammar/grammar_cache.py` | dead — same situation, redundant with `grammar_engine.py` | same as above | n/a | |

### F. Serving / Scheduling

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `--batch-scheduler` (`serving/scheduler.py`) | opt | none — forwarded | `squish quality` under concurrent synthetic load | write a small concurrent-client script if one doesn't already exist in `benchmarks/` |
| `serving/backend_router.py` | dead — `SQUISH_BACKEND` env var read nowhere else | **add flag+forward** or kill if no real multi-backend use case exists today | `squish quality` once wired | |
| `serving/router.py` (`PromptRouter`, only reachable via `squish route <prompt>` diagnostic) | inert | **finish/expose**: give it a real call site in the serve path, not just a CLI diagnostic | `squish quality` (does routing actually change model choice beneficially) | biggest open question in this category — may not be worth promoting past "diagnostic tool" |

### G. Experimental

| Module | Status | Wiring tier | Tool | Notes |
|---|---|---|---|---|
| `experimental/structured_sparsity.py` | inert — mask auto-loaded via `auto_profile`, but `apply_mask()` never called anywhere | **finish stub implementation** — wire `apply_mask()` into the real FFN forward pass | `squish eval` (INT2/INT3 quality with vs. without the mask applied) | |
| `experimental/convert_coreml.py` | dead — a separate, unaudited `squish/loaders/coreml_loader.py` appears to be the real production path | **investigate first**: confirm whether `coreml_loader.py` truly supersedes this before spending wiring effort — likely a kill once confirmed redundant | n/a until investigated | |
| `experimental/torch_ops.py` | dead — CUDA/PyTorch path, squish is MLX-first | kill candidate — no realistic path to relevance on Apple Silicon | n/a | |

## Per-Module Protocol

For every ledger row above (skip the pre-triaged-kill and phantom groups — those go
straight to the Removal Recipe with no measurement):

1. **Wire if needed** — per the row's `Wiring tier`. Use the [Wiring Recipe](#wiring-recipe).
   Skip if the tier says "none."
2. **Baseline** — run the row's `Tool` with the optimization OFF. Record the number(s).
3. **Optimization-enabled** — rerun with the flag ON, same model/config. For latency/
   throughput measurements, use ≥50 requests — smaller samples are noise, not signal.
4. **Compare** against the matching threshold in [Decision Thresholds](#decision-thresholds).
5. **Act**:
   - KILL → [Removal Recipe](#removal-recipe), exactly.
   - KEEP (either tier) → [Wiring Recipe](#wiring-recipe), exactly (if not already fully
     wired in step 1) plus the promote/document steps.
6. **Report** in the [Output Format](#output-format-per-module).

## Decision Thresholds

- **Quantization**: promote if it matches/beats the shipped format's accuracy at
  equal-or-smaller bit-width (±0.5pp arc_easy vs. the nearest shipped comparator);
  opt-in if within 2pp of the relevant [inviolable gate](#inviolable-gates) *and* offers
  a documented size/speed win; kill if >2pp below with no offsetting win. The inviolable
  gates always win regardless of any other number.
- **KV cache**: promote if ≥10% throughput/memory improvement with <1dB SNR loss vs.
  fp16; opt-in if 3–10%, or a real win on one axis with a documented cost on another;
  kill if <3% delta with no quality win.
- **Serving/latency**: promote if ≥8% p95 latency or tok/s gain with no regression
  elsewhere; opt-in if the gain only shows under specific conditions (name them
  explicitly in the report); kill if <3% delta on both p50 and p95 over ≥50 requests.
- **Speculative decoding**: promote if ≥15% tok/s uplift with ≥60% acceptance rate on a
  representative prompt mix; opt-in if it only works well with a specific draft model
  the user must supply anyway; kill if acceptance is too low for net uplift.
- **Stub-dependent modules** (`inert`/`broken` tier): not scoreable until the stub is
  finished. Apply the relevant category's bar above once it is.

## Removal Recipe

For every KILL decision, in one changeset:

1. **`CHANGELOG.md`** — new version entry, `### Removed` section, one bullet per
   deleted symbol/flag/file naming exactly what's gone and the measured number that
   drove the decision (e.g. *"removed `--foo`: measured 1.8% p95 delta over 50
   requests, below the 3% kill threshold"*). Explicitly note anything "considered but
   kept" and why — mirror Wave 124's `_ProductionProfiler`-excluded pattern
   (`CHANGELOG.md:3875`).
2. **New regression test** — `tests/test_wave<N>_<description>.py` that reads the
   affected source file(s) as raw text and asserts each deleted symbol/flag string is
   now absent (the exact pattern `tests/test_wave119_dead_stub_purge.py` and
   `tests/test_wave124_orphan_global_purge.py` already use).
3. **`squish/__init__.py` `_LAZY_IMPORTS` cleanup** — if the killed module has a
   corresponding entry in the dict (line 88+), delete it too. This step is new relative
   to prior purge waves: 7 stale entries from Wave 119 still linger in the registry
   today with zero cleanup precedent — don't repeat that rot on this round's kills.
4. **Version bump** — `pyproject.toml` + `squish/__init__.py`'s `__version__`, kept in
   sync (current: 9.34.14).
5. **Line/module-count delta** noted in the CHANGELOG entry, matching the "(-N lines)"
   convention every prior Wave entry uses.
6. **Confirm Konjo gates stay green** (coverage ≥80%, complexity/size/DRY) — deleting
   dead code should only help these; just confirm no import breakage elsewhere.

## Wiring Recipe

For every KEEP decision on a currently unreachable/inert/broken module, replicating the
confirmed-working `--draft-model` pattern:

1. **`squish/cli.py`** — add `p_run.add_argument("--foo", ...)` near the existing block
   (`p_run` starts ~`cli.py:6715`; e.g. `--draft-model` is defined at `cli.py:6722-6723`)
   and mirror it in `p_serve` (~`cli.py:6870`; e.g. `cli.py:6877`), matching
   `server.py`'s own default/type for that flag exactly.
2. **`squish/cli.py`'s `cmd_run` forward-list** (function starts `cli.py:1638`, the
   subprocess-command-building block runs ~`cli.py:1906-1985`) — add
   `if args.foo: cmd += ["--foo", args.foo]` (or `.append("--foo")` for boolean flags,
   see the `--draft-model` forward at `cli.py:1924-1925` or the multi-value
   `--block-kv-cache` forward at `cli.py:1931-1935` as templates).
3. If `inert` (stub logic never invoked): finish the real call site *first* — e.g. wire
   `structured_sparsity.apply_mask()` into the actual FFN forward pass, or replace
   `layer_overlap_loader`'s stub `load_fn` with real weight dispatch — before the flag
   means anything measurable.
4. If `dead` with zero flag anywhere in `server.py` either: define the flag in
   `server.py`'s own argparse first (match a neighboring flag's style — the block
   runs roughly `server.py:4852-5367`), then repeat steps 1–2.
5. **`CHANGELOG.md`** — `### Added` entry with the measured benefit that justified
   promotion, with the exact number.
6. **`MODULES.md`** — update the module's row with its new reachability status.
7. **A real test** that exercises the flag end-to-end (invokes the CLI parser, confirms
   forwarding, and/or smoke-tests server.py's handling) — not just an import-succeeds
   test.
8. **Confirm Konjo gates stay green** after the addition (new code needs coverage).

## Output Format (per module)

```
module: <name>
category: <one of the 7 above>
decision: KEEP-AND-PROMOTE | KEEP-AS-OPT-IN | KILL
baseline: <numbers, with units>
with-optimization: <numbers, with units>
delta: <% or absolute, whichever the threshold uses>
threshold-applied: <the exact rule from Decision Thresholds that decided it>
action-taken: <files touched, CHANGELOG entry added, test added/removed>
artifact-status-update: <new status this module should carry going forward>
```

## Final Summary Table

One row per module worked: `name | old status | new status | decision | key metric
delta`. This table is what reconciles back into the source audit artifact.

## Execution Strategy

1. **Pass 1 — Wire** (no hardware dependency — this part *can* run anywhere, including
   a non-Apple-Silicon dev box, if you want to split the work): apply every
   "add flag+forward" / "finish stub" / "fix ImportError" item from the ledger's
   `Wiring tier` column across all categories. This is pure code editing.
2. **Pass 2 — Measure** (must run on the Mac, one category at a time, serially):
   download Qwen2.5-1.5B (+ Qwen2.5-0.5B for speculative decoding) once, reuse across
   every module in that category's batch. Run baseline → optimization-enabled → decide
   → act, per the protocol above, before moving to the next category.
3. Do categories in this order (cheapest/most-certain first, to build momentum and bank
   easy wins before the harder investigation-required rows): **C (Quantization)** → **A
   (KV Cache)** → **F (Serving)** → **B (Speculative)** → **E (Context/Grammar)** → **D
   (Hardware)** → **G (Experimental)**.
4. The pre-triaged kill groups (VPTQ/dfloat11, coconut/self_consistency, all 15 phantom
   `_LAZY_IMPORTS` entries) skip Pass 2 entirely — go straight to the Removal Recipe,
   ideally as an early, fast first commit to clear noise out of the way before the real
   measurement work starts.

## Provenance

This handoff was generated from a 9-agent audit of squish's optimization modules,
published as `squish-wiring-audit.html` (an interactive, filterable table — not
required to complete this task, since the full module list is inline above, but useful
as a visual cross-reference if you have it open). Repo state at generation time:
commit `415647f`, version `9.34.14`.
