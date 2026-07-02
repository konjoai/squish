# Memory governor eviction sprint — complete (v9.34.9 - v9.34.12)

All 5 phases from the original sprint brief are landed. `MemoryGovernor`
now actually drives eviction, context sizing, and request shedding across
all four pressure levels (NORMAL/WARNING/URGENT/CRITICAL), matching its own
docstring intent instead of being read only by `/health`.

## What's built and verified
- **Phase 1** (v9.34.9): `BlockKVCache.set_hot_max_bytes(n)` /
  `PromptKVStore.set_max_bytes(n)` — thread-safe, live-adjustable cache
  budgets that evict immediately when shrunk.
- **Phase 2** (v9.34.9-10): `squish/server.py::_on_memory_pressure_change`
  — WARNING shrinks caches to 50%, URGENT to 20%, both always shrinking
  from the same originally-captured baseline regardless of escalation
  direction. NORMAL restores exactly. CRITICAL doesn't shrink caches
  further (that's Phase 4's job).
- **Phase 3** (v9.34.10): `squish/server.py::_effective_max_kv_size()` —
  per-request `max_kv_size` ceiling, capped at `governor.budget_tokens()`
  whenever pressure isn't NORMAL (including CRITICAL). Never raises the
  configured ceiling.
- **Phase 4** (v9.34.11): `squish/server.py::_MemoryPressureShedMiddleware`
  — rejects new requests with HTTP 503 at CRITICAL (reject-only, no
  queueing), exempting `/health`/`/v1/metrics`. In-flight requests are
  never aborted.
- **Phase 5** (v9.34.12): concurrency safety review found and fixed one
  real (if not-yet-reachable) TOCTOU race in Phase 2's baseline-capture
  logic (`_pressure_callback_lock`). Stress tests prove cache/budget
  invariants and response integrity hold under concurrent pressure storms
  racing concurrent request traffic.

## Test coverage (all in `tests/serving/`)
- `test_memory_governor_wiring.py` (17 cases) — WARNING/URGENT/NORMAL
  cache-shrink and restore, real cache instances + mocked-call precision.
- `test_effective_max_kv_size.py` (10 cases) — ceiling computation across
  all pressure levels, never-raise guarantee, real-governor end-to-end case.
- `test_critical_request_shedding.py` (13 cases) — shedding, exemptions,
  CORS-header survival, in-flight-request survival (single transition).
- `test_phase5_concurrency_stress.py` (2 cases) — the same invariants
  under real concurrent load and rapid pressure storms.

## Explicitly out of scope for this sprint (flag for a future one if needed)
- **Request queueing/backpressure at CRITICAL.** The sprint brief scoped
  this out explicitly as "a separate, larger design question" — CRITICAL
  currently only rejects, never queues.
- **10 benchmark-matrix cells** (`r*_c16000`, `r*_c32000` in
  `benchmarks/ollama_vs_squish/matrix`) measure stale behavior on
  memory-constrained hosts now that live eviction exists — see CHANGELOG
  9.34.9 for the full list. Not re-run per this sprint's explicit non-goal.
- **`ruff format --check` pre-existing drift** on `squish/server.py`,
  `squish/kv/block_kv_cache.py`, `squish/kv/prompt_kv_cache.py` — confirmed
  via `git stash` to predate this sprint (local ruff 0.15.20 vs whatever
  version last formatted the repo's hand-aligned dataclass style). Not this
  sprint's regression.
- **`/v1/models`, `/v1/tokenize`, and other cheap-but-unlisted endpoints**
  are shed (503) at CRITICAL along with the generation routes, since
  Phase 4 used a small observability allowlist (`/health`, `/v1/metrics`)
  rather than a generation-route denylist. Intentional simplicity
  tradeoff — revisit only if it proves operationally annoying.
- **Fraction values (WARNING=50%, URGENT=20%) are starting points**, not
  derived from fleet telemetry. Revisit with real production pressure data
  if/when available.
