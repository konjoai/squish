# Next session — memory governor eviction sprint, continuation

## Where this left off
Phases 1-4 are landed in v9.34.11: live-adjustable cache budgets, WARNING/
URGENT cache shrink, `budget_tokens()` as a per-request context ceiling, and
CRITICAL request shedding via `_MemoryPressureShedMiddleware`. Only Phase 5
(concurrency stress pass) remains — the sprint brief scopes it as a
dedicated verification pass across everything built in Phases 1-4, not new
functionality, but it's still worth a checkpoint since a stress-test finding
could surface a real bug in the already-shipped phases.

## What's built and verified
- `BlockKVCache.set_hot_max_bytes(n)` / `PromptKVStore.set_max_bytes(n)` —
  thread-safe, evict immediately. (v9.34.9)
- `squish/server.py::_on_memory_pressure_change` — NORMAL (restore),
  WARNING (shrink to 50%), URGENT (shrink to 20%), both always shrinking
  from the same originally-captured baseline regardless of escalation
  direction. CRITICAL doesn't shrink caches further. (v9.34.9-10)
- `squish/server.py::_effective_max_kv_size()` — per-request `max_kv_size`
  ceiling, capped at `governor.budget_tokens()` whenever pressure isn't
  NORMAL (including CRITICAL). Never raises the configured ceiling. (v9.34.10)
- `squish/server.py::_MemoryPressureShedMiddleware` — rejects new requests
  with HTTP 503 at CRITICAL, except `/health`/`/v1/metrics`. In-flight
  requests are unaffected; this is shedding, not queueing. (v9.34.11)
- Kill-test evidence: `tests/serving/test_memory_governor_wiring.py` (17
  cases), `tests/serving/test_effective_max_kv_size.py` (10 cases),
  `tests/serving/test_critical_request_shedding.py` (13 cases, new).

## What's next (needs approval before starting — see original sprint brief)
1. **Phase 5**: concurrency stress test across all of Phases 1-4 together.
   Fire rapid simulated pressure transitions (NORMAL→WARNING→URGENT→
   CRITICAL→NORMAL, including skipped levels like NORMAL→CRITICAL directly)
   on one thread while concurrent mock requests are in flight on others.
   Assert (not just "didn't crash"):
   - `BlockKVCache`/`PromptKVStore` never observe `hot_bytes >
     hot_max_bytes` (beyond the single-entry floor) mid-storm.
   - `_original_hot_max_bytes`/`_original_prompt_max_bytes` never drift from
     the true original value no matter how many transitions fire.
   - `_effective_max_kv_size()` never returns a value larger than whatever
     the governor's current level should allow at the instant it's called.
   - No request that was genuinely in-flight before a CRITICAL transition
     gets a 503 (harder version of the existing single-transition test in
     `test_critical_request_shedding.py` — do it under real thread
     contention, many requests, many transitions).
   Existing single-transition tests already cover the sequential-correctness
   case for each phase individually; Phase 5 is specifically about
   concurrent/racing correctness, which those don't exercise.

## Also flagged, not acted on
- 10 benchmark-matrix cells (`r*_c16000`, `r*_c32000` in
  `benchmarks/ollama_vs_squish/matrix`) now measure stale behavior on
  memory-constrained hosts — see CHANGELOG 9.34.9 for the full list. Re-run
  is a separate future sprint, not this one.
- `ruff format --check` reports pre-existing drift on
  `squish/server.py`, `squish/kv/block_kv_cache.py`, and
  `squish/kv/prompt_kv_cache.py` unrelated to this sprint's diff — confirmed
  via `git stash` that the same files fail identically on `main` before this
  sprint's changes (local ruff 0.15.20 vs whatever version last formatted the
  repo's hand-aligned dataclass style). Not fixed here; flagged so it isn't
  mistaken for damage caused by this PR.
- `/v1/models`, `/v1/tokenize`, and other cheap-but-unlisted endpoints are
  shed (503) at CRITICAL along with the generation routes, since Phase 4
  used a small observability allowlist rather than a generation-route
  denylist (see CHANGELOG 9.34.11 Ledger). Flagged as an intentional
  simplicity tradeoff, not a bug — revisit only if it proves operationally
  annoying in practice.
