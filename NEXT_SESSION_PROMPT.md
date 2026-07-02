# Next session — memory governor eviction sprint, continuation

## Where this left off
Phase 1, Phase 2 (both WARNING and URGENT halves), and Phase 3
(`budget_tokens()` as a per-request context ceiling) are landed in v9.34.10.
Phase 4 (CRITICAL request shedding) and Phase 5 (concurrency stress pass)
remain — the sprint brief calls these out as the riskier, less mechanical
pieces and asked for a checkpoint before them.

## What's built and verified
- `BlockKVCache.set_hot_max_bytes(n)` / `PromptKVStore.set_max_bytes(n)` —
  thread-safe, evict immediately. (v9.34.9)
- `squish/server.py::_on_memory_pressure_change` — registered as a
  `MemoryGovernor` callback in Phase 13B startup. Handles `LEVEL_NORMAL`
  (restore), `LEVEL_WARNING` (shrink to 50%), and `LEVEL_URGENT` (shrink to
  20%) — both cache-budget knobs shrink from the same originally-captured
  baseline regardless of how pressure escalates/de-escalates.
  `LEVEL_CRITICAL` does not shrink the caches further (Ledger: that's
  Phase 4's job).
- `squish/server.py::_effective_max_kv_size()` — per-request ceiling on
  `max_kv_size`, applied at the `_generate_tokens` chokepoint. Caps at
  `governor.budget_tokens()` whenever pressure is not NORMAL (including
  CRITICAL, since Phase 4 doesn't exist yet to reject those requests
  another way). Never raises the configured ceiling, only lowers it.
- Kill-test evidence: `tests/serving/test_memory_governor_wiring.py` (17
  cases total) + `tests/serving/test_effective_max_kv_size.py` (10 cases,
  new) + unit tests for both setters.

## What's next (needs approval before starting — see original sprint brief)
1. **Phase 4**: CRITICAL request shedding (HTTP 503), reject-only (no
   queueing). In-flight requests must be allowed to finish. The cleanest
   single chokepoint found during discovery is a small `BaseHTTPMiddleware`
   registered on `app` (see the pattern at `squish/server.py:3172` — note
   that's an unrelated optional `squash.governor` middleware, not this
   sprint's `MemoryGovernor`; don't conflate the two). A middleware fires
   before every route handler regardless of registration order, so it
   naturally excludes anything already past dispatch (in-flight generation).
   The alternative — checking at each of the ~6 individual handler
   `_state.model is None` 503 sites — is more scattered and easy to miss a
   route; prefer the middleware. This is a one-way-door behavior change
   (rejects requests that previously would have been accepted) — needs its
   own version bump and explicit Ledger entries per the original brief.
2. **Phase 5**: concurrency stress test across everything built so far
   (rapid simulated pressure transitions racing concurrent mock requests,
   assertions not just "didn't crash" — check cache-limit invariants hold
   and `_effective_max_kv_size()` never observes a torn/inconsistent state).

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
