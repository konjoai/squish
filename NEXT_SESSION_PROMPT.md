# Next session — memory governor eviction sprint, continuation

## Where this left off
Phase 1 (live-adjustable cache-limit setters) and the WARNING half of Phase 2
(wiring `MemoryGovernor` WARNING transitions to real hot-tier/prompt-cache
shrink) are landed in v9.34.9. This was a deliberate kill-test gate: the
sprint brief asked for approval before building the riskier remaining pieces.

## What's built and verified
- `BlockKVCache.set_hot_max_bytes(n)` — thread-safe, evicts immediately.
- `PromptKVStore.set_max_bytes(n)` / `.max_bytes` — same, disk-backed.
- `squish/server.py::_on_memory_pressure_change` — registered as a
  `MemoryGovernor` callback in Phase 13B startup (`squish/server.py` around
  the `# ── Phase 13B: macOS Memory Governor` marker). Handles `LEVEL_NORMAL`
  (restore) and `LEVEL_WARNING` (shrink to 50%) only. `LEVEL_URGENT` and
  `LEVEL_CRITICAL` fall through to a deliberate no-op.
- Kill-test evidence: `tests/serving/test_memory_governor_wiring.py` (10
  cases, real cache instances) + unit tests for both new setters.

## What's next (needs approval before starting — see original sprint brief)
1. **Phase 2 URGENT half**: shrink further (propose 20% of original —
   confirm/adjust) and start calling `governor.budget_tokens()`.
2. **Phase 3**: wire `budget_tokens()` as a ceiling on a new request's
   context/KV size. The chokepoint is `squish/server.py:_generate_tokens`
   (shared by chat completions, completions, agent-run, and the
   Ollama-compat routes) — specifically the `_sg_kwargs["max_kv_size"] = ...`
   assignment currently sourced only from the startup-computed `_max_kv_size`
   global. Apply `min(_max_kv_size, governor.budget_tokens())` only when the
   governor is present and not at NORMAL.
3. **Phase 4**: CRITICAL request shedding (HTTP 503), reject-only (no
   queueing). In-flight requests must be allowed to finish. The cleanest
   single chokepoint found during discovery is a small `BaseHTTPMiddleware`
   registered on `app` (see the pattern at `squish/server.py:3172` — note
   that's an unrelated optional `squash.governor` middleware, not this
   sprint's `MemoryGovernor`; don't conflate the two). A middleware fires
   before every route handler regardless of registration order, so it
   naturally excludes anything already past dispatch (in-flight generation).
   The alternative — checking at each of the ~6 individual handler
   `_state.model is None` 503 sites — is more scattered and easy to miss a
   route; prefer the middleware.
4. **Phase 5**: concurrency stress test across all of the above (rapid
   simulated pressure transitions racing concurrent mock requests, assertions
   not just "didn't crash").

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
