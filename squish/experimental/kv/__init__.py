"""
squish.experimental.kv

Research implementations not yet wired into the default inference path.
These modules are valid Python and can be imported explicitly by advanced
users, but are not activated by any default ``squish serve`` flag and have
not been validated on all hardware.

Promotion criteria (to move back to ``squish/kv/``):
  1. Benchmarked end-to-end on Qwen2.5-1.5B, Qwen3-4B, Qwen3-8B
  2. Full unit + integration tests passing
  3. Memory and latency within regression gates (BENCHMARK_REFERENCE.md)
  4. Explicit review before merge

90-day review date: 2026-06-25
"""
