#!/usr/bin/env python3
"""Cold-prefill micro-benchmark — settles the v5.1 open question.

The v5.1 harness reuses ONE prompt per phase across 5 TTFT runs, so the
reported "TTFT (cold)" median for a 4000-token prompt is dominated by
*cache hits* (ollama's in-memory KV prefix cache; squish's pkv/block).
A 242 ms "TTFT" on 4000 tokens is physically a cache hit, not a real
cold prefill (4000-token prefill of a 7B is seconds of GPU compute).

This script measures the *true* cold prefill by sending genuinely novel
prompts — each unique from token 0 (uuid nonce + randomly-assembled
filler) so neither engine's prefix cache can hit. For each engine we
record, per prompt:

  * cold  = first send  (KV cache cold for this prefix)  ← the real number
  * warm  = second send (exact-match cache hit)          ← the illusion

We run ollama and squish_daemon (no squish caches) for an apples-to-apples
cold-prefill comparison, then squish_block to show squish's architectural
sidestep (it only prefills the non-matching suffix).

Output: results/benchmarks_v5_1_1/cold_prefill/<UTC>.json + stdout table.
"""
from __future__ import annotations

import json
import random
import statistics as stats
import time
import uuid
from pathlib import Path
from typing import Any

# Reuse the v5.1 harness plumbing verbatim (same binaries, model, ports).
import bench_v5_1 as B

# ── Config ────────────────────────────────────────────────────────────────────

N_PROMPTS      = 5          # distinct novel prompts (each measured cold + warm)
TARGET_TOKENS  = 4000       # match the v5.1 p4000 phase
WARM_RUNS      = 2          # repeats after the cold send, to confirm cache hit

OUT_DIR = B.REPO_ROOT / "results" / "benchmarks_v5_1_1" / "cold_prefill"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A pool of varied clauses; randomly assembling them yields prose that is
# token-distinct from prompt to prompt, so no shared prefix survives.
_CLAUSES = [
    "the distributed scheduler reconciles partitions across availability zones",
    "a lock-free ring buffer absorbs bursts without backpressure stalls",
    "the migration replays the write-ahead log under a fenced epoch token",
    "telemetry spans propagate through the gateway with sampled baggage",
    "the planner rewrites the predicate before the index is consulted",
    "garbage collection pauses are amortized across the eviction window",
    "the consensus group elects a leader once the quorum lease expires",
    "vectorized kernels fuse the projection into a single memory pass",
    "the rate limiter sheds load when the token bucket drains to zero",
    "checkpoint snapshots are deduplicated against the content address store",
    "the retry budget decays geometrically after each upstream timeout",
    "shard rebalancing throttles itself to protect the tail latency budget",
    "the parser folds constant subexpressions during semantic analysis",
    "speculative execution discards the branch once the guard misverifies",
    "the cache coherence protocol invalidates the line on the remote write",
    "backfill jobs coalesce adjacent ranges to minimize random seeks",
    "the feature flag gates the rollout behind a sticky cohort hash",
    "compaction merges overlapping sstables into a wider sorted run",
    "the circuit breaker half-opens to probe the recovering dependency",
    "differential dataflow incrementally maintains the materialized view",
]


def _build_novel_prompt(idx: int, tok: "Any") -> str:
    """A ~TARGET_TOKENS prompt that is unique from token 0.

    Seeded per idx for reproducibility, but each idx produces distinct
    content (and a unique uuid nonce up front) so no prefix is shared with
    the warmup, with other prompts, or with anything an engine cached.
    """
    rng = random.Random(idx * 7919 + 17)
    nonce = uuid.uuid4().hex
    head = (
        f"Session {nonce}. You are auditing an unfamiliar service. "
        f"The following notes were captured during incident {idx}:\n"
    )
    parts = [head]
    while True:
        sentence = ", ".join(rng.choice(_CLAUSES) for _ in range(rng.randint(3, 6)))
        parts.append(sentence.capitalize() + ". ")
        if len(tok.encode("".join(parts))) >= TARGET_TOKENS:
            break
    parts.append("\nGiven all of the above, name the single most important risk.")
    return "".join(parts)


def _ttft(cfg_id: str, prompt: str) -> float | None:
    d = B.run_one_request(cfg_id, prompt, max_tokens=1)
    return d.get("ttft_s")


def _measure_config(cfg_id: str, prompts: list[str]) -> dict[str, Any]:
    cfg = B.CONFIGS[cfg_id]
    B.log(f"=== {cfg_id} : start server ===")
    B.kill_all_serving()
    log_path = B.LOG_DIR / f"coldprefill_{cfg_id}_{B.TS}.log"
    proc, sampler = cfg["start"](log_path)
    per_prompt: list[dict[str, Any]] = []
    try:
        if not B.wait_ready(cfg["ready_url"], timeout=300):
            raise RuntimeError(f"{cfg_id} did not become ready")
        # Warm the model into RAM + JIT the kernels with a short, unrelated
        # prompt so the first long prompt measures prefill, not model load.
        B.log("  ready; warming model into memory")
        cfg["stream"]("Warm up the runtime, please.", max_tokens=8)

        for i, prompt in enumerate(prompts):
            cold = _ttft(cfg_id, prompt)                       # first send → cold
            warms = [_ttft(cfg_id, prompt) for _ in range(WARM_RUNS)]  # cache hits
            warm_med = stats.median([w for w in warms if w]) if any(warms) else None
            per_prompt.append({"idx": i, "cold_s": cold, "warm_s": warm_med})
            B.log(
                f"  prompt {i}: cold={_fmt(cold)}  warm={_fmt(warm_med)}  "
                f"(speedup {cold / warm_med:.0f}x)" if (cold and warm_med) else
                f"  prompt {i}: cold={_fmt(cold)}  warm={_fmt(warm_med)}"
            )
    finally:
        B.stop_server(proc, sampler)

    cold_vals = [p["cold_s"] for p in per_prompt if p["cold_s"]]
    warm_vals = [p["warm_s"] for p in per_prompt if p["warm_s"]]
    return {
        "label":      cfg["label"],
        "per_prompt": per_prompt,
        "cold_median_s": stats.median(cold_vals) if cold_vals else None,
        "cold_min_s":    min(cold_vals) if cold_vals else None,
        "cold_max_s":    max(cold_vals) if cold_vals else None,
        "warm_median_s": stats.median(warm_vals) if warm_vals else None,
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 1000:.0f}ms" if v < 1 else f"{v:.2f}s"


def main() -> None:
    B.log(f"Cold-prefill bench — {N_PROMPTS} novel ~{TARGET_TOKENS}-token prompts")
    from mlx_lm import load
    _, tok = load(B.SQUISH_MODEL_INT4)
    prompts = [_build_novel_prompt(i, tok) for i in range(N_PROMPTS)]
    tok_counts = [len(tok.encode(p)) for p in prompts]
    B.log(f"  prompt token counts: {tok_counts}")

    configs = ["ollama", "squish_daemon", "squish_block"]
    results: dict[str, Any] = {
        "timestamp": B.TS,
        "host": "Apple M3 MacBook Pro 16 GB",
        "target_tokens": TARGET_TOKENS,
        "n_prompts": N_PROMPTS,
        "prompt_token_counts": tok_counts,
        "note": "cold = first send (true cold prefill); warm = exact-match cache hit.",
        "configs": {},
    }
    for cfg_id in configs:
        results["configs"][cfg_id] = _measure_config(cfg_id, prompts)

    out = OUT_DIR / f"{B.TS}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    B.log(f"Wrote {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f"# True cold prefill — {N_PROMPTS} novel ~{TARGET_TOKENS}-token prompts (M3 16 GB)")
    print(f"{'Engine':<26} | {'cold p50':>10} | {'cold min':>10} | {'cold max':>10} | {'warm p50':>10}")
    print("-" * 78)
    for cfg_id in configs:
        c = results["configs"][cfg_id]
        print(f"{c['label']:<26} | {_fmt(c['cold_median_s']):>10} | "
              f"{_fmt(c['cold_min_s']):>10} | {_fmt(c['cold_max_s']):>10} | "
              f"{_fmt(c['warm_median_s']):>10}")
    print()


if __name__ == "__main__":
    main()
