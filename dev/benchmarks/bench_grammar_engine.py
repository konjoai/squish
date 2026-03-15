#!/usr/bin/env python3
"""dev/benchmarks/bench_grammar_engine.py

Per-token grammar mask latency benchmark.

Measures three operation modes at vocab_size=32000 across batch sizes
1, 4, 16, and 64:

  (a) no-op pass-through  — GrammarEngine.constrain_logits with state=None
                            (fallback path; should be near-zero overhead)
  (b) numpy mask apply    — boolean forbidden-token mask applied to a
                            (batch_size, vocab_size) float32 logits array
  (c) bitmask AND         — uint32 bitmask AND with a precomputed
                            context-independent mask, matching the
                            ``bitmask &= self._independent_mask`` line in
                            GrammarEngine._apply_combined_mask

Results are saved as JSON to
    dev/results/grammar_engine_bench.json

and printed as a human-readable table to stdout.

Usage
-----
    python dev/benchmarks/bench_grammar_engine.py

No external dependencies beyond numpy.  Always exits 0.
xgrammar is not required; the benchmark runs correctly whether or not it
is installed.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 32_000
BATCH_SIZES = [1, 4, 16, 64]
ITERATIONS  = 1_000

_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "grammar_engine_bench.json"

# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _logits(batch_size: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Return a float32 logits array of shape (batch_size, vocab_size)."""
    return np.random.randn(batch_size, vocab_size).astype(np.float32)


def _bool_mask(vocab_size: int = VOCAB_SIZE, forbidden_frac: float = 0.05) -> np.ndarray:
    """Return a boolean 1-D mask; True = token is forbidden."""
    mask = np.zeros(vocab_size, dtype=bool)
    n    = max(1, int(vocab_size * forbidden_frac))
    idx  = np.random.choice(vocab_size, size=n, replace=False)
    mask[idx] = True
    return mask


def _bitmask(vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Return a uint32 bitmask of shape (1, ceil(vocab_size/32))."""
    words = math.ceil(vocab_size / 32)
    return np.full((1, words), np.uint32(0xFFFFFFFF), dtype=np.uint32)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_us(fn: Any, iters: int, warmup: int = 20) -> float:
    """
    Call fn() `iters` times and return the mean latency in microseconds.

    A fixed `warmup` number of calls are made first and excluded from timing.
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1_000_000.0


# ---------------------------------------------------------------------------
# Benchmark A — no-op pass-through
# ---------------------------------------------------------------------------

def _load_engine() -> Any:
    """
    Return a GrammarEngine instance.  The engine will be in fallback mode
    (self._available == False) when xgrammar is not installed, which is the
    mode being benchmarked here.  If xgrammar *is* installed we still force
    the pass-through path by passing state=None to constrain_logits.
    """
    try:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from squish.grammar.grammar_engine import GrammarEngine  # noqa: PLC0415
        try:
            engine = GrammarEngine(None)  # type: ignore[arg-type]
        except Exception:
            engine = GrammarEngine.__new__(GrammarEngine)
            engine._available = False  # type: ignore[attr-defined]
        return engine
    except Exception:
        return None


def bench_noop_passthrough(batch_size: int) -> float:
    """
    Measure latency of GrammarEngine.constrain_logits(logits, state=None).

    The grammar engine checks ``if not self._available or state is None``
    and returns the logits array immediately — no masking occurs.  This
    measures the Python-level guard cost only.

    A plain numpy array is passed as logits; the function returns it
    unchanged regardless of its type when state is None.
    """
    engine = _load_engine()
    arr    = _logits(batch_size)

    if engine is not None and hasattr(engine, "constrain_logits"):
        def _call() -> None:
            engine.constrain_logits(arr, None)
    else:
        # Fallback: simulate the two-condition guard manually so the
        # benchmark never raises even if the import completely fails.
        available = False
        state: Any = None

        def _call() -> None:
            if not available or state is None:
                return

    return _time_us(_call, ITERATIONS)


# ---------------------------------------------------------------------------
# Benchmark B — numpy mask application
# ---------------------------------------------------------------------------

def bench_numpy_mask_apply(batch_size: int) -> float:
    """
    Measure latency of applying a boolean forbidden-token mask to a
    (batch_size, vocab_size) float32 logits array in-place.

    This simulates the logical equivalent of
    ``xgrammar.apply_token_bitmask_inplace`` implemented in pure numpy:
    tokens whose bitmask bit is 0 are set to -1e9 (negative infinity proxy).
    """
    arr  = _logits(batch_size)
    mask = _bool_mask()           # shape: (VOCAB_SIZE,) — True = forbidden

    def _call() -> None:
        arr[:, mask] = np.float32(-1e9)

    return _time_us(_call, ITERATIONS)


# ---------------------------------------------------------------------------
# Benchmark C — bitmask AND operation
# ---------------------------------------------------------------------------

def bench_bitmask_and(batch_size: int) -> float:  # noqa: ARG001
    """
    Measure latency of a uint32 bitmask AND with the precomputed
    context-independent mask.

    Simulates ``bitmask &= self._independent_mask`` from
    GrammarEngine._apply_combined_mask.  The bitmask shape is
    (1, ceil(vocab_size/32)) — independent of batch_size; the parameter
    is accepted only for a consistent table structure.

    Uses np.bitwise_and with an explicit output buffer so no allocation
    occurs inside the timed loop.
    """
    bitmask          = _bitmask()
    independent_mask = _bitmask()
    # Make the independent mask non-trivial: clear ~5% of words
    words   = independent_mask.shape[1]
    n_clear = max(1, words // 20)
    clear_i = np.random.choice(words, size=n_clear, replace=False)
    independent_mask[0, clear_i] = np.uint32(0xAAAAAAAA)

    result = np.empty_like(bitmask)

    def _call() -> None:
        np.bitwise_and(bitmask, independent_mask, out=result)

    return _time_us(_call, ITERATIONS)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_W_OP    = 26
_COL_W_BATCH =  5
_COL_W_US    = 10
_COL_W_CPS   = 13
_SEP_WIDTH   = _COL_W_OP + _COL_W_BATCH + _COL_W_US + _COL_W_CPS + 8


def _sep() -> str:
    return "-" * _SEP_WIDTH


def _header() -> str:
    return (
        f"  {'Operation':<{_COL_W_OP}}"
        f" {'Batch':>{_COL_W_BATCH}}"
        f"  {'µs / call':>{_COL_W_US}}"
        f"  {'calls / s':>{_COL_W_CPS}}"
    )


def _row(label: str, batch_size: int, us: float) -> str:
    cps = 1_000_000.0 / us if us > 0.0 else float("inf")
    return (
        f"  {label:<{_COL_W_OP}}"
        f" {batch_size:>{_COL_W_BATCH}}"
        f"  {us:>{_COL_W_US}.3f}"
        f"  {cps:>{_COL_W_CPS},.0f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)

    benchmarks = [
        ("noop_passthrough", "no-op pass-through",  bench_noop_passthrough),
        ("numpy_mask_apply", "numpy mask apply",     bench_numpy_mask_apply),
        ("bitmask_and",      "bitmask AND",          bench_bitmask_and),
    ]

    measurements: list[dict[str, Any]] = []

    print(_sep())
    print("Grammar Engine — Per-Token Mask Latency Benchmark")
    print(f"  vocab_size = {VOCAB_SIZE:,}  |  iterations per cell = {ITERATIONS:,}")
    print(_sep())
    print(_header())
    print(_sep())

    for key, label, fn in benchmarks:
        for bs in BATCH_SIZES:
            us  = fn(bs)
            cps = 1_000_000.0 / us if us > 0.0 else float("inf")
            print(_row(label, bs, us))
            measurements.append(
                {
                    "operation":   key,
                    "label":       label,
                    "batch_size":  bs,
                    "latency_us":  round(us, 4),
                    "calls_per_s": round(cps, 1),
                }
            )
        print()

    print(_sep())

    results: dict[str, Any] = {
        "vocab_size":   VOCAB_SIZE,
        "iterations":   ITERATIONS,
        "batch_sizes":  BATCH_SIZES,
        "measurements": measurements,
    }

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
