"""Shared harness for the perf battle-test suite (benchmarks/perf/battle_test.py).

Provides rigorous timing (warmup + p50/p95/p99/stddev), hardware capture, a
results-JSON writer, and a verdict model — per `.claude/rules/benchmarking.md`
(≥5 warmup, report percentiles, >5% p95 regression = hard stop).
"""
from __future__ import annotations

import json
import platform
import statistics as st
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Stat:
    """Summary of a sample of per-call durations (seconds)."""

    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    stddev: float

    @classmethod
    def of(cls, samples: "list[float]") -> "Stat":
        s = sorted(samples)
        pct = lambda p: s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]  # noqa: E731
        return cls(
            n=len(s), p50=pct(50), p95=pct(95), p99=pct(99),
            mean=st.mean(s), stddev=(st.pstdev(s) if len(s) > 1 else 0.0),
        )


def measure(fn, *, warmup: int, reps: int, per_call_div=lambda out: 1) -> "Stat":
    """Time ``fn`` over ``reps`` runs after ``warmup`` warmups.

    ``per_call_div(out)`` lets a call that produces N units (e.g. tokens) report
    per-unit time — return the unit count from the call's result.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        samples.append(dt / max(1, per_call_div(out)))
    return Stat.of(samples)


@dataclass
class Case:
    """One optimization's battle-test result."""

    name: str
    pr: str                       # which PR/commit landed it
    metric: str                   # what we measured (e.g. "decode tok/s", "prefill ms")
    baseline: float               # baseline value
    optimized: float              # optimized value
    speedup: float                # optimized/baseline in the "higher is better" sense
    lossless: "bool | None"       # byte-identical to baseline (None = N/A)
    verdict: str                  # PASS / FAIL / WARN
    detail: str = ""
    extra: dict = field(default_factory=dict)


def verdict(*, lossless: "bool | None", speedup: float, min_speedup: float,
            regress_guard: bool = False) -> str:
    """PASS/FAIL/WARN from correctness + speedup.

    A broken correctness contract is always FAIL. ``regress_guard`` cases only
    require *no* >5% regression (speedup ≥ 0.95); others require ``min_speedup``.
    """
    if lossless is False:
        return "FAIL"
    floor = 0.95 if regress_guard else min_speedup
    if speedup >= floor:
        return "PASS"
    return "WARN" if speedup >= floor * 0.9 else "FAIL"


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:  # chip + RAM on macOS
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["ram_gb"] = round(ram / 1024**3, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import mlx.core as mx
        info["mlx"] = getattr(mx, "__version__", "?")
    except ImportError:
        pass
    return info


def write_results(cases: "list[Case]", hw: dict, tag: str, results_root: Path,
                  stamp: str) -> Path:
    """Write a results JSON under ``results_root/<stamp>_<tag>/`` (never overwrite)."""
    out_dir = results_root / f"{stamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag, "stamp": stamp, "hardware": hw,
        "passed": sum(c.verdict == "PASS" for c in cases),
        "failed": sum(c.verdict == "FAIL" for c in cases),
        "warned": sum(c.verdict == "WARN" for c in cases),
        "cases": [asdict(c) for c in cases],
    }
    path = out_dir / "battle_test.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_table(cases: "list[Case]") -> None:
    glyph = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
    print(f"\n{'optimization':30}{'PR':7}{'metric':18}{'baseline':>11}"
          f"{'optimized':>11}{'x':>8}{'lossless':>10}  verdict")
    print("─" * 108)
    for c in cases:
        loss = "—" if c.lossless is None else ("yes" if c.lossless else "NO")
        print(f"{c.name:30}{c.pr:7}{c.metric:18}{c.baseline:>11.2f}"
              f"{c.optimized:>11.2f}{c.speedup:>7.2f}x{loss:>10}  "
              f"{glyph.get(c.verdict, c.verdict)} {c.verdict}")
        if c.detail:
            print(f"{'':30}{'':7}↳ {c.detail}")
