"""Thermal control — the strongest part of the existing methodology, preserved.

The M3 throttles under sustained load, so whichever engine runs hotter looks
slower. The existing harnesses (``bench_thermal_h2h``, ``bench_p4000_iso``)
defend against this with: a cooldown before every config from a near-baseline
temperature, a settle between phases, and a first-vs-last drift probe. This
module keeps all of that and adds the sprint's requirements: a ~50 degC baseline
gate, live die-temperature logging, and an explicit drift check with a 1.7%
ceiling.

Temperature reading prefers ``macmon`` (sudoless, works reliably on Apple
Silicon) and degrades gracefully from there — if no sensor command is available
at all (non-mac, macmon not installed, no sudo for powermetrics) it returns
``None`` and the harness logs "temp unavailable" rather than crashing. The
drift arithmetic and both sensor-output parsers are pure and unit-tested.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

BASELINE_TARGET_C = 50.0
DRIFT_CEILING_PCT = 1.7
DEFAULT_COOLDOWN_S = 120
DEFAULT_SETTLE_S = 25


# ── temperature reading ───────────────────────────────────────────────────────


def parse_powermetrics_temp(text: str) -> float | None:
    """Extract a CPU/GPU die temperature (degC) from powermetrics SMC output."""
    for line in text.splitlines():
        low = line.lower()
        if "die temperature" in low or "cpu die temperature" in low or "gpu die temperature" in low:
            for tok in line.replace(":", " ").split():
                try:
                    return float(tok)
                except ValueError:
                    continue
    return None


def parse_macmon_temp(text: str) -> float | None:
    """Extract the max CPU/GPU die temperature (degC) from a ``macmon pipe`` JSON line.

    ``macmon`` (https://github.com/vladkens/macmon) is a sudoless Apple Silicon
    sensor CLI — the tool this project's earlier thermally-controlled head-to-head
    (commit b9b5d8e) validated cooldowns against. Takes the max of cpu/gpu die temp
    since either can be the throttling driver depending on workload.
    """
    line = text.strip().splitlines()[-1] if text.strip() else ""
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    temp = data.get("temp", {})
    readings = [
        v
        for v in (temp.get("cpu_temp_avg"), temp.get("gpu_temp_avg"))
        if isinstance(v, (int, float))
    ]
    return max(readings) if readings else None


PLAUSIBLE_DIE_TEMP_RANGE_C = (5.0, 110.0)


def _plausible_die_temp(t: float) -> bool:
    """Reject readings outside a plausible CPU/GPU die temperature range.

    Some sensor CLIs (e.g. ``osx-cpu-temp`` on Apple Silicon, where the
    Intel-only SMC keys it reads don't exist) exit 0 and print a fixed
    ``0.0`` rather than failing — a bogus reading, not a real one. Treating
    that as valid would make ``wait_for_baseline`` pass instantly every time,
    silently turning the thermal gate into a no-op. Better to fall through to
    the next probe, or report "no sensor" honestly, than trust an implausible
    number.
    """
    lo, hi = PLAUSIBLE_DIE_TEMP_RANGE_C
    return lo <= t <= hi


def read_die_temp_c() -> float | None:
    """Best-effort die temperature in degC, trying several mac sensor tools.

    Order: macmon (sudoless, reliable on Apple Silicon — preferred), powermetrics
    (needs sudo), osx-cpu-temp, istats (the latter two read Intel-only SMC keys and
    are unreliable/non-functional on Apple Silicon, kept only as a last resort on
    Intel Macs). Any failure, or an implausible reading, falls through to the next;
    all-fail returns None.
    """
    probes: list[list[str]] = [
        ["macmon", "pipe", "-s", "1"],
        ["sudo", "-n", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "1"],
        ["osx-cpu-temp"],
        ["istats", "cpu", "temp", "--value-only"],
    ]
    for cmd in probes:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode != 0:
            continue
        if cmd[0] == "macmon":
            t = parse_macmon_temp(out.stdout)
            if t is not None and _plausible_die_temp(t):
                return t
        elif cmd[0] in ("osx-cpu-temp", "istats"):
            for tok in out.stdout.replace("°C", " ").replace("C", " ").split():
                try:
                    t = float(tok)
                except ValueError:
                    continue
                if _plausible_die_temp(t):
                    return t
        else:
            t = parse_powermetrics_temp(out.stdout)
            if t is not None and _plausible_die_temp(t):
                return t
    return None


# ── drift check ───────────────────────────────────────────────────────────────


@dataclass
class DriftResult:
    first: float
    last: float
    pct: float
    ceiling_pct: float
    passed: bool

    def describe(self) -> str:
        verdict = "OK" if self.passed else "FAIL"
        return (
            f"drift {self.first:.3g} -> {self.last:.3g} "
            f"({self.pct:+.2f}%, ceiling {self.ceiling_pct}%) [{verdict}]"
        )


def drift_check(first: float, last: float, ceiling_pct: float = DRIFT_CEILING_PCT) -> DriftResult:
    """First-vs-last drift on a repeated measurement (e.g. ollama-first vs -last).

    A small magnitude means the cooldowns reset and the cross-config comparison
    sat on equal thermal footing. ``passed`` iff |pct| <= ceiling.
    """
    if first == 0:
        pct = 0.0 if last == 0 else float("inf")
    else:
        pct = (last - first) / first * 100.0
    return DriftResult(first, last, pct, ceiling_pct, abs(pct) <= ceiling_pct)


# ── live temperature log + baseline gate ──────────────────────────────────────


@dataclass
class ThermalLog:
    samples: list[tuple[float, float, str]] = field(default_factory=list)

    def record(self, temp_c: float | None, label: str) -> None:
        if temp_c is not None:
            self.samples.append((time.time(), temp_c, label))

    def max_temp(self) -> float | None:
        return max((t for _, t, _ in self.samples), default=None)

    def as_rows(self) -> list[dict[str, object]]:
        return [{"t": t, "temp_c": c, "label": lbl} for t, c, lbl in self.samples]


class TemperatureSampler(threading.Thread):
    """Background die-temp logger; appends to a ThermalLog at a fixed cadence."""

    def __init__(
        self,
        log: ThermalLog,
        label: str,
        interval_s: float = 5.0,
        reader: Callable[[], float | None] = read_die_temp_c,
    ) -> None:
        super().__init__(daemon=True)
        self.log = log
        self.label = label
        self.interval_s = interval_s
        self.reader = reader
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.log.record(self.reader(), self.label)
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


def wait_for_baseline(
    target_c: float = BASELINE_TARGET_C,
    timeout_s: float = 600.0,
    poll_s: float = 10.0,
    reader: Callable[[], float | None] = read_die_temp_c,
    log: Callable[[str], None] = print,
) -> bool:
    """Block until die temp <= target_c (or timeout). True if baseline reached.

    If no sensor is available the gate is skipped (returns True) and the absence
    is logged, so a sensorless machine still runs — but the writeup must note the
    baseline was time-based, not temperature-gated, on that host.
    """
    first = reader()
    if first is None:
        log("  [thermal] no die-temp sensor available; baseline gate SKIPPED")
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        t = reader()
        if t is None:
            log("  [thermal] sensor dropped out mid-wait; proceeding")
            return True
        if t <= target_c:
            log(f"  [thermal] baseline reached: {t:.1f} degC <= {target_c:.0f}")
            return True
        log(f"  [thermal] waiting for baseline: {t:.1f} degC > {target_c:.0f}")
        time.sleep(poll_s)
    log(f"  [thermal] baseline NOT reached within {timeout_s:.0f}s")
    return False


def cooldown(
    seconds: int, kill_fn: Callable[[], None] | None = None, log: Callable[[str], None] = print
) -> None:
    """Idle with all servers down so the next config starts near baseline."""
    if kill_fn is not None:
        kill_fn()
    log(f"  [thermal] cooldown {seconds}s (idle, servers down)")
    time.sleep(seconds)
