"""Peak-RSS sampling, KV-cache memory, and the fit / degraded / OOM classifier.

At long contexts (32k on 16 GB especially) a system may fit, degrade through the
macOS memory governor (compression / swap, surviving but slow), or OOM outright.
The sprint requires recording which of the three happened per cell *without
letting the harness crash* — a non-fit is a result, and how each system degrades
is itself a finding.

``RSSSampler`` mirrors the proven sampler in ``bench_v5_1`` (whole process tree,
50 ms cadence) but imports psutil lazily so this module loads on any platform.
``classify_memory_status`` is pure and unit-tested.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


# ── peak RSS sampler ──────────────────────────────────────────────────────────


class RSSSampler(threading.Thread):
    """Sample peak resident memory of a process tree at 50 ms cadence."""

    def __init__(self, root_pid: int) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self._stop_event = threading.Event()
        self.peak_bytes = 0
        self.samples = 0

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_event.is_set():
            try:
                tree = root.memory_info().rss
                for child in root.children(recursive=True):
                    try:
                        tree += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self.peak_bytes = max(self.peak_bytes, tree)
            self.samples += 1
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


# ── KV-cache memory readout ───────────────────────────────────────────────────


def kv_cache_mb_from_metrics(metrics: dict[str, float]) -> float | None:
    """Squish reports KV-cache memory directly; Ollama does not (None)."""
    v = metrics.get("squish_kv_cache_memory_mb")
    return float(v) if v is not None else None


# ── fit / degraded / OOM classification ───────────────────────────────────────


@dataclass
class MemoryStatus:
    status: str  # "fit" | "degraded_via_governor" | "oom"
    peak_rss_bytes: int
    ram_bytes: int
    headroom_frac: float
    note: str = ""

    @property
    def fit(self) -> bool:
        return self.status == "fit"


def classify_memory_status(
    *,
    peak_rss_bytes: int,
    ram_bytes: int,
    request_failed: bool,
    oom_signal: bool = False,
    governor_event: bool = False,
    decode_tps: float | None = None,
    baseline_tps: float | None = None,
    tps_collapse_frac: float = 0.5,
) -> MemoryStatus:
    """Classify a cell's memory outcome from observed signals.

    * ``oom`` — the request failed with an OOM-class signal, or RSS exceeded RAM
      and the request did not complete.
    * ``degraded_via_governor`` — completed, but a governor/memory-pressure event
      was logged, OR RSS sat within ~5% of RAM, OR decode throughput collapsed
      past ``tps_collapse_frac`` of baseline (the signature of compression/swap).
    * ``fit`` — completed with headroom and no degradation signal.
    """
    headroom = 1.0 - (peak_rss_bytes / ram_bytes) if ram_bytes > 0 else 0.0

    if oom_signal or (request_failed and peak_rss_bytes >= ram_bytes):
        return MemoryStatus(
            "oom",
            peak_rss_bytes,
            ram_bytes,
            headroom,
            "request did not complete under memory pressure",
        )
    if request_failed:
        return MemoryStatus(
            "oom", peak_rss_bytes, ram_bytes, headroom, "request failed (treated as non-fit)"
        )

    collapsed = (
        decode_tps is not None
        and baseline_tps is not None
        and baseline_tps > 0
        and decode_tps < tps_collapse_frac * baseline_tps
    )
    near_cap = headroom <= 0.05
    if governor_event or near_cap or collapsed:
        reasons = []
        if governor_event:
            reasons.append("governor/pressure event logged")
        if near_cap:
            reasons.append(f"RSS within {headroom:.1%} of RAM")
        if collapsed:
            reasons.append("decode throughput collapsed vs baseline")
        return MemoryStatus(
            "degraded_via_governor", peak_rss_bytes, ram_bytes, headroom, "; ".join(reasons)
        )
    return MemoryStatus("fit", peak_rss_bytes, ram_bytes, headroom)


# ── OOM signal detection from a server log tail ───────────────────────────────

_OOM_MARKERS = (
    "out of memory",
    "oom",
    "metal error",
    "failed to allocate",
    "insufficient memory",
    "mlx.core.error",
    "killed",
    "std::bad_alloc",
)
_GOVERNOR_MARKERS = (
    "memory pressure",
    "memory governor",
    "swapping",  # active swap verb; avoids matching benign "free_swap=0 B" log lines
    "paged out",
    "jetsam",
)


def scan_log_for_signals(log_text: str) -> tuple[bool, bool]:
    """Return (oom_signal, governor_event) from a lowercased server-log scan."""
    low = log_text.lower()
    oom = any(m in low for m in _OOM_MARKERS)
    gov = any(m in low for m in _GOVERNOR_MARKERS)
    return oom, gov
