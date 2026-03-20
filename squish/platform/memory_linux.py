"""squish/platform/memory_linux.py — Linux memory pressure governor.

Provides a /proc/meminfo + cgroup v1/v2-aware memory governor for Linux,
analogous to the macOS MemoryGovernor (vm_stat-based).  Designed for
containerised and bare-metal GPU inference workloads.

Classes
───────
LinuxMemLevel     — Pressure level enum (OK / MODERATE / HIGH / CRITICAL).
LinuxMemSnapshot  — Parsed /proc/meminfo + cgroup watermark snapshot.
LinuxMemConfig    — Configuration dataclass.
LinuxMemGovernorStats — Runtime stats.
LinuxMemGovernor  — Main governor: start/stop, register handlers.

Usage::

    gov = LinuxMemGovernor()
    gov.register_handler(LinuxMemLevel.HIGH, lambda snap: print("high!"))
    gov.start()
    snap = gov.snapshot()
    gov.stop()
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Memory pressure levels
# ---------------------------------------------------------------------------

class LinuxMemLevel(IntEnum):
    """Memory pressure levels, ordered by severity."""
    OK       = 0
    MODERATE = 1
    HIGH     = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinuxMemSnapshot:
    """Read-only snapshot of current Linux memory state."""
    total_gb:     float   # MemTotal
    available_gb: float   # MemAvailable
    used_gb:      float   # total - available
    usage_ratio:  float   # used / total
    cgroup_limit_gb:  Optional[float]   # cgroup memory.limit (or None)
    cgroup_usage_gb:  Optional[float]   # cgroup memory.usage (or None)
    swap_total_gb:    float
    swap_free_gb:     float
    level:        LinuxMemLevel

    @property
    def pressure_pct(self) -> float:
        """Used memory as percentage of total."""
        return self.usage_ratio * 100.0


# ---------------------------------------------------------------------------
# Config and stats
# ---------------------------------------------------------------------------

@dataclass
class LinuxMemConfig:
    """Configuration for LinuxMemGovernor.

    Attributes
    ----------
    poll_interval_s:
        How often to read /proc/meminfo. Default 1.0 s.
    moderate_threshold:
        usage_ratio above which MODERATE is declared. Default 0.65.
    high_threshold:
        usage_ratio above which HIGH is declared. Default 0.80.
    critical_threshold:
        usage_ratio above which CRITICAL is declared. Default 0.92.
    respect_cgroup:
        If True, use cgroup limit as the total when inside a container.
    """
    poll_interval_s:     float = 1.0
    moderate_threshold:  float = 0.65
    high_threshold:      float = 0.80
    critical_threshold:  float = 0.92
    respect_cgroup:      bool  = True

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError(
                f"poll_interval_s must be > 0, got {self.poll_interval_s}"
            )
        thresholds = [
            self.moderate_threshold,
            self.high_threshold,
            self.critical_threshold,
        ]
        for name, val in zip(
            ("moderate_threshold", "high_threshold", "critical_threshold"),
            thresholds,
        ):
            if not (0.0 < val < 1.0):
                raise ValueError(
                    f"{name} must be in (0, 1), got {val}"
                )
        if not (
            self.moderate_threshold
            < self.high_threshold
            < self.critical_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy: "
                "moderate < high < critical"
            )


@dataclass
class LinuxMemGovernorStats:
    """Runtime statistics for LinuxMemGovernor."""
    snapshots_taken:    int   = 0
    handler_calls:      int   = 0
    level_transitions:  int   = 0
    last_poll_ms:       float = 0.0

    @property
    def avg_polls_per_handler_call(self) -> float:
        if self.handler_calls == 0:
            return float(self.snapshots_taken)
        return self.snapshots_taken / self.handler_calls


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------

_NOOP_PLATFORM = sys.platform != "linux"


class LinuxMemGovernor:
    """Poll /proc/meminfo and cgroup memory files; call handlers on pressure.

    On non-Linux platforms the governor operates in no-op mode — all methods
    succeed silently, and ``snapshot()`` returns a zeroed-out snapshot at
    ``LinuxMemLevel.OK``.

    Usage::

        cfg = LinuxMemConfig(poll_interval_s=0.5)
        gov = LinuxMemGovernor(cfg)
        gov.register_handler(LinuxMemLevel.HIGH, my_handler)
        gov.start()
        # ... inference runs ...
        gov.stop()
    """

    def __init__(self, config: Optional[LinuxMemConfig] = None) -> None:
        self._cfg   = config or LinuxMemConfig()
        self.stats  = LinuxMemGovernorStats()

        self._handlers: Dict[LinuxMemLevel, List[Callable[[LinuxMemSnapshot], None]]] = {
            lvl: [] for lvl in LinuxMemLevel
        }
        self._last_level: LinuxMemLevel = LinuxMemLevel.OK
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock    = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_handler(
        self,
        level: LinuxMemLevel,
        handler: Callable[[LinuxMemSnapshot], None],
    ) -> None:
        """Register a callback for a given pressure level."""
        with self._lock:
            self._handlers[level].append(handler)

    def start(self) -> None:
        """Start background polling thread."""
        if _NOOP_PLATFORM:
            return
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="squish-linux-memgov"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> LinuxMemSnapshot:
        """Return a fresh memory snapshot (or a zeroed no-op snapshot)."""
        if _NOOP_PLATFORM:
            return self._noop_snapshot()
        return self._read_snapshot()

    @property
    def current_level(self) -> LinuxMemLevel:
        """Last observed memory pressure level."""
        return self._last_level

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            snap = self._read_snapshot()
            self.stats.last_poll_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.snapshots_taken += 1

            if snap.level != self._last_level:
                self.stats.level_transitions += 1
            self._last_level = snap.level

            with self._lock:
                handlers = list(self._handlers.get(snap.level, []))
            for h in handlers:
                h(snap)
                self.stats.handler_calls += 1

            time.sleep(self._cfg.poll_interval_s)

    def _read_snapshot(self) -> LinuxMemSnapshot:
        meminfo   = self._parse_proc_meminfo()
        cg_limit  = self._read_cgroup_limit()
        cg_usage  = self._read_cgroup_usage()

        total_kb  = meminfo.get("MemTotal", 0)
        avail_kb  = meminfo.get("MemAvailable", 0)
        swaptot   = meminfo.get("SwapTotal", 0)
        swapfree  = meminfo.get("SwapFree", 0)

        total_gb  = total_kb  / 1e6
        avail_gb  = avail_kb  / 1e6
        used_gb   = max(total_gb - avail_gb, 0.0)

        # Use cgroup limit as ceiling when inside a container
        if self._cfg.respect_cgroup and cg_limit is not None and cg_limit > 0:
            effective_total = cg_limit
            effective_used  = cg_usage if cg_usage is not None else used_gb
        else:
            effective_total = total_gb
            effective_used  = used_gb

        ratio = effective_used / effective_total if effective_total > 0 else 0.0
        level = self._classify(ratio)

        return LinuxMemSnapshot(
            total_gb        = total_gb,
            available_gb    = avail_gb,
            used_gb         = used_gb,
            usage_ratio     = ratio,
            cgroup_limit_gb = cg_limit,
            cgroup_usage_gb = cg_usage,
            swap_total_gb   = swaptot / 1e6,
            swap_free_gb    = swapfree / 1e6,
            level           = level,
        )

    def _classify(self, ratio: float) -> LinuxMemLevel:
        if ratio >= self._cfg.critical_threshold:
            return LinuxMemLevel.CRITICAL
        if ratio >= self._cfg.high_threshold:
            return LinuxMemLevel.HIGH
        if ratio >= self._cfg.moderate_threshold:
            return LinuxMemLevel.MODERATE
        return LinuxMemLevel.OK

    @staticmethod
    def _parse_proc_meminfo() -> Dict[str, int]:
        result: Dict[str, int] = {}
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        try:
                            result[key] = int(parts[1])
                        except ValueError:
                            pass
        except Exception:
            pass
        return result

    @staticmethod
    def _read_cgroup_limit() -> Optional[float]:
        """Read cgroup v2 memory.max or cgroup v1 memory.limit_in_bytes."""
        _HUGE = 9_223_372_036_854_771_712  # "max" placeholder in cgroup v2
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ):
            try:
                raw = open(path).read().strip()
                val = int(raw)
                if val >= _HUGE:
                    return None   # unlimited
                return round(val / 1e9, 2)
            except Exception:
                continue
        return None

    @staticmethod
    def _read_cgroup_usage() -> Optional[float]:
        for path in (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ):
            try:
                return round(int(open(path).read().strip()) / 1e9, 2)
            except Exception:
                continue
        return None

    @staticmethod
    def _noop_snapshot() -> LinuxMemSnapshot:
        return LinuxMemSnapshot(
            total_gb=0.0, available_gb=0.0, used_gb=0.0,
            usage_ratio=0.0, cgroup_limit_gb=None, cgroup_usage_gb=None,
            swap_total_gb=0.0, swap_free_gb=0.0, level=LinuxMemLevel.OK,
        )

    def __repr__(self) -> str:
        return (
            f"LinuxMemGovernor("
            f"running={self._running}, "
            f"level={self._last_level.name}, "
            f"noop={_NOOP_PLATFORM})"
        )
