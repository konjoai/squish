#!/usr/bin/env python3
"""
squish/memory_governor.py

macOS memory-pressure watchdog for proactive KV-cache and context management.

Monitors ``kern.memorypressure`` via ``sysctl`` and available physical RAM via
``vm_stat``.  A lightweight background daemon thread polls both sources every
*poll_interval* seconds and fires registered callbacks whenever the pressure
level changes.

Pressure levels (mirroring the macOS kernel constants)
───────────────────────────────────────────────────────
  0  NORMAL   — comfortable headroom; full KV cache / context allowed
  1  WARNING  — low free pages; consider evicting speculative cache entries
  2  URGENT   — memory compressor is active; shed non-essential buffers
  4  CRITICAL — swap pressure; emergency context truncation / request shedding

When ``sysctl`` or ``vm_stat`` are unavailable (Linux CI, containers) every
method returns safe defaults (``LEVEL_NORMAL``, large ``available_gb``).

Usage::

    from squish.serving.memory_governor import MemoryGovernor

    gov = MemoryGovernor(poll_interval=5.0)
    gov.start()

    # Query current state
    print(gov.pressure_level)   # 0 / 1 / 2 / 4
    print(gov.available_gb)     # e.g. 9.2
    print(gov.is_under_pressure)

    # React to changes
    gov.add_callback(lambda level: print(f"pressure → {level}"))

    gov.stop()
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "MemoryGovernor",
    "MemorySnapshot",
    "LEVEL_NORMAL",
    "LEVEL_WARNING",
    "LEVEL_URGENT",
    "LEVEL_CRITICAL",
]

# ── Pressure level constants ──────────────────────────────────────────────────

LEVEL_NORMAL:   int = 0
LEVEL_WARNING:  int = 1
LEVEL_URGENT:   int = 2
LEVEL_CRITICAL: int = 4

_VALID_LEVELS: frozenset[int] = frozenset({LEVEL_NORMAL, LEVEL_WARNING, LEVEL_URGENT, LEVEL_CRITICAL})

# Fallback available memory when vm_stat is unreadable (16 GB)
_DEFAULT_AVAILABLE_GB: float = 16.0


# ── Point-in-time snapshot ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable snapshot of memory state at a single poll instant.

    Attributes
    ----------
    pressure_level : int
        macOS ``kern.memorypressure`` value: 0, 1, 2, or 4.
    available_gb : float
        Approximate immediately reclaimable RAM (free + inactive + purgeable
        pages × page_size) in gibibytes.
    wired_gb : float
        Wired (pinned, non-reclaimable) RAM in gibibytes.
    timestamp : float
        ``time.monotonic()`` at poll time.
    """

    pressure_level: int   = LEVEL_NORMAL
    available_gb:   float = _DEFAULT_AVAILABLE_GB
    wired_gb:       float = 0.0
    timestamp:      float = field(default_factory=time.monotonic)

    @property
    def is_under_pressure(self) -> bool:
        """``True`` when pressure_level ≥ ``LEVEL_WARNING``."""
        return self.pressure_level >= LEVEL_WARNING


# ── vm_stat / sysctl helpers ──────────────────────────────────────────────────


def _run(cmd: list[str]) -> str:
    """Run *cmd* and return stdout; return empty string on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout
    except Exception:
        return ""


def _read_pressure_level() -> int:
    """Return the current ``kern.memorypressure`` value (0/1/2/4).

    Falls back to ``LEVEL_NORMAL`` when ``sysctl`` is unavailable.
    """
    raw = _run(["sysctl", "-n", "kern.memorypressure"]).strip()
    try:
        val = int(raw)
        return val if val in _VALID_LEVELS else LEVEL_NORMAL
    except (ValueError, TypeError):
        return LEVEL_NORMAL


def _read_vm_stat() -> tuple[float, float]:
    """Parse ``vm_stat`` output and return ``(available_gb, wired_gb)``."""
    raw = _run(["vm_stat"])
    if not raw:
        return _DEFAULT_AVAILABLE_GB, 0.0

    # Extract page size from the header line, e.g.:
    #   "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
    page_size = 4096  # default; Apple Silicon uses 16384
    for line in raw.splitlines():
        if "page size of" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "size" and i + 2 < len(parts):
                    try:
                        page_size = int(parts[i + 2])
                    except ValueError:
                        pass
            break

    def _pages(keyword: str) -> int:
        for line in raw.splitlines():
            if line.strip().startswith(keyword):
                val = line.split(":")[-1].strip().rstrip(".")
                try:
                    return int(val)
                except ValueError:
                    return 0
        return 0

    free_pages       = _pages("Pages free")
    inactive_pages   = _pages("Pages inactive")
    speculative_pages = _pages("Pages speculative")
    purgeable_pages  = _pages("Pages purgeable")
    wired_pages      = _pages("Pages wired down")

    available_pages = (
        free_pages + inactive_pages + speculative_pages + purgeable_pages
    )
    _gib = 1 << 30
    available_gb = available_pages * page_size / _gib
    wired_gb     = wired_pages     * page_size / _gib
    return available_gb, wired_gb


# ── MemoryGovernor ────────────────────────────────────────────────────────────


class MemoryGovernor:
    """Background memory-pressure watcher for Apple Silicon (macOS).

    Starts a daemon thread that polls ``sysctl kern.memorypressure`` and
    ``vm_stat`` every *poll_interval* seconds.  Registered callbacks are
    invoked on the polling thread whenever the pressure level changes.

    Parameters
    ----------
    poll_interval : float
        Seconds between polls (default 5.0).
    warning_threshold : int
        Minimum pressure level considered "under pressure" for
        :attr:`is_under_pressure` (default ``LEVEL_WARNING = 1``).
    """

    def __init__(
        self,
        poll_interval:     float = 5.0,
        warning_threshold: int   = LEVEL_WARNING,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive; got {poll_interval}")
        self._interval  = poll_interval
        self._threshold = warning_threshold
        self._lock      = threading.Lock()
        self._snapshot  = MemorySnapshot()
        self._callbacks: list[Callable[[int], None]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> MemoryGovernor:
        """Start the background polling thread.

        Returns *self* for chaining::

            gov = MemoryGovernor().start()
        """
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        # Populate snapshot immediately so callers don't see stale defaults.
        self._poll_once()
        self._thread = threading.Thread(
            target=self._loop,
            name="squish-mem-governor",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the polling thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None

    def _loop(self) -> None:
        """Polling loop — runs on the daemon thread."""
        while not self._stop_event.wait(timeout=self._interval):
            self._poll_once()

    def _poll_once(self) -> None:
        """Single poll: read pressure + vm_stat, update snapshot, fire callbacks."""
        level                  = _read_pressure_level()
        available_gb, wired_gb = _read_vm_stat()
        snap = MemorySnapshot(
            pressure_level=level,
            available_gb=available_gb,
            wired_gb=wired_gb,
            timestamp=time.monotonic(),
        )
        prev_level: int
        with self._lock:
            prev_level    = self._snapshot.pressure_level
            self._snapshot = snap
        if level != prev_level:
            for cb in list(self._callbacks):
                try:
                    cb(level)
                except Exception:
                    pass

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def snapshot(self) -> MemorySnapshot:
        """Most-recent :class:`MemorySnapshot`; updated every poll cycle."""
        with self._lock:
            return self._snapshot

    @property
    def pressure_level(self) -> int:
        """Current macOS memory-pressure level (0 / 1 / 2 / 4)."""
        return self.snapshot.pressure_level

    @property
    def available_gb(self) -> float:
        """Approximate immediately reclaimable RAM in gibibytes."""
        return self.snapshot.available_gb

    @property
    def wired_gb(self) -> float:
        """Wired (non-reclaimable) RAM in gibibytes."""
        return self.snapshot.wired_gb

    @property
    def is_under_pressure(self) -> bool:
        """``True`` when :attr:`pressure_level` ≥ the configured threshold."""
        return self.pressure_level >= self._threshold

    # ── Callbacks ─────────────────────────────────────────────────────────

    def add_callback(self, fn: Callable[[int], None]) -> None:
        """Register *fn* to be called with the new pressure level on change.

        Parameters
        ----------
        fn : callable(level: int) -> None
        """
        self._callbacks.append(fn)

    def remove_callback(self, fn: Callable[[int], None]) -> None:
        """Remove a previously registered callback (no-op if not found)."""
        try:
            self._callbacks.remove(fn)
        except ValueError:
            pass

    # ── Utility ───────────────────────────────────────────────────────────

    def budget_tokens(self, bytes_per_token: float = 512.0) -> int:
        """Estimate a safe KV-cache token budget from available memory.

        Reserves 1 GB for the OS and model weights overhead, then divides
        the remainder by *bytes_per_token* to yield a token count.

        Parameters
        ----------
        bytes_per_token : float
            Bytes consumed per token in the KV cache.  For a 7B model at
            INT8 KV (128 dim × 32 layers × 2 × 1 byte) = 8192 bytes.
            Default 512 is a conservative safe-default.

        Returns
        -------
        int
            Estimated maximum KV-cache tokens that fit in available RAM.
            Always ≥ 0.
        """
        _gib = 1 << 30
        headroom_gb = 1.0
        usable = max(0.0, self.available_gb - headroom_gb)
        return max(0, int(usable * _gib / bytes_per_token))

    def __repr__(self) -> str:
        snap = self.snapshot
        return (
            f"MemoryGovernor("
            f"level={snap.pressure_level}, "
            f"available={snap.available_gb:.1f}GB, "
            f"wired={snap.wired_gb:.1f}GB, "
            f"under_pressure={self.is_under_pressure})"
        )
