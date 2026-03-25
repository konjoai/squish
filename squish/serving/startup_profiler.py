"""squish/serving/startup_profiler.py — Startup timing and phase profiling.

Measures wall-clock time for each major initialization phase so slow startup
can be diagnosed and optimized.  Enabled by setting ``SQUISH_TRACE_STARTUP=1``.

Public API
──────────
StartupPhase        — enum of named startup phases
StartupTimer        — context manager that records a phase duration
StartupReport       — aggregates timings; ``slowest(n)`` + ``to_dict()``
measure_import_ms   — time a fresh module import (for import-chain analysis)
_global_report      — module-level report written to by server.py
"""
from __future__ import annotations

__all__ = [
    "StartupPhase",
    "StartupTimer",
    "StartupReport",
    "measure_import_ms",
]

import os
import time
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# StartupPhase
# ---------------------------------------------------------------------------

class StartupPhase(str, Enum):
    """Named phases of the server startup sequence."""
    IMPORTS         = "imports"
    CONFIG          = "config"
    HW_DETECT       = "hw_detect"
    MODEL_LOAD      = "model_load"
    KV_CACHE_INIT   = "kv_cache_init"
    METAL_WARMUP    = "metal_warmup"
    DRAFT_HEAD      = "draft_head"
    HTTP_BIND       = "http_bind"
    OTHER           = "other"


# ---------------------------------------------------------------------------
# Internal timing entry
# ---------------------------------------------------------------------------

class _Entry:
    __slots__ = ("phase", "label", "start_ms", "end_ms")

    def __init__(self, phase: str, label: str, start_ms: float) -> None:
        self.phase    = phase
        self.label    = label
        self.start_ms = start_ms
        self.end_ms   = start_ms  # updated on __exit__

    @property
    def elapsed_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {
            "phase":      self.phase,
            "label":      self.label,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


# ---------------------------------------------------------------------------
# StartupReport
# ---------------------------------------------------------------------------

class StartupReport:
    """Accumulates :class:`_Entry` objects from :class:`StartupTimer`.

    Parameters
    ----------
    enabled:
        If ``False`` all timers are no-ops (startup overhead is zero).
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._entries: list[_Entry] = []

    def add(self, entry: _Entry) -> None:
        if self._enabled:
            self._entries.append(entry)

    @property
    def total_ms(self) -> float:
        """Total elapsed time across all recorded entries."""
        return float(sum(e.elapsed_ms for e in self._entries))

    def slowest(self, n: int = 5) -> list[_Entry]:
        """Return the *n* slowest entries sorted descending by elapsed_ms."""
        return sorted(self._entries, key=lambda e: e.elapsed_ms, reverse=True)[:n]

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict suitable for the ``/v1/startup-profile`` endpoint."""
        entries = [e.to_dict() for e in self._entries]
        return {
            "enabled":    self._enabled,
            "total_ms":   round(self.total_ms, 3),
            "phase_count": len(self._entries),
            "entries":    entries,
            "slowest_5":  [e.to_dict() for e in self.slowest(5)],
        }


# ---------------------------------------------------------------------------
# StartupTimer — context manager
# ---------------------------------------------------------------------------

class StartupTimer:
    """Context manager that records a startup phase into a :class:`StartupReport`.

    Usage::

        report = StartupReport()
        with StartupTimer(report, StartupPhase.MODEL_LOAD, "Load qwen3:8b"):
            ... load model ...

    When the report is disabled (``enabled=False``) this is a no-op.
    """

    def __init__(
        self,
        report: StartupReport,
        phase: str | StartupPhase,
        label: str = "",
    ) -> None:
        self._report = report
        self._phase  = phase.value if isinstance(phase, StartupPhase) else str(phase)
        self._label  = label or self._phase
        self._entry: Optional[_Entry] = None

    def __enter__(self) -> "StartupTimer":
        if self._report._enabled:
            self._entry = _Entry(
                phase=self._phase,
                label=self._label,
                start_ms=time.monotonic() * 1000,
            )
        return self

    def __exit__(self, *_) -> None:
        if self._entry is not None:
            self._entry.end_ms = time.monotonic() * 1000
            self._report.add(self._entry)
        return None  # do not suppress exceptions


# ---------------------------------------------------------------------------
# measure_import_ms
# ---------------------------------------------------------------------------

def measure_import_ms(module_name: str) -> float:
    """Return the time in milliseconds to import *module_name* from scratch.

    If the module is already in ``sys.modules`` (cached) the function returns
    ``0.0`` without unloading the module (unloading is dangerous).

    Parameters
    ----------
    module_name:
        Fully-qualified module name, e.g. ``"squish.catalog"``.

    Returns
    -------
    float
        Import time in milliseconds, or ``0.0`` if already imported.
    """
    import sys
    import importlib

    if module_name in sys.modules:
        return 0.0

    start = time.monotonic()
    try:
        importlib.import_module(module_name)
    except ImportError:
        return 0.0
    return (time.monotonic() - start) * 1000


# ---------------------------------------------------------------------------
# Module-level global report (written to by server.py if SQUISH_TRACE_STARTUP=1)
# ---------------------------------------------------------------------------

_ENABLED = os.environ.get("SQUISH_TRACE_STARTUP", "0") not in ("0", "", "false", "no")
_global_report = StartupReport(enabled=_ENABLED)
