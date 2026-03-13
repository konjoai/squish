#!/usr/bin/env python3
"""
squish/power_monitor.py

macOS power-source detection and mode-based server configuration.

Provides:
    PowerMonitor — polls ``pmset`` every *poll_interval_s* seconds to track
                   whether the Mac is on AC or battery power and at what level.

    PowerModeConfig — frozen dataclass holding per-mode resource limits.

    POWER_CONFIGS — ``dict[str, PowerModeConfig]`` with three predefined
                    profiles: ``"performance"``, ``"balanced"``, ``"battery"``.

    apply_mode(mode_str, server_globals) — mutates a dict of server module
        globals to enforce the chosen mode's limits.  Used by server.py startup
        and by the auto-mode background timer.

Non-macOS platforms
───────────────────
``get_power_source()`` and ``get_battery_level()`` return safe defaults
(``"ac"`` / ``1.0``) when ``pmset`` is unavailable so the server starts cleanly
on Linux CI runners.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass

# ── Per-mode resource profiles ────────────────────────────────────────────────

@dataclass(frozen=True)
class PowerModeConfig:
    """Immutable collection of per-mode inference resource limits."""

    eagle_k_draft: int          # number of speculative draft tokens
    batch_window_ms: float      # batch coalescing window (milliseconds)
    metal_cache_fraction: float # Metal GPU cache cap as fraction of total
    thinking_budget: int        # Qwen3 thinking token budget (-1 = unlimited)
    chunk_prefill_threshold: int  # min seq_len to trigger chunked prefill


POWER_CONFIGS: dict[str, PowerModeConfig] = {
    "performance": PowerModeConfig(
        eagle_k_draft=5,
        batch_window_ms=20.0,
        metal_cache_fraction=0.25,
        thinking_budget=-1,
        chunk_prefill_threshold=512,
    ),
    "balanced": PowerModeConfig(
        eagle_k_draft=3,
        batch_window_ms=30.0,
        metal_cache_fraction=0.20,
        thinking_budget=256,
        chunk_prefill_threshold=512,
    ),
    "battery": PowerModeConfig(
        eagle_k_draft=2,
        batch_window_ms=50.0,
        metal_cache_fraction=0.15,
        thinking_budget=64,
        chunk_prefill_threshold=256,
    ),
}


def apply_mode(mode_str: str, server_globals: dict) -> None:
    """
    Mutate *server_globals* to enforce the resource limits of *mode_str*.

    This is intentionally a thin mapping — server.py reads the values written
    here at each decode step, so the mutation is immediately effective for the
    next request.

    Parameters
    ----------
    mode_str:
        One of ``"performance"``, ``"balanced"``, or ``"battery"``.
        Unknown keys are silently ignored so callers never crash on bad input.
    server_globals:
        A ``dict`` whose keys match the server.py module-level global names
        that correspond to the config fields.  Typically ``globals()`` from
        server.py, but any dict works (enabling unit tests without a running
        server).
    """
    cfg = POWER_CONFIGS.get(mode_str)
    if cfg is None:
        return
    server_globals["_thinking_budget"]          = cfg.thinking_budget
    server_globals["_chunk_prefill_threshold"]  = cfg.chunk_prefill_threshold


# ── macOS pmset wrapper ───────────────────────────────────────────────────────

class PowerMonitor:
    """
    Polls ``pmset -g batt`` on macOS to determine power source and battery
    charge level.

    Thread-safe: ``_lock`` protects the cached pmset output.  Background
    polling is **not** started automatically — call :meth:`start_polling` when
    the server is ready.

    On non-macOS or when ``pmset`` is not found the methods return safe
    defaults (``"ac"`` / ``1.0``) so the server still starts.
    """

    def __init__(self, poll_interval_s: float = 30.0) -> None:
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._raw: str = ""          # last pmset output
        self._last_poll: float = 0.0
        self._timer: threading.Timer | None = None
        # Eagerly fetch once so callers get real data immediately.
        self._refresh()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Run ``pmset -g batt`` and cache the output."""
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            with self._lock:
                self._raw = result.stdout
                self._last_poll = time.monotonic()
        except Exception:
            # pmset unavailable (Linux, timeout, permission) → keep previous cache
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def get_power_source(self) -> str:
        """
        Return ``"battery"`` when running on battery, ``"ac"`` otherwise.

        Parses the ``Now drawing from '...'`` line from pmset output.
        """
        with self._lock:
            raw = self._raw
        if "Battery Power" in raw:
            return "battery"
        return "ac"

    def get_battery_level(self) -> float:
        """
        Return battery charge as a fraction between 0.0 and 1.0.

        Returns ``1.0`` when on AC power or when the level cannot be parsed.
        """
        with self._lock:
            raw = self._raw
        m = re.search(r"(\d+)%", raw)
        if m:
            return int(m.group(1)) / 100.0
        return 1.0

    def get_recommended_mode(self) -> str:
        """
        Return the recommended ``PowerModeConfig`` key for current conditions.

        Rules
        ──────
        * ``"performance"`` — on AC power
        * ``"balanced"``    — on battery, level ≥ 50%
        * ``"battery"``     — on battery, level < 50%
        """
        source = self.get_power_source()
        if source != "battery":
            return "performance"
        level = self.get_battery_level()
        return "battery" if level < 0.5 else "balanced"

    # ── Background polling ────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """
        Start a recurring background refresh.

        Schedules :meth:`_refresh` every ``poll_interval_s`` seconds using a
        ``threading.Timer`` chain.  Safe to call multiple times — if a timer is
        already running this is a no-op.
        """
        if self._timer is not None:
            return
        self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._poll_interval_s, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        """Timer callback: refresh then schedule the next tick."""
        self._refresh()
        self._schedule_next()

    def stop_polling(self) -> None:
        """Cancel any pending background refresh timer."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
