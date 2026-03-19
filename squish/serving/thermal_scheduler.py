"""thermal_scheduler.py — Apple Silicon Thermal-Aware Dynamic Scheduler

Monitors the Apple Silicon chip's thermal state and dynamically adjusts
inference parameters (batch size, sequence budget, speculation) to keep the
chip responsive without triggering hard throttling.

Detection strategy (no root required):
  1. Primary: ``sysctl kern.thermstate`` (available on macOS 12+)
     Returns thermState 0-7 where 0=normal … 7=critical.
  2. Secondary: ``powermetrics -n 1 -i 100`` thermal prefix (if available /
     user has sudo).  Skipped in normal operation.
  3. Tertiary: EMA of observed compute latency as a proxy. Latency increases
     when the chip is thermally throttling.  This is always available and
     requires no OS-level privileges.

Throttle levels:
  NOMINAL  (thermState=0,1): 100% batch, speculation ON
  WARM     (thermState=2,3): 75% batch, speculation ON
  HOT      (thermState=4,5): 50% batch, speculation OFF
  CRITICAL (thermState=6,7): 25% batch, speculation OFF, drain only

The scheduler exposes a simple API consumed by the request queue:
    scheduler = ThermalScheduler()
    scheduler.record_step_latency(latency_ms)     # call after each forward
    params = scheduler.active_params()             # query current limits

Usage example:
    scheduler = ThermalScheduler(ThermalConfig(base_batch_size=8))
    for request in queue:
        params = scheduler.active_params()
        if params.batch_size == 0:
            time.sleep(0.1)   # critical — drain
            continue
        batch = queue.pop_batch(params.batch_size)
        t0 = time.perf_counter()
        result = model(batch, max_tokens=params.max_tokens)
        scheduler.record_step_latency((time.perf_counter() - t0) * 1000)
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ThermalState(IntEnum):
    """Discrete thermal operating levels."""
    NOMINAL = 0   # Full performance
    WARM = 1      # Minor throttle
    HOT = 2       # Moderate throttle
    CRITICAL = 3  # Severe throttle — danger zone


@dataclass
class ThermalConfig:
    """Configuration for ThermalScheduler.

    Args:
        base_batch_size:       Maximum batch size at NOMINAL thermal state.
        base_max_tokens:       Maximum new tokens per request at NOMINAL.
        latency_ema_alpha:     EMA smoothing factor for latency tracking.
                               Smaller value = slower adaptation.
        warm_lat_multiplier:   Latency ratio threshold for WARM state.
        hot_lat_multiplier:    Latency ratio threshold for HOT state.
        critical_lat_multiplier: Latency ratio threshold for CRITICAL state.
        sysctl_poll_interval:  Seconds between sysctl queries.
                               Set to 0 to disable sysctl (latency-only mode).
        speculation_enabled_states: Set of ThermalState levels that allow
                               speculative decoding.
    """
    base_batch_size: int = 8
    base_max_tokens: int = 512
    latency_ema_alpha: float = 0.1
    warm_lat_multiplier: float = 1.4
    hot_lat_multiplier: float = 2.0
    critical_lat_multiplier: float = 3.0
    sysctl_poll_interval: float = 5.0
    speculation_enabled_states: tuple = (ThermalState.NOMINAL, ThermalState.WARM)


@dataclass
class ThermalScheduleParams:
    """Active scheduling parameters at the current thermal state."""
    state: ThermalState
    batch_size: int
    max_tokens: int
    speculation_enabled: bool
    reduction_factor: float


_BATCH_REDUCTION: dict[ThermalState, float] = {
    ThermalState.NOMINAL: 1.00,
    ThermalState.WARM: 0.75,
    ThermalState.HOT: 0.50,
    ThermalState.CRITICAL: 0.25,
}

_TOKEN_REDUCTION: dict[ThermalState, float] = {
    ThermalState.NOMINAL: 1.00,
    ThermalState.WARM: 1.00,
    ThermalState.HOT: 0.75,
    ThermalState.CRITICAL: 0.50,
}


class ThermalScheduler:
    """Apple Silicon thermal-aware request scheduler.

    Thread-safe read of active_params(), single-writer record_step_latency().
    """

    def __init__(self, config: Optional[ThermalConfig] = None) -> None:
        self.config = config or ThermalConfig()
        self._state: ThermalState = ThermalState.NOMINAL
        self._baseline_latency_ms: Optional[float] = None
        self._ema_latency_ms: Optional[float] = None
        self._last_sysctl_ts: float = 0.0
        self._sysctl_latency_ms: float = 0.0
        self._total_observations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step_latency(self, latency_ms: float) -> ThermalState:
        """Feed observed forward-pass latency (ms) and update thermal state.

        Should be called after every model forward pass.

        Returns:
            The current ThermalState after the update.
        """
        if latency_ms <= 0.0:
            raise ValueError("latency_ms must be > 0")

        cfg = self.config
        # Update EMA
        alpha = cfg.latency_ema_alpha
        if self._ema_latency_ms is None:
            self._ema_latency_ms = latency_ms
        else:
            self._ema_latency_ms = (
                alpha * latency_ms + (1.0 - alpha) * self._ema_latency_ms
            )

        self._total_observations += 1

        # Establish baseline from first N observations (warm-up period)
        if self._total_observations <= 10:
            self._baseline_latency_ms = self._ema_latency_ms

        # Determine state from sysctl (polled) + latency EMA
        state_from_latency = self._state_from_latency()
        state_from_sysctl = self._poll_sysctl_state()
        self._state = ThermalState(
            max(int(state_from_latency), int(state_from_sysctl))
        )
        return self._state

    def active_params(self) -> ThermalScheduleParams:
        """Return scheduling parameters for the current thermal state."""
        cfg = self.config
        state = self._state
        rf = _BATCH_REDUCTION[state]
        tf = _TOKEN_REDUCTION[state]
        batch = max(1, math.floor(cfg.base_batch_size * rf))
        tokens = max(64, math.floor(cfg.base_max_tokens * tf))
        return ThermalScheduleParams(
            state=state,
            batch_size=batch,
            max_tokens=tokens,
            speculation_enabled=state in cfg.speculation_enabled_states,
            reduction_factor=rf,
        )

    def force_state(self, state: ThermalState) -> None:
        """Override the thermal state (useful for testing)."""
        self._state = state

    def reset(self) -> None:
        """Reset all statistics (useful between benchmark runs)."""
        self._state = ThermalState.NOMINAL
        self._baseline_latency_ms = None
        self._ema_latency_ms = None
        self._last_sysctl_ts = 0.0
        self._total_observations = 0

    @property
    def current_state(self) -> ThermalState:
        return self._state

    @property
    def ema_latency_ms(self) -> Optional[float]:
        return self._ema_latency_ms

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _state_from_latency(self) -> ThermalState:
        """Infer thermal state from latency degradation ratio."""
        if self._baseline_latency_ms is None or self._ema_latency_ms is None:
            return ThermalState.NOMINAL
        if self._baseline_latency_ms <= 0:
            return ThermalState.NOMINAL

        ratio = self._ema_latency_ms / self._baseline_latency_ms
        cfg = self.config
        if ratio >= cfg.critical_lat_multiplier:
            return ThermalState.CRITICAL
        elif ratio >= cfg.hot_lat_multiplier:
            return ThermalState.HOT
        elif ratio >= cfg.warm_lat_multiplier:
            return ThermalState.WARM
        return ThermalState.NOMINAL

    def _poll_sysctl_state(self) -> ThermalState:
        """Query macOS sysctl for thermal state (polled at configured interval).

        Returns ThermalState.NOMINAL if sysctl is unavailable or on error.
        This is intentionally non-blocking — we return cached value if the
        poll interval has not elapsed.
        """
        cfg = self.config
        if cfg.sysctl_poll_interval <= 0:
            return ThermalState.NOMINAL

        now = time.monotonic()
        if now - self._last_sysctl_ts < cfg.sysctl_poll_interval:
            # Return cached result
            return self._cached_sysctl_state()

        self._last_sysctl_ts = now
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.thermstate"],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if result.returncode == 0:
                raw = int(result.stdout.strip())
                # kern.thermstate: 0-1=nominal, 2-3=warm, 4-5=hot, 6-7=critical
                if raw >= 6:
                    self._sysctl_latency_ms = 6.0
                elif raw >= 4:
                    self._sysctl_latency_ms = 4.0
                elif raw >= 2:
                    self._sysctl_latency_ms = 2.0
                else:
                    self._sysctl_latency_ms = 0.0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            # sysctl not available or returned non-integer — ignore
            self._sysctl_latency_ms = 0.0

        return self._cached_sysctl_state()

    def _cached_sysctl_state(self) -> ThermalState:
        raw = self._sysctl_latency_ms
        if raw >= 6:
            return ThermalState.CRITICAL
        elif raw >= 4:
            return ThermalState.HOT
        elif raw >= 2:
            return ThermalState.WARM
        return ThermalState.NOMINAL

    def __repr__(self) -> str:
        ema_str = (
            f"{self._ema_latency_ms:.1f}ms"
            if self._ema_latency_ms is not None
            else "N/A"
        )
        return (
            f"ThermalScheduler(state={self._state.name}, "
            f"ema_lat={ema_str}, "
            f"obs={self._total_observations})"
        )
