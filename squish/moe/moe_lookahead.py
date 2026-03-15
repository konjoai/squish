# [Experimental] This module is part of Squish v6+ (Phase 14).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""MoE Lookahead Router — Phase 14.

Combines :class:`~squish.moe.sparse_moe.SparseMoERouter` with an
Exponential Moving Average (EMA) delta projection of the hidden state to
prefetch likely experts *before* the next forward call arrives.

The key idea:

1. After each :meth:`MoELookaheadRouter.route` call the module tracks the
   mean hidden state ``h_mean`` across the batch and maintains an EMA of the
   frame-to-frame delta ``delta = h_mean_t - h_mean_{t-1}``.
2. :meth:`~MoELookaheadRouter.predict_lookahead` projects the current hidden
   state ``k`` steps into the future (``h + k * ema_delta``) and routes each
   projection through the underlying sparse router to predict which experts
   will be needed.
3. :meth:`~MoELookaheadRouter.prefetch_set` returns the union of all predicted
   expert indices as a :class:`frozenset` and records it so that the *next*
   :meth:`~MoELookaheadRouter.route` call can evaluate the prefetch hit rate.

Usage::

    import numpy as np
    from squish.moe.moe_lookahead import MoELookaheadConfig, MoELookaheadRouter

    cfg    = MoELookaheadConfig(n_experts=8, top_k=2, hidden_dim=256)
    router = MoELookaheadRouter(cfg)

    rng           = np.random.default_rng(0)
    hidden_states = rng.standard_normal((16, 256)).astype(np.float32)

    # Pre-fetch experts for the *next* step
    pset = router.prefetch_set(hidden_states)
    print("prefetch set:", pset)

    # Route the actual step (evaluates hit rate automatically)
    indices, weights, aux_loss = router.route(hidden_states)
    print(router.stats())
"""

from __future__ import annotations

__all__ = ["MoELookaheadConfig", "MoELookaheadRouter"]

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from squish.moe.sparse_moe import MoEConfig, SparseMoERouter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoELookaheadConfig:
    """Configuration for :class:`MoELookaheadRouter`.

    Attributes:
        n_experts: Total number of expert networks.
        top_k: Number of experts each token is routed to.
        hidden_dim: Dimension of the input hidden states.
        lookahead_steps: Default number of future steps to predict.
        ema_alpha: Smoothing factor for the EMA delta update.  Must be in
            ``(0, 1]``; a value of ``1.0`` disables smoothing (raw delta).
        load_balance_weight: Coefficient for the auxiliary load-balancing
            loss passed through to the underlying :class:`SparseMoERouter`.
    """

    n_experts:           int   = 8
    top_k:               int   = 2
    hidden_dim:          int   = 256
    lookahead_steps:     int   = 3
    ema_alpha:           float = 0.1
    load_balance_weight: float = 0.01

    # Watchdog: auto-disable lookahead when rolling hit rate is persistently low.
    # Set watchdog_window=0 to disable the watchdog entirely.
    watchdog_window:             int   = 50    # rolling window size (route steps)
    watchdog_disable_threshold:  float = 0.40  # disable when rolling hit rate < this
    watchdog_enable_threshold:   float = 0.60  # re-enable when rolling hit rate >= this

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(
                f"top_k must be >= 1; got {self.top_k}"
            )
        if self.n_experts < self.top_k:
            raise ValueError(
                f"n_experts ({self.n_experts}) must be >= top_k ({self.top_k})"
            )
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1; got {self.hidden_dim}"
            )
        if self.lookahead_steps < 1:
            raise ValueError(
                f"lookahead_steps must be >= 1; got {self.lookahead_steps}"
            )
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1]; got {self.ema_alpha}"
            )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class MoELookaheadRouter:
    """Top-k sparse MoE router with EMA-delta hidden-state lookahead.

    The router wraps a :class:`~squish.moe.sparse_moe.SparseMoERouter` and
    augments it with:

    * **EMA tracking** — a running estimate of how the mean hidden state
      changes between consecutive ``route()`` calls.
    * **Lookahead prediction** — project the current hidden state forward by
      ``k * ema_delta`` for ``k = 1 … steps`` and route each projection to
      predict future expert demand.
    * **Prefetch hit-rate accounting** — track how often a prefetch set
      predicted by :meth:`prefetch_set` actually contained the experts
      selected by the following :meth:`route` call.

    Args:
        config: :class:`MoELookaheadConfig` controlling all behaviour.
    """

    def __init__(self, config: MoELookaheadConfig) -> None:
        self._config = config

        # Build underlying sparse router from a compatible MoEConfig
        sparse_cfg = MoEConfig(
            n_experts=config.n_experts,
            top_k=config.top_k,
            hidden_dim=config.hidden_dim,
            load_balance_weight=config.load_balance_weight,
        )
        self._router = SparseMoERouter(sparse_cfg)

        # EMA state — None until the first route() call has been made
        self._prev_h:    Optional[np.ndarray] = None   # shape (hidden_dim,)
        self._ema_delta: Optional[np.ndarray] = None   # shape (hidden_dim,)

        # Pending prefetch set — set by prefetch_set(), consumed by route()
        self._pending_prefetch: Optional[frozenset] = None

        # Hit-rate and prefetch-size counters
        self._prefetch_hits:    int = 0
        self._prefetch_total:   int = 0
        self._prefetch_set_sum: int = 0
        self._prefetch_calls:   int = 0

        # Watchdog: rolling window of per-step hit fractions (0.0–1.0 each)
        _w = config.watchdog_window
        self._watchdog_window: deque[float] = deque(maxlen=_w) if _w > 0 else deque(maxlen=1)
        self._watchdog_active: bool = _w > 0   # False when window=0 means watchdog off
        self._lookahead_disabled: bool = False  # toggled by watchdog

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_2d(self, hidden_states: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return *(arr_2d, was_1d)* — always yields a 2-D array."""
        arr = np.asarray(hidden_states, dtype=np.float32)
        if arr.ndim == 1:
            return arr[np.newaxis, :], True
        return arr, False

    def _update_ema(self, h_mean: np.ndarray) -> None:
        """Update ``_ema_delta`` and ``_prev_h`` given the current mean hidden
        state *h_mean* (shape ``(hidden_dim,)``)."""
        alpha = self._config.ema_alpha
        if self._prev_h is None:
            # First call: no history yet — delta initialised to zero.
            # _ema_delta is guaranteed to be None here (same lifecycle as
            # _prev_h), so we simply zero-initialise it.
            self._ema_delta = np.zeros(self._config.hidden_dim, dtype=np.float32)
        else:
            # _ema_delta is always set on the same call that first sets
            # _prev_h, so it is guaranteed non-None at this point.
            delta = h_mean - self._prev_h
            self._ema_delta = (
                alpha * delta + (1.0 - alpha) * self._ema_delta
            ).astype(np.float32)
        self._prev_h = h_mean.copy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        hidden_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Route hidden states to the top-k experts.

        Accepts both 1-D ``(hidden_dim,)`` and 2-D ``(batch, hidden_dim)``
        inputs.  The return shape mirrors the input shape: 1-D input yields
        1-D index/weight arrays; 2-D input yields 2-D arrays.

        Side effects (in order):

        1. If a pending prefetch set exists (set by a prior
           :meth:`prefetch_set` call) evaluate the hit rate against the
           actual expert selections and update counters.
        2. Update the EMA delta from the mean hidden state across the batch.

        Args:
            hidden_states: Float array of shape ``(hidden_dim,)`` or
                ``(batch, hidden_dim)``.

        Returns:
            Tuple ``(top_k_indices, top_k_weights, aux_loss)`` matching the
            input batch dimension.
        """
        arr_2d, was_1d = self._to_2d(hidden_states)

        indices, weights, aux_loss = self._router.route(arr_2d)

        # --- Evaluate pending prefetch before updating EMA -----------------
        if self._pending_prefetch is not None:
            actual_experts: set[int] = set(indices.flatten().tolist())
            hits = len(self._pending_prefetch & actual_experts)
            total = len(actual_experts)
            self._prefetch_hits  += hits
            self._prefetch_total += total
            self._pending_prefetch = None

            # Watchdog: record per-step hit fraction and check rolling mean
            if self._watchdog_active and total > 0:
                step_rate = hits / total
                self._watchdog_window.append(step_rate)
                cfg = self._config
                if len(self._watchdog_window) >= cfg.watchdog_window:
                    rolling_mean = sum(self._watchdog_window) / len(self._watchdog_window)
                    if not self._lookahead_disabled and rolling_mean < cfg.watchdog_disable_threshold:
                        self._lookahead_disabled = True
                    elif self._lookahead_disabled and rolling_mean >= cfg.watchdog_enable_threshold:
                        self._lookahead_disabled = False

        # --- Update EMA state -----------------------------------------------
        h_mean = arr_2d.mean(axis=0)  # (hidden_dim,)
        self._update_ema(h_mean)

        # --- Return matching shape ------------------------------------------
        if was_1d:
            return indices, weights, aux_loss
        return indices, weights, aux_loss

    def predict_lookahead(
        self,
        hidden_states: np.ndarray,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """Predict expert indices for the next *steps* forward passes.

        This method does **not** modify the EMA state or the pending-prefetch
        slot — it is a pure read-ahead query.

        Args:
            hidden_states: Float array of shape ``(hidden_dim,)`` or
                ``(batch, hidden_dim)``.
            steps: Number of lookahead steps.  Defaults to
                :attr:`MoELookaheadConfig.lookahead_steps`.

        Returns:
            Int64 array of shape ``(steps, batch, top_k)`` where ``batch=1``
            when the input was 1-D.
        """
        if steps is None:
            steps = self._config.lookahead_steps

        arr_2d, _ = self._to_2d(hidden_states)
        batch = arr_2d.shape[0]

        # Use zero delta when no EMA history is available yet
        delta = (
            self._ema_delta
            if self._ema_delta is not None
            else np.zeros(self._config.hidden_dim, dtype=np.float32)
        )

        results: list[np.ndarray] = []
        for k in range(1, steps + 1):
            projected = (arr_2d + k * delta).astype(np.float32)
            # Call the underlying router directly to avoid EMA/prefetch
            # side-effects that self.route() would trigger.
            indices_k, _weights_k, _loss_k = self._router.route(projected)
            results.append(indices_k)  # (batch, top_k)

        return np.stack(results, axis=0)  # (steps, batch, top_k)

    def prefetch_set(
        self,
        hidden_states: np.ndarray,
        steps: Optional[int] = None,
    ) -> frozenset:
        """Return the union of all expert indices predicted across all steps.

        The result is stored internally so that the *next* call to
        :meth:`route` can evaluate the prefetch hit rate automatically.

        Args:
            hidden_states: Float array of shape ``(hidden_dim,)`` or
                ``(batch, hidden_dim)``.
            steps: Number of lookahead steps.  Defaults to
                :attr:`MoELookaheadConfig.lookahead_steps`.

        Returns:
            A :class:`frozenset` of integer expert indices.
        """
        lookahead = self.predict_lookahead(hidden_states, steps=steps)
        # lookahead shape: (steps, batch, top_k)
        pset: frozenset
        if self._lookahead_disabled:
            # Watchdog has disabled lookahead — return empty set, don't record
            return frozenset()
        pset = frozenset(int(x) for x in lookahead.flatten().tolist())

        self._pending_prefetch = pset
        self._prefetch_set_sum += len(pset)
        self._prefetch_calls   += 1

        return pset

    def reset(self) -> None:
        """Reset EMA state and pending prefetch.

        The route-call statistics (total_route_calls, total_tokens) and
        hit-rate counters are preserved across resets.  The watchdog state
        (rolling window, disabled flag) is also preserved across resets so
        that repeated context resets don't re-enable a legitimately disabled
        lookahead.  Call :meth:`reset_watchdog` to explicitly clear watchdog
        state.
        """
        self._prev_h           = None
        self._ema_delta        = None
        self._pending_prefetch = None

    def reset_watchdog(self) -> None:
        """Clear watchdog state and re-enable lookahead.

        Call this after a routing-quality improvement (e.g. load a new model,
        change task distribution) to give the watchdog a fresh start.
        """
        self._watchdog_window.clear()
        self._lookahead_disabled = False

    @property
    def lookahead_disabled(self) -> bool:
        """True when the watchdog has disabled lookahead due to low hit rate."""
        return self._lookahead_disabled

    def stats(self) -> dict:
        """Return a snapshot of routing and prefetch-hit statistics.

        Returns:
            A dict with the following keys:

            * ``hit_rate`` (float) — fraction of actual expert selections that
              appeared in the most recent prefetch set.  Returns ``-1.0`` when
              no measurements have been taken yet.
            * ``prefetch_hits`` (int) — cumulative number of hit experts.
            * ``prefetch_total`` (int) — cumulative number of actual expert
              selections evaluated against prefetch sets.
            * ``avg_prefetch_set_size`` (float) — mean size of prefetch sets
              returned by :meth:`prefetch_set`.  ``0.0`` when no calls have
              been made.
            * ``prefetch_calls`` (int) — number of :meth:`prefetch_set` calls.
            * ``total_route_calls`` (int) — total :meth:`route` calls delegated
              to the underlying :class:`SparseMoERouter`.
            * ``total_tokens`` (int) — total tokens routed.
        """
        hit_rate: float
        if self._prefetch_total == 0:
            hit_rate = -1.0
        else:
            hit_rate = self._prefetch_hits / self._prefetch_total

        avg_prefetch_set_size: float
        if self._prefetch_calls == 0:
            avg_prefetch_set_size = 0.0
        else:
            avg_prefetch_set_size = self._prefetch_set_sum / self._prefetch_calls

        sparse_stats = self._router.stats
        rolling_mean: float
        if len(self._watchdog_window) == 0:
            rolling_mean = -1.0
        else:
            rolling_mean = sum(self._watchdog_window) / len(self._watchdog_window)
        return {
            "hit_rate":              hit_rate,
            "prefetch_hits":         self._prefetch_hits,
            "prefetch_total":        self._prefetch_total,
            "avg_prefetch_set_size": avg_prefetch_set_size,
            "prefetch_calls":        self._prefetch_calls,
            "total_route_calls":     sparse_stats.total_route_calls,
            "total_tokens":          sparse_stats.total_tokens,
            "watchdog_rolling_hit_rate": rolling_mean,
            "lookahead_disabled":    self._lookahead_disabled,
        }
