#!/usr/bin/env python3
"""
squish/cost_estimator.py

CostEstimator — Per-request compute cost estimation for billing.

Each inference request has a compute cost proportional to three factors:

  * **Prefill cost** — proportional to the number of prompt tokens, which
    must all be processed in a single parallel forward pass.
  * **Decode cost** — proportional to the number of generated tokens, each
    requiring an incremental forward pass.
  * **KV memory cost** — proportional to the product of KV memory consumed
    (in MB) and the duration of that occupancy (in milliseconds), reflecting
    the opportunity cost of holding memory that other requests cannot use.

:class:`RequestCostEstimator` accumulates per-request costs so that billing
summaries can be produced at any point.

Example usage::

    from squish.cost_estimator import RequestCostEstimator, CostModel

    model = CostModel(
        prefill_cost_per_token=0.001,
        decode_cost_per_token=0.002,
        kv_cost_per_mb_ms=0.0001,
        currency="credits",
    )
    estimator = RequestCostEstimator(model)

    cost = estimator.estimate(
        request_id="req-001",
        n_prefill_tokens=512,
        n_decode_tokens=128,
        kv_mb=64.0,
        duration_ms=350.0,
    )
    print(f"total={cost.total_cost:.4f} {model.currency}")
    print(estimator.stats)
"""

from __future__ import annotations

__all__ = [
    "CostModel",
    "RequestCost",
    "RequestCostEstimator",
    "CostStats",
]

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """Pricing parameters for per-request cost estimation.

    All cost fields are in arbitrary *currency* units.

    Attributes:
        prefill_cost_per_token:  Cost per prompt token processed.
        decode_cost_per_token:   Cost per generated token decoded.
        kv_cost_per_mb_ms:       Cost per megabyte of KV memory per millisecond
                                 of request lifetime.
        currency:                Human-readable currency label (default "credits").
    """

    prefill_cost_per_token: float = 0.001
    decode_cost_per_token: float = 0.002
    kv_cost_per_mb_ms: float = 0.0001
    currency: str = "credits"

    def __post_init__(self) -> None:
        if self.prefill_cost_per_token < 0.0:
            raise ValueError(
                f"prefill_cost_per_token must be >= 0, "
                f"got {self.prefill_cost_per_token}"
            )
        if self.decode_cost_per_token < 0.0:
            raise ValueError(
                f"decode_cost_per_token must be >= 0, "
                f"got {self.decode_cost_per_token}"
            )
        if self.kv_cost_per_mb_ms < 0.0:
            raise ValueError(
                f"kv_cost_per_mb_ms must be >= 0, got {self.kv_cost_per_mb_ms}"
            )
        if not self.currency:
            raise ValueError("currency must be a non-empty string")


# ---------------------------------------------------------------------------
# Request cost
# ---------------------------------------------------------------------------

@dataclass
class RequestCost:
    """Cost breakdown for a single inference request.

    Attributes:
        request_id:   Caller-supplied request identifier.
        prefill_cost: Cost attributed to prompt-token processing.
        decode_cost:  Cost attributed to generated-token decoding.
        kv_cost:      Cost attributed to KV-memory occupancy over time.
    """

    request_id: str
    prefill_cost: float
    decode_cost: float
    kv_cost: float

    @property
    def total_cost(self) -> float:
        """Sum of prefill, decode, and KV memory costs."""
        return self.prefill_cost + self.decode_cost + self.kv_cost


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CostStats:
    """Cumulative billing statistics collected by :class:`RequestCostEstimator`.

    Attributes:
        total_requests:      Number of requests estimated so far.
        total_cost:          Aggregate cost across all requests.
        total_prefill_tokens: Cumulative prefill tokens across all requests.
        total_decode_tokens:  Cumulative decode tokens across all requests.
    """

    total_requests: int = 0
    total_cost: float = 0.0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0

    @property
    def avg_cost_per_request(self) -> float:
        """Average cost per request.

        Returns 0.0 when no requests have been estimated yet.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class RequestCostEstimator:
    """Stateful per-request cost estimator.

    Computes the cost of each inference request from its token counts, KV
    memory consumption, and wall-clock duration, then accumulates those
    costs in :attr:`stats`.

    Args:
        model: A :class:`CostModel` instance with pricing parameters.
    """

    def __init__(self, model: CostModel) -> None:
        self._model = model
        self._stats = CostStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        request_id: str,
        n_prefill_tokens: int,
        n_decode_tokens: int,
        kv_mb: float,
        duration_ms: float,
    ) -> RequestCost:
        """Estimate the cost of a completed inference request.

        Args:
            request_id:       Caller-supplied unique request identifier.
            n_prefill_tokens: Number of prompt tokens processed (>= 0).
            n_decode_tokens:  Number of tokens generated (>= 0).
            kv_mb:            Peak KV memory consumed by this request in MB
                              (>= 0).
            duration_ms:      End-to-end request wall-clock time in
                              milliseconds (>= 0).

        Returns:
            A :class:`RequestCost` with the full cost breakdown.

        Raises:
            ValueError: if any numeric argument is negative or
                        ``request_id`` is empty.
        """
        if not request_id:
            raise ValueError("request_id must be a non-empty string")
        if n_prefill_tokens < 0:
            raise ValueError(
                f"n_prefill_tokens must be >= 0, got {n_prefill_tokens}"
            )
        if n_decode_tokens < 0:
            raise ValueError(
                f"n_decode_tokens must be >= 0, got {n_decode_tokens}"
            )
        if kv_mb < 0.0:
            raise ValueError(f"kv_mb must be >= 0, got {kv_mb}")
        if duration_ms < 0.0:
            raise ValueError(
                f"duration_ms must be >= 0, got {duration_ms}"
            )

        m = self._model
        prefill_cost = n_prefill_tokens * m.prefill_cost_per_token
        decode_cost = n_decode_tokens * m.decode_cost_per_token
        kv_cost = kv_mb * duration_ms * m.kv_cost_per_mb_ms

        cost = RequestCost(
            request_id=request_id,
            prefill_cost=prefill_cost,
            decode_cost=decode_cost,
            kv_cost=kv_cost,
        )

        # Update cumulative stats.
        self._stats.total_requests += 1
        self._stats.total_cost += cost.total_cost
        self._stats.total_prefill_tokens += n_prefill_tokens
        self._stats.total_decode_tokens += n_decode_tokens

        return cost

    @property
    def stats(self) -> CostStats:
        """Cumulative billing statistics (updated in place)."""
        return self._stats
