"""squish/serving/token_budget_scheduler.py

TokenBudgetScheduler — KV-Budget Preemption with CPU Swap.

Reference
---------
Han et al. "Token Budget Estimation for Transformer Inference."
(Also related to vLLM preemption & PagedAttention token budget management.)

Algorithm
---------
The token budget scheduler assigns each running request a *token budget* —
the maximum total KV slots it may occupy.  When the cluster-wide KV memory
is under pressure:

1. **Token pruning** — request's KV cache is compacted by evicting the
   lowest-importance tokens (ranked by accumulated attention weight).
2. **Swap to CPU** — if pruning is insufficient, the entire KV cache for a
   low-priority request is swapped to CPU DRAM.
3. **Re-admit** — swapped requests are brought back in FIFO order as memory
   allows.

Key properties
--------------
* ``register(request_id, max_tokens)`` — add a request with a token budget.
* ``record_attention(request_id, token_importance)`` — update importance.
* ``enforce(available_slots)`` → list of evicted (request_id, token_count) pairs.
* ``swap_out(request_id)`` / ``swap_in(request_id)`` — CPU offload simulation.
* NumPy-only; importance tracked per token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TokenBudgetSchedulerConfig",
    "RequestBudget",
    "TokenBudgetScheduler",
]


@dataclass
class TokenBudgetSchedulerConfig:
    """Configuration for :class:`TokenBudgetScheduler`.

    Attributes:
        total_kv_slots: Total KV token slots in the system.
        prune_fraction: Fraction of a request's tokens to evict on pressure.
        swap_threshold: KV occupancy fraction above which CPU swap starts.
    """

    total_kv_slots: int = 32768
    prune_fraction: float = 0.2
    swap_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in (0, 1)")
        if not 0.0 < self.swap_threshold <= 1.0:
            raise ValueError("swap_threshold must be in (0, 1]")


@dataclass
class RequestBudget:
    """Per-request KV token budget state.

    Attributes:
        request_id: Unique identifier.
        max_tokens: Assigned token budget.
        token_importance: Importance score per KV slot (higher = keep).
        swapped: Whether the KV cache is currently on CPU.
        priority: Lower = higher priority.
    """

    request_id: int
    max_tokens: int
    token_importance: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    swapped: bool = False
    priority: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.token_importance)


class TokenBudgetScheduler:
    """KV-budget-based token eviction and CPU-swap scheduler.

    Parameters
    ----------
    config:
        TokenBudgetSchedulerConfig.
    """

    def __init__(self, config: Optional[TokenBudgetSchedulerConfig] = None) -> None:
        self._cfg = config or TokenBudgetSchedulerConfig()
        self._requests: Dict[int, RequestBudget] = {}
        self._swapped_cpu: Dict[int, np.ndarray] = {}  # request_id → importance array

    @property
    def config(self) -> TokenBudgetSchedulerConfig:
        return self._cfg

    def register(self, request_id: int, max_tokens: int, priority: int = 0) -> None:
        """Register a new request with a token budget."""
        self._requests[request_id] = RequestBudget(
            request_id=request_id,
            max_tokens=max_tokens,
            priority=priority,
        )

    def record_attention(
        self, request_id: int, token_importance: np.ndarray
    ) -> None:
        """Update cumulative token importance scores.

        Parameters
        ----------
        token_importance:
            Scalar importance per KV token (length = current token count).
            Higher = more important to retain.
        """
        req = self._requests.get(request_id)
        if req is None:
            return
        req.token_importance = np.asarray(token_importance, dtype=np.float32)

    def total_kv_used(self) -> int:
        """Count total active KV slots across all non-swapped requests."""
        return sum(
            r.n_tokens for r in self._requests.values() if not r.swapped
        )

    def enforce(self, available_slots: Optional[int] = None) -> List[Tuple[int, int]]:
        """Run budget enforcement: prune low-importance tokens or swap requests.

        Parameters
        ----------
        available_slots:
            Override total_kv_slots; uses config value if None.

        Returns
        -------
        List of (request_id, n_tokens_freed) pairs for each eviction action.
        """
        total = available_slots or self._cfg.total_kv_slots
        evictions: List[Tuple[int, int]] = []
        occupancy = self.total_kv_used() / max(total, 1)

        if occupancy <= self._cfg.swap_threshold:
            return evictions

        # Sort candidates: highest priority (lowest value) last — evict low prio first
        candidates = sorted(
            [r for r in self._requests.values() if not r.swapped],
            key=lambda r: -r.priority,
        )
        for req in candidates:
            if self.total_kv_used() / total <= self._cfg.swap_threshold:
                break
            if req.n_tokens == 0:
                continue
            n_prune = max(1, int(req.n_tokens * self._cfg.prune_fraction))
            freed = self._prune_tokens(req, n_prune)
            if freed > 0:
                evictions.append((req.request_id, freed))
            # If still over threshold, swap the entire request
            if self.total_kv_used() / total > self._cfg.swap_threshold:
                self._swap_out(req)
                evictions.append((req.request_id, req.n_tokens))
        return evictions

    def swap_out(self, request_id: int) -> bool:
        """Manually swap a request's KV cache to CPU.

        Returns True if successful.
        """
        req = self._requests.get(request_id)
        if req is None or req.swapped:
            return False
        self._swap_out(req)
        return True

    def swap_in(self, request_id: int) -> bool:
        """Restore a swapped request's KV cache from CPU.

        Returns True if successful.
        """
        req = self._requests.get(request_id)
        if req is None or not req.swapped:
            return False
        cpu_imp = self._swapped_cpu.pop(request_id, None)
        if cpu_imp is not None:
            req.token_importance = cpu_imp
        req.swapped = False
        return True

    def _prune_tokens(self, req: RequestBudget, n_prune: int) -> int:
        """Remove the ``n_prune`` least important tokens from req.

        Returns number of tokens freed.
        """
        if req.n_tokens <= n_prune:
            req.token_importance = np.empty(0, dtype=np.float32)
            return req.n_tokens
        keep_idx = np.argsort(req.token_importance)[n_prune:]
        req.token_importance = req.token_importance[keep_idx]
        return n_prune

    def _swap_out(self, req: RequestBudget) -> None:
        self._swapped_cpu[req.request_id] = req.token_importance.copy()
        req.token_importance = np.empty(0, dtype=np.float32)
        req.swapped = True

    def unregister(self, request_id: int) -> None:
        """Remove a completed request."""
        self._requests.pop(request_id, None)
        self._swapped_cpu.pop(request_id, None)
