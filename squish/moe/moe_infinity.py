"""squish/moe/moe_infinity.py

MoEInfinityOffload — Activation-Pattern Expert Prefetch for Offloaded MoE.

Reference
---------
Xiao et al. "MoE-Infinity: Offloading-Efficient MoE Model Serving."
arXiv 2401.14361 (2024).

Algorithm
---------
Expert weights are kept on CPU ("host") memory and loaded to GPU ("device")
on demand.  The key insight is that expert activation patterns repeat
predictably across tokens, so predictions derived from router logits can
drive *prefetch*: experts likely to be used in the next step are
transferred before their turn, hiding transfer latency.

This module simulates CPU/GPU transfer with Python dictionaries and tracks:

* ``prefetch_hit_rate`` — fraction of forward calls that found the expert
  already on device from a prior prefetch.
* ``n_on_device`` — number of experts currently residing in the device cache.

Integration points:

1. ``store_expert(expert_id, weight)`` — register an expert on "CPU".
2. ``prefetch(expert_ids)`` — move a list of experts to "device".
3. ``predict_next_experts(router_logits, k)`` — top-k routing prediction.
4. ``forward(token_hidden, expert_id)`` — run one expert; auto-loads if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "MoEInfinityConfig",
    "MoEInfinityOffload",
]


@dataclass
class MoEInfinityConfig:
    """Configuration for :class:`MoEInfinityOffload`.

    Attributes:
        n_experts: Total expert count.
        expert_dim: Input/output dimension for each expert.
        hidden_dim: Hidden dimension inside each expert FFN.
        top_k: Default top-k for routing.
        seed: RNG seed.
    """

    n_experts: int = 8
    expert_dim: int = 512
    hidden_dim: int = 2048
    top_k: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1; got {self.n_experts}")
        if self.expert_dim < 1:
            raise ValueError(f"expert_dim must be >= 1; got {self.expert_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {self.hidden_dim}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")


class MoEInfinityOffload:
    """Offloaded Mixture-of-Experts with activation-pattern expert prefetch.

    Expert weights live on "CPU" (a plain dict) and are loaded to "device"
    (another dict) on demand or via explicit ``prefetch()``.

    Example::

        cfg = MoEInfinityConfig(n_experts=4, expert_dim=16, hidden_dim=64)
        moe = MoEInfinityOffload(cfg)

        rng = np.random.default_rng(0)
        for i in range(cfg.n_experts):
            w = rng.standard_normal((cfg.expert_dim, cfg.hidden_dim)).astype(np.float32)
            moe.store_expert(i, w)

        token = rng.standard_normal(cfg.expert_dim).astype(np.float32)
        out = moe.forward(token, expert_id=0)
    """

    def __init__(self, config: Optional[MoEInfinityConfig] = None) -> None:
        self._cfg = config or MoEInfinityConfig()
        # CPU storage: expert_id -> (expert_dim, hidden_dim) weight matrix
        self._cpu_store: Dict[int, np.ndarray] = {}
        # Device cache: expert_id -> (expert_dim, hidden_dim)
        self._device_cache: Dict[int, np.ndarray] = {}
        self._total_forwards: int = 0
        self._prefetch_hits: int = 0

    @property
    def config(self) -> MoEInfinityConfig:
        return self._cfg

    @property
    def n_on_device(self) -> int:
        """Number of experts currently in the device cache."""
        return len(self._device_cache)

    @property
    def prefetch_hit_rate(self) -> float:
        """Fraction of ``forward()`` calls that found the expert on device.

        Returns 0.0 if ``forward()`` has never been called.
        """
        if self._total_forwards == 0:
            return 0.0
        return self._prefetch_hits / self._total_forwards

    def store_expert(self, expert_id: int, weight: np.ndarray) -> None:
        """Register an expert weight on CPU.

        Args:
            expert_id: Integer in ``[0, n_experts)``.
            weight: ``(expert_dim, hidden_dim)`` weight matrix.
        """
        self._cpu_store[expert_id] = np.asarray(weight, dtype=np.float32)

    def prefetch(self, expert_ids: List[int]) -> None:
        """Pre-load experts from CPU to device.

        Experts already on device are left unchanged.

        Args:
            expert_ids: List of expert indices to load.
        """
        for eid in expert_ids:
            if eid not in self._device_cache and eid in self._cpu_store:
                self._device_cache[eid] = self._cpu_store[eid]

    def evict(self, expert_ids: List[int]) -> None:
        """Remove experts from the device cache.

        Useful for testing or bounded-cache simulations.
        """
        for eid in expert_ids:
            self._device_cache.pop(eid, None)

    def predict_next_experts(
        self, router_logits: np.ndarray, k: Optional[int] = None
    ) -> List[int]:
        """Return the top-k expert indices predicted by router logits.

        Args:
            router_logits: ``(n_experts,)`` or ``(n_tokens, n_experts)``.
                Multi-token logits are averaged across the token dimension.
            k: Number of experts; defaults to ``config.top_k``.

        Returns:
            Sorted list of ``k`` expert indices.
        """
        if k is None:
            k = self._cfg.top_k
        logits = np.asarray(router_logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits.mean(axis=0)
        k = min(k, self._cfg.n_experts)
        top_k_idx = np.argpartition(logits, -k)[-k:]
        return sorted(top_k_idx.tolist())

    def _load_expert(self, expert_id: int) -> np.ndarray:
        """Ensure an expert is on device and return its weight."""
        if expert_id not in self._device_cache:
            if expert_id not in self._cpu_store:
                raise KeyError(
                    f"Expert {expert_id} not found in CPU store; "
                    "call store_expert() first."
                )
            self._device_cache[expert_id] = self._cpu_store[expert_id]
        return self._device_cache[expert_id]

    def forward(self, token_hidden: np.ndarray, expert_id: int) -> np.ndarray:
        """Run a single expert on one token's hidden state.

        The expert is a single linear projection followed by ReLU:
        ``output = relu(token_hidden @ W)``.

        Args:
            token_hidden: ``(expert_dim,)`` input vector.
            expert_id: Which expert to run.

        Returns:
            ``(expert_dim,)`` output (projected back via transpose).
        """
        self._total_forwards += 1
        was_on_device = expert_id in self._device_cache
        W = self._load_expert(expert_id)
        if was_on_device:
            self._prefetch_hits += 1
        h = np.asarray(token_hidden, dtype=np.float32)
        hidden = np.maximum(0.0, h @ W)          # (hidden_dim,)
        out = hidden @ W.T                        # (expert_dim,)
        return out
