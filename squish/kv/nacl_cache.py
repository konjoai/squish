"""squish/kv/nacl_cache.py

NaCLCache — Non-Attention Cache with Lossless random eviction + reserve.

KV cache management policy that combines:

* **Anchor tokens** (first ``k_anchor`` positions): never evicted.
* **Recent tokens** (last ``k_recent`` positions): never evicted.
* **Middle tokens**: subject to uniform random eviction when budget is exceeded.

O(1) eviction decision per token.  The non-evictable reserve (anchors +
recents) guarantees that critical context (BOS tokens, immediately preceding
tokens) is always present.

Reference
---------
Xu et al., "NaCl: Non-Attention Cache with Lossless Compression."
NeurIPS 2024. arXiv:2408.16527, 2024.
"""

from __future__ import annotations

__all__ = ["NaCLConfig", "NaCLState", "NaCLCache"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NaCLConfig:
    """Configuration for NaCLCache.

    Parameters
    ----------
    max_budget:
        Maximum number of KV entries retained at any time.
    k_anchor:
        Number of leading (anchor) tokens that are never evicted.
    k_recent:
        Number of trailing (recent) tokens that are never evicted.
    n_heads:
        Number of KV heads.
    head_dim:
        Per-head dimension.
    seed:
        RNG seed for random eviction.
    """

    max_budget: int = 512
    k_anchor: int = 4
    k_recent: int = 4
    n_heads: int = 8
    head_dim: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_budget < 1:
            raise ValueError("max_budget must be >= 1")
        if self.k_anchor < 0:
            raise ValueError("k_anchor must be >= 0")
        if self.k_recent < 0:
            raise ValueError("k_recent must be >= 0")
        if self.k_anchor + self.k_recent > self.max_budget:
            raise ValueError("k_anchor + k_recent must be <= max_budget")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class NaCLState:
    """Mutable state for NaCLCache.

    Attributes
    ----------
    K_cache:
        Key cache, shape ``(budget, n_heads, head_dim)``.
    V_cache:
        Value cache, shape ``(budget, n_heads, head_dim)``.
    n_tokens:
        Number of tokens currently stored (not exceeding max_budget).
    """

    K_cache: ndarray
    V_cache: ndarray
    n_tokens: int = 0

    @property
    def is_full(self) -> bool:
        """Return True when the cache has reached max capacity."""
        return self.n_tokens >= self.K_cache.shape[0]

    @property
    def utilization(self) -> float:
        """Fraction of cache slots occupied."""
        total = self.K_cache.shape[0]
        return self.n_tokens / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class NaCLCache:
    """NaCL KV cache with anchor + recent reserve and random eviction.

    Parameters
    ----------
    config:
        ``NaCLConfig`` instance.
    """

    def __init__(self, config: NaCLConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def new_state(self) -> NaCLState:
        """Create a fresh empty NaCLState."""
        cfg = self.config
        return NaCLState(
            K_cache=np.zeros((cfg.max_budget, cfg.n_heads, cfg.head_dim), dtype=np.float32),
            V_cache=np.zeros((cfg.max_budget, cfg.n_heads, cfg.head_dim), dtype=np.float32),
            n_tokens=0,
        )

    def update(self, k: ndarray, v: ndarray, state: NaCLState) -> NaCLState:
        """Append a new KV pair, evicting if necessary.

        Parameters
        ----------
        k:
            Key, shape ``(n_heads, head_dim)``.
        v:
            Value, shape ``(n_heads, head_dim)``.
        state:
            Current cache state.

        Returns
        -------
        Updated ``NaCLState``.
        """
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if k.shape != (self.config.n_heads, self.config.head_dim):
            raise ValueError(f"k must be ({self.config.n_heads}, {self.config.head_dim})")
        if v.shape != (self.config.n_heads, self.config.head_dim):
            raise ValueError(f"v must be ({self.config.n_heads}, {self.config.head_dim})")

        K_cache = state.K_cache.copy()
        V_cache = state.V_cache.copy()
        n = state.n_tokens

        if n < self.config.max_budget:
            # Still space — just append
            K_cache[n] = k
            V_cache[n] = v
            return NaCLState(K_cache=K_cache, V_cache=V_cache, n_tokens=n + 1)

        # Cache full — evict a random middle position
        evicted_state = self._evict_random(
            NaCLState(K_cache=K_cache, V_cache=V_cache, n_tokens=n)
        )
        K_cache = evicted_state.K_cache
        V_cache = evicted_state.V_cache
        n_after = evicted_state.n_tokens

        K_cache[n_after] = k
        V_cache[n_after] = v
        return NaCLState(K_cache=K_cache, V_cache=V_cache, n_tokens=n_after + 1)

    def evict_if_needed(self, state: NaCLState) -> NaCLState:
        """Evict one random middle token if cache is full."""
        if not state.is_full:
            return state
        return self._evict_random(state)

    def get_kv(self, state: NaCLState) -> Tuple[ndarray, ndarray]:
        """Return the currently stored K and V tensors.

        Returns
        -------
        K:
            Shape ``(n_tokens, n_heads, head_dim)``.
        V:
            Shape ``(n_tokens, n_heads, head_dim)``.
        """
        n = state.n_tokens
        return state.K_cache[:n].copy(), state.V_cache[:n].copy()

    def _evict_random(self, state: NaCLState) -> NaCLState:
        """Evict a uniformly random non-reserved position.

        Reserved positions: first ``k_anchor`` and last ``k_recent`` of the
        currently stored tokens.  If no evictable positions exist, the oldest
        middle token (position k_anchor) is removed.
        """
        n = state.n_tokens
        ka = self.config.k_anchor
        kr = self.config.k_recent
        # Evictable indices: [k_anchor, n - k_recent)
        evict_start = ka
        evict_end = max(evict_start, n - kr)

        if evict_start >= evict_end:
            # No evictable middle — fallback: remove position k_anchor
            evict_idx = min(ka, n - 1)
        else:
            evict_idx = int(self._rng.integers(evict_start, evict_end))

        K_cache = state.K_cache.copy()
        V_cache = state.V_cache.copy()

        # Shift entries left to fill gap
        K_cache[evict_idx : n - 1] = K_cache[evict_idx + 1 : n]
        V_cache[evict_idx : n - 1] = V_cache[evict_idx + 1 : n]
        # Zero the vacated slot at position n-1
        K_cache[n - 1] = 0.0
        V_cache[n - 1] = 0.0

        return NaCLState(K_cache=K_cache, V_cache=V_cache, n_tokens=n - 1)
