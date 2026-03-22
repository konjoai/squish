"""squish/serving/preble_router.py

PrebeleRouter — Prefix-cache-aware request routing for multi-instance serving.

Reference
---------
Zhang et al. "Preble: Efficient Distribution of Prompt Sharing LLM Serving."
arXiv:2407.00023, 2024.

Algorithm
---------
In a multi-instance deployment each server maintains its own KV cache.
Without coordination, a request that shares a long prefix with a cached
sequence on server A may be routed to server B, forcing B to recompute the
prefix from scratch.

Preble solves this with a stateful router that:

1. Maintains a KV-cache occupancy map per server — which token-prefix hashes
   are warm on which server.
2. For each incoming request, hashes its prefixes at multiple granularities
   (document chunk level, ~1K tokens).
3. Scores each candidate server by the fraction of prefix hashes that are
   already warm there.
4. Routes to the server with the highest KV overlap score (tie-break by
   current load).
5. Updates the occupancy map after routing.

Result: ~50% reduction in prefix recomputation at 4+ replica deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PrebeleConfig:
    """Configuration for PrebeleRouter.

    Parameters
    ----------
    n_servers:
        Number of backend servers / replicas.
    chunk_size:
        Number of tokens per prefix chunk (used for hash granularity).
    max_cache_entries_per_server:
        Maximum number of prefix hashes tracked per server.
    load_weight:
        Weight of load-balancing term vs KV-overlap term in routing score.
        0.0 = route purely by KV overlap; 1.0 = route purely by load.
    """

    n_servers: int = 4
    chunk_size: int = 64
    max_cache_entries_per_server: int = 10_000
    load_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.n_servers < 1:
            raise ValueError("n_servers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {self.chunk_size}")
        if self.load_weight < 0:
            raise ValueError(f"load_weight must be >= 0; got {self.load_weight}")


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """Result of a routing decision.

    Parameters
    ----------
    server_id:
        Index of the selected server (0-based).
    overlap_score:
        Fraction of request's prefix hashes already warm on the selected
        server (0.0–1.0).
    current_load:
        Number of in-flight requests on the selected server at decision time.
    """

    server_id: int
    overlap_score: float
    current_load: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class PrebeleRouter:
    """Prefix-cache-aware multi-server request router.

    Parameters
    ----------
    config:
        PrebeleRouter configuration.
    """

    def __init__(self, config: Optional[PrebeleConfig] = None) -> None:
        self._cfg = config or PrebeleConfig()
        n = self._cfg.n_servers
        # Set of warm prefix hashes per server
        self._cache_maps: list[set[int]] = [set() for _ in range(n)]
        # In-flight request count per server
        self._loads: list[int] = [0] * n

    @property
    def config(self) -> PrebeleConfig:
        return self._cfg

    @property
    def server_loads(self) -> list[int]:
        """Current in-flight request counts per server."""
        return list(self._loads)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hash_chunk(self, token_ids: list[int], start: int, end: int) -> int:
        """Hash a token sub-sequence to a stable integer."""
        return hash(tuple(token_ids[start:end]))

    def _prefix_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute chunk-level prefix hashes for a token sequence."""
        cs = self._cfg.chunk_size
        n = len(token_ids)
        hashes = []
        for start in range(0, n, cs):
            hashes.append(self._hash_chunk(token_ids, start, min(start + cs, n)))
        return hashes

    def _overlap_score(self, server_id: int, req_hashes: list[int]) -> float:
        """Fraction of request hashes that are warm on server_id."""
        if not req_hashes:
            return 0.0
        warm = sum(1 for h in req_hashes if h in self._cache_maps[server_id])
        return warm / len(req_hashes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, token_ids: list[int]) -> RouteResult:
        """Route a request to the best server.

        Parameters
        ----------
        token_ids:
            Input token ids for the incoming request.

        Returns
        -------
        RouteResult
        """
        req_hashes = self._prefix_hashes(token_ids)
        n = self._cfg.n_servers
        max_load = max(self._loads) + 1  # avoid division by zero

        best_server = 0
        best_score = -np.inf

        for s in range(n):
            overlap = self._overlap_score(s, req_hashes)
            normalized_load = self._loads[s] / max_load
            score = (1.0 - self._cfg.load_weight) * overlap - self._cfg.load_weight * normalized_load
            if score > best_score:
                best_score = score
                best_server = s

        # Register the request's hashes on the chosen server
        cache_map = self._cache_maps[best_server]
        for h in req_hashes:
            cache_map.add(h)
            # Evict oldest entries if over capacity (simple set truncation)
            if len(cache_map) > self._cfg.max_cache_entries_per_server:
                # Remove an arbitrary element (hash sets have no order, use pop)
                cache_map.pop()

        self._loads[best_server] += 1

        return RouteResult(
            server_id=best_server,
            overlap_score=max(0.0, float(self._overlap_score(best_server, req_hashes))),
            current_load=self._loads[best_server],
        )

    def complete_request(self, server_id: int) -> None:
        """Signal that a request on server_id has completed.

        Parameters
        ----------
        server_id:
            Server that finished processing.
        """
        self._loads[server_id] = max(0, self._loads[server_id] - 1)

    def warm_cache(self, server_id: int, token_ids: list[int]) -> None:
        """Manually mark a token sequence as warm on a specific server.

        Useful for pre-warming the router state when a server already has
        a KV sequence in its cache.

        Parameters
        ----------
        server_id:
            Server with the warm KV sequence.
        token_ids:
            Token ids of the cached prefix.
        """
        hashes = self._prefix_hashes(token_ids)
        for h in hashes:
            self._cache_maps[server_id].add(h)

    def cache_stats(self) -> list[int]:
        """Return the number of warm prefix hashes per server."""
        return [len(m) for m in self._cache_maps]
