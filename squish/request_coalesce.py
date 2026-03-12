#!/usr/bin/env python3
"""
squish/request_coalesce.py

RequestCoalesce — Merge requests sharing long common prefixes.

When multiple requests share a common prefix (e.g., the same system prompt or
few-shot examples), processing the shared prefix once and branching after it
saves prefill FLOPs proportional to the number of requests that share the prefix.

This module buffers incoming requests and, when :meth:`PrefixCoalescer.coalesce`
is called, groups them by their longest common prefix (LCP).  Requests are
clustered greedily: starting from the longest possible shared prefix, each
cluster collects all requests that share at least ``min_shared_tokens`` tokens
with the cluster's representative.  Requests that do not share enough tokens
with any existing cluster are placed in singleton groups.

The returned :class:`CoalesceGroup` objects describe the shared prefix (which is
prefilled once) and the per-request branch tokens (each prefilled from the
branching point).

Example usage::

    from squish.request_coalesce import CoalesceConfig, PrefixCoalescer

    config = CoalesceConfig(min_shared_tokens=4, max_group_size=4)
    coalescer = PrefixCoalescer(config)

    system = [10, 20, 30, 40, 50]
    coalescer.add_request("req-1", system + [100, 101])
    coalescer.add_request("req-2", system + [200, 201])
    coalescer.add_request("req-3", system + [300])

    groups = coalescer.coalesce()
    for g in groups:
        print(f"shared={len(g.shared_prefix)} tokens, requests={g.request_ids}")
    print(coalescer.stats)
"""

from __future__ import annotations

__all__ = [
    "CoalesceConfig",
    "CoalesceGroup",
    "PrefixCoalescer",
    "CoalesceStats",
]

from dataclasses import dataclass, field

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoalesceConfig:
    """Configuration for the :class:`PrefixCoalescer`.

    Attributes:
        min_shared_tokens: Minimum number of tokens two requests must share at
                           their leading prefix for them to be coalesced into
                           the same group.  Requests sharing fewer tokens are
                           placed in separate groups.  Must be >= 1.
        max_group_size:    Maximum number of requests in a single coalesced
                           group.  Once a group reaches this size, additional
                           requests that would otherwise qualify are placed in
                           new groups.  Must be >= 2.
    """

    min_shared_tokens: int = 32
    max_group_size: int = 8

    def __post_init__(self) -> None:
        if self.min_shared_tokens < 1:
            raise ValueError(
                f"min_shared_tokens must be >= 1, got {self.min_shared_tokens}"
            )
        if self.max_group_size < 2:
            raise ValueError(
                f"max_group_size must be >= 2, got {self.max_group_size}"
            )


# ---------------------------------------------------------------------------
# Coalesce group
# ---------------------------------------------------------------------------


@dataclass
class CoalesceGroup:
    """A set of requests that share a common prefix and will be co-processed.

    Attributes:
        shared_prefix: The token IDs shared by all requests in this group.
                       This prefix is prefilled once before branching.
        request_ids:   Ordered list of request identifiers in the group.
        branch_tokens: Per-request list of token IDs that follow the shared
                       prefix.  ``branch_tokens[i]`` belongs to
                       ``request_ids[i]``.
    """

    shared_prefix: list[int]
    request_ids: list[str]
    branch_tokens: list[list[int]]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CoalesceStats:
    """Cumulative statistics for the :class:`PrefixCoalescer`.

    Attributes:
        total_requests:      Total number of requests passed to
                             :meth:`add_request`.
        total_groups_formed: Total number of :class:`CoalesceGroup` objects
                             returned across all :meth:`coalesce` calls.
        total_tokens_saved:  Cumulative number of prefix tokens whose prefill
                             computation was avoided by coalescing (i.e. for
                             each group, ``len(shared_prefix) * (n_requests - 1)``
                             tokens were not re-prefilled).
    """

    total_requests: int = 0
    total_groups_formed: int = 0
    total_tokens_saved: int = 0


# ---------------------------------------------------------------------------
# Coalescer
# ---------------------------------------------------------------------------


class PrefixCoalescer:
    """Groups buffered requests by longest common prefix before dispatch.

    Incoming requests are buffered via :meth:`add_request`.  Calling
    :meth:`coalesce` drains the buffer, groups requests by shared prefix, and
    returns a list of :class:`CoalesceGroup` objects ready for dispatch.  After
    :meth:`coalesce` returns, the internal buffer is empty.

    Algorithm overview:

    1. Sort pending requests by their token list (lexicographic) to bring
       requests with similar prefixes adjacent to each other.
    2. Greedily form groups: start a new group with the first ungrouped request.
       Scan forward and add each subsequent request to the current group if
       the LCP with the group's shared prefix is >= ``min_shared_tokens`` and
       the group has not yet reached ``max_group_size``.
    3. When a candidate's LCP falls below ``min_shared_tokens`` or the group is
       full, close the current group and start a new one.

    Args:
        config: A :class:`CoalesceConfig` describing grouping thresholds.
    """

    def __init__(self, config: CoalesceConfig) -> None:
        self._config = config
        # Buffer: list of (request_id, tokens) in insertion order.
        self._pending: list[tuple[str, list[int]]] = []
        self._stats = CoalesceStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request_id: str, tokens: list[int]) -> None:
        """Buffer a request for coalescing.

        Args:
            request_id: Unique identifier for the request.
            tokens:     Full token sequence (prompt + any prefixed context).
                        Must be non-empty.

        Raises:
            ValueError: if *request_id* is empty, *tokens* is empty, or
                        *request_id* is already in the buffer.
        """
        if not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not tokens:
            raise ValueError("tokens must be a non-empty list")
        existing_ids = {rid for rid, _ in self._pending}
        if request_id in existing_ids:
            raise ValueError(
                f"request_id '{request_id}' is already pending in the coalescer"
            )
        self._pending.append((request_id, tokens))
        self._stats.total_requests += 1

    def coalesce(self) -> list[CoalesceGroup]:
        """Group all buffered requests by longest common prefix.

        Drains the internal buffer and returns a list of
        :class:`CoalesceGroup` objects.  Groups are formed greedily as
        described in the class docstring.  Singleton groups are included for
        requests that do not share enough prefix with any neighbour.

        Returns:
            A non-empty list of :class:`CoalesceGroup` objects, or an empty
            list if no requests have been buffered.
        """
        if not self._pending:
            return []

        # Sort by token sequence for LCP-adjacent grouping.
        sorted_reqs = sorted(self._pending, key=lambda item: item[1])
        self._pending = []

        groups: list[CoalesceGroup] = []
        # Track which requests have been assigned.
        assigned = [False] * len(sorted_reqs)

        i = 0
        while i < len(sorted_reqs):
            if assigned[i]:
                i += 1
                continue

            rid_i, tokens_i = sorted_reqs[i]
            # Start a new group anchored at request i.
            group_request_ids = [rid_i]
            group_branch_tokens: list[list[int]] = []
            current_shared = list(tokens_i)
            assigned[i] = True

            # Try to grow the group from subsequent requests.
            j = i + 1
            while j < len(sorted_reqs) and len(group_request_ids) < self._config.max_group_size:
                if assigned[j]:
                    j += 1
                    continue
                rid_j, tokens_j = sorted_reqs[j]
                lcp = _longest_common_prefix_len(current_shared, tokens_j)
                if lcp >= self._config.min_shared_tokens:
                    # Narrow the shared prefix to the LCP of all group members.
                    current_shared = current_shared[:lcp]
                    group_request_ids.append(rid_j)
                    assigned[j] = True
                j += 1

            # Build branch tokens for each group member.
            prefix_len = len(current_shared)
            for rid, tok in sorted_reqs[i:]:
                if rid in group_request_ids:
                    # Find the matching token list.
                    for r2, t2 in sorted_reqs:
                        if r2 == rid:
                            group_branch_tokens.append(t2[prefix_len:])
                            break

            # Re-order branch_tokens to match group_request_ids order.
            rid_to_tokens: dict[str, list[int]] = {}
            for r2, t2 in sorted_reqs:
                if r2 in group_request_ids:
                    rid_to_tokens[r2] = t2[prefix_len:]
            ordered_branch = [rid_to_tokens[rid] for rid in group_request_ids]

            # Compute tokens saved by coalescing.
            n_members = len(group_request_ids)
            tokens_saved = prefix_len * (n_members - 1)
            self._stats.total_tokens_saved += tokens_saved

            groups.append(
                CoalesceGroup(
                    shared_prefix=list(current_shared),
                    request_ids=list(group_request_ids),
                    branch_tokens=ordered_branch,
                )
            )
            self._stats.total_groups_formed += 1
            i += 1

        return groups

    @property
    def n_pending(self) -> int:
        """Number of requests currently buffered and awaiting coalescing."""
        return len(self._pending)

    @property
    def stats(self) -> CoalesceStats:
        """Cumulative coalescing statistics."""
        return self._stats


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Return the length of the longest common prefix of *a* and *b*.

    Args:
        a: First token list.
        b: Second token list.

    Returns:
        An integer in ``[0, min(len(a), len(b))]``.
    """
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n
