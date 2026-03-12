"""ModalityRouter — Per-modality SLO-aware request dispatcher.

Routes inference requests to the correct serving backend based on detected
modality (text, vision, audio, etc.) while enforcing:

  - Per-modality maximum concurrency (``max_concurrent``).
  - Per-modality SLO target (``target_latency_ms``).
  - Priority tie-breaking when multiple modalities are competing for capacity.

When a modality's active request count is at ``max_concurrent`` the request
is *deferred* (returns ``False``); the caller is responsible for re-queuing
or shedding deferred requests.  Concurrency decrements are performed via
:meth:`complete`, which also records the observed latency for SLO tracking.

Typical usage::

    from squish.modality_router import ModalityPolicy, ModalityRouter

    router = ModalityRouter([
        ModalityPolicy(modality="text",   target_latency_ms=50.0,  max_concurrent=8),
        ModalityPolicy(modality="vision", target_latency_ms=200.0, max_concurrent=2),
    ])

    req_id = 42
    routed = router.route(req_id, modality="vision")
    if routed:
        # ... process request ...
        router.complete(req_id, modality="vision", latency_ms=185.0)
    else:
        print("deferred — vision backend at capacity")

    print(router.slo_violation_rate("vision"))
    print(router.stats)
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ModalityPolicy",
    "ModalityRouter",
    "RouterStats",
]

# ---------------------------------------------------------------------------
# ModalityPolicy
# ---------------------------------------------------------------------------


@dataclass
class ModalityPolicy:
    """SLO and capacity policy for a single modality backend.

    Parameters
    ----------
    modality : str
        Unique modality name, e.g. ``"text"``, ``"vision"``, or ``"audio"``.
    target_latency_ms : float
        SLO target in milliseconds.  Completions above this value count as
        violations for :meth:`~ModalityRouter.slo_violation_rate`.
    max_concurrent : int
        Maximum number of in-flight requests for this modality.  New requests
        are deferred (not accepted) when the active count reaches this limit.
    priority : int
        Serving priority for tie-breaking.  Higher value = served first.
        Not currently used for automatic scheduling but exposed for external
        orchestration logic.
    """

    modality: str
    target_latency_ms: float = 100.0
    max_concurrent: int = 4
    priority: int = 1

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("modality must be a non-empty string")
        if self.target_latency_ms <= 0.0:
            raise ValueError("target_latency_ms must be > 0")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class RouterStats:
    """Aggregate statistics for :class:`ModalityRouter`.

    Attributes
    ----------
    total_routes : int
        Total number of requests successfully routed (accepted).
    total_deferred : int
        Total number of requests deferred because the modality was at capacity.
    total_completions : int
        Total number of :meth:`~ModalityRouter.complete` calls processed.
    """

    total_routes: int = 0
    total_deferred: int = 0
    total_completions: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of route attempts that were accepted (not deferred)."""
        total = self.total_routes + self.total_deferred
        if total == 0:
            return 0.0
        return self.total_routes / total


# ---------------------------------------------------------------------------
# ModalityRouter
# ---------------------------------------------------------------------------


class ModalityRouter:
    """Routes requests to per-modality backends with SLO and capacity enforcement.

    Parameters
    ----------
    policies : list[ModalityPolicy]
        Initial set of modality policies.  Modality names must be unique.

    Raises
    ------
    ValueError
        If *policies* contains duplicate modality names.
    """

    def __init__(self, policies: list[ModalityPolicy]) -> None:
        self._policies: dict[str, ModalityPolicy] = {}
        self._active: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}
        self._stats = RouterStats()

        seen: set[str] = set()
        for p in policies:
            if p.modality in seen:
                raise ValueError(
                    f"Duplicate modality name in policies: '{p.modality}'"
                )
            seen.add(p.modality)
            self._register_policy(p)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def register(self, modality: str, policy: ModalityPolicy) -> None:
        """Add or replace the policy for *modality*.

        If a policy for *modality* already exists it is overwritten.  The
        active count and latency history for the modality are preserved if it
        was already registered.

        Parameters
        ----------
        modality : str
            Modality name to register.  Must match ``policy.modality``.
        policy : ModalityPolicy
            New policy for the modality.

        Raises
        ------
        ValueError
            If *modality* does not match ``policy.modality``.
        """
        if modality != policy.modality:
            raise ValueError(
                f"modality key '{modality}' does not match "
                f"policy.modality '{policy.modality}'"
            )
        self._register_policy(policy)

    def route(self, req_id: int, modality: str) -> bool:
        """Attempt to route request *req_id* to the *modality* backend.

        Parameters
        ----------
        req_id : int
            Opaque request identifier (used for logging/tracing; not stored
            by the router beyond this call).
        modality : str
            Target modality backend.

        Returns
        -------
        bool
            ``True`` if the request was accepted (concurrency slot available);
            ``False`` if deferred (at capacity).

        Raises
        ------
        KeyError
            If *modality* has no registered policy.
        """
        policy = self._get_policy(modality)
        active = self._active[modality]

        if active < policy.max_concurrent:
            self._active[modality] += 1
            self._stats.total_routes += 1
            return True

        self._stats.total_deferred += 1
        return False

    def complete(
        self,
        req_id: int,
        modality: str,
        latency_ms: float,
    ) -> None:
        """Signal that request *req_id* for *modality* has finished.

        Decrements the active concurrency counter and records the observed
        latency for SLO accounting.

        Parameters
        ----------
        req_id : int
            Opaque request identifier (informational only).
        modality : str
            Modality backend that processed the request.
        latency_ms : float
            Observed end-to-end latency in milliseconds.

        Raises
        ------
        KeyError
            If *modality* has no registered policy.
        """
        self._get_policy(modality)  # validates existence
        self._active[modality] = max(0, self._active[modality] - 1)
        self._latencies[modality].append(float(latency_ms))
        self._stats.total_completions += 1

    def slo_violation_rate(self, modality: str) -> float:
        """Return the fraction of completions that exceeded the SLO target.

        Parameters
        ----------
        modality : str
            Modality backend to query.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` if no completions have been
            recorded for this modality.

        Raises
        ------
        KeyError
            If *modality* has no registered policy.
        """
        policy = self._get_policy(modality)
        latencies = self._latencies[modality]
        if not latencies:
            return 0.0
        violations = sum(
            1 for lat in latencies if lat > policy.target_latency_ms
        )
        return violations / len(latencies)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> RouterStats:
        """Current aggregate statistics."""
        return self._stats

    def active_counts(self) -> dict[str, int]:
        """Return a snapshot of current active request counts per modality."""
        return dict(self._active)

    def __repr__(self) -> str:
        modalities = list(self._policies)
        return (
            f"ModalityRouter(modalities={modalities}, "
            f"routes={self._stats.total_routes}, "
            f"deferred={self._stats.total_deferred})"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _register_policy(self, policy: ModalityPolicy) -> None:
        """Store policy; initialise active/latency tracking if new."""
        m = policy.modality
        self._policies[m] = policy
        if m not in self._active:
            self._active[m] = 0
        if m not in self._latencies:
            self._latencies[m] = []

    def _get_policy(self, modality: str) -> ModalityPolicy:
        """Return the policy for *modality* or raise ``KeyError``."""
        try:
            return self._policies[modality]
        except KeyError:
            raise KeyError(
                f"No policy registered for modality '{modality}'. "
                f"Available: {list(self._policies)}"
            ) from None
