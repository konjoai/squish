"""EmbeddingGate — Gated modality-conditional embedding router.

A lightweight routing module that classifies each token (or image patch) into
one of two modality pathways — text (route 0) or vision/tool (route 1) — based
on the magnitude of its activation along a learned gate direction.

Gate computation (per sample):

    gate_score = |embeddings @ gate_weights|   (scalar per sample)
    route       = 0  if gate_score < threshold  (text pathway)
                = 1  otherwise                  (vision/tool pathway)

The gate weights are initialised uniformly to ``1 / embed_dim`` so that the
initial gate score is the mean absolute activation — a sensible default for
the text-vs-vision split before any fine-tuning.

The gated output zeros out *text* tokens in the vision path::

    masked_embeddings = embeddings * route[:, np.newaxis]

Text tokens (route 0) contribute zero to the vision pathway; callers can
select the plain embeddings for the text pathway and the masked embeddings for
the vision pathway.

Typical usage::

    from squish.embedding_gate import GateConfig, EmbeddingGate
    import numpy as np

    cfg  = GateConfig(embed_dim=512, threshold=0.5)
    gate = EmbeddingGate(cfg)

    embeddings = np.random.randn(8, 512).astype(np.float32)
    routes, masked = gate.gate(embeddings)
    # routes: (8,) int — 0=text, 1=vision
    # masked: (8, 512) — vision tokens kept, text tokens zeroed
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GateConfig",
    "EmbeddingGate",
    "GateStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Configuration for :class:`EmbeddingGate`.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embedding vectors.
    threshold : float
        Scalar threshold on the gate score.  Tokens whose
        ``|embedding @ gate_weights|`` is at or above this value are routed
        to the vision pathway (route 1); others go to text (route 0).
    n_routes : int
        Number of routing outputs.  Currently only ``2`` is supported
        (text and vision).
    """

    embed_dim: int = 512
    threshold: float = 0.5
    n_routes: int = 2

    def __post_init__(self) -> None:
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be > 0")
        if self.n_routes != 2:
            raise ValueError(
                "EmbeddingGate currently supports n_routes=2 only; "
                f"got n_routes={self.n_routes}"
            )


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class GateStats:
    """Aggregate statistics for :class:`EmbeddingGate`.

    Attributes
    ----------
    total_gate_calls : int
        Total number of :meth:`~EmbeddingGate.gate` calls.
    total_text_routed : int
        Total tokens routed to the text pathway (route 0).
    total_vision_routed : int
        Total tokens routed to the vision pathway (route 1).
    """

    total_gate_calls: int = 0
    total_text_routed: int = 0
    total_vision_routed: int = 0

    @property
    def vision_fraction(self) -> float:
        """Fraction of all routed tokens sent to the vision pathway."""
        total = self.total_text_routed + self.total_vision_routed
        if total == 0:
            return 0.0
        return self.total_vision_routed / total


# ---------------------------------------------------------------------------
# EmbeddingGate
# ---------------------------------------------------------------------------


class EmbeddingGate:
    """Threshold-based modality gate over a learned projection direction.

    Parameters
    ----------
    config : GateConfig
        Gate configuration.
    """

    def __init__(self, config: GateConfig) -> None:
        self._cfg = config
        # Learnable gate direction initialised uniformly.
        self.gate_weights: np.ndarray = np.full(
            (config.embed_dim,),
            fill_value=1.0 / config.embed_dim,
            dtype=np.float32,
        )
        self._stats = GateStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def gate(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Route each embedding to text or vision pathway.

        Parameters
        ----------
        embeddings : np.ndarray
            Input tensor of shape ``(batch, embed_dim)`` or
            ``(seq, embed_dim)``.

        Returns
        -------
        route : np.ndarray
            Integer array of shape ``(batch,)`` with values ``0`` (text) or
            ``1`` (vision).
        masked_embeddings : np.ndarray
            Same shape as *embeddings*.  Text tokens (route 0) are zeroed out
            so that downstream vision-pathway operations see only active tokens.

        Raises
        ------
        ValueError
            If *embeddings* is not 2-D or its second dimension does not match
            ``embed_dim``.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (batch, embed_dim); got ndim={embeddings.ndim}"
            )
        batch, d = embeddings.shape
        if d != self._cfg.embed_dim:
            raise ValueError(
                f"embeddings second dimension must be {self._cfg.embed_dim}; got {d}"
            )

        # Gate score: |dot product with learned direction|, shape (batch,).
        gate_scores = np.abs(embeddings @ self.gate_weights)  # (batch,)

        # Binary routing decision.
        route = (gate_scores >= self._cfg.threshold).astype(np.int32)  # (batch,)

        # Zero out text tokens in the masked output.
        masked_embeddings = embeddings * route[:, np.newaxis]

        # Update stats.
        n_vision = int(route.sum())
        n_text = batch - n_vision
        self._stats.total_gate_calls += 1
        self._stats.total_text_routed += n_text
        self._stats.total_vision_routed += n_vision

        return route, masked_embeddings

    def update_threshold(self, new_threshold: float) -> None:
        """Update the gate threshold at runtime.

        Parameters
        ----------
        new_threshold : float
            New threshold value.  Must be strictly positive.

        Raises
        ------
        ValueError
            If *new_threshold* is not positive.
        """
        if new_threshold <= 0.0:
            raise ValueError(
                f"threshold must be > 0; got {new_threshold}"
            )
        self._cfg.threshold = new_threshold

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> GateStats:
        """Current aggregate statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"EmbeddingGate(embed_dim={self._cfg.embed_dim}, "
            f"threshold={self._cfg.threshold})"
        )
