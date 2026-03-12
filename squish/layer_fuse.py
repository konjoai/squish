"""LayerFuse — Adjacent transformer layer fusion via weight averaging.

When two consecutive transformer layers have nearly identical weight
distributions — as measured by the cosine similarity of their flattened weight
vectors — they can be replaced by a single layer whose weight matrix is the
element-wise average of the two.  The fused layer captures the shared
representation while cutting the parameter count for that pair in half.

This technique is loosely related to *SliceGPT* and similar post-training
architecture compression approaches that identify redundancy in depth rather
than width.

Usage::

    import numpy as np
    from squish.layer_fuse import FusionConfig, LayerFuser

    cfg    = FusionConfig(similarity_threshold=0.97)
    fuser  = LayerFuser(cfg)

    rng  = np.random.default_rng(0)
    wa   = rng.standard_normal((512, 512)).astype(np.float32)
    wb   = wa + rng.standard_normal((512, 512)).astype(np.float32) * 0.01

    if fuser.should_fuse(wa, wb):
        fused = fuser.fuse(wa, wb)   # (512, 512)
    print(fuser.stats.total_fusions)
"""

from __future__ import annotations

__all__ = [
    "FusionConfig",
    "LayerFuser",
    "FusionStats",
]

from dataclasses import dataclass

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    """Configuration for adjacent-layer weight fusion.

    Attributes:
        hidden_dim: Size of the hidden dimension (informational; used when
            constructing layer pairs outside this class).
        similarity_threshold: Minimum cosine similarity required before two
            layers are considered fusible.  Must be in ``(0, 1]``.
    """

    hidden_dim: int = 512
    similarity_threshold: float = 0.97

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1; got {self.hidden_dim}"
            )
        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1]; "
                f"got {self.similarity_threshold}"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class FusionStats:
    """Cumulative statistics for a :class:`LayerFuser` session.

    Attributes:
        total_similarity_checks: Number of times
            :meth:`LayerFuser.should_fuse` has been called.
        total_fusions: Number of layer pairs that were fused via
            :meth:`LayerFuser.fuse`.
        total_layers_saved: Number of layers eliminated (one per fusion,
            since each fusion replaces two layers with one).
    """

    total_similarity_checks: int = 0
    total_fusions: int = 0
    total_layers_saved: int = 0


# ── Fuser ─────────────────────────────────────────────────────────────────────

class LayerFuser:
    """Identifies and fuses pairs of similar adjacent transformer layers.

    Two layers are considered similar when the cosine similarity of their
    flattened weight vectors is at or above ``config.similarity_threshold``.
    Fusion produces the element-wise mean of the two weight matrices.

    Args:
        config: :class:`FusionConfig` specifying the similarity threshold
            and optional hidden dimension.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self._stats = FusionStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cosine_similarity(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        """Compute the cosine similarity between two weight arrays.

        Both arrays are flattened to 1-D before the computation.

        Args:
            a: First weight array (any shape, float32).
            b: Second weight array (any shape, float32).

        Returns:
            Cosine similarity in [-1, 1].  Values near +1 indicate that the
            two weight tensors are nearly collinear.
        """
        a_flat = np.asarray(a, dtype=np.float32).ravel()
        b_flat = np.asarray(b, dtype=np.float32).ravel()

        dot    = float(np.dot(a_flat, b_flat))
        norm_a = float(np.linalg.norm(a_flat))
        norm_b = float(np.linalg.norm(b_flat))

        return dot / (norm_a * norm_b + 1e-9)

    def should_fuse(
        self, weights_a: np.ndarray, weights_b: np.ndarray
    ) -> bool:
        """Determine whether two layers are similar enough to fuse.

        Computes :meth:`cosine_similarity` and compares to
        ``config.similarity_threshold``.  Each call increments
        ``stats.total_similarity_checks``.

        Args:
            weights_a: Weight matrix of the first layer (any shape, float32).
            weights_b: Weight matrix of the second layer (must have the same
                number of elements as *weights_a*).

        Returns:
            ``True`` if ``cosine_similarity(a, b) >= similarity_threshold``.
        """
        sim = self.cosine_similarity(weights_a, weights_b)
        self._stats.total_similarity_checks += 1
        return sim >= self.config.similarity_threshold

    def fuse(
        self, weights_a: np.ndarray, weights_b: np.ndarray
    ) -> np.ndarray:
        """Fuse two weight matrices by computing their element-wise mean.

        Both matrices are cast to float32 before averaging.  The returned
        array has the same shape as *weights_a*.

        Args:
            weights_a: First weight matrix (any shape, float32).
            weights_b: Second weight matrix (same shape as *weights_a*).

        Returns:
            Float32 array equal to ``(weights_a + weights_b) / 2``.
        """
        a = np.asarray(weights_a, dtype=np.float32)
        b = np.asarray(weights_b, dtype=np.float32)

        self._stats.total_fusions    += 1
        self._stats.total_layers_saved += 1

        return ((a + b) * 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> FusionStats:
        """Cumulative fusion statistics for this instance."""
        return self._stats
