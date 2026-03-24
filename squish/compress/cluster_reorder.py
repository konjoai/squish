# [Experimental] This module is part of Squish v40+ (Wave 66).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""cluster_reorder.py — FFN Weight Column Reordering for Co-activation Clusters.

After :mod:`squish.compress.sparsity_profiler` assigns neurons to clusters,
this module physically sorts the FFN weight tensors so that neurons belonging
to the same cluster occupy *contiguous column ranges* in memory.  This is
essential for ``sparse_gemv.metal``: the shader skips entire cluster column
ranges for inactive clusters, so sequential layout makes those skip operations
a simple range-bounds check with no scatter/gather.

For each FFN layer the three weight matrices involved are:

  W_up   (n_neurons, hidden_dim) — up-projection from hidden space
  W_gate (n_neurons, hidden_dim) — gate-projection (SwiGLU gate path)
  W_down (hidden_dim, n_neurons) — down-projection back to hidden space

The co-activation cluster corresponds to the *neuron dimension* (axis 0 of
W_up/W_gate; axis 1 of W_down).  The reordering permutation is derived from
the cluster assignments and applied identically to all three matrices.

Usage::

    from squish.compress.cluster_reorder import ClusterReorder, ReorderResult
    from squish.compress.sparsity_profiler import LayerSparsityProfile

    profile: LayerSparsityProfile = ...
    w_up   = np.random.randn(4096, 768).astype(np.float32)
    w_gate = np.random.randn(4096, 768).astype(np.float32)
    w_down = np.random.randn(768, 4096).astype(np.float32)

    reorderer = ClusterReorder()
    result = reorderer.reorder(profile, w_up, w_gate, w_down)
    # result.w_up_reordered  : (4096, 768) — neurons sorted by cluster id
    # result.w_down_reordered: (768, 4096) — rows sorted by cluster id
    # result.cluster_boundaries: [u32; n_clusters+1] column ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.compress.sparsity_profiler import LayerSparsityProfile

__all__ = [
    "ReorderResult",
    "ClusterReorder",
    "compute_cluster_permutation",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReorderResult:
    """Outputs from :class:`ClusterReorder`.

    Attributes:
        w_up_reordered: W_up with neuron axis (axis 0) sorted by cluster id.
        w_gate_reordered: W_gate with neuron axis (axis 0) sorted by cluster id.
            None if W_gate was not provided (non-SwiGLU FFN).
        w_down_reordered: W_down with neuron axis (axis 1) sorted by cluster id.
        permutation: (n_neurons,) int32 — the permutation applied.
            ``w_up_reordered[i] = w_up[permutation[i]]``.
        inverse_permutation: (n_neurons,) int32 — inverse of *permutation*.
        cluster_boundaries: (n_clusters+1,) int32 — column offsets in
            the *reordered* layout.  Boundary ``[i:i+1]`` gives the column
            range of cluster ``i``.
        profile: The :class:`LayerSparsityProfile` used for reordering,
            updated in-place with the new ``cluster_boundaries``.
    """

    w_up_reordered: np.ndarray
    w_gate_reordered: Optional[np.ndarray]
    w_down_reordered: np.ndarray
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    cluster_boundaries: np.ndarray
    profile: LayerSparsityProfile


# ---------------------------------------------------------------------------
# Permutation helper
# ---------------------------------------------------------------------------

def compute_cluster_permutation(
    cluster_assignments: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Compute a permutation that groups neurons by cluster id.

    Args:
        cluster_assignments: ``(n_neurons,) int32`` cluster id per neuron.
        n_clusters: Total number of clusters.

    Returns:
        ``(n_neurons,) int32`` permutation such that neurons of cluster 0
        come first, then cluster 1, etc.  Within each cluster, neurons are
        sorted by their original index (stable).
    """
    groups = [
        np.where(cluster_assignments == cid)[0].astype(np.int32)
        for cid in range(n_clusters)
    ]
    return np.concatenate(groups).astype(np.int32) if groups else np.empty(0, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main reorderer
# ---------------------------------------------------------------------------

class ClusterReorder:
    """Reorders FFN weight matrices columns/rows by co-activation cluster.

    After reordering, each cluster occupies a contiguous column range in
    W_up and W_gate (and the corresponding row range in W_down).  The
    ``sparse_gemv.metal`` shader reads these ranges sequentially.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reorder(
        self,
        profile: LayerSparsityProfile,
        w_up: np.ndarray,
        w_down: np.ndarray,
        w_gate: Optional[np.ndarray] = None,
    ) -> ReorderResult:
        """Apply cluster-based reordering to the FFN weight matrices.

        Args:
            profile: :class:`LayerSparsityProfile` from calibration.
            w_up: ``(n_neurons, hidden_dim)`` up-projection weight matrix.
            w_down: ``(hidden_dim, n_neurons)`` down-projection weight matrix.
            w_gate: Optional ``(n_neurons, hidden_dim)`` gate weight matrix
                (SwiGLU only).

        Returns:
            :class:`ReorderResult` with reordered matrices, permutation
            arrays, and updated cluster boundaries.

        Raises:
            ValueError: If matrix and profile neuron counts are inconsistent.
        """
        w_up = np.asarray(w_up)
        w_down = np.asarray(w_down)
        n_neurons_up = w_up.shape[0]
        n_neurons_down = w_down.shape[1]

        if n_neurons_up != profile.n_neurons:
            raise ValueError(
                f"w_up.shape[0]={n_neurons_up} does not match "
                f"profile.n_neurons={profile.n_neurons}"
            )
        if n_neurons_down != profile.n_neurons:
            raise ValueError(
                f"w_down.shape[1]={n_neurons_down} does not match "
                f"profile.n_neurons={profile.n_neurons}"
            )
        if w_gate is not None:
            w_gate = np.asarray(w_gate)
            if w_gate.shape[0] != profile.n_neurons:
                raise ValueError(
                    f"w_gate.shape[0]={w_gate.shape[0]} does not match "
                    f"profile.n_neurons={profile.n_neurons}"
                )

        perm = compute_cluster_permutation(
            profile.cluster_assignments, profile.n_clusters
        )
        inv_perm = np.argsort(perm).astype(np.int32)

        # Apply reordering.
        w_up_r = w_up[perm]
        w_down_r = w_down[:, perm]
        w_gate_r = w_gate[perm] if w_gate is not None else None

        # Recompute cluster boundaries in the new layout.
        sizes = np.array(
            [
                int((profile.cluster_assignments == cid).sum())
                for cid in range(profile.n_clusters)
            ],
            dtype=np.int32,
        )
        boundaries = np.zeros(profile.n_clusters + 1, dtype=np.int32)
        boundaries[1:] = np.cumsum(sizes)

        # Update profile in-place.
        profile.cluster_boundaries = boundaries

        return ReorderResult(
            w_up_reordered=w_up_r,
            w_gate_reordered=w_gate_r,
            w_down_reordered=w_down_r,
            permutation=perm,
            inverse_permutation=inv_perm,
            cluster_boundaries=boundaries,
            profile=profile,
        )

    def apply_permutation_to_tensor(
        self, tensor: np.ndarray, permutation: np.ndarray, axis: int = 0
    ) -> np.ndarray:
        """Apply *permutation* to *tensor* along *axis*.

        Convenience helper for reordering arbitrary tensors (e.g., bias
        vectors) using a pre-computed permutation.

        Args:
            tensor: Tensor to permute.
            permutation: ``(n,) int32`` permutation to apply.
            axis: Axis of *tensor* whose size == ``len(permutation)``.

        Returns:
            Reordered tensor (new array, original unchanged).
        """
        return np.take(tensor, permutation, axis=axis)

    def verify_reorder(
        self,
        original_w_up: np.ndarray,
        result: ReorderResult,
        hidden_state: np.ndarray,
    ) -> float:
        """Verify that the reordered matrices produce the same FFN output.

        Computes ``output_original`` and ``output_reordered`` using the same
        hidden state (dense GEMV, no sparsity) and returns the L1 error.
        A result of 0.0 confirms the reordering is an exact permutation.

        Args:
            original_w_up: The original un-reordered W_up matrix.
            result: :class:`ReorderResult` containing the reordered matrices.
            hidden_state: ``(hidden_dim,)`` float32 input to test.

        Returns:
            L1 error between original and reordered GEMV outputs.
        """
        original_w_up = np.asarray(original_w_up, dtype=np.float32)
        hs = np.asarray(hidden_state, dtype=np.float32)

        out_orig = (original_w_up @ hs).astype(np.float32)
        out_reord = (result.w_up_reordered @ hs).astype(np.float32)

        # The reordered output is a permutation of the original.
        # Re-permuting with inverse_permutation should match exactly.
        out_reord_unshuffled = out_reord[result.inverse_permutation]
        return float(np.abs(out_orig - out_reord_unshuffled).sum())
