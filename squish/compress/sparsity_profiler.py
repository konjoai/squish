# [Experimental] This module is part of Squish v40+ (Wave 66).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""sparsity_profiler.py — FFN Co-activation Calibration + K-Means Clustering.

Wave 66 exploits the *dead-neuron* phenomenon in SwiGLU FFN layers.
Empirically, 40–65% of FFN neurons produce near-zero activations on any
given token (DejaVu, NeurIPS 2023; PowerInfer, SC 2024).  This module bakes
that sparsity into the ``.squizd`` compressed format at calibration time.

Algorithm
─────────
1. Feed 2,000 prompt samples through each FFN layer, recording the post-
   activation magnitudes of all neurons.
2. Compute per-neuron statistics: mean magnitude, activation frequency (
   fraction of samples where |act| > threshold), and pairwise co-activation
   frequency (how often two neurons fire on the same sample).
3. Build a k-means (k=64) clustering over the co-activation frequency
   vectors, grouping neurons that consistently fire together.
4. Return one :class:`LayerSparsityProfile` per layer, carrying:
   - cluster assignment for each neuron
   - per-cluster activation frequency histogram
   - measured dead-neuron fraction
   - expected predictor speedup

The profile is later consumed by:
 - :mod:`squish.compress.cluster_reorder` — sorts weight columns by cluster
 - :mod:`squish.token.sparsity_predictor` — trains the lightweight FFN predictor

Usage::

    from squish.compress.sparsity_profiler import SparsityProfiler, ProfilerConfig

    cfg      = ProfilerConfig(n_samples=2000, n_clusters=64)
    profiler = SparsityProfiler(cfg)

    # activation_fn: callable (hidden_state: np.ndarray) -> np.ndarray
    profile = profiler.profile_layer(activation_fn, hidden_states)
    print(f"Sparsity: {profile.sparsity_ratio:.1%}  Clusters: {len(profile.cluster_boundaries)-1}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ProfilerConfig",
    "ClusterInfo",
    "LayerSparsityProfile",
    "SparsityProfiler",
    "coactivation_matrix",
    "kmeans_cluster",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ACT_THRESHOLD: float = 1e-3   # |activation| below this → neuron "dead"
_DEFAULT_N_CLUSTERS: int = 64
_DEFAULT_N_SAMPLES: int = 2000
_KMEANS_MAX_ITER: int = 100
_KMEANS_TOL: float = 1e-4               # relative centroid shift tolerance


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProfilerConfig:
    """Configuration for :class:`SparsityProfiler`.

    Attributes:
        n_samples: Number of calibration prompt samples.  Profiles stabilise
            at ~2000; minimum recommended is 500.
        n_clusters: Number of k-means clusters per FFN layer.  64 works well
            for models up to 14B; larger models may benefit from 128.
        activation_threshold: Absolute magnitude threshold below which a
            neuron is considered "dead" on a given sample.
        kmeans_max_iter: Maximum k-means iterations.
        kmeans_seed: RNG seed for reproducible k-means initialisation.
    """

    n_samples: int = _DEFAULT_N_SAMPLES
    n_clusters: int = _DEFAULT_N_CLUSTERS
    activation_threshold: float = _DEFAULT_ACT_THRESHOLD
    kmeans_max_iter: int = _KMEANS_MAX_ITER
    kmeans_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")
        if self.activation_threshold < 0.0:
            raise ValueError(
                f"activation_threshold must be >= 0, got {self.activation_threshold}"
            )
        if self.kmeans_max_iter < 1:
            raise ValueError(
                f"kmeans_max_iter must be >= 1, got {self.kmeans_max_iter}"
            )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    """Statistics for a single co-activation cluster.

    Attributes:
        cluster_id: Zero-based cluster index.
        neuron_indices: Sorted array of neuron (column) indices in this cluster.
        activation_frequency: Fraction of calibration samples on which at
            least one neuron in the cluster was active.
        mean_magnitude: Mean |activation| across all active samples and neurons.
    """

    cluster_id: int
    neuron_indices: np.ndarray    # (n_neurons_in_cluster,) int32
    activation_frequency: float
    mean_magnitude: float


@dataclass
class LayerSparsityProfile:
    """Sparsity profile for a single FFN layer.

    Attributes:
        layer_idx: Transformer layer index (-1 if not associated with a layer).
        n_neurons: Total neuron count in the FFN hidden dimension.
        n_clusters: Number of clusters produced.
        cluster_assignments: (n_neurons,) int32 — cluster id per neuron.
        cluster_boundaries: (n_clusters+1,) int32 — start/end neuron index for
            each cluster in the *reordered* layout (post cluster_reorder.py).
            Before reordering, this is a compact sorted boundary array over
            the cluster natural ordering.
        activation_histogram: (n_clusters,) float32 — empirical cluster
            activation frequency from calibration data.
        sparsity_ratio: fraction of neurons that were "dead" on the average
            calibration sample (i.e., |act| < threshold).
        expected_sparsity: same as sparsity_ratio, retained for metadata.
        clusters: List of :class:`ClusterInfo` objects, one per cluster.
    """

    layer_idx: int
    n_neurons: int
    n_clusters: int
    cluster_assignments: np.ndarray     # (n_neurons,) int32
    cluster_boundaries: np.ndarray      # (n_clusters+1,) int32
    activation_histogram: np.ndarray    # (n_clusters,) float32
    sparsity_ratio: float
    expected_sparsity: float
    clusters: List[ClusterInfo]

    def active_clusters_at(self, threshold: float = 0.5) -> int:
        """Count clusters with activation frequency above *threshold*."""
        return int((self.activation_histogram > threshold).sum())

    def to_metadata_bytes(self) -> bytes:
        """Serialise to the binary sparsity metadata block format.

        Wire layout::

            u16  n_clusters
            u32[n_clusters+1]  cluster_boundaries
            f32[n_clusters]    activation_histogram
            f32                expected_sparsity
        """
        import struct as _struct
        header = _struct.pack("<H", self.n_clusters)
        boundaries = self.cluster_boundaries.astype("<i4").tobytes()
        histogram = self.activation_histogram.astype("<f4").tobytes()
        sparsity = _struct.pack("<f", self.expected_sparsity)
        return header + boundaries + histogram + sparsity

    @classmethod
    def from_metadata_bytes(
        cls, data: bytes, layer_idx: int, n_neurons: int
    ) -> "LayerSparsityProfile":
        """Deserialise a binary sparsity metadata block.

        Args:
            data: Bytes produced by :meth:`to_metadata_bytes`.
            layer_idx: Transformer layer index to assign.
            n_neurons: Total neuron count for this layer.
        """
        import struct as _struct
        offset = 0
        (n_clusters,) = _struct.unpack_from("<H", data, offset)
        offset += 2
        boundaries = np.frombuffer(
            data[offset : offset + (n_clusters + 1) * 4], dtype="<i4"
        ).copy()
        offset += (n_clusters + 1) * 4
        histogram = np.frombuffer(
            data[offset : offset + n_clusters * 4], dtype="<f4"
        ).copy()
        offset += n_clusters * 4
        (expected_sparsity,) = _struct.unpack_from("<f", data, offset)
        # Reconstruct flat cluster_assignments from boundaries.
        assignments = np.empty(n_neurons, dtype=np.int32)
        for cid in range(n_clusters):
            start, end = int(boundaries[cid]), int(boundaries[cid + 1])
            assignments[start:end] = cid
        return cls(
            layer_idx=layer_idx,
            n_neurons=n_neurons,
            n_clusters=int(n_clusters),
            cluster_assignments=assignments,
            cluster_boundaries=boundaries,
            activation_histogram=histogram.astype(np.float32),
            sparsity_ratio=float(expected_sparsity),
            expected_sparsity=float(expected_sparsity),
            clusters=[],
        )


# ---------------------------------------------------------------------------
# Co-activation matrix helper
# ---------------------------------------------------------------------------

def coactivation_matrix(
    activation_matrix: np.ndarray, threshold: float = _DEFAULT_ACT_THRESHOLD
) -> np.ndarray:
    """Compute the pairwise co-activation frequency matrix.

    Args:
        activation_matrix: ``(n_samples, n_neurons)`` float32 array of
            post-activation magnitudes.
        threshold: Magnitude below which a neuron is considered inactive.

    Returns:
        ``(n_neurons, n_neurons)`` float32 symmetric matrix where entry
        ``[i, j]`` is the fraction of samples where both neuron i and j
        were active.
    """
    active = (np.abs(activation_matrix) > threshold).astype(np.float32)
    return (active.T @ active) / max(activation_matrix.shape[0], 1)


# ---------------------------------------------------------------------------
# K-means helper (NumPy-only, deterministic)
# ---------------------------------------------------------------------------

def kmeans_cluster(
    features: np.ndarray,
    k: int,
    max_iter: int = _KMEANS_MAX_ITER,
    seed: int = 42,
    tol: float = _KMEANS_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means clustering over float32 feature rows.

    Args:
        features: ``(n, d)`` float32 feature matrix.
        k: Number of clusters.  Clamped to ``min(k, n)``.
        max_iter: Maximum iterations.
        seed: RNG seed for centroid initialisation.
        tol: Convergence tolerance (relative centroid shift).

    Returns:
        Tuple of ``(assignments, centroids)`` where:
        - ``assignments`` is ``(n,) int32`` cluster id per row.
        - ``centroids`` is ``(k, d) float32`` final centroid positions.
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    k = min(k, n)

    # k-means++ initialisation
    centroid_indices = [int(rng.integers(n))]
    for _ in range(k - 1):
        dist_sq = np.array(
            [min(np.sum((features[i] - features[ci]) ** 2) for ci in centroid_indices)
             for i in range(n)],
            dtype=np.float64,
        )
        probs = dist_sq / dist_sq.sum()
        centroid_indices.append(int(rng.choice(n, p=probs)))
    centroids = features[centroid_indices].copy().astype(np.float32)

    assignments = np.zeros(n, dtype=np.int32)
    for iteration in range(max_iter):
        # Assignment step: nearest centroid via squared Euclidean distance.
        diffs = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
        dist_sq = (diffs ** 2).sum(axis=2)                                  # (n, k)
        new_assignments = dist_sq.argmin(axis=1).astype(np.int32)

        # Update step: recompute centroids.
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(n):
            new_centroids[new_assignments[i]] += features[i]
            counts[new_assignments[i]] += 1
        for ci in range(k):
            if counts[ci] > 0:
                new_centroids[ci] /= counts[ci]
            else:
                # Empty cluster: re-seed from a random point.
                new_centroids[ci] = features[int(rng.integers(n))]

        shift = np.linalg.norm(new_centroids - centroids) / (
            np.linalg.norm(centroids) + 1e-12
        )
        centroids = new_centroids
        assignments = new_assignments
        if shift < tol:
            break

    return assignments, centroids


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class SparsityProfiler:
    """Calibration-time FFN sparsity profiler.

    Parameters:
        config: :class:`ProfilerConfig` controlling sample count, cluster
            count, and activation threshold.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.config = config or ProfilerConfig()

    # ------------------------------------------------------------------
    # Core profiling
    # ------------------------------------------------------------------

    def collect_activations(
        self,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        """Apply *activation_fn* to each row in *hidden_states* and collect
        the resulting neuron activation magnitudes.

        Args:
            activation_fn: ``(hidden_state: np.ndarray) -> np.ndarray`` where
                the input is a 1-D hidden state vector and the output is the
                1-D post-activation neuron magnitude vector.
            hidden_states: ``(n_samples, hidden_dim)`` float32 array of input
                hidden states.

        Returns:
            ``(n_samples, n_neurons)`` float32 activation magnitude matrix.
        """
        results: List[np.ndarray] = []
        for hs in hidden_states:
            act = activation_fn(np.asarray(hs, dtype=np.float32))
            results.append(np.abs(act).astype(np.float32))
        if not results:
            return np.empty((0, 1), dtype=np.float32)
        return np.stack(results, axis=0)

    def compute_neuron_stats(
        self, activation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-neuron mean magnitude and firing frequency.

        Args:
            activation_matrix: ``(n_samples, n_neurons)`` float32.

        Returns:
            Tuple of::
                mean_magnitude  : (n_neurons,) float32
                firing_frequency : (n_neurons,) float32  (fraction of samples active)
        """
        act = np.abs(activation_matrix).astype(np.float32)
        mean_mag = act.mean(axis=0)
        firing_freq = (act > self.config.activation_threshold).astype(np.float32).mean(axis=0)
        return mean_mag, firing_freq

    def profile_layer(
        self,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        hidden_states: np.ndarray,
        layer_idx: int = -1,
    ) -> LayerSparsityProfile:
        """Run the full calibration pipeline for a single FFN layer.

        Args:
            activation_fn: Post-activation callable for this layer.
            hidden_states: ``(n_samples, hidden_dim)`` calibration inputs.
                If ``n_samples > config.n_samples`` the first
                ``config.n_samples`` rows are used.
            layer_idx: Transformer layer index for tagging.

        Returns:
            :class:`LayerSparsityProfile` with cluster assignments and
            activation statistics.
        """
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must be 2-D (n_samples, hidden_dim), "
                f"got shape {hidden_states.shape}"
            )
        # Cap to configured sample count.
        cap = self.config.n_samples
        if hidden_states.shape[0] > cap:
            hidden_states = hidden_states[:cap]

        act_matrix = self.collect_activations(activation_fn, hidden_states)
        n_samples, n_neurons = act_matrix.shape
        k = min(self.config.n_clusters, n_neurons)

        # Dead-neuron sparsity ratio.
        active_mask = act_matrix > self.config.activation_threshold
        sparsity_ratio = float(1.0 - active_mask.astype(np.float32).mean())

        # Build the feature matrix for k-means: each neuron is described by
        # its co-activation frequency with respect to all other neurons in the
        # layer.  Using the full co-activation matrix is O(n_neurons²) so we
        # subsample to 128-dim features for scalability.
        mean_mag, firing_freq = self.compute_neuron_stats(act_matrix)
        feature_dim = min(128, n_neurons)
        if feature_dim == n_neurons:
            features = np.stack(
                [
                    firing_freq,                                # neuron's own frequency
                    mean_mag / (mean_mag.max() + 1e-12),        # normalised magnitude
                ],
                axis=1,
            )  # (n_neurons, 2)
        else:
            rng = np.random.default_rng(self.config.kmeans_seed)
            sample_idx = rng.choice(n_neurons, feature_dim, replace=False)
            coact = coactivation_matrix(act_matrix, self.config.activation_threshold)
            features = coact[:, sample_idx].astype(np.float32)  # (n_neurons, feature_dim)

        assignments, _ = kmeans_cluster(
            features, k=k,
            max_iter=self.config.kmeans_max_iter,
            seed=self.config.kmeans_seed,
        )

        # Build per-cluster statistics.
        cluster_activation = np.zeros(k, dtype=np.float32)
        cluster_info_list: List[ClusterInfo] = []
        for cid in range(k):
            mask = assignments == cid
            neuron_idx = np.where(mask)[0].astype(np.int32)
            if len(neuron_idx) == 0:
                cluster_activation[cid] = 0.0
                cluster_info_list.append(
                    ClusterInfo(
                        cluster_id=cid,
                        neuron_indices=neuron_idx,
                        activation_frequency=0.0,
                        mean_magnitude=0.0,
                    )
                )
                continue
            # A cluster is "active" on a sample if any of its neurons fires.
            cluster_active = active_mask[:, neuron_idx].any(axis=1)
            freq = float(cluster_active.mean())
            cluster_activation[cid] = freq
            cluster_info_list.append(
                ClusterInfo(
                    cluster_id=cid,
                    neuron_indices=neuron_idx,
                    activation_frequency=freq,
                    mean_magnitude=float(mean_mag[mask].mean()),
                )
            )

        # Build cluster_boundaries assuming neurons are sorted by cluster id
        # (as they will be after cluster_reorder).
        sizes = np.array([len(ci.neuron_indices) for ci in cluster_info_list], dtype=np.int32)
        boundaries = np.zeros(k + 1, dtype=np.int32)
        boundaries[1:] = np.cumsum(sizes)

        return LayerSparsityProfile(
            layer_idx=layer_idx,
            n_neurons=n_neurons,
            n_clusters=k,
            cluster_assignments=assignments.astype(np.int32),
            cluster_boundaries=boundaries,
            activation_histogram=cluster_activation,
            sparsity_ratio=sparsity_ratio,
            expected_sparsity=sparsity_ratio,
            clusters=cluster_info_list,
        )

    def profile_model(
        self,
        layer_activation_fns: Sequence[Callable[[np.ndarray], np.ndarray]],
        hidden_states: np.ndarray,
    ) -> List[LayerSparsityProfile]:
        """Profile all FFN layers in a model.

        Args:
            layer_activation_fns: One callable per FFN layer.
            hidden_states: ``(n_samples, hidden_dim)`` calibration inputs
                shared across all layers.

        Returns:
            List of :class:`LayerSparsityProfile`, one per layer.
        """
        return [
            self.profile_layer(fn, hidden_states, layer_idx=i)
            for i, fn in enumerate(layer_activation_fns)
        ]
