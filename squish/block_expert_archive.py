"""
squish/block_expert_archive.py

Squish Block-Expert Archive + Self-Learning System — core module.

A `.squish/` bundle directory holds a compressed base model together with a
*block-expert archive*: a set of layer-weight clusters (experts) produced by
K-means over block-level weight similarity.  At inference time a routing
function selects which expert cluster to use for each Transformer block.

Bundle layout
─────────────
::

    ~/.squish/archives/<name>/
        manifest.json          — bundle metadata
        base/                  — base model weights (INT4 / safetensors tier)
        experts/
            block_<l>_k<k>.npy — packed expert weight delta for layer l, cluster k
        router.json            — routing table: block → cluster assignment

Public API
──────────
    BlockExpertConfig       — dataclass of hyper-parameters
    BlockExpertArchive      — create / load / save archives
    ExpertRouter            — select expert for a block at decode time
    ArchiveStats            — lightweight telemetry dataclass
    cluster_block_weights   — K-means clustering driver (numpy-only)
    pack_expert_delta       — delta-compress one expert relative to base
    unpack_expert_delta     — reconstruct full weight from base + delta
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BlockExpertConfig:
    """Hyper-parameters for the Block-Expert Archive.

    Parameters
    ----------
    n_clusters:
        Number of K-means clusters (experts) per Transformer block.
        Typical range 2–8; 4 is a good default.
    n_iter:
        K-means iterations.
    similarity_metric:
        ``"cosine"`` or ``"l2"`` — used to measure block-weight similarity
        during clustering and routing.
    delta_bits:
        Bit-width for storing expert deltas (4 or 8).  4-bit halves delta
        storage; 8-bit gives higher fidelity.
    min_delta_snr_db:
        Minimum acceptable SNR (dB) when validating a packed expert delta.
        Deltas with SNR below this threshold are stored at higher precision.
    router_temperature:
        Softmax temperature for the routing logit distribution.  Lower = more
        winner-takes-all; higher = softer mixing.
    archive_version:
        Bundle format version string.
    """

    n_clusters: int = 4
    n_iter: int = 20
    similarity_metric: str = "cosine"
    delta_bits: int = 8
    min_delta_snr_db: float = 30.0
    router_temperature: float = 0.5
    archive_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be ≥ 1")
        if self.n_iter < 1:
            raise ValueError("n_iter must be ≥ 1")
        if self.similarity_metric not in ("cosine", "l2"):
            raise ValueError("similarity_metric must be 'cosine' or 'l2'")
        if self.delta_bits not in (4, 8):
            raise ValueError("delta_bits must be 4 or 8")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")


# ─────────────────────────────────────────────────────────────────────────────
# Delta compression helpers
# ─────────────────────────────────────────────────────────────────────────────


def pack_expert_delta(
    expert_weight: np.ndarray,
    base_weight: np.ndarray,
    bits: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise (expert − base) to *bits*-bit integers.

    Parameters
    ----------
    expert_weight:
        Full-precision expert weight array.
    base_weight:
        Full-precision base weight array (same shape).
    bits:
        Storage precision — 4 or 8.

    Returns
    -------
    delta_q : np.ndarray
        Quantised delta, dtype int8 (8-bit) or packed uint8 (4-bit nibbles).
    scales : np.ndarray
        Per-row scale factors, shape (nrows,).
    zeros : np.ndarray
        Per-row zero-points, shape (nrows,).
    """
    if expert_weight.shape != base_weight.shape:
        raise ValueError(
            f"Shape mismatch: expert {expert_weight.shape} vs base {base_weight.shape}"
        )

    delta = (expert_weight - base_weight).astype(np.float32)
    if delta.ndim == 1:
        delta = delta.reshape(1, -1)

    nrows, ncols = delta.shape
    clamp = (1 << (bits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit

    row_max = np.abs(delta).max(axis=1, keepdims=True).clip(min=1e-8)
    scales = (row_max / clamp).squeeze(1).astype(np.float32)
    zeros = np.zeros(nrows, dtype=np.float32)

    delta_q_f32 = np.round(delta / row_max * clamp).clip(-clamp, clamp)
    delta_q = delta_q_f32.astype(np.int8)

    if bits == 4:
        # Pack pairs of int4 values into uint8 nibbles
        # Shift to unsigned [0, 15] range before packing
        unsigned = (delta_q_f32.astype(np.int8) + 8).clip(0, 15).astype(np.uint8)
        if ncols % 2 != 0:
            unsigned = np.pad(unsigned, ((0, 0), (0, 1)))
        packed = (unsigned[:, 0::2] & 0x0F) | ((unsigned[:, 1::2] & 0x0F) << 4)
        delta_q = packed

    return delta_q, scales, zeros


def unpack_expert_delta(
    delta_q: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    base_weight: np.ndarray,
    bits: int = 8,
    original_ncols: int | None = None,
) -> np.ndarray:
    """Reconstruct full expert weight from quantised delta + base.

    Parameters
    ----------
    delta_q:
        Quantised delta from :func:`pack_expert_delta`.
    scales:
        Per-row scale factors.
    zeros:
        Per-row zero-points (currently unused, reserved for asymmetric quant).
    base_weight:
        Original base weight array.
    bits:
        Storage precision matching the packing call.
    original_ncols:
        Original column count before nibble padding (required for bits=4).

    Returns
    -------
    np.ndarray
        Reconstructed expert weight, same shape as *base_weight*.
    """
    if bits == 4:
        ncols = original_ncols if original_ncols is not None else delta_q.shape[1] * 2
        lo = (delta_q & 0x0F).astype(np.int8)
        hi = ((delta_q >> 4) & 0x0F).astype(np.int8)
        signed = np.empty((delta_q.shape[0], delta_q.shape[1] * 2), dtype=np.int8)
        signed[:, 0::2] = lo
        signed[:, 1::2] = hi
        signed = (signed - 8).astype(np.float32)
        if signed.shape[1] > ncols:
            signed = signed[:, :ncols]
        delta_f32 = signed * scales[:, None]
    else:
        delta_f32 = delta_q.astype(np.float32) * scales[:, None]

    orig_shape = base_weight.shape
    delta_reshaped = delta_f32.reshape(orig_shape)
    return (base_weight.astype(np.float32) + delta_reshaped).reshape(orig_shape)


def _delta_snr_db(
    expert_weight: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Compute SNR (dB) between original expert weight and reconstructed."""
    signal_power = float(np.mean(expert_weight.astype(np.float32) ** 2))
    noise = expert_weight.astype(np.float32) - reconstructed.astype(np.float32)
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-30:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# K-means clustering over block weight matrices
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance ∈ [0, 2] between flattened arrays."""
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a_f, b_f) / (na * nb))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised L2 distance between flattened arrays."""
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = float(np.linalg.norm(a_f) + np.linalg.norm(b_f))
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(a_f - b_f) / denom)


def cluster_block_weights(
    weight_snapshots: list[np.ndarray],
    n_clusters: int,
    n_iter: int = 20,
    metric: str = "cosine",
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """K-means clustering of weight snapshots for one Transformer block.

    Each element of *weight_snapshots* is a weight matrix captured during a
    different training / fine-tuning step.  We cluster the snapshots to find
    ``n_clusters`` representative expert weight matrices.

    Parameters
    ----------
    weight_snapshots:
        List of weight arrays (all same shape) — one per training snapshot.
    n_clusters:
        Number of expert clusters to produce.
    n_iter:
        Number of K-means iterations.
    metric:
        ``"cosine"`` or ``"l2"``.
    rng:
        Optional seeded Generator for reproducibility.

    Returns
    -------
    labels : list[int]
        Cluster assignment (0 … n_clusters−1) for each snapshot.
    centroids : list[np.ndarray]
        Centroid weight matrix for each cluster (same shape as inputs).
    """
    if not weight_snapshots:
        raise ValueError("weight_snapshots must not be empty")

    rng = rng or np.random.default_rng(42)
    n = len(weight_snapshots)
    k = min(n_clusters, n)

    dist_fn = _cosine_distance if metric == "cosine" else _l2_distance

    # Initialise centroids by k-means++ style seeding
    centroid_indices = [int(rng.integers(0, n))]
    while len(centroid_indices) < k:
        dists = np.array([
            min(dist_fn(weight_snapshots[i], weight_snapshots[c]) for c in centroid_indices)
            for i in range(n)
        ])
        probs = dists ** 2
        total = probs.sum()
        if total < 1e-30:
            break
        probs /= total
        next_idx = int(rng.choice(n, p=probs))
        centroid_indices.append(next_idx)

    centroids = [weight_snapshots[i].astype(np.float32).copy() for i in centroid_indices]
    # Pad to requested k if fewer than k unique snapshots exist
    while len(centroids) < n_clusters:
        centroids.append(centroids[0].copy())

    labels = [0] * n

    for _iter in range(n_iter):
        # Assignment step
        new_labels = []
        for i in range(n):
            dists = [dist_fn(weight_snapshots[i], centroids[c]) for c in range(k)]
            new_labels.append(int(np.argmin(dists)))
        labels = new_labels

        # Update step — recompute centroids as mean of assigned snapshots
        new_centroids = []
        for c in range(k):
            assigned = [weight_snapshots[i] for i in range(n) if labels[i] == c]
            if assigned:
                new_centroids.append(
                    np.mean(np.stack([a.astype(np.float32) for a in assigned], axis=0), axis=0)
                )
            else:
                new_centroids.append(centroids[c].copy())
        centroids = new_centroids

    # Pad to requested n_clusters if k < n_clusters (fewer unique snapshots than requested)
    while len(centroids) < n_clusters:
        centroids.append(centroids[0].copy())

    return labels, centroids


# ─────────────────────────────────────────────────────────────────────────────
# Expert routing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExpertRoutingStats:
    """Telemetry for one routing call."""
    block_idx: int
    selected_cluster: int
    logit_scores: list[float]
    routing_time_us: float


class ExpertRouter:
    """Routes each Transformer block to an expert cluster at decode time.

    Parameters
    ----------
    routing_table:
        Mapping of ``block_idx → list[centroid_embedding]`` produced during
        archiving.  Each centroid embedding is a 1-D float32 array (the
        flattened + normalised centroid weight of that cluster).
    config:
        :class:`BlockExpertConfig` instance.
    """

    def __init__(
        self,
        routing_table: dict[int, list[np.ndarray]],
        config: BlockExpertConfig,
    ) -> None:
        self._table = routing_table
        self._config = config
        self._call_count: int = 0

    @property
    def n_blocks(self) -> int:
        """Number of Transformer blocks registered in the routing table."""
        return len(self._table)

    def route(
        self,
        block_idx: int,
        current_weight: np.ndarray,
    ) -> tuple[int, ExpertRoutingStats]:
        """Select the best expert cluster for *block_idx* given *current_weight*.

        Parameters
        ----------
        block_idx:
            Index of the Transformer block (0-based).
        current_weight:
            Current weight matrix for this block (used to compute similarity
            scores against each centroid).

        Returns
        -------
        cluster_idx : int
            Index of the selected expert cluster.
        stats : ExpertRoutingStats
            Routing telemetry.

        Raises
        ------
        KeyError
            If *block_idx* is not in the routing table.
        """
        if block_idx not in self._table:
            raise KeyError(f"Block {block_idx} not in routing table")

        t0 = time.perf_counter()
        centroids = self._table[block_idx]
        dist_fn = (
            _cosine_distance
            if self._config.similarity_metric == "cosine"
            else _l2_distance
        )

        def _embed(w: np.ndarray) -> np.ndarray:
            """Reduce weight matrix to a 1-D row-mean embedding for routing."""
            return w.mean(axis=0).astype(np.float64) if w.ndim > 1 else w.flatten().astype(np.float64)

        query_emb = _embed(current_weight)

        # Compute similarity score (−distance) for each centroid
        raw_scores = np.array([
            -dist_fn(query_emb, _embed(c)) for c in centroids
        ], dtype=np.float64)

        # Softmax with temperature
        T = self._config.router_temperature
        shifted = raw_scores - raw_scores.max()
        exp_s = np.exp(shifted / T)
        probs = exp_s / exp_s.sum()
        selected = int(np.argmax(probs))

        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._call_count += 1

        stats = ExpertRoutingStats(
            block_idx=block_idx,
            selected_cluster=selected,
            logit_scores=probs.tolist(),
            routing_time_us=elapsed_us,
        )
        return selected, stats

    def reset_stats(self) -> None:
        """Reset the routing call counter."""
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count


# ─────────────────────────────────────────────────────────────────────────────
# Archive stats
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ArchiveStats:
    """Aggregate statistics for a :class:`BlockExpertArchive`."""
    n_blocks: int = 0
    n_experts_total: int = 0
    avg_delta_snr_db: float = 0.0
    archive_size_mb: float = 0.0
    learn_epochs: int = 0
    last_learned: str = ""

    @property
    def avg_experts_per_block(self) -> float:
        if self.n_blocks == 0:
            return 0.0
        return self.n_experts_total / self.n_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Block-Expert Archive
# ─────────────────────────────────────────────────────────────────────────────


class BlockExpertArchive:
    """Container for the block-expert archive bundle.

    A bundle is a directory with the layout::

        <bundle_dir>/
            manifest.json
            experts/
                block_<l>_k<k>.npy    — INT4/INT8 packed delta
                block_<l>_k<k>_sc.npy — scale factors for delta
                block_<l>_k<k>_zp.npy — zero-points for delta
            router.json                — mapping block→[centroids as list-of-list]

    Parameters
    ----------
    bundle_dir:
        Path to the bundle directory.  Created by :meth:`create` or loaded by
        :meth:`load`.
    config:
        :class:`BlockExpertConfig` describing the archive.
    """

    def __init__(self, bundle_dir: Path, config: BlockExpertConfig) -> None:
        self._dir = Path(bundle_dir)
        self._config = config
        # block_idx → {cluster_idx → (delta_q, scales, zeros, original_ncols)}
        self._experts: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {}
        # block_idx → list of centroid embeddings (1-D float32)
        self._router_centroids: dict[int, list[np.ndarray]] = {}
        self._stats = ArchiveStats()
        self._router: ExpertRouter | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def config(self) -> BlockExpertConfig:
        return self._config

    @property
    def stats(self) -> ArchiveStats:
        return self._stats

    @property
    def bundle_dir(self) -> Path:
        return self._dir

    @property
    def router(self) -> ExpertRouter:
        if self._router is None:
            self._router = ExpertRouter(self._router_centroids, self._config)
        return self._router

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        bundle_dir: str | Path,
        block_weights: dict[int, list[np.ndarray]],
        base_weights: dict[int, np.ndarray],
        config: BlockExpertConfig | None = None,
    ) -> BlockExpertArchive:
        """Build a new archive from a dict of per-block weight snapshots.

        Parameters
        ----------
        bundle_dir:
            Destination directory (will be created).
        block_weights:
            ``block_idx → list of weight snapshots`` (one per training step).
        base_weights:
            ``block_idx → base weight`` (the starting point for delta encoding).
        config:
            :class:`BlockExpertConfig` — uses defaults if omitted.

        Returns
        -------
        BlockExpertArchive
            Populated archive, not yet saved to disk.
        """
        config = config or BlockExpertConfig()
        archive = cls(Path(bundle_dir), config)

        snr_values: list[float] = []
        n_experts_total = 0

        for block_idx, snapshots in sorted(block_weights.items()):
            base = base_weights.get(block_idx, snapshots[0])

            # Cluster snapshots → expert centroids
            labels, centroids = cluster_block_weights(
                snapshots,
                n_clusters=config.n_clusters,
                n_iter=config.n_iter,
                metric=config.similarity_metric,
            )

            # Store centroid embeddings (mean-row projection, normalised) for the router.
            # Using mean over all but the last axis gives a (ncols,) vector regardless of
            # the weight shape, matching the _embed() reduction in ExpertRouter.route().
            centroid_embeddings = []
            for c in centroids:
                emb = c.mean(axis=0).astype(np.float32) if c.ndim > 1 else c.astype(np.float32)
                norm = np.linalg.norm(emb)
                centroid_embeddings.append(emb / (norm + 1e-12))
            archive._router_centroids[block_idx] = centroid_embeddings

            # Pack each expert delta
            cluster_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
            for k, centroid in enumerate(centroids):
                orig_ncols = centroid.shape[-1] if centroid.ndim > 1 else centroid.shape[0]
                delta_q, scales, zeros = pack_expert_delta(centroid, base, bits=config.delta_bits)
                reconstructed = unpack_expert_delta(
                    delta_q, scales, zeros, base,
                    bits=config.delta_bits, original_ncols=orig_ncols,
                )
                snr = _delta_snr_db(centroid, reconstructed)
                if snr < config.min_delta_snr_db and config.delta_bits < 8:
                    # Fall back to INT8 for this expert
                    delta_q, scales, zeros = pack_expert_delta(centroid, base, bits=8)
                    reconstructed = unpack_expert_delta(delta_q, scales, zeros, base, bits=8)
                    snr = _delta_snr_db(centroid, reconstructed)
                snr_values.append(snr)
                cluster_map[k] = (delta_q, scales, zeros, orig_ncols)
                n_experts_total += 1

            archive._experts[block_idx] = cluster_map

        # Update stats
        archive._stats = ArchiveStats(
            n_blocks=len(block_weights),
            n_experts_total=n_experts_total,
            avg_delta_snr_db=float(np.mean(snr_values)) if snr_values else 0.0,
            learn_epochs=1,
            last_learned=_utcnow(),
        )
        archive._router = None  # will be rebuilt on first access
        return archive

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the archive to :attr:`bundle_dir`."""
        experts_dir = self._dir / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

        for block_idx, cluster_map in self._experts.items():
            for k, (delta_q, scales, zeros, orig_ncols) in cluster_map.items():
                prefix = experts_dir / f"block_{block_idx}_k{k}"
                np.save(str(prefix) + ".npy", delta_q)
                np.save(str(prefix) + "_sc.npy", scales)
                np.save(str(prefix) + "_zp.npy", zeros)

        # Serialise router centroids as JSON-compatible nested lists
        router_data: dict[str, Any] = {}
        for block_idx, embeddings in self._router_centroids.items():
            router_data[str(block_idx)] = [e.tolist() for e in embeddings]
        (self._dir / "router.json").write_text(json.dumps(router_data, indent=2))

        # Manifest
        manifest = {
            "version": self._config.archive_version,
            "created": _utcnow(),
            "config": {
                "n_clusters": self._config.n_clusters,
                "n_iter": self._config.n_iter,
                "similarity_metric": self._config.similarity_metric,
                "delta_bits": self._config.delta_bits,
                "min_delta_snr_db": self._config.min_delta_snr_db,
                "router_temperature": self._config.router_temperature,
            },
            "stats": {
                "n_blocks": self._stats.n_blocks,
                "n_experts_total": self._stats.n_experts_total,
                "avg_delta_snr_db": self._stats.avg_delta_snr_db,
                "learn_epochs": self._stats.learn_epochs,
                "last_learned": self._stats.last_learned,
            },
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._update_disk_size()

    @classmethod
    def load(cls, bundle_dir: str | Path) -> BlockExpertArchive:
        """Load an archive previously written by :meth:`save`.

        Parameters
        ----------
        bundle_dir:
            Path to the bundle directory containing ``manifest.json``.

        Returns
        -------
        BlockExpertArchive

        Raises
        ------
        FileNotFoundError
            If ``manifest.json`` is missing.
        """
        path = Path(bundle_dir)
        manifest_file = path / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"No manifest.json in {path}")

        manifest = json.loads(manifest_file.read_text())
        cfg_data = manifest.get("config", {})
        config = BlockExpertConfig(
            n_clusters=cfg_data.get("n_clusters", 4),
            n_iter=cfg_data.get("n_iter", 20),
            similarity_metric=cfg_data.get("similarity_metric", "cosine"),
            delta_bits=cfg_data.get("delta_bits", 8),
            min_delta_snr_db=cfg_data.get("min_delta_snr_db", 30.0),
            router_temperature=cfg_data.get("router_temperature", 0.5),
            archive_version=manifest.get("version", "1.0"),
        )

        archive = cls(path, config)

        # Load router centroids
        router_file = path / "router.json"
        if router_file.is_file():
            router_data = json.loads(router_file.read_text())
            archive._router_centroids = {
                int(bi): [np.array(e, dtype=np.float32) for e in embeddings]
                for bi, embeddings in router_data.items()
            }

        # Load expert deltas
        experts_dir = path / "experts"
        if experts_dir.is_dir():
            for npy_file in sorted(experts_dir.glob("block_*_k*.npy")):
                stem = npy_file.stem  # e.g. "block_0_k1"
                if stem.endswith("_sc") or stem.endswith("_zp"):
                    continue
                parts = stem.split("_")
                block_idx = int(parts[1])
                k = int(parts[2][1:])
                scales = np.load(str(experts_dir / f"{stem}_sc.npy"))
                zeros  = np.load(str(experts_dir / f"{stem}_zp.npy"))
                delta_q = np.load(str(npy_file))
                orig_ncols = delta_q.shape[1] * 2 if config.delta_bits == 4 else delta_q.shape[1]
                archive._experts.setdefault(block_idx, {})[k] = (
                    delta_q, scales, zeros, orig_ncols
                )

        # Restore stats
        stats_data = manifest.get("stats", {})
        archive._stats = ArchiveStats(
            n_blocks=stats_data.get("n_blocks", 0),
            n_experts_total=stats_data.get("n_experts_total", 0),
            avg_delta_snr_db=stats_data.get("avg_delta_snr_db", 0.0),
            learn_epochs=stats_data.get("learn_epochs", 0),
            last_learned=stats_data.get("last_learned", ""),
        )
        archive._update_disk_size()
        return archive

    # ── Expert retrieval ──────────────────────────────────────────────────────

    def get_expert_weight(
        self,
        block_idx: int,
        cluster_idx: int,
        base_weight: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct a full expert weight from its packed delta + base.

        Parameters
        ----------
        block_idx:
            Transformer block index.
        cluster_idx:
            Expert cluster index within that block.
        base_weight:
            Base weight array for this block.

        Returns
        -------
        np.ndarray
            Reconstructed expert weight (float32, same shape as *base_weight*).

        Raises
        ------
        KeyError
            If the block or cluster is not in the archive.
        """
        if block_idx not in self._experts:
            raise KeyError(f"Block {block_idx} not in archive")
        if cluster_idx not in self._experts[block_idx]:
            raise KeyError(f"Cluster {cluster_idx} not in block {block_idx}")

        delta_q, scales, zeros, orig_ncols = self._experts[block_idx][cluster_idx]
        return unpack_expert_delta(
            delta_q, scales, zeros, base_weight,
            bits=self._config.delta_bits,
            original_ncols=orig_ncols,
        )

    def num_blocks(self) -> int:
        return len(self._experts)

    def num_experts(self, block_idx: int) -> int:
        return len(self._experts.get(block_idx, {}))

    # ── Incremental learning ──────────────────────────────────────────────────

    def absorb_snapshot(
        self,
        block_idx: int,
        new_weight: np.ndarray,
        base_weight: np.ndarray,
    ) -> int:
        """Absorb a new weight snapshot into block *block_idx*.

        Finds the closest existing centroid, updates it with an EMA step, and
        returns the cluster index that was updated.

        Parameters
        ----------
        block_idx:
            Which Transformer block this weight belongs to.
        new_weight:
            The new weight observation (post fine-tuning snapshot).
        base_weight:
            The base weight for this block (used for delta re-compression).

        Returns
        -------
        int
            The cluster index that was updated.
        """
        if block_idx not in self._router_centroids:
            # New block — add as a single-expert cluster
            emb = new_weight.mean(axis=0).astype(np.float32) if new_weight.ndim > 1 else new_weight.astype(np.float32)
            norm = np.linalg.norm(emb)
            self._router_centroids[block_idx] = [emb / (norm + 1e-12)]
            delta_q, scales, zeros = pack_expert_delta(
                new_weight, base_weight, bits=self._config.delta_bits
            )
            orig_ncols = new_weight.shape[-1] if new_weight.ndim > 1 else new_weight.shape[0]
            self._experts.setdefault(block_idx, {})[0] = (delta_q, scales, zeros, orig_ncols)
            self._stats.learn_epochs += 1
            self._stats.last_learned = _utcnow()
            return 0

        # Find closest centroid via routing
        selected_k, _ = self.router.route(block_idx, new_weight)

        # EMA update of centroid (α = 0.1)
        alpha = 0.1
        old_centroid = self._router_centroids[block_idx][selected_k]
        flat_new = new_weight.mean(axis=0).astype(np.float32) if new_weight.ndim > 1 else new_weight.astype(np.float32)
        norm = np.linalg.norm(flat_new)
        flat_new_norm = flat_new / (norm + 1e-12)
        updated = (1 - alpha) * old_centroid + alpha * flat_new_norm
        updated_norm = np.linalg.norm(updated)
        self._router_centroids[block_idx][selected_k] = updated / (updated_norm + 1e-12)

        # Re-pack delta using the absorbed weight directly as the new expert observation.
        # The EMA-updated routing embedding guides cluster selection; the actual stored
        # delta captures the incoming weight relative to base.
        orig_ncols = new_weight.shape[-1] if new_weight.ndim > 1 else new_weight.shape[0]
        delta_q, scales, zeros = pack_expert_delta(
            new_weight, base_weight, bits=self._config.delta_bits
        )
        self._experts[block_idx][selected_k] = (delta_q, scales, zeros, orig_ncols)

        # Invalidate cached router so centroids are reloaded on next access
        self._router = None
        self._stats.learn_epochs += 1
        self._stats.last_learned = _utcnow()
        return selected_k

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _update_disk_size(self) -> None:
        total = 0
        if self._dir.is_dir():
            for f in self._dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        self._stats.archive_size_mb = total / (1024 * 1024)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dict."""
        return {
            "bundle_dir": str(self._dir),
            "n_blocks": self._stats.n_blocks,
            "n_experts_total": self._stats.n_experts_total,
            "avg_experts_per_block": self._stats.avg_experts_per_block,
            "avg_delta_snr_db": round(self._stats.avg_delta_snr_db, 2),
            "archive_size_mb": round(self._stats.archive_size_mb, 3),
            "learn_epochs": self._stats.learn_epochs,
            "last_learned": self._stats.last_learned,
            "config": {
                "n_clusters": self._config.n_clusters,
                "delta_bits": self._config.delta_bits,
                "similarity_metric": self._config.similarity_metric,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────


def archive_manifest_hash(bundle_dir: str | Path) -> str:
    """Return a short SHA256 digest of the archive's manifest.json."""
    manifest_file = Path(bundle_dir) / "manifest.json"
    if not manifest_file.is_file():
        return ""
    data = manifest_file.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
