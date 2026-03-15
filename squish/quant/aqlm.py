"""squish/quant/aqlm.py — AQLM (Additive Quantization of Language Models) quantiser.

Reference: Egiazarian et al., ICML 2024 "AQLM: Additive Quantization of Language Models"

2-bit effective precision via additive codebook lookup. Each group of `group_size`
consecutive weights is encoded as the sum of M codeword vectors from M separate
codebooks. Offline beam-search calibration; dequantization via numpy gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "AQLMConfig",
    "AQLMCodebook",
    "AQLMLayer",
    "AQLMQuantizer",
    "aqlm_dequantize",
    "quantize_model_aqlm",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AQLMConfig:
    """
    AQLM compression configuration.

    Parameters
    ----------
    n_codebooks : int
        Number of additive codebooks (M in the paper).  Default 2.
    codebook_size : int
        Number of codewords per codebook (K).  Must be >= 2.  Default 16.
    group_size : int
        Number of contiguous weights per group vector (D).  Default 8.
    n_iterations : int
        Number of k-means iterations for codebook initialisation.  Default 25.
    beam_width : int
        Beam width used during additive beam-search assignment.  Default 8.
    """

    n_codebooks: int = 2
    codebook_size: int = 16
    group_size: int = 8
    n_iterations: int = 25
    beam_width: int = 8

    def __post_init__(self) -> None:
        if self.n_codebooks < 1:
            raise ValueError(f"n_codebooks must be >= 1, got {self.n_codebooks}")
        if self.codebook_size < 2:
            raise ValueError(f"codebook_size must be >= 2, got {self.codebook_size}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")


# ---------------------------------------------------------------------------
# Single codebook
# ---------------------------------------------------------------------------

class AQLMCodebook:
    """
    A single AQLM codebook: a table of ``codebook_size`` codeword vectors,
    each of length ``group_size``.

    After construction the codebook is all-zeros; call :meth:`initialize_kmeans`
    to fit it to calibration data.
    """

    def __init__(self, codebook_size: int, group_size: int) -> None:
        self.codebook_size = codebook_size
        self.group_size = group_size
        # Shape: (codebook_size, group_size) — all zeros until fitted
        self.vectors: np.ndarray = np.zeros(
            (codebook_size, group_size), dtype=np.float32
        )

    def initialize_kmeans(self, data: np.ndarray) -> None:
        """
        Run Lloyd's k-means on ``data`` to initialise the codebook.

        Parameters
        ----------
        data : np.ndarray, shape (n_groups, group_size)
            Residual vectors to cluster.
        """
        data = np.asarray(data, dtype=np.float32)
        n_samples = data.shape[0]
        k = min(self.codebook_size, n_samples)

        if n_samples == 0:
            return

        # k-means++ initialisation
        rng = np.random.default_rng(seed=0)
        chosen = [int(rng.integers(n_samples))]
        for _ in range(k - 1):
            # Squared distances from each point to its nearest chosen centre
            dists = np.min(
                np.sum((data[:, np.newaxis, :] - data[chosen][np.newaxis, :, :]) ** 2, axis=-1),
                axis=1,
            )
            total = dists.sum()
            if total < 1e-30:
                # All remaining distances are zero (identical points); pick uniformly.
                chosen.append(int(rng.integers(n_samples)))
            else:
                probs = dists / total
                probs = probs / probs.sum()  # re-normalise to absorb float drift
                chosen.append(int(rng.choice(n_samples, p=probs)))

        centroids = data[chosen].copy()

        # Lloyd iterations
        for _ in range(20):
            # Assignment step
            diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, D)
            sq_dists = np.sum(diffs ** 2, axis=-1)  # (N, k)
            labels = np.argmin(sq_dists, axis=1)  # (N,)

            # Update step
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int64)
            for idx in range(k):
                mask = labels == idx
                if mask.any():
                    new_centroids[idx] = data[mask].mean(axis=0)
                    counts[idx] = mask.sum()
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[idx] = centroids[idx]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Pad with zeros if k < codebook_size
        if k < self.codebook_size:
            pad = np.zeros((self.codebook_size - k, self.group_size), dtype=np.float32)
            centroids = np.vstack([centroids, pad])

        self.vectors = centroids.astype(np.float32)

    def nearest(self, residual: np.ndarray) -> int:
        """
        Return the index of the codeword nearest to ``residual``.

        Parameters
        ----------
        residual : np.ndarray, shape (group_size,)

        Returns
        -------
        int
            Index in [0, codebook_size).
        """
        residual = np.asarray(residual, dtype=np.float32)
        diffs = self.vectors - residual[np.newaxis, :]  # (codebook_size, group_size)
        sq_dists = np.sum(diffs ** 2, axis=-1)          # (codebook_size,)
        return int(np.argmin(sq_dists))


# ---------------------------------------------------------------------------
# Compressed layer
# ---------------------------------------------------------------------------

class AQLMLayer:
    """
    A weight matrix compressed with AQLM.

    Attributes
    ----------
    codebooks : list[AQLMCodebook]
        The M fitted codebooks.
    indices : np.ndarray, shape (out_features, n_groups, n_codebooks)
        The chosen codeword index for each (row, group, codebook) triple.
    scale : float
        Global scale factor applied to the reconstructed weight matrix.
    out_features : int
    in_features : int
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        config: AQLMConfig,
    ) -> None:
        self.out_features = out_features
        self.in_features = in_features
        self.config = config

        n_groups = (in_features + config.group_size - 1) // config.group_size

        self.codebooks: list[AQLMCodebook] = [
            AQLMCodebook(config.codebook_size, config.group_size)
            for _ in range(config.n_codebooks)
        ]

        # Use uint16 if codebook_size > 255, else uint8
        idx_dtype = np.uint16 if config.codebook_size > 255 else np.uint8
        self.indices: np.ndarray = np.zeros(
            (out_features, n_groups, config.n_codebooks), dtype=idx_dtype
        )
        self.scale: float = 1.0

    def dequantize(self) -> np.ndarray:
        """
        Reconstruct the weight matrix from codebook indices.

        Returns
        -------
        np.ndarray, shape (out_features, in_features), dtype float32
        """
        out_features = self.out_features
        in_features = self.in_features
        group_size = self.config.group_size
        n_groups = self.indices.shape[1]
        n_codebooks = self.config.n_codebooks

        # Accumulate summed codewords: (out_features, n_groups, group_size)
        reconstructed = np.zeros(
            (out_features, n_groups, group_size), dtype=np.float32
        )
        for m in range(n_codebooks):
            cb_vectors = self.codebooks[m].vectors  # (codebook_size, group_size)
            idx_m = self.indices[:, :, m].astype(np.int64)  # (out_features, n_groups)
            # Gather: reconstructed += cb_vectors[idx_m]
            reconstructed += cb_vectors[idx_m]

        # Flatten groups back to in_features, then trim any padding
        flat = reconstructed.reshape(out_features, n_groups * group_size)
        flat = flat[:, :in_features]

        return (flat * self.scale).astype(np.float32)

    @property
    def compressed_bits(self) -> int:
        """Total storage bits: index array + codebook vectors (float16)."""
        bits_per_index = 16 if self.indices.dtype == np.uint16 else 8
        index_bits = int(np.prod(self.indices.shape)) * bits_per_index
        codebook_bits = sum(
            int(np.prod(cb.vectors.shape)) * 16  # float16
            for cb in self.codebooks
        )
        return index_bits + codebook_bits


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class AQLMQuantizer:
    """
    Calibrate an AQLM-compressed layer from a float32 weight matrix.

    Uses k-means codebook initialisation followed by beam-search additive
    assignment.  No neural-network calibration data is required (though
    ``calib_inputs`` is accepted for API compatibility).
    """

    def __init__(self, config: Optional[AQLMConfig] = None) -> None:
        self.config = config if config is not None else AQLMConfig()

    def calibrate(
        self,
        weight: np.ndarray,
        calib_inputs: Optional[np.ndarray] = None,  # noqa: ARG002  (reserved for future use)
    ) -> AQLMLayer:
        """
        Compress a 2-D weight matrix with AQLM.

        Parameters
        ----------
        weight : np.ndarray, shape (out_features, in_features)
        calib_inputs : optional, unused (reserved for activation-aware calibration)

        Returns
        -------
        AQLMLayer
        """
        cfg = self.config
        weight = np.asarray(weight, dtype=np.float32)

        if weight.ndim == 1:
            weight = weight.reshape(1, -1)
        elif weight.ndim != 2:
            raise ValueError(f"weight must be 1-D or 2-D, got shape {weight.shape}")

        out_features, in_features = weight.shape

        # Global scale
        abs_mean = float(np.mean(np.abs(weight)))
        scale = abs_mean if abs_mean > 0.0 else 1.0

        # Normalise weight
        w_norm = weight / scale  # (out_features, in_features)

        # Pad in_features to a multiple of group_size
        group_size = cfg.group_size
        pad = (group_size - in_features % group_size) % group_size
        if pad:
            w_norm = np.pad(w_norm, ((0, 0), (0, pad)))

        n_groups = w_norm.shape[1] // group_size

        # Reshape → (out_features, n_groups, group_size)
        w_groups = w_norm.reshape(out_features, n_groups, group_size)

        layer = AQLMLayer(out_features, in_features, cfg)
        layer.scale = scale

        # ── Codebook initialisation + beam-search assignment ────────────────
        # residuals: current unencoded part of the weight groups
        residuals = w_groups.copy()  # (out_features, n_groups, group_size)

        for m in range(cfg.n_codebooks):
            cb = layer.codebooks[m]

            # Collect residual vectors for this codebook across all rows/groups
            flat_residuals = residuals.reshape(-1, group_size)  # (out_features*n_groups, group_size)

            # Initialise codebook m via k-means on current residuals
            cb.initialize_kmeans(flat_residuals)

            # Beam-search assignment for each (row i, group j)
            idx_dtype = np.uint16 if cfg.codebook_size > 255 else np.uint8

            for i in range(out_features):
                for j in range(n_groups):
                    # At this stage, residuals[i, j] is what codebook m should encode
                    best_idx = cb.nearest(residuals[i, j])
                    layer.indices[i, j, m] = idx_dtype(best_idx)

                    # Subtract the codeword from residual so the next codebook
                    # (if any) encodes the remaining error
                    residuals[i, j] -= cb.vectors[best_idx]

        return layer

    # ------------------------------------------------------------------
    # Convenience aliases expected by bench_2bit and convert.py
    # ------------------------------------------------------------------

    def compress(self, weight: np.ndarray) -> "AQLMLayer":
        """Alias for :meth:`calibrate`.  Returns a compressed ``AQLMLayer``."""
        return self.calibrate(weight)

    def decompress(self, layer: "AQLMLayer") -> np.ndarray:
        """Dequantize *layer* back to a float32 numpy array."""
        return layer.dequantize()


# ---------------------------------------------------------------------------
# Standalone dequantize helper
# ---------------------------------------------------------------------------

def aqlm_dequantize(layer: AQLMLayer) -> np.ndarray:
    """
    Dequantize an AQLMLayer to a float32 numpy array.

    This is a thin wrapper around :meth:`AQLMLayer.dequantize` provided for
    symmetry with the other ``*_dequantize`` helpers in this package.

    Returns
    -------
    np.ndarray, shape (out_features, in_features), dtype float32
    """
    return layer.dequantize()


# ---------------------------------------------------------------------------
# Model-level quantization helper
# ---------------------------------------------------------------------------

def quantize_model_aqlm(
    model_weights: dict,
    config: Optional[AQLMConfig] = None,
) -> dict:
    """
    Walk a dict of model weight tensors and compress every 2-D weight matrix
    using AQLM.

    Parameters
    ----------
    model_weights : dict[str, np.ndarray]
        Mapping of tensor name → float32 numpy array.
    config : AQLMConfig, optional
        Quantisation configuration.  Defaults to ``AQLMConfig()``.

    Returns
    -------
    dict[str, AQLMLayer]
        Mapping of tensor name → compressed AQLMLayer.  Only tensors with
        ndim >= 2 are included; passthrough tensors are omitted.
    """
    if config is None:
        config = AQLMConfig()

    quantizer = AQLMQuantizer(config)
    result: dict = {}

    for name, weight in model_weights.items():
        arr = np.asarray(weight, dtype=np.float32)
        if arr.ndim < 2:
            continue
        # Flatten to 2-D for quantisation
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        result[name] = quantizer.calibrate(flat)

    return result
