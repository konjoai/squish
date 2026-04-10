"""squish/quant/aqlm.py — Additive Quantization of Language Models (AQLM).

AQLM (Egiazarian et al., 2024  https://arxiv.org/abs/2401.06118) represents
each linear-layer weight row as the sum of K look-up vectors drawn from K
separate codebooks:

    Ŵ[i, g] = scale · ∑_{k=0}^{K-1}  CB_k[ indices[i, g, k] ]

where:
  i          — output feature index  (0 … out_features-1)
  g          — group index within row (0 … n_groups-1)
  K          — number of codebooks (n_codebooks)
  CB_k       — codebook k; shape (codebook_size, group_size)
  indices    — int16 index tensor; shape (out_features, n_groups, K)
  scale      — global scalar multiplier (float32, learned during quantisation)
  group_size — number of input features per group

Dequantisation complexity: O(out_features · n_groups · K) gather ops — fully
vectorisable with NumPy advanced indexing.

Storage layout in Squish npy-dir archives
-----------------------------------------
  <stem>__aqlm_idx.npy   — int16 array; shape (out_features, n_groups, K)
  <stem>__aqlm_cb.npy    — float32 flat array; layout:
      [scale, float(codebook_size), float(group_size), *cb_vectors...]
      cb_vectors reshape → (K, codebook_size, group_size)

This module is imported lazily by squish.quant.compressed_loader; if it is
absent the loader falls through to other decoding paths without raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Public exports ─────────────────────────────────────────────────────────────

__all__ = [
    "AQLMConfig",
    "AQLMCodebook",
    "AQLMLayer",
    "aqlm_dequantize",
    "AQLMEncoder",
    "encode_weight_matrix",
]


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class AQLMConfig:
    """Hyper-parameters that describe the AQLM quantisation grid.

    Attributes:
        n_codebooks:   number of additive codebooks K (typically 1–4).
        codebook_size: number of vectors per codebook (typically 256 for 8-bit
                       codebook indices, 65536 for 16-bit indices).
        group_size:    number of input features encoded per group vector
                       (typically 8 or 16).
    """

    n_codebooks: int
    codebook_size: int
    group_size: int

    def __post_init__(self) -> None:
        if self.n_codebooks < 1:
            raise ValueError(f"n_codebooks must be ≥ 1, got {self.n_codebooks}")
        if self.codebook_size < 2:
            raise ValueError(f"codebook_size must be ≥ 2, got {self.codebook_size}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")


@dataclass
class AQLMCodebook:
    """One codebook in an AQLMLayer.

    Attributes:
        vectors: float32 ndarray of shape (codebook_size, group_size).
                 Each row is a basis vector that can be selected by index.
    """

    vectors: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))

    def __post_init__(self) -> None:
        if self.vectors.size > 0 and self.vectors.ndim != 2:
            raise ValueError(
                f"AQLMCodebook.vectors must be 2-D (codebook_size, group_size), "
                f"got shape {self.vectors.shape}"
            )


class AQLMLayer:
    """Compressed representation of one weight matrix using AQLM.

    Attributes:
        out_features: number of output channels.
        in_features:  number of input channels (= n_groups * group_size).
        cfg:          AQLMConfig describing the codebook grid.
        scale:        global weight scale; multiply reconstructed weights by this.
        indices:      int16/int32 ndarray; shape (out_features, n_groups, K).
        codebooks:    list of K AQLMCodebook objects.
    """

    def __init__(self, out_features: int, in_features: int, cfg: AQLMConfig) -> None:
        if in_features % cfg.group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({cfg.group_size})"
            )
        self.out_features: int = out_features
        self.in_features: int = in_features
        self.cfg: AQLMConfig = cfg
        self.scale: float = 1.0
        # indices will be assigned by the loader; pre-allocate with zeros
        n_groups = in_features // cfg.group_size
        self.indices: np.ndarray = np.zeros(
            (out_features, n_groups, cfg.n_codebooks), dtype=np.int32
        )
        self.codebooks: List[AQLMCodebook] = [
            AQLMCodebook() for _ in range(cfg.n_codebooks)
        ]

    @property
    def n_groups(self) -> int:
        return self.in_features // self.cfg.group_size


# ── Core dequantisation ────────────────────────────────────────────────────────


def aqlm_dequantize(layer: AQLMLayer) -> np.ndarray:
    """Reconstruct the full-precision weight matrix from an AQLMLayer.

    Algorithm:

        W[i, g*gs : (g+1)*gs] = scale · ∑_{k=0}^{K-1} CB_k[ indices[i, g, k] ]

    where gs = group_size.

    Vectorised via NumPy advanced indexing:
        For each codebook k:  accumulated[i, g, :] += CB_k[ indices[i, g, k] ]

    Complexity: O(out_features · n_groups · K) memory gathers.

    Args:
        layer: AQLMLayer with populated indices, codebooks, and scale.

    Returns:
        float32 ndarray of shape (out_features, in_features).

    Raises:
        ValueError: if indices shape is inconsistent with layer dimensions or
                    codebook vectors have the wrong shape.
    """
    cfg = layer.cfg
    indices = np.asarray(layer.indices)  # (out_features, n_groups, K)

    if indices.ndim != 3:
        raise ValueError(
            f"indices must be 3-D (out_features, n_groups, K), got ndim={indices.ndim}"
        )
    out_features, n_groups, K = indices.shape
    if K != cfg.n_codebooks:
        raise ValueError(
            f"indices.shape[-1]={K} does not match cfg.n_codebooks={cfg.n_codebooks}"
        )
    if len(layer.codebooks) != K:
        raise ValueError(
            f"layer.codebooks length {len(layer.codebooks)} != n_codebooks {K}"
        )

    # Accumulate over K codebooks into shape (out_features, n_groups, group_size)
    accumulated = np.zeros((out_features, n_groups, cfg.group_size), dtype=np.float32)

    for k in range(K):
        cb_vectors = np.asarray(layer.codebooks[k].vectors, dtype=np.float32)
        if cb_vectors.shape != (cfg.codebook_size, cfg.group_size):
            raise ValueError(
                f"codebooks[{k}].vectors shape {cb_vectors.shape} does not match "
                f"expected ({cfg.codebook_size}, {cfg.group_size})"
            )
        idx_k = indices[:, :, k]  # (out_features, n_groups) — integer indices
        # Advanced gather: cb_vectors[idx_k] → (out_features, n_groups, group_size)
        accumulated += cb_vectors[idx_k]

    # Apply global scale and flatten groups → in_features
    return (accumulated * layer.scale).reshape(out_features, n_groups * cfg.group_size)


# ── K-means codebook training ──────────────────────────────────────────────────


def _kmeans_fit(data: np.ndarray, n_clusters: int, seed: int, max_iter: int) -> np.ndarray:
    """Fit K-means on *data* (n_samples, dim) and return cluster centres.

    Tries ``sklearn.cluster.MiniBatchKMeans`` first (10-100× faster than vanilla
    K-means for large matrices).  Falls back to a pure-NumPy Lloyd's-algorithm
    implementation when scikit-learn is unavailable.

    Args:
        data:       float32 array of shape (n_samples, dim).
        n_clusters: number of cluster centres to learn.
        seed:       random seed for reproducibility.
        max_iter:   maximum number of K-means iterations.

    Returns:
        float32 array of shape (n_clusters, dim) — the codebook vectors.
    """
    n_samples = data.shape[0]
    if n_samples <= n_clusters:
        # Edge case: pad centres with zero rows if fewer samples than clusters
        centres = data.astype(np.float32)
        pad = np.zeros((n_clusters - n_samples, data.shape[1]), dtype=np.float32)
        return np.vstack([centres, pad])

    try:
        from sklearn.cluster import MiniBatchKMeans  # optional fast path

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=max_iter,
            n_init="auto",
            batch_size=min(4096, n_samples),
        )
        km.fit(data)
        return km.cluster_centers_.astype(np.float32)

    except ImportError:
        pass  # fall through to pure-numpy implementation

    # ── Pure-NumPy Lloyd's algorithm ────────────────────────────────────────
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centres = data[idx].astype(np.float32)

    for _ in range(max_iter):
        # Assign each sample to its nearest centre (vectorised squared-distance)
        # dist[i, j] = ||data[i] - centres[j]||^2
        diff = data[:, None, :] - centres[None, :, :]          # (N, K, D)
        assignments = np.argmin((diff ** 2).sum(axis=-1), axis=1)  # (N,)

        # Recompute centres as mean of assigned samples
        new_centres = np.zeros_like(centres)
        counts = np.zeros(n_clusters, dtype=np.int32)
        np.add.at(new_centres, assignments, data)
        np.add.at(counts, assignments, 1)

        # Avoid divide-by-zero for empty clusters → keep previous centre
        mask = counts > 0
        new_centres[mask] /= counts[mask, None]
        new_centres[~mask] = centres[~mask]

        if np.allclose(centres, new_centres, atol=1e-6):
            break
        centres = new_centres

    return centres.astype(np.float32)


def _assign(groups: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Return index of nearest centre for each row in *groups*.

    Args:
        groups:  float32 (n_groups, group_size).
        centres: float32 (codebook_size, group_size).

    Returns:
        int32 (n_groups,) nearest-centre indices.
    """
    diff = groups[:, None, :] - centres[None, :, :]    # (G, C, D)
    return np.argmin((diff ** 2).sum(axis=-1), axis=1).astype(np.int32)


def encode_weight_matrix(
    weight: np.ndarray,
    cfg: AQLMConfig,
    *,
    seed: int = 42,
    max_iter: int = 100,
    residual_scale: float = 1.0,
) -> AQLMLayer:
    """Encode a single 2-D weight matrix as an AQLMLayer.

    The algorithm follows the AQLM iterative residual codebook training:

    1. Normalise the weight matrix by its RMS (stored as ``layer.scale``).
    2. For each codebook k = 0 … K-1:
        a. Slice the weight matrix into groups of ``group_size`` input features.
        b. Fit K-means with ``codebook_size`` clusters on the *residual* groups.
        c. Assign each group to its nearest centroid.
        d. Subtract the selected centroid from the residual (next codebook sees
           the error left over from all previous codebooks).
    3. Store indices and codebook vectors in an AQLMLayer.

    Runtime estimates (single thread, no GPU, sklearn MiniBatchKMeans):
        - 1 B  parameters model (~2 GB BF16):  ~2–5  min on M3
        - 1.5B parameters model (~3 GB BF16):  ~5–10 min on M3
        - 3B   parameters model (~6 GB BF16):  ~12–20 min on M3
        - 7B   parameters model (~14 GB BF16): ~30–60 min on M3
    Without sklearn (pure-NumPy Lloyd): roughly 5–20× slower.

    Memory: peak ~2× weight matrix size (residual copy + centres).

    Args:
        weight:          float32 (out_features, in_features) weight matrix.
        cfg:             AQLMConfig with n_codebooks, codebook_size, group_size.
        seed:            random seed for K-means initialisation.
        max_iter:        maximum K-means iterations per codebook.
        residual_scale:  multiply residual by this factor before each codebook
                         training step (default 1.0, i.e. no scaling).

    Returns:
        AQLMLayer with populated indices, codebooks, and scale.

    Raises:
        ValueError: if weight is not 2-D or in_features not divisible by group_size.
    """
    if weight.ndim != 2:
        raise ValueError(f"encode_weight_matrix: weight must be 2-D, got ndim={weight.ndim}")

    out_features, in_features = weight.shape
    if in_features % cfg.group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) not divisible by group_size ({cfg.group_size})"
        )

    n_groups = in_features // cfg.group_size
    layer = AQLMLayer(out_features, in_features, cfg)

    # Global RMS scale normalisation — keeps codebook vectors O(1)
    rms = float(np.sqrt(np.mean(weight.astype(np.float64) ** 2)))
    layer.scale = rms if rms > 1e-12 else 1.0

    # Residual over all segments: shape (out_features * n_groups, group_size)
    w32 = weight.astype(np.float32) / layer.scale
    # Reshape to (N_total_groups, group_size) for joint codebook training
    residual = w32.reshape(out_features * n_groups, cfg.group_size).copy()

    all_indices = np.zeros(
        (out_features * n_groups, cfg.n_codebooks), dtype=np.int32
    )

    for k in range(cfg.n_codebooks):
        r = residual * residual_scale
        logger.debug(
            "AQLMEncoder: codebook %d/%d — fitting %d clusters on %d vectors (dim=%d)",
            k + 1, cfg.n_codebooks, cfg.codebook_size, r.shape[0], cfg.group_size,
        )
        centres = _kmeans_fit(r, cfg.codebook_size, seed=seed + k, max_iter=max_iter)
        idx = _assign(r, centres)

        # Store
        all_indices[:, k] = idx
        layer.codebooks[k] = AQLMCodebook(vectors=centres)

        # Subtract selected centroids from residual
        residual -= centres[idx]

    # Reshape indices back to (out_features, n_groups, K)
    layer.indices = all_indices.reshape(out_features, n_groups, cfg.n_codebooks)

    return layer


class AQLMEncoder:
    """High-level encoder: compress all linear weight matrices in a model directory.

    Iterates over safetensors weight files, identifies linear projection tensors
    by shape (2-D, in_features divisible by group_size), encodes each with
    :func:`encode_weight_matrix`, and writes squish npy-dir files alongside.

    Graceful import failure: instantiating this class when the ``safetensors``
    package is unavailable raises ``ImportError`` with a clear install hint.

    Usage::

        enc = AQLMEncoder(cfg=AQLMConfig(n_codebooks=2, codebook_size=256, group_size=8))
        result = enc.compress_dir(model_dir, output_dir, progress=True)

    The returned ``result`` dict maps ``layer_name → AQLMLayer``.

    Expected runtimes (see :func:`encode_weight_matrix` docstring for detail):
        1.5B model: 5–10 min (sklearn), 30–100 min (pure-NumPy) on M3 single-thread.
    """

    def __init__(
        self,
        cfg: Optional[AQLMConfig] = None,
        *,
        seed: int = 42,
        max_iter: int = 100,
        min_out_features: int = 64,
    ) -> None:
        """
        Args:
            cfg:               AQLM config. Defaults to K=2, C=256, G=8 (≈2 bpw).
            seed:              random seed passed to K-means.
            max_iter:          max K-means iterations per codebook per layer.
            min_out_features:  layers with fewer output features are skipped
                               (embeddings / small projections are usually kept in FP16).
        """
        # Ensure safetensors is importable before doing any work
        try:
            import safetensors.numpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AQLMEncoder requires 'safetensors': pip install safetensors"
            ) from exc

        self.cfg = cfg or AQLMConfig(n_codebooks=2, codebook_size=256, group_size=8)
        self.seed = seed
        self.max_iter = max_iter
        self.min_out_features = min_out_features

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _should_encode(name: str, tensor: np.ndarray, min_out: int) -> bool:
        """Return True if *tensor* is a linear weight worth encoding.

        Criteria:
        - exactly 2-D
        - out_features (dim 0) ≥ min_out_features
        - Name contains 'proj', 'fc', 'dense', 'linear', 'weight', or 'mlp'
          (heuristic to skip embedding tables and layer-norms).
        """
        if tensor.ndim != 2:
            return False
        if tensor.shape[0] < min_out:
            return False
        name_lower = name.lower()
        keywords = ("proj", "fc", "dense", "linear", ".weight", "mlp", "gate")
        return any(kw in name_lower for kw in keywords)

    # ── public API ──────────────────────────────────────────────────────────

    def encode_layer(self, weight: np.ndarray) -> AQLMLayer:
        """Encode a single weight matrix. Convenience wrapper."""
        return encode_weight_matrix(
            weight.astype(np.float32),
            self.cfg,
            seed=self.seed,
            max_iter=self.max_iter,
        )

    def compress_dir(
        self,
        model_dir: "Path",  # noqa: F821 — Path imported at call site
        output_dir: "Path",
        *,
        progress: bool = True,
    ) -> dict:
        """Compress all eligible linear layers in *model_dir* to AQLM npy-dir.

        Reads every ``*.safetensors`` file in *model_dir*, encodes qualifying
        weight tensors, and writes ``<layer_stem>__aqlm_idx.npy`` and
        ``<layer_stem>__aqlm_cb.npy`` to *output_dir*.  Non-quantised tensors
        (LayerNorm, biases, embeddings) are copied as float32 .npy files.

        Storage layout in *output_dir*:
            squish.json              — format metadata
            <stem>__aqlm_idx.npy    — int32 index array (out, n_groups, K)
            <stem>__aqlm_cb.npy     — packed [scale, C, G, *vectors] float32
            <stem>.npy              — passthrough tensors (fp32)

        Args:
            model_dir:  source directory containing safetensors files.
            output_dir: destination directory (created if absent).
            progress:   print per-layer progress to stdout.

        Returns:
            dict mapping ``layer_name`` to ``AQLMLayer`` for all encoded layers.
        """
        import json
        from pathlib import Path

        import safetensors.numpy as stn

        model_dir = Path(model_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result: dict = {}
        n_encoded = 0
        n_passthrough = 0

        sf_files = sorted(model_dir.glob("*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(
                f"No *.safetensors files found in {model_dir}"
            )

        for sf_path in sf_files:
            tensors = stn.load_file(str(sf_path))
            for name, tensor in tensors.items():
                arr = np.asarray(tensor, dtype=np.float32)
                stem = name.replace(".", "_")

                if self._should_encode(name, arr, self.min_out_features):
                    if arr.shape[1] % self.cfg.group_size != 0:
                        # in_features not divisible — fall through to passthrough
                        if progress:
                            print(
                                f"  [skip]  {name}  {arr.shape}  "
                                f"(in_features not divisible by group_size={self.cfg.group_size})"
                            )
                        np.save(str(output_dir / f"{stem}.npy"), arr)
                        n_passthrough += 1
                        continue

                    if progress:
                        print(f"  [aqlm]  {name}  {arr.shape}")

                    layer = encode_weight_matrix(
                        arr, self.cfg,
                        seed=self.seed,
                        max_iter=self.max_iter,
                    )

                    # ── Save index array ────────────────────────────────────
                    np.save(
                        str(output_dir / f"{stem}__aqlm_idx.npy"),
                        layer.indices.astype(np.int32),
                    )

                    # ── Pack codebook: [scale, C, G, *vectors] ──────────────
                    cb_vectors = np.stack(
                        [cb.vectors for cb in layer.codebooks], axis=0
                    )  # (K, C, G) float32
                    header = np.array(
                        [layer.scale, float(self.cfg.codebook_size), float(self.cfg.group_size)],
                        dtype=np.float32,
                    )
                    np.save(
                        str(output_dir / f"{stem}__aqlm_cb.npy"),
                        np.concatenate([header, cb_vectors.ravel()]),
                    )
                    result[name] = layer
                    n_encoded += 1
                else:
                    np.save(str(output_dir / f"{stem}.npy"), arr)
                    n_passthrough += 1

        # ── Write format metadata ──────────────────────────────────────────
        meta = {
            "format": "aqlm",
            "n_codebooks": self.cfg.n_codebooks,
            "codebook_size": self.cfg.codebook_size,
            "group_size": self.cfg.group_size,
            "n_encoded": n_encoded,
            "n_passthrough": n_passthrough,
            "bpw_estimate": round(
                self.cfg.n_codebooks
                * (np.log2(self.cfg.codebook_size) / self.cfg.group_size),
                2,
            ),
        }
        (output_dir / "squish.json").write_text(json.dumps(meta, indent=2))

        if progress:
            print(
                f"\n  ✓  AQLM compression complete: "
                f"{n_encoded} encoded, {n_passthrough} passthrough\n"
                f"     BPW estimate: {meta['bpw_estimate']:.2f}\n"
                f"     Output: {output_dir}"
            )

        return result

