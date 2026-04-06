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

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ── Public exports ─────────────────────────────────────────────────────────────

__all__ = ["AQLMConfig", "AQLMCodebook", "AQLMLayer", "aqlm_dequantize"]


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
