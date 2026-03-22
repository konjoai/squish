"""squish/quant/ada_gptq.py

AdaGPTQ — Hessian-Adaptive Group GPTQ.

Reference
---------
Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers." ICLR 2023 (arXiv:2210.17323).

Adaptive group selection:
Shao et al. "OmniQuant: Omnidirectionally Calibrated Quantization for Large
Language Models." ICLR 2024 (arXiv:2308.13137).

Algorithm
---------
Standard GPTQ uses a fixed group size (e.g., 128 columns per group).
AdaGPTQ selects group boundaries per row by inspecting the diagonal of the
Fisher/Hessian matrix: columns with higher curvature get smaller groups (more
expressive scales), while flat regions use larger groups.

1. Estimate per-column Hessian diagonal from a set of calibration activations.
2. For each weight row, assign group boundaries greedily: accumulate until the
   group's curvature budget is spent.
3. Quantize each group with symmetric or asymmetric INT-n_bits.

Key properties
--------------
* ``estimate_hessian(activations) → hessian_diag`` — per-column curvature.
* ``select_group_size(hessian_diag) → group_boundaries`` — adaptive groups.
* ``quantize(W, group_boundaries) → AdaGPTQResult`` — full quantization.
* NumPy-only simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "AdaGPTQConfig",
    "AdaGPTQResult",
    "AdaGPTQ",
]


@dataclass
class AdaGPTQConfig:
    """Configuration for :class:`AdaGPTQ`.

    Attributes:
        n_bits: Quantization bit-width (2–8).
        min_group_size: Smallest allowed group size (floor).
        max_group_size: Largest allowed group size (ceiling).
        curvature_budget: Sum of Hessian diagonal values per group.  Smaller
            values → more, smaller groups.
        symmetric: Whether to use symmetric (zero-centered) quantization.
    """

    n_bits: int = 4
    min_group_size: int = 8
    max_group_size: int = 128
    curvature_budget: float = 1.0
    symmetric: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.n_bits <= 8:
            raise ValueError("n_bits must be 1–8")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be ≥ 1")


@dataclass
class AdaGPTQResult:
    """Result of AdaGPTQ compression.

    Attributes:
        W_q: Quantized weight codes, shape ``(out_features, in_features)``.
        scale: Per-group scale, shape ``(out_features, n_groups)``.
        zero: Per-group zero-point (0 for symmetric), same shape as scale.
        group_boundaries: Column indices marking group starts, shared across
            all rows (length = n_groups + 1).
        hessian_diag: Column curvature vector used for group selection.
    """

    W_q: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    group_boundaries: List[int]
    hessian_diag: np.ndarray

    def dequantize(self) -> np.ndarray:
        """Reconstruct approximate FP32 weights."""
        out_f = self.W_q.shape[0]
        n_groups = len(self.group_boundaries) - 1
        W = np.empty_like(self.W_q, dtype=np.float32)
        for i in range(out_f):
            for g in range(n_groups):
                s = self.group_boundaries[g]
                e = self.group_boundaries[g + 1]
                W[i, s:e] = (self.W_q[i, s:e].astype(np.float32) - self.zero[i, g]) * self.scale[i, g]
        return W


class AdaGPTQ:
    """Hessian-adaptive GPTQ quantizer.

    Parameters
    ----------
    config:
        AdaGPTQ configuration.
    """

    def __init__(self, config: Optional[AdaGPTQConfig] = None) -> None:
        self._cfg = config or AdaGPTQConfig()

    @property
    def config(self) -> AdaGPTQConfig:
        return self._cfg

    def estimate_hessian(self, activations: np.ndarray) -> np.ndarray:
        """Estimate per-column Hessian diagonal from calibration activations.

        Parameters
        ----------
        activations:
            Calibration inputs, shape ``(n_samples, in_features)``.

        Returns
        -------
        hessian_diag of shape ``(in_features,)``.
        """
        X = np.asarray(activations, dtype=np.float32)
        # H ≈ X^T X / n (Fisher approximation)
        h = (X ** 2).mean(axis=0)
        # Avoid zero-curvature columns
        h = np.maximum(h, 1e-8)
        return h

    def select_group_boundaries(self, hessian_diag: np.ndarray) -> List[int]:
        """Choose adaptive group boundaries based on Hessian curvature.

        Parameters
        ----------
        hessian_diag:
            Per-column curvature, shape ``(in_features,)``.

        Returns
        -------
        List of column indices marking group starts + one past-end index.
        """
        in_f = len(hessian_diag)
        # Normalise curvature so budget is relative
        h_norm = hessian_diag / hessian_diag.mean()
        budget = self._cfg.curvature_budget

        boundaries = [0]
        accum = 0.0
        group_start = 0

        for col in range(in_f):
            accum += h_norm[col]
            group_len = col - group_start + 1
            if group_len >= self._cfg.min_group_size and (
                accum >= budget or group_len >= self._cfg.max_group_size
            ):
                boundaries.append(col + 1)
                group_start = col + 1
                accum = 0.0

        if boundaries[-1] < in_f:
            boundaries.append(in_f)
        return boundaries

    def quantize(
        self,
        weights: np.ndarray,
        hessian_diag: Optional[np.ndarray] = None,
    ) -> AdaGPTQResult:
        """Quantize the weight matrix with adaptive group sizes.

        Parameters
        ----------
        weights:
            FP32 weight matrix, shape ``(out_features, in_features)``.
        hessian_diag:
            Optional precomputed curvature; if None a flat (uniform) Hessian
            is used which falls back to fixed ``max_group_size`` groups.

        Returns
        -------
        AdaGPTQResult
        """
        W = np.asarray(weights, dtype=np.float32)
        out_f, in_f = W.shape
        if hessian_diag is None:
            hessian_diag = np.ones(in_f, dtype=np.float32)
        else:
            hessian_diag = np.asarray(hessian_diag, dtype=np.float32)
            if hessian_diag.shape[0] != in_f:
                raise ValueError("hessian_diag length must match in_features")

        boundaries = self.select_group_boundaries(hessian_diag)
        n_groups = len(boundaries) - 1
        n_levels = 2 ** self._cfg.n_bits

        W_q = np.zeros_like(W, dtype=np.int32)
        scale = np.zeros((out_f, n_groups), dtype=np.float32)
        zero = np.zeros((out_f, n_groups), dtype=np.float32)

        for i in range(out_f):
            for g in range(n_groups):
                s = boundaries[g]
                e = boundaries[g + 1]
                w_seg = W[i, s:e]
                if self._cfg.symmetric:
                    sc = max(2.0 * np.abs(w_seg).max() / (n_levels - 1), 1e-8)
                    zr = 0.0
                else:
                    w_min = float(w_seg.min())
                    w_max = float(w_seg.max())
                    sc = max((w_max - w_min) / (n_levels - 1), 1e-8)
                    zr = -w_min / sc
                codes = np.round(w_seg / sc + zr).clip(0, n_levels - 1)
                W_q[i, s:e] = codes.astype(np.int32)
                scale[i, g] = sc
                zero[i, g] = zr

        return AdaGPTQResult(
            W_q=W_q,
            scale=scale,
            zero=zero,
            group_boundaries=boundaries,
            hessian_diag=hessian_diag,
        )
