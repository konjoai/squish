"""SparseGPT: one-shot second-order Hessian weight pruning (arXiv 2301.00774).

Frantar & Alistarh (ICLR 2023).  Prunes 50-60 % of weights in a single
forward pass by updating the remaining weights post-hoc to compensate for
each removed element.  Stacks with INT4/INT2 quantization to produce a
sparse-quantized model that occupies dense-INT2 DRAM at measurably higher
quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SparseGPTConfig",
    "SparseGPTResult",
    "SparseGPTPruner",
]


@dataclass
class SparseGPTConfig:
    """Configuration for :class:`SparseGPTPruner`.

    Attributes:
        sparsity_ratio: Fraction of weights to set to zero (default 0.5).
        block_size: Column block width for the OBC sweep (default 128).
        update_weights: Apply post-pruning weight update to compensate for
            removed weights (default True).
        structured: If True, enforce 2:4 structured sparsity (every group of
            4 weights has exactly 2 non-zeros) instead of unstructured.
        damp_pct: Hessian diagonal damping as a fraction of the mean diagonal
            (default 0.01 — follows the original paper).
        seed: RNG seed used when synthesising a proxy Hessian (default 0).
    """

    sparsity_ratio: float = 0.5
    block_size: int = 128
    update_weights: bool = True
    structured: bool = False
    damp_pct: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.sparsity_ratio < 1.0:
            raise ValueError(
                f"sparsity_ratio must be in (0, 1), got {self.sparsity_ratio}"
            )
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be >= 1, got {self.block_size}"
            )
        if not 0.0 < self.damp_pct < 1.0:
            raise ValueError(
                f"damp_pct must be in (0, 1), got {self.damp_pct}"
            )


@dataclass
class SparseGPTResult:
    """Outcome of pruning a single weight matrix.

    Attributes:
        sparsity_achieved: Fraction of weights actually zeroed.
        n_params_pruned: Count of pruned (zeroed) weight elements.
        n_params_total: Total element count of the input weight matrix.
        layer_name: Optional name tag for multi-layer reports.
    """

    sparsity_achieved: float
    n_params_pruned: int
    n_params_total: int
    layer_name: str = ""

    @property
    def compression_ratio(self) -> float:
        """Effective density (fraction still non-zero)."""
        return 1.0 - self.sparsity_achieved


class SparseGPTPruner:
    """One-shot SparseGPT pruner.

    Uses an approximate Optimal Brain Compression (OBC) column-sweep to
    identify and zero the least-salient weights while updating the
    survivors to compensate.  When real calibration data is unavailable a
    proxy Hessian is synthesised from the weight matrix itself.

    Usage::

        cfg = SparseGPTConfig(sparsity_ratio=0.5)
        pruner = SparseGPTPruner(cfg)
        W_pruned, result = pruner.prune_weight(W)

    """

    def __init__(self, config: SparseGPTConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune_weight(
        self,
        W: np.ndarray,
        H: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, SparseGPTResult]:
        """Prune a single weight matrix.

        Parameters
        ----------
        W:
            2-D weight matrix of shape ``(rows, cols)``.  Float32/float64.
        H:
            Optional *cols × cols* Hessian matrix representative of the
            activation covariance.  When omitted a proxy is synthesised.

        Returns
        -------
        (W_pruned, result)
        """
        W = np.array(W, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")
        rows, cols = W.shape
        if H is None:
            H = self._synthesise_hessian(W)

        H = self._damp_hessian(H)

        if self.config.structured:
            W_out = self._structured_prune(W, H)
        else:
            W_out = self._unstructured_prune(W, H)

        n_zero = int(np.sum(W_out == 0.0))
        n_total = W_out.size
        sparsity = n_zero / n_total
        result = SparseGPTResult(
            sparsity_achieved=float(sparsity),
            n_params_pruned=n_zero,
            n_params_total=n_total,
        )
        return W_out, result

    def prune_model(
        self,
        weights: Dict[str, np.ndarray],
        hessians: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[SparseGPTResult]]:
        """Prune every weight matrix in *weights*.

        Parameters
        ----------
        weights:
            Mapping ``layer_name → 2-D weight matrix``.
        hessians:
            Optional matching mapping of Hessian matrices.  Layers without
            a provided Hessian use the synthetic proxy.

        Returns
        -------
        (pruned_weights, results)
        """
        hessians = hessians or {}
        pruned: Dict[str, np.ndarray] = {}
        results: List[SparseGPTResult] = []
        for name, W in weights.items():
            H = hessians.get(name)
            W_p, res = self.prune_weight(W, H)
            res.layer_name = name
            pruned[name] = W_p
            results.append(res)
        return pruned, results

    def sparsity_report(
        self, weights: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Return the *current* (already-pruned) sparsity for each layer."""
        report: Dict[str, float] = {}
        for name, W in weights.items():
            total = W.size
            zeros = int(np.sum(W == 0.0))
            report[name] = zeros / total if total > 0 else 0.0
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesise_hessian(self, W: np.ndarray) -> np.ndarray:
        """Build a proxy *cols × cols* Hessian from the weight matrix.

        The empirical WᵀW scaled by the column norm serves as a reasonable
        stand-in when no calibration activations are available.
        """
        cols = W.shape[1]
        H = W.T @ W  # shape (cols, cols)
        scale = np.maximum(np.max(np.abs(np.diag(H))), 1e-6)
        H = H / scale
        return H.astype(np.float32)

    def _damp_hessian(self, H: np.ndarray) -> np.ndarray:
        """Add a relative diagonal damping term to prevent ill-conditioning."""
        diag_mean = float(np.mean(np.diag(H)))
        damp = self.config.damp_pct * abs(diag_mean)
        H_d = H.copy()
        np.fill_diagonal(H_d, np.diag(H_d) + damp)
        return H_d

    def _unstructured_prune(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Magnitude-weighted unstructured pruning with OBC weight update."""
        rows, cols = W.shape
        n_prune = int(round(self.config.sparsity_ratio * cols))
        if n_prune == 0:
            return W.copy()

        W_out = W.copy()
        block = self.config.block_size

        # Invert the Hessian once for the full column range.
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        # Column-sweep: process blocks of columns together.
        for start in range(0, cols, block):
            end = min(start + block, cols)
            W_block = W_out[:, start:end]  # (rows, blk)
            H_inv_block = H_inv[start:end, start:end]  # (blk, blk)

            # Compute per-column salience: ‖W[:, j]‖₂² / H_inv[j,j]
            h_diag = np.maximum(np.diag(H_inv_block), 1e-8)  # (blk,)
            col_scores = np.sum(W_block ** 2, axis=0) / h_diag  # (blk,)

            # Determine which columns (within the block) to zero.
            n_prune_blk = min(n_prune, end - start)
            prune_cols = np.argsort(col_scores)[:n_prune_blk]
            mask = np.ones(end - start, dtype=bool)
            mask[prune_cols] = False

            if self.config.update_weights:
                # Post-pruning weight update: W_remain -= W_pruned @ (H_inv_pruned / H_inv_remain)
                W_pruned_cols = W_block[:, ~mask]
                H_cross = H_inv_block[~mask][:, mask]
                H_remain_diag = np.maximum(h_diag[mask], 1e-8)
                update = W_pruned_cols @ (H_cross / H_remain_diag[np.newaxis, :])
                W_out[:, start:end][:, mask] -= update

            W_out[:, start:end][:, ~mask] = 0.0

        return W_out

    def _structured_prune(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """2:4 structured sparsity: exactly 2 zeros per group of 4 columns."""
        rows, cols = W.shape
        W_out = W.copy()
        # Pad to multiple of 4
        pad = (4 - cols % 4) % 4
        if pad:
            W_out = np.pad(W_out, ((0, 0), (0, pad)))

        padded_cols = W_out.shape[1]
        for g in range(0, padded_cols, 4):
            group = W_out[:, g: g + 4]  # (rows, 4)
            col_norms = np.sum(group ** 2, axis=0)  # (4,)
            prune_idx = np.argsort(col_norms)[:2]
            W_out[:, g: g + 4][:, prune_idx] = 0.0

        return W_out[:, :cols]
