"""squish/quant/short_gpt.py

ShortGPTPruner — Layer-redundancy removal via Block Importance scoring.

Reference
---------
Men et al. "ShortGPT: Layers in Large Language Models are More Redundant
Than You Expect." arXiv:2403.03853, 2024.

Algorithm
---------
For each transformer block b with input hidden state h_in and output h_out:

  BI(b) = 1 − cosine_similarity(h_in, h_out)

A BI close to 0 means the block barely transforms its input (redundant).
The ``removal_fraction`` lowest-BI blocks are removed entirely from the
layer stack.  Blocks are kept contiguous: we preserve their original order
and remove the pruned ones.

This reduces FLOPs proportionally to the number of removed layers and
composes cleanly with SliceGPT (applied before or after column pruning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ShortGPTConfig:
    """Configuration for ShortGPTPruner.

    Parameters
    ----------
    removal_fraction:
        Fraction of transformer layers to remove (0.25 → remove 25%).
    n_calibration_tokens:
        Number of synthetic token vectors used to estimate BI scores.
    hidden_size:
        Hidden dimension of the model (used for synthetic calibration).
    seed:
        RNG seed.
    """

    removal_fraction: float = 0.25
    n_calibration_tokens: int = 256
    hidden_size: int = 4096
    seed: int = 7

    def __post_init__(self) -> None:
        if not 0.0 < self.removal_fraction < 1.0:
            raise ValueError("removal_fraction must be in (0, 1)")


# ---------------------------------------------------------------------------
# Block Importance result
# ---------------------------------------------------------------------------

@dataclass
class BlockImportance:
    """Per-layer block importance scores.

    Parameters
    ----------
    scores:
        Shape ``(n_layers,)`` — BI score per layer in original order.
    layer_indices:
        Original layer indices in descending importance order.
    """

    scores: np.ndarray
    layer_indices: np.ndarray

    def most_redundant(self, k: int) -> np.ndarray:
        """Return indices of the k least important layers."""
        return self.layer_indices[-k:]

    def most_important(self, k: int) -> np.ndarray:
        """Return indices of the k most important layers."""
        return self.layer_indices[:k]


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

class ShortGPTPruner:
    """Layer-removal pruner based on Block Importance scores.

    Parameters
    ----------
    config:
        ShortGPT configuration.
    """

    def __init__(self, config: Optional[ShortGPTConfig] = None) -> None:
        self._cfg = config or ShortGPTConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> ShortGPTConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Mean cosine similarity between matching rows of a and b."""
        a = a.reshape(-1, a.shape[-1]).astype(np.float64)
        b = b.reshape(-1, b.shape[-1]).astype(np.float64)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
        norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
        cos = (a / norm_a * (b / norm_b)).sum(axis=1)
        return float(cos.mean())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_block_importance(
        self,
        layer_inputs: list[np.ndarray],
        layer_outputs: list[np.ndarray],
    ) -> BlockImportance:
        """Compute BI scores from recorded per-layer activations.

        Parameters
        ----------
        layer_inputs:
            List of ``n_layers`` arrays, each shape ``(n_tokens, hidden)``.
        layer_outputs:
            Corresponding outputs; same shapes as inputs.

        Returns
        -------
        BlockImportance
        """
        n_layers = len(layer_inputs)
        scores = np.empty(n_layers, dtype=np.float32)
        for i, (h_in, h_out) in enumerate(zip(layer_inputs, layer_outputs)):
            cos_sim = self._cosine_sim(h_in, h_out)
            scores[i] = 1.0 - cos_sim  # BI

        # Sort by descending importance (highest BI first)
        order = np.argsort(scores)[::-1].astype(np.int32)
        return BlockImportance(scores=scores, layer_indices=order)

    def select_layers_to_remove(
        self, bi: BlockImportance, n_layers: int
    ) -> np.ndarray:
        """Select which layer indices to remove.

        Parameters
        ----------
        bi:
            BlockImportance from ``compute_block_importance``.
        n_layers:
            Total number of layers in the model.

        Returns
        -------
        np.ndarray
            Sorted array of layer indices to *remove*.
        """
        n_remove = max(1, round(n_layers * self._cfg.removal_fraction))
        to_remove = bi.most_redundant(n_remove)
        return np.sort(to_remove)

    def prune_layer_list(
        self,
        layers: list,
        bi: BlockImportance,
    ) -> tuple[list, np.ndarray]:
        """Remove redundant layers from a list.

        Parameters
        ----------
        layers:
            List of layer objects (arbitrary type).
        bi:
            BlockImportance from ``compute_block_importance``.

        Returns
        -------
        tuple[list, np.ndarray]
            (pruned_layers, removed_indices)
        """
        to_remove = set(self.select_layers_to_remove(bi, len(layers)).tolist())
        pruned = [l for i, l in enumerate(layers) if i not in to_remove]
        removed = np.array(sorted(to_remove), dtype=np.int32)
        return pruned, removed

    def calibrate_importance(
        self,
        layer_transforms: list,
        n_layers: Optional[int] = None,
    ) -> BlockImportance:
        """Synthetically estimate BI when real activations are unavailable.

        Each ``layer_transforms[i]`` must be a callable ``f(x) -> x_out``
        where x has shape ``(n_tokens, hidden_size)``.

        Parameters
        ----------
        layer_transforms:
            List of callables representing each layer.
        n_layers:
            Number of layers (inferred from list length if None).

        Returns
        -------
        BlockImportance
        """
        n_layers = n_layers or len(layer_transforms)
        h = self._rng.standard_normal(
            (self._cfg.n_calibration_tokens, self._cfg.hidden_size)
        ).astype(np.float32)

        inputs, outputs = [], []
        x = h.copy()
        for fn in layer_transforms:
            h_in = x.copy()
            x = np.asarray(fn(x), dtype=np.float32)
            inputs.append(h_in)
            outputs.append(x.copy())

        return self.compute_block_importance(inputs, outputs)
