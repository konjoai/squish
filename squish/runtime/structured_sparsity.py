"""squish/runtime/structured_sparsity.py

Wave 82 — Structured FFN Sparsity: pre-computed binary masks.

Loads ``sparse_masks.npz`` produced by the squish compress pipeline and
provides a thin wrapper that:

* reports per-layer sparsity ratios
* applies a binary mask to an FFN output tensor at inference time
* gracefully handles missing layers / partial mask files

Expected .npz key naming (all mask arrays are float32, values in {0.0, 1.0}):

    layer_{i}_gate   — gate-projection mask  [hidden_size]
    layer_{i}_up     — up-projection mask    [hidden_size]
    layer_{i}        — unified mask (fallback; covers gate + up jointly)

A value of ``1.0`` means "keep"; ``0.0`` means "zero out" (prune).
All non-zero values in the loaded mask are treated as "keep".

Zero-masking a neuron's contribution is equivalent to structured pruning of
that channel — the weight row/column stays in memory but its output is
zeroed, so no compute reduction occurs without a custom kernel.  The main
benefit in this implementation is **quality preservation**: the masks were
calibrated offline on representative data and are guaranteed not to remove
neurons that are reliably active, giving better INT2/INT3 quality than
unmasked quantisation.

Usage::

    from squish.runtime.structured_sparsity import StructuredFfnSparsity

    sparsity = StructuredFfnSparsity.from_file("path/to/sparse_masks.npz")
    print(sparsity)   # StructuredFfnSparsity(layers=32, mean_sparsity=0.42)

    # In the inference loop:
    masked = sparsity.apply_mask(layer_idx=5, tensor=ffn_output_numpy_array)

    # Check if a layer has a mask:
    if sparsity.has_mask(layer_idx=5):
        ...

    # Get per-layer sparsity ratio:
    ratio = sparsity.layer_sparsity(layer_idx=5)

Public API
----------
``StructuredFfnSparsity.from_file(path)``  — load from npz file
``StructuredFfnSparsity.n_layers``         — number of masked layers
``StructuredFfnSparsity.mean_sparsity``    — mean fraction of zeroed neurons
``StructuredFfnSparsity.has_mask(i)``      — True if layer i has a mask
``StructuredFfnSparsity.layer_sparsity(i)``— fraction of zeros in layer i mask
``StructuredFfnSparsity.apply_mask(i, t)`` — returns t * mask[i] (numpy/array)
``StructuredFfnSparsity.summary()``        — one-line human-readable summary
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["StructuredFfnSparsity"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_npz_masks(path: str) -> dict[int, np.ndarray]:
    """Parse sparse_masks.npz and return ``{layer_idx: mask_array}``."""
    raw = np.load(path)
    masks: dict[int, np.ndarray] = {}

    for key in raw.files:
        # Accept formats: "layer_5_gate", "layer_5_up", "layer_5", "5"
        k = key.strip()
        parts = k.split("_")
        layer_idx: int | None = None

        if k.isdigit():
            layer_idx = int(k)
        elif parts[0] == "layer" and len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])

        if layer_idx is None:
            continue

        arr = raw[key].astype(np.float32)
        if layer_idx not in masks:
            masks[layer_idx] = arr
        else:
            # Combine multiple masks for the same layer (gate + up → pointwise AND)
            combined_len = max(len(masks[layer_idx]), len(arr))
            m_a = np.ones(combined_len, dtype=np.float32)
            m_b = np.ones(combined_len, dtype=np.float32)
            m_a[:len(masks[layer_idx])] = masks[layer_idx]
            m_b[:len(arr)] = arr
            masks[layer_idx] = (m_a * m_b).astype(np.float32)

    return masks


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class StructuredFfnSparsity:
    """Binary FFN masks loaded from a sparse_masks.npz file.

    Parameters
    ----------
    masks : dict[int, np.ndarray]
        Mapping from layer index to float32 binary mask vector.
    source_path : str
        Path of the .npz file that was loaded (for logging / repr).
    """

    def __init__(
        self,
        masks: dict[int, np.ndarray],
        source_path: str = "",
    ) -> None:
        self._masks: dict[int, np.ndarray] = {
            k: (v != 0).astype(np.float32) for k, v in masks.items()
        }
        self._source_path = source_path
        self._sparsity_by_layer: dict[int, float] = {
            k: float(1.0 - v.mean()) for k, v in self._masks.items()
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "StructuredFfnSparsity":
        """Load from a .npz file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file contains no recognisable layer masks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"sparse_masks.npz not found: {path!r}")
        masks = _load_npz_masks(path)
        if not masks:
            raise ValueError(
                f"No layer masks found in {path!r}. "
                "Expected keys like 'layer_0', 'layer_0_gate', or '0'."
            )
        return cls(masks, source_path=path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of layers with a mask."""
        return len(self._masks)

    @property
    def mean_sparsity(self) -> float:
        """Mean fraction of zeroed neurons across all masked layers."""
        if not self._sparsity_by_layer:
            return 0.0
        return float(np.mean(list(self._sparsity_by_layer.values())))

    # ------------------------------------------------------------------
    # Per-layer API
    # ------------------------------------------------------------------

    def has_mask(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* has an associated mask."""
        return layer_idx in self._masks

    def layer_sparsity(self, layer_idx: int) -> float:
        """Return fraction of zeros in the mask for *layer_idx*.

        Returns 0.0 if the layer has no mask.
        """
        return self._sparsity_by_layer.get(layer_idx, 0.0)

    def get_mask(self, layer_idx: int) -> np.ndarray | None:
        """Return the float32 binary mask for *layer_idx*, or None."""
        return self._masks.get(layer_idx)

    def apply_mask(self, layer_idx: int, tensor: Any) -> Any:
        """Multiply *tensor* by the mask for *layer_idx*.

        If *tensor* is a ``numpy.ndarray``, the result is a numpy array.
        If *tensor* is an MLX array (``mlx.core.array``), the mask is
        converted to an MLX array before multiplication.

        If there is no mask for *layer_idx*, *tensor* is returned unchanged.

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer.
        tensor : numpy.ndarray | mlx.core.array
            Output of the FFN block, shape ``[..., hidden_size]``.

        Returns
        -------
        Same type and shape as *tensor*.
        """
        mask = self._masks.get(layer_idx)
        if mask is None:
            return tensor

        if isinstance(tensor, np.ndarray):
            # Broadcast mask over batch / sequence dimensions
            return tensor * mask

        # MLX path — lazy import to avoid mandatory dependency
        try:
            import mlx.core as mx  # noqa: PLC0415
            if hasattr(tensor, "shape"):
                mlx_mask = mx.array(mask)
                return tensor * mlx_mask
        except ImportError:
            pass

        # Unknown type — return unchanged to avoid silent corruption
        return tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable description."""
        return (
            f"StructuredFfnSparsity("
            f"layers={self.n_layers}, "
            f"mean_sparsity={self.mean_sparsity:.1%})"
        )

    def __repr__(self) -> str:
        return self.summary()
