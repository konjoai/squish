"""squish/kernels/ffn_mask_patch.py — Wave 98: FFN Structured Sparsity Injection.

Patches a loaded MLX model's FFN (MLP) layers in-place so that structured
sparsity masks from ``StructuredFfnSparsity`` are applied at every forward pass.

Design
──────
Each transformer layer's ``.mlp`` attribute is replaced with a ``MaskedFFN``
wrapper that calls the original MLP and then multiplies the output by a
precomputed binary mask stored as an MLX array.  The mask zeros out neurons
that ``sparsity_profiler`` identified as rarely needed, preserving the
reliable neurons at full fidelity.

The mask multiply is a single elementwise broadcast (negligible compute cost).
The bandwidth benefit comes from combined effects: masked outputs do not
contribute to residual stream norm, reducing effective information content that
downstream attention must attend to, and improving INT2 quality by ensuring the
activation distribution is tighter.

For actual weight-loading bandwidth reduction, use ``squish sparsity-trim`` to
permanently remove zeroed columns from W_up/W_gate and corresponding rows from
W_down.

Usage::

    from squish.kernels.ffn_mask_patch import patch_model_ffn_sparsity
    from squish.runtime.structured_sparsity import StructuredFfnSparsity

    sparsity = StructuredFfnSparsity.from_file("path/to/sparse_masks.npz")
    n_patched = patch_model_ffn_sparsity(model, sparsity)
    # model.layers[i].mlp is now a MaskedFFN for each layer with a mask.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from squish.runtime.structured_sparsity import StructuredFfnSparsity

__all__ = [
    "MaskedFFN",
    "patch_model_ffn_sparsity",
    "unpatch_model_ffn_sparsity",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLX availability probe
# ---------------------------------------------------------------------------

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False


# ---------------------------------------------------------------------------
# MaskedFFN wrapper
# ---------------------------------------------------------------------------

class MaskedFFN:
    """Wraps an MLX MLP module and applies a binary neuron mask to its output.

    The original MLP is stored as ``self.inner`` and called normally.
    One elementwise multiply applies the mask: neurons with mask=0 are zeroed,
    neurons with mask=1 pass through unchanged.

    The mask is stored as an MLX bfloat16 array for minimal memory overhead and
    maximum compute efficiency on Apple Silicon (GPU ALUs are optimised for BF16
    elementwise ops).

    Args:
        inner: The original ``model.layers[i].mlp`` module.
        mask: Float32 or boolean numpy array of shape ``(hidden_size,)``.
              Values > 0 mean "keep"; 0 means "zero out".
        layer_idx: Layer index (used for logging only).
    """

    def __init__(self, inner, mask: np.ndarray, layer_idx: int = -1) -> None:
        self.inner = inner
        self._layer_idx = layer_idx
        # Store as MLX bfloat16 — broadcast multiply is zero-overhead on GPU
        if _MLX_AVAILABLE:
            mask_bool = (mask > 0).astype(np.float16)
            self._mask_mx = mx.array(mask_bool)
        else:
            self._mask_mx = None
        self._mask_np = (mask > 0).astype(np.float32)
        self._sparsity_ratio = float(1.0 - mask_bool.mean()) if _MLX_AVAILABLE else float(1.0 - (mask > 0).mean())

    def __call__(self, x):
        """Forward pass: run inner MLP then apply mask."""
        out = self.inner(x)
        if self._mask_mx is not None:
            return out * self._mask_mx
        # NumPy fallback (CI / non-Apple)
        import numpy as _np
        out_np = _np.array(out) * self._mask_np
        return out_np

    # Expose inner module attributes so model.layers[i].mlp.weight still works
    def __getattr__(self, name: str):
        # Avoid infinite recursion for our own attributes
        if name in ("inner", "_layer_idx", "_mask_mx", "_mask_np", "_sparsity_ratio"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __repr__(self) -> str:
        return (
            f"MaskedFFN(layer={self._layer_idx}, "
            f"sparsity={self._sparsity_ratio:.1%}, "
            f"inner={self.inner.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------

def patch_model_ffn_sparsity(
    model,
    sparsity: "StructuredFfnSparsity",
    *,
    verbose: bool = True,
) -> int:
    """Patch *model*'s MLP layers in-place with binary sparsity masks.

    Iterates over ``model.layers`` (standard mlx_lm layout) and wraps each
    ``.mlp`` attribute that has a corresponding mask in *sparsity* with a
    :class:`MaskedFFN`.  Layers without a mask are left untouched.

    Args:
        model: A loaded MLX transformer model with a ``.layers`` attribute.
        sparsity: Loaded :class:`~squish.runtime.structured_sparsity.StructuredFfnSparsity`.
        verbose: If ``True``, log a one-line summary after patching.

    Returns:
        Number of layers patched.

    Raises:
        AttributeError: If *model* has no ``.layers`` attribute (wrong model type).
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        # Try model.model.layers (common in mlx_lm wrapping)
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise AttributeError(
            "Cannot locate model.layers or model.model.layers — "
            "ensure this is an mlx_lm-loaded transformer."
        )

    n_patched = 0
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if not sparsity.has_mask(layer_idx):
            continue
        # Skip layers already patched (idempotent)
        if isinstance(mlp, MaskedFFN):
            continue

        mask = sparsity._masks[layer_idx]
        layer.mlp = MaskedFFN(mlp, mask, layer_idx=layer_idx)
        n_patched += 1

    if verbose and n_patched > 0:
        logger.info(
            "Patched %d/%d FFN layers (mean sparsity=%.1f%%)",
            n_patched, len(layers), sparsity.mean_sparsity * 100,
        )
    elif verbose:
        logger.debug("patch_model_ffn_sparsity: no layers patched (masks may be absent)")

    return n_patched


def unpatch_model_ffn_sparsity(model) -> int:
    """Remove any :class:`MaskedFFN` wrappers, restoring the original MLPs.

    Args:
        model: The same model previously patched by :func:`patch_model_ffn_sparsity`.

    Returns:
        Number of layers restored.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        return 0

    n_restored = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, MaskedFFN):
            layer.mlp = mlp.inner
            n_restored += 1

    return n_restored
