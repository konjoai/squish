"""HybridArchRouter: per-layer dispatch for Jamba/Zamba/Hymba hybrid models.

Hybrid LLMs like Jamba-1.5B interleave transformer attention blocks with
Mamba / RWKV / Hawk SSM blocks at a configurable ratio.  The model's
``config.json`` encodes the per-layer type via a ``layer_types`` list.

HybridArchRouter reads these configs at instantiation and exposes a
single ``route(layer_idx, x, state)`` call that dispatches to the correct
compute path without conditional Python overhead at the call site.

Supported layer type strings (case-insensitive):
  ``"attention"`` / ``"attn"`` / ``"transformer"`` → transformer path
  ``"mamba"`` / ``"mamba2"`` / ``"ssm"``           → Mamba2SSD path
  ``"rwkv"`` / ``"rwkv6"``                         → RWKV6 path
  ``"hawk"`` / ``"rglr"`` / ``"griffin"``          → Hawk path

Reference: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model"
arXiv 2403.19887 (2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "HybridArchConfig",
    "HybridLayerSpec",
    "HybridArchRouter",
]

# Canonical type groups
_ATTN_TYPES = {"attention", "attn", "transformer"}
_MAMBA_TYPES = {"mamba", "mamba2", "ssm"}
_RWKV_TYPES = {"rwkv", "rwkv6", "eagle"}
_HAWK_TYPES = {"hawk", "rglr", "griffin"}


def _canonical(t: str) -> str:
    t = t.lower().strip()
    if t in _ATTN_TYPES:
        return "attention"
    if t in _MAMBA_TYPES:
        return "mamba"
    if t in _RWKV_TYPES:
        return "rwkv"
    if t in _HAWK_TYPES:
        return "hawk"
    raise ValueError(
        f"Unknown layer type '{t}'. "
        f"Supported: {sorted(_ATTN_TYPES | _MAMBA_TYPES | _RWKV_TYPES | _HAWK_TYPES)}"
    )


@dataclass
class HybridLayerSpec:
    """Specification for a single model layer.

    Attributes:
        layer_idx: Zero-based layer index.
        canonical_type: Normalised type string (one of ``"attention"``,
            ``"mamba"``, ``"rwkv"``, ``"hawk"``).
        raw_type: Original type string from config.json.
    """

    layer_idx: int
    canonical_type: str
    raw_type: str


@dataclass
class HybridArchConfig:
    """Configuration parsed from a hybrid model's config.json.

    Attributes:
        layer_types: Ordered list of raw layer type strings.  Length must equal
            the total number of model layers.
        model_name: Optional model identifier.
        seed: Unused; for API consistency.
    """

    layer_types: Optional[List[str]] = None
    model_name: str = ""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.layer_types is not None:
            if not self.layer_types:
                raise ValueError("layer_types must be non-empty")
            # Validate all types are recognised
            for t in self.layer_types:
                _canonical(t)


class HybridArchRouter:
    """Route inference calls to the correct kernel for each layer.

    Usage::

        config = HybridArchConfig(
            layer_types=["attention", "mamba", "mamba", "attention", "mamba"],
            model_name="jamba-1.5b",
        )
        router = HybridArchRouter(config)

        # Register compute callables (one per canonical type)
        router.register("attention", my_attn_fn)
        router.register("mamba", my_mamba_fn)

        # At inference time
        for layer_idx in range(n_layers):
            out = router.route(layer_idx, x=hidden_state, state=layer_states[layer_idx])
    """

    def __init__(self, config: HybridArchConfig) -> None:
        self.config = config
        layer_types = config.layer_types or []
        self._specs: List[HybridLayerSpec] = [
            HybridLayerSpec(
                layer_idx=i,
                canonical_type=_canonical(t),
                raw_type=t,
            )
            for i, t in enumerate(layer_types)
        ]
        self._handlers: Dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def register(self, canonical_type: str, fn: Callable[..., Any]) -> None:
        """Register a compute callable for a layer type.

        Args:
            canonical_type: One of ``"attention"``, ``"mamba"``, ``"rwkv"``,
                ``"hawk"``.
            fn: Callable ``(x, state, **kwargs) -> (output, new_state)``.
        """
        valid = {"attention", "mamba", "rwkv", "hawk"}
        if canonical_type not in valid:
            raise ValueError(f"canonical_type must be one of {valid}")
        self._handlers[canonical_type] = fn

    def route(
        self,
        layer_idx: int,
        x: np.ndarray,
        state: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a forward call for the given layer.

        Args:
            layer_idx: Layer index (0-based).
            x: Input tensor to the layer.
            state: Per-layer recurrent state (None for attention layers).
            **kwargs: Additional arguments forwarded to the handler.

        Returns:
            Whatever the registered handler returns.

        Raises:
            IndexError: If layer_idx is out of range.
            KeyError: If no handler is registered for the layer type.
        """
        if layer_idx < 0 or layer_idx >= len(self._specs):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {len(self._specs)})"
            )
        spec = self._specs[layer_idx]
        ct = spec.canonical_type
        if ct not in self._handlers:
            raise KeyError(
                f"No handler registered for type '{ct}' at layer {layer_idx}. "
                f"Register one with router.register('{ct}', fn)."
            )
        return self._handlers[ct](x, state, **kwargs)

    def layer_type(self, layer_idx: int) -> str:
        """Return canonical type for layer ``layer_idx``."""
        return self._specs[layer_idx].canonical_type

    def count_by_type(self) -> Dict[str, int]:
        """Return counts of each layer type."""
        counts: Dict[str, int] = {}
        for spec in self._specs:
            counts[spec.canonical_type] = counts.get(spec.canonical_type, 0) + 1
        return counts

    def attention_ratio(self) -> float:
        """Fraction of layers that are attention (not SSM)."""
        n_attn = sum(1 for s in self._specs if s.canonical_type == "attention")
        return n_attn / len(self._specs)

    @classmethod
    def from_layer_types(cls, layer_types: List[str], model_name: str = "") -> "HybridArchRouter":
        """Convenience constructor from a plain list of type strings."""
        return cls(HybridArchConfig(layer_types=layer_types, model_name=model_name))
