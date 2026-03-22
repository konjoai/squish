"""squish/token/calm_exit.py

AdaptiveCALM — Confidence-Adaptive Computation via Early Exits.

Reference
---------
Schuster et al. "Confident Adaptive Language Modeling."
NeurIPS 2022.

Algorithm
---------
CALM allows each token to exit (return its representation) at any
transformer layer once the model is *confident* enough.  Confidence is
measured as the maximum softmax probability of the intermediate hidden
state projected to the vocabulary.

A simple approximation used here: the per-layer hidden state is passed
through a softmax (treating the ``d_model``-dimensional vector as unnormalised
logits over a proxy vocabulary) and confidence is the maximum component.
When confidence exceeds ``confidence_threshold`` and at least
``min_layers`` layers have been processed, computation halts.

The module tracks an ``exit_histogram`` counting how often each layer is
chosen across all ``forward()`` calls.

This module provides:

1. ``AdaptiveCALM.forward(x, layer_fns)`` → ``CALMResult``.
2. ``AdaptiveCALM.confidence_at_layer(hidden)`` → float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

__all__ = [
    "CALMConfig",
    "CALMResult",
    "AdaptiveCALM",
]


@dataclass
class CALMConfig:
    """Configuration for :class:`AdaptiveCALM`.

    Attributes:
        n_layers: Total number of transformer layers available.
        d_model: Hidden state dimension.
        confidence_threshold: Exit once max-softmax ≥ this value.
        min_layers: Minimum number of layers that must run before exit.
        seed: Unused; kept for API consistency.
    """

    n_layers: int = 32
    d_model: int = 4096
    confidence_threshold: float = 0.9
    min_layers: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {self.n_layers}")
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1; got {self.d_model}")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in (0, 1]; got {self.confidence_threshold}"
            )
        if self.min_layers < 1:
            raise ValueError(f"min_layers must be >= 1; got {self.min_layers}")
        if self.min_layers >= self.n_layers:
            raise ValueError(
                f"min_layers ({self.min_layers}) must be < n_layers ({self.n_layers})"
            )


@dataclass
class CALMResult:
    """Result of one :meth:`AdaptiveCALM.forward` call.

    Attributes:
        output: Final hidden state ``(d_model,)`` at the exit layer.
        exit_layer: Zero-based index of the layer where computation stopped.
        confidence: Confidence score at the exit layer.
        flop_ratio: Fraction of layers executed (``exit_layer + 1) / n_layers``.
    """

    output: np.ndarray
    exit_layer: int
    confidence: float
    flop_ratio: float


class AdaptiveCALM:
    """Confidence-adaptive early exit for transformer inference.

    ``layer_fns`` passed to ``forward()`` are callables with signature
    ``(hidden: np.ndarray) -> np.ndarray`` each transforming a single
    ``(d_model,)`` hidden state.

    Example::

        import numpy as np
        cfg = CALMConfig(n_layers=8, d_model=16, confidence_threshold=0.8, min_layers=2)
        calm = AdaptiveCALM(cfg)

        rng = np.random.default_rng(0)
        # Toy layers: each scales + shifts
        layer_fns = [
            lambda h, i=i: h * 0.9 + rng.standard_normal(h.shape) * 0.01
            for i in range(8)
        ]
        x = rng.standard_normal(16).astype(np.float32)
        result = calm.forward(x, layer_fns)
        print(result.exit_layer, result.confidence)
    """

    def __init__(self, config: Optional[CALMConfig] = None) -> None:
        self._cfg = config or CALMConfig()
        self._exit_histogram: np.ndarray = np.zeros(
            self._cfg.n_layers, dtype=np.int64
        )

    @property
    def config(self) -> CALMConfig:
        return self._cfg

    @property
    def exit_histogram(self) -> np.ndarray:
        """``(n_layers,)`` count of how often each layer was the exit."""
        return self._exit_histogram

    def confidence_at_layer(self, hidden: np.ndarray) -> float:
        """Compute confidence (max softmax) for a hidden state.

        The hidden vector is treated as unnormalised logits.  The maximum
        softmax probability is a proxy for how "peaked" the distribution is.

        Args:
            hidden: ``(d_model,)`` float array.

        Returns:
            Maximum softmax probability in ``[1/d_model, 1.0]``.
        """
        h = np.asarray(hidden, dtype=np.float64)
        h = h - h.max()
        exp_h = np.exp(h)
        softmax = exp_h / (exp_h.sum() + 1e-12)
        return float(softmax.max())

    def forward(
        self,
        x: np.ndarray,
        layer_fns: List[Callable[[np.ndarray], np.ndarray]],
    ) -> CALMResult:
        """Run early-exit forward pass.

        Args:
            x: ``(d_model,)`` token hidden state.
            layer_fns: List of per-layer callable transformations.  Length
                must equal ``config.n_layers``.

        Returns:
            :class:`CALMResult` with output, exit layer, confidence, and
            FLOPs ratio.

        Raises:
            ValueError: If ``len(layer_fns) != config.n_layers``.
        """
        if len(layer_fns) != self._cfg.n_layers:
            raise ValueError(
                f"Expected {self._cfg.n_layers} layer functions, "
                f"got {len(layer_fns)}"
            )
        h = np.asarray(x, dtype=np.float32)
        conf = 0.0
        exit_layer = self._cfg.n_layers - 1
        for layer_idx, fn in enumerate(layer_fns):
            h = np.asarray(fn(h), dtype=np.float32)
            conf = self.confidence_at_layer(h)
            if (
                layer_idx + 1 >= self._cfg.min_layers
                and conf >= self._cfg.confidence_threshold
            ):
                exit_layer = layer_idx
                break
        self._exit_histogram[exit_layer] += 1
        flop_ratio = (exit_layer + 1) / self._cfg.n_layers
        return CALMResult(
            output=h,
            exit_layer=exit_layer,
            confidence=conf,
            flop_ratio=flop_ratio,
        )
