"""LayerwiseDecode — Layer-by-layer decode with configurable early exit.

Instead of always running all N transformer layers, decoding can exit at an
earlier layer for "easy" tokens where the model is already highly confident.
A lightweight linear probe maps the hidden state at each layer to a compact
logit space; if the resulting softmax max-probability exceeds
``exit_threshold`` and the current layer index is at or beyond
``min_exit_layer``, the forward pass short-circuits, saving the remaining
``N − layer_idx`` matrix-multiply operations.

Reference:
    Schuster et al., "Confident Adaptive Language Modeling", NeurIPS 2022.
    https://arxiv.org/abs/2207.07061

Usage::

    import numpy as np
    from squish.layerwise_decode import LayerwiseDecoder, LayerwiseConfig, LayerStream

    cfg     = LayerwiseConfig(n_layers=32, hidden_dim=4096, exit_threshold=0.9)
    decoder = LayerwiseDecoder(cfg)

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal(4096).astype(np.float32)
    stream  = LayerStream(hidden=hidden, layer_idx=0, confidence=0.0)

    for layer_idx in range(cfg.n_layers):
        layer_w = rng.standard_normal((4096, 4096)).astype(np.float32) * 0.01
        stream = decoder.process_layer(stream, layer_w)
        if decoder.should_exit(stream.hidden, layer_idx):
            stream = LayerStream(
                hidden=stream.hidden,
                layer_idx=stream.layer_idx,
                confidence=stream.confidence,
                exited_early=True,
            )
            break

    decoder.record_token(exited_early=stream.exited_early)
    print(decoder.stats)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["LayerwiseConfig", "LayerStream", "LayerwiseDecoder", "DecodeStats"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LayerwiseConfig:
    """Configuration for layer-wise early-exit decoding.

    Attributes:
        n_layers: Total number of transformer layers.
        hidden_dim: Model hidden dimension.
        exit_threshold: Softmax confidence (max-probability) required to
            trigger early exit.  Must be in ``(0, 1]``.
        min_exit_layer: Earliest layer index at which early exit is permitted.
            Must satisfy ``0 <= min_exit_layer < n_layers``.
        probe_vocab: Output dimensionality of the lightweight exit probe
            (``hidden_dim → probe_vocab``).  Smaller values make the probe
            cheaper; larger values improve exit-confidence estimation.
    """

    n_layers: int = 32
    hidden_dim: int = 4096
    exit_threshold: float = 0.9
    min_exit_layer: int = 16
    probe_vocab: int = 256

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {self.n_layers}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {self.hidden_dim}")
        if not (0.0 < self.exit_threshold <= 1.0):
            raise ValueError(
                f"exit_threshold must be in (0, 1]; got {self.exit_threshold}"
            )
        if self.min_exit_layer < 0 or self.min_exit_layer >= self.n_layers:
            raise ValueError(
                f"min_exit_layer must be in [0, n_layers); "
                f"got {self.min_exit_layer} with n_layers={self.n_layers}"
            )
        if self.probe_vocab < 2:
            raise ValueError(f"probe_vocab must be >= 2; got {self.probe_vocab}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerStream:
    """State passed between layers during layerwise decode.

    Attributes:
        hidden: Hidden-state vector, shape ``(hidden_dim,)`` float32.
        layer_idx: Index of the layer that produced this hidden state
            (0-based; incremented by :meth:`LayerwiseDecoder.process_layer`).
        confidence: Softmax max-probability computed by the exit probe.
            Initialise to ``0.0`` before the first forward pass.
        exited_early: ``True`` when decoding terminated before the final layer.
    """

    hidden: np.ndarray
    layer_idx: int
    confidence: float
    exited_early: bool = False


@dataclass
class DecodeStats:
    """Accumulated statistics for :class:`LayerwiseDecoder`.

    Attributes:
        total_tokens: Number of tokens that have been fully decoded and
            recorded via :meth:`LayerwiseDecoder.record_token`.
        early_exits: Number of those tokens that triggered early exit.
        total_layers_run: Cumulative count of layer forward-passes across
            all calls to :meth:`LayerwiseDecoder.process_layer`.
    """

    total_tokens: int = 0
    early_exits: int = 0
    total_layers_run: int = 0

    @property
    def avg_exit_layer(self) -> float:
        """Average layer index at which tokens exited (layers run / tokens)."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_layers_run / self.total_tokens

    @property
    def early_exit_rate(self) -> float:
        """Fraction of decoded tokens that triggered early exit."""
        if self.total_tokens == 0:
            return 0.0
        return self.early_exits / self.total_tokens


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class LayerwiseDecoder:
    """Layer-by-layer decode controller with confidence-based early exit.

    A lightweight linear probe of shape ``(hidden_dim, probe_vocab)`` is
    initialised with Xavier-uniform weights.  At each layer the caller may
    query :meth:`should_exit`; if the probe confidence exceeds
    ``config.exit_threshold`` *and* the current layer index is at or beyond
    ``config.min_exit_layer``, the method returns ``True`` and the caller
    should stop the forward pass.

    The probe and layer-residual weights are seeded once at construction and
    are intentionally kept fixed (they represent a pre-trained exit head in a
    real deployment).

    Args:
        config: :class:`LayerwiseConfig` instance.
        rng: Optional NumPy random generator for reproducible probe
            initialisation.  Defaults to ``np.random.default_rng(0)``.
    """

    def __init__(
        self,
        config: LayerwiseConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        if rng is None:
            rng = np.random.default_rng(0)

        # Xavier-uniform initialisation for the exit probe
        limit = math.sqrt(6.0 / (config.hidden_dim + config.probe_vocab))
        self._probe: np.ndarray = rng.uniform(
            -limit,
            limit,
            size=(config.hidden_dim, config.probe_vocab),
        ).astype(np.float32)

        self._stats = DecodeStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_exit(self, hidden: np.ndarray, layer_idx: int) -> bool:
        """Determine whether decoding should exit at the current layer.

        Projects ``hidden`` through the exit probe, applies softmax, and
        compares the maximum probability against ``config.exit_threshold``.
        Early exit is suppressed for any layer index below
        ``config.min_exit_layer``.

        Args:
            hidden: Current hidden state, shape ``(hidden_dim,)`` float32.
            layer_idx: Zero-based index of the current layer.

        Returns:
            ``True`` if the confidence threshold is met and the layer index
            satisfies the minimum-exit-layer constraint.

        Raises:
            ValueError: If ``hidden`` has an unexpected shape or ``layer_idx``
                is negative.
        """
        if hidden.shape != (self.config.hidden_dim,):
            raise ValueError(
                f"Expected hidden shape ({self.config.hidden_dim},); "
                f"got {hidden.shape}"
            )
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative; got {layer_idx}")
        if layer_idx < self.config.min_exit_layer:
            return False

        logits = hidden.astype(np.float32) @ self._probe  # (probe_vocab,)
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        confidence = float(exp_l.max() / exp_l.sum())
        return confidence >= self.config.exit_threshold

    def process_layer(
        self,
        stream: LayerStream,
        layer_weights: np.ndarray,
    ) -> LayerStream:
        """Apply a single transformer layer to the stream state.

        The layer is modelled as a linear transform with a residual
        connection::

            new_hidden = hidden + hidden @ layer_weights.T

        After the update the probe is evaluated and the resulting confidence
        score is stored in the returned :class:`LayerStream`.

        Args:
            stream: Input :class:`LayerStream` for this layer.
            layer_weights: Weight matrix, shape
                ``(hidden_dim, hidden_dim)`` float32.

        Returns:
            Updated :class:`LayerStream` with incremented ``layer_idx``,
            new ``hidden`` state, and updated ``confidence``.

        Raises:
            ValueError: If ``layer_weights`` has an unexpected shape.
        """
        cfg = self.config
        if layer_weights.shape != (cfg.hidden_dim, cfg.hidden_dim):
            raise ValueError(
                f"layer_weights must have shape ({cfg.hidden_dim}, {cfg.hidden_dim}); "
                f"got {layer_weights.shape}"
            )

        h = stream.hidden.astype(np.float32)
        # Linear transform + residual
        new_h = (h + h @ layer_weights.T).astype(np.float32)

        # Evaluate probe for updated confidence
        logits = new_h @ self._probe  # (probe_vocab,)
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        confidence = float(exp_l.max() / exp_l.sum())

        self._stats.total_layers_run += 1

        return LayerStream(
            hidden=new_h,
            layer_idx=stream.layer_idx + 1,
            confidence=confidence,
            exited_early=stream.exited_early,
        )

    def record_token(self, exited_early: bool) -> None:
        """Record the completion of a single decoded token.

        Should be called once per token after the per-layer loop concludes
        (either at early exit or after all layers).

        Args:
            exited_early: ``True`` if this token triggered early exit.
        """
        self._stats.total_tokens += 1
        if exited_early:
            self._stats.early_exits += 1

    def reset_stats(self) -> None:
        """Reset all accumulated decode statistics to zero."""
        self._stats = DecodeStats()

    @property
    def stats(self) -> DecodeStats:
        """Current accumulated :class:`DecodeStats`."""
        return self._stats
