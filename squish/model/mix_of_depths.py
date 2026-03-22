"""Mixture of Depths: dynamic per-token layer routing (arXiv 2404.02258).

Raposo et al., TMLR 2024.  A lightweight router decides at each transformer
layer which tokens should be processed by that layer and which can skip it via
a residual bypass.  At a 50 % skip budget this halves effective FLOPs while
maintaining near-identical perplexity.  Composable with any quantization and
KV optimization strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "MixtureOfDepthsConfig",
    "MoDLayerResult",
    "MixtureOfDepths",
]


@dataclass
class MixtureOfDepthsConfig:
    """Configuration for :class:`MixtureOfDepths`.

    Attributes:
        n_layers: Number of transformer layers in the model.
        skip_ratio: Fraction of tokens that skip each layer (default 0.5).
        router_dim: Hidden dimension used by the learned router projection.
        router_type: ``"linear"`` (dot-product score) or ``"threshold"``
            (fixed threshold on the last hidden-state norm).
        min_active_tokens: Minimum tokens guaranteed to be processed per layer
            even when their scores fall below the budget (default 1).
        seed: RNG seed for weight initialisation (default 0).
    """

    n_layers: int = 32
    skip_ratio: float = 0.5
    router_dim: int = 64
    router_type: str = "linear"
    min_active_tokens: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if not 0.0 <= self.skip_ratio < 1.0:
            raise ValueError(
                f"skip_ratio must be in [0, 1), got {self.skip_ratio}"
            )
        if self.router_dim < 1:
            raise ValueError(f"router_dim must be >= 1, got {self.router_dim}")
        if self.router_type not in ("linear", "threshold"):
            raise ValueError(
                f"router_type must be 'linear' or 'threshold', "
                f"got {self.router_type!r}"
            )
        if self.min_active_tokens < 1:
            raise ValueError(
                f"min_active_tokens must be >= 1, got {self.min_active_tokens}"
            )


@dataclass
class MoDLayerResult:
    """Result of routing one sequence through one layer.

    Attributes:
        layer_idx: Zero-based layer index.
        n_tokens: Total tokens in the sequence.
        n_active: Tokens that were processed by the layer.
        n_skipped: Tokens that bypassed the layer.
        skip_mask: Boolean array of shape ``(n_tokens,)`` — True = skipped.
    """

    layer_idx: int
    n_tokens: int
    n_active: int
    n_skipped: int
    skip_mask: np.ndarray

    @property
    def active_ratio(self) -> float:
        return self.n_active / self.n_tokens if self.n_tokens > 0 else 0.0


class MixtureOfDepths:
    """Token-level layer-routing for Mixture-of-Depths inference.

    Each call to :meth:`route` produces a skip mask for the current layer.
    :meth:`apply_layer` then merges the layer output back into the full
    sequence by substituting residual values for skipped positions.

    Router weights are randomly initialised (proxy for a trained router) so
    that the class is fully functional without a loaded model.

    Usage::

        cfg = MixtureOfDepthsConfig(n_layers=32, skip_ratio=0.5)
        mod = MixtureOfDepths(cfg)
        for layer_idx in range(cfg.n_layers):
            result = mod.route(hidden_states, layer_idx)
            # ... run transformer layer on active tokens ...
            layer_out = layer(hidden_states[~result.skip_mask])
            hidden_states = mod.apply_layer(hidden_states, layer_out, result)

    """

    def __init__(self, config: MixtureOfDepthsConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        # Per-layer router projection: (router_dim,) weights
        self._router_weights: Dict[int, np.ndarray] = {
            i: self._rng.standard_normal(config.router_dim).astype(np.float32)
            for i in range(config.n_layers)
        }
        self._stats: Dict[int, List[float]] = {
            i: [] for i in range(config.n_layers)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        hidden_states: np.ndarray,
        layer_idx: int,
    ) -> MoDLayerResult:
        """Compute the skip mask for *hidden_states* at *layer_idx*.

        Parameters
        ----------
        hidden_states:
            Array of shape ``(n_tokens, hidden_dim)`` or ``(n_tokens,)`` for
            1-D proxy inputs.
        layer_idx:
            Zero-based layer index (0 ≤ layer_idx < n_layers).

        Returns
        -------
        :class:`MoDLayerResult` with a boolean skip mask.
        """
        if not 0 <= layer_idx < self.config.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.config.n_layers})"
            )
        hs = np.atleast_2d(hidden_states).astype(np.float32)
        n_tokens = hs.shape[0]

        scores = self._score_tokens(hs, layer_idx)  # (n_tokens,)
        n_skip = max(
            0,
            min(
                n_tokens - self.config.min_active_tokens,
                int(round(self.config.skip_ratio * n_tokens)),
            ),
        )
        # Skip the *lowest*-scored tokens.
        skip_indices = np.argsort(scores)[:n_skip]
        skip_mask = np.zeros(n_tokens, dtype=bool)
        skip_mask[skip_indices] = True

        n_active = int(np.sum(~skip_mask))
        n_skipped = int(np.sum(skip_mask))
        self._stats[layer_idx].append(n_active / n_tokens if n_tokens > 0 else 1.0)

        return MoDLayerResult(
            layer_idx=layer_idx,
            n_tokens=n_tokens,
            n_active=n_active,
            n_skipped=n_skipped,
            skip_mask=skip_mask,
        )

    def apply_layer(
        self,
        hidden_states: np.ndarray,
        layer_output: np.ndarray,
        result: MoDLayerResult,
    ) -> np.ndarray:
        """Merge *layer_output* (active tokens only) back into *hidden_states*.

        Skipped token positions retain their original values (residual bypass).

        Parameters
        ----------
        hidden_states:
            Original sequence ``(n_tokens, hidden_dim)``.
        layer_output:
            Transformer output for active tokens ``(n_active, hidden_dim)``.
        result:
            The :class:`MoDLayerResult` from :meth:`route`.
        """
        out = hidden_states.copy()
        active_positions = np.where(~result.skip_mask)[0]
        if layer_output.shape[0] != len(active_positions):
            raise ValueError(
                f"layer_output has {layer_output.shape[0]} rows but "
                f"result has {len(active_positions)} active positions"
            )
        out[active_positions] = layer_output
        return out

    def expected_flop_ratio(self) -> float:
        """Fraction of FLOPs relative to a dense model (1 − skip_ratio)."""
        return 1.0 - self.config.skip_ratio

    def reset_stats(self) -> None:
        """Clear accumulated per-layer active-ratio statistics."""
        for i in self._stats:
            self._stats[i].clear()

    def layer_stats(self) -> Dict[int, float]:
        """Return mean active-ratio per layer over all :meth:`route` calls."""
        return {
            i: float(np.mean(v)) if v else 0.0
            for i, v in self._stats.items()
        }

    def router_weight(self, layer_idx: int) -> np.ndarray:
        """Return the router projection vector for *layer_idx* (read-only view)."""
        return self._router_weights[layer_idx].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_tokens(
        self, hidden_states: np.ndarray, layer_idx: int
    ) -> np.ndarray:
        """Compute importance scores for each token.

        ``"linear"`` mode: project each token onto the layer's router vector.
        ``"threshold"`` mode: use the L2 norm of the hidden state.
        """
        if self.config.router_type == "threshold":
            return np.linalg.norm(hidden_states, axis=-1)

        w = self._router_weights[layer_idx]  # (router_dim,)
        dim = hidden_states.shape[-1]
        if w.shape[0] != dim:
            # Adapt projection dimension by truncating / zero-padding.
            pad = max(0, dim - w.shape[0])
            w_adapted = np.pad(w[: min(dim, w.shape[0])], (0, pad))
        else:
            w_adapted = w
        return hidden_states.astype(np.float32) @ w_adapted
