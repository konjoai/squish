"""squish/moe/shared_expert.py

SharedExpertMoE — Shared + Routed Expert combination (DeepSeek-V2 style).

Every token is processed by ``n_shared`` always-active shared experts whose
output is summed in.  In parallel, a top-K router dispatches each token to
``top_k`` of the ``n_routed`` specialised experts.  The two paths are added
to form the final output, following the mixture formulation in DeepSeek-V2.

Reference
---------
DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-
Experts Language Model." arXiv:2405.04434, 2024.
"""

from __future__ import annotations

__all__ = ["SharedExpertConfig", "SharedExpertMoE"]

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SharedExpertConfig:
    """Configuration for SharedExpertMoE.

    Parameters
    ----------
    d_model:
        Residual-stream / hidden dimension.
    n_routed:
        Number of specialised (routed) experts.
    n_shared:
        Number of always-active shared experts.
    top_k:
        How many routed experts each token is dispatched to.
    expand_factor:
        Inner FFN hidden dimension = round(d_model * expand_factor).
    seed:
        RNG seed for weight initialisation.
    """

    d_model: int = 256
    n_routed: int = 8
    n_shared: int = 2
    top_k: int = 2
    expand_factor: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.n_routed < 1:
            raise ValueError("n_routed must be >= 1")
        if self.n_shared < 1:
            raise ValueError("n_shared must be >= 1")
        if self.top_k < 1 or self.top_k > self.n_routed:
            raise ValueError("top_k must be in [1, n_routed]")
        if self.expand_factor <= 0.0:
            raise ValueError("expand_factor must be > 0")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SharedExpertMoE:
    """Shared + Routed MoE forward pass (NumPy reference implementation).

    Architecture
    ------------
    out = shared_path(x) + routed_path(x)

    shared_path: each of the n_shared experts runs on every token; their
    outputs are averaged.

    routed_path: a linear router selects top_k experts for each token;
    outputs are weighted-summed with renormalised softmax weights.

    Parameters
    ----------
    config:
        ``SharedExpertConfig`` instance.
    """

    def __init__(self, config: SharedExpertConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d = config.d_model
        h = max(1, round(d * config.expand_factor))

        # Shared expert FFNs: (n_shared, d→h) up-projection and (n_shared, h→d) down
        scale = float(d) ** -0.5
        self._shared_up: ndarray = (
            rng.standard_normal((config.n_shared, h, d)).astype(np.float32) * scale
        )
        self._shared_down: ndarray = (
            rng.standard_normal((config.n_shared, d, h)).astype(np.float32) * scale
        )

        # Routed expert FFNs: (n_routed, d→h) up and (n_routed, h→d) down
        self._routed_up: ndarray = (
            rng.standard_normal((config.n_routed, h, d)).astype(np.float32) * scale
        )
        self._routed_down: ndarray = (
            rng.standard_normal((config.n_routed, d, h)).astype(np.float32) * scale
        )

        # Router weight: (n_routed, d)
        self._router_w: ndarray = (
            rng.standard_normal((config.n_routed, d)).astype(np.float32) * scale
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: ndarray) -> tuple[ndarray]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(T, d_model)``.

        Returns
        -------
        out:
            Output tensor of shape ``(T, d_model)``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config.d_model:
            raise ValueError(
                f"x must be (T, {self.config.d_model}), got {x.shape}"
            )
        shared_out = self._shared_forward(x)
        routed_out = self._routed_forward(x)
        return (shared_out + routed_out,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shared_forward(self, x: ndarray) -> ndarray:
        """Always-active shared expert path; averages n_shared expert outputs."""
        # x: (T, d)
        accumulator = np.zeros_like(x)
        for e in range(self.config.n_shared):
            # Up-projection: (T, d) @ (d, h) = (T, h)
            h_state = x @ self._shared_up[e].T
            h_state = np.maximum(0.0, h_state)  # ReLU
            # Down-projection: (T, h) @ (h, d) = (T, d)
            accumulator = accumulator + h_state @ self._shared_down[e].T
        return accumulator / self.config.n_shared

    def _routed_forward(self, x: ndarray) -> ndarray:
        """Top-K routed expert path."""
        indices, weights = self._router(x)  # (T, top_k), (T, top_k)
        T = x.shape[0]
        out = np.zeros_like(x)
        for k_idx in range(self.config.top_k):
            exp_indices = indices[:, k_idx]  # (T,)
            exp_weights = weights[:, k_idx]  # (T,)
            for e_id in range(self.config.n_routed):
                mask = exp_indices == e_id
                if not mask.any():
                    continue
                x_sel = x[mask]  # (m, d)
                h_state = x_sel @ self._routed_up[e_id].T
                h_state = np.maximum(0.0, h_state)
                e_out = h_state @ self._routed_down[e_id].T  # (m, d)
                out[mask] += e_out * exp_weights[mask, np.newaxis]
        return out

    def _router(self, x: ndarray) -> tuple[ndarray, ndarray]:
        """Compute top-K routing indices and renormalised softmax weights.

        Returns
        -------
        indices:
            Shape ``(T, top_k)``.
        weights:
            Shape ``(T, top_k)``, sums to 1 across top_k dimension.
        """
        # logits: (T, n_routed)
        logits = x @ self._router_w.T
        # softmax for stability in numerics
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        top_k = self.config.top_k
        # argsort descending, take first top_k
        sorted_indices = np.argsort(-probs, axis=-1)
        indices = sorted_indices[:, :top_k]
        raw_weights = np.take_along_axis(probs, indices, axis=-1)
        # renormalise so top_k weights sum to 1
        weight_sum = raw_weights.sum(axis=-1, keepdims=True)
        weights = raw_weights / np.where(weight_sum > 0, weight_sum, 1.0)
        return indices, weights
