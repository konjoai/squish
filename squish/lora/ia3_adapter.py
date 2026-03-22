"""squish/lora/ia3_adapter.py

IA3Adapter — Infused Adapter via Inhibiting and Amplifying Inner Activations.

Reference
---------
Liu et al. "(IA)³: Few-Shot Parameter-Efficient Fine-Tuning is Better and
Cheaper than In-Context Learning." NeurIPS 2022 (arXiv 2205.05638).

Algorithm
---------
IA³ inserts three learned element-wise scale vectors into a transformer:

* ``l_k`` — scales key activations:  ``K' = K * l_k``
* ``l_v`` — scales value activations: ``V' = V * l_v``
* ``l_ff`` — scales the inner FFN activations after the non-linearity.

All other weights stay frozen.  Because the scales start at ones, the
adapter is a no-op at initialisation.  At inference, scales can be
*merged* back into the underlying weight matrices so there is zero
overhead: ``W_k_merged = W_k * l_k[None, :]``.

Multiple adapters can be composed by element-wise multiplication of their
scales via ``ia3_compose()``.

This module provides:

1. ``IA3Adapter`` — holds the three scale vectors.
2. ``ia3_compose(adapters)`` — compose multiple adapters into a single one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "IA3Config",
    "IA3Adapter",
    "ia3_compose",
]


@dataclass
class IA3Config:
    """Configuration for :class:`IA3Adapter`.

    Attributes:
        d_model: Key/value feature dimension.
        d_ff: Inner FFN dimension.
        seed: RNG seed for reproducibility.
    """

    d_model: int = 4096
    d_ff: int = 16384
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1; got {self.d_model}")
        if self.d_ff < 1:
            raise ValueError(f"d_ff must be >= 1; got {self.d_ff}")


class IA3Adapter:
    """IA³ adapter: three learned scale vectors for K, V, and FF activations.

    Scales are initialised to ones so the adapter is a no-op at the start
    of training.  In this simulation the scales are fixed random
    perturbations around 1.0 to enable testing of the merge/apply paths.

    Example::

        cfg = IA3Config(d_model=64, d_ff=256)
        adapter = IA3Adapter(cfg)

        K = np.random.randn(8, 64)   # (seq_len, d_model)
        K_scaled = adapter.apply_k(K)

        W_k = np.random.randn(64, 64)
        W_k_merged, _, _ = adapter.merge_to_base(W_k, W_k.copy(), np.random.randn(64, 256))
    """

    def __init__(self, config: Optional[IA3Config] = None) -> None:
        self._cfg = config or IA3Config()
        rng = np.random.default_rng(self._cfg.seed)
        # Initialise near 1.0 with small noise so tests can verify non-trivial scaling.
        self._scale_k: np.ndarray = (
            1.0 + rng.standard_normal(self._cfg.d_model).astype(np.float32) * 0.1
        )
        self._scale_v: np.ndarray = (
            1.0 + rng.standard_normal(self._cfg.d_model).astype(np.float32) * 0.1
        )
        self._scale_ff: np.ndarray = (
            1.0 + rng.standard_normal(self._cfg.d_ff).astype(np.float32) * 0.1
        )

    @property
    def config(self) -> IA3Config:
        return self._cfg

    @property
    def scale_k(self) -> np.ndarray:
        """Learned key scale ``(d_model,)``."""
        return self._scale_k

    @property
    def scale_v(self) -> np.ndarray:
        """Learned value scale ``(d_model,)``."""
        return self._scale_v

    @property
    def scale_ff(self) -> np.ndarray:
        """Learned FFN inner scale ``(d_ff,)``."""
        return self._scale_ff

    def apply_k(self, K: np.ndarray) -> np.ndarray:
        """Scale key activations.

        Args:
            K: ``(..., d_model)`` key tensor.

        Returns:
            ``K * scale_k`` broadcast along the last dimension.
        """
        return np.asarray(K, dtype=np.float32) * self._scale_k

    def apply_v(self, V: np.ndarray) -> np.ndarray:
        """Scale value activations.

        Args:
            V: ``(..., d_model)`` value tensor.
        """
        return np.asarray(V, dtype=np.float32) * self._scale_v

    def apply_ff(self, h: np.ndarray) -> np.ndarray:
        """Scale inner FFN activations.

        Args:
            h: ``(..., d_ff)`` hidden tensor after the non-linearity.
        """
        return np.asarray(h, dtype=np.float32) * self._scale_ff

    def merge_to_base(
        self,
        W_k: np.ndarray,
        W_v: np.ndarray,
        W_ff: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge scales into existing weight matrices (inference optimisation).

        For a linear ``y = x @ W``, the scaled output is
        ``y' = (x @ W) * l = x @ (W * l[None, :])``.

        Args:
            W_k: ``(in, d_model)`` key projection matrix.
            W_v: ``(in, d_model)`` value projection matrix.
            W_ff: ``(in, d_ff)`` FFN projection matrix.

        Returns:
            Triple ``(W_k_merged, W_v_merged, W_ff_merged)``.
        """
        W_k = np.asarray(W_k, dtype=np.float32)
        W_v = np.asarray(W_v, dtype=np.float32)
        W_ff = np.asarray(W_ff, dtype=np.float32)
        return (
            W_k * self._scale_k[np.newaxis, :],
            W_v * self._scale_v[np.newaxis, :],
            W_ff * self._scale_ff[np.newaxis, :],
        )

    def zero_scales(self) -> None:
        """Reset all scales to zero (merge-to-zero for ablation studies)."""
        self._scale_k[:] = 0.0
        self._scale_v[:] = 0.0
        self._scale_ff[:] = 0.0

    def reset_to_identity(self) -> None:
        """Reset all scales to one (identity / no-op)."""
        self._scale_k[:] = 1.0
        self._scale_v[:] = 1.0
        self._scale_ff[:] = 1.0


def ia3_compose(adapters: List[IA3Adapter]) -> IA3Adapter:
    """Compose multiple IA³ adapters into a single equivalent adapter.

    Composition is element-wise product of the scale vectors, which is
    mathematically equivalent to applying each adapter sequentially.

    Args:
        adapters: Non-empty list of :class:`IA3Adapter` instances.  All
            must share the same ``d_model`` / ``d_ff``.

    Returns:
        A new :class:`IA3Adapter` whose scales are the products of all
        input adapter scales.  The underlying config is copied from the
        first adapter.

    Raises:
        ValueError: If *adapters* is empty or configs are incompatible.
    """
    if not adapters:
        raise ValueError("ia3_compose() requires at least one adapter")
    cfg0 = adapters[0].config
    for i, a in enumerate(adapters[1:], start=1):
        if a.config.d_model != cfg0.d_model or a.config.d_ff != cfg0.d_ff:
            raise ValueError(
                f"Adapter {i} has incompatible config "
                f"(d_model={a.config.d_model}, d_ff={a.config.d_ff}) "
                f"vs first adapter (d_model={cfg0.d_model}, d_ff={cfg0.d_ff})"
            )
    composed = IA3Adapter(IA3Config(d_model=cfg0.d_model, d_ff=cfg0.d_ff, seed=0))
    sk = np.ones(cfg0.d_model, dtype=np.float32)
    sv = np.ones(cfg0.d_model, dtype=np.float32)
    sff = np.ones(cfg0.d_ff, dtype=np.float32)
    for a in adapters:
        sk = sk * a.scale_k
        sv = sv * a.scale_v
        sff = sff * a.scale_ff
    composed._scale_k = sk
    composed._scale_v = sv
    composed._scale_ff = sff
    return composed
