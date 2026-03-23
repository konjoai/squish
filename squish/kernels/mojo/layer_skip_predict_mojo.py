"""layer_skip_predict_mojo.py — Mojo-accelerated LayerSkip confidence predictor.

Wraps ``squish/kernels/mojo/kernels/layer_skip_predict.mojo`` via MojoBridge
(Wave 59b).
Falls back to NumPy when the Mojo library is unavailable.

MojoLayerSkipPredict replaces the Python list comprehension
``[sigmoid(dot(w, x)) for w in weights]`` in ``skip_layer_predictor.py``
``predict()`` with ``parallelize(n_layers)`` tasks; each task uses
``@parameter`` on ``n_features`` (16, 32) and ``vectorize`` SIMD FMA
dot-product + scalar sigmoid; ~8× for n_layers=32, n_features=32.

Also covers ``deja_vu_sparse.py`` inference forward pass.

Reference:
    Elhoushi et al. (ACL 2024, arXiv 2404.16710) — LayerSkip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["LayerSkipConfig", "MojoLayerSkipPredict"]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("layer_skip_predict")


@dataclass
class LayerSkipConfig:
    n_layers: int = 32
    n_features: int = 32


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _numpy_predict(
    weights: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """NumPy fallback: per-layer dot-product + sigmoid."""
    dots = weights @ features    # (n_layers,)
    return (1.0 / (1.0 + np.exp(-dots))).astype(np.float32)


class MojoLayerSkipPredict:
    """LayerSkip per-layer confidence predictor (Mojo → NumPy fallback).

    Maintains a weight matrix ``(n_layers, n_features)`` for all layers.

    Args:
        config: :class:`LayerSkipConfig`.
    """

    def __init__(self, config: Optional[LayerSkipConfig] = None) -> None:
        self._cfg = config or LayerSkipConfig()
        self._weights = np.zeros(
            (self._cfg.n_layers, self._cfg.n_features), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute per-layer skip confidence: ``sigmoid(weights[l] @ features)``.

        Args:
            features: ``(n_features,)`` float32 hidden-state feature vector.
            weights: Override internal weights with ``(n_layers, n_features)``
                float32 array (optional).

        Returns:
            ``(n_layers,)`` float32 confidence scores in ``[0, 1]``.
        """
        f = np.ascontiguousarray(features.ravel(), dtype=np.float32)
        w = np.ascontiguousarray(
            weights if weights is not None else self._weights,
            dtype=np.float32,
        )
        if w.ndim != 2 or w.shape[1] != f.shape[0]:
            raise ValueError(
                f"weights shape {w.shape} incompatible with features length {f.shape[0]}"
            )
        if _kernel is not None:
            try:
                result = _kernel(w, f)
                return np.asarray(result, dtype=np.float32).ravel()
            except Exception:
                pass
        return _numpy_predict(w, f)

    def update_weights(self, weights: np.ndarray) -> None:
        """Replace internal weight matrix.

        Args:
            weights: ``(n_layers, n_features)`` float32 array.
        """
        self._weights = np.asarray(weights, dtype=np.float32)

    def weights(self) -> np.ndarray:
        """Return internal weight matrix ``(n_layers, n_features)``."""
        return self._weights

    def n_layers(self) -> int:
        return self._cfg.n_layers

    def n_features(self) -> int:
        return self._cfg.n_features

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
