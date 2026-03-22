"""squish/speculative/dynamic_spec_len.py

DynamicSpecLen — Online Adaptive Draft-Length Router.

Reference
---------
Xing et al. "Dynamic Speculation Lookahead Accelerates Speculative
Decoding." NAACL 2024 (arXiv:2405.04304).

Algorithm
---------
Fixed speculative draft lengths K are suboptimal: short K wastes latency
savings while long K wastes target verification budget when acceptance
rate is low.  DynamicSpecLen predicts the optimal K per token:

1. Extract a lightweight feature from the draft model hidden state
   (e.g., top-1 probability, entropy, token frequency).
2. A small learned router predicts: "given these features, how many
   draft tokens will be accepted on average?"
3. The prediction is clipped to [K_min, K_max] and used as the draft
   length for that step.
4. Online updates use the observed acceptance count as the supervision
   signal.

Key properties
--------------
* NumPy-only.
* ``k_min`` / ``k_max`` — draft length bounds.
* ``n_features`` — number of router input features.
* ``lr`` — online learning rate for router updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DynamicSpecLenConfig",
    "DynamicSpecLen",
]


@dataclass
class DynamicSpecLenConfig:
    """Configuration for :class:`DynamicSpecLen`.

    Attributes:
        min_spec_len: Minimum draft length.
        max_spec_len: Maximum draft length.
        n_features: Number of router input features.
        lr: Online learning rate.
        hidden_size: Hidden layer size for the 2-layer router MLP.
        vocab_size: Vocabulary size (used for feature extraction).
    """

    min_spec_len: int = 1
    max_spec_len: int = 8
    n_features: int = 8
    lr: float = 5e-3
    hidden_size: int = 16
    vocab_size: int = 32000

    def __post_init__(self) -> None:
        if self.min_spec_len < 1 or self.max_spec_len < self.min_spec_len:
            raise ValueError("min_spec_len >= 1 and max_spec_len >= min_spec_len required")


class DynamicSpecLen:
    """Lightweight router that predicts optimal speculative draft length.

    Parameters
    ----------
    config:
        DynamicSpecLen configuration.
    seed:
        RNG seed.
    """

    def __init__(self, config: Optional[DynamicSpecLenConfig] = None, seed: int = 0) -> None:
        self._cfg = config or DynamicSpecLenConfig()
        rng = np.random.default_rng(seed)
        # 2-layer MLP: features → hidden → 1
        scale1 = 1.0 / np.sqrt(self._cfg.n_features)
        scale2 = 1.0 / np.sqrt(self._cfg.hidden_size)
        self._W1: np.ndarray = rng.standard_normal(
            (self._cfg.hidden_size, self._cfg.n_features)
        ).astype(np.float32) * scale1
        self._b1: np.ndarray = np.zeros(self._cfg.hidden_size, dtype=np.float32)
        self._W2: np.ndarray = rng.standard_normal(
            (1, self._cfg.hidden_size)
        ).astype(np.float32) * scale2
        self._b2: np.ndarray = np.zeros(1, dtype=np.float32)
        self._prediction_history: List[int] = []
        self._actual_history: List[int] = []

    @property
    def config(self) -> DynamicSpecLenConfig:
        return self._cfg

    def _forward(self, features: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Forward pass, returning prediction and intermediate activations."""
        h1 = np.tanh(self._W1 @ features + self._b1)  # (hidden,)
        out = float((self._W2 @ h1 + self._b2)[0])
        return out, h1, features

    def extract_features(self, logits: np.ndarray) -> np.ndarray:
        """Extract routing features from draft logits.

        Parameters
        ----------
        logits:
            Draft model logits, shape ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Feature vector, shape ``(n_features,)``.
        """
        logits = np.asarray(logits, dtype=np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        top_p = float(probs.max())
        entropy = float(-(probs * np.log(probs.clip(min=1e-10))).sum())
        top5 = probs[np.argpartition(probs, -5)[-5:]]
        features = np.concatenate([
            [top_p, entropy],
            top5,
            [float(np.log(logits.shape[0]))],  # log vocab_size
        ]).astype(np.float32)
        # Pad or truncate to n_features
        if len(features) < self._cfg.n_features:
            features = np.pad(features, (0, self._cfg.n_features - len(features)))
        else:
            features = features[: self._cfg.n_features]
        return features

    def predict(self, logits: np.ndarray) -> int:
        """Predict the optimal draft length K for this token.

        Parameters
        ----------
        logits:
            Draft model logits, shape ``(vocab_size,)``.

        Returns
        -------
        int
            Predicted draft length in [k_min, k_max].
        """
        features = self.extract_features(logits)
        raw, _, _ = self._forward(features)
        k_range = self._cfg.max_spec_len - self._cfg.min_spec_len
        k = int(np.round(np.clip(raw, 0.0, 1.0) * k_range + self._cfg.min_spec_len))
        k = int(np.clip(k, self._cfg.min_spec_len, self._cfg.max_spec_len))
        self._prediction_history.append(k)
        return k

    def update(self, logits: np.ndarray, actual_accepted: int) -> None:
        """Update the router with the observed acceptance count.

        Parameters
        ----------
        logits:
            Draft logits used at the prediction step.
        actual_accepted:
            Actual number of tokens accepted (0 … k_max).
        """
        features = self.extract_features(logits)
        raw, h1, _ = self._forward(features)
        target = (actual_accepted - self._cfg.min_spec_len) / max(
            self._cfg.max_spec_len - self._cfg.min_spec_len, 1
        )
        target = float(np.clip(target, 0.0, 1.0))
        # MSE gradient
        error = raw - target
        d_out = error
        d_W2 = d_out * h1[None, :]
        d_b2 = np.array([d_out])
        d_h1 = d_out * self._W2.squeeze(0)
        d_h1_pre = d_h1 * (1.0 - h1 ** 2)  # tanh derivative
        d_W1 = d_h1_pre[:, None] * features[None, :]
        d_b1 = d_h1_pre
        self._W1 -= self._cfg.lr * d_W1
        self._b1 -= self._cfg.lr * d_b1
        self._W2 -= self._cfg.lr * d_W2
        self._b2 -= self._cfg.lr * d_b2
        self._actual_history.append(actual_accepted)

    def mean_predicted_k(self) -> float:
        if not self._prediction_history:
            return float(self._cfg.min_spec_len)
        return float(np.mean(self._prediction_history))

    def reset_stats(self) -> None:
        self._prediction_history.clear()
        self._actual_history.clear()
