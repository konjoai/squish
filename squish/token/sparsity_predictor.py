# [Experimental] This module is part of Squish v40+ (Wave 66).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""sparsity_predictor.py — Lightweight Per-Layer Cluster Activity Predictor.

For each FFN layer, a small linear classifier predicts *which co-activation
clusters are active* given the current hidden-state vector.  The predictor is
trained offline (during calibration, after :mod:`squish.compress.sparsity_profiler`
has assigned neurons to clusters) and its float16 weight matrix is stored
inside the ``.squizd`` sparsity metadata block.

At inference time the predictor is evaluated *before* the FFN GEMV, producing
a ``(n_clusters,) bool`` active-cluster mask that is passed to
``sparse_gemv.metal`` via an MTLBuffer.  The predictor itself is fast: it is
a single ``(d_model, n_clusters)`` matrix-vector multiply in float16, taking
~0.5–1 ms on an M-series GPU — well within the 5–10 % overhead budget vs the
40–50 % FFN bandwidth it saves.

Usage::

    from squish.token.sparsity_predictor import PredictorConfig, SparsityPredictor

    cfg = PredictorConfig(d_model=4096, n_clusters=64)
    pred = SparsityPredictor(cfg)

    # Train from calibration data.
    pred.train(hidden_states, cluster_active_labels)

    # Inference.
    mask = pred.predict(hidden_state)   # (n_clusters,) bool

    # Serialise into .squizd sparsity metadata block.
    raw = pred.to_bytes()
    pred2 = SparsityPredictor.from_bytes(raw, cfg)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = [
    "PredictorConfig",
    "SparsityPredictor",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MAGIC = b"SQPRED\x01\x00"  # 8-byte magic for serialised predictor blocks


@dataclass
class PredictorConfig:
    """Configuration for :class:`SparsityPredictor`.

    Attributes:
        d_model: Hidden-state dimension (== predictor input size).
        n_clusters: Number of co-activation clusters (== predictor output size).
        threshold: Logit threshold above which a cluster is predicted active.
            Default 0.0 (sign test).
        learning_rate: SGD step size for :meth:`SparsityPredictor.train`.
        n_epochs: Training epochs.
        batch_size: Mini-batch size during training.
    """

    d_model: int
    n_clusters: int
    threshold: float = 0.0
    learning_rate: float = 0.01
    n_epochs: int = 20
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class SparsityPredictor:
    """Single-layer linear predictor for co-activation cluster activity.

    Parameters
    ----------
    config:
        :class:`PredictorConfig` describing model and training hyperparameters.
    """

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        # Weight matrix: (d_model, n_clusters) float16.
        # Initialised to zero; set via train() or from_bytes().
        self._weights: np.ndarray = np.zeros(
            (config.d_model, config.n_clusters), dtype=np.float16
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, hidden_state: np.ndarray) -> np.ndarray:
        """Predict the active cluster mask for a single hidden state.

        Args:
            hidden_state: ``(d_model,)`` float32 hidden-state vector.

        Returns:
            ``(n_clusters,)`` bool array — True for clusters predicted active.

        Raises:
            ValueError: If *hidden_state* shape does not match *d_model*.
        """
        hs = np.asarray(hidden_state, dtype=np.float32).ravel()
        if hs.shape[0] != self.config.d_model:
            raise ValueError(
                f"hidden_state length {hs.shape[0]} != d_model {self.config.d_model}"
            )
        # float16 weights × float32 input → accumulate in float32.
        logits = self._weights.astype(np.float32).T @ hs  # (n_clusters,)
        return logits > self.config.threshold

    def predict_batch(self, hidden_states: np.ndarray) -> np.ndarray:
        """Predict active cluster masks for a batch of hidden states.

        Args:
            hidden_states: ``(n_samples, d_model)`` float32 matrix.

        Returns:
            ``(n_samples, n_clusters)`` bool matrix.
        """
        hs = np.asarray(hidden_states, dtype=np.float32)
        if hs.ndim == 1:
            hs = hs[np.newaxis, :]
        if hs.shape[1] != self.config.d_model:
            raise ValueError(
                f"hidden_states.shape[1]={hs.shape[1]} != d_model={self.config.d_model}"
            )
        logits = hs @ self._weights.astype(np.float32)  # (n, n_clusters)
        return logits > self.config.threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        hidden_states: np.ndarray,
        cluster_labels: np.ndarray,
        *,
        verbose: bool = False,
    ) -> List[float]:
        """Fit the predictor weight matrix using mini-batch SGD.

        Each cluster is a binary classification problem (active / inactive).
        The loss is binary cross-entropy with sigmoid output.  All clusters
        are trained jointly in a single weight matrix.

        Args:
            hidden_states: ``(n_samples, d_model)`` float32 calibration data.
            cluster_labels: ``(n_samples, n_clusters)`` float32 targets in
                {0, 1} — 1 if cluster c was active for sample i, else 0.
            verbose: If True, print epoch loss.

        Returns:
            List of per-epoch average losses.
        """
        X = np.asarray(hidden_states, dtype=np.float32)
        Y = np.asarray(cluster_labels, dtype=np.float32)

        if X.shape[1] != self.config.d_model:
            raise ValueError("hidden_states.shape[1] != d_model")
        if Y.shape[1] != self.config.n_clusters:
            raise ValueError("cluster_labels.shape[1] != n_clusters")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("hidden_states and cluster_labels sample counts differ")

        n_samples = X.shape[0]
        # Start from current weights (supports warm-start via multiple calls).
        W = self._weights.astype(np.float32)  # (d_model, n_clusters)
        lr = self.config.learning_rate
        epoch_losses: List[float] = []

        rng = np.random.default_rng(42)
        for epoch in range(self.config.n_epochs):
            perm = rng.permutation(n_samples)
            X_s, Y_s = X[perm], Y[perm]
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                Xb = X_s[start:end]   # (b, d_model)
                Yb = Y_s[start:end]   # (b, n_clusters)

                logits = Xb @ W       # (b, n_clusters)
                probs  = _sigmoid(logits)              # (b, n_clusters)
                diff   = probs - Yb                    # gradient wrt logits
                grad   = Xb.T @ diff / (end - start)   # (d_model, n_clusters)

                W -= lr * grad

                # BCE loss: -[y log p + (1-y) log(1-p)], clipped for stability.
                eps = 1e-7
                loss = -np.mean(
                    Yb * np.log(probs + eps) + (1 - Yb) * np.log(1 - probs + eps)
                )
                total_loss += float(loss)
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            if verbose:
                print(f"  [SparsityPredictor] epoch {epoch+1}/{self.config.n_epochs} loss={avg_loss:.4f}")

        # Store as float16.
        self._weights = W.astype(np.float16)
        return epoch_losses

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def accuracy(
        self, hidden_states: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """Compute element-wise binary accuracy over a test set.

        Args:
            hidden_states: ``(n_samples, d_model)`` float32 input matrix.
            cluster_labels: ``(n_samples, n_clusters)`` float32 ground-truth
                labels in {0, 1}.

        Returns:
            Fraction of (sample, cluster) pairs predicted correctly in [0, 1].
        """
        preds = self.predict_batch(hidden_states)
        targets = np.asarray(cluster_labels, dtype=bool)
        return float(np.mean(preds == targets))

    def recall(
        self, hidden_states: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """Compute per-cluster recall (active cluster detection rate).

        A false negative means an *active* cluster was predicted inactive —
        that cluster's FFN contribution would be silently dropped.  This
        metric should stay above ~0.95 for acceptable model quality.

        Returns:
            Average per-cluster recall across all clusters in [0, 1].
        """
        preds = self.predict_batch(hidden_states).astype(np.float32)
        targets = np.asarray(cluster_labels, dtype=np.float32)
        # Per-cluster recall: TP / (TP + FN)
        tp = np.sum(preds * targets, axis=0)
        fn = np.sum((1 - preds) * targets, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cluster = np.where((tp + fn) > 0, tp / (tp + fn), 1.0)
        return float(np.mean(per_cluster))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialise the predictor weight matrix to a byte string.

        Wire format::

            [8B  magic "SQPRED\\x01\\x00"]
            [4B  d_model    uint32 LE]
            [4B  n_clusters uint32 LE]
            [4B  threshold  float32 LE]
            [d_model × n_clusters × 2B  float16 LE, row-major]

        Returns:
            Bytes suitable for embedding in a ``.squizd`` metadata block.
        """
        header = struct.pack(
            "<8sIIf",
            _MAGIC,
            self.config.d_model,
            self.config.n_clusters,
            self.config.threshold,
        )
        weights_bytes = self._weights.astype("<f2").tobytes()
        return header + weights_bytes

    @classmethod
    def from_bytes(cls, data: bytes, config: "PredictorConfig") -> "SparsityPredictor":
        """Deserialise a predictor from bytes produced by :meth:`to_bytes`.

        Args:
            data: Byte string as returned by :meth:`to_bytes`.
            config: :class:`PredictorConfig` with the correct ``d_model`` and
                ``n_clusters`` (used to instantiate the predictor).

        Returns:
            A :class:`SparsityPredictor` with the stored weight matrix loaded.

        Raises:
            ValueError: If the magic header is wrong or sizes mismatch.
        """
        header_size = 8 + 4 + 4 + 4  # magic + d_model + n_clusters + threshold
        if len(data) < header_size:
            raise ValueError(f"Data too short: {len(data)} bytes")

        magic, d_model, n_clusters, threshold = struct.unpack_from("<8sIIf", data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")
        if d_model != config.d_model:
            raise ValueError(
                f"Stored d_model={d_model} != config.d_model={config.d_model}"
            )
        if n_clusters != config.n_clusters:
            raise ValueError(
                f"Stored n_clusters={n_clusters} != config.n_clusters={config.n_clusters}"
            )

        expected_weights = d_model * n_clusters * 2  # float16 = 2 bytes each
        raw_weights = data[header_size : header_size + expected_weights]
        if len(raw_weights) != expected_weights:
            raise ValueError(
                f"Expected {expected_weights} weight bytes, got {len(raw_weights)}"
            )

        weights = np.frombuffer(raw_weights, dtype="<f2").astype(np.float16)
        weights = weights.reshape(d_model, n_clusters)

        pred = cls(config)
        pred._weights = weights
        return pred

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def predictor_weights(self) -> np.ndarray:
        """``(d_model, n_clusters)`` float16 weight matrix (read-only view)."""
        return self._weights

    @predictor_weights.setter
    def predictor_weights(self, value: np.ndarray) -> None:
        w = np.asarray(value, dtype=np.float16)
        if w.shape != (self.config.d_model, self.config.n_clusters):
            raise ValueError(
                f"Expected shape ({self.config.d_model}, {self.config.n_clusters}), "
                f"got {w.shape}"
            )
        self._weights = w

    def __repr__(self) -> str:
        return (
            f"SparsityPredictor(d_model={self.config.d_model}, "
            f"n_clusters={self.config.n_clusters}, "
            f"threshold={self.config.threshold})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids overflow for large negative x."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
