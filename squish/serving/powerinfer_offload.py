"""squish/serving/powerinfer_offload.py

PowerInferOffload — ReLU-Sparsity Hot/Cold Neuron Split.

Reference
---------
Song et al. "PowerInfer: Fast Large Language Model Serving with a Consumer-
Grade GPU." SOSP 2024 (arXiv:2312.12456).

Algorithm
---------
MLP layers in LLMs with ReLU activations exhibit *neuron sparsity*: on
average only 5–15 % of neurons are non-zero for any token.  PowerInfer:

1. **Profiling phase** — collect activation frequency for each neuron across
   a large corpus.  Neurons with high frequency are *hot*; others are *cold*.
2. **Placement** — hot neurons (typically top ~5 %) reside in GPU VRAM;
   cold neurons reside in CPU DRAM.
3. **Inference** — for each MLP call:
   a. Predict which neurons will activate (can be done with a small router).
   b. Pre-fetch only the predicted cold neurons.
   c. Execute the sparse MLP.

Key properties
--------------
* ``profile(activations)`` — measure per-neuron activation frequency.
* ``plan(hot_fraction)`` → ``NeuronPlan`` — assign neurons to GPU/CPU.
* ``sparse_forward(x, W_up, W_down, activation_mask)`` — simulate sparse MLP.
* NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PowerInferOffloadConfig",
    "NeuronPlan",
    "PowerInferOffload",
]


@dataclass
class PowerInferOffloadConfig:
    """Configuration for :class:`PowerInferOffload`.

    Attributes:
        n_neurons: Number of neurons in the MLP layer (hidden_dim).
        hot_fraction: Fraction of neurons assigned to GPU (0–1).
        predictor_threshold: Activation frequency above which a neuron is
            considered *hot* regardless of the hot_fraction cap.
        relu_threshold: Activation value threshold below which output is 0.
    """

    n_neurons: int = 4096
    hot_fraction: float = 0.1
    predictor_threshold: float = 0.5
    relu_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hot_fraction <= 1.0:
            raise ValueError("hot_fraction must be in [0, 1]")


@dataclass
class NeuronPlan:
    """Neuron placement plan.

    Attributes:
        hot_indices: Neuron indices assigned to GPU.
        cold_indices: Neuron indices assigned to CPU.
        activation_freq: Per-neuron activation frequency from profiling.
    """

    hot_indices: np.ndarray
    cold_indices: np.ndarray
    activation_freq: np.ndarray

    @property
    def n_hot(self) -> int:
        return len(self.hot_indices)

    @property
    def n_cold(self) -> int:
        return len(self.cold_indices)


class PowerInferOffload:
    """ReLU-sparsity hot/cold neuron offload manager.

    Parameters
    ----------
    config:
        PowerInferOffloadConfig.
    """

    def __init__(self, config: Optional[PowerInferOffloadConfig] = None) -> None:
        self._cfg = config or PowerInferOffloadConfig()
        self._freq: Optional[np.ndarray] = None
        self._plan: Optional[NeuronPlan] = None

    @property
    def config(self) -> PowerInferOffloadConfig:
        return self._cfg

    def profile(self, activations: np.ndarray) -> np.ndarray:
        """Measure per-neuron activation frequency.

        Parameters
        ----------
        activations:
            MLP intermediate activations, shape ``(n_samples, n_neurons)``.
            Values > relu_threshold are considered "active".

        Returns
        -------
        Per-neuron activation frequency, shape ``(n_neurons,)``.
        """
        A = np.asarray(activations, dtype=np.float32)
        active = (A > self._cfg.relu_threshold).astype(np.float32)
        self._freq = active.mean(axis=0)
        return self._freq

    def plan(self, activation_freq: Optional[np.ndarray] = None) -> NeuronPlan:
        """Partition neurons into hot (GPU) and cold (CPU) sets.

        Parameters
        ----------
        activation_freq:
            Optional externally computed frequency; otherwise uses the last
            result from :meth:`profile`.

        Returns
        -------
        NeuronPlan
        """
        if activation_freq is not None:
            self._freq = np.asarray(activation_freq, dtype=np.float32)
        if self._freq is None:
            # Default: uniform frequency
            self._freq = np.full(self._cfg.n_neurons, 0.1, dtype=np.float32)

        n = self._cfg.n_neurons
        n_hot = max(1, int(n * self._cfg.hot_fraction))
        sorted_idx = np.argsort(self._freq)[::-1]
        hot_idx = sorted_idx[:n_hot]
        cold_idx = sorted_idx[n_hot:]
        self._plan = NeuronPlan(
            hot_indices=hot_idx,
            cold_indices=cold_idx,
            activation_freq=self._freq,
        )
        return self._plan

    def sparse_forward(
        self,
        x: np.ndarray,
        W_up: np.ndarray,
        W_down: np.ndarray,
        predict_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate sparse MLP forward pass.

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, in_features)``.
        W_up:
            Up-projection weight, shape ``(n_neurons, in_features)``.
        W_down:
            Down-projection weight, shape ``(in_features, n_neurons)``.
        predict_mask:
            Boolean mask of shape ``(n_neurons,)`` indicating which neurons
            to compute.  If None, uses all neurons.

        Returns
        -------
        Output of shape ``(batch, in_features)``.
        """
        if predict_mask is None:
            predict_mask = np.ones(W_up.shape[0], dtype=bool)

        active_idx = np.where(predict_mask)[0]
        W_up_sparse = W_up[active_idx]  # (k, in_f)
        W_down_sparse = W_down[:, active_idx]  # (in_f, k)

        h = np.maximum(x @ W_up_sparse.T, self._cfg.relu_threshold)  # (batch, k)
        return h @ W_down_sparse.T  # (batch, in_f)
