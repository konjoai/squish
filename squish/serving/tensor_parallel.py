"""squish/serving/tensor_parallel.py

TensorParallel — Activation-Memory-Optimal Tensor Sharding.

Reference
---------
Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language
Models Using Model Parallelism." arXiv:1909.08053 (2019).

Algorithm
---------
Megatron-style column/row tensor parallelism splits large matrix multiplications
across ``world_size`` devices:

* **Column parallel** — split the weight matrix along the *output* dimension.
  Each device computes a shard of the output.  An AllGather is needed to
  reconstruct the full output.
* **Row parallel** — split the weight matrix along the *input* dimension.
  Each device receives a slice of the input.  An AllReduce sum combines partial
  outputs.

This module provides a CPU NumPy simulation:
* ``split_weights_column(W)`` → list of shards.
* ``split_weights_row(W)`` → list of shards.
* ``column_forward(x, shards)`` → full output after simulated AllGather.
* ``row_forward(x_shards, shards)`` → full output after simulated AllReduce.

Key properties
--------------
* No real inter-device communication — all-reduce simulated as a local sum.
* ``world_size`` — number of shards / virtual devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "TensorParallelConfig",
    "TensorParallel",
]


@dataclass
class TensorParallelConfig:
    """Configuration for :class:`TensorParallel`.

    Attributes:
        world_size: Number of tensor-parallel shards (virtual devices).
    """

    world_size: int = 4

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be ≥ 1")


class TensorParallel:
    """Megatron-style tensor parallelism simulator.

    Parameters
    ----------
    config:
        TensorParallelConfig.
    """

    def __init__(self, config: Optional[TensorParallelConfig] = None) -> None:
        self._cfg = config or TensorParallelConfig()

    @property
    def config(self) -> TensorParallelConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Weight sharding
    # ------------------------------------------------------------------

    def split_weights_column(self, W: np.ndarray) -> List[np.ndarray]:
        """Column-parallel split: divide W along out_features.

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.

        Returns
        -------
        List of ``world_size`` shards, each of shape
        ``(out_features // world_size, in_features)``.
        """
        W = np.asarray(W, dtype=np.float32)
        ws = self._cfg.world_size
        out_f = W.shape[0]
        shard_size = (out_f + ws - 1) // ws
        shards = []
        for i in range(ws):
            s = i * shard_size
            e = min(s + shard_size, out_f)
            shards.append(W[s:e, :].copy())
        return shards

    def split_weights_row(self, W: np.ndarray) -> List[np.ndarray]:
        """Row-parallel split: divide W along in_features.

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.

        Returns
        -------
        List of ``world_size`` shards, each of shape
        ``(out_features, in_features // world_size)``.
        """
        W = np.asarray(W, dtype=np.float32)
        ws = self._cfg.world_size
        in_f = W.shape[1]
        shard_size = (in_f + ws - 1) // ws
        shards = []
        for i in range(ws):
            s = i * shard_size
            e = min(s + shard_size, in_f)
            shards.append(W[:, s:e].copy())
        return shards

    def split_input_row(self, x: np.ndarray) -> List[np.ndarray]:
        """Split input activations along the feature dimension for row-parallel."""
        x = np.asarray(x, dtype=np.float32)
        ws = self._cfg.world_size
        in_f = x.shape[-1]
        shard_size = (in_f + ws - 1) // ws
        shards = []
        for i in range(ws):
            s = i * shard_size
            e = min(s + shard_size, in_f)
            shards.append(x[..., s:e].copy())
        return shards

    # ------------------------------------------------------------------
    # Simulated collective operations
    # ------------------------------------------------------------------

    def column_forward(
        self,
        x: np.ndarray,
        shards: List[np.ndarray],
        bias_shards: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Column-parallel linear forward with simulated AllGather.

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, in_features)``.
        shards:
            Column-parallel weight shards from :meth:`split_weights_column`.
        bias_shards:
            Optional bias shards matching ``shards``.

        Returns
        -------
        Full output, shape ``(batch, out_features)``.
        """
        outputs = []
        for i, shard in enumerate(shards):
            out_shard = np.asarray(x, dtype=np.float32) @ shard.T
            if bias_shards is not None:
                out_shard = out_shard + bias_shards[i][None, :]
            outputs.append(out_shard)
        return np.concatenate(outputs, axis=-1)

    def row_forward(
        self,
        x_shards: List[np.ndarray],
        weight_shards: List[np.ndarray],
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Row-parallel linear forward with simulated AllReduce.

        Parameters
        ----------
        x_shards:
            Input activation shards (from :meth:`split_input_row`).
        weight_shards:
            Row-parallel weight shards from :meth:`split_weights_row`.
        bias:
            Optional full bias, shape ``(out_features,)``.

        Returns
        -------
        Full output after AllReduce, shape ``(batch, out_features)``.
        """
        partial_sum = None
        for x_shard, w_shard in zip(x_shards, weight_shards):
            partial = np.asarray(x_shard, dtype=np.float32) @ w_shard.T
            if partial_sum is None:
                partial_sum = partial
            else:
                partial_sum = partial_sum + partial
        assert partial_sum is not None
        if bias is not None:
            partial_sum = partial_sum + bias[None, :]
        return partial_sum

    def all_reduce(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Simulate AllReduce sum across shards.

        Parameters
        ----------
        tensors:
            List of arrays with the same shape.

        Returns
        -------
        Element-wise sum.
        """
        result = tensors[0].copy()
        for t in tensors[1:]:
            result = result + t
        return result
