"""squish/serving/flexgen_offload.py

FlexGenOffload — LP-Optimal CPU/Disk Weight Placement for Large-Model Inference.

Reference
---------
Sheng et al. "FlexGen: High-Throughput Generative Inference of Large Language
Models with a Single GPU." ICML 2023 (arXiv:2303.06865).

Algorithm
---------
FlexGen formulates inference throughput as a linear program where the decision
variables encode the fraction of each tensor (weights, KV cache, activations)
placed on GPU, CPU DRAM, or disk.  Constraints enforce memory capacities;
the objective maximises generation throughput.

This module provides:
* ``FlexGenOffload`` — a policy that decides, for each layer tensor, which
  device tier it resides on and pre-fetches it to GPU when needed.
* A simplified greedy policy (instead of a full LP solver) that fills GPU
  memory first, then DRAM, then disk.

Key properties
--------------
* ``plan()`` assigns tensors to tiers and returns a ``OffloadPlan``.
* ``prefetch(layer_idx)`` simulates moving the required tensors to GPU.
* ``evict(layer_idx)`` moves tensors back to their home tier.
* NumPy-only simulation; no actual IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DeviceTier",
    "FlexGenOffloadConfig",
    "OffloadPlan",
    "FlexGenOffload",
]


class DeviceTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class FlexGenOffloadConfig:
    """Configuration for :class:`FlexGenOffload`.

    Attributes:
        gpu_memory_gb: Available GPU memory in GiB.
        cpu_memory_gb: Available CPU DRAM in GiB.
        n_layers: Number of transformer layers.
        weight_bytes_per_layer: Approximate weight bytes per layer.
        kv_bytes_per_layer: Approximate KV cache bytes per layer.
        prefetch_bandwidth_gbps: Simulated PCIe bandwidth (GB/s).
    """

    gpu_memory_gb: float = 24.0
    cpu_memory_gb: float = 64.0
    n_layers: int = 32
    weight_bytes_per_layer: int = 500_000_000  # 500 MB
    kv_bytes_per_layer: int = 50_000_000  # 50 MB
    prefetch_bandwidth_gbps: float = 16.0  # PCIe 4.0 ×16


@dataclass
class OffloadPlan:
    """Tensor placement plan produced by :class:`FlexGenOffload`.

    Attributes:
        weight_tier: Device tier for each layer's weights.
        kv_tier: Device tier for each layer's KV cache.
        gpu_util: Fraction of GPU memory used (0–1).
        cpu_util: Fraction of CPU memory used (0–1).
    """

    weight_tier: List[DeviceTier]
    kv_tier: List[DeviceTier]
    gpu_util: float
    cpu_util: float


class FlexGenOffload:
    """Greedy LP-inspired tensor offload policy for large-model inference.

    Parameters
    ----------
    config:
        FlexGenOffload configuration.
    """

    def __init__(self, config: Optional[FlexGenOffloadConfig] = None) -> None:
        self._cfg = config or FlexGenOffloadConfig()
        self._plan: Optional[OffloadPlan] = None
        self._on_gpu: Dict[int, bool] = {}

    @property
    def config(self) -> FlexGenOffloadConfig:
        return self._cfg

    def plan(self) -> OffloadPlan:
        """Compute a greedy tensor placement plan.

        Returns
        -------
        OffloadPlan
        """
        cfg = self._cfg
        gpu_cap = int(cfg.gpu_memory_gb * (1 << 30))
        cpu_cap = int(cfg.cpu_memory_gb * (1 << 30))
        gpu_used = 0
        cpu_used = 0

        weight_tiers: List[DeviceTier] = []
        kv_tiers: List[DeviceTier] = []

        for layer in range(cfg.n_layers):
            wb = cfg.weight_bytes_per_layer
            kb = cfg.kv_bytes_per_layer

            # Weights
            if gpu_used + wb <= gpu_cap:
                weight_tiers.append(DeviceTier.GPU)
                gpu_used += wb
            elif cpu_used + wb <= cpu_cap:
                weight_tiers.append(DeviceTier.CPU)
                cpu_used += wb
            else:
                weight_tiers.append(DeviceTier.DISK)

            # KV cache
            if gpu_used + kb <= gpu_cap:
                kv_tiers.append(DeviceTier.GPU)
                gpu_used += kb
            elif cpu_used + kb <= cpu_cap:
                kv_tiers.append(DeviceTier.CPU)
                cpu_used += kb
            else:
                kv_tiers.append(DeviceTier.DISK)

        self._plan = OffloadPlan(
            weight_tier=weight_tiers,
            kv_tier=kv_tiers,
            gpu_util=gpu_used / gpu_cap,
            cpu_util=cpu_used / cpu_cap,
        )
        return self._plan

    def prefetch(self, layer_idx: int) -> float:
        """Simulate pre-fetching a layer to GPU.

        Returns
        -------
        Simulated transfer time in milliseconds.
        """
        if self._plan is None:
            self.plan()
        assert self._plan is not None

        tier = self._plan.weight_tier[layer_idx]
        if tier == DeviceTier.GPU:
            self._on_gpu[layer_idx] = True
            return 0.0
        bw_bytes = self._cfg.prefetch_bandwidth_gbps * (1 << 30)
        transfer_s = self._cfg.weight_bytes_per_layer / bw_bytes
        self._on_gpu[layer_idx] = True
        return transfer_s * 1000.0

    def evict(self, layer_idx: int) -> None:
        """Evict a layer from the simulated GPU cache."""
        self._on_gpu.pop(layer_idx, None)

    def is_on_gpu(self, layer_idx: int) -> bool:
        """Return True if layer is currently resident in GPU."""
        return self._on_gpu.get(layer_idx, False)
