"""SSMQuantizer: calibration-aware quantisation for SSM parameter matrices.

Linear-recurrent architectures (Mamba2, RWKV-6, Hawk, DeltaNet) have a
fundamentally different quantisation sensitivity profile compared to
transformer feedforward weights:

  * **Δ (time-step projection):** Spans only [dt_min, dt_max] after softplus;
    INT8 is sufficient.
  * **A_log (diagonal state matrix log):** Negative scalar per head; INT4
    covers the empirical range [-5, 0] without clipping.
  * **B, C projections:** Per-tensor dynamic range; INT4 with per-tensor scale.
  * **conv1d weights:** INT4 (short kernel, low rank).
  * **Recurrent hidden states:** Kept at FP16 (not quantised) because they
    accumulate over O(1) steps and precision loss compounds.

Calibration is performed by recording the empirical min/max of each tensor
over a small calibration batch (default 64 samples) and then computing the
optimal scale/zero-point for the desired bit width.

Reference: Lin et al., "QuaSSM: Efficient Quantization of State Space Models
for Language Modeling" arXiv 2408.09871, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "SSMQuantConfig",
    "SSMQuantState",
    "SSMQuantizer",
]

# Supported tensor roles and their default bit widths
_DEFAULT_BITS: Dict[str, int] = {
    "dt": 8,
    "A_log": 4,
    "B": 4,
    "C": 4,
    "conv1d": 4,
    "state": 16,  # FP16 — no quantisation
}


@dataclass
class SSMQuantConfig:
    """Configuration for :class:`SSMQuantizer`.

    Attributes:
        bits_per_role: Mapping from tensor role name to bit width.  Override
            to change the default precision for individual tensor types.
        calibration_samples: Number of batches to collect during calibration.
        symmetric: Use symmetric (zero-point = 0) quantisation.
        seed: RNG seed.
    """

    bits_per_role: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_BITS)
    )
    calibration_samples: int = 64
    symmetric: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        for role, bits in self.bits_per_role.items():
            if bits not in {1, 2, 4, 8, 16}:
                raise ValueError(
                    f"bits for role '{role}' must be in {{1,2,4,8,16}}, got {bits}"
                )
        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be ≥ 1, got {self.calibration_samples}"
            )


@dataclass
class SSMQuantState:
    """Per-tensor calibration state.

    Attributes:
        scales: Mapping from tensor role → learned scale factor.
        zero_points: Mapping from tensor role → zero-point (0 for symmetric).
        n_calibration_steps: Number of calibration batches seen.
    """

    scales: Dict[str, float] = field(default_factory=dict)
    zero_points: Dict[str, float] = field(default_factory=dict)
    n_calibration_steps: int = 0

    @property
    def is_calibrated(self) -> bool:
        return len(self.scales) > 0

    @property
    def calibrated_roles(self) -> list:
        return list(self.scales.keys())


class SSMQuantizer:
    """Calibrate-and-quantise SSM parameter tensors.

    Usage::

        cfg = SSMQuantConfig()
        quant = SSMQuantizer(cfg)
        cstate = quant.new_state()

        # Calibration phase
        for batch in calibration_batches:
            quant.observe(batch, cstate)
        quant.finalise(cstate)

        # Quantise a Δ tensor
        dt_quant, meta = quant.quantize_tensor(dt_weights, "dt", cstate)
        # Dequantise
        dt_fp32 = quant.dequantize_tensor(dt_quant, meta)
    """

    def __init__(self, config: SSMQuantConfig) -> None:
        self.config = config
        self._min_acc: Dict[str, float] = {}
        self._max_acc: Dict[str, float] = {}
        self._n_obs: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> SSMQuantState:
        return SSMQuantState()

    def observe(
        self,
        tensors: Dict[str, np.ndarray],
        state: SSMQuantState,
    ) -> SSMQuantState:
        """Accumulate min/max statistics from a calibration batch.

        Args:
            tensors: Dict mapping role names to float32 arrays.
            state: Calibration state (mutated in-place; also returned).

        Returns:
            Updated ``SSMQuantState``.
        """
        for role, arr in tensors.items():
            arr = np.asarray(arr, dtype=np.float32)
            mn, mx = float(arr.min()), float(arr.max())
            if role not in self._min_acc:
                self._min_acc[role] = mn
                self._max_acc[role] = mx
                self._n_obs[role] = 1
            else:
                self._min_acc[role] = min(self._min_acc[role], mn)
                self._max_acc[role] = max(self._max_acc[role], mx)
                self._n_obs[role] += 1
        state.n_calibration_steps += 1
        return state

    def finalise(self, state: SSMQuantState) -> SSMQuantState:
        """Compute scales and zero points from accumulated statistics."""
        for role, mn in self._min_acc.items():
            mx = self._max_acc[role]
            bits = self.config.bits_per_role.get(role, 8)
            if bits == 16:
                state.scales[role] = 1.0
                state.zero_points[role] = 0.0
                continue
            n_levels = (1 << bits) - 1
            if self.config.symmetric:
                abs_max = max(abs(mn), abs(mx), 1e-8)
                half = n_levels // 2
                scale = abs_max / half
                state.scales[role] = scale
                state.zero_points[role] = 0.0
            else:
                scale = (mx - mn) / n_levels if mx != mn else 1.0
                zero = -round(mn / scale)
                state.scales[role] = scale
                state.zero_points[role] = float(zero)
        return state

    def quantize_tensor(
        self,
        arr: np.ndarray,
        role: str,
        state: SSMQuantState,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Quantise a tensor given calibrated state.

        Args:
            arr: Float32 tensor to quantise.
            role: Tensor role key.
            state: Calibrated state.

        Returns:
            ``(quantised_array, meta)`` where meta contains scale/zero-point.
        """
        if not state.is_calibrated:
            raise RuntimeError("SSMQuantizer must be finalised before quantize_tensor")
        arr = np.asarray(arr, dtype=np.float32)
        bits = self.config.bits_per_role.get(role, 8)
        if bits == 16:
            return arr.astype(np.float16), {"scale": 1.0, "zero": 0.0, "bits": 16}

        scale = state.scales.get(role, 1.0)
        zero = state.zero_points.get(role, 0.0)
        n_levels = (1 << bits) - 1
        half = n_levels // 2
        q = np.round(arr / scale + zero).clip(-half, half).astype(np.int8)
        return q, {"scale": scale, "zero": zero, "bits": bits}

    def dequantize_tensor(
        self,
        q: np.ndarray,
        meta: Dict[str, float],
    ) -> np.ndarray:
        """Dequantise a previously quantised tensor.

        Args:
            q: Quantised int8 array.
            meta: Metadata dict from :meth:`quantize_tensor`.

        Returns:
            Float32 reconstructed array.
        """
        bits = int(meta.get("bits", 8))
        if bits == 16:
            return np.asarray(q, dtype=np.float32)
        scale = float(meta["scale"])
        zero = float(meta["zero"])
        return (q.astype(np.float32) - zero) * scale

    def compression_ratio(
        self, role: str, original_bits: int = 32
    ) -> float:
        """Theoretical compression ratio for a given role."""
        bits = self.config.bits_per_role.get(role, 8)
        return original_bits / max(bits, 1)
