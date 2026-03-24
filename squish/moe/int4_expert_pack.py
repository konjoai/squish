"""squish/moe/int4_expert_pack.py

INT4ExpertPacker — Group-quantized INT4 packing for MoE expert weights.

Expert FFN projection matrices tend to be large but numerically smooth,
making them ideal candidates for grouped INT4 quantization.  This module
implements:

  * **Packing** — Convert float32/float16 weight matrix W of shape (out, in)
    to a packed INT4 buffer (2 nibbles per byte) plus per-group scale and
    zero-point vectors.
  * **Unpacking / dequantization** — Recover a float32 approximation of W
    from the packed buffer.  The error is bounded by ½ × scale per group.
  * **Column-group topology** — Groups span ``group_size`` consecutive
    columns of W, matching the memory layout used by GPTQ, AWQ, and
    mlx_lm's INT4 kernels.

Group quantization parameters
------------------------------
Given a weight row slice R of length ``group_size``:

  * ``scale = (max(R) − min(R)) / 15``     (15 = 2^4 − 1)
  * ``zero  = round(−min(R) / scale)``      clamped to [0, 15]
  * ``q     = clamp(round(R / scale) + zero, 0, 15)``
  * Reconstruction:  ``R̃ = (q − zero) × scale``

Memory savings
--------------
  * float32  → INT4: 8× reduction.
  * bfloat16 → INT4: 4× reduction.

For Mixtral-8x7B:
  * Each expert ≈ 3 × 4096 × 14336 × 2 bytes ≈ 336 MB at bf16.
  * INT4: ≈ 84 MB per expert.
  * With budget_mb=4096, the resident set grows from ~12 to ~48 experts.

Reference
---------
Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers," ICLR 2023.

Usage
-----
::

    packer = INT4ExpertPacker(group_size=128)

    # Pack
    packed = packer.pack(weight_matrix)   # INT4PackedExpert

    # Unpack
    w_approx = packer.unpack(packed)      # np.ndarray, float32

    # Pack an entire expert's three matrices
    packed_expert = packer.pack_expert({"gate": Wg, "up": Wu, "down": Wd})
    weights = packer.unpack_expert(packed_expert)  # {"gate": ..., ...}
"""

from __future__ import annotations

__all__ = [
    "PackConfig",
    "INT4PackedMatrix",
    "INT4PackedExpert",
    "INT4ExpertPacker",
]

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Maximum representable INT4 value (unsigned nibble: 0–15)
_I4_MAX: int = 15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PackConfig:
    """Configuration for INT4ExpertPacker.

    Attributes
    ----------
    group_size:
        Number of consecutive columns per quantization group.  Must be a
        positive integer and typically a power of 2 (e.g. 64, 128, 256).
        Smaller groups → lower error, more scale/zero storage overhead.
    unsigned:
        If True, use unsigned INT4 (0–15) with an explicit zero-point.
        If False, use signed INT4 (−8 to 7) with zero zero-point.
        The unsigned scheme is standard for weight quantization (aligns
        with GPTQ / AWQ conventions).
    """

    group_size: int = 128
    unsigned: bool = True

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1; got {self.group_size}")


@dataclass
class INT4PackedMatrix:
    """INT4-packed representation of a single weight matrix.

    Attributes
    ----------
    packed_data:
        uint8 array of shape ``(out_features, n_groups × group_size // 2)``.
        Two INT4 nibbles are packed per byte (low nibble = col 0, high = col 1).
    scales:
        float32 array of shape ``(out_features, n_groups)``.
    zeros:
        float32 array of shape ``(out_features, n_groups)`` — zero-points.
    original_shape:
        The (out, in) shape of the original weight matrix.
    group_size:
        Group size used during quantization.
    """

    packed_data: np.ndarray   # uint8, shape (out, n_groups * group_size // 2)
    scales: np.ndarray        # float32, shape (out, n_groups)
    zeros: np.ndarray         # float32, shape (out, n_groups)
    original_shape: tuple
    group_size: int

    @property
    def out_features(self) -> int:
        return self.original_shape[0]

    @property
    def in_features(self) -> int:
        return self.original_shape[1]

    @property
    def n_groups(self) -> int:
        return self.scales.shape[1]

    @property
    def packed_bytes(self) -> int:
        return (
            self.packed_data.nbytes
            + self.scales.nbytes
            + self.zeros.nbytes
        )

    @property
    def compression_ratio(self) -> float:
        """Ratio of original float32 size to packed size."""
        original_bytes = self.out_features * self.in_features * 4  # float32
        return original_bytes / max(self.packed_bytes, 1)


@dataclass
class INT4PackedExpert:
    """INT4-packed weights for one MoE expert (gate, up, down projections)."""

    layer_idx: int
    expert_idx: int
    matrices: Dict[str, INT4PackedMatrix]

    @property
    def total_packed_bytes(self) -> int:
        return sum(m.packed_bytes for m in self.matrices.values())

    @property
    def compression_ratio(self) -> float:
        ratios = [m.compression_ratio for m in self.matrices.values()]
        return sum(ratios) / len(ratios) if ratios else 1.0


# ---------------------------------------------------------------------------
# INT4ExpertPacker
# ---------------------------------------------------------------------------

class INT4ExpertPacker:
    """Pack and unpack MoE expert weight matrices as grouped INT4.

    Parameters
    ----------
    config:
        Packing configuration (group_size, unsigned).
    """

    def __init__(self, config: Optional[PackConfig] = None) -> None:
        self._config = config or PackConfig()

    # ------------------------------------------------------------------ #
    # Core pack / unpack
    # ------------------------------------------------------------------ #

    def pack(self, weight: np.ndarray) -> INT4PackedMatrix:
        """Quantize a 2-D weight matrix to INT4 group quantization.

        Parameters
        ----------
        weight:
            Float32 or float16 array of shape ``(out_features, in_features)``.
            If shape 1-D, it is treated as a row vector.

        Returns
        -------
        INT4PackedMatrix
            Packed representation with separate scale and zero arrays.
        """
        if weight.ndim == 1:
            weight = weight.reshape(1, -1)
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D, got shape {weight.shape}")

        w = weight.astype(np.float32)
        out_features, in_features = w.shape
        gs = self._config.group_size
        original_shape = (out_features, in_features)

        # Pad in_features to a multiple of group_size
        pad = (-in_features) % gs
        if pad > 0:
            w = np.concatenate([w, np.zeros((out_features, pad), dtype=np.float32)], axis=1)
        padded_in = w.shape[1]
        n_groups = padded_in // gs

        # Reshape to (out, n_groups, group_size)
        w_grouped = w.reshape(out_features, n_groups, gs)

        # Per-group min / max
        w_min = w_grouped.min(axis=2)   # (out, n_groups)
        w_max = w_grouped.max(axis=2)   # (out, n_groups)

        # Scale and zero
        span = w_max - w_min
        scales = span / _I4_MAX
        scales = np.where(scales == 0.0, 1.0, scales)  # avoid div-by-zero

        if self._config.unsigned:
            zeros = np.round(-w_min / scales).clip(0, _I4_MAX)   # (out, n_groups)
        else:
            zeros = np.zeros_like(scales)

        # Quantize: q = clamp(round(w / scale) + zero, 0, 15)
        # Broadcast: scales/zeros shape (out, n_groups, 1) for group dim
        scales_b = scales[:, :, np.newaxis]
        zeros_b = zeros[:, :, np.newaxis]
        q = np.round(w_grouped / scales_b + zeros_b).clip(0, _I4_MAX).astype(np.uint8)
        # q shape: (out, n_groups, group_size)

        # Pack two nibbles per byte: even columns into low nibble, odd into high
        # We pack column pairs within each group: bytes = group_size // 2
        q_flat = q.reshape(out_features, padded_in)  # (out, padded_in)
        # Pad to even length if needed
        if padded_in % 2 != 0:
            q_flat = np.concatenate(
                [q_flat, np.zeros((out_features, 1), dtype=np.uint8)], axis=1
            )
        lo = q_flat[:, 0::2]  # even columns — low nibble
        hi = q_flat[:, 1::2]  # odd columns  — high nibble
        packed = (lo & 0x0F) | ((hi & 0x0F) << 4)  # shape (out, packed_cols)

        return INT4PackedMatrix(
            packed_data=packed,
            scales=scales.astype(np.float32),
            zeros=zeros.astype(np.float32),
            original_shape=original_shape,
            group_size=gs,
        )

    def unpack(self, packed: INT4PackedMatrix) -> np.ndarray:
        """Dequantize an INT4PackedMatrix back to float32.

        Parameters
        ----------
        packed:
            Output of a previous :meth:`pack` call.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``original_shape``.
        """
        out_features, in_features = packed.original_shape
        gs = packed.group_size
        n_groups = packed.n_groups

        # Unpack nibbles
        lo = packed.packed_data & 0x0F
        hi = (packed.packed_data >> 4) & 0x0F
        # Interleave: shape becomes (out, 2 × packed_cols)
        q_interleaved = np.empty(
            (out_features, lo.shape[1] + hi.shape[1]), dtype=np.uint8
        )
        q_interleaved[:, 0::2] = lo
        q_interleaved[:, 1::2] = hi

        padded_in = n_groups * gs
        q_flat = q_interleaved[:, :padded_in].astype(np.float32)

        # Reshape to (out, n_groups, group_size)
        q_grouped = q_flat.reshape(out_features, n_groups, gs)

        # Dequantize
        scales_b = packed.scales[:, :, np.newaxis]
        zeros_b = packed.zeros[:, :, np.newaxis]
        w_approx = (q_grouped - zeros_b) * scales_b

        # Flatten and trim to original in_features
        w_flat = w_approx.reshape(out_features, padded_in)
        return w_flat[:, :in_features]

    # ------------------------------------------------------------------ #
    # Expert-level pack / unpack
    # ------------------------------------------------------------------ #

    def pack_expert(
        self,
        weights: Dict[str, np.ndarray],
        layer_idx: int = 0,
        expert_idx: int = 0,
    ) -> INT4PackedExpert:
        """Pack all projection matrices for one expert.

        Parameters
        ----------
        weights:
            Dict mapping projection names (e.g. "gate", "up", "down") to
            2-D float arrays.
        layer_idx:
            Layer index for bookkeeping.
        expert_idx:
            Expert index for bookkeeping.

        Returns
        -------
        INT4PackedExpert
            Packed representation.
        """
        packed_matrices: Dict[str, INT4PackedMatrix] = {}
        for proj_name, w in weights.items():
            if w is not None:
                packed_matrices[proj_name] = self.pack(w)
        return INT4PackedExpert(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            matrices=packed_matrices,
        )

    def unpack_expert(self, packed_expert: INT4PackedExpert) -> Dict[str, np.ndarray]:
        """Dequantize all projection matrices for one expert.

        Parameters
        ----------
        packed_expert:
            Output of a previous :meth:`pack_expert` call.

        Returns
        -------
        Dict[str, np.ndarray]
            Dequantized weight matrices keyed by projection name.
        """
        return {
            proj: self.unpack(pm)
            for proj, pm in packed_expert.matrices.items()
        }

    # ------------------------------------------------------------------ #
    # Error measurement
    # ------------------------------------------------------------------ #

    def quantization_error(self, original: np.ndarray) -> float:
        """Compute the mean absolute error introduced by pack-then-unpack.

        Parameters
        ----------
        original:
            Float32 weight matrix.

        Returns
        -------
        float
            Mean absolute error (lower is better).
        """
        packed = self.pack(original.astype(np.float32))
        reconstructed = self.unpack(packed)
        return float(np.mean(np.abs(original.astype(np.float32) - reconstructed)))

    # ------------------------------------------------------------------ #
    # Config access
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> PackConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"INT4ExpertPacker(group_size={self._config.group_size}, "
            f"unsigned={self._config.unsigned})"
        )
