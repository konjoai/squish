"""INT3Linear — MLX module that keeps INT3 weights as uint8 in Metal.

Stores 3-bit asymmetric codes (one code per uint8 byte, values 0–7) and
per-group float16 scales / zeros.  The dequantization:

    w_dq[out_i, k] = scales[out_i, k // gs] * codes[out_i, k]
                   + zeros[out_i, k // gs]          (per group, gs = group_size)

is fused with the GEMV/GEMM in the MLX JIT graph — no BF16 staging buffer
is ever written to Metal cache; the codes live as uint8 until the metal
shader for the matmul reads them.

Memory footprint vs BF16 (1-byte codes, 2-byte BF16):
    1.5B model codes ≈ 750 MB  (vs ~1.5 GB BF16 at 1 byte/weight stored as
                                 uint8 vs 2 bytes BF16 × ~750M linear weights)
    + scales/zeros ≈ (~1.5B // group_size) × 4 bytes ≈ 45 MB for gs=64
    Total ≈ ~800 MB vs ~3 GB BF16 → ~3.75× reduction in Metal resident set

API:
    INT3Linear is a drop-in for mlx.nn.Linear.  The weight attribute stores
    uint8 codes, not float weights.  Callers that inspect dtype should expect
    mx.uint8 for weight.  The __call__ signature is identical to nn.Linear.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = ["INT3Linear"]


class INT3Linear(nn.Module):
    """INT3 asymmetric quantized linear layer for MLX.

    Weights are stored as uint8 (one 3-bit code per byte, values 0–7) with
    per-group float16 scales and zeros.  Dequantization is fused into the
    MLX JIT graph — weights stay compact in Metal unified memory and are
    never expanded to a full BF16 matrix.

    Args:
        weight: uint8 array, shape ``(out_features, in_features)``.
            Each element is a 3-bit code in ``[0, 7]`` stored in a byte.
        scales: float16 array, shape ``(out_features, n_groups)`` where
            ``n_groups = in_features // group_size``.
        zeros:  float16 array, shape ``(out_features, n_groups)``.
        bias:   optional array, shape ``(out_features,)``.  If provided it
            must be broadcast-compatible with the matmul output.

    Raises:
        TypeError: if ``weight.dtype`` is not ``mx.uint8``.
        ValueError: if shape constraints (n_groups, divisibility) are violated.

    Notes:
        Dequantization formula (per-group asymmetric):
            ``w_dq = scales * codes + zeros``
        This is the same convention used by ``squish.quant.int3_runtime``:
        ``codes`` are unsigned 0–7 (NOT signed –3 to +3), so the ``zeros``
        term absorbs the bias rather than a symmetric pivot.
    """

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        zeros: mx.array,
        bias: Optional[mx.array] = None,
    ) -> None:
        super().__init__()
        if weight.dtype != mx.uint8:
            raise TypeError(
                f"INT3Linear weight must be uint8, got {weight.dtype}.  "
                "Pass the raw code array from __q3.npy, not dequantized floats."
            )
        if weight.ndim != 2:
            raise ValueError(
                f"INT3Linear weight must be 2-D (out, in), got shape {weight.shape}"
            )
        n_out, n_in = weight.shape
        if scales.shape != zeros.shape:
            raise ValueError(
                f"scales and zeros must have matching shapes, "
                f"got {scales.shape} vs {zeros.shape}"
            )
        if scales.ndim != 2 or scales.shape[0] != n_out:
            raise ValueError(
                f"scales must be 2-D (out_features, n_groups), got {scales.shape}"
            )
        n_groups = scales.shape[1]
        if n_in % n_groups != 0:
            raise ValueError(
                f"in_features ({n_in}) must be divisible by n_groups ({n_groups})"
            )

        self.weight = weight   # (out, in) uint8
        self.scales = scales   # (out, n_groups) float16
        self.zeros  = zeros    # (out, n_groups) float16
        if bias is not None:
            self.bias = bias

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def group_size(self) -> int:
        return self.in_features // int(self.scales.shape[1])

    # ── Forward ────────────────────────────────────────────────────────────────

    def __call__(self, x: mx.array) -> mx.array:
        """Dequantize weights and compute x @ W.T (+ bias).

        The dequantize + matmul chain is fused by the MLX JIT — Metal only
        ever materialises the output vector, not the full BF16 weight matrix.

        Args:
            x: input array, shape ``(..., in_features)``.

        Returns:
            Output array, shape ``(..., out_features)``, same dtype as ``x``.
        """
        n_out, n_in = self.weight.shape
        gs = n_in // self.scales.shape[1]

        # Dequantize: w_dq[i, k] = scales[i, k//gs] * codes[i, k] + zeros[i, k//gs]
        # Reshape for group-wise broadcast multiplication:
        #   codes  (n_out, n_groups, gs)  →  scale/zero broadcast → (n_out, n_groups, gs)
        w = self.weight.reshape(n_out, -1, gs).astype(mx.bfloat16)  # (n_out, n_groups, gs)
        s = self.scales[:, :, None].astype(mx.bfloat16)             # (n_out, n_groups,  1)
        z = self.zeros[:, :, None].astype(mx.bfloat16)              # (n_out, n_groups,  1)
        w_dq = (w * s + z).reshape(n_out, n_in)                     # (n_out, n_in)  bfloat16

        y = x @ w_dq.T
        if hasattr(self, "bias"):
            y = y + self.bias
        return y

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        bias_str = "bias" if hasattr(self, "bias") else "no bias"
        return (
            f"INT3Linear("
            f"in={self.in_features}, out={self.out_features}, "
            f"gs={self.group_size}, {bias_str})"
        )
