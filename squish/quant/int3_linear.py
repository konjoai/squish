"""INT3Linear — MLX module using mx.quantized_matmul for in-Metal 3-bit inference.

Accepts uint8 codes (one 3-bit code per byte, values 0–7) plus per-group
float16 scales / zeros at construction time.

Fast path (group_size ∈ {32, 64, 128}):
    Converts to MLX's native uint32 bit-packed format with MLX-convention biases.
    Every forward call uses ``mx.quantized_matmul`` — the BF16 weight matrix is
    NEVER materialised in Metal unified memory at inference time.

Fallback path (group_size < 32 — only used for tiny test tensors):
    Keeps uint8 codes and uses BF16 dequant+matmul.  Production models always
    use group_size=64 so this path is never taken at inference time.

Squish decode convention (per-group asymmetric):
    w_dq[i, k] = codes[i, k] * scales[i, k//gs] + zeros[i, k//gs]

MLX quantized_matmul decode convention:
    w_dq[i, k] = scales[i, k//gs] * codes[i, k] + biases[i, k//gs]

Equivalence: biases = zeros (same zero-point, same scales, same codes)

Memory footprint (1.5B model, gs=64):
    uint32 packed codes ≈ 750 MB × (3/8) = ~281 MB  (3 bits/weight, uint32-packed)
    scales + biases     ≈ 45 MB  (float16, same as before)
    Total ≈ ~326 MB vs ~3 GB BF16 → ~9× reduction; no per-token spike.

API:
    INT3Linear is a drop-in for mlx.nn.Linear.  The ``weight`` attribute stores
    either MLX-native uint32 packed codes (fast path) or raw uint8 codes (fallback).
    The ``__call__`` signature is identical to nn.Linear.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = ["INT3Linear"]

# Group sizes supported by mx.quantized_matmul Metal kernels for bits=3.
_MLX_SUPPORTED_GS: frozenset[int] = frozenset({32, 64, 128})


def _pack_codes_uint32(codes_np, bits: int = 3):
    """Pack ``(n_out, n_in)`` uint8 codes into ``(n_out, n_packed)`` uint32.

    MLX bit-stream layout: element *i* occupies bits ``[i*bits, i*bits+bits)``
    of the packed bit-stream, packed LSB-first with cross-word boundary support.
    Identical pack order to ``mx.quantize`` internals.

    Fully vectorised — no Python loop over n_in. Eliminating the per-column
    loop drops first-load time on a 1.5B model from ~1.75M iterations to a
    single numpy scatter-or operation.

    Args:
        codes_np: uint8 array of shape ``(n_out, n_in)`` with values in
            ``[0, 2**bits - 1]``.
        bits:     number of bits per element (default 3).

    Returns:
        uint32 array of shape ``(n_out, n_packed)`` where
        ``n_packed = ceil(n_in * bits / 32)``.
    """
    import numpy as np  # deferred: only needed at load time, not inference
    n_out, n_in = codes_np.shape
    n_packed = (n_in * bits + 31) // 32
    packed = np.zeros((n_out, n_packed), dtype=np.uint32)
    # Vectorised scatter: compute word index and bit offset for every column
    # simultaneously, then accumulate with np.add.at for overflow columns.
    cols    = np.arange(n_in, dtype=np.int64)
    bit_pos = cols * bits                         # (n_in,)
    w       = (bit_pos >> 5).astype(np.int64)     # word index per column
    b       = (bit_pos & 31).astype(np.uint32)    # bit offset within word
    codes_u32 = codes_np.astype(np.uint32)        # (n_out, n_in)
    mask    = np.uint32((1 << bits) - 1)
    # Main contribution: shift each column's codes into position and OR into packed.
    # packed[:, w[i]] |= (codes_u32[:, i] & mask) << b[i]  — vectorised via
    # transposing to (n_in, n_out), scaling by shift, then scatter-adding.
    shifted = (codes_u32 & mask) << b           # (n_out, n_in) — broadcast b
    np.add.at(packed.T, w, shifted.T)           # scatter-add into correct words
    # Overflow contribution: bits that spill past the 32-bit word boundary.
    overflow = b.astype(np.int64) + bits - 32   # (n_in,) — negative when no spill
    spill_mask = overflow > 0
    if spill_mask.any():
        sp_cols   = cols[spill_mask]            # column indices with overflow
        sp_ov     = overflow[spill_mask].astype(np.uint32)
        sp_w1     = w[spill_mask] + 1           # next word index
        sp_shift  = np.uint32(bits) - sp_ov    # right-shift amount
        sp_vals   = codes_u32[:, sp_cols] >> sp_shift  # (n_out, n_spill)
        np.add.at(packed.T, sp_w1, sp_vals.T)
    return packed


class INT3Linear(nn.Module):
    """INT3 asymmetric quantized linear layer — zero-BF16 inference via mx.quantized_matmul.

    At construction time the uint8 codes are repacked into MLX's native uint32
    bit-packed format (when group_size ∈ {32, 64, 128}) and zeros are converted to
    MLX biases.  Every forward call delegates to ``mx.quantized_matmul`` which
    dequantizes inside the Metal shader — the full BF16 weight matrix is NEVER
    materialised at inference time.

    For non-standard group sizes (gs < 32, used only in tiny test tensors), the
    module falls back to BF16 dequant+matmul to preserve correctness.

    Args:
        weight: uint8 array, shape ``(out_features, in_features)``.
            Each element is a 3-bit code in ``[0, 7]`` stored in a uint8 byte.
        scales: float16 array, shape ``(out_features, n_groups)`` where
            ``n_groups = in_features // group_size``.
        zeros:  float16 array, shape ``(out_features, n_groups)``.
        bias:   optional array, shape ``(out_features,)``.

    Raises:
        TypeError: if ``weight.dtype`` is not ``mx.uint8``.
        ValueError: if shape constraints (n_groups, divisibility) are violated.

    Notes:
        Fast path (gs ∈ {32, 64, 128}): ``weight`` is stored as uint32 packed,
        ``scales`` is float16, ``biases`` is float16 (= squish zeros, the zero-point
        offset).  MLX decode: ``w = scale * code + biases``, identical to squish
        decode: ``w = code * scale + zeros``.
        Fallback path (gs < 32): ``weight`` is uint8, ``scales`` and ``zeros``
        are float16.
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

        gs = n_in // n_groups
        self._n_out = n_out
        self._n_in  = n_in
        self._gs    = gs

        if gs in _MLX_SUPPORTED_GS:
            # ── Fast path: pack to MLX uint32 + map zeros → MLX biases ──────
            # squish: w = codes * scale + zeros
            # MLX:    w = scale * code + biases  (biases IS the squish zero-point)
            # Equivalence: mlx_scales = squish_scales, mlx_biases = squish_zeros
            # Codes are packed bit-for-bit identically in both conventions.
            import numpy as np  # deferred: load-time only, not inference
            codes_np  = np.array(weight, dtype=np.uint8)
            scales_np = np.array(scales, dtype=np.float32)
            zeros_np  = np.array(zeros,  dtype=np.float32)

            packed_np = _pack_codes_uint32(codes_np, bits=3)

            self.weight = mx.array(packed_np)                       # uint32 packed
            self.scales = mx.array(scales_np.astype(np.float16))    # float16
            self.biases = mx.array(zeros_np.astype(np.float16))     # float16 (= squish zeros)
            self._fast  = True
        else:
            # ── Fallback path: keep uint8, use BF16 dequant+matmul ───────────
            # Only reached for non-standard group sizes (gs=8, gs=16) which are
            # used in small test tensors but never in production models.
            self.weight = weight                           # uint8
            self.scales = scales.astype(mx.float16)       # float16
            self.zeros  = zeros.astype(mx.float16)        # float16
            self._fast  = False

        if bias is not None:
            self.bias = bias

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def out_features(self) -> int:
        return self._n_out

    @property
    def in_features(self) -> int:
        return self._n_in

    @property
    def group_size(self) -> int:
        return self._gs

    # ── Forward ────────────────────────────────────────────────────────────────

    def __call__(self, x: mx.array) -> mx.array:
        """Compute x @ W.T (+ bias).

        Fast path (gs ∈ {32, 64, 128}): dequantizes inside the Metal shader via
        ``mx.quantized_matmul`` — zero BF16 allocation at inference time.

        Fallback path (gs < 32): BF16 dequant+matmul (only for non-production
        tiny tensors where Metal kernel is unsupported).

        Args:
            x: input array, shape ``(..., in_features)``.

        Returns:
            Output array, shape ``(..., out_features)``.
        """
        if self._fast:
            y = mx.quantized_matmul(
                x,
                self.weight,
                scales=self.scales,
                biases=self.biases,
                transpose=True,
                group_size=self._gs,
                bits=3,
            )
        else:
            # BF16 fallback for non-standard group sizes (gs < 32).
            w = self.weight.reshape(self._n_out, -1, self._gs).astype(mx.bfloat16)
            s = self.scales[:, :, None].astype(mx.bfloat16)
            z = self.zeros[:, :, None].astype(mx.bfloat16)
            w_dq = (w * s + z).reshape(self._n_out, self._n_in)
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
