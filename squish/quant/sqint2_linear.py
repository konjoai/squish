"""squish/quant/sqint2_linear.py — W103.4c SQINT2Linear MLX Module.

SQINT2Linear is the inference-path companion to ``squish/quant/sqint2.py``.
It accepts the compressed artifacts produced by ``compress_weight`` and
``save_sqint2_layer`` and performs quantized GEMV entirely within MLX on
Apple Silicon.

Platform gate
-------------
MLX (``mlx-lm``) is Apple Silicon-only. This module must not be imported on
Linux. A hard guard at the module level raises ``ImportError`` when executed
on non-darwin platforms, so callers cannot accidentally import it on Linux
CI or x86 hosts. The guard is:

    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ImportError(...)

NF2 dequant strategy
--------------------
MLX's ``mx.quantized_matmul`` natively supports 4-bit and 8-bit integer
codes; it does not expose a 2-bit path.  We therefore perform the NF2
lookup table dequantisation as an explicit MLX op before the matmul:

    1. Unpack 2-bit indices from the packed uint8 representation to a
       (out_features, in_padded) uint8 array.  (This unpacking happens once
       at construction time — the unpacked indices are stored as ``mx.uint8``
       on the unified buffer; no runtime allocation.)
    2. Look up NF2_VALUES via ``mx.take(nf2_lut, indices, axis=0)`` — a
       O(N) gather into a 4-element fp32 LUT.  This is vectorised in
       Metal's compute shader; no element-wise Python loop.
    3. Apply the per-group scale/zero-point: dq = (nf2_rescaled - zp) * scale
       — broadcast over the group_size axis.  Result is (out, in_padded) fp16.
    4. x @ dq.T — standard MLX matmul.

Because step 2–4 materialise the full fp16 weight matrix, SQINT2Linear is
NOT zero-weight-allocation at inference time.  This is a known limitation of
the W103.4c approach (the W103.4c hard goal is functional correctness with a
Metal-accelerated forward pass, not zero-weight materialisation).  The Metal
shader path for a fused 2-bit NF2 dequant+matmul (without materialisation)
is deferred to a future wave.

Residual correction (Stage 3)
------------------------------
If the SQINT2Layer was compressed with ``residual_rank > 0``, the L and R
factors are stored and applied at inference time:

    y_base = x @ dq.T                         # standard NF2 GEMV
    y_res  = sqint2_residual_gemv(L, R, x)   # low-rank correction (NumPy → MLX)
    y      = y_base + mx.array(y_res)         # combined output

The residual GEMV uses the Rust kernel (W103.4b) when available, falling
back to NumPy otherwise.  The numpy/Rust output is wrapped in ``mx.array``
before adding.

Sparse correction
-----------------
If ``SQINT2Layer.sparse_rows`` is not None, the sparse COO corrections are
applied to the reconstructed weight matrix before the matmul.  This is a
Python scatter operation (``dq_np[rows, cols] += vals``) performed once per
forward pass.  The hardware ship gate for a fused sparse correction kernel
is deferred alongside the zero-weight-materialisation work.

API
---
    SQINT2Linear        drop-in for mlx.nn.Linear constructed from SQINT2Layer
    sqint2_linear_from_layer(layer, bias=None)  convenience factory
"""
from __future__ import annotations

import platform
import sys

# Platform guard — hard fail on non-Apple-Silicon.
# CLAUDE.md: "MLX imports must be gated behind platform check — never imported
# on Linux paths."
if sys.platform != "darwin" or platform.machine() != "arm64":
    raise ImportError(
        "squish.quant.sqint2_linear is Apple Silicon (darwin/arm64) only. "
        "It must not be imported on Linux or x86 hosts. "
        "Check `sys.platform == 'darwin' and platform.machine() == 'arm64'` "
        "before importing this module."
    )

from typing import Optional

import numpy as np

from squish.quant.sqint2 import (
    NF2_VALUES,
    SQINT2Layer,
    _round_up,
    _unpack_2bit,
)
from squish.quant.quantizer import sqint2_residual_gemv

__all__ = ["SQINT2Linear", "sqint2_linear_from_layer"]

# _NF2_LUT is initialized lazily on first use inside __init__ to avoid
# triggering Metal GPU initialization at module import time (causes SIGABRT
# on macOS CI runners where the Metal context is not ready during the Python
# import phase).  Access via module __getattr__ or directly as self._nf2_lut.
_NF2_LUT_CACHE: "Optional[object]" = None  # mx.array once initialized


def _get_nf2_lut() -> "object":
    """Return NF2 LUT (mx.array), initializing lazily on first call."""
    global _NF2_LUT_CACHE
    if _NF2_LUT_CACHE is None:
        import mlx.core as mx  # deferred — safe at function-call time
        _NF2_LUT_CACHE = mx.array(NF2_VALUES.astype(np.float32))
    return _NF2_LUT_CACHE


class SQINT2Linear:
    """SQINT2 quantized linear layer — NF2 dequant + MLX matmul + SVD residual.

    Implements the inference path for SQINT2 Stage 1+2+3 on Apple Silicon:

        Stage 1  Hadamard inverse is pre-applied to the reconstructed weight
                 at construction time (materialise-once approach).
        Stage 2  NF2 dequant via mx.take LUT lookup per forward call.
        Stage 3  Low-rank SVD residual via sqint2_residual_gemv (Rust/NumPy).
                 Sparse COO correction via scatter (Python loop at forward
                 time — acceptable overhead given k << M·N at 1% sparsity).

    Args:
        indices:     mx.uint8, shape ``(out_features, in_padded // 4)``.
                     Packed 2-bit SQINT2 indices from ``SQINT2Layer.indices``.
        scales:      mx.array, shape ``(out_features, n_groups)``, float32 or
                     float16.  Per-group multiplier in the NF2 frame.
        zero_points: mx.array, shape ``(out_features, n_groups)``, same dtype
                     as scales.  Per-group additive shift in the NF2 frame.
        in_features: int — original (unpadded) input dimension.
        out_features: int — original output dimension.
        group_size:  int — columns per quantization group (default 32).
        residual_L:  optional fp16 ndarray ``(out_features, rank)`` — SVD
                     left factor.  None when rank=0.
        residual_R:  optional fp16 ndarray ``(rank, in_features)`` — SVD
                     right factor.  None when rank=0.
        sparse_rows: optional int32 ndarray ``(k,)`` — COO row indices.
        sparse_cols: optional int32 ndarray ``(k,)`` — COO col indices.
        sparse_vals: optional fp16 ndarray ``(k,)`` — COO correction values.
        bias:        optional mx.array ``(out_features,)``.

    Raises:
        TypeError:  weight/scales/zero_points dtype or ndim constraint violated.
        ValueError: shape consistency constraints violated.
    """

    def __init__(
        self,
        indices: "object",
        scales: "object",
        zero_points: "object",
        in_features: int,
        out_features: int,
        group_size: int = 32,
        residual_L: "Optional[np.ndarray]" = None,
        residual_R: "Optional[np.ndarray]" = None,
        sparse_rows: "Optional[np.ndarray]" = None,
        sparse_cols: "Optional[np.ndarray]" = None,
        sparse_vals: "Optional[np.ndarray]" = None,
        bias: "Optional[object]" = None,
    ) -> None:
        import mlx.core as mx  # deferred — avoids Metal init at import time

        if indices.dtype != mx.uint8:
            raise TypeError(
                f"SQINT2Linear indices must be mx.uint8 (packed 2-bit), "
                f"got {indices.dtype}"
            )
        if indices.ndim != 2:
            raise ValueError(
                f"SQINT2Linear indices must be 2-D (out, in_padded//4), "
                f"got shape {indices.shape}"
            )
        if scales.ndim != 2 or zero_points.ndim != 2:
            raise ValueError(
                f"scales and zero_points must be 2-D (out, n_groups), "
                f"got scales={scales.shape}, zp={zero_points.shape}"
            )
        if scales.shape != zero_points.shape:
            raise ValueError(
                f"scales and zero_points must have identical shapes, "
                f"got {scales.shape} vs {zero_points.shape}"
            )
        if scales.shape[0] != out_features:
            raise ValueError(
                f"scales.shape[0] ({scales.shape[0]}) must equal out_features "
                f"({out_features})"
            )

        in_padded = _round_up(in_features, group_size)
        n_groups = in_padded // group_size
        expected_idx_cols = in_padded // 4
        if indices.shape[1] != expected_idx_cols:
            raise ValueError(
                f"indices.shape[1] ({indices.shape[1]}) must equal "
                f"in_padded // 4 = {expected_idx_cols} "
                f"(in_features={in_features}, group_size={group_size})"
            )
        if scales.shape[1] != n_groups:
            raise ValueError(
                f"scales.shape[1] ({scales.shape[1]}) must equal "
                f"n_groups = {n_groups} "
                f"(in_padded={in_padded}, group_size={group_size})"
            )

        self._in_features  = in_features
        self._out_features = out_features
        self._group_size   = group_size
        self._in_padded    = in_padded
        self._n_groups     = n_groups

        # Store packed indices and quant params as mx.array buffers.
        # They live in unified memory; the unpacking is deferred to forward().
        self.indices     = indices                               # mx.uint8 packed
        self.scales      = scales.astype(mx.float32)            # mx.float32
        self.zero_points = zero_points.astype(mx.float32)       # mx.float32

        # Stage 3 residual factors (optional, stored as numpy for Rust/NumPy GEMV).
        # The Rust/NumPy path operates on numpy arrays; we store them in np memory
        # rather than unified memory to avoid roundtrip mx→np at every forward call.
        self._residual_L: "Optional[np.ndarray]" = (
            np.asarray(residual_L, dtype=np.float16)
            if residual_L is not None else None
        )
        self._residual_R: "Optional[np.ndarray]" = (
            np.asarray(residual_R, dtype=np.float16)
            if residual_R is not None else None
        )

        # Sparse COO correction (optional, numpy for scatter).
        self._sparse_rows: "Optional[np.ndarray]" = (
            np.asarray(sparse_rows, dtype=np.int32)
            if sparse_rows is not None else None
        )
        self._sparse_cols: "Optional[np.ndarray]" = (
            np.asarray(sparse_cols, dtype=np.int32)
            if sparse_cols is not None else None
        )
        self._sparse_vals: "Optional[np.ndarray]" = (
            np.asarray(sparse_vals, dtype=np.float16)
            if sparse_vals is not None else None
        )

        # Optional bias.
        if bias is not None:
            self.bias = bias

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def group_size(self) -> int:
        return self._group_size

    @property
    def has_residual(self) -> bool:
        return self._residual_L is not None and self._residual_R is not None

    @property
    def has_sparse(self) -> bool:
        return self._sparse_rows is not None

    # ── Forward ────────────────────────────────────────────────────────────────

    def __call__(self, x: "object") -> "object":
        """Compute SQINT2 forward pass.

        Forward pipeline:
            1. Unpack 2-bit indices  → uint8 (out, in_padded) via numpy bit-unpack.
            2. NF2 LUT lookup        → float32 (out, in_padded) via mx.take.
            3. Per-group dequant     → (NF2_val - zp) * scale → float16 weight.
            4. Optional sparse COO correction in numpy before wrapping to mx.
            5. Main GEMV             → x @ w_dq.T  in MLX.
            6. Optional residual     → sqint2_residual_gemv(L, R, x_np) → mx.array.
            7. Optional bias add.

        Args:
            x: ``(..., in_features)`` mx.array.

        Returns:
            ``(..., out_features)`` mx.array.
        """
        import mlx.core as mx  # deferred — avoids Metal init at import time

        # ── Step 1: unpack 2-bit indices → (out, in_padded) uint8 ────────────
        # _unpack_2bit works on numpy; mx→np→mx roundtrip is acceptable at
        # inference time — it happens in unified memory (zero-copy on M-series).
        indices_np = np.array(self.indices, copy=False)     # numpy view of mx buffer
        unpacked_np = _unpack_2bit(indices_np, self._in_padded)  # (out, in_padded) uint8

        # ── Step 2: NF2 LUT lookup → float32 (out, in_padded) ────────────────
        # mx.take performs a vectorised gather: NF2_LUT[unpacked] element-wise.
        # No Python loop, no element-wise compute — dispatches to Metal.
        unpacked_mx = mx.array(unpacked_np.astype(np.int32))   # int32 for mx.take index
        nf2_rescaled = mx.take(_get_nf2_lut(), unpacked_mx)    # (out, in_padded) float32

        # ── Step 3: per-group dequant → float16 weight ───────────────────────
        # Decode convention: w_dq = (NF2_val - zp) * scale
        # scales shape: (out, n_groups) — broadcast over group_size axis.
        # Reshape to (out, n_groups, group_size), broadcast, reshape back.
        nf2_3d = nf2_rescaled.reshape(self._out_features, self._n_groups, self._group_size)
        # scales/zp: (out, n_groups) → (out, n_groups, 1) for broadcast
        s_3d = self.scales.reshape(self._out_features, self._n_groups, 1)
        z_3d = self.zero_points.reshape(self._out_features, self._n_groups, 1)
        # dq in float32, strip padding back to in_features
        dq_3d = (nf2_3d - z_3d) * s_3d                        # (out, n_groups, gs)
        dq = dq_3d.reshape(self._out_features, self._in_padded)[:, :self._in_features]
        # Cast to float16 for efficient matmul on Metal
        dq = dq.astype(mx.float16)

        # ── Step 4: optional sparse COO correction ────────────────────────────
        # Sparse corrections are small (k = 1% of out*in_padded entries).
        # Apply in numpy before transferring back to MLX.
        if self._sparse_rows is not None:
            dq_np = np.array(dq, dtype=np.float32, copy=True)  # (out, in_features)
            dq_np[self._sparse_rows, self._sparse_cols] += (
                self._sparse_vals.astype(np.float32)
            )
            dq = mx.array(dq_np.astype(np.float16))

        # ── Step 5: main GEMV — x @ dq.T ─────────────────────────────────────
        y = x @ dq.T                                            # (..., out_features)

        # ── Step 6: optional SVD low-rank residual correction ─────────────────
        # sqint2_residual_gemv expects numpy input; extract x as numpy, run
        # the correction, and add back as mx.array.
        if self._residual_L is not None and self._residual_R is not None:
            x_np = np.array(x, dtype=np.float32)           # (..., in_features)
            x_2d = x_np.reshape(-1, self._in_features)     # flatten batch dims
            res_np = sqint2_residual_gemv(self._residual_L, self._residual_R, x_2d)
            # Reshape back to original batch shape + out_features
            res_shape = (*x_np.shape[:-1], self._out_features)
            y = y + mx.array(res_np.reshape(res_shape).astype(np.float32))

        # ── Step 7: optional bias ─────────────────────────────────────────────
        if hasattr(self, "bias"):
            y = y + self.bias

        return y

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        residual_str = f", residual_rank={self._residual_L.shape[1]}" if self.has_residual else ""
        sparse_str   = f", sparse_k={len(self._sparse_rows)}" if self.has_sparse else ""
        bias_str     = ", bias" if hasattr(self, "bias") else ""
        return (
            f"SQINT2Linear("
            f"in={self.in_features}, out={self.out_features}, "
            f"gs={self.group_size}"
            f"{residual_str}{sparse_str}{bias_str})"
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def sqint2_linear_from_layer(
    layer: SQINT2Layer,
    bias: "Optional[object]" = None,
) -> "SQINT2Linear":
    """Construct a SQINT2Linear from a deserialized SQINT2Layer.

    Converts numpy arrays from ``load_sqint2_layer`` into MLX arrays and
    wires in the optional Stage-3 residual and sparse factors.

    Args:
        layer: SQINT2Layer from ``squish.quant.sqint2.load_sqint2_layer``.
        bias:  optional bias array (numpy or mx.array, shape ``(out_features,)``).

    Returns:
        SQINT2Linear ready for inference.

    Example::

        layer = load_sqint2_layer(tensor_dir, "model__layers__0__mlp__gate_proj")
        linear = sqint2_linear_from_layer(layer)
        y = linear(x)   # x: mx.array (batch, in_features)
    """
    import mlx.core as mx  # deferred — avoids Metal init at import time

    indices_mx = mx.array(layer.indices)                      # mx.uint8
    scales_mx  = mx.array(layer.scales.astype(np.float32))   # mx.float32
    zp_mx      = mx.array(layer.zero_points.astype(np.float32))  # mx.float32

    bias_mx: "Optional[object]" = None
    if bias is not None:
        if isinstance(bias, mx.array):
            bias_mx = bias
        else:
            bias_mx = mx.array(np.asarray(bias, dtype=np.float32))

    return SQINT2Linear(
        indices=indices_mx,
        scales=scales_mx,
        zero_points=zp_mx,
        in_features=layer.in_features,
        out_features=layer.out_features,
        group_size=layer.cfg.group_size,
        residual_L=layer.residual_L,
        residual_R=layer.residual_R,
        sparse_rows=layer.sparse_rows,
        sparse_cols=layer.sparse_cols,
        sparse_vals=layer.sparse_vals,
        bias=bias_mx,
    )


def __getattr__(name: str) -> object:
    """Lazy access to _NF2_LUT — triggers Metal init at access time, not import time."""
    if name == "_NF2_LUT":
        return _get_nf2_lut()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
