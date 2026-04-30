"""SQINT2Linear — MLX inference module for SQINT2-compressed weights (W103.4c).

Drop-in replacement for ``mlx.nn.Linear`` that consumes a ``SQINT2Layer``
(packed 2-bit indices + per-group scales/zero-points + optional rank-r SVD
factors + optional sparse COO triplet) and computes ``y = W · x`` without
ever materialising the full BF16/fp32 weight matrix at inference time.

Forward path
------------
The SQINT2 compress pipeline stores W in the Hadamard-rotated frame:

    W_rot = H_left · W · H_rightᵀ
    W_rot ≈ NF2_dequant(indices, scales, zp) + L · R + sparse_COO

Therefore, for inference (with H orthogonal so Hᵀ = H⁻¹):

    y = W · x
      = H_leftᵀ · W_rot · H_right · x
      = H_leftᵀ · (NF2_dequant(...) · x_rot + L · R · x_rot + sparse · x_rot)
                   └────── base ──────┘   └──────── residual ───────┘
    where x_rot = H_right · x_padded

The base GEMV is computed via a fused-dequant Metal kernel on Apple Silicon
(``mx.fast.metal_kernel``) that unpacks 2-bit codes, applies per-group
asymmetric (NF2 lookup, zp subtraction, scale multiplication), and reduces
along the input axis — all in one shader pass with zero W_rot materialisation.

The residual GEMV uses two MLX paths in sequence:
  1. Low-rank: ``mx.matmul(R, x_rot)`` → ``Rx`` (rank-r vector); then
     ``mx.matmul(L, Rx)`` → ``y_residual``.
  2. Sparse: ``mx.scatter_add`` of ``vals · x_rot[cols]`` into ``rows``.
The residual leg never materialises ``L · R`` as a dense ``(M, N)`` matrix —
at M=N=4096, r=16 the rank-vector path is ~16 KB vs ~64 MB.

Fallback paths
--------------
``mx.fast.metal_kernel`` is required only for the fused-dequant base GEMV.
When unavailable (older mlx, non-Metal platforms, kernel compile failure),
the constructor falls back to a pure-MLX dequant-then-matmul path that
allocates W_rot as fp16 transient — ~10× slower than the fused kernel but
numerically identical to ~1e-3 fp16 roundoff.

When the ``mlx`` package itself is unavailable (the test environment), the
class import succeeds but ``__init__`` raises ``RuntimeError`` — callers
should ``importorskip("mlx.core")`` in tests.

API
---
Constructor accepts a ``SQINT2Layer`` (the canonical path) or the raw
component arrays. Forward matches ``mlx.nn.Linear``:

    sl = SQINT2Linear.from_layer(layer, bias=None)
    y  = sl(x)         # x: (..., in_features) → y: (..., out_features)

Numerical contract
------------------
Forward output equals ``decompress_weight(layer) @ x`` to the precision of
the storage dtypes (fp16 residual factors → ~1e-3 abs/rel; fp32 → ~1e-5).
Verified by ``tests/test_sqint2_linear.py``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from squish.quant.sqint2 import (
    NF2_VALUES,
    SQINT2Layer,
    _round_up,
    _unpack_2bit,
    build_hadamard,
)

try:  # MLX is Apple-Silicon-only; CI on x86 must still import this module.
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
    _HAS_METAL_KERNEL = hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
    _Module = nn.Module
except ImportError:  # pragma: no cover
    _HAS_MLX = False
    _HAS_METAL_KERNEL = False

    class _Module:                                # minimal stand-in so the
        def __init__(self) -> None: pass          # class definition succeeds


__all__ = ["SQINT2Linear"]


# ── Metal kernel: fused NF2 unpack + per-group asymmetric + GEMV ─────────────
#
# Inputs:
#   packed  : uint8 (out, in_pad/4)  — 2-bit indices, 4 per byte (low-bit-first)
#   scales  : float32 (out, n_groups)  — per-group multiplier
#   zp      : float32 (out, n_groups)  — per-group zero-point in NF2 frame
#   x_rot   : float32 (in_pad,)        — already in the Hadamard-rotated frame
#   nf2_lut : float32 (4,)             — NF2_VALUES = {-1.5, -0.5, +0.5, +1.5}
# Outputs:
#   y_rot   : float32 (out,)           — y_rot[m] = Σ_n W_rot[m,n] · x_rot[n]
#
# Threading: one thread per output row. Each thread streams the row's packed
# bytes, unpacks 4 codes per byte, and accumulates ``(NF2[idx]−zp[g])·s[g]·x[n]``
# into a scalar accumulator. Group index ``g = n / group_size`` is updated
# every group_size columns. NF2 lookup is from a const-memory 4-element LUT.
# This is the smallest sensible Konjo kernel — one threadgroup per row, no
# tiling, no shared memory. For a 4096×4096 weight at gs=32, this is ~131 K
# multiplies per row × 4096 rows; total ~537 M FMAs, well under one Metal
# command-buffer dispatch budget on M-series silicon.

_NF2_GEMV_KERNEL_SOURCE = r"""
// sqint2_nf2_gemv — squish.quant.sqint2_linear (W103.4c)
// One thread per output row; streams packed 2-bit indices, dequantises NF2,
// applies per-group asymmetric (zp, scale) and reduces a single x_rot vector.
//
// MLX template constants (compile-time): in_pad, out_dim, group_size.
// Inputs (runtime):  packed, scales, zp, x_rot, nf2_lut.
// Output:            y_rot — (out_dim,) float32.

uint m = thread_position_in_grid.x;
if (m >= (uint)out_dim) return;

const int n_groups = in_pad / group_size;
const int row_packed_off = (int)m * (in_pad / 4);
const int row_group_off  = (int)m * n_groups;

float acc = 0.0f;

for (int g = 0; g < n_groups; ++g) {
    const float s = scales[row_group_off + g];
    const float z = zp[row_group_off + g];
    const int n0  = g * group_size;

    for (int b = 0; b < group_size / 4; ++b) {
        const uint8_t byte = packed[row_packed_off + (n0 / 4) + b];
        const float x0 = x_rot[n0 + b * 4 + 0];
        const float x1 = x_rot[n0 + b * 4 + 1];
        const float x2 = x_rot[n0 + b * 4 + 2];
        const float x3 = x_rot[n0 + b * 4 + 3];
        const float v0 = (nf2_lut[(byte >> 0) & 3u] - z) * s;
        const float v1 = (nf2_lut[(byte >> 2) & 3u] - z) * s;
        const float v2 = (nf2_lut[(byte >> 4) & 3u] - z) * s;
        const float v3 = (nf2_lut[(byte >> 6) & 3u] - z) * s;
        acc += v0 * x0 + v1 * x1 + v2 * x2 + v3 * x3;
    }
}

y_rot[m] = acc;
"""


# ── NumPy reference of the base GEMV (used when MLX absent + cross-check) ────


def _base_gemv_numpy(layer: SQINT2Layer, x_rot: np.ndarray) -> np.ndarray:
    """Compute ``y_rot = NF2_dequant(layer) @ x_rot`` in pure NumPy.

    Matches the Metal kernel byte-for-byte modulo float roundoff. Used by the
    pure-MLX-fallback constructor and by the test suite for parity checks.
    """
    in_pad = _round_up(layer.in_features, layer.cfg.group_size)
    out_dim = layer.indices.shape[0]
    n_groups = in_pad // layer.cfg.group_size

    indices = _unpack_2bit(layer.indices, in_pad)             # (out, in_pad)
    rescaled = NF2_VALUES[indices.astype(np.intp)]            # (out, in_pad)
    rescaled_3d = rescaled.reshape(out_dim, n_groups, layer.cfg.group_size)
    scale_3d = layer.scales[:, :, None].astype(np.float32)
    zp_3d = layer.zero_points[:, :, None].astype(np.float32)
    W_rot = ((rescaled_3d - zp_3d) * scale_3d).reshape(out_dim, in_pad)
    return (W_rot.astype(np.float32) @ x_rot.astype(np.float32)).astype(np.float32)


# ── SQINT2Linear ─────────────────────────────────────────────────────────────


class SQINT2Linear(_Module):
    """MLX inference module for one SQINT2-compressed weight matrix.

    Args:
        indices:        uint8, packed 2-bit codes, shape (out, in_pad // 4).
        scales:         float32, per-group multiplier, shape (out, n_groups).
        zero_points:    float32, per-group zero-point, shape (out, n_groups).
        in_features:    original (unpadded) input width.
        out_features:   output width (no padding on the output side).
        group_size:     columns per quantisation group.
        seed:           Hadamard sign-flip seed (must match the compressor).
        rotate_left:    if True, apply inverse left-Hadamard rotation post-GEMV.
        rotate_right:   if True, apply right-Hadamard rotation pre-GEMV to x.
        residual_L:     optional fp16 / fp32, shape (out, rank). None if rank=0.
        residual_R:     optional fp16 / fp32, shape (rank, in_pad). None if rank=0.
        sparse_rows:    optional int32, COO row indices in rotated frame.
        sparse_cols:    optional int32, COO col indices.
        sparse_vals:    optional fp16 / fp32, COO correction values.
        bias:           optional fp32, shape (out,).

    Notes:
        - ``in_features`` may be < ``in_pad = round_up(in_features, group_size)``.
          The forward pads ``x`` with zeros to ``in_pad`` before rotation.
        - Output is unpadded — ``y.shape[-1] == out_features`` always.
        - The Metal fused-dequant kernel is built lazily on first call; if
          the build fails the module silently falls back to a pure-MLX
          dequant-then-matmul. The fallback path is numerically equivalent
          to the kernel within fp16 storage roundoff.
    """

    def __init__(
        self,
        indices,                         # uint8 (out, in_pad // 4)
        scales,                          # float32 (out, n_groups)
        zero_points,                     # float32 (out, n_groups)
        in_features: int,
        out_features: int,
        group_size: int,
        seed: int,
        rotate_left: bool = True,
        rotate_right: bool = True,
        residual_L=None,                 # fp16 / fp32 (out, rank)
        residual_R=None,                 # fp16 / fp32 (rank, in_pad)
        sparse_rows=None,                # int32
        sparse_cols=None,                # int32
        sparse_vals=None,                # fp16 / fp32
        bias=None,                       # fp32 (out,)
    ) -> None:
        if not _HAS_MLX:
            raise RuntimeError(
                "SQINT2Linear requires mlx (Apple Silicon). "
                "Install with `pip install mlx` on macOS arm64."
            )
        super().__init__()

        # Coerce all numpy inputs to mx.array on the default device.
        self._in_features  = int(in_features)
        self._out_features = int(out_features)
        self._group_size   = int(group_size)
        self._in_pad       = _round_up(self._in_features, self._group_size)
        self._n_groups     = self._in_pad // self._group_size
        self._rotate_left  = bool(rotate_left)
        self._rotate_right = bool(rotate_right)
        self._seed         = int(seed)

        if indices.shape != (self._out_features, self._in_pad // 4):
            raise ValueError(
                f"indices shape {tuple(indices.shape)} != "
                f"({self._out_features}, {self._in_pad // 4})"
            )
        if scales.shape != (self._out_features, self._n_groups):
            raise ValueError(
                f"scales shape {tuple(scales.shape)} != "
                f"({self._out_features}, {self._n_groups})"
            )
        if zero_points.shape != scales.shape:
            raise ValueError("scales and zero_points must have matching shapes")

        # Store packed indices and per-group params on-device.
        self._packed = mx.array(np.ascontiguousarray(indices, dtype=np.uint8))
        self._scales = mx.array(np.ascontiguousarray(scales, dtype=np.float32))
        self._zp     = mx.array(np.ascontiguousarray(zero_points, dtype=np.float32))
        self._nf2    = mx.array(np.ascontiguousarray(NF2_VALUES, dtype=np.float32))

        # Hadamard rotations — re-derive deterministically from cfg.seed.
        # Materialised once at construction (negligible vs full weight).
        if self._rotate_left:
            H_left = build_hadamard(
                self._out_features, np.random.default_rng(self._seed)
            )
            self._H_left = mx.array(H_left.astype(np.float32))
        else:
            self._H_left = None
        if self._rotate_right:
            H_right = build_hadamard(
                self._in_pad, np.random.default_rng(self._seed + 1)
            )
            self._H_right = mx.array(H_right.astype(np.float32))
        else:
            self._H_right = None

        # Residual leg: low-rank L · R.
        if residual_L is not None and residual_R is not None:
            self._L = mx.array(np.asarray(residual_L, dtype=np.float32))
            self._R = mx.array(np.asarray(residual_R, dtype=np.float32))
            self._has_lowrank = True
        else:
            self._L = None
            self._R = None
            self._has_lowrank = False

        # Residual leg: sparse COO.
        if (
            sparse_rows is not None
            and sparse_cols is not None
            and sparse_vals is not None
            and sparse_rows.size > 0
        ):
            self._sp_rows = mx.array(np.ascontiguousarray(sparse_rows, dtype=np.int32))
            self._sp_cols = mx.array(np.ascontiguousarray(sparse_cols, dtype=np.int32))
            self._sp_vals = mx.array(np.ascontiguousarray(sparse_vals, dtype=np.float32))
            self._has_sparse = True
        else:
            self._sp_rows = None
            self._sp_cols = None
            self._sp_vals = None
            self._has_sparse = False

        if bias is not None:
            self.bias = mx.array(np.asarray(bias, dtype=np.float32))

        # Lazy Metal kernel — built on first forward.
        self._metal_kernel: Any = None

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_layer(
        cls,
        layer: SQINT2Layer,
        bias=None,
    ) -> "SQINT2Linear":
        """Build a SQINT2Linear directly from a SQINT2Layer (canonical path)."""
        return cls(
            indices=np.asarray(layer.indices),
            scales=np.asarray(layer.scales),
            zero_points=np.asarray(layer.zero_points),
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=layer.cfg.group_size,
            seed=layer.cfg.seed,
            rotate_left=layer.cfg.rotate_left,
            rotate_right=layer.cfg.rotate_right,
            residual_L=(
                np.asarray(layer.residual_L) if layer.residual_L is not None else None
            ),
            residual_R=(
                np.asarray(layer.residual_R) if layer.residual_R is not None else None
            ),
            sparse_rows=(
                np.asarray(layer.sparse_rows)
                if layer.sparse_rows is not None
                else None
            ),
            sparse_cols=(
                np.asarray(layer.sparse_cols)
                if layer.sparse_cols is not None
                else None
            ),
            sparse_vals=(
                np.asarray(layer.sparse_vals)
                if layer.sparse_vals is not None
                else None
            ),
            bias=bias,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def group_size(self) -> int:
        return self._group_size

    # ── Internal: base GEMV (NF2 fused-dequant) ──────────────────────────────

    def _build_metal_kernel(self) -> None:
        """Compile the fused-dequant Metal kernel on first call."""
        if not _HAS_METAL_KERNEL:
            return
        try:
            self._metal_kernel = mx.fast.metal_kernel(
                name="sqint2_nf2_gemv",
                input_names=["packed", "scales", "zp", "x_rot", "nf2_lut"],
                output_names=["y_rot"],
                source=_NF2_GEMV_KERNEL_SOURCE,
            )
        except Exception:
            # Compile failure → fall back to dequant+matmul. Any future call
            # will short-circuit to the fallback path via ``self._metal_kernel
            # is None``.
            self._metal_kernel = None

    def _base_gemv_metal(self, x_rot_1d):
        """Single-row base GEMV via the fused Metal kernel."""
        out, = self._metal_kernel(
            inputs=[self._packed, self._scales, self._zp, x_rot_1d, self._nf2],
            template=[
                ("int", "in_pad",     self._in_pad),
                ("int", "out_dim",    self._out_features),
                ("int", "group_size", self._group_size),
            ],
            grid=(self._out_features, 1, 1),
            threadgroup=(min(256, self._out_features), 1, 1),
            output_shapes=[(self._out_features,)],
            output_dtypes=[mx.float32],
        )
        return out

    def _base_gemv_mlx(self, x_rot):
        """Pure-MLX dequant-then-matmul base GEMV.

        Materialises ``W_rot`` as fp32 (out, in_pad) per call. Slower than the
        Metal kernel but works on any MLX build and gives a numerical
        cross-check during the kernel-correctness rollout.

        Accepts ``x_rot`` of shape ``(in_pad,)`` or ``(B, in_pad)``.
        """
        # Unpack 4 indices per byte — the bit shifts map to four broadcasted
        # MLX takes against the constant NF2 LUT. Operates on the full packed
        # weight in one shot; MLX fuses the dequant chain.
        p = self._packed                                          # (out, in_pad/4)
        idx0 = (p) & mx.array(np.uint8(3))
        idx1 = (p >> mx.array(np.uint8(2))) & mx.array(np.uint8(3))
        idx2 = (p >> mx.array(np.uint8(4))) & mx.array(np.uint8(3))
        idx3 = (p >> mx.array(np.uint8(6))) & mx.array(np.uint8(3))
        # Stack into (out, in_pad/4, 4) → (out, in_pad)
        stacked = mx.stack([idx0, idx1, idx2, idx3], axis=-1)     # (out, in_pad/4, 4)
        all_idx = stacked.reshape(self._out_features, self._in_pad)
        nf2 = self._nf2[all_idx.astype(mx.uint32)]                 # (out, in_pad)
        # Per-group asymmetric: (nf2 − zp) * scale, broadcast over group axis.
        nf2_3d = nf2.reshape(self._out_features, self._n_groups, self._group_size)
        scale_3d = self._scales.reshape(self._out_features, self._n_groups, 1)
        zp_3d    = self._zp.reshape(self._out_features, self._n_groups, 1)
        W_rot = ((nf2_3d - zp_3d) * scale_3d).reshape(
            self._out_features, self._in_pad
        )
        # GEMV: x_rot is (..., in_pad); we want (..., out)
        return mx.matmul(x_rot, W_rot.T)

    def _base_gemv(self, x_rot):
        """Dispatch the base GEMV: Metal kernel for 1-D, MLX for batched."""
        # The Metal kernel is single-row; for batched x we use the MLX path
        # which is already vectorised and only slightly slower than launching
        # B separate kernel calls for typical decode (B ≤ 4).
        if (
            x_rot.ndim == 1
            and _HAS_METAL_KERNEL
        ):
            if self._metal_kernel is None:
                self._build_metal_kernel()
            if self._metal_kernel is not None:
                try:
                    return self._base_gemv_metal(x_rot.astype(mx.float32))
                except Exception:
                    # First-call kernel failure: cache the fallback decision.
                    self._metal_kernel = None
        return self._base_gemv_mlx(x_rot)

    # ── Internal: residual GEMV (low-rank + sparse) ──────────────────────────

    def _residual_gemv(self, x_rot):
        """Compute ``(L · R) · x_rot + sparse_coo · x_rot`` in MLX.

        Mirrors ``squish.quant.sqint2.sqint2_residual_gemv`` semantically; the
        difference is that this path stays inside MLX so we avoid round-tripping
        through numpy mid-forward. For the ``B == 1`` decode hot path the two
        produce the same answer to within fp16 storage roundoff.
        """
        # Promote to fp32 for accumulation regardless of how factors are stored.
        zero_shape = (
            (self._out_features,) if x_rot.ndim == 1
            else (x_rot.shape[0], self._out_features)
        )
        y = mx.zeros(zero_shape, dtype=mx.float32)

        if self._has_lowrank:
            # x_rot: (in_pad,) or (B, in_pad). Rx: (rank,) or (B, rank).
            if x_rot.ndim == 1:
                Rx = mx.matmul(self._R, x_rot)                     # (rank,)
                y = y + mx.matmul(self._L, Rx)                     # (out,)
            else:
                Rx = mx.matmul(x_rot, self._R.T)                   # (B, rank)
                y = y + mx.matmul(Rx, self._L.T)                   # (B, out)

        if self._has_sparse:
            # COO scatter-add: y[..., rows[i]] += vals[i] · x_rot[..., cols[i]].
            # Done as a one-shot dense add for shape clarity — at sparse_frac=1%
            # for an FFN tensor of (M, N) ≈ (4096, 11008), nnz ≈ 4.5e5; a
            # single mx.scatter_add call dominates the cost regardless of
            # sequencing. Avoid the explicit scatter for B>1 batches: the per-
            # batch correction is independent so we broadcast x_rot along rows.
            x_take = x_rot[..., self._sp_cols.astype(mx.int32)]    # (..., nnz)
            contrib = x_take * self._sp_vals.astype(mx.float32)    # (..., nnz)
            # mx.scatter_add: target (..., out), updates (..., nnz), indices (nnz,)
            y = y.at[..., self._sp_rows.astype(mx.int32)].add(contrib)

        return y

    # ── Forward ───────────────────────────────────────────────────────────────

    def __call__(self, x):
        """Compute ``y = W · x (+ bias)``.

        Args:
            x: ``(in_features,)`` or ``(B, in_features)`` mx.array.

        Returns:
            mx.array of shape ``(out_features,)`` or ``(B, out_features)``.
        """
        if x.shape[-1] != self._in_features:
            raise ValueError(
                f"x last-axis {x.shape[-1]} must equal in_features={self._in_features}"
            )

        # Step 1: pad x to in_pad if needed.
        x32 = x.astype(mx.float32)
        if self._in_pad > self._in_features:
            pad_shape = list(x32.shape[:-1]) + [self._in_pad - self._in_features]
            x32 = mx.concatenate(
                [x32, mx.zeros(pad_shape, dtype=mx.float32)], axis=-1
            )

        # Step 2: right-Hadamard rotation. x_rot = x_padded @ H_rightᵀ.
        if self._H_right is not None:
            x_rot = mx.matmul(x32, self._H_right.T)
        else:
            x_rot = x32

        # Step 3: base + residual in the rotated frame.
        y_rot = self._base_gemv(x_rot)
        if self._has_lowrank or self._has_sparse:
            y_rot = y_rot + self._residual_gemv(x_rot)

        # Step 4: inverse left-Hadamard rotation. y = y_rot @ H_left.
        # (H_leftᵀ · y_rot.T).T == y_rot · H_left for orthogonal H.
        if self._H_left is not None:
            y = mx.matmul(y_rot, self._H_left)
        else:
            y = y_rot

        # Step 5: optional bias add.
        if hasattr(self, "bias"):
            y = y + self.bias

        return y

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        bias_str = "bias" if hasattr(self, "bias") else "no bias"
        return (
            f"SQINT2Linear(in={self._in_features}, out={self._out_features}, "
            f"gs={self._group_size}, "
            f"residual={'L+R' if self._has_lowrank else '-'},"
            f"sparse={'COO' if self._has_sparse else '-'}, {bias_str})"
        )
