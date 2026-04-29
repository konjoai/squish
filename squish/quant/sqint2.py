"""squish/quant/sqint2.py — SQINT2 Stages 1–3: Hadamard + NF2 + SVD residual.

SQINT2 is Squish's coherent INT2 weight-compression format (W103). This module
implements the offline-compress half of stages 1, 2, and 3:

    Stage 1  Hadamard incoherence preprocessing  (this module: encode/decode)
    Stage 2  NF2 per-group asymmetric quantisation  (this module)
    Stage 3  Low-rank SVD + sparse residual correction  (W103.2 — this module)
    Stage 4  Layer-selective mixed precision  (W103.3 — quantizer.py routing)

W103.2 — rank-r SVD + sparse-k residual correction
----------------------------------------------------
After Stage 1+2, the reconstruction error E = W_rotated − dequant(Q_INT2) is
measured in the Hadamard-rotated frame. Two complementary corrections are then
computed and stored compactly alongside the INT2 indices:

  1. **Low-rank SVD correction** (rank r, default 16):
         E ≈ L · R,  L ∈ ℝ^{m × r},  R ∈ ℝ^{r × n}
     Follows the same convention as `squish/quant/milo_quant.py`:
     L = U[:, :r] × S[:r], R = Vt[:r, :].  Singular values absorbed into L.
     Factors stored in ``residual_factor_dtype`` (fp16 default).

  2. **Sparse outlier correction** (top-``sparse_frac`` fraction of |E₂| entries):
         E₂ = E − L·R  (post-SVD residual)
         sparse_vals at (sparse_rows, sparse_cols) stored as fp16 COO triplets.
     At decompress: W_rot_rec[rows, cols] += sparse_vals.astype(fp32).

At inference: W_recon = H_leftᵀ · (dequant_rot + L·R + sparse) · H_right

Measured SNR lift on σ=0.02 IID Gaussian, (1536, 576), g=32, refine=2, 5 seeds:
  Stage 1+2 base:  9.69 dB  (W103.1 gate: ≥ 9.0 dB)
  + SVD rank-16:  +0.30 dB  → 9.99 dB
  + sparse 1%:    +0.24 dB  → 10.23 dB  (W103.2 gate: ≥ 10.0 dB)

Why outliers do NOT benefit further from W103.2: Hadamard rotation whitens any
input distribution (IID, outlier-bearing, low-rank-dominant) by design. After
rotation the post-quantisation residual is IID regardless of original structure —
outlier recovery was already performed by Stage 1. Pre-rotation sparse correction
(fixing outliers in the original domain before quantisation) is scoped to W103.3.

Why naive INT2 fails (the floor)
--------------------------------
Transformer weight matrices contain ~0.1% massive outliers. With only 4 bins
available at 2 bits, the outliers dictate the quantisation scale, collapsing
99.9% of normal weights into 1–2 of the 4 bins. Signal is destroyed; output
becomes incoherent. Squish's own benchmarks (see docs/benchmark_int3_int2.md)
show naive INT2-WOQ at ~7 dB SNR on synthetic σ=0.02 weights — well below
what coherent generation requires. arc_easy under naive INT2 sits at ~26–30%
≈ random across the 0.6B–7B family.

Why SQINT2 works (the ceiling)
------------------------------
Respect the geometry first.

1. **Randomised Hadamard rotation** spreads outlier energy uniformly across
   all weight dimensions before quantisation (Tseng et al. 2024 — QuaRot;
   Tseng & Chee 2024 — QuIP#). The rotation is orthogonal:

       W_rotated = H_left · W · H_rightᵀ        with H·Hᵀ = I

   Inner products are preserved up to a scalar. After rotation, the per-group
   variance is uniform; the bin-collapse failure mode disappears.

2. **NF2 per-group asymmetric quantisation** maps each group's empirical
   [x_min, x_max] onto a 4-symbol codebook NF2 = {−1.5, −0.5, +0.5, +1.5}.
   These symbols are uniformly spaced because — after per-group asymmetric
   scaling that places (x_min, x_max) at (−1.5, +1.5) — the within-group
   distribution is approximately uniform on [−1.5, +1.5] (the extreme order
   statistics of the original Gaussian have been pinned to the boundaries).
   For approximately-uniform data, uniform codebook spacing is the
   Lloyd-Max-optimal choice.

3. **One Lloyd-Max refinement sweep** (default) recomputes the codebook
   centroids as the empirical mean of the assignment buckets, then reassigns.
   This corrects for the small mismatch between the assumed-uniform and
   actual-near-uniform within-group distribution, lifting SNR by ~2–3 dB
   on synthetic Gaussian weights.

Reference papers
----------------
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Tseng et al. 2024,
  arXiv 2404.00456). The Walsh–Hadamard sign-flip construction used here.
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice
  Codebooks (Tseng & Chee 2024, arXiv 2402.04396). Hadamard incoherence
  for sub-3-bit weight compression.
- QLoRA: NF4 Quantile Quantisation (Dettmers et al. 2023, arXiv 2305.14314).
  The NormalFloat construction; NF2 is the 2-bit specialisation.
- INT2.1: Towards Fine-grained 2-bit Quantization for LLMs via Low-Rank
  Residuals (2023). Motivates Stage 3 (low-rank residual; W103.2).

Storage layout
--------------
A compressed SQINT2Layer holds (per FFN weight matrix):

    indices           uint8    packed (out_padded, in_padded // 4)   2 bpw raw
    scales            fp32     shape (out_padded, n_groups)         16 bpw / group
    zero_points       fp32     shape (out_padded, n_groups)         16 bpw / group
    residual_L        fp16     shape (out_padded, rank) or None
    residual_R        fp16     shape (rank, in_padded)  or None
    sparse_rows       int32    shape (k,) or None         COO row indices
    sparse_cols       int32    shape (k,) or None         COO col indices
    sparse_vals       fp16     shape (k,) or None         sparse corrections
    cfg               SQINT2Config

Effective bits per weight — Stage 1+2 base (group_size=32, fp32 scale + zp):

    2.0  (raw indices)
    + 32 / 32  (scale)
    + 32 / 32  (zero-point)
    = 4.0 bpw before residual

Effective bits per weight — Stage 3 added overhead at g=32, M=N=4096:

    + rank·(M+N)·fp16 / (M·N)   ≈  0.125 bpw  (r=16, fp16)
    + sparse_frac·80 bits/entry  ≈  0.80 bpw   (1%, int32+int32+fp16 per entry)
    = 4.925 bpw gross at g=32 fp32 scale+zp

Hitting the SQINT2 spec target of ~2.15 bpw requires compressing scale/zp to
INT8 and using g≥128 (scheduled for W103.3 scale-compression pass). The 4.0 bpw
base figure is unchanged here; W103.2 adds only the residual overhead on top.

Public API
----------
- SQINT2Config         hyper-parameters (group_size, refine_iters, seed, …,
                       residual_rank, residual_factor_dtype, sparse_frac)
- SQINT2Layer          one compressed weight matrix (numpy arrays + cfg);
                       .effective_bpw          base bpw (backward-compat)
                       .effective_bpw_at(M, N) full bpw accounting with residual
- compress_weight      W (float32, out × in) → SQINT2Layer
- decompress_weight    SQINT2Layer → W̃ (float32, out × in)
- snr_db               helper: 10·log10(σ²_signal / σ²_noise)
- NF2_VALUES           the four NF2 codebook symbols
- build_hadamard       lifted from squish/kv/kv_cache.py for compress-time use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Public exports ────────────────────────────────────────────────────────────

__all__ = [
    "NF2_VALUES",
    "SQINT2Config",
    "SQINT2Layer",
    "build_hadamard",
    "apply_hadamard",
    "inverse_hadamard",
    "compress_weight",
    "decompress_weight",
    "compress_weights_sqint2",
    "snr_db",
    "save_sqint2_layer",
    "load_sqint2_layer",
    "SQINT2_FORMAT_VERSION",
    "SQINT2_SUFFIXES",
]


# ── NF2 codebook ──────────────────────────────────────────────────────────────
#
# Four symbols, uniformly spaced on [−1.5, +1.5]. The boundaries match the
# per-group asymmetric-scale convention: x_min → −1.5, x_max → +1.5. Within
# that mapped range the data distribution is approximately uniform, for which
# uniform spacing is Lloyd-Max-optimal.
#
# The codebook is symmetric around 0 — zero-mean weights map to the {−0.5,
# +0.5} interior, large-magnitude weights to the {−1.5, +1.5} boundary.
# ──────────────────────────────────────────────────────────────────────────────

NF2_VALUES: np.ndarray = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


# ── Configuration ─────────────────────────────────────────────────────────────


_RESIDUAL_FACTOR_DTYPES = frozenset({"fp32", "fp16"})


@dataclass
class SQINT2Config:
    """Hyper-parameters describing the SQINT2 compression grid.

    Attributes:
        group_size:    columns per quantisation group (default 32, matches
                       the AWQ pipeline g=32 from `squish/quant/awq.py`).
        seed:          RNG seed for the Hadamard sign-flip construction.
                       Stored on the SQINT2Layer so decompress reproduces H.
        refine_iters:  Lloyd-Max refinement sweeps after the initial NF2
                       assignment (default 1). Each sweep recomputes the
                       four codebook centroids as the mean of their
                       assignment bucket (within the rescaled per-group
                       coordinate frame), then reassigns. Lifts SNR by
                       ~2–3 dB on synthetic Gaussian weights at g=32.
        rotate_left:   apply Hadamard on the output-feature axis (default
                       True). Set False for right-only rotation.
        rotate_right:  apply Hadamard on the input-feature axis (default
                       True). Both-sides rotation matches the user's spec
                       formula `W_rotated = H · W · Hᵀ` generalised for
                       rectangular W as `H_left · W · H_rightᵀ`.

        -- W103.2 Stage-3 residual fields --

        residual_rank:          rank r for truncated SVD of the post-quant
                                residual E = W_rotated − dequant(Q_INT2).
                                0 = Stage 1+2 only (W103.1 behaviour, default).
                                16 = production SQINT2 setting.
        residual_factor_dtype:  storage dtype for L and R factors.
                                "fp16" (default) — half-precision, minimal SNR
                                loss vs fp32 (~0.02 dB on synthetic Gaussian).
                                "fp32" — full precision, used for unit tests.
        sparse_frac:            fraction of |E₂| entries (post-SVD residual) to
                                store as fp16 COO sparse corrections.
                                0.0 = disabled (default). 0.01 = top 1%.
    """

    group_size: int = 32
    seed: int = 42
    refine_iters: int = 1
    rotate_left: bool = True
    rotate_right: bool = True
    # W103.2
    residual_rank: int = 0
    residual_factor_dtype: str = "fp16"
    sparse_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")
        if self.group_size % 4 != 0:
            # 2-bit packing groups four indices into one byte; group_size must
            # be divisible by 4 so per-group buffers align cleanly.
            raise ValueError(
                f"group_size must be divisible by 4 for 2-bit packing, "
                f"got {self.group_size}"
            )
        if self.refine_iters < 0:
            raise ValueError(f"refine_iters must be ≥ 0, got {self.refine_iters}")
        if self.residual_rank < 0:
            raise ValueError(f"residual_rank must be ≥ 0, got {self.residual_rank}")
        if self.residual_factor_dtype not in _RESIDUAL_FACTOR_DTYPES:
            raise ValueError(
                f"residual_factor_dtype must be one of {sorted(_RESIDUAL_FACTOR_DTYPES)}, "
                f"got {self.residual_factor_dtype!r}"
            )
        if not (0.0 <= self.sparse_frac < 1.0):
            raise ValueError(
                f"sparse_frac must be in [0, 1), got {self.sparse_frac}"
            )


# ── Compressed-layer container ────────────────────────────────────────────────


@dataclass
class SQINT2Layer:
    """Compressed representation of one weight matrix (Stage 1+2+3).

    Attributes:
        indices:       uint8, shape (out_features, in_padded // 4). Each byte
                       packs 4 successive 2-bit indices, low-bit-first
                       (index 0 in bits 0–1, index 3 in bits 6–7).
        scales:        float32, shape (out_features, n_groups). Per-group
                       multiplier such that NF2_VALUES[idx] decodes to a
                       value in (rescaled - zp) * scale.
        zero_points:   float32, shape (out_features, n_groups). Per-group
                       additive shift in the rescaled coordinate frame.
        in_features:   original (unpadded) input-feature count.
        out_features:  original (unpadded) output-feature count.
        cfg:           SQINT2Config with seed for Hadamard reconstruction.

        -- W103.2 Stage-3 residual fields (all None when residual_rank == 0) --

        residual_L:    ndarray, shape (out_padded, rank), dtype per
                       cfg.residual_factor_dtype. Left SVD factor. None when
                       cfg.residual_rank == 0.
        residual_R:    ndarray, shape (rank, in_padded), same dtype. None when
                       cfg.residual_rank == 0.
        sparse_rows:   int32 ndarray, shape (k,) or None. COO row indices in
                       the padded rotated frame.
        sparse_cols:   int32 ndarray, shape (k,) or None. COO col indices.
        sparse_vals:   float16 ndarray, shape (k,) or None. Correction values.
    """

    indices: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    in_features: int
    out_features: int
    cfg: SQINT2Config
    # W103.2 residual — all None for Stage 1+2 only layers
    residual_L: "np.ndarray | None" = None
    residual_R: "np.ndarray | None" = None
    sparse_rows: "np.ndarray | None" = None
    sparse_cols: "np.ndarray | None" = None
    sparse_vals: "np.ndarray | None" = None

    @property
    def n_groups(self) -> int:
        in_padded = _round_up(self.in_features, self.cfg.group_size)
        return in_padded // self.cfg.group_size

    @property
    def effective_bpw(self) -> float:
        """Base bpw: INT2 indices + fp32 scale + fp32 zero-point per group.

        Backward-compatible with the W103.1 definition. Does NOT include the
        Stage-3 residual overhead. Use effective_bpw_at(M, N) for the full
        accounting including L·R factors and sparse corrections.
        """
        gs = self.cfg.group_size
        return 2.0 + 32.0 / gs + 32.0 / gs

    def effective_bpw_at(self, out: int | None = None, in_: int | None = None) -> float:
        """Full bit-width accounting including Stage-3 residual overhead.

        Args:
            out: output-feature count. Defaults to self.out_features.
            in_: input-feature count. Defaults to self.in_features.

        Returns:
            Total bits-per-original-weight: INT2 indices + scale/zp overhead
            + SVD factor storage + sparse COO storage. Use this number when
            comparing total storage against INT3/INT4 baselines.

        Breakdown at g=32, fp32 scale/zp, r=16 fp16, sparse 1%, M=N=4096:
            2.000 bpw  INT2 indices (padded → almost equal original for large M,N)
            1.000 bpw  fp32 scale at g=32  (32/32)
            1.000 bpw  fp32 zero-point at g=32
            0.063 bpw  L factors fp16 (16·4096·16 / 4096²)
            0.063 bpw  R factors fp16
            0.800 bpw  sparse 1% at 80 bits/entry (int32+int32+fp16)
            ─────────
            4.925 bpw  gross at g=32 fp32 scale+zp

        Reaching ~2.15 bpw requires g≥128 with INT8-compressed scale/zp
        (deferred to W103.3 scale-compression pass).
        """
        M = out if out is not None else self.out_features
        N = in_ if in_ is not None else self.in_features
        n_weights = M * N

        in_padded = _round_up(N, self.cfg.group_size)
        out_padded = M
        n_groups = in_padded // self.cfg.group_size

        bits = 0
        # INT2 indices (padded, scaled to hypothetical M, N)
        bits += out_padded * in_padded * 2
        # fp32 scale + zero-point per group
        bits += out_padded * n_groups * 32 * 2
        # SVD factors: project with stored rank to hypothetical M, N
        if self.residual_L is not None and self.residual_R is not None:
            actual_rank = int(self.residual_L.shape[1])
            factor_bits = 16 if self.cfg.residual_factor_dtype == "fp16" else 32
            bits += out_padded * actual_rank * factor_bits   # L: (M, r)
            bits += actual_rank * in_padded * factor_bits    # R: (r, N)
        # Sparse COO: scale by cfg.sparse_frac on hypothetical M, N padded shape.
        # 80 bits/entry = int32 row (32) + int32 col (32) + fp16 val (16).
        if self.cfg.sparse_frac > 0.0 and self.sparse_rows is not None:
            k = max(1, round(out_padded * in_padded * self.cfg.sparse_frac))
            bits += k * 80

        return bits / n_weights


# ── Hadamard primitives ───────────────────────────────────────────────────────
#
# Lifted from `squish/kv/kv_cache.py::HadamardKVCache._build_hadamard` (the
# QuaRot KV path shipped in Wave 19/20). The KV-path version returns float16
# because runtime KV traffic is mixed-precision; here we return float32 for
# the compress-time matmul, which needs the extra precision to avoid SNR
# pollution from the rotation itself. The construction algorithm is identical.
# ──────────────────────────────────────────────────────────────────────────────


def build_hadamard(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (dim, dim) random Hadamard-like orthogonal matrix in float32.

    For power-of-two dim: Walsh–Hadamard via Sylvester construction with a
    random sign-flip column on the right (column-wise diagonal D). The
    resulting matrix is orthogonal: H @ Hᵀ = I.

    For non-power-of-two dim: random orthogonal matrix via QR decomposition
    of a Gaussian. Identical orthogonality property; slightly slower to
    construct, but only paid once per layer.

    Algorithm matches `squish/kv/kv_cache.py::HadamardKVCache._build_hadamard`
    bit-for-bit (other than the float16-vs-float32 return dtype). Lifted
    here rather than imported to avoid a heavyweight cross-module dependency
    from quant onto kv at import time.
    """
    if dim <= 0:
        raise ValueError(f"dim must be ≥ 1, got {dim}")

    if (dim & (dim - 1)) == 0:
        # Power-of-two: Walsh–Hadamard via Sylvester construction
        H = np.array([[1.0]], dtype=np.float32)
        while H.shape[0] < dim:
            H = np.block([[H, H], [H, -H]])
        H /= np.sqrt(dim)
        # Random column-wise sign flip → randomises the rotation while
        # preserving orthogonality (D is its own inverse).
        signs = rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32)
        H = H * signs[np.newaxis, :]
    else:
        # General fallback: random orthogonal matrix via QR of Gaussian.
        G = rng.standard_normal((dim, dim)).astype(np.float32)
        H, _ = np.linalg.qr(G)

    return H.astype(np.float32, copy=False)


def apply_hadamard(
    W: np.ndarray, H_left: np.ndarray | None, H_right: np.ndarray | None
) -> np.ndarray:
    """Apply two-sided Hadamard rotation: W_rot = H_left · W · H_rightᵀ.

    Either side may be None (skip that rotation). Both rotations are
    isometric, so element-wise variance of W is preserved. Outlier energy
    that was concentrated in a few channels of W gets spread uniformly
    across all channels of W_rot.

    Compute is in float32 throughout — the rotation must not introduce
    SNR loss of its own; a downcast to bf16 inside the rotation matmul
    would cost ~1–2 dB on synthetic Gaussian inputs.
    """
    out = W.astype(np.float32, copy=False)
    if H_left is not None:
        out = H_left.astype(np.float32, copy=False) @ out
    if H_right is not None:
        out = out @ H_right.astype(np.float32, copy=False).T
    return out


def inverse_hadamard(
    W_rot: np.ndarray, H_left: np.ndarray | None, H_right: np.ndarray | None
) -> np.ndarray:
    """Invert the two-sided rotation: W = H_leftᵀ · W_rot · H_right.

    For orthogonal H, Hᵀ = H⁻¹, so this exactly inverts apply_hadamard
    in infinite precision and to within ~1e-7 in float32.
    """
    out = W_rot.astype(np.float32, copy=False)
    if H_left is not None:
        out = H_left.astype(np.float32, copy=False).T @ out
    if H_right is not None:
        out = out @ H_right.astype(np.float32, copy=False)
    return out


# ── 2-bit packing helpers ─────────────────────────────────────────────────────
#
# Pack 4 successive 2-bit indices into one uint8. Layout (low-bit-first):
#
#     byte = (idx[3] << 6) | (idx[2] << 4) | (idx[1] << 2) | idx[0]
#
# This matches the convention used by squish/quant/quantizer.py for INT4
# nibble packing (low nibble first), extended to 2-bit pairs.
# ──────────────────────────────────────────────────────────────────────────────


def _pack_2bit(indices: np.ndarray) -> np.ndarray:
    """Pack a uint8 array of 2-bit indices (last axis) into uint8 bytes."""
    if indices.dtype != np.uint8:
        indices = indices.astype(np.uint8)
    if indices.shape[-1] % 4 != 0:
        raise ValueError(
            f"last-axis length must be divisible by 4 for 2-bit packing, "
            f"got shape {indices.shape}"
        )
    grouped = indices.reshape(*indices.shape[:-1], -1, 4)
    packed = (
        grouped[..., 0]
        | (grouped[..., 1] << 2)
        | (grouped[..., 2] << 4)
        | (grouped[..., 3] << 6)
    ).astype(np.uint8)
    return packed


def _unpack_2bit(packed: np.ndarray, n_columns: int) -> np.ndarray:
    """Inverse of _pack_2bit. Returns indices uint8 with last-axis = n_columns."""
    if packed.dtype != np.uint8:
        packed = packed.astype(np.uint8)
    if n_columns % 4 != 0:
        raise ValueError(
            f"n_columns must be divisible by 4 for 2-bit unpacking, got {n_columns}"
        )
    expected_packed = n_columns // 4
    if packed.shape[-1] != expected_packed:
        raise ValueError(
            f"packed last-axis {packed.shape[-1]} does not match "
            f"n_columns={n_columns} (expected {expected_packed})"
        )
    out_shape = (*packed.shape[:-1], n_columns)
    out = np.empty(out_shape, dtype=np.uint8)
    out[..., 0::4] = packed & 0b11
    out[..., 1::4] = (packed >> 2) & 0b11
    out[..., 2::4] = (packed >> 4) & 0b11
    out[..., 3::4] = (packed >> 6) & 0b11
    return out


# ── NF2 per-group asymmetric quantisation core ────────────────────────────────


def _round_up(n: int, multiple: int) -> int:
    """Round n up to the nearest multiple of `multiple`."""
    return ((n + multiple - 1) // multiple) * multiple


def _nf2_quantise_groups(
    groups: np.ndarray, refine_iters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise (M, group_size) float32 groups onto NF2.

    Step 1 — per-group asymmetric scale + zero-point:

        scale[g] = (xmax[g] - xmin[g]) / (NF2_max - NF2_min)
                 = (xmax[g] - xmin[g]) / 3.0
        zp[g]    = NF2_min - xmin[g] / scale[g]
                 = -1.5 - xmin[g] / scale[g]

    so that the rescaled value  y = x / scale + zp  satisfies
    y_min = -1.5 and y_max = +1.5 — the NF2 boundary.

    Step 2 — initial assignment: snap each rescaled value to the nearest
    NF2 codebook entry.

    Step 3 — Lloyd-Max refinement (refine_iters sweeps): for each codebook
    bucket, set its centroid to the empirical mean of the rescaled values
    assigned to it, then reassign. Recovers the mismatch between the
    assumed-uniform and actually-near-uniform per-group distribution.

    Empty groups (xmax == xmin) are handled with scale = 1.0, zp = 0.0,
    indices = 1 (NF2 = -0.5 ≈ 0). This collapses to lossless storage of
    constant-zero weight patches.

    Returns:
        indices       uint8, shape (M, group_size). Values in {0, 1, 2, 3}.
        scales        float32, shape (M,).
        zero_points   float32, shape (M,).
    """
    if groups.ndim != 2:
        raise ValueError(f"groups must be 2-D (M, group_size), got ndim={groups.ndim}")

    M, gs = groups.shape
    g32 = groups.astype(np.float32, copy=False)

    xmin = g32.min(axis=1)            # (M,)
    xmax = g32.max(axis=1)            # (M,)
    span = xmax - xmin                # (M,)

    # Constant groups: protect against divide-by-zero.
    safe = span > 0
    scale = np.where(safe, span / 3.0, 1.0).astype(np.float32)
    zp = np.where(safe, -1.5 - xmin / scale, 0.0).astype(np.float32)

    # Rescale into the NF2 frame: y ∈ approximately [-1.5, +1.5].
    rescaled = g32 / scale[:, None] + zp[:, None]

    # Track the working codebook (one shared codebook across all groups —
    # the per-group adaptation lives in scale + zp). Lloyd-Max refinement
    # lets the codebook drift slightly off the {-1.5, -0.5, +0.5, +1.5}
    # initial points to better fit the observed rescaled distribution.
    codebook = NF2_VALUES.astype(np.float32, copy=True)

    # Step 2: initial nearest-neighbour assignment.
    # diff: (M, gs, 4); pick argmin over the codebook axis.
    indices = _assign_to_codebook(rescaled, codebook)

    # Step 3: Lloyd-Max refinement.
    for _ in range(refine_iters):
        new_codebook = codebook.copy()
        # Aggregate over ALL groups so the codebook is a single shared grid
        # — adapting it per-group would defeat the per-group scale/zp design.
        flat_rescaled = rescaled.reshape(-1)
        flat_indices = indices.reshape(-1)
        for k in range(NF2_VALUES.shape[0]):
            mask = flat_indices == k
            if mask.any():
                new_codebook[k] = float(flat_rescaled[mask].mean())
        # Keep the codebook sorted so dequant indexing remains monotonic.
        new_codebook.sort()
        codebook = new_codebook
        indices = _assign_to_codebook(rescaled, codebook)

    # Bake the refined codebook back into scale/zp so dequant can use the
    # static NF2_VALUES array. We seek scale', zp' such that:
    #
    #     codebook[k] · scale + (NF2_VALUES[k] - codebook[k]) · scale ?= NF2_VALUES[k] · scale'
    #
    # Closed form: rescaled value y maps to codebook[k] under refinement,
    # which decodes to (codebook[k] - zp) * scale in original space. We want
    # NF2_VALUES[k] to decode to the same value via (NF2_VALUES[k] - zp') * scale'.
    # That's a 2-point linear system in scale', zp':
    #
    #     (NF2_VALUES[0] - zp') * scale' = (codebook[0] - zp) * scale
    #     (NF2_VALUES[3] - zp') * scale' = (codebook[3] - zp) * scale
    #
    # With NF2_VALUES = [-1.5, -0.5, 0.5, 1.5] (diff = 3.0):
    #
    #     scale' = ((codebook[3] - codebook[0]) / 3.0) * scale
    #     zp'    = NF2_VALUES[0] - (codebook[0] - zp) * scale / scale'
    if refine_iters > 0:
        cb_lo = float(codebook[0])
        cb_hi = float(codebook[-1])
        cb_span = cb_hi - cb_lo
        if cb_span > 0:
            new_scale = (cb_span / 3.0) * scale
            # Avoid divide-by-zero when new_scale collapses for an all-equal group
            safe_new = new_scale > 0
            decoded_lo = (cb_lo - zp) * scale  # original-space value of cb[0]
            new_zp = np.where(
                safe_new,
                NF2_VALUES[0] - decoded_lo / np.where(safe_new, new_scale, 1.0),
                zp,
            ).astype(np.float32)
            scale = np.where(safe_new, new_scale, scale).astype(np.float32)
            zp = new_zp

    # Constant-group override (AFTER bake). NF2 has no zero entry, so an
    # all-zero or otherwise constant group would naively decode to ±0.5×scale,
    # leaving residual energy that the inverse Hadamard then amplifies to
    # √(out·in)·0.5 ≈ 20+ on a 32×64 matrix. Force constant groups to a fixed
    # codebook index (1, NF2 = −0.5) and pick scale = 1, zp = NF2[1] − c so
    # dequant is exact: (NF2[1] − zp) · 1 = c. Round-trip becomes lossless on
    # constant inputs (the natural correctness floor).
    const_mask = ~safe
    if const_mask.any():
        indices[const_mask, :] = 1
        scale[const_mask] = 1.0
        zp[const_mask] = float(NF2_VALUES[1]) - xmin[const_mask].astype(np.float32)

    return indices.astype(np.uint8, copy=False), scale, zp


def _assign_to_codebook(rescaled: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-neighbour assignment. rescaled: (M, gs); codebook: (4,)."""
    # Vectorised over all groups simultaneously. Memory cost: 4 × rescaled.size
    # floats — acceptable for compress-time use on transformer-scale weights
    # (peak ~500 MB on a 4096×14336 layer at fp32; well within budget).
    diffs = rescaled[..., None] - codebook[None, None, :]   # (M, gs, 4)
    return np.argmin(np.abs(diffs), axis=-1).astype(np.uint8)


# ── Stage-3 residual helpers ─────────────────────────────────────────────────
#
# Low-rank SVD convention: L = U[:, :r] * S[:r], R = Vt[:r, :].
# Singular values absorbed into L — matches LowRankCompensator in milo_quant.py.
#
# Factor storage: fp16 (halves vs fp32; SNR loss < 0.05 dB on Gaussian) or
# fp32 (used in unit tests for exact comparison). FP8 not in numpy; use fp16
# as the minimal-loss compact representation.
# ──────────────────────────────────────────────────────────────────────────────


def _pack_factor(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Store an SVD factor in the requested compact dtype."""
    if dtype == "fp16":
        return arr.astype(np.float16)
    return arr.astype(np.float32)  # "fp32"


def _unpack_factor(arr: np.ndarray) -> np.ndarray:
    """Restore a stored SVD factor to float32 for computation."""
    return arr.astype(np.float32, copy=False)


def _compute_stage3_residual(
    E: np.ndarray,
    cfg: "SQINT2Config",
) -> "Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]":
    """Compute the Stage-3 SVD + sparse residual correction from error matrix E.

    E is the reconstruction error in the Hadamard-rotated frame:
        E = W_rotated − dequant(Q_INT2)

    The correction is split into two parts:
      1. Rank-r SVD:  E ≈ L · R  with L = U[:, :r]·S[:r], R = Vt[:r, :]
      2. Sparse top-k%: top |E₂| entries of the post-SVD residual E₂ = E − L·R

    Both are stored compactly (fp16 factors, fp16 COO values) and applied
    additively to the rotated reconstruction at decompress time.

    Returns:
        (residual_L, residual_R, sparse_rows, sparse_cols, sparse_vals).
        Any component is None when the corresponding cfg field is zero/off.

    Raises:
        RuntimeError: if np.linalg.svd fails (near-singular matrix). Returns
                      zero factors rather than propagating, matching MiLo.
    """
    L_out: "np.ndarray | None" = None
    R_out: "np.ndarray | None" = None
    E2 = E  # remaining residual — updated after SVD if rank > 0

    if cfg.residual_rank > 0:
        rank = min(cfg.residual_rank, min(E.shape[0], E.shape[1]))
        try:
            u, s, vt = np.linalg.svd(E.astype(np.float32), full_matrices=False)
        except np.linalg.LinAlgError:
            # Near-zero matrix or SVD failure — zero factors, no correction.
            L_out = _pack_factor(np.zeros((E.shape[0], rank), dtype=np.float32), cfg.residual_factor_dtype)
            R_out = _pack_factor(np.zeros((rank, E.shape[1]), dtype=np.float32), cfg.residual_factor_dtype)
            # E2 stays as E — sparse can still correct from original residual
        else:
            raw_L = (u[:, :rank] * s[:rank]).astype(np.float32)  # (m, r)
            raw_R = vt[:rank, :].astype(np.float32)               # (r, n)
            L_out = _pack_factor(raw_L, cfg.residual_factor_dtype)
            R_out = _pack_factor(raw_R, cfg.residual_factor_dtype)
            # Reconstruct in fp32 to get accurate E2 for sparse step
            E2 = (E.astype(np.float32)
                  - _unpack_factor(L_out) @ _unpack_factor(R_out))

    rows_out: "np.ndarray | None" = None
    cols_out: "np.ndarray | None" = None
    vals_out: "np.ndarray | None" = None

    if cfg.sparse_frac > 0.0:
        k = max(1, round(E2.size * cfg.sparse_frac))
        flat_e2 = E2.reshape(-1).astype(np.float32)
        # np.argpartition is O(N) — much faster than full argsort for large matrices.
        top_k_idx = np.argpartition(np.abs(flat_e2), -k)[-k:]
        top_r, top_c = np.unravel_index(top_k_idx, E2.shape)
        rows_out = top_r.astype(np.int32)
        cols_out = top_c.astype(np.int32)
        vals_out = flat_e2[top_k_idx].astype(np.float16)

    return L_out, R_out, rows_out, cols_out, vals_out


# ── Public compress / decompress ──────────────────────────────────────────────


def compress_weight(W: np.ndarray, cfg: SQINT2Config | None = None) -> SQINT2Layer:
    """Compress a 2-D weight matrix with SQINT2 Stage 1+2.

    Args:
        W:    float32 ndarray of shape (out_features, in_features).
        cfg:  SQINT2Config; defaults to group_size=32, refine_iters=1, seed=42.

    Returns:
        SQINT2Layer with packed indices, per-group scales + zero-points,
        and the seed needed to reconstruct H_left and H_right at decompress.

    Pipeline:
        1. Pad in_features and out_features up to the next multiple of
           group_size and 4 respectively (Hadamard needs power-of-two for
           Walsh–Hadamard; the QR fallback handles the rest).
        2. Build H_left (out × out) and H_right (in × in) from cfg.seed.
        3. W_rot = H_left · W · H_rightᵀ in float32.
        4. NF2 per-group asymmetric quantise W_rot → indices, scales, zps.
        5. Pack indices to uint8.
    """
    if cfg is None:
        cfg = SQINT2Config()
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (out, in), got ndim={W.ndim}")

    out_features, in_features = W.shape

    # Pad columns to multiple of group_size so groups divide evenly.
    in_padded = _round_up(in_features, cfg.group_size)
    out_padded = out_features  # rows are not grouped; no row padding needed
    pad_in = in_padded - in_features

    if pad_in > 0:
        W_padded = np.pad(
            W.astype(np.float32, copy=False),
            ((0, 0), (0, pad_in)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        W_padded = W.astype(np.float32, copy=False)

    # Build Hadamard rotations. Both seeds derive from cfg.seed so a single
    # int suffices to reproduce them at decompress time.
    rng_left = np.random.default_rng(cfg.seed)
    rng_right = np.random.default_rng(cfg.seed + 1)
    H_left = build_hadamard(out_padded, rng_left) if cfg.rotate_left else None
    H_right = build_hadamard(in_padded, rng_right) if cfg.rotate_right else None

    # Stage 1: rotate.
    W_rot = apply_hadamard(W_padded, H_left, H_right)

    # Stage 2: NF2 per-group asymmetric quantise.
    n_groups = in_padded // cfg.group_size
    grouped = W_rot.reshape(out_padded * n_groups, cfg.group_size)
    indices_flat, scales_flat, zps_flat = _nf2_quantise_groups(
        grouped, refine_iters=cfg.refine_iters
    )

    # Reshape and pack.
    indices = indices_flat.reshape(out_padded, in_padded)
    scales = scales_flat.reshape(out_padded, n_groups)
    zero_points = zps_flat.reshape(out_padded, n_groups)
    packed = _pack_2bit(indices)

    # Stage 3: low-rank SVD + sparse residual correction (W103.2).
    res_L = res_R = sp_rows = sp_cols = sp_vals = None
    if cfg.residual_rank > 0 or cfg.sparse_frac > 0.0:
        # Reconstruct dequant_rot from quantised indices to get the error E.
        nf2_flat = NF2_VALUES[indices_flat.astype(np.intp)]          # (M, gs) fp32
        dq_flat = (nf2_flat - zps_flat[:, None]) * scales_flat[:, None]
        dequant_rot = dq_flat.reshape(out_padded, in_padded)
        E = W_rot - dequant_rot
        res_L, res_R, sp_rows, sp_cols, sp_vals = _compute_stage3_residual(E, cfg)

    return SQINT2Layer(
        indices=packed,
        scales=scales,
        zero_points=zero_points,
        in_features=in_features,
        out_features=out_features,
        cfg=cfg,
        residual_L=res_L,
        residual_R=res_R,
        sparse_rows=sp_rows,
        sparse_cols=sp_cols,
        sparse_vals=sp_vals,
    )


def decompress_weight(layer: SQINT2Layer) -> np.ndarray:
    """Reconstruct a float32 weight matrix from a SQINT2Layer.

    Pipeline (inverse of compress_weight):
        1. Unpack 2-bit indices to uint8.
        2. Look up NF2_VALUES → rescaled float32.
        3. Per-group inverse asymmetric: x_rot = (NF2 - zp) * scale.
        3b. [Stage 3] Add L·R SVD correction to rotated reconstruction.
        3c. [Stage 3] Add sparse COO corrections to rotated reconstruction.
        4. Inverse Hadamard: W_recon = H_leftᵀ · W_rot_corrected · H_right.
        5. Strip column padding back to original in_features.

    The output is the float32 reconstruction of the original W; exact
    reconstruction up to NF2 quantisation error (Stage 1+2) plus residual
    storage error (Stage 3, ~1e-4 on fp16 factors) plus ~1e-7 rotation roundoff.
    """
    cfg = layer.cfg
    in_padded = _round_up(layer.in_features, cfg.group_size)
    out_padded = layer.out_features
    n_groups = in_padded // cfg.group_size

    # Step 1+2: unpack + NF2 lookup.
    indices = _unpack_2bit(layer.indices, in_padded)            # (out, in_padded)
    rescaled = NF2_VALUES[indices.astype(np.intp)]              # (out, in_padded)

    # Step 3: per-group inverse asymmetric.
    rescaled_3d = rescaled.reshape(out_padded, n_groups, cfg.group_size)
    scale_3d = layer.scales[:, :, None]
    zp_3d = layer.zero_points[:, :, None]
    W_rot = ((rescaled_3d - zp_3d) * scale_3d).reshape(out_padded, in_padded)

    # Step 3b: SVD residual correction in rotated frame.
    if layer.residual_L is not None and layer.residual_R is not None:
        L_fp32 = _unpack_factor(layer.residual_L)   # (out_padded, rank)
        R_fp32 = _unpack_factor(layer.residual_R)   # (rank, in_padded)
        W_rot = W_rot + L_fp32 @ R_fp32

    # Step 3c: sparse COO correction in rotated frame.
    if layer.sparse_rows is not None:
        W_rot = W_rot.copy()  # avoid in-place mutation of a view
        W_rot[layer.sparse_rows, layer.sparse_cols] += (
            layer.sparse_vals.astype(np.float32)
        )

    # Step 4: inverse Hadamard. Re-derive H from cfg.seed deterministically.
    rng_left = np.random.default_rng(cfg.seed)
    rng_right = np.random.default_rng(cfg.seed + 1)
    H_left = build_hadamard(out_padded, rng_left) if cfg.rotate_left else None
    H_right = build_hadamard(in_padded, rng_right) if cfg.rotate_right else None
    W_recon_padded = inverse_hadamard(W_rot, H_left, H_right)

    # Step 5: strip column padding.
    return W_recon_padded[:, : layer.in_features].astype(np.float32, copy=False)


# ── SNR helper ────────────────────────────────────────────────────────────────


def snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    """Signal-to-noise ratio in decibels: 10·log10(mean(s²) / mean((s-r)²)).

    Used by the W103.1 unit-test gate (≥ 12 dB on synthetic σ=0.02 weights).
    Returns +inf for exact reconstruction, raises if signal is identically zero.
    """
    if signal.shape != recon.shape:
        raise ValueError(
            f"shape mismatch: signal {signal.shape} vs recon {recon.shape}"
        )
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    sig_power = float(np.mean(s * s))
    if sig_power == 0.0:
        raise ValueError("signal power is zero — SNR is undefined")
    noise = s - r
    noise_power = float(np.mean(noise * noise))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(sig_power / noise_power))


# ── W103.3 — per-tensor compress pipeline ─────────────────────────────────────
#
# compress_weights_sqint2() is the core W103.3 dispatch: it routes every tensor
# in a weight dict through MixedPrecisionRouter and applies the correct codec.
# Pure in-memory (no filesystem I/O) so it is unit-testable with synthetic dicts.
#
# Helper codecs (INT3, INT4) are pure-NumPy so the pipeline runs without the
# squish_quant Rust extension.  At W103.4, the Metal inference path will pick up
# the uint8 INT3 codes from npy-dir and bit-pack them inside the Metal shader.
# ──────────────────────────────────────────────────────────────────────────────


def _npy_safe_key(tensor_name: str) -> str:
    """Convert a dotted tensor name to a filesystem-safe key.

    Identical to ``squish.convert.safe_key()``, lifted here to avoid an
    import cycle between squish.quant.sqint2 and squish.convert at import time.
    """
    return tensor_name.replace(".", "__")


def _int3_quantize_numpy(
    W: np.ndarray, group_size: int = 32
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Pure-NumPy asymmetric INT3 group quantization (8 levels, codes 0–7).

    Convention (matches INT3Linear in squish/quant/int3_linear.py):
        scale = (xmax - xmin) / 7.0     per group
        zero  = xmin                     per group
        code  = round(clamp((x - zero) / scale, 0, 7))
        decode: w = code * scale + zero

    Storage: uint8 codes (one byte per weight — no bit-packing at this stage).
    INT3Linear bit-packs at load time via _pack_codes_uint32; W103.4 Metal path
    does the same.

    Args:
        W:          float32, shape (out_features, in_features).
        group_size: columns per quantization group; must divide in_features.

    Returns:
        codes:  uint8 (out_features, in_features), values in [0, 7]
        scales: float32 (out_features, n_groups)
        zeros:  float32 (out_features, n_groups)   — the per-group xmin offset
    """
    M, N = W.shape
    in_pad = _round_up(N, group_size)
    W_pad = (np.pad(W.astype(np.float32), ((0, 0), (0, in_pad - N)))
             if in_pad > N else W.astype(np.float32))
    n_groups = in_pad // group_size
    grouped = W_pad.reshape(M * n_groups, group_size)

    xmin = grouped.min(axis=1)
    xmax = grouped.max(axis=1)
    span = xmax - xmin
    scale = np.where(span > 0, span / 7.0, 1.0).astype(np.float32)
    zero = np.where(span > 0, xmin, 0.0).astype(np.float32)

    rescaled = np.clip((grouped - zero[:, None]) / scale[:, None], 0.0, 7.0)
    codes = np.round(rescaled).astype(np.uint8)

    return (
        codes.reshape(M, in_pad)[:, :N],
        scale.reshape(M, n_groups),
        zero.reshape(M, n_groups),
    )


def _int3_dequantize_numpy(
    codes: np.ndarray, scales: np.ndarray, zeros: np.ndarray
) -> np.ndarray:
    """Inverse of _int3_quantize_numpy. decode: w = code * scale + zero."""
    M = codes.shape[0]
    in_features = codes.shape[1]
    n_groups = scales.shape[1]
    gs = in_features // n_groups if n_groups > 0 else in_features
    codes_f = codes.astype(np.float32)
    if n_groups > 0 and in_features == n_groups * gs:
        codes_3d = codes_f.reshape(M, n_groups, gs)
        recon = (codes_3d * scales[:, :, None] + zeros[:, :, None]).reshape(M, in_features)
    else:
        recon = codes_f * scales[:, :1] + zeros[:, :1]
    return recon.astype(np.float32)


def _int4_quantize_numpy(
    W: np.ndarray, group_size: int = 32
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Pure-NumPy asymmetric INT4 group quantization (16 levels, nibble-packed).

    Convention:
        scale = (xmax - xmin) / 15.0    per group
        offset= xmin                     per group (= gmin)
        code  = round(clamp((x - offset) / scale, 0, 15))
        decode: w = offset + code * scale

    Storage: nibble-packed uint8, shape (out, in // 2). Low nibble first.
    Same layout as squish_quant Rust INT4 output (__q4a suffix).

    Args:
        W:          float32, shape (out_features, in_features).
        group_size: columns per group; must divide in_features.

    Returns:
        packed:  uint8 (out_features, in_features // 2)
        scales:  float32 (out_features, n_groups)
        offsets: float32 (out_features, n_groups)   — per-group xmin
    """
    M, N = W.shape
    in_pad = _round_up(N, max(group_size, 2))  # nibble-pack needs even N
    W_pad = (np.pad(W.astype(np.float32), ((0, 0), (0, in_pad - N)))
             if in_pad > N else W.astype(np.float32))
    n_groups = in_pad // group_size
    grouped = W_pad.reshape(M * n_groups, group_size)

    xmin = grouped.min(axis=1)
    xmax = grouped.max(axis=1)
    span = xmax - xmin
    scale = np.where(span > 0, span / 15.0, 1.0).astype(np.float32)
    offset = np.where(span > 0, xmin, 0.0).astype(np.float32)

    rescaled = np.clip((grouped - offset[:, None]) / scale[:, None], 0.0, 15.0)
    codes = np.round(rescaled).astype(np.uint8).reshape(M, in_pad)[:, :N]

    # Nibble-pack: two 4-bit codes per byte, low nibble first.
    N_even = N + (N % 2)
    if N_even > N:
        codes = np.pad(codes, ((0, 0), (0, 1)))
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)

    return (
        packed.astype(np.uint8),
        scale.reshape(M, n_groups),
        offset.reshape(M, n_groups),
    )


def _int4_dequantize_numpy(
    packed: np.ndarray, scales: np.ndarray, offsets: np.ndarray, in_features: int
) -> np.ndarray:
    """Inverse of _int4_quantize_numpy. decode: w = offset + code * scale."""
    M = packed.shape[0]
    n_groups = scales.shape[1]
    # Unpack nibbles
    lo = (packed & 0x0F).astype(np.float32)
    hi = ((packed >> 4) & 0x0F).astype(np.float32)
    codes = np.empty((M, packed.shape[1] * 2), dtype=np.float32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    codes = codes[:, :in_features]
    gs = in_features // n_groups if n_groups > 0 else in_features
    if n_groups > 0 and in_features == n_groups * gs:
        codes_3d = codes.reshape(M, n_groups, gs)
        recon = (offsets[:, :, None] + codes_3d * scales[:, :, None]).reshape(M, in_features)
    else:
        recon = offsets[:, :1] + codes * scales[:, :1]
    return recon.astype(np.float32)


def compress_weights_sqint2(
    weights: "dict[str, np.ndarray]",
    n_layers: int,
    sqint2_cfg: "SQINT2Config | None" = None,
    int3_group_size: int = 32,
    int4_group_size: int = 32,
) -> "Tuple[dict[str, np.ndarray], dict[str, str], dict[str, int]]":
    """Route and compress all model weights using W103.3 mixed precision.

    Pure in-memory: reads from *weights* dict, returns compressed arrays ready
    for npy-dir writing.  No filesystem access — caller supplies weights and
    writes the output.

    Format dispatch via MixedPrecisionRouter(n_layers):
        "sqint2" (gate_proj, up_proj, non-boundary) →
            __sqint2_idx.npy  uint8  packed 2-bit INT2 indices
            __sqint2_scales   float32 per-group scales
            __sqint2_zp       float32 per-group zero-points
            __sqint2_L        fp16 SVD left factor  (if residual_rank > 0)
            __sqint2_R        fp16 SVD right factor (if residual_rank > 0)
            __sqint2_srows    int32 sparse row indices  (if sparse_frac > 0)
            __sqint2_scols    int32 sparse col indices  (if sparse_frac > 0)
            __sqint2_svals    fp16 sparse values        (if sparse_frac > 0)
            __shape           int64 original shape
        "int3" (q/k/v/o_proj, non-boundary) →
            __q3  uint8 codes [0,7], decode: w = code * scale + zero
            __s3  float32 per-group scales
            __z3  float32 per-group zero-point offsets (xmin)
            __shape
        "int4" (boundary layers, down_proj, norms) →
            __q4  uint8 nibble-packed [0,15], decode: w = offset + code * scale
            __s4  float32 per-group scales
            __z4  float32 per-group offsets (xmin)
            __shape
        None (embed_tokens, lm_head, 1D tensors, non-.weight) →
            __pt  float16 passthrough
            __shape

    Args:
        weights:        Dict mapping tensor name → float32/float16/bfloat16 ndarray.
        n_layers:       Total number of transformer decoder blocks.
        sqint2_cfg:     SQINT2Config for the Stage 1+2+3 path. Defaults to
                        group_size=32, refine_iters=2, residual_rank=16,
                        sparse_frac=0.01 (production SQINT2 settings).
        int3_group_size: Group size for the INT3 tier (default 32, matches validated
                         mlx_lm.convert baseline).
        int4_group_size: Group size for the INT4 tier (default 32).

    Returns:
        arrays:     Flat dict ``{"{safe_key}{suffix}": numpy_array}`` ready for
                    ``squish.convert.write_npy_dir(output_dir, arrays, manifest)``.
        manifest:   ``{original_name: safe_key}`` mapping for manifest.json.
        fmt_counts: ``{"sqint2": N, "int3": N, "int4": N, "skip": N}`` summary.
    """
    from squish.quant.quantizer import MixedPrecisionRouter

    if sqint2_cfg is None:
        sqint2_cfg = SQINT2Config(
            group_size=32,
            refine_iters=2,
            residual_rank=16,
            sparse_frac=0.01,
        )

    router = MixedPrecisionRouter(n_layers)
    arrays: "dict[str, np.ndarray]" = {}
    manifest: "dict[str, str]" = {}
    fmt_counts: "dict[str, int]" = {"sqint2": 0, "int3": 0, "int4": 0, "skip": 0}

    for name, tensor in weights.items():
        sk = _npy_safe_key(name)
        manifest[name] = sk
        arr = tensor.astype(np.float32)
        shape_arr = np.array(arr.shape, dtype=np.int64)

        fmt = router.format_for(name)

        # Non-quantizable: 1D tensors always pass through regardless of router verdict.
        if arr.ndim < 2:
            fmt = None

        arrays[f"{sk}__shape"] = shape_arr

        if fmt == "sqint2":
            layer = compress_weight(arr, sqint2_cfg)
            arrays[f"{sk}__sqint2_idx"] = layer.indices
            arrays[f"{sk}__sqint2_scales"] = layer.scales
            arrays[f"{sk}__sqint2_zp"] = layer.zero_points
            # Per-layer meta header (W103.4a). Encodes cfg.seed, which
            # `load_sqint2_layer` needs to reconstruct the Hadamard rotation
            # deterministically — without it, decompression rebuilds H
            # against the default seed and produces incoherent output.
            arrays[f"{sk}__sqint2_meta"] = _meta_array(layer)
            if layer.residual_L is not None:
                arrays[f"{sk}__sqint2_L"] = layer.residual_L
                arrays[f"{sk}__sqint2_R"] = layer.residual_R
            if layer.sparse_rows is not None:
                arrays[f"{sk}__sqint2_srows"] = layer.sparse_rows
                arrays[f"{sk}__sqint2_scols"] = layer.sparse_cols
                arrays[f"{sk}__sqint2_svals"] = layer.sparse_vals
            fmt_counts["sqint2"] += 1

        elif fmt == "int3":
            codes, scales3, zeros3 = _int3_quantize_numpy(arr, int3_group_size)
            arrays[f"{sk}__q3"] = codes
            arrays[f"{sk}__s3"] = scales3
            arrays[f"{sk}__z3"] = zeros3
            fmt_counts["int3"] += 1

        elif fmt == "int4":
            packed4, scales4, offsets4 = _int4_quantize_numpy(arr, int4_group_size)
            arrays[f"{sk}__q4"] = packed4
            arrays[f"{sk}__s4"] = scales4
            arrays[f"{sk}__z4"] = offsets4
            fmt_counts["int4"] += 1

        else:  # None — passthrough
            arrays[f"{sk}__pt"] = arr.astype(np.float16)
            fmt_counts["skip"] += 1

    return arrays, manifest, fmt_counts


# ── W103.4a — Disk serialization (npy-dir format) ─────────────────────────────
#
# A SQINT2Layer is persisted as a small set of `.npy` files sharing a common
# safe-key prefix `{sk}` and a `__sqint2_*` suffix. Detection in
# `squish/quant/compressed_loader.py` triggers on `{sk}__sqint2_idx.npy`.
#
#   {sk}__sqint2_idx.npy        uint8   (out, in_padded // 4)   packed indices
#   {sk}__sqint2_scales.npy     fp32    (out, n_groups)         per-group scales
#   {sk}__sqint2_zp.npy         fp32    (out, n_groups)         per-group zp
#   {sk}__sqint2_meta.npy       fp64    (16,)                   header — see below
#   {sk}__sqint2_L.npy       optional fp16/fp32 (out_pad, r) SVD left factor
#   {sk}__sqint2_R.npy       optional fp16/fp32 (r, in_pad)  SVD right factor
#   {sk}__sqint2_srows.npy      optional int32  (k,)            COO rows
#   {sk}__sqint2_scols.npy      optional int32  (k,)            COO cols
#   {sk}__sqint2_svals.npy      optional fp16   (k,)            COO values
#
# Meta layout (float64 to keep one homogeneous array — config ints fit exactly
# up to 2^53; using fp64 also accommodates `sparse_frac`):
#
#   [0]  format version (= SQINT2_FORMAT_VERSION)
#   [1]  in_features
#   [2]  out_features
#   [3]  group_size
#   [4]  seed
#   [5]  refine_iters
#   [6]  rotate_left   (0.0 or 1.0)
#   [7]  rotate_right  (0.0 or 1.0)
#   [8]  residual_rank
#   [9]  residual_factor_dtype_code (16.0 = fp16, 32.0 = fp32)
#   [10] sparse_frac
#   [11..15] reserved (zero)
#
# Forward compatibility: readers must check meta[0] and refuse versions newer
# than the highest they support. Future fields may use the reserved slots.
# ──────────────────────────────────────────────────────────────────────────────

SQINT2_FORMAT_VERSION: float = 1.0

# Tuple of suffixes (without the leading `{sk}`) — useful for tests and
# discovery code that needs to enumerate or clean up SQINT2 layer files.
SQINT2_SUFFIXES: Tuple[str, ...] = (
    "__sqint2_idx.npy",
    "__sqint2_scales.npy",
    "__sqint2_zp.npy",
    "__sqint2_meta.npy",
    "__sqint2_L.npy",
    "__sqint2_R.npy",
    "__sqint2_srows.npy",
    "__sqint2_scols.npy",
    "__sqint2_svals.npy",
)

_DTYPE_CODE_TO_NAME = {16.0: "fp16", 32.0: "fp32"}
_DTYPE_NAME_TO_CODE = {v: k for k, v in _DTYPE_CODE_TO_NAME.items()}


def _meta_array(layer: "SQINT2Layer") -> np.ndarray:
    cfg = layer.cfg
    meta = np.zeros(16, dtype=np.float64)
    meta[0] = SQINT2_FORMAT_VERSION
    meta[1] = float(layer.in_features)
    meta[2] = float(layer.out_features)
    meta[3] = float(cfg.group_size)
    meta[4] = float(cfg.seed)
    meta[5] = float(cfg.refine_iters)
    meta[6] = 1.0 if cfg.rotate_left else 0.0
    meta[7] = 1.0 if cfg.rotate_right else 0.0
    meta[8] = float(cfg.residual_rank)
    meta[9] = float(_DTYPE_NAME_TO_CODE[cfg.residual_factor_dtype])
    meta[10] = float(cfg.sparse_frac)
    return meta


def save_sqint2_layer(layer: "SQINT2Layer", tensor_dir, safe_key: str) -> None:
    """Persist a SQINT2Layer to disk as a set of `.npy` files.

    Writes the four mandatory files (`idx`, `scales`, `zp`, `meta`) plus any
    optional residual / sparse files present on the layer. The directory must
    already exist; tensors are written with `np.save` (no compression — these
    files are mmap'd at load time for fast cold-start, which precludes zip
    compression at the npy-file level).

    Args:
        layer: SQINT2Layer to serialise.
        tensor_dir: pathlib.Path-like directory that will receive the files.
        safe_key: filesystem-safe tensor key (the `{sk}` prefix; see
                  `squish/quant/compressed_loader.py:_safe_key_to_original`).
    """
    tensor_dir = Path(tensor_dir)
    if not tensor_dir.is_dir():
        raise FileNotFoundError(f"tensor_dir does not exist: {tensor_dir}")
    if "/" in safe_key or "\\" in safe_key:
        raise ValueError(
            f"safe_key must not contain path separators, got {safe_key!r}"
        )

    indices = np.ascontiguousarray(layer.indices, dtype=np.uint8)
    scales = np.ascontiguousarray(layer.scales, dtype=np.float32)
    zero_points = np.ascontiguousarray(layer.zero_points, dtype=np.float32)

    np.save(tensor_dir / f"{safe_key}__sqint2_idx.npy", indices)
    np.save(tensor_dir / f"{safe_key}__sqint2_scales.npy", scales)
    np.save(tensor_dir / f"{safe_key}__sqint2_zp.npy", zero_points)
    np.save(tensor_dir / f"{safe_key}__sqint2_meta.npy", _meta_array(layer))

    has_lowrank = layer.residual_L is not None and layer.residual_R is not None
    if has_lowrank:
        factor_dtype = np.float16 if layer.cfg.residual_factor_dtype == "fp16" else np.float32
        L = np.ascontiguousarray(layer.residual_L, dtype=factor_dtype)
        R = np.ascontiguousarray(layer.residual_R, dtype=factor_dtype)
        np.save(tensor_dir / f"{safe_key}__sqint2_L.npy", L)
        np.save(tensor_dir / f"{safe_key}__sqint2_R.npy", R)

    has_sparse = (
        layer.sparse_rows is not None
        and layer.sparse_cols is not None
        and layer.sparse_vals is not None
    )
    if has_sparse:
        np.save(
            tensor_dir / f"{safe_key}__sqint2_srows.npy",
            np.ascontiguousarray(layer.sparse_rows, dtype=np.int32),
        )
        np.save(
            tensor_dir / f"{safe_key}__sqint2_scols.npy",
            np.ascontiguousarray(layer.sparse_cols, dtype=np.int32),
        )
        np.save(
            tensor_dir / f"{safe_key}__sqint2_svals.npy",
            np.ascontiguousarray(layer.sparse_vals, dtype=np.float16),
        )


def load_sqint2_layer(tensor_dir, safe_key: str) -> "SQINT2Layer":
    """Reconstruct a SQINT2Layer from on-disk `.npy` files.

    Inverse of :func:`save_sqint2_layer`. Validates the format version,
    reconstructs `SQINT2Config` from the meta header, and attaches optional
    residual / sparse arrays when their files are present.

    Args:
        tensor_dir: directory containing the `{safe_key}__sqint2_*.npy` files.
        safe_key:   safe-key prefix.

    Returns:
        SQINT2Layer with cfg, indices, scales, zero_points populated. Residual
        and sparse fields are populated when their files are present and zero
        otherwise (matching the contract of :func:`compress_weight`).
    """
    tensor_dir = Path(tensor_dir)

    idx_path = tensor_dir / f"{safe_key}__sqint2_idx.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"missing SQINT2 idx file: {idx_path}. A SQINT2 layer requires "
            f"`__sqint2_idx.npy`, `__sqint2_scales.npy`, `__sqint2_zp.npy`, "
            f"and `__sqint2_meta.npy`."
        )
    meta_path = tensor_dir / f"{safe_key}__sqint2_meta.npy"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing SQINT2 meta file: {meta_path}")

    meta = np.load(meta_path)
    if meta.shape != (16,):
        raise ValueError(
            f"SQINT2 meta has unexpected shape {meta.shape}, expected (16,)"
        )
    version = float(meta[0])
    if version > SQINT2_FORMAT_VERSION:
        raise ValueError(
            f"SQINT2 file format version {version} is newer than this build "
            f"supports ({SQINT2_FORMAT_VERSION}). Upgrade `squish`."
        )
    if version < 1.0:
        raise ValueError(f"SQINT2 file format version {version} is invalid (< 1.0)")

    in_features = int(meta[1])
    out_features = int(meta[2])
    dtype_code = float(meta[9])
    if dtype_code not in _DTYPE_CODE_TO_NAME:
        raise ValueError(
            f"unknown residual_factor_dtype_code {dtype_code} in meta"
        )

    cfg = SQINT2Config(
        group_size=int(meta[3]),
        seed=int(meta[4]),
        refine_iters=int(meta[5]),
        rotate_left=bool(meta[6]),
        rotate_right=bool(meta[7]),
        residual_rank=int(meta[8]),
        residual_factor_dtype=_DTYPE_CODE_TO_NAME[dtype_code],
        sparse_frac=float(meta[10]),
    )

    indices = np.load(idx_path)
    scales = np.load(tensor_dir / f"{safe_key}__sqint2_scales.npy")
    zero_points = np.load(tensor_dir / f"{safe_key}__sqint2_zp.npy")

    if indices.dtype != np.uint8:
        raise ValueError(f"SQINT2 indices must be uint8, got {indices.dtype}")
    if scales.dtype != np.float32:
        scales = scales.astype(np.float32, copy=False)
    if zero_points.dtype != np.float32:
        zero_points = zero_points.astype(np.float32, copy=False)

    in_padded = _round_up(in_features, cfg.group_size)
    expected_idx_shape = (out_features, in_padded // 4)
    if indices.shape != expected_idx_shape:
        raise ValueError(
            f"SQINT2 indices shape {indices.shape} does not match meta "
            f"(out_features={out_features}, in_padded//4={in_padded // 4}); "
            f"expected {expected_idx_shape}"
        )

    resL_path = tensor_dir / f"{safe_key}__sqint2_L.npy"
    resR_path = tensor_dir / f"{safe_key}__sqint2_R.npy"
    residual_L = np.load(resL_path) if resL_path.exists() else None
    residual_R = np.load(resR_path) if resR_path.exists() else None
    if (residual_L is None) != (residual_R is None):
        raise ValueError(
            "SQINT2 residual factors are partially present — both `__sqint2_L.npy` "
            "and `__sqint2_R.npy` must be saved together (or neither)."
        )

    srows_path = tensor_dir / f"{safe_key}__sqint2_srows.npy"
    scols_path = tensor_dir / f"{safe_key}__sqint2_scols.npy"
    svals_path = tensor_dir / f"{safe_key}__sqint2_svals.npy"
    sparse_present = [p.exists() for p in (srows_path, scols_path, svals_path)]
    if any(sparse_present) and not all(sparse_present):
        raise ValueError(
            "SQINT2 sparse triplet is incomplete — `srows`, `scols`, `svals` "
            "must all be present together (or none)."
        )
    if all(sparse_present):
        sparse_rows = np.load(srows_path).astype(np.int32, copy=False)
        sparse_cols = np.load(scols_path).astype(np.int32, copy=False)
        sparse_vals = np.load(svals_path).astype(np.float16, copy=False)
        if not (sparse_rows.shape == sparse_cols.shape == sparse_vals.shape):
            raise ValueError(
                f"SQINT2 sparse triplet shape mismatch: rows={sparse_rows.shape}, "
                f"cols={sparse_cols.shape}, vals={sparse_vals.shape}"
            )
    else:
        sparse_rows = sparse_cols = sparse_vals = None

    return SQINT2Layer(
        indices=indices,
        scales=scales,
        zero_points=zero_points,
        in_features=in_features,
        out_features=out_features,
        cfg=cfg,
        residual_L=residual_L,
        residual_R=residual_R,
        sparse_rows=sparse_rows,
        sparse_cols=sparse_cols,
        sparse_vals=sparse_vals,
    )
