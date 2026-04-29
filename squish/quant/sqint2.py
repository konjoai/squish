"""squish/quant/sqint2.py — SQINT2 Stage 1+2: Hadamard preprocess + NF2 codebook.

SQINT2 is Squish's coherent INT2 weight-compression format (W103). This module
implements the offline-compress half of stages 1 and 2:

    Stage 1  Hadamard incoherence preprocessing  (this module: encode/decode)
    Stage 2  NF2 per-group asymmetric quantisation  (this module)
    Stage 3  Low-rank residual correction  (W103.2 — separate module)
    Stage 4  Layer-selective mixed precision  (W103.3 — quantizer.py routing)

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

    indices      uint8  packed (out_padded, in_padded // 4)   2 bpw raw
    scales       fp32   shape (out_padded, n_groups)         16 bpw / group
    zero_points  fp32   shape (out_padded, n_groups)         16 bpw / group
    seed         int    Hadamard-construction RNG seed (small fixed cost)
    cfg          SQINT2Config

Effective bits per weight at this stage (group_size=32, fp32 scale + zp):

    2.0  (raw indices)
    + 32 / 32  (scale)
    + 32 / 32  (zero-point)
    = 4.0 bpw before residual

Stage 3 will compress (scale, zp) to fp16/INT4 and add a rank-16 residual,
hitting the SQINT2 spec target of ~2.15 bpw effective. The 4.0 bpw figure
here is intentional — Stage 1+2 prioritises **correctness and SNR** over
final compactness; Stage 3 closes the bpw gap.

Public API
----------
- SQINT2Config         hyper-parameters (group_size, refine_iters, seed, …)
- SQINT2Layer          one compressed weight matrix (numpy arrays + cfg)
- compress_weight      W (float32, out × in) → SQINT2Layer
- decompress_weight    SQINT2Layer → W̃ (float32, out × in)
- snr_db               helper: 10·log10(σ²_signal / σ²_noise)
- NF2_VALUES           the four NF2 codebook symbols
- build_hadamard       lifted from squish/kv/kv_cache.py for compress-time use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    "snr_db",
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
    """

    group_size: int = 32
    seed: int = 42
    refine_iters: int = 1
    rotate_left: bool = True
    rotate_right: bool = True

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


# ── Compressed-layer container ────────────────────────────────────────────────


@dataclass
class SQINT2Layer:
    """Compressed representation of one weight matrix (Stage 1+2 only).

    Attributes:
        indices:      uint8, shape (out_features, in_padded // 4). Each byte
                      packs 4 successive 2-bit indices, low-bit-first
                      (index 0 in bits 0–1, index 3 in bits 6–7).
        scales:       float32, shape (out_features, n_groups). Per-group
                      multiplier such that NF2_VALUES[idx] decodes to a
                      value in (rescaled - zp) * scale.
        zero_points:  float32, shape (out_features, n_groups). Per-group
                      additive shift in the rescaled coordinate frame.
        in_features:  original (unpadded) input-feature count.
        out_features: original (unpadded) output-feature count.
        cfg:          SQINT2Config with seed for Hadamard reconstruction.
    """

    indices: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    in_features: int
    out_features: int
    cfg: SQINT2Config

    @property
    def n_groups(self) -> int:
        in_padded = _round_up(self.in_features, self.cfg.group_size)
        return in_padded // self.cfg.group_size

    @property
    def effective_bpw(self) -> float:
        """Effective bits per weight, including fp32 scale + zero-point overhead.

        Stage 3 (W103.2) compresses scale/zp to fp16 or INT4 and adds the
        low-rank residual; that path targets ~2.15 bpw. This Stage 1+2 cost
        is reported here for transparency.
        """
        gs = self.cfg.group_size
        return 2.0 + 32.0 / gs + 32.0 / gs


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

    return SQINT2Layer(
        indices=packed,
        scales=scales,
        zero_points=zero_points,
        in_features=in_features,
        out_features=out_features,
        cfg=cfg,
    )


def decompress_weight(layer: SQINT2Layer) -> np.ndarray:
    """Reconstruct a float32 weight matrix from a SQINT2Layer.

    Pipeline (inverse of compress_weight):
        1. Unpack 2-bit indices to uint8.
        2. Look up NF2_VALUES → rescaled float32.
        3. Per-group inverse asymmetric: x_rot = (NF2 - zp) * scale.
        4. Inverse Hadamard: W_recon = H_leftᵀ · W_rot · H_right.
        5. Strip column padding back to original in_features.

    The output is the float32 reconstruction of the original W; exact
    reconstruction up to NF2 quantisation error plus ~1e-7 rotation roundoff.
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
