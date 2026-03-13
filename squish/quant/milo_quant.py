"""
squish/milo_quant.py

MiLo — Mixture of Low-Rank Compensators for INT3 MoE Inference.

Based on:
  "MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank
  Compensators"
  arXiv:2504.02658 (MLSys 2025)
  github.com/Supercomputing-System-AI-Lab/MiLo

Problem
-------
INT4 quantization fails on Mixture-of-Experts models for a fundamental reason:
MoE weight distributions are bimodal — expert weights and router weights have
very different statistics, but INT4 quantizes them with the same scheme.  The
per-expert weight variance is also higher than in dense models because each
expert specialises on a subset of the input distribution.

INT3 naive quantization makes this worse: 3 bits (8 levels) vs 4 bits (16
levels) doubles the quantization step size, significantly increasing round-off
error on outlier expert weights.

MiLo Solution: Quantize-Then-Compensate
-----------------------------------------
1. **Quantize aggressively to INT3** using calibration-free GPTQ (fast,
   no data required).
2. **Compute residual error** per weight matrix:
   ``Error = W_FP32 - W_INT3``
3. **Decompose residual into low-rank form**:
   ``Error ≈ A × B``   where A ∈ ℝ^{m×r}, B ∈ ℝ^{r×n}, rank r ≪ min(m,n)
4. **At inference**, correct output:
   ``output = W_INT3 @ input + B @ (A @ input)``

The rank is selected *adaptively* per weight matrix:
  - Dense router weights (small variance) → low rank (r ≈ 4)
  - Active expert weights (high variance) → higher rank (r ≈ 16)
  - Rank is chosen to meet a target reconstruction SNR threshold

MiLo on Dense Models (Qwen3-8B INT4)
--------------------------------------
The Quantize-Then-Compensate principle is not exclusive to MoE.  Applied to
dense INT4 models, low-rank compensators recover accuracy lost in quantization
for the most error-prone weight matrices (attention projection layers, which
have highest activation variance).  MiLo + INT4 is effectively GPTQ with
structured low-rank error correction.

Integer Representation
-----------------------
This module stores INT3 weights packed into ``uint8`` bytes using 3 bits per
value (zero-waste packing):
  - 8 values per 3 bytes (24 bits / 3 bits = 8 values)
  - ``pack_int3(arr)`` → ``uint8[ceil(n/8)*3]``
  - ``unpack_int3(packed, n)`` → ``float32[n]``

Provides
--------
  MiLoConfig            — configuration parameters
  LowRankCompensator    — stores A, B matrices; applies correction
  MiLoQuantizer         — quantize + compensate for one weight matrix
  MiLoStats             — accuracy and compression tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MiLoConfig",
    "LowRankCompensator",
    "MiLoQuantizer",
    "MiLoStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MiLoConfig:
    """Configuration for MiLo quantization.

    Parameters
    ----------
    target_bits:
        Target quantization bit-width for weight matrices.  Default 3 (INT3).
        Use 4 for INT4 + compensator (denser accuracy recovery on Qwen3-8B).
    max_rank:
        Maximum low-rank compensator rank.  Default 16.
    min_rank:
        Minimum rank, even for low-variance matrices.  Default 2.
    snr_threshold_db:
        Target reconstruction SNR in dB.  The rank is increased until the
        SNR of the reconstructed weight (W_INT3 + A×B) versus W_FP32 meets
        this threshold, up to ``max_rank``.  Default 40 dB.
    group_size:
        INT3/INT4 quantization group size (values sharing one scale + zero).
        Default 64 (matches Squish INT4 group convention).
    adaptive_rank:
        If True, pick rank per matrix to hit ``snr_threshold_db``.
        If False, always use ``max_rank``.  Default True.
    """

    target_bits:      int   = 3
    max_rank:         int   = 16
    min_rank:         int   = 2
    snr_threshold_db: float = 40.0
    group_size:       int   = 64
    adaptive_rank:    bool  = True

    def __post_init__(self) -> None:
        if self.target_bits not in (3, 4, 8):
            raise ValueError(
                f"target_bits must be 3, 4, or 8, got {self.target_bits}"
            )
        if self.min_rank < 1:
            raise ValueError(f"min_rank must be >= 1, got {self.min_rank}")
        if self.max_rank < self.min_rank:
            raise ValueError(
                f"max_rank ({self.max_rank}) must be >= min_rank ({self.min_rank})"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


# ---------------------------------------------------------------------------
# INT3 packing helpers
# ---------------------------------------------------------------------------

def pack_int3(arr: np.ndarray) -> np.ndarray:
    """Pack ``arr`` of uint8 values in [0, 7] into 3-bit packed uint8 bytes.

    Packing layout: 8 INT3 values → 3 bytes (zero wasted bits).

    Parameters
    ----------
    arr:
        1-D uint8 array with values in [0, 7].  Length need not be a multiple
        of 8 — the last group is zero-padded.

    Returns
    -------
    uint8 array of length ``ceil(len(arr) / 8) * 3``.
    """
    arr = np.asarray(arr, dtype=np.uint8)
    n = len(arr)
    # Pad to multiple of 8
    pad = (8 - n % 8) % 8
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    n_groups = len(arr) // 8
    out = np.zeros(n_groups * 3, dtype=np.uint8)
    for g in range(n_groups):
        v = arr[g * 8 : g * 8 + 8].astype(np.uint32)
        # Pack 8 × 3-bit values into 24 bits stored in 3 bytes
        bits24 = (
            (v[0])
            | (v[1] << 3)
            | (v[2] << 6)
            | (v[3] << 9)
            | (v[4] << 12)
            | (v[5] << 15)
            | (v[6] << 18)
            | (v[7] << 21)
        )
        out[g * 3]     = np.uint8(bits24 & 0xFF)
        out[g * 3 + 1] = np.uint8((bits24 >> 8) & 0xFF)
        out[g * 3 + 2] = np.uint8((bits24 >> 16) & 0xFF)
    return out


def unpack_int3(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack ``packed`` (from :func:`pack_int3`) back to uint8 values in [0, 7].

    Parameters
    ----------
    packed:
        uint8 array produced by :func:`pack_int3`.
    n:
        Number of values to unpack (original length before padding).

    Returns
    -------
    uint8 array of length *n* with values in [0, 7].
    """
    packed = np.asarray(packed, dtype=np.uint8)
    n_groups = len(packed) // 3
    out = np.zeros(n_groups * 8, dtype=np.uint8)
    mask3 = np.uint32(0x7)
    for g in range(n_groups):
        b0, b1, b2 = (
            np.uint32(packed[g * 3]),
            np.uint32(packed[g * 3 + 1]),
            np.uint32(packed[g * 3 + 2]),
        )
        bits24 = b0 | (b1 << 8) | (b2 << 16)
        for j in range(8):
            out[g * 8 + j] = np.uint8((bits24 >> (j * 3)) & mask3)
    return out[:n]


def _quantize_int_n(arr: np.ndarray, bits: int, group_size: int = 64
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform symmetric quantization to *bits* bits per value, group-wise.

    Returns
    -------
    (quantized_uint8, scales, zeros)  — scales/zeros per group (float32)
    """
    flat = arr.reshape(-1)
    n = len(flat)
    n_groups = math.ceil(n / group_size)
    scales = np.zeros(n_groups, dtype=np.float32)
    zeros  = np.zeros(n_groups, dtype=np.float32)
    levels = (1 << bits) - 1

    quant_flat = np.zeros(n, dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end   = min(start + group_size, n)
        chunk = flat[start:end].astype(np.float32)
        vmax  = np.abs(chunk).max()
        if vmax == 0.0:
            vmax = 1.0
        scale = vmax / (levels / 2)
        zp    = levels // 2
        q     = np.clip(np.round(chunk / scale + zp), 0, levels).astype(np.int32)
        quant_flat[start:end] = q.astype(np.float32)
        scales[g] = scale
        zeros[g]  = float(zp)

    # For INT3 packing return uint8; for others just return float32 indices
    q_int = quant_flat.astype(np.uint8)
    if bits == 3:
        returned_q = pack_int3(q_int)
    else:
        returned_q = q_int
    return returned_q, scales, zeros


def _dequantize_int_n(
    q_packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    n: int,
    bits: int,
    group_size: int = 64,
    original_shape: tuple | None = None,
) -> np.ndarray:
    """Dequantize weight packed by :func:`_quantize_int_n`."""
    if bits == 3:
        q_int = unpack_int3(q_packed, n).astype(np.float32)
    else:
        q_int = q_packed[:n].astype(np.float32)

    flat = np.zeros(n, dtype=np.float32)
    n_groups = math.ceil(n / group_size)
    for g in range(n_groups):
        start = g * group_size
        end   = min(start + group_size, n)
        flat[start:end] = (q_int[start:end] - zeros[g]) * scales[g]

    if original_shape is not None:
        flat = flat.reshape(original_shape)
    return flat


# ---------------------------------------------------------------------------
# Low-Rank Compensator
# ---------------------------------------------------------------------------

class LowRankCompensator:
    """Compact low-rank representation of the quantization residual error.

    Stores matrices A ∈ ℝ^{m×r} and B ∈ ℝ^{r×n} such that the residual
    is approximately equal to A @ B.

    The inference correction is:
      ``corrected_output = quant_output + B @ (A @ input)``

    Parameters
    ----------
    a:
        Left factor, shape ``(m, rank)``.
    b:
        Right factor, shape ``(rank, n)``.
    scale:
        Optional multiplicative scale applied before adding compensation.
        Default 1.0.
    """

    __slots__ = ("_a", "_b", "_scale")

    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        scale: float = 1.0,
    ) -> None:
        self._a     = np.asarray(a, dtype=np.float32)
        self._b     = np.asarray(b, dtype=np.float32)
        self._scale = float(scale)

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def rank(self) -> int:
        return self._a.shape[1]

    @property
    def scale(self) -> float:
        return self._scale

    def apply(self, base_output: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """Add the low-rank correction to *base_output*.

        Parameters
        ----------
        base_output:
            Output of the quantized weight matrix multiply, shape ``(m,)``.
        input_vec:
            Input to the weight multiply, shape ``(n,)``.

        Returns
        -------
        np.ndarray — ``base_output + scale * (A @ (B @ input_vec))``

        This implements the MiLo correction formula:
            corrected = W_int3 @ x + A @ (B @ x)
        where A ∈ ℝ^{m×r} and B ∈ ℝ^{r×n}.
        """
        # B @ input_vec: (r, n) @ (n,) → (r,)
        projected = self._b @ input_vec      # (r,)
        # A @ projected: (m, r) @ (r,) → (m,)
        correction = self._a @ projected     # (m,)
        return base_output + self._scale * correction

    def memory_bytes(self) -> int:
        """Return total memory used by A and B arrays."""
        return self._a.nbytes + self._b.nbytes

    def reconstruction_snr_db(self, residual: np.ndarray) -> float:
        """Compute SNR (in dB) of this compensator on *residual*.

        SNR = 10 * log10( ||residual||² / ||residual - A@B||² )

        Parameters
        ----------
        residual:
            The true W_FP32 - W_INT3 matrix (same shape as A @ B).

        Returns
        -------
        float — SNR in dB (higher is better).
        """
        approx  = self._a @ self._b
        err     = residual.reshape(approx.shape) - approx
        sig_pow = np.sum(residual ** 2)
        err_pow = np.sum(err ** 2)
        if err_pow == 0.0:
            return float("inf")
        return float(10 * np.log10(sig_pow / (err_pow + 1e-30)))


# ---------------------------------------------------------------------------
# MiLoQuantizer
# ---------------------------------------------------------------------------

class MiLoQuantizer:
    """Quantize a weight matrix with MiLo INT3 + low-rank compensator.

    Parameters
    ----------
    config:
        :class:`MiLoConfig` instance.  Defaults to standard MiLo defaults.
    """

    def __init__(self, config: MiLoConfig | None = None) -> None:
        self._cfg = config or MiLoConfig()

    @property
    def config(self) -> MiLoConfig:
        return self._cfg

    def quantize(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, LowRankCompensator]:
        """Quantize *weight* to INT3 and compute a low-rank compensator.

        Parameters
        ----------
        weight:
            2-D float32 weight matrix, shape ``(m, n)``.
        name:
            Optional label (used only by :class:`MiLoStats`).

        Returns
        -------
        (q_packed, scales, zeros, compensator)
          - q_packed: INT3-packed uint8 array (output of :func:`pack_int3`)
          - scales:   float32 per-group scale factors
          - zeros:    float32 per-group zero-point offsets
          - compensator: :class:`LowRankCompensator` for inference correction
        """
        w = np.asarray(weight, dtype=np.float32)
        orig_shape = w.shape
        n = w.size

        # ── Step 1: Symmetric INT3 quantization ──────────────────────────────
        q_packed, scales, zeros = _quantize_int_n(
            w, self._cfg.target_bits, self._cfg.group_size
        )

        # ── Step 2: Dequantize and compute residual ───────────────────────────
        w_dequant = _dequantize_int_n(
            q_packed, scales, zeros, n,
            self._cfg.target_bits, self._cfg.group_size, orig_shape
        )
        residual = w - w_dequant  # shape: (m, n)

        # ── Step 3: Adaptive low-rank decomposition via truncated SVD ─────────
        compensator = self._low_rank_decompose(residual)
        return q_packed, scales, zeros, compensator

    def dequantize(
        self,
        q_packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        n: int,
        original_shape: tuple,
    ) -> np.ndarray:
        """Dequantize a packed weight matrix (without applying compensator).

        Parameters
        ----------
        q_packed:
            Packed uint8 array from :meth:`quantize`.
        scales, zeros:
            Per-group arrays from :meth:`quantize`.
        n:
            Total number of weight elements (``m * n``).
        original_shape:
            Shape tuple to restore the output to.

        Returns
        -------
        float32 numpy array, shape *original_shape*.
        """
        return _dequantize_int_n(
            q_packed, scales, zeros, n,
            self._cfg.target_bits, self._cfg.group_size, original_shape
        )

    def reconstruction_snr(
        self,
        weight: np.ndarray,
        q_packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        compensator: LowRankCompensator,
    ) -> float:
        """Measure SNR (dB) of the compensated reconstruction vs *weight*.

        Parameters
        ----------
        weight:
            Original weight matrix (float32).
        q_packed, scales, zeros:
            Quantized form from :meth:`quantize`.
        compensator:
            Low-rank compensator from :meth:`quantize`.

        Returns
        -------
        float — SNR in dB.
        """
        w = np.asarray(weight, dtype=np.float32)
        n = w.size
        w_dq = _dequantize_int_n(
            q_packed, scales, zeros, n,
            self._cfg.target_bits, self._cfg.group_size, w.shape
        )
        # Compensated reconstruction ≈ w_dq + A @ B
        w_compensated = w_dq + compensator.a @ compensator.b
        err = w - w_compensated
        sig_pow = float(np.sum(w ** 2))
        err_pow = float(np.sum(err ** 2))
        if err_pow == 0.0:
            return float("inf")
        return float(10 * np.log10(sig_pow / (err_pow + 1e-30)))

    # ── Private ────────────────────────────────────────────────────────────────

    def _low_rank_decompose(self, residual: np.ndarray) -> LowRankCompensator:
        """Compute adaptive-rank SVD of *residual* using :attr:`config` thresholds.

        Uses numpy's truncated SVD via ``np.linalg.svd(full_matrices=False)``.
        The rank is the minimum r ∈ [min_rank, max_rank] that satisfies
        ``snr_threshold_db``.  If ``adaptive_rank=False``, always uses
        ``max_rank``.

        Returns
        -------
        LowRankCompensator with A = U[:, :r] * S[:r] and B = Vt[:r, :]
        """
        res = residual.astype(np.float32)
        if res.ndim == 1:
            res = res.reshape(1, -1)

        max_rank = min(self._cfg.max_rank, min(*res.shape))
        min_rank = min(self._cfg.min_rank, max_rank)

        try:
            u, s, vt = np.linalg.svd(res, full_matrices=False)
        except np.linalg.LinAlgError:
            # SVD failed (e.g. near-zero matrix) — return rank-1 zero compensator
            m, n = res.shape
            return LowRankCompensator(
                np.zeros((m, 1), dtype=np.float32),
                np.zeros((1, n), dtype=np.float32),
            )

        if not self._cfg.adaptive_rank:
            r = max_rank
            a = u[:, :r] * s[:r]
            b = vt[:r, :]
            return LowRankCompensator(a, b)

        # Adaptive: increase rank until SNR target met or max_rank reached
        sig_pow = float(np.sum(res ** 2))
        r = min_rank
        while r <= max_rank:
            a = u[:, :r] * s[:r]  # (m, r)  ← absorb singular values into A
            b = vt[:r, :]          # (r, n)
            approx  = a @ b
            err_pow = float(np.sum((res - approx) ** 2))
            snr_db  = 10 * np.log10(sig_pow / (err_pow + 1e-30)) if err_pow else 99.0
            if snr_db >= self._cfg.snr_threshold_db:
                break
            r += 1
        r = min(r, max_rank)
        a = u[:, :r] * s[:r]
        b = vt[:r, :]
        return LowRankCompensator(a, b)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class MiLoStats:
    """Accuracy and compression statistics for MiLo quantization.

    Attributes
    ----------
    n_matrices:    Number of weight matrices quantized.
    avg_snr_db:    Average reconstruction SNR in dB.
    avg_rank:      Average low-rank compensator rank.
    total_weight_bytes:      Total bytes of original FP32 weights.
    total_quant_bytes:       Total bytes of quantized weights (packed INT3).
    total_compensator_bytes: Total bytes of all A+B matrices.
    """

    n_matrices:               int   = 0
    _snr_sum:                 float = field(default=0.0, repr=False)
    _rank_sum:                int   = field(default=0, repr=False)
    total_weight_bytes:       int   = 0
    total_quant_bytes:        int   = 0
    total_compensator_bytes:  int   = 0

    @property
    def avg_snr_db(self) -> float:
        return self._snr_sum / self.n_matrices if self.n_matrices else 0.0

    @property
    def avg_rank(self) -> float:
        return self._rank_sum / self.n_matrices if self.n_matrices else 0.0

    @property
    def compression_ratio(self) -> float:
        """Total (quant + compensator) bytes / original weight bytes."""
        if self.total_weight_bytes == 0:
            return 0.0
        return (self.total_quant_bytes + self.total_compensator_bytes) / self.total_weight_bytes

    def record(
        self,
        snr_db: float,
        rank: int,
        weight_bytes: int,
        quant_bytes: int,
        comp_bytes: int,
    ) -> None:
        """Record statistics for one quantized weight matrix."""
        self.n_matrices              += 1
        self._snr_sum                += snr_db
        self._rank_sum               += rank
        self.total_weight_bytes      += weight_bytes
        self.total_quant_bytes       += quant_bytes
        self.total_compensator_bytes += comp_bytes

    def reset(self) -> None:
        """Reset all counters."""
        self.n_matrices              = 0
        self._snr_sum                = 0.0
        self._rank_sum               = 0
        self.total_weight_bytes      = 0
        self.total_quant_bytes       = 0
        self.total_compensator_bytes = 0
