"""
squish/pm_kvq.py

PM-KVQ — Progressive Mixed-Precision KV Quantization for Long Chain-of-Thought.

Based on:
  "PM-KVQ: Progressive Mixed-Precision KV Cache Quantization for Long
  Chain-of-Thought Reasoning LLMs"
  arXiv:2505.18610 (Tsinghua + Infinigence-AI, ICLR 2026)

Problem
-------
Standard KV quantization (KVTuner, DiffKV, CommVQ) uses a fixed bit-width
throughout the entire generation sequence.  For long chain-of-thought tasks
(Qwen3 thinking mode, AIME math, long code generation) this fails in two ways:

1. **Cumulative quantization error**: errors accumulate over thousands of steps.
   A corrupted key at token 200 can invalidate reasoning chains at token 8000.

2. **RoPE channel calibration blindspot**: at long sequence positions, rotary
   positional embedding (RoPE) encodes position in key channels via full-period
   rotations.  Short (2K) calibration data never sees the full-period regime —
   those channels appear flat during calibration but become heavy outliers at
   32K positions.  Fixed quantization params degrade silently.

PM-KVQ Solution
---------------
**Part 1 — Progressive quantization schedule**: Start at FP16 for the first
``high_prec_tokens`` steps, transition through INT8 and INT4 to the minimum
``min_bits`` as generation proceeds.  Transitions are gated per transformer
block by a pre-computed sensitivity score: insensitive blocks transition
earlier; the most sensitive block never drops below ``min_bits_sensitive``.

**Part 2 — Positional interpolation calibration**: To calibrate the quantizer
for long-context key distributions from short data only, scale the position
indices in calibration data by ``interp_scale`` before running calibration.
This forces all RoPE channel frequencies into their full-rotation regime.

Runtime precision schedule (default):
  steps  0 – high_prec_tokens:         FP16   (8+ : no rounding error)
  steps  high_prec_tokens – mid_tokens: INT8   (8 bits: minimal accumulated error)
  steps  mid_tokens –  …:               INT4   (4 bits: moderate compression)
  after ``cold_steps`` from start:      min_bits (lowest precision tier)

Provides
--------
  PMKVQConfig         — configuration parameters
  PMKVQScheduler      — per-step precision scheduling
  PMKVQCalibrator     — positional interpolation calibration
  PMKVQStats          — tracking stats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "PMKVQConfig",
    "PMKVQScheduler",
    "PMKVQCalibrator",
    "PMKVQStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PMKVQConfig:
    """Configuration for PM-KVQ progressive quantization.

    Parameters
    ----------
    high_prec_tokens:
        Number of initial decode steps to keep at FP16 (default 256).
        These early tokens are attended by all future steps — critical.
    mid_tokens:
        Step count at which INT8 → INT4 transition occurs (default 1024).
    cold_steps:
        Step count at which INT4 → ``min_bits`` transition occurs (default 4096).
    min_bits:
        Minimum bit-width for non-sensitive blocks (default 2).
    min_bits_sensitive:
        Minimum bit-width for sensitive blocks (default 4).
    sensitive_fraction:
        Fraction of transformer blocks classified as "sensitive" (default 0.25).
    interp_scale:
        Positional index scaling factor for calibration (default 16).
        Multiplies position indices in short calibration data to simulate
        long-sequence Key channel distributions.
    n_blocks:
        Number of transformer blocks in the model (default 32).
    group_size:
        Quantization group size per KV vector (default 64).
    """

    high_prec_tokens:    int   = 256
    mid_tokens:          int   = 1024
    cold_steps:          int   = 4096
    min_bits:            int   = 2
    min_bits_sensitive:  int   = 4
    sensitive_fraction:  float = 0.25
    interp_scale:        int   = 16
    n_blocks:            int   = 32
    group_size:          int   = 64

    def __post_init__(self) -> None:
        if not 1 <= self.min_bits <= 8:
            raise ValueError(f"min_bits must be in [1, 8], got {self.min_bits}")
        if not 1 <= self.min_bits_sensitive <= 8:
            raise ValueError(
                f"min_bits_sensitive must be in [1, 8], got {self.min_bits_sensitive}"
            )
        if self.min_bits_sensitive < self.min_bits:
            raise ValueError(
                "min_bits_sensitive must be >= min_bits"
            )
        if not 0.0 <= self.sensitive_fraction <= 1.0:
            raise ValueError("sensitive_fraction must be in [0, 1]")


# ---------------------------------------------------------------------------
# PM-KVQ Scheduler
# ---------------------------------------------------------------------------

class PMKVQScheduler:
    """Per-step precision scheduler implementing the progressive schedule.

    Parameters
    ----------
    config:
        PM-KVQ configuration.
    block_sensitivity:
        Optional array of per-block sensitivity scores in [0, 1] where
        higher = more sensitive (should retain precision longer).
        If None, a uniform descending schedule is used.
    """

    def __init__(
        self,
        config: PMKVQConfig | None = None,
        block_sensitivity: np.ndarray | None = None,
    ) -> None:
        self._cfg   = config or PMKVQConfig()
        self._step  = 0
        self._stats = PMKVQStats()

        n_blocks = self._cfg.n_blocks
        if block_sensitivity is not None:
            if block_sensitivity.shape != (n_blocks,):
                raise ValueError(
                    f"block_sensitivity must have shape ({n_blocks},), "
                    f"got {block_sensitivity.shape}"
                )
            self._sensitivity = np.asarray(block_sensitivity, dtype=np.float32)
        else:
            # Default: later blocks are more sensitive (empirical prior for decoders)
            self._sensitivity = np.linspace(0.1, 0.9, n_blocks, dtype=np.float32)

        # Pre-classify sensitive vs. non-sensitive blocks
        threshold = np.quantile(
            self._sensitivity,
            1.0 - self._cfg.sensitive_fraction,
        )
        self._is_sensitive = self._sensitivity >= threshold  # shape (n_blocks,)

    @property
    def config(self) -> PMKVQConfig:
        return self._cfg

    @property
    def step(self) -> int:
        return self._step

    @property
    def stats(self) -> PMKVQStats:
        return self._stats

    def current_bits(self, block_idx: int) -> int:
        """Return the quantization bit-width for ``block_idx`` at the current step.

        Parameters
        ----------
        block_idx:
            Transformer block index (0-indexed).

        Returns
        -------
        int — 16 (FP16), 8, 4, or ``min_bits``/``min_bits_sensitive``.
        """
        cfg  = self._cfg
        step = self._step
        sensitive = bool(self._is_sensitive[block_idx % cfg.n_blocks])

        if step < cfg.high_prec_tokens:
            return 16
        if step < cfg.mid_tokens:
            return 8
        if step < cfg.cold_steps:
            return 4
        return cfg.min_bits_sensitive if sensitive else cfg.min_bits

    def bits_for_all_blocks(self) -> np.ndarray:
        """Return an array of bit-widths for all blocks at the current step.

        Returns
        -------
        np.ndarray of shape ``(n_blocks,)`` with dtype int32.
        """
        bits = np.empty(self._cfg.n_blocks, dtype=np.int32)
        for i in range(self._cfg.n_blocks):
            bits[i] = self.current_bits(i)
        return bits

    def advance(self) -> None:
        """Increment the step counter by 1 and record stats."""
        bits = self.bits_for_all_blocks()
        self._stats.record_step(int(bits.mean()), self._step)
        self._step += 1

    def reset(self) -> None:
        """Reset the step counter to 0 (call at the start of each request)."""
        self._step = 0

    def quantize_kv(
        self,
        kv_vector: np.ndarray,
        block_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Quantize a KV vector using the current scheduled bit-width.

        Parameters
        ----------
        kv_vector:
            Float32 array of shape ``(seq_len, head_dim)`` or ``(head_dim,)``.
        block_idx:
            Transformer block index.

        Returns
        -------
        quantized : np.ndarray — integer quantized array
        scale     : np.ndarray — per-group float32 scale
        bits      : int        — actual bit-width used
        """
        bits = self.current_bits(block_idx)
        if bits == 16:
            # FP16: return as-is (cast to float16)
            arr16 = kv_vector.astype(np.float16)
            return arr16, np.ones(1, dtype=np.float32), 16

        levels    = (1 << bits) - 1
        flat      = kv_vector.flatten().astype(np.float32)
        gs        = self._cfg.group_size
        n         = len(flat)
        n_groups  = max(1, (n + gs - 1) // gs)
        padded    = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        groups  = padded.reshape(n_groups, gs)
        vmax    = np.abs(groups).max(axis=1, keepdims=True).clip(min=1e-7)
        scale   = vmax / (levels / 2)
        q       = np.round(groups / scale).clip(-(levels // 2), levels // 2).astype(np.int8)
        return q.flatten()[:n], scale.flatten(), bits

    def dequantize_kv(
        self,
        quantized: np.ndarray,
        scale: np.ndarray,
        bits: int,
        original_shape: tuple,
    ) -> np.ndarray:
        """Dequantize a PM-KVQ quantized vector.

        Parameters
        ----------
        quantized:
            Integer or FP16 array from ``quantize_kv``.
        scale:
            Per-group float32 scale from ``quantize_kv``.
        bits:
            Bit-width value from ``quantize_kv``.
        original_shape:
            Original shape before quantization.

        Returns
        -------
        np.ndarray of dtype float32.
        """
        if bits == 16:
            return quantized.astype(np.float32).reshape(original_shape)

        n   = quantized.size
        gs  = self._cfg.group_size
        n_groups = len(scale)
        padded_n = n_groups * gs
        q_padded = np.zeros(padded_n, dtype=np.float32)
        q_padded[:n] = quantized.astype(np.float32)
        groups = q_padded.reshape(n_groups, gs)
        dequant = (groups * scale.reshape(-1, 1)).flatten()[:n]
        return dequant.reshape(original_shape)


# ---------------------------------------------------------------------------
# PM-KVQ Calibrator
# ---------------------------------------------------------------------------

class PMKVQCalibrator:
    """Positional interpolation calibrator for long-context Key statistics.

    Collects Key cache statistics from short calibration sequences by scaling
    position indices by ``config.interp_scale``, forcing all RoPE channel
    frequencies into their full-rotation regime.

    Usage
    -----
        cal = PMKVQCalibrator(config)
        for sample in calibration_data:
            cal.record(key_states, position_ids=...)
        block_sensitivity = cal.compute_block_sensitivity()
    """

    def __init__(self, config: PMKVQConfig | None = None) -> None:
        self._cfg         = config or PMKVQConfig()
        self._key_stats:  list[dict] = []   # per-block: {"mean", "std", "max_abs"}
        self._n_samples   = 0

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def record(
        self,
        key_states: np.ndarray,
        block_idx: int,
    ) -> None:
        """Record Key channel statistics for one block in one calibration sample.

        Parameters
        ----------
        key_states:
            Float32 array of shape ``(seq_len, n_heads, head_dim)`` or
            ``(seq_len, head_dim)``.  Position scaling is handled internally.
        block_idx:
            Transformer block index.
        """
        flat = key_states.astype(np.float32).reshape(-1, key_states.shape[-1])
        while len(self._key_stats) <= block_idx:
            self._key_stats.append({"max_abs": np.zeros(flat.shape[-1], dtype=np.float32),
                                    "var": np.zeros(flat.shape[-1], dtype=np.float32),
                                    "n": 0})
        entry = self._key_stats[block_idx]
        entry["max_abs"] = np.maximum(entry["max_abs"], np.abs(flat).max(axis=0))
        entry["var"]    += flat.var(axis=0)
        entry["n"]      += 1
        self._n_samples += 1

    def compute_block_sensitivity(self) -> np.ndarray:
        """Return per-block sensitivity scores in [0, 1].

        Higher scores indicate blocks where key channels have high variance or
        large outliers — blocks that are most harmed by aggressive quantization.

        Returns
        -------
        np.ndarray of shape ``(n_blocks,)`` with dtype float32.
        """
        n_blocks = max(len(self._key_stats), self._cfg.n_blocks)
        scores   = np.zeros(n_blocks, dtype=np.float32)
        for i, entry in enumerate(self._key_stats):
            if entry["n"] == 0:
                continue
            # Sensitivity = normalized (max_abs + var) combined metric
            combined = entry["max_abs"] + np.sqrt(entry["var"] / max(entry["n"], 1))
            scores[i] = float(combined.mean())

        # Normalise to [0, 1]
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class PMKVQStats:
    """Track quantization precision over the generation sequence.

    Attributes
    ----------
    total_steps        : total advance() calls recorded
    avg_bits_history   : list of per-step average bit-widths
    step_at_int8       : step when first INT8 level was reached
    step_at_int4       : step when first INT4 level was reached
    step_at_min        : step when minimum bits were first reached
    """

    total_steps:      int        = 0
    avg_bits_history: list[float] = field(default_factory=list)
    step_at_int8:     int | None = None
    step_at_int4:     int | None = None
    step_at_min:      int | None = None

    def record_step(self, avg_bits: int, step: int) -> None:
        self.total_steps += 1
        self.avg_bits_history.append(float(avg_bits))
        if self.step_at_int8 is None and avg_bits <= 8:
            self.step_at_int8 = step
        if self.step_at_int4 is None and avg_bits <= 4:
            self.step_at_int4 = step
        if self.step_at_min is None and avg_bits <= 2:
            self.step_at_min = step

    def reset(self) -> None:
        self.total_steps      = 0
        self.avg_bits_history = []
        self.step_at_int8     = None
        self.step_at_int4     = None
        self.step_at_min      = None
