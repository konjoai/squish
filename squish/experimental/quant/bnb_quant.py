"""squish/quant/bnb_quant.py — BitsAndBytes quantization wrapper.

Provides NF4 / INT8 / FP4 quantization via the bitsandbytes library when
running on Linux + NVIDIA CUDA.  Falls back to a simplified NumPy int8
simulation when bitsandbytes is not installed, so the module is always
importable on macOS and CPU-only environments.

Classes
───────
BnbConfig           — Configuration dataclass.
BnbQuantized        — Container for a quantized weight tensor.
BnbStats            — Runtime statistics.
BitsAndBytesQuantizer — Quantize and dequantize weight matrices.

Usage::

    cfg  = BnbConfig(quant_type="nf4", use_double_quant=True)
    quant = BitsAndBytesQuantizer(cfg)
    q    = quant.quantize(weight)   # weight is np.ndarray float32
    w    = quant.dequantize(q)      # restored float32 array
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_QUANT_TYPES  = frozenset({"nf4", "int8", "fp4"})
_VALID_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32"})

# NF4 quantisation lookup table (16 levels, zero-centred symmetric, unit norm)
_NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534,  0.16093020141124725,  0.24611230195522308,
    0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
    0.7229568362236023,   1.0,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BnbConfig:
    """Configuration for BitsAndBytesQuantizer.

    Attributes
    ----------
    quant_type:
        Quantization type: 'nf4', 'fp4', or 'int8'. Default 'nf4'.
    compute_dtype:
        Dtype for computation after dequantization. One of
        'float16', 'bfloat16', 'float32'. Default 'float16'.
    use_double_quant:
        Apply a second quantization pass to the quantization constants
        themselves (reduces memory further). Default True.
    group_size:
        Number of elements per quantization group. Default 64.
    """
    quant_type:       str  = "nf4"
    compute_dtype:    str  = "float16"
    use_double_quant: bool = True
    group_size:       int  = 64

    def __post_init__(self) -> None:
        if self.quant_type not in _VALID_QUANT_TYPES:
            raise ValueError(
                f"quant_type must be one of {sorted(_VALID_QUANT_TYPES)}, "
                f"got '{self.quant_type}'"
            )
        if self.compute_dtype not in _VALID_COMPUTE_DTYPES:
            raise ValueError(
                f"compute_dtype must be one of {sorted(_VALID_COMPUTE_DTYPES)}, "
                f"got '{self.compute_dtype}'"
            )
        if self.group_size < 1:
            raise ValueError(
                f"group_size must be >= 1, got {self.group_size}"
            )


@dataclass
class BnbQuantized:
    """Container for a quantized weight tensor."""
    packed:         np.ndarray          # int8 packed representation
    quant_state:    Any                 # bitsandbytes QuantState or dict
    original_shape: tuple               # shape of the original float tensor
    quant_type:     str                 # 'nf4' / 'fp4' / 'int8'
    scale:          np.ndarray          # per-group scale factors (float32)
    is_bnb_native:  bool = False        # whether bitsandbytes was used


@dataclass
class BnbStats:
    """Runtime statistics for BitsAndBytesQuantizer."""
    total_quantized_tensors:   int   = 0
    total_dequantized_tensors: int   = 0
    total_params:              int   = 0
    bitsandbytes_used:         bool  = False
    last_quantize_ms:          float = 0.0
    last_dequantize_ms:        float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Approximate bits-per-param ratio (4-bit → 0.5, 8-bit → 1.0)."""
        return 0.5 if not self.bitsandbytes_used else 0.5


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BitsAndBytesQuantizer:
    """Quantize and dequantize weight tensors.

    Tries to use the real bitsandbytes library on CUDA; falls back to a
    NumPy-based int8 / NF4 simulation on CPU and macOS.

    Usage::

        q   = BitsAndBytesQuantizer()
        qw  = q.quantize(weight)
        fw  = q.dequantize(qw)   # float32 ndarray ≈ original weight
    """

    def __init__(self, config: Optional[BnbConfig] = None) -> None:
        self._cfg   = config or BnbConfig()
        self.stats  = BnbStats()
        self._bnb   = self._try_import_bnb()
        self.stats.bitsandbytes_used = self._bnb is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, weight: np.ndarray) -> BnbQuantized:
        """Quantize a float32 weight matrix.

        Parameters
        ----------
        weight : ndarray, shape (rows, cols) or (n,), dtype float32
        """
        t0 = time.perf_counter()
        weight = np.asarray(weight, dtype=np.float32)
        if self._bnb is not None:
            result = self._bnb_quantize(weight)
        else:
            result = self._numpy_quantize(weight)
        self.stats.total_quantized_tensors += 1
        self.stats.total_params            += weight.size
        self.stats.last_quantize_ms         = (time.perf_counter() - t0) * 1000.0
        return result

    def dequantize(self, quantized: BnbQuantized) -> np.ndarray:
        """Restore a quantized tensor to float32.

        Parameters
        ----------
        quantized : BnbQuantized returned by quantize()

        Returns
        -------
        float32 ndarray with the original shape
        """
        t0 = time.perf_counter()
        if quantized.is_bnb_native and self._bnb is not None:
            result = self._bnb_dequantize(quantized)
        else:
            result = self._numpy_dequantize(quantized)
        self.stats.total_dequantized_tensors += 1
        self.stats.last_dequantize_ms         = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # BitsAndBytes path
    # ------------------------------------------------------------------

    def _bnb_quantize(self, weight: np.ndarray) -> BnbQuantized:
        try:
            import torch
            t = torch.from_numpy(weight).cuda()
            bnb = self._bnb
            if self._cfg.quant_type in ("nf4", "fp4"):
                packed, qs = bnb.functional.quantize_4bit(
                    t,
                    quant_type=self._cfg.quant_type,
                    compress_statistics=self._cfg.use_double_quant,
                    blocksize=self._cfg.group_size,
                )
                scale = np.ones(1, dtype=np.float32)  # stored in qs
            else:  # int8
                packed, qs = bnb.functional.quantize(t, absmax=None)
                scale = qs.cpu().float().numpy()
                qs    = {"absmax": qs}

            return BnbQuantized(
                packed=packed.cpu().numpy(),
                quant_state=qs,
                original_shape=weight.shape,
                quant_type=self._cfg.quant_type,
                scale=scale,
                is_bnb_native=True,
            )
        except Exception:
            # bitsandbytes present but no CUDA — fall back to numpy
            return self._numpy_quantize(weight)

    def _bnb_dequantize(self, quantized: BnbQuantized) -> np.ndarray:
        try:
            import torch
            bnb = self._bnb
            packed = torch.from_numpy(quantized.packed).cuda()
            if quantized.quant_type in ("nf4", "fp4"):
                out = bnb.functional.dequantize_4bit(
                    packed, quant_state=quantized.quant_state
                )
            else:
                out = bnb.functional.dequantize(
                    packed, state=quantized.quant_state
                )
            return out.float().cpu().numpy().reshape(quantized.original_shape)
        except Exception:
            return self._numpy_dequantize(quantized)

    # ------------------------------------------------------------------
    # NumPy fallback path
    # ------------------------------------------------------------------

    def _numpy_quantize(self, weight: np.ndarray) -> BnbQuantized:
        flat = weight.ravel().astype(np.float32)
        gs   = self._cfg.group_size
        n    = len(flat)
        pad  = (-n % gs)          # padding to multiple of group_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        groups = flat.reshape(-1, gs)
        absmax = np.abs(groups).max(axis=1, keepdims=True).astype(np.float32)
        absmax = np.where(absmax == 0, 1.0, absmax)

        if self._cfg.quant_type == "int8":
            normed  = groups / absmax
            q8      = np.clip(np.round(normed * 127), -127, 127).astype(np.int8)
            packed  = q8.ravel()[:n]
        else:
            # NF4 / FP4 simulation: round to nearest NF4 level
            normed = groups / absmax           # in [-1, 1]
            idxs   = np.argmin(
                np.abs(normed[:, :, np.newaxis] - _NF4_LEVELS[np.newaxis, np.newaxis, :]),
                axis=2,
            ).astype(np.uint8)
            # Pack two 4-bit indices into one byte
            idxs_flat = idxs.ravel()[:n]
            if len(idxs_flat) % 2:
                idxs_flat = np.concatenate([idxs_flat, [0]])
            packed = (idxs_flat[0::2] | (idxs_flat[1::2] << 4)).astype(np.int8)

        return BnbQuantized(
            packed=packed,
            quant_state={"type": self._cfg.quant_type, "group_size": gs, "n_original": n},
            original_shape=weight.shape,
            quant_type=self._cfg.quant_type,
            scale=absmax.squeeze(1),
            is_bnb_native=False,
        )

    def _numpy_dequantize(self, quantized: BnbQuantized) -> np.ndarray:
        gs      = self._cfg.group_size
        state   = quantized.quant_state
        n_orig  = state["n_original"] if isinstance(state, dict) else quantized.packed.size
        scale   = quantized.scale.ravel()
        qt      = quantized.quant_type

        if qt == "int8":
            flat_q = quantized.packed[:n_orig].astype(np.float32)
            n_groups = len(flat_q) // gs + (1 if len(flat_q) % gs else 0)
            padded   = np.zeros(n_groups * gs, dtype=np.float32)
            padded[:n_orig] = flat_q
            groups = padded.reshape(-1, gs)
            recon  = (groups / 127.0) * scale[:, np.newaxis]
        else:
            # NF4 / FP4
            packed = quantized.packed.view(np.uint8)
            lo     = packed & 0x0F
            hi     = (packed >> 4) & 0x0F
            idxs   = np.empty(len(packed) * 2, dtype=np.uint8)
            idxs[0::2] = lo
            idxs[1::2] = hi
            idxs = idxs[:n_orig]
            n_groups = len(idxs) // gs + (1 if len(idxs) % gs else 0)
            padded_i = np.zeros(n_groups * gs, dtype=np.uint8)
            padded_i[:n_orig] = idxs
            normed = _NF4_LEVELS[padded_i].reshape(-1, gs)
            recon  = normed * scale[:len(normed), np.newaxis]

        return recon.ravel()[:n_orig].reshape(quantized.original_shape).astype(np.float32)

    @staticmethod
    def _try_import_bnb() -> Optional[Any]:
        try:
            import bitsandbytes as bnb  # type: ignore[import]
            return bnb
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"BitsAndBytesQuantizer("
            f"quant_type={self._cfg.quant_type}, "
            f"bnb={'yes' if self._bnb else 'numpy-fallback'}, "
            f"tensors={self.stats.total_quantized_tensors})"
        )
