"""squish/kernels/universal_attn.py — Universal attention router.

Routes attention computation to the best available backend for the current
platform: Metal Flash Attention on macOS, CUDA Flash Attention on Linux+GPU,
or pure-NumPy softmax as a universal fallback.

Classes
───────
UniversalAttnConfig  — Configuration dataclass.
UniversalAttnStats   — Runtime call statistics.
UniversalAttention   — Router class; call .forward(q, k, v).

Usage::

    from squish.kernels.universal_attn import UniversalAttention

    attn = UniversalAttention()
    out, lse = attn.forward(q, k, v)   # q/k/v: (seq, heads, d) float32
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_VALID_PREFER = frozenset({"auto", "metal", "cuda", "numpy"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class UniversalAttnConfig:
    """Configuration for UniversalAttention.

    Attributes
    ----------
    causal:
        Apply causal mask. Default True.
    prefer_implementation:
        One of 'auto', 'metal', 'cuda', 'numpy'.  'auto' picks the best
        available backend at construction time.
    dropout:
        Attention dropout probability [0, 1). Passed to CUDA backend only.
    """
    causal:                bool  = True
    prefer_implementation: str   = "auto"
    dropout:               float = 0.0

    def __post_init__(self) -> None:
        if self.prefer_implementation not in _VALID_PREFER:
            raise ValueError(
                f"prefer_implementation must be one of "
                f"{sorted(_VALID_PREFER)}, got '{self.prefer_implementation}'"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class UniversalAttnStats:
    """Runtime statistics for UniversalAttention."""
    total_calls:   int   = 0
    metal_calls:   int   = 0
    cuda_calls:    int   = 0
    numpy_calls:   int   = 0
    last_call_ms:  float = 0.0

    @property
    def active_backend(self) -> str:
        """Backend with the most calls."""
        counts = {
            "metal": self.metal_calls,
            "cuda":  self.cuda_calls,
            "numpy": self.numpy_calls,
        }
        return max(counts, key=lambda k: counts[k]) if self.total_calls else "none"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UniversalAttention:
    """Platform-routing attention: Metal → CUDA → NumPy.

    This class wraps MetalFlashAttention (macOS) and CUDAFlashAttention
    (Linux) behind a single stable ``forward()`` API.  The active backend
    is selected once at construction time based on available hardware.

    Input shapes
    ─────────────
    3-D : (seq_len, num_heads, head_dim)  ← preferred
    2-D : (seq_len, head_dim)             ← treated as single-head
    """

    def __init__(self, config: Optional[UniversalAttnConfig] = None) -> None:
        self._cfg    = config or UniversalAttnConfig()
        self.stats   = UniversalAttnStats()
        self._backend, self._backend_name = self._init_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention output and log-sum-exp.

        Parameters
        ----------
        q, k, v : float32 numpy arrays
            Shape (seq_len, num_heads, head_dim) or (seq_len, head_dim).
        mask : optional bool array, shape (seq_len, seq_len)

        Returns
        -------
        (output, lse) with the same shape contract as CUDAFlashAttention.
        """
        t0 = time.perf_counter()
        result = self._dispatch(q, k, v, mask)
        self.stats.last_call_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.total_calls  += 1
        return result

    @property
    def backend_name(self) -> str:
        """Active backend identifier string."""
        return self._backend_name

    def reset_stats(self) -> None:
        self.stats = UniversalAttnStats()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        name = self._backend_name
        if name == "metal" and self._backend is not None:
            self.stats.metal_calls += 1
            return self._call_metal(q, k, v)
        if name == "cuda" and self._backend is not None:
            self.stats.cuda_calls += 1
            return self._backend.forward(q, k, v, mask)
        self.stats.numpy_calls += 1
        return self._numpy_forward(q, k, v, mask)

    def _call_metal(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to MetalFlashAttention."""
        try:
            return self._backend.forward(q, k, v)
        except Exception:
            # Metal call failed — degrade to numpy
            self._backend_name = "numpy"
            self.stats.numpy_calls += 1
            self.stats.metal_calls -= 1
            return self._numpy_forward(q, k, v, None)

    # ------------------------------------------------------------------
    # NumPy fallback (identical to CUDAFlashAttention._numpy_attention)
    # ------------------------------------------------------------------

    def _numpy_forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)

        single_head = q.ndim == 2
        if single_head:
            q = q[:, np.newaxis, :]
            k = k[:, np.newaxis, :]
            v = v[:, np.newaxis, :]

        seq, heads, d = q.shape
        scale = 1.0 / np.sqrt(d)

        q_h = q.transpose(1, 0, 2).astype(np.float32)
        k_h = k.transpose(1, 0, 2).astype(np.float32)
        v_h = v.transpose(1, 0, 2).astype(np.float32)

        scores = np.matmul(q_h, k_h.transpose(0, 2, 1)) * scale

        if self._cfg.causal:
            causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
            scores[:, causal_mask] = -1e9
        if mask is not None:
            scores[:, ~mask] = -1e9

        scores -= scores.max(-1, keepdims=True)
        exp_s  = np.exp(scores)
        denom  = exp_s.sum(-1, keepdims=True)
        probs  = exp_s / (denom + 1e-8)
        lse    = np.log(denom + 1e-8).squeeze(-1).transpose(1, 0)

        out = np.matmul(probs, v_h).transpose(1, 0, 2)

        if single_head:
            out = out[:, 0, :]
            lse = lse[:, 0]

        return out.astype(np.float32), lse.astype(np.float32)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self):
        """Select and construct the best backend."""
        prefer = self._cfg.prefer_implementation

        if prefer in ("auto", "metal") and sys.platform == "darwin":
            try:
                from squish.kernels.metal_flash_attn import MetalFlashAttention  # type: ignore
                from squish.kernels.metal_flash_attn import MetalFlashConfig
                cfg = MetalFlashConfig(causal=self._cfg.causal)
                return MetalFlashAttention(cfg), "metal"
            except Exception:
                pass

        if prefer in ("auto", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    from squish.kernels.cuda_flash_attn import (
                        CUDAFlashAttention,
                        CUDAFlashConfig,
                    )
                    cfg = CUDAFlashConfig(
                        causal=self._cfg.causal,
                        dropout=self._cfg.dropout,
                    )
                    return CUDAFlashAttention(cfg), "cuda"
            except Exception:
                pass

        if prefer == "cuda" and not self._cuda_available():
            # Explicit CUDA requested but not available — still use CUDAFlashAttention
            # which will gracefully fall back to its NumPy backend
            try:
                from squish.kernels.cuda_flash_attn import CUDAFlashAttention, CUDAFlashConfig
                cfg = CUDAFlashConfig(causal=self._cfg.causal, implementation="numpy")
                return CUDAFlashAttention(cfg), "cuda"
            except Exception:
                pass

        return None, "numpy"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        return (
            f"UniversalAttention("
            f"backend={self._backend_name}, "
            f"causal={self._cfg.causal}, "
            f"calls={self.stats.total_calls})"
        )
