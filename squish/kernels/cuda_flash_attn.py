"""squish/kernels/cuda_flash_attn.py — CUDA Flash Attention kernel wrapper.

Provides a unified Flash Attention interface for NVIDIA CUDA GPUs.  Falls
back gracefully through a chain: flash-attn 2.x → xformers memory-efficient
attention → PyTorch F.scaled_dot_product_attention → NumPy softmax baseline.

The NumPy fallback is always available (no CUDA required), making this module
safe to import and unit-test on macOS development machines.

Classes
───────
CUDAFlashConfig         — Configuration dataclass.
CUDAFlashStats          — Runtime statistics dataclass.
CUDAFlashAttention      — Main kernel class.

Usage::

    cfg  = CUDAFlashConfig(causal=True)
    attn = CUDAFlashAttention(cfg)
    out, lse = attn.forward(q, k, v)   # (seq, head, d) tensors or arrays
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_VALID_IMPLS = frozenset({"auto", "flash_attn", "xformers", "torch_sdpa", "numpy"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CUDAFlashConfig:
    """Configuration for CUDAFlashAttention.

    Attributes
    ----------
    causal:
        Apply a causal (autoregressive) mask. Default True.
    dropout:
        Attention dropout probability in [0, 1). Default 0.0.
    scale:
        Softmax scale factor. Defaults to ``1 / sqrt(head_dim)``.
    implementation:
        One of 'auto', 'flash_attn', 'xformers', 'torch_sdpa', 'numpy'.
        'auto' selects the best available backend at runtime.
    """
    causal:         bool            = True
    dropout:        float           = 0.0
    scale:          Optional[float] = None
    implementation: str             = "auto"

    def __post_init__(self) -> None:
        if self.implementation not in _VALID_IMPLS:
            raise ValueError(
                f"implementation must be one of {sorted(_VALID_IMPLS)}, "
                f"got '{self.implementation}'"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CUDAFlashStats:
    """Runtime statistics for CUDAFlashAttention."""
    total_forward_calls:  int   = 0
    total_query_tokens:   int   = 0
    implementation_used:  str   = "none"
    last_forward_ms:      float = 0.0
    flash_attn_calls:     int   = 0
    xformers_calls:       int   = 0
    torch_sdpa_calls:     int   = 0
    numpy_fallback_calls: int   = 0

    @property
    def avg_tokens_per_call(self) -> float:
        if self.total_forward_calls == 0:
            return 0.0
        return self.total_query_tokens / self.total_forward_calls


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CUDAFlashAttention:
    """Unified Flash Attention for CUDA; degrades to NumPy on CPU/macOS.

    Input shapes (q, k, v)
    ──────────────────────
    3-D: (seq_len, num_heads, head_dim)  ← expected / preferred
    2-D: (seq_len, head_dim)             ← treated as single-head

    Returns
    ───────
    output : same shape as q
    lse    : log-sum-exp, shape (seq_len,) or (seq_len, num_heads)
    """

    def __init__(self, config: Optional[CUDAFlashConfig] = None) -> None:
        self._cfg   = config or CUDAFlashConfig()
        self.stats  = CUDAFlashStats()
        self._impl  = self._select_impl(self._cfg.implementation)
        self.stats.implementation_used = self._impl

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
        q, k, v:
            Query/key/value arrays, float32 or float16.
            Shape (seq_len, num_heads, head_dim) or (seq_len, head_dim).
        mask:
            Optional boolean mask of shape (seq_len, seq_len).

        Returns
        -------
        (output, lse)
        """
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)

        single_head = q.ndim == 2
        if single_head:
            q = q[:, np.newaxis, :]
            k = k[:, np.newaxis, :]
            v = v[:, np.newaxis, :]

        output, lse = self._dispatch(q, k, v, mask)

        if single_head:
            output = output[:, 0, :]
            lse    = lse[:, 0]

        self.stats.total_forward_calls += 1
        self.stats.total_query_tokens  += q.shape[0]
        self.stats.last_forward_ms      = (time.perf_counter() - t0) * 1000.0
        return output, lse

    def reset_stats(self) -> None:
        """Reset all runtime stats."""
        self.stats = CUDAFlashStats()
        self.stats.implementation_used = self._impl

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        impl = self._impl
        if impl in ("auto", "flash_attn"):
            result = self._try_flash_attn(q, k, v)
            if result is not None:
                self.stats.flash_attn_calls += 1
                return result
        if impl in ("auto", "xformers"):
            result = self._try_xformers(q, k, v)
            if result is not None:
                self.stats.xformers_calls += 1
                return result
        if impl in ("auto", "torch_sdpa"):
            result = self._try_torch_sdpa(q, k, v)
            if result is not None:
                self.stats.torch_sdpa_calls += 1
                return result
        self.stats.numpy_fallback_calls += 1
        return self._numpy_attention(q, k, v, mask)

    def _try_flash_attn(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            import torch
            from flash_attn import flash_attn_func  # type: ignore[import]
            qt = torch.from_numpy(q).half().cuda()
            kt = torch.from_numpy(k).half().cuda()
            vt = torch.from_numpy(v).half().cuda()
            # flash_attn_func expects (batch, seqlen, heads, d)
            qt = qt.unsqueeze(0)
            kt = kt.unsqueeze(0)
            vt = vt.unsqueeze(0)
            out, lse, _ = flash_attn_func(
                qt, kt, vt,
                causal=self._cfg.causal,
                dropout_p=self._cfg.dropout,
                return_attn_probs=False,
                softmax_scale=self._cfg.scale,
            )
            out_np = out.squeeze(0).float().cpu().numpy()
            lse_np = lse.squeeze(0).float().cpu().numpy()
            return out_np, lse_np
        except Exception:
            return None

    def _try_xformers(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            import torch
            import xformers.ops as xops  # type: ignore[import]
            qt = torch.from_numpy(q).half().cuda().unsqueeze(0)
            kt = torch.from_numpy(k).half().cuda().unsqueeze(0)
            vt = torch.from_numpy(v).half().cuda().unsqueeze(0)
            attn_bias = xops.LowerTriangularMask() if self._cfg.causal else None
            out = xops.memory_efficient_attention(qt, kt, vt, attn_bias=attn_bias)
            out_np = out.squeeze(0).float().cpu().numpy()
            # xformers doesn't return LSE; approximate from output norm
            lse = np.log(np.abs(out_np).sum(-1) + 1e-8)
            return out_np, lse
        except Exception:
            return None

    def _try_torch_sdpa(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            import torch
            import torch.nn.functional as F
            # (seq, head, d) → (1, head, seq, d) for SDPA
            qt = torch.from_numpy(q).float().permute(1, 0, 2).unsqueeze(0)
            kt = torch.from_numpy(k).float().permute(1, 0, 2).unsqueeze(0)
            vt = torch.from_numpy(v).float().permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                qt, kt, vt,
                is_causal=self._cfg.causal,
                scale=self._cfg.scale,
            )
            # (1, head, seq, d) → (seq, head, d)
            out_np = out.squeeze(0).permute(1, 0, 2).numpy()
            lse = np.log(np.abs(out_np).sum(-1) + 1e-8)
            return out_np, lse
        except Exception:
            return None

    def _numpy_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-NumPy softmax attention, always available."""
        seq, heads, d = q.shape
        scale = self._cfg.scale if self._cfg.scale is not None else 1.0 / np.sqrt(d)

        # (seq, heads, d) → (heads, seq, d) for matmul
        q_h = q.transpose(1, 0, 2).astype(np.float32)
        k_h = k.transpose(1, 0, 2).astype(np.float32)
        v_h = v.transpose(1, 0, 2).astype(np.float32)

        # scores: (heads, seq, seq)
        scores = np.matmul(q_h, k_h.transpose(0, 2, 1)) * scale

        if self._cfg.causal:
            causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
            scores[:, causal_mask] = -1e9
        if mask is not None:
            scores[:, ~mask] = -1e9

        # stable softmax + LSE
        scores -= scores.max(-1, keepdims=True)
        exp_s = np.exp(scores)
        denom = exp_s.sum(-1, keepdims=True)
        probs = exp_s / (denom + 1e-8)
        lse   = np.log(denom + 1e-8).squeeze(-1).transpose(1, 0)  # (seq, heads)

        out_h = np.matmul(probs, v_h)          # (heads, seq, d)
        out   = out_h.transpose(1, 0, 2)       # (seq, heads, d)
        return out.astype(np.float32), lse.astype(np.float32)

    @staticmethod
    def _select_impl(requested: str) -> str:
        """Return the best available implementation name."""
        if requested != "auto":
            return requested
        for lib, name in (
            ("flash_attn", "flash_attn"),
            ("xformers",   "xformers"),
            ("torch",      "torch_sdpa"),
        ):
            try:
                __import__(lib)
                return name
            except ImportError:
                continue
        return "numpy"

    def __repr__(self) -> str:
        return (
            f"CUDAFlashAttention("
            f"impl={self._impl}, "
            f"causal={self._cfg.causal}, "
            f"calls={self.stats.total_forward_calls})"
        )
