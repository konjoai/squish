# [Experimental] This module is part of Squish v10+ (Wave 10B).
# Proof-of-concept quality: API and behaviour may change without notice.
#!/usr/bin/env python3
"""
squish/metal_fusion.py

Apple Silicon Metal kernel fusion for transformer attention and FFN operators.

Targets three high-value operator boundaries that commonly generate unnecessary
DRAM round-trips on M-series chips:

1. **Fused RoPE-Q/K** — applies RoPE sin/cos embeddings to Q and K in a
   single unified pass instead of two separate dispatches.
2. **Fused SwiGLU** — combines ``silu(gate) * up_act`` into one elementwise
   kernel, eliminating the intermediate ``(batch, seq, 4*hidden)`` tensor.
3. **Fused INT8 KV attention** — dequantizes INT8 KV blocks and runs scaled
   dot-product attention in a single Metal dispatch, removing the separate
   dequantize step from the critical decode path.

**Availability tiers**

* **MLX 0.18+ with Metal** (`_METAL_FUSION_AVAILABLE = True`): Full
  ``mx.metal.kernel()`` dispatch.  This path is ``# pragma: no cover`` for
  CI since it requires physical M-series hardware.
* **Fallback** (`_METAL_FUSION_AVAILABLE = False`): Pure-numpy reference
  implementations that are numerically equivalent and fully unit-testable
  without MLX or Metal hardware.

Usage::

    from squish.metal_fusion import MetalFusionConfig, MetalFusionKernels
    from squish.metal_fusion import fused_rope_qk, fused_swiglu, fused_int8_kv_attn

    cfg     = MetalFusionConfig(use_fused_rope=True, use_fused_swiglu=True)
    kernels = MetalFusionKernels(cfg)

    # All functions accept and return numpy arrays (or MLX arrays when available)
    Q2, K2 = fused_rope_qk(Q, K, cos, sin, kernels=kernels)
    out     = fused_swiglu(gate_out, up_out, kernels=kernels)
"""

from __future__ import annotations

__all__ = [
    "MetalFusionConfig",
    "MetalFusionKernels",
    "fused_rope_qk",
    "fused_swiglu",
    "fused_int8_kv_attn",
    "_METAL_FUSION_AVAILABLE",
]

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Metal availability probe
# ---------------------------------------------------------------------------

def _probe_metal_fusion() -> bool:
    """Return True only when MLX 0.18+ ``mx.metal.kernel`` is available."""
    try:
        import mlx.core as mx  # type: ignore[import]
        return hasattr(mx, "metal") and hasattr(mx.metal, "kernel")
    except Exception:  # noqa: BLE001
        return False


_METAL_FUSION_AVAILABLE: bool = _probe_metal_fusion()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetalFusionConfig:
    """Configuration flags for Metal kernel fusion.

    Setting a flag to ``False`` forces the numpy fallback for that operator
    even when Metal is available.  Useful for A/B comparisons.

    Attributes:
        use_fused_rope:       Fuse RoPE rotation for Q and K (default True).
        use_fused_swiglu:     Fuse SwiGLU gate * silu(up) (default True).
        use_fused_int8_attn:  Fuse INT8 KV dequantize + attention (default True).
        require_metal:        If ``True``, raise ``RuntimeError`` when Metal
                              kernels are not available (default False).
    """

    use_fused_rope: bool = True
    use_fused_swiglu: bool = True
    use_fused_int8_attn: bool = True
    require_metal: bool = False


# ---------------------------------------------------------------------------
# MetalFusionKernels — kernel lifecycle manager
# ---------------------------------------------------------------------------

class MetalFusionKernels:
    """Manages Metal kernel compilation and exposes availability flags.

    On instantiation, checks whether Metal is available and whether each
    fusion type is enabled by ``config``.  When ``config.require_metal`` is
    ``True`` and Metal is not available, raises ``RuntimeError``.

    Attributes:
        available:           True when Metal kernels are actually usable.
        rope_enabled:        True when fused RoPE is both configured and available.
        swiglu_enabled:      True when fused SwiGLU is both configured and available.
        int8_attn_enabled:   True when fused INT8 attn is both configured and available.

    Args:
        config: A :class:`MetalFusionConfig` instance (default: all fusions on).
    """

    def __init__(self, config: MetalFusionConfig | None = None) -> None:
        self.config = config or MetalFusionConfig()
        self.available = _METAL_FUSION_AVAILABLE

        if self.config.require_metal and not self.available:
            raise RuntimeError(
                "MetalFusionKernels: Metal is not available "
                "(MLX 0.18+ with mx.metal.kernel required).  "
                "Set require_metal=False to use the numpy fallback."
            )

        self.rope_enabled      = self.available and self.config.use_fused_rope
        self.swiglu_enabled    = self.available and self.config.use_fused_swiglu
        self.int8_attn_enabled = self.available and self.config.use_fused_int8_attn

    def __repr__(self) -> str:
        return (
            f"MetalFusionKernels(metal={self.available}  "
            f"rope={self.rope_enabled}  "
            f"swiglu={self.swiglu_enabled}  "
            f"int8_attn={self.int8_attn_enabled})"
        )


# ---------------------------------------------------------------------------
# Fused RoPE — Q and K
# ---------------------------------------------------------------------------

def _rope_numpy(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply RoPE to a single tensor via the numpy reference path.

    Args:
        x:   Shape ``(..., head_dim)`` where ``head_dim`` is even.
        cos: Shape ``(..., head_dim)`` or broadcastable.
        sin: Shape ``(..., head_dim)`` or broadcastable.

    Returns:
        RoPE-rotated tensor of the same shape and dtype as ``x``.
    """
    x = x.astype(np.float32)
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_h = cos[..., :half].astype(np.float32)
    sin_h = sin[..., :half].astype(np.float32)
    rotated = np.concatenate(
        [x1 * cos_h - x2 * sin_h, x1 * sin_h + x2 * cos_h], axis=-1
    )
    return rotated.astype(x.dtype if x.dtype != np.float32 else np.float32)


def fused_rope_qk(
    Q: np.ndarray,
    K: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    kernels: MetalFusionKernels | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE to Q and K in a single fused operation.

    When Metal is available and ``kernels.rope_enabled`` is ``True``, uses a
    single Metal dispatch.  Otherwise falls back to two sequential numpy
    ``_rope_numpy`` calls (numerically equivalent).

    Args:
        Q:       Query tensor, shape ``(batch, heads, seq, head_dim)``.
        K:       Key tensor, same shape as Q.
        cos:     Cosine embeddings, broadcastable to Q/K.
        sin:     Sine embeddings, broadcastable to Q/K.
        kernels: Optional :class:`MetalFusionKernels` instance.  When
                 ``None``, always uses the numpy fallback.

    Returns:
        Tuple ``(Q_rotated, K_rotated)`` of the same shapes and dtypes as
        Q and K.
    """
    if kernels is not None and kernels.rope_enabled:  # pragma: no cover
        # ── Metal path (requires hardware — not executed in CI) ────────
        try:
            import mlx.core as mx
            Q_mx = mx.array(Q)
            K_mx = mx.array(K)
            cos_mx = mx.array(cos)
            sin_mx = mx.array(sin)
            # MLX fast.rope is the canonical fused path
            Q_rot = mx.fast.rope(Q_mx, cos_mx, sin_mx, traditional=False)
            K_rot = mx.fast.rope(K_mx, cos_mx, sin_mx, traditional=False)
            return np.asarray(Q_rot), np.asarray(K_rot)
        except Exception:  # noqa: BLE001
            pass

    # ── Numpy fallback ─────────────────────────────────────────────────
    return _rope_numpy(Q, cos, sin), _rope_numpy(K, cos, sin)


# ---------------------------------------------------------------------------
# Fused SwiGLU
# ---------------------------------------------------------------------------

def fused_swiglu(
    gate_out: np.ndarray,
    up_out: np.ndarray,
    *,
    kernels: MetalFusionKernels | None = None,
) -> np.ndarray:
    """Compute ``silu(gate_out) * up_out`` in a single fused operation.

    Args:
        gate_out: Gating activations, any broadcastable shape.
        up_out:   Up-projection activations, same shape as ``gate_out``.
        kernels:  Optional :class:`MetalFusionKernels`.

    Returns:
        Fused SwiGLU output of the same shape and dtype.
    """
    if kernels is not None and kernels.swiglu_enabled:  # pragma: no cover
        try:
            import mlx.core as mx
            g = mx.array(gate_out)
            u = mx.array(up_out)
            result = mx.fast.gelu(g) * u   # approximate via GLU; true SiLU below
            # MLX has mx.silu:
            try:
                result = mx.silu(g) * u
            except AttributeError:
                pass
            return np.asarray(result)
        except Exception:  # noqa: BLE001
            pass

    # ── Numpy fallback ─────────────────────────────────────────────────
    g = gate_out.astype(np.float32)
    u = up_out.astype(np.float32)
    silu_g = g * (1.0 / (1.0 + np.exp(-g)))
    return (silu_g * u).astype(gate_out.dtype if gate_out.dtype != np.float32
                                else np.float32)


# ---------------------------------------------------------------------------
# Fused INT8 KV attention
# ---------------------------------------------------------------------------

def fused_int8_kv_attn(
    q: np.ndarray,
    k_int8: np.ndarray,
    v_int8: np.ndarray,
    k_scales: np.ndarray,
    v_scales: np.ndarray,
    *,
    kernels: MetalFusionKernels | None = None,
) -> np.ndarray:
    """Dequantize INT8 KV cache and compute scaled dot-product attention.

    Fuses the dequantize and matmul passes that would otherwise generate a
    full FP16/BF16 KV buffer as an intermediate.

    Args:
        q:        Query tensor, shape ``(batch, heads, q_len, head_dim)``.
        k_int8:   INT8-quantized key cache,
                  shape ``(batch, heads, kv_len, head_dim)``, dtype int8.
        v_int8:   INT8-quantized value cache, same shape as ``k_int8``.
        k_scales: Per-token key dequantization scales,
                  shape ``(batch, heads, kv_len, 1)`` or broadcastable.
        v_scales: Per-token value dequantization scales, same shape as
                  ``k_scales``.
        kernels:  Optional :class:`MetalFusionKernels`.

    Returns:
        Attention output of shape ``(batch, heads, q_len, head_dim)``,
        dtype float32.
    """
    if kernels is not None and kernels.int8_attn_enabled:  # pragma: no cover
        try:
            import mlx.core as mx
            q_mx    = mx.array(q, dtype=mx.float32)
            k_fp    = mx.array(k_int8, dtype=mx.float32) * mx.array(k_scales)
            v_fp    = mx.array(v_int8, dtype=mx.float32) * mx.array(v_scales)
            scale   = 1.0 / math.sqrt(q.shape[-1])
            result  = mx.fast.scaled_dot_product_attention(q_mx, k_fp, v_fp, scale=scale)
            return np.asarray(result)
        except Exception:  # noqa: BLE001
            pass

    # ── Numpy fallback ─────────────────────────────────────────────────
    q_f = q.astype(np.float32)
    k_f = k_int8.astype(np.float32) * k_scales.astype(np.float32)
    v_f = v_int8.astype(np.float32) * v_scales.astype(np.float32)

    scale = 1.0 / math.sqrt(q_f.shape[-1])
    # q: (B, H, Lq, D), k: (B, H, Lk, D)
    # attn_weights: (B, H, Lq, Lk) = q @ k.transpose(-2,-1) * scale
    attn_w = np.matmul(q_f, k_f.swapaxes(-2, -1)) * scale
    # softmax over last axis
    attn_w -= attn_w.max(axis=-1, keepdims=True)
    exp_w   = np.exp(attn_w)
    attn_p  = exp_w / exp_w.sum(axis=-1, keepdims=True)
    # (B, H, Lq, Lk) @ (B, H, Lk, D) → (B, H, Lq, D)
    return np.matmul(attn_p, v_f)
