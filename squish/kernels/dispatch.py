"""
squish/kernels/dispatch.py

Phase 6 — Structured Kernel Dispatch

Motivation
──────────
Performance-critical operations in squish (softmax, top-p nucleus filter,
INT8 quantization, outer-product fast-weight updates) are implemented as
private helper functions scattered across multiple modules.  This prevents
easy substitution with faster backends (MLX, or future Mojo kernels).

This module provides a *structured kernel registry*:

  1. Each operation is registered under a stable **name** with a specific
     **backend** (NUMPY, MLX, or MOJO).
  2. ``dispatch(name, *args)`` selects the best available backend
     automatically (AUTO) or a specific one if requested.
  3. New backend implementations can be registered at any time — the call
     sites do not change.
  4. The registry is a thin Python dict; no C extensions required.

Usage example
─────────────
    from squish.kernels.dispatch import register_kernel, dispatch, KernelBackend

    @register_kernel("softmax", KernelBackend.NUMPY)
    def numpy_softmax(logits, temperature=1.0):
        ...

    tok_probs = dispatch("softmax", logits, temperature=1.0)

Backend priority
────────────────
When backend=AUTO, the registry tries backends in this order:
    MOJO > MLX > NUMPY

If the preferred backend is not registered for a given kernel, the next
available one is tried.  If none are available, a ``KeyError`` is raised.

Mojo integration path
─────────────────────
To add a Mojo kernel:
    1. Compile the Mojo module and expose it via a Python binding.
    2. Register it:
           @register_kernel("softmax", KernelBackend.MOJO)
           def mojo_softmax(logits, temperature=1.0):
               return mojo_ext.softmax(logits, temperature)
    3. Done — ``dispatch("softmax", ...)`` will now prefer MOJO automatically.

The existing NUMPY fallback remains available if the Mojo binary is absent
(e.g., CI environments without the Mojo toolchain).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class KernelBackend(Enum):
    """
    Ordered priority (highest first) used by ``dispatch`` in AUTO mode.
    """
    NUMPY = "numpy"
    MLX   = "mlx"
    MOJO  = "mojo"
    AUTO  = "auto"   # pseudo-backend: selects best available


# Backend preference order for AUTO dispatch (highest priority first)
_AUTO_ORDER: list[KernelBackend] = [
    KernelBackend.MOJO,
    KernelBackend.MLX,
    KernelBackend.NUMPY,
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class KernelRegistry:
    """
    Thread-safe (read-dominated) registry of named kernel implementations.

    Use :func:`register_kernel` (module-level decorator) rather than
    instantiating this class directly.
    """

    def __init__(self) -> None:
        # _reg[kernel_name][backend] = callable
        self._reg: dict[str, dict[KernelBackend, Callable]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name:    str,
        backend: KernelBackend = KernelBackend.NUMPY,
    ) -> Callable:
        """
        Decorator: register ``func`` as the *backend* implementation of *name*.

        Parameters
        ----------
        name    : Stable kernel name (e.g. ``"softmax"``, ``"int8_quantize"``).
        backend : Which backend this function implements.  Must not be AUTO.

        Returns
        -------
        The original function unchanged (transparent decorator).
        """
        if backend is KernelBackend.AUTO:
            raise ValueError("Cannot register a kernel with backend=AUTO")

        def _decorator(func: Callable) -> Callable:
            self._reg.setdefault(name, {})[backend] = func
            logger.debug("kernel registered: %s [%s]", name, backend.value)
            return func

        return _decorator

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(
        self,
        name:    str,
        *args:   Any,
        backend: KernelBackend = KernelBackend.AUTO,
        **kwargs: Any,
    ) -> Any:
        """
        Call the named kernel, selecting the *backend* implementation.

        Parameters
        ----------
        name    : Kernel name.
        *args   : Positional arguments forwarded to the kernel.
        backend : ``AUTO`` (default) selects the highest-priority available
                  backend; any other value forces that specific backend.
        **kwargs: Keyword arguments forwarded to the kernel.

        Raises
        ------
        KeyError : No implementation found for *name* (and *backend*).
        """
        impls = self._reg.get(name)
        if not impls:
            raise KeyError(f"No kernel registered under name {name!r}")

        if backend is not KernelBackend.AUTO:
            fn = impls.get(backend)
            if fn is None:
                available = [b.value for b in impls]
                raise KeyError(
                    f"Kernel {name!r} has no {backend.value!r} implementation. "
                    f"Available: {available}"
                )
            return fn(*args, **kwargs)

        # AUTO: try in priority order
        for b in _AUTO_ORDER:
            fn = impls.get(b)
            if fn is not None:
                return fn(*args, **kwargs)

        raise KeyError(
            f"Kernel {name!r} has no available implementation "
            f"(registered backends: {[b.value for b in impls]})"
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def available_backends(self, name: str) -> list[KernelBackend]:
        """Return backends registered for *name*, in AUTO priority order."""
        impls = self._reg.get(name, {})
        return [b for b in _AUTO_ORDER if b in impls]

    def list_kernels(self) -> dict[str, list[str]]:
        """
        Return a dict mapping kernel name → list of registered backend names.

        Useful for logging / debug endpoints.
        """
        return {
            name: [b.value for b in self.available_backends(name)]
            for name in sorted(self._reg)
        }

    def is_registered(self, name: str, backend: KernelBackend = KernelBackend.AUTO) -> bool:
        """Return True if *name* has at least one (or a specific) backend."""
        if name not in self._reg:
            return False
        if backend is KernelBackend.AUTO:
            return bool(self._reg[name])
        return backend in self._reg[name]


# ---------------------------------------------------------------------------
# Module-level singleton + convenience wrappers
# ---------------------------------------------------------------------------

#: Global kernel registry.  Import and use ``register_kernel`` / ``dispatch``.
kernel_registry = KernelRegistry()


def register_kernel(
    name:    str,
    backend: KernelBackend = KernelBackend.NUMPY,
) -> Callable:
    """
    Module-level decorator that registers a function in the global
    :data:`kernel_registry`.

    Example::

        @register_kernel("softmax", KernelBackend.NUMPY)
        def _numpy_softmax(logits: np.ndarray, temperature: float = 1.0):
            ...
    """
    return kernel_registry.register(name, backend)


def dispatch(
    name:    str,
    *args:   Any,
    backend: KernelBackend = KernelBackend.AUTO,
    **kwargs: Any,
) -> Any:
    """
    Dispatch *name* via the global :data:`kernel_registry`.

    See :meth:`KernelRegistry.dispatch` for full documentation.
    """
    return kernel_registry.dispatch(name, *args, backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# Built-in numpy kernel registrations
# ---------------------------------------------------------------------------
#
# Register the core squish sampling kernels with the NUMPY backend.
# These back the StructuredSampler when MLX / Mojo are unavailable.

import numpy as _np  # noqa: E402  (deferred to avoid polluting module namespace)

from squish.sampling.sampler import (  # noqa: E402
    _apply_rep_penalty,
    _apply_top_k,
    _apply_top_p,
    _softmax_f32,
)


@register_kernel("softmax", KernelBackend.NUMPY)
def _numpy_softmax(logits: _np.ndarray, temperature: float = 1.0) -> _np.ndarray:
    """Temperature-scaled softmax → float32 probabilities."""
    logits = _np.asarray(logits, dtype=_np.float32)
    if temperature != 1.0 and temperature > 0.0:
        logits = logits / _np.float32(temperature)
    return _softmax_f32(logits)


@register_kernel("top_k", KernelBackend.NUMPY)
def _numpy_top_k(logits: _np.ndarray, k: int) -> _np.ndarray:
    """Top-k logit masking → float32."""
    return _apply_top_k(_np.asarray(logits, dtype=_np.float32), k)


@register_kernel("top_p", KernelBackend.NUMPY)
def _numpy_top_p(probs: _np.ndarray, top_p: float) -> _np.ndarray:
    """Nucleus (top-p) probability filtering → float32."""
    return _apply_top_p(_np.asarray(probs, dtype=_np.float32), top_p)


@register_kernel("rep_penalty", KernelBackend.NUMPY)
def _numpy_rep_penalty(
    logits:  _np.ndarray,
    window:  list[int],
    penalty: float,
) -> _np.ndarray:
    """Repetition penalty on logits."""
    return _apply_rep_penalty(_np.asarray(logits, dtype=_np.float32), window, penalty)


@register_kernel("int8_quantize", KernelBackend.NUMPY)
def _numpy_int8_quantize(
    x: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Per-channel INT8 symmetric quantization.

    Parameters
    ----------
    x : (..., D) float32 tensor

    Returns
    -------
    q : (..., D) int8 quantized values
    s : (..., 1) float32 per-row scale factors
    """
    x   = _np.asarray(x, dtype=_np.float32)
    s   = _np.max(_np.abs(x), axis=-1, keepdims=True).clip(min=1e-8)
    q   = _np.clip(_np.round(x / s * 127.0), -127, 127).astype(_np.int8)
    return q, s.astype(_np.float32)


@register_kernel("int8_dequantize", KernelBackend.NUMPY)
def _numpy_int8_dequantize(
    q: _np.ndarray,
    s: _np.ndarray,
) -> _np.ndarray:
    """
    Inverse of ``int8_quantize``.

    Parameters
    ----------
    q : (..., D) int8
    s : (..., 1) float32 scale factors

    Returns
    -------
    x : (..., D) float32 reconstructed values
    """
    return (q.astype(_np.float32) / 127.0 * s).astype(_np.float32)


# ---------------------------------------------------------------------------
# Phase 3 — Q-Filter geometric scoring kernel
# ---------------------------------------------------------------------------

@register_kernel("svd_score", KernelBackend.NUMPY)
def _numpy_svd_score(
    kproj:  _np.ndarray,
    recent: _np.ndarray,
    basis:  _np.ndarray,
) -> _np.ndarray:
    """
    Compute per-token geometric relevance scores for Q-Filter eviction.

    Projects recent keys through the SVD basis to form a proxy query
    direction, then computes cosine similarity against every cached
    projected key.

    Parameters
    ----------
    kproj  : (n_heads, n_tokens, rank) float32 — projected cached keys
    recent : (n_heads, n_recent, head_dim) float32 — most-recent keys as
             proxy query direction
    basis  : (n_heads, rank, head_dim) float32 — SVD projection basis

    Returns
    -------
    scores : (n_tokens,) float32 — higher = more geometrically relevant
    """
    kproj  = _np.asarray(kproj,  dtype=_np.float32)
    recent = _np.asarray(recent, dtype=_np.float32)
    basis  = _np.asarray(basis,  dtype=_np.float32)

    # Project recent keys through SVD basis → (n_heads, n_recent, rank)
    rk_proj = _np.einsum("hnd,hrd->hnr", recent, basis)
    # Mean over recent window → proxy query direction (n_heads, rank)
    q_hat = rk_proj.mean(axis=1)

    # Cosine similarity: kproj (n_heads, n_tokens, rank) vs q_hat (n_heads, rank)
    dots   = _np.einsum("htr,hr->ht", kproj, q_hat)           # (n_heads, n_tokens)
    k_norm = _np.linalg.norm(kproj, axis=-1)                  # (n_heads, n_tokens)
    q_norm = _np.linalg.norm(q_hat, axis=-1)[:, _np.newaxis]  # (n_heads, 1)
    denom  = _np.maximum(k_norm * q_norm, _np.float32(1e-8))
    cos    = dots / denom                                       # (n_heads, n_tokens)
    return cos.mean(axis=0).astype(_np.float32)                # (n_tokens,)


# ---------------------------------------------------------------------------
# Phase 4 — ReDrafter GRU step kernel
# ---------------------------------------------------------------------------

def _numpy_sigmoid(x: _np.ndarray) -> _np.ndarray:
    """Numerically stable float32 sigmoid (clip-based, avoids overflow)."""
    x_c = _np.clip(_np.asarray(x, dtype=_np.float32), -88.0, 88.0)
    return (_np.float32(1.0) / (_np.float32(1.0) + _np.exp(-x_c))).astype(_np.float32)


@register_kernel("gru_step", KernelBackend.NUMPY)
def _numpy_gru_step(
    x: _np.ndarray,
    h: _np.ndarray,
    W: _np.ndarray,
    U: _np.ndarray,
    b: _np.ndarray,
) -> _np.ndarray:
    """
    Single GRU recurrent cell step.

    Implements the standard GRU equations:
        r = σ(W_r·x + U_r·h + b_r)
        z = σ(W_z·x + U_z·h + b_z)
        n = tanh(W_n·x + b_n + r ⊙ (U_n·h))
        h' = (1 − z) ⊙ h + z ⊙ n

    Parameters
    ----------
    x : (input_dim,)  float32 — input embedding
    h : (hidden_dim,) float32 — previous hidden state
    W : (3*hidden_dim, input_dim)  float32 — input weight matrix [r; z; n]
    U : (3*hidden_dim, hidden_dim) float32 — recurrent weight matrix [r; z; n]
    b : (3*hidden_dim,) float32 — bias [b_r; b_z; b_n]

    Returns
    -------
    h_new : (hidden_dim,) float32 — updated hidden state
    """
    x = _np.asarray(x, dtype=_np.float32)
    h = _np.asarray(h, dtype=_np.float32)
    W = _np.asarray(W, dtype=_np.float32)
    U = _np.asarray(U, dtype=_np.float32)
    b = _np.asarray(b, dtype=_np.float32)

    hd = h.shape[0]
    gates_x = W @ x + b          # (3*hd,)
    gates_h = U @ h               # (3*hd,)

    r = _numpy_sigmoid(gates_x[:hd]        + gates_h[:hd])
    z = _numpy_sigmoid(gates_x[hd:2 * hd]  + gates_h[hd:2 * hd])
    n = _np.tanh(gates_x[2 * hd:] + r * gates_h[2 * hd:]).astype(_np.float32)

    return ((_np.float32(1.0) - z) * h + z * n).astype(_np.float32)


# ---------------------------------------------------------------------------
# Phase 5 — TTT Fast Weight kernels
# ---------------------------------------------------------------------------

@register_kernel("outer_product_update", KernelBackend.NUMPY)
def _numpy_outer_product_update(
    W_f:    _np.ndarray,
    keys:   _np.ndarray,
    values: _np.ndarray,
    lr:     float,
    decay:  float,
) -> _np.ndarray:
    """
    In-place outer-product fast-weight update (returns updated W_f).

    Update rule:
        W_f ← decay · W_f  +  lr · Σ_t( v_t ⊗ k_t )

    Parameters
    ----------
    W_f    : (n_heads, head_dim, head_dim) float32 — current fast-weight matrix
    keys   : (n_heads, n_tokens, head_dim) float32 — absorbed key vectors
    values : (n_heads, n_tokens, head_dim) float32 — absorbed value vectors
    lr     : per-absorption learning rate
    decay  : multiplicative decay < 1.0 applied before the new update

    Returns
    -------
    W_f_new : (n_heads, head_dim, head_dim) float32 — updated matrix
    """
    W_f    = _np.asarray(W_f,    dtype=_np.float32)
    keys   = _np.asarray(keys,   dtype=_np.float32)
    values = _np.asarray(values, dtype=_np.float32)

    if decay < 1.0:
        W_f = W_f * _np.float32(decay)

    delta = _np.einsum("hte,htd->hed", values, keys, dtype=_np.float32)
    return (W_f + _np.float32(lr) * delta).astype(_np.float32)


@register_kernel("fast_weight_query", KernelBackend.NUMPY)
def _numpy_fast_weight_query(
    W_f:     _np.ndarray,
    queries: _np.ndarray,
) -> _np.ndarray:
    """
    Query compressed fast-weight memory.

    Computes  out[h] = W_f[h] @ q[h]  (linear attention over absorbed history).

    Parameters
    ----------
    W_f     : (n_heads, head_dim, head_dim) float32
    queries : (n_heads, head_dim) float32

    Returns
    -------
    out : (n_heads, head_dim) float32
    """
    W_f     = _np.asarray(W_f,     dtype=_np.float32)
    queries = _np.asarray(queries, dtype=_np.float32)
    return _np.einsum("hed,hd->he", W_f, queries, dtype=_np.float32).astype(_np.float32)
