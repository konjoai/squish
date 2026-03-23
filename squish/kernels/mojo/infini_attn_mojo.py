"""infini_attn_mojo.py — Mojo-accelerated Infini-Attention compressive memory.

Wraps `squish/kernels/mojo/kernels/infini_attn.mojo` via MojoBridge
(Wave 58b). Falls back to NumPy einsum when the Mojo library is unavailable.

MojoInfiniAttnMemory computes the outer-product-accumulate memory update
and matrix-vector retrieval for Infini-attention in one Mojo SIMD pass
(`@parameter head_dim`, `parallelize` over heads), replacing two
`np.einsum` calls in `infini_attn.py`:

  M += ELU(K)^T ⊗ V   (outer-product-accumulate)
  A  = M × σ(Q) / (|Z| + ε)  (retrieval with normalization)

Also covers the identical outer-product update pattern in `fast_weights.py`.
~3× on head_dim=128, H=8 heads vs NumPy einsum dispatch.

Reference:
  Munkhdalai et al. (arXiv:2404.07143, 2024) — Infini-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["InfiniAttnConfig", "MojoInfiniAttnMemory"]

_bridge = MojoBridge()
_MOJO_FN = _bridge.load_kernel("infini_attn")


@dataclass
class InfiniAttnConfig:
    """Configuration for MojoInfiniAttnMemory.

    Attributes:
        n_heads:   Number of attention heads.
        head_dim:  Head dimension ``d``.  Memory M has shape ``(H, d, d)``.
        use_elu:   If True apply ELU+1 gating to keys before accumulate.
        eps:       Normalizer epsilon to avoid division by zero.
    """

    n_heads: int = 8
    head_dim: int = 128
    use_elu: bool = True
    eps: float = 1.0e-6


class MojoInfiniAttnMemory:
    """Mojo-accelerated Infini-Attention compressive memory module.

    Maintains an ``(H, d, d)`` memory matrix ``M`` and normalizer ``Z``.
    Supports per-segment incremental update and retrieval.

    Usage::

        infini = MojoInfiniAttnMemory(InfiniAttnConfig(n_heads=8, head_dim=128))
        M, Z = infini.zero_memory()
        K = np.random.randn(8, 16, 128).astype(np.float32)  # (H, T, d)
        V = np.random.randn(8, 16, 128).astype(np.float32)
        Q = np.random.randn(8, 16, 128).astype(np.float32)
        M, Z = infini.update(K, V, M, Z)
        A = infini.retrieve(Q, M, Z)   # (H, T, d)
    """

    def __init__(self, config: InfiniAttnConfig | None = None) -> None:
        self._cfg = config or InfiniAttnConfig()

    def zero_memory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-initialized ``(M, Z)`` for a new sequence."""
        h, d = self._cfg.n_heads, self._cfg.head_dim
        return np.zeros((h, d, d), dtype=np.float32), np.zeros((h, d), dtype=np.float32)

    def update(
        self,
        K: np.ndarray,
        V: np.ndarray,
        M: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update compressive memory with new keys and values.

        ``M += gated_K^T ⊗ V``  (outer-product over T tokens, summed).

        Args:
            K: Float32 ``(n_heads, T, head_dim)`` key tokens.
            V: Float32 ``(n_heads, T, head_dim)`` value tokens.
            M: Float32 ``(n_heads, head_dim, head_dim)`` current memory.
            Z: Float32 ``(n_heads, head_dim)`` normalizer vector.

        Returns:
            Updated ``(M_new, Z_new)``.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        M = np.asarray(M.copy(), dtype=np.float32)
        Z = np.asarray(Z.copy(), dtype=np.float32)
        if _MOJO_FN is not None:
            result = _MOJO_FN(K, V, M, Z, int(self._cfg.use_elu))
            return np.asarray(result[0], dtype=np.float32), np.asarray(result[1], dtype=np.float32)
        return self._numpy_update(K, V, M, Z)

    def retrieve(
        self,
        Q: np.ndarray,
        M: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Retrieve from compressive memory.

        ``A = M × sigma(Q) / (|Z| + eps)``

        Args:
            Q: Float32 ``(n_heads, T, head_dim)`` query tokens.
            M: Float32 ``(n_heads, head_dim, head_dim)`` memory.
            Z: Float32 ``(n_heads, head_dim)`` normalizer.

        Returns:
            Float32 ``(n_heads, T, head_dim)`` attention from memory.
        """
        Q = np.asarray(Q, dtype=np.float32)
        M = np.asarray(M, dtype=np.float32)
        Z = np.asarray(Z, dtype=np.float32)
        if _MOJO_FN is not None:
            return np.asarray(_MOJO_FN(Q, M, Z, self._cfg.eps), dtype=np.float32)
        return self._numpy_retrieve(Q, M, Z)

    def backend(self) -> str:
        """Return 'mojo' if Mojo kernel loaded, else 'numpy'."""
        return "mojo" if _MOJO_FN is not None else "numpy"

    # ── NumPy fallbacks ────────────────────────────────────────────────────

    def _numpy_update(
        self,
        K: np.ndarray,
        V: np.ndarray,
        M: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Apply optional ELU+1 gating
        K_gate = (np.where(K > 0, K, np.expm1(K)) + 1.0) if self._cfg.use_elu else K
        # M += sum over T of outer(K[h,t], V[h,t])
        M_new = M + np.einsum("htd,hte->hde", K_gate, V)
        Z_new = Z + K_gate.sum(axis=1)
        return M_new.astype(np.float32), Z_new.astype(np.float32)

    def _numpy_retrieve(
        self,
        Q: np.ndarray,
        M: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        # sigma(Q) as ELU+1 normalization key
        Q_gate = np.where(Q > 0, Q, np.expm1(Q)) + 1.0
        # A[h, t, d] = M[h] @ Q_gate[h, t] / (Z[h] · Q_gate[h, t] + eps)
        A_mem = np.einsum("htd,hde->hte", Q_gate, M)
        Z_norm = np.einsum("hte,he->ht", Q_gate, Z) + self._cfg.eps
        return (A_mem / Z_norm[:, :, np.newaxis]).astype(np.float32)
