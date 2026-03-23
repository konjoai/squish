"""squish/attention/infini_attn.py

InfiniAttention — Memory-efficient infinite-context transformer with
segment-level compressive memory and local attention
(Munkhdalai et al., ICML 2024 / arXiv:2404.07143).

Reference
---------
"Leave No Context Behind: Efficient Infinite Context Transformers with
Infini-attention." Munkhdalai et al., ICML 2024 (arXiv:2404.07143).

Algorithm
---------
InfiniAttention combines:

1. **Local (dot-product) attention** over the current segment of length L.
2. **Compressive memory** M updated with each processed segment:
      M_new = M + K^T @ V    (associative key-value memory)
      Z_new = Z + sum(K, axis=0)  (normalisation accumulator)

3. **Memory retrieval**:
      A_mem = softmax(Q @ M_prev) / (Q @ Z_prev + eps)

4. **Fusion** via a learnable scalar (β):
      A_final = sigmoid(β) * A_mem + (1 - sigmoid(β)) * A_local

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* Memory is bounded — O(d²) per head regardless of sequence length.
* ``segment_len`` — tokens processed per segment (local attention window).
* ``n_heads``     — number of attention heads.
* ``head_dim``    — dimension per head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "InfiniAttentionConfig",
    "InfiniAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class InfiniAttentionConfig:
    """Configuration for :class:`InfiniAttention`.

    Attributes:
        segment_len: Tokens per segment for local attention.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        causal: Whether to apply causal masking within each segment.
        beta_init: Initial value of the learnable fusion gate β.
    """

    segment_len: int = 64
    n_heads: int = 8
    head_dim: int = 64
    causal: bool = True
    beta_init: float = 0.0

    def __post_init__(self) -> None:
        if self.segment_len < 1:
            raise ValueError(f"segment_len must be ≥ 1; got {self.segment_len}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")

    @property
    def segment_size(self) -> int:  # server.py compatibility alias
        return self.segment_len

    @property
    def memory_dim(self) -> int:  # server.py compatibility alias
        return self.n_heads * self.head_dim


# ── InfiniAttention ───────────────────────────────────────────────────────────


class InfiniAttention:
    """Segment-level infinite-context attention with compressive memory.

    Example::

        cfg = InfiniAttentionConfig(segment_len=32, n_heads=4, head_dim=8)
        attn = InfiniAttention(cfg)

        # Process a long sequence in segments
        for Q_seg, K_seg, V_seg in segment_iterator(Q, K, V, seg_len=32):
            out_seg = attn.forward(Q_seg, K_seg, V_seg)

        attn.reset_memory()  # clear between document boundaries
    """

    def __init__(self, config: Optional[InfiniAttentionConfig] = None) -> None:
        self.config = config or InfiniAttentionConfig()
        H, d = self.config.n_heads, self.config.head_dim
        # Compressive memory: (H, d, d)
        self._M: np.ndarray = np.zeros((H, d, d), dtype=np.float32)
        # Normalisation accumulator: (H, d)
        self._Z: np.ndarray = np.zeros((H, d), dtype=np.float32)
        # Learnable fusion gate (scalar per head)
        self._beta: np.ndarray = np.full(H, self.config.beta_init, dtype=np.float32)
        self._n_segments: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Process one segment of Q/K/V with local + memory attention.

        Args:
            Q: ``(n_heads, T, head_dim)`` — query tensor for this segment.
            K: ``(n_heads, T, head_dim)`` — key tensor for this segment.
            V: ``(n_heads, T, head_dim)`` — value tensor for this segment.

        Returns:
            ``(n_heads, T, head_dim)`` fused attention output.

        Raises:
            ValueError: If tensors have incompatible shapes.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        H, T, d = Q.shape
        if H != self.config.n_heads:
            raise ValueError(f"Expected n_heads={self.config.n_heads}; got H={H}")
        if K.shape[1] != T or V.shape[1] != T:
            raise ValueError("Q, K, V must have equal sequence length")

        scale = 1.0 / np.sqrt(d)

        # ── Memory retrieval (from previous segments) ──────────────────────
        # A_mem: (H, T, d) — memory-based attention output
        # Q @ M: (H, T, d)  (Q is (H,T,d), M is (H,d,d))
        QM = np.einsum("htd,hde->hte", Q, self._M)   # (H, T, d)
        # Q @ Z: (H, T)
        QZ = np.einsum("htd,hd->ht", Q, self._Z)     # (H, T)
        # Normalised memory read
        A_mem = QM / (np.abs(QZ)[..., None] + 1e-9)  # (H, T, d)

        # ── Local attention over current segment ───────────────────────────
        scores = np.einsum("hqd,hkd->hqk", Q, K) * scale  # (H, T, T)
        if self.config.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[:, mask] = -1e30
        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_w = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9)
        A_local = np.einsum("hqk,hkd->hqd", attn_w, V)    # (H, T, d)

        # ── Fusion via sigmoid(β) ──────────────────────────────────────────
        beta_s = 1.0 / (1.0 + np.exp(-self._beta))  # (H,)
        out = beta_s[:, None, None] * A_mem + (1.0 - beta_s[:, None, None]) * A_local

        # ── Update compressive memory with current K/V ─────────────────────
        # M += K^T @ V  →  (H, d, d)
        self._M += np.einsum("htd,hte->hde", K, V)
        # Z += sum(K, axis=T)  → (H, d)
        self._Z += K.sum(axis=1)
        self._n_segments += 1

        return out.astype(np.float32)

    def reset_memory(self) -> None:
        """Clear the compressive memory (e.g., at document boundaries)."""
        H, d = self.config.n_heads, self.config.head_dim
        self._M = np.zeros((H, d, d), dtype=np.float32)
        self._Z = np.zeros((H, d), dtype=np.float32)
        self._n_segments = 0

    def memory_bytes(self) -> int:
        """DRAM bytes used by the compressive memory state."""
        return self._M.nbytes + self._Z.nbytes

    @property
    def n_segments(self) -> int:
        """Number of segments processed since last :meth:`reset_memory`."""
        return self._n_segments

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"InfiniAttention(segment_len={cfg.segment_len}, "
            f"n_heads={cfg.n_heads}, head_dim={cfg.head_dim}, "
            f"n_segments={self._n_segments})"
        )
