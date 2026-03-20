"""squish/token/sep_llm_compress.py

SepLLMCompress — KV Compression via Separator Token Retention.

Reference
---------
Chen et al. "SepLLM: Accelerate Large Language Models by Compressing One
Separator Token per Two Layers." ICLR 2025.

Algorithm
---------
LLM attention is heavily focused on separator tokens (punctuation, special
tokens like ``<|im_sep|>``, ``\n\n``, etc.) and the most recent few tokens.
SepLLM exploits this by:

1. **Identifying separator positions** in the input token sequence.
2. On **alternating layers** (even layers), retaining KV only for:
   - Separator token positions.
   - The most recent ``recent_window`` tokens.
3. On **odd layers**, retaining the full KV (unchanged).

This achieves ~2× KV reduction for instruction-following workloads because
separators are relatively sparse.

Key properties
--------------
* NumPy-only simulation; the separator detection is token-id based.
* ``sep_token_ids`` — set of token ids treated as separators.
* ``recent_window`` — unconditionally retain the last N KV positions.
* ``compress_even_layers`` — if True, compress even-indexed layers (default);
  if False, compress odd-indexed layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "SepLLMConfig",
    "SepLLMCompress",
]

# ── Default separator token ids (common special tokens) ───────────────────────
_DEFAULT_SEP_IDS: Set[int] = {
    13,    # '\n' (common in LLaMA tokenizers)
    2,     # </s>
    1,     # <s>
    29871, # space-newline in SentencePiece
    198,   # '\n' in GPT-2/Qwen tokenizers
    271,   # '\\n\\n'
    151643, # <|im_end|> in Qwen
    151644, # <|im_start|> in Qwen
}

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SepLLMConfig:
    """Configuration for :class:`SepLLMCompress`.

    Attributes:
        sep_token_ids: Set of token ids treated as separators.
        recent_window: Number of most-recent tokens always retained.
        compress_even_layers: Compress even-indexed layers; leave odd layers full.
        n_heads: Number of attention heads per layer.
        head_dim: Attention head dimension.
    """

    sep_token_ids: Set[int] = field(default_factory=lambda: set(_DEFAULT_SEP_IDS))
    recent_window: int = 64
    compress_even_layers: bool = True
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.recent_window < 1:
            raise ValueError(
                f"recent_window must be ≥ 1; got {self.recent_window}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class SepLLMCompress:
    """Separator-aware alternating-layer KV compression.

    Example::

        cfg   = SepLLMConfig(recent_window=16, n_heads=2, head_dim=8)
        comp  = SepLLMCompress(cfg)

        token_ids = [1, 42, 13, 55, 99, 2]  # 13 and 2 are separators
        K = np.random.randn(2, 6, 8).astype(np.float32)
        V = np.random.randn(2, 6, 8).astype(np.float32)

        K_out, V_out, kept_idx = comp.compress(
            layer_id=0, token_ids=token_ids, K=K, V=V
        )
    """

    def __init__(self, config: Optional[SepLLMConfig] = None) -> None:
        self.config = config or SepLLMConfig()

    # ── Core ──────────────────────────────────────────────────────────────────

    def compress(
        self,
        layer_id: int,
        token_ids: List[int],
        K: np.ndarray,
        V: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compress K/V for a given layer according to the SepLLM policy.

        Args:
            layer_id: Layer index used to decide whether to compress.
            token_ids: List of token ids for the current KV positions.
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.

        Returns:
            Tuple of:
            - ``K_out``: ``(n_heads, K', head_dim)`` compressed keys.
            - ``V_out``: ``(n_heads, K', head_dim)`` compressed values.
            - ``kept_idx``: ``(K',)`` array of retained position indices.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        S = K.shape[1]

        if not self._should_compress(layer_id):
            return K, V, np.arange(S, dtype=np.int32)

        kept = self._kept_indices(token_ids, S)
        return K[:, kept, :], V[:, kept, :], kept

    def _should_compress(self, layer_id: int) -> bool:
        """Return True if this layer should be compressed."""
        if self.config.compress_even_layers:
            return layer_id % 2 == 0
        else:
            return layer_id % 2 == 1

    def _kept_indices(self, token_ids: List[int], S: int) -> np.ndarray:
        """Compute which positions to retain.

        Returns:
            Sorted integer array of retained position indices.
        """
        cfg = self.config
        sep_ids = cfg.sep_token_ids
        recent_w = min(cfg.recent_window, S)

        kept: Set[int] = set()
        # Separator positions
        for i, tok in enumerate(token_ids[:S]):
            if tok in sep_ids:
                kept.add(i)
        # Recent window (force keep)
        for i in range(S - recent_w, S):
            kept.add(i)

        if not kept:
            # Degenerate: keep at least the last token
            kept.add(S - 1)

        return np.array(sorted(kept), dtype=np.int32)

    # ── Introspection ─────────────────────────────────────────────────────────

    def compression_ratio(self, token_ids: List[int], layer_id: int = 0) -> float:
        """Estimate the KV compression ratio for a given token sequence.

        Args:
            token_ids: Sequence of token ids.
            layer_id: Layer to simulate (affects whether compression applies).

        Returns:
            Fraction of positions retained in [0, 1].
        """
        S = len(token_ids)
        if not self._should_compress(layer_id):
            return 1.0
        kept = self._kept_indices(token_ids, S)
        return len(kept) / max(S, 1)

    def __repr__(self) -> str:
        return (
            f"SepLLMCompress(recent_window={self.config.recent_window}, "
            f"n_sep_ids={len(self.config.sep_token_ids)})"
        )
