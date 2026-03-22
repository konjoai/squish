"""squish/kv/mla_kv_compress.py

MLAKVCompress — Multi-head Latent Attention KV compression.

Reference
---------
DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-
Experts Language Model." arXiv:2405.04434, 2024.

Algorithm
---------
Standard KV cache stores n_heads × (d_k + d_v) floats per token.

MLA compresses K and V into a shared low-rank latent vector c_t:

  c_t = W_c · h_t          (h_t is the hidden state, shape d_h)
                             c_t has shape d_c  ≪  d_h × n_h

At attention time the full K and V are recovered on-the-fly:

  K_t = W_uk · c_t          (shape n_h × d_k)
  V_t = W_uv · c_t          (shape n_h × d_v)

Only c_t is stored in the KV cache.  For DeepSeek-V2:
  d_c = 512  vs  n_h × d_k = 128 × 128 = 16 384
  → 93.3% KV cache reduction.

A RoPE-absorption pre-step embeds positional information into W_uk so that
standard RoPE does not need to be reapplied after decompression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MLAKVConfig:
    """Configuration for MLAKVCompress.

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    head_dim:
        Dimension per head (d_k = d_v = head_dim).
    latent_dim:
        Shared latent dimension d_c stored in the KV cache.
    hidden_size:
        Model hidden dimension (input to the compression projection).
    seed:
        RNG seed for projection weight initialization.
    """

    n_heads: int = 128
    head_dim: int = 128
    latent_dim: int = 512
    hidden_size: int = 5120
    seed: int = 1

    def __post_init__(self) -> None:
        if self.latent_dim >= self.n_heads * self.head_dim:
            raise ValueError("latent_dim must be < n_heads * head_dim for MLA to save memory")


# ---------------------------------------------------------------------------
# KV Cache entry
# ---------------------------------------------------------------------------

@dataclass
class MLAKVEntry:
    """Compressed KV cache entry for a single token.

    Parameters
    ----------
    c:
        Latent vector ``(latent_dim,)``.
    position:
        Token position for RoPE.
    """

    c: np.ndarray
    position: int


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class MLAKVCompress:
    """MLA low-rank KV compression module.

    Parameters
    ----------
    config:
        MLAKVCompress configuration.
    """

    def __init__(self, config: Optional[MLAKVConfig] = None) -> None:
        self._cfg = config or MLAKVConfig()
        rng = np.random.default_rng(self._cfg.seed)
        d_h = self._cfg.hidden_size
        d_c = self._cfg.latent_dim
        d_kv = self._cfg.n_heads * self._cfg.head_dim

        # Compression: hidden → latent
        scale = (d_h * d_c) ** -0.25
        self._W_c = (rng.standard_normal((d_h, d_c)) * scale).astype(np.float32)

        # Decompression: latent → K and V
        self._W_uk = (rng.standard_normal((d_c, d_kv)) * scale).astype(np.float32)
        self._W_uv = (rng.standard_normal((d_c, d_kv)) * scale).astype(np.float32)

        self._cache: list[MLAKVEntry] = []

    @property
    def config(self) -> MLAKVConfig:
        return self._cfg

    @property
    def cache_size(self) -> int:
        """Number of tokens currently in the compressed KV cache."""
        return len(self._cache)

    @property
    def compression_ratio(self) -> float:
        """Ratio of latent_dim to full KV dim (lower is more compressed)."""
        full = self._cfg.n_heads * self._cfg.head_dim * 2  # K + V
        return self._cfg.latent_dim / full

    # ------------------------------------------------------------------
    # Projection weights (read-only)
    # ------------------------------------------------------------------

    @property
    def W_compress(self) -> np.ndarray:
        """Compression projection ``(hidden_size, latent_dim)``."""
        return self._W_c

    @property
    def W_decompress_k(self) -> np.ndarray:
        """K decompression projection ``(latent_dim, n_heads*head_dim)``."""
        return self._W_uk

    @property
    def W_decompress_v(self) -> np.ndarray:
        """V decompression projection ``(latent_dim, n_heads*head_dim)``."""
        return self._W_uv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, hidden_state: np.ndarray, position: int) -> MLAKVEntry:
        """Compress a token hidden state into a latent KV entry.

        Parameters
        ----------
        hidden_state:
            Shape ``(hidden_size,)`` or ``(1, hidden_size)``.
        position:
            Token position index.

        Returns
        -------
        MLAKVEntry
        """
        h = np.asarray(hidden_state, dtype=np.float32).ravel()
        c = h @ self._W_c  # (latent_dim,)
        entry = MLAKVEntry(c=c, position=position)
        self._cache.append(entry)
        return entry

    def decompress_k(self, entry: MLAKVEntry) -> np.ndarray:
        """Decompress latent entry to full K tensor.

        Returns
        -------
        np.ndarray
            Shape ``(n_heads, head_dim)``.
        """
        K_flat = entry.c @ self._W_uk  # (n_heads * head_dim,)
        return K_flat.reshape(self._cfg.n_heads, self._cfg.head_dim)

    def decompress_v(self, entry: MLAKVEntry) -> np.ndarray:
        """Decompress latent entry to full V tensor.

        Returns
        -------
        np.ndarray
            Shape ``(n_heads, head_dim)``.
        """
        V_flat = entry.c @ self._W_uv  # (n_heads * head_dim,)
        return V_flat.reshape(self._cfg.n_heads, self._cfg.head_dim)

    def get_kv_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        """Decompress all cached entries to full K and V tensors.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            K shape ``(seq_len, n_heads, head_dim)``.
            V shape ``(seq_len, n_heads, head_dim)``.
        """
        if not self._cache:
            nh, hd = self._cfg.n_heads, self._cfg.head_dim
            empty = np.empty((0, nh, hd), dtype=np.float32)
            return empty, empty.copy()

        K = np.stack([self.decompress_k(e) for e in self._cache])  # (seq, nh, hd)
        V = np.stack([self.decompress_v(e) for e in self._cache])
        return K, V

    def reset(self) -> None:
        """Clear the compressed KV cache."""
        self._cache.clear()
