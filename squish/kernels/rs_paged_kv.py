"""squish/kernels/rs_paged_kv.py — Rust-backed paged KV-cache gather.

Wraps ``squish_quant_rs.paged_kv_gather_f32`` with a NumPy fallback
that implements the same non-contiguous block reconstruction logic.

Reference: Kwon et al., "Efficient Memory Management for Large Language
Model Serving with PagedAttention." SOSP 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PagedKVConfig",
    "RustPagedKVGather",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "paged_kv_gather_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_gather(
    kv_pool: np.ndarray,    # (max_blocks, n_heads, block_size, head_dim) flat→2D
    page_table: np.ndarray, # (n_pages,) int32
    n_heads: int,
    block_size: int,
    head_dim: int,
    n_valid_tokens: int,
) -> np.ndarray:
    out = np.empty((n_valid_tokens, n_heads, head_dim), dtype=np.float32)
    pool_4d = kv_pool.reshape(-1, n_heads, block_size, head_dim)
    for tok in range(n_valid_tokens):
        page = int(page_table[tok // block_size])
        pos_in_page = tok % block_size
        out[tok] = pool_4d[page, :, pos_in_page, :]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class PagedKVConfig:
    """Configuration for :class:`RustPagedKVGather`.

    Attributes:
        n_heads:    Number of KV attention heads.
        block_size: Tokens per physical page.
        head_dim:   Head dimension.
    """

    n_heads: int = 8
    block_size: int = 16
    head_dim: int = 128


class RustPagedKVGather:
    """Rust-accelerated paged KV-cache block gather.

    Reconstructs a contiguous ``(n_valid_tokens, n_heads, head_dim)`` tensor
    from non-contiguous physical pages using a page table.
    Falls back to a NumPy loop when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[PagedKVConfig] = None) -> None:
        self._cfg = config or PagedKVConfig()

    def gather(
        self,
        kv_pool: np.ndarray,
        page_table: np.ndarray,
        n_valid_tokens: int,
        n_heads: Optional[int] = None,
        block_size: Optional[int] = None,
        head_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Gather K or V tensor from paged physical storage.

        Args:
            kv_pool:        Physical KV pool – shape
                            ``(max_blocks, n_heads, block_size, head_dim)``
                            or any 2D reshape thereof; float32.
            page_table:     Physical block IDs for this sequence ``(n_pages,)``
                            int32.
            n_valid_tokens: Number of tokens to reconstruct.
            n_heads:        Override config n_heads.
            block_size:     Override config block_size.
            head_dim:       Override config head_dim.

        Returns:
            Contiguous tensor ``(n_valid_tokens, n_heads, head_dim)`` float32.
            (Returned flat (n_valid_tokens * n_heads * head_dim,) for the Rust
            path; reshaped automatically.)
        """
        nh = int(n_heads) if n_heads is not None else self._cfg.n_heads
        bs = int(block_size) if block_size is not None else self._cfg.block_size
        hd = int(head_dim) if head_dim is not None else self._cfg.head_dim
        pool_f = np.ascontiguousarray(kv_pool, dtype=np.float32).reshape(-1, hd)
        pt_f = np.ascontiguousarray(page_table, dtype=np.int32).ravel()
        if _HAS_RUST:
            raw = _sq.paged_kv_gather_f32(pool_f, pt_f, nh, bs, hd, n_valid_tokens)
            return np.asarray(raw, dtype=np.float32).reshape(n_valid_tokens, nh, hd)
        pool_4d = pool_f.reshape(-1, nh, bs, hd)
        return _numpy_gather(pool_4d.reshape(-1, hd), pt_f, nh, bs, hd, n_valid_tokens)

    # ── properties ───────────────────────────────────────────────────────────

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def block_size(self) -> int:
        return self._cfg.block_size

    def head_dim(self) -> int:
        return self._cfg.head_dim

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
