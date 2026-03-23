"""squish/kernels/mojo/paged_kv_mojo.py — Mojo-backed paged KV-cache gather.

Wraps the ``paged_kv_gather`` Mojo kernel via MojoBridge with a NumPy fallback.

Reference: Kwon et al., "Efficient Memory Management for Large Language Model
Serving with PagedAttention." SOSP 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "PagedKVMojoConfig",
    "MojoPagedKVGather",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("paged_kv_gather")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_gather(
    kv_pool: np.ndarray,
    page_table: np.ndarray,
    n_heads: int,
    block_size: int,
    head_dim: int,
    n_valid_tokens: int,
) -> np.ndarray:
    pool_4d = kv_pool.reshape(-1, n_heads, block_size, head_dim)
    out = np.empty((n_valid_tokens, n_heads, head_dim), dtype=np.float32)
    for tok in range(n_valid_tokens):
        page = int(page_table[tok // block_size])
        pos = tok % block_size
        out[tok] = pool_4d[page, :, pos, :]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class PagedKVMojoConfig:
    """Configuration for :class:`MojoPagedKVGather`.

    Attributes:
        n_heads:    Number of KV attention heads.
        block_size: Tokens per physical page.
        head_dim:   Head dimension.
    """

    n_heads: int = 8
    block_size: int = 16
    head_dim: int = 128


class MojoPagedKVGather:
    """Mojo-backed paged KV-cache block gather.

    Uses a Mojo kernel with ``parallelize`` over tokens and vectorised
    head-dim copy for efficient non-contiguous physical block reconstruction.
    Falls back to a NumPy loop when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[PagedKVMojoConfig] = None) -> None:
        self._cfg = config or PagedKVMojoConfig()

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

        Returns:
            ``(n_valid_tokens, n_heads, head_dim)`` float32.
        """
        nh = int(n_heads) if n_heads is not None else self._cfg.n_heads
        bs = int(block_size) if block_size is not None else self._cfg.block_size
        hd = int(head_dim) if head_dim is not None else self._cfg.head_dim
        pool_f = np.ascontiguousarray(kv_pool, dtype=np.float32).reshape(-1, nh, bs, hd)
        pt_f = np.ascontiguousarray(page_table, dtype=np.int32).ravel()
        if _kernel is not None:
            pool_2d = pool_f.reshape(-1, hd)
            out_buf = np.zeros((n_valid_tokens, nh, hd), dtype=np.float32)
            _kernel(
                pool_2d.ctypes.data, pt_f.ctypes.data,
                out_buf.ctypes.data, nh, bs, hd, n_valid_tokens,
            )
            return out_buf
        return _numpy_gather(pool_f.reshape(-1, hd), pt_f, nh, bs, hd, n_valid_tokens)

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def block_size(self) -> int:
        return self._cfg.block_size

    def head_dim(self) -> int:
        return self._cfg.head_dim

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
