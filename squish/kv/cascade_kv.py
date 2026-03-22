"""squish/kv/cascade_kv.py

CascadeKV — Two-Level Cascade KV Cache for Shared-Prefix Batches.

Reference
---------
Juravsky et al. "Cascade: Memory Bandwidth Efficient Shared Prefixes for
LLM Inference." MLSys 2024 (arXiv:2406.19078).

Algorithm
---------
When multiple requests share a common prefix (system prompt, few-shot
examples, long document), the KV cache for that prefix can be computed
once and stored in a shared *Level-0* (L0) block.  Per-request context is
stored in a separate *Level-1* (L1) block.

At attention time, a two-pass FlashAttention is used:

1. Compute softmax(Q @ K_L0.T) and partial output O_L0.
2. Compute softmax(Q @ K_L1.T) and partial output O_L1.
3. Merge the two partial outputs using log-sum-exp renormalisation.

This yields numerically identical results to combining the KV tensors but
avoids copying the shared prefix into each request's KV buffer.

Key properties
--------------
* NumPy-only simulation.
* ``prefix_len`` — number of tokens in the shared prefix (L0 block).
* ``max_request_len`` — maximum per-request (L1) token count.
* ``n_heads`` — number of attention heads.
* ``head_dim`` — dimension per head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CascadeKVConfig",
    "CascadeKVBlock",
    "CascadeKV",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class CascadeKVConfig:
    """Configuration for :class:`CascadeKV`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        prefix_len: Maximum shared-prefix length (L0 block capacity).
        max_request_len: Maximum per-request KV token count (L1 block capacity).
        scale: Attention scale factor (default 1/sqrt(head_dim)).
    """

    n_heads: int = 8
    head_dim: int = 64
    prefix_len: int = 512
    max_request_len: int = 1024
    scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.scale is None:
            object.__setattr__(self, "scale", float(self.head_dim ** -0.5))


# ── Block ─────────────────────────────────────────────────────────────────────


@dataclass
class CascadeKVBlock:
    """A single KV block (L0 or L1 level).

    Attributes:
        keys: Shape ``(n_heads, seq_len, head_dim)``.
        values: Shape ``(n_heads, seq_len, head_dim)``.
        length: Number of valid tokens stored.
    """

    keys: np.ndarray
    values: np.ndarray
    length: int = 0

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Append a single token's K/V tensors.

        Parameters
        ----------
        k, v: Shape ``(n_heads, head_dim)``.
        """
        if self.length >= self.capacity:
            raise OverflowError(f"CascadeKVBlock is full (capacity={self.capacity})")
        self.keys[:, self.length, :] = k
        self.values[:, self.length, :] = v
        self.length += 1

    def reset(self) -> None:
        self.length = 0


# ── Module ────────────────────────────────────────────────────────────────────


class CascadeKV:
    """Two-level cascade KV cache.

    Parameters
    ----------
    config:
        Cascade KV configuration.
    """

    def __init__(self, config: Optional[CascadeKVConfig] = None) -> None:
        self._cfg = config or CascadeKVConfig()
        h, d = self._cfg.n_heads, self._cfg.head_dim
        # L0: shared prefix block
        self._l0 = CascadeKVBlock(
            keys=np.zeros((h, self._cfg.prefix_len, d), dtype=np.float32),
            values=np.zeros((h, self._cfg.prefix_len, d), dtype=np.float32),
        )
        # L1: per-request blocks keyed by request_id
        self._l1: Dict[int, CascadeKVBlock] = {}
        self._next_id: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> CascadeKVConfig:
        return self._cfg

    @property
    def prefix_length(self) -> int:
        return self._l0.length

    def set_shared_prefix(self, keys: np.ndarray, values: np.ndarray) -> None:
        """Store shared-prefix KV tensors into the L0 block.

        Parameters
        ----------
        keys, values:
            Shape ``(n_heads, prefix_len, head_dim)`` or
            ``(prefix_len, n_heads, head_dim)`` (auto-transposed).
        """
        k = np.asarray(keys, dtype=np.float32)
        v = np.asarray(values, dtype=np.float32)
        if k.ndim == 3 and k.shape[0] != self._cfg.n_heads:
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)
        plen = k.shape[1]
        if plen > self._cfg.prefix_len:
            raise ValueError(f"prefix_len {plen} exceeds capacity {self._cfg.prefix_len}")
        self._l0.keys[:, :plen, :] = k
        self._l0.values[:, :plen, :] = v
        self._l0.length = plen

    def create_request(self) -> int:
        """Allocate a new per-request L1 block.

        Returns
        -------
        int
            Request ID to use with :meth:`append_token` and :meth:`attend`.
        """
        req_id = self._next_id
        self._next_id += 1
        h, d = self._cfg.n_heads, self._cfg.head_dim
        self._l1[req_id] = CascadeKVBlock(
            keys=np.zeros((h, self._cfg.max_request_len, d), dtype=np.float32),
            values=np.zeros((h, self._cfg.max_request_len, d), dtype=np.float32),
        )
        return req_id

    def append_token(
        self,
        request_id: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Append a single decoded token's KV to an L1 block.

        Parameters
        ----------
        request_id: Request ID from :meth:`create_request`.
        key, value: Shape ``(n_heads, head_dim)``.
        """
        self._l1[request_id].append(
            np.asarray(key, dtype=np.float32),
            np.asarray(value, dtype=np.float32),
        )

    def attend(self, request_id: int, query: np.ndarray) -> np.ndarray:
        """Compute two-level cascade attention for one query step.

        Parameters
        ----------
        request_id:
            Request ID for the L1 per-request block.
        query:
            Shape ``(n_heads, head_dim)``.

        Returns
        -------
        np.ndarray
            Attention output, shape ``(n_heads, head_dim)``.
        """
        q = np.asarray(query, dtype=np.float32)  # (n_heads, head_dim)
        scale = float(self._cfg.scale)

        def _partial_attn(K: np.ndarray, V: np.ndarray, length: int):
            if length == 0:
                return None, None
            K_valid = K[:, :length, :]  # (n_heads, length, head_dim)
            V_valid = V[:, :length, :]
            scores = np.einsum("hd,hsd->hs", q, K_valid) * scale  # (n_heads, length)
            max_s = scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            sum_s = exp_s.sum(axis=-1, keepdims=True)
            w = exp_s / sum_s  # (n_heads, length)
            out = np.einsum("hs,hsd->hd", w, V_valid)  # (n_heads, head_dim)
            lse = np.log(sum_s.squeeze(-1)) + max_s.squeeze(-1)  # (n_heads,)
            return out, lse

        l1_block = self._l1[request_id]
        o_l0, lse_l0 = _partial_attn(self._l0.keys, self._l0.values, self._l0.length)
        o_l1, lse_l1 = _partial_attn(l1_block.keys, l1_block.values, l1_block.length)

        if o_l0 is None and o_l1 is None:
            return np.zeros((self._cfg.n_heads, self._cfg.head_dim), dtype=np.float32)
        if o_l0 is None:
            return o_l1
        if o_l1 is None:
            return o_l0

        # LSE-merge: output = (exp(lse0)*O0 + exp(lse1)*O1) / (exp(lse0) + exp(lse1))
        lse_max = np.maximum(lse_l0, lse_l1)
        w0 = np.exp(lse_l0 - lse_max)[:, None]
        w1 = np.exp(lse_l1 - lse_max)[:, None]
        merged = (w0 * o_l0 + w1 * o_l1) / (w0 + w1)
        return merged.astype(np.float32)

    def free_request(self, request_id: int) -> None:
        """Release per-request L1 block."""
        self._l1.pop(request_id, None)

    def reset_prefix(self) -> None:
        """Clear the shared prefix L0 block."""
        self._l0.reset()

    def active_requests(self) -> List[int]:
        return list(self._l1.keys())
