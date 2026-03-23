"""squish/kv/radix_attn.py

RadixAttentionCache — Radix-tree KV prefix deduplication across concurrent
requests (SGLang / RadixAttention, Zheng et al., SOSP 2024).

Reference
---------
"SGLang: Efficient Execution of Structured Language Model Programs."
Zheng et al., SOSP 2024 (arXiv:2312.07104).

Algorithm
---------
1. Each request arrives with a token sequence.
2. The cache traverses a radix tree keyed by token ID sequences.
3. The longest common prefix is found and its KV tensors reused.
4. Only the suffix (novel tokens) need to be computed and inserted.
5. Leaf nodes are evicted LRU-style when the token budget is exceeded.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``max_tokens`` — total KV token budget across all cached prefixes.
* ``n_heads`` / ``head_dim`` — standard attention dimensionality.
* Thread-safe via simple dict-based state (simulation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "RadixAttentionConfig",
    "RadixNode",
    "RadixAttentionCache",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RadixAttentionConfig:
    """Configuration for :class:`RadixAttentionCache`.

    Attributes:
        max_tokens: Total KV token budget across all cached prefixes.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    max_tokens: int = 4096
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be ≥ 1; got {self.max_tokens}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")

    @property
    def max_nodes(self) -> int:  # server.py compatibility alias
        return self.max_tokens


# ── Radix tree node ───────────────────────────────────────────────────────────


class RadixNode:
    """A node in the radix KV-cache tree.

    Attributes:
        token_ids: Token ID sequence stored at this edge.
        K: Cached key tensor ``(n_heads, len(token_ids), head_dim)``.
        V: Cached value tensor ``(n_heads, len(token_ids), head_dim)``.
        children: Child nodes keyed by first token ID of their edge.
        last_access: Timestamp of last access for LRU eviction.
    """

    def __init__(
        self,
        token_ids: Tuple[int, ...],
        K: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
    ) -> None:
        self.token_ids: Tuple[int, ...] = token_ids
        self.K: Optional[np.ndarray] = K
        self.V: Optional[np.ndarray] = V
        self.children: Dict[int, "RadixNode"] = {}
        self.last_access: float = time.monotonic()

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


# ── Cache ─────────────────────────────────────────────────────────────────────


class RadixAttentionCache:
    """Radix-tree KV prefix cache.

    Example::

        cfg = RadixAttentionConfig(max_tokens=256, n_heads=4, head_dim=8)
        cache = RadixAttentionCache(cfg)
        tokens = [1, 2, 3, 4, 5]
        K = np.random.randn(4, 5, 8).astype(np.float32)
        V = np.random.randn(4, 5, 8).astype(np.float32)
        cache.insert(tokens, K, V)
        prefix_len = cache.match_prefix([1, 2, 3, 6, 7])
        K_pre, V_pre = cache.lookup(tokens[:prefix_len])
    """

    def __init__(self, config: Optional[RadixAttentionConfig] = None) -> None:
        self.config = config or RadixAttentionConfig()
        self._root: RadixNode = RadixNode(token_ids=())
        self._n_cached_tokens: int = 0
        self._n_hits: int = 0
        self._n_misses: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(
        self,
        token_ids: List[int],
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Insert a full KV sequence into the radix tree.

        Args:
            token_ids: Token ID sequence (prefix + new tokens).
            K: ``(n_heads, len(token_ids), head_dim)``.
            V: ``(n_heads, len(token_ids), head_dim)``.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        if K.shape[1] != len(token_ids):
            raise ValueError(
                f"K.shape[1]={K.shape[1]} must equal len(token_ids)={len(token_ids)}"
            )
        self._evict_if_needed(len(token_ids))
        self._insert_node(self._root, tuple(token_ids), K, V)

    def match_prefix(self, token_ids: List[int]) -> int:
        """Return the length of the longest cached prefix of *token_ids*.

        Args:
            token_ids: Query token sequence.

        Returns:
            Number of leading tokens already cached (0 if no match).
        """
        matched = self._match(self._root, tuple(token_ids), 0)
        if matched > 0:
            self._n_hits += 1
        else:
            self._n_misses += 1
        return matched

    def lookup(self, token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve cached KV tensors for an exact token prefix.

        Args:
            token_ids: Token sequence that must be an exact cached prefix.

        Returns:
            ``(K, V)`` concatenated along the sequence dimension.

        Raises:
            KeyError: If *token_ids* is not an exact cached prefix.
        """
        k_parts: List[np.ndarray] = []
        v_parts: List[np.ndarray] = []
        found = self._collect(self._root, tuple(token_ids), k_parts, v_parts)
        if not found:
            self._n_misses += 1
            raise KeyError(f"Token prefix not cached: {token_ids[:8]}...")
        self._n_hits += 1
        K = np.concatenate(k_parts, axis=1) if k_parts else np.empty(
            (self.config.n_heads, 0, self.config.head_dim), dtype=np.float32
        )
        V = np.concatenate(v_parts, axis=1) if v_parts else np.empty(
            (self.config.n_heads, 0, self.config.head_dim), dtype=np.float32
        )
        return K, V

    def n_cached_tokens(self) -> int:
        """Total number of token positions currently cached."""
        return self._n_cached_tokens

    def clear(self) -> None:
        """Evict all cached KV tensors."""
        self._root = RadixNode(token_ids=())
        self._n_cached_tokens = 0

    @property
    def n_hits(self) -> int:
        return self._n_hits

    @property
    def n_misses(self) -> int:
        return self._n_misses

    def hit_rate(self) -> float:
        """Cache hit rate over all lookups."""
        total = self._n_hits + self._n_misses
        return self._n_hits / total if total > 0 else 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _insert_node(
        self,
        node: RadixNode,
        token_ids: Tuple[int, ...],
        K: np.ndarray,
        V: np.ndarray,
        offset: int = 0,
    ) -> None:
        if offset >= len(token_ids):
            return
        first = token_ids[offset]
        if first not in node.children:
            child = RadixNode(
                token_ids=token_ids[offset:],
                K=K[:, offset:, :],
                V=V[:, offset:, :],
            )
            node.children[first] = child
            self._n_cached_tokens += len(token_ids) - offset
            return
        child = node.children[first]
        child.last_access = time.monotonic()
        # Find common prefix length between child edge and remaining tokens
        edge = child.token_ids
        remaining = token_ids[offset:]
        cmn = 0
        while cmn < len(edge) and cmn < len(remaining) and edge[cmn] == remaining[cmn]:
            cmn += 1
        if cmn == len(edge):
            # Edge fully consumed — recurse deeper into child
            self._insert_node(child, token_ids, K, V, offset + cmn)
        else:
            # Split: create intermediate node for shared prefix
            shared_tokens = edge[:cmn]
            new_child = RadixNode(
                token_ids=shared_tokens,
                K=K[:, offset : offset + cmn, :],
                V=V[:, offset : offset + cmn, :],
            )
            # Reattach old child under new split node
            old_rest = RadixNode(
                token_ids=edge[cmn:],
                K=child.K[:, cmn:, :] if child.K is not None else None,
                V=child.V[:, cmn:, :] if child.V is not None else None,
            )
            old_rest.children = child.children
            new_child.children[edge[cmn]] = old_rest
            node.children[first] = new_child
            # Insert new suffix
            if offset + cmn < len(token_ids):
                new_suffix = RadixNode(
                    token_ids=token_ids[offset + cmn :],
                    K=K[:, offset + cmn :, :],
                    V=V[:, offset + cmn :, :],
                )
                new_child.children[token_ids[offset + cmn]] = new_suffix
                self._n_cached_tokens += len(token_ids) - (offset + cmn)

    def _match(
        self, node: RadixNode, token_ids: Tuple[int, ...], matched: int
    ) -> int:
        if not token_ids:
            return matched
        first = token_ids[0]
        if first not in node.children:
            return matched
        child = node.children[first]
        edge = child.token_ids
        cmn = 0
        while cmn < len(edge) and cmn < len(token_ids) and edge[cmn] == token_ids[cmn]:
            cmn += 1
        if cmn == len(edge):
            return self._match(child, token_ids[cmn:], matched + cmn)
        return matched + cmn

    def _collect(
        self,
        node: RadixNode,
        token_ids: Tuple[int, ...],
        k_parts: List[np.ndarray],
        v_parts: List[np.ndarray],
    ) -> bool:
        if not token_ids:
            return True
        first = token_ids[0]
        if first not in node.children:
            return False
        child = node.children[first]
        edge = child.token_ids
        if not token_ids[: len(edge)] == edge:
            return False
        if child.K is not None:
            k_parts.append(child.K)
            v_parts.append(child.V)  # type: ignore[arg-type]
        return self._collect(child, token_ids[len(edge) :], k_parts, v_parts)

    def _evict_if_needed(self, n_new: int) -> None:
        if self._n_cached_tokens + n_new <= self.config.max_tokens:
            return
        need = self._n_cached_tokens + n_new - self.config.max_tokens
        self._evict_lru(self._root, need)

    def _evict_lru(self, node: RadixNode, need: int) -> int:
        """Evict leaf nodes LRU until *need* tokens have been freed. Returns freed count."""
        if need <= 0:
            return 0
        freed = 0
        to_remove = []
        for key, child in node.children.items():
            if not child.children:
                # Leaf — evict
                freed_here = child.n_tokens
                self._n_cached_tokens -= freed_here
                freed += freed_here
                to_remove.append(key)
                if freed >= need:
                    break
            else:
                freed += self._evict_lru(child, need - freed)
                if freed >= need:
                    break
        for key in to_remove:
            del node.children[key]
        return freed

    def __repr__(self) -> str:
        return (
            f"RadixAttentionCache(max_tokens={self.config.max_tokens}, "
            f"n_cached={self._n_cached_tokens}, hit_rate={self.hit_rate():.2%})"
        )
