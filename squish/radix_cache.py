"""Phase 2B — RadixTree: token-prefix trie cache.

Supersedes the SHA-256 LRU ``_PrefixCache`` in server.py while maintaining
full backward compatibility with its ``get()`` / ``put()`` / ``hits`` / ``size``
/ ``_maxsize`` interface.

Two distinct capabilities are unified here:

1. **Exact-match text response cache** (backward compat with ``_PrefixCache``)
   ``get(prompt_str)`` / ``put(prompt_str, text, finish)``
   Uses an ``OrderedDict``-backed LRU — O(1) operations, same semantics as
   the old cache.

2. **Token-prefix trie for KV reuse** (new — requires PagedKVCache)
   ``find_prefix(token_ids)`` → ``(prefix_len, block_refs)``
   ``insert_prefix(token_ids, block_refs)``
   Token sequences are stored in a compressed radix trie.  On a prefix hit
   the matching physical KV blocks can be forked into the new request's
   ``PageBlockTable``, eliminating the prefill pass for those tokens.
   Integration with ``PagedKVCache`` is handled by the server dispatch layer.

Thread safety
-------------
Both facilities are individually lock-protected.  Callers must not hold one
lock while acquiring the other.

Usage — text cache (drop-in for ``_PrefixCache``)::

    rt = RadixTree(maxsize=512)
    rt.put(prompt_str, response_text, "stop")
    cached = rt.get(prompt_str)   # (text, finish) | None

Usage — prefix trie (requires PagedKVCache in server.py)::

    rt.insert_prefix(token_ids, block_refs)
    prefix_len, block_refs = rt.find_prefix(token_ids)
"""

from __future__ import annotations

import collections
import hashlib
import time
import threading

__all__ = ["RadixNode", "RadixTree"]


# ── Radix trie ────────────────────────────────────────────────────────────────


class RadixNode:
    """One node in the token-id radix trie."""

    __slots__ = (
        "edge_tokens",   # list[int]: token-ids on the edge leading INTO this node
        "children",      # dict[int, RadixNode]: first-token-of-edge → child
        "block_refs",    # list[int]: physical KV block indices for this prefix
        "last_access",   # float: time.monotonic() of last hit
        "ref_count",     # int: generation requests currently holding these blocks
    )

    def __init__(self, edge_tokens: list[int] | None = None) -> None:
        self.edge_tokens: list[int]          = edge_tokens or []
        self.children:    dict[int, "RadixNode"] = {}
        self.block_refs:  list[int]          = []
        self.last_access: float              = time.monotonic()
        self.ref_count:   int                = 0

    def touch(self) -> None:
        self.last_access = time.monotonic()


# ── RadixTree ─────────────────────────────────────────────────────────────────


class RadixTree:
    """
    Thread-safe combined response cache + token-prefix trie.

    Parameters
    ----------
    maxsize : int
        Maximum number of stored text-response entries (LRU eviction applies).
        Set to 0 to disable the text response cache entirely.
    """

    def __init__(self, maxsize: int = 512) -> None:
        # ── Text response cache (backward compat with _PrefixCache) ──────────
        self._maxsize  = maxsize   # public attr — server.py writes to it directly
        self._cache: collections.OrderedDict[str, tuple[str, str]] = (
            collections.OrderedDict()
        )
        self._str_lock = threading.Lock()

        # ── Token-prefix trie ─────────────────────────────────────────────────
        self._root     = RadixNode()
        self._trie_lock = threading.Lock()

        # ── Metrics ──────────────────────────────────────────────────────────
        self.hits:        int = 0   # exact-match text hits
        self.misses:      int = 0   # exact-match text misses
        self.prefix_hits: int = 0   # token-prefix trie hits

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _key(prompt: str) -> str:
        # blake2b (128-bit) — ~3× faster than sha256, sufficient for a local
        # string-keyed cache with ≤4096 slots.
        return hashlib.blake2b(prompt.encode(), digest_size=16).hexdigest()

    # ── Text response cache API (drop-in for _PrefixCache) ────────────────────

    def get(self, prompt: str) -> tuple[str, str] | None:
        """
        Exact-match text response lookup.
        Returns ``(response_text, finish_reason)`` or ``None``.
        Equivalent to the old ``_PrefixCache.get()``.
        """
        k = self._key(prompt)
        with self._str_lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                self.hits += 1
                return self._cache[k]
            self.misses += 1
            return None

    def put(self, prompt: str, response: str, finish: str) -> None:
        """
        Store a full text response.
        Equivalent to the old ``_PrefixCache.put()``.
        Silently drops the entry when ``_maxsize == 0``.
        """
        if self._maxsize <= 0:
            return
        k = self._key(prompt)
        with self._str_lock:
            if k in self._cache:
                self._cache.move_to_end(k)
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)   # O(1) LRU evict
            self._cache[k] = (response, finish)

    def clear(self) -> None:
        """Flush both the text cache and the prefix trie."""
        with self._str_lock:
            self._cache.clear()
        with self._trie_lock:
            self._root = RadixNode()

    @property
    def size(self) -> int:
        """Number of stored response entries (text cache)."""
        with self._str_lock:
            return len(self._cache)

    # ── Token-prefix trie API (new — KV block reuse) ──────────────────────────

    def insert_prefix(
        self,
        token_ids: list[int],
        block_refs: list[int],
    ) -> None:
        """
        Associate physical KV block references with *token_ids*.

        Used by the server after a request completes to record which blocks
        cover the prompt prefix.  On a subsequent request sharing this prefix
        the matching blocks can be forked into the new request's
        ``PageBlockTable`` to skip prefill.
        """
        if not token_ids or not block_refs:
            return
        with self._trie_lock:
            node = self._trie_insert(token_ids)
            node.block_refs = list(block_refs)
            node.touch()

    def find_prefix(
        self,
        token_ids: list[int],
    ) -> tuple[int, list[int]]:
        """
        Find the longest stored prefix of *token_ids*.

        Returns ``(prefix_len, block_refs)``:

        * ``prefix_len`` — number of leading tokens matched (0 = no match).
        * ``block_refs``  — physical KV block indices for the matched prefix
          (empty when no KV prefix was stored or paged attention is off).
        """
        if not token_ids:
            return 0, []
        with self._trie_lock:
            prefix_len, node = self._trie_find_longest(token_ids)
            if prefix_len > 0 and node is not None and node.block_refs:
                node.touch()
                self.prefix_hits += 1
                return prefix_len, list(node.block_refs)
            return 0, []

    def evict_prefix_lru(self, n: int = 1) -> int:
        """
        Evict up to *n* trie nodes with block_refs by LRU.
        Returns the count actually evicted.
        Useful when ``PagedKVCache.free_count`` drops below a threshold.
        """
        with self._trie_lock:
            return self._trie_evict_lru(n)

    # ── Internal trie operations ──────────────────────────────────────────────

    def _trie_insert(self, token_ids: list[int]) -> RadixNode:
        """Walk / create the trie path for *token_ids*; return the leaf node."""
        node      = self._root
        remaining = list(token_ids)

        while remaining:
            first = remaining[0]
            child = node.children.get(first)

            if child is None:
                # No child for this token — attach a new leaf
                new_node = RadixNode(edge_tokens=list(remaining))
                node.children[first] = new_node
                return new_node

            edge   = child.edge_tokens
            common = sum(1 for a, b in zip(remaining, edge) if a == b)
            # Calculate common length properly (stop at first mismatch)
            common = 0
            for a, b in zip(remaining, edge):
                if a != b:
                    break
                common += 1

            if common == len(edge):
                # Entire edge matched — descend
                node      = child
                remaining = remaining[common:]
            else:
                # Partial match — split the edge at `common`
                split = RadixNode(edge_tokens=edge[:common])
                node.children[first] = split

                # Re-attach old child with the unmatched suffix
                child.edge_tokens = edge[common:]
                split.children[edge[common]] = child

                if remaining[common:]:
                    new_leaf = RadixNode(edge_tokens=remaining[common:])
                    split.children[remaining[common]] = new_leaf
                    return new_leaf
                return split

        return node

    def _trie_find_longest(
        self, token_ids: list[int]
    ) -> tuple[int, RadixNode | None]:
        """
        Return ``(n_matched, node)`` for the longest prefix of *token_ids*
        present in the trie.  *node* is the deepest matched node.
        """
        node      = self._root
        consumed  = 0
        remaining = list(token_ids)

        best_node     = None
        best_consumed = 0

        while remaining:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                break

            edge   = child.edge_tokens
            common = 0
            for a, b in zip(remaining, edge):
                if a != b:
                    break
                common += 1

            consumed  += common
            remaining  = remaining[common:]
            node       = child

            if node.block_refs:
                best_node     = node
                best_consumed = consumed

            if common < len(edge):
                # Partial edge match — cannot descend further
                break

        if best_node is not None:
            return best_consumed, best_node
        return consumed, node if consumed > 0 else None

    def _collect_trie_nodes_with_blocks(self) -> list[RadixNode]:
        """BFS: collect all trie nodes that have block_refs."""
        result: list[RadixNode] = []
        stack = [self._root]
        while stack:
            n = stack.pop()
            if n.block_refs:
                result.append(n)
            stack.extend(n.children.values())
        return result

    def _trie_evict_lru(self, n: int) -> int:
        """Evict up to *n* LRU trie nodes (block_refs cleared). Returns count."""
        nodes = sorted(
            self._collect_trie_nodes_with_blocks(),
            key=lambda x: x.last_access,
        )
        evicted = 0
        for node in nodes[:n]:
            if node.ref_count > 0:
                continue
            node.block_refs = []
            evicted += 1
        return evicted
