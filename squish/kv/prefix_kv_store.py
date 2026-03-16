"""
squish/kv/prefix_kv_store.py

Phase 13 — RadixTree KV prefix reuse.

PrefixKVStore wires the RadixTree token-prefix trie (radix_cache.py) to an
in-memory LRU dictionary of QuantizedKVCache snapshots.  When a new request's
prompt shares a prefix with a stored snapshot the decode loop can:

  1. Restore the cached KV state via ``restore()``.
  2. Run the model only on the *delta* (new) tokens instead of the full prompt.
  3. After prefill, call ``store()`` to snapshot the KV for future reuse.

This converts O(n_prefix) → O(n_delta) prefill work for repeated prefixes such
as system prompts, few-shot examples, or shared conversation histories.

Thread safety
-------------
All public methods are protected by a single ``threading.Lock``.  The lock is
held only for the in-memory dictionary operations; the ``RadixTree`` trie has
its own lock and is called outside this lock to avoid lock ordering issues.

Usage
-----
    from squish.kv.prefix_kv_store import PrefixKVStore
    from squish.kv.radix_cache import RadixTree

    rt = RadixTree(maxsize=512)       # shared with text-response cache
    store = PrefixKVStore(rt, max_snapshots=8)

    # After prefill:
    store.store(input_ids, kv_cache)

    # Before next prefill:
    prefix_len, snap_id = store.find(input_ids)
    if prefix_len > 0:
        store.restore(snap_id, kv_cache)
        # then prefill only input_ids[prefix_len:]
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from squish.kv.kv_cache import QuantizedKVCache
    from squish.kv.radix_cache import RadixTree

__all__ = ["PrefixKVStore"]

# Minimum number of tokens a prefix must cover to be worth a snapshot.
# Short prefixes (< 8 tokens) provide negligible prefill savings.
_MIN_PREFIX_TOKENS: int = 8


class PrefixKVStore:
    """
    In-memory LRU store of post-prefill KV snapshots, keyed by token-prefix.

    Parameters
    ----------
    radix_tree : RadixTree
        Shared ``RadixTree`` instance (also used as the text-response cache
        in ``server.py``).  KV snapshots are registered in its token-prefix
        trie via ``insert_prefix`` / retrieved via ``find_prefix``.
    max_snapshots : int
        Maximum number of KV snapshots to keep in memory simultaneously.
        Each snapshot costs roughly ``n_layers × n_tokens × head_dim × 2 B``
        (FP16 equivalent) in system RAM.  Default 8.
    min_prefix_tokens : int
        Minimum prefix length required before a snapshot is stored.  Prefixes
        shorter than this offer negligible prefill savings.  Default 8.
    """

    def __init__(
        self,
        radix_tree: RadixTree,
        max_snapshots: int = 8,
        min_prefix_tokens: int = _MIN_PREFIX_TOKENS,
    ) -> None:
        self._tree = radix_tree
        self._max = max(1, max_snapshots)
        self._min_prefix = max(1, min_prefix_tokens)
        # snapshot_id → QuantizedKVCache clone
        self._store:   dict[int, QuantizedKVCache] = {}
        # LRU order: _order[0] is oldest
        self._order:   list[int] = []
        self._next_id: int = 0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def find(self, input_ids: list[int]) -> tuple[int, int | None]:
        """
        Look up the longest stored prefix of *input_ids*.

        Returns
        -------
        (prefix_len, snapshot_id)
            ``prefix_len`` — how many leading tokens are covered by the snapshot
            (0 when no usable match exists).
            ``snapshot_id`` — integer key for ``restore()`` (``None`` when no match).
        """
        if len(input_ids) <= self._min_prefix:
            return 0, None
        prefix_len, block_refs = self._tree.find_prefix(input_ids)
        if prefix_len < self._min_prefix or not block_refs:
            return 0, None
        snap_id = block_refs[0]
        with self._lock:
            if snap_id not in self._store:
                # Trie has a stale entry (snapshot was evicted) — ignore.
                return 0, None
            # Touch LRU
            if snap_id in self._order:
                self._order.remove(snap_id)
                self._order.append(snap_id)
        return prefix_len, snap_id

    def restore(self, snapshot_id: int, target: QuantizedKVCache) -> bool:
        """
        Copy the snapshot identified by *snapshot_id* into *target* in-place.

        Returns ``True`` on success, ``False`` if the snapshot was evicted.
        The caller must call ``target.reset()`` before this method.
        """
        with self._lock:
            snap = self._store.get(snapshot_id)
            if snap is None:
                return False
        # restore_from() is safe outside the lock: the snapshot is immutable
        # (its numpy arrays are never mutated after being stored).
        target.restore_from(snap)
        return True

    def store(self, input_ids: list[int], cache: QuantizedKVCache) -> int | None:
        """
        Clone *cache* and register the snapshot keyed by *input_ids*.

        Does nothing and returns ``None`` when *input_ids* is too short to be
        worth snapshotting.

        Parameters
        ----------
        input_ids : list[int]
            Full prompt token IDs that were used to produce *cache*.
        cache : QuantizedKVCache
            Post-prefill cache to snapshot.

        Returns
        -------
        int | None
            The new snapshot_id, or ``None`` if skipped.
        """
        if len(input_ids) < self._min_prefix:
            return None

        snap = cache.clone_snapshot()

        with self._lock:
            # LRU eviction
            while len(self._store) >= self._max:
                old_id = self._order.pop(0)
                self._store.pop(old_id, None)
            snap_id = self._next_id
            self._next_id += 1
            self._store[snap_id] = snap
            self._order.append(snap_id)

        # Register in the trie *outside* our lock to respect lock ordering.
        # block_refs = [snap_id] is the convention used by find() above.
        self._tree.insert_prefix(list(input_ids), [snap_id])
        return snap_id

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict:
        """Return a diagnostic summary dictionary."""
        with self._lock:
            n = len(self._store)
        return {
            "snapshots_stored": n,
            "max_snapshots": self._max,
            "prefix_hits": self._tree.prefix_hits,
            "min_prefix_tokens": self._min_prefix,
        }
