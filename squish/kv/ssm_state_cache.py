"""SSMStateCache: unified recurrent state store for SSM/linear-RNN architectures.

Different architecture families carry different recurrent state layouts:
  * Mamba2: ``conv_state`` (d_inner, d_conv-1) + ``ssm_state`` (n_heads, d_head, d_state)
  * RWKV-6: ``time_state`` (n_heads, head_dim, d_state)
  * Hawk/Griffin: ``h`` (d_state,)
  * xLSTM sLSTM: ``c`` (n_heads, head_dim) + ``n`` + ``m``
  * xLSTM mLSTM: ``C`` (d, d) + ``n`` (d,)
  * TTT: ``W`` (md, md) + ``velocity``
  * DeltaNet: ``W`` (n_heads, head_dim, d_state)

SSMStateCache normalises all of these into a versioned byte-buffer keyed by
session ID, enabling O(state_dim) session restoration at arbitrary positions —
decoupling conversation length from on-device memory.

Cache entries use Python ``bytes`` for serialisation; in production these would
be DMA-copied to CPU DRAM or SSD.  The LRU eviction policy keeps the N most
recently active sessions in cache.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "SSMStateCacheConfig",
    "SSMCacheEntry",
    "SSMStateCache",
]

_SUPPORTED_ARCH = {"mamba2", "rwkv6", "hawk", "xlstm_slstm", "xlstm_mlstm", "ttt", "deltanet"}


@dataclass
class SSMStateCacheConfig:
    """Configuration for :class:`SSMStateCache`.

    Attributes:
        max_sessions: Maximum number of sessions to keep in the LRU cache.
        compress: If True, state bytes are stored as-is (placeholder for FP8
            compression in production).
        seed: Unused; for API consistency.
    """

    max_sessions: int = 256
    compress: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_sessions < 1:
            raise ValueError(f"max_sessions must be ≥ 1, got {self.max_sessions}")


@dataclass
class SSMCacheEntry:
    """A single session's serialised recurrent state.

    Attributes:
        session_id: String identifier for the session.
        arch: Architecture family name (one of :py:data:`_SUPPORTED_ARCH`).
        state_bytes: Serialised state payload.
        n_tokens: Number of tokens the state represents.
        timestamp: Monotonic last-access time.
    """

    session_id: str
    arch: str
    state_bytes: bytes
    n_tokens: int = 0
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def size_bytes(self) -> int:
        return len(self.state_bytes)


class SSMStateCache:
    """LRU state cache supporting multiple SSM architecture families.

    Usage::

        cfg = SSMStateCacheConfig(max_sessions=64)
        cache = SSMStateCache(cfg)

        # Serialise a Hawk state
        state = HawkState(h=np.zeros(256))
        cache.put("session-1", "hawk", {"h": state.h}, n_tokens=512)

        # Restore
        arrays = cache.get("session-1")
        if arrays:
            state.h = arrays["h"]
    """

    def __init__(self, config: SSMStateCacheConfig) -> None:
        self.config = config
        self._store: Dict[str, SSMCacheEntry] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def put(
        self,
        session_id: str,
        arch: str,
        arrays: Dict[str, np.ndarray],
        n_tokens: int = 0,
    ) -> None:
        """Serialise and cache the recurrent state for a session.

        Args:
            session_id: Unique session identifier.
            arch: Architecture key (e.g. ``"hawk"``, ``"mamba2"``).
            arrays: Named numpy arrays constituting the recurrent state.
            n_tokens: Number of tokens represented by this state.
        """
        if arch not in _SUPPORTED_ARCH:
            raise ValueError(
                f"Unknown arch '{arch}'; supported: {sorted(_SUPPORTED_ARCH)}"
            )
        payload = self._serialize(arrays)
        if session_id in self._store:
            entry = self._store[session_id]
            entry.state_bytes = payload
            entry.n_tokens = n_tokens
            entry.timestamp = time.monotonic()
        else:
            if len(self._store) >= self.config.max_sessions:
                self._evict_lru()
            self._store[session_id] = SSMCacheEntry(
                session_id=session_id,
                arch=arch,
                state_bytes=payload,
                n_tokens=n_tokens,
            )

    def get(
        self, session_id: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Restore cached state arrays for a session.

        Returns:
            Dict of named arrays on hit, or ``None`` on miss.
        """
        entry = self._store.get(session_id)
        if entry is None:
            self._misses += 1
            return None
        entry.timestamp = time.monotonic()
        self._hits += 1
        return self._deserialize(entry.state_bytes)

    def delete(self, session_id: str) -> bool:
        """Remove a session from cache. Returns True if it existed."""
        return self._store.pop(session_id, None) is not None

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "sessions": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "total_bytes": sum(e.size_bytes for e in self._store.values()),
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Serialisation helpers (NumPy .npz in-memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(arrays: Dict[str, np.ndarray]) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    @staticmethod
    def _deserialize(payload: bytes) -> Dict[str, np.ndarray]:
        buf = io.BytesIO(payload)
        npz = np.load(buf)
        return {k: npz[k] for k in npz.files}

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        if not self._store:
            return
        lru_key = min(self._store, key=lambda sid: self._store[sid].timestamp)
        del self._store[lru_key]

    # ------------------------------------------------------------------
    # Container helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._store
