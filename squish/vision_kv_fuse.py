"""squish/vision_kv_fuse.py

VisionKVFuse — Fused KV cache with separate modality sub-caches and
modality-aware eviction for vision-language model inference.

Vision-language models (VLMs) such as LLaVA and Flamingo process two distinct
token streams: a text stream (queries, answers, system prompts) and a vision
stream (image patch embeddings produced by a ViT encoder).  At long context
lengths the KV cache becomes the dominant memory consumer.  A naïve unified
cache treats text and vision tokens identically, which wastes capacity on
low-utility vision keys and values that are rarely revisited after the initial
visual-encoding phase.

VisionKVFuse partitions the KV cache into two fixed-capacity sub-buffers — one
for text tokens and one for vision tokens.  Each buffer is backed by a
pre-allocated NumPy array of shape ``(n_heads, capacity, head_dim)`` to avoid
heap fragmentation and enable O(1) slot appends.  When a sub-buffer reaches
capacity an ``OverflowError`` is raised, giving the caller an explicit signal
to trigger modality-level eviction before re-appending.  Because vision tokens
are typically only relevant during the initial cross-attention phase, callers
may configure a smaller ``vision_capacity`` relative to ``text_capacity``.

The :meth:`VisionKVFuseCache.reset` method supports per-modality invalidation,
so a fresh image can be injected mid-conversation without flushing accumulated
text context.  This enables efficient multi-turn VLM inference where the text
history is preserved across image turns while each new image gets a clean
vision sub-cache.

Example usage::

    import numpy as np
    from squish.vision_kv_fuse import ModalityConfig, VisionKVFuseCache

    cfg   = ModalityConfig(text_capacity=512, vision_capacity=256,
                           n_heads=8, head_dim=64)
    cache = VisionKVFuseCache(cfg)

    key = np.random.randn(8, 64).astype(np.float32)
    val = np.random.randn(8, 64).astype(np.float32)
    cache.append("text",   key, val)
    cache.append("vision", key, val)

    tk, tv = cache.get_kv("text")    # shape (8, 1, 64)
    vk, vv = cache.get_kv("vision")  # shape (8, 1, 64)
    print(f"memory_ratio={cache.memory_ratio:.3f}")
    print(cache.stats)
"""

from __future__ import annotations

__all__ = ["ModalityConfig", "VisionKVFuseCache", "VisionKVFuseStats"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModalityConfig:
    """Configuration for the fused modality KV cache.

    Attributes:
        text_capacity:   Maximum number of text-token KV slots.
        vision_capacity: Maximum number of vision-token KV slots.
        n_heads:         Number of attention heads.
        head_dim:        Dimension of each attention head.
    """

    text_capacity: int = 512
    vision_capacity: int = 256
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.text_capacity < 1:
            raise ValueError(
                f"text_capacity must be >= 1, got {self.text_capacity}"
            )
        if self.vision_capacity < 1:
            raise ValueError(
                f"vision_capacity must be >= 1, got {self.vision_capacity}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class VisionKVFuseStats:
    """Aggregate statistics for a :class:`VisionKVFuseCache`.

    Attributes:
        text_appends:   Total successful text-token appends.
        vision_appends: Total successful vision-token appends.
        total_resets:   Total :meth:`~VisionKVFuseCache.reset` calls.
    """

    text_appends: int = 0
    vision_appends: int = 0
    total_resets: int = 0


# ---------------------------------------------------------------------------
# Internal sentinel for valid modalities
# ---------------------------------------------------------------------------

_VALID_MODALITIES: frozenset[str] = frozenset({"text", "vision"})


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class VisionKVFuseCache:
    """Fused KV cache with independent pre-allocated sub-buffers per modality.

    Both sub-buffers are allocated at construction time as contiguous NumPy
    arrays of shape ``(n_heads, capacity, head_dim)``.  Tokens are appended
    sequentially; no in-place eviction is performed — the caller is responsible
    for calling :meth:`reset` and re-populating after eviction.

    Args:
        config: A :class:`ModalityConfig` instance controlling capacities and
                tensor dimensions.
    """

    def __init__(self, config: ModalityConfig) -> None:
        self._cfg = config
        h, dh = config.n_heads, config.head_dim

        self._text_keys   = np.zeros((h, config.text_capacity,   dh), dtype=np.float32)
        self._text_vals   = np.zeros((h, config.text_capacity,   dh), dtype=np.float32)
        self._vision_keys = np.zeros((h, config.vision_capacity, dh), dtype=np.float32)
        self._vision_vals = np.zeros((h, config.vision_capacity, dh), dtype=np.float32)

        self._text_fill:   int = 0
        self._vision_fill: int = 0

        self._text_appends:   int = 0
        self._vision_appends: int = 0
        self._total_resets:   int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        modality: str,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Append a single-token KV pair to the specified modality sub-cache.

        Args:
            modality: ``"text"`` or ``"vision"``.
            key:      Array of shape ``(n_heads, head_dim)`` (float32).
            value:    Array of shape ``(n_heads, head_dim)`` (float32).

        Raises:
            ValueError:    If *modality* is not ``"text"`` or ``"vision"``, or
                           if *key* / *value* have unexpected shapes.
            OverflowError: If the sub-cache for *modality* is already full.
        """
        self._validate_modality(modality)
        self._validate_kv_shape(key, "key")
        self._validate_kv_shape(value, "value")

        if modality == "text":
            if self._text_fill >= self._cfg.text_capacity:
                raise OverflowError(
                    f"Text KV sub-cache is full "
                    f"(capacity={self._cfg.text_capacity})."
                )
            self._text_keys[:, self._text_fill, :] = key
            self._text_vals[:, self._text_fill, :] = value
            self._text_fill   += 1
            self._text_appends += 1
        else:
            if self._vision_fill >= self._cfg.vision_capacity:
                raise OverflowError(
                    f"Vision KV sub-cache is full "
                    f"(capacity={self._cfg.vision_capacity})."
                )
            self._vision_keys[:, self._vision_fill, :] = key
            self._vision_vals[:, self._vision_fill, :] = value
            self._vision_fill   += 1
            self._vision_appends += 1

    def get_kv(self, modality: str) -> tuple[np.ndarray, np.ndarray]:
        """Return all stored KV pairs for *modality*.

        Args:
            modality: ``"text"`` or ``"vision"``.

        Returns:
            A ``(keys, values)`` tuple, each of shape
            ``(n_heads, n_tokens, head_dim)`` where *n_tokens* equals the
            current fill level for the requested modality.

        Raises:
            ValueError: If *modality* is not a valid value.
        """
        self._validate_modality(modality)
        if modality == "text":
            return (
                self._text_keys[:, : self._text_fill, :].copy(),
                self._text_vals[:, : self._text_fill, :].copy(),
            )
        return (
            self._vision_keys[:, : self._vision_fill, :].copy(),
            self._vision_vals[:, : self._vision_fill, :].copy(),
        )

    def reset(self, modality: Optional[str] = None) -> None:
        """Reset one or both modality sub-caches.

        Resetting only zeroes the fill counter; the underlying buffer memory
        is not cleared, so it will be overwritten on the next :meth:`append`.

        Args:
            modality: ``"text"`` to reset only text, ``"vision"`` to reset
                      only vision, or ``None`` (default) to reset both.

        Raises:
            ValueError: If *modality* is not ``"text"``, ``"vision"``, or
                        ``None``.
        """
        if modality is not None:
            self._validate_modality(modality)

        if modality is None or modality == "text":
            self._text_fill = 0
        if modality is None or modality == "vision":
            self._vision_fill = 0

        self._total_resets += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def text_len(self) -> int:
        """Current number of stored text-token KV slots."""
        return self._text_fill

    @property
    def vision_len(self) -> int:
        """Current number of stored vision-token KV slots."""
        return self._vision_fill

    @property
    def memory_ratio(self) -> float:
        """Ratio of vision fill to text fill at the time of access.

        Returns ``0.0`` when no text tokens are present to avoid division by
        zero.  A value greater than 1.0 indicates more vision tokens than text
        tokens are currently cached.
        """
        if self._text_fill == 0:
            return 0.0
        return self._vision_fill / self._text_fill

    @property
    def stats(self) -> VisionKVFuseStats:
        """Return a snapshot of accumulated append and reset counters."""
        return VisionKVFuseStats(
            text_appends=self._text_appends,
            vision_appends=self._vision_appends,
            total_resets=self._total_resets,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_modality(self, modality: str) -> None:
        if modality not in _VALID_MODALITIES:
            raise ValueError(
                f"modality must be 'text' or 'vision', got {modality!r}."
            )

    def _validate_kv_shape(self, arr: np.ndarray, name: str) -> None:
        expected = (self._cfg.n_heads, self._cfg.head_dim)
        if arr.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {arr.shape}."
            )
