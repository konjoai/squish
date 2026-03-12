"""squish/multimodal_batch.py

MultiModalBatch — Heterogeneous batch scheduler for mixed text-only and
text+vision inference requests.

Serving a mixture of text-only and vision-language requests in the same batch
introduces severe padding waste: vision requests carry large image token
sequences that must be zero-padded into text-only slots (or vice-versa),
inflating memory bandwidth and compute costs.  The problem is especially acute
on hardware where attention kernels operate on fixed-shape tensors, as a single
vision request in an otherwise text-only batch forces every slot to allocate
the full ``max_vision_tokens`` worth of KV capacity.

MultiModalBatch separates incoming requests into two independent queues — one
for text-only requests and one for vision (text+image) requests.  When the
scheduler is asked for the next batch it selects the larger queue as the
preferred source and returns up to ``max_batch_size`` slots drawn exclusively
from that queue.  Homogeneous batches (all text or all vision) eliminate
cross-modality padding, maximising hardware utilisation.  The scheduler never
mixes modalities within a single :meth:`~MultiModalBatcher.next_batch` call,
so downstream kernels can safely specialise on a single token layout.

Requests are represented as :class:`BatchSlot` dataclass instances that carry
the request identifier, declared modality, and pre-validated tensor length
metadata.  Validation at enqueue time ensures that requests violating the
configured length limits are rejected immediately, preventing silent truncation
later in the pipeline.

Example usage::

    from squish.multimodal_batch import BatchConfig, MultiModalBatcher

    cfg     = BatchConfig(max_batch_size=8, max_text_len=2048,
                          max_vision_tokens=256)
    batcher = MultiModalBatcher(cfg)

    batcher.add_request(req_id=0, modality="text",   text_len=512)
    batcher.add_request(req_id=1, modality="vision", text_len=128, vision_tokens=196)
    batcher.add_request(req_id=2, modality="vision", text_len=64,  vision_tokens=196)

    batch = batcher.next_batch()   # returns the 2 vision slots (larger queue)
    print([s.req_id for s in batch])
    print(batcher.stats)
"""

from __future__ import annotations

__all__ = ["BatchConfig", "BatchSlot", "MultiModalBatcher", "BatchStats"]

from collections import deque
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Configuration for the heterogeneous batch scheduler.

    Attributes:
        max_batch_size:    Maximum number of request slots per batch.
        max_text_len:      Maximum allowed ``text_len`` for any request.
        max_vision_tokens: Maximum allowed ``vision_tokens`` for any vision
                           request.
    """

    max_batch_size: int = 8
    max_text_len: int = 2048
    max_vision_tokens: int = 256

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        if self.max_text_len < 1:
            raise ValueError(
                f"max_text_len must be >= 1, got {self.max_text_len}"
            )
        if self.max_vision_tokens < 1:
            raise ValueError(
                f"max_vision_tokens must be >= 1, got {self.max_vision_tokens}"
            )


# ---------------------------------------------------------------------------
# Batch slot
# ---------------------------------------------------------------------------


@dataclass
class BatchSlot:
    """A single request slot awaiting dispatch.

    Attributes:
        req_id:         Caller-assigned request identifier.
        modality:       ``"text"`` or ``"vision"``.
        text_len:       Number of text tokens in the request.
        vision_tokens:  Number of vision patch tokens.  Always ``0`` for
                        text-only requests.
    """

    req_id: int
    modality: str
    text_len: int
    vision_tokens: int = 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BatchStats:
    """Aggregate scheduling statistics for a :class:`MultiModalBatcher`.

    Attributes:
        total_batches:      Total number of :meth:`~MultiModalBatcher.next_batch`
                            calls that returned at least one slot.
        total_text_slots:   Cumulative text-only slots dispatched.
        total_vision_slots: Cumulative vision slots dispatched.
    """

    total_batches: int = 0
    total_text_slots: int = 0
    total_vision_slots: int = 0


# ---------------------------------------------------------------------------
# Batcher
# ---------------------------------------------------------------------------

_VALID_MODALITIES: frozenset[str] = frozenset({"text", "vision"})


class MultiModalBatcher:
    """Heterogeneous batch scheduler with modality-preference queue selection.

    Maintains separate FIFO queues for text-only and vision requests.  On each
    :meth:`next_batch` call the larger queue is drained first, producing
    homogeneous batches that minimise padding waste.

    Args:
        config: A :class:`BatchConfig` instance.
    """

    def __init__(self, config: BatchConfig) -> None:
        self._cfg = config
        self._text_queue:   deque[BatchSlot] = deque()
        self._vision_queue: deque[BatchSlot] = deque()
        self._stats = BatchStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        req_id: int,
        modality: str,
        text_len: int,
        vision_tokens: int = 0,
    ) -> None:
        """Enqueue a new inference request.

        Args:
            req_id:        Caller-assigned integer request identifier.
            modality:      ``"text"`` or ``"vision"``.
            text_len:      Number of text tokens.  Must be ``<= max_text_len``.
            vision_tokens: Number of vision patch tokens.  Must be
                           ``<= max_vision_tokens`` when *modality* is
                           ``"vision"``.

        Raises:
            ValueError: If *modality* is invalid, *text_len* exceeds the
                        configured limit, or *vision_tokens* exceeds the
                        configured limit.
        """
        if modality not in _VALID_MODALITIES:
            raise ValueError(
                f"modality must be 'text' or 'vision', got {modality!r}."
            )
        if text_len > self._cfg.max_text_len:
            raise ValueError(
                f"text_len {text_len} exceeds max_text_len "
                f"{self._cfg.max_text_len}."
            )
        if vision_tokens > self._cfg.max_vision_tokens:
            raise ValueError(
                f"vision_tokens {vision_tokens} exceeds max_vision_tokens "
                f"{self._cfg.max_vision_tokens}."
            )

        slot = BatchSlot(
            req_id=req_id,
            modality=modality,
            text_len=text_len,
            vision_tokens=vision_tokens,
        )
        if modality == "text":
            self._text_queue.append(slot)
        else:
            self._vision_queue.append(slot)

    def next_batch(self) -> list[BatchSlot]:
        """Return the next homogeneous batch of up to ``max_batch_size`` slots.

        Prefers whichever modality queue has more pending requests.  If both
        queues are equally long the vision queue is preferred.  Returns an
        empty list when both queues are empty.

        Returns:
            A list of :class:`BatchSlot` instances, all of the same modality.
        """
        if not self._text_queue and not self._vision_queue:
            return []

        # Prefer vision when vision queue >= text queue (or text is empty).
        if len(self._vision_queue) >= len(self._text_queue):
            source = self._vision_queue
        else:
            source = self._text_queue

        batch: list[BatchSlot] = []
        while source and len(batch) < self._cfg.max_batch_size:
            batch.append(source.popleft())

        if batch:
            self._stats.total_batches += 1
            for slot in batch:
                if slot.modality == "text":
                    self._stats.total_text_slots += 1
                else:
                    self._stats.total_vision_slots += 1

        return batch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_text(self) -> int:
        """Number of text-only requests waiting in the queue."""
        return len(self._text_queue)

    @property
    def pending_vision(self) -> int:
        """Number of vision requests waiting in the queue."""
        return len(self._vision_queue)

    @property
    def stats(self) -> BatchStats:
        """Return a snapshot of cumulative scheduling statistics."""
        return BatchStats(
            total_batches=self._stats.total_batches,
            total_text_slots=self._stats.total_text_slots,
            total_vision_slots=self._stats.total_vision_slots,
        )
