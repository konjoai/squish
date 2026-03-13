# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/streaming_chunk.py

StreamingChunk — Sub-token-latency chunked streaming with backpressure.

Rather than sending tokens one at a time (high per-token overhead) or
buffering all output until generation completes (high time-to-first-byte),
chunk streaming sends groups of tokens as soon as a full chunk is ready.
When the downstream consumer is slow, the producer receives a backpressure
signal so it can throttle generation rather than unboundedly buffer output.

:class:`BackpressureBuffer` handles the single-producer / single-consumer
token queue.  :class:`ChunkedStreamer` is a stateless utility for
splitting a pre-collected token list into chunk-sized slices.

Example usage::

    from squish.streaming_chunk import ChunkedStreamer, ChunkConfig

    config = ChunkConfig(chunk_size=4, max_buffer=64)
    streamer = ChunkedStreamer(config)

    token_ids = list(range(20))
    chunks = streamer.stream(token_ids)
    print(chunks)   # [[0,1,2,3], [4,5,6,7], ..., [16,17,18,19]]
    print(streamer.stats)

    # Backpressure example:
    from squish.streaming_chunk import BackpressureBuffer
    buf = BackpressureBuffer(config)
    for t in range(65):          # 65 > max_buffer=64
        ok = buf.push(t)
        if not ok:
            print(f"backpressure at token {t}")
            break
"""

from __future__ import annotations

__all__ = [
    "ChunkConfig",
    "BackpressureBuffer",
    "ChunkedStreamer",
    "StreamStats",
]

from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """Configuration for chunked token streaming.

    Attributes:
        chunk_size:  Number of tokens to accumulate before flushing a chunk.
        max_buffer:  Maximum tokens buffered before backpressure is signalled.
    """

    chunk_size: int = 4
    max_buffer: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(
                f"chunk_size must be >= 1, got {self.chunk_size}"
            )
        if self.max_buffer < self.chunk_size:
            raise ValueError(
                f"max_buffer ({self.max_buffer}) must be >= "
                f"chunk_size ({self.chunk_size})"
            )


# ---------------------------------------------------------------------------
# Backpressure buffer
# ---------------------------------------------------------------------------

class BackpressureBuffer:
    """Single-producer / single-consumer token buffer with backpressure.

    Tokens are pushed by the producer (inference engine) and flushed by the
    consumer (network sender).  When the buffer reaches ``max_buffer`` tokens,
    :meth:`push` returns ``False`` to signal that the producer should pause.

    Args:
        config: A :class:`ChunkConfig` instance controlling sizes.
    """

    def __init__(self, config: ChunkConfig) -> None:
        self._config = config
        self._buffer: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, token_id: int) -> bool:
        """Attempt to add *token_id* to the buffer.

        Args:
            token_id: Integer token identifier to enqueue.

        Returns:
            ``True`` if the token was accepted.
            ``False`` if the buffer was already full (backpressure); the
            token is *not* added in this case.
        """
        if len(self._buffer) >= self._config.max_buffer:
            return False
        self._buffer.append(token_id)
        return True

    def flush(self) -> list[int]:
        """Return and consume tokens from the buffer.

        If the buffer holds at least ``chunk_size`` tokens, the first
        ``chunk_size`` tokens are returned and removed.  If fewer tokens
        are buffered (or :meth:`flush` is called explicitly to drain), all
        buffered tokens are returned and the buffer is cleared.

        Returns:
            A list of token IDs (possibly empty if the buffer is empty).
        """
        if not self._buffer:
            return []
        n = min(len(self._buffer), self._config.chunk_size)
        chunk = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return chunk

    def peek_size(self) -> int:
        """Return the number of tokens currently held in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Discard all buffered tokens without returning them."""
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class StreamStats:
    """Cumulative streaming statistics collected by :class:`ChunkedStreamer`.

    Attributes:
        total_tokens_streamed: Total token IDs processed across all
                               :meth:`ChunkedStreamer.stream` calls.
        total_chunks:          Total chunks emitted.
        backpressure_events:   Number of :meth:`BackpressureBuffer.push`
                               calls that returned ``False``.
    """

    total_tokens_streamed: int = 0
    total_chunks: int = 0
    backpressure_events: int = 0

    @property
    def avg_chunk_size(self) -> float:
        """Average tokens per emitted chunk.

        Returns 0.0 when no chunks have been produced yet.
        """
        if self.total_chunks == 0:
            return 0.0
        return self.total_tokens_streamed / self.total_chunks


# ---------------------------------------------------------------------------
# Chunked streamer
# ---------------------------------------------------------------------------

class ChunkedStreamer:
    """Stateless token-list splitter with cumulative statistics.

    Splits a pre-collected list of token IDs into fixed-size chunks.  The
    final chunk may be smaller than ``chunk_size`` if the total token count
    is not an exact multiple.

    Args:
        config: A :class:`ChunkConfig` instance controlling chunk size.
    """

    def __init__(self, config: ChunkConfig) -> None:
        self._config = config
        self._stats = StreamStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self, token_ids: List[int]) -> List[List[int]]:
        """Split *token_ids* into sequential chunks.

        Args:
            token_ids: Flat list of integer token IDs to chunk.  May be
                       empty, in which case an empty list is returned.

        Returns:
            A list of sub-lists, each of length at most ``chunk_size``.
            The last sub-list may be shorter than ``chunk_size``.

        Raises:
            TypeError: if any element of ``token_ids`` is not an integer.
        """
        for i, tok in enumerate(token_ids):
            if not isinstance(tok, int):
                raise TypeError(
                    f"token_ids[{i}] must be int, got {type(tok).__name__!r}"
                )

        chunks: list[list[int]] = []
        step = self._config.chunk_size
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start : start + step]
            if chunk:
                chunks.append(chunk)

        self._stats.total_tokens_streamed += len(token_ids)
        self._stats.total_chunks += len(chunks)
        return chunks

    @property
    def stats(self) -> StreamStats:
        """Cumulative streaming statistics (updated in place)."""
        return self._stats
