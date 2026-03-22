"""SSMStateOffload: disk/memory offload of recurrent states for long contexts.

Recurrent architectures like Mamba2 maintain a fixed-size state vector whose
memory cost does not grow with sequence length — but for very long sessions
(e.g., millions of tokens) the per-layer state must be checkpointed and
restored in chunks so that a single GPU/CPU device never needs to hold all
states at once.

SSMStateOffload checkpoints a complete dict of layer states every
``segment_len`` tokens, serialises them (optionally down-casting to float16
to halve bandwidth), and stores the bytes in an ``OffloadSegment``.  On
restore the most-recent segment is deserialised and the states handed back
to the calling loop.

This module is intentionally storage-backend agnostic: it keeps segments
in an in-process dict.  Production code can subclass and override
``_write_segment`` / ``_read_segment`` to use mmap, object storage, etc.

Reference: Waleffe et al., "An Empirical Study of Mamba-based Language
Models", arXiv 2406.07887 (2024).
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "SSMStateOffloadConfig",
    "OffloadSegment",
    "SSMStateOffload",
]


@dataclass
class SSMStateOffloadConfig:
    """Configuration for SSMStateOffload.

    Attributes:
        segment_len: Number of tokens between state checkpoints.
        compress_fp16: Down-cast float32 arrays to float16 before serialising
            (halves storage at the cost of minor precision).
        max_segments_per_session: Hard cap on stored segments per session.
            Oldest segments are evicted once the cap is reached.
        seed: Unused; for API consistency.
    """

    segment_len: int = 2048
    compress_fp16: bool = True
    max_segments_per_session: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.segment_len < 1:
            raise ValueError("segment_len must be >= 1")
        if self.max_segments_per_session < 1:
            raise ValueError("max_segments_per_session must be >= 1")


@dataclass
class OffloadSegment:
    """A serialised snapshot of all layer states at a given token boundary.

    Attributes:
        session_id: Unique identifier for the generating session.
        segment_idx: Zero-based index of this segment within the session.
        state_bytes: Serialised state payload (numpy .npz format).
        n_tokens: Total tokens processed when this snapshot was taken.
        timestamp: Wall-clock time at snapshot creation (``time.time()``).
    """

    session_id: str
    segment_idx: int
    state_bytes: bytes
    n_tokens: int
    timestamp: float

    @property
    def size_bytes(self) -> int:
        return len(self.state_bytes)


class SSMStateOffload:
    """Checkpoint and restore per-layer SSM states across token segments.

    Args:
        config: SSMStateOffloadConfig controlling segment length and format.
    """

    def __init__(self, config: SSMStateOffloadConfig) -> None:
        self.config = config
        # {session_id: [OffloadSegment, ...]} ordered oldest→newest
        self._store: Dict[str, List[OffloadSegment]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """Create a new offload session.

        Returns:
            A unique session ID string (UUID4 hex).
        """
        sid = uuid.uuid4().hex
        self._store[sid] = []
        return sid

    def delete_session(self, session_id: str) -> None:
        """Remove all stored segments for a session.

        Args:
            session_id: Session to delete.

        Raises:
            KeyError: If session_id does not exist.
        """
        if session_id not in self._store:
            raise KeyError(f"Unknown session '{session_id}'")
        del self._store[session_id]

    # ------------------------------------------------------------------
    # Offload / restore
    # ------------------------------------------------------------------

    def maybe_offload(
        self,
        session_id: str,
        layer_states: Dict[str, np.ndarray],
        n_tokens_total: int,
    ) -> bool:
        """Snapshot states if a segment boundary has been crossed.

        A snapshot is triggered when ``n_tokens_total`` is a positive
        multiple of ``config.segment_len``.

        Args:
            session_id: Session this state belongs to.
            layer_states: Dict mapping layer name / index to state array.
            n_tokens_total: Cumulative token count after the current step.

        Returns:
            ``True`` if a checkpoint was written, ``False`` otherwise.
        """
        if session_id not in self._store:
            raise KeyError(f"Unknown session '{session_id}'")
        if n_tokens_total <= 0:
            return False
        if n_tokens_total % self.config.segment_len != 0:
            return False

        seg_idx = n_tokens_total // self.config.segment_len - 1
        payload = self._compress(layer_states)
        segment = OffloadSegment(
            session_id=session_id,
            segment_idx=seg_idx,
            state_bytes=payload,
            n_tokens=n_tokens_total,
            timestamp=time.time(),
        )
        self._write_segment(session_id, segment)
        return True

    def restore(
        self,
        session_id: str,
        segment_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Restore layer states from a stored segment.

        Args:
            session_id: Session to restore from.
            segment_idx: Which segment to restore; defaults to the most
                recent if ``None``.

        Returns:
            Dict mapping layer name to state array.

        Raises:
            KeyError: If session does not exist or has no segments.
            IndexError: If the requested segment_idx does not exist.
        """
        if session_id not in self._store:
            raise KeyError(f"Unknown session '{session_id}'")
        segments = self._store[session_id]
        if not segments:
            raise KeyError(f"Session '{session_id}' has no stored segments")

        if segment_idx is None:
            seg = segments[-1]
        else:
            matches = [s for s in segments if s.segment_idx == segment_idx]
            if not matches:
                raise IndexError(
                    f"Segment {segment_idx} not found for session '{session_id}'"
                )
            seg = matches[0]

        return self._decompress(seg.state_bytes)

    def latest_segment(self, session_id: str) -> Optional[OffloadSegment]:
        """Return the most recent OffloadSegment for a session, or None."""
        if session_id not in self._store:
            raise KeyError(f"Unknown session '{session_id}'")
        segs = self._store[session_id]
        return segs[-1] if segs else None

    def segments_for_session(self, session_id: str) -> int:
        """Return the number of stored segments for a session."""
        if session_id not in self._store:
            raise KeyError(f"Unknown session '{session_id}'")
        return len(self._store[session_id])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        n_sessions = len(self._store)
        total_segments = sum(len(v) for v in self._store.values())
        total_bytes = sum(
            seg.size_bytes for segs in self._store.values() for seg in segs
        )
        return {
            "n_sessions": n_sessions,
            "total_segments": total_segments,
            "total_bytes_stored": total_bytes,
            "compress_fp16": self.config.compress_fp16,
            "segment_len": self.config.segment_len,
        }

    # ------------------------------------------------------------------
    # Internal helpers (overridable for custom storage backends)
    # ------------------------------------------------------------------

    def _write_segment(self, session_id: str, segment: OffloadSegment) -> None:
        """Store segment in the in-memory dict, evicting oldest if needed."""
        segs = self._store[session_id]
        segs.append(segment)
        # Evict oldest if over cap
        cap = self.config.max_segments_per_session
        if len(segs) > cap:
            self._store[session_id] = segs[-cap:]

    def _compress(self, arrays: Dict[str, np.ndarray]) -> bytes:
        """Serialise arrays to .npz bytes, optionally casting to float16."""
        to_save: Dict[str, np.ndarray] = {}
        for k, v in arrays.items():
            if self.config.compress_fp16 and v.dtype == np.float32:
                to_save[k] = v.astype(np.float16)
            else:
                to_save[k] = v
        buf = io.BytesIO()
        np.savez_compressed(buf, **to_save)
        return buf.getvalue()

    def _decompress(self, data: bytes) -> Dict[str, np.ndarray]:
        """Deserialise .npz bytes back to float32 arrays."""
        buf = io.BytesIO(data)
        npz = np.load(buf, allow_pickle=False)
        return {k: npz[k].astype(np.float32) for k in npz.files}
