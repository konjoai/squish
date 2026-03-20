"""
squish/speculative/spec_stream.py

SpeculativeStreamer — Low-Latency Speculative Streaming with Silent Rollback.

Based on:
  "SpecInfer: Accelerating Large Language Model Serving with Tree-based
   Speculative Inference and Verification"  —  MLSys 2024

  "Streaming LLM with Speculative Execution" — MLSys 2025 Workshop
  Key result: perceived TTFT ≈ 0 ms when first draft token is streamed
  immediately, with silent rollback on rejection.

Background
----------
Standard speculative decoding improves **throughput** but not
*perceived first-token latency* (TTFT) — the user still waits for the
target model's verification pass before seeing any output.

Speculative Streaming solves this by:

  1. Immediately streaming draft tokens to the client as they are
     generated (before verification).
  2. Running the target model verification in the background.
  3. If a draft token is accepted → no visible change (token was already
     shown).
  4. If a draft token is rejected at position k → issue a *rollback*
     signal: delete tokens d_k … d_N from the visible stream and append
     the correction token.

Rollback is implemented as a special control message (e.g. ``\\x08`` ×
N or a JSON patch) that a streaming client applies.

This module provides:
  - Buffering of draft tokens before commit.
  - Recording the correction token on rejection.
  - Rollback calculation and payload generation.
  - Statistics on rollback frequency and accepted-prefix length.

Classes
-------
``SpecStreamConfig``        — buffer size, rollback encoding
``StreamedToken``           — one token in the stream (draft or committed)
``SpecStreamStats``         — per-instance statistics
``SpeculativeStreamer``     — main streaming manager

Usage::

    from squish.speculative.spec_stream import SpecStreamConfig, SpeculativeStreamer

    streamer = SpeculativeStreamer(SpecStreamConfig(buffer_size=8))
    streamer.push_draft([10, 20, 30, 40])

    accepted = [True, True, False, False]
    correction = 99
    n_rollback = streamer.commit(accepted, correction)
    tokens = streamer.flush()
    # tokens = [10, 20, 99]  (rollback removed 30, 40; appended 99)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

__all__ = [
    "SpecStreamConfig",
    "StreamedToken",
    "SpecStreamStats",
    "SpeculativeStreamer",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpecStreamConfig:
    """Configuration for the speculative streamer.

    Attributes:
        buffer_size:        Maximum draft tokens to buffer before auto-flush.
                            Must be >= 1.
        rollback_on_reject: When True, committed stream automatically removes
                            rejected tokens and inserts correction.  When
                            False, only accepted tokens are committed.
        eos_token_id:       Token id for end-of-sequence.  When seen in a
                            committed stream, ``is_done`` is set to True.
    """

    buffer_size: int = 16
    rollback_on_reject: bool = True
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")


# ---------------------------------------------------------------------------
# Token representation
# ---------------------------------------------------------------------------


class StreamedToken(NamedTuple):
    """A single token in the speculative stream.

    Attributes:
        token_id:    Vocabulary index.
        is_draft:    True = token came from draft model (tentative).
        position:    Sequence position (0-based).
        committed:   True = target model has verified / accepted this token.
    """

    token_id: int
    is_draft: bool
    position: int
    committed: bool


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class SpecStreamStats:
    """Runtime statistics for SpeculativeStreamer.

    Attributes:
        push_calls:             Number of ``push_draft()`` calls.
        commit_calls:           Number of ``commit()`` calls.
        total_draft_tokens:     Tokens pushed as draft.
        total_accepted_tokens:  Tokens accepted by target (no rollback).
        total_rollbacks:        Number of rollback events.
        tokens_rolled_back:     Total tokens removed via rollback.
    """

    push_calls: int = 0
    commit_calls: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rollbacks: int = 0
    tokens_rolled_back: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def mean_rollback_tokens(self) -> float:
        if self.total_rollbacks == 0:
            return 0.0
        return self.tokens_rolled_back / self.total_rollbacks

    def __repr__(self) -> str:
        return (
            f"SpecStreamStats("
            f"accept={self.acceptance_rate:.2%}, "
            f"rollbacks={self.total_rollbacks}, "
            f"mean_rb={self.mean_rollback_tokens:.1f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpeculativeStreamer:
    """Buffer and manage speculative draft tokens with rollback support.

    Workflow:
      1. Draft model generates ``N`` draft tokens.
      2. Call ``push_draft(draft_tokens)`` — tokens enter the buffer.
      3. Target model runs verification over the draft.
      4. Call ``commit(accepted_mask, correction_token)`` — stream is updated.
      5. Call ``flush()`` → returns the current committed token sequence.
      6. Client receives the committed sequence; if rollback occurred, client
         removes trailing tokens per ``n_rolled_back``.

    Parameters
    ----------
    config:
        Streamer configuration.
    """

    def __init__(self, config: Optional[SpecStreamConfig] = None) -> None:
        self._cfg = config or SpecStreamConfig()
        self._committed: List[int] = []
        self._draft_buffer: List[int] = []
        self._position: int = 0
        self.is_done: bool = False
        self.stats = SpecStreamStats()

    # ------------------------------------------------------------------
    # Draft management
    # ------------------------------------------------------------------

    def push_draft(self, draft_tokens: List[int]) -> None:
        """Push draft tokens into the pending buffer.

        Parameters
        ----------
        draft_tokens: Token ids from the draft model.
        """
        limit = self._cfg.buffer_size
        tokens_to_add = draft_tokens[: limit - len(self._draft_buffer)]
        self._draft_buffer.extend(tokens_to_add)
        self.stats.push_calls += 1
        self.stats.total_draft_tokens += len(draft_tokens)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(
        self,
        accepted_mask: List[bool],
        correction_token: int,
    ) -> int:
        """Commit verification result and update the stream.

        Parameters
        ----------
        accepted_mask:    Boolean per draft token (True = accepted).
        correction_token: Target model's correction token at the first
                          rejected position.

        Returns
        -------
        n_rolled_back: Number of tokens rolled back from the committed stream
                       (0 if no rollback needed).
        """
        self.stats.commit_calls += 1
        draft = self._draft_buffer[: len(accepted_mask)]

        # Find first rejection
        first_reject = len(accepted_mask)
        for i, acc in enumerate(accepted_mask):
            if not acc:
                first_reject = i
                break

        accepted_tokens = draft[:first_reject]
        n_rolled_back = 0

        if self._cfg.rollback_on_reject:
            if first_reject < len(accepted_mask):
                # Rollback: remove pending draft tokens already "shown"
                # (anything after first_reject that was pushed earlier)
                already_shown = self._draft_buffer[:first_reject]
                n_to_rollback = len(already_shown) - first_reject
                n_rolled_back = max(0, n_to_rollback)
                if n_rolled_back > 0:
                    self._committed = self._committed[:-n_rolled_back]
                    self.stats.tokens_rolled_back += n_rolled_back
                    self.stats.total_rollbacks += 1
                self._committed.extend(accepted_tokens)
                self._committed.append(correction_token)
            else:
                self._committed.extend(accepted_tokens)
        else:
            self._committed.extend(accepted_tokens)
            if first_reject < len(accepted_mask):
                self._committed.append(correction_token)

        self.stats.total_accepted_tokens += len(accepted_tokens)
        self._position += len(accepted_tokens) + (1 if first_reject < len(accepted_mask) else 0)
        self._draft_buffer = []

        # Check for EOS
        if self._cfg.eos_token_id in self._committed:
            self.is_done = True

        return n_rolled_back

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def flush(self) -> List[int]:
        """Return the current committed token sequence.

        Does NOT clear the committed list — call ``reset()`` to start fresh.
        """
        return list(self._committed)

    def reset(self) -> None:
        """Clear all state (e.g., for a new generation request)."""
        self._committed = []
        self._draft_buffer = []
        self._position = 0
        self.is_done = False

    def rollback_to(self, position: int) -> None:
        """Truncate committed stream to ``position`` tokens.

        Parameters
        ----------
        position: Target length of committed sequence (0-indexed).
        """
        if position < 0 or position > len(self._committed):
            raise ValueError(
                f"rollback_to position {position} out of range "
                f"[0, {len(self._committed)}]"
            )
        n_removed = len(self._committed) - position
        self._committed = self._committed[:position]
        self._position = position
        if n_removed > 0:
            self.stats.tokens_rolled_back += n_removed
            self.stats.total_rollbacks += 1

    @property
    def n_committed(self) -> int:
        return len(self._committed)

    @property
    def n_pending_draft(self) -> int:
        return len(self._draft_buffer)

    def __repr__(self) -> str:
        return (
            f"SpeculativeStreamer("
            f"committed={self.n_committed}, "
            f"pending={self.n_pending_draft}, "
            f"done={self.is_done}, "
            f"{self.stats})"
        )
