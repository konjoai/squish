"""ReasoningKVManager: differentiated KV-cache quantisation for thinking vs answer tokens.

Reasoning models (DeepSeek-R1, QwQ-32B, Qwen3-8B-thinking) produce many
more *thinking* tokens than answer tokens.  Because thinking tokens are
never decoded to the user they tolerate aggressive lossy compression.
This module provides an enum-driven KV manager that stores thinking-segment
KV entries at 2-bit precision and answer-segment entries at full fp16.

The quantisation stub uses symmetric quantisation with per-group scale
factors (group_size=32 by default) matching the technique in
``squish.kv.quantized_cache``.

Reference: Motivated by s1 / DeepSeek-R1 serving practices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ReasoningKVConfig",
    "ReasoningKVSegment",
    "ReasoningKVState",
    "ReasoningKVManager",
]


class ReasoningKVSegment(Enum):
    """Cache segment label."""

    THINKING = "thinking"
    ANSWER = "answer"


@dataclass
class ReasoningKVConfig:
    """Configuration for :class:`ReasoningKVManager`.

    Attributes:
        thinking_bits: Quantisation bits for thinking-segment vectors.
        answer_bits: Storage bits for answer-segment vectors (``16`` = fp16 stub).
        boundary_token: Token string that triggers segment transition from
            thinking to answer.
        group_size: Quantisation group size (must divide head_dim).
        seed: RNG seed.
    """

    thinking_bits: int = 2
    answer_bits: int = 16
    boundary_token: str = "</think>"
    group_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.thinking_bits not in {1, 2, 4, 8}:
            raise ValueError(
                f"thinking_bits must be one of {{1, 2, 4, 8}}, got {self.thinking_bits}"
            )
        if self.answer_bits not in {8, 16, 32}:
            raise ValueError(
                f"answer_bits must be one of {{8, 16, 32}}, got {self.answer_bits}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")


@dataclass
class ReasoningKVState:
    """Per-sequence KV-cache state managed by :class:`ReasoningKVManager`.

    Thinking-segment entries are stored as quantised ``(scale, codes)`` pairs;
    answer-segment entries are stored as fp32 NumPy arrays (fp16 stub).

    Attributes:
        segment: Current active segment.
        thinking_k: Quantised key entries (one tuple per token).
        thinking_v: Quantised value entries.
        answer_k: FP32 key vectors for answer segment.
        answer_v: FP32 value vectors for answer segment.
        boundary_position: Token index at which thinking → answer transition occurred.
    """

    segment: ReasoningKVSegment = ReasoningKVSegment.THINKING
    thinking_k: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    thinking_v: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    answer_k: List[np.ndarray] = field(default_factory=list)
    answer_v: List[np.ndarray] = field(default_factory=list)
    boundary_position: Optional[int] = None

    @property
    def n_thinking_tokens(self) -> int:
        return len(self.thinking_k)

    @property
    def n_answer_tokens(self) -> int:
        return len(self.answer_k)

    @property
    def total_tokens(self) -> int:
        return self.n_thinking_tokens + self.n_answer_tokens

    @property
    def compression_ratio(self) -> float:
        """Ratio of thinking tokens to total (higher = more compression achieved)."""
        total = self.total_tokens
        return self.n_thinking_tokens / total if total > 0 else 0.0


class ReasoningKVManager:
    """Store each KV entry with segment-appropriate precision.

    Usage::

        cfg = ReasoningKVConfig(thinking_bits=2, boundary_token="</think>")
        mgr = ReasoningKVManager(cfg)
        state = mgr.new_state()
        for token_str, k, v in zip(tokens, keys, values):
            mgr.update(k, v, token_str, state)
        k_full, v_full = mgr.get_kv(state)
    """

    def __init__(self, config: ReasoningKVConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> ReasoningKVState:
        """Return a fresh per-sequence cache state."""
        return ReasoningKVState()

    def update(
        self,
        k: np.ndarray,
        v: np.ndarray,
        token_str: str,
        state: ReasoningKVState,
    ) -> None:
        """Append one KV entry at current segment precision.

        Parameters
        ----------
        k, v:
            Key/value vectors of shape ``(head_dim,)`` or ``(n_heads, head_dim)``.
        token_str:
            Decoded token string.  If it matches ``config.boundary_token``
            the segment transitions from THINKING to ANSWER.
        state:
            Mutable cache state.
        """
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)

        if state.segment is ReasoningKVSegment.THINKING:
            state.thinking_k.append(self._quantize(k))
            state.thinking_v.append(self._quantize(v))
            if token_str == self.config.boundary_token:
                state.segment = ReasoningKVSegment.ANSWER
                state.boundary_position = state.total_tokens - 1
        else:
            state.answer_k.append(k.copy())
            state.answer_v.append(v.copy())

    def get_kv(
        self, state: ReasoningKVState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantise all entries and return concatenated KV arrays.

        Returns
        -------
        (k, v):
            Arrays of shape ``(total_tokens, head_dim)`` in fp32.
        """
        k_parts: List[np.ndarray] = [
            self._dequantize(scale, codes) for scale, codes in state.thinking_k
        ]
        v_parts: List[np.ndarray] = [
            self._dequantize(scale, codes) for scale, codes in state.thinking_v
        ]
        k_parts.extend(state.answer_k)
        v_parts.extend(state.answer_v)

        if not k_parts:
            empty = np.empty((0,), dtype=np.float32)
            return empty, empty

        k_out = np.stack(k_parts, axis=0)
        v_out = np.stack(v_parts, axis=0)
        return k_out, v_out

    def memory_summary(self, state: ReasoningKVState) -> Dict[str, object]:
        """Return a dict describing memory usage for this state."""
        bits_per_thinking = self.config.thinking_bits
        bits_per_answer = self.config.answer_bits
        thinking_bytes = state.n_thinking_tokens * bits_per_thinking / 8
        answer_bytes = state.n_answer_tokens * bits_per_answer / 8
        return {
            "segment": state.segment.value,
            "n_thinking_tokens": state.n_thinking_tokens,
            "n_answer_tokens": state.n_answer_tokens,
            "thinking_bytes_approx": thinking_bytes,
            "answer_bytes_approx": answer_bytes,
            "compression_ratio": state.compression_ratio,
            "boundary_position": state.boundary_position,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize(
        self, vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric group-wise quantisation to ``thinking_bits`` bits.

        Returns ``(scale, codes)`` where *scale* has shape ``(n_groups,)``
        and *codes* is int8 with the same total elements as *vec*.
        """
        flat = vec.ravel().astype(np.float32)
        gs = self.config.group_size
        bits = self.config.thinking_bits
        # Pad to multiple of group_size
        pad = (-len(flat)) % gs
        if pad:
            flat = np.pad(flat, (0, pad))
        groups = flat.reshape(-1, gs)
        max_abs = np.abs(groups).max(axis=1, keepdims=True).clip(min=1e-8)
        scale = max_abs.squeeze(1)
        n_levels = (1 << bits) - 1  # e.g. 3 for 2-bit
        codes = np.round(groups / max_abs * (n_levels / 2)).astype(np.int8)
        return scale, codes

    def _dequantize(
        self, scale: np.ndarray, codes: np.ndarray
    ) -> np.ndarray:
        """Inverse of :meth:`_quantize`."""
        bits = self.config.thinking_bits
        n_levels = (1 << bits) - 1
        gs = self.config.group_size
        groups = codes.astype(np.float32) / (n_levels / 2) * scale[:, np.newaxis]
        return groups.reshape(-1)
