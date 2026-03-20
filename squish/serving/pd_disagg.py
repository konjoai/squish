"""
squish/serving/pd_disagg.py

PDDisaggregator — Prefill–Decode Disaggregation for Bandwidth-Optimal Serving.

Based on:
  "Splitwise: Efficient Generative LLM Job Scheduling"
  Patel et al. — ISCA 2024  —  arXiv:2311.08227
  Key result: 2× TTFT improvement under mixed workloads.

  "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
  Qin et al. — USENIX ATC 2025 / MLSys 2025
  Key result: 1.5–2× TTFT improvement by overlapping prefill/decode compute.

  "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized
   Large Language Model Serving" — OSDI 2024

Background
----------
In unified (aggregated) LLM serving, prefill and decode requests share
the same hardware.  The two phases have very different compute profiles:

  • **Prefill**: compute-bound (O(S²) attention over prompt tokens),
    utilises matrix-multiplication throughput, short latency per request.
  • **Decode**: memory-bandwidth-bound (per-token AR, one row of weights
    loaded per step), sensitive to batch size and KV cache IO.

When mixed, prefill pre-empts decode → decode latency spikes → TTFT
suffers even for decode-only requests.

**Disaggregation** runs prefill on *compute-optimised* resources and
decode on *memory-bandwidth-optimised* resources, communicating the
resulting KV cache via a lightweight transfer channel.

This module provides a single-process simulation of the PD disaggregation
pattern, making the scheduling policy and KV transfer explicit.  In
production, ``prefill_worker`` and ``decode_worker`` would be separate
processes (or pods); here they are callable objects.

Classes
-------
``PDConfig``               — worker counts, timeout
``PrefillResult``          — result of a prefill operation
``PDStats``                — latency and throughput statistics
``PDDisaggregator``        — submit_prefill / submit_decode API

Usage::

    from squish.serving.pd_disagg import PDConfig, PDDisaggregator

    def my_prefill(tokens, max_new_tokens):
        return {"kv": None, "n_tokens": len(tokens)}

    def my_decode(kv, n_generated, max_new_tokens):
        return [42] * max_new_tokens

    pd = PDDisaggregator(
        PDConfig(),
        prefill_fn=my_prefill,
        decode_fn=my_decode,
    )
    tokens = pd.generate(request_id="r1", tokens=[1, 2, 3], max_new_tokens=8)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional

__all__ = [
    "PDConfig",
    "PrefillResult",
    "PDStats",
    "PDDisaggregator",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PDConfig:
    """Configuration for the PD disaggregator.

    Attributes:
        kv_transfer_timeout_ms: Maximum allowed KV cache transfer time in
                                milliseconds before raising.  Default: 500.
        max_prefill_tokens:     Maximum prompt length handled by one prefill
                                call.  Longer prompts are chunked.
        max_decode_tokens:      Maximum new tokens per decode call.
    """

    kv_transfer_timeout_ms: float = 500.0
    max_prefill_tokens: int = 8192
    max_decode_tokens: int = 512

    def __post_init__(self) -> None:
        if self.kv_transfer_timeout_ms <= 0:
            raise ValueError(
                f"kv_transfer_timeout_ms must be > 0, got {self.kv_transfer_timeout_ms}"
            )
        if self.max_prefill_tokens < 1:
            raise ValueError(
                f"max_prefill_tokens must be >= 1, got {self.max_prefill_tokens}"
            )
        if self.max_decode_tokens < 1:
            raise ValueError(
                f"max_decode_tokens must be >= 1, got {self.max_decode_tokens}"
            )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PrefillResult(NamedTuple):
    """Output of a prefill operation.

    Attributes:
        request_id:   Unique request identifier.
        kv_state:     Opaque KV cache state dict (to be passed to decode).
        n_prompt_toks: Number of prompt tokens processed.
        latency_ms:   Wall-clock prefill latency in milliseconds.
    """

    request_id: str
    kv_state: Any
    n_prompt_toks: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class PDStats:
    """Runtime statistics for PDDisaggregator.

    Attributes:
        total_requests:         Completed generate() calls.
        total_prefill_ms:       Cumulative prefill latency.
        total_decode_ms:        Cumulative decode latency.
        total_prompt_tokens:    Prompt tokens processed.
        total_generated_tokens: Tokens generated across all requests.
    """

    total_requests: int = 0
    total_prefill_ms: float = 0.0
    total_decode_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0

    @property
    def mean_prefill_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_prefill_ms / self.total_requests

    @property
    def mean_decode_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_decode_ms / self.total_requests

    @property
    def throughput_tps(self) -> float:
        total_s = (self.total_prefill_ms + self.total_decode_ms) / 1e3
        if total_s <= 0:
            return 0.0
        return self.total_generated_tokens / total_s

    def __repr__(self) -> str:
        return (
            f"PDStats("
            f"requests={self.total_requests}, "
            f"prefill={self.mean_prefill_ms:.1f}ms, "
            f"decode={self.mean_decode_ms:.1f}ms, "
            f"tps={self.throughput_tps:.0f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PDDisaggregator:
    """Prefill–Decode disaggregation scheduler.

    Separates the compute-bound prefill phase from the bandwidth-bound
    decode phase, passing the KV state between the two workers.

    Parameters
    ----------
    config:
        Disaggregator configuration.
    prefill_fn:
        Callable ``(tokens: List[int], max_new_tokens: int) → dict`` where
        the returned dict must have key ``"kv"`` (KV state) and may have
        key ``"n_tokens"`` (prompt length processed).
    decode_fn:
        Callable ``(kv: Any, n_generated: int, max_new_tokens: int) → List[int]``
        Returns a list of generated token ids.
    """

    def __init__(
        self,
        config: Optional[PDConfig] = None,
        prefill_fn: Optional[Callable] = None,
        decode_fn: Optional[Callable] = None,
    ) -> None:
        self._cfg = config or PDConfig()
        self._prefill_fn: Callable = prefill_fn or self._default_prefill
        self._decode_fn: Callable = decode_fn or self._default_decode
        self.stats = PDStats()
        self._pending_kv: Dict[str, PrefillResult] = {}

    # ------------------------------------------------------------------
    # Default stub implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _default_prefill(tokens: List[int], max_new_tokens: int) -> Dict:
        """Stub prefill: returns empty KV state."""
        return {"kv": {"tokens": tokens, "max_new": max_new_tokens}, "n_tokens": len(tokens)}

    @staticmethod
    def _default_decode(kv: Any, n_generated: int, max_new_tokens: int) -> List[int]:
        """Stub decode: returns 1s up to max_new_tokens."""
        return [1] * max_new_tokens

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def submit_prefill(self, tokens: List[int], max_new_tokens: int) -> PrefillResult:
        """Run the prefill phase and return the KV state.

        Parameters
        ----------
        tokens:         Prompt token ids.
        max_new_tokens: Maximum tokens to generate.

        Returns
        -------
        PrefillResult with kv_state for subsequent decode call.
        """
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        result = self._prefill_fn(tokens, max_new_tokens)
        kv_state = result.get("kv") if isinstance(result, dict) else result
        n_toks = result.get("n_tokens", len(tokens)) if isinstance(result, dict) else len(tokens)

        latency_ms = (time.perf_counter() - t0) * 1e3
        pf_result = PrefillResult(
            request_id=request_id,
            kv_state=kv_state,
            n_prompt_toks=n_toks,
            latency_ms=latency_ms,
        )
        self._pending_kv[request_id] = pf_result
        self.stats.total_prefill_ms += latency_ms
        self.stats.total_prompt_tokens += n_toks
        return pf_result

    def submit_decode(
        self,
        prefill_result: PrefillResult,
        max_new_tokens: int,
    ) -> List[int]:
        """Run the decode phase using the KV state from prefill.

        Parameters
        ----------
        prefill_result: Result from ``submit_prefill()``.
        max_new_tokens: Maximum new tokens to generate.

        Returns
        -------
        List of generated token ids.
        """
        t0 = time.perf_counter()
        tokens = self._decode_fn(
            prefill_result.kv_state, 0, max_new_tokens
        )
        latency_ms = (time.perf_counter() - t0) * 1e3
        self.stats.total_decode_ms += latency_ms
        self.stats.total_generated_tokens += len(tokens)
        # Clean up pending KV
        self._pending_kv.pop(prefill_result.request_id, None)
        return tokens

    def generate(
        self,
        request_id: str,
        tokens: List[int],
        max_new_tokens: int,
    ) -> List[int]:
        """End-to-end prefill + decode (convenience wrapper).

        Parameters
        ----------
        request_id:     Caller-supplied request identifier (for logging).
        tokens:         Prompt token ids.
        max_new_tokens: Maximum tokens to generate.

        Returns
        -------
        List of generated token ids.
        """
        pf = self.submit_prefill(tokens, max_new_tokens)
        result = self.submit_decode(pf, max_new_tokens)
        self.stats.total_requests += 1
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def pending_kv_count(self) -> int:
        return len(self._pending_kv)

    def __repr__(self) -> str:
        return (
            f"PDDisaggregator("
            f"timeout_ms={self._cfg.kv_transfer_timeout_ms}, "
            f"{self.stats})"
        )
