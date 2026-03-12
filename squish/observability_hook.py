#!/usr/bin/env python3
"""
squish/observability_hook.py

ObservabilityHook — Zero-overhead inference tracing with OpenTelemetry-compatible spans.

Inference pipelines benefit from structured tracing to diagnose latency regressions,
identify slow decode steps, and measure acceptance rates in speculative decoding.
This module provides a lightweight, in-process span collector whose output format
is compatible with the OpenTelemetry data model, making it easy to forward spans
to an OTLP-compliant backend (Jaeger, Grafana Tempo, etc.) via a thin adapter.

The :class:`SpanCollector` records start and end times for named spans, attaches
arbitrary key/value attributes, and exports all finished spans as a list of
JSON-serialisable dictionaries.  The :class:`InferenceTracer` is a typed facade
that creates well-known spans for the three main inference phases: prefill,
decode, and speculative-decode verification.

Example usage::

    import time
    from squish.observability_hook import SpanCollector, InferenceTracer, SpanKind

    collector = SpanCollector(max_spans=1000)
    tracer = InferenceTracer(collector)

    span = tracer.trace_prefill(seq_len=512)
    # ... run prefill ...
    collector.finish(span)

    span = tracer.trace_decode(step=0)
    # ... run decode step ...
    collector.finish(span)

    records = collector.export()
    print(f"collected {len(records)} spans")
    print(tracer.stats)
"""

from __future__ import annotations

__all__ = [
    "SpanKind",
    "Span",
    "SpanCollector",
    "InferenceTracer",
    "TracerStats",
]

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions


# ---------------------------------------------------------------------------
# Span kind constants
# ---------------------------------------------------------------------------


class SpanKind:
    """Named constants for well-known inference span kinds.

    Each constant is a plain string so spans can be filtered by kind using
    standard string comparisons without importing this module.

    Attributes:
        PREFILL:   KV-computation and attention for the prompt tokens.
        DECODE:    Single auto-regressive decode step.
        VERIFY:    Speculative-decode verification pass.
        KV_LOOKUP: KV cache lookup or prefix-cache query.
    """

    PREFILL: str = "prefill"
    DECODE: str = "decode"
    VERIFY: str = "verify"
    KV_LOOKUP: str = "kv_lookup"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """An OpenTelemetry-compatible trace span for a single inference operation.

    Attributes:
        span_id:    Unique identifier for this span (UUID4 string).
        kind:       Span kind string (see :class:`SpanKind`).
        start_time: Monotonic start timestamp in seconds (``time.monotonic()``).
        end_time:   Monotonic end timestamp in seconds.  Zero until the span is
                    finished via :meth:`SpanCollector.finish`.
        attributes: Arbitrary key/value pairs attached to this span.  Values
                    must be JSON-serialisable (str, int, float, bool, None).
    """

    span_id: str
    kind: str
    start_time: float
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration of the span in milliseconds.

        Returns 0.0 if the span has not yet been finished (``end_time == 0``).
        """
        if self.end_time == 0.0:
            return 0.0
        return (self.end_time - self.start_time) * 1_000.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class TracerStats:
    """Cumulative counters for spans created by an :class:`InferenceTracer`.

    Attributes:
        total_spans:    Total spans started (across all kinds).
        prefill_spans:  Spans with kind ``SpanKind.PREFILL``.
        decode_spans:   Spans with kind ``SpanKind.DECODE``.
    """

    total_spans: int = 0
    prefill_spans: int = 0
    decode_spans: int = 0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class SpanCollector:
    """In-process span store with bounded capacity and JSON export.

    Spans are started via :meth:`record`, finished via :meth:`finish`, and
    exported as a list of dictionaries via :meth:`export`.  Once the number of
    finished spans reaches ``max_spans``, the oldest half is evicted (circular
    buffer behaviour) to prevent unbounded memory growth.

    Args:
        max_spans: Maximum number of finished spans to retain.  Must be >= 1.
    """

    def __init__(self, max_spans: int = 10_000) -> None:
        if max_spans < 1:
            raise ValueError(f"max_spans must be >= 1, got {max_spans}")
        self._max_spans = max_spans
        self._finished: list[Span] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, kind: str, **attrs: Any) -> Span:
        """Start a new span of the given *kind* with optional attributes.

        The span is *not* automatically added to the finished list; callers
        must invoke :meth:`finish` when the operation completes.

        Args:
            kind:   A span kind string.  Should be one of the :class:`SpanKind`
                    constants but any non-empty string is accepted.
            **attrs: Arbitrary keyword arguments stored as span attributes.
                    Values must be JSON-serialisable.

        Returns:
            A new :class:`Span` with ``end_time == 0.0``.

        Raises:
            ValueError: if *kind* is empty.
        """
        if not kind:
            raise ValueError("kind must be a non-empty string")
        return Span(
            span_id=str(uuid.uuid4()),
            kind=kind,
            start_time=time.monotonic(),
            end_time=0.0,
            attributes=dict(attrs),
        )

    def finish(self, span: Span) -> None:
        """Mark *span* as finished by recording its end timestamp.

        Sets ``span.end_time`` to the current monotonic time and appends the
        span to the finished list.  If the finished list has reached
        ``max_spans``, the oldest half is discarded before appending.

        Args:
            span: The :class:`Span` to finish.  It must not have been finished
                  already (``end_time == 0.0``).

        Raises:
            ValueError: if *span* has already been finished.
        """
        if span.end_time != 0.0:
            raise ValueError(
                f"span '{span.span_id}' has already been finished "
                f"(end_time={span.end_time})"
            )
        span.end_time = time.monotonic()
        if len(self._finished) >= self._max_spans:
            # Evict oldest half to keep memory bounded.
            self._finished = self._finished[self._max_spans // 2 :]
        self._finished.append(span)

    def export(self) -> list[dict[str, Any]]:
        """Return all finished spans as JSON-serialisable dictionaries.

        Each dictionary contains the keys: ``span_id``, ``kind``,
        ``start_time``, ``end_time``, ``duration_ms``, and ``attributes``.

        Returns:
            A new list of dictionaries.  Modifying the returned list does not
            affect the collector's internal state.
        """
        return [
            {
                "span_id": s.span_id,
                "kind": s.kind,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": s.duration_ms,
                "attributes": dict(s.attributes),
            }
            for s in self._finished
        ]

    def clear(self) -> None:
        """Remove all finished spans from the collector."""
        self._finished.clear()

    @property
    def n_spans(self) -> int:
        """Number of finished spans currently held in the collector."""
        return len(self._finished)


# ---------------------------------------------------------------------------
# Inference tracer
# ---------------------------------------------------------------------------


class InferenceTracer:
    """Typed facade over :class:`SpanCollector` for common inference phases.

    Provides convenience methods for the three main span kinds used in LLM
    inference pipelines.  All spans must still be finished by calling
    ``collector.finish(span)`` when the timed operation completes.

    Args:
        collector: The :class:`SpanCollector` that backs this tracer.
    """

    def __init__(self, collector: SpanCollector) -> None:
        self._collector = collector
        self._stats = TracerStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace_prefill(self, seq_len: int) -> Span:
        """Start a prefill span for a sequence of *seq_len* tokens.

        Args:
            seq_len: Number of tokens in the prompt being prefilled.
                     Must be >= 1.

        Returns:
            An unfinished :class:`Span` with kind ``SpanKind.PREFILL``.

        Raises:
            ValueError: if *seq_len* < 1.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        span = self._collector.record(SpanKind.PREFILL, seq_len=seq_len)
        self._stats.total_spans += 1
        self._stats.prefill_spans += 1
        return span

    def trace_decode(self, step: int) -> Span:
        """Start a decode span for decode step number *step*.

        Args:
            step: Zero-based decode step index.  Must be >= 0.

        Returns:
            An unfinished :class:`Span` with kind ``SpanKind.DECODE``.

        Raises:
            ValueError: if *step* < 0.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        span = self._collector.record(SpanKind.DECODE, step=step)
        self._stats.total_spans += 1
        self._stats.decode_spans += 1
        return span

    def trace_verify(self, n_draft: int) -> Span:
        """Start a speculative-decode verification span.

        Args:
            n_draft: Number of draft tokens being verified in this pass.
                     Must be >= 1.

        Returns:
            An unfinished :class:`Span` with kind ``SpanKind.VERIFY``.

        Raises:
            ValueError: if *n_draft* < 1.
        """
        if n_draft < 1:
            raise ValueError(f"n_draft must be >= 1, got {n_draft}")
        span = self._collector.record(SpanKind.VERIFY, n_draft=n_draft)
        self._stats.total_spans += 1
        return span

    @property
    def stats(self) -> TracerStats:
        """Cumulative span counters for this tracer."""
        return self._stats

    @property
    def collector(self) -> SpanCollector:
        """The underlying :class:`SpanCollector`."""
        return self._collector
