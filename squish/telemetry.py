"""squish/telemetry.py — Structured tracing and span recording for Squish.

Every code path that runs inside Squish can be wrapped in a ``trace_span``
context manager (or decorated with ``@trace_span("name")``) to record its
start time, end time, parent/child relationship, arbitrary tags, and any
exception that propagates out.

After a run the collected spans can be:
  • Printed as a rich console tree with per-span timing bars.
  • Exported as a Chrome DevTools Trace Event Format JSON file that opens
    directly at https://speedscope.app or ``chrome://tracing`` as a flame
    graph, showing every module and its relative timing at a glance.

Usage
─────
Environment variables:
    SQUISH_TRACE=1          Enable span tracing (default: off).

Programmatic:
    from squish.telemetry import trace_span, get_tracer, configure_tracing

    configure_tracing(True)

    with trace_span("load-model", model="qwen3:8b") as span:
        model = load(path)
        span.set_tag("params", 8_000_000_000)

    @trace_span("generate")
    def gen(prompt: str) -> str: ...

    # Async context manager
    async with trace_span("request"):
        ...

    # Print tree to terminal
    get_tracer().print_trace()

    # Save Chrome-format flame graph
    get_tracer().save_trace("/tmp/squish-trace.json")

    # Reset for next run (useful in tests)
    from squish.telemetry import reset_tracer
    reset_tracer()
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Span",
    "Tracer",
    "trace_span",
    "get_tracer",
    "reset_tracer",
    "configure_tracing",
    "TRACING_ENABLED",
]

# ── Global tracing toggle ────────────────────────────────────────────────────
# Initialised from SQUISH_TRACE env var; overridable at runtime via
# configure_tracing().
TRACING_ENABLED: bool = os.environ.get("SQUISH_TRACE", "").strip() not in (
    "", "0", "false", "no",
)


def configure_tracing(enabled: bool) -> None:
    """Enable or disable span recording at runtime.

    This is process-global.  Typical use: call ``configure_tracing(True)``
    early in ``main()`` when ``--trace`` is present on the command line.
    """
    global TRACING_ENABLED
    TRACING_ENABLED = enabled


# ── Active-span context variable (async- and thread-safe) ───────────────────
# ContextVar propagates automatically across asyncio Tasks and is thread-local
# across OS threads, giving correct parent-span tracking in both concurrency
# models without any extra synchronisation.
_CURRENT_SPAN: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "squish_current_span", default=None,
)


# ── Span ─────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """A timed, attributed unit of work.

    Attributes
    ----------
    id            : Unique hex string (uuid4).
    parent_id     : ID of the enclosing span, or ``None`` for root spans.
    name          : Human-readable operation name.
    start_time    : ``time.perf_counter()`` at span creation.
    end_time      : ``time.perf_counter()`` when ``finish()`` was called.
    tags          : Arbitrary key/value metadata (string keys).
    events        : ``(offset_ms, message)`` tuples recorded mid-span.
    status        : ``"ok"`` or ``"error"``.
    error_type    : Exception class name once ``set_error`` is called.
    error_message : ``str(exc)`` once ``set_error`` is called.
    thread_id     : ``threading.get_ident()`` captured at creation.
    """

    id:            str
    parent_id:     str | None
    name:          str
    start_time:    float
    end_time:      float | None              = field(default=None)
    tags:          dict[str, Any]            = field(default_factory=dict)
    events:        list[tuple[float, str]]   = field(default_factory=list)
    status:        str                       = "ok"
    error_type:    str | None                = field(default=None)
    error_message: str | None                = field(default=None)
    thread_id:     int                       = field(default_factory=threading.get_ident)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def duration_ms(self) -> float | None:
        """Elapsed milliseconds, or ``None`` while the span is still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1_000

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def set_tag(self, key: str, value: Any) -> None:
        """Attach an arbitrary key/value attribute to this span."""
        self.tags[key] = value

    def add_event(self, message: str) -> None:
        """Record a timestamped log line inside this span."""
        offset_ms = (time.perf_counter() - self.start_time) * 1_000
        self.events.append((round(offset_ms, 3), message))

    def set_error(self, exc: BaseException) -> None:
        """Mark this span as errored and capture the exception details."""
        self.status        = "error"
        self.error_type    = type(exc).__name__
        self.error_message = str(exc)

    def finish(self) -> None:
        """Stamp ``end_time`` if not already set (idempotent)."""
        if self.end_time is None:
            self.end_time = time.perf_counter()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id":            self.id,
            "parent_id":     self.parent_id,
            "name":          self.name,
            "start_ms":      round(self.start_time * 1_000, 3),
            "end_ms":        (
                round(self.end_time * 1_000, 3)
                if self.end_time is not None else None
            ),
            "duration_ms":   (
                round(self.duration_ms, 3)
                if self.duration_ms is not None else None
            ),
            "status":        self.status,
            "error_type":    self.error_type,
            "error_message": self.error_message,
            "tags":          self.tags,
            "events":        self.events,
            "thread_id":     self.thread_id,
        }

    def to_chrome_event(self, epoch: float) -> dict[str, Any]:
        """Return a Chrome DevTools *complete event* (``ph="X"``) entry.

        Parameters
        ----------
        epoch : ``time.perf_counter()`` recorded at :class:`Tracer` creation.
                Used as the zero baseline so all timestamps are trace-relative.

        The returned dict belongs in the ``traceEvents`` list of the file
        opened at https://speedscope.app or ``chrome://tracing``.
        """
        ts_us  = (self.start_time - epoch) * 1_000_000   # μs from epoch
        dur_us = (self.duration_ms or 0.0) * 1_000       # μs
        return {
            "name": self.name,
            "ph":   "X",                                  # complete event
            "ts":   round(ts_us, 1),
            "dur":  round(dur_us, 1),
            "pid":  os.getpid(),
            "tid":  self.thread_id,
            "args": {**self.tags, "status": self.status},
        }


# ── Noop span ────────────────────────────────────────────────────────────────

class _NoopSpan:
    """Drop-in for :class:`Span` returned when tracing is disabled.

    All mutation methods are no-ops; accessing ``status`` returns ``"ok"``.
    A single shared singleton :data:`_NOOP_SPAN` is used to avoid allocation.
    """

    status: str = "ok"

    def set_tag(self, key: str, value: Any) -> None:
        """No-op."""

    def add_event(self, message: str) -> None:
        """No-op."""

    def set_error(self, exc: BaseException) -> None:
        """No-op."""

    def finish(self) -> None:
        """No-op."""


_NOOP_SPAN: _NoopSpan = _NoopSpan()


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """Thread-safe span collector with built-in CLI and file visualisation.

    Obtain the process-global singleton with :func:`get_tracer`.
    Create a fresh one with :func:`reset_tracer` (useful in tests).
    """

    def __init__(self) -> None:
        self._spans:      list[Span]    = []
        self._lock:       threading.Lock = threading.Lock()
        self._epoch:      float          = time.perf_counter()  # trace zero point
        self._wall_epoch: float          = time.time()          # wall-clock anchor

    # ── Span lifecycle ────────────────────────────────────────────────────────

    def start_span(self, name: str, **tags: Any) -> Span:
        """Create a new :class:`Span`, register it, and return it unfinished."""
        parent = _CURRENT_SPAN.get()
        span   = Span(
            id         = uuid.uuid4().hex,
            parent_id  = parent.id if parent is not None else None,
            name       = name,
            start_time = time.perf_counter(),
            tags       = dict(tags),
        )
        with self._lock:
            self._spans.append(span)
        return span

    def finish_span(self, span: Span) -> None:
        """Finish *span* (delegates to :meth:`Span.finish`)."""
        span.finish()

    # ── Data access ───────────────────────────────────────────────────────────

    def spans(self) -> list[Span]:
        """Return a thread-safe snapshot of all recorded spans (finished or not)."""
        with self._lock:
            return list(self._spans)

    def slowest_spans(self, n: int = 10) -> list[Span]:
        """Return up to *n* finished spans ordered by duration (slowest first)."""
        finished = [s for s in self.spans() if s.end_time is not None]
        return sorted(finished, key=lambda s: s.duration_ms or 0.0, reverse=True)[:n]

    def clear(self) -> None:
        """Remove all recorded spans and reset timing baselines."""
        with self._lock:
            self._spans.clear()
        self._epoch      = time.perf_counter()
        self._wall_epoch = time.time()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable trace summary."""
        try:
            from squish import __version__ as _ver
        except Exception:
            _ver = "unknown"
        return {
            "squish_version": _ver,
            "epoch_wall":     self._wall_epoch,
            "spans":          [s.to_dict() for s in self.spans()],
        }

    def to_chrome_trace(self) -> dict[str, Any]:
        """Return a Chrome DevTools Trace Event Format dict.

        Save with :meth:`save_trace` then open at https://speedscope.app or
        ``chrome://tracing`` → "Load" to see the full flame graph.
        """
        events = [
            s.to_chrome_event(self._epoch)
            for s in self.spans()
            if s.end_time is not None
        ]
        try:
            from squish import __version__ as _ver
        except Exception:
            _ver = "unknown"
        return {
            "traceEvents":     events,
            "displayTimeUnit": "ms",
            "metadata":        {"squish_version": _ver},
        }

    def save_trace(self, path: str | Path) -> None:
        """Write the Chrome-format flame graph JSON to *path*."""
        Path(path).write_text(
            json.dumps(self.to_chrome_trace(), indent=2),
            encoding="utf-8",
        )

    # ── Visualisation ─────────────────────────────────────────────────────────

    def print_trace(self, console: Any = None) -> None:
        """Print a rich hierarchical span tree to *console* (or stdout).

        Parameters
        ----------
        console : :class:`rich.console.Console` instance, or ``None`` to
                  create one writing to stdout.
        """
        from rich.console import Console

        if console is None:
            console = Console()

        spans = self.spans()
        if not spans:
            console.print("[dim]No spans recorded.[/dim]")
            return

        _render_span_tree(spans, self._epoch, console)


# ── Rich rendering ────────────────────────────────────────────────────────────

def _render_span_tree(spans: list[Span], epoch: float, console: Any) -> None:
    """Render *spans* as a :class:`rich.tree.Tree` plus a slowest-spans table."""
    from rich.table import Table
    from rich.tree  import Tree

    # Build parent → children index
    children: dict[str | None, list[Span]] = {}
    for s in spans:
        children.setdefault(s.parent_id, []).append(s)

    # Full timeline bounds for proportional Gantt bars
    t_min   = min(s.start_time for s in spans)
    t_max   = max((s.end_time or s.start_time) for s in spans)
    total_s = max(t_max - t_min, 1e-9)

    BAR_WIDTH = 14

    def _speed_color(ms: float | None) -> str:
        if ms is None:
            return "dim"
        if ms < 50:
            return "green"
        if ms < 500:
            return "yellow"
        return "red"

    def _timing_bar(span: Span) -> str:
        """Proportional ASCII Gantt bar across the full trace timeline."""
        offset    = span.start_time - t_min
        dur       = (span.end_time - span.start_time) if span.end_time else 0.0
        s_pos     = int((offset / total_s) * BAR_WIDTH)
        e_pos     = min(BAR_WIDTH, s_pos + max(1, int((dur / total_s) * BAR_WIDTH)))
        return " " * s_pos + "█" * (e_pos - s_pos) + " " * (BAR_WIDTH - e_pos)

    def _add_children(node: Any, parent_id: str | None) -> None:
        for span in sorted(children.get(parent_id, []), key=lambda s: s.start_time):
            ms      = span.duration_ms
            color   = _speed_color(ms)
            dur_str = f"{ms:.1f}ms" if ms is not None else "…"
            bar     = _timing_bar(span)
            label   = (
                f"[{color}]{span.name}[/{color}]"
                f"  [{color}]{dur_str:>9}[/{color}]"
                f"  [dim]{bar}[/dim]"
            )
            if span.status == "error":
                label += (
                    f"  [bold red]✗ {span.error_message or span.error_type}[/bold red]"
                )
            child_node = node.add(label)
            _add_children(child_node, span.id)

    tree = Tree(
        f"[bold cyan]Squish Trace[/bold cyan]  "
        f"[dim]{len(spans)} span{'s' if len(spans) != 1 else ''}  "
        f"timeline: {total_s * 1_000:.1f}ms[/dim]"
    )
    _add_children(tree, None)
    console.print(tree)

    # Slowest spans summary table
    finished = [s for s in spans if s.end_time is not None]
    if finished:
        slowest = sorted(finished, key=lambda s: s.duration_ms or 0.0, reverse=True)[:10]
        table   = Table(
            title="Slowest Spans", show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Span",     style="cyan",  no_wrap=True)
        table.add_column("Duration", justify="right")
        table.add_column("Status",   justify="center")
        for s in slowest:
            dur_str      = f"{s.duration_ms:.1f}ms" if s.duration_ms is not None else "?"
            status_style = "green" if s.status == "ok" else "bold red"
            table.add_row(
                s.name, dur_str,
                f"[{status_style}]{s.status}[/{status_style}]",
            )
        console.print(table)


# ── Span context manager / decorator ─────────────────────────────────────────

class _SpanContext:
    """Sync/async context manager **and** decorator factory for a named span.

    Returned by :func:`trace_span`.  Checking :data:`TRACING_ENABLED` is
    deferred to entry time so toggling tracing at runtime always takes effect
    on the *next* span entry, even for pre-decorated functions.
    """

    def __init__(self, name: str, tags: dict[str, Any]) -> None:
        self._name  = name
        self._tags  = tags
        self._span: Span | None                    = None
        self._token: contextvars.Token[Span | None] | None = None

    # ── Sync context manager ──────────────────────────────────────────────────

    def __enter__(self) -> Span | _NoopSpan:
        if not TRACING_ENABLED:
            self._span = None
            return _NOOP_SPAN
        self._span  = _GLOBAL_TRACER.start_span(self._name, **self._tags)
        self._token = _CURRENT_SPAN.set(self._span)
        return self._span

    def __exit__(
        self,
        exc_type: type | None,
        exc_val:  BaseException | None,
        exc_tb:   object,
    ) -> None:
        if self._span is None:
            return
        if self._token is not None:
            _CURRENT_SPAN.reset(self._token)
            self._token = None
        if exc_val is not None:
            self._span.set_error(exc_val)
        self._span.finish()
        self._span = None

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> Span | _NoopSpan:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val:  BaseException | None,
        exc_tb:   object,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    # ── Decorator ─────────────────────────────────────────────────────────────

    def __call__(self, func: Any) -> Any:
        """Wrap *func* so that each invocation runs inside a new span context.

        Works with both regular and ``async`` functions.  The span name and
        tags are those captured at decoration time.
        """
        name = self._name
        tags = self._tags

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with _SpanContext(name, tags):
                    return await func(*args, **kwargs)
            return _async_wrapper

        @functools.wraps(func)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _SpanContext(name, tags):
                return func(*args, **kwargs)
        return _sync_wrapper


# ── Public factory ────────────────────────────────────────────────────────────

def trace_span(name: str, **tags: Any) -> _SpanContext:
    """Return a context manager / decorator that wraps work in a named span.

    Parameters
    ----------
    name : Operation label shown in the trace tree and flame graph.
    tags : Optional key/value metadata attached to the span.

    Examples
    --------
    Context manager::

        with trace_span("model.load", backend="mlx") as span:
            model = load(path)
            span.set_tag("params", 8_000_000_000)

    Decorator::

        @trace_span("inference.generate")
        def generate(prompt: str) -> str:
            ...

    Async::

        async with trace_span("server.request", endpoint="/v1/chat"):
            response = await handle(request)
    """
    return _SpanContext(name, tags)


# ── Global tracer singleton ───────────────────────────────────────────────────

_GLOBAL_TRACER: Tracer = Tracer()


def get_tracer() -> Tracer:
    """Return the process-global :class:`Tracer` singleton."""
    return _GLOBAL_TRACER


def reset_tracer() -> Tracer:
    """Replace the global tracer with a fresh one and return it.

    Intended for use in tests to isolate span collections between test cases.
    """
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = Tracer()
    return _GLOBAL_TRACER
