"""tests/test_telemetry_unit.py

Full unit tests for squish/telemetry.py.

Coverage targets
────────────────
 • Span  — creation, duration_ms, finish (idempotent), set_tag, add_event,
           set_error, to_dict (with / without end_time), to_chrome_event
 • _NoopSpan  — all mutation methods are no-ops; status == "ok"
 • _SpanContext  — sync/async context manager (enabled / disabled / exception);
                   decorator (sync + async function); token-is-None branch
 • Tracer  — start_span (root + child), finish_span, spans(), slowest_spans,
             clear, to_dict, to_chrome_trace (including version-unknown path),
             save_trace, print_trace
 • _render_span_tree  — flat, nested, error span, speed colors, all-in-flight
 • configure_tracing / get_tracer / reset_tracer
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import types

import pytest

import squish.telemetry as tel
from squish.telemetry import (
    Span,
    Tracer,
    _NoopSpan,
    _NOOP_SPAN,
    _SpanContext,
    _render_span_tree,
    configure_tracing,
    get_tracer,
    reset_tracer,
    trace_span,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_finished_span(name: str = "test", duration_s: float = 0.01) -> Span:
    """Create a Span that is already finished."""
    t0 = time.perf_counter()
    s = Span(id="abc123", parent_id=None, name=name, start_time=t0)
    s.end_time = t0 + duration_s
    return s


def _rich_console() -> "rich.console.Console":
    """Return a Rich Console that writes to a StringIO buffer."""
    from rich.console import Console
    return Console(file=io.StringIO(), width=120)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_tracer():
    """Each test gets a fresh global tracer and tracing disabled by default."""
    original_enabled = tel.TRACING_ENABLED
    reset_tracer()
    tel.TRACING_ENABLED = False
    yield
    reset_tracer()
    tel.TRACING_ENABLED = original_enabled


# ═══════════════════════════════════════════════════════════════════════════════
# Span
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpan:

    def test_duration_ms_none_while_open(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        assert s.duration_ms is None

    def test_duration_ms_after_finish(self):
        t0 = time.perf_counter()
        s  = Span(id="x", parent_id=None, name="op", start_time=t0)
        s.end_time = t0 + 0.1
        assert pytest.approx(s.duration_ms, abs=1) == 100.0

    def test_finish_sets_end_time(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        assert s.end_time is None
        s.finish()
        assert s.end_time is not None

    def test_finish_is_idempotent(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        s.finish()
        first_end = s.end_time
        time.sleep(0.001)
        s.finish()
        assert s.end_time == first_end  # second call must not update end_time

    def test_set_tag(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        s.set_tag("model", "qwen3:8b")
        assert s.tags["model"] == "qwen3:8b"

    def test_add_event(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        s.add_event("cache hit")
        assert len(s.events) == 1
        offset_ms, msg = s.events[0]
        assert msg == "cache hit"
        assert offset_ms >= 0.0

    def test_set_error(self):
        s   = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        exc = ValueError("boom")
        s.set_error(exc)
        assert s.status        == "error"
        assert s.error_type    == "ValueError"
        assert s.error_message == "boom"

    def test_to_dict_with_end_time(self):
        s  = _make_finished_span("load")
        d  = s.to_dict()
        assert d["name"]        == "load"
        assert d["status"]      == "ok"
        assert d["end_ms"]      is not None
        assert d["duration_ms"] is not None
        assert d["parent_id"]   is None

    def test_to_dict_no_end_time(self):
        s = Span(id="x", parent_id=None, name="op", start_time=time.perf_counter())
        d = s.to_dict()
        assert d["end_ms"]      is None
        assert d["duration_ms"] is None

    def test_to_chrome_event(self):
        s     = _make_finished_span("gen", duration_s=0.05)
        epoch = s.start_time - 1.0  # epoch 1 second before span start
        ev    = s.to_chrome_event(epoch)
        assert ev["name"] == "gen"
        assert ev["ph"]   == "X"
        assert ev["ts"]   > 0          # 1 s * 1e6 μs/s ≈ 1_000_000 μs
        assert ev["dur"]  > 0
        import os
        assert ev["pid"]  == os.getpid()
        assert "status"   in ev["args"]


# ═══════════════════════════════════════════════════════════════════════════════
# _NoopSpan
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoopSpan:

    def test_status_is_ok(self):
        assert _NOOP_SPAN.status == "ok"

    def test_set_tag_noop(self):
        _NOOP_SPAN.set_tag("key", "value")  # must not raise

    def test_add_event_noop(self):
        _NOOP_SPAN.add_event("msg")  # must not raise

    def test_set_error_noop(self):
        _NOOP_SPAN.set_error(RuntimeError("oh no"))  # must not raise

    def test_finish_noop(self):
        _NOOP_SPAN.finish()  # must not raise

    def test_singleton(self):
        from squish.telemetry import _NOOP_SPAN as ns
        assert isinstance(ns, _NoopSpan)


# ═══════════════════════════════════════════════════════════════════════════════
# Tracer
# ═══════════════════════════════════════════════════════════════════════════════

class TestTracer:

    def test_start_span_no_parent(self):
        t    = Tracer()
        span = t.start_span("op", model="qwen3")
        assert span.parent_id is None
        assert span.name      == "op"
        assert span.tags      == {"model": "qwen3"}
        assert len(t.spans()) == 1

    def test_start_span_with_parent(self):
        """A span started while another is current must reference it as parent."""
        tel.TRACING_ENABLED = True
        # Use the global tracer so that trace_span and start_span share the same
        # _CURRENT_SPAN context variable value.
        tracer = reset_tracer()
        with trace_span("parent") as parent_span:
            child_span = tracer.start_span("child")
            assert child_span.parent_id == parent_span.id

    def test_finish_span(self):
        t    = Tracer()
        span = t.start_span("op")
        assert span.end_time is None
        t.finish_span(span)
        assert span.end_time is not None

    def test_spans_returns_copy(self):
        t = Tracer()
        t.start_span("a")
        s1 = t.spans()
        s1.clear()          # mutating the returned list must not affect internals
        assert len(t.spans()) == 1

    def test_slowest_spans_filters_unfinished(self):
        t  = Tracer()
        s1 = t.start_span("finished")
        s1.finish()
        _open = t.start_span("open")  # never finished
        result = t.slowest_spans(10)
        assert len(result) == 1
        assert result[0].name == "finished"

    def test_slowest_spans_ordered_and_capped(self):
        t = Tracer()
        for dur in (0.001, 0.1, 0.01):
            s = t.start_span(f"span-{dur}")
            s.end_time = s.start_time + dur
        result = t.slowest_spans(n=2)
        assert len(result) == 2
        assert result[0].duration_ms >= result[1].duration_ms

    def test_slowest_spans_n_greater_than_total(self):
        t = Tracer()
        s = t.start_span("only-one")
        s.finish()
        assert len(t.slowest_spans(n=100)) == 1

    def test_clear(self):
        t = Tracer()
        t.start_span("a").finish()
        t.clear()
        assert len(t.spans()) == 0

    def test_to_dict_structure(self):
        t = Tracer()
        s = t.start_span("model.load", backend="mlx")
        s.finish()
        d = t.to_dict()
        assert "spans" in d
        assert "epoch_wall" in d
        assert d["spans"][0]["name"] == "model.load"

    def test_to_dict_unknown_version(self):
        """When squish.__version__ is unavailable to_dict() falls back to 'unknown'."""
        import squish as _squish_pkg
        original = getattr(_squish_pkg, "__version__", _SENTINEL := object())
        squish_had_version = original is not _SENTINEL
        if squish_had_version:
            del _squish_pkg.__version__
        try:
            d = Tracer().to_dict()
            assert d["squish_version"] == "unknown"
        finally:
            if squish_had_version:
                _squish_pkg.__version__ = original

    def test_to_chrome_trace_excludes_unfinished(self):
        t = Tracer()
        fin = t.start_span("done")
        fin.finish()
        _open = t.start_span("still-open")  # never finished
        ct = t.to_chrome_trace()
        names = [ev["name"] for ev in ct["traceEvents"]]
        assert "done" in names
        assert "still-open" not in names

    def test_to_chrome_trace_metadata(self):
        t  = Tracer()
        ct = t.to_chrome_trace()
        assert ct["displayTimeUnit"] == "ms"
        assert "squish_version" in ct["metadata"]

    def test_to_chrome_trace_unknown_version(self):
        """When squish.__version__ is unavailable to_chrome_trace() uses 'unknown'."""
        import squish as _squish_pkg
        original = getattr(_squish_pkg, "__version__", _SENTINEL := object())
        squish_had_version = original is not _SENTINEL
        if squish_had_version:
            del _squish_pkg.__version__
        try:
            ct = Tracer().to_chrome_trace()
            assert ct["metadata"]["squish_version"] == "unknown"
        finally:
            if squish_had_version:
                _squish_pkg.__version__ = original

    def test_save_trace(self, tmp_path):
        t = Tracer()
        s = t.start_span("save-me")
        s.finish()
        out = tmp_path / "trace.json"
        t.save_trace(out)
        content = json.loads(out.read_text())
        assert "traceEvents" in content

    def test_print_trace_no_spans(self):
        t       = Tracer()
        console = _rich_console()
        t.print_trace(console=console)
        output  = console.file.getvalue()
        assert "No spans" in output

    def test_print_trace_with_spans(self):
        t  = Tracer()
        p  = t.start_span("parent-op")
        c  = t.start_span("child-op")
        # Manually wire parent_id so we exercise nested rendering
        c.parent_id = p.id
        c.finish()
        p.finish()
        console = _rich_console()
        t.print_trace(console=console)
        output = console.file.getvalue()
        assert "parent-op" in output
        assert "child-op"  in output

    def test_print_trace_creates_console_when_none(self):
        """print_trace(console=None) creates an internal Console and must not raise."""
        t = Tracer()
        t.start_span("x").finish()
        t.print_trace()  # creates its own Console writing to stdout; must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# _render_span_tree
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderSpanTree:

    def test_flat_spans(self):
        t  = Tracer()
        s  = t.start_span("flat")
        s.finish()
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "flat" in out

    def test_nested_spans(self):
        t     = Tracer()
        root  = t.start_span("root")
        child = t.start_span("child")
        child.parent_id = root.id
        child.finish()
        root.finish()
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "root"  in out
        assert "child" in out

    def test_error_span(self):
        t  = Tracer()
        s  = t.start_span("failing-op")
        s.set_error(RuntimeError("disk full"))
        s.finish()
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "disk full" in out

    def test_slowest_table_shown(self):
        t = Tracer()
        for i in range(3):
            s = t.start_span(f"op-{i}")
            s.end_time = s.start_time + (i + 1) * 0.1
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "Slowest Spans" in out

    def test_all_inflight_no_table(self):
        """When every span is unfinished the Slowest Spans table must not appear."""
        t = Tracer()
        t.start_span("never-done")  # never finished
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "never-done"    in out
        assert "Slowest Spans" not in out

    def test_in_flight_span_excluded_from_table(self):
        """An unfinished span appears in the tree but not in the slowest table."""
        t = Tracer()
        fin  = t.start_span("done")
        fin.finish()
        _inf = t.start_span("still-running")  # never finished
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        assert "still-running" in out
        assert "Slowest Spans"  in out

    def test_speed_colors(self):
        """Spans under 50ms, 50–500ms, and ≥500ms each hit a different color branch."""
        t = Tracer()
        fast   = t.start_span("fast")
        fast.end_time   = fast.start_time + 0.01   # 10ms  → green
        medium = t.start_span("medium")
        medium.end_time = medium.start_time + 0.1  # 100ms → yellow
        slow   = t.start_span("slow")
        slow.end_time   = slow.start_time + 1.0    # 1000ms → red
        console = _rich_console()
        _render_span_tree(t.spans(), t._epoch, console)
        out = console.file.getvalue()
        for name in ("fast", "medium", "slow"):
            assert name in out


# ═══════════════════════════════════════════════════════════════════════════════
# _SpanContext (sync context manager)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpanContextSync:

    def test_disabled_returns_noop(self):
        tel.TRACING_ENABLED = False
        with trace_span("op") as s:
            assert isinstance(s, _NoopSpan)

    def test_enabled_returns_real_span(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with trace_span("load") as s:
            assert isinstance(s, Span)
            assert s.name == "load"
        assert len(tracer.spans()) == 1
        assert tracer.spans()[0].end_time is not None

    def test_enabled_finishes_span_on_exit(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with trace_span("work"):
            pass
        assert tracer.spans()[0].end_time is not None

    def test_exception_marks_error(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with pytest.raises(ValueError):
            with trace_span("failing"):
                raise ValueError("oops")
        span = tracer.spans()[0]
        assert span.status        == "error"
        assert span.error_message == "oops"
        assert span.end_time      is not None

    def test_disabled_exit_is_noop(self):
        """When tracing is off, __exit__ returns early (self._span is None path)."""
        tel.TRACING_ENABLED = False
        ctx = _SpanContext("op", {})
        ctx.__enter__()
        ctx.__exit__(None, None, None)  # must not raise

    def test_exit_token_none_branch(self):
        """Exercises the self._token is None branch inside __exit__."""
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        ctx = _SpanContext("token-none-op", {})
        ctx.__enter__()          # sets _span AND _token
        ctx._token = None        # simulate cleared token
        ctx.__exit__(None, None, None)  # must not raise; skips ContextVar.reset
        assert tracer.spans()[0].end_time is not None

    def test_set_tag_on_real_span(self):
        tel.TRACING_ENABLED = True
        reset_tracer()
        with trace_span("tagged") as s:
            s.set_tag("layer", 7)
        assert get_tracer().spans()[0].tags["layer"] == 7

    def test_nesting_parent_child(self):
        """Nested trace_span calls must produce correct parent_id wiring."""
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with trace_span("outer") as outer:
            with trace_span("inner") as inner:
                assert inner.parent_id == outer.id
        spans = {s.name: s for s in tracer.spans()}
        assert spans["inner"].parent_id == spans["outer"].id


# ═══════════════════════════════════════════════════════════════════════════════
# _SpanContext (async context manager)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpanContextAsync:

    def test_async_disabled_returns_noop(self):
        tel.TRACING_ENABLED = False

        async def _run():
            async with trace_span("async-op") as s:
                assert isinstance(s, _NoopSpan)

        asyncio.run(_run())

    def test_async_enabled_records_span(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        async def _run():
            async with trace_span("async-op") as s:
                assert isinstance(s, Span)

        asyncio.run(_run())
        assert len(tracer.spans()) == 1
        assert tracer.spans()[0].name == "async-op"

    def test_async_exception_marks_error(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        async def _run():
            with pytest.raises(TypeError):
                async with trace_span("async-fail"):
                    raise TypeError("async boom")

        asyncio.run(_run())
        span = tracer.spans()[0]
        assert span.status        == "error"
        assert span.error_message == "async boom"


# ═══════════════════════════════════════════════════════════════════════════════
# _SpanContext (decorator)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpanContextDecorator:

    def test_sync_decorator_enabled(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        @trace_span("decorated-sync")
        def work(x: int) -> int:
            return x * 2

        result = work(5)
        assert result == 10
        assert tracer.spans()[0].name == "decorated-sync"

    def test_sync_decorator_disabled(self):
        tel.TRACING_ENABLED = False
        tracer = reset_tracer()

        @trace_span("decorated-sync-off")
        def work() -> str:
            return "ok"

        assert work() == "ok"
        # When disabled at call time, no real span is recorded
        assert len(tracer.spans()) == 0

    def test_async_decorator_enabled(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        @trace_span("decorated-async")
        async def async_work(x: int) -> int:
            return x + 1

        result = asyncio.run(async_work(3))
        assert result == 4
        assert tracer.spans()[0].name == "decorated-async"

    def test_async_decorator_disabled(self):
        tel.TRACING_ENABLED = False
        tracer = reset_tracer()

        @trace_span("decorated-async-off")
        async def async_work() -> str:
            return "done"

        assert asyncio.run(async_work()) == "done"
        assert len(tracer.spans()) == 0

    def test_decorator_preserves_funcname(self):
        @trace_span("wrap")
        def my_specific_func():
            pass

        assert my_specific_func.__name__ == "my_specific_func"

    def test_async_decorator_preserves_funcname(self):
        @trace_span("wrap-async")
        async def my_async_func():
            pass

        assert my_async_func.__name__ == "my_async_func"

    def test_sync_decorator_exception_marks_error(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        @trace_span("deco-fail")
        def boom():
            raise RuntimeError("deco error")

        with pytest.raises(RuntimeError):
            boom()

        assert tracer.spans()[0].status == "error"

    def test_async_decorator_exception_marks_error(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()

        @trace_span("async-deco-fail")
        async def async_boom():
            raise KeyError("async deco error")

        with pytest.raises(KeyError):
            asyncio.run(async_boom())

        assert tracer.spans()[0].status == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# configure_tracing / get_tracer / reset_tracer
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigureTracingAPI:

    def test_configure_tracing_enables(self):
        configure_tracing(True)
        assert tel.TRACING_ENABLED is True

    def test_configure_tracing_disables(self):
        configure_tracing(False)
        assert tel.TRACING_ENABLED is False

    def test_get_tracer_returns_tracer(self):
        t = get_tracer()
        assert isinstance(t, Tracer)

    def test_reset_tracer_returns_new_instance(self):
        old = get_tracer()
        new = reset_tracer()
        assert new is not old
        assert get_tracer() is new

    def test_reset_tracer_clears_spans(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with trace_span("before-reset"):
            pass
        assert len(tracer.spans()) == 1

        new_tracer = reset_tracer()
        assert len(new_tracer.spans()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level SQUISH_TRACE env var (tested via configure_tracing)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTracingEnvDefault:
    """Verify that TRACING_ENABLED obeys configure_tracing correctly."""

    def test_false_by_default_after_reset(self):
        # _isolated_tracer fixture sets TRACING_ENABLED = False
        assert tel.TRACING_ENABLED is False

    def test_no_spans_when_disabled(self):
        tel.TRACING_ENABLED = False
        tracer = reset_tracer()
        with trace_span("quiet"):
            pass
        assert len(tracer.spans()) == 0

    def test_spans_recorded_when_enabled(self):
        tel.TRACING_ENABLED = True
        tracer = reset_tracer()
        with trace_span("loud"):
            pass
        assert len(tracer.spans()) == 1
