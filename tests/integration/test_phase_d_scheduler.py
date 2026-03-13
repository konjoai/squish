"""
tests/test_phase_d_scheduler.py

Unit tests for Phase D Scheduler Upgrades:
  D1 — Double-buffer: _prepared_queue, _prepare_thread, lifecycle changes
  D2 — Cache-aware prefix grouping: BatchScheduler._group_by_prefix
  D3 — Decode-before-prefill ordering: BatchScheduler._sort_decode_first

All tests use mock model + tokenizer — no MLX or real model required.
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock

from squish.serving.scheduler import BatchScheduler, _Request

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scheduler(**kwargs) -> BatchScheduler:
    """Build a BatchScheduler with mock model + tokenizer."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.encode.return_value = [1, 2, 3]
    defaults = dict(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        batch_window_ms=10.0,
    )
    defaults.update(kwargs)
    return BatchScheduler(**defaults)


def _req(
    request_id: str = "r0",
    input_ids: list[int] | None = None,
    generated_ids: list[int] | None = None,
) -> _Request:
    """Build a minimal _Request for testing."""
    r = _Request(
        request_id=request_id,
        input_ids=input_ids if input_ids is not None else [1, 2, 3],
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        stop_ids=[],
        seed=None,
    )
    if generated_ids:
        r.generated_ids.extend(generated_ids)
    return r


# ── D1: Double-buffer fields ──────────────────────────────────────────────────

class TestDoubleBufferInit:
    """D1: verify new __init__ fields are present and correctly configured."""

    def test_prepared_queue_is_queue_instance(self):
        sched = _make_scheduler()
        assert isinstance(sched._prepared_queue, queue.Queue)

    def test_prepared_queue_maxsize_one(self):
        sched = _make_scheduler()
        assert sched._prepared_queue.maxsize == 1

    def test_prepare_thread_initially_none(self):
        sched = _make_scheduler()
        assert sched._prepare_thread is None

    def test_prepared_queue_initially_empty(self):
        sched = _make_scheduler()
        assert sched._prepared_queue.qsize() == 0


class TestDoubleBufferLifecycle:
    """D1: both threads start and stop together."""

    def test_prepare_thread_started_after_start(self):
        sched = _make_scheduler()
        sched.start()
        assert sched._prepare_thread is not None
        assert sched._prepare_thread.is_alive()
        sched.stop(timeout=2.0)

    def test_worker_thread_started_after_start(self):
        sched = _make_scheduler()
        sched.start()
        assert sched._thread is not None
        assert sched._thread.is_alive()
        sched.stop(timeout=2.0)

    def test_is_running_requires_both_threads(self):
        """is_running() should only be True when both threads are alive."""
        sched = _make_scheduler()
        sched.start()
        assert sched.is_running() is True
        sched.stop(timeout=2.0)
        assert sched.is_running() is False

    def test_prepare_thread_none_after_stop(self):
        sched = _make_scheduler()
        sched.start()
        sched.stop(timeout=2.0)
        assert sched._prepare_thread is None

    def test_worker_thread_none_after_stop(self):
        sched = _make_scheduler()
        sched.start()
        sched.stop(timeout=2.0)
        assert sched._thread is None

    def test_stop_without_start_safe_for_both_threads(self):
        sched = _make_scheduler()
        sched.stop(timeout=0.5)  # no threads started — must not raise

    def test_stats_includes_prepared_queue(self):
        sched = _make_scheduler()
        assert "prepared_queue" in sched.stats()

    def test_stats_prepared_queue_zero_initially(self):
        sched = _make_scheduler()
        assert sched.stats()["prepared_queue"] == 0


# ── D2: Cache-aware prefix grouping ──────────────────────────────────────────

class TestGroupByPrefix:
    """D2: _group_by_prefix selects up to max_batch_size, grouping by prefix."""

    def test_empty_pool_returns_empty(self):
        """A pool that fits max_batch_size returns itself unchanged."""
        sched = _make_scheduler(max_batch_size=4)
        selected, leftovers = sched._group_by_prefix([])
        assert selected == []
        assert leftovers == []

    def test_small_pool_no_leftovers(self):
        """Pool <= max_batch_size → all selected, no leftovers."""
        sched = _make_scheduler(max_batch_size=4)
        pool = [_req(f"r{i}", [i + 1, i + 2, i + 3]) for i in range(3)]
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 3
        assert leftovers == []

    def test_exact_max_batch_no_leftovers(self):
        """Pool == max_batch_size → all selected."""
        sched = _make_scheduler(max_batch_size=4)
        pool = [_req(f"r{i}", [i, i + 1, i + 2]) for i in range(4)]
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 4
        assert leftovers == []

    def test_overflow_pool_produces_leftovers(self):
        """Pool > max_batch_size → exactly max_batch selected, rest leftover."""
        sched = _make_scheduler(max_batch_size=2)
        pool = [_req(f"r{i}", [i + 1, i + 2]) for i in range(5)]
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 2
        assert len(leftovers) == 3

    def test_selected_plus_leftovers_partitions_pool(self):
        """Every request in pool appears in exactly one of selected/leftovers."""
        sched = _make_scheduler(max_batch_size=3)
        pool = [_req(f"r{i}", [i + 1, i + 2]) for i in range(7)]
        selected, leftovers = sched._group_by_prefix(pool)
        all_ids = {r.request_id for r in selected + leftovers}
        pool_ids = {r.request_id for r in pool}
        assert all_ids == pool_ids

    def test_same_prefix_pair_preferred_over_solo(self):
        """Two requests with the same 64-token prefix are batched together."""
        sched = _make_scheduler(max_batch_size=2)
        shared = list(range(64))
        req_a = _req("a", shared[:])
        req_b = _req("b", shared[:])
        req_c = _req("c", list(range(64, 128)))  # different prefix
        # Pool exceeds max_batch(2), so prefix grouping kicks in.
        selected, leftovers = sched._group_by_prefix([req_a, req_b, req_c])
        assert len(selected) == 2
        assert req_a in selected
        assert req_b in selected
        assert req_c in leftovers

    def test_largest_group_fills_batch_first(self):
        """When multiple prefix groups exist, the largest cohort wins."""
        sched = _make_scheduler(max_batch_size=3)
        prefix_a = list(range(64))          # 2 requests
        prefix_b = list(range(64, 128))     # 3 requests — larger
        reqs_a = [_req(f"a{i}", prefix_a[:]) for i in range(2)]
        reqs_b = [_req(f"b{i}", prefix_b[:]) for i in range(3)]
        pool = reqs_a + reqs_b              # 5 total, max_batch=3
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 3
        # All 3 from the larger group (B) should fill the batch.
        for r in reqs_b:
            assert r in selected
        for r in reqs_a:
            assert r in leftovers

    def test_batch_full_remainder_are_leftovers(self):
        """Once batch is full, any remaining groups go entirely to leftovers."""
        sched = _make_scheduler(max_batch_size=2)
        shared = list(range(64))
        pool = [_req(f"r{i}", shared[:]) for i in range(5)]
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 2
        assert len(leftovers) == 3
        for r in selected:
            assert r.input_ids == shared

    def test_unique_prefixes_fifo_fallback(self):
        """All unique prefixes → FIFO order preserved across selected + leftovers."""
        sched = _make_scheduler(max_batch_size=3)
        # Each request has a unique 64-token prefix.
        pool = [_req(f"r{i}", [i * 100 + j for j in range(64)]) for i in range(5)]
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 3
        assert len(leftovers) == 2

    def test_new_group_key_created_for_first_occurrence(self):
        """First request for a prefix creates a new group (branch: key not in groups)."""
        sched = _make_scheduler(max_batch_size=4)
        # Two different prefixes — each creates a new group on first occurrence.
        _req("x", list(range(64)))
        _req("y", list(range(64, 128)))
        # Pool equal to max_batch — early return, no grouping needed.
        # Use 5 to force grouping: 3 of x, 2 of y.
        pool_x = [_req(f"x{i}", list(range(64))) for i in range(3)]
        pool_y = [_req(f"y{i}", list(range(64, 128))) for i in range(2)]
        pool = pool_x + pool_y  # 5 > max_batch=4
        selected, leftovers = sched._group_by_prefix(pool)
        assert len(selected) == 4
        # group x (size 3) is selected in full, one from group y fills slot 4
        assert len(leftovers) == 1


# ── D3: Decode-before-prefill ordering ───────────────────────────────────────

class TestSortDecodeFirst:
    """D3: _sort_decode_first places decode requests before prefill requests."""

    def test_empty_batch_returns_empty(self):
        sched = _make_scheduler()
        assert sched._sort_decode_first([]) == []

    def test_all_prefill_order_preserved(self):
        """Requests with no generated tokens remain in their original order."""
        sched = _make_scheduler()
        pool = [_req(f"r{i}") for i in range(3)]  # all prefill
        result = sched._sort_decode_first(pool)
        assert result == pool

    def test_all_decode_order_preserved(self):
        """Requests with generated tokens remain in their original order."""
        sched = _make_scheduler()
        pool = [_req(f"r{i}", generated_ids=[100 + i]) for i in range(3)]
        result = sched._sort_decode_first(pool)
        assert result == pool

    def test_single_decode_before_single_prefill(self):
        prefill = _req("pf")
        decode_ = _req("dec", generated_ids=[42])
        sched = _make_scheduler()
        result = sched._sort_decode_first([prefill, decode_])
        assert result[0] is decode_
        assert result[1] is prefill

    def test_multiple_decode_all_before_prefill(self):
        sched = _make_scheduler()
        prefills = [_req(f"pf{i}") for i in range(2)]
        decodes  = [_req(f"dec{i}", generated_ids=[i]) for i in range(2)]
        # Interleave: pf0, dec0, pf1, dec1
        batch  = [prefills[0], decodes[0], prefills[1], decodes[1]]
        result = sched._sort_decode_first(batch)
        # First two must be the decode requests (order within group preserved).
        assert result[0] is decodes[0]
        assert result[1] is decodes[1]
        assert result[2] is prefills[0]
        assert result[3] is prefills[1]

    def test_total_count_preserved(self):
        """No requests are lost or duplicated."""
        sched = _make_scheduler()
        pool = [
            _req(f"r{i}", generated_ids=([i] if i % 2 == 0 else []))
            for i in range(6)
        ]
        result = sched._sort_decode_first(pool)
        assert len(result) == 6
        assert {r.request_id for r in result} == {r.request_id for r in pool}

    def test_decode_flag_is_presence_of_generated_ids(self):
        """An empty generated_ids list is still treated as prefill."""
        sched = _make_scheduler()
        prefill = _req("pf")         # generated_ids=[]
        decode_ = _req("dec", generated_ids=[99])
        result  = sched._sort_decode_first([prefill, decode_])
        assert result.index(decode_) < result.index(prefill)
