"""
tests/serving/test_phase5_concurrency_stress.py

Kill-test evidence for Phase 5 of the memory-governor eviction sprint: a
concurrency stress pass across everything built in Phases 1-4. Fires rapid,
overlapping pressure transitions (including skipped levels, e.g.
NORMAL -> CRITICAL directly) on one thread while concurrent "request"
threads exercise the BlockKVCache/PromptKVStore eviction paths,
``_effective_max_kv_size()``, and the CRITICAL-shedding middleware on other
threads. Assertions check real invariants — budget/eviction correctness,
exact baseline restoration, no corrupted or 500 responses — not just
"the process didn't crash."
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import squish.server as _srv
from squish.kv.block_kv_cache import BlockKVCache
from squish.kv.prompt_kv_cache import PromptKVStore
from squish.serving.memory_governor import LEVEL_CRITICAL, LEVEL_NORMAL, LEVEL_URGENT, LEVEL_WARNING

_LEVELS = (LEVEL_NORMAL, LEVEL_WARNING, LEVEL_URGENT, LEVEL_CRITICAL)
# Deliberately includes non-adjacent jumps (NORMAL->CRITICAL, CRITICAL->NORMAL)
# so the storm isn't just a tidy escalate/de-escalate cycle.
_STORM_SEQUENCE = (
    LEVEL_NORMAL, LEVEL_CRITICAL, LEVEL_WARNING, LEVEL_NORMAL,
    LEVEL_URGENT, LEVEL_CRITICAL, LEVEL_NORMAL, LEVEL_WARNING,
    LEVEL_URGENT, LEVEL_WARNING, LEVEL_CRITICAL, LEVEL_URGENT,
)


@pytest.fixture(autouse=True)
def _reset_state():
    orig_bkv     = _srv._block_kv_cache
    orig_pkv     = _srv._prompt_kv_store
    orig_hot     = _srv._original_hot_max_bytes
    orig_prompt  = _srv._original_prompt_max_bytes
    orig_gov     = _srv._memory_governor
    orig_max_kv  = _srv._max_kv_size

    _srv._block_kv_cache          = None
    _srv._prompt_kv_store         = None
    _srv._original_hot_max_bytes    = None
    _srv._original_prompt_max_bytes = None
    _srv._memory_governor         = None
    _srv._max_kv_size             = None

    yield

    _srv._block_kv_cache          = orig_bkv
    _srv._prompt_kv_store         = orig_pkv
    _srv._original_hot_max_bytes    = orig_hot
    _srv._original_prompt_max_bytes = orig_prompt
    _srv._memory_governor         = orig_gov
    _srv._max_kv_size             = orig_max_kv


def _fake_block_arrays(block_size, n_layers=2, n_heads=4, head_dim=16):
    keys = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    vals = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    return keys, vals


class TestCacheAndBudgetStormSurviveConcurrency:
    def test_rapid_pressure_transitions_with_concurrent_cache_and_budget_traffic(self, tmp_path):
        # hot_max_bytes sized so shrink fractions actually force eviction
        # (each block is 4096 bytes; see test_memory_governor_wiring.py).
        cache = BlockKVCache(
            cache_dir=tmp_path / "block", block_size=8, model_key="stress",
            hot_max_bytes=64 * 1024,
        )
        store = PromptKVStore(cache_dir=tmp_path / "prompt", max_bytes=10_000_000)
        _srv._block_kv_cache  = cache
        _srv._prompt_kv_store = store
        _srv._max_kv_size     = 32768
        _srv._memory_governor = None  # driven by direct calls below, not a real poller

        errors: list[BaseException] = []
        stop = threading.Event()

        def pressure_storm():
            try:
                i = 0
                # Runs until told to stop (after the work threads below finish),
                # not a fixed count — otherwise this loop (near-instant plain
                # attribute writes) races ahead and finishes before the slower
                # cache/disk work threads even get going, leaving pressure
                # parked at one level for most of the test instead of storming.
                while not stop.is_set():
                    _srv._on_memory_pressure_change(_STORM_SEQUENCE[i % len(_STORM_SEQUENCE)])
                    i += 1
            except BaseException as exc:  # noqa: BLE001 — stress test must capture every failure mode
                errors.append(exc)

        def cache_writer(tid: int):
            try:
                for i in range(40):
                    base = (tid * 1000 + i) * 8
                    ids = list(range(base, base + 8))
                    k, v = _fake_block_arrays(8)
                    cache.store_blocks(ids, [k], [v])
                    match = cache.lookup_prefix(ids)
                    assert match.matched_tokens in (0, 8)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def prompt_writer(tid: int):
            try:
                keys, values = _fake_block_arrays(8)
                for i in range(40):
                    store.put(f"stress-prompt-{tid}-{i}", keys, values, offset=8)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def budget_reader():
            try:
                for _ in range(300):
                    result = _srv._effective_max_kv_size()
                    # Never negative, never above the configured operator ceiling.
                    assert result is None or 0 <= result <= 32768
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        storm_thread = threading.Thread(target=pressure_storm)
        work_threads = (
            [threading.Thread(target=cache_writer, args=(i,)) for i in range(4)]
            + [threading.Thread(target=prompt_writer, args=(i,)) for i in range(4)]
            + [threading.Thread(target=budget_reader) for _ in range(4)]
        )
        storm_thread.start()
        for t in work_threads:
            t.start()
        # Join the bounded work threads first so the storm keeps running for
        # their entire duration; only then signal it to stop.
        for t in work_threads:
            t.join(timeout=30)
        stop.set()
        storm_thread.join(timeout=5)

        assert not errors, f"concurrency stress raised: {errors!r}"

        # Drive pressure back to NORMAL explicitly and confirm EXACT baseline
        # restoration — proof _original_hot_max_bytes/_original_prompt_max_bytes
        # never drifted from the true original despite the storm.
        _srv._on_memory_pressure_change(LEVEL_NORMAL)
        assert cache.stats()["hot_max_bytes"] == 64 * 1024
        assert store.max_bytes == 10_000_000

        # Eviction invariant: the hot tier never exceeds its currently
        # configured budget by more than the single-entry floor allowance.
        stats = cache.stats()
        assert stats["hot_bytes"] <= stats["hot_max_bytes"] or stats["hot_entries"] <= 1


class TestMiddlewareStormSurvivesConcurrency:
    def test_concurrent_requests_during_pressure_storm_never_corrupt_responses(self):
        """Every response across a request storm racing a pressure storm must
        be EXACTLY one of two shapes: 200 (admitted, handler ran to
        completion) or 503 with the exact shed detail (rejected pre-dispatch).
        No 500s, no malformed bodies, no other outcome is structurally
        possible — the middleware only ever short-circuits BEFORE call_next
        or returns call_next's result untouched, so this also proves no
        in-flight request is ever aborted mid-handler."""
        governor = type("G", (), {"pressure_level": LEVEL_NORMAL})()
        _srv._memory_governor = governor

        app = FastAPI()
        app.add_middleware(_srv._MemoryPressureShedMiddleware)

        @app.get("/work")
        def work():
            return {"ok": True}

        errors: list[BaseException] = []
        results: list[tuple[int, dict]] = []
        results_lock = threading.Lock()
        stop = threading.Event()

        def pressure_storm():
            try:
                i = 0
                # Uncapped (runs until told to stop) — a fixed count races
                # ahead of the slower HTTP-request threads and finishes long
                # before they do, leaving pressure parked at one level for
                # most of the test instead of actually storming through it.
                while not stop.is_set():
                    governor.pressure_level = _STORM_SEQUENCE[i % len(_STORM_SEQUENCE)]
                    i += 1
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def requester(client: TestClient):
            try:
                for _ in range(30):
                    r = client.get("/work")
                    with results_lock:
                        results.append((r.status_code, r.json()))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        clients = [TestClient(app, raise_server_exceptions=False) for _ in range(6)]
        storm_thread = threading.Thread(target=pressure_storm)
        requester_threads = [threading.Thread(target=requester, args=(c,)) for c in clients]

        storm_thread.start()
        for t in requester_threads:
            t.start()
        # Join the bounded requester threads first so the storm keeps running
        # for their entire duration; only then signal it to stop.
        for t in requester_threads:
            t.join(timeout=30)
        stop.set()
        storm_thread.join(timeout=5)

        assert not errors, f"concurrency stress raised: {errors!r}"
        assert len(results) == 6 * 30

        expected_shed_body = {
            "detail": "Server under critical memory pressure — request rejected. Try again shortly."
        }
        for status, body in results:
            if status == 200:
                assert body == {"ok": True}
            elif status == 503:
                assert body == expected_shed_body
            else:
                pytest.fail(f"unexpected status {status} with body {body!r}")

        # The storm did pass through CRITICAL — sanity-check at least one
        # request actually got shed, so this test isn't vacuously green.
        assert any(status == 503 for status, _ in results)
        assert any(status == 200 for status, _ in results)
