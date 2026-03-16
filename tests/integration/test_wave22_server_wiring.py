"""tests/test_wave22_server_wiring.py

Verifies that all Wave 22 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 22 modules (Server Wiring · Adaptive Serving · Observability):
  multi_tenant_sched, request_router, cache_warmup, token_budget_gate,
  observability_hook, request_coalesce, adaptive_quantize, health_check,
  fault_tolerance, model_pool, streaming_chunk, cost_estimator,
  sla_monitor, context_cache
"""

import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CacheWarmup
# ---------------------------------------------------------------------------


class TestCacheWarmupWiring:
    def test_warmup_config_fields(self):
        from squish.kv.cache_warmup import WarmupConfig

        cfg = WarmupConfig(top_k=16, min_access_count=2, max_prefix_tokens=64)
        assert cfg.top_k == 16
        assert cfg.min_access_count == 2
        assert cfg.max_prefix_tokens == 64

    def test_access_record_fields(self):
        from squish.kv.cache_warmup import AccessRecord

        rec = AccessRecord(prefix_hash=42, access_count=5, last_access=1.5)
        assert rec.prefix_hash == 42
        assert rec.access_count == 5
        assert rec.last_access == 1.5

    def test_predictor_record_and_candidates(self):
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig

        cfg = WarmupConfig(top_k=4, min_access_count=2, max_prefix_tokens=8)
        predictor = CacheWarmupPredictor(cfg)
        tokens = [1, 2, 3, 4]
        predictor.record_access(tokens, timestamp=0.0)
        predictor.record_access(tokens, timestamp=0.1)
        assert predictor.n_tracked == 1
        candidates = predictor.get_warmup_candidates()
        assert len(candidates) == 1
        assert predictor.stats.total_accesses == 2

    def test_warmup_stats_fields(self):
        from squish.kv.cache_warmup import WarmupStats

        s = WarmupStats(total_accesses=10, cache_warmups_issued=4, predicted_hits=3)
        assert s.total_accesses == 10
        assert s.cache_warmups_issued == 4
        assert s.predicted_hits == 3


# ---------------------------------------------------------------------------
# TokenBudgetGate
# ---------------------------------------------------------------------------


class TestTokenBudgetGateWiring:
    def test_budget_policy_fields(self):
        from squish.token.token_budget_gate import BudgetPolicy

        policy = BudgetPolicy(mode="soft", soft_penalty=0.2, warn_at_fraction=0.8)
        assert policy.mode == "soft"
        assert policy.soft_penalty == 0.2
        assert policy.warn_at_fraction == 0.8

    def test_gate_tick_exhausts(self):
        from squish.token.token_budget_gate import BudgetPolicy, TokenBudgetGate

        policy = BudgetPolicy(mode="hard")
        gate = TokenBudgetGate(max_tokens=5, policy=policy)
        results = [gate.tick() for _ in range(7)]
        # Ticks 1–4 should return True (budget not yet reached at < 5),
        # tick 5 returns False (budget now == max_tokens).
        assert results[4] is False
        assert gate.is_exhausted

    def test_gate_fraction_used(self):
        from squish.token.token_budget_gate import BudgetPolicy, TokenBudgetGate

        policy = BudgetPolicy()
        gate = TokenBudgetGate(max_tokens=10, policy=policy)
        gate.tick()
        gate.tick()
        gate.tick()
        assert abs(gate.fraction_used() - 0.3) < 1e-9
        assert gate.tokens_used == 3
        assert gate.remaining() == 7

    def test_budget_gate_stats(self):
        from squish.token.token_budget_gate import BudgetGateStats

        s = BudgetGateStats(
            total_requests=5,
            total_tokens_gated=50,
            hard_stops=2,
            warnings_issued=3,
        )
        assert s.total_requests == 5
        assert s.hard_stops == 2
        assert s.warnings_issued == 3


# ---------------------------------------------------------------------------
# RequestCoalesce
# ---------------------------------------------------------------------------


class TestRequestCoalesceWiring:
    def test_coalesce_config_defaults(self):
        from squish.serving.request_coalesce import CoalesceConfig

        cfg = CoalesceConfig(min_shared_tokens=4, max_group_size=4)
        assert cfg.min_shared_tokens == 4
        assert cfg.max_group_size == 4

    def test_coalesce_group_fields(self):
        from squish.serving.request_coalesce import CoalesceGroup

        g = CoalesceGroup(
            shared_prefix=[1, 2, 3],
            request_ids=["r1", "r2"],
            branch_tokens=[[4, 5], [6, 7]],
        )
        assert g.shared_prefix == [1, 2, 3]
        assert len(g.request_ids) == 2
        assert len(g.branch_tokens) == 2

    def test_prefix_coalescer_coalesce(self):
        from squish.serving.request_coalesce import CoalesceConfig, PrefixCoalescer

        cfg = CoalesceConfig(min_shared_tokens=3, max_group_size=4)
        coalescer = PrefixCoalescer(cfg)
        shared = [10, 20, 30, 40]
        coalescer.add_request("req-1", shared + [100, 101])
        coalescer.add_request("req-2", shared + [200, 201])
        coalescer.add_request("req-3", shared + [300])
        assert coalescer.n_pending == 3
        groups = coalescer.coalesce()
        assert coalescer.n_pending == 0
        assert len(groups) >= 1
        assert coalescer.stats.total_requests == 3

    def test_coalesce_stats_fields(self):
        from squish.serving.request_coalesce import CoalesceStats

        s = CoalesceStats(
            total_requests=6,
            total_groups_formed=2,
            total_tokens_saved=12,
        )
        assert s.total_requests == 6
        assert s.total_groups_formed == 2
        assert s.total_tokens_saved == 12


# ---------------------------------------------------------------------------
# AdaptiveQuantize
# ---------------------------------------------------------------------------


class TestAdaptiveQuantizeWiring:
    def test_pressure_thresholds_fields(self):
        from squish.quant.adaptive_quantize import PressureThresholds

        t = PressureThresholds(int8_threshold=0.70, int4_threshold=0.85)
        assert t.int8_threshold == 0.70
        assert t.int4_threshold == 0.85

    def test_quant_precision_constants(self):
        from squish.quant.adaptive_quantize import QuantPrecision

        assert QuantPrecision.FP16 == "fp16"
        assert QuantPrecision.INT8 == "int8"
        assert QuantPrecision.INT4 == "int4"

    def test_pressure_monitor_update_and_precision(self):
        from squish.quant.adaptive_quantize import (
            PressureMonitor,
            PressureThresholds,
            QuantPrecision,
        )

        t = PressureThresholds(int8_threshold=0.75, int4_threshold=0.90)
        cap = 4 * 1024 ** 3  # 4 GiB
        monitor = PressureMonitor(t, capacity_bytes=cap)
        monitor.update(0)
        assert monitor.current_precision == QuantPrecision.FP16
        monitor.update(int(0.80 * cap))
        assert monitor.current_precision == QuantPrecision.INT8
        monitor.update(int(0.92 * cap))
        assert monitor.current_precision == QuantPrecision.INT4

    def test_adaptive_quantizer_quantize_dequantize(self):
        from squish.quant.adaptive_quantize import (
            AdaptiveQuantizer,
            PressureMonitor,
            PressureThresholds,
            QuantPrecision,
        )

        rng = np.random.default_rng(0)
        t = PressureThresholds(int8_threshold=0.75, int4_threshold=0.90)
        cap = 4 * 1024 ** 3
        monitor = PressureMonitor(t, capacity_bytes=cap)
        monitor.update(int(0.80 * cap))  # force INT8
        quantizer = AdaptiveQuantizer(monitor)
        x = rng.standard_normal((16, 8)).astype(np.float32)
        q, scale = quantizer.quantize(x)
        assert q.dtype == np.int8
        assert scale > 0.0
        x_approx = quantizer.dequantize(q, scale, QuantPrecision.INT8)
        assert x_approx.shape == x.shape
        assert quantizer.stats.total_quantize_calls == 1
        assert quantizer.stats.int8_calls == 1


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheckWiring:
    def test_health_state_constants(self):
        from squish.serving.health_check import HealthState

        assert HealthState.OK == "ok"
        assert HealthState.DEGRADED == "degraded"
        assert HealthState.CRITICAL == "critical"

    def test_health_metric_state(self):
        from squish.serving.health_check import HealthMetric, HealthState

        m_ok = HealthMetric(name="p99", value=100.0,
                            threshold_warn=500.0, threshold_crit=2000.0)
        assert m_ok.state == HealthState.OK

        m_deg = HealthMetric(name="p99", value=800.0,
                             threshold_warn=500.0, threshold_crit=2000.0)
        assert m_deg.state == HealthState.DEGRADED

        m_crit = HealthMetric(name="p99", value=2500.0,
                              threshold_warn=500.0, threshold_crit=2000.0)
        assert m_crit.state == HealthState.CRITICAL

    def test_inference_health_monitor_record(self):
        from squish.serving.health_check import HealthState, InferenceHealthMonitor

        monitor = InferenceHealthMonitor(
            warn_latency_ms=500.0,
            crit_latency_ms=2000.0,
            warn_error_rate=0.05,
            crit_error_rate=0.20,
        )
        for _ in range(100):
            monitor.record_request(latency_ms=200.0, success=True)
        assert monitor.overall_health() == HealthState.OK
        # Inject a very slow, failing request.
        monitor.record_request(latency_ms=3000.0, success=False)
        # p99 should now be elevated; overall health should not be OK.
        assert monitor.stats.total_requests == 101
        assert monitor.stats.total_errors == 1

    def test_health_stats_error_rate(self):
        from squish.serving.health_check import HealthStats

        s = HealthStats(total_requests=20, total_errors=4)
        assert abs(s.error_rate - 0.2) < 1e-9
        empty = HealthStats()
        assert empty.error_rate == 0.0


# ---------------------------------------------------------------------------
# FaultTolerance
# ---------------------------------------------------------------------------


class TestFaultToleranceWiring:
    def test_fault_policy_fields(self):
        from squish.serving.fault_tolerance import FaultPolicy

        p = FaultPolicy(
            evict_kv_at=0.80,
            disable_draft_at=0.88,
            reduce_batch_at=0.95,
            min_batch_size=2,
        )
        assert p.evict_kv_at == 0.80
        assert p.disable_draft_at == 0.88
        assert p.reduce_batch_at == 0.95
        assert p.min_batch_size == 2

    def test_fault_action_constants(self):
        from squish.serving.fault_tolerance import FaultAction

        assert FaultAction.EVICT_KV == "evict_kv"
        assert FaultAction.DISABLE_DRAFT == "disable_draft"
        assert FaultAction.REDUCE_BATCH == "reduce_batch"
        assert FaultAction.RENEGOTIATE_SLO == "renegotiate_slo"

    def test_fault_handler_evaluate(self):
        from squish.serving.fault_tolerance import FaultAction, FaultHandler, FaultPolicy

        policy = FaultPolicy(evict_kv_at=0.85, disable_draft_at=0.90,
                             reduce_batch_at=0.95, min_batch_size=1)
        handler = FaultHandler(policy)

        # Below all thresholds — no actions.
        actions = handler.evaluate(pressure=0.50, current_batch_size=8)
        assert actions == []

        # Above evict_kv threshold only.
        actions = handler.evaluate(pressure=0.87, current_batch_size=8)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT not in actions

        # Above all thresholds.
        actions = handler.evaluate(pressure=0.96, current_batch_size=8)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT in actions
        assert FaultAction.REDUCE_BATCH in actions

    def test_fault_stats_fields(self):
        from squish.serving.fault_tolerance import FaultHandler, FaultPolicy

        policy = FaultPolicy()
        handler = FaultHandler(policy)
        evicted = handler.apply_evict_kv(16)
        assert evicted == 16
        assert handler.stats.kv_evictions == 16
        assert handler.stats.total_evaluations == 0


# ---------------------------------------------------------------------------
# ModelPool
# ---------------------------------------------------------------------------


class TestModelPoolWiring:
    def test_pool_entry_fields(self):
        from squish.serving.model_pool import PoolEntry

        entry = PoolEntry(model_id="phi-3-mini", size_mb=2048.0,
                          last_accessed=1.0, access_count=3)
        assert entry.model_id == "phi-3-mini"
        assert entry.size_mb == 2048.0
        assert entry.last_accessed == 1.0
        assert entry.access_count == 3

    def test_model_pool_register_acquire_release(self):
        from squish.serving.model_pool import ModelPool

        pool = ModelPool(capacity_mb=8192.0)
        pool.register("llama-3-8b", size_mb=4096.0)
        entry = pool.acquire("llama-3-8b")
        assert entry.model_id == "llama-3-8b"
        assert "llama-3-8b" in pool.loaded_models
        assert pool.stats.cache_misses == 1
        pool.release("llama-3-8b")
        # Second acquire should be a cache hit.
        pool.acquire("llama-3-8b")
        assert pool.stats.cache_hits == 1

    def test_model_pool_evict_lru(self):
        from squish.serving.model_pool import ModelPool

        pool = ModelPool(capacity_mb=4096.0)
        pool.register("small-a", size_mb=1024.0)
        pool.register("small-b", size_mb=1024.0)
        pool.acquire("small-a")
        pool.release("small-a")
        pool.acquire("small-b")
        pool.release("small-b")
        evicted = pool.evict_lru()
        assert evicted is not None
        assert pool.stats.total_evictions == 1

    def test_pool_stats_hit_rate(self):
        from squish.serving.model_pool import PoolStats

        s = PoolStats(total_acquires=10, cache_hits=7, cache_misses=3)
        assert abs(s.hit_rate - 0.7) < 1e-9
        empty = PoolStats()
        assert empty.hit_rate == 0.0


# ---------------------------------------------------------------------------
# StreamingChunk
# ---------------------------------------------------------------------------


class TestStreamingChunkWiring:
    def test_chunk_config_fields(self):
        from squish.streaming.streaming_chunk import ChunkConfig

        cfg = ChunkConfig(chunk_size=8, max_buffer=128)
        assert cfg.chunk_size == 8
        assert cfg.max_buffer == 128

    def test_backpressure_buffer_push_flush(self):
        from squish.streaming.streaming_chunk import BackpressureBuffer, ChunkConfig

        cfg = ChunkConfig(chunk_size=4, max_buffer=6)
        buf = BackpressureBuffer(cfg)
        for i in range(6):
            assert buf.push(i) is True
        # 7th push should trigger backpressure.
        assert buf.push(99) is False
        assert buf.peek_size() == 6
        chunk = buf.flush()
        assert len(chunk) == 4
        assert buf.peek_size() == 2

    def test_chunked_streamer_stream(self):
        from squish.streaming.streaming_chunk import ChunkConfig, ChunkedStreamer

        cfg = ChunkConfig(chunk_size=4)
        streamer = ChunkedStreamer(cfg)
        token_ids = list(range(10))
        chunks = streamer.stream(token_ids)
        assert chunks[0] == [0, 1, 2, 3]
        assert chunks[-1] == [8, 9]
        assert sum(len(c) for c in chunks) == 10
        assert streamer.stats.total_tokens_streamed == 10
        assert streamer.stats.total_chunks == len(chunks)

    def test_stream_stats_avg_chunk_size(self):
        from squish.streaming.streaming_chunk import StreamStats

        s = StreamStats(total_tokens_streamed=20, total_chunks=5)
        assert abs(s.avg_chunk_size - 4.0) < 1e-9
        empty = StreamStats()
        assert empty.avg_chunk_size == 0.0


# ---------------------------------------------------------------------------
# ContextCache
# ---------------------------------------------------------------------------


class TestContextCacheWiring:
    def test_cache_entry_fields(self):
        from squish.kv.context_cache import CacheEntry

        kv = np.zeros((4, 8), dtype=np.float32)
        entry = CacheEntry(
            entry_id="e1",
            token_hash=12345,
            kv_data=kv,
            created_at=time.time(),
            ttl_s=300.0,
        )
        assert entry.entry_id == "e1"
        assert entry.token_hash == 12345
        assert entry.kv_data.shape == (4, 8)
        assert not entry.is_expired

    def test_persistent_context_cache_put_get(self):
        from squish.kv.context_cache import PersistentContextCache

        cache = PersistentContextCache(max_entries=16, default_ttl_s=300.0)
        rng = np.random.default_rng(0)
        tokens = [1, 2, 3, 4, 5]
        kv = rng.standard_normal((4, 5, 8)).astype(np.float32)
        entry_id = cache.put(tokens, kv)
        assert isinstance(entry_id, str) and len(entry_id) > 0
        result = cache.get(tokens)
        assert result is not None
        assert result.shape == kv.shape
        assert cache.stats.hits == 1

    def test_context_cache_evict_expired(self):
        from squish.kv.context_cache import PersistentContextCache

        cache = PersistentContextCache(max_entries=8, default_ttl_s=0.001)
        rng = np.random.default_rng(1)
        for i in range(3):
            kv = rng.standard_normal((2, i + 1, 4)).astype(np.float32)
            cache.put([i + 1, i + 2], kv, ttl_s=0.001)
        time.sleep(0.05)
        n_evicted = cache.evict_expired()
        assert n_evicted == 3
        assert cache.n_entries == 0

    def test_context_cache_stats_hit_rate(self):
        from squish.kv.context_cache import ContextCacheStats

        s = ContextCacheStats(total_puts=5, total_gets=8, hits=6, misses=2)
        assert abs(s.hit_rate - 0.75) < 1e-9
        empty = ContextCacheStats()
        assert empty.hit_rate == 0.0
