"""Tests for Wave 49b serving modules: PromptCache, PipeInfer, Prepack."""

from __future__ import annotations

import unittest

import numpy as np

from squish.serving.prompt_cache import (
    PromptCacheConfig,
    PromptCacheKV,
    PromptCacheResult,
    PromptSchema,
)
from squish.serving.pipe_infer import (
    PipeInferConfig,
    PipeInferRequest,
    PipeInferScheduler,
    PipeInferTick,
)
from squish.serving.prepack import (
    PrepackBatch,
    PrepackConfig,
    PrepackRequest,
    PrepackScheduler,
)


# ---------------------------------------------------------------------------
# PromptCacheConfig
# ---------------------------------------------------------------------------


class TestPromptCacheConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PromptCacheConfig()
        self.assertEqual(cfg.max_schemas, 64)
        self.assertEqual(cfg.kv_dim, 128)
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.seed, 0)

    def test_valid_custom(self):
        cfg = PromptCacheConfig(max_schemas=10, kv_dim=64, n_heads=4, seed=7)
        self.assertEqual(cfg.max_schemas, 10)
        self.assertEqual(cfg.kv_dim, 64)
        self.assertEqual(cfg.n_heads, 4)

    def test_max_schemas_zero_raises(self):
        with self.assertRaises(ValueError):
            PromptCacheConfig(max_schemas=0)

    def test_kv_dim_zero_raises(self):
        with self.assertRaises(ValueError):
            PromptCacheConfig(kv_dim=0)

    def test_n_heads_zero_raises(self):
        with self.assertRaises(ValueError):
            PromptCacheConfig(n_heads=0)


# ---------------------------------------------------------------------------
# PromptSchema
# ---------------------------------------------------------------------------


class TestPromptSchema(unittest.TestCase):
    def test_n_constant_tokens_single_span(self):
        schema = PromptSchema(
            name="test",
            constant_spans=["hello world foo bar"],
            variable_slots=["query"],
        )
        self.assertEqual(schema.n_constant_tokens, 4)

    def test_n_constant_tokens_multiple_spans(self):
        schema = PromptSchema(
            name="test",
            constant_spans=["hello world", "foo bar baz"],
            variable_slots=[],
        )
        self.assertEqual(schema.n_constant_tokens, 5)

    def test_n_constant_tokens_empty_spans(self):
        schema = PromptSchema(name="test", constant_spans=[], variable_slots=["x"])
        self.assertEqual(schema.n_constant_tokens, 0)

    def test_schema_name(self):
        schema = PromptSchema(name="my_schema", constant_spans=[], variable_slots=[])
        self.assertEqual(schema.name, "my_schema")


# ---------------------------------------------------------------------------
# PromptCacheKV
# ---------------------------------------------------------------------------


class TestPromptCacheKV(unittest.TestCase):
    def _make_schema(self, name: str = "s1") -> PromptSchema:
        return PromptSchema(
            name=name,
            constant_spans=["You are helpful. Answer the user query."],
            variable_slots=["query"],
        )

    def setUp(self):
        self.cfg = PromptCacheConfig(max_schemas=4, kv_dim=16, n_heads=4, seed=0)
        self.cache = PromptCacheKV(self.cfg)

    def test_config_property(self):
        self.assertIs(self.cache.config, self.cfg)

    def test_default_init(self):
        cache = PromptCacheKV()
        self.assertIsNotNone(cache.config)

    def test_n_schemas_initial(self):
        self.assertEqual(self.cache.n_schemas, 0)

    def test_n_materialized_initial(self):
        self.assertEqual(self.cache.n_materialized, 0)

    def test_register_schema(self):
        self.cache.register_schema(self._make_schema())
        self.assertEqual(self.cache.n_schemas, 1)

    def test_register_duplicate_raises(self):
        self.cache.register_schema(self._make_schema())
        with self.assertRaises(ValueError):
            self.cache.register_schema(self._make_schema())

    def test_register_max_exceeded_raises(self):
        for i in range(4):
            self.cache.register_schema(self._make_schema(f"s{i}"))
        extra = PromptSchema(name="extra", constant_spans=[], variable_slots=[])
        with self.assertRaises(ValueError):
            self.cache.register_schema(extra)

    def test_list_schemas_empty(self):
        self.assertEqual(self.cache.list_schemas(), [])

    def test_list_schemas_after_register(self):
        self.cache.register_schema(self._make_schema("a"))
        self.cache.register_schema(self._make_schema("b"))
        names = self.cache.list_schemas()
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_materialize_synthetic(self):
        self.cache.register_schema(self._make_schema())
        kv = self.cache.materialize("s1")
        self.assertIsInstance(kv, np.ndarray)
        self.assertEqual(kv.dtype, np.float32)

    def test_materialize_shape(self):
        schema = self._make_schema()
        self.cache.register_schema(schema)
        kv = self.cache.materialize("s1")
        n_tok = schema.n_constant_tokens
        n_heads = self.cfg.n_heads
        head_dim = self.cfg.kv_dim // self.cfg.n_heads
        self.assertEqual(kv.shape, (max(1, n_tok), n_heads, head_dim))

    def test_materialize_custom_kv(self):
        self.cache.register_schema(self._make_schema())
        custom = np.ones((5, 4, 4), dtype=np.float32)
        kv = self.cache.materialize("s1", kv_data=custom)
        np.testing.assert_array_equal(kv, custom)

    def test_materialize_unknown_schema_raises(self):
        with self.assertRaises(KeyError):
            self.cache.materialize("nonexistent")

    def test_lookup_hit(self):
        self.cache.register_schema(self._make_schema())
        self.cache.materialize("s1")
        result = self.cache.lookup("s1")
        self.assertIsInstance(result, PromptCacheResult)
        self.assertTrue(result.hit)

    def test_lookup_hit_kv_not_none(self):
        self.cache.register_schema(self._make_schema())
        self.cache.materialize("s1")
        result = self.cache.lookup("s1")
        self.assertIsNotNone(result.cached_kv)

    def test_lookup_miss_unregistered(self):
        result = self.cache.lookup("unknown")
        self.assertFalse(result.hit)
        self.assertIsNone(result.cached_kv)
        self.assertIsNone(result.schema_name)

    def test_lookup_miss_not_materialized(self):
        self.cache.register_schema(self._make_schema())
        result = self.cache.lookup("s1")
        self.assertFalse(result.hit)

    def test_lookup_n_fresh_tokens(self):
        self.cache.register_schema(self._make_schema())
        self.cache.materialize("s1")
        result = self.cache.lookup("s1", n_variable_tokens=10)
        self.assertEqual(result.n_fresh_tokens, 10)

    def test_lookup_n_cached_tokens(self):
        self.cache.register_schema(self._make_schema())
        kv = self.cache.materialize("s1")
        result = self.cache.lookup("s1")
        self.assertEqual(result.n_cached_tokens, kv.shape[0])

    def test_evict_removes_kv(self):
        self.cache.register_schema(self._make_schema())
        self.cache.materialize("s1")
        self.assertEqual(self.cache.n_materialized, 1)
        self.cache.evict("s1")
        self.assertEqual(self.cache.n_materialized, 0)

    def test_evict_unknown_no_error(self):
        self.cache.evict("ghost")  # should not raise

    def test_evict_then_lookup_miss(self):
        self.cache.register_schema(self._make_schema())
        self.cache.materialize("s1")
        self.cache.evict("s1")
        result = self.cache.lookup("s1")
        self.assertFalse(result.hit)


# ---------------------------------------------------------------------------
# PipeInferConfig
# ---------------------------------------------------------------------------


class TestPipeInferConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PipeInferConfig()
        self.assertEqual(cfg.chunk_size, 128)
        self.assertEqual(cfg.max_decode_steps, 512)
        self.assertEqual(cfg.seed, 0)

    def test_chunk_size_zero_raises(self):
        with self.assertRaises(ValueError):
            PipeInferConfig(chunk_size=0)

    def test_max_decode_steps_zero_raises(self):
        with self.assertRaises(ValueError):
            PipeInferConfig(max_decode_steps=0)

    def test_valid_custom(self):
        cfg = PipeInferConfig(chunk_size=64, max_decode_steps=256)
        self.assertEqual(cfg.chunk_size, 64)


# ---------------------------------------------------------------------------
# PipeInferRequest
# ---------------------------------------------------------------------------


class TestPipeInferRequest(unittest.TestCase):
    def test_basic(self):
        req = PipeInferRequest(request_id="r1", prompt_tokens=256)
        self.assertEqual(req.request_id, "r1")
        self.assertEqual(req.prompt_tokens, 256)
        self.assertEqual(req.n_decode_tokens, 1)

    def test_negative_prompt_tokens_raises(self):
        with self.assertRaises(ValueError):
            PipeInferRequest(request_id="r1", prompt_tokens=-1)

    def test_zero_decode_tokens_raises(self):
        with self.assertRaises(ValueError):
            PipeInferRequest(request_id="r1", prompt_tokens=10, n_decode_tokens=0)


# ---------------------------------------------------------------------------
# PipeInferScheduler
# ---------------------------------------------------------------------------


class TestPipeInferScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = PipeInferConfig(chunk_size=4, max_decode_steps=512, seed=0)
        self.sched = PipeInferScheduler(self.cfg)

    def test_config_property(self):
        self.assertIs(self.sched.config, self.cfg)

    def test_default_init(self):
        s = PipeInferScheduler()
        self.assertIsNotNone(s.config)

    def test_is_done_initially(self):
        self.assertTrue(self.sched.is_done())

    def test_n_active_initially(self):
        self.assertEqual(self.sched.n_active, 0)

    def test_submit_increases_active(self):
        req = PipeInferRequest("r1", prompt_tokens=8)
        self.sched.submit(req)
        self.assertEqual(self.sched.n_active, 1)

    def test_submit_duplicate_raises(self):
        req = PipeInferRequest("r1", prompt_tokens=8)
        self.sched.submit(req)
        with self.assertRaises(ValueError):
            self.sched.submit(PipeInferRequest("r1", prompt_tokens=4))

    def test_is_done_after_submit(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=4))
        self.assertFalse(self.sched.is_done())

    def test_step_returns_ticks(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=8))
        ticks = self.sched.step()
        self.assertIsInstance(ticks, list)
        self.assertGreater(len(ticks), 0)

    def test_step_tick_type(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=4))
        ticks = self.sched.step()
        self.assertIsInstance(ticks[0], PipeInferTick)

    def test_step_tick_request_id(self):
        self.sched.submit(PipeInferRequest("my_req", prompt_tokens=4))
        ticks = self.sched.step()
        self.assertEqual(ticks[0].request_id, "my_req")

    def test_step_empty_queue_returns_empty(self):
        ticks = self.sched.step()
        self.assertEqual(ticks, [])

    def test_first_token_emitted(self):
        # chunk_size=4, prompt=4 → first chunk done in 1 step → decode starts
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=4, n_decode_tokens=1))
        ticks = self.sched.step()
        # At least one tick should have first_token_emitted = True
        emitted = [t for t in ticks if t.first_token_emitted]
        self.assertGreater(len(emitted), 0)

    def test_request_eventually_done(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=16, n_decode_tokens=2))
        for _ in range(30):  # more than enough steps
            self.sched.step()
            if self.sched.is_done():
                break
        self.assertTrue(self.sched.is_done())

    def test_multiple_requests(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=4))
        self.sched.submit(PipeInferRequest("r2", prompt_tokens=8))
        ticks = self.sched.step()
        ids = {t.request_id for t in ticks}
        self.assertIn("r1", ids)
        self.assertIn("r2", ids)

    def test_ttft_estimate_small_prompt(self):
        # prompt fits in one chunk → no improvement
        val = self.sched.ttft_estimate(4)
        self.assertAlmostEqual(val, 1.0)

    def test_ttft_estimate_large_prompt(self):
        # prompt = 2 chunks → TTFT = 0.5
        val = self.sched.ttft_estimate(8)
        self.assertAlmostEqual(val, 0.5)

    def test_ttft_estimate_very_large_prompt(self):
        val = self.sched.ttft_estimate(128)
        self.assertLess(val, 1.0)
        self.assertGreater(val, 0.0)

    def test_ttft_estimate_at_chunk_boundary(self):
        val = self.sched.ttft_estimate(self.cfg.chunk_size)
        self.assertAlmostEqual(val, 1.0)

    def test_n_prefill_tokens_bounded_by_chunk(self):
        self.sched.submit(PipeInferRequest("r1", prompt_tokens=20))
        ticks = self.sched.step()
        for t in ticks:
            self.assertLessEqual(t.n_prefill_tokens, self.cfg.chunk_size)


# ---------------------------------------------------------------------------
# PrepackConfig
# ---------------------------------------------------------------------------


class TestPrepackConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PrepackConfig()
        self.assertEqual(cfg.max_batch_size, 8)
        self.assertEqual(cfg.chunk_size, 128)
        self.assertEqual(cfg.seed, 0)

    def test_max_batch_size_zero_raises(self):
        with self.assertRaises(ValueError):
            PrepackConfig(max_batch_size=0)

    def test_chunk_size_zero_raises(self):
        with self.assertRaises(ValueError):
            PrepackConfig(chunk_size=0)

    def test_valid_custom(self):
        cfg = PrepackConfig(max_batch_size=4, chunk_size=64)
        self.assertEqual(cfg.max_batch_size, 4)


# ---------------------------------------------------------------------------
# PrepackRequest
# ---------------------------------------------------------------------------


class TestPrepackRequest(unittest.TestCase):
    def test_basic(self):
        req = PrepackRequest(request_id="r1", prompt_length=128)
        self.assertEqual(req.request_id, "r1")
        self.assertEqual(req.prompt_length, 128)
        self.assertAlmostEqual(req.arrival_time, 0.0)

    def test_negative_prompt_length_raises(self):
        with self.assertRaises(ValueError):
            PrepackRequest(request_id="r1", prompt_length=-1)

    def test_arrival_time_custom(self):
        req = PrepackRequest("r1", prompt_length=64, arrival_time=3.14)
        self.assertAlmostEqual(req.arrival_time, 3.14)


# ---------------------------------------------------------------------------
# PrepackScheduler
# ---------------------------------------------------------------------------


class TestPrepackScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = PrepackConfig(max_batch_size=3, chunk_size=64, seed=0)
        self.sched = PrepackScheduler(self.cfg)

    def test_config_property(self):
        self.assertIs(self.sched.config, self.cfg)

    def test_default_init(self):
        s = PrepackScheduler()
        self.assertIsNotNone(s.config)

    def test_n_pending_initial(self):
        self.assertEqual(self.sched.n_pending, 0)

    def test_submit_increases_pending(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=100))
        self.assertEqual(self.sched.n_pending, 1)

    def test_submit_duplicate_raises(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=100))
        with self.assertRaises(ValueError):
            self.sched.submit(PrepackRequest("r1", prompt_length=50))

    def test_schedule_empty_raises(self):
        with self.assertRaises(RuntimeError):
            self.sched.schedule()

    def test_schedule_returns_batch(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=100))
        batch = self.sched.schedule()
        self.assertIsInstance(batch, PrepackBatch)

    def test_schedule_reduces_pending(self):
        for i in range(5):
            self.sched.submit(PrepackRequest(f"r{i}", prompt_length=100 + i))
        self.sched.schedule()
        self.assertEqual(self.sched.n_pending, 2)  # 5 - max_batch_size(3)

    def test_schedule_shortest_first(self):
        self.sched.submit(PrepackRequest("long", prompt_length=500))
        self.sched.submit(PrepackRequest("short", prompt_length=10))
        self.sched.submit(PrepackRequest("medium", prompt_length=100))
        batch = self.sched.schedule()
        lengths = [r.prompt_length for r in batch.requests]
        self.assertEqual(lengths, sorted(lengths))

    def test_schedule_total_tokens(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=100))
        self.sched.submit(PrepackRequest("r2", prompt_length=200))
        batch = self.sched.schedule()
        self.assertEqual(batch.total_prefill_tokens, 300)

    def test_schedule_estimated_ttft_positive(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=128))
        batch = self.sched.schedule()
        self.assertGreater(batch.estimated_ttft, 0.0)

    def test_schedule_estimated_ttft_max_length(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=64))
        self.sched.submit(PrepackRequest("r2", prompt_length=128))
        batch = self.sched.schedule()
        # estimated_ttft = max(128) / chunk_size(64) = 2.0
        self.assertAlmostEqual(batch.estimated_ttft, 2.0)

    def test_drain_empty_returns_empty(self):
        batches = self.sched.drain()
        self.assertEqual(batches, [])

    def test_drain_returns_all_batches(self):
        for i in range(7):
            self.sched.submit(PrepackRequest(f"r{i}", prompt_length=10 + i))
        batches = self.sched.drain()
        # 7 requests / max_batch_size=3 → ceil(7/3) = 3 batches
        self.assertEqual(len(batches), 3)

    def test_drain_empties_queue(self):
        for i in range(5):
            self.sched.submit(PrepackRequest(f"r{i}", prompt_length=50 + i))
        self.sched.drain()
        self.assertEqual(self.sched.n_pending, 0)

    def test_drain_batches_are_batch_type(self):
        self.sched.submit(PrepackRequest("r1", prompt_length=100))
        batches = self.sched.drain()
        for b in batches:
            self.assertIsInstance(b, PrepackBatch)

    def test_drain_all_requests_included(self):
        submitted_ids = {f"r{i}" for i in range(6)}
        for rid in submitted_ids:
            self.sched.submit(PrepackRequest(rid, prompt_length=50))
        batches = self.sched.drain()
        scheduled_ids = {r.request_id for b in batches for r in b.requests}
        self.assertEqual(scheduled_ids, submitted_ids)

    def test_schedule_max_batch_size_respected(self):
        for i in range(10):
            self.sched.submit(PrepackRequest(f"r{i}", prompt_length=10 + i))
        batch = self.sched.schedule()
        self.assertLessEqual(len(batch.requests), self.cfg.max_batch_size)

    def test_single_request_drain(self):
        self.sched.submit(PrepackRequest("only", prompt_length=64))
        batches = self.sched.drain()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].requests[0].request_id, "only")


if __name__ == "__main__":
    unittest.main()
