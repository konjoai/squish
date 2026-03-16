"""tests/test_wave20_server_wiring.py

Verifies that all Wave 20 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 20 modules (Model Merging + Composability + Batching + Calibration):
  model_merge, lora_compose, continuous_batching, matryoshka_emb,
  ane_profiler, spec_bench, ppl_tracker, grammar_cache,
  quant_aware, adaptive_budget, vision_tokens, tool_cache,
  distil_spec, batch_embed
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ModelMerge
# ---------------------------------------------------------------------------


class TestModelMergeWiring:
    def test_import(self):
        from squish.lora.model_merge import MergeConfig, ModelMerger

        cfg    = MergeConfig(method="slerp", t=0.5)
        merger = ModelMerger(cfg)
        assert merger is not None

    def test_config_defaults(self):
        from squish.lora.model_merge import MergeConfig

        cfg = MergeConfig()
        assert hasattr(cfg, "method")
        assert hasattr(cfg, "t")
        assert cfg.method in ("slerp", "dare", "ties")
        assert 0.0 <= cfg.t <= 1.0

    def test_slerp_function(self):
        from squish.lora.model_merge import slerp

        rng = np.random.default_rng(0)
        a   = rng.standard_normal((8, 4)).astype(np.float32)
        b   = rng.standard_normal((8, 4)).astype(np.float32)
        out = slerp(a, b, t=0.5)
        assert out.shape == a.shape

    def test_merge_returns_array(self):
        from squish.lora.model_merge import MergeConfig, ModelMerger

        rng    = np.random.default_rng(1)
        cfg    = MergeConfig(method="slerp", t=0.5)
        merger = ModelMerger(cfg)
        w1     = {"w": rng.standard_normal((4, 4)).astype(np.float32)}
        w2     = {"w": rng.standard_normal((4, 4)).astype(np.float32)}
        result = merger.merge(w1, w2)
        assert isinstance(result, dict)
        assert "w" in result
        assert isinstance(result["w"], np.ndarray)


# ---------------------------------------------------------------------------
# LoRACompose
# ---------------------------------------------------------------------------


class TestLoRAComposeWiring:
    def test_import(self):
        from squish.lora.lora_compose import AdapterConfig, LoRAComposer

        cfg     = AdapterConfig(rank=4, alpha=8.0, hidden_dim=16)
        composer = LoRAComposer(hidden_dim=16)
        assert cfg is not None
        assert composer is not None

    def test_adapter_stack(self):
        from squish.lora.lora_compose import AdapterStack

        rng   = np.random.default_rng(0)
        A     = rng.standard_normal((16, 4)).astype(np.float32)
        B     = rng.standard_normal((4, 16)).astype(np.float32)
        stack1 = AdapterStack(name="adapter1", A=A,       B=B,       scaling=1.0)
        stack2 = AdapterStack(name="adapter2", A=A.copy(), B=B.copy(), scaling=0.5)
        assert stack1.name == "adapter1"
        assert stack2.name == "adapter2"
        assert stack1.n_params == A.size + B.size

    def test_compose_forward(self):
        from squish.lora.lora_compose import LoRAComposer

        rng    = np.random.default_rng(2)
        hidden = 16
        rank   = 4
        composer = LoRAComposer(hidden_dim=hidden)
        A1 = rng.standard_normal((hidden, rank)).astype(np.float32)
        B1 = rng.standard_normal((rank, hidden)).astype(np.float32)
        A2 = rng.standard_normal((hidden, rank)).astype(np.float32)
        B2 = rng.standard_normal((rank, hidden)).astype(np.float32)
        composer.add_adapter("a1", A1, B1, scale=0.6)
        composer.add_adapter("a2", A2, B2, scale=0.4)
        x   = rng.standard_normal((2, hidden)).astype(np.float32)
        out = composer.forward(x, weights={"a1": 0.6, "a2": 0.4})
        assert out.shape == (2, hidden)

    def test_composition_stats(self):
        from squish.lora.lora_compose import LoRAComposer

        rng    = np.random.default_rng(3)
        hidden = 8
        rank   = 2
        composer = LoRAComposer(hidden_dim=hidden)
        A = rng.standard_normal((hidden, rank)).astype(np.float32)
        B = rng.standard_normal((rank, hidden)).astype(np.float32)
        composer.add_adapter("test", A, B, scale=1.0)
        x = rng.standard_normal((1, hidden)).astype(np.float32)
        composer.forward(x)
        stats = composer.composition_stats()
        assert stats.n_forward_calls == 1
        assert stats.adapters_used_total >= 1


# ---------------------------------------------------------------------------
# ContinuousBatching
# ---------------------------------------------------------------------------


class TestContinuousBatchingWiring:
    def test_import(self):
        from squish.serving.continuous_batching import CBConfig, CBScheduler

        cfg   = CBConfig(max_batch_size=4, max_seq_len=512)
        sched = CBScheduler(cfg)
        assert sched is not None

    def test_request_state_enum(self):
        from squish.serving.continuous_batching import RequestState

        # Module uses WAITING, RUNNING, FINISHED (no PENDING state)
        assert RequestState.WAITING  is not None
        assert RequestState.RUNNING  is not None
        assert RequestState.FINISHED is not None

    def test_submit_and_step(self):
        from squish.serving.continuous_batching import (
            CBConfig,
            CBScheduler,
            InFlightRequest,
            RequestState,
        )

        cfg   = CBConfig(max_batch_size=2, max_seq_len=128)
        sched = CBScheduler(cfg)
        req   = InFlightRequest(
            request_id="r1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=4,
        )
        assert req.state == RequestState.WAITING
        sched.submit(req)
        batch = sched.step_batch()
        assert len(batch) >= 1
        assert batch[0].state == RequestState.RUNNING

    def test_stats(self):
        from squish.serving.continuous_batching import CBConfig, CBScheduler, InFlightRequest

        cfg   = CBConfig(max_batch_size=4, max_seq_len=256)
        sched = CBScheduler(cfg)
        req   = InFlightRequest(request_id="r1", prompt_tokens=[1], max_new_tokens=2)
        sched.submit(req)
        sched.step_batch()
        sched.complete_token("r1", 42)
        stats = sched.scheduler_stats()
        assert stats.total_submitted == 1
        assert stats.total_tokens_generated == 1


# ---------------------------------------------------------------------------
# MatryoshkaEmb
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbWiring:
    def test_import(self):
        from squish.lora.matryoshka_emb import MatryoshkaEmbedding, MRLConfig

        cfg = MRLConfig(full_dim=256, nested_dims=[64, 128, 256])
        emb = MatryoshkaEmbedding(cfg)
        assert emb is not None

    def test_config_dims(self):
        from squish.lora.matryoshka_emb import MRLConfig

        cfg  = MRLConfig(full_dim=256, nested_dims=[64, 128, 256])
        dims = cfg.nested_dims
        assert dims == sorted(dims)
        assert all(d <= cfg.full_dim for d in dims)

    def test_embed_shape(self):
        from squish.lora.matryoshka_emb import MatryoshkaEmbedding, MRLConfig

        rng      = np.random.default_rng(0)
        full_dim = 128
        cfg      = MRLConfig(full_dim=full_dim, nested_dims=[32, 64, 128])
        emb      = MatryoshkaEmbedding(cfg)
        x        = rng.standard_normal(full_dim).astype(np.float32)
        out      = emb.embed(x)          # default target_dim == full_dim
        assert out.shape == (full_dim,)

    def test_truncate(self):
        from squish.lora.matryoshka_emb import MatryoshkaEmbedding, MRLConfig

        rng      = np.random.default_rng(1)
        full_dim = 128
        cfg      = MRLConfig(full_dim=full_dim, nested_dims=[32, 64, 128])
        emb      = MatryoshkaEmbedding(cfg)
        x        = rng.standard_normal(full_dim).astype(np.float32)
        truncated = emb.embed(x, target_dim=32)
        assert truncated.shape == (32,)


# ---------------------------------------------------------------------------
# ANEProfiler
# ---------------------------------------------------------------------------


class TestANEProfilerWiring:
    def test_import(self):
        from squish.hardware.ane_profiler import ANEProfiler

        profiler = ANEProfiler()
        assert profiler is not None

    def test_op_device_constants(self):
        from squish.hardware.ane_profiler import OpDevice

        assert OpDevice.ANE == "ane"
        assert OpDevice.GPU == "gpu"
        assert OpDevice.CPU == "cpu"

    def test_record_op(self):
        from squish.hardware.ane_profiler import ANEProfiler

        profiler = ANEProfiler()
        profiler.record_op("matmul", shape=(1024, 1024), dtype="float16", latency_us=100.0)
        assert profiler.n_ops == 1

    def test_summary_metrics(self):
        from squish.hardware.ane_profiler import ANEMetrics, ANEProfiler

        # threshold=64 so 128*128=16384 > 64 → ANE; 4*4=16 <= 64 → GPU
        profiler = ANEProfiler(ane_threshold_elements=64)
        profiler.record_op("big_op",   shape=(128, 128), dtype="float16", latency_us=50.0)
        profiler.record_op("small_op", shape=(4, 4),     dtype="float16", latency_us=10.0)
        metrics = profiler.summary()
        assert isinstance(metrics, ANEMetrics)
        assert metrics.total_ops == 2
        assert 0.0 <= metrics.ane_fraction <= 1.0


# ---------------------------------------------------------------------------
# SpecBench
# ---------------------------------------------------------------------------


class TestSpecBenchWiring:
    def test_import(self):
        from squish.speculative.spec_bench import SpecBenchRunner

        runner = SpecBenchRunner(gamma=4)
        assert runner is not None

    def test_default_tasks(self):
        from squish.speculative.spec_bench import SpecBenchRunner, SpecBenchTask

        tasks = SpecBenchRunner.default_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) >= 1
        assert all(isinstance(t, SpecBenchTask) for t in tasks)

    def test_run_single_task(self):
        from squish.speculative.spec_bench import SpecBenchResult, SpecBenchRunner, SpecBenchTask

        runner = SpecBenchRunner(gamma=2)
        task   = SpecBenchTask("qa", prompts=["What is 2+2?", "Define AI."])

        def draft_fn(prompt):
            return [42, 7]

        def target_fn(prompt, draft_tokens):
            return [True, False]

        result = runner.run_task(task, draft_fn, target_fn)
        assert isinstance(result, SpecBenchResult)
        assert result.task_name == "qa"
        assert result.n_prompts == 2

    def test_stats(self):
        from squish.speculative.spec_bench import SpecBenchRunner, SpecBenchStats, SpecBenchTask

        runner = SpecBenchRunner(gamma=2)
        task   = SpecBenchTask("math", prompts=["Compute 3+4."])

        def draft_fn(prompt):
            return [1, 2]

        def target_fn(prompt, draft_tokens):
            return [True, True]

        results = {"math": runner.run_task(task, draft_fn, target_fn)}
        stats   = runner.suite_stats(results)
        assert isinstance(stats, SpecBenchStats)
        assert stats.tasks_run == 1
        assert 0.0 <= stats.overall_acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# PPLTracker
# ---------------------------------------------------------------------------


class TestPPLTrackerWiring:
    def test_import(self):
        from squish.serving.ppl_tracker import PPLTracker

        tracker = PPLTracker(window_size=50, alert_threshold=2.0)
        assert tracker is not None

    def test_config_defaults(self):
        from squish.serving.ppl_tracker import PPLTracker, PPLWindow

        # PPLWindow exposes window_size as a public dataclass field
        window = PPLWindow(window_size=100)
        assert window.window_size == 100

        # PPLTracker enforces alert_threshold > 1.0 in __post_init__
        tracker = PPLTracker(window_size=10, alert_threshold=1.5)
        assert tracker._window.window_size == 10
        assert tracker._alert_threshold > 1.0

    def test_track_and_ppl(self):
        from squish.serving.ppl_tracker import PPLTracker

        rng     = np.random.default_rng(0)
        tracker = PPLTracker(window_size=10, alert_threshold=2.0)
        vocab   = 32
        seq     = 4
        logits     = rng.standard_normal((seq, vocab)).astype(np.float32)
        target_ids = rng.integers(0, vocab, size=seq)
        tracker.record(logits, target_ids)
        ppl = tracker.rolling_ppl
        assert not np.isnan(ppl)
        assert ppl > 0.0

    def test_alert_triggered(self):
        from squish.serving.ppl_tracker import PPLTracker

        rng = np.random.default_rng(1)
        # baseline_ppl=2.0, threshold=1.2 → alert fires when rolling_ppl > 2.4
        # Uniform logits (all 0) → PPL = vocab_size = 16 >> 2.4
        tracker = PPLTracker(window_size=5, alert_threshold=1.2, baseline_ppl=2.0)
        vocab   = 16
        seq     = 3
        for _ in range(5):
            logits     = np.zeros((seq, vocab), dtype=np.float32)
            target_ids = rng.integers(0, vocab, size=seq)
            tracker.record(logits, target_ids)
        assert len(tracker.alerts) > 0


# ---------------------------------------------------------------------------
# GrammarCache
# ---------------------------------------------------------------------------


class TestGrammarCacheWiring:
    def test_import(self):
        from squish.grammar.grammar_cache import GrammarCache

        cache = GrammarCache(vocab_size=1000)
        assert cache is not None

    def test_fsm_state(self):
        from squish.grammar.grammar_cache import FSMState

        state = FSMState(state_id=0, pattern_name="json")
        assert hasattr(state, "state_id")
        assert state.state_id == 0

    def test_register_grammar(self):
        from squish.grammar.grammar_cache import FSMState, GrammarCache

        cache = GrammarCache(vocab_size=100)
        cache.add_pattern("json", r"^\{")
        state = FSMState(state_id=0, pattern_name="json")
        mask  = cache.get_mask(state)
        assert mask is not None
        assert cache.n_states_cached >= 1

    def test_get_allowed_tokens(self):
        from squish.grammar.grammar_cache import FSMState, GrammarCache

        cache = GrammarCache(vocab_size=100)
        cache.add_pattern("regex1", r"[a-z]+")
        state = FSMState(state_id=1, pattern_name="regex1")
        mask  = cache.get_mask(state)
        assert mask.shape == (100,)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# QuantAware
# ---------------------------------------------------------------------------


class TestQuantAwareWiring:
    def test_import(self):
        from squish.quant.quant_aware import QAConfig, QuantAwareCalibrator

        cfg        = QAConfig(method="percentile", percentile=99.0, n_bits=8)
        calibrator = QuantAwareCalibrator(cfg)
        assert calibrator is not None

    def test_config_defaults(self):
        from squish.quant.quant_aware import QAConfig

        cfg = QAConfig()
        assert hasattr(cfg, "method")
        assert hasattr(cfg, "percentile")
        assert cfg.method in ("minmax", "percentile", "mse")
        assert 0.0 < cfg.percentile <= 100.0

    def test_calibrate_returns_scales(self):
        from squish.quant.quant_aware import QAConfig, QuantAwareCalibrator

        rng        = np.random.default_rng(0)
        cfg        = QAConfig(method="percentile", percentile=99.0, n_bits=8)
        calibrator = QuantAwareCalibrator(cfg)
        activations = rng.standard_normal((16, 8)).astype(np.float32)
        calibrator.record(activations)
        scales = calibrator.compute_scales()
        assert scales.dtype == np.float32
        assert scales.shape == (8,)
        assert np.all(scales > 0.0)

    def test_stats(self):
        from squish.quant.quant_aware import QAConfig, QuantAwareCalibrator

        rng        = np.random.default_rng(1)
        cfg        = QAConfig(method="minmax", n_bits=8)
        calibrator = QuantAwareCalibrator(cfg)
        activations = rng.standard_normal((8, 4)).astype(np.float32)
        calibrator.record(activations)
        stats = calibrator.stats()
        assert stats.n_batches > 0


# ---------------------------------------------------------------------------
# AdaptiveBudget
# ---------------------------------------------------------------------------


class TestAdaptiveBudgetWiring:
    def test_import(self):
        from squish.serving.adaptive_budget import AdaptiveBudgetController, BudgetConfig

        cfg  = BudgetConfig(target_latency_ms=100.0, kv_budget_min=256, kv_budget_max=2048)
        ctrl = AdaptiveBudgetController(cfg)
        assert ctrl is not None

    def test_config_defaults(self):
        from squish.serving.adaptive_budget import BudgetConfig

        cfg = BudgetConfig()
        assert hasattr(cfg, "target_latency_ms")
        assert hasattr(cfg, "kv_budget_max")    # spec calls this max_kv_budget
        assert cfg.target_latency_ms > 0.0
        assert cfg.kv_budget_max > cfg.kv_budget_min

    def test_update_returns_budget_state(self):
        from squish.serving.adaptive_budget import (
            AdaptiveBudgetController,
            BudgetConfig,
            BudgetState,
        )

        cfg   = BudgetConfig(target_latency_ms=100.0, kv_budget_min=256, kv_budget_max=2048)
        ctrl  = AdaptiveBudgetController(cfg)
        state = ctrl.step(observed_latency_ms=150.0)
        assert isinstance(state, BudgetState)
        assert state.kv_tokens >= 1
        assert 0.0 <= state.skip_fraction <= 1.0

    def test_stats(self):
        from squish.serving.adaptive_budget import AdaptiveBudgetController, BudgetConfig

        cfg  = BudgetConfig(target_latency_ms=100.0, kv_budget_min=256, kv_budget_max=2048)
        ctrl = AdaptiveBudgetController(cfg)
        ctrl.step(observed_latency_ms=80.0)
        ctrl.step(observed_latency_ms=120.0)
        stats = ctrl.stats()
        assert stats.n_steps > 0


# ---------------------------------------------------------------------------
# ToolCache
# ---------------------------------------------------------------------------


class TestToolCacheWiring:
    def test_import(self):
        from squish.kv.tool_cache import ToolRouter, ToolSchemaCache

        cache  = ToolSchemaCache()
        router = ToolRouter(cache)
        assert cache  is not None
        assert router is not None

    def test_register_and_lookup(self):
        from squish.kv.tool_cache import ToolSchemaCache

        cache = ToolSchemaCache()
        schema_dict = {
            "name":        "get_weather",
            "parameters":  {"city": "string", "unit": "string"},
            "description": "Get current weather.",
        }
        schema_id = cache.register(schema_dict)
        assert isinstance(schema_id, str)
        assert len(schema_id) == 16
        schema = cache.get_by_hash(schema_id)
        assert schema is not None
        assert schema.name == "get_weather"

    def test_hash_dedup(self):
        from squish.kv.tool_cache import ToolSchemaCache

        cache = ToolSchemaCache()
        schema_dict = {
            "name":       "search",
            "parameters": {"query": "string"},
        }
        id1 = cache.register(schema_dict)
        id2 = cache.register(schema_dict)
        assert id1 == id2
        assert cache.n_cached == 1

    def test_route(self):
        from squish.kv.tool_cache import ToolRouter, ToolSchemaCache

        cache = ToolSchemaCache()
        cache.register({"name": "add", "parameters": {"a": "int", "b": "int"}})
        router = ToolRouter(cache)
        result = router.route(
            "add",
            {"a": 1, "b": 2},
            handlers={"add": lambda args: args["a"] + args["b"]},
        )
        assert result == 3


# ---------------------------------------------------------------------------
# DistilSpec
# ---------------------------------------------------------------------------


class TestDistilSpecWiring:
    def test_import(self):
        from squish.speculative.distil_spec import DistilConfig, DistilSpecCalibrator

        cfg        = DistilConfig(n_calibration_steps=10, temperature=2.0)
        calibrator = DistilSpecCalibrator(cfg)
        assert calibrator is not None

    def test_config_defaults(self):
        from squish.speculative.distil_spec import DistilConfig

        cfg = DistilConfig()
        assert hasattr(cfg, "temperature")
        assert hasattr(cfg, "n_calibration_steps")
        assert cfg.temperature > 0.0
        assert cfg.n_calibration_steps >= 1

    def test_calibrate_step(self):
        from squish.speculative.distil_spec import DistilConfig, DistilSpecCalibrator

        rng        = np.random.default_rng(0)
        vocab      = 32
        cfg        = DistilConfig(n_calibration_steps=5, temperature=1.5)
        calibrator = DistilSpecCalibrator(cfg)
        draft_logits  = rng.standard_normal(vocab).astype(np.float32)
        target_logits = rng.standard_normal(vocab).astype(np.float32)
        calibrator.record_step(draft_logits, target_logits)
        assert calibrator.n_steps == 1

    def test_acceptance_gain(self):
        from squish.speculative.distil_spec import DistilConfig, DistilSpecCalibrator

        rng        = np.random.default_rng(2)
        vocab      = 16
        cfg        = DistilConfig(n_calibration_steps=5, temperature=2.0)
        calibrator = DistilSpecCalibrator(cfg)
        for _ in range(3):
            draft  = rng.standard_normal(vocab).astype(np.float32)
            target = rng.standard_normal(vocab).astype(np.float32)
            calibrator.record_step(draft, target)
        gain = calibrator.acceptance_improvement_estimate()
        assert isinstance(gain, float)


# ---------------------------------------------------------------------------
# BatchEmbed
# ---------------------------------------------------------------------------


class TestBatchEmbedWiring:
    def test_import(self):
        from squish.serving.batch_embed import BatchEmbedder, PoolingConfig

        cfg      = PoolingConfig(strategy="mean", hidden_dim=64, normalize=True)
        embedder = BatchEmbedder(cfg)
        assert embedder is not None

    def test_config_defaults(self):
        from squish.serving.batch_embed import PoolingConfig

        cfg = PoolingConfig()
        # The module calls the field 'strategy' (equivalent to 'method' in spec)
        assert hasattr(cfg, "strategy")
        assert hasattr(cfg, "normalize")
        assert cfg.strategy in ("mean", "max", "cls", "weighted")
        assert isinstance(cfg.normalize, bool)

    def test_embed_mean_pooling(self):
        from squish.serving.batch_embed import BatchEmbedder, PoolingConfig

        rng    = np.random.default_rng(0)
        hidden = 32
        cfg    = PoolingConfig(strategy="mean", hidden_dim=hidden, normalize=False)
        embedder = BatchEmbedder(cfg)
        # shape: (batch=4, seq_len=8, hidden_dim=32)
        token_embeddings = rng.standard_normal((4, 8, hidden)).astype(np.float32)
        out = embedder.pool(token_embeddings)
        assert out.shape == (4, hidden)

    def test_all_pooling_methods(self):
        from squish.serving.batch_embed import BatchEmbedder, PoolingConfig

        rng    = np.random.default_rng(1)
        hidden = 16
        seq    = 5
        hs     = rng.standard_normal((seq, hidden)).astype(np.float32)

        for strategy in ("mean", "max", "cls", "weighted"):
            cfg      = PoolingConfig(strategy=strategy, hidden_dim=hidden, normalize=False)
            embedder = BatchEmbedder(cfg)
            out      = embedder.pool_single(hs)
            assert out.shape == (hidden,), (
                f"strategy={strategy!r} produced shape {out.shape}"
            )
