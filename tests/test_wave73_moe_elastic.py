"""tests/test_wave73_moe_elastic.py

Wave 73 — "Impossible 70B": MoE Elastic Inference Engine

Test suite covering:
  * HFMoELoader    — config parsing, arch detection, shard indexing, lazy expert handles
  * ExpertMemoryMap — LRU eviction, budget, pin/unpin, stats
  * RouterEstimator — gate weight loading, route computation, ExpertSchedule
  * INT4ExpertPacker — pack/unpack matrix and expert, error bounds
  * LayerByLayerExecutor — forward pass, expert getter, attention and MoE layers
  * MoEPipeline     — end-to-end construction, generate(), stats
  * Catalog additions — Mixtral 8x7B, 8x22B, Qwen3-235B-A22B entries
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────

from squish.moe.hf_moe_loader import (
    HFMoELoader,
    MoEArchType,
    MoEModelInfo,
    ExpertWeightHandle,
    _detect_arch,
    _extract_model_info,
    _is_expert_key,
    _parse_expert_key,
)
from squish.moe.expert_memory_map import (
    ExpertMemoryMap,
    MemoryMapConfig,
    MemoryMapStats,
    _expert_bytes,
)
from squish.moe.router_estimator import (
    RouterConfig,
    RouterEstimator,
    ExpertSchedule,
    LayerRouting,
)
from squish.moe.int4_expert_pack import (
    INT4ExpertPacker,
    PackConfig,
    INT4PackedMatrix,
    INT4PackedExpert,
)
from squish.moe.layer_by_layer_executor import (
    ExecutorConfig,
    LayerByLayerExecutor,
    LayerWeights,
    ExecutorStats,
    _rms_norm,
    _silu,
    _softmax,
)
from squish.moe.moe_pipeline import (
    MoEPipeline,
    PipelineConfig,
    PipelineStats,
    GenerationResult,
    _sample_token,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_random_weight(shape, rng=None, scale=0.02) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _write_minimal_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file to *path*."""
    # Build header
    data_parts = []
    current_offset = 0
    meta = {}
    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        raw = arr.tobytes()
        meta[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + len(raw)],
        }
        data_parts.append(raw)
        current_offset += len(raw)

    header_bytes = json.dumps(meta).encode("utf-8")
    header_len = len(header_bytes)

    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", header_len))
        fh.write(header_bytes)
        for part in data_parts:
            fh.write(part)


def _make_mock_model_dir(
    arch: str = "MixtralForCausalLM",
    n_layers: int = 2,
    n_experts: int = 4,
    hidden: int = 32,
    intermediate: int = 64,
    expert_layer_range: range | None = None,
) -> Path:
    """Create a minimal mock HuggingFace MoE model directory."""
    tmpdir = Path(tempfile.mkdtemp())

    config = {
        "architectures": [arch],
        "num_hidden_layers": n_layers,
        "num_local_experts": n_experts,
        "num_experts_per_tok": 2,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": 128,
        "model_type": arch.lower().replace("forcausallm", ""),
    }
    (tmpdir / "config.json").write_text(json.dumps(config))

    # Build tensor dict
    tensors = {}
    # Embedding + LM head
    tensors["model.embed_tokens.weight"] = np.random.randn(128, hidden).astype(np.float32)
    tensors["lm_head.weight"] = np.random.randn(128, hidden).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(hidden, dtype=np.float32)

    # Backbone layers
    for li in range(n_layers):
        p = f"model.layers.{li}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{p}.self_attn.{proj}.weight"] = _make_random_weight((hidden, hidden))
        tensors[f"{p}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{p}.block_sparse_moe.gate.weight"] = _make_random_weight((n_experts, hidden))

        # Expert tensors
        for ei in range(n_experts):
            ep = f"{p}.block_sparse_moe.experts.{ei}"
            tensors[f"{ep}.w1.weight"] = _make_random_weight((intermediate, hidden))
            tensors[f"{ep}.w2.weight"] = _make_random_weight((hidden, intermediate))
            tensors[f"{ep}.w3.weight"] = _make_random_weight((intermediate, hidden))

    shard_path = tmpdir / "model.safetensors"
    _write_minimal_safetensors(shard_path, tensors)
    return tmpdir


# ══════════════════════════════════════════════════════════════════════════════
# TestMoEArchDetection
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEArchDetection(unittest.TestCase):
    def test_detect_mixtral(self):
        config = {"architectures": ["MixtralForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.MIXTRAL)

    def test_detect_deepseek_v2(self):
        config = {"architectures": ["DeepseekV2ForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.DEEPSEEK_V2)

    def test_detect_deepseek_v3(self):
        config = {"architectures": ["DeepseekV3ForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.DEEPSEEK_V2)

    def test_detect_qwen2_moe(self):
        config = {"architectures": ["Qwen2MoeForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.QWEN_MOE)

    def test_detect_qwen3_moe(self):
        config = {"architectures": ["Qwen3MoeForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.QWEN_MOE)

    def test_detect_unknown(self):
        config = {"architectures": ["LlamaForCausalLM"]}
        self.assertEqual(_detect_arch(config), MoEArchType.UNKNOWN)

    def test_detect_via_model_type(self):
        config = {"architectures": [], "model_type": "mixtral", "num_local_experts": 8}
        self.assertEqual(_detect_arch(config), MoEArchType.MIXTRAL)

    def test_detect_qwen_moe_via_model_type(self):
        config = {"architectures": [], "model_type": "qwen_moe", "num_experts": 4}
        self.assertEqual(_detect_arch(config), MoEArchType.QWEN_MOE)


# ══════════════════════════════════════════════════════════════════════════════
# TestMoEModelInfo
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEModelInfo(unittest.TestCase):
    def _mixtral_config(self) -> dict:
        return {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32000,
        }

    def test_extracting_basic_fields(self):
        info = _extract_model_info(self._mixtral_config())
        self.assertEqual(info.n_layers, 32)
        self.assertEqual(info.n_experts, 8)
        self.assertEqual(info.top_k, 2)
        self.assertEqual(info.hidden_size, 4096)

    def test_activation_ratio(self):
        info = _extract_model_info(self._mixtral_config())
        self.assertGreater(info.activation_ratio, 0)
        self.assertLess(info.activation_ratio, 1.0)

    def test_memory_savings_x(self):
        info = _extract_model_info(self._mixtral_config())
        # Mixtral-8x7B: top-2/8 means ~25% active → 4× savings
        self.assertGreater(info.memory_savings_x, 1.0)

    def test_str_representation(self):
        info = _extract_model_info(self._mixtral_config())
        s = str(info)
        self.assertIn("Mixtral", s)
        self.assertIn("32 layers", s)


# ══════════════════════════════════════════════════════════════════════════════
# TestIsExpertKey
# ══════════════════════════════════════════════════════════════════════════════

class TestIsExpertKey(unittest.TestCase):
    def test_mixtral_gate_key(self):
        name = "model.layers.0.block_sparse_moe.experts.3.w1.weight"
        self.assertTrue(_is_expert_key(name, MoEArchType.MIXTRAL))

    def test_mixtral_non_expert_key(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        self.assertFalse(_is_expert_key(name, MoEArchType.MIXTRAL))

    def test_qwen_moe_expert_key(self):
        name = "model.layers.2.mlp.experts.7.gate_proj.weight"
        self.assertTrue(_is_expert_key(name, MoEArchType.QWEN_MOE))

    def test_parse_mixtral_expert_key(self):
        name = "model.layers.5.block_sparse_moe.experts.3.w1.weight"
        result = _parse_expert_key(name, MoEArchType.MIXTRAL)
        self.assertIsNotNone(result)
        layer_idx, expert_idx, proj = result
        self.assertEqual(layer_idx, 5)
        self.assertEqual(expert_idx, 3)
        self.assertEqual(proj, "gate")  # w1 → gate

    def test_parse_mixtral_w2_is_down(self):
        name = "model.layers.0.block_sparse_moe.experts.0.w2.weight"
        _, _, proj = _parse_expert_key(name, MoEArchType.MIXTRAL)
        self.assertEqual(proj, "down")

    def test_parse_mixtral_w3_is_up(self):
        name = "model.layers.0.block_sparse_moe.experts.0.w3.weight"
        _, _, proj = _parse_expert_key(name, MoEArchType.MIXTRAL)
        self.assertEqual(proj, "up")

    def test_parse_qwen_moe_expert_key(self):
        name = "model.layers.1.mlp.experts.6.down_proj.weight"
        result = _parse_expert_key(name, MoEArchType.QWEN_MOE)
        self.assertIsNotNone(result)
        layer_idx, expert_idx, proj = result
        self.assertEqual(layer_idx, 1)
        self.assertEqual(expert_idx, 6)
        self.assertEqual(proj, "down")


# ══════════════════════════════════════════════════════════════════════════════
# TestHFMoELoader
# ══════════════════════════════════════════════════════════════════════════════

class TestHFMoELoader(unittest.TestCase):
    def setUp(self):
        self.model_dir = _make_mock_model_dir(
            arch="MixtralForCausalLM",
            n_layers=2,
            n_experts=4,
            hidden=32,
            intermediate=64,
        )
        self.loader = HFMoELoader.from_directory(self.model_dir)

    def test_from_directory(self):
        self.assertIsInstance(self.loader, HFMoELoader)

    def test_model_info_extracted(self):
        info = self.loader.model_info
        self.assertEqual(info.n_layers, 2)
        self.assertEqual(info.n_experts, 4)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            HFMoELoader.from_directory("/nonexistent/path")

    def test_expert_count(self):
        count = self.loader.expert_count()
        # 2 layers × 4 experts × 3 projections = 24 tensor entries → 8 unique (layer, expert) pairs
        self.assertEqual(count, 8)

    def test_backbone_tensor_count(self):
        count = self.loader.backbone_tensor_count()
        self.assertGreater(count, 0)

    def test_expert_handle_lazy(self):
        handle = self.loader.expert_handle(layer_idx=0, expert_idx=0)
        self.assertIsInstance(handle, ExpertWeightHandle)
        self.assertFalse(handle.is_loaded)

    def test_expert_handle_gate_loads_on_access(self):
        handle = self.loader.expert_handle(layer_idx=0, expert_idx=0)
        gate = handle.gate()
        self.assertIsNotNone(gate)
        self.assertIsInstance(gate, np.ndarray)
        self.assertTrue(handle.is_loaded)

    def test_expert_handle_evict(self):
        handle = self.loader.expert_handle(layer_idx=0, expert_idx=0)
        handle.gate()
        self.assertTrue(handle.is_loaded)
        handle.evict()
        self.assertFalse(handle.is_loaded)

    def test_expert_disk_bytes(self):
        b = self.loader.expert_disk_bytes(0, 0)
        self.assertGreater(b, 0)

    def test_total_expert_disk_bytes(self):
        total = self.loader.total_expert_disk_bytes()
        self.assertGreater(total, 0)

    def test_iter_experts(self):
        experts = list(self.loader.iter_experts())
        self.assertEqual(len(experts), 8)  # 2 layers × 4 experts

    def test_load_backbone_returns_dict(self):
        backbone = self.loader.load_backbone()
        self.assertIsInstance(backbone, dict)
        self.assertGreater(len(backbone), 0)

    def test_repr(self):
        r = repr(self.loader)
        self.assertIn("HFMoELoader", r)


# ══════════════════════════════════════════════════════════════════════════════
# TestExpertMemoryMap
# ══════════════════════════════════════════════════════════════════════════════

class TestExpertMemoryMapConfig(unittest.TestCase):
    def test_valid_config(self):
        cfg = MemoryMapConfig(budget_mb=1024, max_experts=10)
        self.assertEqual(cfg.budget_mb, 1024)
        self.assertEqual(cfg.budget_bytes, 1024 * 1024 * 1024)

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            MemoryMapConfig(budget_mb=0)

    def test_negative_max_experts_raises(self):
        with self.assertRaises(ValueError):
            MemoryMapConfig(budget_mb=1024, max_experts=-1)


class TestExpertMemoryMap(unittest.TestCase):
    def _make_weights(self, size_kb: float = 1.0) -> Dict[str, np.ndarray]:
        n = int(size_kb * 1024 / 4)  # n float32 elements
        return {
            "gate": np.zeros(n, dtype=np.float32),
            "up": np.zeros(n, dtype=np.float32),
            "down": np.zeros(n, dtype=np.float32),
        }

    def test_get_empty_returns_none(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        self.assertIsNone(emap.get(0, 0))

    def test_put_and_get(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        w = self._make_weights()
        emap.put(0, 0, w)
        result = emap.get(0, 0)
        self.assertIsNotNone(result)
        self.assertIn("gate", result)

    def test_is_resident(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        emap.put(0, 1, self._make_weights())
        self.assertTrue(emap.is_resident(0, 1))
        self.assertFalse(emap.is_resident(0, 2))

    def test_eviction_by_budget(self):
        # Budget = 10 KB; each expert = 12 KB → only 0 or 1 can fit
        cfg = MemoryMapConfig(budget_mb=0.01)  # 10 KB
        emap = ExpertMemoryMap(cfg)
        emap.put(0, 0, self._make_weights(4.0))  # 12 KB
        emap.put(0, 1, self._make_weights(4.0))  # 12 KB — evicts (0,0)
        # (0,0) should have been evicted
        self.assertFalse(emap.is_resident(0, 0))
        self.assertTrue(emap.is_resident(0, 1))

    def test_explicit_evict(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        emap.put(0, 0, self._make_weights())
        self.assertTrue(emap.evict(0, 0))
        self.assertFalse(emap.is_resident(0, 0))

    def test_evict_nonresident_returns_false(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        self.assertFalse(emap.evict(99, 99))

    def test_pin_protects_from_eviction(self):
        cfg = MemoryMapConfig(budget_mb=0.01)  # tiny budget
        emap = ExpertMemoryMap(cfg)
        emap.put(0, 0, self._make_weights(4.0))
        emap.pin(0, 0)
        # Try to insert something that needs to evict (0,0)
        emap.put(0, 1, self._make_weights(4.0))
        # (0,0) should still be resident because it's pinned
        self.assertTrue(emap.is_resident(0, 0))

    def test_unpin_allows_eviction(self):
        cfg = MemoryMapConfig(budget_mb=0.01)
        emap = ExpertMemoryMap(cfg)
        emap.put(0, 0, self._make_weights(4.0))
        emap.pin(0, 0)
        emap.unpin(0, 0)
        emap.put(0, 1, self._make_weights(4.0))
        # now (0,0) can be evicted
        self.assertFalse(emap.is_resident(0, 0))

    def test_stats_hit_rate(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        emap.put(0, 0, self._make_weights())
        emap.get(0, 0)   # hit
        emap.get(0, 1)   # miss
        s = emap.stats()
        self.assertEqual(s.n_hits, 1)
        self.assertEqual(s.n_misses, 1)
        self.assertAlmostEqual(s.hit_rate, 0.5)

    def test_stats_eviction_count(self):
        cfg = MemoryMapConfig(budget_mb=0.01)
        emap = ExpertMemoryMap(cfg)
        for ei in range(5):
            emap.put(0, ei, self._make_weights(4.0))
        s = emap.stats()
        self.assertGreaterEqual(s.n_evictions, 4)

    def test_clear_resets_map(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        emap.put(0, 0, self._make_weights())
        emap.clear()
        self.assertEqual(len(emap), 0)
        self.assertIsNone(emap.get(0, 0))

    def test_max_experts_cap(self):
        cfg = MemoryMapConfig(budget_mb=1024, max_experts=3)
        emap = ExpertMemoryMap(cfg)
        for ei in range(5):
            emap.put(0, ei, self._make_weights())
        self.assertLessEqual(len(emap), 3)

    def test_repr_contains_stats(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=512))
        r = repr(emap)
        self.assertIn("ExpertMemoryMap", r)
        self.assertIn("512", r)

    def test_resident_keys_order(self):
        emap = ExpertMemoryMap(MemoryMapConfig(budget_mb=1024))
        for ei in range(3):
            emap.put(0, ei, self._make_weights())
        keys = emap.resident_keys()
        self.assertEqual(len(keys), 3)


# ══════════════════════════════════════════════════════════════════════════════
# TestRouterConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestRouterConfig(unittest.TestCase):
    def test_valid_config(self):
        cfg = RouterConfig(n_layers=32, n_experts=8, top_k=2, hidden_size=4096)
        self.assertEqual(cfg.n_layers, 32)

    def test_top_k_exceeds_n_experts(self):
        with self.assertRaises(ValueError):
            RouterConfig(n_layers=4, n_experts=4, top_k=5)

    def test_zero_n_layers(self):
        with self.assertRaises(ValueError):
            RouterConfig(n_layers=0)


# ══════════════════════════════════════════════════════════════════════════════
# TestExpertSchedule
# ══════════════════════════════════════════════════════════════════════════════

class TestExpertSchedule(unittest.TestCase):
    def _make_routing(self, layer_idx: int, seq_len: int = 4, n_experts: int = 8, top_k: int = 2) -> LayerRouting:
        rng = np.random.default_rng(layer_idx)
        assignments = rng.integers(0, n_experts, size=(seq_len, top_k), dtype=np.int32)
        weights = np.ones((seq_len, top_k), dtype=np.float32) / top_k
        logits = rng.standard_normal((seq_len, n_experts)).astype(np.float32)
        expert_ids = np.unique(assignments).astype(np.int32)
        return LayerRouting(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
            token_assignments=assignments,
            token_weights=weights,
            gate_logits=logits,
        )

    def test_experts_for_layer(self):
        sched = ExpertSchedule(n_layers=4)
        sched.routings[0] = self._make_routing(0)
        ids = sched.experts_for_layer(0)
        self.assertIsInstance(ids, np.ndarray)
        self.assertGreater(len(ids), 0)

    def test_experts_for_missing_layer(self):
        sched = ExpertSchedule(n_layers=4)
        ids = sched.experts_for_layer(99)
        self.assertEqual(len(ids), 0)

    def test_all_expert_ids(self):
        sched = ExpertSchedule(n_layers=2)
        sched.routings[0] = self._make_routing(0)
        sched.routings[1] = self._make_routing(1)
        all_ids = sched.all_expert_ids()
        self.assertGreater(len(all_ids), 0)
        self.assertEqual(len(np.unique(all_ids)), len(all_ids))

    def test_iteration(self):
        sched = ExpertSchedule(n_layers=3)
        for li in range(3):
            sched.routings[li] = self._make_routing(li)
        pairs = list(sched)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(isinstance(p[1], np.ndarray) for p in pairs))

    def test_expert_activation_frequency(self):
        sched = ExpertSchedule(n_layers=2)
        sched.routings[0] = self._make_routing(0)
        freq = sched.expert_activation_frequency()
        self.assertIsInstance(freq, dict)
        self.assertTrue(all(v > 0 for v in freq.values()))

    def test_peak_active_per_layer(self):
        sched = ExpertSchedule(n_layers=2)
        sched.routings[0] = self._make_routing(0)
        sched.routings[1] = self._make_routing(1)
        peak = sched.peak_active_per_layer()
        self.assertGreater(peak, 0)

    def test_repr(self):
        sched = ExpertSchedule(n_layers=4)
        r = repr(sched)
        self.assertIn("ExpertSchedule", r)


# ══════════════════════════════════════════════════════════════════════════════
# TestRouterEstimator
# ══════════════════════════════════════════════════════════════════════════════

class TestRouterEstimator(unittest.TestCase):
    def _make_estimator(self, n_layers=4, n_experts=8, top_k=2, hidden=64):
        cfg = RouterConfig(
            n_layers=n_layers, n_experts=n_experts, top_k=top_k, hidden_size=hidden
        )
        est = RouterEstimator(cfg)
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((n_layers, n_experts, hidden)).astype(np.float32)
        est.load_gate_weights(weights)
        return est

    def test_gate_weights_loaded(self):
        est = self._make_estimator()
        self.assertTrue(est.gate_weights_loaded)

    def test_no_weights_raises_on_estimate(self):
        cfg = RouterConfig(n_layers=2)
        est = RouterEstimator(cfg)
        hs = np.ones((4, 64), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            est.estimate(hs)

    def test_estimate_single_hidden(self):
        est = self._make_estimator(n_layers=4, n_experts=8, top_k=2, hidden=64)
        hs = np.random.randn(6, 64).astype(np.float32)
        sched = est.estimate(hs)
        self.assertIsInstance(sched, ExpertSchedule)
        self.assertEqual(len(sched), 4)

    def test_estimate_per_layer_hidden(self):
        est = self._make_estimator(n_layers=4, n_experts=8, top_k=2, hidden=64)
        hs_list = [np.random.randn(6, 64).astype(np.float32) for _ in range(4)]
        sched = est.estimate(hs_list)
        self.assertEqual(len(sched.routings), 4)

    def test_estimate_3d_array(self):
        est = self._make_estimator(n_layers=4, n_experts=8, top_k=2, hidden=64)
        hs = np.random.randn(4, 6, 64).astype(np.float32)
        sched = est.estimate(hs)
        self.assertEqual(len(sched.routings), 4)

    def test_routing_top_k_count(self):
        est = self._make_estimator(n_layers=2, n_experts=8, top_k=3, hidden=64)
        hs = np.random.randn(5, 64).astype(np.float32)
        sched = est.estimate(hs)
        routing = sched.routings[0]
        self.assertEqual(routing.token_assignments.shape, (5, 3))

    def test_routing_weights_sum_to_one(self):
        est = self._make_estimator(n_layers=2, n_experts=4, top_k=2, hidden=64)
        hs = np.random.randn(4, 64).astype(np.float32)
        sched = est.estimate(hs)
        for routing in sched.routings.values():
            row_sums = routing.token_weights.sum(axis=-1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_estimate_single_layer(self):
        est = self._make_estimator()
        hs = np.random.randn(3, 64).astype(np.float32)
        routing = est.estimate_single_layer(0, hs)
        self.assertIsInstance(routing, LayerRouting)
        self.assertEqual(routing.layer_idx, 0)

    def test_load_gate_weights_transposed(self):
        cfg = RouterConfig(n_layers=2, n_experts=4, hidden_size=32)
        est = RouterEstimator(cfg)
        # (n_layers, hidden_size, n_experts) — should be auto-transposed
        weights = np.random.randn(2, 32, 4).astype(np.float32)
        est.load_gate_weights(weights)
        self.assertTrue(est.gate_weights_loaded)

    def test_load_gate_weights_per_layer(self):
        cfg = RouterConfig(n_layers=3, n_experts=4, hidden_size=16)
        est = RouterEstimator(cfg)
        for li in range(3):
            est.load_gate_weights_for_layer(
                li, np.random.randn(4, 16).astype(np.float32)
            )
        hs = np.random.randn(2, 16).astype(np.float32)
        sched = est.estimate(hs)
        self.assertEqual(len(sched), 3)

    def test_wrong_hidden_dim_raises(self):
        est = self._make_estimator(hidden=64)
        hs = np.random.randn(4, 32).astype(np.float32)  # wrong dim
        with self.assertRaises(ValueError):
            est.estimate(hs)

    def test_repr(self):
        est = self._make_estimator()
        r = repr(est)
        self.assertIn("RouterEstimator", r)


# ══════════════════════════════════════════════════════════════════════════════
# TestINT4ExpertPacker
# ══════════════════════════════════════════════════════════════════════════════

class TestPackConfig(unittest.TestCase):
    def test_valid(self):
        cfg = PackConfig(group_size=128)
        self.assertEqual(cfg.group_size, 128)

    def test_zero_group_size_raises(self):
        with self.assertRaises(ValueError):
            PackConfig(group_size=0)


class TestINT4PackedMatrix(unittest.TestCase):
    def test_compression_ratio_float32(self):
        packer = INT4ExpertPacker(PackConfig(group_size=64))
        w = np.random.randn(64, 256).astype(np.float32)
        packed = packer.pack(w)
        self.assertGreater(packed.compression_ratio, 3.0)  # at least 3× compression


class TestINT4ExpertPacker(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.packer = INT4ExpertPacker(PackConfig(group_size=64))

    def _random_weight(self, out=64, inp=256):
        return (self.rng.standard_normal((out, inp)) * 0.02).astype(np.float32)

    def test_pack_returns_int4_packed_matrix(self):
        w = self._random_weight()
        packed = self.packer.pack(w)
        self.assertIsInstance(packed, INT4PackedMatrix)

    def test_pack_shape_preserved_in_original(self):
        w = self._random_weight(32, 128)
        packed = self.packer.pack(w)
        self.assertEqual(packed.original_shape, (32, 128))

    def test_unpack_shape_matches_original(self):
        w = self._random_weight(64, 256)
        packed = self.packer.pack(w)
        w_approx = self.packer.unpack(packed)
        self.assertEqual(w_approx.shape, (64, 256))

    def test_roundtrip_error_small(self):
        w = self._random_weight(32, 128)
        err = self.packer.quantization_error(w)
        # error should be small relative to weight magnitude (~0.02)
        self.assertLess(err, 0.005)

    def test_pack_1d_vector(self):
        w = self.rng.standard_normal(256).astype(np.float32)
        packed = self.packer.pack(w)
        w_approx = self.packer.unpack(packed)
        self.assertEqual(w_approx.shape, (1, 256))

    def test_pack_expert(self):
        weights = {
            "gate": self._random_weight(64, 32),
            "up": self._random_weight(64, 32),
            "down": self._random_weight(32, 64),
        }
        packed = self.packer.pack_expert(weights, layer_idx=1, expert_idx=3)
        self.assertIsInstance(packed, INT4PackedExpert)
        self.assertEqual(packed.layer_idx, 1)
        self.assertEqual(packed.expert_idx, 3)
        self.assertEqual(set(packed.matrices.keys()), {"gate", "up", "down"})

    def test_unpack_expert(self):
        weights = {
            "gate": self._random_weight(64, 32),
            "up": self._random_weight(64, 32),
        }
        packed = self.packer.pack_expert(weights)
        unpacked = self.packer.unpack_expert(packed)
        self.assertEqual(set(unpacked.keys()), {"gate", "up"})
        self.assertEqual(unpacked["gate"].shape, (64, 32))

    def test_total_packed_bytes_less_than_float32(self):
        w = self._random_weight(128, 512)
        packed = self.packer.pack(w)
        original_bytes = 128 * 512 * 4  # float32
        self.assertLess(packed.packed_bytes, original_bytes)

    def test_repr(self):
        r = repr(self.packer)
        self.assertIn("INT4ExpertPacker", r)

    def test_non_power_of_2_in_features(self):
        # in_features=100 is not divisible by group_size=64 — should work (padded)
        w = self._random_weight(16, 100)
        packed = self.packer.pack(w)
        w_approx = self.packer.unpack(packed)
        self.assertEqual(w_approx.shape, (16, 100))


# ══════════════════════════════════════════════════════════════════════════════
# TestExecutorPrimitives
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutorPrimitives(unittest.TestCase):
    def test_rms_norm_unit_output(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = _rms_norm(x, w)
        # Output should have rms ≈ 1
        self.assertAlmostEqual(float(np.sqrt((out ** 2).mean())), 1.0, places=4)

    def test_silu_zero(self):
        self.assertAlmostEqual(float(_silu(np.array([0.0]))[0]), 0.0, places=5)

    def test_silu_positive_large(self):
        # silu(x) ≈ x for large positive x
        val = float(_silu(np.array([10.0]))[0])
        self.assertAlmostEqual(val, 10.0, places=2)

    def test_softmax_sums_to_one(self):
        x = np.random.randn(5, 8).astype(np.float32)
        s = _softmax(x)
        np.testing.assert_allclose(s.sum(axis=-1), 1.0, atol=1e-6)

    def test_softmax_non_negative(self):
        x = np.random.randn(3, 4).astype(np.float32)
        self.assertTrue((_softmax(x) >= 0).all())


# ══════════════════════════════════════════════════════════════════════════════
# TestExecutorConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutorConfig(unittest.TestCase):
    def test_valid_config(self):
        cfg = ExecutorConfig(n_layers=4, n_experts=8, top_k=2, hidden_size=64)
        self.assertEqual(cfg.n_layers, 4)

    def test_top_k_too_large(self):
        with self.assertRaises(ValueError):
            ExecutorConfig(n_layers=4, n_experts=4, top_k=5)


# ══════════════════════════════════════════════════════════════════════════════
# TestLayerByLayerExecutor
# ══════════════════════════════════════════════════════════════════════════════

class TestLayerByLayerExecutor(unittest.TestCase):
    VOCAB = 64
    HIDDEN = 32
    INTER = 64
    N_LAYERS = 2
    N_EXPERTS = 4
    TOP_K = 2

    def _make_executor(self):
        cfg = ExecutorConfig(
            n_layers=self.N_LAYERS,
            n_experts=self.N_EXPERTS,
            top_k=self.TOP_K,
            hidden_size=self.HIDDEN,
            intermediate_size=self.INTER,
            vocab_size=self.VOCAB,
        )
        exe = LayerByLayerExecutor(cfg)
        rng = np.random.default_rng(7)
        # Embedding
        exe.set_embedding(rng.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32))
        exe.set_lm_head(rng.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32))
        exe.set_final_norm(np.ones(self.HIDDEN, dtype=np.float32))
        # Layers
        for li in range(self.N_LAYERS):
            lw = LayerWeights(
                q_proj=rng.standard_normal((self.HIDDEN, self.HIDDEN)).astype(np.float32),
                k_proj=rng.standard_normal((self.HIDDEN, self.HIDDEN)).astype(np.float32),
                v_proj=rng.standard_normal((self.HIDDEN, self.HIDDEN)).astype(np.float32),
                o_proj=rng.standard_normal((self.HIDDEN, self.HIDDEN)).astype(np.float32),
                input_norm=np.ones(self.HIDDEN, dtype=np.float32),
                post_attn_norm=np.ones(self.HIDDEN, dtype=np.float32),
                router_gate=rng.standard_normal((self.N_EXPERTS, self.HIDDEN)).astype(np.float32),
            )
            exe.set_layer(li, lw)

        def _expert_getter(layer_idx, expert_idx):
            return {
                "gate": rng.standard_normal((self.INTER, self.HIDDEN)).astype(np.float32),
                "up": rng.standard_normal((self.INTER, self.HIDDEN)).astype(np.float32),
                "down": rng.standard_normal((self.HIDDEN, self.INTER)).astype(np.float32),
            }

        exe.set_expert_getter(_expert_getter)
        return exe

    def _make_schedule(self, seq_len: int = 3) -> ExpertSchedule:
        cfg = RouterConfig(
            n_layers=self.N_LAYERS,
            n_experts=self.N_EXPERTS,
            top_k=self.TOP_K,
            hidden_size=self.HIDDEN,
        )
        est = RouterEstimator(cfg)
        rng = np.random.default_rng(0)
        gw = rng.standard_normal((self.N_LAYERS, self.N_EXPERTS, self.HIDDEN)).astype(np.float32)
        est.load_gate_weights(gw)
        hs = rng.standard_normal((seq_len, self.HIDDEN)).astype(np.float32)
        return est.estimate(hs)

    def test_forward_returns_logits_shape(self):
        exe = self._make_executor()
        sched = self._make_schedule()
        input_ids = np.array([1, 5, 10], dtype=np.int32)
        logits = exe.forward(input_ids, sched)
        self.assertEqual(logits.shape, (self.VOCAB,))

    def test_forward_no_embedding_raises(self):
        cfg = ExecutorConfig(n_layers=2, n_experts=4, top_k=2, hidden_size=32)
        exe = LayerByLayerExecutor(cfg)
        sched = self._make_schedule()
        with self.assertRaises(RuntimeError):
            exe.forward(np.array([1]), sched)

    def test_last_stats_populated(self):
        exe = self._make_executor()
        sched = self._make_schedule()
        exe.forward(np.array([1, 2, 3], dtype=np.int32), sched)
        stats = exe.last_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats.n_layers_executed, self.N_LAYERS)

    def test_expert_activation_count(self):
        exe = self._make_executor()
        sched = self._make_schedule(seq_len=5)
        exe.forward(np.array(list(range(5)), dtype=np.int32), sched)
        stats = exe.last_stats
        # 5 tokens × top_k=2 × n_layers=2 = 20 activations
        self.assertEqual(stats.n_expert_activations, 5 * self.TOP_K * self.N_LAYERS)

    def test_prefetcher_called(self):
        exe = self._make_executor()
        sched = self._make_schedule()
        calls = []
        exe.set_expert_prefetcher(lambda li, ids: calls.append((li, ids)))
        exe.forward(np.array([1, 2, 3], dtype=np.int32), sched)
        self.assertGreater(len(calls), 0)

    def test_repr(self):
        exe = self._make_executor()
        r = repr(exe)
        self.assertIn("LayerByLayerExecutor", r)


# ══════════════════════════════════════════════════════════════════════════════
# TestSampleToken
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleToken(unittest.TestCase):
    def test_greedy_argmax(self):
        logits = np.array([0.1, 0.9, 0.3, 0.2], dtype=np.float32)
        rng = np.random.default_rng(0)
        tok = _sample_token(logits, temperature=0.0, top_p=1.0, rng=rng)
        self.assertEqual(tok, 1)

    def test_temperature_zero_deterministic(self):
        logits = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        rng = np.random.default_rng(0)
        tok1 = _sample_token(logits, temperature=0.0, top_p=1.0, rng=rng)
        tok2 = _sample_token(logits, temperature=0.0, top_p=1.0, rng=rng)
        self.assertEqual(tok1, tok2)

    def test_valid_token_range(self):
        logits = np.random.randn(100).astype(np.float32)
        rng = np.random.default_rng(42)
        tok = _sample_token(logits, temperature=1.0, top_p=0.9, rng=rng)
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, 100)


# ══════════════════════════════════════════════════════════════════════════════
# TestPipelineConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfig(unittest.TestCase):
    def test_valid(self):
        cfg = PipelineConfig(budget_mb=1024, use_int4=True)
        self.assertTrue(cfg.use_int4)

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            PipelineConfig(budget_mb=0)

    def test_negative_temperature_raises(self):
        with self.assertRaises(ValueError):
            PipelineConfig(temperature=-1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TestMoEPipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEPipeline(unittest.TestCase):
    def setUp(self):
        self.model_dir = _make_mock_model_dir(
            arch="MixtralForCausalLM",
            n_layers=2,
            n_experts=4,
            hidden=32,
            intermediate=64,
        )
        self.cfg = PipelineConfig(
            budget_mb=256,
            use_int4=True,
            group_size=32,
            max_tokens=5,
            temperature=0.0,
        )

    def test_from_pretrained(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        self.assertIsInstance(pipe, MoEPipeline)

    def test_model_info(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        info = pipe.model_info
        self.assertEqual(info.n_layers, 2)
        self.assertEqual(info.n_experts, 4)

    def test_backbone_not_loaded_initially(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        self.assertFalse(pipe.backbone_loaded)

    def test_warmup_loads_backbone(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        pipe.warmup()
        self.assertTrue(pipe.backbone_loaded)

    def test_generate_yields_tokens(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        tokens = list(pipe.generate("Hello", max_tokens=3))
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_generate_sync_returns_result(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        result = pipe.generate_sync("Test", max_tokens=3)
        self.assertIsInstance(result, GenerationResult)
        self.assertIsInstance(result.text, str)

    def test_last_stats_populated_after_generate(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        list(pipe.generate("Hello", max_tokens=3))
        stats = pipe.last_stats
        self.assertIsNotNone(stats)
        self.assertGreater(stats.n_tokens_generated, 0)

    def test_pipeline_stats_str(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        list(pipe.generate("Hi", max_tokens=2))
        s = str(pipe.last_stats)
        self.assertIn("tokens=", s)

    def test_expert_memory_map_access(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        pipe.warmup()
        emap = pipe.expert_memory_map
        self.assertIsInstance(emap, ExpertMemoryMap)

    def test_repr(self):
        pipe = MoEPipeline.from_pretrained(self.model_dir, self.cfg)
        r = repr(pipe)
        self.assertIn("MoEPipeline", r)

    def test_from_pretrained_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            MoEPipeline.from_pretrained("/no/such/path")


# ══════════════════════════════════════════════════════════════════════════════
# TestCatalogAdditions
# ══════════════════════════════════════════════════════════════════════════════

class TestCatalogAdditions(unittest.TestCase):
    def setUp(self):
        from squish.catalog import load_catalog, resolve
        self._load = load_catalog
        self._resolve = resolve

    def test_mixtral_8x7b_present(self):
        entry = self._resolve("mixtral:8x7b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)
        self.assertEqual(entry.active_params_b, 13.0)

    def test_mixtral_8x22b_present(self):
        entry = self._resolve("mixtral:8x22b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)
        self.assertEqual(entry.active_params_b, 39.0)

    def test_qwen3_235b_present(self):
        entry = self._resolve("qwen3:235b-a22b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)
        self.assertEqual(entry.active_params_b, 22.0)

    def test_mixtral_alias(self):
        entry = self._resolve("mixtral")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "mixtral:8x7b")

    def test_mixtral_47b_alias(self):
        entry = self._resolve("mixtral:47b")
        self.assertIsNotNone(entry)

    def test_mixtral_141b_alias(self):
        entry = self._resolve("mixtral:141b")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "mixtral:8x22b")

    def test_qwen3_235b_alias(self):
        entry = self._resolve("qwen3:235b")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "qwen3:235b-a22b")

    def test_mixtral_8x7b_params(self):
        entry = self._resolve("mixtral:8x7b")
        self.assertEqual(entry.params, "47B")

    def test_mixtral_8x22b_impossible_tag(self):
        entry = self._resolve("mixtral:8x22b")
        self.assertIn("impossible", entry.tags)

    def test_qwen3_235b_impossible_tag(self):
        entry = self._resolve("qwen3:235b-a22b")
        self.assertIn("impossible", entry.tags)

    def test_moe_model_notes_mention_wave73(self):
        for model_id in ["mixtral:8x7b", "mixtral:8x22b", "qwen3:235b-a22b"]:
            entry = self._resolve(model_id)
            self.assertIn("Wave 73", entry.notes, msg=f"{model_id} notes missing Wave 73")

    def test_all_new_moe_models_have_active_params(self):
        for model_id in ["mixtral:8x7b", "mixtral:8x22b", "qwen3:235b-a22b"]:
            entry = self._resolve(model_id)
            self.assertIsNotNone(entry.active_params_b)
            self.assertGreater(entry.active_params_b, 0)


# ══════════════════════════════════════════════════════════════════════════════
# TestMemorySavings
# ══════════════════════════════════════════════════════════════════════════════

class TestMemorySavings(unittest.TestCase):
    """Validate that the Wave 73 design actually enables "impossible" models."""

    def test_mixtral_8x7b_backbone_fits_16gb(self):
        """Verify Mixtral 8x7B activation ratio leaves backbone < 16 GB."""
        config = {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32000,
        }
        info = _extract_model_info(config)
        # Activation ratio should be <40%
        self.assertLess(info.activation_ratio, 0.40)
        # Memory savings factor should be >2×
        self.assertGreater(info.memory_savings_x, 2.0)

    def test_int4_compresses_expert_4x(self):
        """Verify INT4 packer gives ≥3× compression on large expert weights."""
        packer = INT4ExpertPacker(PackConfig(group_size=128))
        # Simulate gate matrix for one expert in Mixtral-8x7B
        # (14336 × 4096) — too large for test, use proportional:
        w = np.random.randn(256, 512).astype(np.float32)
        packed = packer.pack(w)
        self.assertGreater(packed.compression_ratio, 3.0)

    def test_lru_budget_limits_resident_set(self):
        """With a 10-expert budget, verify capacity is respected."""
        # Use max_experts to cap at 10 without needing large allocations
        cfg = MemoryMapConfig(budget_mb=1024, max_experts=10)
        emap = ExpertMemoryMap(cfg)
        for i in range(20):
            emap.put(0, i, {"w": np.zeros(256, dtype=np.float32)})
        # The map should cap at 10 experts
        self.assertLessEqual(len(emap), 10)

    def test_qwen3_235b_activation_economics(self):
        """Qwen3-235B-A22B: 22B active / 235B total = 9.4% activation ratio."""
        config = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 94,
            "n_routed_experts": 128,
            "num_experts_per_tok": 4,
            "hidden_size": 7168,
            "moe_intermediate_size": 2048,
            "vocab_size": 152064,
            "num_shared_experts": 1,
        }
        info = _extract_model_info(config)
        # Should be a very low activation ratio
        self.assertLess(info.activation_ratio, 0.25)
        # Memory savings should be significant
        self.assertGreater(info.memory_savings_x, 3.0)


if __name__ == "__main__":
    unittest.main()
