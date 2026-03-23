"""Verify all Config attribute aliases resolve correctly.

Run: python3 dev/verify_config_fixes.py
"""
import importlib

failures = []

# --------------------------------------------------------------------------
# Category 2 — @property aliases
# --------------------------------------------------------------------------
mods_cat2 = [
    ("squish.kv.radix_attn",         "RadixAttentionConfig", {"max_nodes": "max_tokens"}),
    ("squish.speculative.eagle2_spec","EAGLE2Config",         {"gamma": "draft_length", "tree_depth": "max_depth"}),
    ("squish.attention.ring_attn",   "RingAttentionConfig",  {"n_devices": "n_shards"}),
    ("squish.attention.cla_share",   "CLAShareConfig",       {"share_every": "sharing_stride"}),
    ("squish.speculative.lade_decode","LADEConfig",           {"window_size": "n_lookahead", "ngram_size": "n_gram"}),
    ("squish.attention.infini_attn", "InfiniAttentionConfig",{"segment_size": "segment_len"}),
    ("squish.kv.akvq_cache",         "AKVQConfig",           {"min_bits": "low_precision_bits", "max_bits": "high_precision_bits"}),
    ("squish.quant.delta_zip",        "DeltaZipConfig",       {"bits": "quant_bits"}),
    ("squish.kv.wkv_quant",           "WKVQuantConfig",       {"bits": "n_bits"}),
    ("squish.quant.loftq",            "LoFTQConfig",          {"bits": "n_bits"}),
]

for mod, cfg_name, checks in mods_cat2:
    try:
        m = importlib.import_module(mod)
        cfg = getattr(m, cfg_name)()
        for alias, real in checks.items():
            got = getattr(cfg, alias)
            want = getattr(cfg, real)
            assert got == want, f"{alias}={got!r} != {real}={want!r}"
    except Exception as e:
        failures.append(f"{mod}.{cfg_name}: {e}")

# qmoe – bits property (no matching real field, just needs to not raise)
try:
    from squish.moe.qmoe_compress import QMoEConfig
    c = QMoEConfig()
    _ = c.bits
except Exception as e:
    failures.append(f"qmoe_compress.QMoEConfig.bits: {e}")

# memory_dim property on InfiniAttentionConfig
try:
    from squish.attention.infini_attn import InfiniAttentionConfig
    c = InfiniAttentionConfig()
    _ = c.memory_dim
except Exception as e:
    failures.append(f"infini_attn.InfiniAttentionConfig.memory_dim: {e}")

# chunk_size property on RingAttentionConfig
try:
    from squish.attention.ring_attn import RingAttentionConfig
    c = RingAttentionConfig()
    _ = c.chunk_size
except Exception as e:
    failures.append(f"ring_attn.RingAttentionConfig.chunk_size: {e}")

# HybridArchConfig – default layer_types
try:
    from squish.serving.hybrid_arch_router import HybridArchConfig
    c = HybridArchConfig()
    assert c.layer_types is None, f"layer_types default wrong: {c.layer_types!r}"
except Exception as e:
    failures.append(f"hybrid_arch_router.HybridArchConfig: {e}")

# --------------------------------------------------------------------------
# Category 3 — class name aliases
# --------------------------------------------------------------------------
cat3 = [
    ("squish.sampling.prm_beam_search",       ["PRMBeamSearchConfig",   "PRMBeamSearch"]),
    ("squish.vision.cross_modal_attn",         ["CrossModalAttnConfig",  "CrossModalRouter"]),
    ("squish.vision.vlm_spec_decode",          ["VLMSpecDecodeConfig",   "VLMSpecDecode"]),
    ("squish.serving.vlm_scheduler",           ["VLMBatchConfig",        "VLMBatchScheduler"]),
    ("squish.moe.fine_grained_router",         ["FineGrainedMoEConfig",  "FineGrainedMoERouter"]),
    ("squish.moe.expert_offload",              ["ExpertOffloaderConfig", "ExpertOffloader"]),
    ("squish.attention.double_sparse",         ["DoubleSparsityConfig",  "DoubleSparsityAttn"]),
    ("squish.serving.token_budget_scheduler",  ["TokenBudgetConfig",     "TokenBudgetScheduler"]),
    ("squish.sampling.test_time_scale",        ["TestTimeComputeConfig",  "TestTimeComputeRouter"]),
    ("squish.sampling.eta_sampler",            ["EtaConfig",             "EtaSampler"]),
    ("squish.sampling.min_p_sampler",          ["MinPConfig",            "MinPSampler"]),
]

for mod, names in cat3:
    try:
        m = importlib.import_module(mod)
        for n in names:
            getattr(m, n)
    except Exception as e:
        failures.append(f"{mod}: {e}")

# min_p_factor alias in min_p_sampler
try:
    from squish.sampling.min_p_sampler import MinPConfig
    c = MinPConfig()
    assert c.min_p_factor == c.p_min, f"min_p_factor alias wrong"
except Exception as e:
    failures.append(f"min_p_sampler.MinPConfig.min_p_factor: {e}")

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
else:
    print("ALL CHECKS PASSED")
