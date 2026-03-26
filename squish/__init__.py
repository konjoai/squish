"""
Squish — fast compressed model loader and OpenAI-compatible server for Apple Silicon.

Public API:
    load_compressed_model(model_dir, npz_path_or_dir, ...)
    load_from_npy_dir(dir_path, model_dir, ...)
    save_int4_npy_dir(npy_dir, group_size=32, verbose=True)

    compress_npy_dir(tensors_dir, level=3, ...)  # zstd entropy compression
    decompress_npy_dir(tensors_dir, ...)

    run_server(...)   # OpenAI-compatible HTTP server
"""
from __future__ import annotations

__version__ = "9.2.0"

# ── Lazy import registry ───────────────────────────────────────────────────────
# Every public name is loaded on first access via __getattr__.
# This keeps `import squish` fast regardless of how many wave modules exist.
_LAZY_IMPORTS: dict[str, str] = {
    # squish.serving.ada_serve
    "AdaServeConfig":            "squish.serving.ada_serve",
    "AdaServeRequest":           "squish.serving.ada_serve",
    "AdaServeScheduler":         "squish.serving.ada_serve",
    "AdaServeStats":             "squish.serving.ada_serve",
    "SLOTarget":                 "squish.serving.ada_serve",

    # squish.quant.awq
    "apply_awq_to_weights":      "squish.quant.awq",
    "collect_activation_scales": "squish.quant.awq",
    "load_awq_scales":           "squish.quant.awq",
    "save_awq_scales":           "squish.quant.awq",

    # squish.catalog
    "CatalogEntry":              "squish.catalog",
    "list_catalog":              "squish.catalog",
    "load_catalog":              "squish.catalog",
    "pull_model":                "squish.catalog",
    "resolve_model":             "squish.catalog",

    # squish.attention.cla
    "CLAConfig":                 "squish.attention.cla",
    "CLALayerSpec":              "squish.attention.cla",
    "CLASchedule":               "squish.attention.cla",
    "CLAStats":                  "squish.attention.cla",

    # squish.quant.compressed_loader
    "load_compressed_model":     "squish.quant.compressed_loader",
    "load_from_npy_dir":         "squish.quant.compressed_loader",
    "save_int4_npy_dir":         "squish.quant.compressed_loader",

    # squish.speculative.conf_spec
    "ConfSpecConfig":            "squish.speculative.conf_spec",
    "ConfSpecDecision":          "squish.speculative.conf_spec",
    "ConfSpecStats":             "squish.speculative.conf_spec",
    "ConfSpecVerifier":          "squish.speculative.conf_spec",

    # squish.quant.dfloat11
    "CompressedBlock":           "squish.quant.dfloat11",
    "CompressedModel":           "squish.quant.dfloat11",
    "DFloat11Compressor":        "squish.quant.dfloat11",
    "DFloat11Config":            "squish.quant.dfloat11",
    "HuffmanCodec":              "squish.quant.dfloat11",
    "compress_model":            "squish.quant.dfloat11",

    # squish.kv.diffkv
    "CompactedKVSlot":           "squish.kv.diffkv",
    "DiffKVConfig":              "squish.kv.diffkv",
    "DiffKVPolicy":              "squish.kv.diffkv",
    "DiffKVPolicyManager":       "squish.kv.diffkv",
    "DiffKVStats":               "squish.kv.diffkv",
    "HeadSparsityProfile":       "squish.kv.diffkv",
    "TokenImportanceTier":       "squish.kv.diffkv",

    # squish.speculative.dovetail
    "DovetailCPUVerifier":       "squish.speculative.dovetail",
    "DovetailConfig":            "squish.speculative.dovetail",
    "DovetailDecoder":           "squish.speculative.dovetail",
    "DovetailDraftRunner":       "squish.speculative.dovetail",
    "DovetailStats":             "squish.speculative.dovetail",

    # squish.attention.duo_decoding
    "DuoCandidate":              "squish.attention.duo_decoding",
    "DuoCPUVerifier":            "squish.attention.duo_decoding",
    "DuoDecodingConfig":         "squish.attention.duo_decoding",
    "DuoDecodingDecoder":        "squish.attention.duo_decoding",
    "DuoDecodingStats":          "squish.attention.duo_decoding",
    "DuoScheduler":              "squish.attention.duo_decoding",

    # squish.io.entropy
    "compress_npy_dir":          "squish.io.entropy",
    "decompress_npy_dir":        "squish.io.entropy",

    # squish.attention.flash_attention
    "PatchResult":               "squish.attention.flash_attention",
    "attention_status":          "squish.attention.flash_attention",
    "patch_model_attention":     "squish.attention.flash_attention",
    "predict_memory_savings":    "squish.attention.flash_attention",
    "print_memory_table":        "squish.attention.flash_attention",

    # squish.token.forelen
    "EGTPPredictor":             "squish.token.forelen",
    "ForelenConfig":             "squish.token.forelen",
    "ForelenStats":              "squish.token.forelen",
    "PLPPredictor":              "squish.token.forelen",

    # squish.speculative.fr_spec
    "FRSpecCalibrator":          "squish.speculative.fr_spec",
    "FRSpecConfig":              "squish.speculative.fr_spec",
    "FRSpecHead":                "squish.speculative.fr_spec",
    "FRSpecStats":               "squish.speculative.fr_spec",
    "FreqTokenSubset":           "squish.speculative.fr_spec",

    # squish.token.gemfilter
    "AttentionScoreBuffer":      "squish.token.gemfilter",
    "GemFilterConfig":           "squish.token.gemfilter",
    "GemFilterStats":            "squish.token.gemfilter",
    "GemSelector":               "squish.token.gemfilter",

    # squish.token.ipw
    "IPWConfig":                 "squish.token.ipw",
    "IPWMeasurement":            "squish.token.ipw",
    "IPWSummary":                "squish.token.ipw",
    "IPWTracker":                "squish.token.ipw",

    # squish.kv.kv_cache
    "DiskKVCache":               "squish.kv.kv_cache",
    "KVBudgetBroker":            "squish.kv.kv_cache",
    "QuantizedKVCache":          "squish.kv.kv_cache",
    "make_quantized_cache":      "squish.kv.kv_cache",
    "patch_model_kv_cache":      "squish.kv.kv_cache",

    # squish.kv.kv_slab
    "KVPage":                    "squish.kv.kv_slab",
    "KVSlabAllocator":           "squish.kv.kv_slab",

    # squish.kv.kvsharer
    "KVLayerCache":              "squish.kv.kvsharer",
    "KVShareMap":                "squish.kv.kvsharer",
    "KVSharerCalibrator":        "squish.kv.kvsharer",
    "KVSharerConfig":            "squish.kv.kvsharer",
    "KVSharerStats":             "squish.kv.kvsharer",

    # squish.kv.kvtuner
    "KVQuantConfig":             "squish.kv.kvtuner",
    "KVTunerCalibrator":         "squish.kv.kvtuner",
    "KVTunerConfig":             "squish.kv.kvtuner",
    "KVTunerStats":              "squish.kv.kvtuner",
    "LayerSensitivity":          "squish.kv.kvtuner",

    # squish.token.layer_skip
    "ConfidenceEstimator":       "squish.token.layer_skip",
    "EarlyExitConfig":           "squish.token.layer_skip",
    "EarlyExitDecoder":          "squish.token.layer_skip",
    "EarlyExitStats":            "squish.token.layer_skip",

    # squish.hardware.layerwise_loader
    "LayerCache":                "squish.hardware.layerwise_loader",
    "LayerwiseLoader":           "squish.hardware.layerwise_loader",
    "LoadStats":                 "squish.hardware.layerwise_loader",
    "recommend_cache_size":      "squish.hardware.layerwise_loader",
    "shard_model":               "squish.hardware.layerwise_loader",

    # squish.speculative.long_spec
    "LongSpecConfig":            "squish.speculative.long_spec",
    "LongSpecDecoder":           "squish.speculative.long_spec",
    "LongSpecHead":              "squish.speculative.long_spec",
    "LongSpecStats":             "squish.speculative.long_spec",

    # squish.token.lookahead_reasoning
    "LookaheadBatch":            "squish.token.lookahead_reasoning",
    "LookaheadConfig":           "squish.token.lookahead_reasoning",
    "LookaheadReasoningEngine":  "squish.token.lookahead_reasoning",
    "LookaheadStats":            "squish.token.lookahead_reasoning",
    "LookaheadStep":             "squish.token.lookahead_reasoning",

    # squish.lora.lora_manager
    "DareTiesConfig":            "squish.lora.lora_manager",
    "DareTiesMerger":            "squish.lora.lora_manager",
    "LoRAManager":               "squish.lora.lora_manager",

    # squish.speculative.mirror_sd
    "MirrorDraftPipeline":       "squish.speculative.mirror_sd",
    "MirrorFuture":              "squish.speculative.mirror_sd",
    "MirrorSDConfig":            "squish.speculative.mirror_sd",
    "MirrorSDDecoder":           "squish.speculative.mirror_sd",
    "MirrorSDStats":             "squish.speculative.mirror_sd",
    "MirrorVerifyPipeline":      "squish.speculative.mirror_sd",

    # squish.kv.paged_attention
    "BlockAllocator":            "squish.kv.paged_attention",
    "PageBlockTable":            "squish.kv.paged_attention",
    "PagedKVCache":              "squish.kv.paged_attention",

    # squish.kv.paris_kv
    "ParisKVCodebook":           "squish.kv.paris_kv",
    "ParisKVConfig":             "squish.kv.paris_kv",

    # squish.moe.pipo
    "INT4BypassKernel":          "squish.moe.pipo",
    "LayerWeightBuffer":         "squish.moe.pipo",
    "PIPOConfig":                "squish.moe.pipo",
    "PIPOScheduler":             "squish.moe.pipo",

    # squish.speculative.prompt_lookup
    "NGramIndex":                "squish.speculative.prompt_lookup",
    "PromptLookupConfig":        "squish.speculative.prompt_lookup",
    "PromptLookupDecoder":       "squish.speculative.prompt_lookup",
    "PromptLookupStats":         "squish.speculative.prompt_lookup",

    # squish.speculative.qspec
    "ActivationQuantizer":       "squish.speculative.qspec",
    "QSpecConfig":               "squish.speculative.qspec",
    "QSpecDecoder":              "squish.speculative.qspec",
    "QSpecStats":                "squish.speculative.qspec",

    # squish.quant.quantizer
    "QuantizationResult":        "squish.quant.quantizer",
    "dequantize_int4":           "squish.quant.quantizer",
    "get_backend_info":          "squish.quant.quantizer",
    "mean_cosine_similarity":    "squish.quant.quantizer",
    "quantize_embeddings":       "squish.quant.quantizer",
    "quantize_int4":             "squish.quant.quantizer",
    "reconstruct_embeddings":    "squish.quant.quantizer",

    # squish.kv.radix_cache
    "RadixNode":                 "squish.kv.radix_cache",
    "RadixTree":                 "squish.kv.radix_cache",

    # squish.serving.robust_scheduler
    "ABalancedScheduler":        "squish.serving.robust_scheduler",
    "AMaxScheduler":             "squish.serving.robust_scheduler",
    "LengthInterval":            "squish.serving.robust_scheduler",
    "Request":                   "squish.serving.robust_scheduler",
    "RobustSchedulerConfig":     "squish.serving.robust_scheduler",
    "RobustSchedulerStats":      "squish.serving.robust_scheduler",

    # squish.attention.sage_attention
    "KSmoother":                 "squish.attention.sage_attention",
    "SageAttentionConfig":       "squish.attention.sage_attention",
    "SageAttentionKernel":       "squish.attention.sage_attention",
    "SageAttentionStats":        "squish.attention.sage_attention",

    # squish.attention.sage_attention2
    "SageAttention2Config":      "squish.attention.sage_attention2",
    "SageAttention2Kernel":      "squish.attention.sage_attention2",
    "SageAttention2Stats":       "squish.attention.sage_attention2",
    "WarpQuantResult":           "squish.attention.sage_attention2",

    # squish.streaming.seq_packing
    "PackedBatch":               "squish.streaming.seq_packing",
    "PackingConfig":             "squish.streaming.seq_packing",
    "PackingStats":              "squish.streaming.seq_packing",
    "SequencePacker":            "squish.streaming.seq_packing",

    # squish.kv.shadow_kv
    "LandmarkSelector":          "squish.kv.shadow_kv",
    "LowRankKeyCache":           "squish.kv.shadow_kv",
    "ShadowKVCache":             "squish.kv.shadow_kv",
    "ShadowKVConfig":            "squish.kv.shadow_kv",

    # squish.kv.smallkv
    "MarginalVCache":            "squish.kv.smallkv",
    "SaliencyTracker":           "squish.kv.smallkv",
    "SmallKVCache":              "squish.kv.smallkv",
    "SmallKVConfig":             "squish.kv.smallkv",
    "SmallKVStats":              "squish.kv.smallkv",

    # squish.attention.sparge_attn
    "BlockMask":                 "squish.attention.sparge_attn",
    "SpargeAttnConfig":          "squish.attention.sparge_attn",
    "SpargeAttnEngine":          "squish.attention.sparge_attn",
    "SpargeAttnStats":           "squish.attention.sparge_attn",

    # squish.speculative.sparse_spec
    "PillarAttnCache":           "squish.speculative.sparse_spec",
    "SparseSpecConfig":          "squish.speculative.sparse_spec",
    "SparseSpecDecoder":         "squish.speculative.sparse_spec",
    "SparseSpecDrafter":         "squish.speculative.sparse_spec",
    "SparseSpecStats":           "squish.speculative.sparse_spec",

    # squish.speculative.sparse_verify
    "InterDraftReuseCache":      "squish.speculative.sparse_verify",
    "SparseVerifyConfig":        "squish.speculative.sparse_verify",
    "SparseVerifyPass":          "squish.speculative.sparse_verify",
    "SparseVerifyStats":         "squish.speculative.sparse_verify",

    # squish.speculative.spec_reason
    "ReasoningStep":             "squish.speculative.spec_reason",
    "SpecReasonConfig":          "squish.speculative.spec_reason",
    "SpecReasonOrchestrator":    "squish.speculative.spec_reason",
    "SpecReasonStats":           "squish.speculative.spec_reason",
    "StepVerdict":               "squish.speculative.spec_reason",

    # squish.speculative.specontext
    "DistilledRetrievalHead":    "squish.speculative.specontext",
    "SpeContextCache":           "squish.speculative.specontext",
    "SpeContextConfig":          "squish.speculative.specontext",
    "SpeContextStats":           "squish.speculative.specontext",

    # squish.speculative.speculative
    "SpeculativeGenerator":      "squish.speculative.speculative",
    "load_draft_model":          "squish.speculative.speculative",

    # squish.io.split_loader
    "OffloadedLayer":            "squish.io.split_loader",
    "SplitInfo":                 "squish.io.split_loader",
    "SplitLayerLoader":          "squish.io.split_loader",
    "print_layer_profile":       "squish.io.split_loader",
    "profile_model_layers":      "squish.io.split_loader",

    # squish.attention.squeeze_attention
    "BudgetAllocator":           "squish.attention.squeeze_attention",
    "LayerKVBudget":             "squish.attention.squeeze_attention",
    "SqueezeConfig":             "squish.attention.squeeze_attention",
    "SqueezeKVCache":            "squish.attention.squeeze_attention",
    "SqueezeStats":              "squish.attention.squeeze_attention",

    # squish.quant.squeeze_llm
    "OutlierDetector":           "squish.quant.squeeze_llm",
    "SqueezeLLMConfig":          "squish.quant.squeeze_llm",
    "SqueezeLLMLayer":           "squish.quant.squeeze_llm",
    "SqueezeLLMQuantizer":       "squish.quant.squeeze_llm",

    # squish.streaming.streaming_sink
    "SinkConfig":                "squish.streaming.streaming_sink",
    "SinkKVCache":               "squish.streaming.streaming_sink",
    "SinkStats":                 "squish.streaming.streaming_sink",

    # squish.speculative.sub_spec
    "SubSpecConfig":             "squish.speculative.sub_spec",
    "SubSpecDecoder":            "squish.speculative.sub_spec",
    "SubSpecStats":              "squish.speculative.sub_spec",
    "SubstituteLayerProxy":      "squish.speculative.sub_spec",

    # squish.quant.svdq
    "HeadSVDProfile":            "squish.quant.svdq",
    "SVDqCalibrator":            "squish.quant.svdq",
    "SVDqConfig":                "squish.quant.svdq",
    "SVDqPrecisionMap":          "squish.quant.svdq",
    "SVDqStats":                 "squish.quant.svdq",

    # squish.token.token_swift
    "MultiTokenHead":            "squish.token.token_swift",
    "PartialKVManager":          "squish.token.token_swift",
    "TokenSwiftConfig":          "squish.token.token_swift",
    "TokenSwiftDecoder":         "squish.token.token_swift",
    "TokenSwiftStats":           "squish.token.token_swift",

    # squish.speculative.trail
    "TrailConfig":               "squish.speculative.trail",
    "TrailLinearProbe":          "squish.speculative.trail",
    "TrailPredictor":            "squish.speculative.trail",
    "TrailStats":                "squish.speculative.trail",

    # squish.quant.vptq
    "VPTQCodebook":              "squish.quant.vptq",
    "VPTQConfig":                "squish.quant.vptq",
    "VPTQLayer":                 "squish.quant.vptq",
    "VPTQQuantizer":             "squish.quant.vptq",

    # squish.attention.yoco
    "YOCOConfig":                "squish.attention.yoco",
    "YOCOKVStore":               "squish.attention.yoco",
    "YOCOLayerSpec":             "squish.attention.yoco",
    "YOCOSchedule":              "squish.attention.yoco",
    "YOCOStats":                 "squish.attention.yoco",

    # squish.telemetry — structured tracing
    "trace_span":                "squish.telemetry",
    "get_tracer":                "squish.telemetry",
    "reset_tracer":              "squish.telemetry",
    "configure_tracing":         "squish.telemetry",

    # squish.logging_config — structured logging
    "configure_logging":         "squish.logging_config",
    "get_squish_logger":         "squish.logging_config",
}

_lazy_cache: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Load any registered public name on first access (lazy import)."""
    if name in _lazy_cache:
        return _lazy_cache[name]
    if name in _LAZY_IMPORTS:
        import importlib
        mod_name = _LAZY_IMPORTS[name]
        # Special-case aliased names from squish.catalog
        if mod_name == "squish.catalog" and name in ("pull_model", "resolve_model"):
            mod = importlib.import_module(mod_name)
            alias_map = {"pull_model": "pull", "resolve_model": "resolve"}
            obj = getattr(mod, alias_map[name])
            _lazy_cache[name] = obj
            return obj
        try:
            mod = importlib.import_module(mod_name)
        except (ImportError, OSError) as exc:
            raise AttributeError(
                f"module 'squish' has no attribute {name!r} "
                f"(optional dependency {mod_name!r} could not be imported: {exc})"
            ) from None
        obj = getattr(mod, name)
        _lazy_cache[name] = obj
        return obj
    raise AttributeError(f"module 'squish' has no attribute {name!r}")


__all__ = [
    "__version__",
    *_LAZY_IMPORTS,
]
