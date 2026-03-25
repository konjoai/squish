#!/usr/bin/env python3
"""
squish_server.py

OpenAI-compatible HTTP API server for Squish compressed models.

Exposes endpoints:
    GET  /v1/models                    — list loaded model
    GET  /v1/models/{model_id}         — model detail
    POST /v1/chat/completions          — chat (streaming + non-streaming)
    POST /v1/completions               — legacy text completion
    POST /v1/embeddings                — mean-pooled token embeddings
    POST /v1/tokenize                  — tokenize text (non-standard, useful for debugging)
    GET  /v1/metrics                   — Prometheus-compatible metrics
    GET  /health                       — health check with real-time stats

Drop-in replacement for cloud APIs:
    export OPENAI_BASE_URL=http://localhost:11435/v1
    export OPENAI_API_KEY=squish        # or your --api-key value
    # Any OpenAI client now routes to local Squish inference

Usage:
    python3 squish_server.py \\
        --model-dir   ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --compressed-dir ~/models/Qwen2.5-7B-Instruct-bf16-compressed \\
        --port 11435 [--api-key mysecret]

Dependencies:
    pip install fastapi "uvicorn[standard]"
"""
import argparse
import collections
import functools
import hashlib
import hmac
import json
import os
import sys
import logging as _logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# ── Suppress macOS malloc-stack-logging noise in child processes ──────────────
# When MallocStackLogging is set in the environment (e.g. from Instruments or
# Xcode), macOS prints "can't turn off malloc stack logging because it was not
# enabled" to stderr for every forked Python process that did not actually
# enable it.  Strip all Malloc* debug keys before any subprocesses are spawned.
for _k in (
    "MallocStackLogging", "MallocStackLoggingNoCompact",
    "MallocScribble", "MallocPreScribble", "MallocGuardEdges",
    "MallocCheckHeapStart", "MallocCheckHeapEach",
):
    os.environ.pop(_k, None)
del _k

# ── Telemetry (structured span tracing + logging config) ─────────────────────
try:
    from squish.telemetry import configure_tracing as _configure_tracing
    from squish.telemetry import get_tracer         as _get_tracer
    from squish.telemetry import trace_span         as _trace_span
    from squish.logging_config import configure_logging as _configure_logging
    _TELEMETRY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TELEMETRY_AVAILABLE = False
    def _configure_tracing(enabled): pass       # type: ignore[misc]
    def _get_tracer(): return None               # type: ignore[misc]
    def _trace_span(name, **tags): return _NullCtx()  # type: ignore[misc]
    def _configure_logging(**kwargs): pass      # type: ignore[misc]


class _NullCtx:  # pragma: no cover
    """Fallback no-op context manager used when squish.telemetry is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    def __call__(self, f): return f

# ── Ensure the squish package root is importable when run as a script ────────
# cli.py launches this file directly with `python3 .../squish/server.py`, so
# the package parent directory must be on sys.path for `from squish.*` imports.
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:  # pragma: no cover
    sys.path.insert(0, _pkg_root)

# ── Validate dependencies ────────────────────────────────────────────────────

def _require(pkg: str, install: str | None = None) -> None:
    try:
        __import__(pkg)
    except ImportError:  # pragma: no cover
        hint = install or pkg
        print(f"  {_C.PK}✗  Missing dependency:{_C.R}  {_C.W}{pkg}{_C.R}  {_C.DIM}→  pip install {hint}{_C.R}")
        sys.exit(1)

_require("fastapi")

from fastapi import FastAPI, HTTPException, Request, Security  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402

try:
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    _STATIC_FILES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STATIC_FILES_AVAILABLE = False

# ── KV cache (Phase 1.3 — lazily imported to keep startup fast) ──────────────
_kv_cache = None         # QuantizedKVCache | None — set in main() after model load
_paged_kv_cache = None   # PagedKVCache | None — set in main() when --paged-attention
_disk_prompt_cache = None  # DiskKVCache | None — set in main() when --disk-prompt-cache given
_lazy_llm_state = None  # _PruneState | None — set in main() when --lazy-llm given
# Phase 13A: asymmetric INT2 KV cache (agent-kv)
_agent_kv_config: "Any | None" = None   # AgentKVConfig | None — set in main() when --agent-kv

# ── Wave optimization module state (lazily instantiated) ─────────────────────
_prompt_lookup_decoder  = None  # PromptLookupDecoder    — --prompt-lookup
_seq_packer             = None  # SequencePacker         — --seq-packing
_ada_serve_scheduler    = None  # AdaServeScheduler      — --ada-serve
_conf_spec_verifier     = None  # ConfSpecVerifier        — --conf-spec
_kvsharer_map           = None  # KVShareMap             — --kv-share
_kv_slab_allocator      = None  # KVSlabAllocator        — --kv-slab
_paris_kv_codebook      = None  # ParisKVCodebook        — --paris-kv
_streaming_sink_cache   = None  # SinkKVCache            — --streaming-sink
_diffkv_policy_mgr      = None  # DiffKVPolicyManager    — --diff-kv
_smallkv_cache          = None  # SmallKVCache           — --small-kv
_lookahead_engine       = None  # LookaheadReasoningEngine — --lookahead
_spec_reason_orch       = None  # SpecReasonOrchestrator — --spec-reason
_sage_attn_kernel       = None  # SageAttentionKernel     — --sage-attention
_sage_attn2_kernel      = None  # SageAttention2Kernel    — --sage-attention2
_sparge_engine          = None  # SpargeAttnEngine        — --sparge-attention
_squeeze_cache          = None  # SqueezeKVCache          — --squeeze-attention
_yoco_config            = None  # YOCOConfig              — --yoco-kv
_cla_config             = None  # CLAConfig               — --cla
_kvtuner_config         = None  # KVTunerConfig           — --kvtuner
_robust_sched           = None  # AMaxScheduler           — --robust-scheduler
_gemfilter_config       = None  # GemFilterConfig         — --gemfilter
_svdq_config            = None  # SVDqConfig              — --svdq
_sparse_spec_config     = None  # SparseSpecConfig        — --sparse-spec
_sparse_verify_config   = None  # SparseVerifyConfig      — --sparse-verify
_trail_config           = None  # TrailConfig             — --trail
_specontext_config      = None  # SpeContextConfig        — --specontext
_forelen_config         = None  # ForelenConfig           — --forelen
_ipw_config             = None  # IPWConfig               — --ipw
_layer_skip_config      = None  # EarlyExitConfig         — --layer-skip
_long_spec_config       = None  # LongSpecConfig          — --long-spec
_fr_spec_config         = None  # FRSpecConfig            -- --fr-spec  # wave37-test-marker

# ── Wave 41: Prefix Sharing, EAGLE-2, Ring Attention, Token Pruning, MoE, Sink Fusion ──
_radix_attn_cache       = None  # RadixAttentionCache     — --radix-attn
_eagle2_spec            = None  # EAGLE2Spec              — --eagle2
_ring_attn_config       = None  # RingAttention           — --ring-attn
_token_entropy_pruner   = None  # TokenEntropyPruner      — --token-entropy-prune
_pregated_moe_router    = None  # PreGatedMoERouter       — --pregated-moe
_sink_fusion_config     = None  # SinkFusion              — --sink-fusion
_cla_share_config       = None  # CLAShareAttention       — --cla-share
_qmoe_compressor        = None  # QMoECompressor          — --qmoe-compress
_lade_decoder           = None  # LADEDecoder             — --lade
_infini_attn_config     = None  # InfiniAttention         — --infini-attn
_akvq_cache             = None  # AKVQCache               — --akvq
_delta_zip_store        = None  # DeltaZipAdapter         — --delta-zip

# ── Wave 42: Disaggregated Serving, NSA, Medusa, KV Quant, AttentionStore, QAT ──
_medusa_heads           = None  # MedusaHeads             — --medusa-heads
_sarathi_scheduler      = None  # SarathiScheduler        — --sarathi
_nsa_attn_config        = None  # NSAAttention            — --nsa-attn
_flex_prefill_config    = None  # FlexPrefill             — --flex-prefill
_think_cache            = None  # ThinKCache              — --think-cache
_attention_store        = None  # AttentionStore          — --attention-store
_rest_decoder           = None  # RESTDecode              — --rest-decode
_star_attn_config       = None  # StarAttention           — --star-attn
_splitwise_scheduler    = None  # SplitwiseScheduler      — --splitwise
_kvquant_cache          = None  # KVQuantCache            — --kvquant
_efficient_qat          = None  # EfficientQAT            — --efficient-qat
_cache_gen_codec        = None  # CacheGenCodec           — --cache-gen

# ── Wave 43: MTP Decode, Cascade KV, Head Pruner, Paged Attn, Layer Collapse ──
_mtp_decode_w43         = None  # MTPDecode               — --mtp-decode
_cascade_kv             = None  # CascadeKV               — --cascade-kv
_head_pruner            = None  # HeadPruner              — --head-prune
_paged_attn_w43         = None  # PagedAttention          — --paged-attn-w43
_layer_collapse         = None  # LayerCollapse           — --layer-collapse
_relay_attn             = None  # RelayAttention          — --relay-attn
_wkv_quant              = None  # WKVQuant                — --wkv-quant
_tokenized_kv           = None  # TokenizedKVCache        — --tokenized-kv
_cluster_evict_kv       = None  # ClusterEvictKV          — --cluster-evict
_s2_attn                = None  # S2Attention             — --s2-attn
_magic_pig_v2           = None  # MagicPIGv2              — --magic-pig-v2

# ── Wave 44: Marlin GEMM, Spec Rejection, LoFTQ, Online Spec, Dynamic Spec ───
_marlin_gemm            = None  # MarlinGEMM              — --marlin-gemm
_spec_rejection         = None  # SpecRejection           — --spec-rejection
_loftq_config           = None  # LoFTQ                   — --loftq
_online_spec            = None  # OnlineSpec              — --online-spec
_dynamic_spec_len       = None  # DynamicSpecLen          — --dynamic-spec-len
_big_little_llm         = None  # BigLittleLLM            — --big-little
_multi_exit_spec        = None  # MultiExitSpec           — --multi-exit-spec
_pv_tuning              = None  # PVTuning                — --pv-tuning
_hadamard_quant         = None  # HadamardQuant           — --hadamard-quant
_prefix_tree            = None  # PrefixTreeDecode        — --prefix-tree-decode
_spectr_ot              = None  # SpecTrOT                — --spectr-ot
_ada_gptq               = None  # AdaGPTQ                 — --ada-gptq

# ── Wave 45: FlexGen Offload, YaRN, SelfExtend, Orca, MxFP4, FP8Act, CLEXRoPE ─
_flexgen_offload        = None  # FlexGenOffload          — --flexgen-offload
_yarn_rope              = None  # YaRNRoPE                — --yarn-rope
_self_extend            = None  # SelfExtend              — --self-extend
_orca_scheduler         = None  # OrcaScheduler           — --orca-sched
_mx_fp4_quant           = None  # MxFP4                   — --mx-fp4
_fp8_act_quant          = None  # FP8ActQuant             — --fp8-act
_clex_rope              = None  # CLeXRoPE                — --clex-rope
_powerinfer_offload     = None  # PowerInferOffload       — --powerinfer
_grouped_rope           = None  # GroupedRoPE             — --grouped-rope
_tensor_parallel        = None  # TensorParallel          — --tensor-parallel
_fused_bias_gelu        = None  # FusedBiasGELU           — --fused-bias-gelu
_token_budget_sched     = None  # TokenBudgetScheduler    — --token-budget-sched

# ── Wave 46: Model Surgery, Expert Choice, W4A8, MLA KV, MinP, Contrastive ───
_slice_gpt              = None  # SliceGPTPruner          — --slice-gpt
_wanda_pruner           = None  # WandaPruner             — --wanda
_short_gpt              = None  # ShortGPTPruner          — --short-gpt
_w4a8_runtime           = None  # W4A8QuantRuntime        — --w4a8
_expert_choice          = None  # ExpertChoiceRouter      — --expert-choice
_mla_kv_compress        = None  # MLAKVCompress           — --mla-kv
_minp_sampler           = None  # MinPSampler             — --minp
_contrastive_search     = None  # ContrastiveSearch       — --contrastive-search
_razor_attn             = None  # RazorAttention          — --razor-attn
_cache_blend            = None  # CacheBlend              — --cache-blend
_green_kv               = None  # GreenKVEviction         — --green-kv
_preble_router          = None  # PrebeleRouter           — --preble

# ── Wave 47: Mamba2, HGRN2, Lookahead, InfMemory, vAttn, IA3, MoE-Infinity ──
_mamba2_ssm             = None  # Mamba2SSM               — --mamba2-ssm
_hgrn2                  = None  # HGRN2                   — --hgrn2
_lookahead_decode       = None  # LookaheadDecode         — --lookahead-decode
_inf_memory             = None  # InfMemory               — --inf-memory
_v_attn_kv              = None  # vAttentionKV            — --v-attn
_ia3_adapter            = None  # IA3Adapter              — --ia3
_moe_infinity           = None  # MoEInfinityOffload      — --moe-infinity
_mega_blocks            = None  # MegaBlocksSparse        — --mega-blocks
_kgw_watermark          = None  # KGWWatermark            — --kgw-watermark
_typical_sampler        = None  # TypicalSampler          — --typical-sampler
_dora_adapter           = None  # DoRAAdapter             — --dora
_calm_exit              = None  # AdaptiveCALM            — --calm-exit

# ── Wave 48: INT2/INT3 Extreme Quantization ───────────────────────────────────
_spqr_quantizer         = None  # SpQRQuantizer           — --spqr
_auto_round             = None  # AutoRoundQuantizer      — --auto-round
_owq_quantizer          = None  # OWQQuantizer            — --owq
_bit_distiller          = None  # BitDistillerQuant       — --bit-distiller
_zip_lm                 = None  # ZipLMMixedPrecision     — --zip-lm
_gguf_mixed             = None  # GGUFMixedQuantizer      — --gguf-mixed

# ── Wave 49: TTFT Sprint: LLMLingua-2, RECOMP, Selective Context, Prepack ────
_llm_lingua2            = None  # LLMLingua2Compressor    — --llm-lingua2
_recomp_compressor      = None  # RECOMPCompressor        — --recomp
_selective_context      = None  # SelectiveContextCompressor — --selective-context
_prompt_cache_kv        = None  # PromptCacheKV           — --prompt-cache
_pipe_infer             = None  # PipeInferScheduler      — --pipe-infer
_prepack_scheduler      = None  # PrepackScheduler        — --prepack

# ── Wave 50: Bigger-Than-Memory: SparseGPT, MoD, LeanKV, GGUF, WeightStream ─
_sparse_gpt             = None  # SparseGPTPruner         — --sparse-gpt
_mix_of_depths          = None  # MixtureOfDepths         — --mix-of-depths
_lean_kv_quant          = None  # LeanKVQuant             — --lean-kv
_gguf_loader            = None  # GGUFNativeLoader        — --gguf-loader
_weight_stream          = None  # WeightDecompressStream  — --weight-stream
_shard_loader           = None  # ModelShardLoader        — --shard-loader

# ── Wave 51: Test-Time Compute Scaling ────────────────────────────────────────
_budget_forcing         = None  # BudgetForcingDecoder    — --budget-forcing
_test_time_router       = None  # TestTimeComputeRouter   — --test-time-scale
_dvts_search            = None  # DVTSSearch              — --dvts
_chain_of_draft         = None  # ChainOfDraftSampler     — --chain-of-draft
_coconut_decoder        = None  # CoconutDecoder          — --coconut
_prm_beam_search        = None  # PRMBeamSearch           — --prm-beam
_best_of_n              = None  # BestOfNSampler          — --best-of-n
_self_consistency       = None  # SelfConsistencyVoter    — --self-consistency
_thought_budget_gate    = None  # ThoughtBudgetGate       — --thought-budget
_reasoning_kv           = None  # ReasoningKVManager      — --reasoning-kv
_draft_reasoning        = None  # DraftReasoningVerifier  — --draft-reasoning
_parallel_reasoning     = None  # ParallelReasoningScheduler — --parallel-reasoning

# ── Wave 52: Multi-Modal VLM Efficiency ───────────────────────────────────────
_fast_v_pruner          = None  # FastVPruner             — --fast-v
_vision_zip             = None  # VisionZip               — --vision-zip
_llava_prumerge         = None  # LLaVAPruMerge           — --llava-prumerge
_token_packer           = None  # TokenPacker             — --token-packer
_flash_vstream          = None  # FlashVStream            — --flash-vstream
_dynamic_res            = None  # DynamicResEncoder       — --dynamic-res
_visual_kv_quant        = None  # VisualKVQuant           — --visual-kv-quant
_cross_modal_router     = None  # CrossModalRouter        — --cross-modal
_video_kv_reuse         = None  # VideoKVReuse            — --video-kv-reuse
_vlm_spec_decode        = None  # VLMSpecDecode           — --vlm-spec
_vlm_batch_sched        = None  # VLMBatchScheduler       — --vlm-sched
_img_encoder_cache      = None  # ImageEncoderCache       — --img-encoder-cache

# ── Wave 53: Linear Recurrent Architectures ───────────────────────────────────
_rwkv6_channel_mix      = None  # RWKV6ChannelMix         — --rwkv6
_hawk_rnn               = None  # HawkLinearRNN           — --hawk-rnn
_xlstm_block            = None  # xLSTMBlock              — --xlstm
_ttt_layer              = None  # TTTLinearLayer          — --ttt
_delta_net              = None  # DeltaNetLinear          — --delta-net
_ssm_state_cache        = None  # SSMStateCache           — --ssm-cache
_parallel_scan          = None  # ParallelScanKernel      — --parallel-scan
_ssm_quantizer          = None  # SSMQuantizer            — --ssm-quant
_hybrid_arch_router     = None  # HybridArchRouter        — --hybrid-arch
_hymba_dual             = None  # HymbaDualTrack          — --hymba
_ssm_state_offload      = None  # SSMStateOffload         — --ssm-offload

# ── Wave 54: Deep MoE Efficiency, FlashAttn3, DoubleSparsity, ElasticBatch ───
_shared_expert_moe      = None  # SharedExpertMoE         — --shared-expert
_fine_grained_router    = None  # FineGrainedMoERouter    — --fine-grained-moe
_expert_offloader       = None  # ExpertOffloader         — --expert-offload
_expert_merger          = None  # ExpertMerger            — --expert-merge
_lazy_expert            = None  # LazyExpertLoader        — --lazy-expert
_expert_act_cache       = None  # ExpertActivationCache   — --expert-cache
_flash_attn3            = None  # FlashAttn3Kernel        — --flash-attn3
_double_sparse_attn     = None  # DoubleSparsityAttn      — --double-sparse
_lasp_linear_attn       = None  # LASPLinearAttn          — --lasp
_nacl_cache             = None  # NaCLCache               — --nacl-cache
_kv_migration           = None  # KVMigrationManager      — --kv-migration
_elastic_batch          = None  # ElasticBatchController  — --elastic-batch

# ── Wave 55: Advanced Sampling, Emerging Quantization ────────────────────────
_min_p_sampler          = None  # MinPSampler             — --min-p
_mirostat_sampler       = None  # MirostatSampler         — --mirostat
_eta_cutoff             = None  # EtaCutoffSampler        — --eta-cutoff
_cfg_sampler            = None  # CFGLogitsSampler        — --cfg-guidance
_diverse_beam           = None  # DiverseBeamSampler      — --diverse-beam
_bitnet158              = None  # BitNet158Quantizer      — --bitnet158
_spqr_quant_w55         = None  # SpQRQuantizer (spqr_quant) — --spqr-quant
_omniquant              = None  # OmniQuantizer           — --omniquant
_qsparse                = None  # QSparsifier             — --q-sparse
_fp4_quantizer          = None  # FP4Quantizer            — --fp4-quant
_ada_round              = None  # AdaRoundQuantizer       — --ada-round

# ── Wave 37: Wire Everything In ───────────────────────────────────────────────
# Twelve isolation modules from Waves 33–35 wired into the live request path.
_kvtc_manager           = None  # KVTCManager             — --kvtc
_chunk_kv_manager       = None  # ChunkKVManager          — --chunk-kv
_ssd_saguaro            = None  # SSDSaguaro              — --ssd-saguaro
_speculative_streamer   = None  # SpeculativeStreamer      — --spec-stream
_metal_flash_attn       = None  # MetalFlashAttention     — --metal-flash-attn
_deja_vu_sparse_ffn     = None  # DejaVuSparseFFN         — --deja-vu
_jacobi_decoder         = None  # JacobiDecoder           — --jacobi
_mtp_predictor          = None  # MultiTokenPredictor     — --mtp
_layer_overlap_loader   = None  # LayerOverlapLoader      — --layer-overlap
_chip_profile           = None  # ChipProfile             — auto (startup)
_fused_qkv_proj         = None  # FusedQKVProjection      — --fused-qkv
_pd_disaggregator       = None  # PDDisaggregator         — --pd-disagg
# ── Wave 81: blazing-mode globals ─────────────────────────────────────────────
_blazing_mode: bool     = False  # True when --blazing preset is active
_metal_cache_limit_mb: int = 256  # Override via --blazing (drops to 64 MB)

# ── Wave 27: new inference velocity flags ─────────────────────────────────────
_fused_sampler          = None  # FusedSampler            — --fused-sampler (v10: default on)
_fused_sampler_enabled  = True  # on by default; --no-fused-sampler to disable
_cached_make_sampler: "Any" = None  # cached on first successful import from mlx_lm.sample_utils
_cache_warmup_predictor = None  # CacheWarmupPredictor    — tracks prefix access patterns
_cache_warmup_enabled   = True  # on by default; --no-cache-warmup to disable
_tome_config            = None  # TokenMergingConfig      — --token-merge
_tome_state             = None  # TokenMergingState       — per-request merge maps
# Phase 3: cross-session persistent KV cache
_session_kv_cache    = None   # SessionKVCache | None — set in main() when --session-cache-dir given
# Phase 4: prompt compression settings (active when --compress-prompt is set)
_compress_enabled         = False
_compress_ratio           = 0.5
_compress_min_tokens      = 512
_compress_preserve_tokens = 0   # protect first N words from compression (RadixAttention synergy)

# ── Phase E1: Babbling Suppression (February 2026) ───────────────────────────
# Qwen3 architecture is a confirmed "babbler" — emits filler content after the
# task is complete, wasting 44–89% of decode energy.  Three complementary guards:
#   1. EOS probability monitoring: stop when model "wants" to stop (P(eos) > threshold)
#   2. Grammar terminal state: stop when XGrammar FSM accepts (schema is complete)
#   3. Hard token caps: per-task-type maximum output length
_babbling_suppression: bool    = True   # on by default; --no-babbling-suppression to disable
_babbling_eos_threshold: float = 0.30   # EOS softmax probability threshold
_babbling_min_tokens: int      = 10     # never trigger before this many decode steps

# Per-task-type hard token caps (0 = uncapped for that type).
# Tuned from real Squish output distributions.
_TASK_TOKEN_CAPS: dict = {
    "git_commit":  100,
    "devops_plan": 500,
    "code_review": 200,
    "email_draft": 300,
}

# ── Phase E2: Polynomial GELU approximation ──────────────────────────────────
# For GELU-based models, replace erf-based GELU with x * sigmoid(1.702x) —
# a single fused Metal op that the ANE handles at peak throughput.
# No-op for Qwen3 (already uses SiLU = x * sigmoid(x), already ANE-optimal).
_fast_gelu_enabled: bool = True  # on by default; --no-fast-gelu to disable

# ── Phase E3: Semantic response cache ────────────────────────────────────────
# Bypass the model entirely for semantically repeated queries.
# Per-task-type cosine similarity thresholds and response TTLs.
_semantic_cache = None   # SquishSemanticCache | None — set in main()
_SEMANTIC_CACHE_CONFIG: dict = {
    "git_commit":  {"threshold": 0.95, "ttl_hours": 24},
    "devops_plan": {"threshold": 0.88, "ttl_hours": 168},
    "code_review": {"threshold": 0.92, "ttl_hours": 72},
    "email_draft": {"threshold": 0.85, "ttl_hours": 48},
    "default":     {"threshold": 0.92, "ttl_hours": 48},
}

# ── Phase 3A: Chunked prefill (COMPRESS_PATH long sequences) ─────────────────
_chunk_prefill_enabled   = False  # set in main() via --chunk-prefill
_chunk_prefill_threshold = 512    # min token count to trigger chunking (default 512)
_chunk_prefill_size      = 512    # tokens per chunk (default 512)

# ── Phase 3C: MInference sparse attention ─────────────────────────────────────
_minference_enabled      = False  # set in main() via --minference
_minference_threshold    = 1024   # min seq_len to apply sparse attention (default 1024)

# ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────────
_thinking_budget: int = -1            # -1=unlimited, 0=disable thinking, >0=token limit
_think_close_token_id: int | None = None  # ID of </think> token, resolved at model load
# ── Phase A2: explicit MLX rotating KV cache size ────────────────────────────
_max_kv_size: int | None = None       # None = mlx_lm default (4K); set to extend context
# ── Phase A3: concise output mode ────────────────────────────────────────────
_concise_responses: bool = False      # prepend concision prefix + EOS bias
_CONCISION_PREFIX = (
    "Respond with only the requested output. "
    "No preamble, no explanation, no apologies.\n\n"
)
# ── Phase B: Structured output (XGrammar) ────────────────────────────────────
_grammar_engine: "Any | None" = None       # GrammarEngine instance, set at startup
_structured_output_mode: str = "none"      # "none" | "json" | "json-schema"
_structured_output_schema: "dict | None" = None  # parsed JSON schema (json-schema mode)
_req_tool_schema: "dict | None" = None     # per-request override: tool_choice-activated schema
# ── Phase C: Power & Energy Modes ────────────────────────────────────────────
_power_monitor: "Any | None" = None        # PowerMonitor instance (auto mode only)
_power_mode: str = "performance"           # current effective mode name
# ── Phase 13B: macOS Memory Governor ─────────────────────────────────────────
_memory_governor: "Any | None" = None      # MemoryGovernor instance (macOS only)

# ── Conflict-Resolution Routing (Phase 0) ────────────────────────────────────
# Two exclusive request paths prevent incompatible optimizations firing together:
#
#   COMPRESS_PATH  — word count > _compress_threshold AND compress enabled
#       Uses: LLMLingua → chunked prefill → LazyLLM → EAGLE-3/N-gram draft
#       Skips: exact-match prefix cache (compressed text never matches cache)
#       Cache key: pre-compression token hash (future identical calls still hit)
#
#   PREFIX_PATH    — short or previously-cached prompts (default path)
#       Uses: RadixAttention → EAGLE-3/N-gram → LazyLLM (prefill-only mode)
#       Skips: LLMLingua (would invalidate cache keys)
#
# _inference_backend controls Phase 4 hardware dispatch (mutually exclusive):
#   'mlx-eager'    — standard MLX path (default)
#   'mlx-compiled' — mx.compile fused draft+verify decode kernel (Phase 4A)
#   'ane-disagg'   — Core ML ANE prefill + MLX decode (Phase 4B)
_compress_threshold  = 512          # word-count proxy above which COMPRESS_PATH fires
_inference_backend   = "mlx-eager"  # overridden by --inference-backend in main()

# ── Wave 76: Agentic Tool Registry & MCP Server Map ──────────────────────────
# _agent_registry is populated in main() by register_builtin_tools().
# _mcp_servers maps server_id → MCPClient instance (lazily connected).
_agent_registry: "Any | None" = None   # ToolRegistry | None — set in main()
_mcp_servers: dict = {}                # {server_id: MCPClient}

# ── Phase F: Inference Backend Abstraction ───────────────────────────────────

class _InferenceBackend:
    """Base shim for inference backend dispatch.

    Concrete subclasses override ``generate_stream``.  All generation paths
    are hardware-bound and marked ``# pragma: no cover``.  The ``__init__``
    constructors are testable (no hardware required).
    """

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _MLXEagerBackend(_InferenceBackend):
    """Standard MLX Metal eager execution path.

    Stores a reference to the loaded model and tokenizer so the dispatch
    layer can route ``generate_stream`` calls without global lookups.
    """

    def __init__(self, model: "Any", tokenizer: "Any") -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Route through _generate_tokens instead")


class _MLCBackend(_InferenceBackend):
    """MLC-LLM engine path for large-context requests.

    Probes for ``mlc_llm`` at construction time and sets
    :meth:`is_available` accordingly so callers can gate on its presence.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        try:
            import mlc_llm as _mlc  # noqa: F401,PLC0415
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        """Return ``True`` when ``mlc_llm`` was importable at construction time."""
        return self._available

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("MLC backend not yet wired")


_active_backend: "_InferenceBackend | None" = None  # set in main() when dispatching

# ── Batch scheduler (Phase 2.1 — continuous batching) ───────────────────────
_scheduler       = None  # BatchScheduler | None — set in main() when --batch-scheduler given
_QueueFullError  = None  # QueueFullError class — imported alongside BatchScheduler

# ── Terminal colours & ASCII art ──────────────────────────────────────────────
_TTY: bool = sys.stdout.isatty()
_TTY_ERR: bool = sys.stderr.isatty()

# True only when the terminal reliably renders 24-bit (true) colour.
# Terminals that remap the ANSI palette via a colour profile will corrupt
# 24-bit codes that get quantised down to palette indices.  We guard against
# this by requiring an explicit true-colour signal (COLORTERM env-var or a
# known terminal program) before emitting gradient escape sequences.
# Respects the NO_COLOR convention (https://no-color.org).
def _has_truecolor(tty: bool) -> bool:
    """Return True when the terminal reliably renders 24-bit RGB escape codes."""
    return (
        tty
        and "NO_COLOR" not in os.environ
        and (
            os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
            or os.environ.get("TERM_PROGRAM", "") in (
                "iTerm.app", "WezTerm", "Ghostty", "Hyper", "vscode", "warp",
                "Apple_Terminal",
            )
            or "kitty" in os.environ.get("TERM", "")
            or "direct" in os.environ.get("TERM", "")
            or bool(os.environ.get("FORCE_COLOR", ""))
        )
    )

_TRUE_COLOR:     bool = _has_truecolor(_TTY)
_TRUE_COLOR_ERR: bool = _has_truecolor(_TTY_ERR)

try:
    from squish._term import detect_dark_background as _detect_dark_bg_srv
    _IS_DARK_BG: bool = _detect_dark_bg_srv()
except Exception:  # pragma: no cover
    _IS_DARK_BG = True  # dark fallback if import fails during early startup


class _C:
    """Dark-background 24-bit colour constants.  Empty strings on non-true-colour TTYs."""
    _k = lambda s: s if _TRUE_COLOR else ""  # noqa: E731
    DP  = _k("\033[38;2;88;28;135m")    # deep purple   #581C87
    P   = _k("\033[38;2;124;58;237m")   # purple        #7C3AED
    V   = _k("\033[38;2;139;92;246m")   # violet        #8B5CF6
    L   = _k("\033[38;2;167;139;250m")  # lilac         #A78BFA
    MG  = _k("\033[38;2;192;132;252m")  # med-purple    #C084FC
    PK  = _k("\033[38;2;236;72;153m")   # pink          #EC4899
    LPK = _k("\033[38;2;249;168;212m")  # light pink    #F9A8D4
    T   = _k("\033[38;2;34;211;238m")   # teal          #22D3EE
    LT  = _k("\033[38;2;165;243;252m")  # light teal    #A5F3FC
    G   = _k("\033[38;2;52;211;153m")   # mint green    #34D399
    W   = _k("\033[38;2;248;250;252m")  # near-white    #F8FAFC
    SIL = _k("\033[38;2;180;185;210m")  # silver        #B4B9D2
    DIM = _k("\033[38;2;100;116;139m")  # dim slate     #64748B
    B   = _k("\033[1m")                 # bold
    R   = _k("\033[0m")                 # reset all


class _CLight:
    """Light-background 24-bit colour constants — deeper for contrast on white."""
    _k = lambda s: s if _TRUE_COLOR else ""  # noqa: E731
    DP  = _k("\033[38;2;67;20;105m")    # deeper purple  #431469
    P   = _k("\033[38;2;88;28;135m")    # dark purple    #581C87
    V   = _k("\033[38;2;109;40;217m")   # dark violet    #6D28D9
    L   = _k("\033[38;2;124;58;237m")   # purple         #7C3AED
    MG  = _k("\033[38;2;139;92;246m")   # violet         #8B5CF6
    PK  = _k("\033[38;2;157;23;77m")    # deep pink      #9D174D
    LPK = _k("\033[38;2;219;39;119m")   # pink           #DB2777
    T   = _k("\033[38;2;6;182;212m")    # teal           #06B6D4
    LT  = _k("\033[38;2;8;145;178m")    # dark teal      #0891B2
    G   = _k("\033[38;2;16;185;129m")   # green          #10B981
    W   = _k("\033[38;2;15;23;42m")     # near-black     #0F172A
    SIL = _k("\033[38;2;71;85;105m")    # slate          #475569
    DIM = _k("\033[38;2;51;65;85m")     # dim slate      #334155
    B   = _k("\033[1m")                 # bold
    R   = _k("\033[0m")                 # reset all


# Select dark or light palette based on detected terminal background.
if not _IS_DARK_BG:  # pragma: no cover
    _C = _CLight  # type: ignore[misc]


def _gradient(text: str, stops: list) -> str:
    """Interpolate a left-to-right RGB gradient across *text* (true-colour TTY only)."""
    if not _TRUE_COLOR or not text:
        return text
    n = len(text)
    k = len(stops) - 1
    out: list[str] = []
    for i, ch in enumerate(text):
        t = i / max(n - 1, 1)
        seg = min(int(t * k), k - 1)
        frac = t * k - seg
        r1, g1, b1 = stops[seg]
        r2, g2, b2 = stops[seg + 1]
        r = int(r1 + (r2 - r1) * frac)
        g = int(g1 + (g2 - g1) * frac)
        b = int(b1 + (b2 - b1) * frac)
        out.append(f"\033[38;2;{r};{g};{b}m{ch}")
    out.append("\033[0m")
    return "".join(out)


# Purple → pink → teal gradient used for the big logo and accent lines
_LOGO_GRAD = [
    ( 88,  28, 135),   # deep purple
    (124,  58, 237),   # purple
    (139,  92, 246),   # violet
    (192, 100, 220),   # lavender-pink
    (236,  72, 153),   # pink
    ( 34, 211, 238),   # teal
]


def _cprint(color: str, label: str, value: str = "", end: str = "\n") -> None:
    """Print a coloured label + plain value line."""
    R = _C.R
    if value:
        print(f"  {color}{label}{R}  {_C.W}{value}{R}", end=end)
    else:
        print(f"  {color}{label}{R}", end=end)


def _ok(msg: str) -> None:
    """Print a success tick line."""
    print(f"  {_C.G}✓{_C.R}  {_C.W}{msg}{_C.R}")


def _info(label: str, value: str) -> None:
    """Print a key → value config line."""
    print(f"  {_C.L}◈{_C.R}  {_C.DIM}{label:<18}{_C.R}{_C.W}{value}{_C.R}")


def _warn(msg: str) -> None:
    """Print a yellow-ish warning line."""
    print(f"  {_C.PK}⚠{_C.R}  {_C.LPK}{msg}{_C.R}")


def _section(title: str) -> None:
    """Print a dimmed section divider."""
    print(f"  {_C.DIM}{'─' * 52}{_C.R}")
    if title:
        print(f"  {_C.MG}{title}{_C.R}")


def _print_optimization_status() -> None:
    """Print a compact one-line-per-module optimization status table.

    Called once before ``uvicorn.run()`` so users can see which performance
    modules are active and which fell back at a glance.
    """
    # Ensure RadixTree is loaded before we read _prefix_cache._maxsize.
    # This is a no-op when the test suite has already patched _prefix_cache.
    _init_prefix_cache()
    rows: list[tuple[str, bool, str]] = [
        ("fused-sampler",  _fused_sampler_enabled and _fused_sampler is not None,
         "single-pass temperature+top-k+top-p decode kernel"),
        ("chunk-prefill",  _chunk_prefill_enabled,
         f"long-prompt chunking  (threshold={_chunk_prefill_threshold}t)"),
        ("cache-warmup",   _cache_warmup_predictor is not None,
         "predictive KV prefix pre-warming"),
        ("metal-jit-warmup", _state.model is not None,
         "forward-pass forced before first request"),
        ("prefix-cache",   _prefix_cache._maxsize > 0,
         f"exact-match response cache  (cap={_prefix_cache._maxsize})"),
        ("paged-kv",       _paged_kv_cache is not None,
         "block-table KV reuse"),
        ("flash-attn3",    _flash_attn3 is not None,
         "Flash Attention 3 kernel"),
    ]
    _section("Optimization modules")
    for name, active, desc in rows:
        mark  = f"{_C.G}✓{_C.R}" if active else f"{_C.DIM}✗{_C.R}"
        label = f"{_C.W}{name:<20}{_C.R}" if active else f"{_C.DIM}{name:<20}{_C.R}"
        note  = f"{_C.DIM}{desc}{_C.R}" if active else f"{_C.DIM}disabled{_C.R}"
        print(f"  {mark}  {label}{note}")
    print()


def _print_banner() -> None:
    """Print the full ASCII-art startup banner."""
    R  = _C.R
    V  = _C.V;  L  = _C.L;  MG = _C.MG
    T  = _C.T;  PK = _C.PK
    W  = _C.W;  SIL = _C.SIL; DIM = _C.DIM

    print()

    if _TTY:
        # ── Squished character (clamp pressing cube flat — 1-row body = max squish) ──
        # Left connector bars = teal (inputs), right = pink (outputs), body = violet
        print(f"        {SIL}           ╤           {R}")
        print(f"        {SIL}   ╔═══════╧═══════╗   {R}")
        print(f"       {T}════{R}{V}╫{R}{W}   ◕  {R}{MG}˶‿˶{R}{W}  ◕   {R}{V}╫{R}{PK}════{R}")
        print(f"        {V}   ╚═══════════════╝{R}")
        print(f"            {DIM}═══════════════{R}")
        print(f"              {L}✦{R}    {PK}✦{R}    {L}✦{R}")
        print()

        # ── SQUISH gradient logo (box-drawing block font) ─────────────────────
        logo_lines = [
            " ██████╗   ██████╗  ██╗   ██╗  ██╗   ██████╗  ██╗  ██╗",
            "██╔════╝  ██╔═══██╗ ██║   ██║  ██║  ██╔════╝  ██║  ██║",
            "╚█████╗   ██║   ██║ ██║   ██║  ██║  ╚█████╗   ███████║",
            " ╚═══██╗  ██║▄▄ ██║ ██║   ██║  ██║   ╚═══██╗  ██╔══██║",
            "██████╔╝  ╚██████╔╝ ╚██████╔╝  ██║  ██████╔╝  ██║  ██║",
            "╚═════╝    ╚══▀▀═╝   ╚═════╝   ╚═╝  ╚═════╝   ╚═╝  ╚═╝",
        ]
        for line in logo_lines:
            print(f"  {_gradient(line, _LOGO_GRAD)}{R}")
        print()

        sub = "✦  Squish it. Run it. Go.  ✦"
        print(f"            {_gradient(sub, _LOGO_GRAD)}{R}")
        print(f"  {DIM}{'─' * 56}{R}")
    else:
        # Plain-text fallback for non-TTY environments
        print("*** SQUISH — Squish it. Run it. Go.   ***")
        print("-" * 48)

    print()


# ── Verbose inference tracing ─────────────────────────────────────────────────
_trace: bool       = False   # set True by --trace in main()
_trace_tokens: bool = False  # set True by --trace-tokens in main()
_trace_file = None           # IO | None — file handle opened by --trace-file


def _tlog(msg: str) -> None:
    """Write a timestamped trace line to stderr (and _trace_file when set)."""
    _ke = lambda s: s if _TRUE_COLOR_ERR else ""  # noqa: E731
    ts  = f"{_ke(_C.MG)}[{time.strftime('%H:%M:%S')}]{_ke(_C.R)}"
    tag = f"{_ke(_C.V)}SQUISH{_ke(_C.R)}"
    line_color = f"{ts} {tag}  {_ke(_C.W)}{msg}{_ke(_C.R)}"
    line_plain = f"[SQUISH {time.strftime('%H:%M:%S')}] {msg}"
    print(line_color, file=sys.stderr, flush=True)
    if _trace_file is not None:
        try:
            _trace_file.write(line_plain + "\n")
            _trace_file.flush()
        except Exception:
            pass

# ── Model state ──────────────────────────────────────────────────────────────

class _ModelState:
    model        = None
    tokenizer    = None
    model_name   = ""
    loaded_at    = 0.0
    load_time_s  = 0.0
    loader_tag   = "squish"
    requests     = 0
    tokens_gen   = 0
    # Real-time performance tracking
    inflight     = 0          # concurrent requests in flight
    _lock        = threading.Lock()
    # Rolling window: last 20 (tps, ttft_s) samples
    _tps_window: collections.deque = None

    def __init__(self):
        self._tps_window = collections.deque(maxlen=20)

    def record_completion(self, n_tokens: int, duration_s: float, ttft_s: float) -> None:
        tps = n_tokens / max(duration_s, 1e-6)
        with self._lock:
            self._tps_window.append((tps, ttft_s))
            self.tokens_gen += n_tokens
            self.requests   += 1

    @property
    def avg_tps(self) -> float:
        with self._lock:
            items = list(self._tps_window)
        return sum(t for t, _ in items) / len(items) if items else 0.0

    @property
    def avg_ttft(self) -> float:
        with self._lock:
            items = list(self._tps_window)
        return sum(f for _, f in items) / len(items) if items else 0.0

_state = _ModelState()
_API_KEY: str | None = None          # set from --api-key at startup
_bearer  = HTTPBearer(auto_error=False)
_server_args: dict = {}              # CLI args captured at startup; exposed via /debug-info

# ── Draft model state (speculative decoding) ─────────────────────────────────

class _DraftState:
    model      = None
    tokenizer  = None
    model_dir  = ""
    generator  = None   # SpeculativeGenerator instance (created after both models load)
    eagle_head = None   # EagleDraftHead instance (Phase 1B)

_draft = _DraftState()

# ── Prefix cache + RadixTree (Phase 1.4 / Phase 2B) ─────────────────────────
# Exact-match text response cache backed by RadixTree.
# RadixTree is a drop-in replacement for the old _PrefixCache:
#   • get() / put() / hits / size / _maxsize / clear() — same interface
#   • find_prefix(token_ids) / insert_prefix(token_ids, block_refs) — new (Phase 2B)
# When --paged-attention is enabled the server also records KV block refs so
# future requests with matching token prefixes can skip prefill entirely.
#
# Wave 78: import deferred until first use (_init_prefix_cache) to save ~16 ms
# from `import squish.server`.  _PrefixCache is set by _init_prefix_cache and
# exposed via module __getattr__ so test code that accesses _srv._PrefixCache
# before any server function is called still gets the real class.

_RadixTree = None    # populated by _init_prefix_cache()
_prefix_cache = None  # populated by _init_prefix_cache()
# NOTE: _PrefixCache is NOT pre-set in module __dict__; access triggers __getattr__
#       which calls _init_prefix_cache() and then returns the class.


def _init_prefix_cache(maxsize: int = 512) -> None:
    """Lazy-load RadixTree and create the module-level prefix cache instance.

    This is idempotent — subsequent calls are a no-op if the cache is already
    initialised (or has been replaced by a test mock via patch.multiple).
    """
    global _RadixTree, _prefix_cache
    if _prefix_cache is not None:
        return
    from squish.kv.radix_cache import RadixTree as _RT  # noqa: PLC0415
    _RadixTree = _RT
    # Set _PrefixCache in the module namespace for backward-compat test access
    globals()["_PrefixCache"] = _RT
    _prefix_cache = _RT(maxsize=maxsize)


def __getattr__(name: str):
    """Module-level __getattr__: lazily expose _PrefixCache before first init."""
    if name == "_PrefixCache":
        _init_prefix_cache()
        return globals().get("_PrefixCache")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _sample_mx(logits_row, temperature: float, top_p: float) -> int:  # pragma: no cover
    """
    Sample a single token id from an MLX logits vector.

    Parameters
    ----------
    logits_row  : mx.array  shape (vocab_size,)
    temperature : float — <= 0 means greedy argmax
    top_p       : float — nucleus sampling probability mass (1.0 = disabled)

    Returns
    -------
    int token id
    """
    import mlx.core as mx
    import numpy as np
    if temperature <= 0.0 or temperature < 1e-5:
        return int(mx.argmax(logits_row).item())
    probs_np = np.array(mx.softmax(logits_row.astype(mx.float32) / temperature, axis=-1))
    if top_p < 1.0:
        idx    = np.argsort(-probs_np)
        cumsum = np.cumsum(probs_np[idx])
        cutoff = min(int((cumsum <= top_p).sum()) + 1, len(idx))
        mask   = np.zeros_like(probs_np)
        mask[idx[:max(1, cutoff)]] = 1.0
        probs_np = probs_np * mask
        probs_np /= probs_np.sum() + 1e-9
    return int(np.random.choice(len(probs_np), p=probs_np))


def _check_auth(creds: HTTPAuthorizationCredentials | None) -> None:
    """Raise 401 if an API key is configured and the request doesn't match.

    Uses hmac.compare_digest to prevent timing-oracle attacks.
    """
    if _API_KEY is None:
        return
    if creds is None or not hmac.compare_digest(
        creds.credentials.encode(), _API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@functools.lru_cache(maxsize=4)
def _system_fingerprint(model_name: str | None, loaded_at: float) -> str:
    """Stable fingerprint derived from model name + load timestamp.

    Cached with lru_cache so the MD5 is only computed once per unique
    (model_name, loaded_at) pair — not on every streamed token.
    """
    return "sq-" + hashlib.md5(
        f"{model_name}{loaded_at}".encode()
    ).hexdigest()[:8]


# ── Wave 81: blazing-mode helpers ─────────────────────────────────────────────

def _has_quantized_layers(model: "Any") -> bool:
    """Return True if the model has at least one quantized linear layer.

    Checks the first three transformer blocks for an object that carries a
    ``bits`` integer attribute — the signature of ``mlx_lm.QuantizedLinear``.
    Works for any model whose layers are exposed as ``model.model.layers``
    or ``model.layers`` (the standard mlx_lm layout).

    Parameters
    ----------
    model : loaded mlx_lm model or any object with a ``layers`` attribute

    Returns
    -------
    bool — True = at least one quantized layer found; False = no quantization
    detected (e.g. BF16/FP16).
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None) or getattr(model, "layers", None)
    if not layers:
        return False
    for layer in layers[:3]:  # check first 3 transformer blocks only
        for val in vars(layer).values():
            if hasattr(val, "bits") and isinstance(getattr(val, "bits", None), int):
                return True
            for sub in vars(val).values() if hasattr(val, "__dict__") else ():
                if hasattr(sub, "bits") and isinstance(getattr(sub, "bits", None), int):
                    return True
    return False


def _blazing_preset_defaults(
    args: "Any",
    chip_profile: "Any | None" = None,
    ram_gb: float = 0.0,
) -> "Any":
    """Apply Wave-81 blazing-mode defaults to *args* (mutated in-place).

    Called when ``--blazing`` is passed on the CLI.  Applies the minimum set
    of flags needed for sub-3s TTFT with 7/8B models on 16 GB M3:

    * INT2 asymmetric KV cache       (--agent-kv)
    * TTFT-optimised chunk-prefill   (--chunk-prefill-size 128)
    * Fast-GELU approximation        (--fast-gelu)
    * Tight Metal buffer pool        (_metal_cache_limit_mb → 64 MB)
    * Clamp max-KV-context to 4096   (frees ~3 GB vs 32 K default)

    Parameters
    ----------
    args        : argparse Namespace (or any object with setattr)
    chip_profile: optional ``ChipProfile`` from ``ChipDetector.detect()``
    ram_gb      : detected system RAM in GB (0 = unknown)

    Returns
    -------
    The same *args* object, with fields mutated.
    """
    # ── KV cache: INT2 asymmetric (6× footprint reduction vs FP16) ──────────
    setattr(args, "agent_kv", True)

    # ── Chunked prefill: TTFT-optimised size ────────────────────────────────
    ttft_chunk = 128
    if chip_profile is not None and hasattr(chip_profile, "recommended_chunk_prefill_ttft"):
        ttft_chunk = chip_profile.recommended_chunk_prefill_ttft
    setattr(args, "chunk_prefill_size", ttft_chunk)
    setattr(args, "no_chunk_prefill", False)  # ensure chunking is on

    # ── Fast-GELU approximation (Wave 28): x·sigmoid(1.702x) — no change in
    #    perceptible output quality but avoids trigonometric exact computation ─
    setattr(args, "fast_gelu", True)

    # ── Clamp KV context: 4096 is plenty for interactive chat; unclamped
    #    context on 16 GB eats 500 MB+ per request ────────────────────────────
    current_max_kv = getattr(args, "max_kv_size", None)
    if current_max_kv is None or current_max_kv > 4096:
        setattr(args, "max_kv_size", 4096)

    # ── Metal allocator pool: 64 MB covers normal weight-loading churn while
    #    releasing stale buffers aggressively on a 16 GB system ───────────────
    setattr(args, "_blazing_metal_cache_mb", 64)

    return args


def _configure_blazing_mode(args: "Any") -> None:
    """Activate Wave-81 blazing mode when ``--blazing`` was passed.

    Sets ``_blazing_mode`` and ``_metal_cache_limit_mb`` globals, then
    delegates to :func:`_blazing_preset_defaults` for individual flag
    expansion.  Must be called in ``main()`` *before* model loading.
    """
    if not getattr(args, "blazing", False):
        return

    global _blazing_mode, _metal_cache_limit_mb  # noqa: PLW0603
    _blazing_mode = True

    ram_gb: float = 0.0
    try:
        from squish.hardware.chip_detector import ChipDetector as _BlazCD  # noqa: PLC0415
        ram_gb = _BlazCD.detect_ram_gb()
    except Exception:  # noqa: BLE001
        pass

    _blazing_preset_defaults(args, chip_profile=_chip_profile, ram_gb=ram_gb)

    limit_mb: int = int(getattr(args, "_blazing_metal_cache_mb", 64))
    _metal_cache_limit_mb = limit_mb

    _info(
        "blazing",
        (
            f"active  INT2-KV  chunk={getattr(args, 'chunk_prefill_size', 128)}"
            f"  kv-max={getattr(args, 'max_kv_size', 4096)}"
            f"  metal-cache={limit_mb}MB  two-pass-warmup=on"
        ),
    )



# Match on the first 200 chars of the prompt to classify the task.
# Only used to select the right token cap and semantic cache threshold.
_TASK_TYPE_KEYWORDS: dict = {
    "git_commit":  ("write a commit", "commit message", "git commit",
                    "summarize this diff", "write commit", "generate a commit"),
    "devops_plan": ("devops", "kubernetes", "deploy", "infrastructure",
                    "k8s", "argo ", "helm ", "kubectl", "ci/cd"),
    "code_review": ("review this code", "code review", "review the following",
                    "what's wrong with", "find bugs in", "critique this"),
    "email_draft": ("write an email", "draft an email", "email draft",
                    "compose an email", "write a message to"),
}


def _detect_task_type(prompt: str) -> str:
    """Return a task-type key by scanning the first 200 chars of *prompt*."""
    lower = prompt[:200].lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return task_type
    return "default"


# ── Phase E2: Polynomial GELU activation patch ────────────────────────────────


def _apply_fast_gelu(model_dir: str) -> None:  # pragma: no cover
    """
    Replace erf-based GELU activations with *x·sigmoid(1.702x)* — a single
    fused Metal op that the ANE executes at peak throughput.

    Skipped automatically for SiLU/SwiGLU models (Qwen3, LLaMA) because
    their activation is already ``x·sigmoid(x)``, which IS ANE-optimal.
    Only applied when the model config reports a GELU-family ``hidden_act``.
    """
    import json
    try:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            return
        cfg = json.loads(config_path.read_text())
        hidden_act = cfg.get("hidden_act", cfg.get("hidden_activation", "")).lower()
        # SiLU / SwiGLU: already x*sigmoid(x) → no-op
        if not hidden_act or hidden_act in ("silu", "swish", "swiglu"):
            return
        # Only patch GELU-family activations
        if "gelu" not in hidden_act:
            return
        import mlx.core as mx
        import mlx.nn as nn

        def _fast_gelu_fn(x: "mx.array") -> "mx.array":
            """x · σ(1.702x)  — single fused Metal multiply+sigmoid."""
            return x * mx.sigmoid(1.702 * x)

        patched = 0
        for layer in getattr(_state.model, "layers", []):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            for attr in ("act", "act_fn", "activation_fn", "activation"):
                current = getattr(mlp, attr, None)
                if current is nn.gelu or current is getattr(nn, "gelu_approx", None):
                    setattr(mlp, attr, _fast_gelu_fn)
                    patched += 1
        if patched > 0:
            _info("fast-gelu",
                  f"patched {patched} FFN activation layers  "
                  f"({hidden_act} → x·sigmoid(1.702x))")
    except Exception:
        pass   # never block startup on activation patching


def load_model(model_dir: str, compressed_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """Load the Squish compressed model into global state.

    On Apple Silicon (macOS + MLX) the existing MLX-backed compressed_loader
    is used.  On Linux / CUDA / CPU the new PyTorch compressed loader is used
    when the compressed_dir contains a npy-dir (``tensors/`` sub-directory),
    otherwise ``transformers.AutoModelForCausalLM.from_pretrained`` is called
    directly for uncompressed BF16 models.
    """
    import sys as _sys

    t0 = time.perf_counter()
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading model:{_C.R}  {_C.W}{compressed_dir}{_C.R}")

    if _sys.platform == "darwin":
        # ── Apple Silicon path: MLX compressed loader ─────────────────────
        try:
            from .quant.compressed_loader import load_compressed_model as _load_compressed_model
        except ImportError:
            from squish.quant.compressed_loader import load_compressed_model as _load_compressed_model

        model, tokenizer, stats = _load_compressed_model(
            model_dir    = model_dir,
            npz_path     = compressed_dir,
            verbose      = verbose,
            return_stats = True,
        )
        loader_tag = stats.get("loader", "squish")
    else:
        # ── Linux / CUDA / CPU path ────────────────────────────────────────
        compressed_path = Path(compressed_dir)
        _is_npy_dir = (
            compressed_path.is_dir()
            and (
                (compressed_path / "tensors").is_dir()
                or any(compressed_path.glob("*__q4a.npy"))
                or any(compressed_path.glob("*__pt.npy"))
            )
        )

        if _is_npy_dir:
            # Load squish npy-dir via the torch loader
            try:
                from .compressed_loader_torch import load_compressed_model_torch
            except ImportError:
                from squish.compressed_loader_torch import load_compressed_model_torch

            from squish.backend import BE
            model, tokenizer = load_compressed_model_torch(
                npy_dir   = compressed_dir,
                model_dir = model_dir,
                device    = BE.device,
                verbose   = verbose,
            )
            loader_tag = "squish-torch"
        else:
            # Fall back: load BF16 / FP16 model directly via transformers
            from squish.backend import BE
            model, tokenizer = BE.load_model(model_dir)
            loader_tag = "transformers"

    elapsed = time.perf_counter() - t0

    _state.model      = model
    _state.tokenizer  = tokenizer
    _state.model_name = Path(compressed_dir).name
    _state.loaded_at  = time.time()

    _state.load_time_s = elapsed
    _state.loader_tag  = loader_tag
    if verbose:
        _ok(f"Model ready  ({elapsed:.2f}s  loader={loader_tag})")

    # Warn before warm-up so the user understands any extra delay on first run
    if loader_tag.startswith("npy-dir"):
        _warn(
            "First-run: Vectro weight cache not yet built.  "
            "Dequantizing INT4 → float16 and writing finalized cache "
            "(one-time cost, ~10-30s).  Future starts will load in ~3-5s."
        )

    _cap_metal_cache(verbose=verbose)
    _warmup_model(verbose=verbose)


def load_mlx_model(mlx_model_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """
    Load a native mlx_lm model directory directly via ``mlx_lm.load()``.

    This is the memory-efficient path: INT4/INT8 quantized mlx_lm models
    keep weights quantized in Metal (≈4-5 GB for 8B INT4) rather than
    dequantizing to BF16 at load time (≈15 GB).

    Use after converting with::

        python3 -m mlx_lm.convert \\
            --hf-path  <bf16-model-dir> \\
            --mlx-path <mlx-int4-model-dir> \\
            -q --q-bits 4

    Parameters
    ----------
    mlx_model_dir : path to the mlx_lm-format quantized model directory
    """
    import mlx_lm
    t0 = time.perf_counter()
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading mlx_lm model:{_C.R}  {_C.W}{mlx_model_dir}{_C.R}")

    model, tokenizer = mlx_lm.load(mlx_model_dir)
    elapsed = time.perf_counter() - t0

    _state.model      = model
    _state.tokenizer  = tokenizer
    _state.model_name = Path(mlx_model_dir).name
    _state.loaded_at  = time.time()
    _state.load_time_s = elapsed
    _state.loader_tag  = "mlx_lm"
    if verbose:
        _ok(f"Model ready  ({elapsed:.2f}s  loader=mlx_lm)")

    _cap_metal_cache(verbose=verbose)
    _warmup_model(verbose=verbose)


def _cap_metal_cache(verbose: bool = False, limit_mb: int | None = None) -> None:  # pragma: no cover
    """
    Cap the MLX Metal allocator's buffer pool after model load.

    By default MLX keeps an unbounded Metal buffer cache for reuse.  After
    the model is fully loaded and eval'd, this cache can hold gigabytes of
    stale buffers.  Capping it to ``limit_mb`` MB frees that memory back to
    the OS without affecting inference performance (the cache is only used
    for *new* allocations, not existing model weights).

    When ``limit_mb`` is None the global ``_metal_cache_limit_mb`` is used
    (256 MB normally; 64 MB in ``--blazing`` mode).
    """
    if limit_mb is None:
        limit_mb = _metal_cache_limit_mb
    try:
        import gc

        import mlx.core as mx
        gc.collect()
        # eval outstanding lazy ops so nothing is unexpectedly freed
        mx.eval(())
        limit_bytes = limit_mb * 1024 * 1024
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(limit_bytes)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(limit_bytes)
            if verbose:
                print(f"  {_C.DIM}◈  Metal buffer cache capped at {limit_mb} MB{_C.R}")
        gc.collect()
    except Exception:
        pass


def _warmup_model(verbose: bool = False) -> None:  # pragma: no cover
    """Run a short inference pass to force Metal JIT kernel compilation at startup.

    ``mx.compile()`` defers Metal kernel compilation to first real use.  Running
    one ``mlx_lm.stream_generate`` call here forces all relevant Metal kernels —
    including the prefill and KV-cache decode kernels — to compile before the
    first user request, eliminating the 2-5s cold-compile penalty on TTFT.

    Wave 81: Two-pass warmup.  Pass 1 compiles the single-token decode path
    (``max_tokens=1``); pass 2 uses a 33-token prompt to trigger the chunked-
    prefill kernel (chunk boundary compile path) rather than waiting for the
    first real user request to hit it.

    Falls back to a bare ``model(dummy_input)`` call when mlx_lm is unavailable
    (e.g. the Linux/CUDA path or test environments).
    """
    # Guard before the mlx import so tests (and the real server on Linux) return
    # cleanly without triggering an ImportError-based _warn when no model is set.
    if _state.model is None:
        return
    try:
        import mlx.core as mx
        t0 = time.perf_counter()

        # ── Primary: warm up via mlx_lm.stream_generate so the exact code path
        # used during real inference — prefill graph, KV-cache decode graph,
        # and sampler — is compiled here rather than on the first user request.
        try:
            import mlx_lm as _wup_mlx_lm
            _wup_kwargs: dict = {"max_tokens": 1}
            try:
                from mlx_lm.sample_utils import make_sampler as _wup_make_sampler
                _wup_kwargs["sampler"] = _wup_make_sampler(temp=0.0)
            except (ImportError, TypeError):
                _wup_kwargs["temp"] = 0.0

            # ── Pass 1: single-token decode path (compile short decode graph) ─
            _wup_prompt = "Hello"
            for _ in _wup_mlx_lm.stream_generate(
                _state.model, _state.tokenizer, _wup_prompt, **_wup_kwargs
            ):
                pass
            elapsed_p1 = time.perf_counter() - t0

            # ── Pass 2 (Wave 81): 33-token prompt forces the chunked-prefill
            # kernel to compile.  Pass 2 only runs when --blazing mode is on
            # (or when chunk-prefill with small chunk size is active) so we
            # avoid doubling startup time for normal users.
            p2_elapsed = 0.0
            if _blazing_mode or _chunk_prefill_size <= 128:
                _p2_prompt = " ".join(["word"] * 33)  # 33 tokens ≈ 1 chunk
                _p2_t0 = time.perf_counter()
                for _ in _wup_mlx_lm.stream_generate(
                    _state.model, _state.tokenizer, _p2_prompt, **_wup_kwargs
                ):
                    pass
                p2_elapsed = time.perf_counter() - _p2_t0

            elapsed = time.perf_counter() - t0
            if verbose:
                p2_note = f"  +chunked-prefill={p2_elapsed*1000:.0f}ms" if p2_elapsed > 0 else ""
                _ok(
                    f"Metal JIT warm-up  ({elapsed_p1 * 1000:.0f}ms decode"
                    f"{p2_note}  total={elapsed * 1000:.0f} ms)"
                    f"  path=stream_generate"
                )
            return
        except Exception:
            pass  # fall through to bare forward pass below

        # ── Fallback: bare single-token forward pass (no mlx_lm available) ────
        bos_id = None
        if _state.tokenizer is not None:
            bos_id = getattr(_state.tokenizer, "bos_token_id", None)
        bos_id = int(bos_id) if bos_id is not None else 1
        dummy_input = mx.array([[bos_id]])
        logits = _state.model(dummy_input)
        mx.eval(logits)
        del logits
        elapsed = time.perf_counter() - t0
        if verbose:
            _ok(f"Metal JIT warm-up  ({elapsed * 1000:.0f} ms)  path=forward-pass")
    except Exception as _e:
        if verbose:
            _warn(f"[warmup] Skipped: {_e}")


def load_draft_model(draft_model_dir: str, draft_compressed_dir: str = "",  # pragma: no cover
                     verbose: bool = True) -> None:
    """Load the small draft model used for speculative decoding."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from squish.speculative import load_draft_model as _load_draft
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading draft model:{_C.R}  {_C.W}{draft_model_dir}{_C.R}")
    draft_m, draft_tok = _load_draft(
        draft_model_dir,
        draft_compressed_dir or (draft_model_dir + "-compressed"),
        verbose=verbose,
    )
    _draft.model     = draft_m
    _draft.tokenizer = draft_tok
    _draft.model_dir = draft_model_dir
    if verbose:
        _ok("Draft model ready")

    # Build the SpeculativeGenerator now that both models are loaded
    _rebuild_spec_gen()


def load_eagle_head(head_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """Load an EAGLE-3 draft head and wire it into the SpeculativeGenerator."""
    from squish.speculative import EagleDraftHead
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading EAGLE-3 head:{_C.R}  {_C.W}{head_dir}{_C.R}")
    _draft.eagle_head = EagleDraftHead.from_dir(head_dir, _state.model, verbose=verbose)
    if verbose:
        _ok("EAGLE-3 head ready")
    _rebuild_spec_gen()


def _rebuild_spec_gen() -> None:  # pragma: no cover
    """(Re-)create the SpeculativeGenerator from current target + draft state."""
    if _state.model is None:
        _draft.generator = None
        return
    # Require at least one draft source (neural draft model OR EAGLE head)
    if _draft.model is None and _draft.eagle_head is None:
        _draft.generator = None
        return
    from squish.speculative import SpeculativeGenerator
    _draft.generator = SpeculativeGenerator(
        _state.model, _state.tokenizer,
        draft_model=_draft.model, draft_tokenizer=_draft.tokenizer,
        eagle_head=_draft.eagle_head,
    )


# ── Token generation ─────────────────────────────────────────────────────────

def _apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer,
    tools: list[dict] | None = None,
) -> str:
    """Apply chat template if available, fall back to manual formatting.

    When *tools* is provided and the tokenizer supports native tool calling
    (Qwen3, Llama-3.1+), the tools list is passed directly so the model uses
    its trained tool-calling format (e.g. ``<tool_call>`` tags for Qwen3)
    rather than a manually-injected system-prompt JSON schema.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        # Try native tool calling first (Qwen3 / HF models with tools support)
        if tools:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tools                 = tools,
                    tokenize              = False,
                    add_generation_prompt = True,
                )
            except Exception:
                pass  # tokenizer doesn't support tools= → fall through
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = True,
            )
        except Exception:
            pass

    # Manual fallback: Qwen / ChatML format
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _count_tokens(text: str) -> int:
    """Count tokens using the loaded tokenizer. Falls back to word-split estimate."""
    tok = _state.tokenizer
    if tok is None:
        return len(text.split())
    try:
        return len(tok.encode(text))
    except Exception:
        return len(text.split())


def _get_stop_ids(stop: list[str] | str | None) -> list[list[int]]:
    """Convert stop string(s) to lists of token IDs."""
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    tok = _state.tokenizer
    result = []
    for s in stop:
        try:
            ids = tok.encode(s, add_special_tokens=False)
            if ids:
                result.append(ids)
        except Exception:
            pass
    return result


def _build_tool_union_schema(tools: list[dict]) -> dict:
    """Build a minimal JSON schema that enforces a valid tool call object.

    Used by tool_choice="required" or tool_choice={"type":"function","function":{"name":X}}
    to grammar-constrain generation to syntactically valid JSON tool call payloads.
    """
    names = [
        t.get("function", {}).get("name", "")
        for t in tools
        if t.get("function", {}).get("name")
    ]
    return {
        "type": "object",
        "properties": {
            "name": (
                {"type": "string", "enum": names}
                if names else {"type": "string"}
            ),
            "parameters": {"type": "object"},
        },
        "required": ["name"],
    }


def _generate_tokens(  # pragma: no cover
    prompt: str,
    max_tokens: int    = 512,
    temperature: float = 0.7,
    top_p: float       = 0.9,
    stop: list[str] | str | None = None,
    seed: int | None   = None,
    use_cache: bool    = True,
):
    """
    Stream (token_text, finish_reason_or_None) tuples from the MLX model.
    finish_reason is 'stop' (eos hit or stop sequence matched) or
    'length' (max_tokens exhausted).

    Dispatch priority:
      1. Prefix cache (exact-match, deterministic prompts only)
      2. Speculative decoding  (when draft model loaded + temp > 0)
      3. mlx_lm.stream_generate  (mlx_lm >= 0.12)
      4. Manual sampling loop  (fallback)
    """
    model     = _state.model
    tokenizer = _state.tokenizer
    stop_ids  = _get_stop_ids(stop)
    eos_id    = getattr(tokenizer, "eos_token_id", None) or 151645

    # ── Phase E: task-type classification ────────────────────────────────────
    # Detect once per request; used for babbling suppression caps and
    # semantic cache threshold selection.
    _task_type = _detect_task_type(prompt)

    # ── Phase E3: Semantic response cache lookup ──────────────────────────────
    # Check BEFORE any model work.  A warm cache hit returns in <20 ms.
    if _semantic_cache is not None:
        try:
            with _trace_span("gen.semantic_cache"):
                _cached_response = _semantic_cache.lookup(prompt, _task_type)
            if _cached_response is not None:
                for _ch in _cached_response:
                    yield _ch, None
                yield "", "stop"
                return
        except Exception:
            pass  # never block generation on cache lookup failure

    # ── Phase 4: prompt compression ───────────────────────────────────────────
    # Compress long prompts before tokenization to reduce prefill cost.
    # Only applied when --compress-prompt is set and the prompt meets the
    # minimum length threshold.
    #
    # CONFLICT RESOLUTION (LLMLingua ↔ DiskKVCache / prefix cache):
    # Cache keys must use the *original* (pre-compression) prompt so that a
    # future identical request hits the cache even when compression was applied.
    # We capture _orig_prompt NOW, then route based on prompt length.
    _orig_prompt = prompt         # pre-compression canonical text for all cache keys
    _on_compress_path = False     # True → COMPRESS_PATH; False → PREFIX_PATH

    if _compress_enabled:
        _word_count = len(prompt.split())
        if _word_count >= _compress_min_tokens:
            _on_compress_path = True
            try:
                with _trace_span("gen.compress", words=_word_count, ratio=_compress_ratio):
                    from squish.prompt_compressor import compress as _compress_fn
                    prompt = _compress_fn(
                        prompt,
                        ratio=_compress_ratio,
                        # preserve_tokens protects the fixed system-prompt prefix from
                        # compression so that RadixAttention still hits on that prefix
                        # for PREFIX_PATH requests (LLMLingua ↔ RadixAttention synergy).
                        # Controlled by --compress-preserve-tokens (default 0 = disabled).
                        preserve_tokens=_compress_preserve_tokens,
                    )
            except Exception:
                pass  # never block generation on compression failure

    # ── Trace: log request entry ───────────────────────────────────────────────
    _rid = uuid.uuid4().hex[:8]          # short per-request ID for log correlation
    _prompt_tokens_approx = len(prompt.split())
    _logging.getLogger(__name__).info(
        "REQ %s  max_tokens=%d  temp=%.2f  prompt_words≈%d  "
        "thinking_budget=%d",
        _rid, max_tokens, temperature, _prompt_tokens_approx, _thinking_budget,
    )
    if _trace:
        _prompt_preview = prompt[:400].replace("\n", "↵") + ("…" if len(prompt) > 400 else "")
        _tlog(f"REQ {_rid}  max_tokens={max_tokens}  temp={temperature}  "
              f"top_p={top_p}  seed={seed}  prompt_words≈{_prompt_tokens_approx}")
        _tlog(f"REQ {_rid}  prompt: {_prompt_preview}")

    # Reset LazyLLM pruning state for this request (Item 3)
    if _lazy_llm_state is not None:
        _lazy_llm_state.active_mask = None

    # ── Batch scheduler dispatch (Phase 2.1) ──────────────────────────────────
    # Route non-deterministic requests through the coalescing batch scheduler.
    # submit_sync() is a plain blocking generator — compatible with this sync
    # generator function without any async bridge required.
    is_deterministic = (temperature == 0.0 or seed is not None)
    if _scheduler is not None and not is_deterministic:
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → batch-scheduler")
        try:
            yield from _scheduler.submit_sync(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                top_p       = top_p,
                stop_ids    = _get_stop_ids(stop),
                seed        = seed,
            )
        except _QueueFullError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        return

    # ── Prefix cache lookup (Phase 1.4) ──────────────────────────────────────
    # Only cache deterministic outputs (temp==0 or seed fixed) so non-
    # deterministic completions never return stale cached text.
    #
    # CONFLICT RESOLUTION (LLMLingua ↔ prefix cache):
    # Requests on COMPRESS_PATH have a stochastically-compressed prompt whose
    # token sequence differs on every call — prefix caching would never hit.
    # Skip the prefix cache entirely for COMPRESS_PATH requests.
    # Keys always use _orig_prompt so a future identical *uncompressed* request
    # still matches a response that was generated after compression.
    #
    # Safety guard: _prefix_cache may be None if the server was not started via
    # cmd_serve (e.g. direct function calls in tests that skip startup).
    if _prefix_cache is None:
        _init_prefix_cache()
    cache_eligible = (use_cache
                      and (temperature == 0.0 or seed is not None)
                      and not _on_compress_path)
    if cache_eligible:
        with _trace_span("gen.prefix_cache") as _pcs:
            cached = _prefix_cache.get(_orig_prompt)
        _pcs.set_tag("hit", cached is not None)
        if cached is not None:
            full_text, finish_reason = cached
            if _trace:
                _tlog(f"REQ {_rid}  dispatch → prefix-cache HIT  "
                      f"({len(full_text)} chars, finish={finish_reason})")
            for char in full_text:
                yield char, None
            yield "", finish_reason
            return

    # Collect full output so we can populate the cache after generation
    _cache_buf: list[str] = [] if cache_eligible else []
    _sc_buf:    list[str] = []  # Phase E3: full response text for semantic cache
    _last_finish = "stop"

    # Apply optional seed for reproducible generation
    if seed is not None:
        try:
            import mlx.core as mx
            mx.random.seed(seed)
        except Exception:
            pass

    # ── Speculative decoding (Phase 0.2) ─────────────────────────────────────
    # Use when a draft model is loaded AND temperature > 0 (greedy draft on
    # temp==0 benchmarks offers less benefit and adds overhead).
    if _draft.generator is not None and temperature > 0.0:
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → speculative-decoding")
        # ── Wave 37: reset SpeculativeStreamer per request ─────────────────────
        if _speculative_streamer is not None:
            try:
                _speculative_streamer.reset()
            except Exception:
                pass  # never block generation on streamer reset failure
        try:
            with _trace_span("gen.speculative") as _spec_tspan:
                gen = _draft.generator.stream(
                    prompt,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    top_p       = top_p,
                    stop_ids    = stop_ids,
                    seed        = seed,
                )
                for tok_text, finish in gen:
                    if cache_eligible:
                        _cache_buf.append(tok_text)
                        _last_finish = finish or _last_finish
                    if _trace_tokens and tok_text:
                        _tlog(f"REQ {_rid}  tok={tok_text!r}")
                    yield tok_text, finish
                    if finish is not None:
                        if _trace:
                            _n_spec = len(_cache_buf) if _cache_buf else 0
                            _tlog(f"REQ {_rid}  DONE  path=speculative  "
                                  f"tokens={_n_spec}  finish={finish}")
                        _spec_tspan.set_tag("n_tokens",
                                            len(_cache_buf) if _cache_buf else 0)
                        break
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), _last_finish)
            return
        except Exception as _spec_err:
            import logging as _log
            _log.getLogger(__name__).warning("Speculative decoding failed (%s); "
                                             "falling back to standard generation", _spec_err)

    # ── Wave 37: Jacobi parallel decode ─────────────────────────────────────────
    # Activated when --jacobi is set and NO draft model is loaded (the two
    # speculative paths are mutually exclusive — draft takes priority).
    # Jacobi runs full-sequence forward passes (no KV cache), which lets the
    # fixed-point iteration find accepted N-token prefixes in O(N·iter) calls
    # instead of the standard O(N) single-token autoregressive steps.
    if _jacobi_decoder is not None and _draft.generator is None:
        try:
            import mlx.core as _jd_mx
            import numpy as _jd_np
            _jd_model    = _state.model
            _jd_tokenizer = _state.tokenizer
            _jd_input_ids = (
                _jd_tokenizer.encode(prompt)
                if hasattr(_jd_tokenizer, "encode")
                else _jd_tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            _jd_eos_id = getattr(_jd_tokenizer, "eos_token_id", None) or 151645
            _jd_context  = list(_jd_input_ids)
            _jd_step     = 0
            _jd_stop_buf: list[int] = []
            if _trace:
                _tlog(f"REQ {_rid}  dispatch → jacobi-decode")

            def _jd_logits_fn(ctx_ids: list) -> "_jd_np.ndarray":
                _x = _jd_mx.array(ctx_ids, dtype=_jd_mx.int32)[None]
                _lg = _jd_model(_x)
                _jd_mx.eval(_lg)
                return _jd_np.array(_lg[0].astype(_jd_mx.float32))

            while _jd_step < max_tokens:
                try:
                    _jd_accepted, _jd_n_iter = _jacobi_decoder.decode_step(
                        _jd_logits_fn,
                        _jd_context,
                        vocab_size=getattr(_jd_tokenizer, "vocab_size", 32000),
                    )
                except Exception:
                    break
                if not _jd_accepted:
                    break
                for _jd_tok in _jd_accepted:
                    if _jd_tok == _jd_eos_id:
                        if cache_eligible and _cache_buf:
                            _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                        if _trace:
                            _tlog(f"REQ {_rid}  DONE  path=jacobi  "
                                  f"tokens={_jd_step}  finish=stop(eos)")
                        yield "", "stop"
                        return
                    _jd_txt = (
                        _jd_tokenizer.decode([_jd_tok])
                        if hasattr(_jd_tokenizer, "decode")
                        else str(_jd_tok)
                    )
                    if stop_ids:
                        _jd_stop_buf.append(_jd_tok)
                        if any(_jd_stop_buf[-len(s):] == s for s in stop_ids):
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                            if _trace:
                                _tlog(f"REQ {_rid}  DONE  path=jacobi  "
                                      f"tokens={_jd_step}  finish=stop(stop-seq)")
                            yield "", "stop"
                            return
                        if len(_jd_stop_buf) > 64:
                            _jd_stop_buf = _jd_stop_buf[-64:]
                    if _jd_step >= max_tokens - 1:
                        if cache_eligible and _cache_buf:
                            _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
                        if _trace:
                            _tlog(f"REQ {_rid}  DONE  path=jacobi  "
                                  f"tokens={_jd_step + 1}  finish=length")
                        yield _jd_txt, "length"
                        return
                    if cache_eligible:
                        _cache_buf.append(_jd_txt)
                    if _trace_tokens:
                        _tlog(f"REQ {_rid}  tok={_jd_txt!r}")
                    yield _jd_txt, None
                    _jd_context.append(_jd_tok)
                    _jd_step += 1
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
            if _trace:
                _tlog(f"REQ {_rid}  DONE  path=jacobi  tokens={_jd_step}  finish=stop")
            yield "", "stop"
            return
        except Exception as _jd_err:
            import logging as _jdlog
            _jdlog.getLogger(__name__).warning(
                "[jacobi] decode failed (%s); falling back to standard path", _jd_err
            )

    # ── Quantized KV cache generation path ─────────────────────────────────────
    if _kv_cache is not None:
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → kv-cache ({_kv_cache.__class__.__name__})")
        _kv_cache.reset()
        # ── Wave 37: invalidate ChunkKV cross-layer reuse cache per request ─────
        if _chunk_kv_manager is not None:
            try:
                _chunk_kv_manager.invalidate_reuse_cache()
            except Exception:
                pass  # never block generation on chunk-kv invalidation failure
        try:
            import mlx.core as mx
            import numpy as np
            # Tokenize the *original* (pre-compression) prompt for KV/disk cache
            # key derivation, then re-tokenize the (possibly compressed) prompt for
            # the actual model forward pass.  This ensures the disk cache key is
            # stable even when LLMLingua produces a different compressed form.
            _orig_input_ids = (
                tokenizer.encode(_orig_prompt)
                if hasattr(tokenizer, "encode")
                else tokenizer(_orig_prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            input_ids = (
                tokenizer.encode(prompt)
                if hasattr(tokenizer, "encode")
                else tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            layer_caches = _kv_cache._layers
            # ── Wave 27 / Step 1C: record prefix for predictive cache warmup ────
            # Track every prompt prefix so the warmup predictor can pre-tile
            # frequent token sequences into the KV cache before the next hit.
            # We cap at 256 tokens to avoid leaking the full prompt.
            if _cache_warmup_predictor is not None and _cache_warmup_enabled:
                try:
                    import time as _cwtime
                    _cache_warmup_predictor.record_access(
                        list(input_ids[:256]),
                        _cwtime.monotonic(),
                    )
                except Exception:
                    pass  # never block generation on warmup tracking failure
            # ── Phase 3: session KV cache lookup ───────────────────────────────
            # Restore KV state from a prior conversation if a matching session
            # exists.  Key is SHA-256 of the first 2 KB of the ORIGINAL prompt.
            _session_key = None
            if _session_kv_cache is not None:
                try:
                    import hashlib as _hl
                    _session_key = _hl.sha256(_orig_prompt[:2048].encode()).hexdigest()[:32]
                    _sess_result = _session_kv_cache.load_session(_session_key)
                    if _sess_result is not None:
                        _kv_cache.restore_from(_sess_result)
                        if _trace:
                            _tlog(f"REQ {_rid}  session-cache HIT  key={_session_key}")
                    elif _trace:
                        _tlog(f"REQ {_rid}  session-cache MISS  key={_session_key}")
                except Exception:
                    _session_key = None  # never block generation on session error
            # ── Disk prompt-cache lookup (Item 2) ──────────────────────────────
            # On a hit, restore KV state from NVMe and skip prefill (O(n) → O(1))
            _disk_hit_logit = None
            if _disk_prompt_cache is not None:
                try:
                    # Key by the original (pre-compression) token IDs so that
                    # different LLMLingua compressions of the same prompt still hit.
                    _disk_result = _disk_prompt_cache.lookup(_orig_input_ids)
                    if _disk_result is not None:
                        _disk_qkv, _disk_last_logit = _disk_result
                        _kv_cache.restore_from(_disk_qkv)
                        _disk_hit_logit = _disk_last_logit
                        if _trace:
                            _tlog(f"REQ {_rid}  disk-prompt-cache HIT  "
                                  f"orig_tokens={len(_orig_input_ids)}  → skipped prefill")
                    elif _trace:
                        _tlog(f"REQ {_rid}  disk-prompt-cache MISS  orig_tokens={len(_orig_input_ids)}")
                except Exception:
                    pass  # disk lookup error — fall through to normal prefill

            if _disk_hit_logit is not None:
                # Cache hit: use stored logit to sample first token; no prefill needed
                last_logit_mlx = mx.array(_disk_hit_logit, dtype=mx.float32)
                next_id = _sample_mx(last_logit_mlx, temperature, top_p)
            else:
                # Cache miss: run full prefill
                # ── Wave 37: PD-disagg timing — record prefill start ──────────
                _pd_prefill_t0 = time.monotonic() if _pd_disaggregator is not None else 0.0
                # ── Phase 3C: patch sparse attention for long sequences ────────
                # Applied BEFORE prefill; must be unpatched after regardless of
                # the prefill path taken (standard or chunked).
                # Guard: only when NOT using ane-disagg backend (Core ML graphs
                # are pre-compiled and cannot accept Python-level mask injection).
                _minf_restore = None
                if (_minference_enabled
                        and len(input_ids) > _minference_threshold
                        and _inference_backend != "ane-disagg"):
                    try:
                        from squish.minference_patch import (
                            patch_model_minference as _patch_minf,
                        )
                        from squish.minference_patch import (
                            select_pattern_for_sequence as _minf_pattern,
                        )
                        _pattern = _minf_pattern(len(input_ids))
                        _minf_restore = _patch_minf(
                            model,
                            seq_len_threshold=0,   # already gated above
                            pattern=_pattern,
                        )
                        if _trace:
                            _tlog(f"REQ {_rid}  minference PATCHED  "
                                  f"pattern={_pattern}  seq_len={len(input_ids)}")
                    except Exception as _minf_err:
                        import logging as _mlog
                        _mlog.getLogger(__name__).debug(
                            "[minference] patch failed (%s) — dense fallback", _minf_err
                        )
                        _minf_restore = None

                # ── Phase 3A / Wave 27: chunked prefill (all paths, long prompts) ──
                # CRITICAL: spec decode starts only after is_final_chunk=True.
                # Interleaved greedy tokens emitted on non-final chunks DO count
                # toward the output but bypass the speculative decode path.
                # v10 change: condition no longer gates on _on_compress_path —
                # chunked prefill now activates for ANY long prompt when
                # --chunk-prefill is set and seq_len > _chunk_prefill_threshold.
                _last_logit_vec = None   # [vocab_size] mlx array
                if (_chunk_prefill_enabled
                        and len(input_ids) > _chunk_prefill_threshold):
                    try:
                        from squish.chunked_prefill import (
                            ChunkedPrefillConfig as _CPFConfig,
                        )
                        from squish.chunked_prefill import (
                            chunk_prefill as _chunk_prefill_fn,
                        )
                        _cpf_cfg = _CPFConfig(chunk_size=_chunk_prefill_size)
                        if _trace:
                            _tlog(f"REQ {_rid}  chunked-prefill START  "
                                  f"tokens={len(input_ids)}  "
                                  f"chunk={_chunk_prefill_size}")
                        for _clogit, _is_fin in _chunk_prefill_fn(
                                model, input_ids, layer_caches, _cpf_cfg):
                            if _is_fin:
                                _last_logit_vec = _clogit
                            elif _cpf_cfg.interleave_decode:
                                # Yield one greedy token between chunks for TTFT.
                                # CRITICAL: spec decode MUST NOT start here.
                                _il_id = _sample_mx(_clogit, temperature, top_p)
                                _il_tok = (
                                    tokenizer.decode([_il_id])
                                    if hasattr(tokenizer, "decode") else str(_il_id)
                                )
                                if cache_eligible:
                                    _cache_buf.append(_il_tok)
                                yield _il_tok, None
                        if _trace:
                            _tlog(f"REQ {_rid}  chunked-prefill DONE")
                    except Exception as _cpf_err:
                        import logging as _cpflog
                        _cpflog.getLogger(__name__).warning(
                            "[chunk-prefill] failed (%s) — standard prefill", _cpf_err
                        )
                        _last_logit_vec = None  # fall through below

                if _last_logit_vec is None:
                    # Standard single-shot prefill (non-compress path or fallback)
                    with _trace_span("gen.prefill", tokens=len(input_ids)):
                        x = mx.array(input_ids, dtype=mx.int32)[None]
                        logits_full = model(x, cache=layer_caches)
                        mx.eval(logits_full)
                    _last_logit_vec = logits_full[0, -1]

                # ── Phase 3C: restore dense attention after prefill ────────────
                if _minf_restore is not None:
                    try:
                        from squish.minference_patch import (
                            unpatch_model_minference as _unpatch_minf,
                        )
                        _unpatch_minf(model, _minf_restore)
                        if _trace:
                            _tlog(f"REQ {_rid}  minference UNPATCHED")
                    except Exception:
                        pass  # never block generation on unpatch failure
                    _minf_restore = None

                next_id = _sample_mx(_last_logit_vec, temperature, top_p)
                # ── Wave 37: PD-disagg timing — record prefill completion ──────
                if _pd_disaggregator is not None:
                    try:
                        _pd_disaggregator.stats.total_prefill_ms += (
                            time.monotonic() - _pd_prefill_t0
                        ) * 1000.0
                        _pd_disaggregator.stats.total_prompt_tokens += len(input_ids)
                        _pd_disaggregator.stats.total_requests += 1
                    except Exception:
                        pass  # never block generation on stats update failure
                # Persist for future requests in background
                if _disk_prompt_cache is not None:
                    try:
                        _last_logit_np = np.array(_last_logit_vec.astype(mx.float32))
                        # Store under original token IDs for stable cache keys
                        _disk_prompt_cache.store(_orig_input_ids, _kv_cache, _last_logit_np)
                    except Exception:
                        pass
            stop_buf = [next_id]
            # Compile the single-token decode step for faster subsequent calls.
            # layer_caches is captured as a constant closure; the list reference
            # never changes, so mx.compile reuses the compiled graph every step.
            _decode_fn = None
            if not getattr(_state, "_no_compile", False):
                try:
                    _decode_fn = mx.compile(
                        lambda tok_x: model(tok_x, cache=layer_caches)
                    )
                except Exception:
                    pass  # mx.compile unavailable or incompatible — use plain call
            # Phase A1: thinking budget tracking state
            _in_think_block = False
            _think_step_count = 0
            # Phase B: initialise grammar FSM state for this request
            _grammar_state = None
            if _grammar_engine is not None:
                if _req_tool_schema is not None:
                    # tool_choice enforcement: use request-specific tool schema
                    _grammar_state = _grammar_engine.json_schema_grammar(_req_tool_schema)
                elif _structured_output_mode == "json":
                    _grammar_state = _grammar_engine.json_object_grammar()
                elif _structured_output_mode == "json-schema" and _structured_output_schema is not None:
                    _grammar_state = _grammar_engine.json_schema_grammar(_structured_output_schema)
            # Hoist loop-invariant expressions out of the decode loop
            _bs_cap_inv = _TASK_TOKEN_CAPS.get(_task_type, 0) if _babbling_suppression else 0
            _tok_decode_fn = getattr(tokenizer, "decode", None)
            # Pre-compute which layer caches support async prefetch so we avoid
            # a per-layer hasattr() check on every decode step.
            _prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]
            for step in range(max_tokens):
                # ── Phase E1: Hard token cap (babbling suppression) ──────────────
                if _bs_cap_inv > 0 and step >= _bs_cap_inv:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                    if _trace:
                        _tlog(f"REQ {_rid}  babbling-cap  step={step}  task={_task_type}  cap={_bs_cap_inv}")
                    yield "", "stop"
                    return
                tok_text = (
                    _tok_decode_fn([next_id])
                    if _tok_decode_fn is not None
                    else str(next_id)
                )
                # Phase A1: track thinking block boundaries
                if _thinking_budget >= 0:
                    if "<think>" in tok_text:
                        _in_think_block = True
                        _think_step_count = 0
                    elif "</think>" in tok_text:
                        _in_think_block = False
                    elif _in_think_block:
                        _think_step_count += 1
                if next_id == eos_id:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                    # Phase E3: persist clean EOS completion to semantic cache
                    if _semantic_cache is not None and _sc_buf:
                        try:
                            _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                        except Exception:
                            pass
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=kv-cache  tokens={step}  finish=stop(eos)")
                    yield tok_text, "stop"
                    return
                if stop_ids:
                    for seq in stop_ids:
                        if stop_buf[-len(seq):] == seq:
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                            # Phase E3: persist stop-sequence completion to semantic cache
                            if _semantic_cache is not None and _sc_buf:
                                try:
                                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                except Exception:
                                    pass
                            if _trace:
                                _tlog(f"REQ {_rid}  DONE  path=kv-cache  "
                                      f"tokens={step}  finish=stop(stop-seq)")
                            yield "", "stop"
                            return
                    if len(stop_buf) > 64:
                        stop_buf = stop_buf[-64:]
                if step == max_tokens - 1:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=kv-cache  tokens={step + 1}  finish=length")
                    yield tok_text, "length"
                    return
                if cache_eligible:
                    _cache_buf.append(tok_text)
                if _semantic_cache is not None:
                    _sc_buf.append(tok_text)
                if _trace_tokens:
                    _tlog(f"REQ {_rid}  tok={tok_text!r}")
                yield tok_text, None
                x = mx.array([[next_id]], dtype=mx.int32)
                logits = _decode_fn(x) if _decode_fn is not None else model(x, cache=layer_caches)
                mx.eval(logits)
                # Phase A1/A3: apply logit biases before sampling
                _logit_vec = logits[0, -1]
                if (_thinking_budget > 0
                        and _in_think_block
                        and _think_step_count >= _thinking_budget
                        and _think_close_token_id is not None):
                    _lg_np = np.array(_logit_vec.astype(mx.float32))
                    _lg_np[_think_close_token_id] += 100.0
                    _logit_vec = mx.array(_lg_np)
                if _concise_responses and step >= 20:
                    _lg_np = np.array(_logit_vec.astype(mx.float32))
                    _lg_np[eos_id] += 8.0
                    _logit_vec = mx.array(_lg_np)
                # ── Phase E1: EOS probability monitoring (babbling suppression) ──
                if _babbling_suppression and step >= _babbling_min_tokens:
                    _eos_logit_val = float(_logit_vec[eos_id].item())
                    _max_logit_val = float(mx.max(_logit_vec).item())
                    if _eos_logit_val > _max_logit_val - 1.5:  # pre-filter: EOS is near-top
                        _bs_np = np.array(_logit_vec.astype(mx.float32))
                        _bs_shifted = _bs_np - _bs_np.max()
                        _bs_exp = np.exp(np.clip(_bs_shifted, -30, 0))
                        _eos_prob = _bs_exp[eos_id] / (_bs_exp.sum() + 1e-9)
                        if _eos_prob > _babbling_eos_threshold:
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                            # Phase E3: model-chosen stop — cache it
                            if _semantic_cache is not None and _sc_buf:
                                try:
                                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                except Exception:
                                    pass
                            if _trace:
                                _tlog(f"REQ {_rid}  babbling-eos  step={step}  p={_eos_prob:.3f}  task={_task_type}")
                            yield "", "stop"
                            return
                # Phase B: grammar-constrained logits
                if _grammar_engine is not None and _grammar_state is not None:
                    _logit_vec = _grammar_engine.constrain_logits(_logit_vec, _grammar_state)
                # ── Wave 27 / Step 1B: fused single-pass sampling ─────────────
                # When FusedSampler is active, replace the multi-pass
                # temperature + softmax + top-p call with one in-place kernel.
                # We rebuild a FusedSampler with request-specific temperature/top_p
                # on the first decode step (only if they differ from the server
                # defaults), then reuse it for all subsequent steps.
                if (_fused_sampler_enabled
                        and _fused_sampler is not None
                        and temperature > 0.0):
                    try:
                        _logit_np = np.array(_logit_vec.astype(mx.float32))
                        next_id = _fused_sampler.sample(_logit_np)
                    except Exception:
                        next_id = _sample_mx(_logit_vec, temperature, top_p)
                else:
                    next_id = _sample_mx(_logit_vec, temperature, top_p)
                # Phase B: advance grammar FSM after sampling
                if _grammar_engine is not None and _grammar_state is not None:
                    _grammar_state = _grammar_engine.advance(_grammar_state, next_id)
                    # ── Phase E1: Grammar terminal state (babbling suppression) ──
                    if _babbling_suppression and _grammar_state is not None:
                        try:
                            if _grammar_state.is_terminated():
                                if cache_eligible and _cache_buf:
                                    _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                                # Phase E3: FSM-complete response — worth caching
                                if _semantic_cache is not None and _sc_buf:
                                    try:
                                        _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                    except Exception:
                                        pass
                                if _trace:
                                    _tlog(f"REQ {_rid}  babbling-grammar-terminal  step={step}")
                                yield "", "stop"
                                return
                        except AttributeError:
                            pass  # xgrammar version without is_terminated()
                stop_buf.append(next_id)
                # Phase 0C: fire async CPU dequant for next step while we set up
                # the token embedding — hides O(n_old_tokens) numpy cost behind
                # the model's token-embedding + layernorm overhead.
                for _lc in _prefetch_caches:
                    _lc.start_prefetch()
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
            # Phase E3: end-of-loop clean completion — store in semantic cache
            if _semantic_cache is not None and _sc_buf:
                try:
                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                except Exception:
                    pass
            # Phase 3: persist KV state for future sessions (background thread)
            if _session_kv_cache is not None and _session_key is not None:
                try:
                    _session_kv_cache.save_session(_session_key, _kv_cache)
                except Exception:
                    pass
            # ── Wave 37: PD-disagg — record total decoded tokens ──────────────
            if _pd_disaggregator is not None:
                try:
                    _pd_disaggregator.stats.total_generated_tokens += len(_cache_buf) if _cache_buf else 0
                except Exception:
                    pass
            yield "", "stop"
            return
        except Exception as _kv_err:
            import logging as _kv_log
            _kv_log.getLogger(__name__).warning(
                "Quantized KV cache path failed (%s); falling back to stream_generate",
                _kv_err,
            )
            _kv_cache.reset()

    # ── mlx_lm.stream_generate (preferred, available mlx_lm >= 0.12) ────────
    try:
        import mlx_lm
        _logging.getLogger(__name__).info(
            "REQ %s  dispatch → mlx_lm.stream_generate", _rid
        )
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → mlx_lm.stream_generate")
        _sg_kwargs = {}
        if _max_kv_size is not None:
            _sg_kwargs["max_kv_size"] = _max_kv_size
        # mlx_lm >= 0.21 replaced temp/top_p kwargs with a `sampler` callable.
        # Passing temp/top_p directly causes a TypeError in generate_step, which
        # would be silently caught below and fall through to the no-cache manual
        # loop (O(n²) — catastrophically slow).  Always use make_sampler when
        # available; fall back to legacy kwargs only for older mlx_lm.
        global _cached_make_sampler
        if _cached_make_sampler is None:
            try:
                from mlx_lm.sample_utils import make_sampler as _ms
                _cached_make_sampler = _ms
            except (ImportError, TypeError):
                _cached_make_sampler = False  # sentinel: don't retry
        if _cached_make_sampler:
            _sg_kwargs["sampler"] = _cached_make_sampler(temp=temperature, top_p=top_p)
        else:
            # Older mlx_lm that accepted temp/top_p directly
            _sg_kwargs["temp"]   = temperature
            _sg_kwargs["top_p"]  = top_p
        # Pre-compute text-space stop strings; avoids per-token tokenize calls
        _stop_strings: list[str] = (
            [stop] if isinstance(stop, str) else list(stop) if stop else []
        )
        _stop_text_maxlen = max((len(s) for s in _stop_strings), default=0) + 64
        gen = mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt     = prompt,
            max_tokens = max_tokens,
            **_sg_kwargs,
        )
        emitted = 0
        _stop_text_buf: str = ""
        _think_token_count = 0   # tokens inside <think>...</think> blocks
        _in_think_sg = False     # True while inside a thinking block
        _text_getter = None      # resolved on first item: avoids per-token hasattr
        for item in gen:
            # mlx_lm >= 0.19 yields GenerationResult objects; older yields strings.
            # Detect the type once on the first item and reuse the accessor.
            if _text_getter is None:
                _text_getter = (lambda i: i.text) if hasattr(item, "text") else str
            tok_text = _text_getter(item)
            emitted += 1
            # Track thinking tokens for diagnostics
            if "<think>" in tok_text:
                _in_think_sg = True
            elif "</think>" in tok_text:
                _in_think_sg = False
                _logging.getLogger(__name__).info(
                    "REQ %s  thinking block ended  think_tokens=%d", _rid, _think_token_count
                )
            elif _in_think_sg:
                _think_token_count += 1

            # Check stop sequences in text space — no per-token re-tokenization
            if _stop_strings and tok_text:
                _stop_text_buf += tok_text
                if len(_stop_text_buf) > _stop_text_maxlen:
                    _stop_text_buf = _stop_text_buf[-_stop_text_maxlen:]
                if any(s in _stop_text_buf for s in _stop_strings):
                    if cache_eligible:
                        _cache_buf.append(tok_text)
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  "
                              f"finish=stop(stop-seq)")
                    yield "", "stop"
                    return

            if emitted >= max_tokens:
                if cache_eligible:
                    _cache_buf.append(tok_text)
                    _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
                if _trace:
                    _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  finish=length")
                yield tok_text, "length"
                return
            if cache_eligible:
                _cache_buf.append(tok_text)
            if _trace_tokens:
                _tlog(f"REQ {_rid}  tok={tok_text!r}")
            yield tok_text, None
        if cache_eligible and _cache_buf:
            _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
        _logging.getLogger(__name__).info(
            "REQ %s  DONE  path=mlx_lm  tokens=%d  think_tokens=%d  finish=stop(eos)",
            _rid, emitted, _think_token_count,
        )
        if _trace:
            _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  finish=stop(eos)")
        yield "", "stop"
        return
    except (AttributeError, TypeError) as _sg_err:
        import logging as _sg_log
        _sg_log.getLogger(__name__).warning(
            "REQ %s  mlx_lm.stream_generate FAILED (%s: %s); "
            "falling back to O(n²) manual sampling loop — generation will be "
            "catastrophically slow. This usually means an mlx_lm API mismatch.",
            _rid, type(_sg_err).__name__, _sg_err,
        )

    # ── Fallback: manual sampling loop ───────────────────────────────────────
    import mlx.core as mx
    import numpy as np

    import logging as _fb_log
    _fb_log.getLogger(__name__).warning(
        "REQ %s  running O(n²) manual sampling loop — "
        "check mlx_lm version compatibility for stream_generate support.", _rid
    )
    if _trace:
        _tlog(f"REQ {_rid}  dispatch → manual-sampling-loop (fallback)")
    input_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else \
                tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()

    def _sample(logits_row, temp: float, top_p: float) -> int:
        if temp == 0.0:
            return int(mx.argmax(logits_row).item())
        logits_f = logits_row.astype(mx.float32)
        probs_np = np.array(mx.softmax(logits_f / temp, axis=-1))
        if top_p < 1.0:
            idx      = np.argsort(-probs_np)
            cumsum   = np.cumsum(probs_np[idx])
            cutoff   = min(int((cumsum <= top_p).sum()) + 1, len(idx))
            mask     = np.zeros_like(probs_np)
            mask[idx[:max(1, cutoff)]] = 1.0
            probs_np = probs_np * mask
            probs_np /= probs_np.sum()
        return int(np.random.choice(len(probs_np), p=probs_np))

    ids      = list(input_ids)
    stop_buf = []
    for step in range(max_tokens):
        x       = mx.array(ids, dtype=mx.int32)[None]
        logits  = model(x)
        next_id = _sample(logits[0, -1], temperature, top_p)
        if next_id == eos_id:
            if _trace:
                _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step}  finish=stop(eos)")
            yield "", "stop"
            return
        ids.append(next_id)
        tok_text = tokenizer.decode([next_id])

        if stop_ids:
            stop_buf.append(next_id)
            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step}  "
                              f"finish=stop(stop-seq)")
                    yield "", "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

        if step == max_tokens - 1:
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
            if _trace:
                _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step + 1}  finish=length")
            yield tok_text, "length"
            return
        if cache_eligible:
            _cache_buf.append(tok_text)
        if _trace_tokens:
            _tlog(f"REQ {_rid}  tok={tok_text!r}")
        yield tok_text, None

    if cache_eligible and _cache_buf:
        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
    if _trace:
        _tlog(f"REQ {_rid}  DONE  path=manual  tokens={max_tokens}  finish=stop")
    yield "", "stop"


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Squish OpenAI-compatible API",
    description = "Local LLM inference via Squish compressed models",
    version     = "1.0.0",
)

# Allow browser clients (e.g. Open WebUI) to call without CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Ollama compatibility layer (POST /api/chat etc.) ────────────────────────
try:
    from .serving.ollama_compat import mount_ollama as _mount_ollama  # package import
except ImportError:  # pragma: no cover
    from serving.ollama_compat import mount_ollama as _mount_ollama  # direct script run
_mount_ollama(
    app,
    get_state     = lambda: _state,
    get_generate  = lambda: _generate_tokens,
    get_tokenizer = lambda: _state.tokenizer,
)

# ── Web chat UI (/chat) ────────────────────────────────────────────────
if _STATIC_FILES_AVAILABLE:  # pragma: no branch
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():  # pragma: no branch
        app.mount("/static", _StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/chat")
async def web_chat_ui():
    """Serve the single-page web chat interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"error": "Web UI not found. Is squish/static/index.html present?"}, status_code=404)  # pragma: no cover


@app.get("/v1/models")
async def list_models(creds: HTTPAuthorizationCredentials | None = Security(_bearer)):
    _check_auth(creds)
    if _state.model is None:
        return {"object": "list", "data": []}
    return {"object": "list", "data": [_model_card()]}


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    _check_auth(creds)
    if _state.model is None or model_id not in (_state.model_name, "squish"):
        raise HTTPException(404, f"Model '{model_id}' not found")
    return _model_card()  # pragma: no cover


def _model_card() -> dict:
    return {
        "id":         _state.model_name,
        "object":     "model",
        "created":    int(_state.loaded_at),
        "owned_by":   "squish",
        "permission": [],
        "root":       _state.model_name,
        "parent":     None,
        "squish": {
            "loader":      _state.loader_tag,
            "load_time_s": round(_state.load_time_s, 2),
            "requests":    _state.requests,
            "tokens_gen":  _state.tokens_gen,
        },
    }


def _make_chunk(content: str, model: str, cid: str, finish_reason=None,
                _created: int | None = None,
                _fingerprint: str | None = None) -> str:
    """Build an SSE data line in OpenAI streaming format.

    Callers that stream many tokens should pass pre-computed _created and
    _fingerprint to avoid recomputing them on every call.
    """
    chunk = {
        "id":                cid,
        "object":            "chat.completion.chunk",
        "created":           _created if _created is not None else int(time.time()),
        "model":             model,
        "system_fingerprint": _fingerprint if _fingerprint is not None
                               else _system_fingerprint(_state.model_name, _state.loaded_at),
        "choices": [{
            "index":         0,
            "delta":         {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(  # pragma: no cover
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/chat/completions

    Accepts standard OpenAI ChatCompletion request body.
    Returns streaming (stream=true) or non-streaming response.
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body: dict[str, Any] = await request.json()
    messages    = body.get("messages", [])
    max_tokens  = int(body.get("max_tokens", 4096))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))
    stream      = bool(body.get("stream", False))
    stop        = body.get("stop", None)
    seed        = body.get("seed", None)
    model_id    = body.get("model", _state.model_name)
    tools       = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")

    # tool_choice == "none": agent explicitly disables tools for this turn
    if tool_choice == "none":
        tools = []

    # ── Phase A1: /no_think mode (thinking_budget == 0) ──────────────────────
    if _thinking_budget == 0:
        _msgs_copy = []
        _found_sys = False
        for _m in messages:
            if _m.get("role") == "system" and not _found_sys:
                _msgs_copy.append({**_m, "content": (_m.get("content", "") + " /no_think").strip()})
                _found_sys = True
            else:
                _msgs_copy.append(_m)
        if not _found_sys:
            _msgs_copy = [{"role": "system", "content": "/no_think"}] + list(messages)
        messages = _msgs_copy

    # ── Phase A3: concision prefix ────────────────────────────────────────────
    if _concise_responses:
        _msgs_copy = []
        _found_sys = False
        for _m in messages:
            if _m.get("role") == "system" and not _found_sys:
                _msgs_copy.append({**_m, "content": _CONCISION_PREFIX + _m.get("content", "")})
                _found_sys = True
            else:
                _msgs_copy.append(_m)
        if not _found_sys:
            _msgs_copy = [{"role": "system", "content": _CONCISION_PREFIX}] + list(messages)
        messages = _msgs_copy

    if not messages:
        raise HTTPException(400, "'messages' must be a non-empty list")

    # ── Trace: log incoming messages ────────────────────────────────────────
    if _trace:
        for _mi, _m in enumerate(messages):
            _role    = _m.get("role", "?")
            _content = str(_m.get("content", ""))
            _preview = _content[:300].replace("\n", "↵") + ("…" if len(_content) > 300 else "")
            _tlog(f"CHAT [{_role}] msg[{_mi}]: {_preview}")

    # ── Tool calling: inject schema into system prompt ────────────────────
    global _req_tool_schema, _grammar_engine
    _req_tool_schema = None  # cleared per-request
    _client_stream = stream  # remember original before tools forces stream=False
    _native_tools: list[dict] | None = None  # passed to apply_chat_template
    if tools:
        # When tools are requested, force non-streaming so we can inspect
        # the full output before deciding between text and tool_calls.
        stream = False

        # Prefer native tokenizer tool-calling (Qwen3, Llama-3.1+).  If the
        # tokenizer supports tools=, we skip the manual system-prompt injection
        # so the model uses its trained format (e.g. <tool_call> tags for Qwen3).
        # Fall back to format_tools_prompt for non-native tokenizers.
        _tok = _state.tokenizer
        _supports_native = False
        if _tok is not None and hasattr(_tok, "apply_chat_template"):
            try:
                import inspect as _inspect
                _sig = _inspect.signature(_tok.apply_chat_template)
                _supports_native = "tools" in _sig.parameters
            except Exception:
                pass

        if _supports_native:
            _native_tools = tools
            # Ensure generation stops right after the closing tag so the model
            # doesn't append prose after its tool call.
            _tc_stop = ["</tool_call>"]
            if stop is None:
                stop = _tc_stop
            elif isinstance(stop, str):
                stop = [stop] + _tc_stop
            else:
                stop = list(stop) + _tc_stop
        else:
            from squish.serving.tool_calling import format_tools_prompt
            messages = format_tools_prompt(messages, tools)

        # tool_choice grammar enforcement ─────────────────────────────────
        # "required": force model to output a valid tool call JSON object
        # {"type":"function","function":{"name":"X"}}: force schema for X only
        _tc_schema: dict | None = None
        if tool_choice == "required":
            _tc_schema = _build_tool_union_schema(tools)
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            _forced_name = tool_choice.get("function", {}).get("name", "")
            _match = next(
                (t for t in tools if t.get("function", {}).get("name") == _forced_name),
                None,
            )
            if _match:
                _tc_schema = _match.get("function", {}).get("parameters") or {}

        if _tc_schema is not None:
            # Lazily initialise grammar engine if not already active
            if _grammar_engine is None:
                from squish.grammar_engine import GrammarEngine  # noqa: PLC0415
                if GrammarEngine.is_available() and _state.tokenizer is not None:
                    _grammar_engine = GrammarEngine(_state.tokenizer)
            if _grammar_engine is not None:
                _req_tool_schema = _tc_schema

    prompt         = _apply_chat_template(messages, _state.tokenizer, tools=_native_tools)
    prompt_tokens  = _count_tokens(prompt)
    cid            = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    req_start      = time.perf_counter()
    _state.inflight += 1

    if stream:
        # ── Streaming response ────────────────────────────────────────────
        async def event_stream() -> AsyncIterator[str]:
            import asyncio as _aio
            # Pre-compute per-request constant fields once to avoid
            # recomputing MD5 and int(time.time()) on every streamed token.
            _fp      = _system_fingerprint(_state.model_name, _state.loaded_at)
            _created = int(time.time())
            # Opening chunk (role delta)
            role_chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": _created, "model": model_id,
                "system_fingerprint": _fp,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            gen = _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed)
            n_comp   = 0
            ttft_s   = 0.0
            last_finish = "stop"
            try:
                for tok_text, finish in gen:
                    if tok_text:
                        if n_comp == 0:
                            ttft_s = time.perf_counter() - req_start
                        n_comp += 1
                        yield _make_chunk(tok_text, model_id, cid,
                                          _created=_created, _fingerprint=_fp)
                        # Yield control to the event loop so the HTTP layer can
                        # flush the SSE chunk immediately before the next forward
                        # pass blocks the thread.
                        await _aio.sleep(0)
                    if finish is not None:
                        last_finish = finish
                        break
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            finally:
                _state.inflight -= 1
                dur = time.perf_counter() - req_start
                _state.record_completion(n_comp, dur, ttft_s)
                _tps = n_comp / dur if dur > 0 else 0.0
                _logging.getLogger(__name__).info(
                    "CHAT stream id=%s tokens=%d ttft=%.3fs total=%.3fs tps=%.1f finish=%s",
                    cid, n_comp, ttft_s, dur, _tps, last_finish,
                )
                if _trace:
                    _tlog(f"CHAT stream DONE  id={cid}  tokens={n_comp}  "
                          f"ttft={ttft_s:.3f}s  total={dur:.3f}s  tps={_tps:.1f}  "
                          f"finish={last_finish}")
            yield _make_chunk("", model_id, cid, finish_reason=last_finish,
                              _created=_created, _fingerprint=_fp)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type = "text/event-stream",
            headers    = {
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id":     cid,
            },
        )
    else:
        # ── Non-streaming response ────────────────────────────────────────
        full_text    = ""
        last_finish  = "stop"
        ttft_s       = 0.0
        n_comp       = 0
        try:
            for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                if tok_text:
                    if n_comp == 0:
                        ttft_s = time.perf_counter() - req_start
                    n_comp   += 1
                    full_text += tok_text
                if finish is not None:
                    last_finish = finish
                    break
        finally:
            _req_tool_schema = None  # clear per-request tool schema override
            _state.inflight -= 1
            dur = time.perf_counter() - req_start
            _state.record_completion(n_comp, dur, ttft_s)
            _tps = n_comp / dur if dur > 0 else 0.0
            _logging.getLogger(__name__).info(
                "CHAT id=%s tokens=%d ttft=%.3fs total=%.3fs tps=%.1f finish=%s",
                cid, n_comp, ttft_s, dur, _tps, last_finish,
            )
            if _trace:
                _tlog(f"CHAT  DONE  id={cid}  tokens={n_comp}  "
                      f"ttft={ttft_s:.3f}s  total={dur:.3f}s  tps={_tps:.1f}  "
                      f"finish={last_finish}")
                _resp_preview = full_text[:400].replace("\n", "↵") + (
                    "…" if len(full_text) > 400 else "")
                _tlog(f"CHAT  resp: {_resp_preview}")

        comp_tokens = _count_tokens(full_text)

        # ── Tool calling: detect function call in output ──────────────────────
        if tools:
            from squish.serving.tool_calling import (  # noqa: PLC0415
                build_tool_calls_response,
                parse_tool_calls,
                stream_tool_calls_response,
            )
            raw_calls = parse_tool_calls(full_text)
            if raw_calls is not None:
                if _client_stream:
                    # Client requested streaming: replay tool call as SSE deltas
                    return StreamingResponse(
                        stream_tool_calls_response(cid, model_id, raw_calls),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control":     "no-cache",
                            "X-Accel-Buffering": "no",
                            "X-Request-Id":      cid,
                        },
                    )
                return JSONResponse({
                    "id":                 cid,
                    "object":             "chat.completion",
                    "created":            int(time.time()),
                    "model":              model_id,
                    "system_fingerprint": _system_fingerprint(_state.model_name, _state.loaded_at),
                    "choices": [{
                        "index":   0,
                        "message": {
                            "role":       "assistant",
                            "content":    None,
                            "tool_calls": build_tool_calls_response(raw_calls),
                        },
                        "finish_reason": "tool_calls",
                        "logprobs":      None,
                    }],
                    "usage": {
                        "prompt_tokens":     prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens":      prompt_tokens + comp_tokens,
                    },
                })

        return JSONResponse({
            "id":                 cid,
            "object":             "chat.completion",
            "created":            int(time.time()),
            "model":              model_id,
            "system_fingerprint": _system_fingerprint(_state.model_name, _state.loaded_at),
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": full_text},
                "finish_reason": last_finish,
                "logprobs":      None,
            }],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens":      prompt_tokens + comp_tokens,
            },
        })


@app.post("/v1/completions")
async def completions(  # pragma: no cover
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/completions — legacy text completion endpoint.
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body: dict[str, Any] = await request.json()
    prompt      = body.get("prompt", "")
    max_tokens  = int(body.get("max_tokens", 4096))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))
    stream      = bool(body.get("stream", False))
    stop        = body.get("stop", None)
    seed        = body.get("seed", None)
    model_id    = body.get("model", _state.model_name)
    cid         = f"cmpl-{uuid.uuid4().hex[:12]}"
    req_start   = time.perf_counter()
    _state.inflight += 1

    if not prompt:
        raise HTTPException(400, "'prompt' must be a non-empty string")

    if stream:
        # Pre-compute the timestamp once; reuse for every yielded chunk so that
        # all tokens in a single response share the same "created" timestamp.
        _comp_ts = int(time.time())

        def _comp_chunk(text: str, finish_reason=None) -> str:
            chunk = {
                "id": cid, "object": "text_completion",
                "created": _comp_ts, "model": model_id,
                "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        async def comp_stream() -> AsyncIterator[str]:
            last_finish = "stop"
            n_comp = 0
            ttft_s = 0.0
            try:
                for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                    if tok_text:
                        if n_comp == 0:
                            ttft_s = time.perf_counter() - req_start
                        n_comp += 1
                        yield _comp_chunk(tok_text)
                    if finish is not None:
                        last_finish = finish
                        break
            finally:
                _dur = time.perf_counter() - req_start
                _state.inflight -= 1
                _state.record_completion(n_comp, _dur, ttft_s)
                if _trace:
                    _tps = n_comp / _dur if _dur > 0 else 0.0
                    _tlog(f"CMPL stream DONE  id={cid}  tokens={n_comp}  "
                          f"ttft={ttft_s:.3f}s  total={_dur:.3f}s  tps={_tps:.1f}  "
                          f"finish={last_finish}")
            yield _comp_chunk("", finish_reason=last_finish)
            yield "data: [DONE]\n\n"

        return StreamingResponse(comp_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Request-Id": cid})
    else:
        full_text   = ""
        last_finish = "stop"
        n_comp      = 0
        ttft_s      = 0.0
        try:
            for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                if tok_text:
                    if n_comp == 0:
                        ttft_s = time.perf_counter() - req_start
                    n_comp   += 1
                    full_text += tok_text
                if finish is not None:
                    last_finish = finish
                    break
        finally:
            _dur = time.perf_counter() - req_start
            _state.inflight -= 1
            _state.record_completion(n_comp, _dur, ttft_s)
            if _trace:
                _tps = n_comp / _dur if _dur > 0 else 0.0
                _tlog(f"CMPL  DONE  id={cid}  tokens={n_comp}  "
                      f"ttft={ttft_s:.3f}s  total={_dur:.3f}s  tps={_tps:.1f}  "
                      f"finish={last_finish}")

        prompt_tokens = _count_tokens(prompt)
        comp_tokens   = _count_tokens(full_text)

        return JSONResponse({
            "id": cid, "object": "text_completion",
            "created": int(time.time()), "model": model_id,
            "choices": [{"text": full_text, "index": 0, "finish_reason": last_finish}],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens":      prompt_tokens + comp_tokens,
            },
        })


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/embeddings — mean-pooled last-hidden-state embeddings.

    Compatible with OpenAI embeddings API.
    Input: {'input': str | list[str], 'model': '...'}
    Output: {'object':'list', 'data':[{'object':'embedding','embedding':[...],'index':0}]}
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    import mlx.core as mx
    import numpy as np

    body: dict[str, Any] = await request.json()
    inputs   = body.get("input", "")
    model_id = body.get("model", _state.model_name)
    if isinstance(inputs, str):
        inputs = [inputs]

    model     = _state.model
    tokenizer = _state.tokenizer
    results   = []
    total_tokens = 0

    for i, text in enumerate(inputs):
        ids = tokenizer.encode(text) if hasattr(tokenizer, "encode") else \
              tokenizer(text, return_tensors="np")["input_ids"][0].tolist()
        total_tokens += len(ids)

        x = mx.array(ids, dtype=mx.int32)[None]       # (1, seq)
        try:
            # Preferred path: last hidden state (proper semantic embeddings)
            hidden = model.model(x)                           # (1, seq, hidden_dim)
            emb_np = np.array(mx.mean(hidden, axis=1)[0])    # (hidden_dim,)
        except (AttributeError, TypeError):  # pragma: no cover
            try:
                # Second-best: input token embeddings (less useful but available)
                tok_emb = model.model.embed_tokens(x)        # (1, seq, D)
                emb_np  = np.array(mx.mean(tok_emb, axis=1)[0])
            except AttributeError:  # pragma: no cover
                # Last-resort: mean-pool logits (not suitable for similarity tasks)
                logits = model(x)                            # (1, seq, vocab)
                emb_np = np.array(mx.mean(logits[0], axis=0))

        # L2-normalize
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        results.append({
            "object":    "embedding",
            "embedding": emb_np.tolist(),
            "index":     i,
        })

    return JSONResponse({
        "object": "list",
        "model":  model_id,
        "data":   results,
        "usage":  {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── Wave 76: Agent API ──────────────────────────────────────────────────────
# Three endpoints:
#   GET  /v1/agent/tools        — list built-in tools
#   POST /v1/agent/run          — run the multi-step agent loop (SSE)
#   GET  /v1/agent/mcp          — list connected MCP servers
#   POST /v1/agent/mcp          — connect a new MCP server
#   DELETE /v1/agent/mcp/{id}   — disconnect an MCP server


@app.get("/v1/agent/tools")
async def agent_list_tools(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Return the list of built-in agent tools."""
    _check_auth(creds)
    if _agent_registry is None:
        return {"tools": []}
    return {"tools": _agent_registry.to_openai_schemas()}


@app.get("/v1/agent/mcp")
async def agent_list_mcp(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Return the list of connected MCP servers."""
    _check_auth(creds)
    return {
        "servers": [
            {"id": sid, "status": "connected"}
            for sid in _mcp_servers
        ]
    }


@app.post("/v1/agent/mcp")
async def agent_connect_mcp(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Connect a new MCP server and load its tools into the agent registry.

    Body:
        server_id   (str)  — human-readable identifier
        command     (str)  — STDIO: shell command to launch the MCP server
        url         (str)  — SSE:   base HTTP URL for the MCP server
        transport   (str)  — "stdio" (default) or "sse"
    """
    _check_auth(creds)
    if _agent_registry is None:
        raise HTTPException(503, "Agent registry not initialised")

    body: dict = await request.json()
    server_id = str(body.get("server_id", "mcp")).strip()
    command   = str(body.get("command", "")).strip()
    url       = str(body.get("url", "")).strip()
    transport = str(body.get("transport", "stdio")).lower()

    if not command and not url:
        raise HTTPException(400, "Provide 'command' (stdio) or 'url' (sse)")
    if server_id in _mcp_servers:
        raise HTTPException(409, f"MCP server '{server_id}' is already connected")

    try:
        from squish.serving.mcp_client import MCPClient, MCPTransport, MCPToolAdapter  # noqa: PLC0415
        t = MCPTransport.SSE if transport == "sse" else MCPTransport.STDIO
        src = url if t == MCPTransport.SSE else command
        client = MCPClient(src, transport=t, server_id=server_id)
        await client.connect()
        adapter = MCPToolAdapter(client)
        registered = await adapter.load(_agent_registry)
        _mcp_servers[server_id] = client
        return {
            "server_id": server_id,
            "transport": transport,
            "tools_registered": registered,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"MCP connect failed: {exc}") from exc


@app.delete("/v1/agent/mcp/{server_id}")
async def agent_disconnect_mcp(
    server_id: str,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Disconnect an MCP server.  Its tools remain registered for the session."""
    _check_auth(creds)
    client = _mcp_servers.pop(server_id, None)
    if client is None:
        raise HTTPException(404, f"MCP server '{server_id}' not found")
    try:
        from squish.serving.mcp_client import MCPClient  # noqa: PLC0415
        await client.disconnect()
    except Exception:  # noqa: BLE001
        pass
    return {"disconnected": server_id}


@app.post("/v1/agent/run")
async def agent_run(  # pragma: no cover
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Run the multi-step agentic tool-calling loop over SSE.

    POST body (JSON):
        messages    list[dict]  — conversation so far (OpenAI format)
        tools       list[dict]  — extra tool schemas to add (optional)
        max_steps   int         — max tool-call iterations (default 10)
        max_tokens  int         — max tokens per inference step (default 2048)
        temperature float       — sampling temperature (default 0.7)
        top_p       float       — nucleus sampling threshold (default 0.9)
        model       str         — model identifier (informational only)

    SSE event stream format (each event is ``data: <json>\\n\\n``):

        {"type": "text_delta",      "delta": str}
        {"type": "tool_call_start", "call_id": str, "tool_name": str, "arguments": dict}
        {"type": "tool_call_result","call_id": str, "tool_name": str, "result": str,
                                    "error": str|null, "elapsed_ms": float}
        {"type": "step_complete",   "step": int}
        {"type": "done",            "total_steps": int, "total_tool_calls": int}
        {"type": "error",           "message": str}
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")
    if _agent_registry is None:
        raise HTTPException(503, "Agent registry not initialised")

    body: dict = await request.json()
    messages    = list(body.get("messages", []))
    extra_tools = body.get("tools", [])
    max_steps   = int(body.get("max_steps", 10))
    max_tokens  = int(body.get("max_tokens", 2048))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))

    if not messages:
        raise HTTPException(400, "'messages' must be a non-empty list")

    # Merge built-in tools with any caller-supplied schemas
    builtin_schemas = _agent_registry.to_openai_schemas()
    all_tools = builtin_schemas + [
        t for t in extra_tools
        if t.get("function", {}).get("name") not in
        {s["function"]["name"] for s in builtin_schemas}
    ]

    async def _event_stream():
        import json as _json  # noqa: PLC0415
        from squish.serving.tool_calling import (  # noqa: PLC0415
            format_tools_prompt, parse_tool_calls,
        )

        current_messages = list(messages)
        total_tool_calls = 0

        for step in range(1, max_steps + 1):
            # ── Inject tool schemas into the system prompt ─────────────────
            augmented = format_tools_prompt(current_messages, all_tools)
            prompt = _apply_chat_template(augmented, _state.tokenizer)

            # ── Run a non-streaming inference pass ─────────────────────────
            full_text = ""
            try:
                for tok_text, finish in _generate_tokens(
                    prompt, max_tokens, temperature, top_p, None, None
                ):
                    if tok_text:
                        full_text += tok_text
                        yield (
                            "data: "
                            + _json.dumps({"type": "text_delta", "delta": tok_text})
                            + "\n\n"
                        )
                    if finish is not None:
                        break
            except Exception as exc:  # noqa: BLE001
                yield (
                    "data: "
                    + _json.dumps({"type": "error", "message": str(exc)})
                    + "\n\n"
                )
                return

            # ── Check for tool calls in the output ────────────────────────
            tool_calls = parse_tool_calls(full_text) if all_tools else None

            if not tool_calls:
                # No more tool calls — the agent is done
                yield (
                    "data: "
                    + _json.dumps({
                        "type": "done",
                        "total_steps": step,
                        "total_tool_calls": total_tool_calls,
                    })
                    + "\n\n"
                )
                return

            # ── Execute tool calls ────────────────────────────────────────
            import uuid as _uuid  # noqa: PLC0415

            assistant_tool_calls = []
            tool_result_messages = []

            for tc in tool_calls:
                call_id   = f"call_{_uuid.uuid4().hex[:8]}"
                tool_name = tc.get("name", "")
                arguments = tc.get("arguments", {})

                yield (
                    "data: "
                    + _json.dumps({
                        "type":      "tool_call_start",
                        "call_id":   call_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                    })
                    + "\n\n"
                )

                result = _agent_registry.call(tool_name, arguments, call_id=call_id)
                total_tool_calls += 1

                result_text = (
                    str(result.output)
                    if result.ok
                    else f"[ERROR] {result.error}"
                )

                yield (
                    "data: "
                    + _json.dumps({
                        "type":       "tool_call_result",
                        "call_id":    call_id,
                        "tool_name":  tool_name,
                        "result":     result_text,
                        "error":      result.error,
                        "elapsed_ms": result.elapsed_ms,
                    })
                    + "\n\n"
                )

                assistant_tool_calls.append({
                    "id":   call_id,
                    "type": "function",
                    "function": {
                        "name":      tool_name,
                        "arguments": _json.dumps(arguments),
                    },
                })
                tool_result_messages.append({
                    "role":         "tool",
                    "tool_call_id": call_id,
                    "content":      result_text,
                })

            # ── Append turns to conversation history ──────────────────────
            current_messages.append({
                "role":       "assistant",
                "content":    full_text,
                "tool_calls": assistant_tool_calls,
            })
            current_messages.extend(tool_result_messages)

            yield (
                "data: "
                + _json.dumps({"type": "step_complete", "step": step})
                + "\n\n"
            )

        # max_steps exhausted
        yield (
            "data: "
            + _json.dumps({
                "type":    "error",
                "message": f"Agent hit max_steps={max_steps}. Partial results may be available.",
            })
            + "\n\n"
        )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    _battery_level: float | None = None
    if _power_monitor is not None:
        _battery_level = round(_power_monitor.get_battery_level(), 2)
    _mem_available: float | None = None
    _mem_pressure: int | None = None
    if _memory_governor is not None:
        _mem_available = round(_memory_governor.available_gb, 2)
        _mem_pressure  = _memory_governor.pressure_level
    return {
        "status":       "ok" if _state.model is not None else "no_model",
        "model":        _state.model_name,
        "loaded":       _state.model is not None,
        "loader":       _state.loader_tag,
        "load_time_s":  round(_state.load_time_s, 2),
        "requests":     _state.requests,
        "tokens_gen":   _state.tokens_gen,
        "inflight":     _state.inflight,
        "avg_tps":      round(_state.avg_tps, 1),
        "avg_ttft_s":   round(_state.avg_ttft, 3),
        "uptime_s":     round(time.time() - _state.loaded_at, 1) if _state.loaded_at else 0,
        "power_mode":   _power_mode,
        "battery_level": _battery_level,
        "mem_available_gb": _mem_available,
        "mem_pressure":     _mem_pressure,
    }


@app.get("/v1/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics."""
    # Ensure prefix cache is initialised (lazy-load guard for standalone test
    # clients that skip the normal startup path via cmd_serve).
    if _prefix_cache is None:
        _init_prefix_cache()
    now = time.time()
    uptime = round(now - _state.loaded_at, 1) if _state.loaded_at else 0
    lines = [
        "# HELP squish_requests_total Total inference requests served",
        "# TYPE squish_requests_total counter",
        f"squish_requests_total {_state.requests}",
        "# HELP squish_tokens_generated_total Total tokens generated",
        "# TYPE squish_tokens_generated_total counter",
        f"squish_tokens_generated_total {_state.tokens_gen}",
        "# HELP squish_inflight_requests Current in-flight requests",
        "# TYPE squish_inflight_requests gauge",
        f"squish_inflight_requests {_state.inflight}",
        "# HELP squish_avg_tokens_per_second Rolling average tokens/sec (last 20 requests)",
        "# TYPE squish_avg_tokens_per_second gauge",
        f"squish_avg_tokens_per_second {_state.avg_tps:.2f}",
        "# HELP squish_avg_ttft_seconds Rolling average time-to-first-token (last 20 requests)",
        "# TYPE squish_avg_ttft_seconds gauge",
        f"squish_avg_ttft_seconds {_state.avg_ttft:.4f}",
        "# HELP squish_uptime_seconds Server uptime",
        "# TYPE squish_uptime_seconds counter",
        f"squish_uptime_seconds {uptime}",
        "# HELP squish_model_load_seconds Time taken to load the model",
        "# TYPE squish_model_load_seconds gauge",
        f"squish_model_load_seconds {_state.load_time_s:.3f}",
        "# HELP squish_prefix_cache_hits_total Prefix cache exact-match hits",
        "# TYPE squish_prefix_cache_hits_total counter",
        f"squish_prefix_cache_hits_total {_prefix_cache.hits}",
        "# HELP squish_prefix_cache_size Current entries in prefix cache",
        "# TYPE squish_prefix_cache_size gauge",
        f"squish_prefix_cache_size {_prefix_cache.size}",
        "# HELP squish_radix_prefix_hits_total RadixTree token-prefix KV reuse hits",
        "# TYPE squish_radix_prefix_hits_total counter",
        f"squish_radix_prefix_hits_total {_prefix_cache.prefix_hits}",
        "# HELP squish_paged_kv_free_blocks Paged KV cache free block count",
        "# TYPE squish_paged_kv_free_blocks gauge",
        f"squish_paged_kv_free_blocks {_paged_kv_cache.stats()['free_blocks'] if _paged_kv_cache is not None else -1}",
        "# HELP squish_paged_kv_used_blocks Paged KV cache used block count",
        "# TYPE squish_paged_kv_used_blocks gauge",
        f"squish_paged_kv_used_blocks {_paged_kv_cache.stats()['used_blocks'] if _paged_kv_cache is not None else -1}",
        "# HELP squish_spec_draft_loaded Whether a draft model is loaded",
        "# TYPE squish_spec_draft_loaded gauge",
        f"squish_spec_draft_loaded {1 if _draft.generator is not None else 0}",
        "# HELP squish_kv_cache_tokens Current KV cache token count",
        "# TYPE squish_kv_cache_tokens gauge",
        f"squish_kv_cache_tokens {_kv_cache.n_tokens if _kv_cache is not None else 0}",
        "# HELP squish_kv_cache_memory_mb KV cache memory in MB",
        "# TYPE squish_kv_cache_memory_mb gauge",
        f"squish_kv_cache_memory_mb {_kv_cache.memory_mb if _kv_cache is not None else 0:.2f}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/sys-stats")
async def sys_stats():
    """System-level resource metrics using stdlib only (no psutil required)."""
    import shutil as _shutil
    import resource as _resource

    # CPU load averages (1 / 5 / 15 min)
    try:
        load_avg = [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        load_avg = [0.0, 0.0, 0.0]

    # Process RSS memory (bytes on macOS, KB on Linux)
    try:
        rss_raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        rss_mb = round(rss_raw / 1024 / 1024 if sys.platform == "darwin" else rss_raw / 1024, 1)
    except Exception:
        rss_mb = 0.0

    # Disk usage for root filesystem
    try:
        du = _shutil.disk_usage("/")
        disk_used_pct  = round(du.used / du.total * 100, 1)
        disk_free_gb   = round(du.free / 1024 ** 3, 1)
        disk_total_gb  = round(du.total / 1024 ** 3, 1)
    except Exception:
        disk_used_pct = 0.0
        disk_free_gb  = 0.0
        disk_total_gb = 0.0

    return {
        "load_avg":       load_avg,
        "process_rss_mb": rss_mb,
        "disk_used_pct":  disk_used_pct,
        "disk_free_gb":   disk_free_gb,
        "disk_total_gb":  disk_total_gb,
        "pid":            os.getpid(),
    }


@app.get("/debug-info")
async def debug_info():
    """Server configuration and CLI flags for debugging/observability."""
    return {
        "cli_flags":      _server_args,
        "python_version": sys.version,
        "platform":       sys.platform,
        "pid":            os.getpid(),
    }


@app.get("/v1/trace")
async def get_trace(
    format: str = "",
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    GET /v1/trace — return collected span data for bottleneck analysis.

    Query parameters:
        format=chrome   Chrome DevTools Trace Event JSON — open at
                        https://speedscope.app or chrome://tracing
                        for a flamegraph with every module's start/end timing.
        (default)       JSON object: 20 slowest spans + total span count.

    Enable tracing first with the --trace flag or SQUISH_TRACE=1 env var.
    Spans are accumulated across requests; use DELETE /v1/trace to reset.
    """
    _check_auth(creds)
    if not _TELEMETRY_AVAILABLE:
        return JSONResponse(
            {"error": "Telemetry module not available"},
            status_code=503,
        )
    tracer = _get_tracer()
    if format == "chrome":
        return JSONResponse(tracer.to_chrome_trace())
    slowest = tracer.slowest_spans(n=20)
    return JSONResponse({
        "tracing_enabled": _TELEMETRY_AVAILABLE and __import__("squish.telemetry",
                           fromlist=["TRACING_ENABLED"]).TRACING_ENABLED,
        "total_spans": len(tracer.spans()),
        "hint": (
            "Enable tracing with --trace (or SQUISH_TRACE=1), then run requests, "
            "then GET /v1/trace?format=chrome and open at https://speedscope.app"
        ),
        "slowest_spans": [s.to_dict() for s in slowest],
    })


@app.delete("/v1/trace")
async def clear_trace(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """DELETE /v1/trace — clear all accumulated span data and reset the tracer."""
    _check_auth(creds)
    if not _TELEMETRY_AVAILABLE:
        return JSONResponse({"ok": False, "error": "Telemetry module not available"})
    _get_tracer().clear()
    return JSONResponse({"ok": True, "message": "Trace cleared"})


@app.post("/v1/tokenize")
async def tokenize(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/tokenize — tokenize text and return token IDs + count.
    Non-standard endpoint, useful for prompt engineering / debugging.

    Body: {"text": "..."}  or  {"messages": [{"role":"user","content":"..."}]}
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body = await request.json()
    if "messages" in body:
        text = _apply_chat_template(body["messages"], _state.tokenizer)
    elif "text" in body:
        text = body["text"]
    else:
        raise HTTPException(400, "Provide 'text' or 'messages' in request body")

    tok = _state.tokenizer
    try:
        ids = tok.encode(text) if hasattr(tok, "encode") else \
              tok(text, return_tensors="np")["input_ids"][0].tolist()
    except Exception as e:
        raise HTTPException(500, f"Tokenization failed: {e}") from e

    return JSONResponse({
        "token_ids":   ids,
        "token_count": len(ids),
        "model":       _state.model_name,
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(
        description = "Squish OpenAI-compatible inference server",
        formatter_class = argparse.RawTextHelpFormatter,
        epilog = """
Examples:
  # Start server with 7B model
  python3 squish_server.py \\
    --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
    --compressed-dir ~/models/Qwen2.5-7B-Instruct-bf16-compressed

  # Use from any OpenAI client
  export OPENAI_BASE_URL=http://localhost:11435/v1
  export OPENAI_API_KEY=squish
  python3 -c "from openai import OpenAI; c=OpenAI(); print(c.chat.completions.create(model='squish', messages=[{'role':'user','content':'hello'}]).choices[0].message.content)"
"""
    )
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16"))
    ap.add_argument("--compressed-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16-compressed"))
    ap.add_argument("--mlx-model-dir", default="",
                    metavar="DIR",
                    help="Load a native mlx_lm model directory directly (INT4/INT8 quantized).\n"
                         "Keeps weights quantized in Metal (~4-5 GB for 8B INT4) instead of\n"
                         "dequantizing to BF16 (~15 GB via --compressed-dir).\n"
                         "Create with: python3 -m mlx_lm.convert --hf-path <bf16-dir> \\\n"
                         "  --mlx-path <output-dir> -q --q-bits 4\n"
                         "When set, --model-dir and --compressed-dir are ignored.")
    ap.add_argument("--port",    type=int, default=11435)
    ap.add_argument("--host",    default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--api-key", default=None,
                    help="Optional bearer token required on all requests. "
                         "Also readable from the SQUISH_API_KEY environment variable "
                         "(env var preferred — avoids key appearing in ps aux). "
                         "If omitted, no auth is enforced.")
    ap.add_argument("--draft-model", default="",
                    help="Path to small draft model dir for speculative decoding. "
                         "Should share tokeniser family with target (e.g. Qwen2.5-0.5B "
                         "with Qwen2.5-7B). Enables 1.8-2.5× throughput.")
    ap.add_argument("--draft-compressed", default="",
                    help="Compressed dir for the draft model (default: <draft-model>-compressed)")
    ap.add_argument("--eagle-head-dir", default="",
                    help="Path to EAGLE-3 draft head directory (from `squish pull-head`). "
                         "Enables EAGLE-3 speculative decoding (~75-85%% acceptance rate). "
                         "Incompatible with --draft-model.")
    ap.add_argument("--no-prefix-cache", action="store_true", default=False,
                    help="Disable the prefix (exact-match) response cache")
    ap.add_argument("--prefix-cache-size", type=int, default=512,
                    help="LRU prefix cache capacity (default 512 entries)")
    ap.add_argument("--paged-attention", action="store_true", default=False,
                    help="Enable PagedAttention block table for KV prefix reuse. "
                         "Pre-allocates a fixed KV block pool from unified memory.")
    ap.add_argument("--paged-attention-fraction", type=float, default=0.25,
                    help="Fraction of total RAM to allocate for paged KV blocks "
                         "(default 0.25 = 25%%).  Ignored when --paged-attention "
                         "is not set.")
    # ── Phase 3A: Chunked prefill ─────────────────────────────────────────────
    ap.add_argument("--chunk-prefill", action="store_true", default=False,
                    help="(No-op — chunked prefill is now on by default since Wave 75.)\n"
                         "Kept for backward compatibility.  Use --no-chunk-prefill to disable.")
    ap.add_argument("--no-chunk-prefill", action="store_true", default=False,
                    help="Disable chunked prefill for long prompts.\n"
                         "Chunked prefill is on by default (Wave 75) to prevent\n"
                         "event-loop blocking on prompts > --chunk-prefill-threshold tokens.")
    ap.add_argument("--chunk-prefill-threshold", type=int, default=512,
                    metavar="N",
                    help="Minimum prompt token count to trigger chunked prefill\n"
                         "(default 512).  Requests shorter than N use standard\n"
                         "single-shot prefill regardless of --chunk-prefill.")
    ap.add_argument("--chunk-prefill-size", type=int, default=512,
                    metavar="N",
                    help="Tokens per prefill chunk (default 512).")
    # ── Phase 3C: MInference sparse attention ─────────────────────────────────
    ap.add_argument("--minference", action="store_true", default=False,
                    help="Enable MInference-style sparse attention during prefill.\n"
                         "Reduces attention cost from O(n²) to O(n·k) for prompts\n"
                         "longer than --minference-threshold.\n"
                         "Automatically selects the best sparsity pattern.\n"
                         "Incompatible with --inference-backend ane-disagg.")
    ap.add_argument("--minference-threshold", type=int, default=1024,
                    metavar="N",
                    help="Minimum sequence length to activate sparse attention\n"
                         "(default 1024 tokens).")
    # ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────
    ap.add_argument("--thinking-budget", type=int, default=-1, metavar="N",
                    help="Qwen3 thinking token budget (-1=unlimited, 0=disable thinking mode).\n"
                         "0 appends /no_think to system messages (non-thinking mode).\n"
                         ">0 forces </think> after N thinking tokens via logit bias (+100).")
    # ── Phase A2: explicit KV cache size ─────────────────────────────────────
    ap.add_argument("--max-kv-size", type=int, default=None, metavar="N",
                    help="MLX rotating KV cache size in tokens.\n"
                         "MLX defaults to 4096, silently truncating contexts longer than 4K.\n"
                         "Set to 131072 for 128K context. Passed directly to mlx_lm.stream_generate.")
    # ── Phase A3: concise responses ───────────────────────────────────────────
    ap.add_argument("--concise-responses", action="store_true", default=False,
                    help="Prepend a concision directive to every system message and apply\n"
                         "+8.0 EOS logit bias after 20 tokens to reduce verbosity.")
    # ── Phase B: Structured output (XGrammar) ─────────────────────────────────
    ap.add_argument("--structured-output",
                    choices=["none", "json", "json-schema"],
                    default="none",
                    metavar="MODE",
                    help="Constrain model output to structured formats via XGrammar:\n"
                         "  none        — unconstrained (default)\n"
                         "  json        — constrain to any valid JSON object\n"
                         "  json-schema — constrain to the schema given by --structured-output-schema\n"
                         "Requires: pip install 'squish[grammar]'")
    ap.add_argument("--structured-output-schema", type=str, default=None,
                    metavar="PATH",
                    help="Path to a JSON file containing the JSON-schema used when\n"
                         "--structured-output json-schema is set.")
    # ── Phase C: Power & Energy Modes ─────────────────────────────────────────
    ap.add_argument("--power-mode",
                    choices=["performance", "balanced", "battery", "auto"],
                    default="performance",
                    metavar="MODE",
                    help="Inference resource profile:\n"
                         "  performance — maximum throughput (default)\n"
                         "  balanced    — moderate resource use\n"
                         "  battery     — minimal resource use\n"
                         "  auto        — poll pmset every 30 s and switch automatically")
    # ── Phase 1.3: KV cache quantization ─────────────────────────────────────
    ap.add_argument("--kv-cache-mode",
                    choices=["fp16", "int8", "snap"],
                    default="fp16",
                    help="KV cache compression mode:\n"
                         "  fp16  — standard / no compression (default)\n"
                         "  int8  — KIVI: INT8 older tokens, FP16 recent window\n"
                         "  snap  — KIVI+SnapKV: INT8 + importance-based eviction")
    ap.add_argument("--kv-cache-window", type=int, default=64,
                    help="Recent-token FP16 window for int8/snap modes (default 64)")
    ap.add_argument("--kv-cache-budget", type=int, default=4096,
                    help="Max K/V positions in snap mode (default 4096)")
    # Phase 1 SVD compression
    ap.add_argument("--kv-cache-svd-rank", type=int, default=0,
                    metavar="N",
                    help="SVD rank for KV compression: project head_dim → N before INT8.\n"
                         "0 = off (default).  Recommended: 64 for head_dim=128 models.\n"
                         "Requires --kv-cache-mode int8 or snap.")
    # ── Phase 13A: Asymmetric INT2 KV Cache ──────────────────────────────────
    ap.add_argument("--agent-kv", action="store_true", default=False,
                    help="Enable the asymmetric INT2 KV cache (AgentKV):\n"
                         "  Keeps first --agent-kv-sink tokens in FP32 (attention sinks),\n"
                         "  quantizes older tokens to INT2 (history tier), and maintains\n"
                         "  a FP32 local window for the most recent tokens.  Achieves ~6×\n"
                         "  KV footprint reduction vs FP16 on long-context agent loops.\n"
                         "  Automatically enabled by --agent preset.")
    ap.add_argument("--agent-kv-sink", type=int, default=4, metavar="N",
                    help="Number of FP32 attention-sink tokens to preserve (default 4)")
    ap.add_argument("--agent-kv-window", type=int, default=64, metavar="N",
                    help="FP32 local-window token count for AgentKV (default 64)")
    # Phase 2 retrieval attention
    ap.add_argument("--retrieval-attention", action="store_true", default=False,
                    help="Enable retrieval attention: fetch only the top-k most relevant\n"
                         "disk-tier tokens via HNSW ANNS search instead of scanning all\n"
                         "disk tokens.  Requires --disk-prompt-cache.  Needs hnswlib.")
    ap.add_argument("--retrieval-top-k", type=int, default=32,
                    metavar="N",
                    help="ANNS top-k tokens to retrieve from disk tier (default 32)")
    ap.add_argument("--retrieval-hot-window", type=int, default=256,
                    metavar="N",
                    help="Number of most-recent RAM INT8 tokens always returned\n"
                         "(hot window guarantee, default 256)")
    ap.add_argument("--log-level",
                    choices=["critical", "error", "warning", "info", "debug", "trace"],
                    default="warning",
                    help="Uvicorn log verbosity (default: warning)")
    # ── Phase 2.1: Batch scheduler ────────────────────────────────────────────
    ap.add_argument("--batch-scheduler", action="store_true", default=False,
                    help="Enable continuous batching scheduler: collects concurrent\n"
                         "requests within --batch-window-ms and runs them in one\n"
                         "padded forward pass.  Improves throughput ~N× at moderate load.")
    ap.add_argument("--scheduler", choices=["nested-wait", "legacy"],
                    default="nested-wait",
                    help="Scheduler algorithm when --batch-scheduler is enabled:\n"
                         "  nested-wait — Nested WAIT continuous batcher: merges newly-"
                         "prefilled\n"
                         "                requests between decode steps, eliminating inter-"
                         "batch GPU idle\n"
                         "                time.  Lower TTFT under load.  (default)\n"
                         "  legacy      — Original static coalescing-window batcher.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Max concurrent requests per batch (default 8)")
    ap.add_argument("--batch-window-ms", type=float, default=20.0,
                    help="Collect window in ms before starting a batch (default 20)")
    ap.add_argument("--no-compile", action="store_true", default=False,
                    help="Disable mx.compile for the single-token decode step\n"
                         "(useful for debugging or models incompatible with tracing)")
    ap.add_argument("--disk-prompt-cache", default="",
                    metavar="DIR",
                    help="Enable persistent cross-request KV-state prompt cache stored\n"
                         "as compressed .npz files under DIR (on SSD/NVMe).  Repeated\n"
                         "identical prompts skip prefill entirely.  64-entry LRU default.")
    ap.add_argument("--disk-prompt-cache-size", type=int, default=64,
                    metavar="N",
                    help="Max entries in the disk prompt cache (default 64)")
    # Phase 3: persistent cross-session KV cache
    ap.add_argument("--session-cache-dir", default="",
                    metavar="DIR",
                    help="Enable persistent cross-session KV state cache under DIR.\\n"
                         "The session key is auto-derived from the last 8 message\\n"
                         "contents (SHA-256), so no client changes are needed.\\n"
                         "Surviving a server restart resumes generation from the\\n"
                         "cached KV state.")
    # Phase 4: prompt compression
    ap.add_argument("--compress-prompt", action="store_true", default=False,
                    help="Enable prompt compression before prefill.\\n"
                         "Uses TF-IDF sentence scoring by default; delegates to\\n"
                         "LLMLingua if installed (pip install squish[llmlingua]).")
    ap.add_argument("--compress-ratio", type=float, default=0.5,
                    metavar="F",
                    help="Target compression fraction: 0.5 = compress to half the\\n"
                         "token count (default 0.5).  Range: (0, 1).")
    ap.add_argument("--compress-min-tokens", type=int, default=512,
                    metavar="N",
                    help="Only compress prompts longer than N tokens (default 512).")
    ap.add_argument("--compress-preserve-tokens", type=int, default=0,
                    metavar="N",
                    help="Protect the first N words of each prompt from compression.\n"
                         "Set to the typical system-prompt length to keep the prefix\n"
                         "identical across requests for RadixAttention cache hits.")
    # ── Phase E1: Babbling suppression ─────────────────────────────────────────
    ap.add_argument("--babbling-suppression", action="store_true", default=False,
                    help="Stop generation early when the model strongly prefers EOS "
                         "(EOS probability > 30%%), a grammar FSM reaches a terminal "
                         "state, or a per-task token cap is exceeded.\n"
                         "Reduces average energy cost by 44-89%% on short-output tasks.")
    ap.add_argument("--no-babbling-suppression", dest="babbling_suppression",
                    action="store_false",
                    help="Disable babbling suppression (keep generating until max_tokens).")
    ap.add_argument("--babbling-eos-threshold", type=float, default=0.30,
                    metavar="P",
                    help="EOS probability threshold for babbling suppression (default 0.30).")
    ap.add_argument("--babbling-min-tokens", type=int, default=10,
                    metavar="N",
                    help="Never stop early before emitting N tokens (default 10).")
    # ── Phase E2: Polynomial GELU approximation ───────────────────────────────
    ap.add_argument("--fast-gelu", action="store_true", default=False,
                    help="Replace erf-GELU with x·sigmoid(1.702x) for GELU-based models.\n"
                         "No-op for SiLU/SwiGLU models (Qwen3, LLaMA). "
                         "Provides ~3-5%% speedup on GPU, larger on ANE.")
    ap.add_argument("--no-fast-gelu", dest="fast_gelu", action="store_false",
                    help="Disable polynomial GELU approximation.")
    # ── Phase E3: Semantic response cache ─────────────────────────────────────
    ap.add_argument("--semantic-cache", action="store_true", default=False,
                    help="Enable semantic response caching. Semantically similar prompts "
                         "(cosine distance < task threshold) return a cached response, "
                         "delivering 25-250× latency reduction for warm repeat patterns.")
    ap.add_argument("--no-semantic-cache", dest="semantic_cache", action="store_false",
                    help="Disable semantic response cache.")
    ap.add_argument("--semantic-cache-db", default="",
                    metavar="PATH",
                    help="Path to the sqlite-vec semantic cache database "
                         "(default: ~/.squish/response_cache.db).")
    # ── Phase 4: hardware inference backend ──────────────────────────────────
    ap.add_argument("--inference-backend",
                    choices=["mlx-eager", "mlx-compiled", "ane-disagg", "mlc"],
                    default="mlx-eager",
                    metavar="BACKEND",
                    help="Hardware dispatch strategy (default: mlx-eager):\n"
                         "  mlx-eager    — standard MLX Metal execution (safest)\n"
                         "  mlx-compiled — mx.compile fused decode (lower GPU overhead)\n"
                         "  ane-disagg   — Apple Neural Engine prefill + GPU decode\n"
                         "  mlc          — MLC-LLM engine (large-context requests)\n"
                         "mlx-compiled and ane-disagg are mutually exclusive.")
    # ── Item 3: LazyLLM token pruning ─────────────────────────────────────────
    ap.add_argument("--lazy-llm", action="store_true", default=False,
                    help="Enable LazyLLM dynamic token pruning during prefill.\n"
                         "Skips low-importance positions in later transformer layers,\n"
                         "reducing TTFT by ~20-35%% on long prompts.")
    ap.add_argument("--lazy-llm-keep-ratio", type=float, default=0.70,
                    metavar="F",
                    help="Fraction of tokens to keep per layer (default 0.70)")
    ap.add_argument("--lazy-llm-start-layer", type=int, default=2,
                    metavar="N",
                    help="First layer index where pruning is applied (default 2)")
    ap.add_argument("--lazy-llm-revive-window", type=int, default=4,
                    metavar="N",
                    help="Always keep the N most recent tokens active (default 4)")
    # ── Verbose inference tracing ─────────────────────────────────────────────
    ap.add_argument("--trace", action="store_true", default=False,
                    help="Log full per-request detail to stderr: prompt, dispatch path, "
                         "finish reason, TTFT, TPS, and cache hit/miss status.")
    ap.add_argument("--trace-tokens", action="store_true", default=False,
                    help="Also log every generated token text (implies --trace; "
                         "very verbose — useful for debugging output corruption).")
    ap.add_argument("--trace-file", default="",
                    metavar="FILE",
                    help="Append trace output to FILE in addition to stderr. "
                         "Useful when the server stdout/stderr is not visible "
                         "(e.g. when launched by _run_all.py).")
    ap.add_argument("--trace-output", default="",
                    metavar="FILE",
                    help="Save a Chrome DevTools Trace Event Format JSON to FILE on exit. "
                         "Open at https://speedscope.app or chrome://tracing for a "
                         "flame graph showing every module with start/end timing.")

    # ── Wave optimization flags ───────────────────────────────────────────────
    ap.add_argument("--prompt-lookup", action="store_true", default=False,
                    help="Enable n-gram prompt lookup speculative decoding.")
    ap.add_argument("--prompt-lookup-n", type=int, default=3, metavar="N",
                    help="N-gram size for prompt lookup (default: 3).")
    ap.add_argument("--prompt-lookup-k", type=int, default=4, metavar="K",
                    help="Max draft tokens per lookup step (default: 4).")
    ap.add_argument("--seq-packing", action="store_true", default=False,
                    help="Enable sequence packing for higher batch GPU utilisation.")
    ap.add_argument("--seq-packing-budget", type=int, default=2048, metavar="N",
                    help="Token budget per packed batch (default: 2048).")
    ap.add_argument("--ada-serve", action="store_true", default=False,
                    help="Enable SLO-adaptive gamma scheduling for speculative decoding.")
    ap.add_argument("--ada-serve-slo", default="general",
                    choices=["git_commit", "devops_plan", "general", "code_review"],
                    help="Default SLO profile for AdaServe (default: general).")
    ap.add_argument("--conf-spec", action="store_true", default=False,
                    help="Enable confidence-gated speculative step verification.")
    ap.add_argument("--conf-spec-high-gate", type=float, default=0.90, metavar="F",
                    help="Confidence above which steps are auto-accepted (default: 0.90).")
    ap.add_argument("--conf-spec-low-gate", type=float, default=0.50, metavar="F",
                    help="Confidence below which full target verify is used (default: 0.50).")
    ap.add_argument("--kv-share", action="store_true", default=False,
                    help="Enable cross-layer KV sharing (KVSharer).")
    ap.add_argument("--kv-share-every", type=int, default=2, metavar="N",
                    help="Share KV every N layers (default: 2).")
    ap.add_argument("--kv-slab", action="store_true", default=False,
                    help="Enable slab-based KV memory allocator for reduced fragmentation.")
    ap.add_argument("--kv-slab-pages", type=int, default=256, metavar="N",
                    help="Number of slab pages (default: 256).")
    ap.add_argument("--paris-kv", action="store_true", default=False,
                    help="Enable PARIS KV codebook compression.")
    ap.add_argument("--paris-kv-centroids", type=int, default=64, metavar="N",
                    help="PARIS codebook centroid count (default: 64).")
    ap.add_argument("--streaming-sink", action="store_true", default=False,
                    help="Enable StreamingLLM-style sink KV cache.")
    ap.add_argument("--streaming-sink-size", type=int, default=2048, metavar="N",
                    help="Sink KV cache token budget (default: 2048).")
    ap.add_argument("--diff-kv", action="store_true", default=False,
                    help="Enable DiffKV 3-axis differentiated KV precision.")
    ap.add_argument("--small-kv", action="store_true", default=False,
                    help="Enable SmallKV saliency-shift compensation.")
    ap.add_argument("--sage-attention", action="store_true", default=False,
                    help="Enable SageAttention INT8 quantized QK^T computation.")
    ap.add_argument("--sage-attention2", action="store_true", default=False,
                    help="Enable SageAttention2 INT4/FP8 quantized attention.")
    ap.add_argument("--sparge-attention", action="store_true", default=False,
                    help="Enable SpargeAttn sparse+quantized attention.")
    ap.add_argument("--squeeze-attention", action="store_true", default=False,
                    help="Enable SqueezeAttention adaptive KV budget allocation.")
    ap.add_argument("--yoco-kv", action="store_true", default=False,
                    help="Enable YOCO cross-layer KV reuse (you-only-cache-once).")
    ap.add_argument("--cla", action="store_true", default=False,
                    help="Enable Cross-Layer Attention KV sharing.")
    ap.add_argument("--kvtuner", action="store_true", default=False,
                    help="Enable KVTuner adaptive per-layer KV budget.")
    ap.add_argument("--robust-scheduler", action="store_true", default=False,
                    help="Use the robust A-max/A-balanced batch scheduler.")
    ap.add_argument("--gemfilter", action="store_true", default=False,
                    help="Enable GemFilter attention head filtering.")
    ap.add_argument("--svdq", action="store_true", default=False,
                    help="Enable SVD-based KV quantization (SVDQ).")
    ap.add_argument("--sparse-spec", action="store_true", default=False,
                    help="Enable sparse speculative decoding.")
    ap.add_argument("--sparse-verify", action="store_true", default=False,
                    help="Enable sparse draft verification.")
    ap.add_argument("--trail", action="store_true", default=False,
                    help="Enable TRAIL token-importance-aware layer skipping.")
    ap.add_argument("--specontext", action="store_true", default=False,
                    help="Enable SpecContext speculative context extension.")
    ap.add_argument("--forelen", action="store_true", default=False,
                    help="Enable ForeLen forward-looking token length prediction.")
    ap.add_argument("--ipw", action="store_true", default=False,
                    help="Enable IPW importance-weighted prefill compression.")
    ap.add_argument("--layer-skip", action="store_true", default=False,
                    help="Enable LayerSkip early-exit adaptive layer skipping.")
    ap.add_argument("--lookahead", action="store_true", default=False,
                    help="Enable LookaheadReasoning parallel step verification.")
    # ── Wave 27: inference velocity flags ────────────────────────────────────
    ap.add_argument("--no-fused-sampler", action="store_true", default=False,
                    help="Disable fused single-pass token sampling (enabled by default).\n"
                         "The FusedSampler applies temperature, top-k, top-p, min-p, and\n"
                         "rep-penalty in one kernel pass, eliminating intermediate\n"
                         "vocabulary-sized allocations per decode step (~8–12%% speedup).")
    ap.add_argument("--no-cache-warmup", action="store_true", default=False,
                    help="Disable predictive KV prefix pre-warming (enabled by default).\n"
                         "Tracks prefix access patterns and pre-warms the KV cache for\n"
                         "hot paths before each request arrives, reducing TTFT for\n"
                         "repeated system prompts and RAG documents.")
    ap.add_argument("--token-merge", action="store_true", default=False,
                    help="[Beta] Enable Token Merging (ToMe) during prefill.\n"
                         "Merges similar adjacent tokens in attention layers 4–11\n"
                         "via bipartite cosine-similarity matching, reducing prefill\n"
                         "FLOPs by 30–40%% with <2%% quality degradation on seqs ≥256.\n"
                         "Complementary to --chunk-prefill (stack both for max TTFT reduction).")
    ap.add_argument("--tome-r", type=int, default=16, metavar="R",
                    help="Token pairs to merge per ToMe layer (default 16).\n"
                         "Higher R = faster prefill but more quality loss.")
    ap.add_argument("--tome-start-layer", type=int, default=4, metavar="L",
                    help="First transformer layer where token merging is applied (default 4).")
    ap.add_argument("--tome-end-layer", type=int, default=11, metavar="L",
                    help="Last transformer layer where token merging is applied (default 11).")
    ap.add_argument("--lookahead-k", type=int, default=4, metavar="K",
                    help="Lookahead window size (default: 4).")
    ap.add_argument("--spec-reason", action="store_true", default=False,
                    help="Enable SpecReason step-level speculative reasoning.")
    ap.add_argument("--long-spec", action="store_true", default=False,
                    help="Enable LongSpec extended speculative decoding.")
    ap.add_argument("--fr-spec", action="store_true", default=False,
                    help="Enable FR-Spec frequency-based token speculative decoding.")
    # ── Wave 37: Wire Everything In ───────────────────────────────────────────
    ap.add_argument("--kvtc", action="store_true", default=False,
                    help="Enable KV-Transform Coder: PCA+quantize KV cache across all layers.\n"
                         "Reduces KV memory 4–8× at cost of a one-time calibration pass.\n"
                         "Targets 8× TTFT improvement on 8k+ token prompts.")
    ap.add_argument("--kvtc-rank", type=int, default=64, metavar="N",
                    help="PCA rank for KVTC (default 64; recommended: head_dim // 2).")
    ap.add_argument("--kvtc-bits", type=int, default=8, choices=[4, 8],
                    help="Quantisation bits for KVTC coefficients (4 or 8, default 8).")
    ap.add_argument("--chunk-kv", action="store_true", default=False,
                    help="Enable ChunkKV: chunk-level KV importance scoring and eviction.\n"
                         "Scores chunks by max-attention or norm, retains top budget_ratio.\n"
                         "Targets +26.5%% TPS via reduced KV memory footprint.")
    ap.add_argument("--chunk-kv-size", type=int, default=16, metavar="N",
                    help="Tokens per KV chunk for eviction scoring (default 16).")
    ap.add_argument("--chunk-kv-budget", type=float, default=0.5, metavar="F",
                    help="Fraction of KV budget to retain after eviction (default 0.5 = 50%%).")
    ap.add_argument("--ssd-saguaro", action="store_true", default=False,
                    help="Enable SSD-Saguaro speculative decoding: prefetch top-K speculative\n"
                         "outcome branches to eliminate sequential verify latency.\n"
                         "Targets 5× throughput vs autoregressive baseline.")
    ap.add_argument("--spec-stream", action="store_true", default=False,
                    help="Enable SpeculativeStreamer: buffer draft tokens and stream them to\n"
                         "the client before verification. Enables perceived 0ms TTFT.\n"
                         "Silent rollback on rejection; no client protocol changes needed.")
    ap.add_argument("--metal-flash-attn", action="store_true", default=False,
                    help="Enable MetalFlashAttention: tiled fused QK^T·softmax·PV kernel.\n"
                         "No intermediate buffer allocations. 3–5× attention speedup.\n"
                         "NumPy reference path used when Metal is unavailable.")
    ap.add_argument("--deja-vu", action="store_true", default=False,
                    help="Enable DejaVu sparse FFN: lightweight predictor skips inactive\n"
                         "neurons before each FFN forward pass. 30–50%% FFN FLOP reduction.")
    ap.add_argument("--jacobi", action="store_true", default=False,
                    help="Enable Jacobi parallel decode: run N speculative positions in\n"
                         "parallel and commit the longest fixed-point prefix. ~3.4× decode\n"
                         "speedup with no draft model required.")
    ap.add_argument("--jacobi-n", type=int, default=4, metavar="N",
                    help="Parallel position count for Jacobi decode (default 4).")
    ap.add_argument("--jacobi-variant",
                    choices=["jacobi", "gauss_seidel"],
                    default="jacobi",
                    metavar="VARIANT",
                    help="Jacobi iteration variant (jacobi or gauss_seidel, default: jacobi).")
    ap.add_argument("--mtp", action="store_true", default=False,
                    help="Enable Multi-Token Predictor: N auxiliary prediction heads each\n"
                         "forecast the next token independently. 1.7–3× throughput gain.")
    ap.add_argument("--mtp-heads", type=int, default=4, metavar="N",
                    help="Number of MTP auxiliary heads (default 4).")
    ap.add_argument("--layer-overlap", action="store_true", default=False,
                    help="Enable LayerOverlapLoader: prefetch layer N+1 weights during layer\n"
                         "N compute. Eliminates weight-load stalls between transformer layers.")
    ap.add_argument("--layer-overlap-prefetch", type=int, default=2, metavar="N",
                    help="Number of layers to keep pre-fetched ahead (default 2).")
    ap.add_argument("--fused-qkv", action="store_true", default=False,
                    help="Enable FusedQKVProjection: single W_qkv matmul replaces three\n"
                         "separate Q/K/V projections. Reduces input reads by 67%%. +14%% prefill.")
    ap.add_argument("--pd-disagg", action="store_true", default=False,
                    help="Enable PD-Disaggregator: route prefill (compute-bound) and\n"
                         "decode (memory-bound) through separate scheduling paths.\n"
                         "Targets 1.5–2× TTFT under mixed prefill/decode load.")
    # ── Wave 41 flags ─────────────────────────────────────────────────────────
    ap.add_argument("--radix-attn", action="store_true", default=False,
                    help="Enable RadixAttentionCache: prefix-aware KV cache sharing "
                         "via a radix tree. Reduces redundant prefill for shared prefixes.")
    ap.add_argument("--eagle2", action="store_true", default=False,
                    help="Enable EAGLE-2 speculative decoding with dynamic draft tree "
                         "pruning. Typical 2–3× decode throughput improvement.")
    ap.add_argument("--ring-attn", action="store_true", default=False,
                    help="Enable RingAttention: sequence-parallel ring-reduce attention "
                         "for very long contexts across virtual devices.")
    ap.add_argument("--token-entropy-prune", action="store_true", default=False,
                    help="Enable TokenEntropyPruner: drop low-entropy (redundant) tokens "
                         "from the KV cache based on attention entropy.")
    ap.add_argument("--pregated-moe", action="store_true", default=False,
                    help="Enable PreGatedMoE router: pre-gate expert selection before "
                         "FFN to reduce expert activation overhead.")
    ap.add_argument("--sink-fusion", action="store_true", default=False,
                    help="Enable SinkFusion: merge attention sink tokens to reduce "
                         "KV cache memory pressure.")
    ap.add_argument("--cla-share", action="store_true", default=False,
                    help="Enable Cross-Layer Attention sharing (CLA): reuse KV caches "
                         "across adjacent transformer layers.")
    ap.add_argument("--qmoe-compress", action="store_true", default=False,
                    help="Enable QMoECompressor: quantize MoE expert weights to reduce "
                         "memory and improve expert-load throughput.")
    ap.add_argument("--lade", action="store_true", default=False,
                    help="Enable LADE: lookahead-based draft speculative decoding.")
    ap.add_argument("--infini-attn", action="store_true", default=False,
                    help="Enable InfiniAttention: compressive memory segment for "
                         "unbounded context windows.")
    ap.add_argument("--akvq", action="store_true", default=False,
                    help="Enable AKVQCache: adaptive KV quantization with per-layer "
                         "bit-width selection.")
    ap.add_argument("--delta-zip", action="store_true", default=False,
                    help="Enable DeltaZipAdapter: quantize fine-tuned weight deltas "
                         "to compress LoRA-style adapters.")
    # ── Wave 42 flags ─────────────────────────────────────────────────────────
    ap.add_argument("--medusa-heads", action="store_true", default=False,
                    help="Enable MedusaHeads: multiple frozen draft heads for parallel "
                         "speculative decoding (Cai et al., ICML 2024).")
    ap.add_argument("--sarathi", action="store_true", default=False,
                    help="Enable SarathiScheduler: fixed-size chunked prefill with "
                         "decode piggybacking (Agrawal et al., OSDI 2024).")
    ap.add_argument("--nsa-attn", action="store_true", default=False,
                    help="Enable NSAAttention: native sparse attention with compound "
                         "block + window + selected-token pattern (Yuan et al., 2025).")
    ap.add_argument("--flex-prefill", action="store_true", default=False,
                    help="Enable FlexPrefill: per-head context-adaptive sparse prefill "
                         "(Lai et al., arXiv:2502.20766).")
    ap.add_argument("--think-cache", action="store_true", default=False,
                    help="Enable ThinKCache: query-driven key-channel pruning to reduce "
                         "KV memory by ~20%% (Xu et al., EMNLP 2024).")
    ap.add_argument("--attention-store", action="store_true", default=False,
                    help="Enable AttentionStore: session-scoped three-tier KV persistence "
                         "(hot/warm/SSD) for multi-turn reuse (Sheng et al., ACL 2024).")
    ap.add_argument("--rest-decode", action="store_true", default=False,
                    help="Enable RESTDecode: retrieval-based n-gram speculative decoding "
                         "from a token datastore (He et al., NAACL 2024).")
    ap.add_argument("--star-attn", action="store_true", default=False,
                    help="Enable StarAttention: block-partitioned star-topology "
                         "local + anchor attention (Acharya et al., NeurIPS 2024).")
    ap.add_argument("--splitwise", action="store_true", default=False,
                    help="Enable SplitwiseScheduler: prefill/decode phase disaggregation "
                         "into separate resource pools (Patel et al., ISCA 2024).")
    ap.add_argument("--kvquant", action="store_true", default=False,
                    help="Enable KVQuantCache: calibrated low-bit KV quantization "
                         "(Hooper et al., NeurIPS 2024).")
    ap.add_argument("--efficient-qat", action="store_true", default=False,
                    help="Enable EfficientQAT: block-wise QAT with frozen neighbours "
                         "for W4A4 quantization (Chen et al., ECCV 2024).")
    ap.add_argument("--cache-gen", action="store_true", default=False,
                    help="Enable CacheGenCodec: arithmetic-coded KV bitstream "
                         "compression and streaming decoder (Liu et al., SIGCOMM 2024).")

    # ── Wave 43 flags ────────────────────────────────────────────────────────
    ap.add_argument("--mtp-decode", action="store_true", default=False,
                    help="Multi-Token Prediction decode heads (DeepSeek-V3, Wave 43).")
    ap.add_argument("--cascade-kv", action="store_true", default=False,
                    help="Cascade KV: two-level shared-prefix + per-request KV cache (Wave 43).")
    ap.add_argument("--head-prune", action="store_true", default=False,
                    help="Structured head/MLP pruning via importance scoring (Wave 43).")
    ap.add_argument("--paged-attn-w43", action="store_true", default=False,
                    help="PagedAttention Wave-43 block manager (Wave 43).")
    ap.add_argument("--layer-collapse", action="store_true", default=False,
                    help="Layer collapse via cosine-similarity skip scheduling (Wave 43).")
    ap.add_argument("--relay-attn", action="store_true", default=False,
                    help="Relay attention: share softmax output across similar-output layers (Wave 43).")
    ap.add_argument("--wkv-quant", action="store_true", default=False,
                    help="Joint weight+KV quantization W4KV4 (Wave 43).")
    ap.add_argument("--tokenized-kv", action="store_true", default=False,
                    help="Tokenized KV: serialize/deserialize KV via embedding table (Wave 43).")
    ap.add_argument("--cluster-evict", action="store_true", default=False,
                    help="Cluster-based KV eviction with adaptive per-layer budget (Wave 43).")
    ap.add_argument("--s2-attn", action="store_true", default=False,
                    help="S²-Attention: sorted-structured sparse attention (Wave 43).")
    ap.add_argument("--magic-pig-v2", action="store_true", default=False,
                    help="MagicPIG v2: LSH-sampled KV retrieval with adaptive probe budget (Wave 43).")

    # ── Wave 44 flags ────────────────────────────────────────────────────────
    ap.add_argument("--marlin-gemm", action="store_true", default=False,
                    help="Marlin INT4×FP16 tiled GEMM for post-training quantization (Wave 44).")
    ap.add_argument("--spec-rejection", action="store_true", default=False,
                    help="Speculative rejection: parallel draft candidates with early pruning (Wave 44).")
    ap.add_argument("--loftq", action="store_true", default=False,
                    help="LoFTQ: alternating LoRA+W4 quantization optimizer (Wave 44).")
    ap.add_argument("--online-spec", action="store_true", default=False,
                    help="Online speculative decoding with session-adaptive draft distribution (Wave 44).")
    ap.add_argument("--dynamic-spec-len", action="store_true", default=False,
                    help="Dynamic speculation lookahead: adaptive K per token (Wave 44).")
    ap.add_argument("--big-little", action="store_true", default=False,
                    help="Big-Little decoder: route easy tokens to small model (Wave 44).")
    ap.add_argument("--multi-exit-spec", action="store_true", default=False,
                    help="Multi-exit speculative decoding at early transformer layer (Wave 44).")
    ap.add_argument("--pv-tuning", action="store_true", default=False,
                    help="PV-Tuning: proximal-gradient quantized weight optimization W1–2 (Wave 44).")
    ap.add_argument("--hadamard-quant", action="store_true", default=False,
                    help="Hadamard rotation whitening before INT4 GEMM (Wave 44).")
    ap.add_argument("--prefix-tree-decode", action="store_true", default=False,
                    help="Prefix tree decode: parallel static prefix tree paths (Wave 44).")
    ap.add_argument("--spectr-ot", action="store_true", default=False,
                    help="SpecTr: optimal-transport draft-target coupling (Wave 44).")
    ap.add_argument("--ada-gptq", action="store_true", default=False,
                    help="Ada-GPTQ: per-layer adaptive group size for W4 PTQ (Wave 44).")

    # ── Wave 45 flags ────────────────────────────────────────────────────────
    ap.add_argument("--flexgen-offload", action="store_true", default=False,
                    help="FlexGen offload: LP-optimal CPU/SSD weight paging (Wave 45).")
    ap.add_argument("--yarn-rope", action="store_true", default=False,
                    help="YaRN RoPE: extended context via NTK-aware interpolation (Wave 45).")
    ap.add_argument("--self-extend", action="store_true", default=False,
                    help="Self-Extend: grouped attention for context beyond training window (Wave 45).")
    ap.add_argument("--orca-sched", action="store_true", default=False,
                    help="Orca iteration-level scheduler for continuous batching (Wave 45).")
    ap.add_argument("--mx-fp4", action="store_true", default=False,
                    help="MX FP4 microscaling quantization (Wave 45).")
    ap.add_argument("--fp8-act", action="store_true", default=False,
                    help="FP8 activation quantization for W8A8-style inference (Wave 45).")
    ap.add_argument("--clex-rope", action="store_true", default=False,
                    help="CLeX RoPE: continuous length extrapolation (Wave 45).")
    ap.add_argument("--powerinfer", action="store_true", default=False,
                    help="PowerInfer offload: neuron-activation-aware weight streaming (Wave 45).")
    ap.add_argument("--grouped-rope", action="store_true", default=False,
                    help="Grouped RoPE: group-query-aware rotary position embeddings (Wave 45).")
    ap.add_argument("--tensor-parallel", action="store_true", default=False,
                    help="Tensor parallelism across available compute partitions (Wave 45).")
    ap.add_argument("--fused-bias-gelu", action="store_true", default=False,
                    help="Fused bias+GELU kernel (Wave 45).")
    ap.add_argument("--token-budget-sched", action="store_true", default=False,
                    help="Token budget scheduler: per-request KV-budget limits (Wave 45).")

    # ── Wave 46 flags ────────────────────────────────────────────────────────
    ap.add_argument("--slice-gpt", action="store_true", default=False,
                    help="SliceGPT: PCA orthogonal column pruning (Wave 46).")
    ap.add_argument("--wanda", action="store_true", default=False,
                    help="Wanda: weight magnitude × activation RMS pruning (Wave 46).")
    ap.add_argument("--short-gpt", action="store_true", default=False,
                    help="ShortGPT: block importance layer removal (Wave 46).")
    ap.add_argument("--w4a8", action="store_true", default=False,
                    help="W4A8 hybrid-precision quantization runtime (Wave 46).")
    ap.add_argument("--expert-choice", action="store_true", default=False,
                    help="Expert Choice MoE routing: experts select top-k tokens (Wave 46).")
    ap.add_argument("--mla-kv", action="store_true", default=False,
                    help="MLA KV compression: low-rank joint K+V projection (Wave 46).")
    ap.add_argument("--minp", action="store_true", default=False,
                    help="Min-P sampler: dynamic probability floor (Wave 46).")
    ap.add_argument("--contrastive-search", action="store_true", default=False,
                    help="Contrastive search sampling: balance probability and diversity (Wave 46).")
    ap.add_argument("--razor-attn", action="store_true", default=False,
                    help="RazorAttention: retrieval-head classifier for 70%% KV reduction (Wave 46).")
    ap.add_argument("--cache-blend", action="store_true", default=False,
                    help="CacheBlend: partial KV prefix reuse for RAG (Wave 46).")
    ap.add_argument("--green-kv", action="store_true", default=False,
                    help="GreenKV: two-stream importance KV eviction (Wave 46).")
    ap.add_argument("--preble", action="store_true", default=False,
                    help="Preble router: prefix-cache-aware multi-instance request routing (Wave 46).")

    # ── Wave 47 flags ────────────────────────────────────────────────────────
    ap.add_argument("--mamba2-ssm", action="store_true", default=False,
                    help="Mamba2 SSM: Structured State-Space Duality layer (Wave 47).")
    ap.add_argument("--hgrn2", action="store_true", default=False,
                    help="HGRN2: gated linear RNN with state expansion (Wave 47).")
    ap.add_argument("--lookahead-decode", action="store_true", default=False,
                    help="Lookahead decoding: 2D Jacobi window without draft model (Wave 47).")
    ap.add_argument("--inf-memory", action="store_true", default=False,
                    help="InfLLM: block-level external KV memory for 1M+ context (Wave 47).")
    ap.add_argument("--v-attn", action="store_true", default=False,
                    help="vAttention: OS virtual memory KV cache management (Wave 47).")
    ap.add_argument("--ia3", action="store_true", default=False,
                    help="IA3 adapter: learned K/V/FF scale vectors (Wave 47).")
    ap.add_argument("--moe-infinity", action="store_true", default=False,
                    help="MoE-Infinity: activation-aware expert offloading (Wave 47).")
    ap.add_argument("--mega-blocks", action="store_true", default=False,
                    help="MegaBlocks: dropless MoE with block-sparse GEMM (Wave 47).")
    ap.add_argument("--kgw-watermark", action="store_true", default=False,
                    help="KGW watermark: green/red list vocabulary watermarking (Wave 47).")
    ap.add_argument("--typical-sampler", action="store_true", default=False,
                    help="Typical sampling: sample from local typical set (Wave 47).")
    ap.add_argument("--dora", action="store_true", default=False,
                    help="DoRA: magnitude-direction weight decomposition adapter (Wave 47).")
    ap.add_argument("--calm-exit", action="store_true", default=False,
                    help="CALM: per-token confidence-gated early exit (Wave 47).")

    # ── Wave 48 flags ────────────────────────────────────────────────────────
    ap.add_argument("--spqr", action="store_true", default=False,
                    help="SpQR: sparse-quantized representation with outlier FP16 (Wave 48).")
    ap.add_argument("--auto-round", action="store_true", default=False,
                    help="AutoRound: sign-gradient-descent rounding optimizer (Wave 48).")
    ap.add_argument("--owq", action="store_true", default=False,
                    help="OWQ: outlier-aware weight quantization with column promotion (Wave 48).")
    ap.add_argument("--bit-distiller", action="store_true", default=False,
                    help="BitDistiller: KL-divergence self-distillation for INT2 (Wave 48).")
    ap.add_argument("--zip-lm", action="store_true", default=False,
                    help="ZipLM: Hessian-sensitivity mixed-precision layer assignment (Wave 48).")
    ap.add_argument("--gguf-mixed", action="store_true", default=False,
                    help="GGUF mixed-precision: Q2_K/Q3_K/Q4_K block quantization format (Wave 48).")

    # ── Wave 49 flags ────────────────────────────────────────────────────────
    ap.add_argument("--llm-lingua2", action="store_true", default=False,
                    help="LLMLingua-2: token-level binary classifier prompt compression (Wave 49).")
    ap.add_argument("--recomp", action="store_true", default=False,
                    help="RECOMP: extractive+abstractive RAG context compression (Wave 49).")
    ap.add_argument("--selective-context", action="store_true", default=False,
                    help="Selective context: self-information pruning (Wave 49).")
    ap.add_argument("--prompt-cache", action="store_true", default=False,
                    help="PromptCache: schema-defined KV materialization for templates (Wave 49).")
    ap.add_argument("--pipe-infer", action="store_true", default=False,
                    help="PipeInfer: chunked prefill+decode pipeline overlap (Wave 49).")
    ap.add_argument("--prepack", action="store_true", default=False,
                    help="Prepack: completion-order batching for TTFT reduction (Wave 49).")

    # ── Wave 50 flags ────────────────────────────────────────────────────────
    ap.add_argument("--sparse-gpt", action="store_true", default=False,
                    help="SparseGPT: one-shot Hessian weight pruning 50-60%% (Wave 50).")
    ap.add_argument("--mix-of-depths", action="store_true", default=False,
                    help="Mixture-of-Depths: per-token layer routing (Wave 50).")
    ap.add_argument("--lean-kv", action="store_true", default=False,
                    help="LeanKV: asymmetric K(INT4)/V(INT8) cache quantization (Wave 50).")
    ap.add_argument("--gguf-loader", action="store_true", default=False,
                    help="GGUF native loader: Q2_K/Q3_K/Q4_K/Q5_K/Q8_0 format parser (Wave 50).")
    ap.add_argument("--weight-stream", action="store_true", default=False,
                    help="Weight decompress stream: overlapped CPU dequant + GPU compute (Wave 50).")
    ap.add_argument("--shard-loader", action="store_true", default=False,
                    help="Model shard loader: 3-tier GPU-hot/CPU-warm/SSD-cold weight paging (Wave 50).")

    # ── Wave 51 flags ────────────────────────────────────────────────────────
    ap.add_argument("--budget-forcing", action="store_true", default=False,
                    help="Budget forcing: s1-style per-request thinking token budget (Wave 51).")
    ap.add_argument("--test-time-scale", action="store_true", default=False,
                    help="Test-time compute router: difficulty-aware strategy dispatch (Wave 51).")
    ap.add_argument("--dvts", action="store_true", default=False,
                    help="DVTS: diverse verifier tree search for reasoning (Wave 51).")
    ap.add_argument("--chain-of-draft", action="store_true", default=False,
                    help="Chain-of-Draft: ≤7-word per-step reasoning constraint (Wave 51).")
    ap.add_argument("--coconut", action="store_true", default=False,
                    help="COCONUT: continuous latent reasoning decoder (Wave 51).")
    ap.add_argument("--prm-beam", action="store_true", default=False,
                    help="PRM beam search: step-level process reward model guidance (Wave 51).")
    ap.add_argument("--best-of-n", action="store_true", default=False,
                    help="Best-of-N sampling with reward model scoring (Wave 51).")
    ap.add_argument("--self-consistency", action="store_true", default=False,
                    help="Self-consistency: majority voting over K reasoning chains (Wave 51).")
    ap.add_argument("--thought-budget", action="store_true", default=False,
                    help="Thought budget gate: per-segment CoT token limiting (Wave 51).")
    ap.add_argument("--reasoning-kv", action="store_true", default=False,
                    help="Reasoning KV: INT2 thinking-region + FP16 answer-region KV (Wave 51).")
    ap.add_argument("--draft-reasoning", action="store_true", default=False,
                    help="Draft reasoning verifier: CoT-consistency speculative acceptance (Wave 51).")
    ap.add_argument("--parallel-reasoning", action="store_true", default=False,
                    help="Parallel reasoning scheduler: M chains per prompt (Wave 51).")

    # ── Wave 52 flags ────────────────────────────────────────────────────────
    ap.add_argument("--fast-v", action="store_true", default=False,
                    help="FastV: visual token pruning by cross-attention score at layer 2 (Wave 52).")
    ap.add_argument("--vision-zip", action="store_true", default=False,
                    help="VisionZip: context-dependent visual token selection (Wave 52).")
    ap.add_argument("--llava-prumerge", action="store_true", default=False,
                    help="LLaVA-PruMerge: spatial clustering+merge of visual patches (Wave 52).")
    ap.add_argument("--token-packer", action="store_true", default=False,
                    help="TokenPacker: fixed-size cross-attention visual projector (Wave 52).")
    ap.add_argument("--flash-vstream", action="store_true", default=False,
                    help="Flash-VStream: 3-tier video KV memory for streaming video (Wave 52).")
    ap.add_argument("--dynamic-res", action="store_true", default=False,
                    help="Dynamic resolution encoder: aspect-ratio tiling (Wave 52).")
    ap.add_argument("--visual-kv-quant", action="store_true", default=False,
                    help="Visual KV quantization: INT4K+INT6V for visual token blocks (Wave 52).")
    ap.add_argument("--cross-modal", action="store_true", default=False,
                    help="Cross-modal router: affinity-gated visual↔text attention (Wave 52).")
    ap.add_argument("--video-kv-reuse", action="store_true", default=False,
                    help="Video KV reuse: frame-pair cosine similarity KV sharing (Wave 52).")
    ap.add_argument("--vlm-spec", action="store_true", default=False,
                    help="VLM speculative decoding with shared visual prefix (Wave 52).")
    ap.add_argument("--vlm-sched", action="store_true", default=False,
                    help="VLM batch scheduler: image-complexity request binning (Wave 52).")
    ap.add_argument("--img-encoder-cache", action="store_true", default=False,
                    help="Image encoder output cache keyed by SHA-256 (Wave 52).")

    # ── Wave 53 flags ────────────────────────────────────────────────────────
    ap.add_argument("--rwkv6", action="store_true", default=False,
                    help="RWKV-6 Eagle channel-mix layer (Wave 53).")
    ap.add_argument("--hawk-rnn", action="store_true", default=False,
                    help="Hawk/Griffin Real-Gated Linear Recurrence (Wave 53).")
    ap.add_argument("--xlstm", action="store_true", default=False,
                    help="xLSTM: extended LSTM with sLSTM and mLSTM cells (Wave 53).")
    ap.add_argument("--ttt", action="store_true", default=False,
                    help="TTT layer: Test-Time Training linear layer (Wave 53).")
    ap.add_argument("--delta-net", action="store_true", default=False,
                    help="DeltaNet: delta-rule linear recurrent attention (Wave 53).")
    ap.add_argument("--ssm-cache", action="store_true", default=False,
                    help="SSM state cache: unified recurrent state persistence (Wave 53).")
    ap.add_argument("--parallel-scan", action="store_true", default=False,
                    help="Parallel scan kernel: Blelloch prefix scan for SSM prefill (Wave 53).")
    ap.add_argument("--ssm-quant", action="store_true", default=False,
                    help="SSM quantizer: SSM-specific calibration Δ/A/B/C quantization (Wave 53).")
    ap.add_argument("--hybrid-arch", action="store_true", default=False,
                    help="Hybrid arch router: per-layer SSM vs attention dispatch (Wave 53).")
    ap.add_argument("--hymba", action="store_true", default=False,
                    help="Hymba dual-track: parallel mini-SSM + attention heads (Wave 53).")
    ap.add_argument("--ssm-offload", action="store_true", default=False,
                    help="SSM state offload: segment-boundary recurrent state export (Wave 53).")

    # ── Wave 54 flags ────────────────────────────────────────────────────────
    ap.add_argument("--shared-expert", action="store_true", default=False,
                    help="Shared expert MoE: always-active shared + routed experts (Wave 54).")
    ap.add_argument("--fine-grained-moe", action="store_true", default=False,
                    help="Fine-grained MoE router: DeepSeek-V3 auxiliary-loss-free (Wave 54).")
    ap.add_argument("--expert-offload", action="store_true", default=False,
                    help="Expert offloader: top-M LRU expert weight pager (Wave 54).")
    ap.add_argument("--expert-merge", action="store_true", default=False,
                    help="Expert merger: cosine-similarity expert consolidation (Wave 54).")
    ap.add_argument("--lazy-expert", action="store_true", default=False,
                    help="Lazy expert loader: JIT expert weight materialization (Wave 54).")
    ap.add_argument("--expert-cache", action="store_true", default=False,
                    help="Expert activation cache: LRU expert output caching (Wave 54).")
    ap.add_argument("--flash-attn3", action="store_true", default=False,
                    help="FlashAttention-3: pingpong warp scheduling 1.5-2× FA-2 (Wave 54).")
    ap.add_argument("--double-sparse", action="store_true", default=False,
                    help="DoubleSparsity: simultaneous head-level and token-level sparse attn (Wave 54).")
    ap.add_argument("--lasp", action="store_true", default=False,
                    help="LASP: linear attention sequence parallelism via ring topology (Wave 54).")
    ap.add_argument("--nacl-cache", action="store_true", default=False,
                    help="NaCL cache: O(1) random KV eviction with non-evictable reserve (Wave 54).")
    ap.add_argument("--kv-migration", action="store_true", default=False,
                    help="KV migration manager: live KV shard transfer between workers (Wave 54).")
    ap.add_argument("--elastic-batch", action="store_true", default=False,
                    help="Elastic batch controller: adaptive continuous-batch sizing (Wave 54).")

    # ── Wave 55 flags ────────────────────────────────────────────────────────
    ap.add_argument("--min-p", action="store_true", default=False,
                    help="Min-P sampler (min_p_sampler): dynamic probability floor (Wave 55).")
    ap.add_argument("--mirostat", action="store_true", default=False,
                    help="Mirostat sampler: PID perplexity controller (Wave 55).")
    ap.add_argument("--eta-cutoff", action="store_true", default=False,
                    help="Eta-cutoff sampler: entropy-adaptive hard logit cutoff (Wave 55).")
    ap.add_argument("--cfg-guidance", action="store_true", default=False,
                    help="CFG logits sampler: Classifier-Free Guidance for text (Wave 55).")
    ap.add_argument("--diverse-beam", action="store_true", default=False,
                    help="Diverse beam search: G beam groups with diversity penalty (Wave 55).")
    ap.add_argument("--bitnet158", action="store_true", default=False,
                    help="BitNet-b1.58: ternary {-1,0,+1} weight quantization (Wave 55).")
    ap.add_argument("--spqr-quant", action="store_true", default=False,
                    help="SpQR quantizer (spqr_quant): sparse-quantized for frontier models (Wave 55).")
    ap.add_argument("--omniquant", action="store_true", default=False,
                    help="OmniQuant: LWC+LET joint calibration W4A4/W4A8 (Wave 55).")
    ap.add_argument("--q-sparse", action="store_true", default=False,
                    help="Q-Sparse: top-K activation sparsity at matmul time (Wave 55).")
    ap.add_argument("--fp4-quant", action="store_true", default=False,
                    help="FP4 quantization: E2M1 FP4 floating-point weights (Wave 55).")
    ap.add_argument("--ada-round", action="store_true", default=False,
                    help="AdaRound: adaptive rounding via per-weight sigmoid relaxation (Wave 55).")

    ap.add_argument("--lora-adapter", default="", metavar="PATH",
                    help="Path to LoRA adapter directory to load via LoRAManager.")
    ap.add_argument(
        "--quip",
        action="store_true",
        default=False,
        help="[Experimental] Signal that the loaded model was compressed with QuIP# "
             "E8 trellis-coded quantization (squish compress --quip).  "
             "The compressed_loader auto-detects quip_e8.npy files; this flag "
             "disables the finalized-cache write-back to avoid expanding the "
             "QuIP# format into a larger float16 cache on first load.",
    )
    ap.add_argument(
        "--all-optimizations", action="store_true", default=False,
        help=(
            "Enable ALL built-in optimization modules at once. "
            "Activates every attention kernel, KV cache strategy, speculative "
            "decoding variant, and adaptive-layer technique. Equivalent to "
            "passing every --sage-attention, --sparge-attention, --yoco-kv, "
            "--squeeze-attention, --kvtuner, --robust-scheduler, --gemfilter, "
            "--svdq, --sparse-spec, --sparse-verify, --trail, --specontext, "
            "--forelen, --ipw, --layer-skip, --long-spec, --fr-spec, --cla, "
            "--prompt-lookup, --seq-packing, --conf-spec, --kv-share, --kv-slab, "
            "--paris-kv, --streaming-sink, --diff-kv, --small-kv, --lookahead, "
            "--spec-reason flags simultaneously. "
            "Useful for local testing. Modules that fail to init are skipped."
        ),
    )
    # ── Phase 13D: Agent preset ───────────────────────────────────────────────
    ap.add_argument(
        "--agent", action="store_true", default=False,
        help=(
            "Agent-mode preset — enables the full Phase-13 agent stack:\n"
            "  --agent-kv       INT2 asymmetric KV cache (6× footprint reduction)\n"
            "  --chunk-prefill  bounded TTFT for long system prompts\n"
            "  --batch-size 1   dedicated-slot serving for agent loops\n"
            "  --max-kv-size    auto-sized from available UMA (min(32768, free_gb×2048))\n"
            "Designed for 7–14 B models on 16 GB M-series Apple Silicon "
            "running long tool-call agent loops."
        ),
    )

    # ── Wave 81: Blazing preset ───────────────────────────────────────────────
    ap.add_argument(
        "--blazing", action="store_true", default=False,
        help=(
            "Blazing-mode preset — targets sub-3 s TTFT for 7/8B models on "
            "16 GB M3 Apple Silicon:\n"
            "  --agent-kv            INT2 asymmetric KV cache (6× footprint)\n"
            "  --chunk-prefill-size 128  TTFT-optimised chunk size (vs 512/1024 default)\n"
            "  --fast-gelu           fast GELU approximation (no quality change)\n"
            "  --max-kv-size 4096    clamp context to preserve UMA headroom\n"
            "  Metal buffer pool → 64 MB (vs 256 MB default)\n"
            "  Two-pass JIT warmup (decode + chunked-prefill kernels pre-compiled)\n"
            "Requires an INT2/INT3/INT4 quantised model — NOT a raw BF16 model.\n"
            "Convert first:  squish convert-model --blazing-m3 <model>\n"
            "Combines cleanly with --agent for the full agent+speed stack."
        ),
    )

    # ── WhatsApp / Meta Cloud API integration ────────────────────────────────
    ap.add_argument(
        "--whatsapp", action="store_true", default=False,
        help=(
            "Enable the WhatsApp Cloud API webhook (POST /webhook/whatsapp). "
            "Requires --whatsapp-verify-token, --whatsapp-access-token, and "
            "--whatsapp-phone-number-id to function. "
            "Also reads WHATSAPP_VERIFY_TOKEN / WHATSAPP_ACCESS_TOKEN / "
            "WHATSAPP_PHONE_NUMBER_ID env vars as fallback."
        ),
    )
    ap.add_argument(
        "--whatsapp-verify-token", default="",
        help="Custom string set in the Meta App Dashboard to verify webhook ownership. "
             "Fallback: WHATSAPP_VERIFY_TOKEN env var.",
    )
    ap.add_argument(
        "--whatsapp-app-secret", default="",
        help="Meta App Secret (App Settings → Basic → App Secret). "
             "When provided, all incoming webhook payloads are validated with "
             "HMAC-SHA256; requests with missing/wrong signatures are rejected 403. "
             "Fallback: WHATSAPP_APP_SECRET env var.",
    )
    ap.add_argument(
        "--whatsapp-access-token", default="",
        help="Permanent or temporary access token for sending replies via the "
             "Meta Graph API (WhatsApp → API Setup → Access Token). "
             "Fallback: WHATSAPP_ACCESS_TOKEN env var.",
    )
    ap.add_argument(
        "--whatsapp-phone-number-id", default="",
        help="Phone Number ID from the Meta WhatsApp API Setup page. "
             "Fallback: WHATSAPP_PHONE_NUMBER_ID env var.",
    )

    # ── Signal / signal-cli integration ──────────────────────────────────────
    ap.add_argument(
        "--signal", action="store_true", default=False,
        help=(
            "Enable the Signal bot (GET /signal/status). "
            "Requires a running signal-cli JSON-RPC daemon and --signal-account. "
            "Also reads SIGNAL_ACCOUNT / SIGNAL_SOCKET env vars as fallback."
        ),
    )
    ap.add_argument(
        "--signal-account", default="",
        help="E.164 phone number registered in signal-cli (e.g. +12025551234). "
             "Fallback: SIGNAL_ACCOUNT env var.",
    )
    ap.add_argument(
        "--signal-socket", default="127.0.0.1:7583",
        help="signal-cli JSON-RPC daemon address: host:port or UNIX socket path. "
             "Fallback: SIGNAL_SOCKET env var. Default: 127.0.0.1:7583.",
    )

    args = ap.parse_args()

    # Capture parsed CLI flags so /debug-info can expose them at runtime.
    _server_args.update({k: str(v) for k, v in vars(args).items()})

    # ── Expand --all-optimizations into individual flags ─────────────────────
    if getattr(args, "all_optimizations", False):
        _bool_wave_flags = [
            "sage_attention", "sage_attention2", "sparge_attention",
            "squeeze_attention", "yoco_kv", "cla", "kvtuner",
            "robust_scheduler", "gemfilter", "svdq",
            "sparse_spec", "sparse_verify", "trail", "specontext",
            "forelen", "ipw", "layer_skip", "long_spec", "fr_spec",
            "prompt_lookup", "seq_packing", "ada_serve", "conf_spec",
            "kv_share", "kv_slab", "paris_kv", "streaming_sink",
            "diff_kv", "small_kv", "lookahead", "spec_reason",
            # Wave 37: Wire Everything In
            # NOTE: "jacobi" is intentionally excluded — Jacobi decode uses
            # full-context forward passes (no KV cache), making it O(n²) and
            # catastrophically slow for normal generation.  Enable only with
            # the explicit --jacobi flag.
            "kvtc", "chunk_kv", "ssd_saguaro", "spec_stream",
            "metal_flash_attn", "deja_vu", "mtp",
            "layer_overlap", "fused_qkv", "pd_disagg",
            # Wave 41
            "radix_attn", "eagle2", "ring_attn", "token_entropy_prune",
            "pregated_moe", "sink_fusion", "cla_share", "qmoe_compress",
            "lade", "infini_attn", "akvq", "delta_zip",
            # Wave 42
            "medusa_heads", "sarathi", "nsa_attn", "flex_prefill",
            "think_cache", "attention_store", "rest_decode", "star_attn",
            "splitwise", "kvquant", "efficient_qat", "cache_gen",
            # Wave 43
            "mtp_decode", "cascade_kv", "head_prune", "paged_attn_w43",
            "layer_collapse", "relay_attn", "wkv_quant", "tokenized_kv",
            "cluster_evict", "s2_attn", "magic_pig_v2",
            # Wave 44
            "marlin_gemm", "spec_rejection", "loftq", "online_spec",
            "dynamic_spec_len", "big_little", "multi_exit_spec", "pv_tuning",
            "hadamard_quant", "prefix_tree_decode", "spectr_ot", "ada_gptq",
            # Wave 45
            "flexgen_offload", "yarn_rope", "self_extend", "orca_sched",
            "mx_fp4", "fp8_act", "clex_rope", "powerinfer",
            "grouped_rope", "tensor_parallel", "fused_bias_gelu", "token_budget_sched",
            # Wave 46
            "slice_gpt", "wanda", "short_gpt", "w4a8",
            "expert_choice", "mla_kv", "minp", "contrastive_search",
            "razor_attn", "cache_blend", "green_kv", "preble",
            # Wave 47
            "mamba2_ssm", "hgrn2", "lookahead_decode", "inf_memory",
            "v_attn", "ia3", "moe_infinity", "mega_blocks",
            "kgw_watermark", "typical_sampler", "dora", "calm_exit",
            # Wave 48
            "spqr", "auto_round", "owq", "bit_distiller", "zip_lm", "gguf_mixed",
            # Wave 49
            "llm_lingua2", "recomp", "selective_context", "prompt_cache",
            "pipe_infer", "prepack",
            # Wave 50
            "sparse_gpt", "mix_of_depths", "lean_kv", "gguf_loader",
            "weight_stream", "shard_loader",
            # Wave 51
            "budget_forcing", "test_time_scale", "dvts", "chain_of_draft",
            "coconut", "prm_beam", "best_of_n", "self_consistency",
            "thought_budget", "reasoning_kv", "draft_reasoning", "parallel_reasoning",
            # Wave 52
            "fast_v", "vision_zip", "llava_prumerge", "token_packer",
            "flash_vstream", "dynamic_res", "visual_kv_quant", "cross_modal",
            "video_kv_reuse", "vlm_spec", "vlm_sched", "img_encoder_cache",
            # Wave 53
            "rwkv6", "hawk_rnn", "xlstm", "ttt", "delta_net",
            "ssm_cache", "parallel_scan", "ssm_quant", "hybrid_arch",
            "hymba", "ssm_offload",
            # Wave 54
            "shared_expert", "fine_grained_moe", "expert_offload", "expert_merge",
            "lazy_expert", "expert_cache", "flash_attn3", "double_sparse",
            "lasp", "nacl_cache", "kv_migration", "elastic_batch",
            # Wave 55
            "min_p", "mirostat", "eta_cutoff", "cfg_guidance",
            "diverse_beam", "bitnet158", "spqr_quant", "omniquant",
            "q_sparse", "fp4_quant", "ada_round",
        ]
        for _f in _bool_wave_flags:
            if not getattr(args, _f, False):
                setattr(args, _f, True)

    # ── Phase 13D: Expand --agent preset into individual flags ────────────────
    if getattr(args, "agent", False):
        # 1. Asymmetric INT2 KV cache
        args.agent_kv = True
        # 2. Bounded TTFT via chunked prefill (COMPRESS_PATH only)
        args.chunk_prefill = True
        # 3. Single-slot serving — agent loops occupy one context at a time
        if getattr(args, "batch_size", 8) >= 8:   # don't override an explicit lower value
            args.batch_size = 1
        # 4. Auto-size context window from available UMA reported by MemoryGovernor
        if getattr(args, "max_kv_size", None) is None:
            try:
                import sys as _sys
                if _sys.platform == "darwin":
                    from squish.serving.memory_governor import (
                        MemoryGovernor as _MG,  # noqa: PLC0415
                    )
                    _mg_tmp = _MG(poll_interval=60.0).start()
                    _free_gb = _mg_tmp.available_gb
                    _mg_tmp.stop()
                    args.max_kv_size = min(32768, int(_free_gb * 2048))
                else:
                    args.max_kv_size = 8192
            except Exception:  # noqa: BLE001
                args.max_kv_size = 8192
        _info("agent-preset",
              f"active  agent-kv=True  chunk-prefill=True"
              f"  batch={args.batch_size}  max-kv={args.max_kv_size}")

    # ── Wave 81: Blazing mode expansion ───────────────────────────────────────
    _configure_blazing_mode(args)

    global _API_KEY
    # Prefer explicit CLI flag; fall back to SQUISH_API_KEY env var.
    # Reading from env var prevents the secret appearing in `ps aux`.
    _API_KEY = args.api_key or os.environ.get("SQUISH_API_KEY")

    # ── Structured logging ────────────────────────────────────────────────────
    if _TELEMETRY_AVAILABLE:
        _configure_logging(level=getattr(args, "log_level", "warning"))

    # ── Structured span tracing ───────────────────────────────────────────────
    global _trace, _trace_tokens, _trace_file
    _trace        = args.trace or args.trace_tokens
    _trace_tokens = args.trace_tokens
    if _trace and _TELEMETRY_AVAILABLE:
        _configure_tracing(True)
    if args.trace_file:
        try:
            _trace_file = open(args.trace_file, "a", buffering=1)  # noqa: WPS515
        except OSError as _tf_err:
            _warn(f"[trace] Could not open trace file {args.trace_file!r}: {_tf_err}")

    if args.no_prefix_cache:
        _prefix_cache._maxsize = 0
    elif args.prefix_cache_size != 512:
        _prefix_cache._maxsize = args.prefix_cache_size

    # ── Phase 2A/2B: PagedKVCache + RadixTree prefix trie ────────────────────
    global _paged_kv_cache
    if getattr(args, "paged_attention", False) and _state.model is not None:
        try:
            from squish.kv.paged_attention import PagedKVCache as _PagedKVCache
            _paged_kv_cache = _PagedKVCache.from_model(
                _state.model,
                metal_fraction=getattr(args, "paged_attention_fraction", 0.25),
            )
            s = _paged_kv_cache.stats()
            _ok("Paged KV cache ready")
            _info("paged-kv-blocks",
                  f"{s['total_blocks']} blocks  "
                  f"({s['memory_mb']} MB  page={s['page_size']}tok  "
                  f"{s['n_layers']}L×{s['n_kv_heads']}H×{s['head_dim']}d)")
        except Exception as _paged_err:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[paged-attention] could not initialise (%s) — disabled", _paged_err
            )

    _print_banner()

    if getattr(args, "mlx_model_dir", ""):
        _info("model", f"{args.mlx_model_dir}  {_C.DIM}(mlx_lm INT4){_C.R}")
    else:
        _info("model-dir", args.model_dir)
        _info("compressed", args.compressed_dir)
    if args.draft_model:
        _info("draft-model", args.draft_model)
    if getattr(args, "eagle_head_dir", ""):
        _info("eagle-head", args.eagle_head_dir)
    _info("prefix-cache", "disabled" if args.no_prefix_cache else str(args.prefix_cache_size))
    if args.kv_cache_mode != "fp16":
        _info("kv-cache", f"{args.kv_cache_mode}  window={args.kv_cache_window}  budget={args.kv_cache_budget}")
    _info("listen", f"http://{args.host}:{args.port}")
    if _trace:
        _info("trace", f"ON  tokens={'yes' if _trace_tokens else 'no'}"
              f"{'  file=' + args.trace_file if args.trace_file else ''}")
    print()

    with _trace_span("server.model_load",
                     mlx=bool(getattr(args, "mlx_model_dir", "")),
                     model_dir=getattr(args, "mlx_model_dir", "") or args.compressed_dir) as _model_load_span:
        if getattr(args, "mlx_model_dir", ""):
            load_mlx_model(args.mlx_model_dir, verbose=args.verbose)
        else:
            load_model(args.model_dir, args.compressed_dir, verbose=args.verbose)
        # Update span tags to reflect the *actual* loader rather than whether
        # --mlx-model-dir was explicitly passed.  "mlx-native" and "squish-4bit"
        # both use mlx_lm.load() and keep weights in INT4; other loaders may
        # dequantize to bfloat16.  This disambiguates the trace for diagnostics.
        _actual_loader = _state.loader_tag or "unknown"
        _mlx_backed_loaders = frozenset({"mlx-native", "squish-4bit"})
        _model_load_span.set_tag("loader", _actual_loader)
        _model_load_span.set_tag("mlx", _actual_loader in _mlx_backed_loaders)
    _state._no_compile = args.no_compile  # propagate --no-compile flag

    # ── Disk prompt-cache init (Item 2) ──────────────────────────────────────
    global _disk_prompt_cache
    if getattr(args, "disk_prompt_cache", ""):
        try:
            from squish.kv_cache import DiskKVCache as _DiskKVCache
        except ImportError:
            from kv_cache import DiskKVCache as _DiskKVCache  # direct run
        _disk_prompt_cache = _DiskKVCache(
            cache_dir   = args.disk_prompt_cache,
            max_entries = args.disk_prompt_cache_size,
        )
        if args.verbose:
            _info("disk-cache", f"{args.disk_prompt_cache}  {_C.DIM}(max {args.disk_prompt_cache_size} entries){_C.R}")

    # ── LazyLLM token-pruning init (Item 3) ──────────────────────────────────
    global _lazy_llm_state
    if getattr(args, "lazy_llm", False) and _state.model is not None:
        try:
            try:
                from squish.lazy_llm import LazyLLMConfig
                from squish.lazy_llm import patch_model_lazy_llm as _patch_llm
            except ImportError:
                from lazy_llm import LazyLLMConfig
                from lazy_llm import patch_model_lazy_llm as _patch_llm
            _lazy_llm_cfg = LazyLLMConfig(
                keep_ratio    = args.lazy_llm_keep_ratio,
                start_layer   = args.lazy_llm_start_layer,
                revive_window = args.lazy_llm_revive_window,
                verbose       = _trace,   # tie to --trace flag
            )
            _lazy_llm_state = _patch_llm(_state.model, _lazy_llm_cfg)
            if args.verbose:
                _info("lazy-llm", f"keep={args.lazy_llm_keep_ratio}  "
                      f"start_layer={args.lazy_llm_start_layer}  "
                      f"revive={args.lazy_llm_revive_window}")
        except Exception as _llm_err:
            _warn(f"[lazy_llm] Skipped: {_llm_err}")

    if _state.model is not None:
        try:
            from squish.io.split_loader import SplitLayerLoader
            _split_info = SplitLayerLoader.auto_split(_state.model, verbose=True)
            if _split_info:
                _info("cpu/gpu split", f"{_split_info.cpu_count} layers offloaded  "
                      f"GPU={_split_info.gpu_gb:.2f}GB  CPU={_split_info.cpu_gb:.2f}GB")
        except Exception as e:
            if args.verbose:
                _warn(f"[split_loader] Skipped: {e}")

    # ── Phase 2.3: Flash Attention status check ──────────────────────────────
    if _state.model is not None:
        try:
            from squish.attention.flash_attention import patch_model_attention
            patch_model_attention(_state.model, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                _warn(f"[flash_attention] Skipped: {e}")

    # ── Phase 1.3: attach quantized KV cache if requested ─────────────
    global _kv_cache
    if args.kv_cache_mode != "fp16" and _state.model is not None:
        try:
            from squish.kv_cache import patch_model_kv_cache
            _kv_cache = patch_model_kv_cache(
                _state.model,
                mode=args.kv_cache_mode,
                window=args.kv_cache_window,
                budget=args.kv_cache_budget,
                svd_rank=getattr(args, "kv_cache_svd_rank", 0),
                verbose=True,
            )
            _info("kv-cache", f"ready ({args.kv_cache_mode})")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[KV cache] could not attach (%s) — running without KV quantisation", e
            )

    # ── Phase 13A: Asymmetric INT2 KV cache (AgentKV) ────────────────────────
    global _agent_kv_config
    if getattr(args, "agent_kv", False):
        try:
            from squish.kv.agent_kv import AgentKVConfig  # noqa: PLC0415
            _agent_kv_config = AgentKVConfig(
                sink_tokens=getattr(args, "agent_kv_sink", 4),
                window_tokens=getattr(args, "agent_kv_window", 64),
            )
            _info("agent-kv",
                  f"enabled  sink={_agent_kv_config.sink_tokens}"
                  f"  window={_agent_kv_config.window_tokens}")
        except Exception as _akv_exc:  # noqa: BLE001
            _info("agent-kv", f"unavailable ({_akv_exc})")

    # ── Phase 3: persistent cross-session KV cache ────────────────────────────
    global _session_kv_cache
    _session_cache_dir = getattr(args, "session_cache_dir", "")
    if _session_cache_dir:
        try:
            from squish.kv_cache import SessionKVCache as _SessionKVCache
            _session_kv_cache = _SessionKVCache(cache_dir=_session_cache_dir)
            _info("session-cache", f"{_session_cache_dir}")
        except Exception as _e:
            _warn(f"[session-cache] Could not enable: {_e}")

    # ── Phase 4: prompt compression settings ─────────────────────────────────
    global _compress_enabled, _compress_ratio, _compress_min_tokens, _compress_preserve_tokens
    _compress_enabled        = getattr(args, "compress_prompt", False)
    _compress_ratio          = getattr(args, "compress_ratio", 0.5)
    _compress_min_tokens     = getattr(args, "compress_min_tokens", 512)
    _compress_preserve_tokens = getattr(args, "compress_preserve_tokens", 0)
    if _compress_enabled:
        _info("compress", f"ratio={_compress_ratio}  min_tokens={_compress_min_tokens}"
              + (f"  preserve_tokens={_compress_preserve_tokens}" if _compress_preserve_tokens else ""))

    # ── Phase E1: Babbling suppression settings ───────────────────────────────
    global _babbling_suppression, _babbling_eos_threshold, _babbling_min_tokens
    _babbling_suppression    = getattr(args, "babbling_suppression", False)
    _babbling_eos_threshold  = getattr(args, "babbling_eos_threshold", 0.30)
    _babbling_min_tokens     = getattr(args, "babbling_min_tokens", 10)
    if _babbling_suppression:
        _info("babbling-suppression",
              f"enabled  eos_threshold={_babbling_eos_threshold}  min_tokens={_babbling_min_tokens}")

    # ── Phase E2: Polynomial GELU ─────────────────────────────────────────────
    global _fast_gelu_enabled
    _fast_gelu_enabled = getattr(args, "fast_gelu", False)
    if _fast_gelu_enabled and _state.model is not None:
        _model_dir_for_gelu = getattr(args, "model_dir", "") or getattr(args, "mlx_model_dir", "")
        if _model_dir_for_gelu:
            _apply_fast_gelu(_model_dir_for_gelu)

    # ── Phase E3: Semantic response cache ─────────────────────────────────────
    global _semantic_cache
    if getattr(args, "semantic_cache", False):
        try:
            from squish.semantic_cache import SquishSemanticCache  # noqa: PLC0415
            _sc_db = getattr(args, "semantic_cache_db", "") or \
                     str(Path.home() / ".squish" / "response_cache.db")
            _semantic_cache = SquishSemanticCache(db_path=_sc_db)
            _info("semantic-cache", f"enabled  db={_sc_db}")
        except Exception as _sc_err:
            _warn(f"[semantic-cache] Could not enable: {_sc_err}\n"
                  "Install sqlite-vec: pip install 'squish[cache]'")

    # ── Phase 3A: chunked prefill settings ───────────────────────────────────
    # On by default (Wave 75): eliminates event-loop blocking for long prompts.
    # Disable with --no-chunk-prefill; the legacy --chunk-prefill flag is a no-op.
    global _chunk_prefill_enabled, _chunk_prefill_threshold, _chunk_prefill_size
    _chunk_prefill_enabled   = not getattr(args, "no_chunk_prefill", False)
    _chunk_prefill_threshold = getattr(args, "chunk_prefill_threshold", 512)
    _chunk_prefill_size      = getattr(args, "chunk_prefill_size", 512)
    if _chunk_prefill_enabled:
        _info("chunk-prefill",
              f"on-by-default  threshold={_chunk_prefill_threshold}  chunk={_chunk_prefill_size}")

    # ── Phase 3C: MInference settings ────────────────────────────────────────
    global _minference_enabled, _minference_threshold, _inference_backend
    _minference_enabled   = getattr(args, "minference", False)
    _minference_threshold = getattr(args, "minference_threshold", 1024)
    if _minference_enabled:
        if _inference_backend == "ane-disagg":
            _warn("[minference] disabled — incompatible with --inference-backend ane-disagg")
            _minference_enabled = False
        else:
            _info("minference", f"sparse-attention  threshold={_minference_threshold}")

    # ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────
    global _thinking_budget, _think_close_token_id
    _thinking_budget = getattr(args, "thinking_budget", -1)
    if _thinking_budget >= 0 and _state.tokenizer is not None:
        try:
            _think_close_token_id = _state.tokenizer.convert_tokens_to_ids("</think>")
        except Exception:
            _think_close_token_id = None
    if _thinking_budget == 0:
        _info("thinking-budget", "disabled (no_think mode)")
    elif _thinking_budget > 0:
        _info("thinking-budget", f"{_thinking_budget} tokens  close_id={_think_close_token_id}")

    # ── Phase A2: max KV size ─────────────────────────────────────────────────
    global _max_kv_size
    _max_kv_size = getattr(args, "max_kv_size", None)
    if _max_kv_size is not None:
        _info("max-kv-size", f"{_max_kv_size} tokens")

    # ── Phase A3: concise responses ───────────────────────────────────────────
    global _concise_responses
    _concise_responses = getattr(args, "concise_responses", False)
    if _concise_responses:
        _info("concise-responses", "enabled")

    # ── Phase B: Structured output (XGrammar) ─────────────────────────────────
    global _grammar_engine, _structured_output_mode, _structured_output_schema
    _structured_output_mode = getattr(args, "structured_output", "none")
    if _structured_output_mode != "none" and _state.tokenizer is not None:
        from squish.grammar_engine import GrammarEngine  # noqa: PLC0415
        if GrammarEngine.is_available():
            _grammar_engine = GrammarEngine(_state.tokenizer)
            if _structured_output_mode == "json-schema":
                _schema_path = getattr(args, "structured_output_schema", None)
                if _schema_path:
                    import json as _json  # noqa: PLC0415
                    with open(_schema_path) as _sf:
                        _structured_output_schema = _json.load(_sf)
            _info("structured-output", f"mode={_structured_output_mode}")
        else:
            _warn("[structured-output] xgrammar not installed; "
                  "falling back to unconstrained generation. "
                  "Install: pip install 'squish[grammar]'")

    # ── Phase C: Power & Energy Modes ─────────────────────────────────────────
    global _power_monitor, _power_mode
    _power_mode = getattr(args, "power_mode", "performance")
    if _power_mode == "auto":
        from squish.power_monitor import PowerMonitor, apply_mode  # noqa: PLC0415
        _power_monitor = PowerMonitor()
        _initial_mode = _power_monitor.get_recommended_mode()
        apply_mode(_initial_mode, globals())
        _power_mode = _initial_mode
        _info("power-mode", f"auto  initial={_initial_mode}")
        # Background timer: re-evaluate and apply every 30 s
        import threading as _threading  # noqa: PLC0415
        def _power_auto_tick() -> None:
            global _power_mode
            if _power_monitor is None:
                return
            _new_mode = _power_monitor.get_recommended_mode()
            if _new_mode != _power_mode:
                from squish.power_monitor import apply_mode as _am  # noqa: PLC0415
                _am(_new_mode, globals())
                _power_mode = _new_mode
                _info("power-mode", f"switched → {_new_mode}")
            _t = _threading.Timer(30.0, _power_auto_tick)
            _t.daemon = True
            _t.start()
        _pt = _threading.Timer(30.0, _power_auto_tick)
        _pt.daemon = True
        _pt.start()
    elif _power_mode != "performance":
        from squish.power_monitor import apply_mode  # noqa: PLC0415
        apply_mode(_power_mode, globals())
        _info("power-mode", _power_mode)

    # ── Phase 13B: macOS Memory Governor ──────────────────────────────────────
    import sys as _sys
    if _sys.platform == "darwin":
        global _memory_governor
        try:
            from squish.serving.memory_governor import MemoryGovernor  # noqa: PLC0415
            _memory_governor = MemoryGovernor(poll_interval=5.0).start()
            _info("memory-governor",
                  f"started  available={_memory_governor.available_gb:.1f} GB"
                  f"  pressure={_memory_governor.pressure_level}")
        except Exception as _mg_exc:  # noqa: BLE001
            _info("memory-governor", f"unavailable ({_mg_exc})")

    # ── Phase 0C: hardware inference backend ─────────────────────────────────
    _inference_backend = getattr(args, "inference_backend", "mlx-eager")
    if _inference_backend != "mlx-eager":
        _info("inference-backend", _inference_backend)

    # ── Phase 2.1: start batch scheduler if requested ────────────────────────
    global _scheduler
    if args.batch_scheduler and _state.model is not None:
        try:
            from squish.scheduler import BatchScheduler, NestedWaitScheduler
            from squish.scheduler import QueueFullError as _QFE
            global _QueueFullError
            _QueueFullError = _QFE
            _sched_cls = (BatchScheduler
                          if getattr(args, "scheduler", "nested-wait") == "legacy"
                          else NestedWaitScheduler)
            _scheduler = _sched_cls(
                _state.model, _state.tokenizer,
                max_batch_size  = args.batch_size,
                batch_window_ms = args.batch_window_ms,
            )
            _scheduler.start()
            _info("batch-scheduler",
                  f"enabled  algo={getattr(args, 'scheduler', 'nested-wait')}  "
                  f"max_batch={args.batch_size}  window={args.batch_window_ms:.0f}ms")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[Scheduler] could not start (%s) — falling back to sequential mode", e
            )
            _scheduler = None

    if args.draft_model:
        print()
        load_draft_model(args.draft_model, args.draft_compressed, verbose=args.verbose)

    if getattr(args, "eagle_head_dir", ""):
        print()
        load_eagle_head(args.eagle_head_dir, verbose=args.verbose)

    # ── Wave optimization module initialisation ───────────────────────────────
    global _prompt_lookup_decoder, _seq_packer, _ada_serve_scheduler
    global _conf_spec_verifier, _kvsharer_map, _kv_slab_allocator
    global _paris_kv_codebook, _streaming_sink_cache
    global _diffkv_policy_mgr, _smallkv_cache, _lookahead_engine, _spec_reason_orch
    global _sage_attn_kernel, _sage_attn2_kernel, _sparge_engine, _squeeze_cache
    global _yoco_config, _cla_config, _kvtuner_config, _robust_sched
    global _gemfilter_config, _svdq_config, _sparse_spec_config, _sparse_verify_config
    global _trail_config, _specontext_config, _forelen_config, _ipw_config
    global _layer_skip_config, _long_spec_config, _fr_spec_config

    if getattr(args, "prompt_lookup", False):
        try:
            from squish.prompt_lookup import PromptLookupConfig, PromptLookupDecoder
            _plcfg = PromptLookupConfig(
                ngram_min=2,
                ngram_max=getattr(args, "prompt_lookup_n", 3),
                max_speculative=getattr(args, "prompt_lookup_k", 4),
            )
            # PromptLookupDecoder needs the forward callable; defer full init to inference.
            # Store config now; decoder is instantiated on first generation call.
            _prompt_lookup_decoder = _plcfg  # type: ignore[assignment]
            _info("prompt-lookup", f"ngram_max={_plcfg.ngram_max}  max_speculative={_plcfg.max_speculative}")
        except Exception as _e:
            _warn(f"[prompt-lookup] Skipped: {_e}")

    if getattr(args, "seq_packing", False):
        try:
            from squish.seq_packing import PackingConfig, SequencePacker
            _spcfg = PackingConfig(max_packed_length=getattr(args, "seq_packing_budget", 2048))
            _seq_packer = SequencePacker(_spcfg)
            _info("seq-packing", f"max_packed_length={_spcfg.max_packed_length}")
        except Exception as _e:
            _warn(f"[seq-packing] Skipped: {_e}")

    if getattr(args, "ada_serve", False):
        try:
            from squish.ada_serve import BUILT_IN_SLOS, AdaServeConfig, AdaServeScheduler
            _slo_name = getattr(args, "ada_serve_slo", "general")
            _ada_slo = BUILT_IN_SLOS.get(_slo_name, BUILT_IN_SLOS["general"])
            _ada_cfg = AdaServeConfig()
            _ada_serve_scheduler = AdaServeScheduler(_ada_cfg)
            _ada_serve_scheduler.register_slo(_slo_name, _ada_slo)
            _info("ada-serve", f"slo={_slo_name}  min_γ={_ada_cfg.min_gamma}  max_γ={_ada_cfg.max_gamma}")
        except Exception as _e:
            _warn(f"[ada-serve] Skipped: {_e}")

    if getattr(args, "conf_spec", False):
        try:
            from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
            _cscfg = ConfSpecConfig(
                high_gate=getattr(args, "conf_spec_high_gate", 0.90),
                low_gate=getattr(args, "conf_spec_low_gate", 0.50),
            )
            _conf_spec_verifier = ConfSpecVerifier(_cscfg)
            _info("conf-spec", f"high_gate={_cscfg.high_gate}  low_gate={_cscfg.low_gate}")
        except Exception as _e:
            _warn(f"[conf-spec] Skipped: {_e}")

    if getattr(args, "kv_share", False):
        try:
            from squish.kvsharer import KVSharerConfig
            _kvshr_cfg = KVSharerConfig()
            _kvshr_cfg._server_enabled = True
            _kvsharer_map = _kvshr_cfg
            _info("kv-share", f"max_share={_kvshr_cfg.max_share_fraction}  n_layers={_kvshr_cfg.n_layers}")
        except Exception as _e:
            _warn(f"[kv-share] Skipped: {_e}")

    if getattr(args, "kv_slab", False):
        try:
            from squish.kv_slab import KVSlabAllocator
            _kv_slab_allocator = KVSlabAllocator(n_pages=getattr(args, "kv_slab_pages", 256))
            _info("kv-slab", f"pages={getattr(args, 'kv_slab_pages', 256)}")
        except Exception as _e:
            _warn(f"[kv-slab] Skipped: {_e}")

    if getattr(args, "paris_kv", False):
        try:
            from squish.paris_kv import ParisKVCodebook, ParisKVConfig
            _paris_cfg = ParisKVConfig()
            _paris_kv_codebook = ParisKVCodebook(
                dim=128,
                n_codes=getattr(args, "paris_kv_centroids", 64),
                config=_paris_cfg,
            )
            _info("paris-kv", f"codes={_paris_kv_codebook._n_codes}  lr={_paris_cfg.learning_rate}")
        except Exception as _e:
            _warn(f"[paris-kv] Skipped: {_e}")

    if getattr(args, "streaming_sink", False):
        try:
            from squish.streaming_sink import SinkConfig, SinkKVCache
            _sink_cfg = SinkConfig(window_size=getattr(args, "streaming_sink_size", 2048))
            _streaming_sink_cache = SinkKVCache(_sink_cfg)
            _info("streaming-sink", f"sinks={_sink_cfg.num_sinks}  window={_sink_cfg.window_size}")
        except Exception as _e:
            _warn(f"[streaming-sink] Skipped: {_e}")

    if getattr(args, "diff_kv", False):
        try:
            from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
            _diffkv_cfg = DiffKVConfig()
            _diffkv_policy_mgr = DiffKVPolicyManager(_diffkv_cfg)
            _info("diff-kv", f"critical={_diffkv_cfg.critical_fraction}  marginal={_diffkv_cfg.marginal_fraction}")
        except Exception as _e:
            _warn(f"[diff-kv] Skipped: {_e}")

    if getattr(args, "small_kv", False):
        try:
            from squish.smallkv import SmallKVCache, SmallKVConfig
            _smallkv_cfg = SmallKVConfig()
            _smallkv_cache = SmallKVCache(_smallkv_cfg)
            _info("small-kv", f"budget={_smallkv_cfg.kv_budget_fraction}  recall_k={_smallkv_cfg.recall_top_k}")
        except Exception as _e:
            _warn(f"[small-kv] Skipped: {_e}")

    if getattr(args, "lookahead", False):
        try:
            from squish.lookahead_reasoning import LookaheadConfig, LookaheadReasoningEngine
            _la_cfg = LookaheadConfig(lookahead_k=getattr(args, "lookahead_k", 4))
            # draft_fn is wired to the actual model at inference time; store config only
            _la_cfg._server_enabled = True  # marker checked during generation
            _info("lookahead", f"k={_la_cfg.lookahead_k}  family={_la_cfg.model_family}")
        except Exception as _e:
            _warn(f"[lookahead] Skipped: {_e}")

    if getattr(args, "spec_reason", False):
        try:
            from squish.spec_reason import SpecReasonConfig
            _sr_cfg = SpecReasonConfig()
            _sr_cfg._server_enabled = True  # marker checked during generation
            _info("spec-reason", f"min_score={_sr_cfg.min_acceptance_score}  max_draft={_sr_cfg.max_draft_steps}")
        except Exception as _e:
            _warn(f"[spec-reason] Skipped: {_e}")

    # ── Attention and KV kernels ─────────────────────────────────────────────
    if getattr(args, "sage_attention", False):
        try:
            from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
            _sage_attn_kernel = SageAttentionKernel(SageAttentionConfig())
            _info("sage-attention", "INT8 QK^T kernel ready  (~2.1× attention speedup)")
        except Exception as _e:
            _warn(f"[sage-attention] Skipped: {_e}")

    if getattr(args, "sage_attention2", False):
        try:
            from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
            _sage_attn2_kernel = SageAttention2Kernel(SageAttention2Config())
            _info("sage-attention2", "INT4/FP8 kernel ready  (~3.1× attention speedup)")
        except Exception as _e:
            _warn(f"[sage-attention2] Skipped: {_e}")

    if getattr(args, "sparge_attention", False):
        try:
            from squish.sparge_attn import SpargeAttnConfig, SpargeAttnEngine
            _sparge_engine = SpargeAttnEngine(SpargeAttnConfig())
            _info("sparge-attention", "sparse+quantized attention engine ready  (2.5–5× speedup)")
        except Exception as _e:
            _warn(f"[sparge-attention] Skipped: {_e}")

    if getattr(args, "squeeze_attention", False):
        try:
            from squish.squeeze_attention import LayerKVBudget, SqueezeConfig, SqueezeKVCache
            _sq_cfg = SqueezeConfig()
            _sq_budgets = [
                LayerKVBudget(layer_idx=i, token_budget=_sq_cfg.total_kv_budget // _sq_cfg.n_layers)
                for i in range(_sq_cfg.n_layers)
            ]
            _squeeze_cache = SqueezeKVCache(budgets=_sq_budgets, config=_sq_cfg)
            _info("squeeze-attention", f"adaptive KV budget: {_sq_cfg.total_kv_budget} total tokens across {_sq_cfg.n_layers} layers")
        except Exception as _e:
            _warn(f"[squeeze-attention] Skipped: {_e}")

    # ── KV cache strategies ──────────────────────────────────────────────────
    if getattr(args, "yoco_kv", False):
        try:
            from squish.yoco import YOCOConfig
            _yoco_config = YOCOConfig()
            _yoco_config._server_enabled = True
            _info("yoco-kv", f"cross-layer KV reuse enabled  (self_attn_layers={_yoco_config.n_self_attn_layers})")
        except Exception as _e:
            _warn(f"[yoco-kv] Skipped: {_e}")

    if getattr(args, "cla", False):
        try:
            from squish.cla import CLAConfig
            _cla_config = CLAConfig()
            _cla_config._server_enabled = True
            _info("cla", f"cross-layer attention enabled  (sharing_factor={_cla_config.sharing_factor})")
        except Exception as _e:
            _warn(f"[cla] Skipped: {_e}")

    if getattr(args, "kvtuner", False):
        try:
            from squish.kvtuner import KVTunerConfig
            _kvtuner_config = KVTunerConfig()
            _kvtuner_config._server_enabled = True
            _info("kvtuner", f"adaptive KV budget  (target_avg_bits={_kvtuner_config.target_avg_bits})")
        except Exception as _e:
            _warn(f"[kvtuner] Skipped: {_e}")

    if getattr(args, "robust_scheduler", False):
        try:
            from squish.robust_scheduler import AMaxScheduler, RobustSchedulerConfig
            _robust_sched = AMaxScheduler(RobustSchedulerConfig())
            _info("robust-scheduler", f"A-max scheduling enabled  (max_batch_tokens={_robust_sched._config.max_batch_tokens})")
        except Exception as _e:
            _warn(f"[robust-scheduler] Skipped: {_e}")

    if getattr(args, "gemfilter", False):
        try:
            from squish.gemfilter import GemFilterConfig
            _gemfilter_config = GemFilterConfig()
            _gemfilter_config._server_enabled = True
            _info("gemfilter", f"attention head filter  (top_k_tokens={_gemfilter_config.top_k_tokens})")
        except Exception as _e:
            _warn(f"[gemfilter] Skipped: {_e}")

    if getattr(args, "svdq", False):
        try:
            from squish.svdq import SVDqConfig
            _svdq_config = SVDqConfig()
            _svdq_config._server_enabled = True
            _info("svdq", f"SVD KV quantization  (target_avg_bits={_svdq_config.target_avg_bits})")
        except Exception as _e:
            _warn(f"[svdq] Skipped: {_e}")

    # ── Speculative decoding variants ─────────────────────────────────────────
    if getattr(args, "sparse_spec", False):
        try:
            from squish.sparse_spec import SparseSpecConfig
            _sparse_spec_config = SparseSpecConfig()
            _sparse_spec_config._server_enabled = True
            _info("sparse-spec", f"sparse speculative decoding  (gamma={_sparse_spec_config.gamma}  top_k_ratio={_sparse_spec_config.top_k_ratio})")
        except Exception as _e:
            _warn(f"[sparse-spec] Skipped: {_e}")

    if getattr(args, "sparse_verify", False):
        try:
            from squish.sparse_verify import SparseVerifyConfig
            _sparse_verify_config = SparseVerifyConfig()
            _sparse_verify_config._server_enabled = True
            _info("sparse-verify", f"sparse draft verification  (attn_sparsity={_sparse_verify_config.attn_sparsity})")
        except Exception as _e:
            _warn(f"[sparse-verify] Skipped: {_e}")

    if getattr(args, "long_spec", False):
        try:
            from squish.long_spec import LongSpecConfig
            _long_spec_config = LongSpecConfig()
            _long_spec_config._server_enabled = True
            _info("long-spec", f"extended speculative decoding  (gamma={_long_spec_config.gamma}  max_context={_long_spec_config.max_context_len})")
        except Exception as _e:
            _warn(f"[long-spec] Skipped: {_e}")

    if getattr(args, "fr_spec", False):
        try:
            from squish.fr_spec import FRSpecConfig
            _fr_spec_config = FRSpecConfig()
            _fr_spec_config._server_enabled = True
            _info("fr-spec", f"frequency-token speculative  (top_k_fraction={_fr_spec_config.top_k_fraction})")
        except Exception as _e:
            _warn(f"[fr-spec] Skipped: {_e}")

    # ── Token-importance / adaptive-layer strategies ──────────────────────────
    if getattr(args, "trail", False):
        try:
            from squish.trail import TrailConfig
            _trail_config = TrailConfig()
            _trail_config._server_enabled = True
            _info("trail", f"token-importance layer skipping  (probe_layer={_trail_config.probe_layer})")
        except Exception as _e:
            _warn(f"[trail] Skipped: {_e}")

    if getattr(args, "specontext", False):
        try:
            from squish.specontext import SpeContextConfig
            _specontext_config = SpeContextConfig()
            _specontext_config._server_enabled = True
            _info("specontext", f"speculative context retrieval  (topk={_specontext_config.retrieval_topk})")
        except Exception as _e:
            _warn(f"[specontext] Skipped: {_e}")

    if getattr(args, "forelen", False):
        try:
            from squish.forelen import ForelenConfig
            _forelen_config = ForelenConfig()
            _forelen_config._server_enabled = True
            _info("forelen", f"forward length prediction  (buckets={_forelen_config.n_length_buckets})")
        except Exception as _e:
            _warn(f"[forelen] Skipped: {_e}")

    if getattr(args, "ipw", False):
        try:
            from squish.ipw import IPWConfig
            _ipw_config = IPWConfig()
            _ipw_config._server_enabled = True
            _info("ipw", f"importance-weighted prefill  (quality_weight={_ipw_config.quality_weight})")
        except Exception as _e:
            _warn(f"[ipw] Skipped: {_e}")

    if getattr(args, "layer_skip", False):
        try:
            from squish.layer_skip import EarlyExitConfig
            _layer_skip_config = EarlyExitConfig()
            _layer_skip_config._server_enabled = True
            _info("layer-skip", f"early-exit adaptive decoding  (exit_layer={_layer_skip_config.exit_layer}  threshold={_layer_skip_config.confidence_threshold})")
        except Exception as _e:
            _warn(f"[layer-skip] Skipped: {_e}")

    # ── Wave 37: Wire Everything In ───────────────────────────────────────────
    # ChipDetector is always run at startup (no flag required).
    global _chip_profile
    try:
        from squish.hardware.chip_detector import ChipDetector as _ChipDetector
        _cd_inst = _ChipDetector()
        _chip_profile = _cd_inst.detect()
        _info("chip-detector",
              f"{_chip_profile.generation.name}"
              f"  bw={_chip_profile.memory_bandwidth_gbps:.1f} GB/s"
              f"  rec_chunk={_chip_profile.recommended_chunk_prefill}"
              f"  rec_kv_bits={_chip_profile.recommended_kv_bits}")
        # Auto-tune chunk_prefill_size when the user didn't explicitly pick a value.
        if _chunk_prefill_enabled and getattr(args, "chunk_prefill_size", 512) == 512:
            _chunk_prefill_size = _chip_profile.recommended_chunk_prefill
            _info("chip-detector", f"→ chunk_prefill_size auto-tuned to {_chunk_prefill_size}")
    except Exception as _cd_err:
        _info("chip-detector", f"detection unavailable ({_cd_err})")

    # ── Wave 79: Auto-detect optimal settings from hardware + model files ─────
    if not getattr(args, "no_optimize", False):
        try:
            from squish.runtime.auto_profile import ModelCapabilityDetector as _MCD
            from squish.hardware.chip_detector import ChipDetector as _ChipDetW79
            _ram_gb_w79 = _ChipDetW79.detect_ram_gb()
            _auto_profile_inst = _MCD().detect(
                model_dir      = getattr(args, "model_dir", "") or getattr(args, "mlx_model_dir", ""),
                compressed_dir = getattr(args, "compressed_dir", ""),
                chip_profile   = _chip_profile,
                ram_gb         = _ram_gb_w79,
            )
            _auto_profile_inst.apply_defaults(args)
            globals()["_auto_profile"] = _auto_profile_inst
        except Exception:  # noqa: BLE001
            pass  # never block startup on auto-profile failure

    global _kvtc_manager
    if getattr(args, "kvtc", False) and _state.model is not None:
        try:
            from squish.kv.kvtc import KVTCConfig, KVTCManager
            _kvtc_cfg = KVTCConfig(
                rank=getattr(args, "kvtc_rank", 64),
                quant_bits=getattr(args, "kvtc_bits", 8),
            )
            _n_layers_kvtc = (
                getattr(_state.model, "n_layers", None)
                or len(getattr(_state.model, "layers", []))
                or 32
            )
            _kvtc_manager = KVTCManager(_kvtc_cfg, n_layers=_n_layers_kvtc)
            _kvtc_manager._server_enabled = True
            _info("kvtc",
                  f"rank={_kvtc_cfg.rank}  bits={_kvtc_cfg.quant_bits}"
                  f"  layers={_n_layers_kvtc}")
        except Exception as _e:
            _warn(f"[kvtc] Skipped: {_e}")

    global _chunk_kv_manager
    if getattr(args, "chunk_kv", False):
        try:
            from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
            _ckv_cfg = ChunkKVConfig(
                chunk_size=getattr(args, "chunk_kv_size", 16),
                budget_ratio=getattr(args, "chunk_kv_budget", 0.5),
            )
            _chunk_kv_manager = ChunkKVManager(_ckv_cfg)
            _info("chunk-kv",
                  f"chunk_size={_ckv_cfg.chunk_size}  budget={_ckv_cfg.budget_ratio}")
        except Exception as _e:
            _warn(f"[chunk-kv] Skipped: {_e}")

    global _ssd_saguaro
    if getattr(args, "ssd_saguaro", False):
        try:
            from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro
            _ssd_cfg = SSDConfig(
                k_outcomes=4,
                draft_len=8,
                acceptance_threshold=0.3,
            )
            _ssd_saguaro = SSDSaguaro(_ssd_cfg)
            _ssd_saguaro._server_enabled = True
            _info("ssd-saguaro",
                  f"k_outcomes={_ssd_cfg.k_outcomes}  draft_len={_ssd_cfg.draft_len}"
                  f"  threshold={_ssd_cfg.acceptance_threshold}")
        except Exception as _e:
            _warn(f"[ssd-saguaro] Skipped: {_e}")

    global _speculative_streamer
    if getattr(args, "spec_stream", False):
        try:
            from squish.speculative.spec_stream import SpecStreamConfig, SpeculativeStreamer
            _ss_cfg = SpecStreamConfig(buffer_size=16, rollback_on_reject=True)
            _speculative_streamer = SpeculativeStreamer(_ss_cfg)
            _info("spec-stream",
                  f"buffer_size={_ss_cfg.buffer_size}"
                  f"  rollback_on_reject={_ss_cfg.rollback_on_reject}")
        except Exception as _e:
            _warn(f"[spec-stream] Skipped: {_e}")

    global _metal_flash_attn
    if getattr(args, "metal_flash_attn", False) and _state.model is not None:
        try:
            from squish.kernels.metal_flash_attn import MetalFlashAttention, MetalFlashConfig
            _mfa_cfg = MetalFlashConfig(causal=True)
            _metal_flash_attn = MetalFlashAttention(_mfa_cfg)
            _metal_flash_attn._server_enabled = True
            _info("metal-flash-attn",
                  f"block_q={_mfa_cfg.block_q}  block_k={_mfa_cfg.block_k}"
                  f"  causal={_mfa_cfg.causal}")
        except Exception as _e:
            _warn(f"[metal-flash-attn] Skipped: {_e}")

    global _deja_vu_sparse_ffn
    if getattr(args, "deja_vu", False) and _state.model is not None:
        try:
            from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN
            import numpy as _dv_np
            # Use default dimension caps safe for all model sizes.
            _dv_cfg = DejaVuConfig(hidden_size=512, ffn_size=2048)
            _deja_vu_sparse_ffn = DejaVuSparseFFN(_dv_cfg)
            _deja_vu_sparse_ffn._server_enabled = True
            _info("deja-vu",
                  f"hidden={_dv_cfg.hidden_size}  ffn={_dv_cfg.ffn_size}"
                  f"  threshold={_dv_cfg.threshold}")
        except Exception as _e:
            _warn(f"[deja-vu] Skipped: {_e}")

    global _jacobi_decoder
    if getattr(args, "jacobi", False):
        try:
            from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder
            _jd_cfg = JacobiConfig(
                n_tokens=getattr(args, "jacobi_n", 4),
                max_iter=8,
                variant=getattr(args, "jacobi_variant", "jacobi"),
                temperature=0.0,
            )
            _jacobi_decoder = JacobiDecoder(_jd_cfg)
            _info("jacobi",
                  f"n_tokens={_jd_cfg.n_tokens}  max_iter={_jd_cfg.max_iter}"
                  f"  variant={_jd_cfg.variant}")
        except Exception as _e:
            _warn(f"[jacobi] Skipped: {_e}")

    global _mtp_predictor
    if getattr(args, "mtp", False):
        try:
            from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor
            _mtp_cfg = MTPHeadConfig(n_heads=getattr(args, "mtp_heads", 4))
            _mtp_predictor = MultiTokenPredictor(_mtp_cfg)
            _mtp_predictor._server_enabled = True
            _info("mtp",
                  f"n_heads={_mtp_cfg.n_heads}  vocab={_mtp_cfg.vocab_size}"
                  f"  emb_dim={_mtp_cfg.emb_dim}")
        except Exception as _e:
            _warn(f"[mtp] Skipped: {_e}")

    global _layer_overlap_loader
    if getattr(args, "layer_overlap", False) and _state.model is not None:
        try:
            from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader
            _lol_cfg = LayerOverlapConfig(
                prefetch_count=getattr(args, "layer_overlap_prefetch", 2),
            )
            _n_layers_lol = (
                getattr(_state.model, "n_layers", None)
                or len(getattr(_state.model, "layers", []))
                or 32
            )
            _layer_overlap_loader = LayerOverlapLoader(_lol_cfg)
            # Lightweight stub load_fn — actual Metal weight dispatch is via mlx;
            # this wires the infrastructure and stat tracking.
            _layer_overlap_loader.start(
                _n_layers_lol,
                lambda idx: {"layer_idx": idx},
            )
            _info("layer-overlap",
                  f"prefetch_count={_lol_cfg.prefetch_count}  n_layers={_n_layers_lol}")
        except Exception as _e:
            _warn(f"[layer-overlap] Skipped: {_e}")

    global _fused_qkv_proj
    if getattr(args, "fused_qkv", False) and _state.model is not None:
        try:
            from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection
            _qkv_model_args = (
                getattr(_state.model, "args", None)
                or getattr(_state.model, "config", None)
            )
            _fqkv_cfg = FusedQKVConfig(
                d_model=getattr(_qkv_model_args, "hidden_size", 4096) if _qkv_model_args else 4096,
                n_heads=getattr(_qkv_model_args, "num_attention_heads", 32) if _qkv_model_args else 32,
                n_kv_heads=getattr(_qkv_model_args, "num_key_value_heads", 8) if _qkv_model_args else 8,
                d_head=getattr(_qkv_model_args, "head_dim", 128) if _qkv_model_args else 128,
            )
            _fused_qkv_proj = FusedQKVProjection(_fqkv_cfg)
            _fused_qkv_proj._server_enabled = True
            _info("fused-qkv",
                  f"d_model={_fqkv_cfg.d_model}  n_heads={_fqkv_cfg.n_heads}"
                  f"  n_kv_heads={_fqkv_cfg.n_kv_heads}  d_head={_fqkv_cfg.d_head}")
        except Exception as _e:
            _warn(f"[fused-qkv] Skipped: {_e}")

    global _pd_disaggregator
    if getattr(args, "pd_disagg", False):
        try:
            from squish.serving.pd_disagg import PDConfig, PDDisaggregator
            _pd_cfg = PDConfig(
                max_prefill_tokens=8192,
                max_decode_tokens=512,
            )
            _pd_disaggregator = PDDisaggregator(config=_pd_cfg)
            _info("pd-disagg",
                  f"max_prefill={_pd_cfg.max_prefill_tokens}"
                  f"  max_decode={_pd_cfg.max_decode_tokens}")
        except Exception as _e:
            _warn(f"[pd-disagg] Skipped: {_e}")

    # ── Wave 41: Prefix Sharing, EAGLE-2, Ring Attention, Token Pruning ───────
    global _radix_attn_cache
    if getattr(args, "radix_attn", False):
        try:
            from squish.kv.radix_attn import RadixAttentionConfig, RadixAttentionCache
            _ra_cfg = RadixAttentionConfig()
            _radix_attn_cache = RadixAttentionCache(_ra_cfg)
            _info("radix-attn", f"prefix-sharing radix KV tree  (max_nodes={_ra_cfg.max_nodes})")
        except Exception as _e:
            _warn(f"[radix-attn] Skipped: {_e}")

    global _eagle2_spec
    if getattr(args, "eagle2", False):
        try:
            from squish.speculative.eagle2_spec import EAGLE2Config, EAGLE2Spec
            _e2_cfg = EAGLE2Config()
            _eagle2_spec = EAGLE2Spec(_e2_cfg)
            _info("eagle2", f"EAGLE-2 speculative decoding  (gamma={_e2_cfg.gamma}  tree_depth={_e2_cfg.tree_depth})")
        except Exception as _e:
            _warn(f"[eagle2] Skipped: {_e}")

    global _ring_attn_config
    if getattr(args, "ring_attn", False):
        try:
            from squish.attention.ring_attn import RingAttentionConfig, RingAttention
            _ring_cfg = RingAttentionConfig()
            _ring_attn_config = RingAttention(_ring_cfg)
            _info("ring-attn", f"ring-reduce sequence-parallel attention  (n_devices={_ring_cfg.n_devices}  chunk={_ring_cfg.chunk_size})")
        except Exception as _e:
            _warn(f"[ring-attn] Skipped: {_e}")

    global _token_entropy_pruner
    if getattr(args, "token_entropy_prune", False):
        try:
            from squish.token.token_entropy_prune import TokenEntropyConfig, TokenEntropyPruner
            _tep_cfg = TokenEntropyConfig()
            _token_entropy_pruner = TokenEntropyPruner(_tep_cfg)
            _info("token-entropy-prune", f"entropy-based KV cache pruning  (keep_ratio={_tep_cfg.keep_ratio})")
        except Exception as _e:
            _warn(f"[token-entropy-prune] Skipped: {_e}")

    global _pregated_moe_router
    if getattr(args, "pregated_moe", False):
        try:
            from squish.moe.pregated_router import PreGatedMoEConfig, PreGatedMoERouter
            _pgm_cfg = PreGatedMoEConfig()
            _pregated_moe_router = PreGatedMoERouter(_pgm_cfg)
            _info("pregated-moe", f"pre-gated MoE expert routing  (n_experts={_pgm_cfg.n_experts}  top_k={_pgm_cfg.top_k})")
        except Exception as _e:
            _warn(f"[pregated-moe] Skipped: {_e}")

    global _sink_fusion_config
    if getattr(args, "sink_fusion", False):
        try:
            from squish.kv.sink_fusion import SinkFusionConfig, SinkFusion
            _sf_cfg = SinkFusionConfig()
            _sink_fusion_config = SinkFusion(_sf_cfg)
            _info("sink-fusion", f"attention sink token merging  (n_sinks={_sf_cfg.n_sinks})")
        except Exception as _e:
            _warn(f"[sink-fusion] Skipped: {_e}")

    global _cla_share_config
    if getattr(args, "cla_share", False):
        try:
            from squish.attention.cla_share import CLAShareConfig, CLAShareAttention
            _cla_cfg = CLAShareConfig()
            _cla_share_config = CLAShareAttention(_cla_cfg)
            _info("cla-share", f"cross-layer KV sharing  (share_every={_cla_cfg.share_every})")
        except Exception as _e:
            _warn(f"[cla-share] Skipped: {_e}")

    global _qmoe_compressor
    if getattr(args, "qmoe_compress", False):
        try:
            from squish.moe.qmoe_compress import QMoEConfig, QMoECompressor
            _qmoe_cfg = QMoEConfig()
            _qmoe_compressor = QMoECompressor(_qmoe_cfg)
            _info("qmoe-compress", f"quantized MoE experts  (bits={_qmoe_cfg.bits})")
        except Exception as _e:
            _warn(f"[qmoe-compress] Skipped: {_e}")

    global _lade_decoder
    if getattr(args, "lade", False):
        try:
            from squish.speculative.lade_decode import LADEConfig, LADEDecoder
            _lade_cfg = LADEConfig()
            _lade_decoder = LADEDecoder(_lade_cfg)
            _info("lade", f"lookahead speculative decoding  (window={_lade_cfg.window_size}  ngram={_lade_cfg.ngram_size})")
        except Exception as _e:
            _warn(f"[lade] Skipped: {_e}")

    global _infini_attn_config
    if getattr(args, "infini_attn", False):
        try:
            from squish.attention.infini_attn import InfiniAttentionConfig, InfiniAttention
            _ia_cfg = InfiniAttentionConfig()
            _infini_attn_config = InfiniAttention(_ia_cfg)
            _info("infini-attn", f"compressive memory attention  (segment={_ia_cfg.segment_size}  mem_dim={_ia_cfg.memory_dim})")
        except Exception as _e:
            _warn(f"[infini-attn] Skipped: {_e}")

    global _akvq_cache
    if getattr(args, "akvq", False):
        try:
            from squish.kv.akvq_cache import AKVQConfig, AKVQCache
            _akvq_cfg = AKVQConfig()
            _akvq_cache = AKVQCache(_akvq_cfg)
            _info("akvq", f"adaptive KV quantization  (min_bits={_akvq_cfg.min_bits}  max_bits={_akvq_cfg.max_bits})")
        except Exception as _e:
            _warn(f"[akvq] Skipped: {_e}")

    global _delta_zip_store
    if getattr(args, "delta_zip", False):
        try:
            from squish.quant.delta_zip import DeltaZipConfig, DeltaZipAdapter
            _dz_cfg = DeltaZipConfig()
            _delta_zip_store = DeltaZipAdapter(_dz_cfg)
            _info("delta-zip", f"delta weight quantization  (bits={_dz_cfg.bits})")
        except Exception as _e:
            _warn(f"[delta-zip] Skipped: {_e}")

    # ── Wave 42: Disaggregated Serving, NSA, Medusa, KV Quant, QAT ───────────
    global _medusa_heads
    if getattr(args, "medusa_heads", False):
        try:
            from squish.speculative.medusa_heads import MedusaConfig, MedusaHeads
            _mh_cfg = MedusaConfig()
            _medusa_heads = MedusaHeads(_mh_cfg)
            _info("medusa-heads", f"multi-head speculative decoding  (n_heads={_mh_cfg.n_heads}  tree_width={_mh_cfg.tree_width})")
        except Exception as _e:
            _warn(f"[medusa-heads] Skipped: {_e}")

    global _sarathi_scheduler
    if getattr(args, "sarathi", False):
        try:
            from squish.serving.sarathi_scheduler import SarathiConfig, SarathiScheduler
            _sa_cfg = SarathiConfig()
            _sarathi_scheduler = SarathiScheduler(_sa_cfg)
            _info("sarathi", f"chunked prefill scheduler  (chunk={_sa_cfg.chunk_size}  max_batch={_sa_cfg.max_batch_size})")
        except Exception as _e:
            _warn(f"[sarathi] Skipped: {_e}")

    global _nsa_attn_config
    if getattr(args, "nsa_attn", False):
        try:
            from squish.attention.nsa_attn import NSAConfig, NSAAttention
            _nsa_cfg = NSAConfig()
            _nsa_attn_config = NSAAttention(_nsa_cfg)
            _info("nsa-attn", f"native sparse attention  (block={_nsa_cfg.block_size}  window={_nsa_cfg.window_size}  n_selected={_nsa_cfg.n_selected_blocks})")
        except Exception as _e:
            _warn(f"[nsa-attn] Skipped: {_e}")

    global _flex_prefill_config
    if getattr(args, "flex_prefill", False):
        try:
            from squish.attention.flex_prefill import FlexPrefillConfig, FlexPrefill
            _fp_cfg = FlexPrefillConfig()
            _flex_prefill_config = FlexPrefill(_fp_cfg)
            _info("flex-prefill", f"adaptive sparse prefill  (min_keep={_fp_cfg.min_keep_ratio})")
        except Exception as _e:
            _warn(f"[flex-prefill] Skipped: {_e}")

    global _think_cache
    if getattr(args, "think_cache", False):
        try:
            from squish.kv.think_cache import ThinKConfig, ThinKCache
            _tk_cfg = ThinKConfig()
            _think_cache = ThinKCache(_tk_cfg)
            _info("think-cache", f"K-channel pruning  (keep_ratio={_tk_cfg.keep_ratio})")
        except Exception as _e:
            _warn(f"[think-cache] Skipped: {_e}")

    global _attention_store
    if getattr(args, "attention_store", False):
        try:
            from squish.kv.attention_store import AttentionStoreConfig, AttentionStore
            _as_cfg = AttentionStoreConfig()
            _attention_store = AttentionStore(_as_cfg)
            _info("attention-store", f"tiered KV persistence  (hot={_as_cfg.hot_capacity}  warm={_as_cfg.warm_capacity})")
        except Exception as _e:
            _warn(f"[attention-store] Skipped: {_e}")

    global _rest_decoder
    if getattr(args, "rest_decode", False):
        try:
            from squish.speculative.rest_decode import RESTConfig, RESTDecode
            _rd_cfg = RESTConfig()
            _rest_decoder = RESTDecode(_rd_cfg)
            _info("rest-decode", f"retrieval n-gram speculative  (n_gram={_rd_cfg.n_gram}  top_k={_rd_cfg.top_k_draft})")
        except Exception as _e:
            _warn(f"[rest-decode] Skipped: {_e}")

    global _star_attn_config
    if getattr(args, "star_attn", False):
        try:
            from squish.attention.star_attn import StarAttentionConfig, StarAttention
            _sta_cfg = StarAttentionConfig()
            _star_attn_config = StarAttention(_sta_cfg)
            _info("star-attn", f"star-topology local+anchor attention  (block={_sta_cfg.block_size})")
        except Exception as _e:
            _warn(f"[star-attn] Skipped: {_e}")

    global _splitwise_scheduler
    if getattr(args, "splitwise", False):
        try:
            from squish.serving.splitwise_scheduler import SplitwiseConfig, SplitwiseScheduler
            _sw_cfg = SplitwiseConfig()
            _splitwise_scheduler = SplitwiseScheduler(_sw_cfg)
            _info("splitwise", f"prefill/decode disaggregation  (prefill_workers={_sw_cfg.prefill_workers}  decode_workers={_sw_cfg.decode_workers})")
        except Exception as _e:
            _warn(f"[splitwise] Skipped: {_e}")

    global _kvquant_cache
    if getattr(args, "kvquant", False):
        try:
            from squish.kv.kvquant import KVQuantConfig, KVQuantCache
            _kvq_cfg = KVQuantConfig()
            _kvquant_cache = KVQuantCache(_kvq_cfg)
            _info("kvquant", f"calibrated low-bit KV quantization  (bits={_kvq_cfg.bits}  window={_kvq_cfg.calibration_window})")
        except Exception as _e:
            _warn(f"[kvquant] Skipped: {_e}")

    global _efficient_qat
    if getattr(args, "efficient_qat", False):
        try:
            from squish.quant.efficient_qat import EfficientQATConfig, EfficientQAT
            _eqat_cfg = EfficientQATConfig()
            _efficient_qat = EfficientQAT(_eqat_cfg)
            _info("efficient-qat", f"block-wise QAT  (bits={_eqat_cfg.bits}  block={_eqat_cfg.block_size})")
        except Exception as _e:
            _warn(f"[efficient-qat] Skipped: {_e}")

    global _cache_gen_codec
    if getattr(args, "cache_gen", False):
        try:
            from squish.kv.cache_gen import CacheGenConfig, CacheGenCodec
            _cg_cfg = CacheGenConfig()
            _cache_gen_codec = CacheGenCodec(_cg_cfg)
            _info("cache-gen", f"KV bitstream compression  (bits={_cg_cfg.bits}  block={_cg_cfg.block_size})")
        except Exception as _e:
            _warn(f"[cache-gen] Skipped: {_e}")

    # ── Wave 43: MTP Decode, Cascade KV, Head Pruner, Paged Attn, etc. ────────
    global _mtp_decode_w43
    if getattr(args, "mtp_decode", False):
        try:
            from squish.speculative.mtp_decode import MTPConfig, MTPDecode
            _mtp_cfg = MTPConfig()
            _mtp_decode_w43 = MTPDecode(_mtp_cfg)
            _info("mtp-decode", f"multi-token prediction heads  (n_heads={_mtp_cfg.n_heads})")
        except Exception as _e:
            _warn(f"[mtp-decode] Skipped: {_e}")

    global _cascade_kv
    if getattr(args, "cascade_kv", False):
        try:
            from squish.kv.cascade_kv import CascadeKVConfig, CascadeKV
            _ck_cfg = CascadeKVConfig()
            _cascade_kv = CascadeKV(_ck_cfg)
            _info("cascade-kv", "two-level shared-prefix KV cache")
        except Exception as _e:
            _warn(f"[cascade-kv] Skipped: {_e}")

    global _head_pruner
    if getattr(args, "head_prune", False):
        try:
            from squish.model.head_pruner import HeadPrunerConfig, HeadPruner
            _hp_cfg = HeadPrunerConfig()
            _head_pruner = HeadPruner(_hp_cfg)
            _info("head-prune", "structured attention head importance pruning")
        except Exception as _e:
            _warn(f"[head-prune] Skipped: {_e}")

    global _paged_attn_w43
    if getattr(args, "paged_attn_w43", False):
        try:
            from squish.kv.paged_attn import PagedAttnConfig, PagedAttention
            _pa_cfg = PagedAttnConfig()
            _paged_attn_w43 = PagedAttention(_pa_cfg)
            _info("paged-attn-w43", f"PagedAttention Wave-43 block manager  (block_size={_pa_cfg.block_size})")
        except Exception as _e:
            _warn(f"[paged-attn-w43] Skipped: {_e}")

    global _layer_collapse
    if getattr(args, "layer_collapse", False):
        try:
            from squish.model.layer_collapse import LayerCollapseConfig, LayerCollapse
            _lc_cfg = LayerCollapseConfig()
            _layer_collapse = LayerCollapse(_lc_cfg)
            _info("layer-collapse", "cosine-similarity layer skip scheduling")
        except Exception as _e:
            _warn(f"[layer-collapse] Skipped: {_e}")

    global _relay_attn
    if getattr(args, "relay_attn", False):
        try:
            from squish.attention.relay_attn import RelayAttnConfig, RelayAttention
            _ra_cfg = RelayAttnConfig()
            _relay_attn = RelayAttention(_ra_cfg)
            _info("relay-attn", "relay attention cross-layer KV sharing")
        except Exception as _e:
            _warn(f"[relay-attn] Skipped: {_e}")

    global _wkv_quant
    if getattr(args, "wkv_quant", False):
        try:
            from squish.kv.wkv_quant import WKVQuantConfig, WKVQuant
            _wkv_cfg = WKVQuantConfig()
            _wkv_quant = WKVQuant(_wkv_cfg)
            _info("wkv-quant", f"joint W4KV4 weight+KV quantization  (bits={_wkv_cfg.bits})")
        except Exception as _e:
            _warn(f"[wkv-quant] Skipped: {_e}")

    global _tokenized_kv
    if getattr(args, "tokenized_kv", False):
        try:
            from squish.kv.tokenized_kv import TokenizedKVConfig, TokenizedKVCache
            _tkv_cfg = TokenizedKVConfig()
            _tokenized_kv = TokenizedKVCache(_tkv_cfg)
            _info("tokenized-kv", "KV serialized via embedding lookup table")
        except Exception as _e:
            _warn(f"[tokenized-kv] Skipped: {_e}")

    global _cluster_evict_kv
    if getattr(args, "cluster_evict", False):
        try:
            from squish.kv.cluster_evict_kv import ClusterEvictKVConfig, ClusterEvictKV
            _ce_cfg = ClusterEvictKVConfig()
            _cluster_evict_kv = ClusterEvictKV(_ce_cfg)
            _info("cluster-evict", f"cluster-based KV eviction  (n_clusters={_ce_cfg.n_clusters})")
        except Exception as _e:
            _warn(f"[cluster-evict] Skipped: {_e}")

    global _s2_attn
    if getattr(args, "s2_attn", False):
        try:
            from squish.attention.s2_attn import S2AttnConfig, S2Attention
            _s2_cfg = S2AttnConfig()
            _s2_attn = S2Attention(_s2_cfg)
            _info("s2-attn", "sorted-structured sparse attention")
        except Exception as _e:
            _warn(f"[s2-attn] Skipped: {_e}")

    global _magic_pig_v2
    if getattr(args, "magic_pig_v2", False):
        try:
            from squish.kv.magic_pig_v2 import MagicPIGv2Config, MagicPIGv2
            _mpv2_cfg = MagicPIGv2Config()
            _magic_pig_v2 = MagicPIGv2(_mpv2_cfg)
            _info("magic-pig-v2", "LSH-sampled KV retrieval with adaptive probe budget")
        except Exception as _e:
            _warn(f"[magic-pig-v2] Skipped: {_e}")

    # ── Wave 44: Marlin GEMM, Spec Rejection, LoFTQ, Online Spec, etc. ────────
    global _marlin_gemm
    if getattr(args, "marlin_gemm", False):
        try:
            from squish.quant.marlin_gemm import MarlinGEMMConfig, MarlinGEMM
            _mg_cfg = MarlinGEMMConfig()
            _marlin_gemm = MarlinGEMM(_mg_cfg)
            _info("marlin-gemm", "INT4×FP16 tiled GEMM for post-training quantization")
        except Exception as _e:
            _warn(f"[marlin-gemm] Skipped: {_e}")

    global _spec_rejection
    if getattr(args, "spec_rejection", False):
        try:
            from squish.speculative.spec_rejection import SpecRejectionConfig, SpecRejection
            _sr_cfg = SpecRejectionConfig()
            _spec_rejection = SpecRejection(_sr_cfg)
            _info("spec-rejection", "parallel draft candidates with early pruning")
        except Exception as _e:
            _warn(f"[spec-rejection] Skipped: {_e}")

    global _loftq_config
    if getattr(args, "loftq", False):
        try:
            from squish.quant.loftq import LoFTQConfig, LoFTQ
            _lq_cfg = LoFTQConfig()
            _loftq_config = LoFTQ(_lq_cfg)
            _info("loftq", f"alternating LoRA+W4 quantization optimizer  (bits={_lq_cfg.bits})")
        except Exception as _e:
            _warn(f"[loftq] Skipped: {_e}")

    global _online_spec
    if getattr(args, "online_spec", False):
        try:
            from squish.speculative.online_spec import OnlineSpecConfig, OnlineSpec
            _os_cfg = OnlineSpecConfig()
            _online_spec = OnlineSpec(_os_cfg)
            _info("online-spec", "session-adaptive draft distribution speculative decoding")
        except Exception as _e:
            _warn(f"[online-spec] Skipped: {_e}")

    global _dynamic_spec_len
    if getattr(args, "dynamic_spec_len", False):
        try:
            from squish.speculative.dynamic_spec_len import DynamicSpecLenConfig, DynamicSpecLen
            _dsl_cfg = DynamicSpecLenConfig()
            _dynamic_spec_len = DynamicSpecLen(_dsl_cfg)
            _info("dynamic-spec-len", "adaptive speculation lookahead K per token")
        except Exception as _e:
            _warn(f"[dynamic-spec-len] Skipped: {_e}")

    global _big_little_llm
    if getattr(args, "big_little", False):
        try:
            from squish.speculative.big_little_llm import BigLittleLLMConfig, BigLittleLLM
            _bl_cfg = BigLittleLLMConfig()
            _big_little_llm = BigLittleLLM(_bl_cfg)
            _info("big-little", "Big-Little decoder: route easy tokens to small model")
        except Exception as _e:
            _warn(f"[big-little] Skipped: {_e}")

    global _multi_exit_spec
    if getattr(args, "multi_exit_spec", False):
        try:
            from squish.speculative.multi_exit_spec import MultiExitSpecConfig, MultiExitSpec
            _mes_cfg = MultiExitSpecConfig()
            _multi_exit_spec = MultiExitSpec(_mes_cfg)
            _info("multi-exit-spec", "multi-exit speculative decoding at early layer")
        except Exception as _e:
            _warn(f"[multi-exit-spec] Skipped: {_e}")

    global _pv_tuning
    if getattr(args, "pv_tuning", False):
        try:
            from squish.quant.pv_tuning import PVTuningConfig, PVTuning
            _pvt_cfg = PVTuningConfig()
            _pv_tuning = PVTuning(_pvt_cfg)
            _info("pv-tuning", "proximal-gradient quantized weight optimization W1-2")
        except Exception as _e:
            _warn(f"[pv-tuning] Skipped: {_e}")

    global _hadamard_quant
    if getattr(args, "hadamard_quant", False):
        try:
            from squish.quant.hadamard_quant import HadamardQuantConfig, HadamardQuant
            _hq_cfg = HadamardQuantConfig()
            _hadamard_quant = HadamardQuant(_hq_cfg)
            _info("hadamard-quant", "Hadamard rotation whitening before INT4 GEMM")
        except Exception as _e:
            _warn(f"[hadamard-quant] Skipped: {_e}")

    global _prefix_tree
    if getattr(args, "prefix_tree_decode", False):
        try:
            from squish.speculative.prefix_tree_decode import PrefixTreeConfig, PrefixTreeDecode
            _pt_cfg = PrefixTreeConfig()
            _prefix_tree = PrefixTreeDecode(_pt_cfg)
            _info("prefix-tree-decode", "parallel static prefix tree path decode")
        except Exception as _e:
            _warn(f"[prefix-tree-decode] Skipped: {_e}")

    global _spectr_ot
    if getattr(args, "spectr_ot", False):
        try:
            from squish.speculative.spectr_ot import SpecTrOTConfig, SpecTrOT
            _sot_cfg = SpecTrOTConfig()
            _spectr_ot = SpecTrOT(_sot_cfg)
            _info("spectr-ot", "SpecTr: optimal-transport draft-target coupling")
        except Exception as _e:
            _warn(f"[spectr-ot] Skipped: {_e}")

    global _ada_gptq
    if getattr(args, "ada_gptq", False):
        try:
            from squish.quant.ada_gptq import AdaGPTQConfig, AdaGPTQ
            _ag_cfg = AdaGPTQConfig()
            _ada_gptq = AdaGPTQ(_ag_cfg)
            _info("ada-gptq", "Ada-GPTQ: per-layer adaptive group size W4 PTQ")
        except Exception as _e:
            _warn(f"[ada-gptq] Skipped: {_e}")

    # ── Wave 45: FlexGen Offload, YaRN, SelfExtend, Orca, MxFP4, etc. ────────
    global _flexgen_offload
    if getattr(args, "flexgen_offload", False):
        try:
            from squish.serving.flexgen_offload import FlexGenOffloadConfig, FlexGenOffload
            _fgo_cfg = FlexGenOffloadConfig()
            _flexgen_offload = FlexGenOffload(_fgo_cfg)
            _info("flexgen-offload", "LP-optimal CPU/SSD weight paging (FlexGen)")
        except Exception as _e:
            _warn(f"[flexgen-offload] Skipped: {_e}")

    global _yarn_rope
    if getattr(args, "yarn_rope", False):
        try:
            from squish.attention.yarn_rope import YaRNRoPEConfig, YaRNRoPE
            _yr_cfg = YaRNRoPEConfig()
            _yarn_rope = YaRNRoPE(_yr_cfg)
            _info("yarn-rope", "YaRN RoPE: extended context via NTK-aware interpolation")
        except Exception as _e:
            _warn(f"[yarn-rope] Skipped: {_e}")

    global _self_extend
    if getattr(args, "self_extend", False):
        try:
            from squish.attention.self_extend import SelfExtendConfig, SelfExtend
            _se_cfg = SelfExtendConfig()
            _self_extend = SelfExtend(_se_cfg)
            _info("self-extend", "Self-Extend: grouped attention beyond training window")
        except Exception as _e:
            _warn(f"[self-extend] Skipped: {_e}")

    global _orca_scheduler
    if getattr(args, "orca_sched", False):
        try:
            from squish.serving.orca_scheduler import OrcaSchedulerConfig, OrcaScheduler
            _orca_cfg = OrcaSchedulerConfig()
            _orca_scheduler = OrcaScheduler(_orca_cfg)
            _info("orca-sched", "Orca iteration-level continuous batching scheduler")
        except Exception as _e:
            _warn(f"[orca-sched] Skipped: {_e}")

    global _mx_fp4_quant
    if getattr(args, "mx_fp4", False):
        try:
            from squish.quant.mx_fp4 import MxFP4Config, MxFP4
            _mfp4_cfg = MxFP4Config()
            _mx_fp4_quant = MxFP4(_mfp4_cfg)
            _info("mx-fp4", "MX FP4 microscaling quantization")
        except Exception as _e:
            _warn(f"[mx-fp4] Skipped: {_e}")

    global _fp8_act_quant
    if getattr(args, "fp8_act", False):
        try:
            from squish.quant.fp8_act_quant import FP8ActQuantConfig, FP8ActQuant
            _fp8a_cfg = FP8ActQuantConfig()
            _fp8_act_quant = FP8ActQuant(_fp8a_cfg)
            _info("fp8-act", "FP8 activation quantization W8A8-style inference")
        except Exception as _e:
            _warn(f"[fp8-act] Skipped: {_e}")

    global _clex_rope
    if getattr(args, "clex_rope", False):
        try:
            from squish.attention.clex_rope import CLeXRoPEConfig, CLeXRoPE
            _cr_cfg = CLeXRoPEConfig()
            _clex_rope = CLeXRoPE(_cr_cfg)
            _info("clex-rope", "CLeX RoPE: continuous length extrapolation")
        except Exception as _e:
            _warn(f"[clex-rope] Skipped: {_e}")

    global _powerinfer_offload
    if getattr(args, "powerinfer", False):
        try:
            from squish.serving.powerinfer_offload import PowerInferOffloadConfig, PowerInferOffload
            _pi_cfg = PowerInferOffloadConfig()
            _powerinfer_offload = PowerInferOffload(_pi_cfg)
            _info("powerinfer", "PowerInfer: neuron-activation-aware weight streaming")
        except Exception as _e:
            _warn(f"[powerinfer] Skipped: {_e}")

    global _grouped_rope
    if getattr(args, "grouped_rope", False):
        try:
            from squish.attention.grouped_rope import GroupedRoPEConfig, GroupedRoPE
            _gr_cfg = GroupedRoPEConfig()
            _grouped_rope = GroupedRoPE(_gr_cfg)
            _info("grouped-rope", "group-query-aware rotary position embeddings")
        except Exception as _e:
            _warn(f"[grouped-rope] Skipped: {_e}")

    global _tensor_parallel
    if getattr(args, "tensor_parallel", False):
        try:
            from squish.serving.tensor_parallel import TensorParallelConfig, TensorParallel
            _tp_cfg = TensorParallelConfig()
            _tensor_parallel = TensorParallel(_tp_cfg)
            _info("tensor-parallel", "tensor parallelism across compute partitions")
        except Exception as _e:
            _warn(f"[tensor-parallel] Skipped: {_e}")

    global _fused_bias_gelu
    if getattr(args, "fused_bias_gelu", False):
        try:
            from squish.kernels.fused_bias_gelu import FusedBiasGELUConfig, FusedBiasGELU
            _fbg_cfg = FusedBiasGELUConfig()
            _fused_bias_gelu = FusedBiasGELU(_fbg_cfg)
            _info("fused-bias-gelu", "fused bias+GELU kernel")
        except Exception as _e:
            _warn(f"[fused-bias-gelu] Skipped: {_e}")

    global _token_budget_sched
    if getattr(args, "token_budget_sched", False):
        try:
            from squish.serving.token_budget_scheduler import TokenBudgetConfig, TokenBudgetScheduler
            _tbs_cfg = TokenBudgetConfig()
            _token_budget_sched = TokenBudgetScheduler(_tbs_cfg)
            _info("token-budget-sched", "per-request KV-budget preemption scheduler")
        except Exception as _e:
            _warn(f"[token-budget-sched] Skipped: {_e}")

    # ── Wave 46: Model Surgery, Expert Choice, W4A8, MLA KV, MinP, etc. ──────
    global _slice_gpt
    if getattr(args, "slice_gpt", False):
        try:
            from squish.quant.slice_gpt import SliceGPTConfig, SliceGPTPruner
            _sg_cfg = SliceGPTConfig()
            _slice_gpt = SliceGPTPruner(_sg_cfg)
            _info("slice-gpt", "SliceGPT: PCA orthogonal column pruning")
        except Exception as _e:
            _warn(f"[slice-gpt] Skipped: {_e}")

    global _wanda_pruner
    if getattr(args, "wanda", False):
        try:
            from squish.quant.wanda_pruner import WandaConfig, WandaPruner
            _wp_cfg = WandaConfig()
            _wanda_pruner = WandaPruner(_wp_cfg)
            _info("wanda", "Wanda: weight magnitude × activation RMS pruning")
        except Exception as _e:
            _warn(f"[wanda] Skipped: {_e}")

    global _short_gpt
    if getattr(args, "short_gpt", False):
        try:
            from squish.quant.short_gpt import ShortGPTConfig, ShortGPTPruner
            _shg_cfg = ShortGPTConfig()
            _short_gpt = ShortGPTPruner(_shg_cfg)
            _info("short-gpt", "ShortGPT: block importance layer removal")
        except Exception as _e:
            _warn(f"[short-gpt] Skipped: {_e}")

    global _w4a8_runtime
    if getattr(args, "w4a8", False):
        try:
            from squish.quant.w4a8_quant import W4A8Config, W4A8QuantRuntime
            _w4a8_cfg = W4A8Config()
            _w4a8_runtime = W4A8QuantRuntime(_w4a8_cfg)
            _info("w4a8", "W4A8 hybrid-precision quantization runtime")
        except Exception as _e:
            _warn(f"[w4a8] Skipped: {_e}")

    global _expert_choice
    if getattr(args, "expert_choice", False):
        try:
            from squish.moe.expert_choice import ExpertChoiceConfig, ExpertChoiceRouter
            _ec_cfg = ExpertChoiceConfig()
            _expert_choice = ExpertChoiceRouter(_ec_cfg)
            _info("expert-choice", "Expert Choice MoE: experts select top-k tokens")
        except Exception as _e:
            _warn(f"[expert-choice] Skipped: {_e}")

    global _mla_kv_compress
    if getattr(args, "mla_kv", False):
        try:
            from squish.kv.mla_kv_compress import MLAKVConfig, MLAKVCompress
            _mla_cfg = MLAKVConfig()
            _mla_kv_compress = MLAKVCompress(_mla_cfg)
            _info("mla-kv", "MLA KV compression: low-rank joint K+V projection")
        except Exception as _e:
            _warn(f"[mla-kv] Skipped: {_e}")

    global _minp_sampler
    if getattr(args, "minp", False):
        try:
            from squish.sampling.minp_sampler import MinPConfig, MinPSampler
            _minp_cfg = MinPConfig()
            _minp_sampler = MinPSampler(_minp_cfg)
            _info("minp", f"Min-P sampler: dynamic probability floor  (min_p={_minp_cfg.min_p_factor})")
        except Exception as _e:
            _warn(f"[minp] Skipped: {_e}")

    global _contrastive_search
    if getattr(args, "contrastive_search", False):
        try:
            from squish.sampling.contrastive_search import ContrastiveSearchConfig, ContrastiveSearch
            _cs_cfg = ContrastiveSearchConfig()
            _contrastive_search = ContrastiveSearch(_cs_cfg)
            _info("contrastive-search", "contrastive search: balance probability and diversity")
        except Exception as _e:
            _warn(f"[contrastive-search] Skipped: {_e}")

    global _razor_attn
    if getattr(args, "razor_attn", False):
        try:
            from squish.attention.razor_attn import RazorAttentionConfig, RazorAttention
            _rzr_cfg = RazorAttentionConfig()
            _razor_attn = RazorAttention(_rzr_cfg)
            _info("razor-attn", "RazorAttention: retrieval-head classifier for 70% KV reduction")
        except Exception as _e:
            _warn(f"[razor-attn] Skipped: {_e}")

    global _cache_blend
    if getattr(args, "cache_blend", False):
        try:
            from squish.kv.cacheblend import CacheBlendConfig, CacheBlend
            _cb_cfg = CacheBlendConfig()
            _cache_blend = CacheBlend(_cb_cfg)
            _info("cache-blend", "CacheBlend: partial KV prefix reuse for RAG")
        except Exception as _e:
            _warn(f"[cache-blend] Skipped: {_e}")

    global _green_kv
    if getattr(args, "green_kv", False):
        try:
            from squish.kv.green_kv import GreenKVConfig, GreenKVEviction
            _gkv_cfg = GreenKVConfig()
            _green_kv = GreenKVEviction(_gkv_cfg)
            _info("green-kv", "GreenKV: two-stream importance KV eviction")
        except Exception as _e:
            _warn(f"[green-kv] Skipped: {_e}")

    global _preble_router
    if getattr(args, "preble", False):
        try:
            from squish.serving.preble_router import PrebeleConfig, PrebeleRouter
            _prb_cfg = PrebeleConfig()
            _preble_router = PrebeleRouter(_prb_cfg)
            _info("preble", "Preble router: prefix-cache-aware multi-instance routing")
        except Exception as _e:
            _warn(f"[preble] Skipped: {_e}")

    # ── Wave 47: Mamba2, HGRN2, Lookahead, InfMemory, vAttn, IA3, MoE-∞ ──────
    global _mamba2_ssm
    if getattr(args, "mamba2_ssm", False):
        try:
            from squish.attention.mamba2_ssm import Mamba2Config, Mamba2SSM
            _m2_cfg = Mamba2Config()
            _mamba2_ssm = Mamba2SSM(_m2_cfg)
            _info("mamba2-ssm", f"Mamba2 SSM: Structured State-Space Duality  (d_state={_m2_cfg.d_state})")
        except Exception as _e:
            _warn(f"[mamba2-ssm] Skipped: {_e}")

    global _hgrn2
    if getattr(args, "hgrn2", False):
        try:
            from squish.attention.hgrn2 import HGRN2Config, HGRN2
            _hg_cfg = HGRN2Config()
            _hgrn2 = HGRN2(_hg_cfg)
            _info("hgrn2", "HGRN2: gated linear RNN with state expansion")
        except Exception as _e:
            _warn(f"[hgrn2] Skipped: {_e}")

    global _lookahead_decode
    if getattr(args, "lookahead_decode", False):
        try:
            from squish.speculative.lookahead_decode import LookaheadConfig, LookaheadDecode
            _ld_cfg = LookaheadConfig()
            _lookahead_decode = LookaheadDecode(_ld_cfg)
            _info("lookahead-decode", "Lookahead decode: 2D Jacobi window without draft model")
        except Exception as _e:
            _warn(f"[lookahead-decode] Skipped: {_e}")

    global _inf_memory
    if getattr(args, "inf_memory", False):
        try:
            from squish.kv.inf_memory import InfMemoryConfig, InfMemory
            _im_cfg = InfMemoryConfig()
            _inf_memory = InfMemory(_im_cfg)
            _info("inf-memory", "InfLLM: block-level external KV memory for 1M+ context")
        except Exception as _e:
            _warn(f"[inf-memory] Skipped: {_e}")

    global _v_attn_kv
    if getattr(args, "v_attn", False):
        try:
            from squish.kv.v_attention import vAttentionConfig, vAttentionKV
            _va_cfg = vAttentionConfig()
            _v_attn_kv = vAttentionKV(_va_cfg)
            _info("v-attn", "vAttention: OS virtual memory KV cache management")
        except Exception as _e:
            _warn(f"[v-attn] Skipped: {_e}")

    global _ia3_adapter
    if getattr(args, "ia3", False):
        try:
            from squish.lora.ia3_adapter import IA3Config, IA3Adapter
            _ia3_cfg = IA3Config()
            _ia3_adapter = IA3Adapter(_ia3_cfg)
            _info("ia3", "IA3 adapter: learned K/V/FF scale vectors")
        except Exception as _e:
            _warn(f"[ia3] Skipped: {_e}")

    global _moe_infinity
    if getattr(args, "moe_infinity", False):
        try:
            from squish.moe.moe_infinity import MoEInfinityConfig, MoEInfinityOffload
            _mi_cfg = MoEInfinityConfig()
            _moe_infinity = MoEInfinityOffload(_mi_cfg)
            _info("moe-infinity", "MoE-Infinity: activation-aware expert offloading")
        except Exception as _e:
            _warn(f"[moe-infinity] Skipped: {_e}")

    global _mega_blocks
    if getattr(args, "mega_blocks", False):
        try:
            from squish.moe.mega_blocks import MegaBlocksConfig, MegaBlocksSparse
            _mb_cfg = MegaBlocksConfig()
            _mega_blocks = MegaBlocksSparse(_mb_cfg)
            _info("mega-blocks", "MegaBlocks: dropless MoE with block-sparse GEMM")
        except Exception as _e:
            _warn(f"[mega-blocks] Skipped: {_e}")

    global _kgw_watermark
    if getattr(args, "kgw_watermark", False):
        try:
            from squish.serving.kgw_watermark import KGWConfig, KGWWatermark
            _kgw_cfg = KGWConfig()
            _kgw_watermark = KGWWatermark(_kgw_cfg)
            _info("kgw-watermark", "KGW watermark: green/red list vocabulary watermarking")
        except Exception as _e:
            _warn(f"[kgw-watermark] Skipped: {_e}")

    global _typical_sampler
    if getattr(args, "typical_sampler", False):
        try:
            from squish.sampling.typical_sampler import TypicalConfig, TypicalSampler
            _typ_cfg = TypicalConfig()
            _typical_sampler = TypicalSampler(_typ_cfg)
            _info("typical-sampler", "typical sampling: sample from local typical set")
        except Exception as _e:
            _warn(f"[typical-sampler] Skipped: {_e}")

    global _dora_adapter
    if getattr(args, "dora", False):
        try:
            from squish.lora.dora import DoRAConfig, DoRAAdapter
            _dora_cfg = DoRAConfig()
            _dora_adapter = DoRAAdapter(_dora_cfg)
            _info("dora", "DoRA: magnitude-direction weight decomposition adapter")
        except Exception as _e:
            _warn(f"[dora] Skipped: {_e}")

    global _calm_exit
    if getattr(args, "calm_exit", False):
        try:
            from squish.token.calm_exit import CALMConfig, AdaptiveCALM
            _calm_cfg = CALMConfig()
            _calm_exit = AdaptiveCALM(_calm_cfg)
            _info("calm-exit", "CALM: per-token confidence-gated early exit")
        except Exception as _e:
            _warn(f"[calm-exit] Skipped: {_e}")

    # ── Wave 48: INT2/INT3 Extreme Quantization ────────────────────────────────
    global _spqr_quantizer
    if getattr(args, "spqr", False):
        try:
            from squish.quant.spqr import SpQRConfig, SpQRQuantizer
            _spqr_cfg = SpQRConfig()
            _spqr_quantizer = SpQRQuantizer(_spqr_cfg)
            _info("spqr", "SpQR: sparse-quantized representation with outlier FP16")
        except Exception as _e:
            _warn(f"[spqr] Skipped: {_e}")

    global _auto_round
    if getattr(args, "auto_round", False):
        try:
            from squish.quant.auto_round import AutoRoundConfig, AutoRoundQuantizer
            _ar_cfg = AutoRoundConfig()
            _auto_round = AutoRoundQuantizer(_ar_cfg)
            _info("auto-round", "AutoRound: sign-gradient-descent rounding optimizer")
        except Exception as _e:
            _warn(f"[auto-round] Skipped: {_e}")

    global _owq_quantizer
    if getattr(args, "owq", False):
        try:
            from squish.quant.owq import OWQConfig, OWQQuantizer
            _owq_cfg = OWQConfig()
            _owq_quantizer = OWQQuantizer(_owq_cfg)
            _info("owq", "OWQ: outlier-aware weight quantization with column promotion")
        except Exception as _e:
            _warn(f"[owq] Skipped: {_e}")

    global _bit_distiller
    if getattr(args, "bit_distiller", False):
        try:
            from squish.quant.bit_distiller import BitDistillerConfig, BitDistillerQuant
            _bd_cfg = BitDistillerConfig()
            _bit_distiller = BitDistillerQuant(_bd_cfg)
            _info("bit-distiller", "BitDistiller: KL-divergence self-distillation for INT2")
        except Exception as _e:
            _warn(f"[bit-distiller] Skipped: {_e}")

    global _zip_lm
    if getattr(args, "zip_lm", False):
        try:
            from squish.quant.zip_lm import ZipLMConfig, ZipLMMixedPrecision
            _zl_cfg = ZipLMConfig()
            _zip_lm = ZipLMMixedPrecision(_zl_cfg)
            _info("zip-lm", "ZipLM: Hessian-sensitivity mixed-precision layer assignment")
        except Exception as _e:
            _warn(f"[zip-lm] Skipped: {_e}")

    global _gguf_mixed
    if getattr(args, "gguf_mixed", False):
        try:
            from squish.quant.gguf_mixed import GGUFConfig, GGUFMixedQuantizer
            _gm_cfg = GGUFConfig()
            _gguf_mixed = GGUFMixedQuantizer(_gm_cfg)
            _info("gguf-mixed", "GGUF mixed-precision Q2_K/Q3_K/Q4_K block quantization")
        except Exception as _e:
            _warn(f"[gguf-mixed] Skipped: {_e}")

    # ── Wave 49: TTFT Sprint: LLMLingua-2, RECOMP, Selective Context, etc. ────
    global _llm_lingua2
    if getattr(args, "llm_lingua2", False):
        try:
            from squish.serving.llm_lingua2 import LLMLingua2Config, LLMLingua2Compressor
            _ll2_cfg = LLMLingua2Config()
            _llm_lingua2 = LLMLingua2Compressor(_ll2_cfg)
            _info("llm-lingua2", "LLMLingua-2: token-level binary classifier prompt compression")
        except Exception as _e:
            _warn(f"[llm-lingua2] Skipped: {_e}")

    global _recomp_compressor
    if getattr(args, "recomp", False):
        try:
            from squish.serving.recomp import RECOMPConfig, RECOMPCompressor
            _rc_cfg = RECOMPConfig()
            _recomp_compressor = RECOMPCompressor(_rc_cfg)
            _info("recomp", "RECOMP: extractive+abstractive RAG context compression")
        except Exception as _e:
            _warn(f"[recomp] Skipped: {_e}")

    global _selective_context
    if getattr(args, "selective_context", False):
        try:
            from squish.serving.selective_context import SelectiveContextConfig, SelectiveContextCompressor
            _sc_cfg = SelectiveContextConfig()
            _selective_context = SelectiveContextCompressor(_sc_cfg)
            _info("selective-context", "selective context: self-information pruning")
        except Exception as _e:
            _warn(f"[selective-context] Skipped: {_e}")

    global _prompt_cache_kv
    if getattr(args, "prompt_cache", False):
        try:
            from squish.serving.prompt_cache import PromptCacheConfig, PromptCacheKV
            _pc_cfg = PromptCacheConfig()
            _prompt_cache_kv = PromptCacheKV(_pc_cfg)
            _info("prompt-cache", "PromptCache: schema-defined KV materialization for templates")
        except Exception as _e:
            _warn(f"[prompt-cache] Skipped: {_e}")

    global _pipe_infer
    if getattr(args, "pipe_infer", False):
        try:
            from squish.serving.pipe_infer import PipeInferConfig, PipeInferScheduler
            _pi2_cfg = PipeInferConfig()
            _pipe_infer = PipeInferScheduler(_pi2_cfg)
            _info("pipe-infer", "PipeInfer: chunked prefill+decode pipeline overlap")
        except Exception as _e:
            _warn(f"[pipe-infer] Skipped: {_e}")

    global _prepack_scheduler
    if getattr(args, "prepack", False):
        try:
            from squish.serving.prepack import PrepackConfig, PrepackScheduler
            _pp_cfg = PrepackConfig()
            _prepack_scheduler = PrepackScheduler(_pp_cfg)
            _info("prepack", "Prepack: completion-order batching for TTFT reduction")
        except Exception as _e:
            _warn(f"[prepack] Skipped: {_e}")

    # ── Wave 50: Bigger-Than-Memory: SparseGPT, MoD, LeanKV, GGUF, etc. ──────
    global _sparse_gpt
    if getattr(args, "sparse_gpt", False):
        try:
            from squish.model.sparse_gpt import SparseGPTConfig, SparseGPTPruner
            _sgpt_cfg = SparseGPTConfig()
            _sparse_gpt = SparseGPTPruner(_sgpt_cfg)
            _info("sparse-gpt", f"SparseGPT: one-shot Hessian weight pruning  (sparsity={_sgpt_cfg.sparsity_ratio})")
        except Exception as _e:
            _warn(f"[sparse-gpt] Skipped: {_e}")

    global _mix_of_depths
    if getattr(args, "mix_of_depths", False):
        try:
            from squish.model.mix_of_depths import MixtureOfDepthsConfig, MixtureOfDepths
            _mod_cfg = MixtureOfDepthsConfig()
            _mix_of_depths = MixtureOfDepths(_mod_cfg)
            _info("mix-of-depths", "Mixture-of-Depths: per-token layer routing")
        except Exception as _e:
            _warn(f"[mix-of-depths] Skipped: {_e}")

    global _lean_kv_quant
    if getattr(args, "lean_kv", False):
        try:
            from squish.kv.lean_kv import LeanKVConfig, LeanKVQuant
            _lkv_cfg = LeanKVConfig()
            _lean_kv_quant = LeanKVQuant(_lkv_cfg)
            _info("lean-kv", "LeanKV: asymmetric K(INT4)/V(INT8) cache quantization")
        except Exception as _e:
            _warn(f"[lean-kv] Skipped: {_e}")

    global _gguf_loader
    if getattr(args, "gguf_loader", False):
        try:
            from squish.io.gguf_loader import GGUFConfig, GGUFNativeLoader
            _gl_cfg = GGUFConfig()
            _gguf_loader = GGUFNativeLoader(_gl_cfg)
            _info("gguf-loader", "GGUF native loader: Q2_K/Q3_K/Q4_K/Q5_K/Q8_0 format parser")
        except Exception as _e:
            _warn(f"[gguf-loader] Skipped: {_e}")

    global _weight_stream
    if getattr(args, "weight_stream", False):
        try:
            from squish.io.weight_decompress_stream import WeightStreamConfig, WeightDecompressStream
            _ws_cfg = WeightStreamConfig()
            _weight_stream = WeightDecompressStream(_ws_cfg)
            _info("weight-stream", "weight decompress stream: overlapped CPU dequant + GPU compute")
        except Exception as _e:
            _warn(f"[weight-stream] Skipped: {_e}")

    global _shard_loader
    if getattr(args, "shard_loader", False):
        try:
            from squish.io.model_shard_loader import ShardConfig, ModelShardLoader
            _sl_cfg = ShardConfig()
            _shard_loader = ModelShardLoader(_sl_cfg)
            _info("shard-loader", "model shard loader: 3-tier GPU-hot/CPU-warm/SSD-cold weight paging")
        except Exception as _e:
            _warn(f"[shard-loader] Skipped: {_e}")

    # ── Wave 51: Test-Time Compute Scaling ────────────────────────────────────
    global _budget_forcing
    if getattr(args, "budget_forcing", False):
        try:
            from squish.serving.budget_forcing import BudgetForcingConfig, BudgetForcingDecoder
            _bf_cfg = BudgetForcingConfig()
            _budget_forcing = BudgetForcingDecoder(_bf_cfg)
            _info("budget-forcing", f"budget forcing: s1-style thinking token budget  (max={_bf_cfg.max_thinking_tokens})")
        except Exception as _e:
            _warn(f"[budget-forcing] Skipped: {_e}")

    global _test_time_router
    if getattr(args, "test_time_scale", False):
        try:
            from squish.sampling.test_time_scale import TestTimeComputeConfig, TestTimeComputeRouter
            _tts_cfg = TestTimeComputeConfig()
            _test_time_router = TestTimeComputeRouter(_tts_cfg)
            _info("test-time-scale", "test-time compute router: difficulty-aware strategy dispatch")
        except Exception as _e:
            _warn(f"[test-time-scale] Skipped: {_e}")

    global _dvts_search
    if getattr(args, "dvts", False):
        try:
            from squish.sampling.dvts_search import DVTSConfig, DVTSSearch
            _dvts_cfg = DVTSConfig()
            _dvts_search = DVTSSearch(_dvts_cfg)
            _info("dvts", "DVTS: diverse verifier tree search for reasoning")
        except Exception as _e:
            _warn(f"[dvts] Skipped: {_e}")

    global _chain_of_draft
    if getattr(args, "chain_of_draft", False):
        try:
            from squish.sampling.chain_of_draft import ChainOfDraftConfig, ChainOfDraftSampler
            _cod_cfg = ChainOfDraftConfig()
            _chain_of_draft = ChainOfDraftSampler(_cod_cfg)
            _info("chain-of-draft", "Chain-of-Draft: ≤7-word per-step reasoning constraint")
        except Exception as _e:
            _warn(f"[chain-of-draft] Skipped: {_e}")

    global _coconut_decoder
    if getattr(args, "coconut", False):
        try:
            from squish.reasoning.coconut import CoconutConfig, CoconutDecoder
            _coc_cfg = CoconutConfig()
            _coconut_decoder = CoconutDecoder(_coc_cfg)
            _info("coconut", "COCONUT: continuous latent reasoning decoder")
        except Exception as _e:
            _warn(f"[coconut] Skipped: {_e}")

    global _prm_beam_search
    if getattr(args, "prm_beam", False):
        try:
            from squish.sampling.prm_beam_search import PRMBeamSearchConfig, PRMBeamSearch
            _prm_cfg = PRMBeamSearchConfig()
            _prm_beam_search = PRMBeamSearch(_prm_cfg)
            _info("prm-beam", "PRM beam search: step-level process reward model guidance")
        except Exception as _e:
            _warn(f"[prm-beam] Skipped: {_e}")

    global _best_of_n
    if getattr(args, "best_of_n", False):
        try:
            from squish.sampling.best_of_n import BestOfNConfig, BestOfNSampler
            _bon_cfg = BestOfNConfig()
            _best_of_n = BestOfNSampler(_bon_cfg)
            _info("best-of-n", "Best-of-N sampling with reward model scoring")
        except Exception as _e:
            _warn(f"[best-of-n] Skipped: {_e}")

    global _self_consistency
    if getattr(args, "self_consistency", False):
        try:
            from squish.reasoning.self_consistency import SelfConsistencyConfig, SelfConsistencyVoter
            _sc2_cfg = SelfConsistencyConfig()
            _self_consistency = SelfConsistencyVoter(_sc2_cfg)
            _info("self-consistency", "self-consistency: majority voting over K reasoning chains")
        except Exception as _e:
            _warn(f"[self-consistency] Skipped: {_e}")

    global _thought_budget_gate
    if getattr(args, "thought_budget", False):
        try:
            from squish.token.thought_budget_gate import ThoughtBudgetConfig, ThoughtBudgetGate
            _tbg_cfg = ThoughtBudgetConfig()
            _thought_budget_gate = ThoughtBudgetGate(_tbg_cfg)
            _info("thought-budget", "thought budget gate: per-segment CoT token limiting")
        except Exception as _e:
            _warn(f"[thought-budget] Skipped: {_e}")

    global _reasoning_kv
    if getattr(args, "reasoning_kv", False):
        try:
            from squish.kv.reasoning_kv import ReasoningKVConfig, ReasoningKVManager
            _rkv_cfg = ReasoningKVConfig()
            _reasoning_kv = ReasoningKVManager(_rkv_cfg)
            _info("reasoning-kv", "Reasoning KV: INT2 thinking-region + FP16 answer-region KV")
        except Exception as _e:
            _warn(f"[reasoning-kv] Skipped: {_e}")

    global _draft_reasoning
    if getattr(args, "draft_reasoning", False):
        try:
            from squish.speculative.draft_reasoning import DraftReasoningConfig, DraftReasoningVerifier
            _dr_cfg = DraftReasoningConfig()
            _draft_reasoning = DraftReasoningVerifier(_dr_cfg)
            _info("draft-reasoning", "draft reasoning verifier: CoT-consistency speculative acceptance")
        except Exception as _e:
            _warn(f"[draft-reasoning] Skipped: {_e}")

    global _parallel_reasoning
    if getattr(args, "parallel_reasoning", False):
        try:
            from squish.serving.parallel_reasoning import ParallelReasoningConfig, ParallelReasoningScheduler
            _pr_cfg = ParallelReasoningConfig()
            _parallel_reasoning = ParallelReasoningScheduler(_pr_cfg)
            _info("parallel-reasoning", "parallel reasoning scheduler: M chains per prompt")
        except Exception as _e:
            _warn(f"[parallel-reasoning] Skipped: {_e}")

    # ── Wave 52: Multi-Modal VLM Efficiency ───────────────────────────────────
    global _fast_v_pruner
    if getattr(args, "fast_v", False):
        try:
            from squish.vision.fast_v import FastVConfig, FastVPruner
            _fv_cfg = FastVConfig()
            _fast_v_pruner = FastVPruner(_fv_cfg)
            _info("fast-v", "FastV: visual token pruning by cross-attention score")
        except Exception as _e:
            _warn(f"[fast-v] Skipped: {_e}")

    global _vision_zip
    if getattr(args, "vision_zip", False):
        try:
            from squish.vision.vision_zip import VisionZipConfig, VisionZip
            _vz_cfg = VisionZipConfig()
            _vision_zip = VisionZip(_vz_cfg)
            _info("vision-zip", "VisionZip: context-dependent visual token selection")
        except Exception as _e:
            _warn(f"[vision-zip] Skipped: {_e}")

    global _llava_prumerge
    if getattr(args, "llava_prumerge", False):
        try:
            from squish.vision.llava_prumerge import LLaVAPruMergeConfig, LLaVAPruMerge
            _lpm_cfg = LLaVAPruMergeConfig()
            _llava_prumerge = LLaVAPruMerge(_lpm_cfg)
            _info("llava-prumerge", "LLaVA-PruMerge: spatial clustering+merge of visual patches")
        except Exception as _e:
            _warn(f"[llava-prumerge] Skipped: {_e}")

    global _token_packer
    if getattr(args, "token_packer", False):
        try:
            from squish.vision.token_packer import TokenPackerConfig, TokenPacker
            _tp2_cfg = TokenPackerConfig()
            _token_packer = TokenPacker(_tp2_cfg)
            _info("token-packer", "TokenPacker: fixed-size cross-attention visual projector")
        except Exception as _e:
            _warn(f"[token-packer] Skipped: {_e}")

    global _flash_vstream
    if getattr(args, "flash_vstream", False):
        try:
            from squish.vision.flash_vstream import FlashVStreamConfig, FlashVStream
            _fvs_cfg = FlashVStreamConfig()
            _flash_vstream = FlashVStream(_fvs_cfg)
            _info("flash-vstream", "Flash-VStream: 3-tier video KV memory for streaming video")
        except Exception as _e:
            _warn(f"[flash-vstream] Skipped: {_e}")

    global _dynamic_res
    if getattr(args, "dynamic_res", False):
        try:
            from squish.vision.dynamic_resolution import DynamicResConfig, DynamicResEncoder
            _dr2_cfg = DynamicResConfig()
            _dynamic_res = DynamicResEncoder(_dr2_cfg)
            _info("dynamic-res", "dynamic resolution encoder: aspect-ratio tiling")
        except Exception as _e:
            _warn(f"[dynamic-res] Skipped: {_e}")

    global _visual_kv_quant
    if getattr(args, "visual_kv_quant", False):
        try:
            from squish.vision.visual_kv_quant import VisualKVQuantConfig, VisualKVQuant
            _vkq_cfg = VisualKVQuantConfig()
            _visual_kv_quant = VisualKVQuant(_vkq_cfg)
            _info("visual-kv-quant", "Visual KV quantization: INT4K+INT6V for visual token blocks")
        except Exception as _e:
            _warn(f"[visual-kv-quant] Skipped: {_e}")

    global _cross_modal_router
    if getattr(args, "cross_modal", False):
        try:
            from squish.vision.cross_modal_attn import CrossModalAttnConfig, CrossModalRouter
            _cm_cfg = CrossModalAttnConfig()
            _cross_modal_router = CrossModalRouter(_cm_cfg)
            _info("cross-modal", "cross-modal router: affinity-gated visual↔text attention")
        except Exception as _e:
            _warn(f"[cross-modal] Skipped: {_e}")

    global _video_kv_reuse
    if getattr(args, "video_kv_reuse", False):
        try:
            from squish.vision.video_kv_reuse import VideoKVReuseConfig, VideoKVReuse
            _vkr_cfg = VideoKVReuseConfig()
            _video_kv_reuse = VideoKVReuse(_vkr_cfg)
            _info("video-kv-reuse", "video KV reuse: frame-pair cosine similarity KV sharing")
        except Exception as _e:
            _warn(f"[video-kv-reuse] Skipped: {_e}")

    global _vlm_spec_decode
    if getattr(args, "vlm_spec", False):
        try:
            from squish.vision.vlm_spec_decode import VLMSpecDecodeConfig, VLMSpecDecode
            _vsd_cfg = VLMSpecDecodeConfig()
            _vlm_spec_decode = VLMSpecDecode(_vsd_cfg)
            _info("vlm-spec", "VLM speculative decoding with shared visual prefix")
        except Exception as _e:
            _warn(f"[vlm-spec] Skipped: {_e}")

    global _vlm_batch_sched
    if getattr(args, "vlm_sched", False):
        try:
            from squish.serving.vlm_scheduler import VLMBatchConfig, VLMBatchScheduler
            _vlms_cfg = VLMBatchConfig()
            _vlm_batch_sched = VLMBatchScheduler(_vlms_cfg)
            _info("vlm-sched", "VLM batch scheduler: image-complexity request binning")
        except Exception as _e:
            _warn(f"[vlm-sched] Skipped: {_e}")

    global _img_encoder_cache
    if getattr(args, "img_encoder_cache", False):
        try:
            from squish.vision.img_encoder_cache import ImageEncoderCacheConfig, ImageEncoderCache
            _iec_cfg = ImageEncoderCacheConfig()
            _img_encoder_cache = ImageEncoderCache(_iec_cfg)
            _info("img-encoder-cache", "image encoder output cache keyed by SHA-256")
        except Exception as _e:
            _warn(f"[img-encoder-cache] Skipped: {_e}")

    # ── Wave 53: Linear Recurrent Architectures ───────────────────────────────
    global _rwkv6_channel_mix
    if getattr(args, "rwkv6", False):
        try:
            from squish.attention.rwkv_channel_mix import RWKV6Config, RWKV6ChannelMix
            _rwkv_cfg = RWKV6Config()
            _rwkv6_channel_mix = RWKV6ChannelMix(_rwkv_cfg)
            _info("rwkv6", "RWKV-6 Eagle channel-mix layer")
        except Exception as _e:
            _warn(f"[rwkv6] Skipped: {_e}")

    global _hawk_rnn
    if getattr(args, "hawk_rnn", False):
        try:
            from squish.attention.hawk_recurrent import HawkConfig, HawkLinearRNN
            _hawk_cfg = HawkConfig()
            _hawk_rnn = HawkLinearRNN(_hawk_cfg)
            _info("hawk-rnn", "Hawk/Griffin Real-Gated Linear Recurrence")
        except Exception as _e:
            _warn(f"[hawk-rnn] Skipped: {_e}")

    global _xlstm_block
    if getattr(args, "xlstm", False):
        try:
            from squish.attention.xlstm_block import xLSTMConfig, xLSTMBlock
            _xl_cfg = xLSTMConfig()
            _xlstm_block = xLSTMBlock(_xl_cfg)
            _info("xlstm", "xLSTM: extended LSTM with sLSTM and mLSTM cells")
        except Exception as _e:
            _warn(f"[xlstm] Skipped: {_e}")

    global _ttt_layer
    if getattr(args, "ttt", False):
        try:
            from squish.attention.ttt_layer import TTTConfig, TTTLinearLayer
            _ttt_cfg = TTTConfig()
            _ttt_layer = TTTLinearLayer(_ttt_cfg)
            _info("ttt", "TTT layer: Test-Time Training linear layer")
        except Exception as _e:
            _warn(f"[ttt] Skipped: {_e}")

    global _delta_net
    if getattr(args, "delta_net", False):
        try:
            from squish.attention.delta_net import DeltaNetConfig, DeltaNetLinear
            _dn_cfg = DeltaNetConfig()
            _delta_net = DeltaNetLinear(_dn_cfg)
            _info("delta-net", "DeltaNet: delta-rule linear recurrent attention")
        except Exception as _e:
            _warn(f"[delta-net] Skipped: {_e}")

    global _ssm_state_cache
    if getattr(args, "ssm_cache", False):
        try:
            from squish.kv.ssm_state_cache import SSMStateCacheConfig, SSMStateCache
            _ssc_cfg = SSMStateCacheConfig()
            _ssm_state_cache = SSMStateCache(_ssc_cfg)
            _info("ssm-cache", "SSM state cache: unified recurrent state persistence")
        except Exception as _e:
            _warn(f"[ssm-cache] Skipped: {_e}")

    global _parallel_scan
    if getattr(args, "parallel_scan", False):
        try:
            from squish.kernels.parallel_scan_kernel import ParallelScanConfig, ParallelScanKernel
            _ps_cfg = ParallelScanConfig()
            _parallel_scan = ParallelScanKernel(_ps_cfg)
            _info("parallel-scan", "parallel scan kernel: Blelloch prefix scan for SSM prefill")
        except Exception as _e:
            _warn(f"[parallel-scan] Skipped: {_e}")

    global _ssm_quantizer
    if getattr(args, "ssm_quant", False):
        try:
            from squish.quant.ssm_quant import SSMQuantConfig, SSMQuantizer
            _ssq_cfg = SSMQuantConfig()
            _ssm_quantizer = SSMQuantizer(_ssq_cfg)
            _info("ssm-quant", "SSM quantizer: SSM-specific Δ/A/B/C calibration quantization")
        except Exception as _e:
            _warn(f"[ssm-quant] Skipped: {_e}")

    global _hybrid_arch_router
    if getattr(args, "hybrid_arch", False):
        try:
            from squish.serving.hybrid_arch_router import HybridArchConfig, HybridArchRouter
            _ha_cfg = HybridArchConfig()
            _hybrid_arch_router = HybridArchRouter(_ha_cfg)
            _info("hybrid-arch", "hybrid arch router: per-layer SSM vs attention dispatch")
        except Exception as _e:
            _warn(f"[hybrid-arch] Skipped: {_e}")

    global _hymba_dual
    if getattr(args, "hymba", False):
        try:
            from squish.attention.hymba_dual import HymbaConfig, HymbaDualTrack
            _hym_cfg = HymbaConfig()
            _hymba_dual = HymbaDualTrack(_hym_cfg)
            _info("hymba", "Hymba dual-track: parallel mini-SSM + attention heads")
        except Exception as _e:
            _warn(f"[hymba] Skipped: {_e}")

    global _ssm_state_offload
    if getattr(args, "ssm_offload", False):
        try:
            from squish.streaming.ssm_state_offload import SSMStateOffloadConfig, SSMStateOffload
            _sso_cfg = SSMStateOffloadConfig()
            _ssm_state_offload = SSMStateOffload(_sso_cfg)
            _info("ssm-offload", "SSM state offload: segment-boundary recurrent state export")
        except Exception as _e:
            _warn(f"[ssm-offload] Skipped: {_e}")

    # ── Wave 54: Deep MoE Efficiency, FlashAttn3, DoubleSparsity, etc. ────────
    global _shared_expert_moe
    if getattr(args, "shared_expert", False):
        try:
            from squish.moe.shared_expert import SharedExpertConfig, SharedExpertMoE
            _se2_cfg = SharedExpertConfig()
            _shared_expert_moe = SharedExpertMoE(_se2_cfg)
            _info("shared-expert", "shared expert MoE: always-active shared + routed experts")
        except Exception as _e:
            _warn(f"[shared-expert] Skipped: {_e}")

    global _fine_grained_router
    if getattr(args, "fine_grained_moe", False):
        try:
            from squish.moe.fine_grained_router import FineGrainedMoEConfig, FineGrainedMoERouter
            _fgr_cfg = FineGrainedMoEConfig()
            _fine_grained_router = FineGrainedMoERouter(_fgr_cfg)
            _info("fine-grained-moe", "fine-grained MoE router: DeepSeek-V3 auxiliary-loss-free")
        except Exception as _e:
            _warn(f"[fine-grained-moe] Skipped: {_e}")

    global _expert_offloader
    if getattr(args, "expert_offload", False):
        try:
            from squish.moe.expert_offload import ExpertOffloaderConfig, ExpertOffloader
            _eo_cfg = ExpertOffloaderConfig()
            _expert_offloader = ExpertOffloader(_eo_cfg)
            _info("expert-offload", "expert offloader: top-M LRU expert weight pager")
        except Exception as _e:
            _warn(f"[expert-offload] Skipped: {_e}")

    global _expert_merger
    if getattr(args, "expert_merge", False):
        try:
            from squish.moe.expert_merge import ExpertMergeConfig, ExpertMerger
            _em_cfg = ExpertMergeConfig()
            _expert_merger = ExpertMerger(_em_cfg)
            _info("expert-merge", "expert merger: cosine-similarity expert consolidation")
        except Exception as _e:
            _warn(f"[expert-merge] Skipped: {_e}")

    global _lazy_expert
    if getattr(args, "lazy_expert", False):
        try:
            from squish.moe.lazy_expert_load import LazyExpertConfig, LazyExpertLoader
            _le_cfg = LazyExpertConfig()
            _lazy_expert = LazyExpertLoader(_le_cfg)
            _info("lazy-expert", "lazy expert loader: JIT expert weight materialization")
        except Exception as _e:
            _warn(f"[lazy-expert] Skipped: {_e}")

    global _expert_act_cache
    if getattr(args, "expert_cache", False):
        try:
            from squish.moe.expert_cache import ExpertCacheConfig, ExpertActivationCache
            _eac_cfg = ExpertCacheConfig()
            _expert_act_cache = ExpertActivationCache(_eac_cfg)
            _info("expert-cache", "expert activation cache: LRU expert output caching")
        except Exception as _e:
            _warn(f"[expert-cache] Skipped: {_e}")

    global _flash_attn3
    if getattr(args, "flash_attn3", False):
        try:
            from squish.kernels.flash_attn3 import FlashAttn3Config, FlashAttn3Kernel
            _fa3_cfg = FlashAttn3Config()
            _flash_attn3 = FlashAttn3Kernel(_fa3_cfg)
            _info("flash-attn3", "FlashAttention-3: pingpong warp scheduling 1.5-2× FA-2")
        except Exception as _e:
            _warn(f"[flash-attn3] Skipped: {_e}")

    global _double_sparse_attn
    if getattr(args, "double_sparse", False):
        try:
            from squish.attention.double_sparse import DoubleSparsityConfig, DoubleSparsityAttn
            _ds_cfg = DoubleSparsityConfig()
            _double_sparse_attn = DoubleSparsityAttn(_ds_cfg)
            _info("double-sparse", "DoubleSparsity: simultaneous head-level and token-level sparse attn")
        except Exception as _e:
            _warn(f"[double-sparse] Skipped: {_e}")

    global _lasp_linear_attn
    if getattr(args, "lasp", False):
        try:
            from squish.attention.lasp_parallel import LASPConfig, LASPLinearAttn
            _lasp_cfg = LASPConfig()
            _lasp_linear_attn = LASPLinearAttn(_lasp_cfg)
            _info("lasp", "LASP: linear attention sequence parallelism via ring topology")
        except Exception as _e:
            _warn(f"[lasp] Skipped: {_e}")

    global _nacl_cache
    if getattr(args, "nacl_cache", False):
        try:
            from squish.kv.nacl_cache import NaCLConfig, NaCLCache
            _nacl_cfg = NaCLConfig()
            _nacl_cache = NaCLCache(_nacl_cfg)
            _info("nacl-cache", "NaCL cache: O(1) random KV eviction with non-evictable reserve")
        except Exception as _e:
            _warn(f"[nacl-cache] Skipped: {_e}")

    global _kv_migration
    if getattr(args, "kv_migration", False):
        try:
            from squish.serving.kv_migration import KVMigrationConfig, KVMigrationManager
            _kvm_cfg = KVMigrationConfig()
            _kv_migration = KVMigrationManager(_kvm_cfg)
            _info("kv-migration", "KV migration manager: live KV shard transfer between workers")
        except Exception as _e:
            _warn(f"[kv-migration] Skipped: {_e}")

    global _elastic_batch
    if getattr(args, "elastic_batch", False):
        try:
            from squish.serving.elastic_batching import ElasticBatchConfig, ElasticBatchController
            _eb_cfg = ElasticBatchConfig()
            _elastic_batch = ElasticBatchController(_eb_cfg)
            _info("elastic-batch", "elastic batch controller: adaptive continuous-batch sizing")
        except Exception as _e:
            _warn(f"[elastic-batch] Skipped: {_e}")

    # ── Wave 55: Advanced Sampling, Emerging Quantization ─────────────────────
    global _min_p_sampler
    if getattr(args, "min_p", False):
        try:
            from squish.sampling.min_p_sampler import MinPConfig, MinPSampler
            _mp_cfg = MinPConfig()
            _min_p_sampler = MinPSampler(_mp_cfg)
            _info("min-p", f"Min-P sampler (min_p_sampler): dynamic probability floor  (min_p={_mp_cfg.min_p_factor})")
        except Exception as _e:
            _warn(f"[min-p] Skipped: {_e}")

    global _mirostat_sampler
    if getattr(args, "mirostat", False):
        try:
            from squish.sampling.mirostat_sampler import MirostatConfig, MirostatSampler
            _miro_cfg = MirostatConfig()
            _mirostat_sampler = MirostatSampler(_miro_cfg)
            _info("mirostat", "Mirostat sampler: PID perplexity controller")
        except Exception as _e:
            _warn(f"[mirostat] Skipped: {_e}")

    global _eta_cutoff
    if getattr(args, "eta_cutoff", False):
        try:
            from squish.sampling.eta_sampler import EtaConfig, EtaSampler
            _eta_cfg = EtaConfig()
            _eta_cutoff = EtaSampler(_eta_cfg)
            _info("eta-cutoff", "eta-cutoff sampler: entropy-adaptive hard logit cutoff")
        except Exception as _e:
            _warn(f"[eta-cutoff] Skipped: {_e}")

    global _cfg_sampler
    if getattr(args, "cfg_guidance", False):
        try:
            from squish.sampling.cfg_sampler import CFGConfig, CFGLogitsSampler
            _cfg_s_cfg = CFGConfig()
            _cfg_sampler = CFGLogitsSampler(_cfg_s_cfg)
            _info("cfg-guidance", "CFG logits sampler: Classifier-Free Guidance for text")
        except Exception as _e:
            _warn(f"[cfg-guidance] Skipped: {_e}")

    global _diverse_beam
    if getattr(args, "diverse_beam", False):
        try:
            from squish.sampling.diverse_beam import DiverseBeamConfig, DiverseBeamSampler
            _db_cfg = DiverseBeamConfig()
            _diverse_beam = DiverseBeamSampler(_db_cfg)
            _info("diverse-beam", "diverse beam search: G beam groups with diversity penalty")
        except Exception as _e:
            _warn(f"[diverse-beam] Skipped: {_e}")

    global _bitnet158
    if getattr(args, "bitnet158", False):
        try:
            from squish.quant.bitnet_b158 import BitNet158Config, BitNet158Quantizer
            _bn_cfg = BitNet158Config()
            _bitnet158 = BitNet158Quantizer(_bn_cfg)
            _info("bitnet158", "BitNet-b1.58: ternary {-1,0,+1} weight quantization")
        except Exception as _e:
            _warn(f"[bitnet158] Skipped: {_e}")

    global _spqr_quant_w55
    if getattr(args, "spqr_quant", False):
        try:
            from squish.quant.spqr_quant import SpQRConfig, SpQRQuantizer
            _spqrq_cfg = SpQRConfig()
            _spqr_quant_w55 = SpQRQuantizer(_spqrq_cfg)
            _info("spqr-quant", "SpQR quantizer (spqr_quant): sparse-quantized for frontier models")
        except Exception as _e:
            _warn(f"[spqr-quant] Skipped: {_e}")

    global _omniquant
    if getattr(args, "omniquant", False):
        try:
            from squish.quant.omniquant import OmniQuantConfig, OmniQuantizer
            _omni_cfg = OmniQuantConfig()
            _omniquant = OmniQuantizer(_omni_cfg)
            _info("omniquant", "OmniQuant: LWC+LET joint calibration W4A4/W4A8")
        except Exception as _e:
            _warn(f"[omniquant] Skipped: {_e}")

    global _qsparse
    if getattr(args, "q_sparse", False):
        try:
            from squish.quant.q_sparse import QSparseConfig, QSparsifier
            _qs_cfg = QSparseConfig()
            _qsparse = QSparsifier(_qs_cfg)
            _info("q-sparse", "Q-Sparse: top-K activation sparsity at matmul time")
        except Exception as _e:
            _warn(f"[q-sparse] Skipped: {_e}")

    global _fp4_quantizer
    if getattr(args, "fp4_quant", False):
        try:
            from squish.quant.fp4_quant import FP4Config, FP4Quantizer
            _fp4_cfg = FP4Config()
            _fp4_quantizer = FP4Quantizer(_fp4_cfg)
            _info("fp4-quant", "FP4 quantization: E2M1 FP4 floating-point weights")
        except Exception as _e:
            _warn(f"[fp4-quant] Skipped: {_e}")

    global _ada_round
    if getattr(args, "ada_round", False):
        try:
            from squish.quant.ada_round import AdaRoundConfig, AdaRoundQuantizer
            _adr_cfg = AdaRoundConfig()
            _ada_round = AdaRoundQuantizer(_adr_cfg)
            _info("ada-round", "AdaRound: adaptive rounding via per-weight sigmoid relaxation")
        except Exception as _e:
            _warn(f"[ada-round] Skipped: {_e}")

    # ── Wave 27: Inference velocity features ──────────────────────────────────
    # 1B — FusedSampler: replace multi-pass sampling with a single fused kernel
    global _fused_sampler, _fused_sampler_enabled
    _fused_sampler_enabled = not getattr(args, "no_fused_sampler", False)
    if _fused_sampler_enabled:
        try:
            from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
            _fs_cfg = SamplerConfig(
                temperature=max(1e-5, getattr(args, "temperature", 0.7)),
                top_p=getattr(args, "top_p", 0.9),
                repetition_penalty=1.0,
            )
            _fused_sampler = FusedSampler(_fs_cfg)
            _info("fused-sampler", "single-pass temperature+top-k+top-p+rep-penalty  (~10% decode throughput)")
        except Exception as _e:
            _fused_sampler_enabled = False
            _warn(f"[fused-sampler] Skipped: {_e}")

    # 1C — CacheWarmup: track prefix access patterns for TTFT reduction
    global _cache_warmup_predictor, _cache_warmup_enabled
    _cache_warmup_enabled = not getattr(args, "no_cache_warmup", False)
    if _cache_warmup_enabled:
        try:
            from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
            _cw_cfg = WarmupConfig(top_k=32, min_access_count=2, max_prefix_tokens=256)
            _cache_warmup_predictor = CacheWarmupPredictor(_cw_cfg)
            _info("cache-warmup", "predictive KV prefix pre-warming  (top_k=32  min_count=2)")
        except Exception as _e:
            _cache_warmup_enabled = False
            _warn(f"[cache-warmup] Skipped: {_e}")

    # 1D — TokenMerging: bipartite ToMe during prefill
    global _tome_config, _tome_state
    if getattr(args, "token_merge", False):
        try:
            from squish.token.token_merging import TokenMergingConfig, TokenMergingState
            _tome_config = TokenMergingConfig(
                r=getattr(args, "tome_r", 16),
                start_layer=getattr(args, "tome_start_layer", 4),
                end_layer=getattr(args, "tome_end_layer", 11),
                similarity_threshold=0.5,
            )
            _tome_state = TokenMergingState()
            _info("token-merge", (
                f"ToMe bipartite prefill compression  "
                f"r={_tome_config.r}  "
                f"layers={_tome_config.start_layer}–{_tome_config.end_layer}  "
                f"threshold={_tome_config.similarity_threshold}"
            ))
        except Exception as _e:
            _warn(f"[token-merge] Skipped: {_e}")

    if getattr(args, "lora_adapter", ""):
        try:
            from squish.lora.lora_manager import LoRAManager
            _lora_mgr = LoRAManager()
            _lora_mgr.load(args.lora_adapter)
            _info("lora-adapter", f"{args.lora_adapter}")
        except Exception as _e:
            _warn(f"[lora-adapter] Skipped: {_e}")

    # ── Wave 13 — Attention/KV/Token innovations (lazy stubs) ────────────────
    try:
        from squish.attention.duo_attention import DuoAttentionConfig as _DuoAttentionConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.attention.duo_decoding import DuoDecodingConfig as _DuoDecodingConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.kv.shadow_kv import ShadowKVConfig as _ShadowKVConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.kv.pq_cache import PQCacheConfig as _PQCacheConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.kv.spe_cache import SpeCacheConfig as _SpeCacheConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.token.knapspec import KnapSpecConfig as _KnapSpecConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.token.token_merging import TokenMergingConfig as _TokenMergingConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.token.token_swift import TokenSwiftConfig as _TokenSwiftConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.token.c2t import C2TConfig as _C2TConfig  # noqa: F401
    except ImportError:
        pass

    # ── Wave 14 — Quantisation/Speculative extensions (lazy stubs) ───────────
    try:
        from squish.speculative.sub_spec import SubSpecConfig as _SubSpecConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.quant.dfloat11 import DFloat11Config as _DFloat11Config  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.quant.rans_codec import RANSCodec as _RANSCodec  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.speculative.qspec import QSpecConfig as _QSpecConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.speculative.quant_spec import QuantSpecConfig as _QuantSpecConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.speculative.copy_spec import CopySpecConfig as _CopySpecConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.quant.squeeze_llm import SqueezeLLMConfig as _SqueezeLLMConfig  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.quant.nf4_quant import quantize_nf4 as _quantize_nf4  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.quant.spin_quant import run_rotation as _spin_run_rotation  # noqa: F401
    except ImportError:
        pass
    try:
        from squish.speculative.head_infer import HeadType as _HeadType  # noqa: F401
    except ImportError:
        pass

    # ── Signal bot ────────────────────────────────────────────────────────────
    import os as _os
    _signal_enabled = getattr(args, "signal", False)
    if _signal_enabled:
        _signal_account = getattr(args, "signal_account", "") or _os.environ.get("SIGNAL_ACCOUNT", "")
        _signal_socket  = getattr(args, "signal_socket",  "127.0.0.1:7583") or _os.environ.get("SIGNAL_SOCKET", "127.0.0.1:7583")
        try:
            from .serving.signal_cli import mount_signal as _mount_signal  # package import
        except ImportError:  # pragma: no cover
            from serving.signal_cli import mount_signal as _mount_signal    # direct script run
        _mount_signal(
            app,
            get_state     = lambda: _state,
            get_generate  = lambda: _generate_tokens,
            get_tokenizer = lambda: _state.tokenizer,
            account       = _signal_account,
            socket_addr   = _signal_socket,
            system_prompt = "",
        )

    # ── WhatsApp webhook ──────────────────────────────────────────────────────
    _wa_enabled = getattr(args, "whatsapp", False)
    if _wa_enabled:
        _wa_verify_token    = getattr(args, "whatsapp_verify_token",    "") or _os.environ.get("WHATSAPP_VERIFY_TOKEN",    "")
        _wa_app_secret      = getattr(args, "whatsapp_app_secret",      "") or _os.environ.get("WHATSAPP_APP_SECRET",      "")
        _wa_access_token    = getattr(args, "whatsapp_access_token",    "") or _os.environ.get("WHATSAPP_ACCESS_TOKEN",    "")
        _wa_phone_number_id = getattr(args, "whatsapp_phone_number_id", "") or _os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        try:
            from .serving.whatsapp import mount_whatsapp as _mount_whatsapp  # package import
        except ImportError:  # pragma: no cover
            from serving.whatsapp import mount_whatsapp as _mount_whatsapp    # direct script run
        _mount_whatsapp(
            app,
            get_state        = lambda: _state,
            get_generate     = lambda: _generate_tokens,
            get_tokenizer    = lambda: _state.tokenizer,
            verify_token     = _wa_verify_token,
            app_secret       = _wa_app_secret,
            access_token     = _wa_access_token,
            phone_number_id  = _wa_phone_number_id,
            system_prompt    = "",
        )

    # ── Wave 75/79: optimization status — compact auto-profile or full table ──
    _auto_prof = globals().get("_auto_profile")
    if _auto_prof is not None and _state.model is not None:
        # Wave 79: single-line status when auto-profile is active
        _model_label = getattr(_state, "model_name", "") or "model"
        _load_s = getattr(_state, "load_time_s", 0.0) or 0.0
        _status = _auto_prof.status_line(_model_label, _load_s)
        _ok(_status)
    else:
        _print_optimization_status()

    # ── Wave 76: Initialise agent tool registry ───────────────────────────────
    global _agent_registry
    try:
        from squish.agent.tool_registry import ToolRegistry as _ToolRegistry
        from squish.agent.builtin_tools import register_builtin_tools as _reg_tools
        _agent_registry = _ToolRegistry()
        _reg_tools(_agent_registry)
        _info("agent-registry", f"loaded  tools={len(_agent_registry)}")
    except Exception as _ar_exc:  # noqa: BLE001
        _warn(f"[agent-registry] Could not load built-in tools: {_ar_exc}")

    print()
    _section("")
    print(f"  {_C.B}{_gradient('  Server ready!', _LOGO_GRAD)}{_C.R}")
    print()
    _info("API endpoint",  f"{_C.T}http://{args.host}:{args.port}/v1{_C.R}")
    _info("Web chat UI",   f"{_C.T}http://{args.host}:{args.port}/chat{_C.R}")
    _info("Ollama compat", f"{_C.T}http://{args.host}:{args.port}/api/chat{_C.R}")
    if _wa_enabled:
        _info("WhatsApp",     f"{_C.T}http://{args.host}:{args.port}/webhook/whatsapp{_C.R}")
    if _signal_enabled:
        _info("Signal",       f"{_C.T}http://{args.host}:{args.port}/signal/status{_C.R}")
    print()
    print(f"  {_C.DIM}Set in any OpenAI client:{_C.R}")
    print(f"    {_C.MG}OPENAI_BASE_URL{_C.R}=http://{args.host}:{args.port}/v1")
    print(f"    {_C.MG}OPENAI_API_KEY{_C.R}=squish")
    print()

    # When --trace is active and --trace-output is set, print the trace tree
    # after startup (before blocking in uvicorn) so startup timing is visible.
    if _trace and _TELEMETRY_AVAILABLE:
        _info("telemetry", "span tracing enabled — startup spans captured")
        if getattr(args, "trace_output", ""):
            _tracer = _get_tracer()
            if _tracer is not None:
                _tracer.save_trace(args.trace_output)
                _info("trace-output", f"written to {args.trace_output}")

    import uvicorn  # deferred: only needed when actually starting the server
    _require("uvicorn", "uvicorn[standard]")  # validate before use
    uvicorn.run(
        app,
        host      = args.host,
        port      = args.port,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
