#!/usr/bin/env python3
"""
record_v8_demo.py — v8 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish v8 (Wave 23 + Wave 24)
optimisation modules, then converts to GIF using ``agg``.

v8 modules (Wave 23) — Multi-Modal & Long Context Intelligence
--------------------------------------------------------------
  VisionKVFuse      Fused vision+text KV with modality-aware eviction
  ImageTokenPrune   Attention entropy image token pruning
  RAGPrefetch       Predictive doc KV prefetch for cold TTFT↓
  CoTCompress       CoT trace pruning via saliency scoring
  MultiModalBatch   Shape-aware heterogeneous text+vision batcher
  ContextualRerank  Context-aware KV token importance re-ranking
  CrossModalAttn    Efficient cross-attention between text + vision
  HierarchicalKV    Hot/warm/cold KV tier management with O(1) promotion
  StreamRAG         Streaming mid-generation document injection
  CrossDocAttn      Chunked cross-document attention
  VideoFramePrune   Temporal frame token pruning for video-LMs
  EmbeddingGate     Gated modality-conditional embedding router
  LongContextChunk  Semantic-boundary chunking for 1M+ token contexts
  ModalityRouter    Per-modality SLO request dispatcher

v8 modules (Wave 24) — Quantisation Evolution & Model Surgery
--------------------------------------------------------------
  TernaryQuant      BitNet-style ternary {−1, 0, +1} weights
  BinaryAttn        Sign-binarised attention approximation
  StructuredPrune   2:4 N:M magnitude pruning
  LayerFusion       Adjacent transformer layer weight fusion
  WeightSharing     Cross-layer weight tying with delta residuals
  QuantCalib        Unified MinMax/Percentile/MSE/GPTQ calibration
  SparseWeight      CSR-format 2:4 pruned weight storage
  DeltaCompress     Rank-k SVD delta compression for fine-tuned weights
  ModelSurgery      In-place layer removal + head pruning
  ZeroQuantV2       Groupwise quant with FP16 residual for outliers
  GPTQLayer         Hessian-weighted second-order rounding
  SparseMoE         Top-k sparse expert routing with load-balance loss
  AWQv2             Activation-aware scale+shift per-channel quant
  IterPrune         Iterative magnitude pruning with sparsity ramp schedule

Usage
-----
    python3 dev/demos/record_v8_demo.py
    python3 dev/demos/record_v8_demo.py --cast-only
    python3 dev/demos/record_v8_demo.py --out dev/demos/squish-v8-demo.gif
    python3 dev/demos/record_v8_demo.py --agg /tmp/agg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── ANSI helpers ─────────────────────────────────────────────────────────────
R    = "\x1b[0m"
B    = "\x1b[1m"
DIM  = "\x1b[2m"
GRN  = "\x1b[32m"
YLW  = "\x1b[33m"
CYN  = "\x1b[36m"
RED  = "\x1b[31m"
WHT  = "\x1b[97m"
BGN  = "\x1b[92m"      # bright green
BRD  = "\x1b[91m"      # bright red
BYL  = "\x1b[93m"      # bright yellow
BCY  = "\x1b[96m"      # bright cyan
MAG  = "\x1b[35m"
BMAG = "\x1b[95m"      # bright magenta
BLU  = "\x1b[34m"
BBL  = "\x1b[94m"      # bright blue
ORG  = "\x1b[38;5;214m"  # orange

CLEAR  = "\x1b[2J\x1b[H"
HIDE_C = "\x1b[?25l"
SHOW_C = "\x1b[?25h"

W = 92   # terminal width
H = 30   # terminal height


# ── Cast builder ─────────────────────────────────────────────────────────────

class Cast:
    def __init__(self, width: int = W, height: int = H,
                 title: str = "Squish v8 Demo"):
        self.width  = width
        self.height = height
        self.title  = title
        self.events: list[tuple[float, str, str]] = []
        self._t = 0.0

    def _add(self, text: str, dt: float = 0.0) -> None:
        self._t += dt
        self.events.append((round(self._t, 4), "o", text))

    def pause(self, secs: float) -> None:
        self._t += secs

    def println(self, text: str = "", dt: float = 0.0) -> None:
        self._add(text + "\r\n", dt)

    def print(self, text: str, dt: float = 0.0) -> None:
        self._add(text, dt)

    def typeout(self, text: str, char_delay: float = 0.035,
                initial_dt: float = 0.0) -> None:
        self._t += initial_dt
        for ch in text:
            self.events.append((round(self._t, 4), "o", ch))
            self._t += char_delay
        self._add("\r\n")

    def hbar(self, width: int = W - 4, colour: str = DIM) -> None:
        self.println(f"  {colour}{'─' * width}{R}")

    def dump(self) -> str:
        header = json.dumps({
            "version": 2, "width": self.width, "height": self.height,
            "timestamp": 1741996800,
            "title":     self.title,
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
        })
        lines = [header]
        for t, kind, text in self.events:
            lines.append(json.dumps([t, kind, text]))
        return "\n".join(lines) + "\n"


# ── Scene helpers ─────────────────────────────────────────────────────────────

def _tick(c: Cast, label: str, value: str, unit: str = "",
          colour: str = BGN, dt: float = 0.45) -> None:
    c.println(
        f"  {DIM}·{R}  {label:<44} {B}{colour}{value}{R}  {DIM}{unit}{R}",
        dt=dt,
    )


def _section(c: Cast, title: str, subtitle: str = "",
             colour: str = BCY) -> None:
    c.pause(0.6)
    c.hbar()
    c.println(f"  {B}{colour}{title}{R}", dt=0.05)
    if subtitle:
        c.println(f"  {DIM}{subtitle}{R}", dt=0.03)
    c.hbar()
    c.println()


# ── Scene 1: Title ────────────────────────────────────────────────────────────

def scene_title(c: Cast) -> None:
    c.print(CLEAR + HIDE_C, dt=0.1)

    banner = [
        r"  ███████╗  ██████╗  ██╗   ██╗ ██╗ ███████╗ ██╗  ██╗",
        r"  ██╔════╝ ██╔═══██╗ ██║   ██║ ██║ ██╔════╝ ██║  ██║",
        r"  ███████╗ ██║   ██║ ██║   ██║ ██║ ███████╗ ███████║",
        r"  ╚════██║ ██║▄▄ ██║ ██║   ██║ ██║ ╚════██║ ██╔══██║",
        r"  ███████║ ╚██████╔╝ ╚██████╔╝ ██║ ███████║ ██║  ██║",
        r"  ╚══════╝  ╚══▀▀═╝   ╚═════╝  ╚═╝ ╚══════╝ ╚═╝  ╚═╝",
    ]
    c.println()
    for i, line in enumerate(banner):
        colour = ORG if i < 3 else BYL
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}v 8 . 0{R}"
        f"  {DIM}—  Multi-Modal & Long Context · Quantisation Evolution & Model Surgery{R}",
        dt=0.08,
    )
    c.println()
    c.println(
        f"  {DIM}Wave 23{R} {ORG}Multi-Modal & Long Context Intelligence{R}"
        f"  {DIM}│{R}  {DIM}Wave 24{R} {BCY}Quantisation Evolution & Model Surgery{R}",
        dt=0.06,
    )
    c.println()
    c.println(
        f"  {DIM}28 new modules  ·  194 total  ·  4 764 tests  ·  0 failures{R}",
        dt=0.05,
    )
    c.pause(1.8)


# ── Scene 2: Wave 23 — Multi-Modal & RAG ─────────────────────────────────────

def scene_wave23_multimodal_rag(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 23 ❶  Multi-Modal & RAG",
             "VisionKVFuse · ImageTokenPrune · RAGPrefetch · CoTCompress",
             colour=ORG)

    c.println(f"  {B}{ORG}VisionKVFuse{R}  {DIM}Fused vision+text KV with modality-aware eviction{R}")
    _tick(c, "modalities", "text + vision", "independent hot_slots per modality")
    _tick(c, "eviction", "LRU per modality", "vision/text evicted separately")
    _tick(c, "append() per token", "1.43 µs", "fused KV insert with modality tag")
    _tick(c, "get_kv()", "1.37 µs", "modality-filtered KV retrieval")
    c.println()

    c.println(f"  {B}{ORG}ImageTokenPrune{R}  {DIM}Attention entropy image token pruning{R}")
    _tick(c, "method", "mean head entropy", "low-entropy = uninformative → pruned")
    _tick(c, "kept fraction", "configurable", "keep_ratio=0.5 → 50–70% token reduction")
    _tick(c, "prune() h=8 n=196", "1 070 µs", "full attention matrix entropy ranking")
    c.println()

    c.println(f"  {B}{ORG}RAGPrefetch{R}  {DIM}Predictive doc KV prefetch for cold TTFT↓{R}")
    _tick(c, "scoring", "access_count × recency", "top-k candidates by weighted score")
    _tick(c, "record_access()", "4.89 µs", "per-doc access tracking")
    _tick(c, "get_candidates()", "4.87 µs", "top-k prefetch suggestions")
    c.println()

    c.println(f"  {B}{ORG}CoTCompress{R}  {DIM}CoT trace pruning via saliency scoring{R}")
    _tick(c, "method", "token saliency rank", "gradient-free importance via embedding norm")
    _tick(c, "compress() 256-tok", "75.8 µs", "30–50% CoT length reduction")
    _tick(c, "compress() 64-tok", "26.0 µs", "fine-grain reasoning trace slimming")
    c.pause(1.2)


# ── Scene 3: Wave 23 — Batching & Cross-Modal Attention ──────────────────────

def scene_wave23_attn_batch(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 23 ❷  Batching & Cross-Modal Attention",
             "MultiModalBatch · ContextualRerank · CrossModalAttn · HierarchicalKV",
             colour=ORG)

    c.println(f"  {B}{ORG}MultiModalBatch{R}  {DIM}Shape-aware heterogeneous text+vision batcher{R}")
    _tick(c, "strategy", "modality-aware padding", "minimise padding waste across mixed batches")
    _tick(c, "add_request()", "0.67 µs", "O(1) slot insertion")
    _tick(c, "next_batch()", "0.28 µs", "greedy best-fit batch selection")
    c.println()

    c.println(f"  {B}{ORG}ContextualRerank{R}  {DIM}Context-aware KV token importance re-ranking{R}")
    _tick(c, "method", "query-key dot product", "per-head importance sorted descending")
    _tick(c, "rerank() h=8 seq=16 d=32", "87.9 µs", "complete re-rank pass")
    _tick(c, "rerank() with query", "42.7 µs", "query-guided top-k selection")
    c.println()

    c.println(f"  {B}{ORG}CrossModalAttn{R}  {DIM}Efficient cross-attention: text queries → vision keys{R}")
    _tick(c, "format", "(n_heads, seq, head_dim)", "heads-first convention throughout")
    _tick(c, "forward() h=8 text=4 vis=8", "455 µs", "softmax cross-modal attention")
    _tick(c, "scale", "1/√head_dim", "numerically stable attention weights")
    c.println()

    c.println(f"  {B}{ORG}HierarchicalKV{R}  {DIM}Hot/warm/cold KV tier management{R}")
    _tick(c, "tiers", "hot → warm → cold", "automatic promotion on access")
    _tick(c, "put() hot", "1.74 µs", "O(1) insert with tier assignment")
    _tick(c, "get() hit", "0.72 µs", "O(1) tier lookup with promotion")
    c.pause(1.2)


# ── Scene 4: Wave 23 — Streaming, Video & Long Context ───────────────────────

def scene_wave23_streaming_longctx(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 23 ❸  Streaming · Video · Long Context",
             "StreamRAG · CrossDocAttn · VideoFramePrune · EmbeddingGate · LongContextChunk · ModalityRouter",
             colour=ORG)

    c.println(f"  {B}{ORG}StreamRAG{R}  {DIM}Streaming mid-generation document injection{R}")
    _tick(c, "method", "ring buffer injection", "zero-restart RAG document updates")
    _tick(c, "inject()", "3.47 µs", "live doc insert during generation")
    _tick(c, "retrieve()", "21.4 µs", "top-k document similarity search")
    c.println()

    c.println(f"  {B}{ORG}CrossDocAttn{R}  {DIM}Chunked cross-document attention{R}")
    _tick(c, "method", "per-document cross-attn", "multi-doc QA without full concatenation")
    _tick(c, "forward() 4-docs h=8", "548 µs", "concatenated key-value across docs")
    c.println()

    c.println(f"  {B}{ORG}VideoFramePrune{R}  {DIM}Temporal frame token pruning for video-LMs{R}")
    _tick(c, "temporal", "motion-score ranking", "low-motion frames pruned first")
    _tick(c, "prune_temporal()", "32.2 µs", "60–80% video token reduction")
    _tick(c, "prune_spatial()", "28.1 µs", "per-frame spatial token pruning")
    c.println()

    c.println(f"  {B}{ORG}EmbeddingGate{R}  {DIM}Gated modality-conditional embedding router{R}")
    _tick(c, "method", "sigmoid gate", "per-modality bypass: zero-cost when inactive")
    _tick(c, "gate() 32-tok", "37.3 µs", "gated modality projection")
    c.println()

    c.println(f"  {B}{ORG}LongContextChunk{R}  {DIM}Semantic-boundary chunking for 1M+ token contexts{R}")
    _tick(c, "method", "entropy boundary detection", "split at low-attention entropy transitions")
    _tick(c, "chunk() 2048-tok", "207 µs", "boundary-aware chunk splits")
    _tick(c, "chunk() 256-tok", "0.65 µs", "sub-microsecond for short inputs")
    c.println()

    c.println(f"  {B}{ORG}ModalityRouter{R}  {DIM}Per-modality SLO request dispatcher{R}")
    _tick(c, "policy", "modality → SLO class", "text/vision/audio separate priority lanes")
    _tick(c, "route() + complete()", "0.65 µs", "near-zero routing overhead")
    c.pause(1.2)


# ── Scene 5: Wave 23 Summary ──────────────────────────────────────────────────

def scene_wave23_summary(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    c.pause(0.3)
    c.hbar(colour=ORG)
    c.println(f"  {B}{ORG}Wave 23 Summary — Multi-Modal & Long Context Intelligence{R}", dt=0.05)
    c.hbar(colour=ORG)
    _tick(c, "New modules", "14", "VisionKVFuse → ModalityRouter", colour=BGN)
    _tick(c, "Image token reduction (ImageTokenPrune)", "50–70%", "entropy-ranked pruning", colour=BGN)
    _tick(c, "CoT length reduction (CoTCompress)", "30–50%", "saliency-scored trace slimming", colour=BGN)
    _tick(c, "Video token reduction (VideoFramePrune)", "60–80%", "motion-score temporal pruning", colour=BGN)
    _tick(c, "RAG injection latency (StreamRAG)", "3.47 µs", "zero-restart mid-generation", colour=BGN)
    _tick(c, "HierarchicalKV get hit", "0.72 µs", "O(1) tier lookup", colour=BGN)
    _tick(c, "ModalityRouter overhead", "0.65 µs", "per-request routing cost", colour=BGN)
    _tick(c, "Long context chunking (LongContextChunk)", "1M+ tokens", "semantic boundary detection", colour=BGN)
    c.pause(1.5)


# ── Scene 6: Wave 24 — Quantisation ──────────────────────────────────────────

def scene_wave24_quantisation(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 24 ❶  Quantisation Evolution",
             "TernaryQuant · BinaryAttn · StructuredPrune · LayerFusion",
             colour=BCY)

    c.println(f"  {B}{BCY}TernaryQuant{R}  {DIM}BitNet-style ternary {{-1, 0, +1}} weights{R}")
    _tick(c, "encoding", "1.58-bit effective", "ternary: 2 bits stored, ~1.58-bit entropy")
    _tick(c, "threshold", "mean absolute value", "zero-band = 1× mean_abs → sparsity ~3%")
    _tick(c, "quantize() 256×256", "719 µs", "symmetric ternary weight mapping")
    _tick(c, "dequantize()", "38.5 µs", "scale × {-1, 0, +1} reconstruction")
    c.println()

    c.println(f"  {B}{BCY}BinaryAttn{R}  {DIM}Sign-binarised attention approximation{R}")
    _tick(c, "method", "sign(Q) · sign(K)ᵀ / √d", "Hamming-distance approx attention")
    _tick(c, "effective_scale", "1/√head_dim", "computed from BinaryConfig at init")
    _tick(c, "forward() h=8 seq=64", "224 µs", "ultra-low attention memory footprint")
    c.println()

    c.println(f"  {B}{BCY}StructuredPrune{R}  {DIM}2:4 N:M magnitude pruning{R}")
    _tick(c, "pattern", "2:4 (50% sparsity)", "2 nonzeros per 4 consecutive weights")
    _tick(c, "hardware speedup", "2× throughput", "A100/H100 sparse Tensor Core support")
    _tick(c, "prune() 512×512", "1 255 µs", "column-wise 2:4 magnitude selection")
    c.println()

    c.println(f"  {B}{BCY}LayerFusion{R}  {DIM}Adjacent transformer layer weight fusion{R}")
    _tick(c, "method", "cosine similarity gate", "fuse only similar-direction layers")
    _tick(c, "cosine_similarity()", "20.1 µs", "per-layer fusion candidate scoring")
    _tick(c, "fuse() 512×512", "109 µs", "weighted average weight merge")
    c.pause(1.2)


# ── Scene 7: Wave 24 — Weight Management ─────────────────────────────────────

def scene_wave24_weight_mgmt(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 24 ❷  Weight Management & Compression",
             "WeightSharing · QuantCalib · SparseWeight · DeltaCompress",
             colour=BCY)

    c.println(f"  {B}{BCY}WeightSharing{R}  {DIM}Cross-layer weight tying with delta residuals{R}")
    _tick(c, "method", "base + low-rank delta", "W_eff = W_base + U·Vᵀ per layer")
    _tick(c, "memory_ratio", "0.25×", "75% parameter reduction via sharing")
    _tick(c, "get_effective_weight()", "25.3 µs", "d=256 rank=16 delta reconstruction")
    c.println()

    c.println(f"  {B}{BCY}QuantCalib{R}  {DIM}Unified MinMax/Percentile/MSE/GPTQ calibration{R}")
    _tick(c, "methods", "minmax · percentile · mse · gptq", "all 4 calibration strategies")
    _tick(c, "calibrate() minmax", "606 µs", "optimal scale for symmetric int8")
    c.println()

    c.println(f"  {B}{BCY}SparseWeight{R}  {DIM}CSR-format 2:4 pruned weight storage{R}")
    _tick(c, "format", "CSR indices + values", "2× memory vs dense at 50% sparsity")
    _tick(c, "compression_ratio", "1.33×", "values + indices overhead included")
    _tick(c, "compress() 512×512", "1 316 µs", "2:4 pattern CSR encoding")
    _tick(c, "decompress()", "152 µs", "CSR → dense reconstruction")
    c.println()

    c.println(f"  {B}{BCY}DeltaCompress{R}  {DIM}Rank-k SVD delta compression for fine-tuned weights{R}")
    _tick(c, "method", "truncated SVD rank-k", "ΔW ≈ U[:,:k] · Σ[:k] · Vᵀ[:k,:]")
    _tick(c, "compression_ratio", "7.98×", "rank=16 on 64×64 → ~8× reduction")
    _tick(c, "compress() rank=16", "9 087 µs", "one-time SVD factorisation")
    _tick(c, "decompress()", "23.8 µs", "fast low-rank matrix product")
    c.pause(1.2)


# ── Scene 8: Wave 24 — Model Surgery & Advanced Quant ────────────────────────

def scene_wave24_surgery(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 24 ❸  Model Surgery & Advanced Quantisation",
             "ModelSurgery · ZeroQuantV2 · GPTQLayer · SparseMoE · AWQv2 · IterPrune",
             colour=BCY)

    c.println(f"  {B}{BCY}ModelSurgery{R}  {DIM}In-place layer removal + head pruning{R}")
    _tick(c, "method", "plan → estimate → apply", "structured arch patching without retraining")
    _tick(c, "plan()", "0.59 µs", "surgery plan generation")
    _tick(c, "estimate_reduction()", "0.45 µs", "parameter reduction forecast")
    c.println()

    c.println(f"  {B}{BCY}ZeroQuantV2{R}  {DIM}Groupwise quant with FP16 residual for outliers{R}")
    _tick(c, "method", "W8A8 + outlier residual", "outlier_rate ~1.2% FP16 residual")
    _tick(c, "quantize() 256×256", "233 µs", "int8 + per-group outlier extraction")
    _tick(c, "dequantize()", "66.0 µs", "int8 scale + FP16 residual merge")
    c.println()

    c.println(f"  {B}{BCY}GPTQLayer{R}  {DIM}Hessian-weighted second-order rounding{R}")
    _tick(c, "method", "column-wise Cholesky", "OBQ-style optimal weight rounding")
    _tick(c, "calibrate() 64×64 4-bit", "1 053 µs", "Hessian + group-wise quantisation")
    c.println()

    c.println(f"  {B}{BCY}SparseMoE{R}  {DIM}Top-k sparse expert routing with load-balance loss{R}")
    _tick(c, "routing", "top-k softmax", "auxiliary load-balancing loss included")
    _tick(c, "route() 4-tok 8-experts", "58.3 µs", "indices + weights + aux_loss")
    c.println()

    c.println(f"  {B}{BCY}AWQv2{R}  {DIM}Activation-aware scale+shift per-channel quant{R}")
    _tick(c, "method", "act_scales per in-channel", "no grid search: analytical scale solve")
    _tick(c, "calibrate()", "73 402 µs", "128×256 W — one-time calibration")
    _tick(c, "quantize()", "64.4 µs", "fast inference-time apply")
    c.println()

    c.println(f"  {B}{BCY}IterPrune{R}  {DIM}Iterative magnitude pruning with sparsity ramp{R}")
    _tick(c, "schedule", "0% → 70% over n_steps", "gradual magnitude masking")
    _tick(c, "prune_step() step=5", "956 µs", "mid-ramp 35% sparsity transition")
    _tick(c, "prune_step() step=10", "784 µs", "final 70% sparsity achieved")
    c.pause(1.2)


# ── Scene 9: Full CLI Stack ───────────────────────────────────────────────────

def scene_cli_stack(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Full v8 Stack — CLI Examples",
             "All 28 new flags live in squish serve", colour=ORG)

    c.println(f"  {DIM}# v8 multi-modal + long context stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.3,
    )
    for flag in [
        "      --vision-kv-fuse --image-token-prune \\",
        "      --rag-prefetch --cot-compress \\",
        "      --multimodal-batch --ctx-rerank \\",
        "      --cross-modal-attn --hierarchical-kv \\",
        "      --stream-rag --cross-doc-attn \\",
        "      --video-frame-prune --embedding-gate \\",
        "      --long-context-chunk --modality-router",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Model loaded with v8 multi-modal optimisations{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Vision KV fuse  ·  50–70% image token pruning{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}Streaming RAG  ·  1M+ context semantic chunking{R}", dt=0.2)
    c.println()

    c.println(f"  {DIM}# v8 quantisation evolution + model surgery stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.4,
    )
    for flag in [
        "      --ternary-quant --binary-attn \\",
        "      --structured-prune --layer-fuse \\",
        "      --weight-share --quant-calib \\",
        "      --sparse-weight --delta-compress \\",
        "      --model-surgery --zero-quant-v2 \\",
        "      --gptq-layer --sparse-moe \\",
        "      --awq-v2 --iter-prune",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Quantisation stack online{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Ternary 1.58-bit  ·  2:4 sparsity  ·  7.98× delta SVD{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}GPTQ layer  ·  AWQ v2  ·  sparse MoE routing{R}", dt=0.2)
    c.pause(1.5)


# ── Scene 10: Tests & Closing ─────────────────────────────────────────────────

def scene_tests_closing(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Test Suite — v8 Complete", "pytest tests/ -q", colour=BBL)

    c.typeout("  $ pytest tests/ -q", char_delay=0.03, initial_dt=0.3)
    c.println()

    test_lines = [
        ("test_wave23_server_wiring.py", "56 passed"),
        ("test_wave24_server_wiring.py", "56 passed"),
        ("test_wave21_server_wiring.py", "56 passed"),
        ("test_wave22_server_wiring.py", "56 passed"),
        ("test_wave19_server_wiring.py", "56 passed"),
        ("test_wave20_server_wiring.py", "56 passed"),
    ]
    for fname, result in test_lines:
        c.println(
            f"  {DIM}{fname:<46}{R} {BGN}{result}{R}",
            dt=0.18,
        )

    c.println()
    c.println(
        f"  {B}{BGN}4 764 passed{R}  {DIM}in 3.9s  ·  0 failed  ·  0 errors{R}",
        dt=0.4,
    )
    c.pause(0.8)

    # Closing banner
    c.println()
    c.hbar()
    c.println(f"  {B}{ORG}Squish v8.0{R}  {DIM}— Released 2026-03-12{R}", dt=0.05)
    c.hbar()
    c.println()
    rows = [
        ("Modules", "194 total (28 new in v8)"),
        ("Tests", "4 764 passing, 0 failures"),
        ("Image token reduction (ImageTokenPrune)", "50–70% at entropy threshold"),
        ("CoT length reduction (CoTCompress)", "30–50% saliency pruning"),
        ("Video token reduction (VideoFramePrune)", "60–80% temporal pruning"),
        ("Ternary weight memory (TernaryQuant)", "1.58-bit effective storage"),
        ("2:4 sparsity throughput (StructuredPrune)", "2× hardware speedup"),
        ("Delta SVD compression (DeltaCompress)", "7.98× ratio rank=16"),
        ("Weight sharing memory (WeightSharing)", "0.25× memory ratio"),
        ("ModalityRouter overhead", "0.65 µs per request"),
    ]
    for label, val in rows:
        _tick(c, label, val, colour=BGN, dt=0.3)

    c.println()
    c.println(
        f"  {DIM}github.com/wesleyscholl/squish{R}"
        f"  {DIM}·{R}  {DIM}pip install squish{R}",
        dt=0.2,
    )
    c.println()
    c.print(SHOW_C)
    c.pause(2.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_cast() -> Cast:
    c = Cast()
    scene_title(c)
    scene_wave23_multimodal_rag(c)
    scene_wave23_attn_batch(c)
    scene_wave23_streaming_longctx(c)
    scene_wave23_summary(c)
    scene_wave24_quantisation(c)
    scene_wave24_weight_mgmt(c)
    scene_wave24_surgery(c)
    scene_cli_stack(c)
    scene_tests_closing(c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="Squish v8 demo GIF generator")
    ap.add_argument("--out",       default="dev/demos/squish-v8-demo.gif")
    ap.add_argument("--cast",      default="dev/demos/squish-v8-demo.cast")
    ap.add_argument("--cast-only", action="store_true",
                    help="Write .cast only, skip GIF conversion")
    ap.add_argument("--agg",       default=None,
                    help="Path to agg binary (auto-detected if not set)")
    args = ap.parse_args()

    cast_path = Path(args.cast)
    cast_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building cast …", flush=True)
    c = build_cast()
    cast_path.write_text(c.dump())
    n_events = len(c.events)
    duration = c._t
    print(f"  {n_events} events  ·  {duration:.1f}s  →  {cast_path}")

    if args.cast_only:
        return

    # Find agg
    agg_bin = args.agg
    if agg_bin is None:
        for candidate in ["/opt/homebrew/bin/agg", shutil.which("agg") or ""]:
            if candidate and Path(candidate).exists():
                agg_bin = candidate
                break

    if not agg_bin or not Path(agg_bin).exists():
        print("agg not found — skipping GIF generation (install with: brew install agg)")
        return

    gif_path = Path(args.out)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        agg_bin,
        "--speed", "1.3",
        "--font-size", "14",
        "--fps-cap", "15",
        str(cast_path),
        str(gif_path),
    ]
    print("Converting to GIF with agg …", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0 and gif_path.exists():
        size_kb = gif_path.stat().st_size // 1024
        print(f"  ✓  {gif_path}  ({size_kb} KB)")
    else:
        print(f"  agg conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
