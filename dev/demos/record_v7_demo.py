#!/usr/bin/env python3
"""
record_v7_demo.py — v7 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish v7 (Wave 21 + Wave 22)
optimisation modules, then converts to GIF using ``agg``.

v7 modules (Wave 21) — Advanced Memory & Decode
-------------------------------------------------
  TreeVerifier      Batched tree-parallel speculative verification
  KVCompress        Online KV quantisation + pruning during generation
  DynamicNTK        Per-request runtime RoPE base auto-scaling
  QuantSpecDecode   INT4 draft + FP16 verify speculative decode
  SparseAttnIndex   ANN KV retrieval index for sub-linear attention
  MixedPrecisionKV  Per-head INT8/INT4/FP16 KV via sensitivity analysis
  PipelineBubble    Overlapped prefill + decode across pipeline stages
  LayerwiseDecode   Layer-by-layer early-exit decode with multi-stream output
  CodecKV           Learned encode/decode KV codec
  DedupeAttn        Near-duplicate Q/K detection + output reuse
  FlashPrefill      Chunked flash attention for prefill with causal mask
  BudgetSpec        Token-budget-aware speculative decode
  RetentionAttn     Retention-style recurrent state attention
  KVRouter          Cross-instance KV routing for disaggregated serving

v7 modules (Wave 22) — Production Serving & Observability
----------------------------------------------------------
  MultiTenantSched  Fair per-tenant QoS scheduling
  RequestRouter     Load-aware request routing across replicas
  CacheWarmup       Predictive KV cache pre-warming from patterns
  TokenBudgetGate   Hard per-request token budget with graceful truncation
  ObservabilityHook Zero-overhead per-step inference tracing
  RequestCoalesce   Merge requests sharing long common prefixes
  AdaptiveQuantize  Runtime precision switching under memory pressure
  HealthCheck       Degradation-aware server health monitoring
  FaultTolerance    Graceful OOM degradation policy
  ModelPool         Hot model pool with lazy-load + LRU eviction
  StreamingChunk    Sub-token-latency chunked streaming with backpressure
  CostEstimator     Per-request compute cost estimation
  SLAMonitor        Real-time SLA violation detection + remediation
  ContextCache      Persistent cross-session context cache with TTL

Usage
-----
    python3 dev/demos/record_v7_demo.py
    python3 dev/demos/record_v7_demo.py --cast-only
    python3 dev/demos/record_v7_demo.py --out dev/demos/squish-v7-demo.gif
    python3 dev/demos/record_v7_demo.py --agg /tmp/agg
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
                 title: str = "Squish v7 Demo"):
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
        colour = BMAG if i < 3 else MAG
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}v 7 . 0{R}"
        f"  {DIM}—  Advanced Decode · Production Serving · Observability{R}",
        dt=0.08,
    )
    c.println()
    c.println(
        f"  {DIM}Wave 21{R} {BMAG}Advanced Memory & Decode{R}"
        f"  {DIM}│{R}  {DIM}Wave 22{R} {BCY}Production Serving & Observability{R}",
        dt=0.06,
    )
    c.println()
    c.println(
        f"  {DIM}28 new modules  ·  166 total  ·  4 390 tests  ·  0 failures{R}",
        dt=0.05,
    )
    c.pause(1.8)


# ── Scene 2: Wave 21 — Advanced Memory & Speculative Decode ──────────────────

def scene_wave21_memory_decode(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 21 ❶  Advanced Memory & Speculative Decode",
             "TreeVerifier · KVCompress · DynamicNTK · QuantSpecDecode",
             colour=BMAG)

    c.println(f"  {B}{BMAG}TreeVerifier{R}  {DIM}Batched tree-parallel speculative verification{R}")
    _tick(c, "acceptance", "rejection-sampling", "min(1, p_target / p_draft) per branch")
    _tick(c, "strategy", "longest-prefix", "keeps longest jointly-accepted token path")
    _tick(c, "verify() tree depth=3 width=4", "521.7 µs", "structured multi-token acceptance")
    c.println()

    c.println(f"  {B}{BMAG}KVCompress{R}  {DIM}Online KV quantisation + pruning during generation{R}")
    _tick(c, "pruning", "global quantile", "prune low-norm key positions online")
    _tick(c, "quantisation", "symmetric INT8", "scale = abs_max / 127")
    _tick(c, "compress() seq=64 heads=4", "317.6 µs", "prune + quantise")
    _tick(c, "decompress()", "95.0 µs", "dequantise only")
    c.println()

    c.println(f"  {B}{BMAG}DynamicNTK{R}  {DIM}Per-request runtime RoPE base auto-scaling{R}")
    _tick(c, "formula", "NTK-aware", "scale_factor = α·(seq/max) − (α−1), clamped ≥ 1")
    _tick(c, "trigger", ">80% context fill", "auto-extends at context saturation")
    _tick(c, "get_freqs() scaled vs unscaled", "4.3 µs", "no retraining required")
    c.println()

    c.println(f"  {B}{BMAG}QuantSpecDecode{R}  {DIM}INT4 draft + FP16 verify speculative decode{R}")
    _tick(c, "draft memory", "4× reduction", "INT4 vs FP16 draft heads")
    _tick(c, "quantize_draft()", "84.0 µs", "per-channel INT4 sym quant")
    _tick(c, "verify()", "232.0 µs", "dequantise + rejection sampling")
    c.pause(1.2)


# ── Scene 3: Wave 21 — Attention & Retrieval ─────────────────────────────────

def scene_wave21_attention(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 21 ❷  Attention & Retrieval",
             "SparseAttnIndex · MixedPrecisionKV · PipelineBubble · LayerwiseDecode",
             colour=BMAG)

    c.println(f"  {B}{BMAG}SparseAttnIndex{R}  {DIM}ANN KV retrieval index for sub-linear attention{R}")
    _tick(c, "method", "L2-normalised cosine ANN", "np.argpartition O(n) top-k")
    _tick(c, "build() seq=256 heads=4 d=32", "113.7 µs", "index construction")
    _tick(c, "query() top_k=16", "75.0 µs", "sub-linear KV attention cost")
    c.println()

    c.println(f"  {B}{BMAG}MixedPrecisionKV{R}  {DIM}Per-head INT4/INT8/FP16 KV via sensitivity{R}")
    _tick(c, "precision assignment", "variance-based", "min-max normalised → INT4/INT8/FP16")
    _tick(c, "KV memory reduction", "2–4×", "at iso-quality versus full FP16")
    _tick(c, "assign_precisions() 32 heads", "6.6 µs", "one-time calibration step")
    _tick(c, "store() / load()", "0.95 / 0.82 µs", "per-token per-head")
    c.println()

    c.println(f"  {B}{BMAG}PipelineBubble{R}  {DIM}Overlapped 1F1B pipeline with bubble elimination{R}")
    _tick(c, "schedule", "1F1B interleaved", "minimise idle stage gaps")
    _tick(c, "build_schedule()", "13.1 µs", "4 stages, 8 microbatches")
    _tick(c, "bubble_fraction", "27%", "→ near-zero with sufficient microbatches")
    c.println()

    c.println(f"  {B}{BMAG}LayerwiseDecode{R}  {DIM}Layer-by-layer early-exit decode{R}")
    _tick(c, "exit criterion", "softmax confidence", "probe vocab logit threshold")
    _tick(c, "should_exit()", "6.9 µs", "per-layer confidence check")
    _tick(c, "process_layer()", "11.0 µs", "hidden state + optional exit")
    c.pause(1.2)


# ── Scene 4: Wave 21 — Codec, Dedup & Recurrence ────────────────────────────

def scene_wave21_codec(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 21 ❸  Codec · Dedup · Recurrence · Routing",
             "CodecKV · DedupeAttn · FlashPrefill · BudgetSpec · RetentionAttn · KVRouter",
             colour=BMAG)

    c.println(f"  {B}{BMAG}CodecKV{R}  {DIM}Learned k-means++ KV codec with 204× compression{R}")
    _tick(c, "method", "k-means++ codebook", "independent key + value codebooks")
    _tick(c, "compression_ratio", "204.8×", "32·head_dim / log₂(n_codebook)")
    _tick(c, "fit() n=256 heads=4 d=32", "8 560 µs", "one-time training (Lloyd's 20 iter)")
    _tick(c, "encode() / decode()", "62.8 / 2.0 µs", "per-token encode; fast decode")
    c.println()

    c.println(f"  {B}{BMAG}DedupeAttn{R}  {DIM}Near-duplicate Q/K detection + output reuse{R}")
    _tick(c, "cache", "FIFO per head", "cosine similarity threshold lookup")
    _tick(c, "lookup()", "212 µs", "batch cosine similarity check")
    _tick(c, "store()", "3.4 µs", "FIFO cache with max_cache eviction")
    c.println()

    c.println(f"  {B}{BMAG}FlashPrefill{R}  {DIM}Chunked causal flash attention for prefill{R}")
    _tick(c, "memory", "O(seq × chunk)", "not O(seq²) — eliminates OOM on long context")
    _tick(c, "prefill() seq=256 chunk=64", "3 653 µs", "h=4, d=32 causal chunked")
    c.println()

    c.println(f"  {B}{BMAG}RetentionAttn{R}  {DIM}RetNet recurrent state — O(1) per step{R}")
    _tick(c, "state update", "S = γ·S + kᵀ·v", "decay γ=0.9, no growing KV cache")
    _tick(c, "step()", "34.3 µs", "n_heads=4, d=32 — linear recurrence")
    c.println()

    c.println(f"  {B}{BMAG}BudgetSpec{R}  {DIM}Token-budget-aware speculative decode{R}")
    _tick(c, "ramp-down", "linear", "full n_draft below threshold, ramps to 1")
    _tick(c, "step()", "0.51 µs", "near-zero scheduling overhead")
    c.println()

    c.println(f"  {B}{BMAG}KVRouter{R}  {DIM}Cross-instance KV routing for disaggregated serving{R}")
    _tick(c, "method", "SHA-256 consistent hash", "deterministic seq_id → node mapping")
    _tick(c, "route()", "1.2 µs", "prefill/decode always on separate nodes")
    c.pause(1.2)


# ── Scene 5: Wave 21 Summary ──────────────────────────────────────────────────

def scene_wave21_summary(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    c.pause(0.3)
    c.hbar(colour=BMAG)
    c.println(f"  {B}{BMAG}Wave 21 Summary — Advanced Memory & Decode{R}", dt=0.05)
    c.hbar(colour=BMAG)
    _tick(c, "New modules", "14", "TreeVerifier → KVRouter", colour=BGN)
    _tick(c, "INT4 draft memory (QuantSpecDecode)", "4× reduction", "vs FP16 draft", colour=BGN)
    _tick(c, "KV compression (CodecKV)", "204.8× ratio", "learned k-means++ codebook", colour=BGN)
    _tick(c, "KV memory (MixedPrecisionKV)", "2–4×", "per-head precision assignment", colour=BGN)
    _tick(c, "Context extension (DynamicNTK)", "auto", "trigger at 80% context fill", colour=BGN)
    _tick(c, "Attention memory (FlashPrefill)", "O(seq·chunk)", "vs O(seq²) naive", colour=BGN)
    _tick(c, "Recurrence memory (RetentionAttn)", "O(1)", "per step, linear recurrence", colour=BGN)
    _tick(c, "Decode throughput (BudgetSpec)", "budget-aware", "ramp-down to 1 draft near limit", colour=BGN)
    c.pause(1.5)


# ── Scene 6: Wave 22 — Scheduling & Routing ──────────────────────────────────

def scene_wave22_scheduling(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 22 ❶  Scheduling & Routing",
             "MultiTenantSched · RequestRouter · CacheWarmup · TokenBudgetGate",
             colour=BCY)

    c.println(f"  {B}{BCY}MultiTenantSched{R}  {DIM}Fair per-tenant QoS scheduling{R}")
    _tick(c, "policy", "weighted fair queuing", "weight/queue-depth ratio selection")
    _tick(c, "SLO isolation", "per-tenant", "independent SLO violation tracking")
    _tick(c, "next_request()", "0.65 µs", "sub-microsecond scheduler overhead")
    c.println()

    c.println(f"  {B}{BCY}RequestRouter{R}  {DIM}Load-aware request routing across replicas{R}")
    _tick(c, "policy", "least-loaded", "active request count per replica")
    _tick(c, "route() + complete()", "2.1 µs", "atomic load accounting round-trip")
    c.println()

    c.println(f"  {B}{BCY}CacheWarmup{R}  {DIM}Predictive KV cache pre-warming{R}")
    _tick(c, "method", "access-count × recency", "top-k prefix candidates")
    _tick(c, "hashing", "hashlib.md5", "deterministic prefix token fingerprint")
    _tick(c, "record_access()", "0.62 µs", "per-request access tracking")
    _tick(c, "get_warmup_candidates()", "19.6 µs", "top-k by score (cold TTFT ↓)")
    c.println()

    c.println(f"  {B}{BCY}TokenBudgetGate{R}  {DIM}Hard per-request token budget enforcement{R}")
    _tick(c, "control", "tick(n) → bool", "returns False when budget exhausted")
    _tick(c, "warning", "configurable fraction", "warns at X% of budget remaining")
    _tick(c, "tick()", "0.30 µs", "deterministic cost control overhead")
    c.pause(1.2)


# ── Scene 7: Wave 22 — Observability & Resilience ────────────────────────────

def scene_wave22_observability(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 22 ❷  Observability & Resilience",
             "ObservabilityHook · RequestCoalesce · AdaptiveQuantize · HealthCheck",
             colour=BCY)

    c.println(f"  {B}{BCY}ObservabilityHook{R}  {DIM}Zero-overhead per-step inference tracing{R}")
    _tick(c, "format", "OpenTelemetry-compatible", "JSON-serialisable span export")
    _tick(c, "record() + finish()", "3.6 µs", "per span — timestamped start/end")
    c.println()

    c.println(f"  {B}{BCY}RequestCoalesce{R}  {DIM}Merge requests sharing long common prefixes{R}")
    _tick(c, "method", "LCP grouping", "token-by-token longest common prefix")
    _tick(c, "benefit", "shared prefill", "forward pass split at divergence point")
    _tick(c, "add() + coalesce()", "8.2 µs", "4-request buffer coalescing round-trip")
    c.println()

    c.println(f"  {B}{BCY}AdaptiveQuantize{R}  {DIM}Runtime precision switching under memory pressure{R}")
    _tick(c, "thresholds", "FP16 / INT8 / INT4", "configurable used/capacity ratios")
    _tick(c, "quantize() FP16 path", "3.0 µs", "pass-through")
    _tick(c, "quantize() INT8 path", "56.0 µs", "symmetric int8 with scale")
    _tick(c, "quantize() INT4 path", "59.1 µs", "uint8 offset by 7")
    c.println()

    c.println(f"  {B}{BCY}HealthCheck{R}  {DIM}Degradation-aware server health monitoring{R}")
    _tick(c, "metrics", "p50/p99 latency + error rate", "deque(maxlen=1000) rolling window")
    _tick(c, "overall_health()", "5.2 µs", "returns worst of latency/error states")
    _tick(c, "record_request()", "95.8 µs", "per-request latency + error logging")
    c.pause(1.2)


# ── Scene 8: Wave 22 — Infrastructure ────────────────────────────────────────

def scene_wave22_infrastructure(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 22 ❸  Infrastructure & Cost",
             "FaultTolerance · ModelPool · StreamingChunk · CostEstimator · SLAMonitor · ContextCache",
             colour=BCY)

    c.println(f"  {B}{BCY}FaultTolerance{R}  {DIM}Graceful OOM degradation policy{R}")
    _tick(c, "actions (ordered)", "evict_kv → disable_draft → reduce_batch", "progressive")
    _tick(c, "evaluate() high pressure", "0.50 µs", "all 3 actions triggered at 0.95")
    c.println()

    c.println(f"  {B}{BCY}ModelPool{R}  {DIM}Hot model pool with lazy-load + LRU eviction{R}")
    _tick(c, "capacity", "configurable", "LRU eviction on acquire when full")
    _tick(c, "acquire() + release()", "0.58 µs", "zero-reload latency for hot models")
    c.println()

    c.println(f"  {B}{BCY}StreamingChunk{R}  {DIM}Sub-token-latency chunked streaming{R}")
    _tick(c, "backpressure", "push() → bool", "False when buffer at max capacity")
    _tick(c, "stream() 64-token chunk", "3.2 µs", "first-chunk latency minimised")
    _tick(c, "stream() 16-token chunk", "1.3 µs", "sub-chunk granularity")
    c.println()

    c.println(f"  {B}{BCY}CostEstimator{R}  {DIM}Per-request compute cost estimation{R}")
    _tick(c, "model", "prefill + decode + KV·duration", "multi-factor billing formula")
    _tick(c, "estimate()", "1.1 µs", "supports priority queuing + billing")
    c.println()

    c.println(f"  {B}{BCY}SLAMonitor{R}  {DIM}Real-time SLA violation detection + escalation{R}")
    _tick(c, "severity", "warning → critical", "escalates at consecutive breach threshold")
    _tick(c, "record()", "0.26 µs", "per-request violation tracking")
    _tick(c, "check()", "41.3 µs", "violations + severity assessment")
    c.println()

    c.println(f"  {B}{BCY}ContextCache{R}  {DIM}Persistent cross-session context cache with TTL{R}")
    _tick(c, "hashing", "hashlib.md5 over int64 tokens", "deterministic cross-restart keys")
    _tick(c, "eviction", "oldest entry", "capacity-based with TTL expiry")
    _tick(c, "put() / get()", "5.4 / 1.9 µs", "hit_rate=100% on repeated context")
    c.pause(1.2)


# ── Scene 9: Full CLI Stack ───────────────────────────────────────────────────

def scene_cli_stack(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Full v7 Stack — CLI Examples",
             "All 28 new flags live in squish serve", colour=ORG)

    c.println(f"  {DIM}# v7 advanced decode + memory stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.3,
    )
    for flag in [
        "      --tree-verify --kv-compress \\",
        "      --dynamic-ntk \\",
        "      --quant-spec-decode \\",
        "      --sparse-attn-index --mp-kv \\",
        "      --pipeline-bubble --layerwise-decode \\",
        "      --codec-kv --dedupe-attn \\",
        "      --flash-prefill --budget-spec \\",
        "      --retention-attn --kv-router",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Model loaded with v7 advanced decode optimisations{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Tree verification  ·  INT4 draft  ·  204× KV codec{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}Per-head mixed precision  ·  RetNet recurrent attn{R}", dt=0.2)
    c.println()

    c.println(f"  {DIM}# v7 production serving + observability stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.4,
    )
    for flag in [
        "      --multi-tenant --request-router \\",
        "      --cache-warmup --token-budget \\",
        "      --observability --req-coalesce \\",
        "      --adaptive-quant --health-check \\",
        "      --fault-tolerance --model-pool \\",
        "      --streaming-chunk --cost-estimate \\",
        "      --sla-monitor --context-cache",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Production serving layer online{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Multi-tenant QoS  ·  OTel tracing  ·  SLA monitors{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}Fault tolerance  ·  cost estimation  ·  context cache{R}", dt=0.2)
    c.pause(1.5)


# ── Scene 10: Tests & Closing ─────────────────────────────────────────────────

def scene_tests_closing(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Test Suite — v7 Complete", "pytest tests/ -q", colour=BBL)

    c.typeout("  $ pytest tests/ -q", char_delay=0.03, initial_dt=0.3)
    c.println()

    test_lines = [
        ("test_wave21_server_wiring.py", "56 passed"),
        ("test_wave22_server_wiring.py", "56 passed"),
        ("test_wave19_server_wiring.py", "56 passed"),
        ("test_wave20_server_wiring.py", "56 passed"),
        ("test_wave17_server_wiring.py", "56 passed"),
        ("test_wave18_server_wiring.py", "56 passed"),
    ]
    for fname, result in test_lines:
        c.println(
            f"  {DIM}{fname:<46}{R} {BGN}{result}{R}",
            dt=0.18,
        )

    c.println()
    c.println(
        f"  {B}{BGN}4 390 passed{R}  {DIM}in 3.7s  ·  0 failed  ·  0 errors{R}",
        dt=0.4,
    )
    c.pause(0.8)

    # Closing banner
    c.println()
    c.hbar()
    c.println(f"  {B}{BBL}Squish v7.0{R}  {DIM}— Released 2026-03-12{R}", dt=0.05)
    c.hbar()
    c.println()
    rows = [
        ("Modules", "166 total (28 new in v7)"),
        ("Tests", "4 390 passing, 0 failures"),
        ("INT4 draft memory (QuantSpecDecode)", "4× reduction vs FP16"),
        ("KV compression (CodecKV)", "204.8× ratio"),
        ("Mixed-precision KV memory", "2–4× at iso-quality"),
        ("Retention attention memory", "O(1) per step"),
        ("Flash prefill memory", "O(seq × chunk) not O(seq²)"),
        ("Scheduler overhead (MultiTenantSched)", "0.65 µs per request"),
        ("Budget gate overhead (TokenBudgetGate)", "0.30 µs per tick"),
        ("Context cache hit latency", "1.9 µs (100% hit rate)"),
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
    scene_wave21_memory_decode(c)
    scene_wave21_attention(c)
    scene_wave21_codec(c)
    scene_wave21_summary(c)
    scene_wave22_scheduling(c)
    scene_wave22_observability(c)
    scene_wave22_infrastructure(c)
    scene_cli_stack(c)
    scene_tests_closing(c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="Squish v7 demo GIF generator")
    ap.add_argument("--out",       default="dev/demos/squish-v7-demo.gif")
    ap.add_argument("--cast",      default="dev/demos/squish-v7-demo.cast")
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
