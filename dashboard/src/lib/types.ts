/**
 * TypeScript types mirroring squish's API contract.
 * Source of truth lives in squish/server.py.
 */

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
}

export interface ChatChunkChoice {
  index: 0;
  delta: { role?: ChatRole; content?: string };
  finish_reason: null | "stop" | "length" | "tool_calls" | "error";
}

export interface ChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  system_fingerprint?: string;
  choices: ChatChunkChoice[];
}

/** GET /health response. */
export interface HealthResponse {
  status: "ok" | "no_model";
  model: string | null;
  loaded: boolean;
  loader: string;
  load_time_s: number;
  requests: number;
  tokens_gen: number;
  inflight: number;
  avg_tps: number;
  avg_ttft_s: number;
  uptime_s: number;
  power_mode: "performance" | "balanced" | "battery" | "auto";
  battery_level: number | null;
  mem_available_gb: number | null;
  mem_pressure: number | null;
}

/** Per-token timing tracked client-side from the SSE stream. */
export interface StreamedToken {
  text: string;
  /** Time since previous token. First token = ttft. */
  intervalMs: number;
  /** Cumulative tokens-per-second at the moment this token arrived. */
  tps: number;
  /** Wall-clock arrival time (performance.now ms since stream start). */
  atMs: number;
}

export interface ChatTurn {
  id: string;
  role: ChatRole;
  content: string;
  /** For assistant turns: per-token timing data. */
  tokens?: StreamedToken[];
  /** TTFT in seconds. */
  ttftS?: number;
  /** Total wall-clock seconds. */
  totalS?: number;
  finishReason?: string;
  fromMock?: boolean;
  /**
   * For agentic assistant turns (agent mode): the live tool-execution
   * timeline streamed from POST /v1/agent/run. Rendered inline in the chat.
   */
  steps?: AgentStep[];
}

export type StreamState = "idle" | "streaming" | "done" | "error";

/** KV cache modes — read from /health.loader heuristic. */
export type KVMode = "fp16" | "int8" | "int4" | "int3" | "int2" | "snap" | "unknown";

/** Compress benchmark result from the demo server. */
export interface CompressBenchEntry {
  mode: "int8" | "int4" | "int2";
  snr_db: number;
  memory_bytes: number;
  compression_ratio: number;
  elapsed_ms: number;
}

export interface CompressBenchResult {
  ctx_len: number;
  head_dim: number;
  n_heads: number;
  fp16_baseline_bytes: number;
  results: CompressBenchEntry[];
  live: boolean;
}

// ── Agent tool execution (/v1/agent/tools + /v1/agent/run) ──────────────────
/** A single tool exposed by the agent registry (OpenAI tools schema element). */
export interface AgentTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties?: Record<string, { type?: string; description?: string }>;
      required?: string[];
    };
  };
}

/** SSE event types emitted by POST /v1/agent/run. */
export type AgentEvent =
  | { type: "text_delta"; delta: string }
  | { type: "tool_call_start"; call_id: string; tool_name: string; arguments: Record<string, unknown> }
  | {
      type: "tool_call_result";
      call_id: string;
      tool_name: string;
      result: string;
      error: string | null;
      elapsed_ms: number;
    }
  | { type: "step_complete"; step: number }
  | { type: "done"; total_steps: number; total_tool_calls: number }
  | { type: "error"; message: string };

/** A tool call as rendered in the agent timeline (start + result merged). */
export interface AgentToolCall {
  callId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  result?: string;
  error?: string | null;
  elapsedMs?: number;
  done: boolean;
}

/** A single step of an agent run: model thinking text + the tool calls it made. */
export interface AgentStep {
  step: number;
  text: string;
  calls: AgentToolCall[];
  complete: boolean;
}

// ── Tokenizer (/v1/tokenize) ────────────────────────────────────────────────

/** Response from POST /v1/tokenize. */
export interface TokenizeResult {
  token_ids: number[];
  token_count: number;
  model: string;
}

// ── Quality monitor (/v1/quality) ───────────────────────────────────────────

/** Per-model rolling-window quality stats. */
export interface QualityModel {
  model_id: string;
  window_seconds: number;
  n_requests: number;
  n_errors: number;
  error_rate: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  latency_mean_ms: number;
  tokens_per_sec_p50: number;
  tokens_per_sec_mean: number;
  ttft_p50_ms: number;
  ttft_p95_ms: number;
  generated_at: number;
}

/** GET /v1/quality response. */
export interface QualityReport {
  window_seconds: number;
  total_requests: number;
  models: QualityModel[];
  generated_at: number;
}

/** Aggregate cockpit metrics from /v1/metrics. */
export interface CockpitMetrics {
  requests_total: number;
  tokens_total: number;
  inflight: number;
  avg_tps: number;
  avg_ttft_s: number;
  uptime_seconds: number;
  model_load_seconds: number;
  prefix_cache_hits: number;
  prefix_cache_size: number;
  radix_prefix_hits: number;
  paged_kv_free_blocks: number;
  paged_kv_used_blocks: number;
  spec_draft_loaded: boolean;
  kv_cache_tokens: number;
  kv_cache_memory_mb: number;
}

// ── Embeddings (/v1/embeddings) ───────────────────────────────────────────────

export interface EmbeddingData {
  object: "embedding";
  embedding: number[];
  index: number;
}

/** POST /v1/embeddings response (OpenAI-compatible). */
export interface EmbeddingResponse {
  object: "list";
  model: string;
  data: EmbeddingData[];
  usage: { prompt_tokens: number; total_tokens: number };
}

// ── System telemetry (/sys-stats + /model/status) ────────────────────────────

/** GET /sys-stats response (stdlib-only host metrics). */
export interface SysStats {
  load_avg: [number, number, number];
  process_rss_mb: number;
  disk_used_pct: number;
  disk_free_gb: number;
  disk_total_gb: number;
  pid: number;
}

/** GET /model/status response (lightweight load-state probe). */
export interface ModelStatus {
  load_mode: "eager" | "lazy" | "preload_async";
  model_loaded: boolean;
  model: string | null;
  load_time_s: number;
  load_error: string | null;
}

// ── Observability / APM (/v1/obs-report) ──────────────────────────────────────

/** Per-operation latency statistics from the production profiler. */
export interface OpStats {
  n_samples: number;
  mean_ms: number;
  p50_ms: number;
  p99_ms: number;
  p999_ms: number;
  min_ms: number;
  max_ms: number;
}

/** A slow operation flagged above the p99 threshold. */
export interface Bottleneck {
  op: string;
  p99_ms: number;
  mean_ms: number;
  n_samples: number;
  hint: string;
}

/** A single trace span (from squish.telemetry). */
export interface TraceSpan {
  id: string;
  parent_id: string | null;
  name: string;
  start_ms: number;
  end_ms: number | null;
  duration_ms: number | null;
  status: string;
  error_type: string | null;
  error_message: string | null;
}

/** GET /v1/obs-report response. */
export interface ObsReport {
  status: "ok" | "degraded" | "unavailable";
  bottlenecks: Bottleneck[];
  profile: Record<string, OpStats>;
  profiler_ops: string[];
  recent_spans: TraceSpan[];
}

// ── Startup profile (/v1/startup-profile) ─────────────────────────────────────

/** A single timed startup phase. */
export interface StartupEntry {
  phase: string;
  label: string;
  elapsed_ms: number;
}

/** GET /v1/startup-profile response (only populated when SQUISH_TRACE_STARTUP=1). */
export interface StartupProfile {
  enabled: boolean;
  total_ms?: number;
  phase_count?: number;
  entries?: StartupEntry[];
  slowest_5?: StartupEntry[];
  message?: string;
}
