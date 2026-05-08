/**
 * TypeScript types mirroring squish's API contract.
 * Source of truth lives in /Users/wesleyscholl/squish/squish/server.py.
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
