/**
 * Mock fixtures for offline development.
 */
import type { HealthResponse, CompressBenchResult, ChatTurn, StreamedToken } from "./types";

export const MOCK_HEALTH: HealthResponse = {
  status: "ok",
  model: "qwen3:8b-q4",
  loaded: true,
  loader: "mlx_lm",
  load_time_s: 8.42,
  requests: 156,
  tokens_gen: 18924,
  inflight: 0,
  avg_tps: 48.7,
  avg_ttft_s: 0.142,
  uptime_s: 3847.2,
  power_mode: "balanced",
  battery_level: 78.5,
  mem_available_gb: 12.4,
  mem_pressure: 32,
};

export const MOCK_PROM_TEXT = `# HELP squish_requests_total Total inference requests served
# TYPE squish_requests_total counter
squish_requests_total 156

# HELP squish_tokens_generated_total Total tokens generated
# TYPE squish_tokens_generated_total counter
squish_tokens_generated_total 18924

# HELP squish_inflight_requests Current in-flight requests
# TYPE squish_inflight_requests gauge
squish_inflight_requests 0

# HELP squish_avg_tokens_per_second Rolling average tokens/sec
# TYPE squish_avg_tokens_per_second gauge
squish_avg_tokens_per_second 48.67

# HELP squish_avg_ttft_seconds Rolling average time-to-first-token
# TYPE squish_avg_ttft_seconds gauge
squish_avg_ttft_seconds 0.1420

# HELP squish_uptime_seconds Server uptime
# TYPE squish_uptime_seconds counter
squish_uptime_seconds 3847.2

# HELP squish_model_load_seconds Time taken to load the model
# TYPE squish_model_load_seconds gauge
squish_model_load_seconds 8.420

# HELP squish_prefix_cache_hits_total Prefix cache hits
# TYPE squish_prefix_cache_hits_total counter
squish_prefix_cache_hits_total 128

# HELP squish_prefix_cache_size Prefix cache entries
# TYPE squish_prefix_cache_size gauge
squish_prefix_cache_size 5

# HELP squish_radix_prefix_hits_total RadixTree KV reuse
# TYPE squish_radix_prefix_hits_total counter
squish_radix_prefix_hits_total 342

# HELP squish_paged_kv_free_blocks Paged KV free blocks
# TYPE squish_paged_kv_free_blocks gauge
squish_paged_kv_free_blocks 1024

# HELP squish_paged_kv_used_blocks Paged KV used blocks
# TYPE squish_paged_kv_used_blocks gauge
squish_paged_kv_used_blocks 256

# HELP squish_spec_draft_loaded Speculative draft model loaded
# TYPE squish_spec_draft_loaded gauge
squish_spec_draft_loaded 1

# HELP squish_kv_cache_tokens Current KV cache token count
# TYPE squish_kv_cache_tokens gauge
squish_kv_cache_tokens 8192

# HELP squish_kv_cache_memory_mb KV cache memory MB
# TYPE squish_kv_cache_memory_mb gauge
squish_kv_cache_memory_mb 64.50
`;

const SAMPLE_RESPONSE = `Speculative decoding accelerates inference by having a small "draft" model propose several tokens in parallel, then verifying them in a single forward pass of the larger target model. Accepted drafts are committed; rejected ones force a fallback to the target's own next token.`;

export function buildMockChatStream(): { tokens: StreamedToken[]; turn: ChatTurn } {
  const words = SAMPLE_RESPONSE.split(/(?=\s)/); // keep leading whitespace per token
  let cum = 0;
  let cumS = 0;
  const tokens: StreamedToken[] = words.map((text, i) => {
    const noise = Math.sin(i * 0.6) * 4 + Math.cos(i * 1.3) * 2.5;
    const intervalMs = i === 0 ? 142 + noise * 0.4 : 18 + noise + (Math.random() < 0.05 ? 22 : 0);
    cum += intervalMs;
    cumS = cum / 1000;
    const tps = (i + 1) / Math.max(0.001, cumS);
    return { text, intervalMs, tps, atMs: cum };
  });
  const turn: ChatTurn = {
    id: `mock-${Date.now()}`,
    role: "assistant",
    content: tokens.map((t) => t.text).join(""),
    tokens,
    ttftS: tokens[0]?.intervalMs / 1000,
    totalS: cum / 1000,
    finishReason: "stop",
    fromMock: true,
  };
  return { tokens, turn };
}

export function buildMockBenchmark(ctx_len = 2048): CompressBenchResult {
  const head_dim = 128;
  const n_heads = 32;
  const fp16 = ctx_len * head_dim * n_heads * 2;
  return {
    ctx_len, head_dim, n_heads,
    fp16_baseline_bytes: fp16,
    live: false,
    results: [
      { mode: "int8", snr_db: 42.1, memory_bytes: fp16 / 2, compression_ratio: 2.0,  elapsed_ms: 89.2 },
      { mode: "int4", snr_db: 38.9, memory_bytes: fp16 / 4, compression_ratio: 4.0,  elapsed_ms: 92.4 },
      { mode: "int2", snr_db: 31.2, memory_bytes: fp16 / 8, compression_ratio: 8.0,  elapsed_ms: 101.3 },
    ],
  };
}
