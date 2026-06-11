/**
 * squish dashboard API client. Three transports:
 *
 *   1. chatStream     — POST /v1/chat/completions (SSE) — the cinematic path
 *   2. fetchHealth    — GET /health (every 5s)
 *   3. fetchMetrics   — GET /v1/metrics (every 5s, optional)
 *   4. benchmarkKV    — POST /api/benchmark on the demo server
 *
 * All four transparently fall back to mock fixtures when the relevant server
 * is unreachable. The MetaInspector reports `live` vs `mock` for each pane.
 */
import type {
  ChatRequest,
  ChatChunk,
  HealthResponse,
  CompressBenchResult,
  StreamedToken,
  AgentEvent,
  AgentTool,
  TokenizeResult,
  QualityReport,
  EmbeddingResponse,
  SysStats,
  ModelStatus,
} from "./types";
import { parseStreamChunk } from "./sse";
import { parseAgentEvent } from "./agent";
import { summarizeProm } from "./prom";
import {
  MOCK_HEALTH,
  MOCK_PROM_TEXT,
  MOCK_AGENT_TOOLS,
  MOCK_QUALITY,
  MOCK_SYS_STATS,
  MOCK_MODEL_STATUS,
  buildMockChatStream,
  buildMockBenchmark,
  buildMockAgentRun,
  buildMockTokenize,
  buildMockEmbeddings,
} from "./mock";

const SQUISH_API = (import.meta.env.VITE_SQUISH_API as string | undefined) ?? "";
const DEMO_API   = (import.meta.env.VITE_SQUISH_DEMO_API as string | undefined) ?? "";
const API_KEY    = (import.meta.env.VITE_SQUISH_API_KEY as string | undefined) ?? "";

function authHeaders(): HeadersInit {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) h["Authorization"] = `Bearer ${API_KEY}`;
  return h;
}

export interface ChatStreamHandle {
  cancel: () => void;
  done: Promise<{
    text: string;
    tokens: StreamedToken[];
    finishReason?: string;
    fromMock: boolean;
  }>;
}

/**
 * Stream tokens from /v1/chat/completions. Caller subscribes via `onToken`.
 * Falls back to a mock replay if the server is unreachable.
 */
export function chatStream(
  req: ChatRequest,
  onToken: (t: StreamedToken, opts: { fromMock: boolean }) => void,
): ChatStreamHandle {
  const ctrl = new AbortController();
  let cancelled = false;
  const tokens: StreamedToken[] = [];
  let textOut = "";
  let finishReason: string | undefined;

  const done = (async () => {
    try {
      const startedAt = performance.now();
      let lastAt = startedAt;

      const res = await fetch(SQUISH_API + "/v1/chat/completions", {
        method: "POST",
        headers: { ...authHeaders(), Accept: "text/event-stream" },
        body: JSON.stringify({ ...req, stream: true }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`http ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (!cancelled) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        const { frames, rest } = parseStreamChunk(buf);
        buf = rest;
        for (const f of frames) {
          if (f.done) continue;
          const obj = parseChunk(f.json);
          if (!obj) continue;
          const choice = obj.choices?.[0];
          if (!choice) continue;
          const text = choice.delta?.content;
          if (text) {
            const now = performance.now();
            const interval = now - lastAt;
            lastAt = now;
            const tok: StreamedToken = {
              text,
              intervalMs: interval,
              tps: tokens.length === 0 ? 0 : (tokens.length + 1) / Math.max(0.001, (now - startedAt) / 1000),
              atMs: now - startedAt,
            };
            tokens.push(tok);
            textOut += text;
            onToken(tok, { fromMock: false });
          }
          if (choice.finish_reason) finishReason = choice.finish_reason;
        }
      }
      return { text: textOut, tokens, finishReason, fromMock: false };
    } catch {
      if (cancelled) return { text: textOut, tokens, fromMock: true };
      const mock = buildMockChatStream();
      let cum = 0;
      for (const t of mock.tokens) {
        if (cancelled) break;
        await sleep(Math.max(8, Math.min(140, t.intervalMs)));
        cum += t.intervalMs;
        const stamped: StreamedToken = { ...t, atMs: cum };
        tokens.push(stamped);
        textOut += stamped.text;
        onToken(stamped, { fromMock: true });
      }
      return { text: textOut, tokens, finishReason: "stop", fromMock: true };
    }
  })();

  return { cancel: () => { cancelled = true; ctrl.abort(); }, done };
}

function parseChunk(s: string): ChatChunk | null {
  try { return JSON.parse(s) as ChatChunk; } catch { return null; }
}

function sleep(ms: number): Promise<void> { return new Promise((r) => setTimeout(r, ms)); }

export async function fetchHealth(): Promise<{ data: HealthResponse; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/health", { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as HealthResponse;
    return { data, fromMock: false };
  } catch {
    return { data: MOCK_HEALTH, fromMock: true };
  }
}

export async function fetchMetrics(): Promise<{ raw: string; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/v1/metrics", { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const raw = await res.text();
    return { raw, fromMock: false };
  } catch {
    return { raw: MOCK_PROM_TEXT, fromMock: true };
  }
}

export function summarizeMetricsText(raw: string) { return summarizeProm(raw); }

// ── Agent tool execution ────────────────────────────────────────────────────

export async function fetchAgentTools(): Promise<{ tools: AgentTool[]; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/v1/agent/tools", { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const body = (await res.json()) as { tools: AgentTool[] };
    return { tools: body.tools ?? [], fromMock: false };
  } catch {
    return { tools: MOCK_AGENT_TOOLS, fromMock: true };
  }
}

export interface AgentRunHandle {
  cancel: () => void;
  done: Promise<{ fromMock: boolean }>;
}

/**
 * Run the multi-step agent loop over SSE. Each AgentEvent is delivered to
 * `onEvent`. Falls back to a scripted mock run if the server is unreachable.
 */
export function agentRun(
  body: { messages: { role: string; content: string }[]; max_steps?: number; max_tokens?: number; temperature?: number },
  onEvent: (ev: AgentEvent, opts: { fromMock: boolean }) => void,
): AgentRunHandle {
  const ctrl = new AbortController();
  let cancelled = false;

  const done = (async () => {
    try {
      const res = await fetch(SQUISH_API + "/v1/agent/run", {
        method: "POST",
        headers: { ...authHeaders(), Accept: "text/event-stream" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`http ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (!cancelled) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        const { frames, rest } = parseStreamChunk(buf);
        buf = rest;
        for (const f of frames) {
          if (f.done) continue;
          const ev = parseAgentEvent(f.json);
          if (ev) onEvent(ev, { fromMock: false });
        }
      }
      return { fromMock: false };
    } catch {
      if (cancelled) return { fromMock: true };
      for (const { ev, delayMs } of buildMockAgentRun(body.messages)) {
        if (cancelled) break;
        await sleep(delayMs);
        onEvent(ev, { fromMock: true });
      }
      return { fromMock: true };
    }
  })();

  return { cancel: () => { cancelled = true; ctrl.abort(); }, done };
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

export async function tokenizeText(
  text: string,
): Promise<{ data: TokenizeResult; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/v1/tokenize", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as TokenizeResult;
    return { data, fromMock: false };
  } catch {
    return { data: buildMockTokenize(text), fromMock: true };
  }
}

// ── Quality monitor ─────────────────────────────────────────────────────────

export async function fetchQuality(
  window = 3600,
): Promise<{ data: QualityReport; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + `/v1/quality?window=${window}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as QualityReport;
    return { data, fromMock: false };
  } catch {
    return { data: MOCK_QUALITY, fromMock: true };
  }
}

// ── Embeddings ────────────────────────────────────────────────────────────────

export async function embedText(
  inputs: string[],
): Promise<{ data: EmbeddingResponse; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/v1/embeddings", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ input: inputs }),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as EmbeddingResponse;
    return { data, fromMock: false };
  } catch {
    return { data: buildMockEmbeddings(inputs), fromMock: true };
  }
}

// ── System telemetry ──────────────────────────────────────────────────────────

export async function fetchSysStats(): Promise<{ data: SysStats; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/sys-stats", { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as SysStats;
    return { data, fromMock: false };
  } catch {
    return { data: MOCK_SYS_STATS, fromMock: true };
  }
}

export async function fetchModelStatus(): Promise<{ data: ModelStatus; fromMock: boolean }> {
  try {
    const res = await fetch(SQUISH_API + "/model/status", { headers: authHeaders() });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as ModelStatus;
    return { data, fromMock: false };
  } catch {
    return { data: MOCK_MODEL_STATUS, fromMock: true };
  }
}

export async function benchmarkKV(ctx_len: number): Promise<{ data: CompressBenchResult; fromMock: boolean }> {
  try {
    const res = await fetch(DEMO_API + "/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ctx_len }),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const data = (await res.json()) as CompressBenchResult;
    return { data, fromMock: false };
  } catch {
    return { data: buildMockBenchmark(ctx_len), fromMock: true };
  }
}
