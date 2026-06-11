# squish/dashboard

Inference Cockpit — flagship cinematic UI for squish. Vite + React + `@konjoai/ui`. Sprint 4 of the Konjo UI Initiative.

## Stack
React 19 · TypeScript · Vite 8 · Tailwind v4 (`@theme` config) · motion · Vitest 4 · `@konjoai/ui` (file: dep)

## Commands
```bash
npm install
npm run dev          # → http://localhost:5177 (proxies /v1 /health → :11435, /api → :8001)
npm test             # vitest (25 tests)
npm run build        # tsc -b && vite build
npm run typecheck    # tsc -b --noEmit
```

## Critical Constraints
- React, react-dom, and motion are deduped in [vite.config.ts](./vite.config.ts) so the dashboard and `@konjoai/ui` share a singleton.
- `@konjoai/ui` is consumed via `file:../../konjoai-ui`. Tokens come from `@konjoai/ui/styles` — don't redefine.
- Three transports, two ports: `/v1/*` + `/health` + `/api/chat` → squish FastAPI (11435); `/api/benchmark` etc → demo stdlib server (8001).
- **KV-cache mode is CLI-only** — server-startup decision. The dashboard reads it (via /health.loader heuristic) but cannot change it. QuantComparator only visualizes alternatives.
- **KV-cache visualization is aggregate** — totals are honest, per-layer / per-position structure is synthesized. MetaInspector flags this.
- **Latency waterfall reports wall-clock observation** — TTFT + decode sum, not internal stage timing (squish doesn't expose those today).
- All 25 tests + the build must stay green.

## File Map
| Path | Role |
|------|------|
| `src/App.tsx` | Composition + chat state machine + 5s health/metrics/quality polling + section nav |
| `src/views/ChatPanel.tsx` | Conversation with per-token latency hue on assistant turns |
| `src/views/PromptBar.tsx` | Prompt textarea + send + clear |
| `src/views/ThroughputCard.tsx` | tok/s dial + ttft + request stats |
| `src/views/AgentPlayground.tsx` | Live tool execution via /v1/agent/run SSE + /v1/agent/tools palette |
| `src/views/KVCacheView.tsx` | /v1/metrics aggregate + synthesized block grid |
| `src/views/QuantComparator.tsx` | INT8/INT4/INT2 side-by-side via /api/benchmark |
| `src/views/TokenizerLab.tsx` | Live token chips + IDs via /v1/tokenize (debounced) |
| `src/views/LatencyWaterfall.tsx` | Prefill vs decode + per-token bar strip |
| `src/views/QualityMonitor.tsx` | P50/P95/P99 latency + TTFT + error rate via /v1/quality |
| `src/views/ThermalDial.tsx` | Battery + mem pressure + power mode |
| `src/views/ModelInfo.tsx` | Loaded model card (read-only) |
| `src/views/MetaInspector.tsx` | Source labels for every pane (live vs mock) |
| `src/components/AnimatedNumber.tsx` | Count-up tween for live telemetry values |
| `src/components/SectionNav.tsx` | Sticky right-rail scroll-spy navigator |
| `src/lib/types.ts` | TS mirrors of chat + health + metrics + benchmark + agent + tokenize + quality |
| `src/lib/api.ts` | chatStream + fetchHealth + fetchMetrics + fetchQuality + benchmarkKV + agentRun + fetchAgentTools + tokenizeText |
| `src/lib/agent.ts` | Pure AgentEvent → AgentStep[] reducer (testable) |
| `src/lib/sse.ts` | SSE/NDJSON parser (auto-detect, keepalive-tolerant) |
| `src/lib/prom.ts` | Prometheus exposition parser + summarizeProm |
| `src/lib/mock.ts` | MOCK fixtures + buildMockChatStream/Benchmark/AgentRun/Tokenize |

## Backend integration
- `POST /v1/chat/completions` — SSE streaming. OpenAI-compatible `chat.completion.chunk` frames; `data: [DONE]` sentinel. Bearer auth via `Authorization: Bearer <key>` if `--api-key` set.
- `GET /health` — model, loader, requests, tokens_gen, inflight, avg_tps, avg_ttft_s, uptime_s, power_mode, battery_level, mem_available_gb, mem_pressure.
- `GET /v1/metrics` — Prometheus text format. Counters · gauges. Polled every 5s.
- `GET /v1/agent/tools` — built-in agent tool schemas (OpenAI tools array).
- `POST /v1/agent/run` — SSE agent loop. Events: text_delta · tool_call_start · tool_call_result · step_complete · done · error.
- `POST /v1/tokenize` — `{text}` → `{token_ids, token_count, model}`.
- `GET /v1/quality` — rolling-window P50/P95/P99 latency + TTFT + error rate per model. Polled every 5s.
- `POST /api/benchmark` (demo server) — INT8/INT4/INT2 KV-cache compression benchmark. Returns SNR, memory, elapsed.

## When extending
- New panel? Lives in `src/views/`. Always ship a Vitest test.
- New backend shape? Mirror types in [src/lib/types.ts](./src/lib/types.ts), add a mock fixture, then add the API method to [src/lib/api.ts](./src/lib/api.ts) with a mock fallback.
- Future backend lift: when squish exposes per-token KV-mode flags or per-stage latency breakdown, the dashboard's typed slots are ready — only the wire format changes.
- New design token? Add to `@konjoai/ui` (so all flagships inherit), not here.

## Sprint context
This is **Sprint 4** of the 10-sprint Konjo UI Initiative. Sprint 0 = `@konjoai/ui` foundation. Sprint 1 = squash Compliance Bridge. Sprint 2 = miru Mind of the Machine. Sprint 3 = kairu Speed Cockpit. Sprint 5 = kyro RAG Observatory (next).
