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
| `src/App.tsx` | Composition + chat state machine + 5s health/metrics polling |
| `src/views/ChatPanel.tsx` | Conversation with per-token latency hue on assistant turns |
| `src/views/PromptBar.tsx` | Prompt textarea + send + clear |
| `src/views/ThroughputCard.tsx` | tok/s dial + ttft + request stats |
| `src/views/KVCacheView.tsx` | /v1/metrics aggregate + synthesized block grid |
| `src/views/QuantComparator.tsx` | INT8/INT4/INT2 side-by-side via /api/benchmark |
| `src/views/LatencyWaterfall.tsx` | Prefill vs decode + per-token bar strip |
| `src/views/ThermalDial.tsx` | Battery + mem pressure + power mode |
| `src/views/ModelInfo.tsx` | Loaded model card (read-only) |
| `src/views/MetaInspector.tsx` | Source labels for every pane |
| `src/lib/types.ts` | TS mirrors of /v1/chat/completions + /health + /v1/metrics + /api/benchmark |
| `src/lib/api.ts` | chatStream + fetchHealth + fetchMetrics + benchmarkKV |
| `src/lib/sse.ts` | SSE/NDJSON parser (auto-detect, keepalive-tolerant) |
| `src/lib/prom.ts` | Prometheus exposition parser + summarizeProm |
| `src/lib/mock.ts` | MOCK_HEALTH + MOCK_PROM_TEXT + buildMockChatStream + buildMockBenchmark |

## Backend integration
- `POST /v1/chat/completions` — SSE streaming. OpenAI-compatible `chat.completion.chunk` frames; `data: [DONE]` sentinel. Bearer auth via `Authorization: Bearer <key>` if `--api-key` set.
- `GET /health` — model, loader, requests, tokens_gen, inflight, avg_tps, avg_ttft_s, uptime_s, power_mode, battery_level, mem_available_gb, mem_pressure.
- `GET /v1/metrics` — Prometheus text format. Counters · gauges. Polled every 5s.
- `POST /api/benchmark` (demo server) — INT8/INT4/INT2 KV-cache compression benchmark. Returns SNR, memory, elapsed.

## When extending
- New panel? Lives in `src/views/`. Always ship a Vitest test.
- New backend shape? Mirror types in [src/lib/types.ts](./src/lib/types.ts), add a mock fixture, then add the API method to [src/lib/api.ts](./src/lib/api.ts) with a mock fallback.
- Future backend lift: when squish exposes per-token KV-mode flags or per-stage latency breakdown, the dashboard's typed slots are ready — only the wire format changes.
- New design token? Add to `@konjoai/ui` (so all flagships inherit), not here.

## Sprint context
This is **Sprint 4** of the 10-sprint Konjo UI Initiative. Sprint 0 = `@konjoai/ui` foundation. Sprint 1 = squash Compliance Bridge. Sprint 2 = miru Mind of the Machine. Sprint 3 = kairu Speed Cockpit. Sprint 5 = kyro RAG Observatory (next).
