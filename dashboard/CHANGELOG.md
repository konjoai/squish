# Changelog

All notable changes to `@squish/dashboard` are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/).

## [0.1.0] — 2026-05-08

### Added — Sprint 4: Inference Cockpit

The flagship cinematic UI for squish. Real-time chat with per-token latency hue, live KV-cache aggregate from `/v1/metrics`, side-by-side quantization comparator, latency waterfall, Apple Silicon power telemetry — everything the SquishBar shows you in 13 px, expanded into cinema.

- **Repository scaffold** — Vite 8 + React 19 + TypeScript + Tailwind v4 + Vitest 4. Consumes `@konjoai/ui` via `file:../../konjoai-ui`. React and motion are deduped at the resolver to share one singleton.

- **Nine views**:
  - [`<ChatPanel>`](./src/views/ChatPanel.tsx) — conversation history. Assistant turns are painted with per-token latency hue (cool=fast, hot=slow); the TTFT chunk is highlighted violet.
  - [`<PromptBar>`](./src/views/PromptBar.tsx) — prompt textarea with ⌘/ctrl-Enter submission, clear-conversation button.
  - [`<ThroughputCard>`](./src/views/ThroughputCard.tsx) — live tok/s dial (overrides `/health.avg_tps` while streaming), TTFT, requests served, in-flight count.
  - [`<KVCacheView>`](./src/views/KVCacheView.tsx) — live `/v1/metrics` aggregate (paged blocks used vs free, context tokens cached, KV memory MB, prefix cache hits, RadixTree reuse, speculative draft loaded) + a synthesized block grid whose totals match the observed used:free ratio.
  - [`<QuantComparator>`](./src/views/QuantComparator.tsx) — side-by-side INT8/INT4/INT2 KV compression results from `/api/benchmark` with SNR (dB), memory, elapsed, and ratio bars. Highlights the server's currently-running mode.
  - [`<LatencyWaterfall>`](./src/views/LatencyWaterfall.tsx) — prefill (TTFT) vs decode summary bar + per-token interval strip for the most recent assistant turn.
  - [`<ThermalDial>`](./src/views/ThermalDial.tsx) — Apple Silicon power telemetry: memory pressure dial, battery dial (when exposed), power-mode card with free RAM.
  - [`<ModelInfo>`](./src/views/ModelInfo.tsx) — currently-loaded model · loader · KV mode · load time · uptime. Read-only.
  - [`<MetaInspector>`](./src/views/MetaInspector.tsx) — source labels for every pane: server, chat, metrics, kv-bench, kv-cache, status — each independently flagged `live` vs `mock`.

- **Library layer**:
  - [`types.ts`](./src/lib/types.ts) — TS mirrors of `/v1/chat/completions`, `/health`, `/v1/metrics`, `/api/benchmark`. KVMode union covers `fp16 · int8 · int4 · int3 · int2 · snap · unknown`.
  - [`sse.ts`](./src/lib/sse.ts) — SSE/NDJSON parser. Auto-detects, handles `[DONE]` sentinel, comment keepalives, mid-frame buffer continuation.
  - [`prom.ts`](./src/lib/prom.ts) — Prometheus exposition parser. `summarizeProm` rolls all squish metric names into the cockpit shape.
  - [`api.ts`](./src/lib/api.ts) — `chatStream` (cinematic) · `fetchHealth` · `fetchMetrics` · `benchmarkKV`. Each transparently falls back to mocks when the server is unreachable. Bearer-token auth via `VITE_SQUISH_API_KEY`.
  - [`mock.ts`](./src/lib/mock.ts) — `MOCK_HEALTH`, `MOCK_PROM_TEXT`, `buildMockChatStream` (24-token canned response with realistic latency noise), `buildMockBenchmark` (compression ratios + SNR).

- **Honest visualization**:
  - **KV-cache mode is read-only.** squish chooses `--kv-cache-mode` at startup; runtime per-request switching isn't supported. QuantComparator visualizes alternatives without claiming control.
  - **KV-cache visualization is aggregate.** Block grid totals match `paged_kv_used_blocks` / `paged_kv_free_blocks` exactly; per-layer / per-position structure is synthesized for readability. The MetaInspector flags this as `kv-cache: aggregate`.
  - **Latency waterfall observes wall-clock**, not internal stage timing. Prefill = TTFT; decode = sum of inter-token intervals. squish does not emit per-stage timing today.
  - **Per-token latency is real.** Every hue, every dot, every breath of the dial comes from observed inter-chunk timing on the SSE stream.
  - **No model-swap endpoint.** `<ModelInfo>` is read-only.

- **Tests** — 25 Vitest cases covering: SSE/NDJSON frame splitting (incl. comment keepalives + mid-frame), Prometheus parsing, mock-fixture invariants (compression ratio + memory monotonicity), and behavioral tests for `<PromptBar>`, `<ChatPanel>`, `<ModelInfo>`. All green.

- **Docs** — README, CLAUDE.md (operating rules), this changelog.

### Notes

- Sprint 4 of the 10-sprint Konjo UI Initiative.
- All animation respects `prefers-reduced-motion`.
- Three transports, two ports: `/v1/*` /health → squish FastAPI (11435); `/api/benchmark` → demo stdlib server (8001). Vite dev proxy routes them transparently.
