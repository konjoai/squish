# squish · Inference Cockpit

A flagship Konjo UI for **squish** — the Apple Silicon-first local LLM server.

> 局所 (squish) — local · ぎゅっ — squeeze · 縮める — to compress

Real-time chat. KV cache live from `/v1/metrics`. Quantization comparator with side-by-side INT2/INT4/INT8 SNR. Latency waterfall (prefill vs decode). Apple Silicon power telemetry. Everything the SquishBar shows you in 13px, expanded into cinema.

## Quick start

```bash
npm install
npm run dev      # → http://localhost:5177
npm test         # vitest (25 tests)
npm run build    # production build → dist/
```

To wire to a live squish backend:

```bash
# Terminal 1 — start the FastAPI server (default :11435)
squish run qwen3:8b --port 11435 --api-key squish

# Terminal 2 — (optional) start the demo server for /api/benchmark on :8001
cd /Users/wscholl/squish && python demo/server.py

# Terminal 3 — start the dashboard
cd dashboard
VITE_SQUISH_API_KEY=squish npm run dev
```

When either server is unreachable the dashboard transparently falls back to mocks. The MetaInspector reports `live` vs `mock` for each pane independently.

## Stack

`React 19` · `TypeScript` · `Vite 8` · `Tailwind CSS v4` · `motion` · `Vitest`
Built on top of [`@konjoai/ui`](../../konjoai-ui).

## What you'll see

| Panel               | What it shows                                                          |
|---------------------|------------------------------------------------------------------------|
| **Hero**            | The squish promise · violet/cyan/green Konjo gradient                  |
| **ChatPanel**       | Conversation history; assistant tokens painted with per-token latency hue |
| **PromptBar**       | Send prompt with ⌘/ctrl-Enter · clear conversation                     |
| **ThroughputCard**  | Live tok/s dial + TTFT, requests served, in-flight count               |
| **KVCacheView**     | Live `/v1/metrics` aggregate · paged blocks, context usage, prefix cache hits |
| **QuantComparator** | Side-by-side INT8/INT4/INT2 SNR + memory + elapsed via `/api/benchmark` |
| **LatencyWaterfall**| Prefill (TTFT) vs decode for the most recent assistant turn            |
| **ThermalDial**     | Battery · memory pressure · power mode (from `/health`)                |
| **ModelInfo**       | Currently-loaded model · loader · KV mode · uptime                     |
| **MetaInspector**   | Source labels — server / chat / metrics / kv-bench (live vs mock)      |

## Architecture

- **Three transports.** `/v1/chat/completions` (SSE), `/health` + `/v1/metrics` (HTTP), `/api/benchmark` (demo server). The dashboard treats them as independent lanes; each falls back to mocks on its own.

- **Mock-first** ([src/lib/mock.ts](./src/lib/mock.ts)). Every transport has a hand-crafted fallback so the cockpit is always demo-able, even with no Apple Silicon or no model.

- **SSE + NDJSON parser** ([src/lib/sse.ts](./src/lib/sse.ts)). Auto-detects format, handles `[DONE]` sentinel, comment keepalives, and mid-frame buffer continuation.

- **Prometheus parser** ([src/lib/prom.ts](./src/lib/prom.ts)). Tiny exposition-format parser tailored for squish's specific output. `summarizeProm` rolls metrics into the cockpit shape.

## Honesty notes

- **KV-cache mode is CLI-only.** `--kv-cache-mode` at server startup; not runtime per-request. The dashboard reads the active mode from `/health.loader` (heuristic) and shows what it's running. The QuantComparator runs `/api/benchmark` to visualize *what other modes would cost*, but cannot change the server's mode. The current mode is highlighted with a `server` badge.

- **KV-cache visualization is aggregate.** squish's `/v1/metrics` exposes `paged_kv_used_blocks`, `paged_kv_free_blocks`, `kv_cache_tokens`, `kv_cache_memory_mb`. We render block grids whose totals match the observed used:free ratio — synthesized for visualization, but the totals are honest. The MetaInspector flags this as `kv-cache: aggregate`. Per-layer / per-position activity is not exposed by the backend today.

- **Latency waterfall observes wall-clock**, not internal stages. Prefill = TTFT, decode = sum of inter-token intervals. squish does not currently emit per-stage timing (tokenize / draft / verify / detokenize); the waterfall reports what's observable.

- **No model-swap endpoint.** `ModelInfo` is read-only. To swap models, restart squish.

- **Per-token latency is real.** Every dot, every hue, every breath of the throughput dial comes from observed inter-chunk timing on the SSE stream — no synthesis.

## Configuration

- `VITE_SQUISH_API`       — base URL of the FastAPI server (default `""`, leans on dev proxy)
- `VITE_SQUISH_DEMO_API`  — base URL of the demo server (default `""`)
- `VITE_SQUISH_API_KEY`   — bearer token for `/v1/*` and `/health` if `--api-key` set on server

The dev server proxies:
- `/v1/*`, `/health`, `/api/chat` → `http://localhost:11435`
- `/api/*` (benchmark/recommend/compress) → `http://localhost:8001`

## Tests

```bash
npm test
```

Covers: SSE/NDJSON frame splitting (incl. comment keepalives + mid-frame continuation), Prometheus parsing, mock fixture invariants (compression ratio + memory monotonicity), and behavioral tests for `<PromptBar>`, `<ChatPanel>`, `<ModelInfo>`. 25 tests, all green.

See [`CLAUDE.md`](./CLAUDE.md) for operating rules.
