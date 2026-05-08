/**
 * Tiny Prometheus exposition-format parser tailored for squish's /v1/metrics.
 */
import type { CockpitMetrics } from "./types";

interface Row {
  name: string;
  labels: Record<string, string>;
  value: number;
}

export function parseProm(text: string): Map<string, Row[]> {
  const out = new Map<string, Row[]>();
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const row = parseSample(line);
    if (!row) continue;
    const arr = out.get(row.name) ?? [];
    arr.push(row);
    out.set(row.name, arr);
  }
  return out;
}

function parseSample(line: string): Row | null {
  const idx = line.indexOf("{");
  let name: string;
  let rest: string;
  let labels: Record<string, string> = {};
  if (idx === -1) {
    const sp = line.indexOf(" ");
    if (sp === -1) return null;
    name = line.slice(0, sp);
    rest = line.slice(sp + 1).trim();
  } else {
    name = line.slice(0, idx);
    const close = line.indexOf("}", idx);
    if (close === -1) return null;
    labels = parseLabels(line.slice(idx + 1, close));
    rest = line.slice(close + 1).trim();
  }
  const v = Number(rest.split(/\s+/)[0]);
  if (!Number.isFinite(v)) return null;
  return { name, labels, value: v };
}

function parseLabels(s: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const pair of s.split(",")) {
    const eq = pair.indexOf("=");
    if (eq === -1) continue;
    const k = pair.slice(0, eq).trim();
    const v = pair.slice(eq + 1).trim().replace(/^"|"$/g, "");
    if (k) out[k] = v;
  }
  return out;
}

const FIRST = (rows: Row[] | undefined): number => rows?.[0]?.value ?? 0;

export function summarizeProm(text: string): CockpitMetrics {
  const m = parseProm(text);
  return {
    requests_total:        FIRST(m.get("squish_requests_total")),
    tokens_total:          FIRST(m.get("squish_tokens_generated_total")),
    inflight:              FIRST(m.get("squish_inflight_requests")),
    avg_tps:               FIRST(m.get("squish_avg_tokens_per_second")),
    avg_ttft_s:            FIRST(m.get("squish_avg_ttft_seconds")),
    uptime_seconds:        FIRST(m.get("squish_uptime_seconds")),
    model_load_seconds:    FIRST(m.get("squish_model_load_seconds")),
    prefix_cache_hits:     FIRST(m.get("squish_prefix_cache_hits_total")),
    prefix_cache_size:     FIRST(m.get("squish_prefix_cache_size")),
    radix_prefix_hits:     FIRST(m.get("squish_radix_prefix_hits_total")),
    paged_kv_free_blocks:  FIRST(m.get("squish_paged_kv_free_blocks")),
    paged_kv_used_blocks:  FIRST(m.get("squish_paged_kv_used_blocks")),
    spec_draft_loaded:     FIRST(m.get("squish_spec_draft_loaded")) > 0,
    kv_cache_tokens:       FIRST(m.get("squish_kv_cache_tokens")),
    kv_cache_memory_mb:    FIRST(m.get("squish_kv_cache_memory_mb")),
  };
}
