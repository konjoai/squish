import { Dial } from "@konjoai/ui";
import type { HealthResponse } from "../lib/types";

export interface ThroughputCardProps {
  health: HealthResponse;
  liveTps?: number; // current run's tps if streaming
}

/**
 * Live throughput dial backed by /health.avg_tps (rolling avg over last 20
 * requests). When a stream is in flight, `liveTps` overrides the average.
 */
export function ThroughputCard({ health, liveTps }: ThroughputCardProps) {
  const tps = liveTps ?? health.avg_tps;
  const ttft = health.avg_ttft_s;

  const sev = tps >= 50 ? "ok" : tps >= 25 ? "info" : tps >= 10 ? "warn" : "high";

  return (
    <div className="glass-konjo rounded-konjo-lg p-5 grid sm:grid-cols-[auto_1fr] gap-5 items-center">
      <Dial
        value={tps}
        min={0}
        max={Math.max(80, Math.ceil(tps / 10) * 10)}
        unit="tok/s"
        label="Throughput"
        severity={sev}
        format={(v) => v.toFixed(1)}
        size={170}
        sublabel={liveTps != null ? "live" : "avg over last 20"}
      />
      <div className="space-y-3 min-w-0">
        <Stat
          label="time to first token"
          value={ttft > 0 ? `${(ttft * 1000).toFixed(0)} ms` : "—"}
          accent={ttft <= 0.2 ? "var(--color-konjo-good)" : ttft <= 0.5 ? "var(--color-konjo-warm)" : "var(--color-konjo-hot)"}
        />
        <Stat label="requests served" value={`${health.requests}`} />
        <Stat label="tokens generated" value={fmtCompact(health.tokens_gen)} />
        <Stat label="in-flight" value={`${health.inflight}`} accent={health.inflight > 0 ? "var(--color-konjo-accent)" : undefined} />
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-baseline gap-3 border-b border-konjo-line/40 pb-1.5 last:border-b-0">
      <span className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted flex-1 min-w-0 truncate">
        {label}
      </span>
      <span className="text-konjo-mono tabular-nums" style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)" }}>
        {value}
      </span>
    </div>
  );
}

function fmtCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}
