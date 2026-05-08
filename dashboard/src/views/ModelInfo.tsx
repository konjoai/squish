import type { HealthResponse, KVMode } from "../lib/types";

export interface ModelInfoProps {
  health: HealthResponse;
  mode: KVMode;
}

function fmtUptime(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  if (s < 86400) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${h}h ${m}m`;
  }
  return `${Math.floor(s / 86400)}d`;
}

/**
 * The currently-loaded model card. Read-only — squish doesn't expose a
 * runtime model-swap endpoint today.
 */
export function ModelInfo({ health, mode }: ModelInfoProps) {
  const loaded = health.loaded && !!health.model;
  const c = loaded ? "var(--color-konjo-good)" : "var(--color-konjo-warm)";

  return (
    <section className="space-y-3">
      <header>
        <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
          Model
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Loaded at startup · CLI-only swap (no runtime endpoint)
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5">
        <div className="flex items-baseline gap-3">
          <span
            className="inline-block rounded-full"
            style={{ width: 8, height: 8, background: c, boxShadow: `0 0 8px ${c}` }}
          />
          <div
            className="text-konjo-display text-konjo-fg leading-none"
            style={{ fontSize: 22, fontWeight: 600 }}
          >
            {health.model ?? "no model loaded"}
          </div>
          <span
            className="ml-auto text-konjo-mono uppercase tracking-[0.18em] text-[10px]"
            style={{ color: c }}
          >
            {loaded ? "ready" : health.status}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <Stat label="loader"      value={health.loader} />
          <Stat label="kv mode"     value={mode} accent={KV_COLOR[mode]} />
          <Stat label="load time"   value={`${health.load_time_s.toFixed(2)}s`} />
          <Stat label="uptime"      value={fmtUptime(health.uptime_s)} />
        </div>
      </div>
    </section>
  );
}

const KV_COLOR: Record<KVMode, string> = {
  fp16:    "var(--color-konjo-fg)",
  int8:    "var(--color-mode-int8)",
  int4:    "var(--color-mode-int4)",
  int3:    "var(--color-mode-int3)",
  int2:    "var(--color-mode-int2)",
  snap:    "var(--color-konjo-good)",
  unknown: "var(--color-konjo-fg-muted)",
};

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-konjo bg-konjo-surface/60 border border-konjo-line/60 px-3 py-2">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div
        className="text-konjo-mono tabular-nums mt-1"
        style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)", textTransform: accent ? "uppercase" : "none", letterSpacing: accent ? "0.16em" : 0 }}
      >
        {value}
      </div>
    </div>
  );
}
