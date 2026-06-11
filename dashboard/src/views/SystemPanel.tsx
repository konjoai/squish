import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { AnimatedNumber } from "../components/AnimatedNumber";
import type { SysStats, ModelStatus } from "../lib/types";

export interface SystemPanelProps {
  stats: SysStats;
  status: ModelStatus;
  fromMock: boolean;
}

const LOAD_MODE_LABEL: Record<ModelStatus["load_mode"], string> = {
  eager: "eager",
  lazy: "lazy",
  preload_async: "preload · async",
};

/**
 * System panel — host + runtime telemetry from GET /sys-stats and
 * GET /model/status, polled every 5s in App. Stdlib-only metrics (load avg,
 * process RSS, disk) plus the server's load mode and live load state.
 */
export function SystemPanel({ stats, status, fromMock }: SystemPanelProps) {
  const [l1, l5, l15] = stats.load_avg;

  return (
    <section className="space-y-3" id="system">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            System &amp; runtime
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Host metrics via <span className="text-konjo-mono">/sys-stats</span> + load state via{" "}
            <span className="text-konjo-mono">/model/status</span> · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <LoadAvgCard l1={l1} l5={l5} l15={l15} />

        <Tile label="process RSS" accent="var(--color-konjo-violet)">
          <AnimatedNumber value={stats.process_rss_mb} format={fmtMb} />
        </Tile>

        <DiskTile usedPct={stats.disk_used_pct} freeGb={stats.disk_free_gb} totalGb={stats.disk_total_gb} />

        <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-3 py-2.5 flex flex-col gap-1">
          <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">load mode</div>
          <div className="text-konjo-display uppercase tracking-[0.14em]" style={{ fontSize: 14, fontWeight: 700, color: "var(--color-konjo-accent)" }}>
            {LOAD_MODE_LABEL[status.load_mode]}
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            <span
              className="inline-block rounded-full"
              style={{
                width: 6, height: 6,
                background: status.load_error ? "var(--color-konjo-hot)" : status.model_loaded ? "var(--color-konjo-good)" : "var(--color-konjo-warm)",
                boxShadow: `0 0 8px ${status.load_error ? "var(--color-konjo-hot)" : status.model_loaded ? "var(--color-konjo-good)" : "var(--color-konjo-warm)"}`,
              }}
            />
            <span className="text-konjo-mono text-[10px] text-konjo-fg-muted">
              {status.load_error ? "load error" : status.model_loaded ? `loaded · ${status.load_time_s.toFixed(1)}s` : "loading…"}
            </span>
          </div>
          {status.model && (
            <div className="text-konjo-mono text-[10px] text-konjo-fg-faint truncate" title={status.model}>{status.model}</div>
          )}
        </div>
      </div>
    </section>
  );
}

function LoadAvgCard({ l1, l5, l15 }: { l1: number; l5: number; l15: number }) {
  const max = Math.max(1, l1, l5, l15);
  const bars = [
    { label: "1m", v: l1 },
    { label: "5m", v: l5 },
    { label: "15m", v: l15 },
  ];
  const sev = l1 >= 8 ? "var(--color-konjo-hot)" : l1 >= 4 ? "var(--color-konjo-warm)" : "var(--color-konjo-good)";
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-3 py-2.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">cpu load avg</div>
      <div className="text-konjo-display tabular-nums mt-1 leading-none" style={{ fontSize: 24, fontWeight: 600, color: sev }}>
        <AnimatedNumber value={l1} format={(v) => v.toFixed(2)} />
      </div>
      <div className="flex items-end gap-2 mt-2 h-8">
        {bars.map((b) => (
          <div key={b.label} className="flex-1 flex flex-col items-center gap-1">
            <div className="w-full bg-konjo-line/40 rounded-sm flex items-end" style={{ height: 20 }}>
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: `${(b.v / max) * 100}%` }}
                transition={{ duration: 0.5, ease: ease.kanjo }}
                className="w-full rounded-sm"
                style={{ background: sev }}
              />
            </div>
            <span className="text-konjo-mono text-[8px] text-konjo-fg-faint">{b.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DiskTile({ usedPct, freeGb, totalGb }: { usedPct: number; freeGb: number; totalGb: number }) {
  const c = usedPct >= 90 ? "var(--color-konjo-hot)" : usedPct >= 75 ? "var(--color-konjo-warm)" : "var(--color-konjo-good)";
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-3 py-2.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">disk</div>
      <div className="text-konjo-display tabular-nums mt-1 leading-none" style={{ fontSize: 24, fontWeight: 600, color: c }}>
        <AnimatedNumber value={usedPct} format={(v) => `${v.toFixed(1)}%`} />
      </div>
      <div className="h-1.5 rounded-full bg-konjo-line/50 overflow-hidden mt-2">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${usedPct}%` }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
          className="h-full rounded-full"
          style={{ background: c, boxShadow: `0 0 6px ${c}` }}
        />
      </div>
      <div className="text-konjo-mono text-[9px] text-konjo-fg-faint mt-1 tabular-nums">
        {freeGb.toFixed(0)} GB free of {totalGb.toFixed(0)} GB
      </div>
    </div>
  );
}

function Tile({ label, accent, children }: { label: string; accent: string; children: React.ReactNode }) {
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-3 py-2.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div className="text-konjo-display tabular-nums mt-1 leading-none" style={{ fontSize: 24, fontWeight: 600, color: accent }}>
        {children}
      </div>
    </div>
  );
}

function fmtMb(v: number): string {
  return v >= 1024 ? `${(v / 1024).toFixed(2)} GB` : `${Math.round(v)} MB`;
}
