import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { AnimatedNumber } from "../components/AnimatedNumber";
import type { QualityModel, QualityReport } from "../lib/types";

export interface QualityMonitorProps {
  report: QualityReport;
  fromMock: boolean;
}

/**
 * Quality monitor — rolling-window P50/P95/P99 latency + TTFT percentiles and
 * error rate from GET /v1/quality. Polled alongside /health in App. Percentile
 * bars animate to their live values; the headline is the worst-case P99.
 */
export function QualityMonitor({ report, fromMock }: QualityMonitorProps) {
  const model = report.models[0] as QualityModel | undefined;

  return (
    <section className="space-y-3" id="quality">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Quality monitor
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Rolling {Math.round(report.window_seconds / 60)}-min window via{" "}
            <span className="text-konjo-mono">/v1/quality</span> ·{" "}
            <span className="text-konjo-fg">{report.total_requests}</span> requests ·{" "}
            <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      {!model ? (
        <div className="glass-konjo rounded-konjo-lg p-8 text-center text-konjo-fg-muted text-konjo-mono text-[13px]">
          no requests recorded in this window yet
        </div>
      ) : (
        <div className="glass-konjo rounded-konjo-lg p-5 grid lg:grid-cols-2 gap-x-8 gap-y-5">
          <PercentileGroup
            title="latency"
            unit="ms"
            p50={model.latency_p50_ms}
            p95={model.latency_p95_ms}
            p99={model.latency_p99_ms}
            mean={model.latency_mean_ms}
          />
          <PercentileGroup
            title="time to first token"
            unit="ms"
            p50={model.ttft_p50_ms}
            p95={model.ttft_p95_ms}
            mean={null}
          />

          <div className="grid grid-cols-3 gap-2 lg:col-span-2 pt-1">
            <KpiCard
              label="error rate"
              value={model.error_rate * 100}
              format={(v) => `${v.toFixed(2)}%`}
              accent={model.error_rate > 0.05 ? "var(--color-konjo-hot)" : model.error_rate > 0.01 ? "var(--color-konjo-warm)" : "var(--color-konjo-good)"}
              sub={`${model.n_errors} / ${model.n_requests}`}
            />
            <KpiCard
              label="tok/s · p50"
              value={model.tokens_per_sec_p50}
              format={(v) => v.toFixed(1)}
              accent="var(--color-konjo-accent)"
              sub={`mean ${model.tokens_per_sec_mean.toFixed(1)}`}
            />
            <KpiCard
              label="ttft · p95"
              value={model.ttft_p95_ms}
              format={(v) => `${Math.round(v)}ms`}
              accent="var(--color-konjo-violet)"
              sub="95th percentile"
            />
          </div>
        </div>
      )}
    </section>
  );
}

function PercentileGroup({
  title, unit, p50, p95, p99, mean,
}: { title: string; unit: string; p50: number; p95: number; p99?: number; mean: number | null }) {
  const bars = [
    { label: "p50", value: p50, color: "var(--color-konjo-good)" },
    { label: "p95", value: p95, color: "var(--color-konjo-warm)" },
    ...(p99 != null ? [{ label: "p99", value: p99, color: "var(--color-konjo-hot)" }] : []),
  ];
  const max = Math.max(1, ...bars.map((b) => b.value));
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted">{title}</span>
        {mean != null && (
          <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-faint">mean {Math.round(mean)}{unit}</span>
        )}
      </div>
      {bars.map((b) => (
        <div key={b.label} className="flex items-center gap-2">
          <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted" style={{ width: 26 }}>{b.label}</span>
          <div className="flex-1 h-2.5 rounded-full bg-konjo-line/50 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(b.value / max) * 100}%` }}
              transition={{ duration: 0.6, ease: ease.kanjo }}
              className="h-full rounded-full"
              style={{ background: b.color, boxShadow: `0 0 8px ${b.color}` }}
            />
          </div>
          <span className="text-konjo-mono text-[11px] tabular-nums text-right" style={{ width: 64, color: b.color }}>
            <AnimatedNumber value={b.value} format={(v) => `${Math.round(v)}${unit}`} />
          </span>
        </div>
      ))}
    </div>
  );
}

function KpiCard({
  label, value, format, accent, sub,
}: { label: string; value: number; format: (v: number) => string; accent: string; sub: string }) {
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-3 py-2.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div className="text-konjo-display tabular-nums mt-1 leading-none" style={{ fontSize: 24, fontWeight: 600, color: accent }}>
        <AnimatedNumber value={value} format={format} />
      </div>
      <div className="text-konjo-mono text-[9px] text-konjo-fg-faint mt-1">{sub}</div>
    </div>
  );
}
