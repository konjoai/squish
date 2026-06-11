import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { ObsReport, OpStats, TraceSpan } from "../lib/types";

export interface ObservabilityPanelProps {
  report: ObsReport;
  fromMock: boolean;
}

/**
 * Observability panel — live APM view from GET /v1/obs-report. Renders:
 *   - a status badge (ok / degraded / unavailable)
 *   - per-operation latency rows with p50→p99 range bars
 *   - flagged bottlenecks with remediation hints
 *   - a recent-span Gantt timeline (when --trace is enabled)
 *
 * Honest about its data: per-op stats come from the production profiler; the
 * span timeline only populates when span tracing is on (--trace / SQUISH_TRACE=1),
 * otherwise the timeline shows a "tracing disabled" note.
 */
export function ObservabilityPanel({ report, fromMock }: ObservabilityPanelProps) {
  const ops = Object.entries(report.profile);
  const globalMax = Math.max(1, ...ops.map(([, s]) => s.p99_ms));
  const statusColor =
    report.status === "ok" ? "var(--color-konjo-good)" :
    report.status === "degraded" ? "var(--color-konjo-warm)" :
    "var(--color-konjo-fg-muted)";

  return (
    <section className="space-y-3" id="observability">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Observability
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            APM bottleneck report via <span className="text-konjo-mono">/v1/obs-report</span> ·{" "}
            <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
        <span
          className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] px-2.5 py-1 rounded-konjo-sm border"
          style={{ color: statusColor, borderColor: statusColor + "66", background: `color-mix(in oklch, ${statusColor} 8%, transparent)` }}
        >
          {report.status}
        </span>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 grid lg:grid-cols-2 gap-x-8 gap-y-5">
        <div className="space-y-3 min-w-0">
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted">
            per-operation latency · p50 → p99
          </div>
          {ops.length === 0 ? (
            <div className="text-konjo-fg-faint text-konjo-mono text-[12px] py-6">no profiler samples yet</div>
          ) : (
            ops.map(([op, stats]) => <OpRow key={op} op={op} stats={stats} max={globalMax} />)
          )}
        </div>

        <div className="space-y-4 min-w-0">
          <div>
            <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted mb-2">
              bottlenecks
            </div>
            {report.bottlenecks.length === 0 ? (
              <div className="text-konjo-good text-konjo-mono text-[12px] flex items-center gap-2">
                <span style={{ width: 6, height: 6, borderRadius: 99, background: "currentColor", boxShadow: "0 0 8px currentColor", display: "inline-block" }} />
                no operations over threshold
              </div>
            ) : (
              <div className="space-y-2">
                {report.bottlenecks.map((b) => (
                  <motion.div
                    key={b.op}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, ease: ease.kanjo }}
                    className="rounded-konjo border border-konjo-warm/40 bg-konjo-warm/5 px-3 py-2"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-konjo-mono text-[12px] text-konjo-warm truncate">{b.op}</span>
                      <span className="text-konjo-mono text-[11px] tabular-nums text-konjo-warm shrink-0">p99 {b.p99_ms.toFixed(0)}ms</span>
                    </div>
                    <div className="text-konjo-fg-muted text-[11px] mt-1" style={{ lineHeight: 1.45 }}>{b.hint}</div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <div>
            <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted mb-2">
              recent spans
            </div>
            <SpanTimeline spans={report.recent_spans} />
          </div>
        </div>
      </div>
    </section>
  );
}

function OpRow({ op, stats, max }: { op: string; stats: OpStats; max: number }) {
  const left = (stats.p50_ms / max) * 100;
  const width = Math.max(1.5, ((stats.p99_ms - stats.p50_ms) / max) * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-konjo-mono text-[11px] text-konjo-fg truncate">{op}</span>
        <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted shrink-0">
          {stats.n_samples.toLocaleString()}× · p50 {stats.p50_ms.toFixed(1)} · p99 {stats.p99_ms.toFixed(1)}ms
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-konjo-line/40 overflow-hidden">
        <motion.div
          initial={{ width: 0, left: 0 }}
          animate={{ width: `${width}%`, left: `${left}%` }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
          className="absolute top-0 h-full rounded-full"
          style={{ background: "var(--color-konjo-accent)", boxShadow: "0 0 6px var(--color-konjo-glow-accent)" }}
        />
      </div>
    </div>
  );
}

function SpanTimeline({ spans }: { spans: TraceSpan[] }) {
  if (spans.length === 0) {
    return (
      <div className="text-konjo-fg-faint text-konjo-mono text-[11px]" style={{ lineHeight: 1.5 }}>
        span tracing disabled — start with <span className="text-konjo-fg-muted">--trace</span> (or{" "}
        <span className="text-konjo-fg-muted">SQUISH_TRACE=1</span>) to populate the timeline.
      </div>
    );
  }
  const t0 = Math.min(...spans.map((s) => s.start_ms));
  const t1 = Math.max(...spans.map((s) => (s.end_ms ?? s.start_ms)));
  const span = Math.max(1, t1 - t0);
  return (
    <div className="space-y-1.5">
      {spans.map((s, i) => {
        const dur = s.duration_ms ?? 0;
        const left = ((s.start_ms - t0) / span) * 100;
        const width = Math.max(1, (dur / span) * 100);
        const c = s.status === "ok" ? "var(--color-konjo-violet)" : "var(--color-konjo-hot)";
        return (
          <div key={s.id} className="flex items-center gap-2" title={`${s.name} · ${dur.toFixed(1)}ms · ${s.status}`}>
            <span className="text-konjo-mono text-[9px] text-konjo-fg-muted truncate" style={{ width: 96 }}>{s.name}</span>
            <div className="relative flex-1 h-3.5 rounded-sm bg-konjo-line/30 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${width}%` }}
                transition={{ duration: 0.5, ease: ease.kanjo, delay: i * 0.04 }}
                className="absolute top-0 h-full rounded-sm"
                style={{ left: `${left}%`, background: c, boxShadow: `0 0 5px ${c}` }}
              />
            </div>
            <span className="text-konjo-mono text-[9px] tabular-nums text-konjo-fg-muted shrink-0" style={{ width: 52, textAlign: "right" }}>
              {dur.toFixed(0)}ms
            </span>
          </div>
        );
      })}
    </div>
  );
}
