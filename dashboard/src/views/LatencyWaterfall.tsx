import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { ChatTurn } from "../lib/types";

export interface LatencyWaterfallProps {
  turn: ChatTurn | null;
}

/**
 * A two-band waterfall for the most recent assistant turn:
 *   - Prefill (= TTFT, observed wall-clock)
 *   - Decode  (= sum of inter-token intervals after the first)
 *
 * Per-component prefill / decode / detokenize timing is NOT exposed by squish
 * today; we report what's observable and label it honestly.
 */
export function LatencyWaterfall({ turn }: LatencyWaterfallProps) {
  if (!turn || !turn.tokens || turn.tokens.length === 0) {
    return (
      <div className="glass-konjo rounded-konjo-lg p-5 flex items-center justify-center text-konjo-fg-muted" style={{ minHeight: 180 }}>
        <span className="text-konjo-mono text-[12px]">no streamed turn yet</span>
      </div>
    );
  }
  const tokens = turn.tokens;
  const ttftMs = tokens[0]?.intervalMs ?? 0;
  const decodeMs = tokens.slice(1).reduce((a, t) => a + t.intervalMs, 0);
  const totalMs = ttftMs + decodeMs;

  const ttftPct = totalMs > 0 ? (ttftMs / totalMs) * 100 : 0;
  const decodePct = totalMs > 0 ? (decodeMs / totalMs) * 100 : 0;

  // Per-token bar chart
  const max = Math.max(1, ...tokens.map((t) => t.intervalMs));

  return (
    <section className="space-y-3">
      <header>
        <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
          Latency waterfall
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Wall-clock observation · {tokens.length} tokens in {(totalMs / 1000).toFixed(2)}s
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        {/* Stacked summary bar */}
        <div className="space-y-2">
          <div className="flex h-3 rounded-full overflow-hidden border border-konjo-line/60">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${ttftPct}%` }}
              transition={{ duration: 0.6, ease: ease.kanjo }}
              style={{
                background: "var(--color-konjo-violet)",
                boxShadow: "0 0 8px var(--color-konjo-glow-violet)",
              }}
              title={`TTFT — ${ttftMs.toFixed(0)}ms`}
            />
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${decodePct}%` }}
              transition={{ duration: 0.6, ease: ease.kanjo, delay: 0.1 }}
              style={{
                background: "var(--color-konjo-accent)",
                boxShadow: "0 0 8px var(--color-konjo-glow-accent)",
              }}
              title={`decode — ${decodeMs.toFixed(0)}ms`}
            />
          </div>
          <div className="flex justify-between gap-4 text-konjo-mono text-[11px]">
            <Stat label="prefill (ttft)" value={`${ttftMs.toFixed(0)} ms`} accent="var(--color-konjo-violet)" />
            <Stat label="decode"          value={`${decodeMs.toFixed(0)} ms`} accent="var(--color-konjo-accent)" />
            <Stat label="total"           value={`${totalMs.toFixed(0)} ms`} accent="var(--color-konjo-fg)" />
          </div>
        </div>

        {/* Per-token strip */}
        <div>
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted mb-2">
            per-token decode interval
          </div>
          <div className="flex items-end gap-[2px] h-16">
            {tokens.map((t, i) => {
              const h = (t.intervalMs / max) * 100;
              const c = i === 0
                ? "var(--color-konjo-violet)"
                : t.intervalMs > 60
                  ? "var(--color-konjo-hot)"
                  : t.intervalMs > 30
                    ? "var(--color-konjo-warm)"
                    : "var(--color-konjo-accent)";
              return (
                <motion.div
                  key={i}
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  transition={{ duration: 0.3, ease: ease.kanjo, delay: i * 0.005 }}
                  className="origin-bottom flex-1 rounded-sm"
                  style={{
                    height: `${h}%`,
                    minHeight: 1,
                    background: c,
                    boxShadow: `0 0 4px ${c}`,
                  }}
                  title={`#${i} · ${t.intervalMs.toFixed(1)}ms${i === 0 ? " (TTFT)" : ""}`}
                />
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span
        className="inline-block rounded-full"
        style={{ width: 6, height: 6, background: accent, boxShadow: `0 0 6px ${accent}` }}
      />
      <span className="text-konjo-fg-muted">{label}</span>
      <span className="tabular-nums" style={{ color: accent }}>{value}</span>
    </div>
  );
}
