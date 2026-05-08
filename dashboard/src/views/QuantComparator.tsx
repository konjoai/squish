import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { benchmarkKV } from "../lib/api";
import type { CompressBenchResult, KVMode } from "../lib/types";

export interface QuantComparatorProps {
  /** Currently active mode at server start (read-only). */
  serverMode: KVMode;
  /** Callback when fromMock status changes (for MetaInspector). */
  onFromMockChange?: (fromMock: boolean) => void;
}

const CTX_PRESETS = [1024, 2048, 4096, 8192, 16384];

const MODE_COLOR: Record<string, string> = {
  fp16: "var(--color-konjo-fg)",
  int8: "var(--color-mode-int8)",
  int4: "var(--color-mode-int4)",
  int2: "var(--color-mode-int2)",
};

/**
 * Side-by-side comparison of INT8 / INT4 / INT2 KV-cache modes.
 *
 * Honest constraint: squish chooses its KV mode at server startup; it cannot
 * be switched per-request. This view shows what the server is currently
 * running with (`serverMode`) and what it *could* save by switching, via
 * the demo server's /api/benchmark.
 */
export function QuantComparator({ serverMode, onFromMockChange }: QuantComparatorProps) {
  const [ctx, setCtx] = useState<number>(2048);
  const [data, setData] = useState<CompressBenchResult | null>(null);
  const [fromMock, setFromMock] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      const { data, fromMock } = await benchmarkKV(ctx);
      if (cancelled) return;
      setData(data);
      setFromMock(fromMock);
      onFromMockChange?.(fromMock);
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [ctx, onFromMockChange]);

  const baseline = data?.fp16_baseline_bytes ?? 0;
  const maxRatio = Math.max(8, ...((data?.results ?? []).map((r) => r.compression_ratio)));

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Quantization comparator
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Server is running <span className="text-konjo-mono uppercase tracking-[0.18em]" style={{ color: MODE_COLOR[serverMode] ?? "var(--color-konjo-fg-muted)" }}>{serverMode}</span> ·{" "}
            <span className="text-konjo-fg-faint">benchmark from{" "}
              <span className="text-konjo-fg">{fromMock ? "offline mock" : "demo server"}</span>
            </span>
          </p>
        </div>
        <CtxPicker value={ctx} onChange={setCtx} />
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5">
        <div className="grid sm:grid-cols-3 gap-3">
          {(data?.results ?? []).map((r) => {
            const c = MODE_COLOR[r.mode] ?? "var(--color-konjo-fg-muted)";
            const isCurrent = r.mode === serverMode;
            return (
              <motion.div
                key={r.mode}
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: ease.kanjo }}
                className={[
                  "rounded-konjo border p-4 relative",
                  isCurrent ? "border-konjo-accent" : "border-konjo-line/60",
                ].join(" ")}
                style={isCurrent ? { boxShadow: "0 0 16px var(--color-konjo-glow-accent)" } : undefined}
              >
                {isCurrent && (
                  <div className="absolute top-2 right-3 text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-accent">
                    server
                  </div>
                )}
                <div className="text-konjo-display uppercase tracking-[0.22em]" style={{ fontSize: 14, fontWeight: 700, color: c }}>
                  {r.mode}
                </div>
                <div
                  className="text-konjo-display tabular-nums mt-2 leading-none"
                  style={{ fontSize: 36, fontWeight: 600, color: c }}
                >
                  {r.compression_ratio.toFixed(1)}<span className="text-konjo-fg-muted text-[16px]">×</span>
                </div>
                <div className="text-konjo-mono text-[10px] uppercase tracking-[0.18em] text-konjo-fg-muted mt-2">
                  memory
                </div>
                <RatioBar ratio={r.compression_ratio} max={maxRatio} color={c} />
                <div className="text-konjo-mono text-[11px] tabular-nums text-konjo-fg-muted mt-3 grid grid-cols-2 gap-x-3">
                  <span>SNR</span>
                  <span style={{ color: c }} className="text-right">{r.snr_db.toFixed(1)} dB</span>
                  <span>memory</span>
                  <span className="text-right text-konjo-fg">{fmtBytes(r.memory_bytes)}</span>
                  <span>elapsed</span>
                  <span className="text-right text-konjo-fg">{r.elapsed_ms.toFixed(1)} ms</span>
                </div>
              </motion.div>
            );
          })}
        </div>
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-konjo-mono text-[11px] text-konjo-fg-muted mt-3"
            >
              running benchmark @ ctx={ctx}…
            </motion.div>
          )}
        </AnimatePresence>
        <div className="mt-3 text-konjo-mono text-[10px] text-konjo-fg-faint">
          fp16 baseline · {fmtBytes(baseline)} for ctx={data?.ctx_len ?? ctx}, head_dim={data?.head_dim ?? 128}, n_heads={data?.n_heads ?? 32}
        </div>
      </div>
    </section>
  );
}

function CtxPicker({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <div className="inline-flex items-center gap-1 p-1 rounded-konjo bg-konjo-surface border border-konjo-line">
      {CTX_PRESETS.map((c) => (
        <button
          key={c}
          type="button"
          onClick={() => onChange(c)}
          aria-pressed={c === value}
          className={[
            "px-2.5 py-1 rounded-konjo-sm text-konjo-mono uppercase tracking-[0.16em] text-[10px] tabular-nums transition-colors",
            c === value ? "bg-konjo-accent text-konjo-bg" : "text-konjo-fg-muted hover:text-konjo-fg",
          ].join(" ")}
        >
          {c.toLocaleString()}
        </button>
      ))}
    </div>
  );
}

function RatioBar({ ratio, max, color }: { ratio: number; max: number; color: string }) {
  return (
    <div className="h-1.5 rounded-full bg-konjo-line/50 overflow-hidden mt-2">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${(ratio / max) * 100}%` }}
        transition={{ duration: 0.6, ease: ease.kanjo }}
        className="h-full"
        style={{ background: color, boxShadow: `0 0 6px ${color}` }}
      />
    </div>
  );
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${b} B`;
}
