import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { fetchStartupProfile } from "../lib/api";
import { AnimatedNumber } from "../components/AnimatedNumber";
import type { StartupProfile as StartupProfileData, StartupEntry } from "../lib/types";

export interface StartupProfileProps {
  onFromMockChange?: (fromMock: boolean) => void;
}

const PHASE_HUES = [
  "var(--color-konjo-violet)",
  "var(--color-konjo-cool)",
  "var(--color-konjo-accent)",
  "var(--color-konjo-warm)",
  "var(--color-konjo-good)",
  "var(--color-konjo-hot)",
];

/**
 * Startup profile — a cumulative waterfall of the server's cold-start phases
 * from GET /v1/startup-profile. Each phase is a bar offset by the cumulative
 * time before it, so you can see exactly where the boot budget goes (almost
 * always model_load). Fetched once on mount — startup timing is fixed for the
 * server's lifetime. Honest "not enabled" state when SQUISH_TRACE_STARTUP is off.
 */
export function StartupProfile({ onFromMockChange }: StartupProfileProps) {
  const [data, setData] = useState<StartupProfileData | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchStartupProfile().then(({ data, fromMock }) => {
      if (cancelled) return;
      setData(data);
      onFromMockChange?.(fromMock);
    });
    return () => { cancelled = true; };
  }, [onFromMockChange]);

  const entries = data?.entries ?? [];
  const total = data?.total_ms ?? entries.reduce((a, e) => a + e.elapsed_ms, 0);
  const slowest = data?.slowest_5?.[0]?.phase;

  return (
    <section className="space-y-3" id="startup">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Startup profile
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Cold-start waterfall via <span className="text-konjo-mono">/v1/startup-profile</span>
          </p>
        </div>
        {entries.length > 0 && (
          <div className="text-right">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">total boot</div>
            <div className="text-konjo-display tabular-nums" style={{ fontSize: 22, fontWeight: 600, color: "var(--color-konjo-accent)" }}>
              <AnimatedNumber value={total / 1000} format={(v) => `${v.toFixed(2)}s`} />
            </div>
          </div>
        )}
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5">
        {data && data.enabled === false ? (
          <Disabled message={data.message} />
        ) : entries.length === 0 ? (
          <div className="text-konjo-fg-muted text-konjo-mono text-[12px] py-8 text-center">loading startup profile…</div>
        ) : (
          <div className="space-y-2">
            {accumulate(entries).map(({ e, offset }, i) => {
              const c = PHASE_HUES[i % PHASE_HUES.length];
              const isSlowest = e.phase === slowest;
              return (
                <div key={e.phase + i}>
                  <div className="flex items-baseline justify-between mb-1 gap-2">
                    <span className="text-konjo-mono text-[11px] truncate" style={{ color: c }}>
                      {e.phase}
                      {isSlowest && <span className="ml-2 text-konjo-fg-faint uppercase tracking-[0.16em] text-[8px]">slowest</span>}
                    </span>
                    <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted shrink-0">
                      {fmtMs(e.elapsed_ms)}
                    </span>
                  </div>
                  <div className="relative h-3 rounded-sm bg-konjo-line/30 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(e.elapsed_ms / total) * 100}%` }}
                      transition={{ duration: 0.6, ease: ease.kanjo, delay: i * 0.05 }}
                      className="absolute top-0 h-full rounded-sm"
                      style={{
                        left: `${(offset / total) * 100}%`,
                        background: c,
                        boxShadow: isSlowest ? `0 0 8px ${c}` : `0 0 4px ${c}`,
                      }}
                    />
                  </div>
                  <div className="text-konjo-fg-faint text-[10px] mt-0.5 truncate">{e.label}</div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

function Disabled({ message }: { message?: string }) {
  return (
    <div className="text-konjo-fg-muted text-konjo-mono text-[12px] py-6 text-center" style={{ lineHeight: 1.6 }}>
      startup tracing not enabled
      <div className="text-konjo-fg-faint text-[11px] mt-1">
        {message ?? <>start the server with <span className="text-konjo-fg-muted">SQUISH_TRACE_STARTUP=1</span> to capture phase timing</>}
      </div>
    </div>
  );
}

/** Attach a cumulative start offset to each phase (sequential waterfall). */
function accumulate(entries: StartupEntry[]): { e: StartupEntry; offset: number }[] {
  let acc = 0;
  return entries.map((e) => {
    const offset = acc;
    acc += e.elapsed_ms;
    return { e, offset };
  });
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`;
}
