import { Dial } from "@konjoai/ui";
import type { HealthResponse } from "../lib/types";

export interface ThermalDialProps {
  health: HealthResponse;
}

const POWER_COLOR: Record<HealthResponse["power_mode"], string> = {
  performance: "var(--color-power-perf)",
  balanced:    "var(--color-power-balance)",
  battery:     "var(--color-power-bat)",
  auto:        "var(--color-konjo-violet)",
};

/**
 * Apple Silicon power telemetry. Reads /health.battery_level (0–100),
 * /health.mem_pressure (0–100, macOS memory governor), and the active
 * power_mode. CPU temperature is not exposed by squish; we deliberately
 * do not synthesize it.
 */
export function ThermalDial({ health }: ThermalDialProps) {
  const battery = health.battery_level;
  const mp = health.mem_pressure ?? 0;
  const memSev = mp >= 70 ? "high" : mp >= 50 ? "warn" : mp >= 25 ? "info" : "ok";

  return (
    <section className="space-y-3">
      <header>
        <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
          Power & thermals
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Apple Silicon telemetry · battery + memory governor (no CPU temp exposed)
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 grid grid-cols-2 sm:grid-cols-3 gap-4 items-center justify-items-center">
        <Dial
          value={mp}
          min={0}
          max={100}
          unit="%"
          label="Mem pressure"
          severity={memSev}
          format={(v) => v.toFixed(0)}
          size={150}
        />

        {battery != null ? (
          <Dial
            value={battery}
            min={0}
            max={100}
            unit="%"
            label="Battery"
            severity={battery >= 60 ? "ok" : battery >= 25 ? "warn" : "high"}
            format={(v) => v.toFixed(0)}
            size={150}
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-center" style={{ width: 150, height: 150 }}>
            <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted">battery</div>
            <div className="text-konjo-fg-faint mt-2 text-konjo-mono text-[12px]">not exposed</div>
          </div>
        )}

        <PowerModeCard mode={health.power_mode} memAvailGb={health.mem_available_gb} />
      </div>
    </section>
  );
}

function PowerModeCard({ mode, memAvailGb }: { mode: HealthResponse["power_mode"]; memAvailGb: number | null }) {
  const c = POWER_COLOR[mode];
  return (
    <div className="flex flex-col items-center gap-2 p-4 rounded-konjo border border-konjo-line/60 bg-konjo-surface/40 w-full max-w-[200px]">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted">power mode</div>
      <div
        className="text-konjo-display uppercase tracking-[0.16em]"
        style={{ fontSize: 14, fontWeight: 700, color: c }}
      >
        {mode}
      </div>
      <div className="w-full h-1 rounded-full" style={{ background: c, boxShadow: `0 0 8px ${c}` }} />
      <div className="text-konjo-mono text-[10px] uppercase tracking-[0.18em] text-konjo-fg-muted mt-2">free ram</div>
      <div className="text-konjo-mono tabular-nums" style={{ fontSize: 16, fontWeight: 600, color: "var(--color-konjo-fg)" }}>
        {memAvailGb != null ? `${memAvailGb.toFixed(1)} GB` : "—"}
      </div>
    </div>
  );
}
