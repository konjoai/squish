import { useMemo } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { CockpitMetrics, KVMode } from "../lib/types";

export interface KVCacheViewProps {
  metrics: CockpitMetrics;
  mode: KVMode;
  /** Approximate model context capacity for the current model (heuristic). */
  ctxCapacity?: number;
}

/**
 * KV cache visualization. The left side is the AGGREGATE truth from
 * /v1/metrics (kv_cache_tokens, kv_cache_memory_mb, paged_kv_used_blocks,
 * paged_kv_free_blocks). The right side renders a synthesized block grid
 * whose total usage exactly matches the live used:free ratio — it's an
 * honest aggregate visualization, not a per-layer/per-position truth claim.
 *
 * The MetaInspector reports this as `kv: aggregate`.
 */
export function KVCacheView({ metrics, mode, ctxCapacity = 8192 }: KVCacheViewProps) {
  const usedBlocks = metrics.paged_kv_used_blocks;
  const freeBlocks = metrics.paged_kv_free_blocks;
  const totalBlocks = usedBlocks + freeBlocks;
  const usedFrac = totalBlocks > 0 ? usedBlocks / totalBlocks : 0;

  const tokens = metrics.kv_cache_tokens;
  const memMb = metrics.kv_cache_memory_mb;

  // Render a 32×16 block grid synthesized to match the used:free ratio.
  const grid = useMemo(() => buildBlockGrid(usedFrac), [usedFrac]);
  const ctxFrac = Math.max(0, Math.min(1, tokens / Math.max(1, ctxCapacity)));

  const modeColor = MODE_COLOR[mode] ?? "var(--color-konjo-fg-muted)";

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            KV cache
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Aggregate from <span className="text-konjo-mono">/v1/metrics</span> ·{" "}
            mode <span style={{ color: modeColor }} className="text-konjo-mono uppercase tracking-[0.18em]">{mode}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 grid lg:grid-cols-[1fr_auto] gap-5 items-start">
        <div className="space-y-4 min-w-0">
          <Bar
            label="paged blocks · used"
            value={usedBlocks}
            total={totalBlocks}
            color="var(--color-konjo-accent)"
            valueText={`${usedBlocks.toLocaleString()} / ${totalBlocks.toLocaleString()}`}
          />
          <Bar
            label="context · tokens cached"
            value={tokens}
            total={Math.max(tokens, ctxCapacity)}
            color={ctxFrac > 0.85 ? "var(--color-konjo-hot)" : ctxFrac > 0.6 ? "var(--color-konjo-warm)" : "var(--color-konjo-violet)"}
            valueText={`${tokens.toLocaleString()} tok`}
          />
          <Bar
            label="memory"
            value={memMb}
            total={Math.max(memMb, 256)}
            color="var(--color-konjo-good)"
            valueText={`${memMb.toFixed(1)} MB`}
          />

          <div className="grid grid-cols-3 gap-2 pt-2">
            <Stat label="prefix-cache hits" value={fmtCompact(metrics.prefix_cache_hits)} />
            <Stat label="radix reuse"        value={fmtCompact(metrics.radix_prefix_hits)} />
            <Stat
              label="speculative"
              value={metrics.spec_draft_loaded ? "draft loaded" : "off"}
              accent={metrics.spec_draft_loaded ? "var(--color-konjo-good)" : "var(--color-konjo-fg-muted)"}
            />
          </div>
        </div>

        <div className="flex flex-col items-center gap-2 shrink-0">
          <BlockGrid grid={grid} color={modeColor} />
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
            paged kv · used : free
          </div>
        </div>
      </div>
    </section>
  );
}

const MODE_COLOR: Record<KVMode, string> = {
  fp16:    "var(--color-konjo-fg)",
  int8:    "var(--color-mode-int8)",
  int4:    "var(--color-mode-int4)",
  int3:    "var(--color-mode-int3)",
  int2:    "var(--color-mode-int2)",
  snap:    "var(--color-konjo-good)",
  unknown: "var(--color-konjo-fg-muted)",
};

function Bar({
  label, value, total, color, valueText,
}: { label: string; value: number; total: number; color: string; valueText: string }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted">{label}</span>
        <span className="text-konjo-mono text-[11px] tabular-nums" style={{ color }}>{valueText}</span>
      </div>
      <div className="h-2 rounded-full bg-konjo-line/60 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
          className="h-full rounded-full"
          style={{ background: color, boxShadow: `0 0 8px ${color}` }}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-2 py-1.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div className="text-konjo-mono tabular-nums mt-0.5" style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)" }}>
        {value}
      </div>
    </div>
  );
}

function fmtCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return `${Math.round(n)}`;
}

const COLS = 32;
const ROWS = 12;

function buildBlockGrid(usedFrac: number): boolean[][] {
  const total = COLS * ROWS;
  const used = Math.round(total * Math.max(0, Math.min(1, usedFrac)));
  const grid: boolean[][] = Array.from({ length: ROWS }, () => Array(COLS).fill(false));
  let placed = 0;
  // Fill bottom-up, left-to-right for a "stacked" feel.
  outer: for (let r = ROWS - 1; r >= 0; r--) {
    for (let c = 0; c < COLS; c++) {
      if (placed >= used) break outer;
      grid[r][c] = true;
      placed++;
    }
  }
  return grid;
}

function BlockGrid({ grid, color }: { grid: boolean[][]; color: string }) {
  const cell = 10;
  const gap = 2;
  const W = COLS * (cell + gap) - gap;
  const H = ROWS * (cell + gap) - gap;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
      {grid.flatMap((row, r) =>
        row.map((on, c) => (
          <motion.rect
            key={`${r}-${c}`}
            x={c * (cell + gap)}
            y={r * (cell + gap)}
            width={cell}
            height={cell}
            rx={1.5}
            initial={{ opacity: 0 }}
            animate={{ opacity: on ? 1 : 0.18 }}
            transition={{ duration: 0.35, ease: ease.kanjo, delay: (r * COLS + c) * 0.001 }}
            fill={on ? color : "var(--color-konjo-line)"}
            style={on ? { filter: `drop-shadow(0 0 3px ${color})` } : undefined}
          />
        )),
      )}
    </svg>
  );
}
