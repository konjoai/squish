import type { HealthResponse } from "../lib/types";

export interface MetaInspectorProps {
  health: HealthResponse;
  healthFromMock: boolean;
  metricsFromMock: boolean;
  benchFromMock: boolean;
  /** True if the most recent chat turn came from the mock fallback. */
  chatFromMock: boolean;
  agentFromMock: boolean;
  tokFromMock: boolean;
  qualityFromMock: boolean;
}

function StatBlock({
  label, value, accent,
}: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 rounded-konjo bg-konjo-surface/60 border border-konjo-line/60">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
        {label}
      </div>
      <div
        className="text-konjo-mono tabular-nums text-konjo-fg"
        style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)" }}
      >
        {value}
      </div>
    </div>
  );
}

const ok = "var(--color-konjo-good)";
const warn = "var(--color-konjo-warm)";

export function MetaInspector({
  health, healthFromMock, metricsFromMock, benchFromMock, chatFromMock,
  agentFromMock, tokFromMock, qualityFromMock,
}: MetaInspectorProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
      <StatBlock label="server"     value={healthFromMock ? "offline · mock" : "live"} accent={healthFromMock ? warn : ok} />
      <StatBlock label="chat"       value={chatFromMock ? "mock replay" : "live SSE"} accent={chatFromMock ? warn : ok} />
      <StatBlock label="agent"      value={agentFromMock ? "mock run" : "live SSE"} accent={agentFromMock ? warn : ok} />
      <StatBlock label="metrics"    value={metricsFromMock ? "mock" : "live"} accent={metricsFromMock ? warn : ok} />
      <StatBlock label="tokenize"   value={tokFromMock ? "mock bpe" : "live"} accent={tokFromMock ? warn : ok} />
      <StatBlock label="quality"    value={qualityFromMock ? "mock" : "live"} accent={qualityFromMock ? warn : ok} />
      <StatBlock label="kv-bench"   value={benchFromMock ? "offline · mock" : "demo server"} accent={benchFromMock ? warn : "var(--color-konjo-violet)"} />
      <StatBlock label="kv-cache"   value="aggregate" accent="var(--color-konjo-violet)" />
      <StatBlock label="status"     value={health.status === "ok" ? "ready" : health.status} accent={health.status === "ok" ? ok : warn} />
    </div>
  );
}
