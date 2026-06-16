import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import type { AgentStep, AgentToolCall } from "../lib/types";

/**
 * Shared agent step/tool-call timeline rendering.
 *
 * Used by both the AgentPlayground (dedicated agent surface) and the main
 * ChatPanel (agentic chat mode) so the live tool-execution visuals are
 * identical everywhere the agent loop runs.
 */
export function AgentTimeline({ steps, running }: { steps: AgentStep[]; running: boolean }) {
  return (
    <AnimatePresence initial={false}>
      {steps.map((s) => (
        <StepCard key={s.step + "-" + s.calls.length} step={s} running={running} />
      ))}
    </AnimatePresence>
  );
}

export function StepCard({ step, running }: { step: AgentStep; running: boolean }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: ease.kanjo }}
      className="rounded-konjo border border-konjo-line/60 bg-konjo-surface/30 p-3"
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
          step {step.step}
        </span>
        {step.complete ? (
          <span className="text-konjo-mono text-[9px] text-konjo-good">✓</span>
        ) : (
          running && <PulseDot />
        )}
      </div>
      {step.text.trim() && (
        <div className="text-konjo-fg" style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
          {step.text}
        </div>
      )}
      {step.calls.map((c) => (
        <ToolCallCard key={c.callId} call={c} />
      ))}
    </motion.div>
  );
}

export function ToolCallCard({ call }: { call: AgentToolCall }) {
  const failed = call.done && call.error != null;
  const c = failed
    ? "var(--color-konjo-hot)"
    : call.done
    ? "var(--color-konjo-good)"
    : "var(--color-konjo-violet)";
  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, ease: ease.kanjo }}
      className="mt-2 rounded-konjo border bg-konjo-bg/40 overflow-hidden"
      style={{ borderColor: c + "55" }}
    >
      <div
        className="flex items-center gap-2 px-3 py-1.5"
        style={{ background: `color-mix(in oklch, ${c} 8%, transparent)` }}
      >
        <span
          style={{
            width: 5,
            height: 5,
            borderRadius: 99,
            background: c,
            boxShadow: `0 0 6px ${c}`,
            display: "inline-block",
          }}
        />
        <span className="text-konjo-mono text-[11px]" style={{ color: c }}>
          {call.toolName}
        </span>
        {!call.done && <PulseDot />}
        {call.done && call.elapsedMs != null && (
          <span className="ml-auto text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted">
            {call.elapsedMs.toFixed(1)} ms
          </span>
        )}
      </div>
      <div className="px-3 py-2 space-y-1.5">
        <pre
          className="text-konjo-mono text-[10.5px] text-konjo-fg-muted overflow-x-auto m-0"
          style={{ whiteSpace: "pre-wrap" }}
        >
          {JSON.stringify(call.arguments, null, 0)}
        </pre>
        <AnimatePresence>
          {call.done && (
            <motion.pre
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="text-konjo-mono text-[10.5px] overflow-x-auto m-0 pt-1.5 border-t border-konjo-line/40"
              style={{
                whiteSpace: "pre-wrap",
                color: failed ? "var(--color-konjo-hot)" : "var(--color-konjo-fg)",
              }}
            >
              {failed ? call.error : call.result}
            </motion.pre>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

export function PulseDot() {
  return (
    <motion.span
      aria-hidden
      animate={{ opacity: [1, 0.25, 1] }}
      transition={{ duration: 0.9, repeat: Infinity, ease: ease.seishin }}
      className="inline-block rounded-full"
      style={{
        width: 6,
        height: 6,
        background: "var(--color-konjo-violet)",
        boxShadow: "0 0 8px var(--color-konjo-violet)",
      }}
    />
  );
}
