import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { agentRun, fetchAgentTools } from "../lib/api";
import { applyAgentEvent } from "../lib/agent";
import { StepCard } from "./AgentTimeline";
import type { AgentStep, AgentTool } from "../lib/types";

export interface AgentPlaygroundProps {
  /** Reports whether the live agent transport fell back to a mock run. */
  onFromMockChange?: (fromMock: boolean) => void;
}

const SUGGESTIONS = [
  "What version is this project and how is it laid out?",
  "List the top-level files and summarise the README.",
  "Find every TODO in the codebase.",
];

/**
 * Agent Playground — drives POST /v1/agent/run and renders the live tool-call
 * timeline. Tools are discovered from GET /v1/agent/tools. Every event
 * (text_delta, tool_call_start/result, step_complete, done/error) animates in
 * as it arrives, so the model's reasoning and tool execution are visible live.
 */
export function AgentPlayground({ onFromMockChange }: AgentPlaygroundProps) {
  const [tools, setTools] = useState<AgentTool[]>([]);
  const [task, setTask] = useState<string>(SUGGESTIONS[0]);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [running, setRunning] = useState<boolean>(false);
  const [terminal, setTerminal] = useState<{ kind: "done" | "error"; text: string } | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchAgentTools().then(({ tools }) => { if (!cancelled) setTools(tools); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const c = scrollRef.current;
    if (c) c.scrollTop = c.scrollHeight;
  }, [steps, terminal]);

  const run = () => {
    cancelRef.current?.();
    if (task.trim().length === 0) return;
    setSteps([]);
    setTerminal(null);
    setRunning(true);
    const handle = agentRun(
      { messages: [{ role: "user", content: task.trim() }], max_steps: 6 },
      (ev, opts) => {
        onFromMockChange?.(opts.fromMock);
        if (ev.type === "done") {
          setTerminal({ kind: "done", text: `${ev.total_steps} steps · ${ev.total_tool_calls} tool calls` });
          return;
        }
        if (ev.type === "error") {
          setTerminal({ kind: "error", text: ev.message });
          return;
        }
        setSteps((prev) => applyAgentEvent(prev, ev));
      },
    );
    cancelRef.current = handle.cancel;
    handle.done.then(() => setRunning(false)).catch(() => setRunning(false));
  };

  const stop = () => { cancelRef.current?.(); setRunning(false); };

  const totalCalls = steps.reduce((a, s) => a + s.calls.length, 0);

  return (
    <section className="space-y-3" id="agent">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Agent playground
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Live tool execution via <span className="text-konjo-mono">/v1/agent/run</span> ·{" "}
            <span className="text-konjo-fg">{tools.length}</span> tools registered
          </p>
        </div>
        <div className="flex items-center gap-2 text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted">
          {steps.length > 0 && <span>{steps.length} steps · {totalCalls} calls</span>}
        </div>
      </header>

      <ToolPalette tools={tools} />

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div className="flex items-end gap-2">
          <textarea
            rows={2}
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !running) run();
            }}
            placeholder="give the agent a task · ⌘/ctrl-enter to run"
            className="flex-1 bg-konjo-surface/50 border border-konjo-line rounded-konjo px-3 py-2 outline-none text-konjo-fg placeholder:text-konjo-fg-faint resize-none focus:shadow-konjo-glow transition-shadow"
            style={{ fontSize: 14, lineHeight: 1.5 }}
          />
          <button
            type="button"
            onClick={running ? stop : run}
            className={[
              "px-5 py-2.5 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[11px] transition-colors shrink-0",
              running
                ? "bg-konjo-hot/20 text-konjo-hot border border-konjo-hot/50 cursor-pointer"
                : "bg-konjo-accent text-konjo-bg hover:brightness-110 cursor-pointer shadow-konjo-glow",
            ].join(" ")}
          >
            {running ? "stop" : "run agent"}
          </button>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={running}
              onClick={() => setTask(s)}
              className="text-konjo-mono text-[10px] px-2 py-1 rounded-konjo-sm border border-konjo-line/60 text-konjo-fg-muted hover:text-konjo-fg hover:border-konjo-violet/60 transition-colors disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>

        <div ref={scrollRef} className="overflow-y-auto space-y-3" style={{ maxHeight: 460 }}>
          <AnimatePresence initial={false}>
            {steps.map((s) => <StepCard key={s.step + "-" + s.calls.length} step={s} running={running} />)}
          </AnimatePresence>

          {steps.length === 0 && !running && (
            <div className="text-konjo-fg-muted text-konjo-mono text-[13px] py-10 text-center">
              the agent reasons, calls tools, and reads the results — watch it work live.
            </div>
          )}

          <AnimatePresence>
            {terminal && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={[
                  "rounded-konjo border px-3 py-2 text-konjo-mono text-[11px] flex items-center gap-2",
                  terminal.kind === "done"
                    ? "border-konjo-good/50 text-konjo-good"
                    : "border-konjo-hot/50 text-konjo-hot",
                ].join(" ")}
              >
                <span style={{ width: 6, height: 6, borderRadius: 99, background: "currentColor", boxShadow: "0 0 8px currentColor", display: "inline-block" }} />
                {terminal.kind === "done" ? "agent complete" : "agent error"} · {terminal.text}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function ToolPalette({ tools }: { tools: AgentTool[] }) {
  if (tools.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {tools.map((t, i) => (
        <motion.span
          key={t.function.name}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, ease: ease.kanjo, delay: i * 0.03 }}
          title={t.function.description}
          className="text-konjo-mono text-[10px] px-2 py-1 rounded-konjo-sm bg-konjo-surface/60 border border-konjo-line/60 text-konjo-violet"
        >
          {t.function.name}
        </motion.span>
      ))}
    </div>
  );
}

