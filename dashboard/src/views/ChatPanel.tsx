import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import type { ChatTurn } from "../lib/types";

export interface ChatPanelProps {
  turns: ChatTurn[];
  /** The currently-streaming assistant turn — appended live until done. */
  active?: ChatTurn | null;
}

/**
 * Conversation history with per-token color hue on assistant turns.
 *
 * Each token's background opacity reflects its inter-token latency
 * normalized within the turn (cool = fast, hot = slow). The TTFT chunk
 * is highlighted with a violet pip.
 */
export function ChatPanel({ turns, active }: ChatPanelProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom on new content.
  useEffect(() => {
    const c = ref.current;
    if (c) c.scrollTop = c.scrollHeight;
  }, [turns, active?.content]);

  const all = active ? [...turns, active] : turns;

  return (
    <div
      ref={ref}
      className="glass-konjo rounded-konjo-lg p-5 overflow-y-auto"
      style={{ maxHeight: 540, minHeight: 360 }}
    >
      <AnimatePresence initial={false}>
        {all.map((turn, idx) => (
          <motion.div
            key={turn.id}
            layout
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: ease.kanjo }}
            className="mb-5 last:mb-0"
          >
            <RoleHeader turn={turn} streaming={active != null && idx === all.length - 1 && turn.id === active.id} />
            {turn.role === "assistant" && turn.tokens && turn.tokens.length > 0
              ? <AssistantBody turn={turn} />
              : <SimpleBody content={turn.content} muted={turn.role === "user"} />}
          </motion.div>
        ))}
      </AnimatePresence>
      {all.length === 0 && (
        <div className="text-konjo-fg-muted text-konjo-mono text-[13px] py-12 text-center">
          start a conversation — every token is timed.
        </div>
      )}
    </div>
  );
}

function RoleHeader({ turn, streaming }: { turn: ChatTurn; streaming: boolean }) {
  const c = turn.role === "user"
    ? "var(--color-konjo-violet)"
    : turn.role === "system"
    ? "var(--color-konjo-fg-muted)"
    : "var(--color-konjo-accent)";
  return (
    <div className="flex items-baseline gap-2 mb-1.5">
      <span
        className="inline-block rounded-full"
        style={{ width: 6, height: 6, background: c, boxShadow: `0 0 8px ${c}` }}
      />
      <span
        className="text-konjo-mono uppercase tracking-[0.18em] text-[10px]"
        style={{ color: c }}
      >
        {turn.role}
      </span>
      {turn.ttftS != null && (
        <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted ml-2">
          ttft {turn.ttftS.toFixed(2)}s
        </span>
      )}
      {turn.totalS != null && (
        <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted">
          · total {turn.totalS.toFixed(2)}s
        </span>
      )}
      {turn.tokens != null && turn.tokens.length > 0 && (
        <span className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted">
          · {turn.tokens.length} tok
        </span>
      )}
      {streaming && (
        <motion.span
          aria-hidden
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.0, ease: ease.seishin, repeat: Infinity }}
          className="ml-auto text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-accent"
        >
          streaming
        </motion.span>
      )}
    </div>
  );
}

function SimpleBody({ content, muted }: { content: string; muted: boolean }) {
  return (
    <div
      className={muted ? "text-konjo-fg-muted" : "text-konjo-fg"}
      style={{ fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}
    >
      {content}
    </div>
  );
}

function AssistantBody({ turn }: { turn: ChatTurn }) {
  const tokens = turn.tokens ?? [];
  if (tokens.length === 0) return <SimpleBody content={turn.content} muted={false} />;

  // Normalize intervals (skip the first one as it's TTFT, often outsized).
  const decodeIntervals = tokens.slice(1).map((t) => t.intervalMs).sort((a, b) => a - b);
  const p10 = decodeIntervals[Math.floor(decodeIntervals.length * 0.1)] ?? 0;
  const p90 = decodeIntervals[Math.floor(decodeIntervals.length * 0.9)] ?? 1;
  const span = Math.max(0.001, p90 - p10);

  return (
    <div
      style={{ fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-wrap" }}
      className="text-konjo-fg"
    >
      {tokens.map((t, i) => {
        const w = i === 0 ? 0 : Math.max(0, Math.min(1, (t.intervalMs - p10) / span));
        const c = w >= 0.66 ? "var(--color-konjo-hot)" :
                  w >= 0.33 ? "var(--color-konjo-warm)" :
                              "var(--color-konjo-accent)";
        return (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 4, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.2, ease: ease.kanjo }}
            style={{
              color: t.text.trim() ? "var(--color-konjo-fg)" : undefined,
              background: w > 0 ? `color-mix(in oklch, ${c} ${Math.round(w * 35)}%, transparent)` : "transparent",
              borderRadius: w > 0 ? 2 : undefined,
              padding: w > 0 ? "0 1px" : undefined,
            }}
            title={`${t.intervalMs.toFixed(1)}ms · ${t.tps.toFixed(1)} tok/s`}
          >
            {t.text}
          </motion.span>
        );
      })}
    </div>
  );
}
