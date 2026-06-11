/**
 * Agent run event reducer.
 *
 * POST /v1/agent/run streams SSE frames (see lib/sse.ts) whose payloads are
 * AgentEvent JSON objects. This module turns a flat event stream into the
 * stepwise `AgentStep[]` structure the AgentPlayground renders — pure and
 * fully unit-testable, with no React or fetch dependency.
 */
import type { AgentEvent, AgentStep, AgentToolCall } from "./types";

export function parseAgentEvent(json: string): AgentEvent | null {
  try {
    const obj = JSON.parse(json) as AgentEvent;
    return obj && typeof obj.type === "string" ? obj : null;
  } catch {
    return null;
  }
}

/** Immutable reducer: fold one AgentEvent into the running step list. */
export function applyAgentEvent(steps: AgentStep[], ev: AgentEvent): AgentStep[] {
  const next = steps.map((s) => ({ ...s, calls: [...s.calls] }));
  const cur = ensureOpenStep(next);

  switch (ev.type) {
    case "text_delta":
      cur.text += ev.delta;
      return next;

    case "tool_call_start": {
      const call: AgentToolCall = {
        callId: ev.call_id,
        toolName: ev.tool_name,
        arguments: ev.arguments ?? {},
        done: false,
      };
      cur.calls.push(call);
      return next;
    }

    case "tool_call_result": {
      const call = findCall(next, ev.call_id);
      if (call) {
        call.result = ev.result;
        call.error = ev.error;
        call.elapsedMs = ev.elapsed_ms;
        call.done = true;
      }
      return next;
    }

    case "step_complete":
      cur.step = ev.step;
      cur.complete = true;
      return next;

    case "done":
    case "error":
      // Terminal events are surfaced by the caller, not stored as steps.
      return next;
  }
}

/** Find the open (incomplete) step, creating the first one if none exist. */
function ensureOpenStep(steps: AgentStep[]): AgentStep {
  const open = steps.find((s) => !s.complete);
  if (open) return open;
  const created: AgentStep = {
    step: steps.length + 1,
    text: "",
    calls: [],
    complete: false,
  };
  steps.push(created);
  return created;
}

function findCall(steps: AgentStep[], callId: string): AgentToolCall | undefined {
  for (const s of steps) {
    const c = s.calls.find((x) => x.callId === callId);
    if (c) return c;
  }
  return undefined;
}
