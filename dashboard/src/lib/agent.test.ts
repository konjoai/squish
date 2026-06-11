import { describe, it, expect } from "vitest";
import { parseAgentEvent, applyAgentEvent } from "./agent";
import type { AgentEvent, AgentStep } from "./types";

describe("parseAgentEvent", () => {
  it("parses a valid event", () => {
    const ev = parseAgentEvent('{"type":"text_delta","delta":"hi"}');
    expect(ev).toEqual({ type: "text_delta", delta: "hi" });
  });

  it("returns null on malformed json", () => {
    expect(parseAgentEvent("{not json")).toBeNull();
  });

  it("returns null when type is missing", () => {
    expect(parseAgentEvent('{"delta":"x"}')).toBeNull();
  });
});

describe("applyAgentEvent", () => {
  const fold = (events: AgentEvent[]): AgentStep[] =>
    events.reduce<AgentStep[]>((acc, ev) => applyAgentEvent(acc, ev), []);

  it("accumulates text deltas into the first step", () => {
    const steps = fold([
      { type: "text_delta", delta: "Hello" },
      { type: "text_delta", delta: " world" },
    ]);
    expect(steps).toHaveLength(1);
    expect(steps[0].text).toBe("Hello world");
    expect(steps[0].complete).toBe(false);
  });

  it("threads a tool call start + result by call_id", () => {
    const steps = fold([
      { type: "tool_call_start", call_id: "c1", tool_name: "squish_list_dir", arguments: { path: "/" } },
      { type: "tool_call_result", call_id: "c1", tool_name: "squish_list_dir", result: "a b c", error: null, elapsed_ms: 3.5 },
    ]);
    expect(steps[0].calls).toHaveLength(1);
    const call = steps[0].calls[0];
    expect(call.toolName).toBe("squish_list_dir");
    expect(call.arguments).toEqual({ path: "/" });
    expect(call.result).toBe("a b c");
    expect(call.elapsedMs).toBe(3.5);
    expect(call.done).toBe(true);
    expect(call.error).toBeNull();
  });

  it("marks the result as failed when error is present", () => {
    const steps = fold([
      { type: "tool_call_start", call_id: "c2", tool_name: "squish_read_file", arguments: {} },
      { type: "tool_call_result", call_id: "c2", tool_name: "squish_read_file", result: "", error: "no such file", elapsed_ms: 1 },
    ]);
    expect(steps[0].calls[0].error).toBe("no such file");
    expect(steps[0].calls[0].done).toBe(true);
  });

  it("opens a new step after step_complete", () => {
    const steps = fold([
      { type: "text_delta", delta: "step one" },
      { type: "step_complete", step: 1 },
      { type: "text_delta", delta: "step two" },
    ]);
    expect(steps).toHaveLength(2);
    expect(steps[0].complete).toBe(true);
    expect(steps[0].text).toBe("step one");
    expect(steps[1].complete).toBe(false);
    expect(steps[1].text).toBe("step two");
  });

  it("does not mutate the input array (immutable reducer)", () => {
    const before: AgentStep[] = [];
    const after = applyAgentEvent(before, { type: "text_delta", delta: "x" });
    expect(before).toHaveLength(0);
    expect(after).toHaveLength(1);
  });

  it("ignores terminal done/error events without creating extra calls", () => {
    const steps = fold([
      { type: "text_delta", delta: "done soon" },
      { type: "done", total_steps: 1, total_tool_calls: 0 },
    ]);
    expect(steps).toHaveLength(1);
    expect(steps[0].calls).toHaveLength(0);
  });
});
