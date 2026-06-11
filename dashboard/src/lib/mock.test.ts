import { describe, it, expect } from "vitest";
import {
  buildMockChatStream, buildMockBenchmark, MOCK_HEALTH,
  buildMockTokenize, buildMockAgentRun, MOCK_AGENT_TOOLS,
} from "./mock";

describe("MOCK_HEALTH", () => {
  it("reports a loaded model with sane defaults", () => {
    expect(MOCK_HEALTH.status).toBe("ok");
    expect(MOCK_HEALTH.loaded).toBe(true);
    expect(MOCK_HEALTH.avg_tps).toBeGreaterThan(0);
  });
});

describe("buildMockChatStream", () => {
  it("returns a multi-token assistant turn marked fromMock", () => {
    const { turn, tokens } = buildMockChatStream();
    expect(tokens.length).toBeGreaterThan(20);
    expect(turn.role).toBe("assistant");
    expect(turn.fromMock).toBe(true);
    expect(turn.content.length).toBeGreaterThan(0);
  });
  it("intervals are positive", () => {
    const { tokens } = buildMockChatStream();
    for (const t of tokens) expect(t.intervalMs).toBeGreaterThan(0);
  });
});

describe("buildMockBenchmark", () => {
  it("emits compression ratios that descend INT8 → INT4 → INT2", () => {
    const r = buildMockBenchmark(2048);
    const ratios = r.results.map((x) => x.compression_ratio);
    expect(ratios[0]).toBeLessThan(ratios[1]);
    expect(ratios[1]).toBeLessThan(ratios[2]);
  });
  it("memory descends with mode", () => {
    const r = buildMockBenchmark(2048);
    const m = r.results.map((x) => x.memory_bytes);
    expect(m[0]).toBeGreaterThan(m[1]);
    expect(m[1]).toBeGreaterThan(m[2]);
  });
});

describe("buildMockTokenize", () => {
  it("produces one id per token and a matching count", () => {
    const r = buildMockTokenize("hello world!");
    expect(r.token_ids.length).toBe(r.token_count);
    expect(r.token_count).toBeGreaterThan(0);
  });
  it("is deterministic for the same input", () => {
    expect(buildMockTokenize("abc def").token_ids).toEqual(buildMockTokenize("abc def").token_ids);
  });
  it("returns zero tokens for empty input", () => {
    expect(buildMockTokenize("").token_count).toBe(0);
  });
});

describe("buildMockAgentRun", () => {
  it("emits tool calls that each resolve to a result and a terminal done", () => {
    const seq = buildMockAgentRun([{ role: "user", content: "inspect the repo" }]);
    const types = seq.map((s) => s.ev.type);
    expect(types).toContain("tool_call_start");
    expect(types).toContain("tool_call_result");
    expect(types[types.length - 1]).toBe("done");
  });
  it("delays are non-negative", () => {
    for (const s of buildMockAgentRun([{ role: "user", content: "x" }])) {
      expect(s.delayMs).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("MOCK_AGENT_TOOLS", () => {
  it("exposes named function tools", () => {
    expect(MOCK_AGENT_TOOLS.length).toBeGreaterThan(0);
    for (const t of MOCK_AGENT_TOOLS) {
      expect(t.type).toBe("function");
      expect(t.function.name).toMatch(/^squish_/);
    }
  });
});
