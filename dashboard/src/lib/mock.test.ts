import { describe, it, expect } from "vitest";
import { buildMockChatStream, buildMockBenchmark, MOCK_HEALTH } from "./mock";

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
