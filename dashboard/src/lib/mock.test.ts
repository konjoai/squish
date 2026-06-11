import { describe, it, expect } from "vitest";
import {
  buildMockChatStream, buildMockBenchmark, MOCK_HEALTH,
  buildMockTokenize, buildMockAgentRun, MOCK_AGENT_TOOLS,
  buildMockEmbeddings, MOCK_SYS_STATS, MOCK_MODEL_STATUS,
} from "./mock";
import { cosineSimilarity } from "./vector";

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

describe("buildMockEmbeddings", () => {
  it("returns one normalized vector per input", () => {
    const r = buildMockEmbeddings(["hello world", "foo bar baz"]);
    expect(r.data).toHaveLength(2);
    for (const d of r.data) {
      const norm = Math.sqrt(d.embedding.reduce((a, x) => a + x * x, 0));
      expect(norm).toBeCloseTo(1, 5);
    }
  });

  it("scores semantically-overlapping texts higher than unrelated ones", () => {
    const r = buildMockEmbeddings([
      "the cat sat on the mat",
      "the cat sat on the rug",
      "quantize the kv cache to int4",
    ]);
    const [a, b, c] = r.data.map((d) => d.embedding);
    expect(cosineSimilarity(a, b)).toBeGreaterThan(cosineSimilarity(a, c));
  });

  it("is deterministic", () => {
    expect(buildMockEmbeddings(["abc def"]).data[0].embedding)
      .toEqual(buildMockEmbeddings(["abc def"]).data[0].embedding);
  });
});

describe("MOCK_SYS_STATS / MOCK_MODEL_STATUS", () => {
  it("expose plausible host + load-state shapes", () => {
    expect(MOCK_SYS_STATS.load_avg).toHaveLength(3);
    expect(MOCK_SYS_STATS.disk_used_pct).toBeGreaterThanOrEqual(0);
    expect(MOCK_MODEL_STATUS.model_loaded).toBe(true);
    expect(["eager", "lazy", "preload_async"]).toContain(MOCK_MODEL_STATUS.load_mode);
  });
});
