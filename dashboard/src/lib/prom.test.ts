import { describe, it, expect } from "vitest";
import { parseProm, summarizeProm } from "./prom";
import { MOCK_PROM_TEXT } from "./mock";

describe("parseProm", () => {
  it("parses counter rows", () => {
    const m = parseProm(`# TYPE x counter\nx 5\nx 7\n`);
    const x = m.get("x");
    expect(x?.length).toBe(2);
    expect(x?.[0].value).toBe(5);
  });
  it("parses labels", () => {
    const m = parseProm(`x{a="1",b="2"} 9\n`);
    expect(m.get("x")?.[0].labels).toEqual({ a: "1", b: "2" });
  });
});

describe("summarizeProm", () => {
  it("summarizes the squish mock metrics", () => {
    const s = summarizeProm(MOCK_PROM_TEXT);
    expect(s.requests_total).toBe(156);
    expect(s.tokens_total).toBe(18924);
    expect(s.avg_tps).toBeCloseTo(48.67);
    expect(s.kv_cache_tokens).toBe(8192);
    expect(s.kv_cache_memory_mb).toBeCloseTo(64.5);
    expect(s.spec_draft_loaded).toBe(true);
  });
});
