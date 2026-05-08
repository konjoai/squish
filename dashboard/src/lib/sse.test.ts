import { describe, it, expect } from "vitest";
import { parseStreamChunk } from "./sse";

describe("parseStreamChunk", () => {
  it("parses two complete SSE frames", () => {
    const buf = `data: {"a":1}\n\ndata: {"a":2}\n\n`;
    const { frames, rest } = parseStreamChunk(buf);
    expect(frames.length).toBe(2);
    expect(frames[0].json).toBe('{"a":1}');
    expect(frames[1].json).toBe('{"a":2}');
    expect(rest).toBe("");
  });

  it("recognizes [DONE]", () => {
    const { frames } = parseStreamChunk("data: [DONE]\n\n");
    expect(frames.length).toBe(1);
    expect(frames[0].done).toBe(true);
  });

  it("returns rest mid-frame", () => {
    const { frames, rest } = parseStreamChunk(`data: {"a":1`);
    expect(frames.length).toBe(0);
    expect(rest).toBe('data: {"a":1');
  });

  it("parses NDJSON", () => {
    const { frames } = parseStreamChunk(`{"a":1}\n{"a":2}\n`);
    expect(frames.length).toBe(2);
    expect(frames[0].json).toBe('{"a":1}');
  });

  it("ignores comment keepalives", () => {
    const buf = `: keepalive\ndata: {"x":1}\n\n`;
    const { frames } = parseStreamChunk(buf);
    expect(frames.length).toBe(1);
    expect(frames[0].json).toBe('{"x":1}');
  });
});
