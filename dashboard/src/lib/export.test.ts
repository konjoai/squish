import { describe, it, expect } from "vitest";
import { conversationToMarkdown, conversationToJSON } from "./export";
import type { ChatTurn } from "./types";

const AT = Date.UTC(2026, 0, 2, 3, 4, 5); // fixed timestamp for determinism

const turns: ChatTurn[] = [
  { id: "u1", role: "user", content: "What is INT4?" },
  { id: "a1", role: "assistant", content: "A 4-bit quantization.", ttftS: 0.14, totalS: 1.2, tokens: [{ text: "A", intervalMs: 1, tps: 1, atMs: 1 }] },
];

describe("conversationToMarkdown", () => {
  it("includes a header with the turn count and ISO date", () => {
    const md = conversationToMarkdown(turns, AT);
    expect(md).toContain("# squish conversation");
    expect(md).toContain("2 turns");
    expect(md).toContain("2026-01-02T03:04:05");
  });

  it("renders role labels and content", () => {
    const md = conversationToMarkdown(turns, AT);
    expect(md).toContain("**user**");
    expect(md).toContain("What is INT4?");
    expect(md).toContain("**assistant**");
  });

  it("annotates assistant timing metadata", () => {
    const md = conversationToMarkdown(turns, AT);
    expect(md).toContain("ttft 0.14s");
    expect(md).toContain("1 tok");
  });

  it("singularizes a one-turn export", () => {
    const md = conversationToMarkdown([turns[0]], AT);
    expect(md).toMatch(/1 turn(?!s)/);
    expect(md).not.toContain("1 turns");
  });
});

describe("conversationToJSON", () => {
  it("produces parseable JSON with structured turns", () => {
    const obj = JSON.parse(conversationToJSON(turns, AT));
    expect(obj.turn_count).toBe(2);
    expect(obj.exported_at).toBe("2026-01-02T03:04:05.000Z");
    expect(obj.turns[0]).toEqual({ role: "user", content: "What is INT4?" });
    expect(obj.turns[1].ttft_s).toBe(0.14);
    expect(obj.turns[1].token_count).toBe(1);
  });

  it("omits timing keys when absent", () => {
    const obj = JSON.parse(conversationToJSON([turns[0]], AT));
    expect(obj.turns[0]).not.toHaveProperty("ttft_s");
    expect(obj.turns[0]).not.toHaveProperty("token_count");
  });
});
