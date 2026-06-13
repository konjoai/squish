import { describe, it, expect, beforeEach } from "vitest";
import { loadTurns, saveTurns, clearTurns } from "./persist";
import type { ChatTurn } from "./types";

const KEY = "squish.cockpit.conversation.v1";

const turn = (id: string, content = "hi"): ChatTurn => ({ id, role: "user", content });

describe("conversation persistence", () => {
  beforeEach(() => { localStorage.clear(); });

  it("returns [] when nothing is stored", () => {
    expect(loadTurns()).toEqual([]);
  });

  it("round-trips a saved conversation", () => {
    const turns: ChatTurn[] = [turn("u1"), { id: "a1", role: "assistant", content: "yo" }];
    saveTurns(turns);
    expect(loadTurns()).toEqual(turns);
  });

  it("clearTurns removes the conversation", () => {
    saveTurns([turn("u1")]);
    clearTurns();
    expect(loadTurns()).toEqual([]);
  });

  it("drops malformed JSON without throwing", () => {
    localStorage.setItem(KEY, "{not valid json");
    expect(loadTurns()).toEqual([]);
  });

  it("filters out structurally invalid turns", () => {
    localStorage.setItem(KEY, JSON.stringify([
      turn("ok"),
      { id: 5, role: "user", content: "bad id" },
      { id: "x", role: "wizard", content: "bad role" },
      { id: "y", role: "assistant" },
    ]));
    const loaded = loadTurns();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].id).toBe("ok");
  });

  it("ignores a non-array payload", () => {
    localStorage.setItem(KEY, JSON.stringify({ not: "an array" }));
    expect(loadTurns()).toEqual([]);
  });

  it("caps the stored history at 200 turns", () => {
    const many = Array.from({ length: 250 }, (_, i) => turn(`u${i}`));
    saveTurns(many);
    const loaded = loadTurns();
    expect(loaded).toHaveLength(200);
    expect(loaded[loaded.length - 1].id).toBe("u249");
  });
});
