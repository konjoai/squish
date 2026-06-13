import { describe, it, expect } from "vitest";
import { fuzzyMatch, fuzzyRank } from "./fuzzy";

describe("fuzzyMatch", () => {
  it("matches an empty query against anything", () => {
    expect(fuzzyMatch("", "anything").matched).toBe(true);
  });

  it("matches a subsequence and records indices", () => {
    const r = fuzzyMatch("agt", "Go to agent");
    expect(r.matched).toBe(true);
    expect(r.indices.length).toBe(3);
  });

  it("rejects a non-subsequence", () => {
    expect(fuzzyMatch("zzz", "quality").matched).toBe(false);
  });

  it("is case-insensitive", () => {
    expect(fuzzyMatch("KV", "kv cache").matched).toBe(true);
  });

  it("scores a contiguous prefix higher than a scattered match", () => {
    const contiguous = fuzzyMatch("tok", "tokenizer");
    const scattered = fuzzyMatch("tok", "the other kind");
    expect(contiguous.score).toBeGreaterThan(scattered.score);
  });
});

describe("fuzzyRank", () => {
  const items = [
    { haystack: "Go to agent section agent" },
    { haystack: "Go to quality section quality" },
    { haystack: "Go to kv cache section kv" },
  ];

  it("returns everything for an empty query, in order", () => {
    expect(fuzzyRank("", items)).toEqual(items);
  });

  it("filters to fuzzy matches", () => {
    const out = fuzzyRank("quality", items);
    expect(out).toHaveLength(1);
    expect(out[0].haystack).toContain("quality");
  });

  it("ranks the best match first", () => {
    const out = fuzzyRank("kv", items);
    expect(out[0].haystack).toContain("kv");
  });
});
