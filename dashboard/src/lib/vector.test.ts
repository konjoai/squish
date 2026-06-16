import { describe, it, expect } from "vitest";
import { cosineSimilarity, similarityMatrix } from "./vector";

describe("cosineSimilarity", () => {
  it("is 1 for identical vectors", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 6);
  });

  it("is 0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 6);
  });

  it("is -1 for opposite vectors", () => {
    expect(cosineSimilarity([1, 1], [-1, -1])).toBeCloseTo(-1, 6);
  });

  it("returns 0 for a zero vector (degenerate)", () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });

  it("returns 0 for empty input", () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });

  it("is scale-invariant", () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 6);
  });
});

describe("similarityMatrix", () => {
  it("produces a symmetric matrix with a unit diagonal", () => {
    const m = similarityMatrix([[1, 0], [0, 1], [1, 1]]);
    expect(m).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
      expect(m[i][i]).toBeCloseTo(1, 6);
      for (let j = 0; j < 3; j++) expect(m[i][j]).toBeCloseTo(m[j][i], 6);
    }
  });
});
