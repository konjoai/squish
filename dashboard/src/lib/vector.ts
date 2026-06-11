/**
 * Vector helpers for the Embeddings Explorer. Pure functions — no React, no
 * fetch — so the similarity math is fully unit-testable.
 */

/** Cosine similarity of two equal-length vectors. Returns 0 for degenerate input. */
export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

/** Full N×N cosine-similarity matrix for a list of vectors. */
export function similarityMatrix(vectors: number[][]): number[][] {
  return vectors.map((a) => vectors.map((b) => cosineSimilarity(a, b)));
}
