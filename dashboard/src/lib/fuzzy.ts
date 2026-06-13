/**
 * Tiny fuzzy subsequence matcher for the command palette. Pure + testable.
 *
 * A query matches a target if its characters appear in order (not necessarily
 * contiguous). Score rewards contiguous runs, word-boundary hits, and earlier
 * matches so the best command floats to the top.
 */

export interface FuzzyResult {
  matched: boolean;
  score: number;
  /** Indices in the target that were matched — for highlight rendering. */
  indices: number[];
}

export function fuzzyMatch(query: string, target: string): FuzzyResult {
  const q = query.trim().toLowerCase();
  if (q.length === 0) return { matched: true, score: 0, indices: [] };

  const t = target.toLowerCase();
  const indices: number[] = [];
  let score = 0;
  let qi = 0;
  let prevMatch = -2;

  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] !== q[qi]) continue;
    indices.push(ti);
    // Contiguous run bonus.
    if (ti === prevMatch + 1) score += 6;
    // Word-boundary bonus (start, or preceded by a separator).
    if (ti === 0 || /[\s\-_./]/.test(t[ti - 1])) score += 4;
    // Earlier matches score slightly higher.
    score += Math.max(0, 3 - ti * 0.05);
    prevMatch = ti;
    qi++;
  }

  if (qi < q.length) return { matched: false, score: 0, indices: [] };
  // Prefer shorter targets when scores are otherwise close.
  score += Math.max(0, 10 - t.length * 0.1);
  return { matched: true, score, indices };
}

export interface Rankable {
  /** Text the query is matched against (title + optional keywords). */
  haystack: string;
}

/** Filter + rank a list by fuzzy score, preserving original order on ties. */
export function fuzzyRank<T extends Rankable>(query: string, items: T[]): T[] {
  if (query.trim().length === 0) return items;
  return items
    .map((item, i) => ({ item, i, r: fuzzyMatch(query, item.haystack) }))
    .filter((x) => x.r.matched)
    .sort((a, b) => (b.r.score - a.r.score) || (a.i - b.i))
    .map((x) => x.item);
}
