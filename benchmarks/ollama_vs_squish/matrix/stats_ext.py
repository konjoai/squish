"""Pure-Python paired statistics — no numpy/scipy dependency.

The sprint mandates: no headline claim without BOTH a significance test AND an
effect size. This module supplies them for paired samples (the same prompt set
run through both systems), implemented in stdlib only so the result is auditable
and reproducible without a scientific stack:

* ``wilcoxon_signed_rank`` — paired Wilcoxon (exact distribution for small n,
  normal approximation with tie + continuity correction for larger n).
* ``cliffs_delta`` — non-parametric effect size (dominance), with the standard
  magnitude bins.
* ``rank_biserial`` — the effect size that pairs naturally with Wilcoxon.
* ``distribution`` — median, IQR, min/max, and the raw values, so every cell
  ships its full distribution rather than a lone median.

All functions return plain dicts/floats and never raise on empty/degenerate
input — they return ``None`` fields a caller can surface as "insufficient data".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Sequence


# ── helpers ───────────────────────────────────────────────────────────────────


def _median(xs: Sequence[float]) -> float | None:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(xs: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in [0, 100]); matches numpy default."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n == 1:
        return float(s[0])
    rank = (q / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[int(rank)])
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function (stdlib math.erf)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ── distribution summary ──────────────────────────────────────────────────────


def distribution(values: Sequence[float]) -> dict[str, object]:
    """Full distribution summary: median, IQR, spread, and raw values."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
            "mean": None,
            "values": [],
        }
    q1 = _percentile(vals, 25.0)
    q3 = _percentile(vals, 75.0)
    return {
        "n": len(vals),
        "median": _median(vals),
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if (q1 is not None and q3 is not None) else None,
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "values": vals,
    }


# ── Wilcoxon signed-rank (paired) ─────────────────────────────────────────────


def _rank_with_ties(magnitudes: list[float]) -> list[float]:
    """Average ranks (1-based) for ascending magnitudes, ties share mean rank."""
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    ranks = [0.0] * len(magnitudes)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and magnitudes[order[j + 1]] == magnitudes[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _exact_two_sided_p(w_plus: float, n: int) -> float:
    """Exact two-sided p from the full sign-permutation null (no ties).

    Enumerates all 2**n sign assignments of ranks 1..n; counts how many give a
    W+ at least as extreme as observed. Used only for small n (<= 20) where the
    2**n enumeration is cheap and the normal approximation is unreliable.
    """
    ranks = list(range(1, n + 1))
    total = 0.0
    count = 0
    mean = n * (n + 1) / 4.0
    obs_dev = abs(w_plus - mean)
    for signs in product((0, 1), repeat=n):
        wp = sum(r for r, s in zip(ranks, signs) if s)
        if abs(wp - mean) >= obs_dev - 1e-9:
            count += 1
        total += 1
    return min(1.0, count / total)


@dataclass
class WilcoxonResult:
    n_effective: int = 0
    n_zero_dropped: int = 0
    w_plus: float | None = None
    w_minus: float | None = None
    statistic: float | None = None  # min(W+, W-)
    z: float | None = None
    p_value: float | None = None
    method: str = "none"


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
    *,
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test on differences a-b.

    ``a`` and ``b`` are paired samples (same prompt set, both systems). Zero
    differences are dropped (``wilcox`` convention). Returns ``method='exact'``
    for n<=20 with no ties, otherwise the tie-corrected normal approximation
    with continuity correction.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples must match length: {len(a)} != {len(b)}")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0.0]
    n_zero = len(diffs) - len(nonzero)
    n = len(nonzero)
    if zero_method != "wilcox":
        raise ValueError("only the 'wilcox' zero_method is implemented")
    if n == 0:
        return WilcoxonResult(n_effective=0, n_zero_dropped=n_zero, method="degenerate")

    mags = [abs(d) for d in nonzero]
    ranks = _rank_with_ties(mags)
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    statistic = min(w_plus, w_minus)

    has_ties = len(set(mags)) != len(mags)
    if n <= 20 and not has_ties:
        p = _exact_two_sided_p(w_plus, n)
        return WilcoxonResult(n, n_zero, w_plus, w_minus, statistic, None, p, "exact")

    mean_w = n * (n + 1) / 4.0
    # tie correction term for the variance
    tie_term = 0.0
    mags_sorted = sorted(mags)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and mags_sorted[j + 1] == mags_sorted[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_term += t**3 - t
        i = j + 1
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return WilcoxonResult(n, n_zero, w_plus, w_minus, statistic, None, 1.0, "degenerate")
    cc = 0.5  # continuity correction
    z = (statistic - mean_w + cc) / math.sqrt(var_w)
    p = 2.0 * _normal_cdf(z)  # statistic <= mean, so z<=0 -> lower tail, doubled
    p = min(1.0, max(0.0, p))
    return WilcoxonResult(n, n_zero, w_plus, w_minus, statistic, z, p, "normal_approx")


# ── effect sizes ──────────────────────────────────────────────────────────────


def _cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> dict[str, object]:
    """Cliff's delta dominance effect size for two samples (need not be paired).

    delta = (#(a>b) - #(a<b)) / (len(a)*len(b)), in [-1, 1]. Sign convention:
    positive means ``a`` tends to exceed ``b``.
    """
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if not av or not bv:
        return {"delta": None, "magnitude": "none"}
    gt = lt = 0
    for x in av:
        for y in bv:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    delta = (gt - lt) / (len(av) * len(bv))
    return {"delta": delta, "magnitude": _cliffs_magnitude(delta)}


def rank_biserial(res: WilcoxonResult) -> float | None:
    """Matched-pairs rank-biserial correlation from a Wilcoxon result.

    r = (W+ - W-) / (W+ + W-), in [-1, 1]. Positive => a>b dominates.
    """
    if res.w_plus is None or res.w_minus is None:
        return None
    denom = res.w_plus + res.w_minus
    if denom == 0:
        return 0.0
    return (res.w_plus - res.w_minus) / denom


# ── paired comparison bundle ──────────────────────────────────────────────────


@dataclass
class PairedComparison:
    """Everything a headline claim needs: distributions + significance + effect."""

    metric: str
    a_label: str
    b_label: str
    a: dict[str, object] = field(default_factory=dict)
    b: dict[str, object] = field(default_factory=dict)
    wilcoxon: dict[str, object] = field(default_factory=dict)
    cliffs_delta: float | None = None
    cliffs_magnitude: str = "none"
    rank_biserial: float | None = None
    median_ratio: float | None = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        p = self.wilcoxon.get("p_value")
        return p is not None and p < alpha


def compare_paired(
    metric: str,
    a_label: str,
    b_label: str,
    a: Sequence[float],
    b: Sequence[float],
) -> PairedComparison:
    """Full paired comparison for one metric: a vs b on identical prompt sets."""
    da = distribution(a)
    db = distribution(b)
    wl = wilcoxon_signed_rank(a, b)
    cd = cliffs_delta(a, b)
    ma, mb = da.get("median"), db.get("median")
    ratio = (
        (ma / mb)
        if (isinstance(ma, (int, float)) and isinstance(mb, (int, float)) and mb not in (0, None))
        else None
    )
    return PairedComparison(
        metric=metric,
        a_label=a_label,
        b_label=b_label,
        a=da,
        b=db,
        wilcoxon={
            "n_effective": wl.n_effective,
            "n_zero_dropped": wl.n_zero_dropped,
            "statistic": wl.statistic,
            "w_plus": wl.w_plus,
            "w_minus": wl.w_minus,
            "z": wl.z,
            "p_value": wl.p_value,
            "method": wl.method,
        },
        cliffs_delta=cd["delta"],
        cliffs_magnitude=str(cd["magnitude"]),
        rank_biserial=rank_biserial(wl),
        median_ratio=ratio,
    )
