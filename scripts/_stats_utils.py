"""Shared statistical utilities for F4c analysis modules.

Functions:
- bootstrap_ci(values, n_iter, ci) → (mean, lo, hi)
- cohens_d(a, b) → effect size
- cliffs_delta(a, b) → non-parametric effect size
- wilcoxon_signed_rank(a, b) → (W, p)
- mann_whitney_u(a, b) → (U, p)
- normal_cdf(z)
- summarize(values) → dict with mean/median/std/min/max/p25/p75/p95/p99
"""
import math
import random
import statistics


def bootstrap_ci(values, n_iter=10000, ci=0.95):
    """Returns (mean, lower, upper) bootstrap CI."""
    if not values: return (0, 0, 0)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample)/n)
    means.sort()
    lo = means[int(n_iter * (1-ci)/2)]
    hi = means[int(n_iter * (1 - (1-ci)/2))]
    return (statistics.mean(values), lo, hi)


def cohens_d(a, b):
    """Effect size between two arrays (parametric)."""
    if not a or not b: return 0
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = statistics.stdev(a) if len(a) > 1 else 0
    sb = statistics.stdev(b) if len(b) > 1 else 0
    pooled = math.sqrt((sa**2 + sb**2) / 2)
    return (ma - mb) / pooled if pooled > 0 else 0


def cliffs_delta(a, b):
    """Non-parametric effect size, range [-1, +1].
    Interpretation: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large."""
    if not a or not b: return 0
    n_a, n_b = len(a), len(b)
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (n_a * n_b)


def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def wilcoxon_signed_rank(a, b):
    """Paired Wilcoxon signed-rank test. Returns (W, p_two_tailed)."""
    if len(a) != len(b) or not a:
        return (0, 1.0)
    diffs = [(ai - bi) for ai, bi in zip(a, b) if ai != bi]
    if not diffs:
        return (0, 1.0)
    n = len(diffs)
    abs_diffs = sorted([(abs(d), 1 if d > 0 else -1) for d in diffs])
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    W_plus = sum(ranks[i] for i in range(n) if abs_diffs[i][1] > 0)
    W_minus = sum(ranks[i] for i in range(n) if abs_diffs[i][1] < 0)
    W = min(W_plus, W_minus)
    if n > 20:
        mu = n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2*n + 1) / 24)
        z = (W - mu) / sigma
        p = 2 * (1 - normal_cdf(abs(z)))
        return (W, p)
    else:
        return (W, 0.5)


def mann_whitney_u(a, b):
    """Mann-Whitney U test (independent samples). Returns (U, p_two_tailed)."""
    if not a or not b: return (0, 1.0)
    n1, n2 = len(a), len(b)
    combined = sorted([(v, 'a') for v in a] + [(v, 'b') for v in b])
    ranks = {'a': [], 'b': []}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[combined[k][1]].append(avg_rank)
        i = j
    R1 = sum(ranks['a'])
    U1 = R1 - n1*(n1+1)/2
    U2 = n1*n2 - U1
    U = min(U1, U2)
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0: return (U, 1.0)
    z = (U - mu) / sigma
    p = 2 * (1 - normal_cdf(abs(z)))
    return (U, p)


def summarize(values):
    """Returns dict with descriptive stats."""
    if not values: return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": round(statistics.mean(s), 4),
        "median": round(statistics.median(s), 4),
        "std": round(statistics.stdev(s), 4) if n > 1 else 0,
        "min": round(min(s), 4),
        "max": round(max(s), 4),
        "p25": round(s[n//4], 4),
        "p75": round(s[3*n//4], 4),
        "p95": round(s[int(n*0.95)], 4),
        "p99": round(s[int(n*0.99)], 4),
    }


def bonferroni(p_values):
    """Apply Bonferroni correction. Returns list of corrected p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def paired_test_full(ours_vals, baseline_vals, n_bootstrap=10000):
    """Full paired comparison: delta, CI, Cohen's d, Cliff's delta, Wilcoxon, p."""
    if not ours_vals or not baseline_vals or len(ours_vals) != len(baseline_vals):
        return None
    delta = statistics.mean(ours_vals) - statistics.mean(baseline_vals)
    diffs = [a - b for a, b in zip(ours_vals, baseline_vals)]
    delta_mean, delta_lo, delta_hi = bootstrap_ci(diffs, n_iter=n_bootstrap)
    d = cohens_d(ours_vals, baseline_vals)
    cd = cliffs_delta(ours_vals, baseline_vals)
    W, p = wilcoxon_signed_rank(ours_vals, baseline_vals)
    return {
        "n": len(ours_vals),
        "ours_mean": round(statistics.mean(ours_vals), 4),
        "baseline_mean": round(statistics.mean(baseline_vals), 4),
        "delta": round(delta, 4),
        "delta_ci_95": [round(delta_lo, 4), round(delta_hi, 4)],
        "cohens_d": round(d, 4),
        "cliffs_delta": round(cd, 4),
        "wilcoxon_W": W,
        "p_value": round(p, 6),
    }
