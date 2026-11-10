"""stats.py - statistical battery (E5.4).

`bootstrap_ci` is ported verbatim from notebook Cell 5 (the only stat the notebook
had). Everything else is NEW for the revision. scipy is imported LAZILY inside
each function that needs it, so this module imports on a host without scipy
(numpy-only functions still work).
"""
import numpy as np

BOOTSTRAP_N = 10_000


def bootstrap_ci(arr, n_boot=BOOTSTRAP_N, ci=0.95, seed=None):
    """Percentile bootstrap CI (Cell 5). Returns (mean, lo, hi)."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        m = float(np.mean(arr)) if len(arr) else 0.0
        return m, m, m
    rng = np.random.default_rng(seed) if seed is not None else np.random
    means = np.array([np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


def permutation_test(a, b, n=10_000, seed=0):
    """Two-sample permutation test on difference of means (E5.3, R2.3).

    Returns (observed_diff, p_value). Two-sided. Suitable for per-prompt
    acceptance indicators / accepted-length vectors (unpaired).
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = pooled[:na].mean() - pooled[na:].mean()
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return float(obs), (count + 1) / (n + 1)


def wilcoxon_signed_rank(x, y):
    """Paired Wilcoxon signed-rank (E5.4). Returns (statistic, p_value)."""
    from scipy import stats
    res = stats.wilcoxon(np.asarray(x), np.asarray(y))
    return float(res.statistic), float(res.pvalue)


def friedman_holm(*groups):
    """Friedman test + Holm-corrected post-hoc pairwise Wilcoxon (E5.4).

    groups: >=3 equal-length paired samples. Returns dict with friedman stat/p and
    a list of (i, j, stat, p_raw, p_holm) post-hoc rows.
    """
    from scipy import stats
    from itertools import combinations
    arrs = [np.asarray(g, dtype=float) for g in groups]
    fr = stats.friedmanchisquare(*arrs)
    pairs = list(combinations(range(len(arrs)), 2))
    raw = []
    for i, j in pairs:
        w = stats.wilcoxon(arrs[i], arrs[j])
        raw.append((i, j, float(w.statistic), float(w.pvalue)))
    # Holm correction
    order = sorted(range(len(raw)), key=lambda idx: raw[idx][3])
    m = len(raw)
    holm = [None] * len(raw)
    prev = 0.0
    for rank, idx in enumerate(order):
        p_adj = min(1.0, (m - rank) * raw[idx][3])
        p_adj = max(p_adj, prev)
        holm[idx] = p_adj
        prev = p_adj
    posthoc = [(raw[k][0], raw[k][1], raw[k][2], raw[k][3], holm[k]) for k in range(len(raw))]
    return {"friedman_stat": float(fr.statistic), "friedman_p": float(fr.pvalue),
            "posthoc": posthoc}


def spearman(x, y):
    """Spearman rho + p (E6 PPL-ratio vs A_4). Returns (rho, p_value)."""
    from scipy import stats
    res = stats.spearmanr(np.asarray(x), np.asarray(y))
    return float(res.statistic), float(res.pvalue)


def mcnemar(b, c):
    """McNemar test on the two discordant cell counts b, c (P2, per-position top-1).

    Returns (statistic, p_value). Uses the exact binomial for small b+c, else the
    continuity-corrected chi-square (1 dof).
    """
    b = int(b); c = int(c)
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        from scipy import stats
        k = min(b, c)
        p = 2 * stats.binom.cdf(k, n, 0.5)
        return float(min(b, c)), float(min(1.0, p))
    stat = (abs(b - c) - 1) ** 2 / n
    from scipy import stats
    p = float(stats.chi2.sf(stat, 1))
    return float(stat), p


def power_note(alpha, n):
    """CI-width note to justify n (R2.2 supplement).

    Returns dict with the normal-approx 95% half-width for a mean-of-Bernoulli-ish
    acceptance estimate and the implied full CI width. `alpha` is the observed
    acceptance rate, `n` the prompt count.
    """
    alpha = float(alpha); n = int(n)
    half = 1.96 * np.sqrt(max(alpha * (1 - alpha), 1e-12) / n)
    return {"alpha": alpha, "n": n, "ci_half_width_normal": float(half),
            "ci_width_normal": float(2 * half)}
