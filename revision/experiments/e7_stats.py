#!/usr/bin/env python3
"""E5.4 / E7 — statistical battery (PLAN Phase 5; EiC-6).

Consumes the per-prompt accepted-length pickles produced by E3.1 (Falcon
partial-drafter variants, n=150 val-test) and E2 (Qwen attn_subset variants,
n=500) — both use the deterministic val-test ordering, so the per-prompt vectors
are PAIRED across strategies for a given model+k. Runs:
  * Wilcoxon signed-rank: two strategies on the SAME model+prompts.
  * Friedman + Holm post-hoc Wilcoxon: >=3 strategies on the same prompts.
  * Spearman: PPL ratio vs A_4 across models (reads E6 output).
McNemar (per-position top-1) is P2 and omitted unless the data exists.

Output: results/E7/stat_tests.csv (test, comparison, n, statistic, p_value,
effect_size). CPU only.
"""
import os
import sys
import csv
import glob
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
from specdiag import stats  # noqa: E402
from specdiag.io import write_csv  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E7 = os.path.join(REPO, "results", "E7")
OUT = os.path.join(E7, "stat_tests.csv")


def load_pp(path):
    return np.asarray(pickle.load(open(path, "rb"))["accepted_lengths"], float)


def find(dirpath, pattern):
    return sorted(glob.glob(os.path.join(REPO, dirpath, pattern)))


def rank_biserial(a, b):
    """Matched-pairs rank-biserial effect size for Wilcoxon (a vs b)."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    rp = r[d > 0].sum(); rn = r[d < 0].sum()
    tot = rp + rn
    return float((rp - rn) / tot) if tot > 0 else 0.0


def main():
    os.makedirs(E7, exist_ok=True)
    rows = []

    # ---- Friedman + Holm across E3.1 Falcon variants (k=2) ----
    for exp, subdir, k in [("E3.1-falcon", "results/E3/perprompt", 2),
                           ("E2-qwen", "results/E2/perprompt", 2)]:
        files = find(subdir, f"*k{k}.pkl") if exp.startswith("E3") else find(subdir, f"*__k{k}.pkl")
        # keep only files whose vectors share the modal length (paired)
        vecs = {}
        for f in files:
            v = load_pp(f)
            vecs[os.path.basename(f).replace(f"__k{k}.pkl", "").replace(f"_k{k}.pkl", "")] = v
        if len(vecs) >= 3:
            lens = [len(v) for v in vecs.values()]
            n = max(set(lens), key=lens.count)
            names = [nm for nm, v in vecs.items() if len(v) == n]
            groups = [vecs[nm][:n] for nm in names]
            fr = stats.friedman_holm(*groups)
            W = fr["friedman_stat"] / (n * (len(groups) - 1)) if n and len(groups) > 1 else float("nan")
            rows.append({"test": "friedman", "comparison": f"{exp}:{'|'.join(names)}",
                         "n": n, "statistic": round(fr["friedman_stat"], 4),
                         "p_value": round(fr["friedman_p"], 6), "effect_size": round(W, 4)})
            for i, j, w, praw, pholm in fr["posthoc"]:
                rows.append({"test": "wilcoxon_holm_posthoc",
                             "comparison": f"{exp}:{names[i]}_vs_{names[j]}", "n": n,
                             "statistic": round(w, 4), "p_value": round(pholm, 6),
                             "effect_size": round(rank_biserial(groups[i], groups[j]), 4)})
            print(f"  Friedman {exp}: chi2={fr['friedman_stat']:.3f} p={fr['friedman_p']:.4g} "
                  f"(n={n}, {len(groups)} strategies)")

    # ---- targeted Wilcoxon pairs (ssm/linear-only vs layerskip) ----
    pairs = [
        ("falcon ssm_only vs layerskip", "results/E3/perprompt/component_only_1.0__k2.pkl",
         "results/E3/perprompt/layerskip_0.33__k2.pkl"),
        ("qwen linear-only vs layerskip", "results/E2/perprompt/attn_subset_keep0__k2.pkl",
         "results/E2/perprompt/layerskip_0.33__k2.pkl"),
    ]
    for label, pa, pb in pairs:
        fa, fb = os.path.join(REPO, pa), os.path.join(REPO, pb)
        if os.path.exists(fa) and os.path.exists(fb):
            a, b = load_pp(fa), load_pp(fb)
            n = min(len(a), len(b)); a, b = a[:n], b[:n]
            try:
                w, p = stats.wilcoxon_signed_rank(a, b)
            except Exception as e:
                print(f"  wilcoxon {label}: skipped ({type(e).__name__})"); continue
            rows.append({"test": "wilcoxon", "comparison": label, "n": n,
                         "statistic": round(w, 4), "p_value": round(p, 6),
                         "effect_size": round(rank_biserial(a, b), 4)})
            print(f"  Wilcoxon {label}: W={w:.1f} p={p:.4g} n={n}")

    # ---- Spearman from E6 ----
    e6 = os.path.join(REPO, "results/E6/ppl_vs_acceptance.csv")
    if os.path.exists(e6):
        r = list(csv.DictReader(open(e6)))
        x = [float(z["log10_ppl_ratio"]) for z in r]; y = [float(z["A_4"]) for z in r]
        rho, p = stats.spearman(x, y)
        rows.append({"test": "spearman", "comparison": "log10_ppl_ratio_vs_A4_allmodels",
                     "n": len(x), "statistic": round(rho, 4), "p_value": round(p, 6),
                     "effect_size": round(rho, 4)})
        print(f"  Spearman ppl_ratio vs A_4: rho={rho:.4f} p={p:.4g} n={len(x)}")

    write_csv(rows, OUT, fieldnames=["test", "comparison", "n", "statistic",
                                     "p_value", "effect_size"])
    print("wrote:", OUT, f"({len(rows)} tests)")


if __name__ == "__main__":
    main()
