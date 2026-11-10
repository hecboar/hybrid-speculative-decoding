#!/usr/bin/env python3
"""E5.3 — two-sample permutation test, Falcon-0.5B vs 3B (PLAN Phase 5; R2.3).

Replaces v1's "statistically indistinguishable (overlapping CIs)" wording with a
proper 10k-permutation test on per-prompt accepted-length vectors (k=4, T=0).
Both vectors are produced by the SAME pipeline at n=150 (Falcon-3B was re-run
because its v1 Drive pickle is aggregate-only — see results/E0/checkpoint_inventory.md).
CPU only.
Output: results/E5/permutation_0.5b_vs_3b.csv
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
from specdiag import stats  # noqa: E402
from specdiag.io import write_csv  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P05 = os.path.join(REPO, "results/E1/perprompt/falcon-h1-0.5b__component_only__k4__T0.0.pkl")
P3B = os.path.join(REPO, "results/E1/perprompt/falcon-h1-3b__component_only__k4__T0.0.pkl")
OUT = os.path.join(REPO, "results/E5/permutation_0.5b_vs_3b.csv")


def main():
    a = np.array(pickle.load(open(P05, "rb"))["accepted_lengths"], float)
    b = np.array(pickle.load(open(P3B, "rb"))["accepted_lengths"], float)
    obs, p = stats.permutation_test(a, b, n=10000, seed=0)
    row = [{"comparison": "falcon-0.5b_vs_3b", "metric": "accepted_length_k4_T0",
            "n_a": len(a), "n_b": len(b), "mean_a": round(a.mean(), 4),
            "mean_b": round(b.mean(), 4), "observed_diff": round(obs, 4),
            "p_value": round(p, 4), "n_perm": 10000,
            "verdict": "not significant" if p > 0.05 else "significant"}]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_csv(row, OUT)
    print(f"0.5b mean_na={a.mean():.4f}  3b mean_na={b.mean():.4f}  "
          f"diff={obs:.4f}  p={p:.4f}  -> "
          f"{'INDISTINGUISHABLE' if p > 0.05 else 'DIFFERENT'}")
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
