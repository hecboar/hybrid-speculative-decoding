#!/usr/bin/env python3
"""E6 correlation — Spearman rho between log10(ppl_ratio) and A_4 across models.

PLAN Phase 1 / R2.8 / R3-(n=2): builds the new Table 6 correlation from OUR own
measurements. A_4 sources, in priority order per model:
  1. results/E1/acceptance_new_models.csv (measured this revision), else
  2. v1 Drive pickle acceptance_v3__<m>__k4__T0.0.pkl (acceptance_rate scalar).
ppl_ratio comes from results/E6/ppl_ablation.csv.

Output: results/E6/ppl_spearman.csv (rho, p_value, n, models) +
        results/E6/ppl_vs_acceptance.csv (per-model joined table for the scatter).
CPU only.
"""
import os
import sys
import csv
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
from specdiag import stats  # noqa: E402
from specdiag.io import write_csv  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E6 = os.path.join(REPO, "results", "E6")
DRIVE = os.path.join(REPO, "checkpoints", "v1_drive")
ACC_CSV = os.path.join(REPO, "results", "E1", "acceptance_new_models.csv")
PPL_CSV = os.path.join(E6, "ppl_ablation.csv")
OUT_SP = os.path.join(E6, "ppl_spearman.csv")
OUT_JOIN = os.path.join(E6, "ppl_vs_acceptance.csv")

# map E6/e1 model key -> v1 pickle model_key stem for fallback A_4
PKL_STEM = {"falcon-h1-0.5b": "falcon-h1-0.5b", "falcon-h1-3b": "falcon-h1-3b",
            "qwen3.5-0.8b": "qwen3.5-0.8b", "qwen2.5-0.5b": "qwen2.5-0.5b"}


def a4_from_csv():
    out = {}
    if os.path.exists(ACC_CSV):
        for r in csv.DictReader(open(ACC_CSV)):
            if str(r["k"]) == "4" and str(r["temperature"]) in ("0.0", "0"):
                out[r["model"]] = float(r["A_k"])
    return out


def a4_from_pickle(model):
    stem = PKL_STEM.get(model)
    if not stem:
        return None
    p = os.path.join(DRIVE, f"acceptance_v3__{stem}__k4__T0.0.pkl")
    if not os.path.exists(p):
        return None
    return float(pickle.load(open(p, "rb"))["acceptance_rate"])


def main():
    if not os.path.exists(PPL_CSV):
        raise SystemExit(f"missing {PPL_CSV} — run e6_ppl.py first")
    ppl = {r["model"]: r for r in csv.DictReader(open(PPL_CSV))}
    a4_csv = a4_from_csv()

    rows, xs, ys = [], [], []
    for m, pr in ppl.items():
        ratio = float(pr["ppl_ratio"]) if pr["ppl_ratio"] else None
        if not ratio or ratio <= 0:
            continue
        a4 = a4_csv.get(m)
        src = "measured"
        if a4 is None:
            a4 = a4_from_pickle(m)
            src = "v1_pickle"
        if a4 is None:
            print(f"  {m}: no A_4 -> skipped from correlation")
            continue
        log10r = float(np.log10(ratio))
        rows.append({"model": m, "arch_paradigm": pr["arch_paradigm"],
                     "ppl_full": pr["ppl_full"], "ppl_no_attn": pr["ppl_no_attn"],
                     "ppl_ratio": round(ratio, 4), "log10_ppl_ratio": round(log10r, 4),
                     "A_4": round(a4, 5), "A_4_source": src})
        xs.append(log10r); ys.append(a4)

    write_csv(rows, OUT_JOIN, fieldnames=["model", "arch_paradigm", "ppl_full",
              "ppl_no_attn", "ppl_ratio", "log10_ppl_ratio", "A_4", "A_4_source"])
    if len(xs) >= 3:
        rho, p = stats.spearman(xs, ys)
        write_csv([{"rho": round(rho, 4), "p_value": round(p, 5), "n": len(xs),
                    "x": "log10_ppl_ratio", "y": "A_4",
                    "models": ";".join(r["model"] for r in rows)}],
                  OUT_SP, fieldnames=["rho", "p_value", "n", "x", "y", "models"])
        print(f"Spearman rho={rho:.4f} p={p:.5f} n={len(xs)}")
    else:
        print(f"only {len(xs)} points — need >=3 for Spearman")
    for r in sorted(rows, key=lambda z: z["log10_ppl_ratio"]):
        print(f"  {r['model']:16s} {r['arch_paradigm']:10s} "
              f"log10(ppl_ratio)={r['log10_ppl_ratio']:+.3f}  A_4={r['A_4']:.4f} "
              f"({r['A_4_source']})")
    print("wrote:", OUT_JOIN, OUT_SP)


if __name__ == "__main__":
    main()
