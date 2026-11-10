#!/usr/bin/env python3
"""E0.6 — Normalize v1 results to the corrected Section 6 metric set.

Reads the immutable v1 baseline CSVs in results/v1/ and emits
results/E0/v1_metrics_normalized.csv, the single basis file for every revised
v1 acceptance/speedup table. No GPU, no third-party deps (stdlib csv only).

Section 6 (PLAN) definitions applied here:
  - A_k        : v1's "acceptance_rate" column, renamed. NOTE: numerically this
                 equals tau_bar / k for ALL 18 v1 conditions (verified, 0 error),
                 i.e. v1's alpha is the MEAN PER-TOKEN accepted fraction
                 (E[na]/k), not literally "fraction of rounds where all k tokens
                 are accepted". The revised manuscript must state this estimator
                 explicitly (see results/E0/README_e06.md).
  - tau_bar    : mean accepted draft tokens per round = v1 "mean_accepted_length".
  - S_theory   : (1 + tau_bar) / (1 + k * c), c = c_param (flop_ratio). This
                 REPLACES the paper's textual geometric-alpha "Eq. 1". The v1
                 theoretical_speedup.csv column already used this exact formula
                 (verified: reproduces all 18 rows with 0 error, incl. the
                 Falcon-H1-0.5B k=2 -> 0.918779 value).
  - break-even : S_theory > 1  <=>  tau_bar / k > c  <=>  A_k > c_param.
"""
import csv
import os

V1 = os.path.join(os.path.dirname(__file__), "..", "results", "v1")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "E0",
                   "v1_metrics_normalized.csv")

ARCH_PARADIGM = {
    "falcon-h1-0.5b": "parallel",
    "qwen3.5-0.8b": "sequential",
    "qwen2.5-0.5b": "transformer",
}

REF_VALUE = ("falcon-h1-0.5b", "2", "0.0", 0.9187786259541987)  # E0.6 sanity check


def load(name):
    with open(os.path.join(V1, name)) as f:
        return list(csv.DictReader(f))


def main():
    accept = load("acceptance_rate.csv")
    theo = {(r["model"], r["k"], r["temperature"]): r
            for r in load("theoretical_speedup.csv")}

    fields = ["model", "arch_paradigm", "strategy", "k", "temperature",
              "A_k", "ci_lo", "ci_hi", "tau_bar", "c_param",
              "s_theory", "break_even_ratio", "break_even_met",
              "reproduces_v1_theo", "n_prompts"]

    out_rows = []
    max_theo_err = 0.0
    ref_ok = None
    for r in accept:
        model = r["model"]
        k = int(r["k"])
        temp = r["temperature"]
        A_k = float(r["acceptance_rate"])
        tau_bar = float(r["mean_accepted_length"])
        tk = theo[(model, r["k"], temp)]
        c_param = float(tk["flop_ratio"])
        s_theory = (1.0 + tau_bar) / (1.0 + k * c_param)
        stored = float(tk["theoretical_speedup"])
        err = abs(s_theory - stored)
        max_theo_err = max(max_theo_err, err)
        break_even_ratio = tau_bar / k
        if (model, r["k"], temp) == REF_VALUE[:3]:
            ref_ok = abs(s_theory - REF_VALUE[3]) < 1e-9
        out_rows.append({
            "model": model,
            "arch_paradigm": ARCH_PARADIGM.get(model, "unknown"),
            "strategy": r["strategy"],
            "k": k,
            "temperature": temp,
            "A_k": f"{A_k:.6g}",
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "tau_bar": f"{tau_bar:.6g}",
            "c_param": f"{c_param:.10g}",
            "s_theory": f"{s_theory:.10g}",
            "break_even_ratio": f"{break_even_ratio:.6g}",
            "break_even_met": "yes" if break_even_ratio > c_param else "no",
            "reproduces_v1_theo": "yes" if err < 1e-9 else f"NO({err:.2e})",
            "n_prompts": r["n_trials"],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # E0.6 verification gate
    print(f"wrote {OUT}  ({len(out_rows)} rows)")
    print(f"max |S_theory - stored v1 theoretical_speedup| over 18 rows = {max_theo_err:.2e}")
    print(f"reference Falcon-H1-0.5B k=2 T=0 reproduces 0.918779 : {ref_ok}")
    n_be = sum(1 for r in out_rows if r["break_even_met"] == "yes")
    print(f"conditions meeting break-even (S_theory>1): {n_be}/{len(out_rows)}")
    assert max_theo_err < 1e-9, "S_theory does NOT reproduce v1 -- investigate"
    assert ref_ok, "0.919 reference reconstruction FAILED"
    print("E0.6 verification PASSED")


if __name__ == "__main__":
    main()
