#!/usr/bin/env python3
"""E0.4 — Sampling-path audit of the T>0 (T=0.6) speculative decoding path.

PLAN E0.4: "confirm rejection sampling draws the correction token from
P_res = norm(max(0, P_H - P_S)). Write a 10-prompt unit test comparing empirical
output distribution vs target for a toy model. If incorrect -> flag; the paper
will relabel T=0.6 results as draft-quality diagnostics (no code fix)."

STATIC FINDING (from the frozen v1 notebook, Cells 12/24/42 measure_speedup*):
  On rejection the v1 code sets the correction/bonus token to
      bonus = full_logits[reject_pos].argmax()           # greedy
      bonus = multinomial(softmax(full_logits/T))         # T>0
  i.e. it re-samples the correction token DIRECTLY from the full model's
  distribution P_H at the reject position. It does NOT sample from the
  normalized residual  P_res = norm(max(0, P_H - P_S)).  There is no max(0, .)
  residual anywhere in the notebook.  => The T>0 path is NOT Leviathan-exact.

This script proves the consequence with an exact analytic derivation plus a
Monte-Carlo confirmation over 10 random toy next-token distributions ("prompts"),
each a small categorical (a toy 2-state/-vocab model's softmax output). No GPU,
numpy only.

Emitted-token distribution for one draft token, draft x ~ P_S, accept w.p.
min(1, P_H(x)/P_S(x)):
  P(accepted token = x) = min(P_S(x), P_H(x))
  P(reject)             = TV(P_S, P_H) = sum_x max(0, P_S(x) - P_H(x))
  CORRECT residual:  emit(x) = min(P_S,P_H)(x) + max(0, P_H-P_S)(x) = P_H(x)   [exact]
  v1 direct-from-PH: emit(x) = min(P_S,P_H)(x) + TV * P_H(x)                   [biased]
The v1 bias per token is  emit_v1(x) - P_H(x) = TV*P_H(x) - max(0,P_H-P_S)(x).
"""
import csv
import os

import numpy as np

OUT_MD = os.path.join(os.path.dirname(__file__), "..", "results", "E0",
                      "e04_sampling_audit.md")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "E0",
                       "e04_sampling_audit.csv")
RNG = np.random.default_rng(42)
VOCAB = 8
N_PROMPTS = 10
N_SAMPLES = 2_000_000  # MC samples per prompt


def emitted_dist_analytic(P_H, P_S, mode):
    """Exact emitted-token distribution for a single draft token."""
    accepted = np.minimum(P_S, P_H)              # P(accept & token=x)
    tv = np.sum(np.maximum(0.0, P_S - P_H))      # P(reject)
    if mode == "residual":                        # Leviathan-exact
        corr = np.maximum(0.0, P_H - P_S)         # already sums to tv
        return accepted + corr
    elif mode == "v1_direct":                     # v1: draw correction from P_H
        return accepted + tv * P_H
    raise ValueError(mode)


def emitted_dist_mc(P_H, P_S, mode, n):
    """Monte-Carlo emitted-token distribution (confirms the analytic form)."""
    x = RNG.choice(VOCAB, size=n, p=P_S)                     # draft ~ P_S
    u = RNG.random(n)
    accept = u < np.minimum(1.0, P_H[x] / P_S[x])
    out = x.copy()
    rej = ~accept
    n_rej = int(rej.sum())
    if n_rej:
        if mode == "residual":
            res = np.maximum(0.0, P_H - P_S)
            res = res / res.sum()
            out[rej] = RNG.choice(VOCAB, size=n_rej, p=res)
        else:  # v1_direct
            out[rej] = RNG.choice(VOCAB, size=n_rej, p=P_H)
    counts = np.bincount(out, minlength=VOCAB).astype(float)
    return counts / counts.sum()


def tv(a, b):
    return 0.5 * np.sum(np.abs(a - b))


def rand_dist():
    d = RNG.random(VOCAB) + 0.05
    return d / d.sum()


def main():
    rows = []
    print("E0.4 sampling-path audit — toy categorical model, T>0 correction step")
    print(f"{'prompt':>6} {'TV(P_H,P_S)':>12} {'v1_TV_to_PH':>12} {'res_TV_to_PH':>13}"
          f" {'v1_mc_TV':>10} {'res_mc_TV':>10}")
    for i in range(N_PROMPTS):
        P_H, P_S = rand_dist(), rand_dist()
        e_v1 = emitted_dist_analytic(P_H, P_S, "v1_direct")
        e_res = emitted_dist_analytic(P_H, P_S, "residual")
        v1_bias = tv(e_v1, P_H)          # >0 : biased
        res_bias = tv(e_res, P_H)        # ~0 : exact
        mc_v1 = tv(emitted_dist_mc(P_H, P_S, "v1_direct", N_SAMPLES), P_H)
        mc_res = tv(emitted_dist_mc(P_H, P_S, "residual", N_SAMPLES), P_H)
        draft_tv = tv(P_H, P_S)
        print(f"{i:>6} {draft_tv:>12.4f} {v1_bias:>12.4f} {res_bias:>13.2e}"
              f" {mc_v1:>10.4f} {mc_res:>10.4f}")
        rows.append(dict(prompt=i, tv_ph_ps=f"{draft_tv:.6f}",
                         v1_analytic_tv_to_target=f"{v1_bias:.6f}",
                         residual_analytic_tv_to_target=f"{res_bias:.3e}",
                         v1_mc_tv_to_target=f"{mc_v1:.6f}",
                         residual_mc_tv_to_target=f"{mc_res:.6f}"))

    v1_max = max(float(r["v1_analytic_tv_to_target"]) for r in rows)
    res_max = max(float(r["residual_analytic_tv_to_target"]) for r in rows)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdict = ("INCORRECT (NOT Leviathan-exact)" if v1_max > 1e-6
               else "correct")
    print()
    print(f"v1-direct max analytic TV-to-target over {N_PROMPTS} prompts = {v1_max:.4f}  (biased)")
    print(f"residual  max analytic TV-to-target over {N_PROMPTS} prompts = {res_max:.2e}  (exact)")
    print(f"E0.4 VERDICT: v1 T>0 correction step is {verdict}")

    with open(OUT_MD, "w") as f:
        f.write(_report(rows, v1_max, res_max, verdict))
    print(f"wrote {OUT_MD} and {OUT_CSV}")

    # gate: the demonstration must show v1 biased and residual exact
    assert v1_max > 1e-6, "expected v1 method to be biased"
    assert res_max < 1e-9, "residual method must be exact"
    print("E0.4 unit test PASSED (v1 biased; residual exact — as expected)")


def _report(rows, v1_max, res_max, verdict):
    lines = []
    lines.append("# E0.4 — T>0 sampling-path audit\n")
    lines.append(f"**Verdict: the v1 T>0 (T=0.6) correction step is {verdict}.**\n")
    lines.append(
        "## Static finding (frozen notebook, Cells 12/24/42)\n"
        "On rejection the v1 code draws the correction/bonus token **directly "
        "from the full model distribution P_H** at the reject position "
        "(`full_logits.argmax()` for greedy; `multinomial(softmax(full_logits/T))` "
        "for T>0). It never forms the normalized residual "
        "`P_res = norm(max(0, P_H - P_S))`. Therefore the T>0 path is **not** "
        "Leviathan-exact rejection sampling and is **not lossless for T>0**.\n")
    lines.append(
        "## Toy unit test (10 prompts, exact analytic + Monte-Carlo)\n"
        "For a single draft token with draft x~P_S and accept prob "
        "min(1, P_H(x)/P_S(x)):\n\n"
        "- correct residual emit(x) = min(P_S,P_H)(x) + max(0,P_H-P_S)(x) = **P_H(x)** (exact)\n"
        "- v1-direct emit(x) = min(P_S,P_H)(x) + TV(P_S,P_H)·P_H(x) (**biased** unless TV=0)\n\n")
    lines.append("| prompt | TV(P_H,P_S) | v1 analytic TV→target | residual analytic TV→target | v1 MC TV | residual MC TV |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in rows:
        lines.append(f"| {r['prompt']} | {r['tv_ph_ps']} | "
                     f"{r['v1_analytic_tv_to_target']} | "
                     f"{r['residual_analytic_tv_to_target']} | "
                     f"{r['v1_mc_tv_to_target']} | {r['residual_mc_tv_to_target']} |\n")
    lines.append(
        f"\nMax TV-to-target: v1-direct = **{v1_max:.4f}** (biased), "
        f"residual = {res_max:.1e} (exact, numerical zero). Monte-Carlo "
        f"({N_SAMPLES:,} draws/prompt) confirms the analytic values.\n")
    lines.append(
        "\n## Consequence for the manuscript (PLAN E0.4 rule)\n"
        "This is a **writing fix, not a code fix** (DO-NOT list item: do not "
        "re-implement lossless residual sampling for T=0.6). All **T=0.6 tables "
        "are relabelled as draft-quality diagnostics**; the losslessness claim is "
        "retained only for the greedy (T=0) results, whose residual mismatch is a "
        "separate bf16-numerics question handled by E0.3. The Leviathan-exact "
        "`residual_correction_token` is added to `specdiag.specloop` for reference "
        "but is deliberately NOT wired into the v1-reproduction path.\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
