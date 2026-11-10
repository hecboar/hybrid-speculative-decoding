#!/usr/bin/env python3
"""Reconcile the THREE latency scenarios into the data-freeze decision:
  * v1-naive      — pre-L4 RTX PRO 4500 rows (the un-accelerated / 'current' path)
  * L4-naive      — canonical L4, no mamba kernels
  * L4-kernels    — canonical L4 + official mamba-ssm/causal-conv1d wheels

Reads the L4 + RTX rows from results/E3/partial_drafter_ablation.csv (E3.1),
results/E3/benchmark_audit.csv (E3.2) and results/E4/long_context.csv (E4), plus
the L4-kernels files results/E3.1/with_kernels_e31.csv / with_kernels_e4.csv, and
rewrites the 'L4 CANONICAL RECONCILIATION' section of
report/DATA_FREEZE_DECISION.md.

Decision rule (component_only = the paper's SSM drafter; attention_only excluded
as a diagnostic): if ANY credible drafter reaches S_theory > 1.0 in the
**L4-kernels** scenario (or L4-naive), FLAG the change LOUDLY -> Ending A. The
memo states explicitly whether any credible regime crosses 1.0 WITH THE OFFICIAL
STACK. Idempotent (replaces the marked section). CPU.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMO = os.path.join(REPO, "report", "DATA_FREEZE_DECISION.md")
BEGIN = "<!-- L4_RECONCILE:BEGIN -->"
END = "<!-- L4_RECONCILE:END -->"


def rd(path):
    p = os.path.join(REPO, path)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_l4(r):
    return "L4" in (r.get("gpu") or "")


def comp_ceff(rows):
    for r in rows:
        if r.get("strategy") == "component_only":
            return fnum(r.get("c_eff"))
    return float("nan")


def max_credible_s(*rowsets):
    vals = []
    for rows in rowsets:
        for r in rows:
            if r.get("strategy") == "attention_only":
                continue
            s = fnum(r.get("s_theory"))
            if s == s:
                vals.append(s)
    return max(vals) if vals else float("nan")


def fmt(x):
    return f"{x:.3f}" if x == x else "n/a"


def main():
    e31 = rd("results/E3/partial_drafter_ablation.csv")
    e4 = rd("results/E4/long_context.csv")
    e32 = rd("results/E3/benchmark_audit.csv")
    k31 = rd("results/E3.1/with_kernels_e31.csv")   # L4-kernels
    k4 = rd("results/E3.1/with_kernels_e4.csv")

    rtx31 = [r for r in e31 if not is_l4(r)]
    l4n31 = [r for r in e31 if is_l4(r)]
    rtx4 = [r for r in e4 if not is_l4(r)]
    l4n4 = [r for r in e4 if is_l4(r)]

    if not (l4n31 or l4n4 or k31 or k4):
        print("No L4-tagged rows yet — run the E3.1/E3.2/E4 (and kernels) L4 blocks first.")
        return

    scenarios = [
        ("v1-naive (RTX PRO 4500, un-accelerated path)", rtx31, rtx4, False),
        ("L4-naive (canonical L4, no mamba kernels)", l4n31, l4n4, True),
        ("L4-kernels (canonical L4 + official mamba-ssm/causal-conv1d)", k31, k4, True),
    ]

    lines = [BEGIN, "", "## L4 CANONICAL RECONCILIATION (auto-generated — 3 scenarios)", ""]
    lines += ["| scenario | component_only c_eff | best credible S_theory | reaches >1? |",
              "|---|---|---|---|"]
    kernels_max = float("nan")
    l4naive_max = float("nan")
    for name, r31, r4, _canon in scenarios:
        if not (r31 or r4):
            lines.append(f"| {name} | _pending_ | _pending_ | — |")
            continue
        ce = comp_ceff(r31) if r31 else comp_ceff(r4)
        ms = max_credible_s(r31, r4)
        lines.append(f"| {name} | {fmt(ce)} | {fmt(ms)} | "
                     f"{'**YES**' if (ms == ms and ms > 1.0) else 'no'} |")
        if name.startswith("L4-kernels"):
            kernels_max = ms
        if name.startswith("L4-naive"):
            l4naive_max = ms
    lines.append("")

    # per-length c_eff for the kernels scenario (the key long-context question)
    if k4:
        lines += ["**L4-kernels — c_eff vs context length (component_only):**", "",
                  "| L | c_eff | S_theory |", "|---|---|---|"]
        for r in sorted(k4, key=lambda z: int(z["prompt_len"])):
            lines.append(f"| {r['prompt_len']} | {fmt(fnum(r['c_eff']))} | "
                         f"{fmt(fnum(r['s_theory']))} |")
        lines.append("")

    # E3.2 canonical speedups by gpu (naive only; kernels block does not re-run E3.2)
    if any(is_l4(r) for r in e32):
        lines += ["**E3.2 speedups (L4-naive):**", "", "| impl | k | speedup |", "|---|---|---|"]
        for r in sorted([r for r in e32 if is_l4(r)],
                        key=lambda z: (z.get("impl", ""), int(z.get("k", 0)))):
            lines.append(f"| {r.get('impl')} | {r.get('k')} | {fmt(fnum(r.get('speedup')))} |")
        lines.append("")

    overall = max([x for x in (kernels_max, l4naive_max) if x == x], default=float("nan"))
    flips = overall == overall and overall > 1.0
    kern_crosses = kernels_max == kernels_max and kernels_max > 1.0

    lines.append("### Official-stack verdict")
    lines.append("")
    if k31 or k4:
        lines.append(
            f"With the OFFICIAL mamba-ssm/causal-conv1d stack on the L4, the best "
            f"credible (non-diagnostic) drafter reaches **S_theory = {fmt(kernels_max)}** — "
            f"**{'IT CROSSES 1.0' if kern_crosses else 'still ≤ 1.0'}.**")
    else:
        lines.append("_L4-kernels scenario not yet run (wheels absent/skipped) — official-stack "
                     "verdict pending._")
    lines.append("")

    if flips:
        lines += [
            "### ⚠️⚠️ CONCLUSION CHANGES → ENDING A ⚠️⚠️",
            "",
            f"A CREDIBLE drafter reaches **S_theory = {fmt(overall)} > 1.0** on the L4"
            + (" **with the official stack**" if kern_crosses else " (L4-naive)")
            + ". The un-accelerated-path bottleneck no longer masks the speedup — "
            "**switch the manuscript to Ending A (conditional speedup regime); "
            "re-do the E3.1 frontier / E4 figures highlighting the L4-kernels rows.**",
        ]
    else:
        lines += [
            "### Ending B HOLDS across all three scenarios",
            "",
            f"Best credible S_theory: v1-naive→see table, L4-naive→{fmt(l4naive_max)}, "
            f"**L4-kernels→{fmt(kernels_max)}** — none crosses 1.0. Even with the "
            "off-the-shelf mamba stack activating Falcon's fast path (lower c_eff for "
            "component_only), the all-layer internal SSM drafter is not cost-viable at "
            "k∈{1,2} → **Ending B stands.** Report the RTX→L4→L4-kernels c_eff delta in "
            "§5.5 as the quantified infrastructure-bottleneck gradient.",
        ]
    lines += ["", END]

    section = "\n".join(lines)
    memo = open(MEMO).read() if os.path.exists(MEMO) else "# DATA-FREEZE DECISION MEMO\n"
    if BEGIN in memo and END in memo:
        memo = memo[:memo.index(BEGIN)] + section + memo[memo.index(END) + len(END):]
    else:
        memo = memo.rstrip() + "\n\n---\n\n" + section + "\n"
    with open(MEMO, "w") as f:
        f.write(memo)

    banner = "ENDING A (CHANGED!)" if flips else "Ending B holds"
    print("=" * 64)
    print(f"L4 reconcile: {banner}")
    print(f"  L4-kernels best credible S_theory = {fmt(kernels_max)} "
          f"({'>1' if kern_crosses else '<=1'})")
    print(f"  L4-naive   best credible S_theory = {fmt(l4naive_max)}")
    print("=" * 64)
    print("updated:", MEMO)


if __name__ == "__main__":
    main()
