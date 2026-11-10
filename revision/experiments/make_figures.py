#!/usr/bin/env python3
"""Regenerate paper figures (PLAN §7.3). Reads the results CSVs and writes PDF+PNG
into figures/. Each figure is skipped (with a note) if its source CSV is absent,
so this can run at any point. CPU only.

Figures:
  fig_acceptance_vs_k        - A_k vs k, all models (E1 + v1 pickles)
  fig_e2_collapse            - Qwen A_k vs #attention layers kept (E2)
  fig_e6_scatter             - log10(ppl_ratio) vs A_4, by paradigm (E6)
  fig_e3_frontier            - acceptance-cost frontier: tau/k vs c_eff (E3.1 + E2)
  fig_e3_2_breakdown         - optimized timing breakdown stacked bars (E3.2)
  fig_e4_ceff_vs_len         - c_eff vs context length (E4)
"""
import os
import sys
import csv
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
# Embed TrueType (Type 42) instead of matplotlib's default Type 3 fonts: crisper,
# text-extractable, and passes publisher (Elsevier/IEEE) preflight checks.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)
DRIVE = os.path.join(REPO, "checkpoints", "v1_drive")

PAR_C = {"parallel": "#2563eb", "sequential": "#dc2626", "transformer": "#6b7280"}


def rd(path):
    p = os.path.join(REPO, path)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


ACCEPT_COL = {
    "falcon-h1-0.5b": "#1f77b4", "falcon15": "#2ca02c",
    "falcon-h1-3b": "#17becf", "zamba2": "#9467bd",
    "granite": "#d62728", "qwen3.5-0.8b": "#ff7f0e",
    "qwen35_2b": "#8c564b", "lfm2": "#e377c2",
}


def fig_acceptance_vs_k():
    """Fig. 1 (R1.2/R3.1): parallel vs sequential panels, k in {2,4}, shared
    y-axis, per-model CI shading, plus the v1 pure-Transformer control."""
    e1 = rd("results/E1/acceptance_new_models.csv")
    if not e1:
        print("  skip acceptance_vs_k (no E1 csv)"); return
    v1 = rd("results/v1/acceptance_rate.csv") or []
    KS = [2, 4]
    par, seq = {}, {}
    for r in e1:
        if str(r["temperature"]) not in ("0.0", "0") or int(r["k"]) not in KS:
            continue
        d = {"parallel": par, "sequential": seq}.get(r["arch_paradigm"])
        if d is None:
            continue
        d.setdefault(r["model"], {})[int(r["k"])] = (
            float(r["A_k"]), float(r["ci_lo"]), float(r["ci_hi"]))
    ctrl = {}
    for r in v1:
        if r["model"] == "qwen2.5-0.5b" and str(r["temperature"]) in ("0.0", "0") \
                and int(r["k"]) in KS:
            ctrl[int(r["k"])] = (float(r["acceptance_rate"]),
                                 float(r["ci_lo"]), float(r["ci_hi"]))
    gran = seq.get("granite", {}).get(4)
    gran_a4 = gran[0] if gran else None

    def draw(ax, series, title, legend_loc="upper right"):
        for m in sorted(series):
            d = series[m]; ks = sorted(d)
            col = ACCEPT_COL.get(m, "#333333"); emph = (m == "granite")
            A = [d[k][0] for k in ks]; lo = [d[k][1] for k in ks]; hi = [d[k][2] for k in ks]
            if len(ks) >= 2:
                ax.plot(ks, A, "-", color=col, lw=2.6 if emph else 1.5,
                        marker="*" if emph else "o", ms=13 if emph else 6,
                        mec="black" if emph else col, mew=1.0 if emph else 0,
                        label=m + (" ★" if emph else ""), zorder=6 if emph else 3)
                ax.fill_between(ks, lo, hi, color=col, alpha=0.18, zorder=1)
            else:                                    # single-k point (falcon-h1-3b)
                k = ks[0]
                ax.errorbar([k], [A[0]], yerr=[[A[0] - lo[0]], [hi[0] - A[0]]], fmt="D",
                            color=col, ms=6, capsize=3, label=f"{m} (k=4 only)", zorder=3)
        if ctrl:
            ks = sorted(ctrl); A = [ctrl[k][0] for k in ks]
            lo = [ctrl[k][1] for k in ks]; hi = [ctrl[k][2] for k in ks]
            ax.plot(ks, A, "s--", color="#7f7f7f", lw=1.4,
                    label="qwen2.5-0.5b (Transformer control, both panels)", zorder=2)
            ax.fill_between(ks, lo, hi, color="#7f7f7f", alpha=0.12, zorder=1)
        if gran_a4 is not None:                      # cross-panel comparison guide
            ax.axhline(gran_a4, color="#d62728", ls=":", lw=1.1, alpha=0.55, zorder=0)
        ax.set_title(title); ax.set_xlabel("draft length k")
        ax.set_xticks(KS); ax.set_xlim(1.7, 4.3); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc=legend_loc)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    draw(axL, par, "Parallel hybrids", legend_loc="lower left")
    draw(axR, seq, "Sequential hybrids", legend_loc="upper right")
    axL.set_ylim(0, 0.78)                             # shared identical y-axis
    axL.set_ylabel("A_k = tau_bar/k (mean acceptance rate)")
    if gran_a4 is not None:
        axL.text(4.25, 0.10,
                 f"reference: granite = {gran_a4:.2f}\n(plotted in sequential panel →)",
                 color="#d62728", fontsize=6.5, ha="right", va="center")
    fig.suptitle("Mean acceptance rate vs draft length k (val-test, greedy) — parallel vs sequential",
                 fontsize=12)
    if gran_a4 is not None:
        fig.text(0.5, -0.02,
                 f"granite (sequential, A_4={gran_a4:.2f}) exceeds falcon15 (parallel, 0.21): "
                 "acceptance tracks functional attention-dependence, not the parallel/sequential "
                 "label; the red dotted line marks granite's level for cross-panel "
                 "comparison. The pure-Transformer control (qwen2.5-0.5b) is drawn in BOTH panels "
                 "as a shared reference baseline. k=8 for the original models is reported in the tables; "
                 "curves are capped at k∈{2,4} for uniform coverage across all models.",
                 ha="center", va="top", fontsize=8, style="italic", wrap=True)
    save(fig, "fig_acceptance_vs_k")


def fig_e2_collapse():
    rows = rd("results/E2/qwen_draft_variants.csv")
    if not rows:
        print("  skip e2_collapse (no E2 csv)"); return
    sub = [r for r in rows if r["strategy"].startswith("attn_subset") and str(r["k"]) == "4"]
    if not sub:
        print("  skip e2_collapse (no attn_subset k=4 rows)"); return
    # keep=0 (linear-only, all 6 attention layers removed) has keep_policy '-' and
    # is the collapse point — it MUST be shown. It is shared by both keep-policies.
    k0 = next((r for r in sub if int(r["n_attn_kept"]) == 0), None)
    k0A = float(k0["A_k"]) if k0 else None
    fig, axL = plt.subplots(figsize=(7.2, 4.8))
    axR = axL.twinx()
    palette = {"keep_first_last": "#1f77b4", "top_ppl": "#22a2c3"}
    for pol in ("keep_first_last", "top_ppl"):
        pts = [(int(r["n_attn_kept"]), float(r["A_k"])) for r in sub
               if r["keep_policy"] == pol and r["n_attn_kept"] != ""]
        if k0A is not None:
            pts.append((0, k0A))                     # shared collapse point
        pts = sorted(set(pts))
        axL.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=palette[pol],
                 ms=7, label=f"A_4 — keep policy: {pol}")
    # c_param per keep level (one value per level; independent of policy)
    cp = {}
    for r in sub:
        cp[int(r["n_attn_kept"])] = float(r["c_param"])
    cp = sorted(cp.items())
    axR.plot([k for k, _ in cp], [v for _, v in cp], "s--", color="#d62728", ms=6,
             label="c_param (drafter cost / full model)")
    # annotations: collapse point, full-cost point, conclusion
    if k0A is not None:
        axL.annotate(f"keep=0 (linear-only): A_4={k0A:.2f}\n→ COLLAPSE (attention fully removed)",
                     (0, k0A), xytext=(0.35, 0.135), fontsize=8, fontweight="bold", color="#b30000",
                     arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.2))
    axR.annotate("keep=5 ≈ full-model cost\n(c_param=0.96)", (5, cp[-1][1]),
                 xytext=(3.35, 0.90), fontsize=8, color="#d62728", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
    axL.text(1.5, 0.37, "no acceptance/cost sweet spot:\nusable-A_4 points cost ≈ full model;\n"
             "the only cheaper point (keep=0) collapsed",
             fontsize=8, ha="center", va="center", style="italic",
             bbox=dict(boxstyle="round", fc="#fff7e6", ec="#e0a800", alpha=0.95))
    axL.set_xlabel("# attention layers kept of 6 (keep=0 → linear-only)")
    axL.set_ylabel("A_4 = tau_bar/4 (mean acceptance rate, k=4)")
    axR.set_ylabel("c_param (drafter cost / full-model params)", color="#d62728")
    axR.tick_params(axis="y", labelcolor="#d62728")
    axL.set_xticks([0, 2, 4, 5]); axL.set_ylim(0, 0.62); axR.set_ylim(0.70, 1.0)
    axL.set_title("Qwen3.5 sequential collapse: acceptance vs cost (attn_subset, k=4)")
    hL, lL = axL.get_legend_handles_labels(); hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, fontsize=7.5, loc="upper left")
    axL.grid(alpha=0.3)
    save(fig, "fig_e2_collapse")


def _isotonic_nonincreasing(xv, yv):
    """Monotone non-increasing fit (PAVA) — a rank-consistent trend guide for the
    Spearman relationship; NOT a linear regression. Returns sorted (x, fitted_y)."""
    order = np.argsort(xv)
    xs = np.asarray(xv, float)[order]
    yneg = (-np.asarray(yv, float)[order]).tolist()   # fit non-decreasing on -y
    vals, wts = [], []
    for v in yneg:
        cv, cw = v, 1.0
        while vals and vals[-1] > cv:
            pv, pw = vals.pop(), wts.pop()
            cv, cw = (cv * cw + pv * pw) / (cw + pw), cw + pw
        vals.append(cv); wts.append(cw)
    fitted = []
    for v, w in zip(vals, wts):
        fitted += [v] * int(round(w))
    return xs, -np.asarray(fitted)


def fig_e6_scatter():
    rows = rd("results/E6/ppl_vs_acceptance.csv")
    if not rows:
        print("  skip e6_scatter (no E6 csv)"); return
    sp = rd("results/E6/ppl_spearman.csv")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    xs = [float(r["log10_ppl_ratio"]) for r in rows]
    ys = [float(r["A_4"]) for r in rows]
    # monotone trend guide (rank-based, matches Spearman)
    tx, ty = _isotonic_nonincreasing(xs, ys)
    ax.plot(tx, ty, color="#999999", lw=1.6, ls="--", zorder=1,
            label="monotone trend (rank-based guide, not a fit)")
    # per-model label offsets to de-clutter the upper-left cluster
    LAB = {
        "falcon-h1-0.5b": (2, 7, "left"),
        "falcon-h1-3b":   (-4, -12, "right"),
        "qwen2.5-0.5b":   (8, -9, "left"),
        "falcon15":       (6, -3, "left"),
        "zamba2":         (7, 3, "left"),
        "qwen35_2b":      (7, 3, "left"),
        "qwen3.5-0.8b":   (7, 3, "left"),
        "lfm2":           (-6, 6, "right"),
    }
    for r in rows:
        x, y = float(r["log10_ppl_ratio"]), float(r["A_4"])
        c = PAR_C.get(r["arch_paradigm"], "#000")
        is_granite = r["model"] == "granite"
        ax.scatter(x, y, color=c, s=240 if is_granite else 60,
                   edgecolors="black" if is_granite else "none",
                   linewidths=1.3 if is_granite else 0, marker="*" if is_granite else "o",
                   zorder=4)
        if is_granite:
            continue  # identified by the callout below (avoid a duplicate overlapping label)
        dx, dy, ha = LAB.get(r["model"], (5, 3, "left"))
        ax.annotate(r["model"], (x, y), fontsize=6, xytext=(dx, dy),
                    textcoords="offset points", ha=ha)
    # granite callout: sequential beats a parallel hybrid -> breaks the taxonomy
    g = next((r for r in rows if r["model"] == "granite"), None)
    if g:
        gx, gy = float(g["log10_ppl_ratio"]), float(g["A_4"])
        ax.annotate("granite (SEQUENTIAL) A_4=0.35\nbeats falcon15 (parallel) 0.21\n→ qualifies parallel/sequential taxonomy",
                    (gx, gy), xytext=(1.15, 0.15), fontsize=7.5, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
    if sp:
        s = sp[0]
        ax.text(0.97, 0.95, f"Spearman ρ = {float(s['rho']):.2f},  p = 0.026,  n = {s['n']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#666", alpha=0.95))
    for par, c in PAR_C.items():
        ax.scatter([], [], color=c, label=par)
    ax.set_xlabel("log10( PPL_no_attn / PPL_full )   (functional attention-dependence)")
    ax.set_ylabel("A_4 = tau_bar/4 (mean acceptance rate, k=4)")
    ax.set_title("Acceptance vs functional attention-dependence")
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.grid(alpha=0.3)
    save(fig, "fig_e6_scatter")


# ---- Redesigned acceptance-cost frontier (R2.0/R2.6) -----------------------
# Diagnosis of the old fig_e3_frontier clutter:
#   (1) label = strategy[:12] truncated "component_only" AND
#       "component_only_partial" both to "component_on" (collision + dropped "ly");
#   (2) partial_drafter_ablation.csv holds BOTH an RTX/naive and an L4/naive block,
#       so every strategy was plotted twice.
#   => "component_on/k1" appeared 8x (4 variants x 2 GPUs), "attention_on/k1" 2x.
# Fixes below: keep only the canonical L4-naive block, full distinct names,
# colour+marker encoding with ONE legend, text only on the kernels + best-credible
# points, attention_only greyed as a diagnostic.

FRONTIER_COLORS = {
    "component_only": "#1f77b4",
    "component_only_partial_0.75": "#2ca02c",
    "component_only_partial_0.5": "#9467bd",
    "component_only_partial_0.25": "#8c564b",
    "layerskip_0.33": "#17becf",
    "early_exit_0.5": "#e377c2",
}


def _frontier_falcon_measured():
    """L4-naive Falcon variants (E3.1). name -> sorted [(c_eff, tau/k, k, s_theory)]."""
    rows = rd("results/E3/partial_drafter_ablation.csv") or []
    series = {}
    for r in rows:
        if "L4" not in (r.get("gpu") or ""):   # canonical scenario only (drop RTX)
            continue
        strat = r["strategy"]
        name = f"component_only_partial_{r['frac']}" if strat == "component_only_partial" else strat
        try:
            ce = float(r["c_eff"]); k = int(r["k"]); tk = float(r["tau_bar"]) / k
            s = float(r["s_theory"])
        except Exception:
            continue
        series.setdefault(name, []).append((ce, tk, k, s))
    for v in series.values():
        v.sort(key=lambda t: t[2])
    return series


def _frontier_kernels():
    """L4-kernels component_only (E3.1). sorted [(c_eff, tau/k, k, s_theory)]."""
    rows = rd("results/E3.1/with_kernels_e31.csv") or []
    pts = []
    for r in rows:
        if r.get("strategy") != "component_only":
            continue
        try:
            pts.append((float(r["c_eff"]), float(r["tau_bar"]) / int(r["k"]),
                        int(r["k"]), float(r["s_theory"])))
        except Exception:
            continue
    pts.sort(key=lambda t: t[2])
    return pts


def _frontier_attention_kernels():
    """L4-kernels attention_only (B-6): [(c_eff, tau/k, k, s_theory)]."""
    rows = rd("results/E3.1/attention_only_kernels.csv") or []
    pts = []
    for r in rows:
        try:
            pts.append((float(r["c_eff"]), float(r["tau_bar_frozen"]) / int(r["k"]),
                        int(r["k"]), float(r["s_theory"])))
        except Exception:
            continue
    pts.sort(key=lambda t: t[2])
    return pts


def _frontier_qwen_param():
    """Qwen3.5 E2 variants on c_param. [(c_param, tau/k)]."""
    rows = rd("results/E2/qwen_draft_variants.csv") or []
    pts = []
    for r in rows:
        try:
            pts.append((float(r["c_param"]), float(r["tau_bar"]) / int(r["k"])))
        except Exception:
            continue
    return pts


def _draw_frontier_falcon(ax, annotate=True):
    """Falcon measured-c_eff points + kernels star. Caller draws diagonal/limits."""
    series = _frontier_falcon_measured()
    for name, color in FRONTIER_COLORS.items():
        if name not in series:
            continue
        pts = series[name]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color,
                ms=7, lw=1.2, label=name, zorder=3)
    if "attention_only" in series:                 # diagnostic ARTIFACT — red X
        pts = series["attention_only"]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker="X", s=150,
                   color="#d62728", edgecolors="black", linewidths=0.6, zorder=4,
                   label="attention_only (diagnostic artifact — not a credible drafter)")
        if annotate:
            cx, cy = pts[len(pts) // 2][0], pts[len(pts) // 2][1]
            ax.annotate("attention_only — ARTIFACT\n(cheap only: discards SSM scan)", (cx, cy),
                        xytext=(0.24, 0.29), fontsize=7.5, color="#d62728", fontweight="bold",
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
    kern = _frontier_kernels()
    for (ce, tk, k, s) in kern:
        ax.scatter(ce, tk, s=360 if k == 1 else 190, marker="*", color="#ffcc00",
                   edgecolors="black", linewidths=1.1, zorder=6,
                   label="component_only (L4-kernels)" if k == 1 else None)
    akern = _frontier_attention_kernels()               # B-6: attention_only under kernels
    for (ce, tk, k, s) in akern:
        ax.scatter(ce, tk, s=170 if k == 1 else 110, marker="P", color="#ff7f0e",
                   edgecolors="black", linewidths=0.9, zorder=6,
                   label="attention_only (L4-kernels — inverts method)" if k == 1 else None)
    if annotate and kern:
        ce, tk, k, s = kern[0]
        ax.annotate("L4-kernels k=1:\nc_eff 0.784, " r"$\bar{\tau}$" " 0.760 → S=0.986",
                    (ce, tk), xytext=(0.15, 1.05), fontsize=8.5, fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    if annotate and series.get("component_only"):
        best = [q for q in series["component_only"] if q[2] == 1]
        if best:
            ce, tk, k, s = best[0]
            # lower-right open space; both anchor and target sit below the
            # diagonal so the leader never crosses the break-even line.
            ax.annotate(f"best credible naive\ncomponent_only k=1 · S={s:.3f}",
                        (ce, tk), xytext=(1.18, 0.30), fontsize=8, ha="right", va="center",
                        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1))


def _frontier_decorate(ax, xlabel):
    lim = 1.2
    ax.plot([0, lim], [0, lim], "k--", lw=1.9, zorder=4, label=r"break-even $\bar{\tau}/k = c$")
    ax.fill_between([0, lim], [0, lim], lim, alpha=0.08, color="green", zorder=0)
    ax.text(0.10, 0.52, "net-speedup region\n(above the line)", color="green",
            fontsize=8, rotation=45, rotation_mode="anchor", alpha=0.85, va="bottom")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\bar{\tau}\,/\,k$   (acceptance per draft token)")
    ax.grid(alpha=0.3)


def fig_e3_frontier_1panel():
    if not rd("results/E3/partial_drafter_ablation.csv"):
        print("  skip e3_frontier_1panel (no E3.1 csv)"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_frontier_falcon(ax, annotate=True)
    q = _frontier_qwen_param()
    if q:
        ax.scatter([p[0] for p in q], [p[1] for p in q], s=45, marker="s",
                   color="#d62728", alpha=0.8, zorder=2,
                   label="Qwen3.5 variants — c_param (param-derived)")
    _frontier_decorate(ax, "drafter cost c   —   ○ measured c_eff (Falcon)   □ c_param (Qwen, param-derived)")
    ax.set_title("Acceptance–cost frontier: no credible drafter crosses break-even")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=True)
    save(fig, "fig_e3_frontier_1panel")


def _build_frontier_2panel(caption=False):
    """Canonical 2-panel frontier. Left: Falcon measured c_eff (the break-even
    claim). Right: Qwen on c_param, context only. Returns fig; caller saves."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 7.2), sharey=True)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)
    _draw_frontier_falcon(axL, annotate=True)
    _frontier_decorate(axL, "measured c_eff (draft/verify, CUDA events)")
    axL.set_title("Falcon-H1-0.5B — measured c_eff")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7.5, frameon=True)
    q = _frontier_qwen_param()
    if q:
        axR.scatter([p[0] for p in q], [p[1] for p in q], s=55, marker="s",
                    color="#d62728", zorder=3, label="Qwen3.5 variants")
    _frontier_decorate(axR, "c_param (parameter-count ratio)")
    axR.set_ylabel("")
    axR.set_title("Qwen3.5-0.8B — c_param (parameter-derived)")
    axR.legend(loc="upper right", fontsize=8, frameon=True)
    # Note: the panel-by-panel explanation lives in the LaTeX \caption (self-contained),
    # so no redundant in-figure text block is drawn here.
    return fig


def fig_e3_frontier():
    """Canonical acceptance-cost frontier (2-panel; author-approved 2026-07-14)."""
    if not rd("results/E3/partial_drafter_ablation.csv"):
        print("  skip e3_frontier (no E3.1 csv)"); return
    save(_build_frontier_2panel(caption=True), "fig_e3_frontier")


def fig_e3_frontier_2panel():
    if not rd("results/E3/partial_drafter_ablation.csv"):
        print("  skip e3_frontier_2panel (no E3.1 csv)"); return
    save(_build_frontier_2panel(caption=False), "fig_e3_frontier_2panel")


def fig_e3_2_breakdown():
    rows = rd("results/E3/benchmark_audit.csv")
    if not rows:
        print("  skip e3_2_breakdown (no E3.2 csv)"); return
    # canonical L4 only (the CSV also holds non-canonical RTX rows; mixing them
    # averaged two GPUs together and mislabelled the figure as non-canonical).
    opt = [r for r in rows if r["impl"] == "optimized" and r["draft_ms_tok"]
           and "L4" in (r.get("gpu") or "")]
    if not opt:
        print("  skip e3_2_breakdown (no optimized L4 rows)"); return
    by = {}
    for r in opt:
        by.setdefault(int(r["k"]), []).append(r)
    ks = sorted(by); phases = ["draft_ms_tok", "verify_ms_tok", "accept_ms_tok", "overhead_ms_tok"]
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#6b7280"]
    ph_label = {"draft_ms_tok": "draft", "verify_ms_tok": "verify",
                "accept_ms_tok": "accept ($\\approx$0.3 ms/tok, negligible)",
                "overhead_ms_tok": "overhead"}
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    bottom = np.zeros(len(ks))
    for ph, c in zip(phases, colors):
        vals = np.array([np.mean([float(r[ph]) for r in by[k]]) for k in ks])
        ax.bar([str(k) for k in ks], vals, bottom=bottom, label=ph_label[ph], color=c)
        bottom += vals
    # AR baseline reference: 1 full forward per token = 1000 / ar_tok_s (L4)
    ar_tok_s = np.mean([float(r["ar_tok_s"]) for r in opt])
    ar_ms = 1000.0 / ar_tok_s
    ax.axhline(ar_ms, color="black", ls="--", lw=1.5, zorder=5,
               label=f"AR baseline ({ar_tok_s:.1f} tok/s ≈ {ar_ms:.1f} ms/tok)")
    ax.set_xlabel("k (draft length)"); ax.set_ylabel("ms / emitted token")
    ax.set_title("Optimized spec timing breakdown (canonical NVIDIA L4)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    save(fig, "fig_e3_2_breakdown")


def fig_e4_ceff_vs_len():
    rows = rd("results/E4/long_context.csv")
    if not rows:
        print("  skip e4_ceff_vs_len (no E4 csv)"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append((int(r["prompt_len"]), float(r["c_eff"])))
    for s, pts in by.items():
        pts = sorted(set(pts)); ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                                        label=f"{s} (naive)")
    # L4-kernels scenario: component_only c_eff drops to ~0.78-0.84 (official
    # mamba-ssm/causal-conv1d stack), so the figure now shows both scenarios.
    kern = rd("results/E3.1/with_kernels_e4.csv")
    if kern:
        pts = sorted({(int(r["prompt_len"]), float(r["c_eff"])) for r in kern
                      if r.get("strategy") == "component_only"})
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "*-", color="#f59e0b",
                    markersize=11, markeredgecolor="black", markeredgewidth=0.5,
                    label="component_only (L4-kernels)")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([512, 1024, 2048, 4096]))
    ax.xaxis.set_major_formatter(FixedFormatter(["512", "1024", "2048", "4096"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.74, 1.01)
    ax.annotate("c_eff ≈ flat in L in BOTH stacks\n(naive ~0.99; kernels 0.78–0.84)\n→ recurrent-state advantage does not materialise",
                xy=(540, 0.905), fontsize=8, ha="left", va="center", color="#222",
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.92))
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("c_eff (draft/full fwd time)")
    ax.set_title("c_eff vs context length (naive vs L4-kernels)")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.80), framealpha=0.95); ax.grid(alpha=0.3)
    fig.text(0.5, -0.03,
             "L4-kernels per-length values are n=30 point estimates; "
             "the 0.78→0.84 rise is within point-estimate noise, not a trend.",
             ha="center", fontsize=7, style="italic")
    save(fig, "fig_e4_ceff_vs_len")


def main():
    print("Regenerating figures ->", FIG)
    for f in (fig_acceptance_vs_k, fig_e2_collapse, fig_e6_scatter,
              fig_e3_frontier, fig_e3_frontier_1panel, fig_e3_frontier_2panel,
              fig_e3_2_breakdown, fig_e4_ceff_vs_len):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {f.__name__} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
