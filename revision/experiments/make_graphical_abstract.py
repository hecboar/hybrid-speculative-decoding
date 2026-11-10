#!/usr/bin/env python3
"""Generate the Elsevier graphical abstract (single landscape panel).

Reads the same source as main-text Figure 2 (results/E6/ppl_vs_acceptance.csv) so the
graphical abstract can never diverge from the paper. Output: figures/graphical_abstract.png
at ~2656x1062 px (Elsevier minimum is 531h x 1328w; we render proportionally larger).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "E6", "ppl_vs_acceptance.csv")
OUT = os.path.join(ROOT, "figures", "graphical_abstract.png")

PAR_C = {"parallel": "#2A6F97", "sequential": "#E8743B", "transformer": "#7A7F87"}
LABEL = {"parallel": "Parallel hybrid", "sequential": "Sequential hybrid",
         "transformer": "Transformer control"}


def load():
    with open(SRC, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    fig = plt.figure(figsize=(13.28, 5.31), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    tx = fig.add_subplot(gs[0, 1]); tx.axis("off")

    for r in rows:
        x, y = float(r["log10_ppl_ratio"]), float(r["A_4"])
        par = r["arch_paradigm"]
        is_g = r["model"] == "granite"
        ax.scatter(x, y, s=430 if is_g else 150, color=PAR_C.get(par, "#000"),
                   marker="*" if is_g else "o",
                   edgecolors="white", linewidths=2.0 if is_g else 1.4, zorder=5)
    # granite callout (short, hugging the star)
    g = next(r for r in rows if r["model"] == "granite")
    ax.annotate("granite: attention-light sequential\nbeats parallel hybrids",
                (float(g["log10_ppl_ratio"]), float(g["A_4"])),
                xytext=(0.74, 0.435), fontsize=10, fontweight="bold", va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                shrinkA=3, shrinkB=8))
    # Spearman box in the open area on the right, clear of the granite callout
    ax.text(0.50, 0.66, "Spearman $\\rho = -0.79$\n(n = 8 hybrids)",
            transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.95))

    ax.set_xlabel("Functional attention-dependence\n$\\log_{10}$( PPL$_{\\mathrm{no\\text{-}attn}}$ / PPL$_{\\mathrm{full}}$ )",
                  fontsize=12.5)
    ax.set_ylabel("Speculative acceptance  $A_4 = \\bar{\\tau}_4/4$", fontsize=12.5)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.03, 0.46)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markeredgecolor="white", markersize=11, label=LABEL[k])
               for k, c in PAR_C.items()]
    ax.legend(handles=handles, fontsize=10.5, loc="upper right", frameon=True)

    # takeaway panel
    tx.text(0.0, 0.93, "Component-Aware Self-Speculative\nDecoding for Hybrid LMs",
            fontsize=16.5, fontweight="bold", va="top", color="#14212E")
    tx.text(0.0, 0.66,
            "A hybrid model's own SSM / linear-attention\n"
            "subgraph is a training-free internal drafter.",
            fontsize=12.5, va="top", color="#222")
    tx.text(0.0, 0.45,
            "Acceptance is governed by measured\n"
            "attention-dependence — not the\n"
            "parallel / sequential label.",
            fontsize=12.5, va="top", color="#222")
    tx.text(0.0, 0.18,
            "Viable in the right architectures, but\n"
            "cost-bound: best $S_{\\mathrm{theory}} = 0.986$ (<1)\n"
            "even with official SSM kernels active.",
            fontsize=12.5, va="top", color="#8A3324", fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.20)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    # Flatten RGBA -> RGB on white so portals never see an alpha channel
    try:
        from PIL import Image
        im = Image.open(OUT).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        Image.alpha_composite(bg, im).convert("RGB").save(OUT)
    except Exception as e:  # pragma: no cover
        print(f"warn: could not flatten to RGB ({e})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
