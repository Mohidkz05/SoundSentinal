"""Builds the paper's data figures from paper/figdata/*.csv.

    ~/tools/docvenv/bin/python paper/figures/make_figures.py

One column of IEEEtran is 3.5 in. Vector PDF out, Times-like serif to match
the body. Palette validated with the dataviz checker (blue #2a6bb8 / orange
#b8641e: CVD ΔE 22.8, contrast ≥ 3:1); the private series also carries a hatch
so the pair still separates in greyscale print.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figdata"

BLUE, ORANGE = "#2a6bb8", "#b8641e"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"

plt.rcParams.update({
    # STIX ships with matplotlib and matches the Times body of IEEEtran.
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def fig_itw():
    data = rows("fig3_itw_eer.csv")
    labels = [r["model"] for r in data]
    eer = [float(r["itw_eer_percent"]) for r in data]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    y = range(len(labels))[::-1]
    ax.barh(list(y), eer, height=0.62, color=BLUE, zorder=2)
    ax.axvline(50, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(50, len(labels) - 0.35, "chance", color=MUTED, fontsize=7,
            ha="center", va="bottom")
    for yi, v in zip(y, eer):
        ax.text(v + 1.0, yi, f"{v:.2f}%", va="center", fontsize=7, color=INK,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=3)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.set_xlabel("In-the-Wild EER (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig3_itw.pdf")
    plt.close(fig)


def fig_dp():
    data = rows("fig2_dp_per_attack.csv")
    attacks = [r["attack"] for r in data]
    nonp = [float(r["non_private"]) for r in data]
    dp = [float(r["dp"]) for r in data]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    x = range(len(attacks))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], nonp, w, color=BLUE, label="Non-private", zorder=2)
    ax.bar([i + w / 2 + 0.01 for i in x], dp, w, color=ORANGE, hatch="//////",
           edgecolor="white", lw=0, label=r"DP-SGD, $\varepsilon=0.48$", zorder=2)
    ax.set_xticks(list(x), attacks, fontsize=7)
    ax.set_ylabel("EER (%)")
    ax.set_ylim(0, 50)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig2_dp.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_itw()
    fig_dp()
    print("wrote fig2_dp.pdf, fig3_itw.pdf")
