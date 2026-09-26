#!/usr/bin/env python3
"""Measured-only figures for the revised paper draft."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R6 = pd.read_csv(os.path.join(HERE, "r6_symmetric.csv"))
R3 = pd.read_csv(os.path.join(
    ROOT, "h0_measurement", "bugs", "2_towards_real_evictor", "R3-cells.csv"))

COL = {
    "llama33-70b": "#12414f", "mistral-7b": "#1f6b52",
    "llama31-8b": "#1d8a9c", "qwen15-moe-a2.7b": "#b0651c",
    "qwen3-8b": "#7c3aed", "qwen3-30b-a3b-2507": "#c2334d",
}
NAME = {
    "llama33-70b": "Llama-3.3-70B", "mistral-7b": "Mistral-7B",
    "llama31-8b": "Llama-3.1-8B", "qwen15-moe-a2.7b": "Qwen1.5-MoE",
    "qwen3-8b": "Qwen3-8B", "qwen3-30b-a3b-2507": "Qwen3-30B-A3B",
}
ORDER = list(COL)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 5.8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, stem):
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(HERE, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def regime_figure():
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.78))
    a0, a1 = ax
    a0.axhline(1.0, color="#999999", lw=0.6, ls=":")
    for model in ORDER:
        g = R6[R6.model == model].sort_values("ctx")
        size = 10 + 5 * (np.log2(g.ctx) - np.log2(R6.ctx.min()))
        a0.plot(g.dead2, g.portfolio_gain, color=COL[model], alpha=0.85)
        a0.scatter(g.dead2, g.portfolio_gain, s=size, color=COL[model],
                   edgecolor="white", linewidth=0.35, zorder=3, label=NAME[model])
        if len(g) > 1:
            p, q = g.iloc[-2], g.iloc[-1]
            a0.annotate("", xy=(q.dead2, q.portfolio_gain),
                        xytext=(p.dead2, p.portfolio_gain),
                        arrowprops=dict(arrowstyle="-|>", color=COL[model],
                                        lw=0.7, mutation_scale=6))
    a0.set(xlabel="dead 2-bit tier, $D_2$ (\\% of heads)",
           ylabel="hindsight portfolio gain $G_+$", xlim=(0, 82), ylim=(0.95, 3.35),
           title="(a) Tier extinction orders interior value")
    a0.text(78, 1.06, "$\\rho=-0.978$", ha="right", color="#555555")
    a0.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.5,
              handletextpad=0.2, labelspacing=0.18, borderaxespad=0.0)

    a1.axhspan(0, 15, color="#1d6f80", alpha=0.06, lw=0)
    a1.axhline(15, color="#1d6f80", lw=0.55, ls=":")
    a1.axhline(35, color="#c2334d", lw=0.55, ls=":")
    for model in ORDER:
        g = R6[R6.model == model].sort_values("ctx")
        yerr = np.vstack([g.band - g.band_lo, g.band_hi - g.band])
        a1.errorbar(g.ctx / 1024, g.band, yerr=yerr, color=COL[model],
                    marker="o", ms=2.5, lw=0.9, elinewidth=0.45, capsize=1.0)
    a1.set_xscale("log", base=2)
    a1.set_xticks([4, 8, 16, 32, 64, 128, 256])
    a1.set_xticklabels(["4k", "8k", "16k", "32k", "64k", "128k", "256k"])
    a1.set(xlabel="context length", ylabel="heads with gain $\\geq2\\times$ (\\%)",
           ylim=(0, 96), title="(b) Context moves fixed models toward a corner")
    a1.text(4.4, 36.5, "GO", color="#c2334d", fontsize=5.5)
    a1.text(4.4, 3.2, "STOP", color="#1d6f80", fontsize=5.5)
    save(fig, "fig1_regime")


def evidence_figure():
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.62))
    a0, a1 = ax
    g = R3.sort_values("dead2")
    a0.plot(g.dead2, g.band_or, "o-", ms=2.6, color="#777777", label="all-oracle")
    a0.plot(g.dead2, g.band_pr, "s-", ms=2.5, color="#b0651c",
            label="asymmetric (E2)")
    a0.plot(g.dead2, g.band_pp, "o-", ms=2.8, color="#1d6f80",
            label="symmetric lagged")
    a0.axhline(15, color="#999999", lw=0.5, ls=":")
    a0.axhline(35, color="#999999", lw=0.5, ls=":")
    a0.set(xlabel="dead 2-bit tier, $D_2$ (\\% of heads)",
           ylabel="heads with gain $\\geq2\\times$ (\\%)", xlim=(0, 78), ylim=(0, 98),
           title="(a) The edge survives symmetric information")
    a0.legend(frameon=False, loc="upper right", borderaxespad=0.1)

    labels = ["L31\\n8k samp.", "L31\\n8k greedy", "L31\\n32k",
              "L31\\n128k", "Q3\\n8k", "Q3\\n32k"]
    stale = np.array([3.03, 3.18, 4.30, 3.94, 2.52, 2.85])
    x = np.arange(len(labels))
    a1.bar(x, stale, color="#b0651c", alpha=0.82, width=0.66,
           label="frozen allocation / lag-1")
    a1.scatter(x, np.ones_like(x), color="#12414f", marker="D", s=13,
               zorder=3, label="route p90 regret")
    a1.axhline(1, color="#777777", lw=0.55, ls=":")
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set(ylabel="relative output error", ylim=(0.8, 4.65),
           title="(b) Slow route, fast token allocation")
    a1.legend(frameon=False, loc="upper left", borderaxespad=0.1)
    save(fig, "fig2_evidence")


if __name__ == "__main__":
    regime_figure()
    evidence_figure()
    print("wrote measured-only paper figures")
