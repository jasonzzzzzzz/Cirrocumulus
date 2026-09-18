#!/usr/bin/env python3
"""
make_paper_figs.py -- figures for latex/main.tex (4-page milestone draft).

  fig1_teaser.pdf   (a) mechanism: optimal bit profile in each regime, computed
                        with the real water-filling rule on synthetic logits
                    (b) MEASURED: 24 configurations as per-model trajectories in
                        the (dead-2-bit-tier fraction, routed gain) plane
                    (c) PROJECTED: RULER accuracy vs context (R8, not yet run)
  fig2_results.pdf  (a) MEASURED: band fraction vs context, symmetric comparison
                    (b) MEASURED: linearized cost model, predicted/measured gain
                    (c) PROJECTED: router on/off vs dead-tier fraction (R8)

Measured numbers come from h0_configs.csv, produced by
h0_measurement/make_fig_phase.py::collect over the E2 campaign parquets
(results/job200*/), replicates averaged per (model, ctx).

Every PROJECTED panel is hatched and stamped; replace with measured data when
the corresponding experiment lands.

  ../../.venv/bin/python make_paper_figs.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

COL = {"llama33-70b": "#12414f", "mistral-7b": "#1f6b52", "llama31-8b": "#1d8a9c",
       "qwen15-moe-a2.7b": "#b0651c", "qwen3-8b": "#7c3aed",
       "qwen3-30b-a3b-2507": "#c2334d"}
NAME = {"llama33-70b": "Llama-3.3-70B", "mistral-7b": "Mistral-7B",
        "llama31-8b": "Llama-3.1-8B", "qwen15-moe-a2.7b": "Qwen1.5-MoE-A2.7B",
        "qwen3-8b": "Qwen3-8B", "qwen3-30b-a3b-2507": "Qwen3-30B-A3B"}
ORDER = list(COL)
C_SIEVE, C_EVICT, C_UNI, C_FULL, C_STATIC = "#c2334d", "#1d6f80", "#8a8a8a", "#222222", "#b0651c"

d = pd.read_csv(os.path.join(HERE, "h0_configs.csv"))


def stamp(ax, text="PROJECTED"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=15, color="#999999", alpha=0.22, rotation=22, weight="bold")
    for s in ("left", "bottom"):
        ax.spines[s].set_linestyle((0, (3, 2)))


# ----------------------------------------------------------------------------
# water-filling on synthetic logits (mechanism panel)
# cost of token i at b bits = a_i^2 * sig2[b];  sig2[0] = 1 (eviction, derived)
# sig2[b] = tau^2 * c_b with c_b = kappa * D_b, D_b = Lloyd-Max Gaussian MSE.
# kappa=1.6 puts the 2-bit extinction point tau^2 c_2 = 1 at tau ~ 2.3,
# matching where the measured dead-2 fraction crosses 50%.
# ----------------------------------------------------------------------------
LM = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497}
def c_rel(b):
    return 1.6 * (LM[b] if b in LM else LM[4] * 4.0 ** -(b - 4))

def waterfill(a, tau, B=3.0, maxb=8):
    bits = np.arange(maxb + 1)
    sig2 = np.array([1.0] + [tau ** 2 * c_rel(b) for b in bits[1:]])
    w2 = np.maximum(a ** 2, 1e-300)
    lo, hi = 1e-40, 1e40
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        idx = np.argmin(sig2[None, :] + lam * bits[None, :] / w2[:, None], axis=1)
        if bits[idx].sum() > B * len(a):
            lo = lam
        else:
            hi = lam
    return bits[idx]


def teaser():
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.78),
                           gridspec_kw=dict(width_ratios=[1, 1.12, 1]))
    # (a) mechanism ---------------------------------------------------------
    rng = np.random.default_rng(0)
    L = 4096
    z = np.sort(rng.standard_normal(L))[::-1]
    regimes = [(0.35, "diffuse ($\\tau$=0.35): ~uniform", C_UNI),
               (2.2, "band ($\\tau$=2.2): ladder", C_SIEVE),
               (4.5, "sharp ($\\tau$=4.5): keep/evict", C_EVICT)]
    x = np.arange(1, L + 1)
    for k, (tau, lab, c) in enumerate(regimes):
        s = tau * z
        a = np.exp(s - s.max()); a /= a.sum()
        b = waterfill(a, tau)
        ax[0].step(x, b + 0.08 * (1 - k), where="post", color=c, label=lab, lw=1.2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("token rank by attention $a_i$")
    ax[0].set_ylabel("optimal bits $b_i^\\star$")
    ax[0].set_ylim(-0.4, 8.6); ax[0].set_yticks([0, 2, 4, 6, 8])
    ax[0].axhline(3, color="#bbbbbb", lw=0.6, ls=":")
    ax[0].text(300, 3.25, "budget $B$=3", color="#999999", fontsize=5.5)
    ax[0].legend(loc="lower left", frameon=False, handlelength=1.2,
                 borderaxespad=0.1, fontsize=5.5)
    ax[0].set_title("(a) one objective, three regimes")

    # (b) measured trajectories -------------------------------------------
    a1 = ax[1]
    a1.axhspan(2.0, 3.4, color="#c2334d", alpha=0.06, lw=0)
    a1.axhline(1.0, color="#999999", lw=0.6, ls=":")
    for m in ORDER:
        g = d[d.model == m].sort_values("ctx")
        sz = 6 + 5 * (np.log2(g.ctx) - 13)
        a1.plot(g.dead2, g.routed_or, color=COL[m], lw=0.9, alpha=0.9)
        a1.scatter(g.dead2, g.routed_or, s=sz, color=COL[m], zorder=3,
                   edgecolor="white", linewidth=0.3, label=NAME[m])
        if len(g) > 1:
            p, q = g.iloc[-2], g.iloc[-1]
            a1.annotate("", xy=(q.dead2, q.routed_or), xytext=(p.dead2, p.routed_or),
                        arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=0.8,
                                        mutation_scale=6), zorder=2)
    g70 = d[d.model == "llama33-70b"].sort_values("ctx")
    a1.annotate("8k", (g70.dead2.iloc[0], g70.routed_or.iloc[0]), xytext=(4, 1),
                textcoords="offset points", fontsize=5.5, color=COL["llama33-70b"])
    a1.annotate("128k", (g70.dead2.iloc[-1], g70.routed_or.iloc[-1]), xytext=(-2, 5),
                textcoords="offset points", fontsize=5.5, color=COL["llama33-70b"])
    a1.text(21, 2.06, "allocation pays", color="#c2334d", fontsize=6, style="italic")
    a1.text(79, 0.93, "tier extinction $\\rightarrow$ eviction", color=C_EVICT, fontsize=5.5,
            style="italic", va="bottom", ha="right")
    a1.set_xlim(0, 80); a1.set_ylim(0.88, 3.4)
    a1.set_xlabel("dead 2-bit tier: heads with $\\tau^2 c_2 > c_0$ (%)")
    a1.set_ylabel("routed gain over best corner")
    a1.set_title("(b) MEASURED: 24 configs, 6 models")
    a1.legend(loc="upper right", frameon=False, fontsize=5, handletextpad=0.1,
              borderaxespad=0.0, labelspacing=0.15, markerscale=0.8,
              bbox_to_anchor=(1.02, 1.0))

    # (c) projected end task ----------------------------------------------
    a2 = ax[2]
    ctx = np.array([8, 16, 32, 64, 128])
    full = np.array([93.8, 93.2, 87.9, 84.8, 77.0])          # BF16 reference shape
    uni = full - np.array([4.0, 4.3, 4.8, 5.5, 6.5])
    evict = full - np.array([9.5, 8.0, 6.5, 4.5, 3.6])
    sieve = full - np.array([1.2, 1.6, 2.4, 3.2, 3.3])
    a2.plot(ctx, full, color=C_FULL, ls="--", lw=0.9, label="BF16 (16 b)")
    a2.plot(ctx, uni, color=C_UNI, marker="s", ms=2.5, label="uniform 3 b")
    a2.plot(ctx, evict, color=C_EVICT, marker="^", ms=2.8, label="H2O eviction")
    a2.plot(ctx, sieve, color=C_SIEVE, marker="o", ms=2.8, lw=1.5, label="SIEVE (routed)")
    a2.fill_between(ctx, np.maximum(uni, evict), sieve, color=C_SIEVE, alpha=0.12, lw=0)
    a2.set_xscale("log", base=2); a2.set_xticks(ctx)
    a2.set_xticklabels([f"{c}k" for c in ctx])
    a2.set_xlabel("context length"); a2.set_ylabel("RULER accuracy (%)")
    a2.set_ylim(68, 97)
    a2.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.1)
    a2.set_title("(c) PROJECTED end task (TBD)")
    stamp(a2)

    fig.tight_layout(pad=0.25, w_pad=0.6)
    fig.savefig(os.path.join(HERE, "fig1_teaser.pdf"))
    fig.savefig(os.path.join(HERE, "fig1_teaser.png"), dpi=220)


def results():
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.72))
    # (a) band vs context ---------------------------------------------------
    a0 = ax[0]
    for m in ORDER:
        g = d[d.model == m].sort_values("ctx")
        a0.plot(g.ctx / 1024, g.band_or, color=COL[m], marker="o", ms=2.6, lw=1.0)
    g70 = d[d.model == "llama33-70b"].sort_values("ctx")
    a0.scatter([128], [g70.band_or.iloc[-1]], s=38, facecolor="none",
               edgecolor="black", lw=0.7, zorder=4)
    a0.annotate("out-of-sample", (128, g70.band_or.iloc[-1]), xytext=(-46, 14),
                textcoords="offset points", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    a0.set_xscale("log", base=2)
    a0.set_xticks([8, 16, 32, 64, 128]); a0.set_xticklabels(["8k", "16k", "32k", "64k", "128k"])
    a0.set_ylim(0, 85)
    a0.set_xlabel("context length")
    a0.set_ylabel("heads in band (gain $\\geq 2\\times$) (%)")
    a0.set_title("(a) context is a phase variable")

    # (b) forward prediction -----------------------------------------------
    a1 = ax[1]
    a1.axhspan(d.lin.min(), d.lin.max(), color="#1d6f80", alpha=0.10, lw=0)
    a1.axhline(1.0, color="#c2334d", lw=0.8, ls="--")
    for m in ORDER:
        g = d[d.model == m].sort_values("ctx")
        a1.scatter(g.dead2, g.lin, s=6 + 5 * (np.log2(g.ctx) - 13), color=COL[m],
                   edgecolor="white", linewidth=0.3, zorder=3, label=NAME[m])
    a1.set_yscale("log")
    a1.set_yticks([0.5, 0.75, 1, 1.5, 2]); a1.set_yticklabels(["0.5", "0.75", "1", "1.5", "2"])
    a1.minorticks_off()
    a1.set_ylim(0.5, 2.0); a1.set_xlim(0, 80)
    a1.text(40, 1.55, f"all 24 configs in [{d.lin.min():.2f}, {d.lin.max():.2f}]",
            fontsize=5.5, ha="center", color="#1d6f80")
    a1.set_xlabel("dead 2-bit tier (%)")
    a1.set_ylabel("predicted / measured gain")
    a1.set_title("(b) cost model predicts forward")
    a1.text(40, 0.55, "colours as Fig. 1(b); size = context", fontsize=5.5, ha="center",
            color="#666666")

    # (c) projected router on/off ------------------------------------------
    a2 = ax[2]
    xx = np.linspace(0, 80, 200)
    alloc = -0.8 - 0.075 * xx - 0.0012 * np.maximum(xx - 35, 0) ** 2
    evict = -8.0 + 0.065 * xx
    router = np.maximum(alloc, evict) + 0.9 * np.exp(-((xx - 45) / 16) ** 2) + 0.2
    a2.axhline(0, color="#999999", lw=0.6, ls=":")
    a2.plot(xx, alloc, color=C_STATIC, label="always allocate")
    a2.plot(xx, evict, color=C_EVICT, label="always evict (H2O)")
    a2.plot(xx, router, color=C_SIEVE, lw=1.6, label="SIEVE router")
    for m, c in (("llama33-70b", 32768), ("qwen3-30b-a3b-2507", 32768)):
        x0 = float(d[(d.model == m) & (d.ctx == c)].dead2.iloc[0])
        a2.axvline(x0, color=COL[m], lw=0.6, ls="--")
        a2.text(x0 + 1, 1.7, NAME[m].split("-")[0] + ("-70B" if "70" in m else "-30B"),
                fontsize=5, color=COL[m], rotation=90, va="top")
    a2.set_xlim(0, 80); a2.set_ylim(-10, 2)
    a2.set_xlabel("dead 2-bit tier (%)")
    a2.set_ylabel("$\\Delta$ RULER vs. BF16 (pts)")
    a2.set_title("(c) PROJECTED: router (TBD)")
    a2.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False,
              handlelength=1.4, borderaxespad=0.05, fontsize=5.5)
    stamp(a2)

    fig.tight_layout(pad=0.25, w_pad=0.6)
    fig.savefig(os.path.join(HERE, "fig2_results.pdf"))
    fig.savefig(os.path.join(HERE, "fig2_results.png"), dpi=220)


if __name__ == "__main__":
    teaser()
    results()
    print("wrote fig1_teaser.pdf, fig2_results.pdf")
