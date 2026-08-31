#!/usr/bin/env python3
"""
make_fig_phase.py -- docs/fig5_phase.png, the headline figure, from valid data.

Reproducible companion to the pitch/proposal. Regenerate after any campaign:

  python h0_measurement/make_fig_phase.py \
      "h0_measurement/results/job199*/*.parquet" -o docs/fig5_phase.png

Three panels, each carrying one of the paper's three empirical claims:

  LEFT   band fraction vs DEAD-2-BIT-TIER fraction. This is the phase axis. It is
         derived (tau^2*c_b vs the derived eviction cost c0=1), it contains no L,
         and it holds across both architecture and context length at rho=-0.96.
         The retired variable phi=n95/L is shown inset for contrast: on valid data
         phi correlates with the outcome at +0.26 -- the WRONG SIGN.
  CENTRE the context sweep. One model crosses all three verdicts with nothing
         varied but L, because tau rises monotonically with L and tau is what
         drives tier extinction.
  RIGHT  the closed-form ladder width tau/ln2 against measurement.
"""
from __future__ import annotations
import argparse, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F_STOP, F_GO, BAND_MIN = 15.0, 35.0, 2.0
COL = {"llama31-8b": "#1d6f80", "llama33-70b": "#12414f", "mistral-7b": "#1f6b52",
       "qwen15-moe-a2.7b": "#a8611f", "qwen3-8b": "#7c3aed",
       "qwen3-30b-a3b-2507": "#c2334d"}


def collect(pats):
    rows = []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            d = pd.read_parquet(f)
            if "gain_best3" not in d:
                continue
            for (m, c), g in d.groupby(["model", "ctx"]):
                ph = g.groupby(["layer", "head"]).median(numeric_only=True)
                gb = ph["gain_best3"].dropna()
                if not len(gb):
                    continue
                rows.append(dict(model=m, ctx=int(c),
                                 band=100 * (gb >= BAND_MIN).mean(),
                                 dead2=100 * ph["evict_beats_b2"].mean(),
                                 tau=ph["tau"].median(), lad=ph["ladder_bits"].median(),
                                 phi=100 * ph["eff_frac"].median()))
    return pd.DataFrame(rows).drop_duplicates(["model", "ctx"]).reset_index(drop=True)


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="docs/fig5_phase.png")
    a = ap.parse_args()
    t = collect(a.inputs)
    if t.empty:
        raise SystemExit("no measurement parquet matched")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # ---- LEFT: the phase axis
    for m, g in t.groupby("model"):
        ax[0].scatter(g.dead2, g.band, s=26 + 34 * np.log2(g.ctx / 4096),
                      color=COL.get(m, "#666"), label=m, zorder=3,
                      edgecolor="white", linewidth=.6)
    ax[0].axhspan(F_GO, 100, color="#1f6b52", alpha=.07, zorder=0)
    ax[0].axhspan(0, F_STOP, color="#c2334d", alpha=.07, zorder=0)
    ax[0].axhline(F_GO, color="#1f6b52", ls="--", lw=1)
    ax[0].axhline(F_STOP, color="#c2334d", ls="--", lw=1)
    ax[0].text(97, F_GO + 2, "GO", fontsize=8, color="#1f6b52", ha="right")
    ax[0].text(97, F_STOP - 5, "STOP", fontsize=8, color="#c2334d", ha="right")
    ax[0].set_xlabel("dead 2-bit tier: heads with $\\tau^2 c_2 > c_0{=}1$   (%)")
    ax[0].set_ylabel("% heads in band (interior beats both corners)")
    ax[0].set_title(f"The phase axis is derived, and L-free\n"
                    f"Spearman = {spearman(t.dead2, t.band):+.3f} over {len(t)} "
                    f"(model, ctx) points", fontsize=9.5)
    ax[0].legend(fontsize=6.2, loc="upper right", framealpha=.9)
    ins = ax[0].inset_axes([.11, .06, .26, .23])
    ins.scatter(t.phi, t.band, s=6, color="#9b2c3a")
    ins.set_title(f"retired: $\\varphi=n_{{95}}/L$   {spearman(t.phi, t.band):+.2f}",
                  fontsize=5.8, pad=2)
    ins.tick_params(labelsize=4.5)

    # ---- CENTRE: context sweep
    for m, g in t.groupby("model"):
        if len(g) < 3:
            continue
        g = g.sort_values("ctx")
        ax[1].plot(g.ctx / 1024, g.band, "o-", color=COL.get(m, "#666"), lw=2,
                   label=f"{m}", zorder=3)
    ax[1].axhspan(F_GO, 100, color="#1f6b52", alpha=.07)
    ax[1].axhspan(0, F_STOP, color="#c2334d", alpha=.07)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("context length (k tokens)")
    ax[1].set_ylabel("% heads in band")
    ax[1].set_title("Context length is a phase variable\n"
                    "one model, GO $\\to$ NARROW $\\to$ STOP", fontsize=9.5)
    ax[1].legend(fontsize=6.6, loc="upper right")

    # ---- RIGHT: the closed form
    pred = t.tau / np.log(2)
    err = 100 * (pred - t.lad).abs() / t.lad
    for m, g in t.groupby("model"):
        ax[2].scatter(g.tau / np.log(2), g.lad, s=30, color=COL.get(m, "#666"),
                      zorder=3, edgecolor="white", linewidth=.6)
    lim = [min(pred.min(), t.lad.min()) * .95, max(pred.max(), t.lad.max()) * 1.05]
    ax[2].plot(lim, lim, ls="--", color="#c2334d", lw=1.3)
    ax[2].set_xlabel("predicted ladder width  $\\tau/\\ln 2$  (bits)")
    ax[2].set_ylabel("measured ladder width (bits)")
    ax[2].set_title(f"Closed form, no fitted parameters\n"
                    f"worst error {err.max():.1f}%, median {err.median():.1f}%",
                    fontsize=9.5)

    for x in ax:
        x.grid(alpha=.22)
        x.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}  ({len(t)} points)")
    print(f"  Spearman(dead2, band) = {spearman(t.dead2, t.band):+.3f}")
    print(f"  Spearman(phi,   band) = {spearman(t.phi,   t.band):+.3f}")
    print(f"  ladder worst error    = {err.max():.2f}%")


if __name__ == "__main__":
    main()
