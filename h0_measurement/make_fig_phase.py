#!/usr/bin/env python3
"""
make_fig_phase.py -- docs/fig5_phase.png, the headline figure, from valid data.

Reproducible companion to the pitch/proposal. Regenerate after any campaign:

  python h0_measurement/make_fig_phase.py \
      "h0_measurement/results/job199*/*.parquet" -o docs/fig5_phase.png

Three panels, each carrying one of the paper's three empirical claims:

  LEFT   band fraction vs DEAD-2-BIT-TIER fraction. This is the phase axis. It is
         derived (tau^2*c_b vs the derived eviction cost c0=1), it contains no L,
         and it holds across both architecture and context length. Plotted TWICE:
         once against the oracle eviction corner and once against a real evictor.
         The two series differ by 10-29 points of band and the correlation is
         unmoved -- that invariance is the robustness claim, so the figure has to
         show both. The retired variable phi=n95/L is inset for contrast: it is
         null-to-wrong-signed either way.
  CENTRE the context sweep, against the real evictor (the verdict corner), with
         the oracle shown faint for contrast.
  RIGHT  the closed-form ladder width tau/ln2 against measurement. Corner-
         independent: it is a property of the quantizer, not of the baseline.

The band column is chosen per file: gain_best_practical3 when the run carried a
practical corner, else gain_best3. Rows where only SOME evictors scored are
dropped first (report.drop_partial_corners) -- on the first decode step only
`recency` has a score, so min-over-corners silently collapses onto the weakest
one and the band reads ~29 points high.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F_STOP, F_GO, BAND_MIN = 15.0, 35.0, 2.0
COL = {"llama31-8b": "#1d6f80", "llama33-70b": "#12414f", "mistral-7b": "#1f6b52",
       "qwen15-moe-a2.7b": "#a8611f", "qwen3-8b": "#7c3aed",
       "qwen3-30b-a3b-2507": "#c2334d"}


def collect(pats):
    from report import drop_partial_corners
    rows, n_blank = [], 0
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            d = pd.read_parquet(f)
            if "gain_best3" not in d:
                continue
            if "quantized" in d:
                d = d[d.quantized]
            d, nb = drop_partial_corners(d)
            n_blank += nb
            for (m, c), g in d.groupby(["model", "ctx"]):
                ph = g.groupby(["layer", "head"]).median(numeric_only=True)
                gb = ph["gain_best3"].dropna()
                if not len(gb):
                    continue
                pr = (ph["gain_best_practical3"].dropna()
                      if "gain_best_practical3" in ph else pd.Series(dtype=float))
                rows.append(dict(model=m, ctx=int(c),
                                 band_or=100 * (gb >= BAND_MIN).mean(),
                                 band_pr=(100 * (pr >= BAND_MIN).mean()
                                          if len(pr) else np.nan),
                                 dead2=100 * ph["evict_beats_b2"].mean(),
                                 tau=ph["tau"].median(), lad=ph["ladder_bits"].median(),
                                 phi=100 * ph["eff_frac"].median()))
    if n_blank:
        print(f"  dropped the practical aggregate on {n_blank:,} partial-corner row(s)")
    t = pd.DataFrame(rows).drop_duplicates(["model", "ctx"]).reset_index(drop=True)
    # `band` is the VERDICT series: the real evictor where it exists, else oracle.
    t["band"] = t.band_pr.where(t.band_pr.notna(), t.band_or)
    t["has_pr"] = t.band_pr.notna()
    return t


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

    # ---- LEFT: the phase axis, under BOTH corner definitions.
    # The point of showing both: the two series are 10-29 points apart and the
    # correlation is the same. A regime variable that survives that shift in its
    # own target is not an artifact of how the baseline was specified.
    pr = t[t.has_pr]
    for m, g in t.groupby("model"):
        ax[0].scatter(g.dead2, g.band, s=26 + 34 * np.log2(g.ctx / 4096),
                      color=COL.get(m, "#666"), label=m, zorder=3,
                      edgecolor="white", linewidth=.6)
    if len(pr):
        ax[0].scatter(pr.dead2, pr.band_or, s=26 + 34 * np.log2(pr.ctx / 4096),
                      facecolor="none", edgecolor="#8a97a0", linewidth=.9,
                      zorder=2, label="same points, oracle corner")
        for _, r in pr.iterrows():
            ax[0].plot([r.dead2, r.dead2], [r.band_or, r.band_pr],
                       color="#8a97a0", lw=.7, alpha=.55, zorder=1)
    ax[0].axhspan(F_GO, 100, color="#1f6b52", alpha=.07, zorder=0)
    ax[0].axhspan(0, F_STOP, color="#c2334d", alpha=.07, zorder=0)
    ax[0].axhline(F_GO, color="#1f6b52", ls="--", lw=1)
    ax[0].axhline(F_STOP, color="#c2334d", ls="--", lw=1)
    ax[0].text(97, F_GO + 2, "GO", fontsize=8, color="#1f6b52", ha="right")
    ax[0].text(97, F_STOP - 5, "STOP", fontsize=8, color="#c2334d", ha="right")
    ax[0].set_xlabel("dead 2-bit tier: heads with $\\tau^2 c_2 > c_0{=}1$   (%)")
    ax[0].set_ylabel("% heads in band (interior beats both corners)")
    sub = f"Spearman = {spearman(t.dead2, t.band):+.3f} over {len(t)} (model, ctx) points"
    if len(pr) > 2:
        sub = (f"Spearman {spearman(pr.dead2, pr.band_pr):+.3f} vs a real evictor, "
               f"{spearman(pr.dead2, pr.band_or):+.3f} vs the oracle\n"
               f"{len(pr)} (model, ctx) points; the corner moves the band 10-29 pts, "
               f"not the axis")
    ax[0].set_title("The phase axis is derived, L-free, and\n"
                    "invariant to the baseline definition\n" + sub, fontsize=8.6)
    ax[0].legend(fontsize=6.2, loc="upper right", framealpha=.9)
    ins = ax[0].inset_axes([.11, .06, .26, .23])
    ins.scatter(t.phi, t.band, s=6, color="#9b2c3a")
    ins.set_title(f"retired: $\\varphi=n_{{95}}/L$   {spearman(t.phi, t.band):+.2f}",
                  fontsize=5.8, pad=2)   # still null-to-wrong-signed
    ins.tick_params(labelsize=4.5)

    # ---- CENTRE: context sweep
    for m, g in t.groupby("model"):
        if len(g) < 3:
            continue
        g = g.sort_values("ctx")
        ax[1].plot(g.ctx / 1024, g.band, "o-", color=COL.get(m, "#666"), lw=2,
                   label=f"{m}", zorder=3)
        if g.has_pr.all():
            ax[1].plot(g.ctx / 1024, g.band_or, "o:", color=COL.get(m, "#666"),
                       lw=1, ms=3, alpha=.45, zorder=2)
    ax[1].axhspan(F_GO, 100, color="#1f6b52", alpha=.07)
    ax[1].axhspan(0, F_STOP, color="#c2334d", alpha=.07)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("context length (k tokens)")
    ax[1].set_ylabel("% heads in band")
    slopes = []
    for m, g in t.groupby("model"):
        if len(g) >= 3:
            g = g.sort_values("ctx")
            slopes.append(np.polyfit(np.log2(g.ctx), g.band, 1)[0])
    ax[1].set_title("Context length is a phase variable\n"
                    f"solid = real evictor (the verdict), dotted = oracle\n"
                    f"{min(slopes):+.1f} to {max(slopes):+.1f} band-pts per ctx doubling",
                    fontsize=8.6)
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
    print(f"wrote {a.out}  ({len(t)} points, {int(t.has_pr.sum())} with a real evictor)")
    print(f"  Spearman(dead2, band  practical) = {spearman(pr.dead2, pr.band_pr):+.3f}"
          if len(pr) > 2 else "")
    print(f"  Spearman(dead2, band  oracle)    = {spearman(t.dead2, t.band_or):+.3f}")
    print(f"  Spearman(phi,   band)            = {spearman(t.phi, t.band):+.3f}")
    print(f"  ladder worst error               = {err.max():.2f}%")


if __name__ == "__main__":
    main()
