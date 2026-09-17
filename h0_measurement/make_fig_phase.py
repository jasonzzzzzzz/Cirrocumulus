#!/usr/bin/env python3
"""
make_fig_phase.py -- docs/fig5_phase.png, the headline figure, from valid data.

Reproducible companion to the pitch/proposal. Regenerate after any campaign:

  python h0_measurement/make_fig_phase.py \
      "h0_measurement/results/job199*/*.parquet" -o docs/fig5_phase.png

Four panels:

  0  ROUTED GAIN vs dead-2-bit-tier fraction. The phase axis against the robust
     statistic. dead-2 is derived (tau^2*c_b vs the derived eviction cost c0=1)
     and contains no L. Reported with model identity PARTIALLED OUT, because 24
     configurations nested in 6 models are not 24 independent points -- and the
     within-model relationship is stronger than the pooled one. phi=n95/L inset
     for contrast: still null-to-wrong-signed.
  1  BAND FRACTION vs dead-2. The same story in the thresholded statistic, kept
     because the GO/STOP verdicts are defined on it, but demoted: it is a count
     at a 2x boundary and a 1.02-1.41x change in corner error moves it 6-29 pts.
     Both panels plot the oracle corner hollow for contrast.
  2  the context sweep, against the real evictor (the verdict corner).
  3  the theory's FORWARD prediction: the linearized cost model against the
     exactly-recomputed gain. This replaced a tau/ln2 panel that plotted x
     against x -- std_i(log2 a_i) = tau/ln2 identically; see
     sievelib/alloc.py and tests/test_units.py::test_ladder_identity.

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
                                 phi=100 * ph["eff_frac"].median(),
                                 # median measured gain, and the linearized cost
                                 # model's ratio to it -- the forward prediction
                                 # that replaces the tau/ln2 identity (panel 3).
                                 gain_or=float(gb.median()),
                                 gain_pr=(float(pr.median()) if len(pr) else np.nan),
                                 # routed gain = geometric mean of max(gain,1):
                                 # the paper's own magnitude statistic, and NOT a
                                 # thresholded count, so it does not inherit the
                                 # band fraction's fragility at the 2x boundary.
                                 routed_or=float(np.exp(np.log(np.maximum(gb, 1.0)).mean())),
                                 routed_pr=(float(np.exp(np.log(np.maximum(pr, 1.0)).mean()))
                                            if len(pr) else np.nan),
                                 lin=(float(ph["lin_ratio3"].median())
                                      if "lin_ratio3" in ph else np.nan)))
    if n_blank:
        print(f"  dropped the practical aggregate on {n_blank:,} partial-corner row(s)")
    t = pd.DataFrame(rows).drop_duplicates(["model", "ctx"]).reset_index(drop=True)
    # `band` is the VERDICT series: the real evictor where it exists, else oracle.
    t["band"] = t.band_pr.where(t.band_pr.notna(), t.band_or)
    t["has_pr"] = t.band_pr.notna()
    # `gain` follows `band`: the honest corner where it exists, else the oracle.
    t["gain"] = t.gain_pr.where(t.gain_pr.notna(), t.gain_or)
    t["routed"] = t.routed_pr.where(t.routed_pr.notna(), t.routed_or)
    return t


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def partial_spearman(x, y, group):
    """Spearman with `group` residualised out of BOTH ranks.

    24 configurations nested in 6 models are not 24 independent points, and a
    reviewer will say so. This answers it: regress each rank on model one-hots
    and correlate the residuals, i.e. the purely WITHIN-model relationship.
    """
    r = lambda v: pd.Series(v).rank().values
    X = pd.get_dummies(pd.Series(list(group)), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(X)), X])
    res = []
    for v in (r(x), r(y)):
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        res.append(v - X @ beta)
    return float(np.corrcoef(res[0], res[1])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="docs/fig5_phase.png")
    a = ap.parse_args()
    t = collect(a.inputs)
    if t.empty:
        raise SystemExit("no measurement parquet matched")
    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.6))

    # ---- PANEL 0: the phase axis against the ROBUST statistic.
    # Band fraction is a thresholded count, and we report ourselves that a
    # 1.02-1.41x change in corner error moves it 6-29 points because the per-head
    # gain distribution is dense at the 2x boundary. So the primary panel uses
    # ROUTED GAIN -- geometric mean of max(gain,1), the paper's own magnitude
    # statistic and not a count. The thresholded version is panel 1, demoted.
    pr = t[t.has_pr]
    for m, g in t.groupby("model"):
        ax[0].scatter(g.dead2, g.routed, s=26 + 34 * np.log2(g.ctx / 4096),
                      color=COL.get(m, "#666"), label=m, zorder=3,
                      edgecolor="white", linewidth=.6)
    if len(pr):
        ax[0].scatter(pr.dead2, pr.routed_or, s=26 + 34 * np.log2(pr.ctx / 4096),
                      facecolor="none", edgecolor="#8a97a0", linewidth=.9,
                      zorder=2, label="same points, oracle corner")
        for _, r in pr.iterrows():
            ax[0].plot([r.dead2, r.dead2], [r.routed_or, r.routed_pr],
                       color="#8a97a0", lw=.7, alpha=.55, zorder=1)
    ax[0].axhline(1.0, color="#c2334d", ls=":", lw=1.2)
    ax[0].text(.99, .02, "no gain", fontsize=7, color="#c2334d", ha="right",
               va="bottom", transform=ax[0].transAxes)
    ax[0].set_xlabel("dead 2-bit tier: heads with $\\tau^2 c_2 > c_0{=}1$   (%)")
    ax[0].set_ylabel("routed gain  (geo-mean of max(gain,1))")
    # The partial correlation is the headline number here: 24 configs nested in
    # 6 models are not 24 independent points, and residualising model identity
    # makes the relationship STRONGER, not weaker.
    pp = partial_spearman(t.dead2, t.routed, t.model)
    wm = [spearman(g.dead2, g.routed) for _, g in t.groupby("model") if len(g) >= 3]
    ax[0].set_title("The phase axis, against the robust statistic\n"
                    f"Spearman {spearman(t.dead2, t.routed):+.3f} pooled, "
                    f"{pp:+.3f} with model identity partialled out\n"
                    f"within-model: {min(wm):+.2f} to {max(wm):+.2f} "
                    f"across {len(wm)} models", fontsize=8.6)
    ax[0].legend(fontsize=6.2, loc="upper right", framealpha=.9)
    ins = ax[0].inset_axes([.13, .07, .25, .22])
    ins.scatter(t.phi, t.routed, s=6, color="#9b2c3a")
    ins.set_title(f"retired: $\\varphi=n_{{95}}/L$   {spearman(t.phi, t.routed):+.2f}",
                  fontsize=5.8, pad=2)
    ins.tick_params(labelsize=4.5)

    # ---- PANEL 1: the same story in the FRAGILE statistic, kept because the
    # GO/STOP verdicts are defined on it and the oracle->practical shift is what
    # removed every STOP.
    for m, g in t.groupby("model"):
        ax[1].scatter(g.dead2, g.band, s=26 + 34 * np.log2(g.ctx / 4096),
                      color=COL.get(m, "#666"), zorder=3,
                      edgecolor="white", linewidth=.6)
    if len(pr):
        ax[1].scatter(pr.dead2, pr.band_or, s=26 + 34 * np.log2(pr.ctx / 4096),
                      facecolor="none", edgecolor="#8a97a0", linewidth=.9, zorder=2)
        for _, r in pr.iterrows():
            ax[1].plot([r.dead2, r.dead2], [r.band_or, r.band_pr],
                       color="#8a97a0", lw=.7, alpha=.55, zorder=1)
    ax[1].axhspan(F_GO, 100, color="#1f6b52", alpha=.07, zorder=0)
    ax[1].axhspan(0, F_STOP, color="#c2334d", alpha=.07, zorder=0)
    ax[1].axhline(F_GO, color="#1f6b52", ls="--", lw=1)
    ax[1].axhline(F_STOP, color="#c2334d", ls="--", lw=1)
    ax[1].text(.99, F_GO + 2, "GO", fontsize=8, color="#1f6b52", ha="right",
               transform=ax[1].get_yaxis_transform())
    ax[1].text(.99, F_STOP - 6, "STOP", fontsize=8, color="#c2334d", ha="right",
               transform=ax[1].get_yaxis_transform())
    ax[1].set_xlabel("dead 2-bit tier   (%)")
    ax[1].set_ylabel("% heads in band (thresholded at 2$\\times$)")
    ax[1].set_title("Secondary: the thresholded count\n"
                    f"Spearman {spearman(t.dead2, t.band):+.3f} practical, "
                    f"{spearman(t.dead2, t.band_or):+.3f} oracle\n"
                    "hollow = oracle corner; every STOP is an oracle artifact",
                    fontsize=8.6)

    # ---- CENTRE: context sweep
    for m, g in t.groupby("model"):
        if len(g) < 3:
            continue
        g = g.sort_values("ctx")
        ax[2].plot(g.ctx / 1024, g.band, "o-", color=COL.get(m, "#666"), lw=2,
                   label=f"{m}", zorder=3)
        if g.has_pr.all():
            ax[2].plot(g.ctx / 1024, g.band_or, "o:", color=COL.get(m, "#666"),
                       lw=1, ms=3, alpha=.45, zorder=2)
    ax[2].axhspan(F_GO, 100, color="#1f6b52", alpha=.07)
    ax[2].axhspan(0, F_STOP, color="#c2334d", alpha=.07)
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("context length (k tokens)")
    ax[2].set_ylabel("% heads in band")
    slopes = []
    for m, g in t.groupby("model"):
        if len(g) >= 3:
            g = g.sort_values("ctx")
            slopes.append(np.polyfit(np.log2(g.ctx), g.band, 1)[0])
    ax[2].set_title("Context length is a phase variable\n"
                    f"solid = real evictor (the verdict), dotted = oracle\n"
                    f"{min(slopes):+.1f} to {max(slopes):+.1f} band-pts per ctx doubling",
                    fontsize=8.6)
    ax[2].legend(fontsize=6.6, loc="upper right")

    # ---- RIGHT: the theory's only forward prediction
    #
    # This panel used to plot tau/ln2 against the measured ladder width. That is
    # x against x: log2 a_i = s_i/ln2 - log2 Z with Z constant in i, so
    # std_i(log2 a_i) = tau/ln2 IDENTICALLY (8e-16 per head over all 24 configs,
    # pinned by tests/test_units.py::test_ladder_identity). The only empirical
    # content was the value term, which we separately measure at <=0.035 bits, so
    # the advertised "~1% agreement" measured a term we call negligible.
    #
    # Replaced with the real thing: the LINEARIZED COST MODEL against the
    # EXACTLY-RECOMPUTED gain. lin_ratio = predicted/measured, so predicted =
    # lin_ratio * measured, and nothing here is algebraically forced -- the cost
    # model could have been off by 5x, as it was before the probe fix.
    if "lin" in t and t.lin.notna().any():
        u = t.dropna(subset=["lin", "gain"])
        ax[3].scatter(u.gain, u.lin * u.gain, s=30,
                      color=[COL.get(m, "#666") for m in u.model],
                      zorder=3, edgecolor="white", linewidth=.6)
        lo = float(min(u.gain.min(), (u.lin * u.gain).min())) * .9
        hi = float(max(u.gain.max(), (u.lin * u.gain).max())) * 1.1
        ax[3].plot([lo, hi], [lo, hi], ls="--", color="#c2334d", lw=1.3,
                   label="exact agreement")
        # Band = the MEASURED envelope of the ratio, not a round number chosen to
        # contain it. If a future campaign widens the spread, the band widens too.
        rlo, rhi = float(u.lin.min()), float(u.lin.max())
        ax[3].fill_between([lo, hi], [lo * rlo, hi * rlo], [lo * rhi, hi * rhi],
                           color="#4ea8b8", alpha=.13, lw=0,
                           label=f"measured envelope {rlo:.2f}–{rhi:.2f}$\\times$")
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ax[3].set_xscale("log"); ax[3].set_yscale("log")
        ticks = [t for t in (1, 1.5, 2, 3, 5, 8) if lo <= t <= hi]
        for axis in (ax[3].xaxis, ax[3].yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax[3].set_xlabel("measured gain over best corner @3b")
        ax[3].set_ylabel("predicted gain, linearized cost model")
        ax[3].set_title(f"The theory's forward prediction\n"
                        f"ratio {u.lin.min():.2f}–{u.lin.max():.2f} across "
                        f"{len(u)} configurations", fontsize=9.5)
        ax[3].legend(fontsize=6.4, loc="upper left")
    else:
        ax[3].text(.5, .5, "lin_ratio3 not present in these parquets",
                   ha="center", va="center", fontsize=8, transform=ax[3].transAxes)
        ax[3].set_title("The theory's forward prediction", fontsize=9.5)

    for x in ax:
        x.grid(alpha=.22)
        x.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}  ({len(t)} points, {int(t.has_pr.sum())} with a real evictor)")
    print(f"  Spearman(dead2, ROUTED)          = {spearman(t.dead2, t.routed):+.3f}"
          f"   partial (model out) = {partial_spearman(t.dead2, t.routed, t.model):+.3f}")
    print(f"  Spearman(dead2, band  practical) = {spearman(pr.dead2, pr.band_pr):+.3f}"
          f"   partial (model out) = {partial_spearman(t.dead2, t.band, t.model):+.3f}"
          if len(pr) > 2 else "")
    print(f"  Spearman(dead2, band  oracle)    = {spearman(t.dead2, t.band_or):+.3f}")
    print(f"  Spearman(ladder, ROUTED)         = {spearman(t.lad, t.routed):+.3f}"
          f"   partial (model out) = {partial_spearman(t.lad, t.routed, t.model):+.3f}")
    print(f"  Spearman(phi,   band)            = {spearman(t.phi, t.band):+.3f}")
    u = t.dropna(subset=["lin", "gain"])
    if len(u):
        print(f"  lin_ratio3 (forward prediction)  = {u.lin.min():.2f}-{u.lin.max():.2f}"
              f"  over {len(u)} configs")
    # The tau/ln2 "prediction" is an identity (std_i(log2 a_i) = tau/ln2 exactly);
    # it is no longer plotted or reported here. See alloc.py and
    # tests/test_units.py::test_ladder_identity.


if __name__ == "__main__":
    main()
