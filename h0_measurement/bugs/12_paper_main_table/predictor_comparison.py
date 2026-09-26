#!/usr/bin/env python3
"""R10 anchor: does D2 order the interior's value better than simpler predictors?

    OMP_NUM_THREADS=8 .venv/bin/python h0_measurement/bugs/12_paper_main_table/predictor_comparison.py

Cells: the 25 symmetric regime-map cells in latex/figures/r6_symmetric.csv
(target: hindsight portfolio gain G+ = `portfolio_gain`, and the band).
Predictors are cell medians of per-head medians, computed on the same rows
boundary.py uses (quantized, complete corners, floor_maxb runs, R3 corner set):

  D2          % heads whose 2-bit tier is dominated (evict_beats_b2)   [ours]
  ladder      median value-aware ladder width, i.e. the spread of 0.5*log2 w;
              for log-normal w, log(AM/GM of w) = var(ln w)/2, so this ranks
              cells like RateQuant's AM/GM heterogeneity predictor
  tau         attention concentration (median)
  entropy     attention entropy (median; lower = more concentrated)
  n95_frac    tokens holding 95% of attention mass, as a fraction of L
  log2_ctx    context-only baseline

Protocol, fixed before looking at the result: (1) pooled Spearman with G+ over
25 cells; (2) Spearman partialled on model identity; (3) leave-one-model-out:
fit log G+ = a + b*x on five models, predict the held-out model's cells, report
the mean absolute error in log G+ (and as a multiplicative factor) and the
within-held-out-model Spearman. A context-only predictor has no cross-model
information, so it is the baseline any regime coordinate must beat.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
sys.path.insert(0, H0)
sys.path.insert(0, os.path.join(H0, "bugs", "6_pin_sharp_boundary"))
from report import drop_partial_corners, cfg_value          # noqa: E402
from make_fig_phase import spearman, partial_spearman       # noqa: E402
import boundary as BD                                        # noqa: E402

COLS = ["evict_beats_b2", "ladder_bits", "tau", "entropy", "n95", "L"]


def cell_predictors(files):
    """(model, ctx) -> per-head-median predictor sums, pooled over replicate runs."""
    acc = {}
    for f in files:
        if not BD.fixed_interior(f):
            continue
        try:
            d = pd.read_parquet(f, columns=None)
        except Exception:
            continue
        if "quantized" in d:
            d = d[d["quantized"]]
        d, _ = drop_partial_corners(d)
        if "gain_best_practical3" in d and d["gain_best_practical3"].notna().any():
            d = d[d["gain_best_practical3"].notna()]
        have = [c for c in COLS if c in d]
        if "evict_beats_b2" not in have:
            continue
        for (m, c), g in d.groupby(["model", "ctx"]):
            if cfg_value(g, "evictors") != "oracle,accum":
                continue
            ph = g.groupby(["layer", "head"])[have].median().reset_index()
            acc.setdefault((m, int(c)), []).append(ph)
    rows = []
    for (m, c), phs in acc.items():
        ph = pd.concat(phs)
        r = dict(model=m, ctx=c, n_runs=len(phs),
                 D2=100 * ph["evict_beats_b2"].mean(),
                 ladder=ph["ladder_bits"].median() if "ladder_bits" in ph else np.nan,
                 tau=ph["tau"].median() if "tau" in ph else np.nan,
                 entropy=ph["entropy"].median() if "entropy" in ph else np.nan,
                 n95_frac=(ph["n95"] / ph["L"]).median() if {"n95", "L"} <= set(ph) else np.nan,
                 log2_ctx=np.log2(c))
        rows.append(r)
    return pd.DataFrame(rows)


def lomo(tab, x, y="lg"):
    errs, within = [], []
    for m in tab.model.unique():
        tr, te = tab[tab.model != m], tab[tab.model == m]
        b, a = np.polyfit(tr[x], tr[y], 1)
        errs.extend(np.abs(a + b * te[x] - te[y]))
        if len(te) >= 3:
            within.append(spearman(te[x], te[y]))
    return float(np.mean(errs)), (float(np.mean(within)) if within else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="*", default=[
        os.path.join(H0, "results", "job*", "h0_*.parquet")])
    ap.add_argument("--out", default=os.path.join(HERE, "predictor_comparison.csv"))
    a = ap.parse_args()
    files = sorted({f for p in a.glob for f in glob.glob(p)})
    pred = cell_predictors(files)
    r6 = pd.read_csv(os.path.join(ROOT, "latex", "figures", "r6_symmetric.csv"))
    tab = r6.merge(pred, on=["model", "ctx"], how="left", suffixes=("", "_re"))
    miss = tab[tab.D2.isna()]
    if len(miss):
        print("cells without predictors:", miss[["model", "ctx"]].values.tolist())
    tab = tab.dropna(subset=["D2"])
    dd = (tab.D2 - tab.dead2).abs()
    print(f"cells matched: {len(tab)}/25; |D2 recomputed - D2 paper|: median {dd.median():.2f}, "
          f"max {dd.max():.2f} pts")
    tab["lg"] = np.log(tab.portfolio_gain)
    tab["D2"] = tab.dead2                    # the paper's value, for the comparison
    tab.to_csv(a.out, index=False)
    print(f"\n{'predictor':10s} {'rho(G+)':>8s} {'rho|model':>9s} {'rho(band)':>9s} "
          f"{'LOMO |err| logG+':>17s} {'x factor':>8s} {'within-model rho':>16s}")
    for x in ["D2", "ladder", "tau", "entropy", "n95_frac", "log2_ctx"]:
        if tab[x].isna().any():
            print(f"{x:10s}  (missing in {int(tab[x].isna().sum())} cells)")
            continue
        e, w = lomo(tab, x)
        print(f"{x:10s} {spearman(tab[x], tab.portfolio_gain):+8.3f} "
              f"{partial_spearman(tab[x], tab.portfolio_gain, tab.model):+9.3f} "
              f"{spearman(tab[x], tab.band):+9.3f} {e:17.3f} {np.exp(e):8.3f}x "
              f"{w:+16.3f}")


if __name__ == "__main__":
    main()
