#!/usr/bin/env python3
"""
R3-figure.py -- figures and numbers for R3, the symmetric cell.

  python h0_measurement/bugs/2_towards_real_evictor/R3-figure.py \
      -o h0_measurement/bugs/2_towards_real_evictor/

THE CAMPAIGNS. Three of them ran the same cells with three different treatments
of the positions a lagged score has never seen (R3-report.md section 2):

  floor  job214217*  the FIXED campaign: unseen positions held at maxb and the
                     rest water-filled on the remaining budget. Every column
                     here is valid; the .json carries
                     "interior_unseen_policy": "floor_maxb".
  zero   job2140*, job21406*   unseen positions scored 0, so the interior
                     EVICTED the token appended each step. Interior columns
                     invalid; its oracle / E2-cell columns are fine and are kept
                     here as independent replicates.
  bump   job92*      unseen positions scored max+1 (ordinal). The controlled
                     test found this harmless, and panel 4 confirms it at scale.

Interior columns from a run without the fix are written with an `invalid_`
prefix so they cannot be quoted by accident.

Six panels:
  1  the three cells vs ctx   oracle / E2 (corner demoted) / symmetric (both
                              demoted), per model -- what R3 was built to show.
  2  phase axis               dead-2 vs the symmetric band, every cell.
  3  where the lag cost lands per-head interior lag cost vs realised gain.
  4  the fix, at scale        interior lag cost by campaign on shared cells.
  5  staleness sweep          corner, interior and band vs lag k.
  6  router miscalibration    heads an oracle-calibrated router sends to the
                              interior that do not pay for themselves.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
sys.path.insert(0, H0); sys.path.insert(0, ROOT)
from report import drop_partial_corners                     # noqa: E402

BAND_MIN, F_GO, F_STOP = 2.0, 35.0, 15.0
ORDER = ["mistral-7b", "llama33-70b", "qwen15-moe-a2.7b", "llama31-8b",
         "qwen3-8b", "qwen3-30b-a3b-2507"]
COL = {"mistral-7b": "#1f6b52", "llama33-70b": "#6a4c93", "qwen15-moe-a2.7b": "#a8611f",
       "llama31-8b": "#1d6f80", "qwen3-8b": "#8a8f2a", "qwen3-30b-a3b-2507": "#c2334d"}
SHORT = {"mistral-7b": "mistral-7b", "llama33-70b": "llama33-70b",
         "qwen15-moe-a2.7b": "qwen15-moe", "llama31-8b": "llama31-8b",
         "qwen3-8b": "qwen3-8b", "qwen3-30b-a3b-2507": "qwen3-30b"}
NEED = ["model", "ctx", "prompt", "family", "step", "quantized", "layer", "head",
        "needle_hit", "tau", "evict_beats_b2", "gain_best3", "gain_best_practical3",
        "gain_e3_oracle_frac", "gain_e3_accum_frac", "gain_pp3_accum",
        "gain_pp_sym3_accum", "interior_lag_cost3_accum", "evict_frac3",
        "evict_frac_pp3_accum", "synthetic", "n_practical", "evictors"]
LAGS = [1, 2, 4, 8]


def read(f, extra=()):
    try:
        import fastparquet
        pf = fastparquet.ParquetFile(f)
        have = set(pf.columns)
        return pf.to_pandas(columns=[c for c in (*NEED, *extra) if c in have])
    except ImportError:
        return pd.read_parquet(f)


def sidecar(f):
    js = f[:-len(".parquet")] + ".json"
    return json.load(open(js)) if os.path.exists(js) else {}


def corner_tag(f):
    return (sidecar(f).get("corner") or {}).get("tag")


def treatment(f):
    """How this run treated positions its lagged score never saw."""
    if sidecar(f).get("interior_unseen_policy") == "floor_maxb":
        return "floor"
    return "bump" if os.path.basename(os.path.dirname(f)).startswith("job92") else "zero"


def per_head(d):
    """Per-head medians over rows where EVERY configured corner scored (step 4),
    so the oracle and practical columns describe the same rows."""
    d = d[d["quantized"].astype(bool)]
    d, _ = drop_partial_corners(d)
    full = d[d["gain_best_practical3"].notna()]
    return full.groupby(["layer", "head"]).median(numeric_only=True)


def band(s):
    s = s.dropna()
    return 100.0 * float((s >= BAND_MIN).mean()) if len(s) else np.nan


def verdict(x):
    return "GO" if x >= F_GO else ("STOP" if x < F_STOP else "NARROW")


def summarise(f):
    d = read(f)
    treat = treatment(f)
    pre = "" if treat == "floor" else "invalid_"
    g = per_head(d)
    lag = g.get("interior_lag_cost3_accum", pd.Series(dtype=float)).dropna()
    inb = g["gain_best_practical3"] >= BAND_MIN
    niah = d[d["family"] == "niah"]
    hit = niah.groupby("prompt")["needle_hit"].max() if "needle_hit" in niah else pd.Series(dtype=bool)
    rec = dict(
        treatment=treat, run=os.path.basename(os.path.dirname(f)),
        model=str(d["model"].iloc[0]), ctx=int(d["ctx"].iloc[0]), heads=len(g),
        retrieved=int(hit.sum()), prompts=int(len(hit)),
        band_or=band(g["gain_best3"]), band_pr=band(g["gain_best_practical3"]),
        dead2=100.0 * float(g["evict_beats_b2"].mean()), tau=float(g["tau"].median()),
        price=float((g["gain_e3_accum_frac"] / g["gain_e3_oracle_frac"]).median())
        if "gain_e3_accum_frac" in g else np.nan,
        evict_frac=float(g["evict_frac3"].median()) if "evict_frac3" in g else np.nan)
    rec.update({
        pre + "band_pp": band(g.get("gain_pp3_accum", pd.Series(dtype=float))),
        pre + "band_sym": band(g.get("gain_pp_sym3_accum", pd.Series(dtype=float))),
        pre + "lag_med": float(lag.median()) if len(lag) else np.nan,
        pre + "lag_p90": float(lag.quantile(.9)) if len(lag) else np.nan,
        pre + "lag_inband": float(lag[inb.reindex(lag.index)].median()) if len(lag) else np.nan,
        pre + "lag_outband": float(lag[~inb.reindex(lag.index)].median()) if len(lag) else np.nan,
        pre + "rho_lag_gain": (float(np.corrcoef(lag.rank(),
                                                 g.loc[lag.index, "gain_best_practical3"].rank())[0, 1])
                               if len(lag) > 8 else np.nan),
        pre + "router_over": 100.0 * float((inb & (g.get("gain_pp3_accum") < BAND_MIN)).mean())
        if "gain_pp3_accum" in g else np.nan,
        pre + "evict_frac_pp": float(g["evict_frac_pp3_accum"].median())
        if "evict_frac_pp3_accum" in g else np.nan})
    return g, rec


def load(patterns, want_tag="or-ac_f", need_col=None):
    """want_tag pins the corner (the fixed campaign is all `or-ac_f`); earlier
    campaigns used other corner sets -- job92* ran five -- so they are selected
    by the column the comparison needs instead."""
    runs, heads = [], {}
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        if want_tag is not None and corner_tag(f) != want_tag:
            continue
        if need_col is not None:
            try:
                import fastparquet
                if need_col not in fastparquet.ParquetFile(f).columns:
                    continue
            except ImportError:
                pass
        try:
            g, r = summarise(f)
        except Exception as e:                       # truncated copy, etc.
            print(f"skip {f}: {type(e).__name__}: {str(e)[:80]}")
            continue
        runs.append(r)
        heads[(r["model"], r["ctx"], r["treatment"], r["run"])] = g
    return pd.DataFrame(runs), heads


def lag_sweep(patterns):
    """Per k at steps 8..14, where every lag is ready -- the same rows for each k."""
    out = []
    ex = ([f"interior_lag_cost3_lag{k}" for k in LAGS]
          + [f"gain_e3_lag{k}_frac" for k in LAGS] + [f"gain_pp3_lag{k}" for k in LAGS])
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        if not str(corner_tag(f)).startswith("or-la-"):
            continue
        d = read(f, ex)
        d = d[d["quantized"].astype(bool) & (d["step"] >= 8)]
        g = d.groupby(["layer", "head"]).median(numeric_only=True)
        inb = g["gain_best3"] >= BAND_MIN
        for k in LAGS:
            if f"gain_e3_lag{k}_frac" not in g:
                continue
            L = g[f"interior_lag_cost3_lag{k}"]
            out.append(dict(
                model=str(d["model"].iloc[0]), ctx=int(d["ctx"].iloc[0]), k=k,
                treatment=treatment(f), heads=len(g),
                corner_over_oracle=float((g[f"gain_e3_lag{k}_frac"]
                                          / g["gain_e3_oracle_frac"]).median()),
                lag_med=float(L.median()), lag_inband=float(L[inb].median()),
                band_pp=band(g[f"gain_pp3_lag{k}"])))
    return pd.DataFrame(out)


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=[os.path.join(H0, "results", "job214217*", "*.parquet")],
                    help="the FIXED campaign (floor_maxb)")
    ap.add_argument("--old", nargs="+", default=[
        os.path.join(H0, "results", "job92*", "*.parquet"),
        os.path.join(H0, "results", "job21400*", "*.parquet"),
        os.path.join(H0, "results", "job21406*", "*.parquet")],
        help="earlier campaigns, for panel 4 and the reproducibility check")
    ap.add_argument("-o", "--out", default=HERE)
    a = ap.parse_args()

    new, new_heads = load(a.results)
    old, old_heads = load(a.old, want_tag=None, need_col="interior_lag_cost3_accum")
    if new.empty:
        raise SystemExit(f"no fixed R3 parquet matched {a.results}")
    if (new.treatment != "floor").any():
        raise SystemExit("--results matched runs without the fresh-token fix")
    runs = pd.concat([new, old], ignore_index=True)
    runs.to_csv(os.path.join(a.out, "R3-per-run.csv"), index=False)

    num = [c for c in new.columns if new[c].dtype.kind in "fi" and c != "ctx"]
    cells = (new.groupby(["model", "ctx"])[num].mean()
             .join(new.groupby(["model", "ctx"]).size().rename("runs")).reset_index())
    cells["order"] = cells["model"].map(ORDER.index)
    cells = cells.sort_values(["ctx", "order"]).drop(columns="order").reset_index(drop=True)
    for c, lab in (("band_or", "v_or"), ("band_pr", "v_pr"), ("band_pp", "v_pp")):
        cells[lab] = cells[c].map(verdict)
    cells.to_csv(os.path.join(a.out, "R3-cells.csv"), index=False)

    pd.set_option("display.width", 250)
    ff = lambda x: f"{x:.2f}"
    show = ["model", "ctx", "heads", "band_or", "band_pr", "band_pp", "band_sym",
            "v_or", "v_pr", "v_pp", "dead2", "tau", "price", "lag_med", "lag_inband",
            "lag_outband", "rho_lag_gain", "router_over"]
    print("=== R3, the FIXED campaign: one row per (model, ctx) ===")
    print(cells[show].to_string(index=False, float_format=ff))

    keep = cells["band_pr"] - cells["band_pp"]            # what demoting the interior costs
    gave = cells["band_pr"] - cells["band_or"]            # what demoting the corner gave
    print(f"\ncorner demotion (E2) adds {gave.min():.1f}-{gave.max():.1f} band points; "
          f"demoting the interior takes back {keep.min():.1f}-{keep.max():.1f}, i.e. "
          f"{100 * (keep / gave).median():.0f}% of it (median cell).")
    print(f"symmetric cell vs oracle cell: {(cells.band_pp - cells.band_or).min():+.1f} to "
          f"{(cells.band_pp - cells.band_or).max():+.1f} points")
    print(f"\nphase axis over {len(cells)} cells: Spearman(dead2, symmetric band) = "
          f"{spearman(cells.dead2, cells.band_pp):+.3f}   (E2 cell "
          f"{spearman(cells.dead2, cells.band_pr):+.3f}, oracle cell "
          f"{spearman(cells.dead2, cells.band_or):+.3f})")
    for ctx, c in cells.groupby("ctx"):
        if len(c) >= 3:
            print(f"   at ctx {ctx:>6}: {spearman(c.dead2, c.band_pp):+.2f} over {len(c)} models")

    # ---- the fix at scale: same cells, three treatments ------------------------
    piv = (runs.assign(cell=runs.model + " @" + (runs.ctx // 1024).astype(str) + "k")
           .pivot_table(index="cell", columns="treatment",
                        values=["lag_med", "invalid_lag_med", "lag_inband",
                                "invalid_lag_inband", "band_or", "band_pr"],
                        aggfunc="mean"))
    print("\n=== interior lag cost by treatment, shared cells "
          "(floor = fixed, bump = job92*, zero = job2140*/21406*) ===")
    cmp = pd.DataFrame({
        "floor": piv.get(("lag_med", "floor")), "bump": piv.get(("invalid_lag_med", "bump")),
        "zero": piv.get(("invalid_lag_med", "zero")),
        "floor_inband": piv.get(("lag_inband", "floor")),
        "bump_inband": piv.get(("invalid_lag_inband", "bump"))}).dropna(how="all")
    print(cmp.to_string(float_format=ff))
    sh = cmp.dropna(subset=["floor", "bump"])
    if len(sh):
        print(f"bump/floor on the {len(sh)} shared cells: "
              f"{(sh.bump / sh.floor).min():.2f}-{(sh.bump / sh.floor).max():.2f}x "
              f"(the controlled test predicted ~1.00)")
    rep = pd.DataFrame({"floor": piv.get(("band_or", "floor")),
                        "zero": piv.get(("band_or", "zero")),
                        "bump": piv.get(("band_or", "bump"))}).dropna(thresh=2)
    if len(rep):
        d = (rep.max(axis=1) - rep.min(axis=1)).max()
        print(f"oracle-cell band reproduces across campaigns to {d:.1f} points "
              f"(max over {len(rep)} shared cells)")

    sweep = lag_sweep(a.results)
    if not sweep.empty:
        sweep.to_csv(os.path.join(a.out, "R3-sweep.csv"), index=False)
        print("\n=== staleness sweep (steps 8-14, every lag ready) ===")
        print(sweep.to_string(index=False, float_format=ff))

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.6)); ax = ax.ravel()
    zones = lambda i: (ax[i].axhspan(F_GO, 100, color="#1f6b52", alpha=.06),
                       ax[i].axhspan(0, F_STOP, color="#c2334d", alpha=.06),
                       ax[i].axhline(F_GO, color="#1f6b52", ls="--", lw=.9),
                       ax[i].axhline(F_STOP, color="#c2334d", ls="--", lw=.9))
    ticks = sorted(cells.ctx.unique())

    # P1 -- the three cells
    for m in ORDER:
        c = cells[cells.model == m].sort_values("ctx")
        if c.empty:
            continue
        x = np.log2(c.ctx)
        ax[0].plot(x, c.band_pr, "--", color=COL[m], lw=1.1, alpha=.55)
        ax[0].plot(x, c.band_or, ":", color=COL[m], lw=1.1, alpha=.55)
        ax[0].plot(x, c.band_pp, "-o", color=COL[m], lw=2.0, ms=5, label=SHORT[m])
    zones(0)
    ax[0].set_xticks(np.log2(ticks)); ax[0].set_xticklabels([f"{t // 1024}k" for t in ticks])
    ax[0].set_xlabel("context length"); ax[0].set_ylabel("% heads in band")
    ax[0].set_ylim(0, 100); ax[0].legend(fontsize=7, ncol=2, loc="upper right")
    ax[0].set_title("SYMMETRIC cell (solid) keeps about half of what demoting\n"
                    "the corner gave (dashed = E2, dotted = both oracle)", fontsize=9.5)

    # P2 -- phase axis on the symmetric cell
    for r in cells.itertuples():
        ax[1].scatter(r.dead2, r.band_pp, s=40 + 25 * np.log2(r.ctx / 4096),
                      color=COL[r.model], edgecolor="white", lw=.7, zorder=3)
        ax[1].scatter(r.dead2, r.band_pr, s=26, facecolor="none",
                      edgecolor=COL[r.model], lw=.8, zorder=2)
    zones(1)
    ax[1].set_xlabel("dead 2-bit tier: heads with $\\tau^2c_2>c_0{=}1$  (%)")
    ax[1].set_ylabel("% heads in band, symmetric cell")
    ax[1].set_title(f"Phase axis, {len(cells)} cells (size = ctx)\n"
                    f"Spearman {spearman(cells.dead2, cells.band_pp):+.2f} symmetric, "
                    f"{spearman(cells.dead2, cells.band_pr):+.2f} E2 (hollow)", fontsize=9.5)

    # P3 -- where the lag cost lands
    for (m, ctx, tr, run), g in new_heads.items():
        if ctx != min(cells[cells.model == m].ctx) or "interior_lag_cost3_accum" not in g:
            continue
        j = g[["gain_best_practical3", "interior_lag_cost3_accum"]].dropna()
        ax[2].scatter(j.iloc[:, 0], j.iloc[:, 1], s=4, alpha=.3, linewidths=0,
                      color=COL.get(m, "#666"), label=f"{SHORT.get(m, m)} {ctx // 1024}k")
    ax[2].axvline(BAND_MIN, color="#e0a838", ls="--", lw=1.5)
    ax[2].axhline(1.0, color="#8a97a0", ls=":", lw=1.2)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("gain over the best practical corner (oracle interior)")
    ax[2].set_ylabel("interior lag cost  $err_{wf}^{lagged}/err_{wf}$")
    ax[2].legend(fontsize=6, loc="upper left", markerscale=3)
    ax[2].set_title("The lag cost lands on the heads that MATTER\n"
                    r"$\rho$(lag, gain) = "
                    f"{cells.rho_lag_gain.min():+.2f} to {cells.rho_lag_gain.max():+.2f}; "
                    f"out-of-band {cells.lag_outband.min():.2f}-{cells.lag_outband.max():.2f}x, "
                    f"in-band {cells.lag_inband.min():.2f}-{cells.lag_inband.max():.2f}x",
                    fontsize=9.5)

    # P4 -- the fix at scale
    data, labels, colors = [], [], []
    tcol = {"bump": "#8a97a0", "zero": "#c2334d", "floor": "#1f6b52"}
    # only cells where ALL THREE treatments exist, so each triplet is a
    # like-for-like comparison and the axis stays readable
    have = {t: {(m, c) for (m, c, tt, _) in list(new_heads) + list(old_heads) if tt == t}
            for t in ("bump", "zero", "floor")}
    shared = sorted(have["bump"] & have["zero"] & have["floor"],
                    key=lambda k: (ORDER.index(k[0]), k[1]))
    for m, c in shared:
        for tr in ("bump", "zero", "floor"):
            src = new_heads if tr == "floor" else old_heads
            gs = [g for (mm, cc, tt, _), g in src.items()
                  if (mm, cc, tt) == (m, c, tr) and "interior_lag_cost3_accum" in g]
            if not gs:
                continue
            data.append(pd.concat(gs)["interior_lag_cost3_accum"].dropna().clip(lower=.5).values)
            labels.append(f"{SHORT[m]} {c // 1024}k\n{tr}"); colors.append(tcol[tr])
    if data:
        bp = ax[3].boxplot(data, whis=(10, 90), showfliers=False, patch_artist=True, widths=.6)
        for p_, c_ in zip(bp["boxes"], colors):
            p_.set_facecolor(c_); p_.set_alpha(.65)
        ax[3].set_xticks(range(1, len(labels) + 1))
        ax[3].set_xticklabels(labels, fontsize=6.5, rotation=90)
        ax[3].set_yscale("log"); ax[3].axhline(1, color="k", lw=.6)
    ax[3].set_ylabel("interior lag cost (per head)")
    ax[3].set_title("The fix, at scale: the cells all three campaigns ran\n"
                    "unseen token bumped (job92*) / zeroed (job2140*) / floored (FIXED)",
                    fontsize=9.5)

    # P5 -- staleness sweep
    if not sweep.empty:
        for (m, ctx), s in sweep.groupby(["model", "ctx"]):
            lab = f"{SHORT.get(m, m)} {ctx // 1024}k"
            ax[4].plot(s.k, s.corner_over_oracle, "-o", color=COL.get(m, "#666"),
                       lw=1.7, alpha=1 if ctx == 8192 else .55, label=f"{lab}: corner")
            ax[4].plot(s.k, s.lag_inband, "--s", color=COL.get(m, "#666"), lw=1.2,
                       alpha=1 if ctx == 8192 else .55, label=f"{lab}: interior, in-band")
        ax[4].set_xscale("log", base=2); ax[4].set_xticks(LAGS); ax[4].set_xticklabels(LAGS)
        ax[4].axhline(1, color="#8a97a0", ls=":")
    ax[4].set_xlabel("lag k (decode steps since the score was taken)")
    ax[4].set_ylabel("error / oracle (median head)")
    ax[4].legend(fontsize=5.6, loc="upper left")
    ax[4].set_title("Staleness is real and rises with k\n"
                    "cost(k) is NOT flat: re-budgeting is load-bearing", fontsize=9.5)

    # P6 -- router miscalibration
    x = np.arange(len(cells))
    ax[5].bar(x, cells.router_over, .6, color=[COL[m] for m in cells.model],
              edgecolor="white", lw=.6)
    for xi, v in zip(x, cells.router_over):
        ax[5].text(xi, v + .3, f"{v:.0f}", ha="center", fontsize=6)
    ax[5].set_xticks(x)
    ax[5].set_xticklabels([f"{SHORT[m]} {c // 1024}k" for m, c in zip(cells.model, cells.ctx)],
                          fontsize=6.5, rotation=90)
    ax[5].set_ylabel("% of all heads")
    ax[5].set_title("Router miscalibration: heads an oracle-calibrated threshold\n"
                    "sends to the interior that do not pay for themselves", fontsize=9.5)

    for axi in ax:
        axi.grid(alpha=.22); axi.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(a.out, "R3-fig.png")
    fig.savefig(p, dpi=150)
    print(f"\nwrote {p}\nwrote {os.path.join(a.out, 'R3-per-run.csv')}\n"
          f"wrote {os.path.join(a.out, 'R3-cells.csv')}"
          + (f"\nwrote {os.path.join(a.out, 'R3-sweep.csv')}" if not sweep.empty else ""))


if __name__ == "__main__":
    main()
