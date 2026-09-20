#!/usr/bin/env python3
"""
R6 reader -- WHERE is the sharp boundary, and how well is it pinned?

    python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
        "h0_measurement/results/<JOB>/*.parquet" [...] \
        [--targets sym,e2,oracle] [--score accum] [--evictors oracle,accum] \
        [--boot 2000] [--csv out.csv]

report.py / make_fig_phase.py show that band fraction falls with the dead
2-bit tier fraction (rho = -0.93 to -0.98). Neither says WHERE it crosses the
verdict lines, nor how sure we are. This does, three ways, per target:

  FIT        a logistic curve  band = 100*sigmoid(a + b*dead2), one unit weight
             per (model, ctx) cell -- NOT per head: the 70B has 13x the heads of
             qwen15-moe and would otherwise be the whole fit. The crossing
             d*(F) = (logit F - a)/b for F = STOP (15%) and GO (35%).
  BRACKET    model-free. lo = highest dead-2 of any cell still >= F, hi = lowest
             dead-2 of any cell already < F. lo < hi: a GAP, the boundary is
             only bracketed and nothing inside it was measured. lo >= hi: an
             OVERLAP, cells of different models sit on both sides at the same
             dead-2, i.e. architecture scatter, not missing data.
  PER MODEL  each model's own curve, interpolated where it crosses F. C1 claims
             the LOCATION is universal; these should agree within the
             measurement interval if it is.

Two uncertainty intervals on d*, 90%:
  meas   layer-cluster bootstrap inside every cell (heads of one layer share a
         residual stream, so heads are not independent draws), refit each
         replicate. What a re-measurement of the same cells would give.
  arch   the same, plus resampling MODELS with replacement. 7 models is few,
         so this is coarse -- but it is the honest one: it asks whether the
         boundary would move with a different set of architectures, which is
         exactly the claim C1 makes. Neither includes seed variance (R7).

TARGETS. Which gain the band counts (all at B bits, BAND_MIN = 2x):
  sym     gain_pp<B>_<score>  interior AND corner on the lagged score -- the
          only comparison without information asymmetry (R3). THE HEADLINE.
  e2      gain_best_practical<B>  practical corner, oracle interior
  oracle  gain_best<B>        oracle corner, oracle interior
R3-report.md: e2 sits 6-27 pts above oracle on the same rows; where sym lands
between them is what R3's re-run decides, and it moves the boundary with it.

PROVENANCE. `sym` is read ONLY from parquets whose .json carries
"interior_unseen_policy": "floor_maxb" (R3-report.md section 2). job214* lack
it (the fresh token was evicted: interior lag cost 5-13x) and job92* lack it
(the old rank bump: plausible but unconfirmed). Their gain_pp* columns are
blanked on load; their oracle / e2 columns are valid and kept.
--allow-unfixed reads them anyway and stamps every sym line PROVISIONAL.

ROWS. Quantized rows where every configured corner scored (step 4 in the
default decode), for EVERY target -- the same rows R3-report.md section 1 uses,
so oracle and practical describe the same heads. dead-2 is read on those rows
too. Cells failing the input-validity gate (report.family_gate) are listed and
dropped from the fit. Tier-`debug` models (models.yaml) are dropped unless
--include-debug. Replicate runs of one (model, ctx, corner set) are averaged
and their spread printed.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))           # h0_measurement/
sys.path.insert(0, H0)
from report import (drop_partial_corners, family_gate, cfg_value,  # noqa: E402
                    BAND_MIN, F_STOP, F_GO)
from make_fig_phase import spearman, partial_spearman              # noqa: E402

LINES = {"STOP": 100 * F_STOP, "GO": 100 * F_GO}


def target_cols(score, B):
    return {"sym": f"gain_pp{B}_{score}", "e2": f"gain_best_practical{B}",
            "oracle": f"gain_best{B}"}


def fixed_interior(parquet):
    js = parquet[: -len(".parquet")] + ".json"
    try:
        with open(js) as fh:
            return json.load(fh).get("interior_unseen_policy") == "floor_maxb"
    except (OSError, ValueError):
        return False


def debug_tags():
    import yaml
    with open(os.path.join(H0, "models.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return {m["tag"] for m in cfg["models"] if m.get("tier") == "debug"}


# ------------------------------------------------------------------ loading ---
def load_cells(files, a):
    """One record per (file, model, ctx): point sums per LAYER, so the
    bootstrap can resample layers without touching the parquet again."""
    cols = target_cols(a.score, a.B)
    skip = set() if a.include_debug else debug_tags()
    out, notes = [], []
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception as e:                       # truncated / still writing
            notes.append(f"SKIP {f}: unreadable ({type(e).__name__})")
            continue
        run = os.path.basename(os.path.dirname(f))
        fixed = fixed_interior(f)
        gates = family_gate(d)
        if "quantized" in d:
            d = d[d["quantized"]]
        d, _ = drop_partial_corners(d)
        pp = [c for c in d.columns if c.startswith("gain_pp")]
        if pp and not fixed and not a.allow_unfixed:
            d = d.drop(columns=pp)
            notes.append(f"{run}/{os.path.basename(f)}: no floor_maxb marker -- "
                         f"symmetric columns ignored (oracle/e2 kept)")
        # complete-corner rows only, for every target (R3-report.md section 1)
        pr = cols["e2"]
        if pr in d and d[pr].notna().any():
            d = d[d[pr].notna()]
        for (m, c), g in d.groupby(["model", "ctx"]):
            if m in skip:
                continue
            ev = cfg_value(g, "evictors")
            if a.evictors != "any" and ev != a.evictors:
                notes.append(f"{run}: {m}@{int(c)} corner set '{ev}' != "
                             f"--evictors {a.evictors}; skipped")
                continue
            ph = g.groupby(["layer", "head"]).median(numeric_only=True).reset_index()
            if "evict_beats_b2" not in ph:
                continue
            lay = np.sort(ph["layer"].unique())
            li = np.searchsorted(lay, ph["layer"].values)

            def per_layer(v):
                return np.bincount(li, weights=v, minlength=len(lay))

            dv = ph["evict_beats_b2"].notna().values
            rec = dict(run=run, model=m, ctx=int(c), evictors=ev,
                       fixed=fixed, n_heads=len(ph),
                       n_all=per_layer(dv.astype(float)),
                       dead=per_layer(np.nan_to_num(ph["evict_beats_b2"].values)),
                       gate=gates.get((m, int(c)), {"passed": True, "reason": ""}),
                       t={})
            for t, col in cols.items():
                if col not in ph or not ph[col].notna().any():
                    continue
                gv = ph[col].values
                ok = np.isfinite(gv)
                gz = np.where(ok, gv, 1.0)
                rec["t"][t] = dict(
                    n=per_layer(ok.astype(float)),
                    inb=per_layer((ok & (gz >= BAND_MIN)).astype(float)),
                    lr=per_layer(np.where(ok, np.log(np.maximum(gz, 1.0)), 0.0)))
            out.append(rec)
    return out, notes


# ---------------------------------------------------------------- estimates ---
def point_and_boot(rec, t, W):
    """(dead2, band, routed) as point estimates and as bootstrap replicates.
    W: (nboot, n_layers) multinomial layer counts, or None."""
    s = rec["t"][t]
    def est(w):
        n_all = w @ rec["n_all"]; n = w @ s["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (100 * (w @ rec["dead"]) / n_all, 100 * (w @ s["inb"]) / n,
                    np.exp((w @ s["lr"]) / n))
    one = np.ones(len(rec["n_all"]))
    return est(one), (est(W) if W is not None else None)


def build_table(recs, t, nboot, rng):
    """Pool replicate runs of one (model, ctx, corner set). Returns the cell
    table and replicate arrays dead[b, cell], band[b, cell], routed[b, cell]."""
    groups = {}
    for r in recs:
        if t in r["t"]:
            groups.setdefault((r["model"], r["ctx"], r["evictors"]), []).append(r)
    rows, D, Bd, Rt = [], [], [], []
    for (m, c, ev), rs in sorted(groups.items()):
        pts, reps = [], []
        for r in rs:
            L = len(r["n_all"])
            W = rng.multinomial(L, np.full(L, 1.0 / L), size=nboot).astype(float)
            p, b = point_and_boot(r, t, W)
            pts.append(p); reps.append(b)
        p = np.mean(pts, axis=0)
        b = [np.mean([x[k] for x in reps], axis=0) for k in range(3)]
        bands = [x[1] for x in pts]
        gate_ok = all(r["gate"]["passed"] for r in rs)
        rows.append(dict(model=m, ctx=c, evictors=ev, runs=len(rs),
                         run_ids="+".join(r["run"] for r in rs),
                         dead2=p[0], dead2_lo=np.nanpercentile(b[0], 5),
                         dead2_hi=np.nanpercentile(b[0], 95),
                         band=p[1], band_lo=np.nanpercentile(b[1], 5),
                         band_hi=np.nanpercentile(b[1], 95),
                         band_rep_sd=(float(np.std(bands, ddof=1))
                                      if len(bands) > 1 else np.nan),
                         routed=p[2], gate=gate_ok,
                         gate_reason="" if gate_ok else "; ".join(
                             r["gate"].get("reason", "") for r in rs),
                         fixed=all(r["fixed"] for r in rs)))
        D.append(b[0]); Bd.append(b[1]); Rt.append(b[2])
    tab = pd.DataFrame(rows)
    return tab, np.array(D).T, np.array(Bd).T, np.array(Rt).T


# ---------------------------------------------------------------------- fit ---
def fit_logistic(x, p):
    """Unit-weight Bernoulli-deviance fit of p = sigmoid(a + b x). Returns (a, b)
    or None. Standardised x for conditioning; p may be exactly 0 or 1."""
    x = np.asarray(x, float); p = np.asarray(p, float)
    ok = np.isfinite(x) & np.isfinite(p)
    x, p = x[ok], np.clip(p[ok], 0, 1)
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return None
    xm, xs = x.mean(), x.std()
    X = np.column_stack([np.ones_like(x), (x - xm) / xs])
    beta = np.zeros(2)
    for _ in range(200):
        s = 1 / (1 + np.exp(-np.clip(X @ beta, -40, 40)))
        H = (X * (s * (1 - s))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (s - p))
        beta -= np.clip(step, -5, 5)
        if np.abs(step).max() < 1e-10:
            break
    return beta[0] - beta[1] * xm / xs, beta[1] / xs


def pava_decreasing(y):
    """Pool-adjacent-violators: the closest non-increasing sequence to y (equal
    weights), i.e. isotonic regression under the only shape C1 actually claims."""
    val = list(map(float, y)); cnt = [1] * len(y)
    i = 0
    while i < len(val) - 1:
        if val[i] < val[i + 1] - 1e-12:            # violation: pool the two blocks
            n = cnt[i] + cnt[i + 1]
            val[i] = (val[i] * cnt[i] + val[i + 1] * cnt[i + 1]) / n
            cnt[i] = n
            del val[i + 1], cnt[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.repeat(val, cnt)


def iso_crossing(x, y, F):
    """Where the MONOTONE fit of band-vs-dead2 crosses F. Free of any functional
    form, and by construction inside the bracket -- the logistic can land outside
    it when the curve is flat at one end, which is misspecification, not a
    measurement. NaN if the data never cross F."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    o = np.argsort(x); x, y = x[o], pava_decreasing(y[o])
    for i in range(len(x) - 1):
        if y[i] >= F > y[i + 1]:
            d = y[i] - y[i + 1]
            return float(x[i] + (x[i + 1] - x[i]) * ((y[i] - F) / d if d > 0 else .5))
    return np.nan


def crossing(ab, F):
    if ab is None or not np.isfinite(ab[1]) or ab[1] >= 0:
        return np.nan                       # no declining curve, no boundary
    f = F / 100
    return (np.log(f / (1 - f)) - ab[0]) / ab[1]


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return (np.nan, np.nan, 0.0)
    return (np.percentile(v, 5), np.percentile(v, 95), len(v))


def bracket_range(tab, F):
    """(lo, hi, is_gap). A GAP -- last cell above F below the first cell below F
    -- is the only case that constrains the crossing: no data live inside it, so
    a d* outside it contradicts the cells themselves. An OVERLAP means models
    disagree over a range, and a pooled crossing may legitimately sit outside."""
    above, below = tab[tab.band >= F], tab[tab.band < F]
    if not len(above) or not len(below):
        return (np.nan, np.nan, False)
    lo, hi = above.dead2.max(), below.dead2.min()
    return (lo, hi, True) if lo < hi else (hi, lo, False)


def bracket(tab, F):
    above, below = tab[tab.band >= F], tab[tab.band < F]
    if not len(above) or not len(below):
        side = "above" if len(above) else "below"
        ext = tab.dead2.max() if len(above) else tab.dead2.min()
        return f"every cell is {side} {F:.0f}% (dead-2 up to/from {ext:.1f}%) " \
               f"-- the line is not crossed in these data"
    lo, hi = above.dead2.max(), below.dead2.min()
    if lo < hi:
        return f"GAP      [{lo:.1f}, {hi:.1f}]  width {hi-lo:.1f} pts -- no cell " \
               f"inside; the boundary is only bracketed"
    inv = sum(int(x > y) for x in above.dead2 for y in below.dead2)
    return f"OVERLAP  [{hi:.1f}, {lo:.1f}]  width {lo-hi:.1f} pts -- {inv} " \
           f"inverted pair(s): different models on both sides at the same dead-2"


def per_model_crossings(tab, F):
    out = []
    for m, g in tab.groupby("model"):
        g = g.sort_values("dead2")
        x, y = g.dead2.values, g.band.values - F
        hits = [x[i] + (x[i+1] - x[i]) * y[i] / (y[i] - y[i+1])
                for i in range(len(g) - 1) if y[i] >= 0 > y[i+1]]
        if hits:
            s = ", ".join(f"{h:.1f}" for h in hits)
            out.append(f"    {m:22s} crosses at dead-2 {s}%"
                       + ("   (NON-MONOTONE: crosses more than once)"
                          if len(hits) > 1 else ""))
        elif (y >= 0).all():
            out.append(f"    {m:22s} above {F:.0f}% throughout -> boundary > "
                       f"{x.max():.1f}%  ({len(g)} cells)")
        elif (y < 0).all():
            out.append(f"    {m:22s} below {F:.0f}% throughout -> boundary < "
                       f"{x.min():.1f}%  ({len(g)} cells)")
        else:
            out.append(f"    {m:22s} rises back through {F:.0f}% "
                       f"(band increases with dead-2 here)")
    return out


def arch_boot(tab, D, Bd, nboot, rng, F, how):
    models = tab.model.values
    uniq = np.unique(models)
    idx = {m: np.where(models == m)[0] for m in uniq}
    out = np.full(nboot, np.nan)
    for b in range(nboot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        if len(set(pick)) < 2:
            continue
        cells = np.concatenate([idx[m] for m in pick])
        out[b] = how(D[b, cells], Bd[b, cells], F)
    return out


def est_iso(x, band, F):
    return iso_crossing(x, band, F)


def est_logit(x, band, F):
    return crossing(fit_logistic(x, band / 100), F)


# --------------------------------------------------------------------- main ---
def report_target(recs, t, a, rng):
    tab, D, Bd, Rt = build_table(recs, t, a.boot, rng)
    name = {"sym": f"SYMMETRIC CELL  (gain_pp{a.B}_{a.score}: interior and corner "
                   f"both on the lagged score)",
            "e2": f"E2 CELL  (gain_best_practical{a.B}: practical corner, "
                  f"oracle interior)",
            "oracle": f"ORACLE CELL  (gain_best{a.B}: oracle corner, oracle "
                      f"interior)"}[t]
    print("\n" + "=" * 96 + f"\n{name}\n" + "=" * 96)
    if tab.empty:
        print("  no cells carry this target"
              + ("  -- the symmetric cell needs runs with "
                 "\"interior_unseen_policy\": \"floor_maxb\" (the R3 re-run)"
                 if t == "sym" else ""))
        return None
    if t == "sym" and not tab.fixed.all():
        print("  PROVISIONAL -- --allow-unfixed: some cells predate the "
              "fresh-token floor (R3-report.md section 2)")
    bad = tab[~tab.gate]
    for _, r in bad.iterrows():
        print(f"  DROPPED (input-validity gate): {r.model}@{r.ctx}: {r.gate_reason}")
    keep = tab.gate.values
    tab, D, Bd, Rt = tab[keep].reset_index(drop=True), D[:, keep], Bd[:, keep], Rt[:, keep]
    # One (model, ctx) must contribute ONE cell. It can appear twice when runs
    # with different corner sets are pooled (--evictors any): the E2 campaign's
    # five-corner cells and R3's accum-only cells are the same head population
    # measured against different competitors, and counting both would weight
    # that cell twice in a fit whose unit is the cell.
    dup = tab.groupby(["model", "ctx"]).evictors.nunique()
    dup = dup[dup > 1]
    if len(dup):
        winner = tab.evictors.value_counts().index[0]
        print(f"  WARNING: {len(dup)} (model, ctx) appear under more than one "
              f"corner set; keeping '{winner}' and dropping the rest. Pass "
              f"--evictors to choose. Affected: "
              + ", ".join(f"{m}@{c}" for m, c in dup.index))
        k = (tab.evictors == winner).values
        tab, D, Bd, Rt = tab[k].reset_index(drop=True), D[:, k], Bd[:, k], Rt[:, k]

    print(f"  {'model':22s} {'ctx':>7s}  {'dead-2 %  [90%]':>20s}  "
          f"{'band %  [90%]':>20s}  routed  runs  rep-sd")
    for _, r in tab.sort_values("dead2").iterrows():
        print(f"  {r.model:22s} {r.ctx:>7,}  {r.dead2:6.1f} [{r.dead2_lo:5.1f},"
              f"{r.dead2_hi:5.1f}]  {r.band:6.1f} [{r.band_lo:5.1f},{r.band_hi:5.1f}]"
              f"  {r.routed:5.2f}x  {r.runs:>3d}  "
              + (f"{r.band_rep_sd:5.2f}" if np.isfinite(r.band_rep_sd) else "    -"))
    nm = tab.model.nunique()
    if len(tab) < 3:
        # A PILOT reads this branch: the per-cell numbers above are the point of
        # the run (does this cell land where the sheet predicted?), and a
        # correlation or a crossing over one or two cells would be noise.
        print(f"\n  {len(tab)} cell(s) -- too few to fit a boundary; the table "
              f"above is the result. Check each cell's dead-2 against the "
              f"prediction in report.md section 4.")
        return tab
    print(f"\n  {len(tab)} cells, {nm} models.  Spearman(dead-2, band) "
          f"{spearman(tab.dead2, tab.band):+.3f}"
          + (f"   model partialled out {partial_spearman(tab.dead2, tab.band, tab.model):+.3f}"
             if len(tab) > nm + 1 else "")
          + f"   Spearman(dead-2, routed) {spearman(tab.dead2, tab.routed):+.3f}")

    # routed gain at the crossing: log-linear in dead-2 (the robust statistic,
    # R2), so the boundary is also stated in a number that is not a count
    k, c0 = np.polyfit(tab.dead2, np.log(tab.routed), 1)
    x, y = tab.dead2.values, tab.band.values
    for line, F in LINES.items():
        print(f"\n  {line} line ({F:.0f}% of heads in band)")
        rows = []
        for nm, how in (("monotone", est_iso), ("logistic", est_logit)):
            d = how(x, y, F)
            meas = ci([how(D[b], Bd[b], F) for b in range(a.boot)])
            arch = ci(arch_boot(tab, D, Bd, a.boot, rng, F, how))
            rows.append((nm, d, meas, arch))
            if not np.isfinite(d):
                print(f"    {nm:9s} no crossing in these data")
                continue
            ext = "" if x.min() <= d <= x.max() else "  EXTRAPOLATED"
            print(f"    {nm:9s} d* = {d:5.1f}% dead-2{ext}   90% meas "
                  f"[{meas[0]:5.1f},{meas[1]:5.1f}] w{meas[1]-meas[0]:4.1f}"
                  f"   90% arch [{arch[0]:5.1f},{arch[1]:5.1f}] "
                  f"w{arch[1]-arch[0]:4.1f}   ({100*arch[2]/a.boot:.0f}% fit)")
        print(f"    routed gain at the monotone d*: "
              f"{np.exp(c0 + k * rows[0][1]):.2f}x" if np.isfinite(rows[0][1])
              else "    routed gain at d*: n/a")
        print(f"    bracket  {bracket(tab, F)}")
        # The bracket is data, the curves are models. A crossing outside the
        # bracket means the functional form is wrong, not that the boundary is
        # there -- the logistic does this when the band flattens at one end.
        b_lo, b_hi, is_gap = bracket_range(tab, F)
        for nm, d, _, _ in rows:
            if is_gap and np.isfinite(d) and not (b_lo - .05 <= d <= b_hi + .05):
                print(f"    WARNING: the {nm} d* ({d:.1f}%) falls OUTSIDE the "
                      f"bracket [{b_lo:.1f}, {b_hi:.1f}] -- that form is "
                      f"misspecified here; quote the monotone fit and the bracket")
        print("\n".join(per_model_crossings(tab, F)))
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--targets", default="sym,e2,oracle")
    ap.add_argument("--score", default="accum",
                    help="interior score of the symmetric cell (gain_pp<B>_<score>)")
    ap.add_argument("--evictors", default="oracle,accum",
                    help="keep only cells with exactly this corner set -- the "
                         "e2 band moves up to 6 pts with it (R3-report 3.4); "
                         "'any' to disable")
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unfixed", action="store_true",
                    help="read gain_pp* from runs without the floor_maxb marker "
                         "(job92*/job214*) -- PROVISIONAL")
    ap.add_argument("--include-debug", action="store_true")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    files = sorted({f for p in a.inputs for f in glob.glob(p)
                    if f.endswith(".parquet")
                    and not os.path.basename(f).startswith("validity_")})
    if not files:
        raise SystemExit("no measurement parquet matched")
    recs, notes = load_cells(files, a)
    print(f"{len(files)} parquet file(s), {len(recs)} (run, model, ctx) record(s)")
    for n in notes:
        print("  " + n)
    rng = np.random.default_rng(a.seed)
    out = []
    for t in [x.strip() for x in a.targets.split(",") if x.strip()]:
        tab = report_target(recs, t, a, rng)
        if tab is not None and len(tab):
            out.append(tab.assign(target=t))
    if a.csv and out:
        pd.concat(out, ignore_index=True).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
