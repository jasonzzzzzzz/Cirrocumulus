#!/usr/bin/env python3
"""S8 -- the reader for the two co-design columns (bugs/co-design/plan.md).

    .venv/bin/python h0_measurement/bugs/co-design/gqa_cascade.py \
        "h0_measurement/results/<WAVE4_JOBS>/*.parquet" \
        --csv h0_measurement/reports/codesign.csv

It answers two questions and nothing else:

  G  IS PER-HEAD ROUTING REALIZABLE?  Every gain this project reports is per
     QUERY head, but K is quantized once per KV head and a tiered cache stores
     one bit-width per (KV head, token). `grp_*_over_head` is what the shared
     allocation costs against the per-head one it replaces -- the R12 number.

  C  DOES A CURRENT-QUERY SCORE OFF THE BASE TIER PAY?  `closed` is the share of
     the lag penalty that the cascade removes, at each base width.

TWO THINGS THIS READER INSISTS ON, because both were got wrong while building it:

  * READ THE MARGINAL, NOT THE CONFLATED RATIO. `grp_pp_*cost` divides by
    `err_wf` and so carries the LAG cost as well as the grouping, and the lag
    tail is heavy -- at n_rep 2 the conflated ratio reads 1.165 median / 124x
    max while the grouping's own marginal reads 1.005 / 4.99x. The lag tail is
    R3's measurement, not R12's.

  * WEIGHT BY IN-BAND HEADS. An out-of-band head never uses the interior
    (R3-report 2.2: its lag cost is 1.00-1.04x), so its group cost is a number
    about a code path nobody runs. Every table below prints all-heads beside
    in-band and the in-band column is the one that decides.

The group band and the per-head band are NOT the same statistic: the group
corner is constrained too (plan.md B2), which costs the corner more than the
interior, so the group band comes out HIGHER. They are printed side by side and
never differenced.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from sievelib.alloc import BAND_MIN                                 # noqa: E402


def load_runs(files, a):
    """One record per parquet that actually carries the co-design columns."""
    recs, notes = [], []
    for f in sorted(files):
        js = f[:-8] + ".json"
        side = {}
        if os.path.isfile(js):
            try:
                side = json.load(open(js))
            except Exception:
                pass
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            notes.append(f"unreadable, skipped: {f} ({type(e).__name__})")
            continue
        if "quantized" in d:
            d = d[d.quantized]
        if not len(d):
            continue
        cols = set(d.columns)
        if not any(c.startswith(("grp_", "cs_b", "csv_b")) for c in cols):
            notes.append(f"no co-design columns (run it with SIEVE_GROUP_ALLOC / "
                         f"SIEVE_COARSE_BITS): {os.path.basename(f)}")
            continue
        if side and not side.get("group_alloc") and "coarse_bits" not in side:
            notes.append(f"columns present but the sidecar has no marker: {f}")
        # One decode step, so a head contributes once. Later steps of a dense
        # run are the same head under a longer history and would double-weight
        # it -- the guard boundary.py applies for the same reason.
        steps = sorted(d.step.unique()) if "step" in d else [None]
        step = a.step if a.step is not None else (steps[-1] if len(steps) else None)
        if step is not None and "step" in d:
            if step not in steps:
                notes.append(f"step {step} absent in {os.path.basename(f)} "
                             f"(has {steps}); using {steps[-1]}")
                step = steps[-1]
            d = d[d.step == step]
        if not len(d):
            continue
        # input-validity gate: a cell whose needle was not retrieved is not a
        # measurement of anything (R6 section 5, R4 section 5).
        gate = True
        if "family" in d and (d.family == "niah").any() and "needle_mass" in d:
            gate = bool(d.loc[d.family == "niah", "needle_mass"].notna().any())
        recs.append(dict(file=f, job=os.path.basename(os.path.dirname(f)),
                         model=str(d.model.iloc[0]), ctx=int(d.ctx.iloc[0]),
                         n_rep=int(d.n_rep.iloc[0]) if "n_rep" in d else None,
                         step=step, gate=gate, d=d,
                         n_prompts=side.get("n_prompts"),
                         coarse=side.get("coarse_bits") or [],
                         group=bool(side.get("group_alloc"))))
    return recs, notes


def _q(v, qs=(0.5, 0.9)):
    v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        return [float("nan")] * len(qs)
    return [float(v.quantile(q)) for q in qs]


def _band(v):
    v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
    return 100.0 * float((v >= BAND_MIN).mean()) if len(v) else float("nan")


def report_cell(key, recs, a):
    model, ctx = key
    d = pd.concat([r["d"] for r in recs], ignore_index=True)
    B = a.B
    n_rep = int(d.n_rep.iloc[0]) if "n_rep" in d and d.n_rep.notna().any() else None
    lag_col = f"interior_lag_cost{B}_{a.score}"
    gain_col = f"gain_pp{B}_{a.score}"
    inb = (d[gain_col] >= BAND_MIN) if gain_col in d else pd.Series(False, index=d.index)
    n_all, n_in = len(d), int(inb.sum())
    jobs = ",".join(sorted({r["job"] for r in recs}))
    bad = [r["job"] for r in recs if not r["gate"]]
    print(f"\n=== {model} @ {ctx:,}   n_rep={n_rep}   {n_all} head-rows, "
          f"{n_in} in band   [{jobs}]"
          + ("   INPUT-VALIDITY GATE FAILED: " + ",".join(bad) if bad else ""))
    if n_in < a.min_inband:
        print(f"    only {n_in} in-band heads (< {a.min_inband}): the in-band "
              f"column below is an anecdote, not a statistic")

    row = dict(model=model, ctx=ctx, n_rep=n_rep, jobs=jobs, n_heads=n_all,
               n_inband=n_in, gate_ok=not bad)

    # ---------------- G: what the shared allocation costs -------------------
    marg = sorted(c for c in d.columns if c.endswith(f"over_head{B}"))
    if marg or f"grp_or_cost{B}" in d:
        print(f"    {'G  the grouping OWN cost (vs the per-head alloc it replaces)':<52s}"
              f"{'all heads':>18s}{'in band':>18s}")
        if f"grp_or_cost{B}" in d:
            # `or` has no per-head twin to divide by -- under oracle information
            # the per-head allocation IS err_wf, so the cost ratio already is
            # the marginal.
            m, p = _q(d[f"grp_or_cost{B}"]); mi, pi = _q(d.loc[inb, f"grp_or_cost{B}"])
            print(f"      {'oracle info (pure constraint)':<50s}"
                  f"{m:8.3f} p90{p:7.3f}{mi:8.3f} p90{pi:7.3f}")
            row.update(grp_or_med=m, grp_or_p90=p, grp_or_med_inband=mi)
        for c in marg:
            nm = c[len("grp_"):-len(f"_over_head{B}")]
            m, p = _q(d[c]); mi, pi = _q(d.loc[inb, c])
            print(f"      {nm:<50s}{m:8.3f} p90{p:7.3f}{mi:8.3f} p90{pi:7.3f}")
            row[f"grp_{nm}_marg_med"] = m
            row[f"grp_{nm}_marg_med_inband"] = mi
        # the conflated ratio, printed once so nobody re-derives it by hand
        cf = f"grp_pp_{a.score}_cost{B}"
        if cf in d and lag_col in d:
            print(f"      {'(conflated grp/err_wf -- carries the LAG tail, do not quote)':<50s}"
                  f"{_q(d[cf])[0]:8.3f}{'':10s}{_q(d.loc[inb, cf])[0]:8.3f}")

    # ---------------- the bands, on a MATCHED corner set --------------------
    # The group corner is ranked by the group-summed INTERIOR score, i.e. by
    # `accum` alone. In the five-corner runs the per-head competitor
    # (`err_practical<B>`) is a min over five, so comparing the two bands as
    # shipped hands the group side an easier corner and flatters it -- the same
    # class of error as plan.md B2, pointing the other way. The accum-only
    # per-head corner is reconstructible exactly from `err_e<B>_<score>_frac`
    # (R7 plan.md 2, verified max|diff| = 0.0), so the matched band is free.
    # ...and on a matched RANKING RULE. The group corner ranks by the group-summed
    # lagged SENSITIVITY w2p (unseen floored), so its per-head twin is the
    # `<score>_w2p` corner, not the raw-attention `<score>` corner. Found on wave
    # 4's n_rep = 1 cell: against the raw corner the two bands read 37.6 vs 37.1
    # where they must be identical; against the w2p corner they are 37.1 = 37.1
    # and the corner errors agree to 0.0 (plan.md 7.0d).
    gg = f"gain_grp_pp_{a.score}_{B}"
    e_un, e_acc = f"err_uniform{B}", f"err_e{B}_{a.score}_w2p_frac"
    ph_acc = f"err_wf_pp{B}_{a.score}"
    if gg in d and {e_un, e_acc, ph_acc} <= set(d.columns):
        ref = np.minimum(d[e_un], d[e_acc]) / d[ph_acc].clip(lower=1e-12)
        bh, bg = _band(ref), _band(d[gg])
        n_corn = int(d.n_practical.iloc[0]) if "n_practical" in d else 1
        print(f"    {'band %, BOTH vs the accum w2p-ranked corner (per-head | group)':<52s}"
              f"{bh:8.1f}{bg:10.1f}")
        row.update(band_perhead_matched=bh, band_group=bg)
        if gain_col in d and n_corn > 1:
            print(f"      (as-shipped gain_pp{B}_{a.score} is a min over {n_corn} corners: "
                  f"{_band(d[gain_col]):.1f}% -- not comparable to the group band)")
    elif gain_col in d and gg in d:
        bh, bg = _band(d[gain_col]), _band(d[gg])
        print(f"    {'band % (per-head) vs (group) -- corner sets may differ':<52s}"
              f"{bh:8.1f}{bg:10.1f}")
        row.update(band_perhead=bh, band_group=bg)

    # ---------------- C: how much of the lag penalty the cascade closes -----
    if lag_col in d:
        widths = sorted({int(m.group(1)) for c in d.columns
                         for m in [re.match(rf"cs_b(\d+)_cost{B}$", c)] if m})
        if widths:
            lag = d[lag_col]
            head = (lag - 1.0).clip(lower=1e-9)
            print(f"    {'C  share of the lag penalty closed  (1.0 = as good as an exact score)':<52s}"
                  f"{'all heads':>18s}{'in band':>18s}")
            for bc in widths:
                for kind, col in (("bound  cs", f"cs_b{bc}_cost{B}"),
                                  ("deployable csv", f"csv_b{bc}_{a.score}_cost{B}")):
                    if col not in d:
                        continue
                    cl = ((lag - d[col]) / head).clip(-2, 2)
                    m, _ = _q(cl); mi, _ = _q(cl[inb])
                    cm, _ = _q(d[col]); cmi, _ = _q(d.loc[inb, col])
                    flag = "  <- worse than the lagged score" if mi < 0 else ""
                    print(f"      bc={bc}  {kind:<42s}{m:+8.2f} ({cm:.3f}x)"
                          f"{mi:+8.2f} ({cmi:.3f}x){flag}")
                    tag = f"{'cs' if kind.startswith('bound') else 'csv'}_b{bc}"
                    row[f"closed_{tag}"] = m
                    row[f"closed_{tag}_inband"] = mi
                    row[f"cost_{tag}_inband"] = cmi
            print(f"      (per-head lagged interior, the baseline: "
                  f"{_q(lag)[0]:.3f}x all / {_q(lag[inb])[0]:.3f}x in band)")
            row["lag_med"] = _q(lag)[0]
            row["lag_med_inband"] = _q(lag[inb])[0]

    # ---------------- the two together --------------------------------------
    best = [(row.get(f"closed_csv_b{bc}_inband", float('nan')), bc)
            for bc in sorted({int(m.group(1)) for c in d.columns
                              for m in [re.match(rf"csv_b(\d+)_", c)] if m})]
    best = [(v, bc) for v, bc in best if np.isfinite(v)]
    if best:
        v, bc = max(best)
        gc = f"gain_grp_csv_b{bc}_{a.score}_{B}"
        if gc in d and gg in d:
            # BOTH of these use the GROUP corner, so the difference is what the
            # cascade adds to the storable design. Comparing either against the
            # per-head band would be comparing different competitors -- the
            # mistake this file's header exists to prevent.
            print(f"    the storable design, group corner throughout: band "
                  f"{_band(d[gg]):.1f}% lagged -> {_band(d[gc]):.1f}% with the "
                  f"cascade at bc={bc}")
            if "band_perhead_matched" in row:
                print(f"      (against {row['band_perhead_matched']:.1f}% for the per-head "
                      f"allocation on the SAME corner and ranking rule)")
            row["band_group_cascade"] = _band(d[gc])
            row["best_bc"] = bc
    return row


def verdicts(rows, a):
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 78 + "\nVERDICTS (plan.md section 8)")
    col = f"grp_pp_{a.score}_marg_med_inband"
    if col in df and df[col].notna().any():
        v = df[col].dropna()
        worst = float(v.max())
        print(f"\nG  grouping's own marginal cost, in-band, worst cell: {worst:.3f}x")
        if worst <= 1.15:
            print("   -> REALIZABLE. C4's per-head router stands; report the group")
            print("      band beside the per-head band, naming both.")
        elif worst <= 2.0:
            print("   -> REALIZABLE AT A DISCOUNT. Every headline gain shrinks by")
            print("      this factor; make the GROUP band the paper's y-axis and")
            print("      re-fit the phase law on it (dead-2 is unchanged).")
        else:
            print("   -> PER-HEAD ROUTING IS PARTLY FICTIONAL for GQA models (R12).")
            print("      The router must decide per KV head from group-summed w2,")
            print("      and R8 must run the group allocation or it measures")
            print("      something a cache cannot store.")
        if "n_rep" in df and df.n_rep.notna().sum() > 1:
            g = df.dropna(subset=["n_rep", col]).groupby("n_rep")[col].median()
            if len(g) > 1:
                print("   dose-response in n_rep (the mechanism check): "
                      + "  ".join(f"{int(k)}:{v:.3f}x" for k, v in g.items()))
                print("   " + ("rising with n_rep -> head heterogeneity inside a "
                               "group is the mechanism" if g.is_monotonic_increasing
                               else "NOT monotone in n_rep -> the cost is not "
                                    "simply group size; look at the cells"))
    ccols = sorted(c for c in df.columns if c.startswith("closed_csv_b")
                   and c.endswith("_inband"))
    if ccols:
        print("\nC  share of the lag penalty closed by the deployable cascade, in band:")
        for c in ccols:
            bc = c[len("closed_csv_b"):-len("_inband")]
            v = df[c].dropna()
            if not len(v):
                continue
            print(f"   bc={bc}: median over cells {v.median():+.2f}"
                  f"   (range {v.min():+.2f} .. {v.max():+.2f})")
        best = max(((df[c].median(), c) for c in ccols
                    if df[c].notna().any()), default=(float("nan"), ""))
        if best[1]:
            m, c = best
            bc = c[len("closed_csv_b"):-len("_inband")]
            if m >= 0.5:
                print(f"   -> BUILD IT at bc={bc} ({m:+.2f}). Then price the base-tier")
                print("      read against re-budgeting more often (R3 section 2.3).")
            elif m >= 0.2:
                print(f"   -> HYBRID OR NOT AT ALL (best {m:+.2f} at bc={bc}): blend the")
                print("      lagged and coarse scores, or spend the base tier only on")
                print("      in-band heads.")
            else:
                print(f"   -> DROP IT (best {m:+.2f}). The lag-only design with")
                print("      re-budgeting every few steps stands; the gap to the")
                print("      oracle interior is not reachable from the base tier.")
        neg = [c for c in ccols if df[c].notna().any() and df[c].median() < 0]
        if neg:
            print("   NOTE: " + ", ".join(c[len('closed_'):-len('_inband')] for c in neg)
                  + " is NEGATIVE -- at that width the coarse score is a WORSE"
                    " allocator than last step's attention. That is a result about"
                    " the base tier's width (R10), not a bug.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--score", default="accum",
                    help="the lagged interior score the columns are keyed on")
    ap.add_argument("--step", type=int, default=None,
                    help="decode step to read (default: the last measured one, "
                         "so one row per head)")
    ap.add_argument("--min-inband", type=int, default=20,
                    help="below this the in-band column is called an anecdote")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    files = sorted({f for p in a.inputs for f in glob.glob(p)
                    if f.endswith(".parquet")
                    and not os.path.basename(f).startswith("validity_")})
    if not files:
        raise SystemExit("no measurement parquet matched")
    recs, notes = load_runs(files, a)
    print(f"{len(files)} parquet file(s), {len(recs)} with co-design columns")
    for n in notes:
        print("  " + n)
    if not recs:
        raise SystemExit(
            "nothing to read. These columns exist only on runs submitted with "
            "SIEVE_GROUP_ALLOC=1 / SIEVE_COARSE_BITS=... -- see "
            "h0_measurement/bugs/co-design/script.sh wave 4.")
    cells = {}
    for r in recs:
        cells.setdefault((r["model"], r["ctx"]), []).append(r)
    rows = [report_cell(k, cells[k], a) for k in sorted(cells)]
    rows = [r for r in rows if r]
    verdicts(rows, a)
    if a.csv and rows:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
