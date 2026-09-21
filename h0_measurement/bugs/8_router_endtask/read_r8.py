#!/usr/bin/env python3
"""The R8 reader -- P0's budget pilot, and later every phase.

    .venv/bin/python h0_measurement/bugs/8_router_endtask/read_r8.py \
        "h0_measurement/results/r8job*/r8_*.parquet" [--csv out.csv] [--p2]

Prints, per (model, ctx, task): accuracy of every (arm, budget) with a 90%
bootstrap interval over prompts, then applies plan.md section 5's P0 rule:

  VALID     FP must be >= --fp-min on a task, or compression cannot be measured
            on it (the model cannot do it uncompressed).
  WHERE     the budget at which UNIFORM falls into [--lo, --hi]: the budget R8
            must run at, because above it every arm is ~100% and nothing
            separates, and below it every arm is ~0.
  TOO EASY  uniform above --hi even at the smallest budget: the tasks need more
            keys / values / hops before anything else is worth running.

The interval is over prompts, the unit of independent sampling here (R7 plan.md
B3). With 20 prompts per task a 90% interval is roughly +-15 points at 50%, so a
difference between two arms is reported as real only when the intervals clear.

--p2 adds plan.md section 7's decision table, from the P2 evaluation runs (and the
r8heads_*.parquet next to each r8_*.parquet):

  PAIRED    every arm minus uniform / evict / the best fixed policy, as a paired
            bootstrap over prompts (the arms share the prompt, its prefill and
            its haystack -- pairing removes the prompt's own difficulty).
  P-1       router_calib >= max(uniform, evict, interior) in every cell, within noise
  P-2       router_calib - best fixed, against the cell's symmetric band (GO->STOP)
  P-3       router_calib - interior in the STOP cells
  P-4       Spearman(per-cell output-error gain over uniform, per-cell accuracy
            gain over uniform) > 0.7. A CELL is (model, ctx, task, B); the
            accuracy is its RATE over prompts. Not per prompt: one prompt's
            pass/fail is a knife-edge that no error statistic separates (plan.md
            10.3). Reported for four head-error statistics -- mean (what the
            water-fill minimises), median, p99, and answer-weighted -- and for the
            router arms alone (the "routed gain" P-4 names) and for every arm.
  P-5       router_oracle - router_calib
"""
from __future__ import annotations
import argparse, glob, os, sys

import numpy as np
import pandas as pd


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(bs, .05)), float(np.quantile(bs, .95))


# symmetric band (%) and phase per P2 cell -- plan.md 2.5, R3/R6 reports
BAND = {("llama31-8b", 8192): (54.5, "GO"), ("llama31-8b", 32768): (34.0, "NARROW"),
        ("llama31-8b", 131072): (20.2, "NARROW"), ("qwen3-8b", 8192): (16.5, "NARROW/STOP"),
        ("qwen3-8b", 32768): (11.5, "STOP")}
FIXED = ("uniform", "evict", "interior")
ROUTERS = ("router_calib", "router_oracle")
ARM_ORDER = ("uniform", "evict", "evict_h2o", "interior", "interior_pool",
             "interior_cascade", "router_oracle", "router_calib")


def paired(a, b, n=2000, seed=0):
    """mean(a - b) over prompts present in both, with a 90% paired bootstrap."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if not len(j):
        return float("nan"), float("nan"), float("nan"), 0
    d = (j.a - j.b).to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.quantile(bs, .05)), float(np.quantile(bs, .95)), len(d)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan"), int(ok.sum())
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def spearman_ci(x, y, n=2000, seed=0):
    """bootstrap over cells -- with ~20-40 cells, a rho of 0.7 has a wide interval"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rs = [spearman(x[i], y[i])[0] for i in rng.integers(0, len(x), size=(n, len(x)))]
    rs = np.asarray([r for r in rs if np.isfinite(r)])
    return (float(np.quantile(rs, .05)), float(np.quantile(rs, .95))) if len(rs) else (float("nan"),) * 2


def reliability(dvecs, n_split=200, seed=0):
    """Split-half reliability of the per-cell accuracy gain, Spearman-Brown corrected.

    Each cell's accuracy gain is a mean of ~20 paired pass/fail differences, so it
    carries binomial noise of ~0.1-0.15 -- the same size as the gains themselves.
    That noise caps the Spearman ANY error statistic can reach: a perfect proxy
    scores about sqrt(reliability), not 1. P-4's 0.7 is read against this
    ceiling, and rho / sqrt(reliability) is reported as the disattenuated value.
    Prompts are split at random into halves; the two halves' per-cell gains are
    rank-correlated; repeated n_split times; median taken."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_split):
        a_, b_ = [], []
        for d in dvecs:
            if len(d) < 4:
                continue
            perm = rng.permutation(len(d))
            h = len(d) // 2
            a_.append(d[perm[:h]].mean()); b_.append(d[perm[h:2 * h]].mean())
        r, n = spearman(a_, b_)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return float("nan")
    r = float(np.median(rs))
    return 2 * r / (1 + r) if r > -1 else float("nan")


def head_stats(files):
    """per (model, ctx, prompt, task, arm, B): four aggregates of the per-head error"""
    out = []
    for f in files:
        hf = os.path.join(os.path.dirname(f), os.path.basename(f).replace("r8_", "r8heads_", 1))
        if not os.path.isfile(hf):
            continue
        model, ctx = pd.read_parquet(f, columns=["model", "ctx"]).iloc[0]
        h = pd.read_parquet(hf)
        h["w"] = h.ans_mass.where(np.isfinite(h.ans_mass), np.nan)
        h["we"] = h.err * h.w
        g = h.groupby(["prompt_idx", "task", "arm", "B"])
        st = pd.DataFrame({"mean": g.err.mean(), "median": g.err.median(),
                           "p99": g.err.quantile(.99),
                           "answer": g.we.sum(min_count=1) / g.w.sum(min_count=1)})
        out.append(st.reset_index().assign(model=model, ctx=ctx))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def p2_report(df, files, metric, fp_min, verbose=False):
    print(f"\n{'#' * 78}\nP2 -- plan.md section 7's decision table\n{'#' * 78}")
    hs = head_stats(files)
    if not len(hs):
        print("  (no r8heads_*.parquet next to the inputs: P-4 needs R8_HEAD_ERROR=1)")
    cells, p1_fail, dvec = [], [], []
    for (model, ctx), g in df.groupby(["model", "ctx"]):
        band, phase = BAND.get((model, int(ctx)), (float("nan"), "?"))
        print(f"\n{model} @ {int(ctx):,}   band {band}%  {phase}")
        if "frac_interior" in g:
            fi = g[g.arm.isin(ROUTERS)].groupby(["arm", "B"]).frac_interior.mean()
            if len(fi):
                print("  share of KV heads routed to the interior: "
                      + ", ".join(f"{a} B={B} {v:.0%}" for (a, B), v in fi.items()))
        for task, t in g.groupby("task"):
            m = metric if (metric == "score" or task in ("niah_single", "niah_multikey")) else "score"
            fpm = t[t.arm == "fp"][m].mean()
            if not fpm >= fp_min:
                print(f"  {task:16s} skipped: FP {fpm:.2f} < {fp_min}")
                continue
            for B in sorted(b for b in t.B.unique() if b > 0):
                tb = t[t.B == B]
                acc = {x: tb[tb.arm == x].set_index("prompt_idx")[m]
                       for x in ARM_ORDER if x in set(tb.arm)}
                if "uniform" not in acc:
                    continue
                means = {x: v.mean() for x, v in acc.items()}
                fixed = [x for x in FIXED if x in acc]
                best = max(fixed, key=lambda x: means[x])
                line = "  ".join(f"{x} {means[x]:.2f}" for x in acc)
                print(f"  {task:16s} B={B}  {line}")
                for x in acc:
                    if x == "uniform":
                        continue
                    d, lo, hi, n = paired(acc[x], acc["uniform"])
                    db, lob, hib, _ = paired(acc[x], acc[best]) if x != best else (0, 0, 0, 0)
                    mark = "  <-- beats uniform" if lo > 0 else ("  <-- LOSES to uniform" if hi < 0 else "")
                    if verbose or x in ("interior", "router_calib"):
                        print(f"      {x:17s} - uniform {d:+.2f} [{lo:+.2f},{hi:+.2f}]"
                          f"   - best fixed ({best}) {db:+.2f} [{lob:+.2f},{hib:+.2f}]{mark}")
                    row = dict(model=model, ctx=int(ctx), band=band, phase=phase, task=task,
                               B=B, arm=x, acc=means[x], acc_uniform=means["uniform"],
                               d_acc=d, d_lo=lo, d_hi=hi, n=n, best_fixed=best,
                               d_best=db, d_best_lo=lob, d_best_hi=hib,
                               d_interior=(paired(acc[x], acc["interior"])[0]
                                           if "interior" in acc and x != "interior" else np.nan))
                    if len(hs):
                        hc = hs[(hs.model == model) & (hs.ctx == ctx) & (hs.task == task) & (hs.B == B)]
                        for stat in ("mean", "median", "p99", "answer"):
                            ex = hc[hc.arm == x].set_index("prompt_idx")[stat]
                            eu = hc[hc.arm == "uniform"].set_index("prompt_idx")[stat]
                            j = pd.concat([eu.rename("u"), ex.rename("x")], axis=1, join="inner")
                            j = j[(j.u > 0) & (j.x > 0)]
                            # error gain over uniform: geometric mean over prompts of
                            # err_uniform / err_arm (log-symmetric; > 0 = the arm errs less)
                            row[f"gain_{stat}"] = (float(np.log(j.u / j.x).mean())
                                                   if len(j) else np.nan)
                    cells.append(row)
                    j = pd.concat([acc[x].rename("a"), acc["uniform"].rename("b")], axis=1,
                                  join="inner").dropna()
                    dvec.append((j.a - j.b).to_numpy(float))
                    if x == "router_calib" and hi < 0 and db < 0 and hib < 0:
                        p1_fail.append((model, int(ctx), task, B, best, db))
    if not cells:
        print("\n  no P2 cells (need uniform plus at least one other arm)")
        return pd.DataFrame()
    C = pd.DataFrame(cells)
    rc = C[C.arm == "router_calib"]

    print(f"\n{'-' * 78}")
    if len(rc):
        clear = rc[rc.d_best_hi < 0]
        print(f"P-1  router_calib >= best fixed policy within noise: "
              f"{len(rc) - len(clear)}/{len(rc)} cells"
              + ("" if not len(clear) else "   FAILS at " + "; ".join(
                  f"{r.model}@{r.ctx} {r.task} B={r.B} vs {r.best_fixed} {r.d_best:+.2f}"
                  for r in clear.itertuples())))
        by = rc.groupby(["model", "ctx", "band", "phase"]).d_best.mean().reset_index()
        rho, n = spearman(by.band, by.d_best)
        print("P-2  router_calib - best fixed, mean over tasks and budgets, by cell:")
        for r in by.sort_values("band", ascending=False).itertuples():
            print(f"       {r.model:11s} @{r.ctx:>7,}  band {r.band:5.1f} {r.phase:12s} {r.d_best:+.3f}")
        print(f"     Spearman(band, gain) = {rho:+.2f} over {n} cells "
              f"(P-2 predicts > 0: largest in GO)" if n >= 4 else
              f"     ({n} cells: too few for a rank correlation; read the column)")
        st = rc[rc.phase.str.contains("STOP")]
        if len(st) and st.d_interior.notna().any():
            print(f"P-3  router_calib - interior in STOP cells: mean {st.d_interior.mean():+.3f} over "
                  f"{len(st)} (cell, task, B)  (P-3 predicts > 0)")
        ro = C[C.arm == "router_oracle"].set_index(["model", "ctx", "task", "B"]).acc
        rj = rc.set_index(["model", "ctx", "task", "B"]).acc
        gap = (ro - rj).dropna()
        if len(gap):
            print(f"P-5  router_oracle - router_calib: mean {gap.mean():+.3f}, max {gap.max():+.3f} "
                  f"over {len(gap)} cells  (P-5 predicts <= a few points)")
    else:
        print("P-1..P-3, P-5: no router_calib rows (a calibration run, or --p2-pilot)")

    print("P-4  Spearman(per-cell error gain over uniform, per-cell accuracy gain over uniform)")
    gcols = [c for c in C.columns if c.startswith("gain_")]
    if not gcols:
        print("     no per-head errors -- rerun with R8_HEAD_ERROR=1")
    for label, mask in (("routed (router_calib, router_oracle)", C.arm.isin(ROUTERS).to_numpy()),
                        ("every arm vs uniform", np.ones(len(C), bool))):
        sub = C[mask]
        rel = reliability([d for d, k in zip(dvec, mask) if k])
        ceil = np.sqrt(rel) if rel > 0 else float("nan")
        print(f"     {label}: accuracy-gain split-half reliability {rel:.2f} -> "
              f"ceiling on any proxy's rho ~{ceil:.2f}")
        for gc in gcols:
            rho, n = spearman(sub[gc], sub.d_acc)
            lo, hi = spearman_ci(sub[gc], sub.d_acc)
            dis = rho / ceil if np.isfinite(ceil) and ceil > 0 else float("nan")
            verdict = ("" if not np.isfinite(rho) else
                       "  HOLDS" if lo > 0.7 else
                       "  HOLDS after disattenuation" if np.isfinite(dis) and dis > 0.7 and lo > 0 else
                       "  consistent with 0.7" if hi >= 0.7 > lo and rho > 0 else "  FAILS")
            print(f"       {gc[5:]:7s} rho {rho:+.2f} [{lo:+.2f},{hi:+.2f}]  disattenuated "
                  f"{dis:+.2f}  n={n} cells{verdict}")
    print("     (a cell = model x ctx x task x B, accuracy = its rate over prompts. "
          "\"HOLDS\" = the interval's lower end is above 0.7; \"after disattenuation\" = "
          "rho / ceiling > 0.7 with rho clearly > 0 -- report both, and the ceiling, "
          "never the disattenuated value alone)")
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--metric", default="score", choices=("score", "first_ok"),
                    help="score = RULER string match (primary); first_ok = the first "
                         "number is the right one (single/multikey only)")
    ap.add_argument("--fp-min", type=float, default=0.95)
    ap.add_argument("--lo", type=float, default=0.50)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--csv", default="")
    ap.add_argument("--p2", action="store_true",
                    help="plan.md section 7's decision table (P-1..P-5) from P2 runs")
    ap.add_argument("--p2-csv", default="", help="write the per-cell P2 table here")
    ap.add_argument("--p2-verbose", action="store_true",
                    help="paired lines for every arm (default: interior and router_calib)")
    a = ap.parse_args()

    files = sorted({f for p in a.inputs for f in glob.glob(p) if f.endswith(".parquet")})
    if not files:
        sys.exit("no R8 parquet matched")
    df = pd.concat([pd.read_parquet(f).assign(src=os.path.basename(os.path.dirname(f)))
                    for f in files], ignore_index=True)
    # P0 and P2-cal share prompts 0-9 on llama31-8b @32k: one row per (cell, prompt,
    # arm, B), the latest job winning, or every paired comparison double-counts
    key = ["model", "ctx", "task", "prompt_idx", "arm", "B"]
    dup = df.duplicated(key, keep="last")
    if dup.any():
        print(f"WARNING: {int(dup.sum()):,} duplicate (cell, prompt, arm, B) rows across jobs "
              f"{sorted(df[dup].src.unique())} -- keeping the latest job's. Pass one "
              f"phase's job directories to read it cleanly.")
        df = df.sort_values("src").drop_duplicates(key, keep="last").reset_index(drop=True)
    print(f"{len(files)} file(s), {len(df):,} rows, "
          f"{df.prompt_idx.nunique()} prompt(s), metric = {a.metric}")

    out = []
    for (model, ctx), g in df.groupby(["model", "ctx"]):
        print(f"\n{'=' * 78}\n{model} @ {int(ctx):,}")
        # bits audit -- every compressed arm must have spent ~B per context token
        au = g[g.arm != "fp"].groupby(["arm", "B"]).bits_per_token.agg(["min", "max"])
        off = au[(au["max"] > au.index.get_level_values("B") + 1e-6)]
        print("  bits audit: " + ("OK, every arm spent <= B bits per context token"
                                 if not len(off) else f"OVERSPENT {off.to_dict()}"))
        for task, t in g.groupby("task"):
            m = a.metric if (a.metric == "score" or task in ("niah_single", "niah_multikey")) else "score"
            fp = t[t.arm == "fp"][m]
            fpm, fplo, fphi = boot(fp)
            valid = fpm >= a.fp_min
            print(f"\n  {task:16s} n={t.prompt_idx.nunique()} prompts   FP {fpm:5.2f} "
                  f"[{fplo:.2f}, {fphi:.2f}]"
                  + ("" if valid else f"   <- INVALID: FP below {a.fp_min}; the model "
                                      f"cannot do this task uncompressed"))
            budgets = sorted(b for b in t.B.unique() if b > 0)
            arms = [x for x in ARM_ORDER if x in set(t.arm)]
            print(f"    {'B':>3s}" + "".join(f"{x:>22s}" for x in arms))
            uni = {}
            for B in budgets:
                cells = []
                for x in arms:
                    mm, lo, hi = boot(t[(t.arm == x) & (t.B == B)][m])
                    cells.append(f"{mm:6.2f} [{lo:.2f},{hi:.2f}]")
                    out.append(dict(model=model, ctx=ctx, task=task, arm=x, B=B,
                                    mean=mm, lo=lo, hi=hi, fp=fpm, metric=m))
                    if x == "uniform":
                        uni[B] = (mm, lo, hi)
                print(f"    {B:3d}" + "".join(f"{c:>22s}" for c in cells))
            if task == "niah_multikey":
                dr = t[t.arm != "fp"].groupby(["arm", "B"]).distractor.mean()
                worst = dr.idxmax() if len(dr) else None
                if worst is not None and dr.max() > 0:
                    print(f"    distractor named: up to {100 * dr.max():.0f}% "
                          f"({worst[0]} B={worst[1]}) -- the wrong needle survived")
            # the P0 rule
            if not valid:
                continue
            if uni:
                inside = [int(B) for B, (mm, _, _) in uni.items() if a.lo <= mm <= a.hi]
                bmin = min(uni)
                if inside:
                    print(f"    -> P0: uniform falls into [{a.lo:.2f}, {a.hi:.2f}] at "
                          f"B = {inside}. R8 runs HERE for {task}.")
                elif uni[bmin][0] > a.hi:
                    print(f"    -> P0: TOO EASY -- uniform still {uni[bmin][0]:.2f} at "
                          f"B = {bmin}. Add keys / values / hops before running P1.")
                else:
                    print(f"    -> P0: uniform jumps past the window (no budget in "
                          f"[{a.lo:.2f}, {a.hi:.2f}]); add an intermediate budget.")
            if "evict" in arms and uni:
                sep = [B for B in budgets
                       if not (np.isnan(uni[B][0]))
                       and ((boot(t[(t.arm == "evict") & (t.B == B)][m])[1] > uni[B][2])
                            or (boot(t[(t.arm == "evict") & (t.B == B)][m])[2] < uni[B][1]))]
                print(f"    evict vs uniform, intervals clear at B = {sep or 'none'}")

    if a.csv and out:
        pd.DataFrame(out).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    if a.p2:
        C = p2_report(df, files, a.metric, a.fp_min, a.p2_verbose)
        if a.p2_csv and len(C):
            C.to_csv(a.p2_csv, index=False)
            print(f"\nwrote {a.p2_csv}")


if __name__ == "__main__":
    main()
