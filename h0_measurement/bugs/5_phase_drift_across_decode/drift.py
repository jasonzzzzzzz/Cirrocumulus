#!/usr/bin/env python3
"""
R5 reader -- does a head's phase drift across a long decode?

    python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
        "h0_measurement/results/<JOB>/*.parquet" [...] [--calib 1] [--csv out.csv]

report.py cannot answer this: per_head() takes a median over EVERY row of a
(model, ctx, layer, head), steps included, so a drift is averaged away before it
is ever looked at. This keeps `step` as an axis.

One block per (model, ctx, decode_temp). Per measured step:

  L, dlog2L     cache length, and how far the generation has grown it
  tau, ladder   median over heads; dtau is against the calibration step
  n95, dead1/2  median support; fraction of heads whose 1-/2-bit tier is dead
  band%         heads with gain_best_practical<B> >= 2 (report.py's BAND_MIN)
  flip_gen      heads whose ROUTE (interior iff gain >= 2) differs from the
                calibration step IN THE SAME GENERATION (per prompt, pooled)
  flip_rtr      the same, on per-head medians over prompts -- what an offline
                router calibrated once would see
  rho           Spearman over heads of gain(t) vs gain(calib), per-head medians

flip_rtr / rho / regret90 / band% / gain use only prompts that are present at
EVERY step after the post-EOS filter. Otherwise a prompt that ends early drops
out of the per-head median mid-table and reads as a phase change.
  regret90      p90 over heads of err(route fixed at calib) / err(route re-read
                at t by the same 2x rule). Depends only on gain(t) and the two
                routes: interior costs max(1, 1/g) against the better option,
                baseline max(1, g); 1.0 wherever the route did not flip.
  lag           median interior_lag_cost<B>_<score> on in-band heads (R3's toll)
  eos%          rows past the first EOS (excluded unless --keep-post-eos)
  d4            median distinct-4-gram fraction of the text so far (loop check)

The floor for flip_gen/flip_rtr is the SAME statistic between the calibration
step and the next measured step: per-head values carry up to 80% GPU
nondeterminism (R3-report.md finding 5), so a flip rate means nothing until it
exceeds that floor. Drift = growth above the floor with t.

L-growth test (the "double duty" of R5): where one (model, temp) was run at
several ctx, the across-ctx slope dtau/dlog2(L) at the calibration step predicts
how far tau should move when a generation grows the cache by G tokens. That
prediction is printed beside the measured dtau at the last step.

Bridge (dense run carrying both accum and last_step): per step, how the
last_step corner the sparse runs are forced onto compares with the campaign's
accum corner -- median gain ratio and band% under each.
"""
from __future__ import annotations
import argparse, glob, math, re, sys
import numpy as np
import pandas as pd

BAND_MIN = 2.0      # keep identical to report.py / alloc.py


def load(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        sys.exit(f"no parquet matched {patterns}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c, v in (("decode_temp", 0.0), ("past_eos", False), ("decode_ban_eos", False),
                 ("gen_distinct4", np.nan), ("schedule", "dense")):
        if c not in df:
            df[c] = v
    print(f"{len(files)} parquet, {len(df):,} rows")
    return df


def spearman(x, y):
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return float("nan")
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])


def regret(g, interior):
    """err(chosen route) / err(better route at this step), g = err_base/err_wf."""
    return np.where(interior, np.maximum(1.0, 1.0 / g), np.maximum(1.0, g))


def drift_regret(g, calib_interior):
    """Cost of keeping the calibration-time route instead of re-reading it now."""
    return regret(g, calib_interior) / regret(g, g >= BAND_MIN)


def flip(a, b):
    m = a.notna() & b.notna()
    if not bool(m.any()):
        return float("nan")
    return float(((a[m] >= BAND_MIN) != (b[m] >= BAND_MIN)).mean())


def pct(x):
    return "   -" if x is None or not math.isfinite(x) else f"{100 * x:4.0f}"


def num(x, f="{:6.2f}"):
    return "     -" if x is None or not math.isfinite(x) else f.format(x)


def block(g, B, calib, lagcols, eos_by_step):
    gcol = f"gain_best_practical{B}"
    keys = ["layer", "head"]
    steps = sorted(g.step.unique())
    if calib not in steps:
        later = [s for s in steps if s >= calib]
        if not later:
            print(f"  calibration step {calib} not measured; steps {steps}")
            return None
        calib = later[0]
    have_g = gcol in g and g[gcol].notna().any()
    if not have_g:
        print(f"  no {gcol} in these rows -- routing columns skipped")
    nxt = [s for s in steps if s > calib]
    per_prompt = g.set_index(["prompt", "family"] + keys)
    # balanced panel for the router-level columns -- see the docstring
    n_steps = g.groupby("prompt").step.nunique()
    full = n_steps.index[n_steps == len(steps)]
    if len(full) == 0:
        print("  WARNING: no prompt survives every step; router columns use "
              "whatever prompts each step has")
        full = n_steps.index
    elif len(full) < len(n_steps):
        print(f"  router columns on {len(full)}/{len(n_steps)} prompts (the rest "
              f"end before the last step)")
    per_head = (g[g.prompt.isin(full)]
                .groupby(["step"] + keys).median(numeric_only=True))

    c_pp = per_prompt[per_prompt.step == calib]
    c_ph = per_head.xs(calib, level="step")

    def flips(s):
        t_pp = per_prompt[per_prompt.step == s]
        t_ph = per_head.xs(s, level="step")
        if not have_g:
            return (float("nan"),) * 4
        a, b = c_pp[gcol].align(t_pp[gcol], join="inner")
        fg = flip(a, b)
        a2, b2 = c_ph[gcol].align(t_ph[gcol], join="inner")
        fr = flip(a2, b2)
        rho = spearman(a2, b2)
        m = a2.notna() & b2.notna()
        rg = (float(np.quantile(drift_regret(b2[m].to_numpy(),
                                             a2[m].to_numpy() >= BAND_MIN), .9))
              if int(m.sum()) else float("nan"))
        return fg, fr, rho, rg

    if nxt and have_g:
        fg0, fr0, _, _ = flips(nxt[0])
        print(f"  calibration step {calib}; noise floor (step {calib} vs {nxt[0]}): "
              f"flip_gen {pct(fg0).strip()}%  flip_rtr {pct(fr0).strip()}%")
    print(f"  {'step':>5} {'L':>7} {'dlog2L':>6} {'tau':>6} {'dtau':>6} "
          f"{'ladder':>6} {'n95':>6} {'dead1':>5} {'dead2':>5} {'band%':>5} "
          f"{'gain':>6} {'flipG':>5} {'flipR':>5} {'rho':>5} {'rgr90':>6} "
          f"{'lag':>6} {'eos%':>4} {'d4':>5}")
    L0 = float(g.loc[g.step == calib, "L"].median())
    tau0 = float(c_ph["tau"].median())
    out = []
    for s in steps:
        r = g[g.step == s]
        ph = per_head.xs(s, level="step")
        L = float(r["L"].median())
        band = (float((ph[gcol].dropna() >= BAND_MIN).mean())
                if have_g and ph[gcol].notna().any() else float("nan"))
        med_g = float(ph[gcol].median()) if have_g else float("nan")
        fg, fr, rho, rg = flips(s) if s != calib else (0.0, 0.0, 1.0, 1.0)
        lag = float("nan")
        for lc in lagcols:
            if lc in ph and have_g:
                inb = ph[gcol] >= BAND_MIN
                if bool(inb.any()):
                    lag = float(ph.loc[inb, lc].median())
                break
        dead = {b: (float(ph[f"evict_beats_b{b}"].mean())
                    if f"evict_beats_b{b}" in ph else float("nan")) for b in (1, 2)}
        row = dict(step=int(s), L=L, dlog2L=math.log2(L / L0),
                   tau=float(ph["tau"].median()),
                   dtau=float(ph["tau"].median()) - tau0,
                   ladder=float(ph["ladder_bits"].median()) if "ladder_bits" in ph
                   else float("nan"),
                   n95=float(ph["n95"].median()) if "n95" in ph else float("nan"),
                   dead1=dead[1], dead2=dead[2], band=band, gain=med_g,
                   flip_gen=fg, flip_rtr=fr, rho=rho, regret90=rg, lag=lag,
                   eos=float(eos_by_step.get(s, float("nan"))), d4=float(r["gen_distinct4"].median()))
        out.append(row)
        print(f"  {s:>5} {L:>7,.0f} {row['dlog2L']:>6.3f} {row['tau']:>6.3f} "
              f"{row['dtau']:>+6.3f} {num(row['ladder'])} {num(row['n95'], '{:6.0f}')} "
              f"{pct(dead[1]):>5} {pct(dead[2]):>5} {pct(band):>5} {num(med_g)} "
              f"{pct(fg):>5} {pct(fr):>5} {num(rho, '{:5.2f}')} {num(rg)} "
              f"{num(lag)} {pct(row['eos'])} {num(row['d4'], '{:5.2f}')}")
    return pd.DataFrame(out)


def bridge(g, B):
    """accum vs last_step corners at the same steps, same rows."""
    ca, cl = f"gain_e{B}_accum_frac", f"gain_e{B}_last_step_frac"
    cu = f"gain_u{B}"
    if not all(c in g for c in (ca, cl, cu)):
        return
    print(f"  BRIDGE -- last_step (what a sparse run can field) vs accum (the "
          f"campaign's corner), per-head medians:")
    print(f"  {'step':>5} {'ratio':>6} {'band_acc':>8} {'band_last':>9}")
    ph = g.groupby(["step", "layer", "head"])[[ca, cl, cu]].median()
    for s, x in ph.groupby(level="step"):
        x = x.dropna()
        if not len(x):
            continue
        ba = np.minimum(x[cu], x[ca]); bl = np.minimum(x[cu], x[cl])
        print(f"  {s:>5} {float((x[cl] / x[ca]).median()):>6.3f} "
              f"{100 * float((ba >= BAND_MIN).mean()):>8.1f} "
              f"{100 * float((bl >= BAND_MIN).mean()):>9.1f}")
    print("  ratio > 1: last_step is the WEAKER corner, so a sparse run's band% "
          "reads high by the band gap shown.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--B", type=int, default=3, help="budget, bits/token")
    ap.add_argument("--calib", type=int, default=1,
                    help="calibration step. Default 1: last_step has no history "
                         "at step 0, so no practical gain exists there.")
    ap.add_argument("--family", default="cont")
    ap.add_argument("--keep-post-eos", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()

    df = load(a.parquet)
    df = df[df.family == a.family] if a.family != "all" else df
    if "quantized" in df:
        df = df[df.quantized.astype(bool)]
    df = df.assign(_eos=df.past_eos.astype(bool))
    lagcols = sorted(c for c in df.columns
                     if re.match(rf"^interior_lag_cost{a.B}_", c))
    tables = []
    for (mdl, ctx, T), g in df.groupby(["model", "ctx", "decode_temp"]):
        ev = ",".join(sorted(g.evictors.astype(str).unique())) if "evictors" in g else "?"
        n_eos = int(g.groupby("prompt")._eos.any().sum())
        print(f"\n=== {mdl}  ctx {ctx:,}  T={T:g}  [{ev}]  "
              f"{g.prompt.nunique()} prompts, {n_eos} reached EOS"
              f"{' (EOS banned)' if bool(g.decode_ban_eos.any()) else ''} "
              f"{'(post-EOS rows kept)' if a.keep_post_eos else '(post-EOS rows dropped)'}")
        gg = g if a.keep_post_eos else g[~g._eos]
        t = block(gg, a.B, a.calib, lagcols, g.groupby("step")._eos.mean())
        bridge(gg, a.B)
        if t is not None:
            tables.append(t.assign(model=mdl, ctx=ctx, decode_temp=T))

    if not tables:
        return
    res = pd.concat(tables, ignore_index=True)

    # ---- L-growth test: does tau move within a generation as the across-ctx
    # slope says it should? Needs >= 2 ctx for one (model, temp).
    print("\nL-GROWTH TEST -- across-ctx slope at the calibration step vs the "
          "within-generation move at the last step")
    any_ = False
    for (mdl, T), r in res.groupby(["model", "decode_temp"]):
        c0 = r[r.dlog2L.abs() < 1e-9].drop_duplicates("ctx")
        if c0.ctx.nunique() < 2:
            continue
        slope = float(np.polyfit(np.log2(c0.L), c0.tau, 1)[0])
        for ctx, rr in r.groupby("ctx"):
            last = rr.loc[rr.step.idxmax()]
            pred = slope * float(last.dlog2L)
            any_ = True
            print(f"  {mdl:22s} T={T:g} ctx {ctx:>7,}  slope {slope:+.3f}/oct  "
                  f"G={int(last.step):>5}  dlog2L {last.dlog2L:.3f}  "
                  f"dtau predicted {pred:+.3f}  measured {last.dtau:+.3f}")
    if not any_:
        print("  (needs one model at >= 2 ctx values)")
    if a.csv:
        res.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
