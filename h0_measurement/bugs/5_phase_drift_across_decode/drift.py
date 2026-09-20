#!/usr/bin/env python3
"""
R5 reader -- does a head's phase drift across a long decode?

    python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
        "h0_measurement/results/<JOB>/*.parquet" [...] [--calib 1] [--csv out.csv]

report.py cannot answer this: per_head() takes a median over EVERY row of a
(model, ctx, layer, head), steps included, so a drift is averaged away before it
is ever looked at. This keeps `step` as an axis.

One block per (model, ctx, decode_temp, schedule, corner set) -- a sparse
drift run and the dense bridge at the same (model, ctx, T) are different
experiments and must never share a table. Per measured step:

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
  lag1          median interior_lag_cost<B>_last_step on in-band heads: the cost
                of re-budgeting from ONE step ago.
  froz          the same for `first` -- the score frozen at the first probed
                step. THE C4 NUMBER: what keeping the prefill-time allocation
                costs t steps later. froz/lag1 is what re-budgeting buys.
                BLANKED past its horizon: the frozen score has never seen the t
                generated tokens, so the interior floors them at maxb, and that
                floor costs maxb*t bits out of B*L. At 8k and t = 4096 it eats
                89% of the budget and leaves 0.5 b/token for everything else --
                `froz` would then be measuring the floor policy, not staleness.
                Past --max-floor-share of the budget the column reads `-`;
                the horizon is t <= B*share*L/maxb, i.e. ~300 steps at 8k,
                ~1200 at 32k and ~4900 at 128k for B = 3.
                Shown ONLY for runs with the fresh-token fix ("interior_unseen_
                policy": "floor_maxb" in the .json). Earlier runs gave the
                token appended each step 0 bits -- evicted it -- and read 3-10x
                here for that reason alone (bugs/2/R3-report.md section 2), so
                their interior columns are blanked on load. The corner columns
                this table routes on (gain_best_practical) were never affected:
                last_step's only unseen token is the current one, which the
                corner always protected.
  eos%          rows past the first EOS (excluded unless --keep-post-eos)
  d4            median distinct-4-gram fraction of the text so far (loop check)

LOOPS ARE EXCLUDED, NOT JUST FLAGGED. Banning EOS pushes a model into repetition
over thousands of tokens: in the first R5 campaign qwen3-30b @32k fell to
d4 = 0.05-0.27 by step 4096 on all three prompts, and one llama31-8b @8k prompt
to 0.04. A loop is a real attention regime but not the one a deployed sampler
lives in, and it reads as phase drift. Any (prompt, step) at or below
--min-distinct4 is dropped, along with every LATER step of that prompt, since a
loop does not recover. `n_loop` in the header says how many were dropped.

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
import argparse, glob, json, math, os, re, sys
import numpy as np
import pandas as pd

BAND_MIN = 2.0      # keep identical to report.py / alloc.py


INTERIOR = re.compile(r"^(interior_lag_cost|gain_pp|in_band_pp|err_wf_pp|gain_u_pp|"
                      r"evict_frac_pp|unseen_frac_pp)")


def r5_gain(df, B):
    # NOTE the corner columns of `first` (gain_e<B>_first_frac and its w2p twin)
    # are NOT a baseline anyone would field: `first` bumps every position its
    # snapshot never saw, so past t = B*L/maxb the corner keeps ONLY generated
    # tokens and its error explodes by construction. `first` is here as an
    # INTERIOR score. Read interior_lag_cost<B>_first (`froz`), not its corner.
    """The band/route column, pinned to the `last_step` (TOVA) corner.

    `gain_best_practical` is a min over EVERY practical corner, so adding
    `first` to the evictor list (it must be an evictor to be an interior score)
    would silently strengthen the competitor and lower the band against the
    first campaign and against the bridge. Recomputing min(uniform, last_step)
    keeps the R5 verdict the same quantity across campaigns."""
    ls, un = f"gain_e{B}_last_step_frac", f"gain_u{B}"
    col = f"gain_best_practical{B}"
    if ls in df and un in df and df[ls].notna().any():
        df = df.copy()
        df[col] = np.minimum(df[un], df[ls])
    return df


def drop_loops(df, thr):
    """Truncate each run at the first step ANY of its prompts degenerates.

    Per-prompt truncation would leave the later steps supported by whichever
    prompts happened not to loop yet, and the per-head medians would then be
    comparing different prompt sets step to step -- which is itself a drift.
    The whole run stops at the earliest loop instead, so every step in the table
    rests on the same prompts."""
    if "gen_distinct4" not in df or thr <= 0:
        return df, 0, {}
    key = ["model", "ctx", "decode_temp", "schedule"]
    d4 = df.groupby(key + ["prompt", "step"]).gen_distinct4.median().reset_index()
    bad = d4[d4.gen_distinct4.notna() & (d4.gen_distinct4 <= thr)]
    if bad.empty:
        return df, 0, {}
    cut = bad.groupby(key).step.min().rename("cut").reset_index()
    m = df.merge(cut, on=key, how="left")
    keep = m.cut.isna() | (m.step < m.cut)
    where = {tuple(r[k] for k in key): int(r.cut) for _, r in cut.iterrows()}
    return df[keep.to_numpy()], int((~keep).sum()), where


def read_one(f):
    """One parquet, with its interior columns blanked unless the run has the
    fresh-token fix -- see `lag` in the module docstring."""
    d = pd.read_parquet(f)
    js = f[:-len(".parquet")] + ".json"
    pol = json.load(open(js)).get("interior_unseen_policy") if os.path.exists(js) else None
    if pol != "floor_maxb":
        cols = [c for c in d.columns if INTERIOR.match(c)]
        if cols:
            d[cols] = np.nan
            print(f"  note: {os.path.relpath(f)} predates the fresh-token fix -- "
                  f"{len(cols)} interior columns blanked (lag shows '-')")
    return d


def load(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        sys.exit(f"no parquet matched {patterns}")
    df = pd.concat([read_one(f) for f in files], ignore_index=True).copy()
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


def block(g, B, calib, lagcols, eos_by_step, lag1col=None, frozcol=None,
          maxb=8, max_floor_share=0.10):
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
          f"{'lag1':>6} {'froz':>6} {'eos%':>4} {'d4':>5}")
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
        def inband_cost(col):
            if not col or col not in ph or not have_g:
                return float("nan")
            inb = ph[gcol] >= BAND_MIN
            return float(ph.loc[inb, col].median()) if bool(inb.any()) else float("nan")

        lag = inband_cost(lag1col or (lagcols[0] if lagcols else None))
        froz = inband_cost(frozcol)
        # past its horizon the frozen column prices the maxb floor, not staleness
        if frozcol and math.isfinite(froz):
            uc = frozcol.replace("interior_lag_cost", "unseen_frac_pp")
            share = (maxb / B) * float(ph[uc].median()) if uc in ph else float("nan")
            if math.isfinite(share) and share > max_floor_share:
                froz = float("nan")
        dead = {b: (float(ph[f"evict_beats_b{b}"].mean())
                    if f"evict_beats_b{b}" in ph else float("nan")) for b in (1, 2)}
        row = dict(step=int(s), L=L, dlog2L=math.log2(L / L0),
                   tau=float(ph["tau"].median()),
                   dtau=float(ph["tau"].median()) - tau0,
                   ladder=float(ph["ladder_bits"].median()) if "ladder_bits" in ph
                   else float("nan"),
                   n95=float(ph["n95"].median()) if "n95" in ph else float("nan"),
                   dead1=dead[1], dead2=dead[2], band=band, gain=med_g,
                   flip_gen=fg, flip_rtr=fr, rho=rho, regret90=rg, lag=lag, froz=froz,
                   eos=float(eos_by_step.get(s, float("nan"))), d4=float(r["gen_distinct4"].median()))
        out.append(row)
        print(f"  {s:>5} {L:>7,.0f} {row['dlog2L']:>6.3f} {row['tau']:>6.3f} "
              f"{row['dtau']:>+6.3f} {num(row['ladder'])} {num(row['n95'], '{:6.0f}')} "
              f"{pct(dead[1]):>5} {pct(dead[2]):>5} {pct(band):>5} {num(med_g)} "
              f"{pct(fg):>5} {pct(fr):>5} {num(rho, '{:5.2f}')} {num(rg)} "
              f"{num(lag)} {num(froz)} {pct(row['eos'])} {num(row['d4'], '{:5.2f}')}")
    return pd.DataFrame(out)


def bridge(g, B):
    """accum vs last_step corners at the same steps, same rows."""
    ca, cl = f"gain_e{B}_accum_frac", f"gain_e{B}_last_step_frac"
    cu = f"gain_u{B}"
    # present AND populated: after concat, a sparse run carries the bridge's
    # columns as all-NaN, which is not a bridge
    if not all(c in g and g[c].notna().any() for c in (ca, cl, cu)):
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
    ap.add_argument("--maxb", type=int, default=8,
                    help="top tier, for the froz horizon (models.yaml maxb)")
    ap.add_argument("--max-floor-share", type=float, default=0.10,
                    help="blank `froz` once the maxb floor on unseen positions "
                         "costs more than this share of the head's budget")
    ap.add_argument("--min-distinct4", type=float, default=0.5,
                    help="drop a prompt from the first step whose distinct-4 "
                         "fraction falls to this or below (0 = keep loops)")
    ap.add_argument("--csv")
    a = ap.parse_args()

    df = load(a.parquet)
    df = df[df.family == a.family] if a.family != "all" else df
    df = r5_gain(df, a.B)
    df, n_loop, cuts = drop_loops(df, a.min_distinct4)
    if frozcol := next((c for c in df.columns
                        if c.startswith(f"interior_lag_cost{a.B}_first")), None):
        uc = frozcol.replace("interior_lag_cost", "unseen_frac_pp")
        if uc in df:
            hz = df.groupby(["model", "ctx"]).apply(
                lambda x: x.loc[(a.maxb / a.B) * x[uc] <= a.max_floor_share, "step"].max()
                if x[uc].notna().any() else float("nan"), include_groups=False)
            print("`froz` horizon (last step where the maxb floor costs <= "
                  f"{a.max_floor_share:.0%} of the budget):")
            for (mdl, ctx), st in hz.items():
                print(f"   {mdl} ctx {ctx:,}: step {'-' if pd.isna(st) else int(st)}")
    if n_loop:
        print(f"dropped {n_loop:,} rows to looping text (distinct-4 <= "
              f"{a.min_distinct4:g}; a loop is not a phase). Runs truncated at:")
        for (mdl, ctx, T, sched), st in sorted(cuts.items()):
            print(f"   {mdl} ctx {ctx:,} T={T:g} {sched}: first loop at step {st}, "
                  f"table stops before it")
    if "quantized" in df:
        df = df[df.quantized.astype(bool)]
    df = df.assign(_eos=df.past_eos.astype(bool))
    lagcols = sorted(c for c in df.columns
                     if re.match(rf"^interior_lag_cost{a.B}_", c))
    lag1col = next((c for c in lagcols if c.endswith(("_last_step", "_lag1"))), None)
    frozcol = next((c for c in lagcols if c.endswith("_first")), None)
    tables = []
    if "evictors" not in df:
        df = df.assign(evictors="?")
    for (mdl, ctx, T, sched, ev), g in df.groupby(
            ["model", "ctx", "decode_temp", "schedule", "evictors"]):
        n_eos = int(g.groupby("prompt")._eos.any().sum())
        print(f"\n=== {mdl}  ctx {ctx:,}  T={T:g}  {sched}  [{ev}]  "
              f"{g.prompt.nunique()} prompts, {n_eos} reached EOS"
              f"{' (EOS banned)' if bool(g.decode_ban_eos.any()) else ''} "
              f"{'(post-EOS rows kept)' if a.keep_post_eos else '(post-EOS rows dropped)'}")
        gg = g if a.keep_post_eos else g[~g._eos]
        t = block(gg, a.B, a.calib, [c for c in (lag1col, frozcol) if c] or lagcols,
                  g.groupby("step")._eos.mean(), lag1col, frozcol,
                  a.maxb, a.max_floor_share)
        bridge(gg, a.B)
        if t is not None:
            tables.append(t.assign(model=mdl, ctx=ctx, decode_temp=T,
                                   schedule=sched, evictors=ev))

    if not tables:
        return
    res = pd.concat(tables, ignore_index=True)

    # ---- L-growth test: does tau move within a generation as the across-ctx
    # slope says it should? Needs >= 2 ctx for one (model, temp).
    print("\nL-GROWTH TEST -- across-ctx slope at the calibration step vs the "
          "within-generation move at the last step")
    any_ = False
    for (mdl, T, sched, ev), r in res.groupby(["model", "decode_temp", "schedule", "evictors"]):
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
