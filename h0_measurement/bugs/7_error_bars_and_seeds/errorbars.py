#!/usr/bin/env python3
"""
R7 reader -- how wide is a band fraction, and does its verdict survive?

    python h0_measurement/bugs/7_error_bars_and_seeds/errorbars.py \
        "h0_measurement/results/<JOB>/*.parquet" [...] \
        [--targets sym_acc,sym_min5,e2_acc,e2_min5,oracle] [--score accum] \
        [--B 3] [--boot 2000] [--csv out.csv]

report.py and boundary.py print a band as a point. It is not one. THREE things
move it, and only the third is a "seed":

  PROMPTS      which documents were drawn. Measured by comparing disjoint
               prompt BLOCKS of equal size (R7 runs 2-3 blocks per cell, and
               the R3 cell is the rot_seed=0 block of the same cell).
  CORNER SET   which evictors were configured. `gain_best_practical` /
               `gain_pp` take a min over the CONFIGURED practical corners, so a
               five-corner run reads lower than an accum-only one -- 18.9 vs
               14.4 on qwen3-30b @128k, which is NARROW vs STOP. Both versions
               are computed here from the same rows, by reconstruction:
                   e2_acc  = min(err_uniform, err_e_accum) / err_wf
                   sym_acc = min(err_uniform, err_e_accum) / err_wf_pp_accum
  HEADS        layer-cluster bootstrap inside a block (heads of one layer share
               a residual stream), as in bugs/6/boundary.py.
  rotation     rot_seed: the same documents under a different quantizer
               rotation, plus ordinary rerun noise. Reported separately because
               the ROADMAP proposed it as THE error bar; it is the smallest one.

BLOCK SIZES MUST MATCH. A head's gain is a median over its rows, so a band over
more prompts is a different statistic, not a better sample of the same one: on
the E2 cell of qwen3-30b @128k it falls 21.0 -> 18.9 from 1 prompt to 4. Blocks
are therefore compared at the reference size (the smallest block seen for that
cell), and the pooled estimate is printed beside them, labelled, never mixed in.

TARGETS (at B bits, band = gain >= report.BAND_MIN):
  sym_acc    interior and corner both lagged, accum corner -- THE HEADLINE
  sym_min5   the same, corner = min over every practical corner in the run
  e2_acc     practical corner (accum), oracle interior
  e2_min5    practical corner (min over the run's set), oracle interior
  oracle     oracle corner and interior
  routed     median routed gain (R2's robust statistic), same blocks

ROWS. Quantized rows where every configured corner scored, per
report.drop_partial_corners -- the R3-report.md section 1 selection, so every
target describes the same heads. Cells failing report.family_gate are dropped
and named. `sym_*` is read only from runs whose .json carries
"interior_unseen_policy": "floor_maxb" (R3-report.md section 2); others keep
their oracle / e2 columns.
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

LINES = {"STOP": 100 * F_STOP, "GO": 100 * F_GO}
TARGETS = ("sym_acc", "sym_min5", "e2_acc", "e2_min5", "oracle")


# ------------------------------------------------------------------ loading ---
def sidecar(parquet):
    js = parquet[: -len(".parquet")] + ".json"
    try:
        with open(js) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def gain_columns(d, score, B):
    """The five band targets as per-row gain series, by reconstruction where a
    column does not exist in this run's corner set."""
    out = {}
    eu = d.get(f"err_uniform{B}")
    wf = d.get(f"err_wf{B}")
    acc = d.get(f"err_e{B}_{score}_frac")
    wfp = d.get(f"err_wf_pp{B}_{score}")
    if f"gain_best{B}" in d:
        out["oracle"] = d[f"gain_best{B}"]
    if f"gain_best_practical{B}" in d:
        out["e2_min5"] = d[f"gain_best_practical{B}"]
    if f"gain_pp{B}_{score}" in d:
        out["sym_min5"] = d[f"gain_pp{B}_{score}"]
    if eu is not None and acc is not None:
        num = np.minimum(eu, acc)
        if wf is not None:
            out["e2_acc"] = num / wf.clip(lower=1e-12)
        if wfp is not None:
            out["sym_acc"] = num / wfp.clip(lower=1e-12)
    return {k: v for k, v in out.items() if v is not None and pd.notna(v).any()}


def load_runs(files, a):
    """One record per (file, model, ctx) = one RUN of one cell. A run holds one
    or more prompt BLOCKS; chunk_blocks() cuts them, because R7's default sheet
    puts all three blocks of a cell in ONE job (n_prompts = 3x the reference)."""
    recs, notes = [], []
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            notes.append(f"SKIP {f}: unreadable ({type(e).__name__})")
            continue
        run = os.path.basename(os.path.dirname(f))
        side = sidecar(f)
        fixed = side.get("interior_unseen_policy") == "floor_maxb"
        rot = side.get("rot_seed", (side.get("config") or {}).get("rot_seed", 0))
        gates = family_gate(d)
        if "quantized" in d:
            d = d[d["quantized"]]
        d, _ = drop_partial_corners(d)
        pr = f"gain_best_practical{a.B}"
        if pr in d and d[pr].notna().any():
            d = d[d[pr].notna()]
        if not len(d):
            notes.append(f"SKIP {run}/{os.path.basename(f)}: no complete rows")
            continue
        for (m, c), g in d.groupby(["model", "ctx"]):
            cols = gain_columns(g, a.score, a.B)
            if not fixed and not a.allow_unfixed:
                for k in ("sym_acc", "sym_min5"):
                    cols.pop(k, None)
            prompts = sorted(int(x) for x in g["prompt"].unique())
            ph = g.groupby(["layer", "head"])
            lay = np.sort(g["layer"].unique())
            # THE HAYSTACK IS KEYED ON THE CORPUS, NOT JUST THE INDEX. The book
            # order is shuffled with _seed("corpus-order", corpus_sha)
            # (sievelib/prompts.py), so re-staging the corpus makes prompt k a
            # DIFFERENT document: llama31-8b p0 is martin-chuzzlewit under
            # corpus b524da5e... and anna-karenina under 0a26bc1e.... Two runs
            # therefore share a sample only if they share a corpus_sha.
            sha = (str(g["corpus_sha"].iloc[0]) if "corpus_sha" in g
                   else str(side.get("corpus_sha", "")))
            recs.append(dict(
                run=run, model=m, ctx=int(c), rot_seed=int(rot or 0),
                corpus_sha=sha,
                fixed=fixed, prompts=tuple(prompts), n_prompts=len(prompts),
                evictors=cfg_value(g, "evictors"),
                gate=gates.get((m, int(c)), {"passed": True, "reason": ""}),
                docs=frozenset(g["corpus_doc"].unique())
                if "corpus_doc" in g else frozenset(),
                frame=g, layers=lay, n_heads=ph.ngroups, cols=cols))
    return recs, notes


# ---------------------------------------------------------------- estimates ---
def per_head(frame, gain, prompts=None):
    """Per-(layer, head) MEDIAN gain over a prompt subset -- computed ONCE.
    Median first, then count, which is the order report.py uses."""
    d, g = frame, gain
    if prompts is not None:
        keep = d["prompt"].isin(prompts).values
        d, g = d[keep], np.asarray(g)[keep]
    ph = pd.Series(np.asarray(g), index=d.index).groupby(
        [d["layer"], d["head"]]).median()
    ph = ph[np.isfinite(ph.values)]
    return ph


def chunk_blocks(recs, size, drop_partial=True):
    """Cut every run into consecutive prompt blocks of exactly `size`.

    THE POINT. R7's default sheet runs one job per cell at n_prompts = 3 x the
    reference block, so prompts 0..n-1 / n..2n-1 / 2n..3n-1 are three samples
    inside ONE parquet. Reading a run as a single block would either compare a
    12-prompt band against a 4-prompt one (a different statistic, plan.md B3)
    or silently drop the two independent blocks. A remainder shorter than
    `size` is dropped and named rather than reported as a short block.
    """
    out, notes = [], []
    for r in recs:
        pr = list(r["prompts"])
        nb = len(pr) // size
        if nb == 0:
            notes.append(f"{r['run']}: only {len(pr)} prompt(s), shorter than "
                         f"the reference block ({size}) -- dropped")
            continue
        for i in range(nb):
            blk = tuple(pr[i * size:(i + 1) * size])
            b = dict(r)
            b["prompts"], b["n_prompts"], b["block"] = blk, size, i
            b["docs"] = frozenset(
                r["frame"].loc[r["frame"]["prompt"].isin(blk), "corpus_doc"].unique()
            ) if "corpus_doc" in r["frame"] else frozenset()
            out.append(b)
        rem = len(pr) - nb * size
        if rem and drop_partial:
            notes.append(f"{r['run']}: {rem} leftover prompt(s) "
                         f"{pr[nb * size:]} dropped (partial block)")
    return out, notes


def block_estimates(rec, t, a, rng):
    """Point band, median routed gain, and layer-bootstrap band replicates for
    ONE prompt block (already cut to the reference size by chunk_blocks).

    The bootstrap resamples LAYERS (heads of one layer share a residual stream,
    so heads are not independent draws). It reuses the per-head medians through
    per-layer counts -- the same construction as bugs/6/boundary.py -- rather
    than re-medianing the frame per replicate.
    """
    ph = per_head(rec["frame"], rec["cols"][t], rec["prompts"])
    if not len(ph):
        return np.nan, np.nan, np.full(a.boot, np.nan)
    lay = np.asarray(ph.index.get_level_values(0))
    uniq = np.unique(lay)
    li = np.searchsorted(uniq, lay)
    v = ph.values
    n = np.bincount(li, minlength=len(uniq)).astype(float)
    inb = np.bincount(li, weights=(v >= BAND_MIN).astype(float),
                      minlength=len(uniq))
    L = len(uniq)
    W = rng.multinomial(L, np.full(L, 1.0 / L), size=a.boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = 100.0 * (W @ inb) / (W @ n)
    return 100.0 * float((v >= BAND_MIN).mean()), float(np.median(v)), reps


def combine(blocks, reps, extra_sd=0.0):
    """90% interval for the MEAN band over blocks.

    Two independent components, and neither may swallow the other:
      layers   every block of a cell measures THE SAME HEADS, so the layer
               bootstrap is correlated across blocks, not independent: a
               second prompt block buys nothing on the head axis. The
               replicates are therefore drawn JOINTLY (report_target hands
               every block an rng with the same seed, so replicate i resamples
               the same layers in each) and averaged replicate by replicate --
               dividing a per-block sd by sqrt(k) would understate it.
      prompts  the spread BETWEEN block means, as an sd of the mean.
    Concatenating the blocks' replicates instead would fold the between-block
    spread into the "layer" term and then add it a second time.
    """
    ok = [np.asarray(r, float) for r in reps]
    ok = [r for r in ok if np.isfinite(r).any()]
    blocks = [b for b in blocks if np.isfinite(b)]
    if not ok or not blocks:
        return np.nan, np.nan
    m = min(len(r) for r in ok)
    joint = np.nanmean(np.vstack([r[:m] for r in ok]), axis=0)
    joint = joint[np.isfinite(joint)]
    sd_layer = float(np.std(joint, ddof=1)) if len(joint) > 1 else 0.0
    sd_block = (float(np.std(blocks, ddof=1)) / np.sqrt(len(blocks))
                if len(blocks) > 1 else 0.0)
    sd = float(np.sqrt(sd_layer ** 2 + sd_block ** 2 + extra_sd ** 2))
    return float(np.mean(blocks)) - 1.645 * sd, float(np.mean(blocks)) + 1.645 * sd


def verdict(lo, hi):
    """What the interval permits, not what the point estimate says."""
    if not np.isfinite(lo):
        return "?"
    out = []
    for name, F in LINES.items():
        if lo < F <= hi:
            out.append(f"STRADDLES {name} ({F:.0f})")
    if out:
        return "; ".join(out)
    if hi < LINES["STOP"]:
        return "STOP, clear"
    if lo >= LINES["GO"]:
        return "GO, clear"
    return "NARROW, clear of both lines"


# --------------------------------------------------------------------- main ---
def report_cell(key, recs, a, rng, rows):
    model, ctx = key
    print("\n" + "=" * 96)
    print(f"{model} @ {ctx:,}")
    print("=" * 96)
    bad = [r for r in recs if not r["gate"]["passed"]]
    for r in bad:
        print(f"  DROPPED (input-validity gate): {r['run']}: {r['gate']['reason']}")
    recs = [r for r in recs if r["gate"]["passed"]]
    if not recs:
        return
    # The reference block size: what the R3/E2 cell this is compared against
    # used, i.e. the smallest run here, unless --block-size says otherwise.
    size = a.block_size or min(r["n_prompts"] for r in recs)
    runs, recs = recs, None
    recs, cut_notes = chunk_blocks(runs, size)
    for n in cut_notes:
        print(f"  note: {n}")
    if not recs:
        print(f"  no run has {size} prompt(s) -- nothing to compare")
        return
    print(f"  {len(runs)} run(s) -> {len(recs)} prompt block(s) of {size}")
    for r in sorted(recs, key=lambda x: (x["run"], x["prompts"])):
        print(f"    {r['run']:<14} prompts {r['prompts'][0]}-{r['prompts'][-1]}"
              f"  rot_seed={r['rot_seed']}  corpus={r['corpus_sha'][:8]}"
              f"  corners={r['evictors']}  {'fixed' if r['fixed'] else 'PRE-FIX'}")
    shas = {r["corpus_sha"] for r in recs}
    if len(shas) > 1:
        print(f"    WARNING: {len(shas)} different corpora here "
              f"({', '.join(sorted(s[:8] for s in shas))}). The book order is "
              f"seeded on corpus_sha (sievelib/prompts.py), so the SAME prompt "
              f"index reads a DIFFERENT document in each -- blocks across "
              f"corpora are independent samples, never a rotation control.")
    # Blocks are disjoint WINDOWS; past the number of full-window books they
    # re-read a book at a different offset (sievelib/prompts.py). Say so.
    for i, x in enumerate(recs):
        for y in recs[i + 1:]:
            if x["prompts"] == y["prompts"] or not (x["docs"] & y["docs"]):
                continue
            print(f"    note: blocks {x['prompts'][0]}-{x['prompts'][-1]} and "
                  f"{y['prompts'][0]}-{y['prompts'][-1]} share "
                  f"{len(x['docs'] & y['docs'])} document(s) at different "
                  f"offsets -- not fully independent samples")

    for t in a.targets:
        # `*_min5` is a min over the CONFIGURED practical corners, so runs with
        # different corner sets measure different things and must not be pooled
        # (on qwen3-30b @128k that difference is 18.9 vs 14.5 -- NARROW vs STOP).
        # The reconstructed accum-only targets are corner-set independent.
        groups = {}
        for r in recs:
            if t in r["cols"]:
                groups.setdefault(r["evictors"] if t.endswith("min5") else "", []
                                  ).append(r)
        for ev, have in sorted(groups.items()):
            report_target(t, ev, have, size, a, rng, rows, model, ctx)

    # THE CORNER-SET COMPONENT, measured WITHIN one run so the rows are
    # identical: accum alone against the min over every corner that run
    # configured. This is the 18.9-vs-14.4 that R3-report.md 3.4 flagged.
    for acc, mn in (("e2_acc", "e2_min5"), ("sym_acc", "sym_min5")):
        for r in recs:
            if acc not in r["cols"] or mn not in r["cols"]:
                continue
            if len(str(r["evictors"]).split(",")) <= 2:    # oracle + accum only
                continue
            va = block_estimates(r, acc, a, rng)[0]
            vb = block_estimates(r, mn, a, rng)[0]
            print(f"\n  corner set ({r['run']}, block "
                  f"{r['prompts'][0]}-{r['prompts'][-1]}, {r['evictors']}): "
                  f"{acc} {va:5.1f}%  vs  {mn} {vb:5.1f}%   "
                  f"({vb - va:+.1f} pts from the corner definition alone)")


def report_target(t, ev, have, size, a, rng, rows, model, ctx):
    pts, routed, reps = [], [], []
    for r in have:
        # a FRESH rng with the same seed per block: identical layer draws, so
        # the replicates line up across blocks and combine() can average them
        # replicate by replicate (see its docstring)
        p, ro, rp = block_estimates(r, t, a, np.random.default_rng(a.seed))
        pts.append(p); routed.append(ro); reps.append(rp)
        rows.append(dict(model=model, ctx=ctx, target=t, run=r["run"],
                         rot_seed=r["rot_seed"], block_lo=r["prompts"][0],
                         block_hi=r["prompts"][-1], n_prompts=size,
                         evictors=r["evictors"], band=p, routed=ro))
    # THE SAME SAMPLE is (corpus, prompt indices) -- not the indices alone,
    # because the corpus_sha shuffles which book each index reads. Blocks that
    # share it differ only in rotation / rerun; blocks that do not are
    # different documents, i.e. part of the prompt component.
    same = {}
    for r, p in zip(have, pts):
        same.setdefault((r["corpus_sha"], r["prompts"]), []).append(p)
    rot_sd = [float(np.std(v, ddof=1)) for v in same.values() if len(v) > 1]
    rot_sd = float(np.mean(rot_sd)) if rot_sd else np.nan
    # different prompt blocks -> the sampling component
    uniq = [np.mean(v) for v in same.values()]
    pr_sd = float(np.std(uniq, ddof=1)) if len(uniq) > 1 else np.nan
    lo, hi = combine(pts, reps)
    # the same cell read over EVERY prompt of one run: lower noise, but a
    # different statistic (the band falls as prompts are added, plan.md B3)
    pooled = None
    runs = {r["run"]: r for r in have}
    if len(runs) == 1:
        r0 = next(iter(runs.values()))
        allp = tuple(sorted(int(x) for x in r0["frame"]["prompt"].unique()))
        if len(allp) > size:
            ph = per_head(r0["frame"], r0["cols"][t], allp)
            pooled = (100.0 * float((ph.values >= BAND_MIN).mean()), len(allp))
    label = t if not ev else f"{t} [{ev}]"
    print(f"\n  {label:38s} band {np.mean(pts):5.1f}%  90% [{lo:5.1f}, {hi:5.1f}]"
          f"   routed {np.mean(routed):4.2f}x")
    print(f"      {len(pts)} block(s): " + " ".join(f"{p:5.1f}" for p in pts)
          + ("   sd(prompt blocks) %4.2f" % pr_sd if np.isfinite(pr_sd)
             else "   sd(prompt blocks) n/a -- every block reads the SAME "
                  "prompts, so this interval covers layers only")
          + ("   sd(same prompts, rerun/rotation) %4.2f" % rot_sd
             if np.isfinite(rot_sd) else ""))
    print(f"      -> {verdict(lo, hi)}")
    if pooled is not None:
        print(f"      pooled over all {pooled[1]} prompts of the run: "
              f"{pooled[0]:5.1f}%  -- a LOWER-NOISE statistic, not a replicate "
              f"(the band falls as prompts are added)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--score", default="accum")
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0,
                    help="prompts per block (default: the smallest run in the "
                         "cell, i.e. the reference sample it is compared "
                         "against). A SMALLER value splits the runs already on "
                         "disk into several blocks and prices the prompt "
                         "component with no GPU time -- at the cost of "
                         "measuring a noisier statistic (plan.md B3), so use it "
                         "to size the effect, not to report a band.")
    ap.add_argument("--allow-unfixed", action="store_true",
                    help="read the symmetric targets from runs without the "
                         "floor_maxb marker (job92*/job214*) -- PROVISIONAL")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    a.targets = [t.strip() for t in a.targets.split(",") if t.strip()]

    files = sorted({f for p in a.inputs for f in glob.glob(p)
                    if f.endswith(".parquet")
                    and not os.path.basename(f).startswith("validity_")})
    if not files:
        raise SystemExit("no measurement parquet matched")
    recs, notes = load_runs(files, a)
    print(f"{len(files)} parquet file(s), {len(recs)} run(s)")
    for n in notes:
        print("  " + n)
    cells = {}
    for r in recs:
        cells.setdefault((r["model"], r["ctx"]), []).append(r)
    rng = np.random.default_rng(a.seed)
    rows = []
    for key in sorted(cells):
        report_cell(key, cells[key], a, rng, rows)
    print("\nAn interval that straddles a line means the VERDICT is not "
          "reportable for that cell, whatever the point estimate reads.")
    if a.csv and rows:
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
