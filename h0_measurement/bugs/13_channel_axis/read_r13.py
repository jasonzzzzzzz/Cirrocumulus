#!/usr/bin/env python3
"""
read_r13.py -- R13 reader: validity gates V1-V5 and the frozen decision table
(plan.md section 5).

  read_r13.py --validate DIR            (worker, after each cell; COMPLETE gate)
  read_r13.py --pilot JOB               (V1-V4 on the excluded pilot)
  read_r13.py --main JOB                (all cells -> main_<JOB>.{json,txt})
"""
from __future__ import annotations
import argparse, glob, json, math, os, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
BUDGETS = (2, 3, 4)
TIERS = (0, 1, 2, 3, 4, 5, 6, 8)
CELLS = [("llama31-8b", 8192), ("llama31-8b", 32768), ("qwen3-8b", 8192),
         ("qwen3-8b", 32768), ("mistral-7b", 8192), ("qwen15-moe-a2.7b", 8192)]


def load(d):
    return (pd.read_parquet(os.path.join(d, "heads.parquet")),
            pd.read_parquet(os.path.join(d, "groups.parquet")),
            json.load(open(os.path.join(d, "checks.json"))))


def validate_dir(d, min_rows=1):
    hd, gd, ck = load(d)
    fails = []
    if not ck.get("l2", {}).get("pass"):
        fails.append("V4 L2 capture")
    if not ck["v1_max_rel"] <= 1e-3:
        fails.append(f"V1 rc_u != TurboQuant(no norm corr): {ck['v1_max_rel']:.2e}")
    if not ck["v2_max_rel_key_err8"] < 0.01:
        fails.append(f"V2 8-bit channel key error {ck['v2_max_rel_key_err8']:.3f}")
    for B in BUDGETS:
        for col in [c for c in gd.columns if c.startswith("bits_") and c.endswith(str(B))]:
            if (gd[col] > B + 1e-9).any():
                fails.append(f"V3 {col} overspends")
            if not col.startswith("bits_t_") and gd[col].median() < B - 0.05:
                fails.append(f"V3 {col} median spend {gd[col].median():.3f} < {B - 0.05}")
    if len(hd) < min_rows:
        fails.append(f"V5 only {len(hd)} head rows")
    errs = [c for c in hd.columns if c.startswith("err_")]
    if not np.isfinite(hd[errs].to_numpy()).all():
        fails.append("non-finite errors")
    return fails, hd, gd, ck


def cell_stats(hd, gd):
    out = {"n_head_rows": int(len(hd)), "n_group_rows": int(len(gd))}
    for B in BUDGETS:
        for fam in ("ch", "rc"):
            for arm in ("wf", "cal", "xcal", "ks"):
                g = hd[f"err_{fam}_u{B}"] / hd[f"err_{fam}_{arm}{B}"].clip(lower=1e-12)
                out[f"gain_{fam}_{arm}{B}"] = float(g.median())
                out[f"gm_{fam}_{arm}{B}"] = float(np.exp(np.log(g).mean()))
                out[f"band_{fam}_{arm}{B}"] = float((g >= 2).mean())
            out[f"err_{fam}_u{B}"] = float(hd[f"err_{fam}_u{B}"].median())
        g = hd[f"err_t_u{B}"] / hd[f"err_t_wf{B}"].clip(lower=1e-12)
        out[f"gain_t_wf{B}"] = float(g.median())
        out[f"band_t_wf{B}"] = float((g >= 2).mean())
        out[f"err_t_u{B}"] = float(hd[f"err_t_u{B}"].median())
        for col in [c for c in gd.columns if c.startswith("share") and c.endswith(str(B))]:
            out[f"mean_{col}"] = float(gd[col].mean())
        for col in [c for c in gd.columns if c.startswith("bits_") and c.endswith(str(B))]:
            out[f"median_{col}"] = float(gd[col].median())
    for t in (1, 2, 3):
        for fam in ("ch", "rc"):
            out[f"dead{t}_{fam}"] = float(gd[f"dead{t}_{fam}"].mean())
            out[f"dead{t}_{fam}_ks"] = float(gd[f"dead{t}_{fam}_ks"].mean())
        out[f"dead{t}_tok"] = float((hd[f"sig2_{t}"] > 1.0).mean())
    out["tau_median"] = float(hd["tau"].median())
    # Q5: per (layer, KV head) units
    u = hd.assign(lc=np.log(hd["err_ch_u3"] / hd["err_ch_wf3"].clip(lower=1e-12)),
                  lt=np.log(hd["err_t_u3"] / hd["err_t_wf3"].clip(lower=1e-12)))
    u = u.groupby(["layer", "kv_head"])[["lc", "lt"]].median()
    out["q5_spearman"] = float(u["lc"].rank().corr(u["lt"].rank()))
    out["q5_units"] = int(len(u))
    return out


def verdicts(S):
    """S: {(model, ctx): stats}. Returns the frozen plan.md section 5 table."""
    cells = list(S.values())
    n = len(cells)
    v = {}
    d2 = sum(c["dead2_ch"] >= 0.05 for c in cells)
    d1 = sum(c["dead1_ch"] >= 0.05 and c["dead2_ch"] < 0.05 for c in cells)
    v["Q1"] = ("C1_DEAD_CH" if d2 >= 3 else "ONLY_1BIT_DEAD" if d1 >= 3 else "NO_DEAD_CH")
    moves, tok = [], []
    for m in ("llama31-8b", "qwen3-8b"):
        a, b = S.get((m, 8192)), S.get((m, 32768))
        if a and b:
            moves.append(100 * (b["dead2_ch"] - a["dead2_ch"]))
            tok.append(100 * (b["dead2_tok"] - a["dead2_tok"]))
    if len(moves) == 2 and all(abs(x) >= 3 for x in moves) and np.sign(moves[0]) == np.sign(moves[1]):
        q2 = "L_DRIVEN"
    elif len(moves) == 2 and all(abs(x) < 1.5 for x in moves):
        q2 = "L_FREE"
    else:
        q2 = "MIXED"
    v["Q2"] = q2 + ("" if any(abs(x) >= 3 for x in tok) else " (control_flat)")
    v["Q2_dead2_ch_move_pts"] = moves
    v["Q2_dead2_tok_move_pts"] = tok
    best = [max(c["gain_ch_xcal2"], c["gain_ch_ks2"]) for c in cells]
    v["Q3"] = ("PAYS" if sum(x >= 1.20 for x in best) >= 3 else
               "NEGLIGIBLE" if all(x < 1.05 for x in best) else "MARGINAL")
    v["Q3_best_deployable_gain_B2"] = best
    rc = [c["gain_rc_xcal3"] for c in cells]
    v["Q4"] = ("SURVIVES" if sum(x >= 1.10 for x in rc) >= 3 else
               "CONSUMED" if all(x < 1.03 for x in rc) else "MARGINAL")
    v["Q4_rc_xcal_gain_B3"] = rc
    rho = [c["q5_spearman"] for c in cells]
    v["Q5"] = "ORTHOGONAL" if sum(abs(x) < 0.3 for x in rho) >= 4 else "COUPLED"
    v["Q5_spearman"] = rho
    v["n_cells"] = n
    return v


def fmt_table(S):
    lines = []
    hdr = (f"{'cell':26s} {'rows':>6s} {'tau':>5s} | dead1/2/3 ch  | dead1/2/3 rc  | dead1/2/3 tok"
           f" | gain B2: ch_wf ch_cal ch_xcal ch_ks | B3: rc_wf rc_cal rc_xcal rc_ks | t_wf B2/B3 | rho5")
    lines.append(hdr)
    for (m, ctx), c in S.items():
        lines.append(
            f"{m + '@' + str(ctx):26s} {c['n_head_rows']:6d} {c['tau_median']:5.2f} | "
            f"{100*c['dead1_ch']:4.1f} {100*c['dead2_ch']:4.1f} {100*c['dead3_ch']:4.1f} | "
            f"{100*c['dead1_rc']:4.1f} {100*c['dead2_rc']:4.1f} {100*c['dead3_rc']:4.1f} | "
            f"{100*c['dead1_tok']:4.1f} {100*c['dead2_tok']:4.1f} {100*c['dead3_tok']:4.1f} | "
            f"{c['gain_ch_wf2']:5.2f} {c['gain_ch_cal2']:5.2f} {c['gain_ch_xcal2']:5.2f} {c['gain_ch_ks2']:5.2f} | "
            f"{c['gain_rc_wf3']:5.2f} {c['gain_rc_cal3']:5.2f} {c['gain_rc_xcal3']:5.2f} {c['gain_rc_ks3']:5.2f} | "
            f"{c['gain_t_wf2']:5.2f}/{c['gain_t_wf3']:5.2f} | {c['q5_spearman']:+.2f}")
    return "\n".join(lines)


def run_main(job):
    dirs = sorted(glob.glob(str(RESULTS / f"r13_main_{job}_*")))
    S, problems = {}, {}
    for d in dirs:
        if not os.path.exists(os.path.join(d, "COMPLETE")):
            problems[d] = ["not COMPLETE"]
            continue
        fails, hd, gd, ck = validate_dir(d, min_rows=1000)
        if fails:
            problems[d] = fails
            continue
        S[(ck["model"], int(ck["ctx"]))] = cell_stats(hd, gd)
    missing = [c for c in CELLS if c not in S]
    status = "valid" if not missing and not problems else "invalid_r13"
    v = verdicts(S) if S else {}
    res = {"protocol": "r13_channel_axis_v1", "job": job, "status": status,
           "missing_cells": missing, "problems": problems, "verdicts": v,
           "cells": {f"{m}@{c}": s for (m, c), s in S.items()}}
    json.dump(res, open(HERE / f"main_{job}.json", "w"), indent=1)
    txt = [f"R13 main job {job}: status={status}  missing={missing}"]
    if problems:
        txt.append(f"problems: {json.dumps(problems, indent=1)}")
    if S:
        txt += ["", fmt_table(S), ""]
        txt += [f"{k}: {val}" for k, val in v.items()]
    open(HERE / f"main_{job}.txt", "w").write("\n".join(txt) + "\n")
    print("\n".join(txt))
    return 0 if status == "valid" else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate")
    g.add_argument("--pilot")
    g.add_argument("--main")
    a = ap.parse_args()
    if a.validate:
        fails, *_ = validate_dir(a.validate)
        if fails:
            print("INVALID:", fails); sys.exit(1)
        print("cell valid (V1-V4)")
    elif a.pilot:
        ds = glob.glob(str(RESULTS / f"r13_pilot_{a.pilot}_*"))
        if not ds or not all(os.path.exists(os.path.join(d, "COMPLETE")) for d in ds):
            print("pilot incomplete"); sys.exit(1)
        for d in ds:
            fails, hd, gd, ck = validate_dir(d)
            if fails:
                print("pilot INVALID:", fails); sys.exit(1)
            s = cell_stats(hd, gd)
            print(f"pilot ok: {d}  rows={len(hd)}  V1={ck['v1_max_rel']:.2e}")
            print(fmt_table({(ck['model'], ck['ctx']): s}))
    else:
        sys.exit(run_main(a.main))


if __name__ == "__main__":
    main()
