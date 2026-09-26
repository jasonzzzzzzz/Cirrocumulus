#!/usr/bin/env python3
"""Allocator knife-edge characterisation (CPU, existing parquets only).

EXPLORATORY diagnostic, not a gate. The saved rows hold per-head summaries,
not per-token scores, so the water filler cannot be re-run under synthetic
noise-table perturbations. Two NATURAL small perturbations of the allocator's
action/noise table are in the data, measured in-process on identical q/K/V:

  P1  R11 A2 (job 988607): per-width noise c4/c6/c8 changed by the nested
      codebook (typically 0-5%), budgets B = 1..4, 4 cells, 22 prompts.
  P2  R10 (job 986347):   tier 1 removed (`full` -> `no1`; tier 1 holds
      <= 0.1% of physical tokens), B = 2, 3, same 4 model/ctx cells, 22 prompts.

Unit = one physical KV group (cell, prompt, family, layer, kv_head) at step 4:
one allocation shared by its query heads. Group error = RMS over those heads.

Reports, per perturbation, cell and budget:
  * the group ratio distribution and the rate of |ratio| >= 1.5x, 2x, 4x;
  * how concentrated the cell-level change in sum(err^2) is (top-k groups);
  * where events sit (layer third, family, eviction fraction, gain);
  * for P1, mono-arm non-monotonicity in B (a knife-edge without perturbation);
  * a slot-stratified group bootstrap: synthetic prompts draw each
    (family, layer, kv_head) slot from a random observed prompt, then the frozen
    per-prompt O_total is computed -> P(prompt O_total > 10% / 20%) and the
    distribution of an n-prompt cell mean.

usage: .venv/bin/python h0_measurement/bugs/11_nested_code_overhead/knife_edge.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "h0_measurement" / "results"
G = ["prompt", "family", "layer", "kv_head"]
SLOT = ["family", "layer", "kv_head"]
R11_JOB, R10_JOB = "988607", "986347"
R11_B, R10_B = (1, 2, 3, 4), (2, 3)
SUF = "__nested3"
DRAWS = 20_000
RNG = np.random.Generator(np.random.PCG64(11))


def pfx(label: str, B: int) -> str:
    return f"grp_tier_{label}_csv_b4_accum_{B}"


def load(path_glob: str) -> pd.DataFrame:
    f = glob.glob(path_glob)
    assert len(f) == 1, path_glob
    df = pd.read_parquet(f[0])
    df = df[df["step"].astype(int) == 4].copy()
    return df


def group_frame(df: pd.DataFrame, a_cols: dict, b_cols: dict, extra: dict) -> pd.DataFrame:
    """RMS over the query heads of each physical group for arm a / arm b."""
    work = df[G + ["layer"]].copy() if "layer" not in G else df[G].copy()
    agg = {}
    for B, c in a_cols.items():
        work[f"a{B}"] = df[c].astype(float) ** 2
        agg[f"a{B}"] = "mean"
    for B, c in b_cols.items():
        work[f"b{B}"] = df[c].astype(float) ** 2
        agg[f"b{B}"] = "mean"
    for k, (c, how) in extra.items():
        work[k] = df[c].astype(float)
        agg[k] = how
    g = work.groupby(G, as_index=False).agg(agg)
    for B in a_cols:
        g[f"a{B}"] = np.sqrt(g[f"a{B}"])
    for B in b_cols:
        g[f"b{B}"] = np.sqrt(g[f"b{B}"])
    g["nlayer"] = int(df["layer"].max()) + 1
    return g


def rates(r: np.ndarray) -> dict:
    lr = np.log2(r)
    return {
        "n": int(len(r)),
        "q01": float(np.quantile(r, .01)), "q50": float(np.median(r)),
        "q99": float(np.quantile(r, .99)), "max": float(r.max()), "min": float(r.min()),
        "up_1.5x": float((r >= 1.5).mean()), "up_2x": float((r >= 2).mean()),
        "up_4x": float((r >= 4).mean()),
        "down_1.5x": float((r <= 1 / 1.5).mean()), "down_2x": float((r <= .5).mean()),
        "abs_log2_gt1": float((np.abs(lr) >= 1).mean()),
    }


def concentration(a: np.ndarray, b: np.ndarray) -> dict:
    d = b ** 2 - a ** 2
    tot = float(d.sum())
    absd = np.sort(np.abs(d))[::-1]
    base = float((a ** 2).sum())
    return {"delta_sumsq_rel": tot / base,
            "top1_share_of_abs_change": float(absd[0] / absd.sum()),
            "top10_share_of_abs_change": float(absd[:10].sum() / absd.sum()),
            "top1pct_share_of_abs_change": float(absd[: max(1, len(absd) // 100)].sum() / absd.sum())}


def where(g: pd.DataFrame, r: np.ndarray, thr: float = 2.0) -> dict:
    ev = (r >= thr) | (r <= 1 / thr)
    out = {"events": int(ev.sum())}
    third = (g["layer"].to_numpy() * 3 // g["nlayer"].to_numpy())
    out["rate_by_layer_third"] = {int(t): float(ev[third == t].mean()) for t in range(3)}
    out["rate_by_family"] = {f: float(ev[(g["family"] == f).to_numpy()].mean())
                             for f in sorted(g["family"].unique())}
    if "evict" in g:
        ef = g["evict"].to_numpy()
        bins = [0, .25, .5, .75, .9, 1.01]
        out["rate_by_evict_frac"] = {f"{lo:.2f}-{hi:.2f}": float(ev[(ef >= lo) & (ef < hi)].mean())
                                     if ((ef >= lo) & (ef < hi)).any() else None
                                     for lo, hi in zip(bins[:-1], bins[1:])}
    if "gain" in g:
        ga = g["gain"].to_numpy()
        qs = np.quantile(ga, [0, .5, .9, .99, 1])
        out["rate_by_gain_quantile"] = {lab: float(ev[(ga >= lo) & (ga <= hi)].mean())
                                        for lab, lo, hi in (("<p50", qs[0], qs[1]), ("p50-p90", qs[1], qs[2]),
                                                            ("p90-p99", qs[2], qs[3]), (">p99", qs[3], qs[4]))}
    if ev.any():
        top = g.loc[ev].assign(ratio=r[ev]).sort_values("ratio", key=lambda s: -np.abs(np.log(s))).head(5)
        out["largest"] = top[G + ["ratio"]].to_dict("records")
    return out


def matched_rate(budgets, logE_mono, logE_target):
    bs = np.asarray(budgets, float)
    y = np.asarray(logE_mono, float)
    for i in range(len(bs) - 1):
        lo, hi = min(y[i], y[i + 1]), max(y[i], y[i + 1])
        if lo <= logE_target <= hi and y[i] != y[i + 1]:
            t = (y[i] - logE_target) / (y[i] - y[i + 1])
            return bs[i] + t * (bs[i + 1] - bs[i])
    i = 0 if logE_target > y[0] else len(bs) - 2
    t = (y[i] - logE_target) / (y[i] - y[i + 1])
    return bs[i] + t * (bs[i + 1] - bs[i])


def slot_bootstrap(g: pd.DataFrame, f3_by_prompt: dict, B: int, n_list=(4, 6, 16, 20)) -> dict:
    """Synthetic prompts: each slot drawn from a random observed prompt."""
    prompts = sorted(g["prompt"].unique())
    slots = g.groupby(SLOT)
    # slot x prompt arrays of squared error summed over the slot's heads.
    keys = list(slots.groups)
    idx = {p: i for i, p in enumerate(prompts)}
    A = {b: np.zeros((len(keys), len(prompts))) for b in R11_B}
    N = {b: np.zeros((len(keys), len(prompts))) for b in R11_B}
    W = np.zeros((len(keys), len(prompts)))
    for k, (key, blk) in enumerate(slots):
        for _, row in blk.iterrows():
            j = idx[row["prompt"]]
            W[k, j] = row["nheads"]
            for b in R11_B:
                A[b][k, j] = row["nheads"] * row[f"a{b}"] ** 2
                N[b][k, j] = row["nheads"] * row[f"b{b}"] ** 2
    f3 = np.array([f3_by_prompt[p] for p in prompts])
    pick = RNG.integers(0, len(prompts), size=(DRAWS, len(keys)))
    rows = np.arange(len(keys))[None, :]
    wsum = W[rows, pick].sum(1)
    Em = {b: np.sqrt(A[b][rows, pick].sum(1) / wsum) for b in R11_B}
    En = np.sqrt(N[B][rows, pick].sum(1) / wsum)
    f3s = f3[pick].mean(1)          # tier-3 fraction of the synthetic prompt
    O = np.empty(DRAWS)
    for d in range(DRAWS):
        bstar = matched_rate(R11_B, [math.log(Em[b][d]) for b in R11_B], math.log(En[d]))
        O[d] = (B + f3s[d]) / bstar - 1
    out = {"prompt_O_total_q50": float(np.median(O)),
           "prompt_O_total_q95": float(np.quantile(O, .95)),
           "prompt_O_total_q99": float(np.quantile(O, .99)),
           "P_prompt_gt_10pct": float((O > .10).mean()),
           "P_prompt_gt_20pct": float((O > .20).mean())}
    for n in n_list:
        m = O[RNG.integers(0, DRAWS, size=(DRAWS, n))].mean(1)
        out[f"cellmean_n{n}_q95"] = float(np.quantile(m, .95))
        out[f"P_cellmean_n{n}_gt_10pct"] = float((m > .10).mean())
        out[f"P_cellmean_n{n}_gt_20pct"] = float((m > .20).mean())
    return out


def main() -> None:
    out = {"note": "EXPLORATORY allocator knife-edge diagnostic; not a gate",
           "P1_nested_codebook_988607": {}, "P2_no1_986347": {}}
    lines = ["Allocator knife-edge characterisation (exploratory; CPU; existing parquets)", ""]

    # ---------------- P1: R11 in-process A/B
    lines.append("P1  nested codebook (R11 job 988607, step 4, physical KV groups)")
    for cell in range(4):
        df = load(str(RES / f"r11ab_main_{R11_JOB}_{cell}" / "h0_*.parquet"))
        name = f"{df['model'].iloc[0]}@{int(df['ctx'].iloc[0])}"
        a = {B: f"err_wf_{pfx('nested3', B)}" for B in R11_B}
        b = {B: f"err_wf_{pfx('nested3', B)}{SUF}" for B in R11_B}
        df["_one"] = 1.0
        df["_cpert"] = np.maximum.reduce([np.abs(np.log(df[f"c{w}_abs{SUF}"] / df[f"c{w}_abs"]))
                                          for w in (4, 6, 8)])
        extra = {"nheads": ("_one", "sum"), "cpert": ("_cpert", "max")}
        res_cell = {}
        for B in R11_B:
            df[f"_ev{B}"] = df[f"evict_frac_{pfx('nested3', B)}"]
            df[f"_gn{B}"] = (np.minimum(df[f"err_uniform{B}"] if f"err_uniform{B}" in df else np.inf,
                                        df[f"err_e{B}_grp_pp_accum_frac"])
                             / df[f"err_wf_{pfx('nested3', B)}"])
            extra[f"evict{B}"] = (f"_ev{B}", "mean")
            extra[f"gain{B}"] = (f"_gn{B}", "mean")
        g = group_frame(df, a, b, extra)
        f3 = {}
        for B in R11_B:
            phys = df.drop_duplicates(G)
            col = f"tier_frac_{pfx('nested3', B)}_b3{SUF}"
            f3[B] = ((phys[col] * phys["L"]).groupby(phys["prompt"]).sum()
                     / phys["L"].groupby(phys["prompt"]).sum()).to_dict()
        # non-monotone budget curve in the mono arm
        nonmono = float(np.mean([(g[f"a{B+1}"] > 1.05 * g[f"a{B}"]).mean() for B in R11_B[:-1]]))
        res_cell["mono_nonmonotone_in_B_rate"] = nonmono
        res_cell["noise_table_perturbation_median_abs_log"] = float(g["cpert"].median())
        for B in R11_B:
            r = (g[f"b{B}"] / g[f"a{B}"]).to_numpy()
            gg = g.rename(columns={f"evict{B}": "evict", f"gain{B}": "gain"})
            rb = {"rates": rates(r), "concentration": concentration(g[f"a{B}"].to_numpy(), g[f"b{B}"].to_numpy()),
                  "where": where(gg, r)}
            if B in (2, 3):
                rb["slot_bootstrap"] = slot_bootstrap(g, f3[B], B)
            res_cell[f"B{B}"] = rb
        out["P1_nested_codebook_988607"][name] = res_cell
        lines.append(f"  {name}: groups/budget={len(g)}  noise-table |log| median="
                     f"{res_cell['noise_table_perturbation_median_abs_log']:.3f}  "
                     f"mono non-monotone in B={100*nonmono:.2f}%")
        for B in (3, 2):
            rb = res_cell[f"B{B}"]
            rt, cc = rb["rates"], rb["concentration"]
            lines.append(f"    B={B}: ratio q01/q50/q99 {rt['q01']:.3f}/{rt['q50']:.3f}/{rt['q99']:.3f} "
                         f"max {rt['max']:.2f}  >=2x {100*rt['up_2x']:.3f}%  <=0.5x {100*rt['down_2x']:.3f}%  "
                         f"top1 group share of |dSSE| {100*cc['top1_share_of_abs_change']:.1f}%  "
                         f"top1% {100*cc['top1pct_share_of_abs_change']:.1f}%")
            w = rb["where"]
            lines.append(f"          2x events={w['events']} by layer third {w['rate_by_layer_third']}  "
                         f"by family {w['rate_by_family']}")
            sb = rb["slot_bootstrap"]
            lines.append(f"          slot bootstrap: prompt O_total q50 {100*sb['prompt_O_total_q50']:+.2f}% "
                         f"q95 {100*sb['prompt_O_total_q95']:+.2f}% q99 {100*sb['prompt_O_total_q99']:+.2f}%  "
                         f"P(prompt>10%) {100*sb['P_prompt_gt_10pct']:.2f}%  P(prompt>20%) "
                         f"{100*sb['P_prompt_gt_20pct']:.2f}%  | cell mean n=4 q95 "
                         f"{100*sb['cellmean_n4_q95']:+.2f}%  n=16 q95 {100*sb['cellmean_n16_q95']:+.2f}%  "
                         f"P(n=16 mean>10%) {100*sb['P_cellmean_n16_gt_10pct']:.2f}%")

    # ---------------- P2: R10 full -> no1
    lines += ["", "P2  tier 1 removed, full -> no1 (R10 job 986347, step 4, physical KV groups)"]
    for task in range(4):
        df = load(str(RES / f"r10_tiers_main_{R10_JOB}_{task}" / "h0_*.parquet"))
        name = f"{df['model'].iloc[0]}@{int(df['ctx'].iloc[0])}"
        df["_one"] = 1.0
        a = {B: f"err_wf_{pfx('full', B)}" for B in R10_B}
        b = {B: f"err_wf_{pfx('no1', B)}" for B in R10_B}
        extra = {"nheads": ("_one", "sum")}
        for B in R10_B:
            df[f"_t1{B}"] = df[f"tier_frac_{pfx('full', B)}_b1"]
            extra[f"t1_{B}"] = (f"_t1{B}", "mean")
            df[f"_ev{B}"] = df[f"evict_frac_{pfx('full', B)}"]
            extra[f"evict{B}"] = (f"_ev{B}", "mean")
        g = group_frame(df, a, b, extra)
        res_cell = {}
        lines.append(f"  {name}: groups/budget={len(g)}")
        for B in (3, 2):
            r = (g[f"b{B}"] / g[f"a{B}"]).to_numpy()
            used = (g[f"t1_{B}"] > 0).to_numpy()
            gg = g.rename(columns={f"evict{B}": "evict"})
            rb = {"rates": rates(r), "rates_groups_not_using_tier1": rates(r[~used]) if (~used).any() else None,
                  "frac_groups_using_tier1": float(used.mean()),
                  "concentration": concentration(g[f"a{B}"].to_numpy(), g[f"b{B}"].to_numpy()),
                  "where": where(gg, r)}
            res_cell[f"B{B}"] = rb
            rt, cc = rb["rates"], rb["concentration"]
            nu = rb["rates_groups_not_using_tier1"]
            lines.append(f"    B={B}: groups using tier 1 {100*rb['frac_groups_using_tier1']:.1f}%  "
                         f"ratio q01/q50/q99 {rt['q01']:.3f}/{rt['q50']:.3f}/{rt['q99']:.3f} max {rt['max']:.2f} "
                         f">=2x {100*rt['up_2x']:.3f}% <=0.5x {100*rt['down_2x']:.3f}%  "
                         f"(groups NOT using tier 1: changed {100*(1-(nu['q01']==1==nu['q99'])) if nu else 0:.0f}%, "
                         f"max {nu['max'] if nu else float('nan'):.3f})  top1 share {100*cc['top1_share_of_abs_change']:.1f}%")
        out["P2_no1_986347"][name] = res_cell

    text = "\n".join(lines) + "\n"
    (HERE / "knife_edge.txt").write_text(text)
    (HERE / "knife_edge.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(text)


if __name__ == "__main__":
    main()
