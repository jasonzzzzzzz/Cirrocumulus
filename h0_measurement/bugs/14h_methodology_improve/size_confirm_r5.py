#!/usr/bin/env python3
"""The confirmatory campaign's size from R5's variance (plan.md, amendment "R6 (draft)"). From R5's
per-unit paired differences (and R5.3's where it has run): the units per model x family needed to show
a tail design non-inferior to FP8 on KL per span token, and the accuracy floor, with power from
  (a) the normal approximation, at R5's mean and SD and at a conservative pair (SD at its 80% upper
      confidence limit, the mean at its 80% upper bound), and
  (b) simulation of the confirmatory test itself (the unit bootstrap's one-sided 95% upper bound)
      drawing units from R5's empirical per-unit differences.
Writes findings/R5_sizing.{md,json}. No model, CPU only.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/size_confirm_r5.py [--r53 TAG:JOB ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

from statistics import NormalDist

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_stage1h_r5 as R  # noqa: E402

R5_BLOCKS = [("h5llama128_r1", 1059655), ("h5llama128_r1", 1059656), ("h5llama128_r3a", 1059657),
             ("h5llama128_r3a", 1059658), ("h5llama128_r3b", 1059659), ("h5llama128_hm", 1059660),
             ("h5qwen32_r2", 1059662), ("h5qwen32_r3a", 1059663), ("h5qwen32_r3a", 1059664),
             ("h5qwen32_r3b", 1059665)]               # R5's valid blocks (lbv2 1059661 failed V)
R53_BLOCKS = [("h53llama128_r1", 22773130)]
DESIGNS = [("qread2t4kqT_v4", 0.125), ("tail4x_v4", 0.125), ("tail3x_v3", 0.125), ("tail2x_v2", 0.125),
           ("tail2x_v2", 0.25)]                       # R5's tail arm (= tail4_v4) and R5.3's candidates
REFS = [("uniform+v4", 4.0), ("kivi4_v4", 4.0)]      # baselines, for the plan's secondary comparisons
MARGINS = [(1.25, 2e-4), (1.0, 2e-4), (1.5, 5e-4)]    # (rho, kappa): mean KL_d < rho mean KL_FP8 + kappa; first = primary
ALPHA, POWER_MODEL = 0.05, 0.80                       # one-sided; overall power per model (all its families)
ACC_DELTAS = (0.02, 0.05)                             # accuracy margins (FP-correct units, pooled per model)
SIX_ARMS = ["fp", "fp8kv", "qread2t4kqT_v4", "uniform+v4", "kivi4_v4", "fp_noise"]
U = ["prompt_idx", "task"]
N_GRID = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480]


def load(blocks):
    parts, secs = [], []
    for tag, job in blocks:
        d = os.path.join(R.RESULTS, f"r14s1h_{tag}_{job}")
        sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
        if not sp:
            print(f"missing {tag}_{job}")
            continue
        side = json.load(open(sp[0]))
        x = pd.read_parquet(os.path.join(d, side["parquet"]))
        x = x.assign(model=side["model"], cell=tag.split("_")[1], family=x.task.map(R.FAMILY))
        parts.append(x[["model", "cell", "family"] + U + ["arm", "B", "kl_span_tok", "a_span_nll_tok", "score"]])
        six = x[x.arm.isin(SIX_ARMS)]
        t6 = six.assign(t=six.t_arm.fillna(0) + six.t_tf.fillna(0)).groupby(U).t.sum()
        pre = x.groupby(U).agg(p=("t_prefill", "first"), c=("t_precompute", "first"))
        secs.append(pd.DataFrame(dict(model=side["model"], family=x.groupby(U).family.first(),
                                      s6=t6 + pre.p + pre.c.fillna(0))).reset_index())
    return pd.concat(parts, ignore_index=True), pd.concat(secs, ignore_index=True)


def paired(rows, arm, B, rho):
    """Per unit: y = KL_d - rho KL_FP8, dKL, accuracy change on FP-correct units, per model x family."""
    out = {}
    for (m, f), g in rows.groupby(["model", "family"]):
        a = g[(g.arm == arm) & np.isclose(g.B, B)].drop_duplicates(["cell"] + U).set_index(["cell"] + U)
        e = g[g.arm == "fp8kv"].drop_duplicates(["cell"] + U).set_index(["cell"] + U)
        fp = g[g.arm == "fp"].drop_duplicates(["cell"] + U).set_index(["cell"] + U)
        idx = a.index.intersection(e.index)
        if not len(idx):
            continue
        ok = np.isfinite(a.kl_span_tok[idx].values) & np.isfinite(e.kl_span_tok[idx].values)
        kd, k8 = a.kl_span_tok[idx].values[ok], e.kl_span_tok[idx].values[ok]
        corr = (fp.score.reindex(idx) >= 1 - 1e-9).values
        dacc = (a.score[idx] - e.score[idx]).values[corr]
        out[(m, f)] = dict(y=kd - rho * k8, kd=kd, k8=k8, dacc=dacc, cells=sorted({i[0] for i in idx}))
    return out


def z_(p):
    return NormalDist().inv_cdf(p)


def t_(p, nu):
    """Student t quantile, first-order Cornish-Fisher (t_0.8 at 9 df: 0.882, exact 0.883); no scipy here."""
    z = z_(p)
    return z + (z ** 3 + z) / (4 * nu)


def chi2_(p, k):
    """chi-square quantile by bisection on torch's regularized incomplete gamma."""
    lo, hi = 0.0, max(10.0, 10.0 * k)
    for _ in range(200):
        mid = (lo + hi) / 2
        if float(torch.special.gammainc(torch.tensor(k / 2, dtype=torch.float64), torch.tensor(mid / 2, dtype=torch.float64))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def n_normal(mu, sd, kappa, power):
    if mu >= kappa:
        return math.inf
    z = z_(1 - ALPHA) + z_(power)
    return max(5, math.ceil((z * sd / (kappa - mu)) ** 2))


def conservative(y):
    m = len(y)
    sd = y.std(ddof=1)
    sd_c = sd * math.sqrt((m - 1) / chi2_(0.20, m - 1))
    mu_c = y.mean() + t_(0.80, m - 1) * sd / math.sqrt(m)
    return mu_c, sd_c


def sim_power(y, n, kappa, sims=600, boots=400, seed=0):
    """Share of simulated confirmatory runs (n units drawn from R5's units) whose unit-bootstrap one-sided
    95% upper bound on the mean of y is below kappa."""
    rng = np.random.default_rng(seed)
    hits = 0
    for c in range(0, sims, 100):
        k = min(100, sims - c)
        draw = y[rng.integers(0, len(y), (k, n))]                          # [k, n]
        bi = rng.integers(0, n, (k, boots, n))
        bm = np.take_along_axis(np.repeat(draw[:, None, :], boots, 1), bi, 2).mean(-1)
        hits += int((np.quantile(bm, 1 - ALPHA, axis=1) < kappa).sum())
    return hits / sims


def n_sim(y, kappa, power):
    if y.mean() >= kappa:
        return math.inf
    for n in N_GRID:
        if sim_power(y, n, kappa) >= power:
            return n
    return math.inf


def acc_size(dacc, delta):
    """FP-correct units for the accuracy floor: one-sided 95% lower bound of the mean change (per unit, partial
    credit allowed) > -ACC_DELTA with 80% power, at R5's mean and variance; with no change seen in R5, one
    unit lost out of the units seen; never fewer than 3 / ACC_DELTA (no loss in n units bounds the loss rate
    below 3 / n)."""
    ACC_DELTA = delta
    m = len(dacc)
    mu = float(dacc.mean()) if m else 0.0
    var = float(dacc.var(ddof=1)) if m > 1 else 0.0
    if var == 0.0:
        mu, var = -1.0 / max(m, 1), 1.0 / max(m, 1)
    if mu <= -ACC_DELTA:
        return math.inf, var, mu
    z = z_(1 - ALPHA) + z_(0.8)
    return max(math.ceil(3 / ACC_DELTA), math.ceil(z * z * var / (mu + ACC_DELTA) ** 2)), var, mu


def fmt(n):
    return "∞" if not np.isfinite(n) else str(int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r53", nargs="*", default=[f"{t}:{j}" for t, j in R53_BLOCKS])
    ap.add_argument("--out-stem", default=os.path.join(R.FINDINGS, "R5_sizing"))
    a = ap.parse_args()
    r53 = [(x.split(":")[0], int(x.split(":")[1])) for x in a.r53]
    rows, secs = load(R5_BLOCKS + r53)
    fams = {m: sorted(g.family.dropna().unique()) for m, g in rows[rows.arm == "fp"].groupby("model")}
    res = dict(margins=MARGINS, alpha=ALPHA, power_model=POWER_MODEL, acc_deltas=ACC_DELTAS, r5=R5_BLOCKS, r53=r53,
               groups=[], totals=[], accuracy=[], cost=[])
    for arm, B in DESIGNS + REFS:
        for rho, kappa in MARGINS:
            P = paired(rows, arm, B, rho)
            for (m, f), p in sorted(P.items()):
                K = len(fams[m])
                pw = POWER_MODEL ** (1 / K)
                y = p["y"]
                mu, sd = float(y.mean()), float(y.std(ddof=1)) if len(y) > 1 else float("nan")
                mu_c, sd_c = conservative(y) if len(y) > 2 else (mu, sd)
                rec = dict(arm=arm, B=B, rho=rho, kappa=kappa, model=m, family=f, cells=p["cells"], m=len(y),
                           power_group=pw, kl_d=float(p["kd"].mean()), kl_fp8=float(p["k8"].mean()), mean_y=mu, sd_y=sd,
                           n_normal=n_normal(mu, sd, kappa, pw), n_normal_cons=n_normal(mu_c, sd_c, kappa, pw),
                           n_sim=n_sim(y, kappa, pw) if (rho, kappa) == MARGINS[0] and len(y) >= 5 else None)
                res["groups"].append(rec)
    # per model totals for each design at the primary margin; the plan's unit count per family = the max of
    # the point-estimate methods, rounded up to a multiple of 10 (the conservative count is reported beside it)
    prim = [g for g in res["groups"] if (g["rho"], g["kappa"]) == MARGINS[0]]
    s_unit = secs.groupby(["model", "family"]).s6.mean()
    for (arm, B, m), gs in pd.DataFrame(prim).groupby(["arm", "B", "model"]):
        plan = {r.family: max(r.n_normal, r.n_sim if r.n_sim is not None else 0) for r in gs.itertuples()}
        plan = {f: (math.inf if not np.isfinite(n) else int(math.ceil(n / 10) * 10)) for f, n in plan.items()}
        cons = {r.family: r.n_normal_cons for r in gs.itertuples()}
        tot = sum(plan.values())
        gpu_h = sum(plan[f] * s_unit.get((m, f), s_unit.xs(m).mean()) for f in plan if np.isfinite(plan[f])) / 3600
        res["totals"].append(dict(arm=arm, B=B, model=m, families=len(plan), units=tot, per_family=plan,
                                  conservative=cons, gpu_h_6arms=gpu_h,
                                  missing=[f for f in fams[m] if f not in plan]))
    P = paired(rows, DESIGNS[0][0], DESIGNS[0][1], 1.0)
    for m in fams:
        for pool, keep in (("all families", lambda f: True), ("without aggregation", lambda f: f != "aggregation")):
            d = np.concatenate([p["dacc"] for (mm, f), p in P.items() if mm == m and keep(f)])
            for delta in ACC_DELTAS:
                n, var, mu = acc_size(d, delta)
                res["accuracy"].append(dict(model=m, arm=DESIGNS[0][0], pool=pool, delta=delta, fp_correct_units=len(d),
                                            losses=int((d < 0).sum()), gains=int((d > 0).sum()), mean=mu,
                                            sd=math.sqrt(var), n_fp_correct=n))
    for (m, f), s in s_unit.items():
        res["cost"].append(dict(model=m, family=f, s_per_unit_6arms=float(s)))
    _write(res, a.out_stem)


def _write(res, stem):
    with open(stem + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    rho0, k0 = MARGINS[0]
    L = ["# R5 → the confirmatory campaign's size (draft input to plan.md amendment \"R6 (draft)\")", "",
         f"`size_confirm_r5.py`; R5's valid blocks and R5.3's finished ones ({', '.join(f'{t}_{j}' for t, j in res['r53'])}). "
         f"Claim per model × family: mean KL/token(design) < {rho0} × mean KL/token(FP8) + {k0} (one-sided "
         f"α = {ALPHA}, unit bootstrap upper bound, paired per unit); all families of a model must pass "
         f"(intersection–union: no multiplicity correction, so each family is powered at {POWER_MODEL}^(1/K)).",
         "Units are prompt × task units of the R5 cells (R5 used 5–20 per group). n_sim draws units from R5's own "
         "per-unit differences, so a group with few or tied units gives a coarse answer; n_cons uses the SD at "
         "its 80% upper confidence limit and the mean at its 80% upper bound. ∞ = R5's mean already misses the "
         "margin (non-inferiority not expected to hold there).", ""]
    for arm, B in DESIGNS + REFS:
        gs = [g for g in res["groups"] if g["arm"] == arm and g["B"] == B and (g["rho"], g["kappa"]) == MARGINS[0]]
        if not gs:
            continue
        L += [f"## {arm}@{B:g} (margin {rho0} × FP8 + {k0})", "",
              "| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in gs:
            L.append(f"| {g['model']} | {g['family']} | {g['m']} | {g['kl_d']:.4f} | {g['kl_fp8']:.4f} | {g['mean_y']:+.4f} | "
                     f"{g['sd_y']:.4f} | {g['power_group']:.3f} | {fmt(g['n_normal'])} | {fmt(g['n_normal_cons'])} | "
                     f"{fmt(g['n_sim']) if g['n_sim'] is not None else '—'} |")
        L.append("")
        for t in [t for t in res["totals"] if t["arm"] == arm and t["B"] == B]:
            L.append(f"- **{t['model']}**: planned units {fmt(t['units'])} over {t['families']} families "
                     f"({', '.join(f'{f} {fmt(n)}' for f, n in t['per_family'].items())}); 6-arm GPU time "
                     f"≈ {t['gpu_h_6arms']:.1f} h + model loads"
                     + (f"; no R5.3 data yet for {', '.join(t['missing'])}" if t["missing"] else ""))
        L.append("")
    L += ["## Sensitivity to the margin (normal approximation, point estimates; units per family)", "",
          "| design | model | family | " + " | ".join(f"{r} × FP8 + {k}" for r, k in MARGINS) + " |",
          "|---|---|---|" + "---|" * len(MARGINS)]
    keyed = {}
    for g in res["groups"]:
        keyed.setdefault((g["arm"], g["B"], g["model"], g["family"]), {})[(g["rho"], g["kappa"])] = g["n_normal"]
    for (arm, B, m, f), d in keyed.items():
        L.append(f"| {arm}@{B:g} | {m} | {f} | " + " | ".join(fmt(d.get(mg, math.inf)) for mg in MARGINS) + " |")
    L += ["", "## Accuracy floor (design − FP8 on FP-correct units, pooled per model; lower bound > −δ, 80% power)", "",
          "| model | design | pool | δ | FP-correct units in R5 | units lower / higher | mean Δ | SD Δ | FP-correct units needed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in res["accuracy"]:
        L.append(f"| {a['model']} | {a['arm']} | {a['pool']} | {a['delta']} | {a['fp_correct_units']} | {a['losses']} / "
                 f"{a['gains']} | {a['mean']:+.4f} | {a['sd']:.4f} | {fmt(a['n_fp_correct'])} |")
    L += ["", "## Seconds per unit for six arms (FP, FP8, the design, dense 4/4, KIVI-4, fp_noise; prefill included)", "",
          "| model | family | s / unit |", "|---|---|---|"]
    L += [f"| {c['model']} | {c['family']} | {c['s_per_unit_6arms']:.0f} |" for c in res["cost"]]
    with open(stem + ".md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {stem}.md / .json")


if __name__ == "__main__":
    main()
