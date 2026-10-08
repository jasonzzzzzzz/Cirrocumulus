#!/usr/bin/env python3
"""R14 Stage 1h R5 readers (plan.md amendment "R5" and "R5 freeze"). EXPLORATORY: no labels.

    python read_stage1h_r5.py --pilot TAG:JOB [TAG:JOB ...] [--out-stem findings/R5_pilot]
    python read_stage1h_r5.py --r5 TAG:JOB [TAG:JOB ...]    [--out-stem findings/R5_reader]

--r5 is the frozen R5 read (its rules are in plan.md, "R5 freeze", and in the functions below
read_r5); blocks failing V are excluded and reported in an appendix; a POST-HOC section, marked as
such, follows. --pilot is the pilot / validity reader described here:

Per pilot job (results/r14s1h_<TAG>_<JOB>/):
  VALIDITY (hypothesis V; any failure is a bug, and the reader exits 1):
    every planned arm ran on every unit; the probe's pass is FP (|kl_all| < 1e-4);
    Lemma 1's and Lemma 3's bounds, the certificate (eps_bar >= eps) and the score bound
    (|s_hat - s| <= b) hold on every measured head and step; the worst-case controller trace
    never leaves its target (viol = 0); injected missed mass never exceeds its target and reaches
    it somewhere; the tail arm evicts nothing; on a GPU, peak memory <= 76 GiB per device.
  COST: peak GPU memory per device; seconds per unit (prefill + precompute + every arm's run,
    replay and A2 own-order replays) and by arm; the projected wall time of each planned R5
    cell (CELLS_R5) from the slowest pilot unit of its model and suite.
  FIRST LOOK (no labels): KL against the injected missed mass (log-log slope; the four layer
    quarters' KL against all layers'), certificate tightness and controller cost, the tail arm and
    Quest against the system (dP per token).
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS = os.path.join(ROOT, "h0_measurement", "results")
FINDINGS = os.path.join(HERE, "findings")
SETUP_H = 0.25
PEAK_LIMIT = 76.0
CELLS_R5 = [   # cell, preset, suite, units  (plan.md amendment "R5")
    ("Llama 128K RULER", "h5llama128", "r3", 40),
    ("Llama 128K harder RULER", "h5llama128", "r3", 40),
    ("Llama 128K cwe/fwe", "h5llama128", "r3b", 20),
    ("Llama 128K HELMET", "h5llama128", "r4", 25),
    ("Llama 128K LongBench v2", "h5llama128", "r4", 20),
    ("Qwen 32K RULER + harder", "h5qwen32", "r3", 80),
    ("Qwen 32K cwe/fwe", "h5qwen32", "r3b", 20),
]


def load(tag, job, root=RESULTS):
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    side_p = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(side_p) != 1:
        raise SystemExit(f"{d}: expected one sidecar, found {side_p}")
    side = json.load(open(side_p[0]))
    out = dict(side=side, main=pd.read_parquet(os.path.join(d, side["parquet"])))
    for k, f in (side.get("r5_files") or {}).items():
        out[k] = pd.read_parquet(os.path.join(d, f))
    return out


def validity(x, problems, w):
    d, side = x["main"], x["side"]
    plan = [tuple(p) for p in side["plan"]]
    units = d[d.arm == "fp"][["prompt_idx", "task"]].drop_duplicates()
    if len(d) != len(units) * len(plan):
        problems.append(f"{w}: {len(d)} rows for {len(units)} units x {len(plan)} arms")
    pr = d[d.arm == "probe"]
    if not len(pr) or not (pr.kl_all.abs() < 1e-4).all():
        problems.append(f"{w}: the probe's pass is not FP (kl_all {pr.kl_all.tolist()})")
    h = x.get("probe_heads")
    if h is None or not len(h):
        problems.append(f"{w}: no probe records")
    else:
        for name, ok in (("Lemma 1", h.err_sys <= h.l1_bound * (1 + 1e-4) + 1e-6),
                         ("Lemma 3", h.err_tail <= h.l3_bound * (1 + 1e-4) + 1e-6),
                         ("certificate", h.epsbar_vote1 >= h.eps_vote1 - 1e-6),
                         ("score bound", h.serr_max <= h.b_max + 1e-4)):
            if not bool(ok.all()):
                problems.append(f"{w}: {name} fails on {int((~ok).sum())} of {len(ok)} head-steps")
    t = x.get("trace")
    if t is None or not len(t):
        problems.append(f"{w}: no controller traces")
    elif float(t[t.bound == "w"].viol.sum()) > 0:
        problems.append(f"{w}: the worst-case controller left its target on {int(t[t.bound == 'w'].viol.sum())} steps")
    ij = x.get("inject")
    if ij is not None and len(ij):
        # never above the target; the target reached by the worst head somewhere (one row more can drop the
        # missed mass well below eps when attention is concentrated, so equality is not required everywhere)
        over = ij[ij.eps_real > ij.eps + 1e-4]
        if len(over):
            problems.append(f"{w}: injected missed mass above its target on {len(over)} head-steps")
        top = ij.groupby(["arm", "eps"]).eps_real.max().reset_index()
        miss = top[(top.eps - top.eps_real) > 1e-3]
        if len(miss):
            problems.append(f"{w}: injection never reached its target: {miss.to_dict('records')}")
    tl = d[d.arm == "qread2t4kqT_v4"]
    if len(tl) and not tl.evict_frac.eq(0).all():
        problems.append(f"{w}: the tail arm evicted rows")


def unit_seconds(d):
    """Seconds per unit: prefill + precompute (once) + every row's arm, replay and own-order replays."""
    tt = d.t_arm.fillna(0) + d.t_tf.fillna(0) + (d.t_own.fillna(0) if "t_own" in d else 0)
    per = d.assign(_t=tt).groupby(["prompt_idx", "task"]).agg(t=("_t", "sum"), pre=("t_prefill", "first"),
                                                              pc=("t_precompute", "first"))
    return (per.t + per.pre + per.pc.fillna(0)).rename("seconds")


def first_look(x):
    d = x["main"]
    out = {}
    fp = d[d.arm == "fp"].set_index(["prompt_idx", "task"])
    inj = d[d.arm == "inj_top"]
    if len(inj):
        slopes = []
        for k, g in inj.groupby(["prompt_idx", "task"]):
            g = g[(g.kl_all > 0) & g.B.gt(0)]
            if len(g) >= 3:
                slopes.append(float(np.polyfit(np.log(g.B.astype(float)), np.log(g.kl_all), 1)[0]))
        out["kl_vs_eps_loglog_slope"] = dict(median=float(np.median(slopes)) if slopes else None, n=len(slopes))
        out["kl_all_by_eps"] = {str(b): round(float(v), 5) for b, v in inj.groupby("B").kl_all.mean().items()}
    q = d[d.arm.str.startswith("inj_top_l")]
    a01 = inj[inj.B == 0.1]
    if len(q) and len(a01):
        out["quarters_kl_sum_over_all"] = round(float(q.kl_all.sum() / max(a01.kl_all.sum(), 1e-12)), 3)
        out["quarters_kl"] = {a: round(float(v), 5) for a, v in q.groupby("arm").kl_all.mean().items()}
    rnd = d[(d.arm == "inj_rnd")]
    if len(rnd):
        out["rnd_over_top_kl"] = {str(b): round(float(g.kl_all.mean() / max(inj[inj.B == b].kl_all.mean(), 1e-12)), 3)
                                  for b, g in rnd.groupby("B")}
    pr = d[d.arm == "probe"]
    out["probe"] = {k: round(float(pr[k].mean()), 4) for k in pr.columns if str(k).startswith("probe_")
                    and pr[k].dtype.kind in "fi"}
    cmp = {}
    for arm in ("qread2t4kq_v4", "qread2t4kqT_v4", "quest_v16", "quest4_v4", "qoraclefp_v16", "uniform+v4",
                "fp8kv"):
        g = d[d.arm == arm]
        for B, gb in g.groupby("B"):
            gb = gb.set_index(["prompt_idx", "task"])
            idx = gb.index.intersection(fp.index)
            dp = (gb.a_span_nll.reindex(idx) - fp.a_span_nll.reindex(idx)) / gb.a_span_ntok.reindex(idx)
            cmp[f"{arm}@{B}"] = dict(dP_tok=round(float(dp.mean()), 4), kl_span_tok=round(float(gb.kl_span_tok.mean()), 4),
                                    acc=round(float(gb.score.mean()), 3), traffic=round(float(gb.traffic_frac.mean()), 4))
    out["arms"] = cmp
    return out


def read_pilot(pairs, out_stem, root=RESULTS):
    problems, res, secs = [], {}, {}
    for tag, job in pairs:
        w = f"{tag}_{job}"
        try:
            x = load(tag, job, root)
        except (SystemExit, FileNotFoundError) as e:
            problems.append(f"{w}: {e}")
            continue
        side, d = x["side"], x["main"]
        validity(x, problems, w)
        us = unit_seconds(d)
        tt = d.t_arm.fillna(0) + d.t_tf.fillna(0) + (d.t_own.fillna(0) if "t_own" in d else 0)
        by_arm = d.assign(_t=tt).groupby("arm")._t.sum().sort_values(ascending=False)
        peaks = [float(v) for v in side.get("peak_gib_dev_max") or []]
        if int(side.get("n_gpus") or 0) > 0 and (not peaks or max(peaks) > PEAK_LIMIT):
            problems.append(f"{w}: peak GPU memory per device {peaks} (limit {PEAK_LIMIT})")
        key = (side["preset_name"], side.get("suite"))
        secs[key] = max(secs.get(key, 0.0), float(us.max()))
        res[w] = dict(preset=side["preset_name"], suite=side.get("suite"), task_cfg=side.get("task_cfg_r5"),
                      units={f"{p}:{t}": round(float(s), 1) for (p, t), s in us.items()}, peak_gib_dev=peaks,
                      seconds_by_arm={a: round(float(v), 1) for a, v in by_arm.items()},
                      first_look=first_look(x))
    proj = {}
    for cell, preset, suite, n in CELLS_R5:
        s = secs.get((preset, suite))
        if s is None:   # same model, another suite: the slowest unit of that model
            s = max((v for (p, _), v in secs.items() if p == preset), default=None)
        proj[cell] = None if s is None else dict(units=n, seconds_per_unit=round(s, 1),
                                                 gpu_hours=round(n * s / 3600 + SETUP_H, 2))
    out = dict(verdict="PASS" if not problems else "FAIL", problems=problems, jobs=res, projection=proj,
               total_gpu_hours=round(sum(v["gpu_hours"] for v in proj.values() if v), 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    md = [f"# R5 pilot — {out['verdict']}", "", "EXPLORATORY; validity is hypothesis V (plan.md amendment R5).", ""]
    md += [f"- problem: {p}" for p in problems] or ["- validity: every check passed"]
    for w, r in res.items():
        md += ["", f"## {w} ({r['preset']}, suite {r['suite']}, cfg {r['task_cfg']})", "",
               f"- units (s): {r['units']}; peak GiB per device {r['peak_gib_dev']}",
               f"- slowest arms (s): {dict(list(r['seconds_by_arm'].items())[:8])}",
               f"- first look: {json.dumps(r['first_look'], default=float)[:3000]}"]
    md += ["", "## Projected cells", "", "| cell | units | s/unit | GPU-h |", "|---|---:|---:|---:|"]
    for c, v in proj.items():
        md.append(f"| {c} | {v['units'] if v else '—'} | {v['seconds_per_unit'] if v else '—'} | "
                  f"{v['gpu_hours'] if v else '—'} |")
    md.append(f"\nTotal ≈ {out['total_gpu_hours']} GPU-h")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if not problems else 1


# ======================================================================= the R5 read
# The frozen read rules (plan.md, "R5 freeze"): exploratory, no labels; unit-clustered bootstrap
# 90% intervals; each model x task family (and cell) separately. Blocks failing V are excluded
# and reported in an appendix. A POST-HOC section (not frozen) follows, marked as such.
FAMILY = {"niah_single": "retrieval", "niah_multikey": "retrieval", "mk_panel": "retrieval",
          "niah_multivalue": "multivalue", "vt": "tracking", "cwe": "aggregation", "fwe": "aggregation",
          "kilt_nq": "rag", "kilt_hotpotqa": "rag", "msmarco_rerank_psg": "rerank", "icl_trec_coarse": "icl",
          "icl_banking77": "icl", "lbv2": "lbv2"}
INJ_LEVELS = (0.003, 0.01, 0.03, 0.1, 0.3)
SYS = "qread2t4kq_v4"
TAIL = "qread2t4kqT_v4"
H4_ARMS = (TAIL, "qoraclefp_v16", "quest_v16", "quest4_v4", "uniform+v4", "fp8kv")
N_BOOT = 2000
UKEY = ["prompt_idx", "task"]


def boot(v, fn=np.median, n=N_BOOT, seed=0):
    """(statistic, 5th, 95th percentile, n) over units resampled with replacement."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [float("nan")] * 3 + [0]
    bs = fn(v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))], axis=1)
    return [float(fn(v)), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)), int(len(v))]


def _ci(b, nd=4):
    return "—" if not b[3] else f"{b[0]:.{nd}f} [{b[1]:.{nd}f}, {b[2]:.{nd}f}] (n={b[3]})"


def _load_block(tag, job, root):
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    side = json.load(open(glob.glob(os.path.join(d, "s1h_evaluate_*.json"))[0]))
    f = side.get("r5_files") or {}
    rd = lambda k, cols=None: pd.read_parquet(os.path.join(d, f[k]), columns=cols) if k in f else None  # noqa: E731
    x = dict(side=side, main=pd.read_parquet(os.path.join(d, side["parquet"])),
             probe_heads=rd("probe_heads", ["err_sys", "l1_bound", "err_tail", "l3_bound", "epsbar_vote1", "eps_vote1",
                                            "serr_max", "b_max", "err_fp8", "rel_fp8", "rel_tail", "rel_sys",
                                            "prompt_idx", "task"]),
             trace=rd("trace"), groups=rd("probe_groups"),
             inject=rd("inject", ["eps_real", "rel_inj", "prompt_idx", "task", "arm", "eps", "layer", "step"]))
    return x


def _block_units(x, tag):
    """Per-unit records of one block: arm rows, H1 inputs, budgets, traces, post-hoc head stats."""
    side, d = x["side"], x["main"]
    model = side["model"]
    d = d.assign(model=model, cell=tag, family=d.task.map(FAMILY))
    out = dict(rows=d[["model", "cell", "family", "prompt_idx", "task", "arm", "B", "a_span_nll_tok", "kl_span_tok",
                       "kl_all", "score", "traffic_frac", "exact_rows_frac"]].copy())
    ij = x["inject"]
    if ij is not None and len(ij):
        out["inj"] = ij.groupby(UKEY + ["arm", "eps"]).agg(eps_mean=("eps_real", "mean"),
                                                            rel_mean=("rel_inj", "mean")).reset_index()
    g = x["groups"]
    if g is not None and len(g):
        rec = {}
        for e in (0.01, 0.1):
            u = g[f"union_{e}"].clip(lower=1)
            for k in ("cert", "cert_hp", "page"):
                rec[f"{k}_over_union_{e}"] = g[f"{k}_{e}"] / u
            rec[f"union_frac_{e}"] = g[f"union_{e}"] / g.ctx
        gg = pd.DataFrame(rec).assign(prompt_idx=g.prompt_idx.values, task=g.task.values)
        out["groups"] = gg.groupby(UKEY).median().reset_index()
        out["group_records"] = g[["union_0.1", "cert_0.1", "cert_hp_0.1", "union_0.01", "cert_0.01", "cert_hp_0.01"]].assign(
            model=model)
    t = x["trace"]
    if t is not None and len(t):
        st = t.steps.clip(lower=1)
        tt = t.assign(rows=t.F_sum / st / t.ctx, refetch=t.fetch_ev / st, cap=t.cap_steps / st, viol=t.viol / st)
        tt["traffic"] = 4.0 / 32 + tt.rows * 20.0 / 32
        out["trace"] = tt.groupby(UKEY + ["bound", "eps"])[["rows", "traffic", "refetch", "cap", "viol"]].mean().reset_index()
    h = x["probe_heads"]
    if h is not None and len(h):
        out["heads"] = h.assign(l3_within_fp8=(h.l3_bound <= h.err_fp8), tail_within_fp8=(h.err_tail <= h.err_fp8),
                                sys_within_fp8=(h.err_sys <= h.err_fp8)).groupby(UKEY)[
            ["l3_within_fp8", "tail_within_fp8", "sys_within_fp8", "rel_tail", "rel_fp8", "rel_sys"]].mean().reset_index()
    for k in ("inj", "groups", "trace", "heads"):
        if k in out:
            out[k] = out[k].assign(model=model, cell=tag, family=out[k].task.map(FAMILY))
    return out


def _slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")


def h1_units(rows):
    """Per unit: the frozen slope (log kl_all on log eps over inj_top, >= 3 levels with KL > 0), the
    quarters' sum over all layers at 0.1, rnd/top at 0.01 and 0.1; post hoc: the same corrected for the
    floor f = KL at eps 0.003 (slope over eps >= 0.03 where KL - f > 0; quarters (sum (KL_q - f)) / (KL - f))."""
    recs = []
    for k, g in rows[rows.arm.str.startswith("inj")].groupby(["model", "cell", "family"] + UKEY):
        top = g[g.arm == "inj_top"].set_index("B").kl_all
        r = dict(zip(["model", "cell", "family"] + UKEY, k))
        ok = top[(top > 0) & top.index.isin(INJ_LEVELS)]
        r["slope"] = _slope(ok.index, ok.values) if len(ok) >= 3 else float("nan")
        f = top.get(0.003, float("nan"))
        ex = (top - f)[(top.index >= 0.03)]
        ex = ex[ex > 0]
        r["slope_floor"] = _slope(ex.index, ex.values) if len(ex) >= 2 else float("nan")
        q = g[g.arm.str.startswith("inj_top_l")]
        a01 = top.get(0.1, float("nan"))
        if len(q) == 4 and a01 > 0:
            r["quarters"] = float(q.kl_all.sum() / a01)
            r["quarters_floor"] = float((q.kl_all - f).sum() / (a01 - f)) if a01 - f > 0 else float("nan")
        rnd = g[g.arm == "inj_rnd"].set_index("B").kl_all
        for e in (0.01, 0.1):
            if e in rnd.index and top.get(e, 0) > 0:
                r[f"rnd_over_top_{e}"] = float(rnd[e] / top[e])
        r["kl_floor"] = float(f)
        recs.append(r)
    return pd.DataFrame(recs)


def link_r2(rows, inj):
    """Per model x family: R^2 of log kl_all on log mean realized eps, and on log mean rel_inj, over
    (unit, injected arm, eps) points with KL > 0."""
    r = rows[rows.arm.isin(["inj_top", "inj_rnd"])].rename(columns={"B": "eps"})
    m = r.merge(inj, on=["model", "cell", "family"] + UKEY + ["arm", "eps"])
    m = m[(m.kl_all > 0) & (m.eps_mean > 0) & (m.rel_mean > 0)]
    out = {}
    for (model, fam), g in m.groupby(["model", "family"]):
        y = np.log(g.kl_all)
        out[f"{model}/{fam}"] = dict(n=int(len(g)), r2_eps=float(np.corrcoef(np.log(g.eps_mean), y)[0, 1] ** 2),
                                     r2_rel=float(np.corrcoef(np.log(g.rel_mean), y)[0, 1] ** 2))
    return out


def arm_table(rows):
    """Per model x cell x family x arm: dP/token and KL/token against FP, accuracy change (FP-correct and
    FP-failed units separately), traffic and exact-row shares; each with its unit bootstrap."""
    fp = rows[rows.arm == "fp"].set_index(["model", "cell"] + UKEY)
    out = []
    for (model, cell, fam, arm, B), g in rows[~rows.arm.isin(["fp", "probe"]) &
                                              ~rows.arm.str.startswith("inj")].groupby(
            ["model", "cell", "family", "arm", "B"]):
        g = g.set_index(["model", "cell"] + UKEY)
        f = fp.reindex(g.index)
        dp = g.a_span_nll_tok - f.a_span_nll_tok
        dacc = g.score - f.score
        corr = (f.score >= 1 - 1e-9)
        out.append(dict(model=model, cell=cell, family=fam, arm=arm, B=float(B), dP_tok=boot(dp, np.mean),
                        KL_tok=boot(g.kl_span_tok, np.mean), dAcc_fp_correct=boot(dacc[corr], np.mean),
                        dAcc_fp_failed=boot(dacc[~corr], np.mean), traffic=float(g.traffic_frac.mean()),
                        exact_rows=float(g.exact_rows_frac.mean())))
    return out


def h4_table(rows):
    """Paired per unit, against the system at 1/8 (the tail arm also at the floor against the system at the
    floor): differences of dP/token, KL/token and accuracy."""
    out = []
    for (model, cell, fam), g in rows.groupby(["model", "cell", "family"]):
        piv = {k: v.set_index(UKEY) for k, v in g.groupby(["arm", "B"])}
        pairs = [((a, b), (SYS, 0.125)) for (a, b) in piv if a in H4_ARMS and a != TAIL]
        pairs += [((TAIL, b), (SYS, b)) for (a, b) in piv if a == TAIL and (SYS, b) in piv]
        for (a, b), ref in pairs:
            if ref not in piv:
                continue
            x, y = piv[(a, b)], piv[ref]
            idx = x.index.intersection(y.index)
            out.append(dict(model=model, cell=cell, family=fam, arm=f"{a}@{b}", against=f"{ref[0]}@{ref[1]}",
                            d_dP_tok=boot(x.a_span_nll_tok[idx] - y.a_span_nll_tok[idx], np.mean),
                            d_KL_tok=boot(x.kl_span_tok[idx] - y.kl_span_tok[idx], np.mean),
                            d_acc=boot(x.score[idx] - y.score[idx], np.mean),
                            traffic=float(x.traffic_frac.mean()), traffic_ref=float(y.traffic_frac.mean())))
    return out


def _analyse(units):
    rows = pd.concat([u["rows"] for u in units], ignore_index=True)
    inj, grp, trc, hds = (pd.concat([u[k] for u in units if k in u], ignore_index=True) if any(k in u for u in units)
                          else pd.DataFrame() for k in ("inj", "groups", "trace", "heads"))
    res = dict(units=int(rows[rows.arm == "fp"].shape[0]))
    h1 = h1_units(rows)
    res["H1"] = {f"{m}/{c}/{f}": dict(slope=boot(g.slope), quarters=boot(g.get("quarters", pd.Series(dtype=float))),
                                      rnd_over_top_0_01=boot(g.get("rnd_over_top_0.01", pd.Series(dtype=float))),
                                      rnd_over_top_0_1=boot(g.get("rnd_over_top_0.1", pd.Series(dtype=float))))
                 for (m, c, f), g in h1.groupby(["model", "cell", "family"])}
    res["H1_link"] = link_r2(rows, inj) if len(inj) else {}
    res["H2"] = {}
    for (m, c, f), g in grp.groupby(["model", "cell", "family"]):
        res["H2"][f"{m}/{c}/{f}"] = {k: boot(g[k]) for k in g.columns if "_over_union_" in k or k.startswith("union_frac")}
    res["H2_controller"] = {}
    for (m, c, f, b, e), g in trc.groupby(["model", "cell", "family", "bound", "eps"]):
        res["H2_controller"][f"{m}/{c}/{f}/{b}@{e}"] = {k: boot(g[k], np.mean) for k in ("rows", "traffic", "refetch", "cap",
                                                                                         "viol")}
    res["H3"] = {}
    gr = pd.concat([u["group_records"] for u in units if "group_records" in u], ignore_index=True) if units else None
    if gr is not None and len(gr):
        for m, g in gr.groupby("model"):
            res["H3"][m] = {f"spearman_union_cert_{e}": float(g[f"union_{e}"].rank().corr(g[f"cert_{e}"].rank()))
                            for e in (0.01, 0.1)}
            res["H3"][m].update({f"spearman_union_certhp_{e}": float(g[f"union_{e}"].rank().corr(g[f"cert_hp_{e}"].rank()))
                                 for e in (0.01, 0.1)})
        if len(trc) and len(grp):
            u = grp.merge(trc[(trc.bound == "hp") & (trc.eps == 0.1)], on=["model", "cell", "family"] + UKEY)
            for m, g in u.groupby("model"):
                res["H3"][m]["spearman_unit_union_frac_vs_ctl_rows"] = float(g["union_frac_0.1"].rank().corr(g.rows.rank()))
    res["H4"] = h4_table(rows)
    res["arms"] = arm_table(rows)
    # ---------------------------------------------------------------- POST HOC (not frozen)
    res["post_hoc"] = dict(
        note="not frozen: added after the blocks finished (plan.md 'R5 read')",
        H1_floor={f"{m}/{c}/{f}": dict(kl_floor=boot(g.kl_floor), slope_floor=boot(g.slope_floor),
                                       quarters_floor=boot(g.get("quarters_floor", pd.Series(dtype=float))))
                  for (m, c, f), g in h1.groupby(["model", "cell", "family"])},
        tail_vs_fp8={f"{m}/{c}/{f}": {k: boot(g[k], np.mean) for k in ("l3_within_fp8", "tail_within_fp8",
                                                                     "sys_within_fp8", "rel_tail", "rel_fp8")}
                     for (m, c, f), g in hds.groupby(["model", "cell", "family"])} if len(hds) else {})
    return res


def _md(res, title):
    L = [f"## {title} ({res['units']} units)", ""]
    L += ["### H1 — KL against the injected missed mass (frozen)", "",
          "| model / cell / family | slope (log KL on log eps) | quarters' KL / all-layer KL | rnd/top @0.01 | rnd/top @0.1 |",
          "|---|---|---|---|---|"]
    for k, v in res["H1"].items():
        L.append(f"| {k} | {_ci(v['slope'], 2)} | {_ci(v['quarters'], 2)} | {_ci(v['rnd_over_top_0_01'], 2)} | "
                 f"{_ci(v['rnd_over_top_0_1'], 2)} |")
    L += ["", "Link (R² of log KL on log mean realized eps / on log mean relative output error):", ""]
    L += [f"- {k}: n={v['n']}, R²(eps) {v['r2_eps']:.2f}, R²(rel error) {v['r2_rel']:.2f}" for k, v in res["H1_link"].items()]
    L += ["", "### H2 — certificate tightness (median budget / shared-set oracle budget, frozen)", "",
          "| model / cell / family | cert@0.01 | cert_hp@0.01 | page@0.01 | cert@0.1 | cert_hp@0.1 | page@0.1 | oracle rows@0.1 |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in res["H2"].items():
        L.append(f"| {k} | " + " | ".join(_ci(v[c], 1) for c in ("cert_over_union_0.01", "cert_hp_over_union_0.01",
                                                                  "page_over_union_0.01", "cert_over_union_0.1",
                                                                  "cert_hp_over_union_0.1", "page_over_union_0.1")) +
                 f" | {_ci(v['union_frac_0.1'], 3)} |")
    L += ["", "Controller emulated on FP's path (exact-row share, traffic of FP16 K+V, re-fetch events per step, "
              "steps above the 25% cap, steps above eps):", "",
          "| model / cell / family / bound@eps | exact rows | traffic | re-fetch | cap | viol |", "|---|---|---|---|---|---|"]
    for k, v in res["H2_controller"].items():
        L.append(f"| {k} | " + " | ".join(_ci(v[c], 3) for c in ("rows", "traffic", "refetch", "cap", "viol")) + " |")
    L += ["", "### H3 — does concentration forecast cost? (Spearman ρ)", ""]
    L += [f"- {m}: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items()) for m, v in res["H3"].items()]
    L += ["", "### H4 — against the system (paired per unit; arm − system, per span token)", "",
          "| model / cell / family | arm | against | Δ dP/tok | Δ KL/tok | Δ accuracy | traffic (arm / ref) |",
          "|---|---|---|---|---|---|---|"]
    for r in res["H4"]:
        L.append(f"| {r['model']}/{r['cell']}/{r['family']} | {r['arm']} | {r['against']} | {_ci(r['d_dP_tok'])} | "
                 f"{_ci(r['d_KL_tok'])} | {_ci(r['d_acc'], 3)} | {r['traffic']:.3f} / {r['traffic_ref']:.3f} |")
    L += ["", "### Accuracy and cost per arm (against FP, per span token)", "",
          "| model / cell / family | arm@B | dP/tok | KL/tok | ΔAcc (FP right) | ΔAcc (FP failed) | traffic | exact rows |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["arms"]:
        L.append(f"| {r['model']}/{r['cell']}/{r['family']} | {r['arm']}@{r['B']:g} | {_ci(r['dP_tok'])} | "
                 f"{_ci(r['KL_tok'])} | {_ci(r['dAcc_fp_correct'], 3)} | {_ci(r['dAcc_fp_failed'], 3)} | "
                 f"{r['traffic']:.3f} | {r['exact_rows']:.3f} |")
    ph = res["post_hoc"]
    L += ["", "### POST HOC (not frozen) — H1 corrected for the bf16 floor; the tail design against FP8", "",
          "| model / cell / family | KL floor (eps 0.003) | slope over eps ≥ 0.03, floor removed | quarters, floor removed |",
          "|---|---|---|---|"]
    for k, v in ph["H1_floor"].items():
        L.append(f"| {k} | {_ci(v['kl_floor'])} | {_ci(v['slope_floor'], 2)} | {_ci(v['quarters_floor'], 2)} |")
    L += ["", "| model / cell / family | Lemma 3 bound ≤ FP8 error | tail error ≤ FP8 error | system error ≤ FP8 error | "
              "rel. error tail | rel. error FP8 |", "|---|---|---|---|---|---|"]
    for k, v in ph["tail_vs_fp8"].items():
        L.append(f"| {k} | " + " | ".join(_ci(v[c], 3) for c in ("l3_within_fp8", "tail_within_fp8", "sys_within_fp8",
                                                                  "rel_tail", "rel_fp8")) + " |")
    return L


def read_r5(pairs, out_stem, root=RESULTS):
    problems, valid, excluded = {}, [], []
    for tag, job in pairs:
        w = f"{tag}_{job}"
        try:
            x = _load_block(tag, job, root)
        except (IndexError, FileNotFoundError, KeyError) as e:
            problems[w] = [f"cannot load: {e}"]
            continue
        pr = []
        validity(x, pr, w)
        u = _block_units(x, tag)
        del x
        (excluded if pr else valid).append(u)
        if pr:
            problems[w] = pr
        print(f"{w}: {'INVALID' if pr else 'valid'}", flush=True)
    out = dict(rules="plan.md 'R5 freeze' (exploratory, no labels)", invalid_blocks=problems)
    md = ["# R14 Stage 1h — R5 read (rules frozen in plan.md, 'R5 freeze'; exploratory, no labels)", "",
          "Unit-clustered bootstrap 90% intervals; each model × cell × task family separately.", ""]
    md += [f"- INVALID (excluded, appendix): {w}: {'; '.join(p)}" for w, p in problems.items()] or ["- every block valid"]
    if valid:
        out["main"] = _analyse(valid)
        md += [""] + _md(out["main"], "Main read: valid blocks")
    if excluded:
        out["appendix"] = _analyse(excluded)
        md += ["", "---", ""] + _md(out["appendix"], "APPENDIX: blocks excluded by V (reported, not read)")
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:8]))
    print(f"wrote {out_stem}.md / .json")
    return 0


# ===================================================================== the R5.3 read
# Rules: plan.md, amendment "R5.3" (frozen before any R5.3 cell's output). Exploratory, no labels;
# unit-clustered bootstrap 90% intervals; each model x cell x task family separately.
R53_REF = ("tail4_v4", 0.125)
R53_Q = {   # question: [(arm, B, against arm, against B)]
    "Q1 exact values on the selected rows": [("tail4x_v4", 0.125, "tail4_v4", 0.125)],
    "Q2 tier precision (against FP8)": [(a, 0.125, "fp8kv", 8.0) for a in ("tail4_v4", "tail4x_v4", "tail3x_v3",
                                                                          "tail2x_v4", "tail2x_v2", "tail2_v2")],
    "Q2b tail against dense at its tier": [("tail4x_v4", 0.125, "uniform+v4", 4.0), ("tail2x_v2", 0.125, "uniform+v2", 2.0)],
    "Q3 the read fraction (2-bit tail)": [("tail2x_v2", 0.0625, "tail2x_v2", 0.125), ("tail2x_v2", 0.25, "tail2x_v2", 0.125)],
    "Q4 selection when nothing is evicted": [("tail2xo_v2", 0.125, "tail2x_v2", 0.125)],
    "Q5 against eviction": [("tail2x_v2", 0.125, "qread2t4kq_v4", 0.125), ("tail2x_v2", 0.125, "qoraclefp_v16", 0.125)],
}


def validity_r53(d, side, problems, w):
    plan = [tuple(p) for p in side["plan"]]
    units = d[d.arm == "fp"][UKEY].drop_duplicates()
    if len(d) != len(units) * len(plan):
        problems.append(f"{w}: {len(d)} rows for {len(units)} units x {len(plan)} arms")
    t5 = d[d.family == "tail5"]
    if len(t5) and not t5.evict_frac.eq(0).all():
        problems.append(f"{w}: a tail arm evicted rows")
    a = d[d.arm == "tail4_v4"].set_index(UKEY).tf_logp
    b = d[d.arm == "qread2t4kqT_v4"].set_index(UKEY).tf_logp
    bad = [k for k in a.index.intersection(b.index)
           if a[k] is not None and b[k] is not None and len(a[k]) == len(b[k])
           and float(np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)), initial=0.0)) > 1e-6]
    if bad:
        problems.append(f"{w}: tail4_v4 differs from qread2t4kqT_v4 on {len(bad)} units (one design, two implementations)")
    peaks = [float(v) for v in side.get("peak_gib_dev_max") or []]
    if int(side.get("n_gpus") or 0) > 0 and (not peaks or max(peaks) > PEAK_LIMIT):
        problems.append(f"{w}: peak GPU memory per device {peaks}")


def read_r53(pairs, out_stem, root=RESULTS):
    problems, parts, excl = {}, [], []
    for tag, job in pairs:
        w = f"{tag}_{job}"
        dd = os.path.join(root, f"r14s1h_{tag}_{job}")
        sp = glob.glob(os.path.join(dd, "s1h_evaluate_*.json"))
        if len(sp) != 1:
            problems[w] = ["cannot load"]
            continue
        side = json.load(open(sp[0]))
        d = pd.read_parquet(os.path.join(dd, side["parquet"]))
        pr = []
        validity_r53(d, side, pr, w)
        rows = d.assign(model=side["model"], cell=tag, family=d.task.map(FAMILY))[
            ["model", "cell", "family", "prompt_idx", "task", "arm", "B", "a_span_nll_tok", "kl_span_tok", "kl_all",
             "score", "traffic_frac", "exact_rows_frac"]]
        (excl if pr else parts).append(rows)
        if pr:
            problems[w] = pr
        print(f"{w}: {'INVALID' if pr else 'valid'}", flush=True)
    out = dict(rules="plan.md amendment 'R5.3' (exploratory, no labels)", invalid_blocks=problems)
    md = ["# R14 Stage 1h — R5.3 read: the tail design's scan (rules: plan.md, amendment 'R5.3')", "",
          "Unit-clustered bootstrap 90% intervals; paired per unit; each model × cell × family separately.", ""]
    md += [f"- INVALID (excluded): {w}: {'; '.join(p)}" for w, p in problems.items()] or ["- every block valid"]

    def analyse(rows, title):
        res = dict(arms=arm_table(rows), questions={})
        for q, pairs_ in R53_Q.items():
            res["questions"][q] = []
            for (m, c, f), g in rows.groupby(["model", "cell", "family"]):
                piv = {k: v.set_index(UKEY) for k, v in g.groupby(["arm", "B"])}
                for a, b, ra, rb in pairs_:
                    if (a, b) not in piv or (ra, rb) not in piv:
                        continue
                    x, y = piv[(a, b)], piv[(ra, rb)]
                    idx = x.index.intersection(y.index)
                    res["questions"][q].append(dict(
                        group=f"{m}/{c}/{f}", arm=f"{a}@{b:g}", against=f"{ra}@{rb:g}",
                        d_KL_tok=boot(x.kl_span_tok[idx] - y.kl_span_tok[idx], np.mean),
                        d_dP_tok=boot(x.a_span_nll_tok[idx] - y.a_span_nll_tok[idx], np.mean),
                        d_acc=boot(x.score[idx] - y.score[idx], np.mean),
                        traffic=float(x.traffic_frac.mean()), traffic_ref=float(y.traffic_frac.mean())))
        L = [f"## {title} ({int((rows.arm == 'fp').sum())} units)", ""]
        for q, recs in res["questions"].items():
            L += [f"### {q}", "", "| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |",
                  "|---|---|---|---|---|---|---|"]
            L += [f"| {r['group']} | {r['arm']} | {r['against']} | {_ci(r['d_KL_tok'])} | {_ci(r['d_dP_tok'])} | "
                  f"{_ci(r['d_acc'], 3)} | {r['traffic']:.3f} / {r['traffic_ref']:.3f} |" for r in recs]
            L.append("")
        L += ["### The accuracy–cost frontier (against FP, per span token; sorted by traffic)", "",
              "| model / cell / family | arm@B | traffic | KL/tok | dP/tok | ΔAcc (FP right) |", "|---|---|---|---|---|---|"]
        for r in sorted(res["arms"], key=lambda r: (r["model"], r["cell"], r["family"], r["traffic"])):
            L.append(f"| {r['model']}/{r['cell']}/{r['family']} | {r['arm']}@{r['B']:g} | {r['traffic']:.3f} | "
                     f"{_ci(r['KL_tok'])} | {_ci(r['dP_tok'])} | {_ci(r['dAcc_fp_correct'], 3)} |")
        return res, L

    if parts:
        out["main"], L = analyse(pd.concat(parts, ignore_index=True), "Main read: valid blocks")
        md += [""] + L
    if excl:
        out["appendix"], L = analyse(pd.concat(excl, ignore_index=True), "APPENDIX: blocks excluded by validity")
        md += ["", "---", ""] + L
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:6]))
    print(f"wrote {out_stem}.md / .json")
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", nargs="+", metavar="TAG:JOB")
    g.add_argument("--r5", nargs="+", metavar="TAG:JOB", help="the frozen R5 read (plan.md 'R5 freeze')")
    g.add_argument("--r53", nargs="+", metavar="TAG:JOB", help="the R5.3 read (plan.md amendment 'R5.3')")
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    pairs = []
    for x in (a.pilot or a.r5 or a.r53):
        if ":" not in x:
            ap.error(f"takes TAG:JOB, not {x!r}")
        pairs.append(tuple(x.split(":", 1)))
    if a.r53:
        sys.exit(read_r53(pairs, a.out_stem or os.path.join(FINDINGS, "R5_3_reader"), a.results_root))
    if a.r5:
        sys.exit(read_r5(pairs, a.out_stem or os.path.join(FINDINGS, "R5_reader"), a.results_root))
    sys.exit(read_pilot(pairs, a.out_stem or os.path.join(FINDINGS, "R5_pilot"), a.results_root))


if __name__ == "__main__":
    main()
