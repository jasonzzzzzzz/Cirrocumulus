#!/usr/bin/env python3
"""R14 Stage 1h reader for R3a (harder synthetic tasks, both models). The rules below are
FROZEN: written 2026-10-04, before any R3a output existed (plan.md, R3a amendment).
Design: s1h3_lib.py. Driver: run_s1h3.py. Everything not stated here is R2's rule
(read_stage1h_r2.py), applied through its functions with s1h3_lib's presets.

    python read_stage1h_r3.py --ladder --llama J1 J2 J3 --qwen J4 J5 J6 [--out-stem .../findings/R3a_levels]
    python read_stage1h_r3.py --r3a --llama A1 A2 --qwen B1 B2 [--levels-json ...] [--r1-json ...]

STEP 1, THE LADDER (excluded from every result). Blocks h3ladder_llama / h3ladder_qwen, one
  per level of s1h3_lib.LEVELS (its task_cfg_r3 names the level), prompts 3200-3204, arms
  FP, D, D_V4. Per (model, task): FP's mean score per level, and D_V4's score minus FP's.
  The main cell's level of each task is s1h3_lib.choose_level over FP's means (lowest level
  with FP in [0.5, 0.95]; else the highest if all are above 0.95 (CEILING_REMAINS), the
  lowest if all are below 0.5 (TOO_HARD), the closest to 0.75 otherwise). mk_panel has one
  level. Output: R3a_levels.json, whose task_cfg per model the main cells must run with.
  Validity: every level once per model; the ladder's arms on every unit; stage 1h, R3a.
STEP 2, THE MAIN CELLS ('r3llama': h3llama at 128K, prompts 9400-9419; 'r3qwen': h3qwen at
  32K, prompts 9420-9439). EXPLORATORY.
  VALIDITY: R2's (R1's with each preset's stop rule), with FP_MIN = 0.5 in place of 0.9 (the
  tasks are made hard on purpose), and every block's task_cfg_r3 equal to R3a_levels.json's
  for its model (INVALID otherwise: the level was not the frozen one).
  NLL / KL LABELS: R2's (read_stage1h_r2.analyse_r2): NEAR_FP and EQUIV at R1's m_FP,
  SYS_VS_DENSE4, REQ1, STORE4, the floor at Qwen's r = 1/2, SEQUENTIAL; and R1's.
  ACCURACY LABELS (co-primary; free-running RULER score per unit; prompt bootstrap, 90%):
    ACC_SYSTEM     the system (qread2t4kq_v4@1/8) minus FP: ACC_NEAR_FP if the interval's lower
                   bound >= -ACC_MARGIN (0.03); ACC_LOSS if its upper bound < -0.03; else
                   ACC_INCONCLUSIVE.
    BEST_DENSE_ACC the same-memory dense 4-bit arm (uniform+v4@4, kivi4_v4@4, kvquant4_v4@4)
                   with the highest mean score in the cell (the strongest baseline).
    SYS_VS_BEST_ACC, SYS_VS_D_ACC, SIMPLE_VS_BEST_ACC (qread4_v4), REQ1_ACC (qread4q_v4 minus
                   qread4_v4), VOTE_LOSS_ACC (qreadfp_v16 minus qoraclefp_v16): accuracy effect
                   labels, _HELPS if the mean >= +0.02 and the interval excludes 0, _HURTS if
                   <= -0.02 and excludes 0, else _NO_EFFECT; plus EQUIV within +-0.03.
  REPORTED: accuracy per task and arm; FP's headroom per task; on mk_panel, the share of
  answers stating another needle's value (confusion) per arm; the FP-FAILED stratum.
AMENDMENT R3a2 (2026-10-05; s1h3_lib's docstring; written before any main-cell output, the
  first ladder void): blocks carry amend_r3 'R3a2' and stop rule 'r8list' (both models);
  NEAR_FP and EQUIV use the CELL'S OWN margin m_cell = s1h_lib.margin_fp(hi90 of fp8kv's mean
  dP in the cell), i.e. FP8 KV's cost on this model and these tasks, clipped to [0.05, 0.10]
  (R1's m_FP was measured on Llama's RULER answer values at 128K). R1's m_FP, when R1's
  reader output exists, is reported beside it (vs_FP_r1); the read no longer waits for it.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import read_stage1h_r2 as RR2  # noqa: E402  (installs R2's grammar and bytes into read_stage1h)
import read_stage1h as R1R  # noqa: E402
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h3_lib as L  # noqa: E402

RESULTS = R1R.RESULTS
FINDINGS = os.path.join(HERE, "findings")
ACC_MARGIN, ACC_EPS = 0.03, 0.02
SYSTEM = RR2.SYSTEM
DENSE4 = R1R.DENSE4
DV4 = R1R.DV4
name, ci = R1R.name, R1C.ci
MODELS = {"llama": ("llama31-8b", "h3ladder_llama", "h3llama"), "qwen": ("qwen3-30b-a3b-2507", "h3ladder_qwen", "h3qwen")}


def validate_h3(d, sides, problems, main=True):
    """R2's validity with s1h3_lib's presets for the stop rule (FP_MIN is set below)."""
    ss = []
    for s in sides:
        stop = L.PRESETS.get(s.get("preset_name"), {}).get("stop", "r8")
        if s.get("stop_rule") != stop:
            problems.append(f"block {s.get('_job')}: stop rule {s.get('stop_rule')}, expected {stop}")
        ss.append(dict(s, stop_rule="r8"))
    out = RR2._ORIG_VALIDATE(d, ss, problems, main)
    pas = d.arm.map(L.parse_arm)
    m = (pas.map(lambda p: p["family"]) == "qreadq").to_numpy()
    if m.any():
        x = d[m]
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(x.B, x.ctx_len)])
        if not (np.allclose(x.read_frac, k, atol=1e-9) and (x.stored_evict_frac == 0).all()):
            problems.append("a qreadq row does not read floor(r C) per KV head")
    return out


R1R.L = L
R1R.validate_h = validate_h3
R1R.FP_MIN = L.HEAD_LO


def _level_of(cfg) -> int:
    cfg = L.parse_task_cfg(L.task_cfg_str(cfg))
    for lv, c in L.LEVELS.items():
        if all(int(c[k]) == int(cfg[k]) for k in c):
            return lv
    raise SystemExit(f"ladder block ran task_cfg {cfg}, not one of s1h3_lib.LEVELS")


# ------------------------------------------------------------------ ladder
def read_ladder(jobs_by_model, out_stem, root=RESULTS):
    out, problems = {}, []
    for mk, jobs in jobs_by_model.items():
        model, tag, main = MODELS[mk]
        per = {}
        for j in jobs:
            d, side = R1R.load_run(tag, j, root)
            if side.get("stage") != "1h" or side.get("amend_r3") != L.AMEND_R3:
                problems.append(f"ladder {mk} {j}: not an R3a block")
            lv = _level_of(side.get("task_cfg_r3") or {})
            if lv in per:
                problems.append(f"ladder {mk}: level {lv} ran twice")
            want = R1R.plan_of(side)
            got = d.groupby(["prompt_idx", "task"]).apply(lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))),
                                                         include_groups=False)
            if any(v != want for v in got):
                problems.append(f"ladder {mk} {j}: a unit lacks the ladder's arms")
            per[lv] = d
        if sorted(per) != sorted(L.LEVELS):
            problems.append(f"ladder {mk}: levels {sorted(per)}, expected {sorted(L.LEVELS)}")
        if problems:
            continue
        rec = dict(model=model, main_preset=main, fp_by_level={}, dv4_minus_fp={}, levels={}, labels={})
        for task in ("niah_multikey", "niah_multivalue", "vt"):
            fpm, gap = {}, {}
            for lv, d in sorted(per.items()):
                x = d[d.task == task]
                if not len(x):
                    continue
                fp = x[x.arm == "fp"].set_index("prompt_idx").score
                dv = x[(x.arm == DV4[0]) & (x.B == DV4[1])].set_index("prompt_idx").score
                fpm[lv] = float(fp.mean())
                gap[lv] = float((dv - fp.reindex(dv.index)).mean())
            lv, lab = L.choose_level(fpm)
            rec["fp_by_level"][task], rec["dv4_minus_fp"][task] = fpm, gap
            rec["levels"][task], rec["labels"][task] = lv, lab
        pan = [(lv, d[d.task == L.PANEL]) for lv, d in sorted(per.items()) if (d.task == L.PANEL).any()]
        if pan:
            x = pan[0][1]
            fp = x[x.arm == "fp"].set_index("prompt_idx").score
            dv = x[(x.arm == DV4[0]) & (x.B == DV4[1])].set_index("prompt_idx").score
            rec["panel"] = dict(fp=float(fp.mean()), dv4_minus_fp=float((dv - fp.reindex(dv.index)).mean()),
                                headroom=bool(L.HEAD_LO <= float(fp.mean()) <= L.HEAD_HI))
        rec["task_cfg"] = L.task_cfg_str(L.main_task_cfg(rec["levels"]))
        out[mk] = rec
    if problems:
        raise SystemExit("INVALID R3a ladder:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rule="s1h3_lib.choose_level", levels=L.LEVELS, head=[L.HEAD_LO, L.HEAD_HI, L.HEAD_TARGET],
                       models=out), fh, indent=1, default=str)
    Lh = ["# R14 Stage 1h — R3a ladder (read_stage1h_r3.py; rules frozen in its docstring)", "",
          "Excluded from every result: it only chooses each task's difficulty for the main cells.", ""]
    for mk, rec in out.items():
        Lh += [f"## {rec['model']} → {rec['main_preset']} with `--task-cfg {rec['task_cfg']}`", "",
               "| task | FP score by level | D_V4 − FP by level | chosen | label |", "|---|---|---|---:|---|"]
        for t in ("niah_multikey", "niah_multivalue", "vt"):
            Lh.append(f"| {t} | {rec['fp_by_level'][t]} | "
                      f"{ {k: round(v, 3) for k, v in rec['dv4_minus_fp'][t].items()} } | {rec['levels'][t]} | "
                      f"{rec['labels'][t]} |")
        if rec.get("panel"):
            p = rec["panel"]
            Lh.append(f"| mk_panel | {p['fp']:.3f} | {p['dv4_minus_fp']:+.3f} | — | "
                      f"{'HEADROOM' if p['headroom'] else 'no headroom'} |")
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh))
    return 0


# ------------------------------------------------------------------ margins
def cell_margin(d):
    """(m_cell, fp8kv's mean dP interval) over the cell's units (R2's bootstrap frame)."""
    P, _, boot, _ = RR2._boot_frame(d)
    c = next((c for c in P.columns if c[0] == "fp8kv"), None)
    if c is None:
        return None, None
    z = boot(P[c])
    return L.margin_fp(z[2]), z


def load_r1(r1_json):
    """R1's reader output, if it exists: (m_FP or None, R1's best dense arm or None)."""
    if not r1_json or not os.path.exists(r1_json):
        return None, None
    r1 = json.load(open(r1_json)).get("r1", {})
    return (float(r1["m_fp"]) if r1.get("m_fp") is not None else None), (r1.get("best_dense") or {}).get("best")


def add_r1_labels(res, m_r1):
    """NEAR_FP and EQUIV at R1's m_FP beside the cell's own (vs_FP_r1)."""
    for p in res["points"].values():
        p["vs_FP_r1"] = dict(label=L.near_fp_label(p["vs_FP"]["nll"], p["vs_FP"]["tail"], m_r1),
                             equiv=L.equiv_label(p["vs_FP"]["nll"], m_r1))


def analyse_nll(d, sides, problems, cell, r1_json):
    """R2's NLL / KL analysis at the cell's own FP8 margin, R1's beside it (amendment R3a2)."""
    m_cell, fp8 = cell_margin(d)
    if m_cell is None:
        problems.append(f"{cell}: no fp8kv arm, so no margin")
        return None
    m_r1, r1_best = load_r1(r1_json)
    res = RR2.analyse_r2(d, sides, m_cell, r1_best)
    res.update(m_cell=m_cell, fp8_dP=fp8, m_r1=m_r1)
    if m_r1 is not None:
        add_r1_labels(res, m_r1)
    return res


def system_nll_line(res, sk):
    p = res["points"][sk]
    out = (f"{p['vs_FP_m']['label']} at the cell's margin {res['m_cell']:.3f} ({ci(p['vs_FP']['nll'])}); "
           f"vs D_V4 {p['vs_D']['label']}")
    if "vs_FP_r1" in p:
        out += f"; at R1's m_FP {res['m_r1']:.3f}: {p['vs_FP_r1']['label']}"
    return out


# ---------------------------------------------------------------- main cells
def _acc_label(z, lab, eps=ACC_EPS):
    if z[0] >= eps and z[1] > 0:
        return f"{lab}_HELPS"
    if z[0] <= -eps and z[2] < 0:
        return f"{lab}_HURTS"
    return f"{lab}_NO_EFFECT"


def analyse_acc(d):
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= L.HEAD_LO)
    v = d[d.task.isin(valid)]
    SC = v.pivot_table(index=R1R.KEY, columns=["arm", "B"], values="score", aggfunc="first")
    pidx = SC.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1), pidx).to_numpy()  # noqa: E731
    boot = lambda s: L1C.boot_ci(pm(s), W)  # noqa: E731
    fpc = ("fp", 0.0)
    arms = {}
    for c in SC.columns:
        arms[name(c)] = dict(acc=boot(SC[c]), dacc=boot(SC[c] - SC[fpc]),
                             by_task={t: round(float(g.mean()), 3) for t, g in SC[c].groupby(level=-1)})
    out = dict(valid_tasks=valid, fp_by_task=fp_cells.round(3).to_dict(), n_units=int(len(SC)), arms=arms, labels={})

    def lab(a, b, nm, equiv=True):
        if a not in SC.columns or b not in SC.columns:
            return
        z = boot(SC[a] - SC[b])
        out["labels"][nm] = dict(diff=z, label=_acc_label(z, nm), a=name(a), b=name(b),
                                 equiv=("EQUIV" if (z[1] >= -ACC_MARGIN and z[2] <= ACC_MARGIN) else "NOT_EQUIV")
                                 if equiv else None)

    if SYSTEM in SC.columns:
        z = boot(SC[SYSTEM] - SC[fpc])
        out["labels"]["ACC_SYSTEM"] = dict(diff=z, a=name(SYSTEM), b="fp@0",
                                           label="ACC_NEAR_FP" if z[1] >= -ACC_MARGIN else
                                           ("ACC_LOSS" if z[2] < -ACC_MARGIN else "ACC_INCONCLUSIVE"))
    dm = {c: float(SC[c].mean()) for c in DENSE4 if c in SC.columns}
    best = max(dm, key=lambda c: (dm[c], c == DENSE4[0])) if dm else None
    out["best_dense_acc"] = name(best) if best else None
    if best:
        lab(SYSTEM, best, "SYS_VS_BEST_ACC")
        lab(("qread4_v4", 0.125), best, "SIMPLE_VS_BEST_ACC")
    lab(SYSTEM, DV4, "SYS_VS_D_ACC")
    lab(("qread4q_v4", 0.125), ("qread4_v4", 0.125), "REQ1_ACC")
    lab(("qreadfp_v16", 0.125), ("qoraclefp_v16", 0.125), "VOTE_LOSS_ACC")
    pan = d[d.task == L.PANEL]
    if len(pan):
        out["panel"] = {f"{a}@{R1R.bk(b)}": dict(acc=round(float(g.score.mean()), 3),
                                                 confused=round(float(g.distractor.astype(float).mean()), 3))
                        for (a, b), g in pan.groupby(["arm", "B"])}
    return out


def read_r3a(jobs_by_model, out_stem, root=RESULTS, levels_json=None, r1_json=RR2.R1_JSON):
    levels_json = levels_json or os.path.join(FINDINGS, "R3a_levels.json")
    if not os.path.exists(levels_json):
        raise SystemExit(f"INVALID R3a read: the ladder's levels {levels_json} are missing")
    lev = json.load(open(levels_json))["models"]
    results, summary, problems = {}, {}, []
    for mk, jobs in jobs_by_model.items():
        model, _, tag = MODELS[mk]
        parts = [R1R.load_run(tag, j, root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        want = L.parse_task_cfg(lev[mk]["task_cfg"])
        for s in sides:
            if L.parse_task_cfg(L.task_cfg_str(s.get("task_cfg_r3") or {})) != want:
                problems.append(f"r3{mk} block {s['_job']}: task_cfg {s.get('task_cfg_r3')}, not the ladder's {want}")
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"r3{mk}: blocks ran different plans")
        R1R.validate_h(d, sides, problems)
        R1R.validate_a2_h(d, problems, f"r3{mk}")
        if problems:
            continue
        res = analyse_nll(d, sides, problems, f"r3{mk}", r1_json)
        if res is None:
            continue
        res["cell"] = f"r3{mk}"
        res["accuracy"] = analyse_acc(d)
        res["task_cfg"] = lev[mk]["task_cfg"]
        results[mk] = res
        a = res["accuracy"]
        for k, z in a["labels"].items():
            summary[f"{k} {mk}"] = z["label"] + (f", {z['equiv']}" if z.get("equiv") else "") + f" ({ci(z['diff'])})"
        summary[f"BEST_DENSE_ACC {mk}"] = a["best_dense_acc"]
        summary[f"FP accuracy {mk}"] = a["fp_by_task"]
        sk = name(SYSTEM)
        if sk in res["points"]:
            summary[f"SYSTEM NLL {mk}"] = system_nll_line(res, sk)
        for k in ("SYS_VS_BEST", "REQ1", "STORE4", "FLOOR_SYS"):
            if k in res["labels_r2"]:
                summary[f"{k} {mk}"] = res["labels_r2"][k]["label"]
    if problems:
        raise SystemExit("INVALID Stage 1h R3a data:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(margin="per cell: s1h_lib.margin_fp(hi90 fp8kv dP)", acc_margin=ACC_MARGIN,
                                  acc_eps=ACC_EPS, fp_min=L.HEAD_LO,
                                  levels_json=levels_json), summary=summary, cells=results), fh, indent=1, default=str)
    Lh = ["# R14 Stage 1h — R3a, harder synthetic tasks (read_stage1h_r3.py; rules frozen in its docstring)", "",
          "EXPLORATORY run: labels guide the design; they are not claims.", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for mk, res in results.items():
        a = res["accuracy"]
        Lh += [f"## r3{mk} — {res['task_cfg']}; FP accuracy {a['fp_by_task']}; {a['n_units']} units", "",
               "| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |", "|---|---|---|---|"]
        Lh += [f"| {k} | {ci(z['acc'])} | {ci(z['dacc'])} | {z['by_task']} |"
               for k, z in sorted(a["arms"].items(), key=lambda kz: -kz[1]["acc"][0])]
        Lh.append("")
        for k, z in a["labels"].items():
            Lh.append(f"**{k}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}"
                      + (f", {z['equiv']}" if z.get("equiv") else ""))
        if a.get("panel"):
            Lh.append(f"**mk_panel** (accuracy, confusion): {a['panel']}")
        Lh += ["", "NLL / KL points:", ""] + R1R._arm_table(res["points"]) + [""]
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh[:4 + len(summary) + 1]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--r3a", action="store_true")
    ap.add_argument("--llama", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--qwen", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--levels-json", default=None)
    ap.add_argument("--r1-json", default=RR2.R1_JSON)
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    jobs = {k: v for k, v in (("llama", a.llama), ("qwen", a.qwen)) if v}
    if not jobs or a.ladder == a.r3a:
        ap.error("give --ladder or --r3a, with --llama and/or --qwen jobs")
    if a.ladder:
        sys.exit(read_ladder(jobs, a.out_stem or os.path.join(FINDINGS, "R3a_levels"), a.results_root))
    sys.exit(read_r3a(jobs, a.out_stem or os.path.join(FINDINGS, "R3a_reader"), a.results_root, a.levels_json,
                      a.r1_json))


if __name__ == "__main__":
    main()
