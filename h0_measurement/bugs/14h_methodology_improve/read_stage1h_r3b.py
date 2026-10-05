#!/usr/bin/env python3
"""R14 Stage 1h reader for R3b (aggregation and latent-association stress tasks, both
models). The rules below are FROZEN: written 2026-10-05, before any R3b output existed
(plan.md, R3b amendment). Design: s1h3b_lib.py. Tasks: tasks_s1h.py. Driver: run_s1h3b.py.
Everything not stated here is R3a's rule (read_stage1h_r3.py), and through it R2's and R1's.

    python read_stage1h_r3b.py --ladder --llama L1 L2 L3 LN --qwen Q1 Q2 Q3 QN [--out-stem .../findings/R3b_levels]
    python read_stage1h_r3b.py --r3b --llama A1 A2 --qwen B1 B2 [--levels-json ...] [--r1-json ...]

STEP 1, THE LADDER (excluded from every result). Per model (h3bladder_llama at 128K,
  h3bladder_qwen at 32K): three level jobs (cwe + fwe on prompts 3210-3217; task_cfg_r3b =
  s1h3b_lib.LEVELS_R3B[level]) and one nolima job (nolima + nolima_direct on 3210-3225);
  arms FP, D, D_V4. Per (model, task): FP's mean score per level and D_V4's minus FP's.
  - cwe, fwe: s1h3b_lib.choose_level_r3b over FP's means (the lowest level with FP in
    [0.5, 0.95]; else the level closest to 0.75, ties to the lower: CEILING_REMAINS /
    TOO_HARD / CLOSEST).
  - nolima: its one FP mean.
  - A task ENTERS step 2 iff FP's mean at its chosen level is >= 0.5; nolima_direct enters
    with nolima (s1h3b_lib.main_tasks_r3b). Output R3b_levels.json: per model the main
    cell's task_cfg (s1h3b_lib.main_cfg_r3b) and tasks. A model with no task entering has
    no main cell (NO_TASKS).
  Validity: each level once per model and one nolima job; the ladder's arms on every unit;
  each job an R3b block (stage 1h, amend_r3b R3b) with its preset's stop rule (r8list for
  Llama, eos_only for Qwen); the nolima job ran both nolima tasks on the same prompts.
STEP 2, THE MAIN CELLS ('r3bllama': h3bllama at 128K, prompts 9440-9459; 'r3bqwen':
  h3bqwen at 32K, prompts 9460-9479). EXPLORATORY.
  VALIDITY: R2's (R1's with each preset's stop rule), except the FP floor: a task whose
  FP mean is below 0.5 in the cell leaves every label (reported as FP_LOW), and the cell
  is INVALID only if no task remains. Every block's task_cfg_r3b and tasks equal
  R3b_levels.json's for its model (INVALID otherwise). A2 self-check replays are required
  when the cell holds a multi-answer task (cwe or fwe).
  NLL / KL LABELS: R2's (read_stage1h_r2.analyse_r2) at R1's m_FP, pooled over the
  remaining tasks (as R3a).
  ACCURACY LABELS: R3a's (read_stage1h_r3.analyse_acc: ACC_SYSTEM, BEST_DENSE_ACC,
  SYS_VS_BEST_ACC, SYS_VS_D_ACC, SIMPLE_VS_BEST_ACC, REQ1_ACC, VOTE_LOSS_ACC), computed
  separately for each family present: AGG = cwe + fwe, LATENT = nolima, DIRECT =
  nolima_direct.
  LEXICAL (needs both nolima tasks in the cell), over the prompts where FP scores 1 on both
  (at least LEX_MIN = 6 prompts, else NO_DATA); prompt bootstrap, 90%:
    LEX_VOTE   [dP(qreadfp_v16@1/8) - dP(qoraclefp_v16@1/8)] on nolima minus the same on
               nolima_direct: does the question's vote lose more when the question shares no
               word with the needle?
    LEX_SYS    dP(system) on nolima minus dP(system) on nolima_direct.
    Labels: _HURTS if the mean >= +0.05 nats and the interval excludes 0; _HELPS if the
    mean <= -0.05 and the interval excludes 0; else _NO_EFFECT. The same two contrasts on
    the score (LEX_VOTE_ACC, LEX_SYS_ACC; the sign flipped, so _HURTS = a larger accuracy
    loss on nolima) with ACC_EPS = 0.02.
  REPORTED: accuracy per task and arm; FP's headroom per task; per task and arm the share of
  the answer's context positions the arm kept (needle_keep; cwe: every occurrence of the 10
  common words) and the share of answers stating a non-answer word of the list (cwe, fwe:
  distractor); the FP-FAILED stratum.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import read_stage1h_r3 as RR3  # noqa: E402  (installs R3a's (and R2's) rules into read_stage1h)
import read_stage1h_r2 as RR2  # noqa: E402
import read_stage1h as R1R  # noqa: E402
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h3b_lib as L  # noqa: E402

RESULTS = R1R.RESULTS
FINDINGS = os.path.join(HERE, "findings")
LEX_MIN, LEX_EPS = 6, 0.05
ACC_EPS = RR3.ACC_EPS
SYSTEM = RR2.SYSTEM
DV4 = R1R.DV4
VOTE_PAIR = (("qreadfp_v16", 0.125), ("qoraclefp_v16", 0.125))
FAMILIES = {"AGG": ("cwe", "fwe"), "LATENT": (L.NOLIMA,), "DIRECT": (L.NOLIMA_DIRECT,)}
name, ci = R1R.name, R1C.ci
MODELS = {"llama": ("llama31-8b", "h3bladder_llama", "h3bllama"),
          "qwen": ("qwen3-30b-a3b-2507", "h3bladder_qwen", "h3bqwen")}


def validate_h3b(d, sides, problems, main=True):
    """R2's validity (R1's plus the qreadq audit) with s1h3b_lib's presets for the stop
    rule; the FP floor is applied by the reader (module docstring), so R1's is not."""
    ss = []
    for s in sides:
        stop = L.PRESETS.get(s.get("preset_name"), {}).get("stop", "r8")
        if s.get("stop_rule") != stop:
            problems.append(f"block {s.get('_job')}: stop rule {s.get('stop_rule')}, expected {stop}")
        ss.append(dict(s, stop_rule="r8", preset_name=None))     # validate_h2 then expects r8
    return RR2.validate_h2(d, ss, problems, False)


R1R.L = L
R1R.validate_h = validate_h3b
R1R.FP_MIN = L.HEAD_LO


def _r3b_side(side, problems, what):
    if side.get("stage") != "1h" or side.get("amend_r3b") != L.AMEND_R3B:
        problems.append(f"{what}: not an R3b block")
    stop = L.PRESETS.get(side.get("preset_name"), {}).get("stop")
    if side.get("stop_rule") != stop:
        problems.append(f"{what}: stop rule {side.get('stop_rule')}, expected {stop}")


def _level_of(cfg) -> int:
    c = L.T.task_config_r3b(**(cfg or {}))
    for lv, x in L.LEVELS_R3B.items():
        if L.T.task_config_r3b(**x) == c:
            return lv
    raise SystemExit(f"ladder block ran task_cfg_r3b {cfg}, not one of s1h3b_lib.LEVELS_R3B")


def _fp_gap(x):
    fp = x[x.arm == "fp"].set_index("prompt_idx").score
    dv = x[(x.arm == DV4[0]) & (x.B == DV4[1])].set_index("prompt_idx").score
    return float(fp.mean()), float((dv - fp.reindex(dv.index)).mean())


# ------------------------------------------------------------------ ladder
def read_ladder(jobs_by_model, out_stem, root=RESULTS):
    out, problems = {}, []
    for mk, jobs in jobs_by_model.items():
        model, tag, main = MODELS[mk]
        per, nol = {}, None
        for j in jobs:
            d, side = R1R.load_run(tag, j, root)
            _r3b_side(side, problems, f"ladder {mk} {j}")
            want = R1R.plan_of(side)
            got = d.groupby(["prompt_idx", "task"]).apply(lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))),
                                                         include_groups=False)
            if any(v != want for v in got):
                problems.append(f"ladder {mk} {j}: a unit lacks the ladder's arms")
            tasks = set(d.task)
            if tasks <= set(L.LEVEL_TASKS):
                lv = _level_of(side.get("task_cfg_r3b"))
                if lv in per:
                    problems.append(f"ladder {mk}: level {lv} ran twice")
                per[lv] = d
            elif tasks == {L.NOLIMA, L.NOLIMA_DIRECT}:
                ps = d.groupby("task").prompt_idx.apply(lambda s: tuple(sorted(set(s))))
                if ps[L.NOLIMA] != ps[L.NOLIMA_DIRECT]:
                    problems.append(f"ladder {mk} {j}: the nolima tasks ran on different prompts")
                if nol is not None:
                    problems.append(f"ladder {mk}: two nolima jobs")
                nol = d
            else:
                problems.append(f"ladder {mk} {j}: tasks {sorted(tasks)} are neither a level job nor the nolima job")
        if sorted(per) != sorted(L.LEVELS_R3B) or nol is None:
            problems.append(f"ladder {mk}: levels {sorted(per)} (expected {sorted(L.LEVELS_R3B)}), "
                            f"nolima job {'present' if nol is not None else 'missing'}")
        if problems:
            continue
        rec = dict(model=model, main_preset=main, fp_by_level={}, dv4_minus_fp={}, levels={}, labels={}, entered={},
                   fp_chosen={})
        for task in L.LEVEL_TASKS:
            fpm, gap = {}, {}
            for lv, d in sorted(per.items()):
                x = d[d.task == task]
                if len(x):
                    fpm[lv], gap[lv] = _fp_gap(x)
            lv, lab = L.choose_level_r3b(fpm)
            rec["fp_by_level"][task], rec["dv4_minus_fp"][task] = fpm, gap
            rec["levels"][task], rec["labels"][task] = lv, lab
            rec["fp_chosen"][task] = fpm.get(lv, float("nan"))
            rec["entered"][task] = L.enters(rec["fp_chosen"][task])
        for task in (L.NOLIMA, L.NOLIMA_DIRECT):
            fp, gap = _fp_gap(nol[nol.task == task])
            rec["fp_by_level"][task], rec["dv4_minus_fp"][task] = {1: fp}, {1: gap}
            rec["fp_chosen"][task] = fp
        rec["entered"][L.NOLIMA] = L.enters(rec["fp_chosen"][L.NOLIMA])
        rec["entered"][L.NOLIMA_DIRECT] = rec["entered"][L.NOLIMA]
        rec["labels"][L.NOLIMA] = "ENTERS" if rec["entered"][L.NOLIMA] else "TOO_HARD"
        rec["task_cfg"] = L.T.cfg_str(L.main_cfg_r3b(rec["levels"]))
        rec["tasks"] = L.main_tasks_r3b(rec["entered"])
        rec["status"] = "RUN" if rec["tasks"] else "NO_TASKS"
        out[mk] = rec
    if problems:
        raise SystemExit("INVALID R3b ladder:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rule="s1h3b_lib.choose_level_r3b + enters", levels=L.LEVELS_R3B,
                       head=[L.HEAD_LO, L.HEAD_HI, L.HEAD_TARGET], models=out), fh, indent=1, default=str)
    Lh = ["# R14 Stage 1h — R3b ladder (read_stage1h_r3b.py; rules frozen in its docstring)", "",
          "Excluded from every result: it only chooses each task's difficulty, and which tasks run, for the "
          "main cells.", ""]
    for mk, rec in out.items():
        Lh += [f"## {rec['model']} → {rec['main_preset']}: {rec['status']}, tasks {','.join(rec['tasks']) or '—'}, "
               f"`--task-cfg {rec['task_cfg']}`", "",
               "| task | FP score by level | D_V4 − FP by level | chosen | label | enters |", "|---|---|---|---:|---|---|"]
        for t in L.R3B_TASKS:
            Lh.append(f"| {t} | { {k: round(v, 3) for k, v in rec['fp_by_level'][t].items()} } | "
                      f"{ {k: round(v, 3) for k, v in rec['dv4_minus_fp'][t].items()} } | {rec['levels'].get(t, '—')} | "
                      f"{rec['labels'].get(t, '(with nolima)')} | {rec['entered'][t]} |")
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh))
    return 0


# ---------------------------------------------------------------- main cells
def _effect(z, lab, eps):
    if z[0] >= eps and z[1] > 0:
        return f"{lab}_HURTS"
    if z[0] <= -eps and z[2] < 0:
        return f"{lab}_HELPS"
    return f"{lab}_NO_EFFECT"


def analyse_lexical(d):
    """LEX_VOTE, LEX_SYS and their accuracy twins (module docstring)."""
    out = {}
    if not {L.NOLIMA, L.NOLIMA_DIRECT} <= set(d.task):
        return out
    P = d.pivot_table(index=["job", "prompt_idx", "task"], columns=["arm", "B"], values=R1R.PRIMARY, aggfunc="first")
    P = P.sub(P[("fp", 0.0)], axis=0)
    SC = d.pivot_table(index=["job", "prompt_idx", "task"], columns=["arm", "B"], values="score", aggfunc="first")
    fpok = SC[("fp", 0.0)].unstack("task")
    keep = fpok.index[(fpok[L.NOLIMA] >= 1) & (fpok[L.NOLIMA_DIRECT] >= 1)]
    out["n_prompts"] = int(len(keep))
    if len(keep) < LEX_MIN:
        out["labels"] = {k: dict(label=f"{k}_NO_DATA") for k in ("LEX_VOTE", "LEX_SYS", "LEX_VOTE_ACC", "LEX_SYS_ACC")}
        return out
    W = BM.boot_weights(keep)

    def per_prompt(T, f):
        x = f(T).unstack("task").reindex(keep)
        return (x[L.NOLIMA] - x[L.NOLIMA_DIRECT]).to_numpy(dtype=float)

    va, vb = VOTE_PAIR
    lab = {}
    for key, T, f, sign, eps in (
            ("LEX_VOTE", P, lambda T: T[va] - T[vb], 1.0, LEX_EPS),
            ("LEX_SYS", P, lambda T: T[SYSTEM], 1.0, LEX_EPS),
            ("LEX_VOTE_ACC", SC, lambda T: T[va] - T[vb], -1.0, ACC_EPS),
            ("LEX_SYS_ACC", SC, lambda T: T[SYSTEM] - T[("fp", 0.0)], -1.0, ACC_EPS)):
        cols = {va, vb} if "VOTE" in key else {SYSTEM}
        if not cols <= set(T.columns):
            continue
        x = sign * per_prompt(T, f)
        if np.isnan(x).any():
            lab[key] = dict(label=f"{key}_NO_DATA", note="a qualifying prompt lacks the value")
            continue
        z = L1C.boot_ci(x, W)
        lab[key] = dict(diff=z, label=_effect(z, key, eps), sign=("+" if sign > 0 else "-") + " (nolima - direct)")
    out["labels"] = lab
    return out


def report_rows(d):
    g = d.groupby(["task", "arm", "B"])
    t = pd.DataFrame(dict(acc=g.score.mean(), needle_keep=g.needle_keep.mean(),
                          distractor=g.distractor.apply(lambda s: float(pd.Series(s).astype(float).mean()))))
    return {f"{task} {a}@{R1R.bk(b)}": {k: round(float(v), 3) for k, v in r.items()}
            for (task, a, b), r in t.iterrows()}


def read_r3b(jobs_by_model, out_stem, root=RESULTS, levels_json=None, r1_json=RR2.R1_JSON):
    levels_json = levels_json or os.path.join(FINDINGS, "R3b_levels.json")
    if not os.path.exists(levels_json):
        raise SystemExit(f"INVALID R3b read: the ladder's levels {levels_json} are missing")
    if not os.path.exists(r1_json):
        raise SystemExit(f"INVALID R3b read: R1's reader output {r1_json} is missing (m_FP comes from R1)")
    lev = json.load(open(levels_json))["models"]
    r1 = json.load(open(r1_json))["r1"]
    m_fp, r1_best = float(r1["m_fp"]), (r1.get("best_dense") or {}).get("best")
    results, summary, problems = {}, {}, []
    for mk, jobs in jobs_by_model.items():
        model, _, tag = MODELS[mk]
        if lev.get(mk, {}).get("status") != "RUN":
            problems.append(f"r3b{mk}: the ladder entered no task for {model}")
            continue
        parts = [R1R.load_run(tag, j, root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        want_cfg, want_tasks = L.T.parse_cfg(lev[mk]["task_cfg"]), list(lev[mk]["tasks"])
        for s in sides:
            _r3b_side(s, problems, f"r3b{mk} block {s['_job']}")
            if L.T.task_config_r3b(**(s.get("task_cfg_r3b") or {})) != want_cfg:
                problems.append(f"r3b{mk} block {s['_job']}: task_cfg {s.get('task_cfg_r3b')}, not the ladder's {want_cfg}")
            if list(s.get("tasks") or []) != want_tasks:
                problems.append(f"r3b{mk} block {s['_job']}: tasks {s.get('tasks')}, not the ladder's {want_tasks}")
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"r3b{mk}: blocks ran different plans")
        R1R.validate_h(d, sides, problems)
        R1R.validate_a2_h(d, problems, f"r3b{mk}", self_check=bool(set(d.task) & set(L.LEVEL_TASKS)))
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        low = sorted(t for t, s in fps.items() if s < L.HEAD_LO)
        if len(low) == len(fps):
            problems.append(f"r3b{mk}: FP is below {L.HEAD_LO} on every task ({fps.round(3).to_dict()})")
        if problems:
            continue
        res = RR2.analyse_r2(d, sides, m_fp, r1_best)
        res.update(cell=f"r3b{mk}", task_cfg=lev[mk]["task_cfg"], tasks=want_tasks, fp_low=low,
                   fp_by_task=fps.round(3).to_dict())
        res["accuracy"] = {}
        for fam, ts in FAMILIES.items():
            sub = d[d.task.isin(ts) & ~d.task.isin(low)]
            if len(sub):
                res["accuracy"][fam] = RR3.analyse_acc(sub)
        res["lexical"] = analyse_lexical(d)
        res["by_task_arm"] = report_rows(d)
        results[mk] = res
        for fam, a in res["accuracy"].items():
            for k, z in a["labels"].items():
                summary[f"{k} {fam} {mk}"] = (z["label"] + (f", {z['equiv']}" if z.get("equiv") else "")
                                              + f" ({ci(z['diff'])})")
            summary[f"BEST_DENSE_ACC {fam} {mk}"] = a["best_dense_acc"]
        for k, z in res["lexical"].get("labels", {}).items():
            summary[f"{k} {mk}"] = z["label"] + (f" ({ci(z['diff'])})" if "diff" in z else "")
        summary[f"FP accuracy {mk}"] = res["fp_by_task"] | ({"FP_LOW": low} if low else {})
        sk = name(SYSTEM)
        if sk in res["points"]:
            p = res["points"][sk]
            summary[f"SYSTEM NLL {mk}"] = (f"{p['vs_FP_m']['label']} at m_FP {m_fp:.3f} ({ci(p['vs_FP']['nll'])}); "
                                           f"vs D_V4 {p['vs_D']['label']}")
        for k in ("SYS_VS_BEST", "REQ1", "STORE4", "FLOOR_SYS"):
            if k in res["labels_r2"]:
                summary[f"{k} {mk}"] = res["labels_r2"][k]["label"]
    if problems:
        raise SystemExit("INVALID Stage 1h R3b data:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(m_fp=m_fp, fp_min=L.HEAD_LO, lex_min=LEX_MIN, lex_eps=LEX_EPS, acc_eps=ACC_EPS,
                                  levels_json=levels_json), summary=summary, cells=results), fh, indent=1, default=str)
    Lh = ["# R14 Stage 1h — R3b, aggregation and latent-association tasks (read_stage1h_r3b.py; rules frozen in "
          "its docstring)", "", "EXPLORATORY run: labels guide the design; they are not claims.", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for mk, res in results.items():
        Lh += [f"## r3b{mk} — tasks {','.join(res['tasks'])}, `{res['task_cfg']}`; FP {res['fp_by_task']}", ""]
        for fam, a in res["accuracy"].items():
            Lh += [f"### {fam} ({', '.join(FAMILIES[fam])}): {a['n_units']} units", "",
                   "| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |", "|---|---|---|---|"]
            Lh += [f"| {k} | {ci(z['acc'])} | {ci(z['dacc'])} | {z['by_task']} |"
                   for k, z in sorted(a["arms"].items(), key=lambda kz: -kz[1]["acc"][0])]
            Lh.append("")
            for k, z in a["labels"].items():
                Lh.append(f"**{k}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}"
                          + (f", {z['equiv']}" if z.get("equiv") else ""))
            Lh.append("")
        lx = res["lexical"]
        if lx:
            Lh += [f"### Lexical overlap ({lx['n_prompts']} prompts with FP right on both nolima tasks)", ""]
            Lh += [f"**{k}**: " + (f"{ci(z['diff'])} → " if "diff" in z else "") + z["label"]
                   for k, z in lx["labels"].items()] + [""]
        Lh += ["### Per task and arm: accuracy, needle_keep, distractor", "", "| task arm@B | values |", "|---|---|"]
        Lh += [f"| {k} | {v} |" for k, v in res["by_task_arm"].items()]
        Lh += ["", "NLL / KL points:", ""] + R1R._arm_table(res["points"]) + [""]
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh[:4 + len(summary) + 1]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--r3b", action="store_true")
    ap.add_argument("--llama", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--qwen", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--levels-json", default=None)
    ap.add_argument("--r1-json", default=RR2.R1_JSON)
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    jobs = {k: v for k, v in (("llama", a.llama), ("qwen", a.qwen)) if v}
    if not jobs or a.ladder == a.r3b:
        ap.error("give --ladder or --r3b, with --llama and/or --qwen jobs")
    if a.ladder:
        sys.exit(read_ladder(jobs, a.out_stem or os.path.join(FINDINGS, "R3b_levels"), a.results_root))
    sys.exit(read_r3b(jobs, a.out_stem or os.path.join(FINDINGS, "R3b_reader"), a.results_root, a.levels_json,
                      a.r1_json))


if __name__ == "__main__":
    main()
