#!/usr/bin/env python3
"""R14 Stage 1h gate and reader for R4 (real tasks at 128K: LongBench v2 and a HELMET subset,
with Quest, the floor system and a closed-book control). The rules below are FROZEN: written
2026-10-05 (amendment R4b), before any R4 output existed (plan.md). Design: s1h4_lib.py.
Tasks: tasks_s1h4.py. Driver: run_s1h4.py. Everything not stated here is R3a2's rule
(read_stage1h_r3.py), R2's and R1's.

    python read_stage1h_r4.py --gate CELL JOB
    python read_stage1h_r4.py --lb2llama A1 A2 --lb2qwen B1 B2 --hmllama C1 C2 --hmqwen D1 D2 [--r1-json ...]

CELLS s1h4_lib.CELLS (any subset may be read). EXPLORATORY.
UNITS item x task; every unit is its own bootstrap cluster (prompt bootstrap within block,
  90%). FAMILIES: lbv2; HELMET's rag (kilt_nq, kilt_hotpotqa), rerank, icl (trec_coarse,
  banking77). Every label is per cell and family: metrics are never pooled across families.
ACCURACY per unit: lbv2 = the FORCED CHOICE (fc_correct), as bug 9's V5-V7 (the greedy letter
  is reported); HELMET = HELMET's metric (SubEM, NDCG@10, exact match). No FP floor: units FP
  gets wrong stay in (the FP-FAILED stratum is reported).
CONTEXT-DEPENDENT units (CTX): FP's accuracy minus the closed-book arm's >= 0.5 on the unit.
  Every accuracy label is also given on CTX units (the _CTX twins, secondary).
NLL / KL: R2's (read_stage1h_r2.analyse_r2) per family at the family's own margin m_cell =
  s1h_lib.margin_fp(hi90 of fp8kv's mean dP in the cell and family); R1's m_FP beside it when
  R1's reader output exists. FP's first line is the answer span. Effect labels on dP
  (|mean| >= 0.05 nats, interval excluding 0): SYS_VS_QUEST (system@1/8 - quest_v16@1/8),
  SCHEDULE (qread4_v4@1/8 - quest4_v4@1/8), FLOOR_VS_FIXED (qread2t4kqF_v4 - system@1/8).
ACCURACY LABELS (R3a's thresholds):
    ACC_SYSTEM / ACC_FLOOR / ACC_QUEST  the arm minus FP: ACC_NEAR_FP if the lower bound >=
                -0.03; ACC_LOSS if the upper bound < -0.03; else ACC_INCONCLUSIVE.
    SYS_VS_QUEST_ACC, FLOOR_VS_QUEST_ACC, SCHEDULE_ACC, SYS_VS_BEST_ACC (the same-memory dense
                4-bit arm with the highest mean accuracy in the cell and family), SYS_VS_D_ACC,
                REQ1_ACC, VOTE_LOSS_ACC: _HELPS if the mean >= +0.02 and the interval excludes
                0, _HURTS if <= -0.02 and excludes 0, else _NO_EFFECT; plus EQUIV within +-0.03.
REPORTED: accuracy per arm and task; lbv2's greedy-letter accuracy, the 4-choice KL to FP and
  the change in the gold choice's log-probability per arm (fc_gold_logp, with its interval);
  the closed-book accuracy and the CTX count; Quest's realized read fraction; the floor's r per
  item; lost answers (R1's LOST, on accuracy).
VALIDITY (any failure -> INVALID): R2's (R1's with the stop rule r8list, the qreadq audit),
  without R1's FP floor, on every row but the floor system's, which is audited at its own r;
  every block an R4b block of its cell's preset, suite and manifest cell, with the frozen
  manifest's sha256, its units the manifest's items in order, the preset's plan, fp_noise
  last; a quest row stores its width dense and reads within 0.01 of r over all layers; lbv2
  rows carry the forced-choice columns and FP's forced choice equals its greedy letter on >=
  90% of units; lbv2 rows record a 32-row vote span. A2 self-checks are not required (no unit
  has more than one answer value). R1's answer-value mask check (the mask found in >= 90% of
  FP's correct answers) holds vacuously when FP answers no unit correctly (amended 2026-10-05,
  before any R4 output: R1's code counted that as 0% and failed every 1-item lbv2 pilot FP
  gets wrong, and HELMET pilots whose re-ranking NDCG is below 1).
GATE (a cell's pilot, excluded; s1h4_lib.pilot_units): R4's validity (no main checks); FP's KL
  0; fp_noise not degenerate (R1's NOISE_DEGENERATE); peak GPU memory <= 76 GiB on every
  device; the projected block (s1h4_lib.block_units x the pilot's seconds per unit of each task
  kind, + 15 minutes) within 90% of the cell's wall time. Writes findings/R4_gate_<cell>.json;
  exits 1 on FAIL, so the cell's blocks (afterok) never start.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import read_stage1h_r3 as RR3  # noqa: E402  (installs R3a's and R2's rules into read_stage1h)
import read_stage1h_r2 as RR2  # noqa: E402
import read_stage1h as R1R  # noqa: E402
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h4_lib as L  # noqa: E402
import tasks_s1h4 as T  # noqa: E402

RESULTS = R1R.RESULTS
FINDINGS = os.path.join(HERE, "findings")
ACC_MARGIN, ACC_EPS = RR3.ACC_MARGIN, RR3.ACC_EPS
QUEST_TOL = 0.01
FC_AGREE_MIN = 0.90
CTX_MIN = 0.5
GATE_SETUP_H = 0.25
SYSTEM = RR2.SYSTEM
FLOOR = (L.FLOOR_ARM, 0.125)
QUEST = ("quest_v16", 0.125)
QUEST4 = ("quest4_v4", 0.125)
SIMPLE = ("qread4_v4", 0.125)
CB = (L.CLOSEDBOOK, 0.0)
DV4, DENSE4 = R1R.DV4, R1R.DENSE4
FPC = ("fp", 0.0)
name, ci = R1R.name, R1C.ci

_ORIG_BYTES = R1R.add_bytes_h


def _with_floor_r(d):
    """A copy whose floor rows carry B = floor_r(C) (the r they read at)."""
    x = d.copy()
    m = (x.arm == L.FLOOR_ARM).to_numpy()
    if m.any():
        x.loc[m, "B"] = [L.floor_r(int(c)) for c in x.loc[m, "ctx_len"]]
    return x


def validate_h4(d, sides, problems, main=True):
    """R2's validity (R1's, the qreadq audit) with r8list and no FP floor, on every row but
    the floor system's; the floor rows audited at their own r; the Quest audit."""
    fl = (d.arm == L.FLOOR_ARM).to_numpy()
    ss = []
    for s in sides:
        stop = L.PRESETS.get(s.get("preset_name"), {}).get("stop")
        if s.get("stop_rule") != stop:
            problems.append(f"block {s.get('_job')}: stop rule {s.get('stop_rule')}, expected {stop}")
        ss.append(dict(s, stop_rule="r8", preset_name=None, plan=[x for x in s["plan"] if x[0] != L.FLOOR_ARM]))
    n0 = len(problems)
    out = RR2.validate_h2(d[~fl], ss, problems, False)
    if not (d[(d.arm == "fp").to_numpy()].score >= 1).any():   # no correct FP answer: nothing to cover
        problems[n0:] = [p for p in problems[n0:] if not p.startswith("FP's answer-value mask")]
    x = d[fl]
    if len(x):
        units = d[d.arm == "fp"].set_index(R1R.KEY).index
        r = np.array([L.floor_r(int(c)) for c in x.ctx_len])
        k = np.array([L1C.qread_keep_count(a, int(c)) / int(c) for a, c in zip(r, x.ctx_len)])
        if not (len(x) == len(units) and set(x.set_index(R1R.KEY).index) == set(units)
                and np.allclose(x.floor_r, r) and np.allclose(x.read_frac, k, atol=1e-9)
                and np.allclose(x.stored_bits_per_token, L.STORE4) and (x.stored_evict_frac == 0).all()
                and np.allclose(x.bits_per_token, x.tier2_bits.astype(float) * x.read_frac, atol=1e-6)):
            problems.append("a floor-system row does not read floor(floor_r(C) C) rows of its tier 2 per KV head "
                            "(or a unit lacks it)")
    q = d[d.family == "quest"]
    if len(q):
        st = q.arm.map(lambda a: float(L.parse_arm(a)["store"])).to_numpy()
        if not (np.allclose(q.stored_bits_per_token, st) and (q.stored_evict_frac == 0).all()):
            problems.append("a quest row does not store its width dense")
        if "quest_read_frac_all" not in q or not (np.abs(q.quest_read_frac_all - q.B) <= QUEST_TOL).all():
            problems.append(f"a quest row reads more than {QUEST_TOL} from r over all layers")
    lb = d[d.task == T.LBV2]
    if len(lb):
        if "fc_correct" not in lb or lb.fc_correct.isna().any():
            problems.append("lbv2 rows lack the forced-choice columns")
        else:
            fp = lb[lb.arm == "fp"]
            agree = float((fp.fc_choice == fp.pred.str.strip().str[:1]).mean())
            if agree < FC_AGREE_MIN:
                problems.append(f"FP's forced choice equals its greedy letter on {agree:.2f} < {FC_AGREE_MIN}")
        if "vote_rows" not in lb or not (lb.vote_rows == T.VOTE_ROWS).all():
            problems.append(f"an lbv2 row does not record a {T.VOTE_ROWS}-row vote span")
    return out


def add_bytes_h4(v):
    """R2's bytes with the floor rows at their own r (the fetch depends on it)."""
    x = _ORIG_BYTES(_with_floor_r(v))
    x["B"] = v["B"].to_numpy()
    return x


R1R.L = RR2.L = L          # R4's arm grammar (quest, the floor arm, closedbook) for R1's and R2's functions
R1R.validate_h = validate_h4
R1R.add_bytes_h = add_bytes_h4
R1R.FP_MIN = 0.0


# ------------------------------------------------------------------ helpers
def analysis_frame(d):
    """score := the unit's accuracy (lbv2: fc_correct; the greedy score kept as score_greedy);
    prompt_idx made unique per (task, item), so every unit is its own cluster."""
    x = d.copy()
    x["score_greedy"] = x["score"]
    m = (x.task == T.LBV2).to_numpy()
    if m.any():
        x.loc[m, "score"] = x.loc[m, "fc_correct"].astype(float)
    x["prompt_idx"] = [T.TASKS.index(t) * 100000 + int(p) for t, p in zip(x.task, x.prompt_idx)]
    x["fam"] = x.task.map(T.FAMILY)
    return x


def fc_kl(d):
    """Per arm, the mean KL of FP's renormalized A/B/C/D distribution to the arm's (lbv2)."""
    x = d[d.task == T.LBV2]
    if not len(x) or "fc_lp" not in x:
        return {}

    def p(lp):
        a = np.asarray(list(lp), dtype=float)
        a = np.exp(a - a.max())
        return a / a.sum()

    fp = {k: p(v) for k, v in x[x.arm == "fp"].set_index(R1R.KEY).fc_lp.items()}
    out = {}
    for (arm, B), g in x[x.arm != "fp"].groupby(["arm", "B"]):
        kl = [float(np.sum(fp[k] * (np.log(fp[k] + 1e-12) - np.log(p(v) + 1e-12))))
              for k, v in g.set_index(R1R.KEY).fc_lp.items() if k in fp]
        out[name((arm, B))] = round(float(np.mean(kl)), 4) if kl else float("nan")
    return out


def _boot(SC):
    pidx = SC.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1), pidx).to_numpy()  # noqa: E731
    return lambda s: L1C.boot_ci(pm(s), W)  # noqa: E731


def gold_logp(x):
    """lbv2: per arm, the change in the gold choice's log-probability vs FP, with its interval."""
    v = x[x.task == T.LBV2]
    if not len(v) or "fc_gold_logp" not in v:
        return {}
    G = v.pivot_table(index=R1R.KEY, columns=["arm", "B"], values="fc_gold_logp", aggfunc="first")
    boot = _boot(G)
    return {name(c): boot(G[c] - G[FPC]) for c in G.columns if c != FPC}


def analyse_acc_r4(d, units=None):
    """R3a's accuracy labels without its FP floor, plus R4's, on all units or a subset."""
    SC = d.pivot_table(index=R1R.KEY, columns=["arm", "B"], values="score", aggfunc="first")
    if units is not None:
        SC = SC[SC.index.isin(units)]
    if len(SC) < 2:
        return dict(n_units=int(len(SC)), labels={}, arms={})
    boot = _boot(SC)
    fpok = SC[FPC] >= 1
    arms = {}
    for c in SC.columns:
        dz = SC[c] - SC[FPC]
        arms[name(c)] = dict(acc=boot(SC[c]), dacc=boot(dz),
                             dacc_fp_failed=round(float(dz[~fpok].mean()), 3) if (~fpok).any() else float("nan"),
                             by_task={t: round(float(g.mean()), 3) for t, g in SC[c].groupby(level=-1)})
    out = dict(fp=round(float(SC[FPC].mean()), 3), n_units=int(len(SC)), n_fp_failed=int((~fpok).sum()), arms=arms,
               labels={})

    def near(c, nm):
        if c in SC.columns:
            z = boot(SC[c] - SC[FPC])
            out["labels"][nm] = dict(diff=z, a=name(c), b="fp@0",
                                     label="ACC_NEAR_FP" if z[1] >= -ACC_MARGIN else
                                     ("ACC_LOSS" if z[2] < -ACC_MARGIN else "ACC_INCONCLUSIVE"))

    def lab(a, b, nm):
        if a in SC.columns and b in SC.columns:
            z = boot(SC[a] - SC[b])
            out["labels"][nm] = dict(diff=z, label=RR3._acc_label(z, nm), a=name(a), b=name(b),
                                     equiv="EQUIV" if (z[1] >= -ACC_MARGIN and z[2] <= ACC_MARGIN) else "NOT_EQUIV")

    near(SYSTEM, "ACC_SYSTEM")
    near(FLOOR, "ACC_FLOOR")
    near(QUEST, "ACC_QUEST")
    lab(SYSTEM, QUEST, "SYS_VS_QUEST_ACC")
    lab(FLOOR, QUEST, "FLOOR_VS_QUEST_ACC")
    lab(SIMPLE, QUEST4, "SCHEDULE_ACC")
    dm = {c: float(SC[c].mean()) for c in DENSE4 if c in SC.columns}
    best = max(dm, key=lambda c: (dm[c], c == DENSE4[0])) if dm else None
    out["best_dense_acc"] = name(best) if best else None
    if best:
        lab(SYSTEM, best, "SYS_VS_BEST_ACC")
    lab(SYSTEM, DV4, "SYS_VS_D_ACC")
    lab(("qread4q_v4", 0.125), SIMPLE, "REQ1_ACC")
    lab(("qreadfp_v16", 0.125), ("qoraclefp_v16", 0.125), "VOTE_LOSS_ACC")
    return out


def ctx_units(x):
    """Units where FP's accuracy minus the closed-book arm's is >= CTX_MIN."""
    S = x.pivot_table(index=R1R.KEY, columns=["arm", "B"], values="score", aggfunc="first")
    if CB not in S.columns:
        return S.index[:0]
    return S.index[(S[FPC] - S[CB]) >= CTX_MIN]


def nll_labels_r4(x):
    P, _, boot, _ = RR2._boot_frame(x)
    out = {}
    for a, b, nm in ((SYSTEM, QUEST, "SYS_VS_QUEST"), (SIMPLE, QUEST4, "SCHEDULE"), (FLOOR, SYSTEM, "FLOOR_VS_FIXED")):
        if a in P.columns and b in P.columns:
            z = boot(P[a] - P[b])
            out[nm] = dict(diff=z, label=L.effect_label(*z, nm), a=name(a), b=name(b))
    return out


def load_cell(cell, jobs, root, problems):
    c = L.CELLS[cell]
    parts = [R1R.load_run(c["preset"], j, root) for j in jobs]
    d = pd.concat([p[0] for p in parts], ignore_index=True)
    sides = [p[1] for p in parts]
    want_cell = L.manifest_cell(c["preset"], c["suite"])
    man = T.manifest()["cells"][want_cell]
    for s in sides:
        w = f"{cell} block {s['_job']}"
        if s.get("stage") != "1h" or s.get("amend_r4") != L.AMEND_R4 or s.get("preset_name") != c["preset"]:
            problems.append(f"{w}: not an {L.AMEND_R4} block of preset {c['preset']}")
        if s.get("suite") != c["suite"] or s.get("manifest_cell") != want_cell or s.get("smoke"):
            problems.append(f"{w}: suite / manifest cell {s.get('suite')} / {s.get('manifest_cell')}, "
                            f"expected {c['suite']} / {want_cell}")
        if s.get("manifest_sha256") != T.MANIFEST_SHA256:
            problems.append(f"{w}: manifest {str(s.get('manifest_sha256'))[:12]}, not the frozen one")
        if R1R.plan_of(s) != sorted((a, L.norm_b(b)) for a, b in L.build_plan(L.PRESETS[c["preset"]])):
            problems.append(f"{w}: plan is not the preset's")
    for (p, task), g in d[d.arm == "fp"].groupby(["prompt_idx", "task"]):
        ids = man.get(task, [])
        if not (0 <= p < len(ids)) or g.corpus_doc.iloc[0] != ids[p][0]:
            problems.append(f"{cell}: unit {task} {p} is not the manifest's item")
            break
    return d, sides


# ------------------------------------------------------------------ the gate
def read_gate(cell, job, root=RESULTS, out_dir=FINDINGS):
    c = L.CELLS[cell]
    problems = []
    d, side = R1R.load_run(c["preset"], job, root)
    if not (side.get("amend_r4") == L.AMEND_R4 and side.get("preset_name") == c["preset"] and not side.get("smoke")):
        problems.append("not an R4b pilot of the cell's preset")
    R1R.validate_h(d, [side], problems, main=False)
    fp = d[d.arm == "fp"]
    if not (fp.kl_all.fillna(0) == 0).all():
        problems.append("FP's KL to itself is not 0")
    degen, n = R1R.noise_degenerate(d)
    if degen:
        problems.append(f"NOISE_DEGENERATE on {n} units")
    peaks = [float(x) for x in side.get("peak_gib_dev_max") or []]
    if not peaks or max(peaks) > L.GATE_PEAK_GIB:
        problems.append(f"peak GPU memory per device {peaks} GiB (limit {L.GATE_PEAK_GIB})")
    per_unit = {}
    for (p, task), g in d.groupby(["prompt_idx", "task"]):
        own = g["t_own"].fillna(0) if "t_own" in g else 0.0
        per_unit[task] = float(g.t_prefill.iloc[0] + g.t_precompute.fillna(0).iloc[0]
                               + (g.t_arm.fillna(0) + g.t_tf.fillna(0) + own).sum())
    kind = lambda t: "rerank" if T.FAMILY[t] == "rerank" else "short"  # noqa: E731
    by_kind = {}
    for t, s in per_unit.items():
        by_kind.setdefault(kind(t), []).append(s)
    by_kind = {k: float(np.mean(v)) for k, v in by_kind.items()}
    block_s = sum(n_ * by_kind.get(kind(t), max(by_kind.values())) for t, n_ in L.block_units(cell).items())
    hh, mm, ss = (int(x) for x in c["wall"].split(":"))
    wall_h = hh + mm / 60 + ss / 3600
    proj_h = block_s / 3600 + GATE_SETUP_H
    if proj_h > L.GATE_WALL_FRAC * wall_h:
        problems.append(f"projected block {proj_h:.2f} h > {L.GATE_WALL_FRAC} x {wall_h:.1f} h")
    verdict = "PASS" if not problems else "FAIL"
    os.makedirs(out_dir, exist_ok=True)
    out = dict(cell=cell, job=str(job), verdict=verdict, problems=problems, peak_gib_dev=peaks,
               seconds_per_unit=per_unit, projected_block_h=round(proj_h, 2), wall_h=wall_h)
    with open(os.path.join(out_dir, f"R4_gate_{cell}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if verdict == "PASS" else 1


# ------------------------------------------------------------------ the read
def read_r4(jobs_by_cell, out_stem, root=RESULTS, r1_json=RR2.R1_JSON):
    results, summary, problems = {}, {}, []
    for cell, jobs in jobs_by_cell.items():
        d, sides = load_cell(cell, jobs, root, problems)
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{cell}: blocks ran different plans")
        R1R.validate_h(d, sides, problems)
        R1R.validate_a2_h(d, problems, cell, self_check=False)
        if problems:
            continue
        x = analysis_frame(d)
        cres = dict(cell=cell, manifest_cell=sides[0].get("manifest_cell"), families={})
        for fam, xf in x.groupby("fam"):
            res = RR3.analyse_nll(xf, sides, problems, f"{cell}/{fam}", r1_json)
            if res is None:
                continue
            ctx = ctx_units(xf)
            res["accuracy"] = analyse_acc_r4(xf)
            res["accuracy_ctx"] = analyse_acc_r4(xf, ctx)
            res["n_ctx"] = int(len(ctx))
            res["closedbook_acc"] = round(float(xf[xf.arm == L.CLOSEDBOOK].score.mean()), 3)
            res["labels_r4"] = nll_labels_r4(xf)
            if fam == "lbv2":
                res["fc_kl"] = fc_kl(xf)
                res["gold_logp"] = gold_logp(xf)
                res["greedy_acc"] = {name(k): round(float(g.score_greedy.mean()), 3) for k, g in xf.groupby(["arm", "B"])}
            q = xf[xf.family == "quest"]
            res["quest_read"] = {a: round(float(g.quest_read_frac_all.mean()), 4) for a, g in q.groupby("arm")}
            fl = xf[xf.arm == L.FLOOR_ARM]
            res["floor_r"] = dict(mean=round(float(fl.floor_r.mean()), 3), min=round(float(fl.floor_r.min()), 3),
                                  max=round(float(fl.floor_r.max()), 3)) if len(fl) else {}
            cres["families"][fam] = res
            a = res["accuracy"]
            key = f"{cell}/{fam}"
            summary[f"FP accuracy {key}"] = (f"{a.get('fp')} ({a['n_units']} units, {a.get('n_fp_failed')} FP-failed; "
                                             f"closed-book {res['closedbook_acc']}, {res['n_ctx']} need the context)")
            for k, z in a["labels"].items():
                summary[f"{k} {key}"] = (z["label"] + (f", {z['equiv']}" if z.get("equiv") else "")
                                         + f" ({ci(z['diff'])})")
            for k, z in res["accuracy_ctx"]["labels"].items():
                summary[f"{k}_CTX {key}"] = z["label"] + f" ({ci(z['diff'])})"
            for k, z in res["labels_r4"].items():
                summary[f"{k} {key}"] = f"{z['label']} ({ci(z['diff'])})"
            sk = name(SYSTEM)
            if sk in res["points"]:
                summary[f"SYSTEM NLL {key}"] = RR3.system_nll_line(res, sk)
        results[cell] = cres
    if problems:
        raise SystemExit("INVALID Stage 1h R4 data:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(margin="per cell and family: s1h_lib.margin_fp(hi90 fp8kv dP)",
                                  acc_margin=ACC_MARGIN, acc_eps=ACC_EPS, ctx_min=CTX_MIN, quest_tol=QUEST_TOL,
                                  manifest_sha256=T.MANIFEST_SHA256), summary=summary, cells=results),
                  fh, indent=1, default=str)
    Lh = ["# R14 Stage 1h — R4, real tasks at 128K (read_stage1h_r4.py; rules frozen in its docstring)", "",
          "EXPLORATORY run: labels guide the design; they are not claims. lbv2 accuracy is the forced choice.", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for cell, cres in results.items():
        for fam, res in cres["families"].items():
            a = res["accuracy"]
            Lh += [f"## {cell} / {fam} — FP {a.get('fp')}; {a['n_units']} units ({a.get('n_fp_failed')} FP-failed, "
                   f"{res['n_ctx']} need the context); margin {res['m_cell']:.3f}", "",
                   "| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |", "|---|---|---|---:|---|"]
            Lh += [f"| {k} | {ci(z['acc'])} | {ci(z['dacc'])} | {z['dacc_fp_failed']} | {z['by_task']} |"
                   for k, z in sorted(a["arms"].items(), key=lambda kz: -kz[1]["acc"][0])]
            Lh.append("")
            for k, z in a["labels"].items():
                Lh.append(f"**{k}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}"
                          + (f", {z['equiv']}" if z.get("equiv") else ""))
            for k, z in res["labels_r4"].items():
                Lh.append(f"**{k}** (NLL, {z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}")
            Lh += ["", f"Quest read fraction (all layers): {res['quest_read']}; floor r: {res['floor_r']}"]
            if res.get("fc_kl"):
                Lh.append(f"4-choice KL to FP: {res['fc_kl']}")
                Lh.append("Δ gold log-prob vs FP: " + ", ".join(f"{k} {ci(z)}" for k, z in res["gold_logp"].items()))
                Lh.append(f"Greedy-letter accuracy: {res['greedy_acc']}")
            Lh += ["", "NLL / KL points:", ""] + R1R._arm_table(res["points"]) + [""]
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh[:4 + len(summary) + 1]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", nargs=2, metavar=("CELL", "JOB"))
    for c in L.CELLS:
        ap.add_argument(f"--{c}", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--r1-json", default=RR2.R1_JSON)
    ap.add_argument("--out-stem", default=os.path.join(FINDINGS, "R4_reader"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.gate:
        if a.gate[0] not in L.CELLS:
            ap.error(f"--gate CELL must be one of {list(L.CELLS)}")
        sys.exit(read_gate(a.gate[0], a.gate[1], a.results_root))
    jobs = {c: getattr(a, c) for c in L.CELLS if getattr(a, c)}
    if not jobs:
        ap.error(f"give --gate CELL JOB, or at least one cell's jobs: {list(L.CELLS)}")
    sys.exit(read_r4(jobs, a.out_stem, a.results_root, a.r1_json))


if __name__ == "__main__":
    main()
