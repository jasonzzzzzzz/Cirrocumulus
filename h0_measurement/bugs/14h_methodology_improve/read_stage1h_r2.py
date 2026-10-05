#!/usr/bin/env python3
"""R14 Stage 1h gate and reader for R2 (Qwen3-30B-A3B at 32K). The rules below are
FROZEN: written 2026-10-04, before any R2 output existed (plan.md, R2 amendment).
Design: s1h2_lib.py. Driver: run_s1h2.py. Everything not stated here is R1's rule
(read_stage1h.py's docstring), applied through read_stage1h's functions, which this
reader runs with s1h2_lib's arm grammar (in this process only).

    python read_stage1h_r2.py --pilot JOB [--pilot-tag h2pilot]
    python read_stage1h_r2.py --r2 J1 J2 [J3 ...] --r2-seeds S0 S1 S2 [--r1-json .../findings/R1_reader.json]

CELLS
  'r2':       h2qwen32, prompts 9300-9319 (2 blocks of 10), more blocks only by SEQUENTIAL;
              all four tasks; rotation seed 0. EXPLORATORY.
  'r2seeds':  h2regress on s1h2_lib.REGRESS_R2 (8234 multikey, 8830 vt, 8215 / 8218 / 8235
              multivalue) at rotation seeds 0, 1, 2.
METRICS, UNITS, INTERVALS, R1-STYLE LABELS: as R1 (dP = a_span_nll primary, dK = kl_span
  co-primary; NEAR_FP / EQUIV at 0.10; MATCHED vs D_L; NOISE; FP8_COST; BEST_DENSE;
  VOTE_LOSS_EXACT / VOTE_LOSS_4; EXACT_KV; LOST with McNemar and types; FP-FAILED).
MARGIN   m_FP is R1's (read from R1's reader output, r1.m_fp). Every point also gets
  NEAR_FP and EQUIV at m_FP. R2 is INVALID without R1's read.
STOP RULE eos_only (Qwen; s1h2_lib's presets), in place of R1's r8.
R2 LABELS (r = 1/8 unless named; effect labels: |mean| >= 0.05 nats and a 90% interval
  excluding 0)
  SYSTEM        qread2t4kq_v4@1/8: NEAR_FP and EQUIV at m_FP, MATCHED vs D_V4.
  SYS_VS_DENSE4 system minus each same-memory dense 4-bit arm (uniform+v4@4, kivi4_v4@4,
                kvquant4_v4@4), effect labels; BEST_DENSE_R2 = s1h_lib.best_dense over
                them in R2 (model-matched, the strongest baseline); SYS_VS_BEST = system
                minus BEST_DENSE_R2 (_HELPS = the system beats the strongest dense 4-bit).
  REQ1          qread4q_v4 - qread4_v4 (the single-tier second question pass; _HELPS).
  STORE4        qread4_v4 - qread_v4 (4- vs 3-bit store, single tier).
  SYS_VS_TT3    system - qread2t_v4 (Stage 1g's fixes vs Stage 1f's 3-bit two-tier read).
  FLOOR_SYS / FLOOR_EXACT / FLOOR_4   the arm at r = 1/2 minus at 1/8 (does Qwen need
                the floor?).
  SEQUENTIAL    metrics_s1h.sequential_step on the system's mean dP interval vs m_FP, with
                the number of prompts read (cap 60): PASS / FAIL / ADD (submit one more
                10-prompt block, 9320 on) / CAP.
  FIX           r2seeds, per unit and arm of s1h2_lib.FIX_ARMS: s1h2_lib.fix_label against
                the failing reference (qread2t_v4@1/8 for 8830, D_V4 otherwise):
                FIXED / NOT_FIXED / NOT_REPRODUCED. VOTE_LOSES_NEEDLE on 8234 if the system
                is NOT_FIXED there while qreadfp_v16 is FIXED.
REPORTED, NOT LABELLED: BRIDGE_1F, per regression unit at seed 0, dP of the arms shared
  with Stage 1f (s1h2_lib.BRIDGE_1F_ARMS) beside Stage 1f's (jobs 1022862, 1022861); the
  system vs R1's BEST_DENSE choice.
VALIDITY: R1's, with the stop rule eos_only; qreadq rows store their width dense and read
  floor(r C) per KV head; r2seeds three distinct rotation seeds with r2's plan.
GATE (pilot, excluded): R1's gate with h2pilot -> h2qwen32 (40 units per block, 3-hour
  limit).
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import read_stage1h as R1R  # noqa: E402
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import read_stage1f as RF  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h2_lib as L  # noqa: E402
import metrics_s1h as M  # noqa: E402

RESULTS = R1R.RESULTS
WALL_H2, BLOCK_UNITS2 = 3.0, 40
R1_JSON = os.path.join(HERE, "findings", "R1_reader.json")
SEQ_CAP = 60
EFFECT_EPS = 0.05
SYSTEM = (L.SYSTEM + "_v4", 0.125)
DENSE4 = R1R.DENSE4
DV4 = R1R.DV4
name, ci, fmt = R1R.name, R1C.ci, R1C.fmt

# read_stage1h's functions with R2's grammar, stop rule and bytes (this process only)
_ORIG_VALIDATE, _ORIG_BYTES = R1R.validate_h, R1R.add_bytes_h


def validate_h2(d, sides, problems, main=True):
    """R1's validity with each block's own preset stop rule, plus the qreadq audit."""
    ss = []
    for s in sides:
        stop = L.PRESETS.get(s.get("preset_name"), {}).get("stop", "r8")
        if s.get("stop_rule") != stop:
            problems.append(f"block {s.get('_job')}: stop rule {s.get('stop_rule')}, expected {stop}")
        ss.append(dict(s, stop_rule="r8"))
    out = _ORIG_VALIDATE(d, ss, problems, main)
    pas = d.arm.map(L.parse_arm)
    m = (pas.map(lambda p: p["family"]) == "qreadq").to_numpy()
    if m.any():
        x = d[m]
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(x.B, x.ctx_len)])
        st = pas[m].map(lambda p: float(p["store"])).to_numpy()
        if not (np.allclose(x.stored_bits_per_token, st) and (x.stored_evict_frac == 0).all()
                and np.allclose(x.read_frac, k, atol=1e-9) and np.allclose(x.bits_per_token, st * x.read_frac,
                                                                            atol=1e-6)):
            problems.append("a qreadq row does not store its width dense and read floor(r C) per KV head")
    return out


def add_bytes_h2(v):
    """R1's bytes; a qreadq row's GPU-stored keys carry every norm, as qread's."""
    v = _ORIG_BYTES(v)
    m = (v.arm.map(lambda a: L.parse_arm(a)["family"]) == "qreadq").to_numpy()
    if m.any():
        x = v[m]
        hd = x.head_dim.to_numpy(dtype=float)
        tail = ((x.window + x.n_question_tokens + x.gen_len / 2.0) * 4 * x.head_dim / x.ctx_len).to_numpy(dtype=float)
        sto = hd / 8 * (x.stored_bits_per_token.to_numpy(dtype=float) + 16.0 / hd) \
            + (1 - x.stored_evict_frac.to_numpy(dtype=float)) * hd / 8 * (x.v_bits.to_numpy(dtype=float)
                                                                       + x.v_side.to_numpy(dtype=float))
        v.loc[m, "bytes_stored"] = sto + tail
        v.loc[m, "bytes_amort"] = v.loc[m, "bytes"] + sto / v.loc[m, "fp_gen_len"].clip(lower=1)
    return v


R1R.L = L
R1R.validate_h = validate_h2
R1R.add_bytes_h = add_bytes_h2
R1R.WALL_H["h2qwen32"] = WALL_H2
R1R.BLOCK_UNITS["h2qwen32"] = BLOCK_UNITS2


# ---------------------------------------------------------------- analysis
def _boot_frame(d):
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= R1R.FP_MIN)
    v = R1R.add_bytes_h(d[d.task.isin(valid)])
    P, S, A, K, SC = R1R._tables(v)
    pidx = P.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1) if s.index.nlevels > 2 else s, pidx).to_numpy()  # noqa: E731
    return P, SC, (lambda s: L1C.boot_ci(pm(s), W)), pidx


def analyse_r2(d, sides, m_fp, r1_best):
    res = R1R.analyse_r1(d, sides)
    res["cell"], res["m_fp"], res["m_fp_source"] = "r2", m_fp, "R1 (Llama-3.1-8B, 128K)"
    res["qwen_fp8_margin"] = L.margin_fp(res["fp8"]["dP"][2]) if res.get("fp8") else None
    P, SC, boot, pidx = _boot_frame(d)
    for k, p in res["points"].items():
        p["vs_FP_m"] = dict(label=L.near_fp_label(p["vs_FP"]["nll"], p["vs_FP"]["tail"], m_fp),
                            equiv=L.equiv_label(p["vs_FP"]["nll"], m_fp))
    lab = res.setdefault("labels_r2", {})

    def diff(a, b, name_):
        if a not in P.columns or b not in P.columns:
            return
        z = boot(P[a] - P[b])
        lab[name_] = dict(diff=z, label=L.effect_label(*z, name_), a=name(a), b=name(b))

    e, f = 0.125, float(L.R2_FLOOR)
    for c in DENSE4:
        diff(SYSTEM, c, f"SYS_VS_{c[0].upper().replace('+', '_')}")
    dm = {name(c): float(P[c].mean()) for c in DENSE4 if c in P.columns}
    best = L.best_dense(dm) if dm else None
    res["best_dense_r2"] = dict(means=dm, best=best)
    if best:
        bc = next(c for c in DENSE4 if name(c) == best)
        diff(SYSTEM, bc, "SYS_VS_BEST")
    if r1_best:
        rc = next((c for c in DENSE4 if name(c) == r1_best), None)
        if rc is not None:
            diff(SYSTEM, rc, "SYS_VS_R1_BEST")
    diff(("qread4q_v4", e), ("qread4_v4", e), "REQ1")
    diff(("qread4_v4", e), ("qread_v4", e), "STORE4")
    diff(SYSTEM, L.TT3, "SYS_VS_TT3")
    diff((SYSTEM[0], f), SYSTEM, "FLOOR_SYS")
    diff(("qreadfp_v16", f), ("qreadfp_v16", e), "FLOOR_EXACT")
    diff(("qread4_v4", f), ("qread4_v4", e), "FLOOR_4")
    if name(SYSTEM) in res["points"]:
        z = res["points"][name(SYSTEM)]
        nll = z["vs_FP"]["nll"]
        res["sequential"] = dict(step=M.sequential_step(nll[1], nll[2], m_fp, len(pidx), SEQ_CAP),
                                 n_prompts=int(len(pidx)), margin=m_fp, nll=nll)
    return res


def analyse_fix(rs):
    """FIX across rotation seeds (module docstring)."""
    per_seed = {}
    for seed, g in rs.groupby("rot_seed"):
        Ps = g.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=R1R.PRIMARY, aggfunc="first")
        Ps = Ps[~Ps[("fp", 0.0)].isna()]
        per_seed[int(seed)] = Ps.sub(Ps[("fp", 0.0)], axis=0)
    units = sorted(set().union(*[set(x.index) for x in per_seed.values()]))
    labels, table = {}, {}
    for u in units:
        ref = L.FIX_REF.get(int(u[0]), DV4)
        ref_by = {s: float(Ps.loc[u, ref]) for s, Ps in per_seed.items() if u in Ps.index and ref in Ps.columns}
        key = f"{u[0]}/{u[1]}"
        labels[key] = {}
        for c in L.FIX_ARMS:
            arm_by = {s: float(Ps.loc[u, c]) for s, Ps in per_seed.items() if u in Ps.index and c in Ps.columns}
            labels[key][name(c)] = L.fix_label(arm_by, ref_by)
        table[key] = {name(c): {s: round(float(Ps.loc[u, c]), 2) for s, Ps in per_seed.items()
                                if u in Ps.index and c in Ps.columns}
                      for c in [DV4, L.TT3] + L.FIX_ARMS}
        table[key]["_ref"] = name(ref)
    vln = None
    k8234 = next((k for k in labels if k.startswith("8234/")), None)
    if k8234:
        z = labels[k8234]
        vln = (z.get(name(SYSTEM)) == "NOT_FIXED" and z.get(name(("qreadfp_v16", 0.125))) == "FIXED")
    return dict(labels=labels, dP=table, seeds=sorted(per_seed), vote_loses_needle_8234=vln)


def bridge_1f(rs, root=RESULTS):
    """BRIDGE_1F (reported): R2's seed-0 regression units beside Stage 1f's."""
    r0 = rs[rs.rot_seed == 0]
    if not len(r0):
        return {}
    Pr = r0.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=R1R.PRIMARY, aggfunc="first")
    Pr = Pr.sub(Pr[("fp", 0.0)], axis=0)
    parts = []
    for tag, job in L.BRIDGE_1F:
        try:
            parts.append(R1R.load_run(tag, job, root, stage="s1f")[0])
        except SystemExit:
            continue
    if not parts:
        return {}
    g = pd.concat(parts, ignore_index=True)
    Pg = g.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=R1R.PRIMARY, aggfunc="first")
    Pg = Pg.sub(Pg[("fp", 0.0)], axis=0)
    out = {}
    for u in Pr.index:
        if u not in Pg.index:
            continue
        out[f"{u[0]}/{u[1]}"] = {name(c): (round(float(Pr.loc[u, c]), 3), round(float(Pg.loc[u, c]), 3))
                                 for c in L.BRIDGE_1F_ARMS if c in Pr.columns and c in Pg.columns}
    return out


# ------------------------------------------------------------------ report
def report(res, fix, br, summary, out_stem):
    Lh = ["# R14 Stage 1h — R2, Qwen3-30B-A3B at 32K (read_stage1h_r2.py; rules frozen in its docstring)", "",
          "EXPLORATORY run: labels guide the design; they are not claims.", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    Lh += [f"## r2 — FP {res['fp']:.3f} ({res['fp_by_task']}), {res['n_prompts']} prompts, {res['n_units']} units; "
           f"m_FP {res['m_fp']:.3f} (from R1)", ""]
    Lh += R1R._arm_table(res["points"]) + [""]
    Lh += ["| arm@B | NEAR_FP at m_FP | EQUIV at m_FP |", "|---|---|---|"]
    Lh += [f"| {k} | {p['vs_FP_m']['label']} | {p['vs_FP_m']['equiv']} |" for k, p in sorted(res["points"].items())]
    Lh.append("")
    for k, z in {**res["labels"], **res["labels_r2"]}.items():
        Lh.append(f"**{k}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}")
    Lh.append("")
    if fix:
        Lh += [f"## Regression units across rotation seeds {fix['seeds']} (dP vs FP)", ""]
        for u, z in fix["labels"].items():
            Lh.append(f"- {u} (reference {fix['dP'][u]['_ref']}): " + ", ".join(f"{a} **{lab}**" for a, lab in z.items()))
        Lh.append("")
        for u, tab in fix["dP"].items():
            Lh.append(f"- {u}: " + "; ".join(f"{a} {v}" for a, v in tab.items() if a != "_ref" and v))
        Lh.append("")
    if br:
        Lh += ["## Beside Stage 1f (seed 0; dP R2, dP Stage 1f)", ""]
        Lh += [f"- {u}: {z}" for u, z in br.items()] + [""]
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def read_r2(jobs, seed_jobs, out_stem, root=RESULTS, r1_json=R1_JSON):
    problems, summary = [], {}
    if not os.path.exists(r1_json):
        raise SystemExit(f"INVALID R2 read: R1's reader output {r1_json} is missing (m_FP comes from R1)")
    r1 = json.load(open(r1_json))["r1"]
    m_fp = float(r1["m_fp"])
    r1_best = (r1.get("best_dense") or {}).get("best")
    parts = [R1R.load_run("h2qwen32", j, root) for j in jobs]
    d = pd.concat([p[0] for p in parts], ignore_index=True)
    sides = [p[1] for p in parts]
    if len({json.dumps(s["plan"]) for s in sides}) != 1:
        problems.append("r2: blocks ran different plans")
    agree, cover = R1R.validate_h(d, sides, problems)
    R1R.validate_a2_h(d, problems, "r2")
    rs = None
    if seed_jobs:
        sp = [R1R.load_run("h2regress", j, root) for j in seed_jobs]
        rs = pd.concat([p[0] for p in sp], ignore_index=True)
        R1R.validate_h(rs, [p[1] for p in sp], problems, main=False)
        R1R.validate_a2_h(rs, problems, "r2seeds", self_check=False)
        seeds_ = [int(p[1].get("rot_seed", -1)) for p in sp]
        if len(set(seeds_)) != len(seeds_):
            problems.append(f"r2seeds: rotation seeds are not distinct: {seeds_}")
        if any(R1R.plan_of(p[1]) != R1R.plan_of(sides[0]) for p in sp):
            problems.append("r2seeds ran another plan than r2")
    if problems:
        raise SystemExit("INVALID Stage 1h R2 data:\n  " + "\n  ".join(problems))
    res = analyse_r2(d, sides, m_fp, r1_best)
    res["fp_replay_agreement"], res["value_cover"] = agree, cover
    fix = analyse_fix(rs) if rs is not None else None
    br = bridge_1f(rs, root) if rs is not None else {}
    sk = name(SYSTEM)
    if sk in res["points"]:
        p = res["points"][sk]
        summary["SYSTEM"] = (f"vs FP at m_FP {res['m_fp']:.3f}: {p['vs_FP_m']['label']} ({ci(p['vs_FP']['nll'])}), "
                             f"{p['vs_FP_m']['equiv']}; vs D_V4 {p['vs_D']['label']}; lost {p['lost']['n']}")
    for k, z in {**res["labels"], **res["labels_r2"]}.items():
        summary[k] = z["label"]
    summary["BEST_DENSE_R2"] = res["best_dense_r2"]["best"]
    if res.get("sequential"):
        summary["SEQUENTIAL"] = f"{res['sequential']['step']} after {res['sequential']['n_prompts']} prompts"
    if res.get("noise"):
        summary["NOISE"] = res["noise"]["label"]
    if res.get("fp8"):
        summary["FP8_COST (Qwen)"] = f"dP {ci(res['fp8']['dP'])}"
    if fix:
        for u, z in fix["labels"].items():
            summary[f"FIX {u} system"] = z.get(sk, "NO_DATA")
        if fix["vote_loses_needle_8234"] is not None:
            summary["VOTE_LOSES_NEEDLE 8234"] = bool(fix["vote_loses_needle_8234"])
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(m_fp=m_fp, m_fp_source=r1_json, effect_eps=EFFECT_EPS, seq_cap=SEQ_CAP,
                                  fix_nats=L.FIX_NATS, primary=R1R.PRIMARY, kl=R1R.KL),
                       summary=summary, r2=res, fix=fix, bridge_1f=br), fh, indent=1, default=str)
    Lh = report(res, fix, br, summary, out_stem)
    print("\n".join(Lh[:4 + len(summary) + 1]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--pilot-tag", default="h2pilot", choices=["h2pilot"])
    ap.add_argument("--r2", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--r2-seeds", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--r1-json", default=R1_JSON)
    ap.add_argument("--out-stem", default=os.path.join(HERE, "findings", "R2_reader"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(R1R.gate(a.pilot, a.pilot_tag, a.results_root))
    if not a.r2:
        ap.error("give --pilot JOB or --r2 JOBS")
    sys.exit(read_r2(a.r2, a.r2_seeds, a.out_stem, a.results_root, a.r1_json))


if __name__ == "__main__":
    main()
