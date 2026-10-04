#!/usr/bin/env python3
"""R14 Stage 1g gate and reader. The rules below are FROZEN: written 2026-10-03,
before any Stage 1g output existed. Design: s1g_lib.py. Driver: run_s1g.py. The
time rules are in s1g_kernel.py, read by bench_s1g_kernel.py.

    python read_stage1g.py --pilot JOB [--pilot-tag gpilot]
    python read_stage1g.py --g128 J1 J2 J3 J4 --g32 J5 J6 J7 J8 [--gregress JR] --out-stem .../stage1g

Result directories: h0_measurement/results/r14s1g_<tag>_<job>/ (files s1g_*).

CELLS
  'g128': Llama-3.1-8B @131072, prompts 9100-9139, 4 blocks of 10, all four tasks.
  'g32':  @32768, prompts 9200-9239.
  'gregress128': g128's plan on Stage 1e-1f's TurboQuant-3 key confusions at 128K
          (8109, 8901, 8937; niah_multikey).
METRIC   Stage 1f's amended statistic (A2): dS(X) = s_set_nll(X) - s_set_nll(fp): the NLL
         of every token of FP's answer from its first answer-value token to the span
         end, scored in the arm's own order when it states FP's values in another
         order. dA on a_sum_nll (Stages 1d-1e) is reported beside it. Prompt-tasks
         without an answer value drop (reported); task cells with FP < 0.9 drop.
COMPARATOR D_L = uniform@3 in X's value lens (a two-tier read: its tier-1 values'
         lens). D4_L = uniform@4 is reported.
INTERVALS Prompt bootstrap within block, 10,000 draws, seed 14; 90% intervals.
MATCHED  hi of mean(dS(X) - dS(D_L)) <= 0.10 and hi of the > 2-nat tail-share difference
         <= 0.05; WORSE if lo > 0.10 or tail lo > 0.05; else INCONCLUSIVE.
NEAR_FP  the same with FP as the comparator (s1g_lib.near_fp_label).
BYTES    per decode step, per context token, per KV head and layer:
         - dense, fp, single-tier reads (3- or 4-bit store), exact-store reads:
           Stage 1e's rule;
         - two-tier: s1g_lib.two_tier_bytes (tier-2 keys + the keep bitmap; values from
           tier 2, or keys-only, tier 1's v bits with their norm) + the tail;
         GPU-stored bytes (rho_mem) are tier 1's (3- or 4-bit keys with norms, values
         at v bits). Host bytes per context token and MB fetched per question and
         layer (s1g_lib.host_bytes, fetch_bytes) are reported.
LABELS (per cell; r = 1/8 unless named; effect labels: |mean| >= 0.05 nats and a 90%
         interval excluding 0, s1g_lib.effect_label)
  G1 QPASS3 = dS(qread2t_v16) - dS(qreadfp_v16)     (Stage 1f's QPASS, again)
     QPASS4 = dS(qread2t4_v16) - dS(qreadfp_v16)    (the question's pass over the 4-bit tier)
     TIER4  = dS(qread2t4_v4) - dS(qread2t_v4)
     STORE4 = dS(qread4_v4) - dS(qread_v4)          (single-tier reads, 4- vs 3-bit store)
  G2 KONLY  = dS(qread2tk_v4) - dS(qread2t_v4);   KONLY8 = dS(qread2tk8_v4) - dS(qread2t_v4)
  G3 REQ    = dS(qread2tq_v4) - dS(qread2t_v4)
  G5 (g32)  FLOOR = dS(qread2t_v4@1/2) - dS(qread2t_v4@1/8); qreadfp_v16@1/2's NEAR_FP label.
  SYSTEM    per cell, qread2t4kq_v4 at r_sys = s1g_lib.floor_r(C) (1/8 at 128K, 1/2 at
            32K): its MATCHED-rule label vs D_V4 and its NEAR_FP label;
            s1g_lib.system_label over both cells -> SYSTEM_NEAR_FP / SYSTEM_MATCHED /
            SYSTEM_FAILS.
  CONFUSION (gregress128): per arm, s1g_lib.confusion_label over the three prompts
            (FIXED / NOT_FIXED / NOT_REPRODUCED), dS per prompt reported.
REPORTED, NOT GATED: family verdicts (cheapest MATCHED point per family: WIN at rho <=
  0.80, TIE <= 1.00); answers below FP against D_L's (paired share interval); answers
  in another order (and replayed); dA beside dS; task scores; bytes.
VALIDITY (any failure -> INVALID)
  - Stage 1e's V1-V7 with Stage 1g's arms: every unit holds the plan's arms once; one
    corpus; FP >= 0.9 on >= 3 of 4 tasks; fp replay agreement >= 0.98; answer-value mask
    on >= 90% of correct FP answers; stop rule r8.
  - Audits: dense = w bits with nothing evicted; single-tier reads store their width (3
    or 4) dense and read floor(r C) per KV head; exact-store reads store 16 bits; two-tier
    rows store tier 1's width dense and read floor(r C) per KV head at the tier-2 key
    width; value twins changed their values.
  - Every block carries amend = A1-A4 and passes Stage 1f's A2 checks; in a main cell
    every block has a self-check replay for every arm NAME, each <= A2_CHECK_TOL nats.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import read_stage1e as RD  # noqa: E402
import read_stage1f as RF  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1g_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"g128": 6.0, "g32": 3.0}
BLOCK_UNITS = {"g128": 40, "g32": 40}
TAIL_NATS, FIXED_NATS, FP_MIN = RD.TAIL_NATS, RD.FIXED_NATS, RD.FP_MIN
DREF = RD.DREF
METRIC = L.METRIC_A2
bk, ci, fmt = R1C.bk, R1C.ci, R1C.fmt


# ------------------------------------------------------------------ loading
def load_run(tag, job, root=RESULTS):
    d = os.path.join(root, f"r14s1g_{tag}_{job}")
    pq = [x for x in glob.glob(os.path.join(d, "s1g_evaluate_*.parquet"))
          if not x.endswith(("_search.parquet", "_searchlog.parquet"))]
    js = glob.glob(os.path.join(d, "s1g_evaluate_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one s1g_evaluate parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    rows["q_role"] = ""
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


# ----------------------------------------------------------------- validity
def validate_g(d, sides, problems, main=True):
    key = ["job", "prompt_idx", "task"]
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(key[1:]).apply(lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))), include_groups=False)
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: units with the wrong arms: {bad[:3]}")
        if len(got) != len(side["prompt_tasks"]):
            problems.append(f"block {side['_job']}: {len(got)} units, expected {len(side['prompt_tasks'])}")
        if side.get("stop_rule") != "r8":
            problems.append(f"block {side['_job']}: stop rule {side.get('stop_rule')}, expected r8 (V7)")
        if side.get("amend") != L.AMEND:
            problems.append(f"block {side['_job']}: not run with amendment {L.AMEND} (A2 columns)")
    if d.duplicated(key + ["arm", "B"]).any():
        problems.append("duplicate rows")
    if d.corpus_sha.nunique() != 1:
        problems.append(f"blocks disagree on corpus: {sorted(d.corpus_sha.unique())}")
    pas = d.arm.map(L.parse_arm)
    fam = pas.map(lambda p: p["family"])
    dn = d[fam == "dense"]
    if not (np.allclose(dn.bits_per_token, dn.B) and (dn.evict_frac == 0).all()):
        problems.append("a dense row is not w bits with nothing evicted")
    k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(d.B, d.ctx_len)])
    for fname in ("qread", "qreadfp", "qread2t"):
        m = (fam == fname).to_numpy()
        if not m.any():
            continue
        x = d[m]
        store = pas[m].map(lambda p: float(p["store"])).to_numpy()
        read_w = pas[m].map(lambda p: float(p["tier2_bits"] if p["family"] == "qread2t" else p["store"])).to_numpy()
        ok = (np.allclose(x.stored_bits_per_token.to_numpy(), store) and (x.stored_evict_frac == 0).all()
              and np.allclose(x.read_frac.to_numpy(), k[m], atol=1e-9)
              and np.allclose(x.bits_per_token.to_numpy(), read_w * x.read_frac.to_numpy(), atol=1e-6))
        if not ok:
            problems.append(f"a {fname} row does not store its tier-1 width dense and read floor(r C) per KV head "
                            f"at its read width")
    tt = d[(fam == "qread2t").to_numpy()]
    if len(tt):
        named = tt.arm.map(lambda a: L.parse_arm(a)["tier2_bits"]).to_numpy(dtype=float)
        kv = tt.arm.map(lambda a: L.parse_arm(a)["kv"]).to_numpy(dtype=bool)
        rq = tt.arm.map(lambda a: L.parse_arm(a)["requestion"]).to_numpy(dtype=bool)
        if not (np.allclose(tt.tier2_bits.to_numpy(dtype=float), named)
                and np.array_equal(tt.keys_only.to_numpy(dtype=bool), ~kv)
                and np.array_equal(tt.requestion.to_numpy(dtype=bool), rq)):
            problems.append("a two-tier row's tier-2 width, keys-only or second-pass flags differ from its name")
    for arm in sorted(d[d.twin.fillna("") != ""].arm.unique()):
        t = d[d.arm == arm]
        b = d[d.arm == L.parse_arm(arm)["base"]]
        m = t.merge(b, on=key + ["B"], suffixes=("", "_b"))
        if len(m) != len(t) or not (m.tf_sum_nll - m.tf_sum_nll_b).abs().mean() > 0:
            problems.append(f"{arm}: missing base rows or values unchanged (V4)")
    fp = d[d.arm == "fp"].dropna(subset=["tf_top1"])
    agree = float((fp.tf_top1 * fp.tf_len).sum() / max(fp.tf_len.sum(), 1))
    if agree < RD.V3_MIN:
        problems.append(f"fp teacher-forced replay agrees with fp's greedy on {agree:.3f} < {RD.V3_MIN}")
    ok_fp = fp[fp.score >= 1]
    cover = float((ok_fp.a_len > 0).mean()) if len(ok_fp) else 0.0
    if cover < RD.V6_MIN:
        problems.append(f"FP's answer-value mask found in {cover:.2f} < {RD.V6_MIN} of correct FP answers")
    if main:
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        if (fps >= FP_MIN).sum() < min(3, len(fps)):
            problems.append(f"FP below {FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree, cover


def validate_a2_g(d, problems, name, self_check=True):
    """Stage 1f's A2 checks, with the self-check required per arm NAME."""
    RF.validate_a2(d, problems, name, self_check=False)
    if not self_check or "a2_check" not in d:
        return
    arms = d[d.arm != "fp"]
    ck = arms.dropna(subset=["a2_check"])
    for job, g in arms.groupby("job"):                 # every block (process) checks its own read paths
        miss = sorted(set(g.arm) - set(ck[ck.job == job].arm))
        if miss:
            problems.append(f"{name}: block {job} has no self-check replay (a2_check) for {miss} (A2)")
    bad = ck[ck.a2_check > L.A2_CHECK_TOL]
    if len(bad):
        problems.append(f"{name}: a replay through replay_ids differs from the first by up to "
                        f"{bad.a2_check.max():.3f} nats ({sorted(set(bad.arm))}) (A2)")


# ---------------------------------------------------------------- analysis
def add_bytes_g(v):
    """BYTES (module docstring): read per step, GPU-stored, host, fetched per question."""
    v = v.copy()
    hd = v.head_dim.to_numpy(dtype=float)
    d8 = hd / 8.0
    pas = v.arm.map(L.parse_arm)
    fam = pas.map(lambda p: p["family"]).to_numpy()
    vb = v.v_bits.to_numpy(dtype=float)
    vs = np.where(vb >= 16, 0.0, 16.0 / hd)
    tail = ((v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len).to_numpy(dtype=float)
    ks = v.key_side.to_numpy(dtype=float)
    rf = v.read_frac.to_numpy(dtype=float)
    byt = d8 * (v.bits_per_token.to_numpy(dtype=float) + ks) + (1 - v.evict_frac.to_numpy(dtype=float)) * d8 * (vb + vs)
    sk = np.where(fam == "qread", 16.0 / hd, np.where(fam == "qreadfp", 0.0, ks))
    sto = (d8 * (v.stored_bits_per_token.to_numpy(dtype=float) + sk)
           + (1 - v.stored_evict_frac.to_numpy(dtype=float)) * d8 * (vb + vs))
    host, fetch = np.zeros(len(v)), np.zeros(len(v))
    for i in np.where(fam == "qread2t")[0]:
        p = pas.iloc[i]
        byt[i] = L.two_tier_bytes(rf[i], p["tier2_bits"], p["kv"], p["v_bits"], int(hd[i]))["total"]
        sto[i] = d8[i] * (p["store"] + 16.0 / hd[i]) + d8[i] * (vb[i] + vs[i])
        host[i] = L.host_bytes(p["tier2_bits"], p["kv"], int(hd[i]))
        fetch[i] = L.fetch_bytes(float(v.B.iloc[i]), int(v.ctx_len.iloc[i]), p["tier2_bits"], p["kv"], int(hd[i]))
    v["bytes"], v["bytes_stored"], v["bytes_host"] = byt + tail, sto + tail, host
    v["fetch_head"] = fetch                       # bytes per KV head and layer, once per question
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


def _points_g(dA, v, pidx, n_kv):
    """RF._points with Stage 1g's arm names: each point's statistics vs D_L, dA
    beside dS, answers below FP vs D_L, answers in another order (replayed)."""
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1) if s.index.nlevels > 2 else s, pidx).to_numpy()  # noqa: E731
    cols = ["bytes", "bytes_stored", "bytes_amort", "bytes_host", "fetch_head", "evict_frac", "kept_width",
            "needle_keep", "score"]
    means = v.groupby(["arm", "B"])[cols].mean()
    At = RF._unit_table(v, dA, L.METRIC_FROZEN)
    dAlt = At.sub(At[("fp", 0.0)], axis=0)
    S = RF._unit_table(v, dA, "score")
    below = S.lt(S[("fp", 0.0)] - 1e-9, axis=0).astype(float)
    RO = RF._unit_table(v.assign(_ro=v.ans_reordered.fillna(False).astype(float)), dA, "_ro")
    RP = RF._unit_table(v.assign(_rp=v.own_replay.fillna(False).astype(float)), dA, "_rp")
    pts = {}
    for c in dA.columns:
        arm, B = c
        if arm == "fp" or dA[c].isna().all():
            continue
        pa = L.parse_arm(arm)
        Dc = (DREF[pa["lens"]], 3.0)
        p = RD._point_stats(dA, c, Dc, W, pm, means)
        p.update(arm=arm, B=float(B), lens=pa["lens"], family=pa["family"], score=float(means.loc[c, "score"]),
                 bytes_host=float(means.loc[c, "bytes_host"]),
                 fetch_mb_layer=float(means.loc[c, "fetch_head"]) * n_kv / 1e6,
                 dA_by_task={t: float(g.mean()) for t, g in dA[c].groupby(level=-1)})
        p["alt"] = dict(stat=L.METRIC_FROZEN, dA=L1C.boot_ci(pm(dAlt[c]), W),
                        vs_D=L1C.boot_ci(pm(dAlt[c] - dAlt[Dc]), W) if Dc in dAlt and c != Dc else None)
        p["below_fp"] = dict(n=int(below[c].sum()), n_D=int(below[Dc].sum()) if Dc in below else None,
                             diff=L1C.boot_ci(pm(below[c] - below[Dc]), W) if Dc in below and c != Dc else None)
        p["reordered"] = dict(n=int(RO[c].sum()), replayed=int(RP[c].sum()))
        p["vs_FP"] = dict(nll=L1C.boot_ci(pm(dA[c]), W),
                          tail=L1C.boot_ci(pm((dA[c] > TAIL_NATS).astype(float)), W))
        p["vs_FP"]["label"] = L.near_fp_label(p["vs_FP"]["nll"], p["vs_FP"]["tail"])
        pts[f"{arm}@{bk(B)}"] = p
    return pts, W, pm


def analyse_cell(name, d, sides):
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= FP_MIN)
    v = add_bytes_g(d[d.task.isin(valid)])
    dA = RF._dA(v, ["job", "prompt_idx", "task"], col=METRIC)
    pidx = dA.index.droplevel("task").unique()
    pts, W, pm = _points_g(dA, v, pidx, int(sides[0].get("n_kv_heads", 8)))
    C = int(sides[0]["ctx"])
    res = dict(cell=name, metric=METRIC, ctx=C, fp=float(fp_cells.mean()), fp_by_task=fp_cells.round(4).to_dict(),
               valid_tasks=valid, n_prompts=int(len(pidx)), n_prompt_tasks=int(len(dA)), points=pts,
               families=RF.families(pts), labels={})

    def diff(a, b, lab):
        if a not in dA.columns or b not in dA.columns:
            return
        z = L1C.boot_ci(pm(dA[a] - dA[b]), W)
        res["labels"][lab] = dict(diff=z, label=L.effect_label(*z, lab), a=f"{a[0]}@{bk(a[1])}",
                                  b=f"{b[0]}@{bk(b[1])}")

    e = 0.125
    diff(("qread2t_v16", e), ("qreadfp_v16", e), "QPASS3")
    diff(("qread2t4_v16", e), ("qreadfp_v16", e), "QPASS4")
    diff(("qread2t4_v4", e), ("qread2t_v4", e), "TIER4")
    diff(("qread4_v4", e), ("qread_v4", e), "STORE4")
    diff(("qread2tk_v4", e), ("qread2t_v4", e), "KONLY")
    diff(("qread2tk8_v4", e), ("qread2t_v4", e), "KONLY8")
    diff(("qread2tq_v4", e), ("qread2t_v4", e), "REQ")
    rf = L.norm_b(L.floor_r(C))                      # the presets put the floor arms exactly there
    if float(rf) > e:
        diff(("qread2t_v4", float(rf)), ("qread2t_v4", e), "FLOOR")
        fpk = f"qreadfp_v16@{bk(rf)}"
        if fpk in pts:
            res["labels"]["FLOOR_FP"] = dict(label=pts[fpk]["vs_FP"]["label"], nll=pts[fpk]["vs_FP"]["nll"])
    sk = f"{L.SYSTEM}_v4@{bk(rf)}"
    if sk in pts:
        res["system"] = dict(point=sk, vs_D=pts[sk]["vs_D"]["label"], vs_FP=pts[sk]["vs_FP"]["label"],
                             nll_vs_FP=pts[sk]["vs_FP"]["nll"], nll_vs_D=pts[sk]["vs_D"]["nll"],
                             rho=pts[sk]["vs_D"]["rho"], rho_mem=pts[sk]["vs_D"]["rho_mem"],
                             fetch_mb_layer=pts[sk]["fetch_mb_layer"], below_fp=pts[sk]["below_fp"])
    return res


def analyse_confusion(rd, D=("uniform", 3.0)):
    """CONFUSION over the regression block (dS per prompt; one answer value)."""
    A = rd.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=METRIC, aggfunc="first")
    A = A[~A[("fp", 0.0)].isna()]
    dA = A.sub(A[("fp", 0.0)], axis=0)
    out = {}
    for c in dA.columns:
        if c[0] == "fp":
            continue
        per = {f"{p}/{t}": (float(dA.loc[(p, t), c]), float(dA.loc[(p, t), D])) for p, t in dA.index}
        out[f"{c[0]}@{bk(c[1])}"] = dict(label=L.confusion_label(per), dS={k: round(x, 2) for k, (x, _) in per.items()})
    return out


# ------------------------------------------------------------------ report
def _points_table(points):
    Lh = ["| arm@B | lens | bytes read | ρ | ρ GPU mem | host B/token | fetch MB/layer | dS vs FP | NEAR_FP | "
          "dS vs D [90%] | tail | label | dA vs D (1d–1e) | below FP: X / D | reordered (replayed) | score |",
          "|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|---|---:|"]
    for k, p in sorted(points.items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
        z = p.get("vs_D")
        a, b, o = p.get("alt", {}), p.get("below_fp", {}), p.get("reordered", {})
        Lh.append(f"| {k} | {p['lens']} | {p['bytes']:.1f} | {fmt(z and z['rho'], 2)} | {fmt(z and z['rho_mem'], 2)} | "
                  f"{p['bytes_host']:.0f} | {p['fetch_mb_layer']:.1f} | {ci(p['dA'])} | {p['vs_FP']['label']} | "
                  f"{ci(z['nll']) if z else '(D)'} | {p['tail']:.3f} | {z['label'] if z else '—'} | "
                  f"{ci(a['vs_D']) if a.get('vs_D') else '—'} | {b.get('n', '—')} / {b.get('n_D', '—')} | "
                  f"{o.get('n', 0)} ({o.get('replayed', 0)}) | {p['score']:.3f} |")
    return Lh


def report(results, summary, out_stem):
    Lh = ["# R14 Stage 1g — G1–G5 and the system (read_stage1g.py; rules frozen in its docstring)", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for r in results:
        if "confusion" in r and "points" not in r:
            Lh += [f"## {r['cell']} — key confusions (dS vs FP per prompt)", ""]
            Lh += [f"- {k}: **{z['label']}** {z['dS']}" for k, z in r["confusion"].items()] + [""]
            continue
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f} ({r['fp_by_task']}), {r['n_prompts']} prompts, "
               f"{r['n_prompt_tasks']} prompt-tasks; per-unit statistic dS (s_set_nll, A2)", ""]
        Lh += RF._fam_table(r["families"]) + [""] + _points_table(r["points"]) + [""]
        for lab, z in r["labels"].items():
            if "diff" in z:
                Lh.append(f"**{lab}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}")
            else:
                Lh.append(f"**{lab}**: {z['label']} ({ci(z['nll'])})")
        if r.get("system"):
            s = r["system"]
            Lh.append(f"**SYSTEM** {s['point']}: vs D_V4 {s['vs_D']} ({ci(s['nll_vs_D'])}, ρ {fmt(s['rho'], 2)}, "
                      f"ρ GPU mem {fmt(s['rho_mem'], 2)}), vs FP {s['vs_FP']} ({ci(s['nll_vs_FP'])}); fetch "
                      f"{s['fetch_mb_layer']:.1f} MB per layer per question; below FP {s['below_fp']['n']} "
                      f"(D {s['below_fp']['n_D']})")
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def read_cells(cells, out_stem, root=RESULTS, regress=None):
    """cells: name -> (tag, [jobs]); regress: (tag, job) for gregress128 or None."""
    results, problems, summary = [], [], {}
    for name, (tag, jobs) in cells.items():
        parts = [load_run(tag, j, root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{name}: blocks ran different plans")
        agree, cover = validate_g(d, sides, problems)
        validate_a2_g(d, problems, name)
        if problems:
            continue
        r = analyse_cell(name, d, sides)
        r["fp_replay_agreement"], r["value_cover"] = agree, cover
        for lab, z in r["labels"].items():
            summary[f"{lab} {name}"] = z["label"]
        if r.get("system"):
            summary[f"SYSTEM {name}"] = f"vs D {r['system']['vs_D']}, vs FP {r['system']['vs_FP']}"
        results.append(r)
    if regress:
        rd, rside = load_run(regress[0], regress[1], root)
        validate_g(rd, [rside], problems, main=False)
        validate_a2_g(rd, problems, "gregress128", self_check=False)
        g = cells.get("g128")
        if g:
            plan = plan_of(load_run(g[0], g[1][0], root)[1])
            if plan_of(rside) != plan:
                problems.append("gregress128 ran another plan than g128")
        if not problems:
            conf = analyse_confusion(rd)
            results.append(dict(cell="gregress128", confusion=conf))
            for k in (f"uniform@3", "qread_v4@0.125", "qread4_v4@0.125", "qread2t_v4@0.125", "qread2tk_v4@0.125",
                      "qread2tk8_v4@0.125", f"{L.SYSTEM}_v4@0.125"):
                if k in conf:
                    summary[f"CONFUSION {k}"] = conf[k]["label"]
    if problems:
        raise SystemExit("INVALID Stage 1g data:\n  " + "\n  ".join(problems))
    sysc = {r["cell"]: r["system"] for r in results if r.get("system")}
    if sysc:
        summary["G SYSTEM"] = L.system_label(sysc)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(nll_margin=RD.NLL_MARGIN, tail_margin=RD.TAIL_MARGIN, tail_nats=TAIL_NATS,
                                  effect_eps=L.EFFECT_EPS, k_min=L.K_MIN, a2_check_tol=L.A2_CHECK_TOL,
                                  boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED),
                       summary=summary, cells=results), fh, indent=1, default=str)
    Lh = report(results, summary, out_stem)
    print("\n".join(Lh[:2 + len(summary) + 1]))
    return 0


# --------------------------------------------------------------------- gate
def gate(job, tag="gpilot", root=RESULTS):
    d, side = load_run(tag, job, root)
    problems = []
    agree, cover = validate_g(d, [side], problems, main=False)
    multi = d[(d.arm == "fp") & d.task.isin(L.MULTI_ANSWER)].fp_ans_order.fillna("").str.contains(",")
    validate_a2_g(d, problems, f"pilot {job}", self_check=bool(multi.any()))
    pk = side.get("peak_gib_dev_max") or []
    if not pk or not all(x <= PEAK_GIB_MAX for x in pk):
        problems.append(f"peak GPU memory per device {pk} GiB (limit {PEAK_GIB_MAX})")
    main = L.PILOT_OF[tag]
    n_main = len(L.build_plan(L.PRESETS[main]))
    dec = float(d[d.arm != "fp"].t_arm.median())
    tf = float(d.t_tf.median()) + float(d.t_own.fillna(0).mean())
    unit_s = float(d.t_prefill.median()) + float(d.t_precompute.max()) + n_main * (dec + tf)
    proj_h = unit_s * BLOCK_UNITS[main] / 3600
    if proj_h > 0.9 * WALL_H[main]:
        problems.append(f"projected {main} block {proj_h:.1f} h > 90% of {WALL_H[main]} h")
    ck = d.dropna(subset=["a2_check"]) if "a2_check" in d else d.iloc[:0]
    print(f"R14 Stage 1g pilot {job} ({tag} for {main}): {len(d)} rows, plan {len(plan_of(side))} arms, fp replay "
          f"{agree:.3f}, answer-value coverage {cover:.2f}, peak per GPU {pk} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected {main} block {proj_h:.1f} h ({n_main} arms, wall {WALL_H[main]} h); "
          f"self-checks {len(ck)}, largest {ck.a2_check.max() if len(ck) else float('nan'):.2e} nats")
    cols = [c for c in ("score", "s_set_nll", "a_sum_nll", "bits_per_token", "read_frac", "stored_bits_per_token",
                        "t_arm", "t_tf") if c in d]
    print(d.pivot_table(index=["arm", "B"], values=cols, aggfunc="mean").round(3).to_string())
    for r_ in d[d.arm == "fp"].itertuples():
        print(f"  FP {r_.task}: score {r_.score:.2f}, pred {r_.pred[:120]!r}")
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--pilot-tag", default="gpilot", choices=sorted(L.PILOT_OF))
    ap.add_argument("--g128", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--g32", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--gregress", metavar="JOB")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1g"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(gate(a.pilot, a.pilot_tag, a.results_root))
    cells = {}
    if a.g128:
        cells["g128"] = ("g128", a.g128)
    if a.g32:
        cells["g32"] = ("g32", a.g32)
    if not cells:
        ap.error("give --pilot JOB or at least one cell")
    sys.exit(read_cells(cells, a.out_stem, a.results_root, ("gregress128", a.gregress) if a.gregress else None))


if __name__ == "__main__":
    main()
