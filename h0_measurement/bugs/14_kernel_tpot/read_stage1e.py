#!/usr/bin/env python3
"""R14 Stage 1e gate and reader. The rules below are FROZEN: written 2026-10-02,
before any Stage 1e output existed (design: s1e_lib.py; driver: run_s1e.py; the
kernel and its own rule: s1e_kernel.py / bench_s1e_kernel.py).

    python read_stage1e.py --pilot JOB --pilot-tag pilot128e|reusepilot|qwenpilot32e|qwenpilot128
    python read_stage1e.py --tail128 J1 J2 J3 J4 --regress128 JR --cal128 C \
                           --tail32 J5 J6 J7 J8 --regress32 JR2 --cal32 C2          (-> stage1e)
    python read_stage1e.py --reuse128 J1 J2 --reuse32 J3 J4 --out-stem .../stage1e_reuse
    python read_stage1e.py --qwen32 J1 J2 J3 J4 --qwen128 J5 J6 J7 J8 --out-stem .../stage1e_qwen

Result directories: h0_measurement/results/r14s1e_<tag>_<job>/.

CELLS
  E1+E2  Llama-3.1-8B @131072 ('tail128', prompts 8100-8139) and @32768 ('tail32',
         8200-8239), blocks of 10. E2's calibration: prompts 8000-8039 (niah_multikey
         on all 40, the other tasks on 8000-8009). Regression blocks: Stage 1d's
         catastrophic prompt-tasks, niah_multikey (128K: 7020, 7036; 32K: 7112, 7113,
         7117), with the same arms.
  E3     'reuse128' (mixed prompts 8300-8339) and 'reuse32' (8400-8439), blocks of 20.
  E4     Qwen3-30B-A3B-2507 @32768 ('qwen32e', 8200-8239) and @131072 ('qwen128q',
         8100-8139, two GPUs).
METRIC   dA(X) = a_sum_nll(X) - a_sum_nll(fp), Stage 1d's answer-value NLL, per
         prompt-task (E3: per prompt and question). A prompt-task whose FP answer
         holds no expected value drops (count reported). Task cells (E3: question
         cells by task) with FP score < 0.9 drop.
COMPARATOR D_L = uniform@3 in X's value lens (V16 exact values, V4 4-bit values).
INTERVALS Prompt bootstrap: prompts resampled within block, every task (E3: both
         questions) of a prompt together, 10,000 draws, seed 14; 90% percentile
         intervals [lo, hi].
MATCHED  Stage 1c/1d's rule on dA: hi of mean(dA(X) - dA(D_L)) <= 0.10 nats AND hi of
         (tail(X) - tail(D_L)) <= 0.05, tail = share of prompt-tasks with dA > 2.
         WORSE iff the NLL lo > 0.10 or the tail lo > 0.05. INCONCLUSIVE otherwise.
BYTES    Stage 1d's rule, read per decode step per context token per KV head per
         layer; rho = bytes(X) / bytes(D_L). Routers: kept norms + width index.
         Reads and snapq: kept norms + keep bitmap. qreadfp: 16-bit keys of the
         kept rows + bitmap. Stored bytes (rho_mem) reported: the reads store D's
         keys dense, qreadfp stores BF16, snapq stores only its kept rows.
VERDICT  Per family (qread, seq2, nest2, seq3, nest3, sieve, pool, snapq), cell and
         lens, the cheapest MATCHED point: WIN (rho <= 0.80), TIE (<= 1.00), LOSS,
         or NO_POINT. qreadfp is a diagnostic (E1), never a design.
E1 SOURCE, every main cell and every r with qread_v16@r and qreadfp_v16@r:
         G_q = mean(dA(uniform@3) - dA(qread_v16@r)); G_fp = mean(-dA(qreadfp_v16@r));
         Q = G_q - G_fp; prompt-bootstrap intervals. s1e_lib.source_label: NO_GAIN
         unless G_q lo > 0; then BOTH (G_fp lo > 0 and Q lo > 0), DILUTION (G_fp lo
         > 0), QUANT_NOISE (Q lo > 0), else UNRESOLVED. Also qreadfp against FP itself
         (MATCHED rule with FP as the comparator).
E2 TAIL, Llama cells; B_t = 4 at 128K, 3 at 32K:
         - catastrophes(X): fresh prompt-tasks with dA(X) - dA(D_L) > 3 nats;
         - the 2 x 2 at B_t in V16, per prompt-task: nesting effect = ((nest2 - seq2)
           + (nest3 - seq3)) / 2, calibration effect = ((seq3 - seq2) + (nest3 -
           nest2)) / 2, as NEST_* / CAL_* effect labels (|mean| >= 0.05 nats and an
           interval excluding 0);
         - regression: per Stage 1d catastrophic prompt-task, FIXED iff dA(X) - dA(D)
           <= 2 nats (V16);
         - s1e_lib.tail_label for router_nest3_calib@B_t (the full fix) against
           router_seq2_calib@B_t (Stage 1d's routes): TAIL_FIXED iff MATCHED in V16
           and V4, 0 catastrophes and FIXED on every regression prompt-task;
           TAIL_REDUCED iff fewer catastrophes, or a regression prompt-task FIXED that
           seq2 was not, at mean(dA(nest3) - dA(seq2)) <= 0; else TAIL_NOT_FIXED;
         - ROUTER_32K: does any router family WIN in V4 at 32K.
E3 REUSE, per reuse cell and lens with qread_v@r and snapq_v@r: MATCHED-rule labels
         vs D_L per question role (Q1 = the question SnapKV's selection saw, Q2 =
         the other). s1e_lib.reuse_label: REUSE_DIFFERENTIATES if at some r the reads
         are MATCHED on Q1 and Q2 while snapq is WORSE on Q2; REUSE_NO_DIFFERENCE if
         snapq is MATCHED on Q2 at every r; QREAD_FAILS_REUSE if the reads are not
         MATCHED on Q2 at any r; REUSE_MIXED otherwise. Reported: snapq - qread on Q2
         per Q2 task, and both arms' stored bytes.
E4 QWEN  - STOP_FIX: WORKS iff FP's niah_multivalue score >= 0.9 in qwen32e (else
           FAILS and the task drops, as in Stage 1d);
         - QWEN_QREAD_V4 at 32K and 128K: the qread family's V4 verdict;
         - QWEN_TAIL: s1e_lib.tail_label for router_nest2_calib@3 (Stage 1d's critical
           heads, nested) against router_seq2_calib@3; Qwen has no regression block,
           so that condition is vacuous;
         - E1 in both Qwen cells.
VALIDITY (any failure -> INVALID, no verdict)
  V1 Every (prompt, task) -- (prompt, question) for E3 -- holds exactly the plan's
     arms, once, for every planned prompt-task. One corpus and one plan per cell.
     FP >= 0.9 on >= 3 of 4 tasks (Llama), >= 2 (Qwen), both tasks (E3).
  V2 Audits: dense = w bits with nothing evicted; routers spend <= B; reads store 3
     bits dense (qreadfp: 16) and read >= floor(r C) per head (exactly, without
     protection); snapq on Q2 reads exactly floor(r C) per head and stores only those;
     twins match their base.
  V3 fp's teacher-forced replay agrees with fp's greedy on >= 98% of steps.
  V4 Every twin changed its values.
  V5 E2's calibration: routes_seq3 = R0 with heads switched to dense; its base file
     unchanged; no forced search; every block read this calibration's file; every
     nested router's dense heads = R0's + the nested critical set.
  V6 FP's answer-value mask is non-empty for >= 90% of correct FP answers.
  V7 Stop rule: Qwen blocks ran eos_only, Llama blocks r8.
REPORTED, NOT GATED: task scores, lossless, worst-5% share, CVaR, max dA, stored and
  answer-amortised bytes, needle_keep, per-task dA.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1e_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0                        # per GPU
WALL_H = {"tail128": 6.0, "tail32": 3.0, "reuse128": 6.0, "reuse32": 2.5, "qwen32e": 4.0,
          "qwen128q": 10.0}
BLOCK_UNITS = {"tail128": 40, "tail32": 40, "reuse128": 40, "reuse32": 40, "qwen32e": 40,
               "qwen128q": 40}             # prompt-tasks (reuse: prompt-questions) per main block
V3_MIN, V6_MIN = 0.98, 0.90
NLL_MARGIN, TAIL_MARGIN, TAIL_NATS, WORST_FRAC = 0.10, 0.05, 2.0, 0.05
FIXED_NATS = 2.0
FP_MIN = BM.RMT.FP_MIN
DREF = {"V16": "uniform", "V4": "uniform+v4"}
bk, fmt, ci, sha256 = R1C.bk, R1C.fmt, R1C.ci, R1C.sha256


# ------------------------------------------------------------------ loading
def load_run(tag, job, mode="evaluate", root=RESULTS):
    d = os.path.join(root, f"r14s1e_{tag}_{job}")
    pq = [x for x in glob.glob(os.path.join(d, f"s1e_{mode}_*.parquet"))
          if not x.endswith(("_search.parquet", "_searchlog.parquet"))]
    js = glob.glob(os.path.join(d, f"s1e_{mode}_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one s1e_{mode} parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    if "q_role" not in rows:
        rows["q_role"] = ""
    rows["q_role"] = rows.q_role.fillna("")
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def load_cal(tag, job, root=RESULTS):
    rows, side = load_run(tag, job, "calibrate", root)
    d = side["_dir"]
    return (rows, side, pd.read_parquet(os.path.join(d, side["search"])),
            pd.read_parquet(os.path.join(d, side["searchlog"])), json.load(open(side["write_routes"])))


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


def unit_key(kind):
    return ["job", "prompt_idx", "q_role"] if kind == "reuse" else ["job", "prompt_idx", "task"]


# ----------------------------------------------------------------- validity
def validate(d, sides, problems, kind="main", fp_tasks_min=3, main=True):
    key = unit_key(kind)
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(key[1:]).apply(lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))), include_groups=False)
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: units with the wrong arms: {bad[:3]}")
        n_want = len(side["prompt_tasks"]) * (2 if kind == "reuse" else 1)
        if len(got) != n_want:
            problems.append(f"block {side['_job']}: {len(got)} units, expected {n_want}")
        rule = side.get("stop_rule")
        want_rule = "eos_only" if side["model"].startswith("qwen") else "r8"
        if rule != want_rule:
            problems.append(f"block {side['_job']}: stop rule {rule}, expected {want_rule} (V7)")
    if d.duplicated(key + ["arm", "B"]).any():
        problems.append("duplicate rows")
    if d.corpus_sha.nunique() != 1:
        problems.append(f"blocks disagree on corpus: {sorted(d.corpus_sha.unique())}")
    fam = d.arm.map(L.family)
    dn = d[fam == "dense"]
    if not (np.allclose(dn.bits_per_token, dn.B) and (dn.evict_frac == 0).all()):
        problems.append("a dense row is not w bits with nothing evicted")
    rt = d[fam.isin(L.ROUTERS)]
    if (rt.bits_per_token > rt.B + 1e-6).any():
        problems.append("a router row overspends its budget")
    for fname, width in (("qread", L.STORE_WIDTH), ("qreadfp", L.EXACT_WIDTH)):
        qr = d[fam == fname]
        if not len(qr):
            continue
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(qr.B, qr.ctx_len)])
        prot = qr.arm.map(lambda a: L.parse_arm(a)["protect"]).to_numpy()
        ok = (np.allclose(qr.stored_bits_per_token, width) and (qr.stored_evict_frac == 0).all()
              and np.all(qr.read_frac.to_numpy() >= k - 1e-9)
              and np.allclose(qr.read_frac.to_numpy()[~prot], k[~prot], atol=1e-9)
              and np.allclose(qr.bits_per_token, width * qr.read_frac, atol=1e-6))
        if not ok:
            problems.append(f"a {fname} row does not store {width} bits dense and read floor(r C) per head")
    sq = d[fam == "snapq"]
    if len(sq):
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(sq.B, sq.ctx_len)])
        if not (np.allclose(sq.read_frac, k, atol=1e-9) and np.allclose(sq.stored_bits_per_token, sq.bits_per_token)
                and np.allclose(sq.stored_evict_frac, sq.evict_frac)):
            problems.append("a snapq row does not read and store exactly floor(r C) per head")
    for arm in sorted(d[d.twin != ""].arm.unique()):
        t = d[d.arm == arm]
        b = d[d.arm == L.parse_arm(arm)["base"]]
        m = t.merge(b, on=key + ["B"], suffixes=("", "_b"))
        if len(m) != len(t):
            problems.append(f"{arm}: {len(m)} of {len(t)} rows have a base row")
            continue
        if not (np.allclose(m.bits_per_token, m.bits_per_token_b) and np.allclose(m.evict_frac, m.evict_frac_b)):
            problems.append(f"{arm}: bits or eviction differ from its base")
        if not (m.tf_sum_nll - m.tf_sum_nll_b).abs().mean() > 0:
            problems.append(f"{arm}: values did not change anything (V4)")
    fp = d[d.arm == "fp"].dropna(subset=["tf_top1"])
    agree = float((fp.tf_top1 * fp.tf_len).sum() / max(fp.tf_len.sum(), 1))
    if agree < V3_MIN:
        problems.append(f"fp teacher-forced replay agrees with fp's greedy on {agree:.3f} < {V3_MIN}")
    ok_fp = fp[fp.score >= 1]
    cover = float((ok_fp.a_len > 0).mean()) if len(ok_fp) else 0.0
    if cover < V6_MIN:
        problems.append(f"FP's answer-value mask found in {cover:.2f} < {V6_MIN} of correct FP answers")
    if main:
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        need = len(fps) if kind == "reuse" else min(fp_tasks_min, len(fps))
        if (fps >= FP_MIN).sum() < need:
            problems.append(f"FP below {FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree, cover


def validate_routes(sides, cal, problems):
    """V5: every Stage 1d/1e router's dense heads = R0's + its critical set (flat:
    the budget's own; nested: every calibrated budget <= B); the critical sets are
    the routes files' own; every block read the same files; seq3/nest3 read this
    calibration's file."""
    prov = {json.dumps(s.get("routes", {}), sort_keys=True) for s in sides}
    if len(prov) != 1:
        problems.append("blocks read different routes files")
    field = {"1d": "routes_seq2", "1e": "routes_seq3"}
    checked = set()
    for s in sides:
        crit = {src: {L.norm_b(float(k)): [tuple(h) for h in hs] for k, hs in cb.items()}
                for src, cb in s.get("critical", {}).items()}
        r0 = {L.norm_b(float(k)): {tuple(h) for h in hs} for k, hs in s.get("r0_dense_heads", {}).items()}
        for k, heads in s.get("dense_heads", {}).items():
            arm, B = k.split("@")
            if arm not in L.ROUTE_SOURCE:
                continue
            B = L.norm_b(float(B))
            src, nested = L.ROUTE_SOURCE[arm]
            add = L.nested_heads(crit.get(src, {}), B) if nested else crit.get(src, {}).get(B, [])
            want = sorted(r0.get(B, set()) | set(add))
            if B not in r0 or sorted(tuple(h) for h in heads) != want:
                problems.append(f"block {s['_job']}: {k}: dense heads are not R0's plus the "
                                f"{'nested ' if nested else ''}critical set (V5)")
            path, sha = s["routes"][k]["path"], s["routes"][k]["sha256"]
            if (src, path) in checked:
                continue
            checked.add((src, path))
            try:
                if sha256(path) != sha:
                    problems.append(f"{path} changed since the blocks read it")
                fc = L.critical_by_budget(json.load(open(path)), field[src])
                if {b: sorted(v) for b, v in fc.items()} != {b: sorted(v) for b, v in crit.get(src, {}).items()}:
                    problems.append(f"the blocks' Stage {src} critical sets are not {path}'s")
            except OSError as e:
                problems.append(f"routes file unreadable: {e}")
    if cal is not None:
        _, cside, _, _, routes = cal
        meta, base = routes["meta"], routes["meta"]["base_routes"]
        try:
            if sha256(base["path"]) != base["sha256"]:
                problems.append(f"{base['path']} changed since the calibration read it")
        except OSError as e:
            problems.append(f"calibration base routes unreadable: {e}")
            return
        j1b = json.load(open(base["path"]))
        for k in routes["routes_pool"]:
            if routes["routes_pool"][k] != j1b["routes_pool"].get(k):
                problems.append(f"calibration routes_pool@{k} is not the base file's")
            if not L1C.only_densified(routes["routes_pool"][k], routes["routes_seq3"][k]):
                problems.append(f"calibration routes_seq3@{k} changes more than dense switches")
        if meta.get("rule", {}).get("forced"):
            problems.append("the calibration ran with --force-search (mechanics only)")
        path = os.path.abspath(cside["write_routes"])
        csha = sha256(path)
        for s in sides:
            for k_, v in s["routes"].items():
                if k_.startswith(("router_seq3_calib", "router_nest3_calib")) and \
                        (os.path.abspath(v["path"]) != path or v["sha256"] != csha):
                    problems.append(f"block {s['_job']}: {k_} read {v['path']}, not the calibration's {path}")


# ---------------------------------------------------------------- analysis
def add_bytes(v):
    """Stage 1d's byte rule with Stage 1e's families (read, stored, answer-amortised)."""
    v = v.copy()
    d8 = v.head_dim / 8.0
    vs = np.where(v.v_bits >= 16, 0.0, 16.0 / v.head_dim)
    tail = (v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len
    v["bytes"] = d8 * (v.bits_per_token + v.key_side) + (1 - v.evict_frac) * d8 * (v.v_bits + vs) + tail
    fam = v.arm.map(L.family)
    sk = np.where(fam == "qread", 16.0 / v.head_dim, np.where(fam == "qreadfp", 0.0, v.key_side))
    v["bytes_stored"] = (d8 * (v.stored_bits_per_token + sk)
                         + (1 - v.stored_evict_frac) * d8 * (v.v_bits + vs) + tail)
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


def _tables(d, kind):
    """dA (and score) pivoted by unit x (arm, B), FP-dropped units removed."""
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= FP_MIN)
    v = add_bytes(d[d.task.isin(valid)])
    key = unit_key(kind)
    piv = lambda col: v.pivot_table(index=key, columns=["arm", "B"], values=col, aggfunc="first")  # noqa: E731
    A = piv("a_sum_nll")
    drop = A[("fp", 0.0)].isna()
    A = A[~drop]
    dA = A.sub(A[("fp", 0.0)], axis=0)
    S = piv("score").loc[A.index]
    T = (v.pivot_table(index=key, columns=["arm", "B"], values="task", aggfunc="first").loc[A.index]
         if "task" not in key else None)              # reuse: each question's task
    return v, dA, S, T, fp_cells, valid, int(drop.sum())


def _point_stats(dA, col, Dc, W, pm, means):
    x = dA[col].to_numpy()
    p = dict(dA=L1C.boot_ci(pm(dA[col]), W), tail=L1C.tail_share(x, TAIL_NATS),
             worst5_share=L1C.worst_share(x, WORST_FRAC), cvar5=L1C.cvar(x, WORST_FRAC), max_dA=float(np.max(x)),
             bytes=float(means.loc[col, "bytes"]), bytes_stored=float(means.loc[col, "bytes_stored"]),
             bytes_amort=float(means.loc[col, "bytes_amort"]), evict_frac=float(means.loc[col, "evict_frac"]),
             kept_width=float(means.loc[col, "kept_width"]), needle_keep=float(means.loc[col, "needle_keep"]))
    if Dc is not None and Dc in dA.columns and col != Dc:
        nll = L1C.boot_ci(pm(dA[col] - dA[Dc]), W)
        tl = L1C.boot_ci(pm((dA[col] > TAIL_NATS).astype(float) - (dA[Dc] > TAIL_NATS).astype(float)), W)
        p["vs_D"] = dict(D=f"{Dc[0]}@3", nll=nll, tail=tl, label=R1C.matched_label(nll, tl),
                         rho=p["bytes"] / float(means.loc[Dc, "bytes"]),
                         rho_mem=p["bytes_stored"] / float(means.loc[Dc, "bytes_stored"]),
                         cat=L.catastrophes(dA[col], dA[Dc]))
    return p


def families(points):
    out = {}
    for lens in ("V16", "V4"):
        out[lens] = {}
        for fam in L.DEPLOYABLE:
            c = [(k, p) for k, p in points.items() if p["lens"] == lens and p["family"] == fam
                 and p.get("vs_D", {}).get("label") == "MATCHED"]
            kb, pb = min(c, key=lambda kp: kp[1]["bytes"]) if c else (None, None)
            r = pb["vs_D"]["rho"] if pb else None
            out[lens][fam] = dict(point=kb, rho=r, verdict=R1C.rho_verdict(r))
    return out


def analyse_main(name, d, sides, regress=None):
    v, dA, S, T, fp_cells, valid, n_drop = _tables(d, "main")
    pidx = dA.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s, pidx).to_numpy()  # noqa: E731
    means = v.groupby(["arm", "B"])[["bytes", "bytes_stored", "bytes_amort", "evict_frac", "kept_width",
                                     "needle_keep", "score"]].mean()
    pts = {}
    for col in dA.columns:
        arm, B = col
        if arm == "fp":
            continue
        pa = L.parse_arm(arm)
        p = _point_stats(dA, col, (DREF[pa["lens"]], 3.0), W, pm, means)
        p.update(arm=arm, B=float(B), lens=pa["lens"], family=pa["family"], score=float(means.loc[col, "score"]),
                 dA_by_task={t: float(g.mean()) for t, g in dA[col].groupby(level="task")})
        pts[f"{arm}@{bk(B)}"] = p
    res = dict(cell=name, fp=float(fp_cells.mean()), fp_by_task=fp_cells.round(4).to_dict(), valid_tasks=valid,
               n_prompts=int(len(pidx)), n_prompt_tasks=int(len(dA)), nll_dropped=n_drop, points=pts,
               families=families(pts))
    # E1: the source of the read gain
    res["e1"] = {}
    D16 = ("uniform", 3.0)
    for col in dA.columns:
        if col[0] != "qreadfp_v16" or ("qread_v16", col[1]) not in dA.columns or D16 not in dA.columns:
            continue
        r = col[1]
        q16 = ("qread_v16", r)
        gq = L1C.boot_ci(pm(dA[D16] - dA[q16]), W)
        gfp = L1C.boot_ci(pm(-dA[col]), W)
        qq = L1C.boot_ci(pm((dA[D16] - dA[q16]) + dA[col]), W)
        fp_nll = L1C.boot_ci(pm(dA[col]), W)
        fp_tail = L1C.boot_ci(pm((dA[col] > TAIL_NATS).astype(float)), W)
        res["e1"][bk(r)] = dict(G_q=gq, G_fp=gfp, Q=qq, label=L.source_label(gq, gfp, qq),
                                qreadfp_vs_fp=R1C.matched_label(fp_nll, fp_tail),
                                G_q_by_task={t: float(g.mean()) for t, g in (dA[D16] - dA[q16]).groupby(level="task")},
                                G_fp_by_task={t: float(g.mean()) for t, g in (-dA[col]).groupby(level="task")})
    # E2: the router tail (Llama tail cells) / Qwen's nested router
    pr = sides[0]["preset"]
    Bt = pr.get("B_target")
    res["e2"] = None
    if Bt is not None and any(a in L.ROUTE_SOURCE for a, _ in plan_of(sides[0])):
        Bt = float(Bt)
        cols = {k: (a, Bt) for k, a in (("seq2", "router_seq2_calib"), ("nest2", "router_nest2_calib"),
                                         ("seq3", "router_seq3_calib"), ("nest3", "router_nest3_calib"))}
        have = {k: c for k, c in cols.items() if c in dA.columns}
        e2 = dict(B_target=Bt, catastrophes={k: L.catastrophes(dA[c], dA[D16]) for k, c in have.items()},
                  labels={k: pts[f"{c[0]}@{bk(Bt)}"]["vs_D"]["label"] for k, c in have.items()},
                  rho={k: pts[f"{c[0]}@{bk(Bt)}"]["vs_D"]["rho"] for k, c in have.items()})
        if len(have) == 4:
            nest = ((dA[have["nest2"]] - dA[have["seq2"]]) + (dA[have["nest3"]] - dA[have["seq3"]])) / 2
            cal = ((dA[have["seq3"]] - dA[have["seq2"]]) + (dA[have["nest3"]] - dA[have["nest2"]])) / 2
            zn, zc = L1C.boot_ci(pm(nest), W), L1C.boot_ci(pm(cal), W)
            e2["nest_effect"] = dict(diff=zn, label=L.effect_label(*zn, "NEST"))
            e2["cal_effect"] = dict(diff=zc, label=L.effect_label(*zc, "CAL"))
        fix, base = ("nest3", "seq2") if "nest3" in have else ("nest2", "seq2")
        if fix in have and base in have:
            fx = f"{have[fix][0]}@{bk(Bt)}"
            fx4 = f"{have[fix][0]}+v4@{bk(Bt)}"
            both = (pts[fx]["vs_D"]["label"] == "MATCHED"
                    and pts.get(fx4, {}).get("vs_D", {}).get("label") == "MATCHED")
            reg_fix = (regress or {}).get(fx, [])
            reg_base = (regress or {}).get(f"{have[base][0]}@{bk(Bt)}", [])
            e2.update(fix=fix, base=base, fix_matched_both_lenses=both,
                      fix_minus_base=L1C.boot_ci(pm(dA[have[fix]] - dA[have[base]]), W),
                      regress_fixed={"fix": reg_fix, "base": reg_base})
            e2["tail_label"] = L.tail_label(both, e2["catastrophes"][fix], e2["catastrophes"][base],
                                            reg_fix, reg_base, float((dA[have[fix]] - dA[have[base]]).mean()))
        res["e2"] = e2
    return res


def analyse_regress(d):
    """Per arm (V16 and V4), per Stage 1d catastrophic prompt-task: dA(X) - dA(D_L)
    and FIXED (<= 2 nats)."""
    A = d.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values="a_sum_nll", aggfunc="first")
    A = A[~A[("fp", 0.0)].isna()]
    dA = A.sub(A[("fp", 0.0)], axis=0)
    out, fixed = {}, {}
    for col in dA.columns:
        arm, B = col
        if arm == "fp":
            continue
        pa = L.parse_arm(arm)
        Dc = (DREF[pa["lens"]], 3.0)
        if Dc not in dA.columns or col == Dc:
            continue
        diff = (dA[col] - dA[Dc]).round(3)
        out[f"{arm}@{bk(B)}"] = {f"{p}/{t}": float(x) for (p, t), x in diff.items()}
        if pa["lens"] == "V16":
            fixed[f"{arm}@{bk(B)}"] = [bool(x <= FIXED_NATS) for x in diff.to_numpy()]
    return out, fixed


def analyse_reuse(name, d, sides):
    v, dA, S, T, fp_cells, valid, n_drop = _tables(d, "reuse")
    out = dict(cell=name, fp=float(fp_cells.mean()), fp_by_task=fp_cells.round(4).to_dict(), valid_tasks=valid,
               nll_dropped=n_drop, roles={}, reuse={}, q2_by_task={})
    for role in ("Q1", "Q2"):
        sub = dA.xs(role, level="q_role", drop_level=False)
        pidx = sub.index.droplevel("q_role").unique()
        W = BM.boot_weights(pidx)
        pm = lambda s, pidx=pidx: R1C.prompt_means(s.droplevel("q_role") if "q_role" in s.index.names else s,  # noqa: E731
                                                   pidx).to_numpy()
        vv = v[v.q_role == role]
        means = vv.groupby(["arm", "B"])[["bytes", "bytes_stored", "bytes_amort", "evict_frac", "kept_width",
                                          "needle_keep", "score"]].mean()
        pts = {}
        for col in sub.columns:
            arm, B = col
            if arm == "fp" or sub[col].isna().all():
                continue
            pa = L.parse_arm(arm)
            p = _point_stats(sub, col, (DREF[pa["lens"]], 3.0), W, pm, means)
            p.update(arm=arm, B=float(B), lens=pa["lens"], family=pa["family"],
                     score=float(means.loc[col, "score"]))
            pts[f"{arm}@{bk(B)}"] = p
        out["roles"][role] = dict(n_prompts=int(len(pidx)), points=pts, families=families(pts))
    q2 = dA.xs("Q2", level="q_role")
    t2 = T.xs("Q2", level="q_role")
    pidx2 = q2.index.unique()
    for lens, vb in (("V16", 16), ("V4", 4)):
        per_r = {}
        for col in dA.columns:
            if col[0] != f"snapq_v{vb}":
                continue
            r = col[1]
            qa = f"qread_v{vb}@{bk(r)}"
            sa = f"snapq_v{vb}@{bk(r)}"
            z = dict(qread_q1=out["roles"]["Q1"]["points"].get(qa, {}).get("vs_D", {}).get("label"),
                     qread_q2=out["roles"]["Q2"]["points"].get(qa, {}).get("vs_D", {}).get("label"),
                     snapq_q2=out["roles"]["Q2"]["points"].get(sa, {}).get("vs_D", {}).get("label"))
            per_r[bk(r)] = z
            diff = q2[col] - q2[(f"qread_v{vb}", r)]
            by = {}
            for task in sorted(t2[col].dropna().unique()):
                m = (t2[col] == task).to_numpy()
                ix = diff.index[m]
                Wt = BM.boot_weights(ix)
                by[task] = L1C.boot_ci(diff[m].to_numpy(), Wt)
            W2 = BM.boot_weights(pidx2)
            out["q2_by_task"][f"{sa} - {qa}"] = dict(all=L1C.boot_ci(diff.reindex(pidx2).to_numpy(), W2), **by)
        out["reuse"][lens] = dict(per_r=per_r, label=L.reuse_label(per_r))
    return out


# ------------------------------------------------------------------ report
def _points_table(points):
    Lh = ["| arm@B | lens | bytes | ρ | ρ mem | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label | cat>3 "
          "| max dA | score | evicted | needle kept |",
          "|---|---|---:|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for k, p in sorted(points.items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
        z = p.get("vs_D")
        Lh.append(f"| {k} | {p['lens']} | {p['bytes']:.1f} | {fmt(z and z['rho'], 2)} | {fmt(z and z['rho_mem'], 2)} "
                  f"| {ci(p['dA'])} | {ci(z['nll']) if z else '(D)'} | {p['tail']:.3f} | "
                  f"{ci(z['tail']) if z else '—'} | {z['label'] if z else '—'} | {z['cat'] if z else '—'} | "
                  f"{p['max_dA']:.2f} | {p['score']:.3f} | {p['evict_frac']:.1%} | {fmt(p['needle_keep'])} |")
    return Lh


def _fam_table(fams):
    Lh = ["| lens | " + " | ".join(L.DEPLOYABLE) + " |", "|---|" + "---|" * len(L.DEPLOYABLE)]
    for lens, fz in fams.items():
        Lh.append(f"| {lens} | " + " | ".join(z["verdict"] + (f" {z['point']} ρ {z['rho']:.2f}" if z["point"] else "")
                                              for z in (fz[f] for f in L.DEPLOYABLE)) + " |")
    return Lh


def report(results, summary, out_stem, title):
    Lh = [f"# R14 Stage 1e — {title} (read_stage1e.py; rules frozen in its docstring)", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for r in results:
        if "roles" in r:
            Lh += [f"## {r['cell']} — FP {r['fp']:.3f} ({r['fp_by_task']}), dropped {r['nll_dropped']}", ""]
            for lens, z in r["reuse"].items():
                Lh.append(f"**E3 {lens}: {z['label']}** — " + "; ".join(
                    f"r={rr}: qread Q1 {x['qread_q1']}, Q2 {x['qread_q2']}; snapq Q2 {x['snapq_q2']}"
                    for rr, x in z["per_r"].items()))
            for k, z in r["q2_by_task"].items():
                Lh.append(f"- Q2 {k}: " + "; ".join(f"{t} {ci(v)}" for t, v in z.items()))
            for role, rz in r["roles"].items():
                Lh += ["", f"### {role} ({rz['n_prompts']} prompts)", ""] + _fam_table(rz["families"]) + [""] \
                    + _points_table(rz["points"])
            Lh.append("")
            continue
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f} ({r['fp_by_task']}), {r['n_prompts']} prompts, "
               f"{r['n_prompt_tasks']} prompt-tasks (dropped {r['nll_dropped']}), fp replay "
               f"{r['fp_replay_agreement']:.3f}, answer-value coverage {r['value_cover']:.2f}", ""]
        Lh += _fam_table(r["families"]) + [""] + _points_table(r["points"]) + [""]
        for rr, z in r["e1"].items():
            Lh.append(f"**E1** r={rr}: G_q {ci(z['G_q'])}, G_fp {ci(z['G_fp'])}, Q {ci(z['Q'])} → **{z['label']}**; "
                      f"qreadfp vs FP {z['qreadfp_vs_fp']}")
        e2 = r.get("e2")
        if e2:
            s = (f"**E2** B_t={bk(e2['B_target'])}: catastrophes {e2['catastrophes']}; labels {e2['labels']}; "
                 f"ρ { {k: round(v, 3) for k, v in e2['rho'].items()} }")
            if "nest_effect" in e2:
                s += (f"; nesting {ci(e2['nest_effect']['diff'])} → {e2['nest_effect']['label']}; calibration "
                      f"{ci(e2['cal_effect']['diff'])} → {e2['cal_effect']['label']}")
            if "tail_label" in e2:
                s += (f"; {e2['fix']} − {e2['base']} {ci(e2['fix_minus_base'])} → **{e2['tail_label']}**")
            Lh.append(s)
        if r.get("regress"):
            Lh.append("**Regression** (dA − dA_D per Stage 1d catastrophic prompt-task): " + "; ".join(
                f"{k} {v}" for k, v in r["regress"].items()))
        if r.get("cal"):
            Lh.append(f"Calibration: {r['cal']}")
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def cal_summary(cal):
    rows, side, search, slog, routes = cal
    out = {}
    for k in routes["routes_pool"]:
        s = search[search.B.astype(float) == float(k)] if len(search) else search
        out[k] = dict(prompt_tasks=int(len(s)), fail=int(s.fail.sum()) if len(s) else 0,
                      fail_by_task=s.groupby("task").fail.sum().astype(int).to_dict() if len(s) else {},
                      dense_R0=len(L1C.dense_heads(routes["routes_pool"][k])),
                      dense_seq3=len(L1C.dense_heads(routes["routes_seq3"][k])),
                      dense_nest3=len(L1C.dense_heads(routes["routes_nest3"][k])))
    return out


def read_cells(cells, out_stem, root=RESULTS, title="", regress=None, cals=None):
    """cells: name -> (tag, [jobs]); regress: name -> (tag, job); cals: name -> (tag, job)."""
    regress, cals = regress or {}, cals or {}
    results, problems, summary = [], [], {}
    for name, (tag, jobs) in cells.items():
        mode = "reuse" if tag.startswith("reuse") else "evaluate"
        parts = [load_run(tag, j, mode, root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{name}: blocks ran different plans")
        is_qwen = sides[0]["model"].startswith("qwen")
        kind = "reuse" if mode == "reuse" else "main"
        agree, cover = validate(d, sides, problems, kind, fp_tasks_min=2 if is_qwen else 3)
        cal = load_cal(*cals[name], root) if name in cals else None
        if any(a.startswith(("router_seq3", "router_nest3")) for a, _ in plan_of(sides[0])) and cal is None:
            problems.append(f"{name}: Stage 1e routers need their calibration job")
        validate_routes(sides, cal, problems)
        if tag.startswith("tail") and name not in regress:
            problems.append(f"{name}: no regression block given (E2 needs Stage 1d's catastrophic prompt-tasks)")
        reg = None
        if name in regress:
            rd, rside = load_run(regress[name][0], regress[name][1], "evaluate", root)
            if plan_of(rside) != plan_of(sides[0]):
                problems.append(f"{name}: the regression block ran another plan")
            validate(rd, [rside], problems, "main", main=False)
            reg = analyse_regress(rd)
        if problems:
            continue
        if kind == "reuse":
            r = analyse_reuse(name, d, sides)
            for lens, z in r["reuse"].items():
                summary[f"E3 {name} {lens}"] = z["label"]
        else:
            r = analyse_main(name, d, sides, reg[1] if reg else None)
            r["fp_replay_agreement"], r["value_cover"] = agree, cover
            if reg:
                r["regress"] = reg[0]
            if cal is not None:
                r["cal"] = cal_summary(cal)
            for rr, z in r["e1"].items():
                summary[f"E1 {name} r={rr}"] = z["label"]
            if r.get("e2") and "tail_label" in r["e2"]:
                summary[f"{'E4 QWEN_TAIL' if is_qwen else 'E2'} {name}"] = r["e2"]["tail_label"]
            if is_qwen:
                mv = r["fp_by_task"].get("niah_multivalue")
                if sides[0]["ctx"] == 32768:
                    summary["E4 STOP_FIX"] = "WORKS" if mv is not None and mv >= FP_MIN else f"FAILS (FP {mv})"
                summary[f"E4 QWEN_QREAD_V4 {name}"] = r["families"]["V4"]["qread"]["verdict"]
            elif sides[0]["ctx"] == 32768:
                wins = [f for f, z in r["families"]["V4"].items() if z["verdict"] == "WIN" and f in L.ROUTERS]
                summary["E2 ROUTER_32K"] = ("WIN: " + ", ".join(wins)) if wins else "NO_ROUTER_WIN"
        r["fp_replay_agreement"], r["value_cover"] = agree, cover
        results.append(r)
    if problems:
        raise SystemExit("INVALID Stage 1e data:\n  " + "\n  ".join(problems))
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(nll_margin=NLL_MARGIN, tail_margin=TAIL_MARGIN, tail_nats=TAIL_NATS,
                                  cat_nats=L.CAT_NATS, fixed_nats=FIXED_NATS, win=BM.WIN, tie=BM.TIE,
                                  boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED),
                       summary=summary, cells=results), fh, indent=1, default=str)
    Lh = report(results, summary, out_stem, title)
    print("\n".join(Lh[:2 + len(summary) + 1]))
    return 0


# --------------------------------------------------------------------- gate
def gate(job, tag, root=RESULTS):
    mode = "reuse" if tag.startswith("reuse") else "evaluate"
    d, side = load_run(tag, job, mode, root)
    problems = []
    kind = "reuse" if mode == "reuse" else "main"
    agree, cover = validate(d, [side], problems, kind, main=False)
    pk_dev = side.get("peak_gib_dev_max") or []
    if not pk_dev or not all(x <= PEAK_GIB_MAX for x in pk_dev):
        problems.append(f"peak GPU memory per device {pk_dev} GiB (limit {PEAK_GIB_MAX})")
    if tag == "pilot128e":
        try:
            _, cside, search, slog, routes = load_cal(tag, job, root)
            if not (len(search) and search.searched.all() and (search.iters == 1).all()):
                problems.append("the forced search did not run once per prompt-task and budget")
            for B, grp in search.groupby("B"):
                k = bk(B)
                crit = set()
                for s in grp.itertuples():
                    nh = int(((slog.prompt_idx == s.prompt_idx) & (slog.task == s.task)
                              & (slog.B.astype(float) == float(s.B)) & (slog.level == "head")).sum())
                    if nh != s.n_cand:
                        problems.append(f"B={k}: the forced search tested {nh} of {s.n_cand} heads")
                    crit |= {tuple(h) for h in json.loads(s.critical)}
                if routes["routes_seq3"][k] != L1C.apply_critical(routes["routes_pool"][k], sorted(crit)):
                    problems.append(f"B={k}: routes_seq3 is not R0 plus the forced heads")
        except SystemExit as e:
            problems.append(f"pilot calibration missing: {e}")
    main = L.PILOT_OF[tag]
    n_main = len(L.build_plan(L.PRESETS[main]))
    per_q = 2 if mode == "reuse" else 1
    dec = float(d[(d.arm != "fp") & d.copied_from.isna()].t_arm.median()) if "copied_from" in d else \
        float(d[d.arm != "fp"].t_arm.median())
    tf = float(d.t_tf.median())
    unit_s = (float(d.t_prefill.median()) + float(d.t_precompute.max())) / per_q + n_main * (dec + tf)
    proj_h = unit_s * BLOCK_UNITS[main] / 3600
    if proj_h > 0.9 * WALL_H[main]:
        problems.append(f"projected {main} block {proj_h:.1f} h > 90% of {WALL_H[main]} h")
    print(f"R14 Stage 1e pilot {job} ({tag} for {main}): {len(d)} rows, plan {len(plan_of(side))} arms, fp replay "
          f"{agree:.3f}, answer-value coverage {cover:.2f}, peak per GPU {pk_dev} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected {main} block {proj_h:.1f} h ({n_main} arms, wall {WALL_H[main]} h)")
    cols = [c for c in ("score", "a_sum_nll", "bits_per_token", "evict_frac", "read_frac", "t_arm", "t_tf") if c in d]
    idx = ["arm", "B"] + (["q_role"] if kind == "reuse" else [])
    print(d.pivot_table(index=idx, values=cols, aggfunc="mean").round(3).to_string())
    fpr = d[d.arm == "fp"]
    for r_ in fpr.itertuples():
        print(f"  FP {r_.task}{'/' + r_.q_role if r_.q_role else ''}: score {r_.score:.2f}, pred {r_.pred[:120]!r}")
    if tag.startswith("qwen") and "niah_multivalue" in set(fpr.task):
        mv = float(fpr[fpr.task == "niah_multivalue"].score.mean())
        print(f"  stop rule {side.get('stop_rule')}: FP multivalue on the pilot {mv:.2f} "
              f"({'ok' if mv >= FP_MIN else 'WARNING: the stop fix did not recover the pilot answer'})")
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--pilot-tag", default="pilot128e", choices=sorted(L.PILOT_OF))
    for f in ("tail128", "tail32", "reuse128", "reuse32", "qwen32", "qwen128"):
        ap.add_argument(f"--{f}", nargs="+", metavar="JOB", default=[])
    for f in ("regress128", "regress32", "cal128", "cal32"):
        ap.add_argument(f"--{f}", metavar="JOB")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1e"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(gate(a.pilot, a.pilot_tag, a.results_root))
    cells, reg, cal = {}, {}, {}
    spec = (("tail128", "llama31-8b@131072", "tail128"), ("tail32", "llama31-8b@32768", "tail32"),
            ("reuse128", "reuse llama31-8b@131072", "reuse128"), ("reuse32", "reuse llama31-8b@32768", "reuse32"),
            ("qwen32", "qwen3-30b-a3b-2507@32768", "qwen32e"), ("qwen128", "qwen3-30b-a3b-2507@131072", "qwen128q"))
    for flag, name, tag in spec:
        if getattr(a, flag):
            cells[name] = (tag, getattr(a, flag))
    for flag, name, tag in (("regress128", "llama31-8b@131072", "regress128"),
                            ("regress32", "llama31-8b@32768", "regress32")):
        if getattr(a, flag):
            reg[name] = (tag, getattr(a, flag))
    for flag, name, tag in (("cal128", "llama31-8b@131072", "cal128e"), ("cal32", "llama31-8b@32768", "cal32e")):
        if getattr(a, flag):
            cal[name] = (tag, getattr(a, flag))
    if not cells:
        ap.error("give --pilot JOB or at least one cell")
    title = ", ".join(sorted({"E1/E2" if n.startswith("llama") else "E3" if n.startswith("reuse") else "E4"
                              for n in cells}))
    sys.exit(read_cells(cells, a.out_stem, a.results_root, title, reg, cal))


if __name__ == "__main__":
    main()
