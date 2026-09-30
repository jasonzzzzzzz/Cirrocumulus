#!/usr/bin/env python3
"""R14 Stage 1c gate and reader. The rules below are FROZEN: written 2026-09-29,
before any Stage 1c output existed (design: s1c_lib.py; driver: run_s1c.py).

    python read_stage1c.py --pilot JOB
    python read_stage1c.py --main128 J1 J2 J3 J4 --main32 J5 J6 J7 J8 \
        --cal128 C1 --cal32 C2 [--out-stem stage1c]

Result directories: h0_measurement/results/r14s1c_<tag>_<job>/, where tag is
pilot128, cal128, cal32, main128 or main32.

CELLS Llama-3.1-8B @131072 on prompts 5000-5039 and @32768 on prompts 5100-5139,
  in blocks of 10. Both ranges are fresh (no earlier stage read them). The
  calibration uses prompts 0-9. Every arm of a prompt-task runs in one process.
METRIC per prompt-task and arm X:
  dNLL(X) = tf_c_sum_nll(X) - tf_c_sum_nll(fp).
  tf_c_sum_nll is the teacher-forced NLL of FP's own greedy answer, over its
  content tokens (letters or digits), through X's view. Task cells whose FP
  score is < 0.9 drop, as in Stages 0-1b. A prompt-task with an empty FP
  answer drops from the NLL statistics.
COMPARATOR D_L = uniform@3 (TurboQuant-3) in X's value lens L:
  - V16: exact values;
  - V4 / V2: TurboQuant-MSE values at 4 / 2 bits.
INTERVALS Prompt bootstrap: prompts resampled within their block, all tasks of a
  prompt together, 10,000 draws, seed 14; 90% percentile intervals [lo, hi].
MATCHED quality, X against D_L, paired within prompt-tasks:
  - NLL: hi of mean(dNLL(X) - dNLL(D_L)) <= 0.10 nats;
  - TAIL: hi of (tail(X) - tail(D_L)) <= 0.05, where tail is the share of
    prompt-tasks with dNLL > 2 nats.
  X is MATCHED iff both hold. It is WORSE iff the NLL lo > 0.10 or the tail
  lo > 0.05. Otherwise it is INCONCLUSIVE.
BYTES Per context token, per KV head, per layer, read by one decode step (d = head
  dim):
      (d/8)(key bits + key side) + (1 - f)(d/8)(v + v side) + tail.
  - Key side: dense 16/d; routers (1-f)16/d + 3/d; vah and qread (1-f)16/d + 1/d.
  - Value side: 16/d when quantized.
  - The tail is (window + question + answer/2) x 4d bytes / context length.
  - For qread, bits and f are what the answer READS; its stored bytes (memory)
    are D_L's and are reported.
  rho = bytes(X) / bytes(D_L).
VERDICT Per deployable family, cell and lens, take the cheapest MATCHED point. The
  families are the designs vah (the proxy-chosen vah_v* points and the fixed-width
  vahw*_v* points), seq and qread, and the references sieve, pool and union:
  - WIN: rho <= 0.80;
  - TIE: rho <= 1.00;
  - LOSS: otherwise;
  - NO_POINT: no MATCHED point.
  router_pool_oracle is a diagnostic and never counts.
DECISION
  - GO_KERNEL if any family is WIN in V4 in at least one cell. The output names
    each (family, cell); the kernel targets that family's layout. A qread WIN
    needs a sparse-read kernel and saves no memory.
  - Else SCOPE_EXACT_V if any family is WIN in V16.
  - Else STOP_SYSTEMS.
  V2 is reported, not gating.
PRE-REGISTERED QUESTIONS (reported; labels fixed):
  Q1 vah. The proxy-chosen points' mean kept-key width per (cell, rho, lens).
     LOWERS_PRECISION if it falls strictly V16 > V4 > V2 at every (cell, rho);
     MIXED otherwise. Also, at rho = 0.75, mean(dNLL(vahw_w) - dNLL(vah)) for
     w = 3 and 4 in every lens: the precision the budget line prefers. And the
     cheapest MATCHED vah point per lens.
  Q2 seq. Per (cell, B): mean(dNLL(seq) - dNLL(pool)) with its interval.
     SEQ_HELPS if hi < 0, SEQ_HURTS if lo > 0, SEQ_NO_EFFECT otherwise. Also
     the share of the pool-to-oracle gap it closes, mean(pool - seq) /
     mean(pool - oracle), and the same labels for union (UNION_*).
  Q3 qread. Per (cell, lens): the smallest r whose point is MATCHED
     (QREAD_MATCHED@r), else QREAD_NOT_MATCHED. Also mean(dNLL(qread) -
     dNLL(D_L)) per r: the selection's own cost, since qread reads D_L's own
     stored keys and values.
VALIDITY (any failure -> INVALID, no verdict is printed)
  V1 Every (prompt, task) block holds exactly the sidecar plan's (arm, B) rows,
     once, for every planned prompt and task. All blocks of a cell share one
     corpus and one plan. FP >= 0.9 on >= 3 of the 4 tasks.
  V2 Audits:
     - dense = w bits with nothing evicted;
     - routers spend <= B;
     - vah stays within its budget T;
     - qread stores 3 bits with nothing evicted, and reads floor(r C) of C;
     - a twin has its base's bits and eviction.
  V3 fp's teacher-forced replay reproduces fp's greedy answer on >= 98% of the
     answer steps (pooled).
  V4 Every twin changed its values: mean |NLL(twin) - NLL(base)| > 0.
  V5 The calibration:
     - routes_seq and routes_union equal Stage 1b's routes_pool except for heads
       switched to dense;
     - the base routes file is unchanged (sha256);
     - every evaluation block used this calibration's routes file.
REPORTED, NOT GATED
  - the task score, and its paired difference vs D_L and vs FP;
  - the Stage 0 lossless rule;
  - the worst 5%'s share of the damage (Part D), CVaR at 5% and the maximum dNLL;
  - stored bytes (memory) and answer-amortised bytes;
  - needle_keep and the kept-key width;
  - the value quantizer's own cost (fp twins);
  - the calibration statistics;
  - every point against uniform@4 as well.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_model as BM  # noqa: E402
import s1c_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"main128": 10.0, "main32": 4.0}
MAIN_BLOCK_PROMPT_TASKS = 40
V3_MIN = 0.98
NLL_MARGIN, TAIL_MARGIN, TAIL_NATS, WORST_FRAC = 0.10, 0.05, 2.0, 0.05
FP_MIN = BM.RMT.FP_MIN
DEPLOYABLE = L.DESIGNS + L.REFERENCES
DREF = {"V16": "uniform", "V4": "uniform+v4", "V2": "uniform+v2"}


def bk(B) -> str:
    B = float(B)
    return str(int(B)) if B.is_integer() else f"{B:g}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) \
        else f"{x:.{nd}f}"


# ------------------------------------------------------------------ loading
def run_dir(tag, job, root=RESULTS):
    return os.path.join(root, f"r14s1c_{tag}_{job}")


def load_run(tag, job, mode="evaluate", root=RESULTS):
    d = run_dir(tag, job, root)
    pq = [x for x in glob.glob(os.path.join(d, f"s1c_{mode}_*.parquet"))
          if not x.endswith(("_heads.parquet", "_search.parquet", "_searchlog.parquet"))]
    js = glob.glob(os.path.join(d, f"s1c_{mode}_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one s1c_{mode} parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def load_cal(tag, job, root=RESULTS):
    rows, side = load_run(tag, job, "calibrate", root)
    d = side["_dir"]
    search = pd.read_parquet(os.path.join(d, side["search"]))
    slog = pd.read_parquet(os.path.join(d, side["searchlog"]))
    routes = json.load(open(side["write_routes"]))
    return rows, side, search, slog, routes


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


# ----------------------------------------------------------------- validity
def validate(d, sides, problems, main=True):
    """V1-V4; returns fp's replay agreement."""
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(["prompt_idx", "task"]).apply(
            lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))), include_groups=False)
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: (prompt, task) with the wrong arms: {bad[:3]}")
        n_want = side["n_prompts"] * len(side["tasks"])
        if len(got) != n_want:
            problems.append(f"block {side['_job']}: {len(got)} of {n_want} prompt-tasks")
    if d.duplicated(["job", "prompt_idx", "task", "arm", "B"]).any():
        problems.append("duplicate rows")
    if d.corpus_sha.nunique() != 1:
        problems.append(f"blocks disagree on corpus: {sorted(d.corpus_sha.unique())}")
    fam = d.arm.map(L.family)
    dn = d[fam == "dense"]
    if not (np.allclose(dn.bits_per_token, dn.B) and (dn.evict_frac == 0).all()):
        problems.append("a dense row is not w bits with nothing evicted")
    rt = d[fam.isin(L.ROUTER_FAMILIES)]
    if (rt.bits_per_token > rt.B + 1e-6).any():
        problems.append("a router row overspends its budget")
    vh = d[fam == "vah"]
    if len(vh) and (vh.vah_total_bits > vh.vah_T + 1e-9).any():
        problems.append("a vah row exceeds its total budget T")
    qr = d[fam == "qread"]
    if len(qr):
        k = np.array([L.qread_keep_count(r, c) / c for r, c in zip(qr.B, qr.ctx_len)])
        if not (np.allclose(qr.stored_bits_per_token, L.QREAD_STORE_WIDTH)
                and (qr.stored_evict_frac == 0).all()
                and np.allclose(qr.read_frac, k, atol=1e-9)
                and np.allclose(qr.bits_per_token, L.QREAD_STORE_WIDTH * k, atol=1e-6)):
            problems.append("a qread row does not store 3 bits dense and read floor(r C)")
    key = ["job", "prompt_idx", "task"]
    tw = d[d.twin != ""]
    for arm in sorted(tw.arm.unique()):
        t = d[d.arm == arm]
        b = d[d.arm == L.parse_arm(arm)["base"]]
        m = t.merge(b, on=key + ["B"], suffixes=("", "_b"))
        if len(m) != len(t):
            problems.append(f"{arm}: {len(m)} of {len(t)} rows have a base row")
            continue
        if not (np.allclose(m.bits_per_token, m.bits_per_token_b)
                and np.allclose(m.evict_frac, m.evict_frac_b)):
            problems.append(f"{arm}: bits or eviction differ from its base")
        if not (m.tf_sum_nll - m.tf_sum_nll_b).abs().mean() > 0:
            problems.append(f"{arm}: values did not change anything (V4)")
    fp = d[d.arm == "fp"].dropna(subset=["tf_top1"])
    agree = float((fp.tf_top1 * fp.tf_len).sum() / max(fp.tf_len.sum(), 1))
    if agree < V3_MIN:
        problems.append(f"fp teacher-forced replay agrees with fp's greedy on {agree:.3f} < {V3_MIN}")
    if main:
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        if (fps >= FP_MIN).sum() < min(3, len(fps)):
            problems.append(f"FP below {FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree


def validate_cal(cal, sides, budgets, problems):
    """V5 for one cell: the calibration's routes against Stage 1b's base, and
    every evaluation block's provenance against this calibration's file."""
    _, cside, search, _, routes = cal
    meta = routes["meta"]
    base = meta["base_routes"]
    try:
        sha = sha256(base["path"])
    except OSError as e:
        problems.append(f"calibration base routes unreadable: {e}")
        return
    if sha != base["sha256"]:
        problems.append(f"{base['path']} changed since the calibration read it")
    j1b = json.load(open(base["path"]))
    for B in budgets:
        k = bk(B)
        if routes["routes_pool"].get(k) != j1b["routes_pool"].get(k):
            problems.append(f"calibration routes_pool@{k} is not Stage 1b's")
        for field in ("routes_seq", "routes_union"):
            if k not in routes.get(field, {}) or not L.only_densified(routes["routes_pool"][k],
                                                                      routes[field][k]):
                problems.append(f"calibration {field}@{k} changes more than dense switches")
    if meta.get("rule", {}).get("forced"):
        problems.append("the calibration ran with --force-search (mechanics only)")
    path = os.path.abspath(cside["write_routes"])
    csha = sha256(path)
    for s in sides:
        for k_, v in s["routes"].items():
            if k_.startswith(("router_seq_calib", "router_union_calib")) and \
                    (os.path.abspath(v["path"]) != path or v["sha256"] != csha):
                problems.append(f"block {s['_job']}: {k_} read {v['path']}, not the calibration's "
                                f"{path} (or the file changed)")


# ---------------------------------------------------------------- analysis
def add_bytes(v):
    v = v.copy()
    d8 = v.head_dim / 8.0
    vs = np.where(v.v_bits >= 16, 0.0, 16.0 / v.head_dim)
    tail = (v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len
    v["bytes"] = (d8 * (v.bits_per_token + v.key_side) + (1 - v.evict_frac) * d8 * (v.v_bits + vs)
                  + tail)
    fam = v.arm.map(L.family)
    sk = np.where(fam == "qread", 16.0 / v.head_dim, v.key_side)
    v["bytes_stored"] = (d8 * (v.stored_bits_per_token + sk)
                         + (1 - v.stored_evict_frac) * d8 * (v.v_bits + vs) + tail)
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


def prompt_means(frame, pidx):
    return frame.groupby(level=["job", "prompt_idx"]).mean().reindex(pidx)


def matched_label(nll, tail):
    if nll[2] <= NLL_MARGIN and tail[2] <= TAIL_MARGIN:
        return "MATCHED"
    if nll[1] > NLL_MARGIN or tail[1] > TAIL_MARGIN:
        return "WORSE"
    return "INCONCLUSIVE"


def rho_verdict(r):
    if r is None:
        return "NO_POINT"
    return "WIN" if r <= BM.WIN else ("TIE" if r <= BM.TIE else "LOSS")


def analyse_cell(name, d, sides):
    fp_task = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_task.items() if s >= FP_MIN)
    v = add_bytes(d[d.task.isin(valid)])
    key = ["job", "prompt_idx", "task"]
    nll = v.pivot_table(index=key, columns=["arm", "B"], values="tf_c_sum_nll", aggfunc="first")
    sco = v.pivot_table(index=key, columns=["arm", "B"], values="score", aggfunc="first")
    drop = nll.isna().any(axis=1)
    n_drop = int(drop.sum())
    nll, sco_nll = nll[~drop], sco[~drop]
    dn = nll.sub(nll[("fp", 0.0)], axis=0)
    pidx = dn.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    spm = sco.groupby(level=["job", "prompt_idx"]).mean()
    Ws = BM.boot_weights(spm.index)
    ll = BM.lossless_table(spm, Ws)
    means = v.groupby(["arm", "B"])[["bytes", "bytes_stored", "bytes_amort", "bits_per_token",
                                     "evict_frac", "kept_width", "needle_keep"]].mean()

    def pair(a, b, tbl=dn, idx=pidx, w=W):
        return L.boot_ci(prompt_means(tbl[a] - tbl[b], idx).to_numpy(), w)

    def tail_pair(a, b):
        t = (dn[a] > TAIL_NATS).astype(float) - (dn[b] > TAIL_NATS).astype(float)
        return L.boot_ci(prompt_means(t, pidx).to_numpy(), W)

    pts = {}
    for col in dn.columns:
        arm, B = col
        if arm == "fp":
            continue
        pa = L.parse_arm(arm)
        lens, fam = pa["lens"], pa["family"]
        Dc = (DREF[lens], 3.0)
        x = dn[col].to_numpy()
        p = dict(arm=arm, B=float(B), lens=lens, family=fam,
                 bytes=float(means.loc[col, "bytes"]),
                 bytes_stored=float(means.loc[col, "bytes_stored"]),
                 bytes_amort=float(means.loc[col, "bytes_amort"]),
                 key_bits=float(means.loc[col, "bits_per_token"]),
                 evict_frac=float(means.loc[col, "evict_frac"]),
                 kept_width=float(means.loc[col, "kept_width"]),
                 needle_keep=float(means.loc[col, "needle_keep"]),
                 dnll=L.boot_ci(prompt_means(dn[col], pidx).to_numpy(), W),
                 tail=L.tail_share(x, TAIL_NATS), worst5_share=L.worst_share(x, WORST_FRAC),
                 cvar5=L.cvar(x, WORST_FRAC), max_dnll=float(np.max(x)),
                 score=ll[col]["score"], dscore_fp=(ll[col]["delta"], ll[col]["lo"], ll[col]["hi"]),
                 lossless=ll[col]["lossless"])
        if Dc in dn.columns and col != Dc:
            p["vs_D"] = dict(D=f"{Dc[0]}@3", nll=pair(col, Dc), tail=tail_pair(col, Dc),
                             dscore=L.boot_ci(prompt_means(sco_nll[col] - sco_nll[Dc],
                                                           pidx).to_numpy(), W),
                             rho=p["bytes"] / float(means.loc[Dc, "bytes"]),
                             rho_mem=p["bytes_stored"] / float(means.loc[Dc, "bytes_stored"]))
            p["vs_D"]["label"] = matched_label(p["vs_D"]["nll"], p["vs_D"]["tail"])
        D4 = (DREF[lens], 4.0)
        if D4 in dn.columns and col != D4:
            p["vs_uniform4"] = dict(nll=pair(col, D4), tail=tail_pair(col, D4))
        pts[f"{arm}@{bk(B)}"] = p
    res = dict(cell=name, fp=float(fp_task.mean()), fp_by_task=fp_task.round(4).to_dict(),
               valid_tasks=valid, n_prompts=int(len(pidx)), n_prompt_tasks=int(len(dn)),
               nll_dropped=n_drop, points=pts, families={})
    for lens in ("V16", "V4", "V2"):
        res["families"][lens] = {}
        for fam in DEPLOYABLE:
            c = [(k, p) for k, p in pts.items() if p["lens"] == lens and p["family"] == fam
                 and p.get("vs_D", {}).get("label") == "MATCHED"]
            kbest, pbest = min(c, key=lambda kp: kp[1]["bytes"]) if c else (None, None)
            r = pbest["vs_D"]["rho"] if pbest else None
            res["families"][lens][fam] = dict(point=kbest, rho=r, verdict=rho_verdict(r),
                                              n_points=sum(1 for p in pts.values()
                                                           if p["lens"] == lens
                                                           and p["family"] == fam))
    preset = sides[0]["preset"]
    # Q1: the value-aware hybrid's kept width, by lens
    q1 = {}
    for rho in sorted({float(r) for r, _ in preset["vah"]}, reverse=True):
        w = {LEN: pts.get(f"vah_v{vb}@{bk(rho)}", {}).get("kept_width")
             for vb, LEN in ((16, "V16"), (4, "V4"), (2, "V2"))}
        ok = all(w[x] is not None for x in w) and w["V16"] > w["V4"] > w["V2"]
        q1[bk(rho)] = dict(kept_width=w, lowers=bool(ok))
    fixed = {}
    for rho, vb, ww in preset.get("vahw", []):
        a_, b_ = (L.vah_arm(rho, vb, ww), float(rho)), (L.vah_arm(rho, vb), float(rho))
        if a_ in dn.columns and b_ in dn.columns:
            fixed[f"{a_[0]}@{bk(rho)}"] = pair(a_, b_)
    res["q1"], res["q1_fixed_vs_proxy"] = q1, fixed
    # Q2: sequence calibration and the union ablation against the pooled router
    q2 = {}
    for B in preset["seq"]:
        B = float(B)
        pool, seq = ("router_pool_calib", B), ("router_seq_calib", B)
        ora, uni = ("router_pool_oracle", B), ("router_union_calib", B)
        if pool not in dn.columns or seq not in dn.columns:
            continue
        z = dict(seq_minus_pool=pair(seq, pool))
        lo_, hi_ = z["seq_minus_pool"][1], z["seq_minus_pool"][2]
        z["label"] = "SEQ_HELPS" if hi_ < 0 else ("SEQ_HURTS" if lo_ > 0 else "SEQ_NO_EFFECT")
        if uni in dn.columns:
            z["union_minus_pool"] = pair(uni, pool)
            lo_, hi_ = z["union_minus_pool"][1], z["union_minus_pool"][2]
            z["union_label"] = ("UNION_HELPS" if hi_ < 0 else
                                "UNION_HURTS" if lo_ > 0 else "UNION_NO_EFFECT")
        if ora in dn.columns:
            z["oracle_minus_pool"] = pair(ora, pool)
            num = prompt_means(dn[pool] - dn[seq], pidx).to_numpy()
            den = prompt_means(dn[pool] - dn[ora], pidx).to_numpy()
            bn, bd = (W @ num) / W.sum(1), (W @ den) / W.sum(1)
            ok = bd > 0
            gc = bn[ok] / bd[ok]
            z["gap_closed"] = (float(num.mean() / den.mean()) if den.mean() > 0 else float("nan"),
                               *(np.percentile(gc, [5, 95]).tolist() if ok.any()
                                 else [float("nan")] * 2))
        q2[bk(B)] = z
    res["q2"] = q2
    # Q3: question-time reads
    q3 = {}
    for lens in ("V16", "V4", "V2"):
        c = sorted((p["B"], k, p) for k, p in pts.items()
                   if p["family"] == "qread" and p["lens"] == lens)
        if not c:
            continue
        m = [B for B, _, p in c if p.get("vs_D", {}).get("label") == "MATCHED"]
        q3[lens] = dict(label=f"QREAD_MATCHED@{bk(min(m))}" if m else "QREAD_NOT_MATCHED",
                        cost={bk(B): p["vs_D"]["nll"] for B, _, p in c})
    res["q3"] = q3
    # the value quantizer's own cost
    res["fp_value_cost"] = {k: pts[k]["dnll"] for k in ("fp+v4@0", "fp+v2@0") if k in pts}
    return res


def decide(results):
    wins = {lens: [(fam, r["cell"]) for r in results for fam, z in r["families"][lens].items()
                   if z["verdict"] == "WIN"] for lens in ("V16", "V4", "V2")}
    if wins["V4"]:
        dec = "GO_KERNEL"
    elif wins["V16"]:
        dec = "SCOPE_EXACT_V"
    else:
        dec = "STOP_SYSTEMS"
    q1 = all(z["lowers"] for r in results for z in r["q1"].values()) if results else False
    return dict(decision=dec, wins=wins, q1="LOWERS_PRECISION" if q1 else "MIXED")


def cal_summary(cal, budgets):
    rows, side, search, slog, routes = cal
    out = {}
    for B in budgets:
        k = bk(B)
        s = search[search.B.astype(float) == float(B)]
        out[k] = dict(prompt_tasks=int(len(s)), fail=int(s.fail.sum()),
                      searched=int(s.searched.sum()), replays=int(s.n_replays.sum()),
                      rescued=int(((s.fp_min - s.final_min) <= L.TAU_FAIL)[s.fail].sum()),
                      dense_R0=len(L.dense_heads(routes["routes_pool"][k])),
                      dense_seq=len(L.dense_heads(routes["routes_seq"][k])),
                      dense_union=len(L.dense_heads(routes["routes_union"][k])),
                      t_search_h=float(s.t_search.sum() / 3600))
    return out


def ci(t, nd=3):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def report(results, dec, cals, out_stem):
    Lh = ["# R14 Stage 1c (read_stage1c.py; rules frozen in its docstring)", "",
          f"**Decision: {dec['decision']}** · V4 WINs: "
          + (", ".join(f"{f} @ {c}" for f, c in dec["wins"]["V4"]) or "none")
          + " · V16 WINs: " + (", ".join(f"{f} @ {c}" for f, c in dec["wins"]["V16"]) or "none")
          + " · V2 (reported): " + (", ".join(f"{f} @ {c}" for f, c in dec["wins"]["V2"]) or "none"),
          f"**Q1 (value-aware hybrid lowers kept precision as values get cheaper): {dec['q1']}**", ""]
    for r in results:
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f}, {r['n_prompts']} prompts, "
               f"{r['n_prompt_tasks']} prompt-tasks (NLL drops {r['nll_dropped']}), tasks "
               f"{', '.join(r['valid_tasks'])}, fp replay agreement {r['fp_replay_agreement']:.3f}", "",
               "Families: cheapest MATCHED point (NLL hi ≤ 0.10 and tail hi ≤ 0.05 vs "
               "uniform@3 in the lens) and its byte ratio.", "",
               "| lens | " + " | ".join(DEPLOYABLE) + " |", "|---|" + "---|" * len(DEPLOYABLE)]
        for lens, fams in r["families"].items():
            Lh.append(f"| {lens} | " + " | ".join(
                f"{z['verdict']}" + (f" {z['point']} ρ {z['rho']:.2f}" if z["point"] else "")
                for z in (fams[f] for f in DEPLOYABLE)) + " |")
        Lh += ["", "| arm@B | lens | bytes | ρ vs D | ΔNLL vs FP | ΔNLL vs D [90%] | tail | "
                   "Δtail vs D [90%] | label | score | Δscore vs FP | lossless | worst-5% share | "
                   "max ΔNLL | key bits | evicted | kept width | needle kept |",
               "|---|---|---:|---:|---|---|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for k, p in sorted(r["points"].items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
            z = p.get("vs_D")
            Lh.append(
                f"| {k} | {p['lens']} | {p['bytes']:.1f} | {fmt(z and z['rho'], 2)} | "
                f"{ci(p['dnll'])} | {ci(z['nll']) if z else '(D)'} | {p['tail']:.3f} | "
                f"{ci(z['tail']) if z else '—'} | {z['label'] if z else '—'} | {p['score']:.3f} | "
                f"{ci(p['dscore_fp'])} | {'yes' if p['lossless'] else 'no'} | "
                f"{fmt(p['worst5_share'], 2)} | {p['max_dnll']:.2f} | {p['key_bits']:.2f} | "
                f"{p['evict_frac']:.1%} | {p['kept_width']:.2f} | {fmt(p['needle_keep'], 3)} |")
        Lh += ["", "**Q1 (vah kept-key width, V16 / V4 / V2)**: " + "; ".join(
            f"ρ {k}: " + " / ".join(fmt(z['kept_width'][x], 2) for x in ("V16", "V4", "V2"))
            + (" (lowers)" if z["lowers"] else " (not monotone)") for k, z in r["q1"].items())]
        if r.get("q1_fixed_vs_proxy"):
            Lh.append("Fixed kept width vs the proxy's choice (ΔNLL, same budget line): " + ", ".join(
                f"{k} {ci(v)}" for k, v in r["q1_fixed_vs_proxy"].items()))
        for B, z in r["q2"].items():
            s = f"**Q2 B={B}**: seq − pool {ci(z['seq_minus_pool'])} → **{z['label']}**"
            if "union_minus_pool" in z:
                s += f"; union − pool {ci(z['union_minus_pool'])} → {z['union_label']}"
            if "oracle_minus_pool" in z:
                s += (f"; oracle − pool {ci(z['oracle_minus_pool'])}; gap closed by seq "
                      f"{fmt(z['gap_closed'][0], 2)} [{fmt(z['gap_closed'][1], 2)}, "
                      f"{fmt(z['gap_closed'][2], 2)}]")
            Lh.append(s)
        for lens, z in r["q3"].items():
            Lh.append(f"**Q3 {lens}**: {z['label']}; selection cost vs D by r: " + ", ".join(
                f"r={k}: {ci(v)}" for k, v in z["cost"].items()))
        if r.get("fp_value_cost"):
            Lh.append("Value quantizer alone (fp twins, ΔNLL vs FP): " + ", ".join(
                f"{k} {ci(v)}" for k, v in r["fp_value_cost"].items()))
        if r["cell"] in cals:
            Lh.append("Calibration (prompts 0-9): " + "; ".join(
                f"B={B}: {z['fail']}/{z['prompt_tasks']} R0 failures, {z['rescued']} rescued, "
                f"dense heads R0 {z['dense_R0']} → seq {z['dense_seq']} (union {z['dense_union']}), "
                f"{z['replays']} replays, {z['t_search_h']:.2f} h"
                for B, z in cals[r["cell"]].items()))
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def read_main(cells, cal_jobs, out_stem, root=RESULTS):
    results, problems, cals = [], [], {}
    for name, (tag, jobs) in cells.items():
        parts = [load_run(tag, j, root=root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{name}: blocks ran different plans")
        agree = validate(d, sides, problems, main=True)
        budgets = [float(B) for B in sides[0]["preset"]["seq"]]
        if name in cal_jobs:
            cal = load_cal(cal_jobs[name][0], cal_jobs[name][1], root)
            validate_cal(cal, sides, budgets, problems)
            cals[name] = cal_summary(cal, budgets)
        else:
            problems.append(f"{name}: no calibration job given")
        if problems:
            continue
        r = analyse_cell(name, d, sides)
        r["fp_replay_agreement"] = agree
        results.append(r)
    if problems:
        raise SystemExit("INVALID Stage 1c data:\n  " + "\n  ".join(problems))
    dec = decide(results)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(nll_margin=NLL_MARGIN, tail_margin=TAIL_MARGIN,
                                  tail_nats=TAIL_NATS, worst_frac=WORST_FRAC, win=BM.WIN,
                                  tie=BM.TIE, boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED),
                       **dec, calibration=cals, cells=results), fh, indent=1, default=str)
    Lh = report(results, dec, cals, out_stem)
    print("\n".join(Lh[:4]))
    return 0


# --------------------------------------------------------------------- gate
def gate(job, root=RESULTS):
    d, side = load_run("pilot128", job, root=root)
    problems = []
    agree = validate(d, [side], problems, main=False)
    pk = float(d.peak_gib.max()) if "peak_gib" in d else float("nan")
    if not pk <= PEAK_GIB_MAX:
        problems.append(f"peak GPU memory {pk} GiB > {PEAK_GIB_MAX}")
    # the calibration's mechanics (forced search on one prompt-task)
    try:
        _, cside, search, slog, routes = load_cal("pilot128", job, root)
    except SystemExit as e:
        problems.append(f"pilot calibration missing: {e}")
        search = None
    if search is not None:
        if not (len(search) and search.searched.all() and (search.iters == 1).all()):
            problems.append("the forced search did not run once per prompt-task and budget")
        for B, grp in search.groupby("B"):
            k = bk(B)
            crit, orc = set(), set()
            for s in grp.itertuples():
                nh = int(((slog.prompt_idx == s.prompt_idx) & (slog.task == s.task)
                          & (slog.B.astype(float) == float(s.B)) & (slog.level == "head")).sum())
                if nh != s.n_cand:
                    problems.append(f"B={k} p{s.prompt_idx} {s.task}: the forced search tested "
                                    f"{nh} of {s.n_cand} heads")
                c = {tuple(h) for h in json.loads(s.critical)}
                if len(c) != 1:
                    problems.append(f"B={k} p{s.prompt_idx} {s.task}: the forced search added "
                                    f"{len(c)} heads, not 1")
                crit |= c
                orc |= {tuple(h) for h in json.loads(s.oracle_dense)}
            if routes["routes_seq"][k] != L.apply_critical(routes["routes_pool"][k], sorted(crit)):
                problems.append(f"B={k}: routes_seq is not R0 plus the forced heads (union)")
            if routes["routes_union"][k] != L.apply_critical(routes["routes_pool"][k], sorted(orc)):
                problems.append(f"B={k}: routes_union is not R0 plus the oracle's dense heads (union)")
    n_main = len(L.build_plan(L.PRESETS["main128"]))
    dec = float(d[d.arm != "fp"].t_arm.median())
    tf = float(d.t_tf.median())
    per = (float(d.t_prefill.median()) + 2 * float(d.t_precompute.max())
           + n_main * (3 * dec + 2 * tf))
    proj_h = per * MAIN_BLOCK_PROMPT_TASKS / 3600
    if proj_h > 0.9 * WALL_H["main128"]:
        problems.append(f"projected main128 block {proj_h:.1f} h > 90% of {WALL_H['main128']} h")
    print(f"R14 Stage 1c pilot {job}: {len(d)} rows, plan {len(plan_of(side))} arms, "
          f"fp replay agreement {agree:.3f}, peak {pk:.1f} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected main128 block {proj_h:.1f} h "
          f"({n_main} arms x {MAIN_BLOCK_PROMPT_TASKS} prompt-tasks)")
    cols = [c for c in ("score", "tf_c_sum_nll", "bits_per_token", "evict_frac", "kept_width",
                        "needle_keep", "t_arm", "t_tf") if c in d]
    print(d.pivot_table(index=["arm", "B"], values=cols, aggfunc="mean").round(3).to_string())
    if search is not None:
        print(search[["B", "fp_min", "base_min", "final_min", "n_cand", "n_replays",
                      "t_search"]].round(3).to_string())
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--main128", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--main32", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--cal128", metavar="JOB")
    ap.add_argument("--cal32", metavar="JOB")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1c"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot and (a.main128 or a.main32):
        ap.error("--pilot or the main blocks, not both")
    if a.pilot:
        sys.exit(gate(a.pilot, a.results_root))
    cells, cal = {}, {}
    if a.main128:
        cells["llama31-8b@131072"] = ("main128", a.main128)
        if a.cal128:
            cal["llama31-8b@131072"] = ("cal128", a.cal128)
    if a.main32:
        cells["llama31-8b@32768"] = ("main32", a.main32)
        if a.cal32:
            cal["llama31-8b@32768"] = ("cal32", a.cal32)
    if not cells:
        ap.error("give --pilot JOB or at least one of --main128 / --main32")
    sys.exit(read_main(cells, cal, a.out_stem, a.results_root))


if __name__ == "__main__":
    main()
