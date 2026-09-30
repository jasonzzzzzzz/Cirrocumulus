#!/usr/bin/env python3
"""R14 Stage 1d gate and reader. The rules below are FROZEN: written 2026-09-30,
before any Stage 1d output existed (design: s1d_lib.py; driver: run_s1d.py).

    python read_stage1d.py --pilot JOB [--pilot-tag pilot128|qwenpilot]
    python read_stage1d.py --main128 J1 J2 J3 J4 --main32 J5 J6 J7 J8 --cal128 C1 --cal32 C2
    python read_stage1d.py --qwen32 J1 J2 J3 J4 --calq32 C --calq8 C8 --out-stem stage1d_qwen

Result directories: h0_measurement/results/r14s1d_<tag>_<job>/.

CELLS
  - Llama-3.1-8B @131072, prompts 7000-7039, and @32768, prompts 7100-7139.
  - Qwen3-30B-A3B-2507 @32768, prompts 7100-7139: the same indices as Llama's
    32K cell, so the same needles and haystack sources.
  All in blocks of 10. Calibration on prompts 0-9; Qwen is also calibrated at
  8K, with no evaluation there. Every arm of a prompt-task runs in one process.
METRIC (the fix)
  dA(X) = a_sum_nll(X) - a_sum_nll(fp). a_sum_nll is the teacher-forced NLL of
  the answer-VALUE tokens of FP's greedy answer: the tokens that spell an
  expected answer string, up to the end of the answer span.
  - A prompt-task whose FP answer contains no expected value drops from the NLL
    statistics; the count is reported.
  - Task cells with FP score < 0.9 drop.
  - Stage 1c's all-content metric (dC, from tf_c_sum_nll) is reported beside it.
COMPARATOR D_L = uniform@3 in X's value lens (V16 exact values, V4 4-bit values).
INTERVALS Prompt bootstrap: prompts resampled within block, all tasks of a prompt
  together, 10,000 draws, seed 14; 90% percentile intervals [lo, hi].
MATCHED (X vs D_L, paired within prompt-tasks; the Stage 1c rule on dA):
  - NLL: hi of mean(dA(X) - dA(D_L)) <= 0.10 nats;
  - TAIL: hi of (tail(X) - tail(D_L)) <= 0.05, where tail is the share of
    prompt-tasks with dA > 2 nats.
  MATCHED iff both hold. WORSE iff the NLL lo > 0.10 or the tail lo > 0.05.
  INCONCLUSIVE otherwise.
BYTES As Stage 1c: read per decode step, per context token, per KV head, per layer.
  - Router families (seq, seq2, topn, mech) use the router side rule.
  - hvah and the reads use the keep-bitmap rule.
  - Re-selecting reads add the scan of the stored keys every RESEL_K steps:
    (d/8)(3 + 16/d)/k.
  rho = bytes(X) / bytes(D_L).
VERDICT Per deployable family (designs seq2, topn, hvah, qread; references
  sieve, pool, seq, union), cell and lens, take the cheapest MATCHED point:
  WIN (rho <= 0.80), TIE (<= 1.00), LOSS, or NO_POINT. The oracle and the
  mechanism arms are diagnostics.
DECISION (Llama cells only; Qwen is a replication)
  - GO_KERNEL if any family is WIN in V4 in at least one cell; the output names
    each (family, cell).
  - Else SCOPE_EXACT_V if any family is WIN in V16.
  - Else STOP_SYSTEMS.
EFFECT LABELS A paired difference is NAME_HELPS if mean <= -0.05 nats and
  hi < 0; NAME_HURTS if mean >= +0.05 and lo > 0; NAME_NO_EFFECT otherwise.
QUESTIONS (reported; labels fixed)
  QM metric. Every point's label under dA and under dC. The count that change.
     The share of D's dC that is post-span continuation.
  QH hvah vs router_seq2_calib (same mask; the router's own widths) per (B,
     lens), as HVAH_* effect labels (negative = hvah better). Also whether hvah
     is MATCHED at rho <= 0.80 in V4.
  QN head budget at B_low: seq2 -> top32 -> top64 -> union, each with dA vs D
     and rho; the smallest MATCHED one (or NONE).
  QR reads at r = 0.125, V4: qreadp, qreadr and qreadpr each vs qread, as
     PROT_*, RESEL_*, BOTH_* effect labels. Also the smallest MATCHED r per
     variant.
  QX mechanism, per cell with mech arms at B:
     - Rescue fractions f_X = mean(dA(pool) - dA(X)) / mean(dA(pool) - dA(seq2))
       for X = mech_q, mech_a, mech_p, with bootstrap intervals. They are
       defined only if the full rescue mean(dA(pool) - dA(seq2)) >= 0.25 nats
       with lo > 0; otherwise NO_RESCUE_TO_EXPLAIN.
     - Timing label:
       - SETUP: f_q >= 0.5 and f_a <= 0.25;
       - ANSWER_TIME: f_a >= 0.5 and f_q <= 0.25;
       - BOTH_PHASES: f_q >= 0.5 and f_a >= 0.5;
       - SPLIT otherwise.
     - Patch label: PATCH_CONFIRMS if f_p >= 0.5, PATCH_PARTIAL if
       f_p >= 0.25, PATCH_NO otherwise.
     - Anatomy (question pass): the critical heads' mean mass on the key term
       against the other heads of the same layers. KEY_READERS if the ratio of
       means >= 2 with a prompt-task bootstrap lo > 1; else NOT_KEY_READERS.
     - Answer step: LOW_ANSWER_ATTENTION if the critical heads' median rank by
       answer-value mass is in the lower half of all KV heads.
  QQ Qwen replication (Qwen cell). The mech arms run at B_low and at 3: at 2.5
     the rescue candidate is TurboQuant-2, which Qwen's outlier channels may
     break.
     - R1, per calibration budget: SMALL_SET if seq2 adds 1-10% of the KV
       heads; NO_CRITICAL_HEADS if it adds none; LARGE_SET otherwise.
     - R2 SEQ_CLOSES: gap closed = mean(pool - seq2) / mean(pool - oracle)
       >= 0.5 with lo > 0 at B_low.
     - R3 and R4, per mech budget: QX's answer-step, timing and patch labels.
     - R5 LENGTH_STABLE: >= 50% of seq2's critical heads at 32K (any budget)
       are critical at 8K too.
VALIDITY (any failure -> INVALID, no verdict)
  V1 Every (prompt, task) block holds exactly the sidecar plan's rows, once, for
     every planned prompt and task. One corpus and one plan per cell. FP >= 0.9
     on >= 3 of 4 tasks (Llama) or >= 2 of 4 (Qwen).
  V2 Audits:
     - dense = w bits with nothing evicted;
     - routers spend <= B;
     - hvah keeps exactly seq2@B's tokens at its width;
     - reads store 3 bits dense and read >= floor(r C) per head (exactly
       floor(r C) without protection);
     - twins match their base;
     - mech rows carry pool@B's bits and eviction.
  V3 fp's teacher-forced replay agrees with fp's greedy on >= 98% of steps.
  V4 Every twin changed its values.
  V5 Calibration:
     - seq2 and union routes are R0 with heads switched to dense;
     - the base file is unchanged;
     - no forced search;
     - every block used this calibration's file.
  V6 Metric: FP's answer-value mask is non-empty for >= 90% of the prompt-tasks
     FP answers correctly.
  V7 Re-selection: every re-selecting read with more than RESEL_K generated
     tokens re-selected at least once.
REPORTED, NOT GATED
  - task scores and the Stage 0 lossless rule;
  - the worst-5% share, CVaR and max dA;
  - stored and answer-amortised bytes;
  - needle_keep; per-task dA;
  - the Stage 1c metric throughout.
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
import s1d_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"main128": 10.0, "main32": 4.0, "qwen32": 8.0}
MAIN_BLOCK_PROMPT_TASKS = 40
V3_MIN, V6_MIN = 0.98, 0.90
NLL_MARGIN, TAIL_MARGIN, TAIL_NATS, WORST_FRAC = 0.10, 0.05, 2.0, 0.05
RESCUE_MIN, SETUP_HI, SETUP_LO, PATCH_HI, PATCH_LO = 0.25, 0.5, 0.25, 0.5, 0.25
KEY_RATIO, SMALL_SET, GAP_MIN, LENGTH_MIN = 2.0, 0.10, 0.5, 0.5
FP_MIN = BM.RMT.FP_MIN
DEPLOYABLE = L.DESIGNS + L.REFERENCES
DREF = {"V16": "uniform", "V4": "uniform+v4", "V2": "uniform+v2"}
bk, fmt, ci, sha256 = R1C.bk, R1C.fmt, R1C.ci, R1C.sha256


# ------------------------------------------------------------------ loading
def load_run(tag, job, mode="evaluate", root=RESULTS):
    d = os.path.join(root, f"r14s1d_{tag}_{job}")
    pq = [x for x in glob.glob(os.path.join(d, f"s1d_{mode}_*.parquet"))
          if not x.endswith(("_heads.parquet", "_search.parquet", "_searchlog.parquet",
                             "_anatomy.parquet"))]
    js = glob.glob(os.path.join(d, f"s1d_{mode}_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one s1d_{mode} parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def load_anatomy(side):
    f = os.path.join(side["_dir"], side["anatomy"])
    return pd.read_parquet(f).assign(job=side["_job"])


def load_cal(tag, job, root=RESULTS):
    rows, side = load_run(tag, job, "calibrate", root)
    d = side["_dir"]
    return (rows, side, pd.read_parquet(os.path.join(d, side["search"])),
            pd.read_parquet(os.path.join(d, side["searchlog"])), json.load(open(side["write_routes"])))


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


# ----------------------------------------------------------------- validity
def validate(d, sides, problems, main=True, fp_tasks_min=3):
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(["prompt_idx", "task"]).apply(
            lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))), include_groups=False)
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: (prompt, task) with the wrong arms: {bad[:3]}")
        if len(got) != side["n_prompts"] * len(side["tasks"]):
            problems.append(f"block {side['_job']}: {len(got)} prompt-tasks, expected "
                            f"{side['n_prompts'] * len(side['tasks'])}")
    if d.duplicated(["job", "prompt_idx", "task", "arm", "B"]).any():
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
    key = ["job", "prompt_idx", "task"]
    hv = d[fam == "hvah"]
    for arm in sorted(hv.arm.unique()):
        v_ = L.parse_arm(arm)["v_bits"]
        src = "router_seq2_calib" + ("" if v_ == 16 else f"+v{v_}")
        m = d[d.arm == arm].merge(d[d.arm == "router_seq2_calib"], on=key + ["B"], suffixes=("", "_s"))
        w = L.HVAH_WIDTH[v_]
        if len(m) != (d.arm == arm).sum() or not (
                np.allclose(m.evict_frac, m.evict_frac_s, atol=1e-9)
                and np.allclose(m.bits_per_token, w * (1 - m.evict_frac), atol=1e-6)):
            problems.append(f"{arm}: not router_seq2_calib's mask at {w} bits ({src})")
    qr = d[fam == "qread"]
    if len(qr):
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(qr.B, qr.ctx_len)])
        prot = qr.arm.map(lambda a: L.parse_arm(a)["protect"]).to_numpy()
        ok = (np.allclose(qr.stored_bits_per_token, L.STORE_WIDTH) and (qr.stored_evict_frac == 0).all()
              and np.all(qr.read_frac.to_numpy() >= k - 1e-9)
              and np.allclose(qr.read_frac.to_numpy()[~prot], k[~prot], atol=1e-9))
        if not ok:
            problems.append("a read row does not store 3 bits dense and read floor(r C) per head")
        rs = qr[qr.arm.map(lambda a: L.parse_arm(a)["resel"]) & (qr.gen_len > L.RESEL_K)]
        if len(rs) and (rs.n_resel <= 0).any():
            problems.append("a re-selecting read with a long answer never re-selected (V7)")
    mc = d[fam == "mech"]
    if len(mc):
        m = mc.merge(d[d.arm == "router_pool_calib"], on=key + ["B"], suffixes=("", "_p"))
        if len(m) != len(mc) or not (np.allclose(m.bits_per_token, m.bits_per_token_p)
                                     and np.allclose(m.evict_frac, m.evict_frac_p)):
            problems.append("a mech row does not carry router_pool_calib's allocation")
    for arm in sorted(d[d.twin != ""].arm.unique()):
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
    ok_fp = fp[fp.score >= 1]
    cover = float((ok_fp.a_len > 0).mean()) if len(ok_fp) else 0.0
    if cover < V6_MIN:
        problems.append(f"FP's answer-value mask found in {cover:.2f} < {V6_MIN} of correct FP answers")
    if main:
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        if (fps >= FP_MIN).sum() < min(fp_tasks_min, len(fps)):
            problems.append(f"FP below {FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree, cover


def validate_cal(cal, sides, budgets, problems):
    _, cside, _, _, routes = cal
    meta, base = routes["meta"], routes["meta"]["base_routes"]
    try:
        if sha256(base["path"]) != base["sha256"]:
            problems.append(f"{base['path']} changed since the calibration read it")
    except OSError as e:
        problems.append(f"calibration base routes unreadable: {e}")
        return
    j1b = json.load(open(base["path"]))
    for B in budgets:
        k = bk(B)
        if routes["routes_pool"].get(k) != j1b["routes_pool"].get(k):
            problems.append(f"calibration routes_pool@{k} is not the base file's")
        for field in ("routes_seq2", "routes_union"):
            if k not in routes.get(field, {}) or not L1C.only_densified(routes["routes_pool"][k],
                                                                       routes[field][k]):
                problems.append(f"calibration {field}@{k} changes more than dense switches")
    if meta.get("rule", {}).get("forced"):
        problems.append("the calibration ran with --force-search (mechanics only)")
    path = os.path.abspath(cside["write_routes"])
    csha = sha256(path)
    for s in sides:
        for k_, v in s["routes"].items():
            if k_.startswith(("router_seq2_calib", "router_union_calib", "router_top")) and \
                    (os.path.abspath(v["path"]) != path or v["sha256"] != csha):
                problems.append(f"block {s['_job']}: {k_} read {v['path']}, not the calibration's "
                                f"{path} (or the file changed)")


# ---------------------------------------------------------------- analysis
def add_bytes(v):
    """Stage 1c's byte rule (read per step; stored; answer-amortised) with
    Stage 1d's families, plus the re-selection scan."""
    v = v.copy()
    d8 = v.head_dim / 8.0
    vs = np.where(v.v_bits >= 16, 0.0, 16.0 / v.head_dim)
    tail = (v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len
    scan = v.scan_bytes.fillna(0.0) if "scan_bytes" in v else 0.0
    v["bytes"] = (d8 * (v.bits_per_token + v.key_side) + (1 - v.evict_frac) * d8 * (v.v_bits + vs)
                  + tail + scan)
    fam = v.arm.map(L.family)
    sk = np.where(fam == "qread", 16.0 / v.head_dim, v.key_side)
    v["bytes_stored"] = (d8 * (v.stored_bits_per_token + sk)
                         + (1 - v.stored_evict_frac) * d8 * (v.v_bits + vs) + tail)
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


def analyse_cell(name, d, sides, anat, cal):
    fp_task = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_task.items() if s >= FP_MIN)
    v = add_bytes(d[d.task.isin(valid)])
    key = ["job", "prompt_idx", "task"]
    piv = lambda col: v.pivot_table(index=key, columns=["arm", "B"], values=col, aggfunc="first")  # noqa: E731
    A, Cm, S = piv("a_sum_nll"), piv("tf_c_sum_nll"), piv("score")
    drop = A[("fp", 0.0)].isna()
    n_drop = int(drop.sum())
    A, Cm, S = A[~drop], Cm.loc[A.index], S.loc[A.index]
    dA, dC = A.sub(A[("fp", 0.0)], axis=0), Cm.sub(Cm[("fp", 0.0)], axis=0)
    pidx = dA.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s, pidx).to_numpy()  # noqa: E731
    spm = piv("score").groupby(level=["job", "prompt_idx"]).mean()
    ll = BM.lossless_table(spm, BM.boot_weights(spm.index))
    means = v.groupby(["arm", "B"])[["bytes", "bytes_stored", "bytes_amort", "bits_per_token",
                                     "evict_frac", "kept_width", "needle_keep"]].mean()

    def pair(a, b, tbl=dA):
        return L1C.boot_ci(pm(tbl[a] - tbl[b]), W)

    def tail_pair(a, b, tbl=dA):
        return L1C.boot_ci(pm((tbl[a] > TAIL_NATS).astype(float) - (tbl[b] > TAIL_NATS).astype(float)), W)

    pts = {}
    for col in dA.columns:
        arm, B = col
        if arm == "fp":
            continue
        pa = L.parse_arm(arm)
        Dc = (DREF[pa["lens"]], 3.0)
        x = dA[col].to_numpy()
        p = dict(arm=arm, B=float(B), lens=pa["lens"], family=pa["family"],
                 bytes=float(means.loc[col, "bytes"]), bytes_stored=float(means.loc[col, "bytes_stored"]),
                 bytes_amort=float(means.loc[col, "bytes_amort"]),
                 key_bits=float(means.loc[col, "bits_per_token"]),
                 evict_frac=float(means.loc[col, "evict_frac"]),
                 kept_width=float(means.loc[col, "kept_width"]),
                 needle_keep=float(means.loc[col, "needle_keep"]),
                 dA=L1C.boot_ci(pm(dA[col]), W), dC=L1C.boot_ci(pm(dC[col]), W),
                 tail=L1C.tail_share(x, TAIL_NATS), worst5_share=L1C.worst_share(x, WORST_FRAC),
                 cvar5=L1C.cvar(x, WORST_FRAC), max_dA=float(np.max(x)),
                 dA_by_task={t: float(g.mean()) for t, g in dA[col].groupby(level="task")},
                 score=ll[col]["score"], dscore_fp=(ll[col]["delta"], ll[col]["lo"], ll[col]["hi"]),
                 lossless=ll[col]["lossless"])
        if Dc in dA.columns and col != Dc:
            z = dict(D=f"{Dc[0]}@3", nll=pair(col, Dc), tail=tail_pair(col, Dc),
                     nll_c=pair(col, Dc, dC), tail_c=tail_pair(col, Dc, dC),
                     rho=p["bytes"] / float(means.loc[Dc, "bytes"]),
                     rho_mem=p["bytes_stored"] / float(means.loc[Dc, "bytes_stored"]))
            z["label"] = R1C.matched_label(z["nll"], z["tail"])
            z["label_c"] = R1C.matched_label(z["nll_c"], z["tail_c"])
            p["vs_D"] = z
        pts[f"{arm}@{bk(B)}"] = p
    res = dict(cell=name, fp=float(fp_task.mean()), fp_by_task=fp_task.round(4).to_dict(),
               valid_tasks=valid, n_prompts=int(len(pidx)), n_prompt_tasks=int(len(dA)),
               nll_dropped=n_drop, points=pts, families={})
    for lens in ("V16", "V4"):
        res["families"][lens] = {}
        for fam in DEPLOYABLE:
            c = [(k, p) for k, p in pts.items() if p["lens"] == lens and p["family"] == fam
                 and p.get("vs_D", {}).get("label") == "MATCHED"]
            kb, pb = min(c, key=lambda kp: kp[1]["bytes"]) if c else (None, None)
            r = pb["vs_D"]["rho"] if pb else None
            res["families"][lens][fam] = dict(point=kb, rho=r, verdict=R1C.rho_verdict(r))
    pr = sides[0]["preset"]
    Bl = float(pr["B_low"])
    # QM: the metric fix
    changed = [k for k, p in pts.items() if "vs_D" in p and p["vs_D"]["label"] != p["vs_D"]["label_c"]]
    Dk = ("uniform", 3.0)
    post = float((v[v.arm == "uniform"].query("B == 3").post_sum_nll.mean()
                  - v[v.arm == "fp"].post_sum_nll.mean()) / max(float(dC[Dk].mean()), 1e-9)) \
        if "post_sum_nll" in v else float("nan")
    res["qm"] = dict(changed=changed, n_changed=len(changed), post_share_of_D_dC=post)
    # QH: hvah vs the router whose mask it copies
    res["qh"] = {}
    for B, vb in pr["hvah"]:
        a_ = (f"hvah_v{int(vb)}", float(B))
        b_ = ("router_seq2_calib" + ("" if int(vb) == 16 else f"+v{int(vb)}"), float(B))
        if a_ in dA.columns and b_ in dA.columns:
            z = pair(a_, b_)
            res["qh"][f"{a_[0]}@{bk(B)}"] = dict(diff=z, label=L.effect_label(*z, "HVAH"))
    # QN: the head budget at B_low
    ladder = [("router_seq2_calib", Bl)] + [(f"router_top{n}_calib", Bl) for n in pr["topn"]] + \
             [("router_union_calib", Bl)]
    ladder = [c for c in ladder if c in dA.columns]
    res["qn"] = dict(points=[f"{a}@{bk(B)}" for a, B in ladder],
                     smallest_matched=next((f"{a}@{bk(B)}" for a, B in ladder
                                            if pts[f"{a}@{bk(B)}"].get("vs_D", {}).get("label")
                                            == "MATCHED"), "NONE"))
    # QR: reads
    res["qr"] = {}
    base = ("qread_v4", 0.125)
    for var, name in (("qreadp_v4", "PROT"), ("qreadr_v4", "RESEL"), ("qreadpr_v4", "BOTH")):
        c = (var, 0.125)
        if base in dA.columns and c in dA.columns:
            z = pair(c, base)
            res["qr"][var] = dict(diff=z, label=L.effect_label(*z, name))
    for var in ("qread_v4", "qreadp_v4", "qreadr_v4", "qreadpr_v4", "qread_v16"):
        rs = sorted(p["B"] for p in pts.values() if p["arm"] == var
                    and p.get("vs_D", {}).get("label") == "MATCHED")
        res["qr"].setdefault("smallest_matched", {})[var] = bk(rs[0]) if rs else "NONE"
    # QX: mechanism
    res["qx"] = {}
    for B in pr["mech"]:
        B = float(B)
        cols = {k: (a, B) for k, a in (("pool", "router_pool_calib"), ("seq2", "router_seq2_calib"),
                                       ("q", "mech_q"), ("a", "mech_a"), ("p", "mech_p"))}
        if not all(c in dA.columns for c in cols.values()):
            res["qx"][bk(B)] = dict(label="MISSING_ARMS")
            continue
        vals = {k: pm(dA[c]) for k, c in cols.items()}
        full = L1C.boot_ci(vals["pool"] - vals["seq2"], W)
        z = dict(full_rescue=full)
        if not (full[0] >= RESCUE_MIN and full[1] > 0):
            z["label"] = "NO_RESCUE_TO_EXPLAIN"
        else:
            bs = lambda x: (W @ x) / W.sum(1)  # noqa: E731
            den = bs(vals["pool"] - vals["seq2"])
            for k in ("q", "a", "p"):
                f = L.rescue_fraction(vals["pool"], vals["seq2"], vals[k])
                fb = bs(vals["pool"] - vals[k]) / np.where(den > 0, den, np.nan)
                lo, hi = np.nanpercentile(fb, [5, 95])
                z[f"f_{k}"] = (f, float(lo), float(hi))
            fq, fa, fp_ = z["f_q"][0], z["f_a"][0], z["f_p"][0]
            z["label"] = ("SETUP" if fq >= SETUP_HI and fa <= SETUP_LO else
                          "ANSWER_TIME" if fa >= SETUP_HI and fq <= SETUP_LO else
                          "BOTH_PHASES" if fq >= SETUP_HI and fa >= SETUP_HI else "SPLIT")
            z["patch"] = ("PATCH_CONFIRMS" if fp_ >= PATCH_HI else
                          "PATCH_PARTIAL" if fp_ >= PATCH_LO else "PATCH_NO")
        z.update(anatomy_summary(anat, sides[0]["critical"].get(bk(B), []), sides[0]))
        res["qx"][bk(B)] = z
    return res


def anatomy_summary(anat, crit, side):
    """Critical heads vs the other heads of the same layers: question-pass mass
    on the key term; answer-step mass on the answer values and its rank."""
    if anat is None or not len(anat) or not crit:
        return dict(anatomy="none")
    crit = {(int(a), int(b)) for a, b in crit}
    layers = {li for li, _ in crit}
    an = anat.copy()
    an["crit"] = [(li, g) in crit for li, g in zip(an.layer, an.kv_head)]
    q = an[(an.phase == "question") & an.layer.isin(layers)]
    per = q.groupby(["job", "prompt_idx", "task", "crit"]).m_key.mean().unstack("crit").dropna()
    out = {}
    if True in per and False in per:
        ratio = float(per[True].mean() / max(per[False].mean(), 1e-12))
        rng = np.random.default_rng(BM.BOOT_SEED)
        idx = rng.integers(0, len(per), size=(2000, len(per)))
        rb = per[True].to_numpy()[idx].mean(1) / np.maximum(per[False].to_numpy()[idx].mean(1), 1e-12)
        lo, hi = np.percentile(rb, [5, 95])
        out["key_ratio"] = (ratio, float(lo), float(hi))
        out["key_label"] = "KEY_READERS" if ratio >= KEY_RATIO and lo > 1 else "NOT_KEY_READERS"
        for c_ in L.ANAT_CATS:
            col = f"m_{c_}"
            g = q.groupby("crit")[col].mean()
            out[f"q_{c_}"] = (float(g.get(True, np.nan)), float(g.get(False, np.nan)))
    a = an[an.phase == "answer_step"].groupby(["layer", "kv_head"]).m_value.mean()
    if len(a):
        rank = a.rank(ascending=False)
        med = float(np.median([rank[h] for h in crit if h in rank.index]))
        out["answer_value_rank_median"] = med
        out["n_kv_heads"] = int(len(a))
        out["answer_label"] = ("LOW_ANSWER_ATTENTION" if med > len(a) / 2 else "HIGH_ANSWER_ATTENTION")
    return out


def qwen_replication(r, cal32, cal8, pr):
    """R1 per calibration budget; R3 / R4 per mech budget (Qwen runs mech at both)."""
    rows, side, search, slog, routes = cal32
    nkv = side["n_layers"] * side["n_kv_heads"]
    out = {"R1": {}}
    for k in routes["routes_seq2"]:
        added = (set(L1C.dense_heads(routes["routes_seq2"][k]))
                 - set(L1C.dense_heads(routes["routes_pool"][k])))
        out["R1"][k] = dict(added=len(added), share=len(added) / nkv,
                            label=("SMALL_SET" if 0 < len(added) / nkv <= SMALL_SET else
                                   "NO_CRITICAL_HEADS" if not added else "LARGE_SET"))
    out["R3"] = {B: z.get("answer_label", "none") for B, z in r["qx"].items()}
    out["R4"] = {B: (z.get("label"), z.get("patch")) for B, z in r["qx"].items()}
    return out


def decide(results):
    wins = {lens: [(fam, r["cell"]) for r in results for fam, z in r["families"][lens].items()
                   if z["verdict"] == "WIN"] for lens in ("V16", "V4")}
    dec = "GO_KERNEL" if wins["V4"] else ("SCOPE_EXACT_V" if wins["V16"] else "STOP_SYSTEMS")
    return dict(decision=dec, wins=wins)


def gap_closed(d, sides, B):
    """mean(pool - seq2) / mean(pool - oracle) at B, with a prompt bootstrap."""
    key = ["job", "prompt_idx", "task"]
    A = d.pivot_table(index=key, columns=["arm", "B"], values="a_sum_nll", aggfunc="first")
    need = [("router_pool_calib", B), ("router_seq2_calib", B), ("router_pool_oracle", B), ("fp", 0.0)]
    if not all(c in A.columns for c in need):
        return None
    A = A[~A[("fp", 0.0)].isna()]
    pidx = A.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s, pidx).to_numpy()  # noqa: E731
    num, den = pm(A[need[0]] - A[need[1]]), pm(A[need[0]] - A[need[2]])
    bn, bd = (W @ num) / W.sum(1), (W @ den) / W.sum(1)
    ok = bd > 0
    lo, hi = np.percentile(bn[ok] / bd[ok], [5, 95]) if ok.any() else (np.nan, np.nan)
    return (float(num.mean() / den.mean()) if den.mean() > 0 else float("nan"), float(lo), float(hi))


# ------------------------------------------------------------------ report
def report(results, dec, cals, qwen, out_stem):
    Lh = ["# R14 Stage 1d (read_stage1d.py; rules frozen in its docstring)", ""]
    if dec:
        Lh.append(f"**Decision: {dec['decision']}** · V4 WINs: "
                  + (", ".join(f"{f} @ {c}" for f, c in dec["wins"]["V4"]) or "none")
                  + " · V16 WINs: " + (", ".join(f"{f} @ {c}" for f, c in dec["wins"]["V16"]) or "none"))
        Lh.append("")
    for r in results:
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f} ({r['fp_by_task']}), {r['n_prompts']} prompts, "
               f"{r['n_prompt_tasks']} prompt-tasks with an answer value (dropped {r['nll_dropped']}), "
               f"fp replay {r['fp_replay_agreement']:.3f}, answer-value coverage {r['value_cover']:.2f}", "",
               "| lens | " + " | ".join(DEPLOYABLE) + " |", "|---|" + "---|" * len(DEPLOYABLE)]
        for lens, fams in r["families"].items():
            Lh.append(f"| {lens} | " + " | ".join(
                z["verdict"] + (f" {z['point']} ρ {z['rho']:.2f}" if z["point"] else "")
                for z in (fams[f] for f in DEPLOYABLE)) + " |")
        Lh += ["", "| arm@B | lens | bytes | ρ | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label "
                   "| label (1c metric) | score | lossless | kept width | evicted | needle kept |",
               "|---|---|---:|---:|---|---|---:|---|---|---|---:|---|---:|---:|---:|"]
        for k, p in sorted(r["points"].items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
            z = p.get("vs_D")
            Lh.append(f"| {k} | {p['lens']} | {p['bytes']:.1f} | {fmt(z and z['rho'], 2)} | {ci(p['dA'])} | "
                      f"{ci(z['nll']) if z else '(D)'} | {p['tail']:.3f} | {ci(z['tail']) if z else '—'} | "
                      f"{z['label'] if z else '—'} | {z['label_c'] if z else '—'} | {p['score']:.3f} | "
                      f"{'yes' if p['lossless'] else 'no'} | {p['kept_width']:.2f} | "
                      f"{p['evict_frac']:.1%} | {fmt(p['needle_keep'])} |")
        Lh += ["", f"**QM**: {r['qm']['n_changed']} labels change with the metric fix "
                   f"({', '.join(r['qm']['changed']) or 'none'}); post-answer continuation = "
                   f"{fmt(r['qm']['post_share_of_D_dC'], 2)} of D's Stage-1c-metric damage"]
        for k, z in r["qh"].items():
            Lh.append(f"**QH** {k}: hvah − seq2 {ci(z['diff'])} → {z['label']}")
        Lh.append(f"**QN** ladder {' → '.join(r['qn']['points'])}: smallest MATCHED "
                  f"{r['qn']['smallest_matched']}")
        for k, z in r["qr"].items():
            if k != "smallest_matched":
                Lh.append(f"**QR** {k} − qread_v4 @0.125: {ci(z['diff'])} → {z['label']}")
        if "smallest_matched" in r["qr"]:
            Lh.append("**QR** smallest MATCHED r: " + ", ".join(f"{k} {v}" for k, v in
                                                                r["qr"]["smallest_matched"].items()))
        for B, z in r["qx"].items():
            s = f"**QX** B={B}: {z.get('label')}"
            if "full_rescue" in z:
                s += f" (full rescue {ci(z['full_rescue'])}"
                for k in ("q", "a", "p"):
                    if f"f_{k}" in z:
                        s += f"; f_{k} {z[f'f_{k}'][0]:.2f} [{z[f'f_{k}'][1]:.2f}, {z[f'f_{k}'][2]:.2f}]"
                s += ")"
            if "patch" in z:
                s += f"; {z['patch']}"
            if "key_ratio" in z:
                s += (f"; key-term mass, critical / other heads {z['key_ratio'][0]:.2f} "
                      f"[{z['key_ratio'][1]:.2f}, {z['key_ratio'][2]:.2f}] → {z['key_label']}")
            if "answer_label" in z:
                s += (f"; answer-value rank median {z['answer_value_rank_median']:.0f} of "
                      f"{z['n_kv_heads']} → {z['answer_label']}")
            Lh.append(s)
        if r["cell"] in cals:
            Lh.append("Calibration: " + "; ".join(f"B={B}: {z}" for B, z in cals[r["cell"]].items()))
        Lh.append("")
    if qwen:
        Lh += ["## Qwen replication", ""] + [f"- {k}: {v}" for k, v in qwen.items()]
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def cal_summary(cal, budgets):
    rows, side, search, slog, routes = cal
    out = {}
    for B in budgets:
        k = bk(B)
        s = search[search.B.astype(float) == float(B)] if len(search) else search
        out[k] = dict(prompt_tasks=int(len(s)), fail=int(s.fail.sum()) if len(s) else 0,
                      searched=int(s.searched.sum()) if len(s) else 0,
                      dense_R0=len(L1C.dense_heads(routes["routes_pool"][k])),
                      dense_seq2=len(L1C.dense_heads(routes["routes_seq2"][k])),
                      dense_union=len(L1C.dense_heads(routes["routes_union"][k])))
    return out


def read_cells(cells, cal_jobs, out_stem, root=RESULTS, qwen_cal8=None):
    results, problems, cals, qwen = [], [], {}, {}
    for name, (tag, jobs) in cells.items():
        parts = [load_run(tag, j, root=root) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        sides = [p[1] for p in parts]
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{name}: blocks ran different plans")
        is_qwen = sides[0]["model"].startswith("qwen")
        agree, cover = validate(d, sides, problems, main=True, fp_tasks_min=2 if is_qwen else 3)
        budgets = [float(B) for B in sides[0]["preset"]["calib"]]
        cal = None
        if name in cal_jobs:
            cal = load_cal(cal_jobs[name][0], cal_jobs[name][1], root)
            validate_cal(cal, sides, budgets, problems)
            cals[name] = cal_summary(cal, budgets)
        else:
            problems.append(f"{name}: no calibration job given")
        if problems:
            continue
        anat = pd.concat([load_anatomy(s) for s in sides], ignore_index=True)
        r = analyse_cell(name, d, sides, anat, cal)
        r["fp_replay_agreement"], r["value_cover"] = agree, cover
        results.append(r)
        if is_qwen:
            pr = sides[0]["preset"]
            qwen = qwen_replication(r, cal, None, pr)
            qwen["R2"] = dict(gap_closed=gap_closed(d[d.task.isin(r["valid_tasks"])], sides,
                                                     float(pr["B_low"])))
            g = qwen["R2"]["gap_closed"]
            qwen["R2"]["label"] = ("SEQ_CLOSES" if g and g[0] >= GAP_MIN and g[1] > 0 else "SEQ_NOT_CLOSING")
            if qwen_cal8 is not None:
                c8 = load_cal(qwen_cal8[0], qwen_cal8[1], root)[4]
                c32 = cal[4]
                crit = lambda rt: {h for k in rt["routes_seq2"] for h in  # noqa: E731
                                   set(L1C.dense_heads(rt["routes_seq2"][k]))
                                   - set(L1C.dense_heads(rt["routes_pool"][k]))}
                a32, a8 = crit(c32), crit(c8)
                share = len(a32 & a8) / len(a32) if a32 else float("nan")
                qwen["R5"] = dict(n32=len(a32), n8=len(a8), overlap=sorted(a32 & a8), share=share,
                                  label="LENGTH_STABLE" if share == share and share >= LENGTH_MIN
                                  else "LENGTH_UNSTABLE")
    if problems:
        raise SystemExit("INVALID Stage 1d data:\n  " + "\n  ".join(problems))
    llama = [r for r in results if not r["cell"].startswith("qwen")]
    dec = decide(llama) if llama else None
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(nll_margin=NLL_MARGIN, tail_margin=TAIL_MARGIN, tail_nats=TAIL_NATS,
                                  win=BM.WIN, tie=BM.TIE, boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED),
                       decision=dec, calibration=cals, qwen=qwen, cells=results), fh, indent=1,
                  default=str)
    Lh = report(results, dec, cals, qwen, out_stem)
    print("\n".join(Lh[:4]))
    return 0


# --------------------------------------------------------------------- gate
def gate(job, tag="pilot128", root=RESULTS):
    d, side = load_run(tag, job, root=root)
    problems = []
    agree, cover = validate(d, [side], problems, main=False)
    pk = float(d.peak_gib.max()) if "peak_gib" in d else float("nan")
    if not pk <= PEAK_GIB_MAX:
        problems.append(f"peak GPU memory {pk} GiB > {PEAK_GIB_MAX}")
    try:
        _, cside, search, slog, routes = load_cal(tag, job, root)
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
                    problems.append(f"B={k}: the forced search tested {nh} of {s.n_cand} heads")
                c = {tuple(h) for h in json.loads(s.critical)}
                if len(c) != 1:
                    problems.append(f"B={k}: the forced search added {len(c)} heads, not 1")
                crit |= c
                orc |= {tuple(h) for h in json.loads(s.oracle_dense)}
            if routes["routes_seq2"][k] != L1C.apply_critical(routes["routes_pool"][k], sorted(crit)):
                problems.append(f"B={k}: routes_seq2 is not R0 plus the forced heads (union)")
            if routes["routes_union"][k] != L1C.apply_critical(routes["routes_pool"][k], sorted(orc)):
                problems.append(f"B={k}: routes_union is not R0 plus the oracle's dense heads")
    main_preset = "qwen32" if tag.startswith("qwen") else "main128"
    n_main = len(L.build_plan(L.PRESETS[main_preset]))
    dec = float(d[d.arm != "fp"].t_arm.median())
    tf = float(d.t_tf.median())
    per = float(d.t_prefill.median()) + 2 * float(d.t_precompute.max()) + n_main * (3 * dec + 2 * tf)
    proj_h = per * MAIN_BLOCK_PROMPT_TASKS / 3600
    if proj_h > 0.9 * WALL_H[main_preset]:
        problems.append(f"projected {main_preset} block {proj_h:.1f} h > 90% of {WALL_H[main_preset]} h")
    print(f"R14 Stage 1d pilot {job} ({tag}): {len(d)} rows, plan {len(plan_of(side))} arms, fp replay "
          f"{agree:.3f}, answer-value coverage {cover:.2f}, peak {pk:.1f} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected {main_preset} block {proj_h:.1f} h ({n_main} arms)")
    cols = [c for c in ("score", "a_sum_nll", "tf_c_sum_nll", "bits_per_token", "evict_frac",
                        "kept_width", "n_resel", "t_arm", "t_tf") if c in d]
    print(d.pivot_table(index=["arm", "B"], values=cols, aggfunc="mean").round(3).to_string())
    if side.get("dropped_arms"):
        print(f"dropped arms: {side['dropped_arms']}")
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--pilot-tag", default="pilot128")
    ap.add_argument("--main128", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--main32", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--qwen32", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--cal128", metavar="JOB")
    ap.add_argument("--cal32", metavar="JOB")
    ap.add_argument("--calq32", metavar="JOB")
    ap.add_argument("--calq8", metavar="JOB")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1d"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(gate(a.pilot, a.pilot_tag, a.results_root))
    cells, cal = {}, {}
    for flag, name, tag, cflag, ctag in (("main128", "llama31-8b@131072", "main128", "cal128", "cal128"),
                                         ("main32", "llama31-8b@32768", "main32", "cal32", "cal32"),
                                         ("qwen32", "qwen3-30b-a3b-2507@32768", "qwen32", "calq32",
                                          "calq32")):
        if getattr(a, flag):
            cells[name] = (tag, getattr(a, flag))
            if getattr(a, cflag):
                cal[name] = (ctag, getattr(a, cflag))
    if not cells:
        ap.error("give --pilot JOB or at least one cell")
    sys.exit(read_cells(cells, cal, a.out_stem, a.results_root,
                        qwen_cal8=("calq8", a.calq8) if a.calq8 else None))


if __name__ == "__main__":
    main()
