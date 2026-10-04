#!/usr/bin/env python3
"""R14 Stage 1f gate and reader. The rules below are FROZEN: written 2026-10-03,
before any Stage 1f output existed. Design: s1f_lib.py. Driver: run_s1f.py. The
kernel and two-tier timing rules are in s1f_kernel.py, read by bench_s1f_kernel.py.

    python read_stage1f.py --pilot JOB --pilot-tag reusepilotf|ttpilot|qwenpilotf
    python read_stage1f.py --reuse128 J1 J2 J3 J4 --reuse32 J5 J6 J7 J8 --out-stem .../stage1f_reuse
    python read_stage1f.py --qwen32 J1 J2 J3 J4 --qregress JR --qcal JC --out-stem .../stage1f_qwen
    python read_stage1f.py --tt128 J1 J2 J3 J4 --tt32 J5 J6 J7 J8 [--ttregress JR] --out-stem .../stage1f_tt

Result directories: h0_measurement/results/r14s1f_<tag>_<job>/ (files s1f_*).

CELLS
  F3  'reuse128f': mixed prompts 8500-8595, 4 blocks of 24.
      'reuse32f': 8600-8687, 4 blocks of 22.
  F2  'tt128': Llama-3.1-8B, prompts 8900-8939, 4 blocks of 10.
      'tt32': 9000-9039, 4 blocks of 10.
  F4  'qwen32f': Qwen3-30B-A3B-2507 @32768, prompts 8800-8839, 4 blocks of 10.
      Regression block: Stage 1e's Qwen failures 8215, 8218, 8235 (multivalue;
      router deletions) and 8234 (multikey; TurboQuant's key confusion).
      Calibration: prompts 8700-8729, multikey and multivalue on all 30, single
      and vt on 10, budgets 2.5 and 3, stop rule eos_only.
METRIC   dA(X) = a_sum_nll(X) - a_sum_nll(fp), as in Stages 1d-1e. Prompt-tasks
         without an answer value drop (reported). Task cells with FP < 0.9 drop
         (main cells).
COMPARATOR D_L = uniform@3 in X's value lens. A two-tier read's lens is that of
         its tier-1 values, so its GPU store is D_L's.
INTERVALS, MATCHED, VERDICT
  - Intervals: prompt bootstrap within block, 10,000 draws, seed 14; 90%
    intervals.
  - MATCHED: NLL hi <= 0.10 nats and > 2-nat tail-share difference hi <= 0.05.
  - WORSE: NLL lo > 0.10 or tail lo > 0.05.
  - Verdict per family (adds 'qread2t'): WIN at rho <= 0.80, TIE <= 1.00.
BYTES    Stage 1e's rule, read per decode step. A two-tier read reads tier-2
         rows: the selected keys at the tier-2 width plus a keep bitmap, and the
         selected values at the tier-2 width. Its GPU-stored bytes are tier 1's
         (= D_L's); the host bytes of tier 2 are reported.
F3 REUSE (per reuse cell; units = (prompt, role))
  - FP_DROP: a unit whose FP answer scores < 1 drops from its role, for every
    arm. A role keeping < 75% of its units makes the cell INVALID. A role with
    < 80 kept units reports its labels as UNDERPOWERED.
  - Per role and arm: MATCHED-rule label vs D_L.
  - Label, per lens, from the SECOND question only: s1f_lib.reuse_label_q2 over
    every r with qread_v@r and snapq_v@r, where diff = mean over Q2 units of
    dA(snapq) - dA(qread), with its interval:
    - REUSE_DIFFERENTIATES: at some r the reads are MATCHED on Q2 and diff >= 0.10
      with lo > 0;
    - REUSE_NO_DIFFERENCE: diff hi < 0.10 at every r;
    - QREAD_FAILS_REUSE: the reads are MATCHED on Q2 at no r;
    - REUSE_MIXED otherwise.
    The first question is reported, never used.
  - Reported: the same label within each Q2 task (multikey, vt); the nested
    router and the two-tier read per role; stored bytes.
F2 TWO-TIER (tt128, tt32, qwen32f), per cell
  - TT_MATCHED(r): qread2t_v4@r MATCHED vs D_V4.
  - TT_VS_READ(r): effect label of dA(qread2t_v4@r) - dA(qread_v4@r) (|mean| >=
    0.05 and an interval excluding 0).
  - s1f_lib.tt_verdict per r: TT_ADVANTAGE, TT_PARITY, TT_PARITY_COST or
    TT_NOT_MATCHED.
  - QPASS_COST: effect label of dA(qread2t_v16@1/8) - dA(qreadfp_v16@1/8), what a
    low-bit question pass costs when the answer reads exact rows.
  - FP8_TIER: effect label of dA(qread2t8_v4@1/8) - dA(qread2t_v4@1/8).
  - CONFUSION (Qwen regression prompt 8234): FIXED_BY_TT iff dA(qread2t_v16@1/8)
    <= 2 nats, reported beside D's.
F4 QWEN
  - CAL_COVERS_MULTIVALUE: the calibration searched >= 1 multivalue prompt-task
    with an answer value. Else the cell is INVALID.
  - QWEN_TAIL2: s1e_lib.tail_label for router_nest3_calib@3 against
    router_seq2_calib@3, with FIXED (<= 2 nats over D) on the router
    regression prompt-tasks 8215, 8218 and 8235.
  - Reported: catastrophes per router, the heads the calibration added by
    task, nest3 - nest2 (effect label), and the family verdicts (SIEVE
    included).
VALIDITY (any failure -> INVALID)
  - Stage 1e's V1-V7.
  - Two-tier rows store 3 bits dense (tier 1), read exactly floor(r C) rows per
    KV head at the tier-2 width (bits = width x read fraction), with tier2_bits
    16 or 8 as the arm name says.
  - Reuse cells replace V1's FP-per-task rule with FP_DROP.

AMENDMENTS A1-A4 (2026-10-03; s1f_lib's docstring). Written after the reuse128f,
reuse32f, tt128 and tt32 blocks had run under the rules above, which still read
them unchanged; before any Qwen evaluation block or regression block. A cell is
read by what its blocks carry (sidecar amend):
  - every block amended: the per-unit statistic is dS = s_set_nll(X) -
    s_set_nll(fp) (A2), wherever the rules above say dA; dA on a_sum_nll is
    reported beside it;
  - no block amended: dA as above, with a_span_nll (all-token span NLL,
    recomputed from the stored per-token log-probabilities) reported beside it;
  - a mix: INVALID. A regression block must match its cell.
  Reported in every cell, not gated: per arm, the units whose answer scores below
  FP's, against D_L's on the same units, with the paired interval of the share
  difference; in amended cells, the units answered in another order and how many
  of them were replayed in that order.
  A2 VALIDITY (amended cells): every A2 column present; fp rows score their own
  order (s_set_nll = a_span_nll, not reordered); s_set_nll <= a_span_nll; a replay
  in another order only for a reordered answer; in main and reuse cells, every
  arm family has a self-check replay (a2_check) and each is <= A2_CHECK_TOL nats.
  A3 CONFUSION 8109 (tt128's regression block, --ttregress; Llama 8109
  niah_multikey, one answer value, where dA = dS): FIXED_BY_TT iff the two-tier
  read (qread2t_v16@1/8) is within 2 nats of FP while D is not; NOT_FIXED if both
  are not; NOT_REPRODUCED if D is within 2 nats. The block runs tt128's plan; the
  other reads are reported beside it. Without the block the label is absent.
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
import s1c_lib as L1C  # noqa: E402
import s1f_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"reuse128f": 6.0, "reuse32f": 2.5, "tt128": 5.0, "tt32": 2.0, "qwen32f": 4.0}
BLOCK_UNITS = {"reuse128f": 48, "reuse32f": 44, "tt128": 40, "tt32": 40, "qwen32f": 40}
TAIL_NATS, FIXED_NATS = RD.TAIL_NATS, RD.FIXED_NATS
FP_MIN = RD.FP_MIN
DREF = RD.DREF
ROUTER_REGRESS = [(8215, "niah_multivalue"), (8218, "niah_multivalue"), (8235, "niah_multivalue")]
CONFUSION = (8234, "niah_multikey")
CONFUSION_LLAMA = tuple(L.TT_REGRESS[0])     # A3
bk, ci, fmt = R1C.bk, R1C.ci, R1C.fmt


# ------------------------------------------------------------------ loading
def load_run(tag, job, mode="evaluate", root=RESULTS):
    d = os.path.join(root, f"r14s1f_{tag}_{job}")
    pq = [x for x in glob.glob(os.path.join(d, f"s1f_{mode}_*.parquet"))
          if not x.endswith(("_search.parquet", "_searchlog.parquet"))]
    js = glob.glob(os.path.join(d, f"s1f_{mode}_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one s1f_{mode} parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    if "q_role" not in rows:
        rows["q_role"] = ""
    rows["q_role"] = rows.q_role.fillna("")
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def load_cal(tag, job, root=RESULTS, sha=None):
    rows, side = load_run(tag, job, "calibrate", root)
    d = side["_dir"]
    side["_routes_local"] = RD.resolve_routes(side["write_routes"], sha)
    return (rows, side, pd.read_parquet(os.path.join(d, side["search"])),
            pd.read_parquet(os.path.join(d, side["searchlog"])), json.load(open(side["_routes_local"])))


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
        want_rule = "eos_only" if side["model"].startswith("qwen") else "r8"
        if side.get("stop_rule") != want_rule:
            problems.append(f"block {side['_job']}: stop rule {side.get('stop_rule')}, expected {want_rule} (V7)")
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
    tt = d[fam == "qread2t"]
    if len(tt):
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(tt.B, tt.ctx_len)])
        named = tt.arm.map(lambda a: L.parse_arm(a)["tier2_bits"]).to_numpy()
        ok = (np.allclose(tt.stored_bits_per_token, L.STORE_WIDTH) and (tt.stored_evict_frac == 0).all()
              and np.allclose(tt.read_frac.to_numpy(), k, atol=1e-9)
              and np.allclose(tt.tier2_bits.to_numpy(dtype=float), named)
              and np.allclose(tt.bits_per_token, tt.tier2_bits.astype(float) * tt.read_frac, atol=1e-6))
        if not ok:
            problems.append("a two-tier row does not store tier 1 dense and read floor(r C) per head at its "
                            "tier-2 width")
    sq = d[fam == "snapq"]
    if len(sq):
        k = np.array([L1C.qread_keep_count(r, c) / c for r, c in zip(sq.B, sq.ctx_len)])
        if not (np.allclose(sq.read_frac, k, atol=1e-9) and np.allclose(sq.stored_bits_per_token, sq.bits_per_token)
                and np.allclose(sq.stored_evict_frac, sq.evict_frac)):
            problems.append("a snapq row does not read and store exactly floor(r C) per head")
    for arm in sorted(d[d.twin.fillna("") != ""].arm.unique()):
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
    if agree < RD.V3_MIN:
        problems.append(f"fp teacher-forced replay agrees with fp's greedy on {agree:.3f} < {RD.V3_MIN}")
    ok_fp = fp[fp.score >= 1]
    cover = float((ok_fp.a_len > 0).mean()) if len(ok_fp) else 0.0
    if cover < RD.V6_MIN:
        problems.append(f"FP's answer-value mask found in {cover:.2f} < {RD.V6_MIN} of correct FP answers")
    if main and kind != "reuse":
        fps = d[d.arm == "fp"].groupby("task").score.mean()
        if (fps >= FP_MIN).sum() < min(fp_tasks_min, len(fps)):
            problems.append(f"FP below {FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree, cover


def cell_metric(name, sides, problems):
    """A2: the per-unit statistic a cell is read with, by what its blocks carry."""
    amended = [s.get("amend") == L.AMEND for s in sides]
    if amended and all(amended):
        return L.METRIC_A2
    if any(amended):
        problems.append(f"{name}: blocks run before and after amendment {L.AMEND} (mixed)")
    return L.METRIC_FROZEN


def validate_a2(d, problems, name, self_check=True):
    """A2's validity checks for an amended cell (module docstring)."""
    miss = [c for c in L.A2_COLS + ("a2_check",) if c not in d]
    if miss:
        problems.append(f"{name}: amended blocks lack the A2 columns {miss}")
        return
    ro, rp = d.ans_reordered.fillna(False).astype(bool), d.own_replay.fillna(False).astype(bool)
    fp = d[(d.arm == "fp")].dropna(subset=["s_set_nll", "a_span_nll"])
    if not (np.allclose(fp.s_set_nll, fp.a_span_nll) and not ro[d.arm == "fp"].any()):
        problems.append(f"{name}: an fp row does not score its own order (A2)")
    x = d.dropna(subset=["s_set_nll", "a_span_nll"])
    if (x.s_set_nll > x.a_span_nll + 1e-6).any():
        problems.append(f"{name}: s_set_nll above a_span_nll (A2)")
    if (rp & ~ro).any():
        problems.append(f"{name}: a replay in another order for an answer that is not reordered (A2)")
    if self_check:
        arms = d[d.arm != "fp"]
        fams = set(zip(arms.family, arms.twin.fillna("")))
        ck = arms.dropna(subset=["a2_check"])
        got = set(zip(ck.family, ck.twin.fillna("")))
        if fams - got:
            problems.append(f"{name}: no self-check replay (a2_check) for {sorted(fams - got)} (A2)")
        if (ck.a2_check > L.A2_CHECK_TOL).any():
            bad = ck[ck.a2_check > L.A2_CHECK_TOL]
            problems.append(f"{name}: a replay through replay_ids differs from the first by up to "
                            f"{bad.a2_check.max():.3f} nats ({sorted(set(bad.arm))}) (A2)")


def add_span_posthoc(v):
    """Frozen cells: a_span_nll recomputed from the stored per-token log-probabilities
    and FP's masks (every row stores FP's), where a row lacks it."""
    if "a_span_nll" in v and v.a_span_nll.notna().all():
        return v
    v = v.copy()
    v["a_span_nll"] = [L.span_nll(lp, vm, se)[1] if isinstance(vm, str) and lp is not None else np.nan
                       for lp, vm, se in zip(v.tf_logp, v.tf_vmask, v.span_end)]
    return v


# ---------------------------------------------------------------- analysis
def add_bytes(v):
    """Stage 1e's rule; two-tier reads read tier-2 rows (keys and values at the
    tier-2 width) and store tier 1."""
    v = v.copy()
    d8 = v.head_dim / 8.0
    fam = v.arm.map(L.family)
    tt = (fam == "qread2t").to_numpy()
    rv = v.v_bits.to_numpy(dtype=float).copy()
    if "read_v_bits" in v:
        rv[tt] = v.read_v_bits.to_numpy(dtype=float)[tt]
    vs = np.where(rv >= 16, 0.0, np.where(tt, 0.0, 16.0 / v.head_dim))
    tail = (v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len
    v["bytes"] = d8 * (v.bits_per_token + v.key_side) + (1 - v.evict_frac) * d8 * (rv + vs) + tail
    sk = np.where(fam.isin(["qread", "qread2t"]), 16.0 / v.head_dim, np.where(fam == "qreadfp", 0.0, v.key_side))
    vs_store = np.where(v.v_bits >= 16, 0.0, 16.0 / v.head_dim)
    v["bytes_stored"] = (d8 * (v.stored_bits_per_token + sk)
                         + (1 - v.stored_evict_frac) * d8 * (v.v_bits + vs_store) + tail)
    v["bytes_host"] = np.where(tt, d8 * 2 * v.get("tier2_bits", pd.Series(0, index=v.index)).fillna(0).astype(float),
                               0.0)
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


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


def _dA(d, key, keep=None, col=L.METRIC_FROZEN):
    A = d.pivot_table(index=key, columns=["arm", "B"], values=col, aggfunc="first")
    A = A[~A[("fp", 0.0)].isna()]
    if keep is not None:
        A = A[A.index.isin(keep)]
    return A.sub(A[("fp", 0.0)], axis=0)


def _unit_table(v, dA, value):
    """One per-unit column of v pivoted like dA (same units, same arms)."""
    t = v.pivot_table(index=list(dA.index.names), columns=["arm", "B"], values=value, aggfunc="first", dropna=False)
    return t.reindex(index=dA.index, columns=dA.columns)


def _points(dA, v, pidx, col=L.METRIC_FROZEN):
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1) if s.index.nlevels > 2 else s, pidx).to_numpy()  # noqa: E731
    cols = ["bytes", "bytes_stored", "bytes_amort", "bytes_host", "evict_frac", "kept_width", "needle_keep", "score"]
    means = v.groupby(["arm", "B"])[cols].mean()
    # reported beside the cell's statistic (A2): the other one, answers below FP, answers in another order
    alt = L.METRIC_FROZEN if col == L.METRIC_A2 else "a_span_nll"
    vx = add_span_posthoc(v) if alt == "a_span_nll" else v
    At = _unit_table(vx, dA, alt)
    dAlt = At.sub(At[("fp", 0.0)], axis=0)
    S = _unit_table(v, dA, "score")
    below = S.lt(S[("fp", 0.0)] - 1e-9, axis=0).astype(float)
    amended = col == L.METRIC_A2
    if amended:
        RO = _unit_table(v.assign(_ro=v.ans_reordered.fillna(False).astype(float)), dA, "_ro")
        RP = _unit_table(v.assign(_rp=v.own_replay.fillna(False).astype(float)), dA, "_rp")
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
                 dA_by_task={t: float(g.mean()) for t, g in dA[c].groupby(level=-1)} if "task" in dA.index.names
                 else {})
        p["alt"] = dict(stat=alt, dA=L1C.boot_ci(pm(dAlt[c]), W),
                        vs_D=L1C.boot_ci(pm(dAlt[c] - dAlt[Dc]), W) if Dc in dAlt and c != Dc else None)
        p["below_fp"] = dict(n=int(below[c].sum()), n_D=int(below[Dc].sum()) if Dc in below else None,
                             diff=L1C.boot_ci(pm(below[c] - below[Dc]), W) if Dc in below and c != Dc else None)
        if amended:
            p["reordered"] = dict(n=int(RO[c].sum()), replayed=int(RP[c].sum()))
        pts[f"{arm}@{bk(B)}"] = p
    return pts, W, pm


def tt_labels(dA, W, pm, pts):
    """F2 labels for one main cell (dA indexed by job, prompt, task)."""
    out = {}
    for col in dA.columns:
        if col[0] != "qread2t_v4":
            continue
        r = col[1]
        z = dict(matched=pts[f"qread2t_v4@{bk(r)}"]["vs_D"]["label"] == "MATCHED",
                 rho=pts[f"qread2t_v4@{bk(r)}"]["vs_D"]["rho"])
        if ("qread_v4", r) in dA.columns:
            diff = L1C.boot_ci(pm(dA[col] - dA[("qread_v4", r)]), W)
            z["vs_read"] = dict(diff=diff, label=L.effect_label(*diff, "TT"))
            z["verdict"] = L.tt_verdict(z["matched"], z["vs_read"]["label"])
        out[bk(r)] = z
    a, b = ("qread2t_v16", 0.125), ("qreadfp_v16", 0.125)
    if a in dA.columns and b in dA.columns:
        diff = L1C.boot_ci(pm(dA[a] - dA[b]), W)
        out["QPASS_COST"] = dict(diff=diff, label=L.effect_label(*diff, "QPASS"))
    a, b = ("qread2t8_v4", 0.125), ("qread2t_v4", 0.125)
    if a in dA.columns and b in dA.columns:
        diff = L1C.boot_ci(pm(dA[a] - dA[b]), W)
        out["FP8_TIER"] = dict(diff=diff, label=L.effect_label(*diff, "FP8_TIER"))
    return out


def analyse_main(name, d, sides, col=L.METRIC_FROZEN):
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= FP_MIN)
    v = add_bytes(d[d.task.isin(valid)])
    dA = _dA(v, ["job", "prompt_idx", "task"], col=col)
    pidx = dA.index.droplevel("task").unique()
    pts, W, pm = _points(dA, v, pidx, col)
    res = dict(cell=name, metric=col, fp=float(fp_cells.mean()), fp_by_task=fp_cells.round(4).to_dict(),
               valid_tasks=valid, n_prompts=int(len(pidx)), n_prompt_tasks=int(len(dA)), points=pts,
               families=families(pts))
    res["tt"] = tt_labels(dA, W, pm, pts)
    pr = sides[0]["preset"]
    if pr.get("routers"):
        D16 = ("uniform", 3.0)
        res["routers"] = {f"{a}@{bk(B)}": dict(catastrophes=L.catastrophes(dA[(a, float(B))], dA[D16]),
                                               label=pts[f"{a}@{bk(B)}"]["vs_D"]["label"],
                                               rho=pts[f"{a}@{bk(B)}"]["vs_D"]["rho"])
                          for a, B, _ in pr["routers"] if (a, float(B)) in dA.columns}
        a, b = ("router_nest3_calib", 3.0), ("router_nest2_calib", 3.0)
        if a in dA.columns and b in dA.columns:
            diff = L1C.boot_ci(pm(dA[a] - dA[b]), W)
            res["nest3_vs_nest2"] = dict(diff=diff, label=L.effect_label(*diff, "NEST3"))
    return res, dA, W, pm


def analyse_regress(d, col=L.METRIC_FROZEN):
    A = d.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=col, aggfunc="first")
    A = A[~A[("fp", 0.0)].isna()]
    dA = A.sub(A[("fp", 0.0)], axis=0)
    out = {}
    for c in dA.columns:
        if c[0] == "fp":
            continue
        out[f"{c[0]}@{bk(c[1])}"] = {f"{p}/{t}": float(x) for (p, t), x in dA[c].round(3).items()}
    return out, dA


def analyse_confusion(rd, col=L.METRIC_FROZEN):
    """A3: Llama 8109 niah_multikey in tt128's regression block."""
    rr, rdA = analyse_regress(rd, col if col in rd else L.METRIC_FROZEN)
    u = CONFUSION_LLAMA
    if u not in rdA.index:
        return dict(label="NO_UNIT", dA={})
    x = {f"{a}@{bk(B)}": float(rdA.loc[u, (a, B)]) for a, B in rdA.columns if a != "fp"}
    xd, xt = x.get("uniform@3"), x.get("qread2t_v16@0.125")
    if xd is None or xt is None:
        lab = "NO_DATA"
    elif xd <= FIXED_NATS:
        lab = f"NOT_REPRODUCED (D {xd:.2f} nats)"
    else:
        lab = f"{'FIXED_BY_TT' if xt <= FIXED_NATS else 'NOT_FIXED'} (two-tier {xt:.2f} nats, D {xd:.2f})"
    return dict(label=lab, dA=x)


def analyse_reuse(name, d, sides, col=L.METRIC_FROZEN):
    v = add_bytes(d)
    fpsc = v[v.arm == "fp"].set_index(["job", "prompt_idx", "q_role"]).score
    keep = fpsc[fpsc >= 1.0].index
    res = dict(cell=name, metric=col, units={}, roles={}, reuse={}, by_task={}, problems=[])
    for role in ("Q1", "Q2"):
        n = int((fpsc.index.get_level_values("q_role") == role).sum())
        k = int((keep.get_level_values("q_role") == role).sum())
        res["units"][role] = dict(units=n, kept=k, dropped=n - k)
        if n and k < L.REUSE_MIN_KEEP * n:
            res["problems"].append(f"{name} {role}: FP drops leave {k} of {n} units (< {L.REUSE_MIN_KEEP:.0%})")
    dA = _dA(v, ["job", "prompt_idx", "q_role"], keep, col=col)
    T = v.pivot_table(index=["job", "prompt_idx", "q_role"], columns=["arm", "B"], values="task",
                      aggfunc="first").reindex(dA.index)
    for role in ("Q1", "Q2"):
        sub = dA.xs(role, level="q_role")
        vv = v[v.q_role == role]
        pidx = sub.index.unique()
        pts, W, pm = _points(sub, vv, pidx, col)
        res["roles"][role] = dict(n=int(len(sub)), points=pts, families=families(pts),
                                  underpowered=bool(len(sub) < L.REUSE_MIN_UNITS))
    q2 = dA.xs("Q2", level="q_role")
    t2 = T.xs("Q2", level="q_role")[("fp", 0.0)]
    pidx2 = q2.index.unique()
    W2 = BM.boot_weights(pidx2)
    pts2 = res["roles"]["Q2"]["points"]

    def per_lens(vb, mask=None):
        per_r = {}
        for col in q2.columns:
            if col[0] != f"snapq_v{vb}":
                continue
            r = col[1]
            qa = (f"qread_v{vb}", r)
            if qa not in q2.columns:
                continue
            diff = (q2[col] - q2[qa])
            if mask is None:
                z_diff = L1C.boot_ci(diff.reindex(pidx2).to_numpy(), W2)
                lab = pts2[f"qread_v{vb}@{bk(r)}"]["vs_D"]["label"]
            else:
                ix = diff.index[mask]
                z_diff = L1C.boot_ci(diff[mask].to_numpy(), BM.boot_weights(ix))
                Dc = (DREF["V16" if vb == 16 else "V4"], 3.0)
                xq, xd = q2.loc[ix, qa], q2.loc[ix, Dc]
                Wt = BM.boot_weights(ix)
                nll = L1C.boot_ci((xq - xd).to_numpy(), Wt)
                tl_ = L1C.boot_ci(((xq > TAIL_NATS).astype(float) - (xd > TAIL_NATS).astype(float)).to_numpy(), Wt)
                lab = R1C.matched_label(nll, tl_)
            per_r[bk(r)] = dict(qread_q2=lab, diff=z_diff,
                                snapq_q2=pts2.get(f"snapq_v{vb}@{bk(r)}", {}).get("vs_D", {}).get("label"))
        return per_r

    for lens, vb in (("V16", 16), ("V4", 4)):
        per_r = per_lens(vb)
        lab = L.reuse_label_q2(per_r)
        if res["roles"]["Q2"]["underpowered"]:
            lab = f"UNDERPOWERED ({lab})"
        res["reuse"][lens] = dict(per_r=per_r, label=lab)
        res["by_task"][lens] = {}
        for task in sorted(t2.dropna().unique()):
            m = (t2 == task).to_numpy()
            pr_ = per_lens(vb, m)
            res["by_task"][lens][task] = dict(per_r=pr_, label=L.reuse_label_q2(pr_), n=int(m.sum()))
    return res


def cal_summary(cal):
    rows, side, search, slog, routes = cal
    out = {}
    for k in routes["routes_pool"]:
        s = search[search.B.astype(float) == float(k)] if len(search) else search
        heads_by_task = {}
        for r_ in s.itertuples():
            for h in json.loads(r_.critical):
                heads_by_task.setdefault(r_.task, set()).add(tuple(h))
        out[k] = dict(prompt_tasks=int(len(s)), by_task=s.groupby("task").size().to_dict() if len(s) else {},
                      fail_by_task=s.groupby("task").fail.sum().astype(int).to_dict() if len(s) else {},
                      heads_by_task={t: sorted(v) for t, v in heads_by_task.items()},
                      dense_R0=len(L1C.dense_heads(routes["routes_pool"][k])),
                      dense_seq3=len(L1C.dense_heads(routes["routes_seq3"][k])),
                      dense_nest3=len(L1C.dense_heads(routes["routes_nest3"][k])))
    covers = bool(len(search) and (search.task == "niah_multivalue").any())
    return out, covers


# ------------------------------------------------------------------ report
def _points_table(points, metric=L.METRIC_FROZEN):
    st = "dS" if metric == L.METRIC_A2 else "dA"
    alt = next((p["alt"]["stat"] for p in points.values() if "alt" in p), "")
    Lh = [f"| arm@B | lens | bytes read | ρ | ρ GPU mem | host bytes | {st} vs FP | {st} vs D [90%] | tail | "
          f"Δtail [90%] | label | cat>3 | max {st} | score | {alt} vs D [90%] | below FP: X / D, Δshare [90%] "
          f"| reordered (replayed) |",
          "|---|---|---:|---:|---:|---:|---|---|---:|---|---|---:|---:|---:|---|---|---|"]
    for k, p in sorted(points.items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
        z = p.get("vs_D")
        a, b, o = p.get("alt", {}), p.get("below_fp", {}), p.get("reordered")
        alt_d = ci(a["vs_D"]) if a.get("vs_D") else "—"
        below = f"{b.get('n', '—')} / {b.get('n_D', '—')}, " + (ci(b["diff"]) if b.get("diff") else "—")
        reord = f"{o['n']} ({o['replayed']})" if o else "—"
        Lh.append(f"| {k} | {p['lens']} | {p['bytes']:.1f} | {fmt(z and z['rho'], 2)} | {fmt(z and z['rho_mem'], 2)} "
                  f"| {p['bytes_host']:.0f} | {ci(p['dA'])} | {ci(z['nll']) if z else '(D)'} | {p['tail']:.3f} | "
                  f"{ci(z['tail']) if z else '—'} | {z['label'] if z else '—'} | {z['cat'] if z else '—'} | "
                  f"{p['max_dA']:.2f} | {p['score']:.3f} | {alt_d} | {below} | {reord} |")
    return Lh


def _fam_table(fams):
    Lh = ["| lens | " + " | ".join(L.DEPLOYABLE) + " |", "|---|" + "---|" * len(L.DEPLOYABLE)]
    for lens, fz in fams.items():
        Lh.append(f"| {lens} | " + " | ".join(z["verdict"] + (f" {z['point']} ρ {z['rho']:.2f}" if z["point"] else "")
                                              for z in (fz[f] for f in L.DEPLOYABLE)) + " |")
    return Lh


def report(results, summary, out_stem, title):
    Lh = [f"# R14 Stage 1f — {title} (read_stage1f.py; rules frozen in its docstring)", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    for r in results:
        stat = "dS (A2: s_set_nll)" if r.get("metric") == L.METRIC_A2 else "dA (a_sum_nll, frozen)"
        if "roles" in r:
            Lh += [f"## {r['cell']} — units {r['units']}; per-unit statistic {stat}", ""]
            for lens, z in r["reuse"].items():
                Lh.append(f"**F3 {lens}: {z['label']}** — " + "; ".join(
                    f"r={rr}: reads on Q2 {x['qread_q2']}, snapq on Q2 {x['snapq_q2']}, snapq − reads {ci(x['diff'])}"
                    for rr, x in z["per_r"].items()))
                for task, tz in r["by_task"][lens].items():
                    Lh.append(f"- Q2 = {task} ({tz['n']} units): {tz['label']} — " + "; ".join(
                        f"r={rr}: snapq − reads {ci(x['diff'])}, reads {x['qread_q2']}" for rr, x in tz["per_r"].items()))
            for role, rz in r["roles"].items():
                Lh += ["", f"### {role} ({rz['n']} units{', UNDERPOWERED' if rz['underpowered'] else ''})", ""] \
                    + _fam_table(rz["families"]) + [""] + _points_table(rz["points"], r.get("metric"))
            Lh.append("")
            continue
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f} ({r['fp_by_task']}), {r['n_prompts']} prompts, "
               f"{r['n_prompt_tasks']} prompt-tasks; per-unit statistic {stat}", ""] + _fam_table(r["families"]) \
            + [""] + _points_table(r["points"], r.get("metric")) + [""]
        for k, z in r["tt"].items():
            if k in ("QPASS_COST", "FP8_TIER"):
                Lh.append(f"**F2 {k}**: {ci(z['diff'])} → {z['label']}")
            else:
                Lh.append(f"**F2 r={k}**: two-tier MATCHED {z['matched']} (ρ {fmt(z['rho'], 2)})"
                          + (f"; two-tier − single-tier {ci(z['vs_read']['diff'])} → {z['vs_read']['label']} → "
                             f"**{z['verdict']}**" if "vs_read" in z else ""))
        if r.get("routers"):
            Lh.append("**Routers**: " + "; ".join(f"{k}: {z['label']}, ρ {fmt(z['rho'], 2)}, catastrophes "
                                                 f"{z['catastrophes']}" for k, z in r["routers"].items()))
        if r.get("nest3_vs_nest2"):
            Lh.append(f"nest3 − nest2: {ci(r['nest3_vs_nest2']['diff'])} → {r['nest3_vs_nest2']['label']}")
        if r.get("tail"):
            Lh.append(f"**F4 QWEN_TAIL2: {r['tail']['label']}** ({r['tail']})")
        if r.get("regress"):
            Lh.append(f"**Regression** ({stat.split(' ')[0]} vs FP): "
                      + "; ".join(f"{k} {v}" for k, v in r["regress"].items()))
        if r.get("confusion"):
            Lh.append(f"**F2 CONFUSION 8109 (A3)**: {r['confusion']['label']} — dA vs FP: "
                      + "; ".join(f"{k} {v:.2f}" for k, v in r["confusion"]["dA"].items()))
        if r.get("cal"):
            Lh.append(f"**Calibration** (covers multivalue: {r['cal_covers_multivalue']}): {r['cal']}")
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def read_cells(cells, out_stem, root=RESULTS, title="", regress=None, cals=None):
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
        agree, cover = validate(d, sides, problems, kind, fp_tasks_min=3)
        metric = cell_metric(name, sides, problems)
        if metric == L.METRIC_A2:
            validate_a2(d, problems, name)
        e_sha = next((v["sha256"] for s in sides for k_, v in s.get("routes", {}).items()
                      if k_.startswith(("router_seq3_calib", "router_nest3_calib"))), None)
        cal = None
        if name in cals:
            try:
                cal = load_cal(*cals[name], root, sha=e_sha)
            except FileNotFoundError as e:
                problems.append(f"{name}: calibration routes unavailable: {e}")
        if is_qwen and cal is None:
            problems.append(f"{name}: the Qwen cell needs its calibration job")
        RD.validate_routes(sides, cal, problems)
        if problems:
            continue
        if kind == "reuse":
            r = analyse_reuse(name, d, sides, metric)
            problems += r.pop("problems")
            for lens, z in r["reuse"].items():
                summary[f"F3 {name} {lens}"] = z["label"]
        else:
            r, dA, W, pm = analyse_main(name, d, sides, metric)
            for k, z in r["tt"].items():
                summary[f"F2 {name} {k}"] = z.get("verdict", z.get("label"))
            if not is_qwen and name in regress:              # A3: tt128's regression block
                rd, rside = load_run(regress[name][0], regress[name][1], "evaluate", root)
                if plan_of(rside) != plan_of(sides[0]):
                    problems.append(f"{name}: the regression block ran another plan")
                validate(rd, [rside], problems, "main", main=False)
                r["confusion"] = analyse_confusion(rd)
                summary["F2 CONFUSION 8109"] = r["confusion"]["label"]
            if is_qwen:
                cs, covers = cal_summary(cal)
                r["cal"], r["cal_covers_multivalue"] = cs, covers
                if not covers:
                    problems.append(f"{name}: the calibration searched no multivalue prompt-task")
                summary["F4 CAL_COVERS_MULTIVALUE"] = covers
                if name in regress:
                    rd, rside = load_run(regress[name][0], regress[name][1], "evaluate", root)
                    if plan_of(rside) != plan_of(sides[0]):
                        problems.append(f"{name}: the regression block ran another plan")
                    validate(rd, [rside], problems, "main", main=False)
                    if (rside.get("amend") == L.AMEND) != (metric == L.METRIC_A2):
                        problems.append(f"{name}: the regression block and the cell differ on amendment {L.AMEND}")
                    elif metric == L.METRIC_A2:
                        validate_a2(rd, problems, f"{name} regression", self_check=False)
                    rr, rdA = analyse_regress(rd, metric)
                    r["regress"] = rr
                    D16 = ("uniform", 3.0)
                    fx, bs = ("router_nest3_calib", 3.0), ("router_seq2_calib", 3.0)
                    idx = [i for i in ROUTER_REGRESS if i in rdA.index]
                    reg_fix = [bool(rdA.loc[i, fx] - rdA.loc[i, D16] <= FIXED_NATS) for i in idx] if fx in rdA else []
                    reg_base = [bool(rdA.loc[i, bs] - rdA.loc[i, D16] <= FIXED_NATS) for i in idx] if bs in rdA else []
                    both = (r["points"].get("router_nest3_calib@3", {}).get("vs_D", {}).get("label") == "MATCHED"
                            and r["points"].get("router_nest3_calib+v4@3", {}).get("vs_D", {}).get("label")
                            == "MATCHED")
                    cats = r.get("routers", {})
                    r["tail"] = dict(
                        label=L.tail_label(both, cats.get("router_nest3_calib@3", {}).get("catastrophes", 0),
                                           cats.get("router_seq2_calib@3", {}).get("catastrophes", 0), reg_fix,
                                           reg_base, float((dA[fx] - dA[bs]).mean()) if fx in dA and bs in dA else 0.0),
                        regress_fixed=dict(fix=reg_fix, base=reg_base), matched_both_lenses=both)
                    summary["F4 QWEN_TAIL2"] = r["tail"]["label"]
                    if CONFUSION in rdA.index and ("qread2t_v16", 0.125) in rdA:
                        x = float(rdA.loc[CONFUSION, ("qread2t_v16", 0.125)])
                        summary["F2 CONFUSION 8234"] = (f"{'FIXED_BY_TT' if x <= FIXED_NATS else 'NOT_FIXED'} "
                                                        f"(two-tier {x:.2f} nats, D "
                                                        f"{float(rdA.loc[CONFUSION, ('uniform', 3.0)]):.2f})")
                else:
                    problems.append(f"{name}: no regression block given")
        r["fp_replay_agreement"], r["value_cover"] = agree, cover
        results.append(r)
    if problems:
        raise SystemExit("INVALID Stage 1f data:\n  " + "\n  ".join(problems))
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(nll_margin=RD.NLL_MARGIN, tail_margin=RD.TAIL_MARGIN, tail_nats=TAIL_NATS,
                                  reuse_eps=L.REUSE_EPS, reuse_min_units=L.REUSE_MIN_UNITS,
                                  reuse_min_keep=L.REUSE_MIN_KEEP, tt_eps=L.TT_EPS, fixed_nats=FIXED_NATS,
                                  boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED, amendments=L.AMEND,
                                  a2_check_tol=L.A2_CHECK_TOL),
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
    pk = side.get("peak_gib_dev_max") or []
    if not pk or not all(x <= PEAK_GIB_MAX for x in pk):
        problems.append(f"peak GPU memory per device {pk} GiB (limit {PEAK_GIB_MAX})")
    if tag == "qwenpilotf":
        try:
            _, cside, search, slog, routes = load_cal(tag, job, root)
            if not (len(search) and search.searched.all() and (search.iters == 1).all()):
                problems.append("the forced search did not run once per prompt-task and budget")
        except SystemExit as e:
            problems.append(f"pilot calibration missing: {e}")
    main = L.PILOT_OF[tag]
    n_main = len(L.build_plan(L.PRESETS[main]))
    per_q = 2 if mode == "reuse" else 1
    sel = (d.arm != "fp") & (d.copied_from.isna() if "copied_from" in d else True)
    dec = float(d[sel].t_arm.median())
    tf = float(d.t_tf.median())
    unit_s = (float(d.t_prefill.median()) + float(d.t_precompute.max())) / per_q + n_main * (dec + tf)
    proj_h = unit_s * BLOCK_UNITS[main] / 3600
    if proj_h > 0.9 * WALL_H[main]:
        problems.append(f"projected {main} block {proj_h:.1f} h > 90% of {WALL_H[main]} h")
    print(f"R14 Stage 1f pilot {job} ({tag} for {main}): {len(d)} rows, plan {len(plan_of(side))} arms, fp replay "
          f"{agree:.3f}, answer-value coverage {cover:.2f}, peak per GPU {pk} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected {main} block {proj_h:.1f} h ({n_main} arms, wall {WALL_H[main]} h)")
    if side.get("amend") == L.AMEND:
        # the self-check must run wherever FP stated several values (a multi-answer unit)
        multi = d[(d.arm == "fp") & d.task.isin(L.MULTI_ANSWER)].fp_ans_order.fillna("").str.contains(",")
        validate_a2(d, problems, f"pilot {job}", self_check=bool(multi.any()))
        if "a2_check" in d:
            ck = d.dropna(subset=["a2_check"])
            print(f"A2: {int(d.ans_reordered.fillna(False).astype(bool).sum())} rows answered in another order, "
                  f"{int(d.own_replay.fillna(False).astype(bool).sum())} replayed; self-check replays "
                  f"{len(ck)}, largest difference {ck.a2_check.max() if len(ck) else float('nan'):.2e} nats")
    idx = ["arm", "B"] + (["q_role"] if kind == "reuse" else [])
    cols = [c for c in ("score", "a_sum_nll", "s_set_nll", "bits_per_token", "evict_frac", "read_frac", "t_arm",
                        "t_tf", "t_own") if c in d]
    print(d.pivot_table(index=idx, values=cols, aggfunc="mean").round(3).to_string())
    for r_ in d[d.arm == "fp"].itertuples():
        print(f"  FP {r_.task}{'/' + r_.q_role if r_.q_role else ''}: score {r_.score:.2f}, pred {r_.pred[:120]!r}")
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--pilot-tag", default="ttpilot", choices=sorted(L.PILOT_OF))
    for f in ("reuse128", "reuse32", "qwen32", "tt128", "tt32"):
        ap.add_argument(f"--{f}", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--qregress", metavar="JOB")
    ap.add_argument("--qcal", metavar="JOB")
    ap.add_argument("--ttregress", metavar="JOB", help="A3: tt128's regression block (Llama 8109 niah_multikey)")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1f"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(gate(a.pilot, a.pilot_tag, a.results_root))
    cells, reg, cal = {}, {}, {}
    for flag, name, tag in (("reuse128", "reuse llama31-8b@131072", "reuse128f"),
                            ("reuse32", "reuse llama31-8b@32768", "reuse32f"),
                            ("qwen32", "qwen3-30b-a3b-2507@32768", "qwen32f"),
                            ("tt128", "tt llama31-8b@131072", "tt128"), ("tt32", "tt llama31-8b@32768", "tt32")):
        if getattr(a, flag):
            cells[name] = (tag, getattr(a, flag))
    if a.qregress:
        reg["qwen3-30b-a3b-2507@32768"] = ("qregress32f", a.qregress)
    if a.ttregress:
        reg["tt llama31-8b@131072"] = ("ttregress128", a.ttregress)
    if a.qcal:
        cal["qwen3-30b-a3b-2507@32768"] = ("qcal32f", a.qcal)
    if not cells:
        ap.error("give --pilot JOB or at least one cell")
    title = ", ".join(sorted({"F3" if n.startswith("reuse") else "F2" if n.startswith("tt") else "F2/F4"
                              for n in cells}))
    sys.exit(read_cells(cells, a.out_stem, a.results_root, title, reg, cal))


if __name__ == "__main__":
    main()
