#!/usr/bin/env python3
"""R14 Stage 1h gate and reader for R1 (calibration and bridge). The rules below are
FROZEN: written 2026-10-04, before any Stage 1h output existed (plan.md section 5).
Design: s1h_lib.py. Driver: run_s1h.py. Statistics: metrics_s1h.py.

    python read_stage1h.py --pilot JOB [--pilot-tag h1pilot]
    python read_stage1h.py --r1 J1 J2 --r1-seeds S0 S1 S2 [--out-stem .../findings/R1_reader]

Result directories: h0_measurement/results/r14s1h_<tag>_<job>/ (files s1h_*). The
bridge reads Stage 1g's g128 blocks 22465811 (prompts 9100-9109) and 22465812
(9110-9119) from h0_measurement/results/r14s1g_g128_<job>/.

CELLS
  'r1':       h1cal, Llama-3.1-8B @131072, prompts 9100-9119 (2 blocks), all four tasks,
              rotation seed 0. EXPLORATORY: its labels guide the design, never claims.
  'r1seeds':  h1regress on s1h_lib.REGRESS_R1 (8109, 8901, 8937; niah_multikey) at
              rotation seeds 0, 1, 2 (one job each).
UNITS    prompt x task. A unit whose FP answer has no answer value drops (reported).
         Labels use the tasks where FP's mean score >= 0.9; units where FP scores below
         1 also form the FP-FAILED stratum, reported (counts, mean dP per arm), not labelled.
METRICS  per unit, minus FP's value in the same process:
         dP = a_span_nll   PRIMARY (all tokens from FP's first answer-value token to the
                           span end, FP's order, no minimum);
         dK = kl_span      CO-PRIMARY (KL(FP || arm) summed over the same span; FP's is 0);
         dS = s_set_nll    Stage 1f-1g's A2 statistic, for continuity, with the bias of its
                           minimum per arm, mean(dS - dP);
         dA = a_sum_nll    Stages 1d-1e's, reported.
INTERVALS prompt bootstrap within block (bytes_model.boot_weights: 10,000 draws, seed 14),
         90% percentile intervals.
LABELS (R1 itself uses Stage 1g's margin 0.10 throughout)
  NEAR_FP(X)   hi90 mean dP <= 0.10 and hi90 share of units with dP > 2 nats <= 0.05;
               FAR_FROM_FP if lo > 0.10 or the share's lo > 0.05; else INCONCLUSIVE.
  EQUIV_FP(X)  the 90% interval of mean dP lies inside [-0.10, +0.10].
  MATCHED(X)   vs D_L = uniform@3 in X's value lens (V16), uniform+v4@3 (V4; also FP8's
               V8): Stage 1g's rule on dP (hi90 of mean dP - dP_D <= 0.10, tail share
               difference <= 0.05).
  NOISE        fp_noise: mean and 95th percentile of |dP| and of dK; lost answers.
               NOISE_DEGENERATE if on every unit fp_noise's per-token log-probabilities equal
               FP's and its KL is 0 (the gate fails on it; the read reports it).
  FP8_COST     fp8kv: mean dP and dK with intervals; lost answers.
  M_FP         the margin from R2 on: s1h_lib.margin_fp(hi90 of fp8kv's mean dP), i.e.
               clipped to [0.05, 0.10]. Recorded, not applied here.
  BRIDGE       for each (arm, B) of s1h_lib.BRIDGE_ARMS, per unit present in R1 and in
               Stage 1g's blocks (same prompt, task and context length; else INVALID):
               dP(R1) - dP(Stage 1g), bootstrapped over R1's prompts;
               s1h_lib.bridge_label (+-0.05) -> BRIDGE_OK / DRIFT [arms].
  BEST_DENSE   s1h_lib.best_dense over the mean dP of uniform+v4@4, kivi4_v4@4 and
               kvquant4_v4@4 (TurboQuant-4 kept unless beaten by >= 0.02).
  VOTE_LOSS_EXACT = dP(qreadfp_v16) - dP(qoraclefp_v16);  VOTE_LOSS_4 = dP(qread4_v4) -
               dP(qoracle4_v4), r = 1/8: effect label (|mean| >= 0.05 and a 90% interval
               excluding 0: _HURTS = the vote loses) and EQUIV within +-0.05.
  EXACT_KV     dP(qread2t4q_v4) - dP(qread2t4kq_v4): effect label (_HELPS = exact values help).
  CONFUSION    per prompt of r1seeds: D_V4's dP at each seed against the largest dP over
               the >= 4-bit and exact arms (CONF_OK_ARMS); s1h_lib.confusion_seeds_label.
  LOST         per arm: units below FP (score); exact McNemar vs FP (b = below, c = above);
               the system (qread2t4kq_v4@1/8) vs D_V4 on their below-FP units. Each lost
               answer's type: 'confused' (a distractor stated), 'incomplete' (some expected
               values, or a proper prefix of a value FP stated), 'other'.
REPORTED, NOT GATED: dS labels (NEAR_FP, MATCHED) for continuity; bytes read (rho) and
  GPU-stored (rho_mem) vs D_L; per-arm peak memory; dP by task.
VALIDITY (any failure -> INVALID)
  - Every unit holds the plan's arms once; one corpus; stage 1h, amend A1-A4, amend_1h M0;
    stop rule r8; fp_noise last; FP >= 0.9 on >= 3 of 4 tasks (r1); fp replay agreement
    >= 0.98; the answer-value mask on >= 90% of correct FP answers.
  - Audits: dense and kivi/kvquant = their width with nothing evicted; fp8kv 8 bits;
    single-tier and oracle reads store their width dense and read floor(r C) per KV head;
    two-tier rows as Stage 1g; value twins changed their values.
  - KL: FP's kl_all is 0 on every unit; every other arm's is finite and >= 0.
  - A2: Stage 1f's checks; in r1 every block has a self-check replay for every arm name,
    each <= A2_CHECK_TOL nats.
  - r1seeds: three runs at three distinct rotation seeds, each with r1's plan.
GATE (pilot, excluded): validity (non-main), A2 self-checks if a multi-answer FP answer
  exists, FP's KL 0, NOT NOISE_DEGENERATE, peak GPU memory <= 76 GiB per device, and the
  projected h1cal block (40 units) within 90% of its 6-hour limit.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
for _p in (HERE, S1G_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bytes_model as BM  # noqa: E402
import read_stage1c as R1C  # noqa: E402
import read_stage1e as RD  # noqa: E402
import read_stage1f as RF  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1g_lib as L1G  # noqa: E402
import s1h_lib as L  # noqa: E402
import metrics_s1h as M  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"h1cal": 6.0}
BLOCK_UNITS = {"h1cal": 40}
S1G_BRIDGE = ("g128", ("22465811", "22465812"))
PRIMARY, KL, CONT, FROZEN = "a_span_nll", "kl_span", "s_set_nll", "a_sum_nll"
MARGIN_R1, TAIL_MARGIN, TAIL_NATS = 0.10, RD.TAIL_MARGIN, RD.TAIL_NATS
EFFECT_EPS, VOTE_EQ_EPS = 0.05, 0.05
FP_MIN = RD.FP_MIN
DREF_H = {"V16": "uniform", "V4": "uniform+v4", "V8": "uniform+v4"}
SYSTEM = ("qread2t4kq_v4", 0.125)
DV4 = ("uniform+v4", 3.0)
CONF_OK_ARMS = [("uniform+v4", 4.0), ("kivi4_v4", 4.0), ("kvquant4_v4", 4.0), ("fp8kv", 8.0), ("fp+v4", 0.0),
                ("qread4_v4", 0.125), ("qread2t4kq_v4", 0.125), ("qread2t4q_v4", 0.125), ("qreadfp_v16", 0.125),
                ("qoraclefp_v16", 0.125), ("qoracle4_v4", 0.125)]
DENSE4 = [("uniform+v4", 4.0), ("kivi4_v4", 4.0), ("kvquant4_v4", 4.0)]
KEY = ["job", "prompt_idx", "task"]
bk, ci, fmt = R1C.bk, R1C.ci, R1C.fmt


def name(c):
    return f"{c[0]}@{bk(c[1])}"


# ------------------------------------------------------------------ loading
def load_run(tag, job, root=RESULTS, stage="s1h"):
    d = os.path.join(root, f"r14{stage}_{tag}_{job}")
    pq = [x for x in glob.glob(os.path.join(d, f"{stage}_evaluate_*.parquet"))
          if not x.endswith(("_search.parquet", "_searchlog.parquet"))]
    js = glob.glob(os.path.join(d, f"{stage}_evaluate_*.json"))
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one {stage}_evaluate parquet and sidecar, found {pq} {js}")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    rows["q_role"] = ""
    side = json.load(open(js[0]))
    side["_job"], side["_dir"] = str(job), d
    return rows, side


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


# ----------------------------------------------------------------- validity
def validate_h(d, sides, problems, main=True):
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(KEY[1:]).apply(lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))), include_groups=False)
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: units with the wrong arms: {bad[:3]}")
        if len(got) != len(side["prompt_tasks"]):
            problems.append(f"block {side['_job']}: {len(got)} units, expected {len(side['prompt_tasks'])}")
        if side.get("stop_rule") != "r8":
            problems.append(f"block {side['_job']}: stop rule {side.get('stop_rule')}, expected r8")
        if side.get("stage") != "1h" or side.get("amend") != L.AMEND or side.get("amend_1h") != L.AMEND_1H:
            problems.append(f"block {side['_job']}: not a Stage 1h block (stage {side.get('stage')}, amend "
                            f"{side.get('amend')}, amend_1h {side.get('amend_1h')})")
        if [a for a, _ in side["plan"]][-1] != L.NOISE:
            problems.append(f"block {side['_job']}: fp_noise is not the last arm")
    if d.duplicated(KEY + ["arm", "B"]).any():
        problems.append("duplicate rows")
    if d.corpus_sha.nunique() != 1:
        problems.append(f"blocks disagree on corpus: {sorted(d.corpus_sha.unique())}")
    pas = d.arm.map(L.parse_arm)
    fam = pas.map(lambda p: p["family"])
    m = fam.isin(("dense", "kivi", "kvquant", "fp8kv", "fpnoise")).to_numpy()
    x = d[m]
    width = np.where(fam[m] == "dense", x.B.to_numpy(dtype=float),
                     pas[m].map(lambda p: float(p["store"] or 16)).to_numpy())
    if len(x) and not (np.allclose(x.bits_per_token, width) and (x.evict_frac == 0).all()):
        problems.append("a dense-view row (dense, KIVI, KVQuant, FP8, fp_noise) is not its width with nothing evicted")
    k = np.array([L1C.qread_keep_count(r, c) / c if 0 < r <= 1 else 1.0 for r, c in zip(d.B, d.ctx_len)])
    for fname in ("qread", "qreadfp", "qread2t", "qoracle", "qoraclefp"):
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
            problems.append(f"a {fname} row does not store its width dense and read floor(r C) per KV head")
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
        m = t.merge(b, on=KEY + ["B"], suffixes=("", "_b"))
        if len(m) != len(t) or not (m.tf_sum_nll - m.tf_sum_nll_b).abs().mean() > 0:
            problems.append(f"{arm}: missing base rows or values unchanged")
    gen = d[d.fp_gen_len > 0]
    fpk = gen[gen.arm == "fp"]
    if "kl_all" not in d or not (fpk.kl_all == 0).all():
        problems.append("KL: an FP row's KL to itself is not 0 (or the KL columns are missing)")
    elif not (np.isfinite(gen[gen.arm != "fp"].kl_all).all() and (gen[gen.arm != "fp"].kl_all >= 0).all()):
        problems.append("KL: an arm's kl_all is not finite and >= 0")
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


def validate_a2_h(d, problems, name_, self_check=True):
    """Stage 1f's A2 checks, with the self-check required per arm NAME and block."""
    RF.validate_a2(d, problems, name_, self_check=False)
    if not self_check or "a2_check" not in d:
        return
    arms = d[d.arm != "fp"]
    ck = arms.dropna(subset=["a2_check"])
    for job, g in arms.groupby("job"):
        miss = sorted(set(g.arm) - set(ck[ck.job == job].arm))
        if miss:
            problems.append(f"{name_}: block {job} has no self-check replay (a2_check) for {miss} (A2)")
    bad = ck[ck.a2_check > L.A2_CHECK_TOL]
    if len(bad):
        problems.append(f"{name_}: a replay through the arm's replay path differs from the first by up to "
                        f"{bad.a2_check.max():.3f} nats ({sorted(set(bad.arm))}) (A2)")


def noise_degenerate(d) -> tuple:
    """(degenerate, units compared): fp_noise's per-token log-probs equal FP's on
    every unit and its KL is 0 everywhere."""
    f = d[d.arm == "fp"].set_index(KEY).tf_logp
    z = d[d.arm == "fp_noise"].set_index(KEY)
    same = []
    for k, lp in z.tf_logp.items():
        a, b = f.get(k), lp
        if a is None or b is None:
            continue
        same.append(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
    kl0 = bool((z.kl_all.fillna(0) == 0).all())
    return bool(same) and all(same) and kl0, len(same)


# ---------------------------------------------------------------- analysis
def add_bytes_h(v):
    """Bytes per decode step, per context token, per KV head and layer (Stage 1g's
    rule with Stage 1h's arms: their own key and value side bits), GPU-stored bytes,
    and two-tier host / fetch bytes."""
    v = v.copy()
    hd = v.head_dim.to_numpy(dtype=float)
    d8 = hd / 8.0
    pas = v.arm.map(L.parse_arm)
    fam = pas.map(lambda p: p["family"]).to_numpy()
    vb = v.v_bits.to_numpy(dtype=float)
    vs = v.v_side.to_numpy(dtype=float)
    tail = ((v.window + v.n_question_tokens + v.gen_len / 2.0) * 4 * v.head_dim / v.ctx_len).to_numpy(dtype=float)
    ks = v.key_side.to_numpy(dtype=float)
    rf = v.read_frac.to_numpy(dtype=float)
    byt = d8 * (v.bits_per_token.to_numpy(dtype=float) + ks) + (1 - v.evict_frac.to_numpy(dtype=float)) * d8 * (vb + vs)
    sk = np.where(np.isin(fam, ("qread", "qoracle")), 16.0 / hd, np.where(np.isin(fam, ("qreadfp", "qoraclefp")), 0.0, ks))
    sto = (d8 * (v.stored_bits_per_token.to_numpy(dtype=float) + sk)
           + (1 - v.stored_evict_frac.to_numpy(dtype=float)) * d8 * (vb + vs))
    host, fetch = np.zeros(len(v)), np.zeros(len(v))
    for i in np.where(fam == "qread2t")[0]:
        p = pas.iloc[i]
        byt[i] = L1G.two_tier_bytes(rf[i], p["tier2_bits"], p["kv"], p["v_bits"], int(hd[i]))["total"]
        sto[i] = d8[i] * (p["store"] + 16.0 / hd[i]) + d8[i] * (vb[i] + vs[i])
        host[i] = L1G.host_bytes(p["tier2_bits"], p["kv"], int(hd[i]))
        fetch[i] = L1G.fetch_bytes(float(v.B.iloc[i]), int(v.ctx_len.iloc[i]), p["tier2_bits"], p["kv"], int(hd[i]))
    v["bytes"], v["bytes_stored"], v["bytes_host"] = byt + tail, sto + tail, host
    v["fetch_head"] = fetch
    v["bytes_amort"] = v.bytes + (v.bytes_stored - tail) / v.fp_gen_len.clip(lower=1)
    return v


def _fp_values(task, fp_pred):
    pat = r"\b[A-Z]{5}\b" if task == "vt" else r"\b\d{5,}\b"
    return re.findall(pat, str(fp_pred))


def failure_type_row(r, task, fp_pred) -> str:
    """LOST's type for one lost unit (module docstring)."""
    if bool(r.get("distractor", False)):
        return "confused"
    h, n = r.get("hits"), r.get("n_expected")
    if h is not None and n is not None and h == h and n == n and 0 < int(h) < int(n):
        return "incomplete"
    t = M.failure_type(r["pred"], _fp_values(task, fp_pred))
    return "incomplete" if t == "incomplete" else ("confused" if t == "confused" else "other")


def _tables(v):
    P = RF._dA(v, KEY, col=PRIMARY)
    S = RF._dA(v, KEY, col=CONT).reindex(index=P.index, columns=P.columns)
    A = RF._dA(v, KEY, col=FROZEN).reindex(index=P.index, columns=P.columns)
    K = RF._unit_table(v, P, KL)
    SC = RF._unit_table(v, P, "score")
    return P, S, A, K, SC


def analyse_r1(d, sides):
    fp_cells = d[d.arm == "fp"].groupby("task").score.mean()
    valid = sorted(t for t, s in fp_cells.items() if s >= FP_MIN)
    v = add_bytes_h(d[d.task.isin(valid)])
    P, S, A, K, SC = _tables(v)
    pidx = P.index.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1) if s.index.nlevels > 2 else s, pidx).to_numpy()  # noqa: E731
    boot = lambda s: L1C.boot_ci(pm(s), W)  # noqa: E731
    cols = ["bytes", "bytes_stored", "bytes_amort", "bytes_host", "fetch_head", "evict_frac", "kept_width",
            "needle_keep", "score", "peak_gib_arm"]
    means = v.groupby(["arm", "B"])[cols].mean()
    preds = v.set_index(KEY + ["arm", "B"]).pred
    unit_rows = v.set_index(KEY + ["arm", "B"])
    fpc = ("fp", 0.0)
    pts = {}
    for c in P.columns:
        if c == fpc or P[c].isna().all():
            continue
        pa = L.parse_arm(c[0])
        Dc = (DREF_H[pa["lens"]], 3.0)
        p = RD._point_stats(P, c, Dc, W, pm, means)
        if "vs_D" in p:
            p["vs_D"]["D"] = name(Dc)
        nll, tail = boot(P[c]), boot((P[c] > TAIL_NATS).astype(float))
        p.update(arm=c[0], B=float(c[1]), lens=pa["lens"], family=pa["family"], score=float(means.loc[c, "score"]),
                 peak_gib_arm=float(means.loc[c, "peak_gib_arm"]) if "peak_gib_arm" in means else float("nan"),
                 dP_by_task={t: float(g.mean()) for t, g in P[c].groupby(level=-1)})
        p["vs_FP"] = dict(nll=nll, tail=tail, label=L.near_fp_label(nll, tail, MARGIN_R1),
                          equiv=L.equiv_label(nll, MARGIN_R1))
        kk = K[c].astype(float)
        p["kl"] = dict(span=boot(kk), q95=float(np.nanpercentile(kk, 95)))
        p["dS"] = dict(vs_FP=boot(S[c]), label=L.near_fp_label(boot(S[c]), boot((S[c] > TAIL_NATS).astype(float))),
                       min_bias=float((S[c] - P[c]).mean()))
        if Dc in S.columns and c != Dc:
            zs = boot(S[c] - S[Dc])
            p["dS"]["vs_D"] = dict(nll=zs, label=R1C.matched_label(zs, boot((S[c] > TAIL_NATS).astype(float)
                                                                           - (S[Dc] > TAIL_NATS).astype(float))))
        p["dA_frozen"] = boot(A[c])
        lost, gain = M.lost_mask(SC[fpc], SC[c]), SC[c].to_numpy() > SC[fpc].to_numpy() + 1e-9
        types = Counter(failure_type_row(dict(unit_rows.loc[k + c]), k[2], preds.get(k + fpc, ""))
                        for k in P.index[lost])
        p["lost"] = dict(n=int(lost.sum()), n_gain=int(gain.sum()), mcnemar=M.mcnemar_exact(lost, gain),
                         types=dict(types))
        pts[name(c)] = p
    res = dict(cell="r1", ctx=int(sides[0]["ctx"]), fp=float(fp_cells.mean()), fp_by_task=fp_cells.round(4).to_dict(),
               valid_tasks=valid, n_prompts=int(len(pidx)), n_units=int(len(P)), points=pts, labels={},
               dropped_no_value=int(d[(d.arm == "fp") & d.task.isin(valid)][PRIMARY].isna().sum()))

    def diff(a, b, lab, eq=None):
        if a not in P.columns or b not in P.columns:
            return
        z = boot(P[a] - P[b])
        res["labels"][lab] = dict(diff=z, label=L.effect_label(*z, lab), a=name(a), b=name(b),
                                  equiv=(L.equiv_label(z, eq) if eq else None))

    e = 0.125
    diff(("qreadfp_v16", e), ("qoraclefp_v16", e), "VOTE_LOSS_EXACT", VOTE_EQ_EPS)
    diff(("qread4_v4", e), ("qoracle4_v4", e), "VOTE_LOSS_4", VOTE_EQ_EPS)
    diff(("qread2t4q_v4", e), ("qread2t4kq_v4", e), "EXACT_KV")
    nz = ("fp_noise", 0.0)
    if nz in P.columns:
        deg, n_cmp = noise_degenerate(v)
        ap = P[nz].abs()
        res["noise"] = dict(mean_abs_dP=float(ap.mean()), q95_abs_dP=float(np.percentile(ap, 95)),
                            dP=boot(P[nz]), dK=boot(K[nz].astype(float)),
                            q95_dK=float(np.nanpercentile(K[nz].astype(float), 95)), lost=pts[name(nz)]["lost"],
                            degenerate=deg, units_compared=n_cmp,
                            label="NOISE_DEGENERATE" if deg else "NOISE_MEASURED")
    f8 = ("fp8kv", 8.0)
    if f8 in P.columns:
        z = pts[name(f8)]
        res["fp8"] = dict(dP=z["vs_FP"]["nll"], dK=z["kl"]["span"], lost=z["lost"], label=z["vs_FP"]["label"])
        res["m_fp"] = L.margin_fp(z["vs_FP"]["nll"][2])
    dm = {name(c): float(P[c].mean()) for c in DENSE4 if c in P.columns}
    if dm:
        res["best_dense"] = dict(means=dm, best=L.best_dense(dm))
    if SYSTEM in P.columns and DV4 in P.columns:
        ls, ld = M.lost_mask(SC[fpc], SC[SYSTEM]), M.lost_mask(SC[fpc], SC[DV4])
        res["system_vs_D"] = dict(point=name(SYSTEM), mcnemar=M.mcnemar_exact(ls, ld), lost=int(ls.sum()),
                                  lost_D=int(ld.sum()))
    # FP-FAILED stratum, over every task (also those dropped for FP < 0.9)
    va = d.copy()
    Pa = RF._dA(va, KEY, col=PRIMARY)
    SCa = RF._unit_table(va, Pa, "score")
    failed = SCa[fpc] < 1.0
    res["fp_failed"] = dict(n_units=int(failed.sum()), of=int(len(Pa)),
                            by_task={t: int(g.sum()) for t, g in failed.groupby(level=-1)},
                            mean_dP={name(c): float(Pa.loc[failed, c].mean()) for c in Pa.columns
                                     if c != fpc and failed.any()})
    return res


def analyse_bridge(v1, root=RESULTS, s1g=S1G_BRIDGE, problems=None):
    """BRIDGE (module docstring): R1 minus Stage 1g, per unit, per bridged arm."""
    problems = [] if problems is None else problems
    g = pd.concat([load_run(s1g[0], j, root, stage="s1g")[0] for j in s1g[1]], ignore_index=True)
    P1 = RF._dA(v1, KEY, col=PRIMARY)
    Pg = RF._dA(g, KEY, col=PRIMARY)
    i1 = P1.index.droplevel("job")
    ig = Pg.index.droplevel("job")
    n1 = v1[v1.arm == "fp"].set_index(["prompt_idx", "task"]).n_context_tokens
    ng = g[g.arm == "fp"].set_index(["prompt_idx", "task"]).n_context_tokens
    sg = set(ig)
    common = [k for k in i1 if k in sg]
    if not common:
        problems.append("bridge: no unit of R1 is in Stage 1g's blocks")
        return {}
    if any(int(n1.get(k, -1)) != int(ng.get(k, -2)) for k in common):
        problems.append("bridge: a shared unit has a different context length in R1 and Stage 1g (not the same prompt)")
        return {}
    sc_ = set(common)
    keep = P1.index[[k in sc_ for k in i1]]
    pidx = keep.droplevel("task").unique()
    W = BM.boot_weights(pidx)
    pm = lambda s: R1C.prompt_means(s.droplevel(-1), pidx).to_numpy()  # noqa: E731
    Pg2 = Pg.copy()
    Pg2.index = ig
    diffs, rows = {}, {}
    for arm, B in L.BRIDGE_ARMS:
        c = (arm, float(B))
        if c not in P1.columns or c not in Pg2.columns:
            continue
        x1 = P1.loc[keep, c]
        xg = Pg2.loc[[k for k in keep.droplevel("job")], c].to_numpy()
        dd = pd.Series(x1.to_numpy() - xg, index=keep)
        diffs[name(c)] = L1C.boot_ci(pm(dd), W)
        rows[name(c)] = dict(r1=float(x1.mean()), s1g=float(np.nanmean(xg)), diff=diffs[name(c)])
    lab, bad = L.bridge_label(diffs)
    return dict(label=lab, drifting=bad, arms=rows, n_units=len(keep), s1g_jobs=list(s1g[1]))


def analyse_seeds(rs):
    """CONFUSION across rotation seeds (module docstring)."""
    out, table = {}, {}
    per_seed = {}
    for seed, g in rs.groupby("rot_seed"):
        Ps = g.pivot_table(index=["prompt_idx", "task"], columns=["arm", "B"], values=PRIMARY, aggfunc="first")
        Ps = Ps[~Ps[("fp", 0.0)].isna()]
        per_seed[int(seed)] = Ps.sub(Ps[("fp", 0.0)], axis=0)
    units = sorted(set().union(*[set(x.index) for x in per_seed.values()]))
    for u in units:
        d_by, ok_by = {}, {}
        for s, Ps in per_seed.items():
            if u not in Ps.index or DV4 not in Ps.columns:
                continue
            d_by[s] = float(Ps.loc[u, DV4])
            oks = [float(Ps.loc[u, c]) for c in CONF_OK_ARMS if c in Ps.columns]
            ok_by[s] = max(oks) if oks else float("inf")
        key = f"{u[0]}/{u[1]}"
        out[key] = L.confusion_seeds_label(d_by, ok_by)
        table[key] = {name(c): {s: round(float(Ps.loc[u, c]), 2) for s, Ps in per_seed.items()
                                if u in Ps.index and c in Ps.columns}
                      for c in [("uniform", 3.0), DV4, ("qread_v4", 0.125)] + CONF_OK_ARMS}
    return dict(labels=out, dP=table, seeds=sorted(per_seed))


# ------------------------------------------------------------------ report
def _arm_table(points):
    Lh = ["| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | "
          "MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |",
          "|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|"]
    for k, p in sorted(points.items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
        z = p.get("vs_D")
        lo = p["lost"]
        Lh.append(f"| {k} | {p['lens']} | {fmt(z and z['rho'], 2)} | {fmt(z and z['rho_mem'], 2)} | "
                  f"{ci(p['vs_FP']['nll'])} | {p['vs_FP']['label']} | {p['vs_FP']['equiv']} | {ci(p['kl']['span'])} | "
                  f"{ci(z['nll']) if z else '(D)'} | {z['label'] if z else '—'} | {lo['n']} / {lo['n_gain']} "
                  f"({lo['mcnemar']['p']:.3f}) | {lo['types'] or '—'} | {ci(p['dS']['vs_FP'])} | "
                  f"{p['dS']['min_bias']:+.3f} | {p['score']:.3f} | {fmt(p['peak_gib_arm'], 1)} |")
    return Lh


def report(res, bridge, seeds, summary, out_stem):
    Lh = ["# R14 Stage 1h — R1, calibration and bridge (read_stage1h.py; rules frozen in its docstring)", "",
          "EXPLORATORY run: labels guide the design; they are not claims.", ""]
    Lh += [f"- **{k}**: {v}" for k, v in summary.items()] + [""]
    Lh += [f"## r1 — FP {res['fp']:.3f} ({res['fp_by_task']}), {res['n_prompts']} prompts, {res['n_units']} units; "
           f"primary dP = a_span_nll, co-primary KL span", ""]
    Lh += _arm_table(res["points"]) + [""]
    for lab, z in res["labels"].items():
        Lh.append(f"**{lab}** ({z['a']} − {z['b']}): {ci(z['diff'])} → {z['label']}"
                  + (f", {z['equiv']} within ±{VOTE_EQ_EPS}" if z.get("equiv") else ""))
    if res.get("noise"):
        n = res["noise"]
        Lh.append(f"**NOISE** ({n['label']}, {n['units_compared']} units): |dP| mean {n['mean_abs_dP']:.4f}, 95th "
                  f"percentile {n['q95_abs_dP']:.4f}; dP {ci(n['dP'])}; KL span {ci(n['dK'])}, 95th percentile "
                  f"{n['q95_dK']:.4f}; lost {n['lost']['n']}")
    if res.get("fp8"):
        f = res["fp8"]
        Lh.append(f"**FP8_COST**: dP {ci(f['dP'])} ({f['label']}); KL span {ci(f['dK'])}; lost {f['lost']['n']}. "
                  f"**M_FP** (from R2 on) = {res['m_fp']:.3f}")
    if res.get("best_dense"):
        Lh.append(f"**BEST_DENSE**: {res['best_dense']['best']} (mean dP {res['best_dense']['means']})")
    if res.get("system_vs_D"):
        s = res["system_vs_D"]
        Lh.append(f"**SYSTEM vs D_V4 (lost answers)**: {s['lost']} vs {s['lost_D']}, McNemar p {s['mcnemar']['p']:.3f}")
    ff = res["fp_failed"]
    Lh += ["", f"**FP-FAILED stratum**: {ff['n_units']} of {ff['of']} units ({ff['by_task']}); mean dP there: "
           + ", ".join(f"{k} {v:+.2f}" for k, v in ff["mean_dP"].items() if v == v), ""]
    if bridge:
        Lh += [f"## Bridge to Stage 1g ({bridge['n_units']} units; jobs {bridge['s1g_jobs']}): **{bridge['label']}**"
               + (f" — drifting: {bridge['drifting']}" if bridge["drifting"] else ""), "",
               "| arm@B | dP R1 | dP Stage 1g | R1 − 1g [90%] |", "|---|---:|---:|---|"]
        Lh += [f"| {k} | {z['r1']:+.3f} | {z['s1g']:+.3f} | {ci(z['diff'])} |" for k, z in bridge["arms"].items()]
        Lh.append("")
    if seeds:
        Lh += [f"## Key confusions across rotation seeds {seeds['seeds']} (dP vs FP)", ""]
        Lh += [f"- {u}: **{lab}**" for u, lab in seeds["labels"].items()] + [""]
        for u, tab in seeds["dP"].items():
            Lh.append(f"- {u}: " + "; ".join(f"{a} {v}" for a, v in tab.items() if v))
        Lh.append("")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def read_r1(jobs, seed_jobs, out_stem, root=RESULTS, s1g=S1G_BRIDGE):
    problems, summary = [], {}
    parts = [load_run("h1cal", j, root) for j in jobs]
    d = pd.concat([p[0] for p in parts], ignore_index=True)
    sides = [p[1] for p in parts]
    if len({json.dumps(s["plan"]) for s in sides}) != 1:
        problems.append("r1: blocks ran different plans")
    agree, cover = validate_h(d, sides, problems)
    validate_a2_h(d, problems, "r1")
    rs = None
    if seed_jobs:
        sp = [load_run("h1regress", j, root) for j in seed_jobs]
        rs = pd.concat([p[0] for p in sp], ignore_index=True)
        validate_h(rs, [p[1] for p in sp], problems, main=False)
        validate_a2_h(rs, problems, "r1seeds", self_check=False)
        seeds_ = [int(p[1].get("rot_seed", -1)) for p in sp]
        if len(set(seeds_)) != len(seeds_):
            problems.append(f"r1seeds: rotation seeds are not distinct: {seeds_}")
        if any(plan_of(p[1]) != plan_of(sides[0]) for p in sp):
            problems.append("r1seeds ran another plan than r1")
    if problems:
        raise SystemExit("INVALID Stage 1h R1 data:\n  " + "\n  ".join(problems))
    res = analyse_r1(d, sides)
    res["fp_replay_agreement"], res["value_cover"] = agree, cover
    bridge = analyse_bridge(d[d.task.isin(res["valid_tasks"])], root, s1g, problems)
    if problems:
        raise SystemExit("INVALID Stage 1h R1 bridge:\n  " + "\n  ".join(problems))
    seeds = analyse_seeds(rs) if rs is not None else None
    for lab, z in res["labels"].items():
        summary[lab] = z["label"] + (f", {z['equiv']}" if z.get("equiv") else "")
    if res.get("noise"):
        summary["NOISE"] = (f"{res['noise']['label']}: |dP| 95th pct {res['noise']['q95_abs_dP']:.4f}, "
                            f"KL span 95th pct {res['noise']['q95_dK']:.4f}")
    if res.get("fp8"):
        summary["FP8_COST"] = f"dP {ci(res['fp8']['dP'])}, {res['fp8']['label']}"
        summary["M_FP (R2 on)"] = f"{res['m_fp']:.3f}"
    if res.get("best_dense"):
        summary["BEST_DENSE"] = res["best_dense"]["best"]
    sk = name(SYSTEM)
    if sk in res["points"]:
        p = res["points"][sk]
        summary["SYSTEM"] = (f"vs FP {p['vs_FP']['label']} ({ci(p['vs_FP']['nll'])}), {p['vs_FP']['equiv']}; vs D_V4 "
                             f"{p['vs_D']['label']}; lost {p['lost']['n']}")
    summary["BRIDGE"] = bridge.get("label", "NO_DATA") + (f" {bridge['drifting']}" if bridge.get("drifting") else "")
    if seeds:
        for u, lab in seeds["labels"].items():
            summary[f"CONFUSION {u}"] = lab
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(dict(rules=dict(margin_r1=MARGIN_R1, tail_margin=TAIL_MARGIN, tail_nats=TAIL_NATS,
                                  effect_eps=EFFECT_EPS, vote_equiv_eps=VOTE_EQ_EPS, bridge_eps=L.BRIDGE_EPS,
                                  margin_floor=L.MARGIN_FLOOR, margin_cap=L.MARGIN_CAP, primary=PRIMARY, kl=KL,
                                  a2_check_tol=L.A2_CHECK_TOL, boot_reps=BM.BOOT_REPS, boot_seed=BM.BOOT_SEED),
                       summary=summary, r1=res, bridge=bridge, seeds=seeds), fh, indent=1, default=str)
    Lh = report(res, bridge, seeds, summary, out_stem)
    print("\n".join(Lh[:4 + len(summary) + 1]))
    return 0


# --------------------------------------------------------------------- gate
def gate(job, tag="h1pilot", root=RESULTS):
    d, side = load_run(tag, job, root)
    problems = []
    agree, cover = validate_h(d, [side], problems, main=False)
    multi = d[(d.arm == "fp") & d.task.isin(L.MULTI_ANSWER)].fp_ans_order.fillna("").str.contains(",")
    validate_a2_h(d, problems, f"pilot {job}", self_check=bool(multi.any()))
    deg, n_cmp = noise_degenerate(d)
    if deg:
        problems.append(f"NOISE_DEGENERATE: fp_noise equals FP on all {n_cmp} units (amend the noise arm before R1)")
    pk = side.get("peak_gib_dev_max") or []
    if not pk or not all(x <= PEAK_GIB_MAX for x in pk):
        problems.append(f"peak GPU memory per device {pk} GiB (limit {PEAK_GIB_MAX})")
    main = L.PILOT_OF[tag]
    per_unit = (d.groupby(KEY).apply(lambda g: float((g.t_arm + g.t_tf + g.t_own.fillna(0)).sum()), include_groups=False)
                + float(d.t_prefill.median()) + float(d.t_precompute.max()))
    unit_s = float(per_unit.median())
    proj_h = unit_s * BLOCK_UNITS[main] / 3600
    if proj_h > 0.9 * WALL_H[main]:
        problems.append(f"projected {main} block {proj_h:.1f} h > 90% of {WALL_H[main]} h")
    ck = d.dropna(subset=["a2_check"]) if "a2_check" in d else d.iloc[:0]
    print(f"R14 Stage 1h pilot {job} ({tag} for {main}): {len(d)} rows, plan {len(plan_of(side))} arms, fp replay "
          f"{agree:.3f}, answer-value coverage {cover:.2f}, peak per GPU {pk} GiB, unit {unit_s:.0f} s, projected "
          f"{main} block {proj_h:.1f} h (wall {WALL_H[main]} h); noise {'DEGENERATE' if deg else 'measured'} on "
          f"{n_cmp} units; self-checks {len(ck)}, largest {ck.a2_check.max() if len(ck) else float('nan'):.2e} nats")
    cols = [c for c in ("score", PRIMARY, KL, "kl_all", "bits_per_token", "read_frac", "t_arm", "t_tf", "peak_gib_arm")
            if c in d]
    print(d.pivot_table(index=["arm", "B"], values=cols, aggfunc="mean").round(4).to_string())
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
    ap.add_argument("--pilot-tag", default="h1pilot", choices=sorted(L.PILOT_OF))
    ap.add_argument("--r1", nargs="+", metavar="JOB", default=[])
    ap.add_argument("--r1-seeds", nargs="*", metavar="JOB", default=[])
    ap.add_argument("--out-stem", default=os.path.join(HERE, "findings", "R1_reader"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.pilot:
        sys.exit(gate(a.pilot, a.pilot_tag, a.results_root))
    if not a.r1:
        ap.error("give --pilot JOB or --r1 JOBS")
    sys.exit(read_r1(a.r1, a.r1_seeds, a.out_stem, a.results_root))


if __name__ == "__main__":
    main()
