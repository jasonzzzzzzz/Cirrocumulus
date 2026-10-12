#!/usr/bin/env python3
"""R7b's reader and gate (plan_r7b.md). Written 2026-10-12 before any R7b output existed; the rules freeze with the run
(lib_r7b.AMEND_R7B = 'R7b'). The primary test, its calibration, the group verdict and the key secondary are R6's
(../../14h_methodology_improve/read_stage1h_r6.py, frozen 2026-10-10), copied here with the design replaced; the gates and
the reports are R7b's additions (theory_r7b.md §6.3).

    python read_r7b.py --r7b TAG:JOB [TAG:JOB ...] [--out-stem findings/R7b_reader]
    python read_r7b.py --gate TAG:JOB               (a group's pilot, lib_r7b.R7B_PILOTS; findings/R7b_gate_<group>.json)
    python read_r7b.py --explore TAG:JOB [...]      (the exploratory blocks; findings/R7b_explore.{json,md})

CLAIM (primary, D7). For each claim group (Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K, Qwen3-30B-A3B at 128K), the single
  store dnest42ux_v4 at 1/8 (keys a 4-bit Lloyd-Max base plus a 2-bit dithered residual, values 4-bit dithered; the vote
  over the store itself; the vote's rows exact, the rest from the store, unbiased; traffic 0.398 of FP16 K+V; memory 0.32
  at rest, 0.45 during a question) is non-inferior to FP8 KV (fp8kv, 0.5) on KL per span token.
UNITS. Prompt x task, each in exactly one block; families as R6 (read_stage1h_r5.FAMILY): retrieval (niah_single,
  niah_multikey, mk_panel), multivalue, tracking (vt), aggregation (cwe, fwe), rag (kilt_nq, kilt_hotpotqa), rerank
  (msmarco_rerank_psg), icl (icl_trec_coarse, icl_banking77). Planned (lib_r7b.R7B_LAYOUT): Llama 128K retrieval 60,
  multivalue 30, tracking 30, aggregation 30, rag 60, icl 60, rerank 30; Qwen 32K retrieval 80, multivalue 300, tracking
  40, aggregation 30; Qwen 128K rag 60, icl 60, rerank 30 (+ rag 40 and icl 60 with R7B_EXTRA_LAYOUT). Re-ranking units are
  R6's items 11-40 (no fresh msmarco query exists).
PRIMARY TEST (R6's), per claim group x family. y_u = KL_tok(design) - 1.5 KL_tok(FP8) per unit (both from the unit's own
  process; units with a non-finite value in either arm are dropped and counted). U = the 95th percentile of 2,000
  unit-bootstrap means of y (resampling units with replacement; numpy default_rng(0)): a one-sided 95% upper bound. PASS if
  U < 5e-4, else FAIL.
CALIBRATION (R6's). The same test on fp_noise (FP rerun) in the same group and family: if fp_noise's U >= 5e-4 the family
  is NOT_TESTABLE at this precision.
GROUP (R6's). NONINFERIOR if every testable family is PASS and at least half of the group's planned families are
  testable; NOT_SHOWN if any testable family is FAIL; INSUFFICIENT otherwise. Each group is its own claim.
KEY SECONDARY (R6's accuracy), per claim group: the FP-correct units (FP's score >= 1 - 1e-9) of every family but
  aggregation; d_u = score(design) - score(FP8); L = the 5th percentile of 2,000 unit-bootstrap means. At delta = 0.02:
  ACC_PASS if L > -delta and n >= 3 / delta (= 150), ACC_FAIL if L <= -delta, else ACC_UNDERPOWERED. Reported at 0.05.
GATES (D3; R7b's addition), per claim group and baseline b: d_u = KL_tok(design) - KL_tok(b) over the group's units (every
  family); p_b = (#{bootstrap means of d >= 0} + 1) / (2,001) (one-sided, the 2,000 unit-bootstrap means as above); Holm's
  step-down at alpha = 0.05 over the group's TESTED baselines: BEATS where H0 (mean d >= 0) is rejected, else NOT_SHOWN.
  Tested baselines: KIVI-4 (kivi4_v4@4), dense Lloyd-Max 4/4 (uniform+v4@4, TurboQuant-MSE), Quest at the design's
  traffic (quest_v16@0.3359375) and dense dithered 7/6 (ddense7u_v6@7, 0.406) in every group (GATE_DENSE_GROUPS;
  plan_r7b.md §8 decision 3; theory_r7b.md §6.3, §5.4b: Corollary G, full-depth totals).
REPORTED, NOT TESTED: E4, R6's design dtail6ux_v4 in the same process: KL(design) / KL(E4) per group and family (the
  ratio of mean KLs with a 95% unit-bootstrap interval, and the geometric mean of per-unit ratios); the primary
  statistic for E4, the gate baselines and uniform@4; the accuracy secondary for every arm; per group x family x arm,
  KL/tok, dP/tok and accuracy changes with 90% intervals, traffic.
EXPLORATORY (--explore; theory_r7b.md §7 P-b7-P-b9): E1 cnest42ux_v4_t1 / _t2 (teacher-forced): KL/tok against the
  design's on the same units, traffic (cr_traffic), the per-query-head misses against delta (COVER_OK if misses <= delta x
  checks + 3 sqrt(delta checks) + 3), the exact-read fallback share; E2 dnest43ux_v4 / design; E3 wnest_v4 / dnest41ux_v4
  (equal traffic). The calibration outputs are not in the code pin: a block whose outputs changed while it ran is
  EXCLUDED, and a group's blocks must have read the same outputs (data_identical; a warning otherwise).
VALIDITY (per block; a failing block is EXCLUDED and reported, the rest are read): an R7b block (sidecar r7b.amend =
  AMEND_R7B) whose tag, preset, suite and tasks are a row of the layout; its plan is the preset's (lib_r7b.build_plan);
  every planned arm on every unit and nothing else; its units are R7b's planned units for the tag; the dnest, dtail and
  ddense arms evict nothing; every dithered arm's dither_seed is lib_r7b.unit_seed(task, prompt); peak GPU memory <= 76 GiB
  per device; HELMET blocks carry R7b's manifest (tasks_r7b.MANIFEST_SHA256); every block's code manifest is the same. A
  unit in two valid blocks is read from the first (sorted by job) and reported.
GATE (per claim group: the group's pilot, lib_r7b.R7B_PILOTS, run by the pilot's own job; excluded): validity in pilot
  mode; FP's KL to itself 0; fp_noise not degenerate; peak <= 76 GiB per device; every core (and R7B_EXTRA) block of the
  group projected within 90% of its wall (gate_projection): per task, its prompts per process x the pilot's seconds per
  unit x GATE_UNIT_S[group][task] / GATE_UNIT_S[group][the pilot's task], x GATE_SLACK, + 15 min. GATE_UNIT_S is R6's
  measured time per unit with R7b's core arms (explore_r7b.py cost): a task's answer length sets its time, which varies 5x
  across tasks. Writes findings/R7b_gate_<group>.json (with the gate's job and the code pin's sha256) and exits 1 on
  FAIL, so the group's blocks (afterok on the pilot's job) never start.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)
METH = os.path.join(os.path.dirname(R7), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _p in (R7, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import lib_r7b as LB  # noqa: E402
import read_stage1h_r5 as R5  # noqa: E402  (FAMILY, arm_table, _ci; read-only)

H0 = os.path.dirname(os.path.dirname(R7))
RESULTS = os.path.join(H0, "results")
FINDINGS = os.path.join(HERE, "findings")
DESIGN = (LB.DESIGN[0], float(LB.DESIGN[1]))
E4 = (LB.E4_ARM[0], float(LB.E4_ARM[1]))
REF, NOISE = ("fp8kv", 8.0), ("fp_noise", 0.0)
RHO, KAPPA = LB.R6_MARGIN
N_BOOT, SEED, ALPHA = 2000, 0, 0.05
ACC_DELTAS, ACC_KEY, ACC_EXCLUDE = (0.02, 0.05), 0.02, ("aggregation",)
PEAK_LIMIT = 76.0
GATE_ARMS = [(a, float(b)) for a, b in LB.GATES]
GATE_DENSE_GROUPS = ("llama128", "qwen32", "qwen128")      # every group (plan_r7b.md §8 decision 3)
REPORTED = [E4] + GATE_ARMS + [("uniform", 4.0)]
GATE_SETUP_H, GATE_WALL_FRAC, GATE_SLACK = 0.25, 0.9, 1.1        # R6's blocks: 3-11 min setup; elapsed / unit times 1.1
GATE_UNIT_S = {                                                     # findings/cost_r7b.json "unit_s" (seconds per unit)
    "llama128": dict(cwe=298, fwe=149, icl_banking77=70, icl_trec_coarse=68, kilt_hotpotqa=99, kilt_nq=107, mk_panel=86,
                     msmarco_rerank_psg=331, niah_multikey=89, niah_multivalue=247, niah_single=89, vt=261),
    "qwen32": dict(cwe=67, fwe=47, mk_panel=46, niah_multikey=44, niah_multivalue=177, niah_single=43, vt=75),
    "qwen128": dict(icl_banking77=91, icl_trec_coarse=81, kilt_hotpotqa=123, kilt_nq=121, msmarco_rerank_psg=280)}
FAMILY = R5.FAMILY
UKEY = ["prompt_idx", "task"]
SEEDED = ("dnest", "wnest", "ddense", "cnest", "kcal", "dtail")


def planned_families(layout=None):
    out = {}
    for _n, tag, preset, *_r in (LB.R7B_LAYOUT if layout is None else layout):
        g = LB.R7B_GROUP_OF[preset]
        for t in _r[3].split(","):
            out.setdefault(g, set()).add(FAMILY[t])
    return {g: sorted(v) for g, v in out.items()}


def _boot_means(y):
    y = np.asarray(y, float)
    return y[np.random.default_rng(SEED).integers(0, len(y), (N_BOOT, len(y)))].mean(1)


def upper(y):
    """One-sided 95% unit-bootstrap upper bound of the mean (and the mean, n)."""
    y = np.asarray(y, float)
    return float(y.mean()), float(np.quantile(_boot_means(y), 1 - ALPHA)), int(len(y))


def lower(y):
    y = np.asarray(y, float)
    return float(y.mean()), float(np.quantile(_boot_means(y), ALPHA)), int(len(y))


# ------------------------------------------------------------------ loading and validity
def _layout_row(tag, preset, suite, tasks, layout):
    for row in layout:
        if row[1] == tag and row[2] == preset and row[4] == suite and row[6] == tasks:
            return row
    return None


def load_block(tag, job, root=RESULTS):
    d = os.path.join(root, f"r14r7b_{tag}_{job}")
    sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(sp) != 1:
        return None, None
    side = json.load(open(sp[0]))
    return side, pd.read_parquet(os.path.join(d, side["parquet"]))


def validity(tag, side, d, problems, layout=None, pilot=False, smoke=False):
    r7b = side.get("r7b") or {}
    preset = side.get("preset_name")
    amend_ok = r7b.get("amend") == LB.AMEND_R7B or (smoke and str(r7b.get("amend", "")).startswith("R7b"))
    if not amend_ok or preset not in LB.R7B_PRESETS or (preset.endswith("smoke") and not smoke):
        problems.append(f"not an R7b block (r7b.amend {r7b.get('amend')!r}, preset {preset!r})")
        return
    layout = LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT if layout is None else layout
    tasks = ",".join(side.get("tasks") or [])
    if not smoke and _layout_row(tag, preset, side.get("suite"), tasks, LB.R7B_PILOTS if pilot else layout) is None:
        problems.append(f"({tag}, {preset}, {side.get('suite')}, {tasks}) is not a row of the {'pilots' if pilot else 'layout'}")
    plan = [tuple(p) for p in side.get("plan", [])]
    want = [tuple(p) for p in LB.build_plan(LB.PRESETS[preset])]
    if [(a, float(b)) for a, b in plan] != [(a, float(b)) for a, b in want]:
        problems.append(f"plan {plan} is not the preset's {want}")
    units = d[d.arm == "fp"][UKEY].drop_duplicates()
    for a, b in want:
        n = d[(d.arm == a) & np.isclose(d.B.astype(float), float(b))][UKEY].drop_duplicates().shape[0]
        if n != len(units):
            problems.append(f"{a}@{b:g} on {n} of {len(units)} units")
    if len(d) != len(units) * len(want):
        problems.append(f"{len(d)} rows for {len(units)} units x {len(want)} arms")
    if not (pilot or smoke):
        pu = LB.r7b_units(layout)
        bad = [(int(p), t) for p, t in units.itertuples(index=False) if int(p) not in pu.get((tag, t), set())]
        if bad:
            problems.append(f"units outside R7b's plan for {tag}: {bad[:5]}")
    fam = d.arm.map(lambda a: LB.family(a))
    tier = d[fam.isin(["dnest", "wnest", "dtail", "ddense"])]
    if len(tier) and float(tier.evict_frac.fillna(0).abs().max()) > 0:
        problems.append("a tier arm evicted rows")
    seeded = d[fam.isin(SEEDED)]
    if len(seeded):
        want_seed = [LB.unit_seed(t, p) for t, p in zip(seeded.task, seeded.prompt_idx)]
        got = seeded["dither_seed"] if "dither_seed" in seeded else pd.Series([None] * len(seeded))
        if [int(x) if x == x and x is not None else None for x in got] != want_seed:
            problems.append("a dithered arm did not use the unit's seed (lib_r7b.unit_seed)")
    peaks = [float(v) for v in side.get("peak_gib_dev_max") or []]
    if int(side.get("n_gpus") or 0) > 0 and (not peaks or max(peaks) > PEAK_LIMIT):
        problems.append(f"peak GPU memory per device {peaks}")
    if side.get("suite") == "r4":
        import tasks_r7b as T7B
        if side.get("manifest_sha256") != T7B.MANIFEST_SHA256 or r7b.get("manifest_r7b") != T7B.MANIFEST_SHA256:
            problems.append(f"HELMET manifest {str(side.get('manifest_sha256'))[:12]} is not R7b's")


# ------------------------------------------------------------------ the tests
def ni_test(g, arm, B):
    """(mean, U, n, dropped) of y = KL_tok(arm) - RHO KL_tok(FP8) on one group x family's units (R6's)."""
    a = g[(g.arm == arm) & np.isclose(g.B, B)].set_index(["block"] + UKEY).kl_span_tok
    e = g[(g.arm == REF[0])].set_index(["block"] + UKEY).kl_span_tok
    idx = a.index.intersection(e.index)
    y = (a[idx] - RHO * e[idx]).values
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan"), float("nan"), 0, int((~ok).sum())
    m, u, n = upper(y[ok])
    return m, u, n, int((~ok).sum())


def acc_test(rows, arm, B):
    """Accuracy change against FP8 on the FP-correct units outside aggregation (R6's): (mean, L, n) and labels."""
    x = rows[~rows.family.isin(ACC_EXCLUDE)]
    fp = x[x.arm == "fp"].set_index(["block"] + UKEY).score
    a = x[(x.arm == arm) & np.isclose(x.B, B)].set_index(["block"] + UKEY).score
    e = x[x.arm == REF[0]].set_index(["block"] + UKEY).score
    idx = fp.index[fp >= 1 - 1e-9].intersection(a.index).intersection(e.index)
    d = (a[idx] - e[idx]).values
    if not len(d):
        return dict(n=0, mean=float("nan"), lower=float("nan"), labels={str(dl): "ACC_UNDERPOWERED" for dl in ACC_DELTAS})
    m, lo, n = lower(d)
    lab = {str(dl): ("ACC_FAIL" if lo <= -dl else "ACC_PASS" if n >= math.ceil(3 / dl) else "ACC_UNDERPOWERED")
           for dl in ACC_DELTAS}
    return dict(n=n, mean=m, lower=lo, losses=int((d < 0).sum()), gains=int((d > 0).sum()), labels=lab)


def _paired(rows, a, b, col="kl_span_tok"):
    x = rows[(rows.arm == a[0]) & np.isclose(rows.B, a[1])].set_index(["block"] + UKEY)[col]
    y = rows[(rows.arm == b[0]) & np.isclose(rows.B, b[1])].set_index(["block"] + UKEY)[col]
    idx = x.index.intersection(y.index)
    xv, yv = x[idx].values.astype(float), y[idx].values.astype(float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    return xv[ok], yv[ok]


def gate_tests(rg, group):
    """Per baseline: mean d, the one-sided 95% upper bound, the bootstrap p-value; Holm over the tested baselines."""
    res = {}
    for b in GATE_ARMS:
        x, y = _paired(rg, DESIGN, b)
        if not len(x):
            res[f"{b[0]}@{b[1]:g}"] = dict(n=0, label="NO_DATA")
            continue
        d = x - y
        bs = _boot_means(d)
        tested = b[0] != LB.DENSE_GATE[0] or group in GATE_DENSE_GROUPS
        res[f"{b[0]}@{b[1]:g}"] = dict(n=int(len(d)), mean=float(d.mean()), upper=float(np.quantile(bs, 1 - ALPHA)),
                                       p=float((np.sum(bs >= 0) + 1) / (N_BOOT + 1)), design_lower=float(np.mean(d < 0)),
                                       tested=bool(tested), traffic=float(rg[rg.arm == b[0]].traffic_frac.mean()))
    tested = sorted((v["p"], k) for k, v in res.items() if v.get("tested"))
    k = len(tested)
    stop = False
    for i, (p, name) in enumerate(tested):
        rej = (not stop) and p <= ALPHA / (k - i)
        stop = stop or not rej
        res[name].update(holm_level=ALPHA / (k - i), label="BEATS" if rej else "NOT_SHOWN")
    for v in res.values():
        if v.get("n") and not v.get("tested"):
            v["label"] = "REPORTED"
    return res


def ratio_report(rg, a, b):
    """KL(a) / KL(b) on paired units: the ratio of means with a 95% unit-bootstrap interval, the geometric mean of per-unit
    ratios (units where both are > 0), n."""
    x, y = _paired(rg, a, b)
    if not len(x) or y.sum() <= 0:
        return dict(n=int(len(x)))
    idx = np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))
    rb = x[idx].sum(1) / np.maximum(y[idx].sum(1), 1e-300)
    pos = (x > 0) & (y > 0)
    gm = float(np.exp(np.mean(np.log(x[pos] / y[pos])))) if pos.any() else float("nan")
    return dict(n=int(len(x)), ratio_of_means=float(x.sum() / y.sum()), lo=float(np.quantile(rb, ALPHA / 2)),
                hi=float(np.quantile(rb, 1 - ALPHA / 2)), gmean=gm)


def _load_valid(pairs, root, smoke, layout=None, datas=None):
    """Valid blocks' rows; also whether every block's code manifest is the same. datas (a dict, filled in place): each
    block's r7b.data_sha256 (the calibration outputs its arms read, hashed at its start), for the exploratory reader's
    identity check; with datas, a block whose calibration outputs changed while it ran (r7b.data_changed) is EXCLUDED
    (core blocks may see them change, and do not read them)."""
    excluded, parts, seen, dups, manifests = {}, [], set(), [], {}
    for tag, job in sorted(pairs, key=lambda p: (p[0], str(p[1]))):
        w = f"{tag}_{job}"
        side, d = load_block(tag, job, root)
        if side is None:
            excluded[w] = ["cannot load"]
            continue
        pr = []
        validity(tag, side, d, pr, layout=layout, smoke=smoke)
        if datas is not None and (side.get("r7b") or {}).get("data_changed"):
            pr.append("the calibration outputs (data/tau_*, alloc_*) changed while the block ran")
        if pr:
            excluded[w] = pr
            print(f"{w}: EXCLUDED: {'; '.join(pr)}", flush=True)
            continue
        group = LB.R7B_GROUP_OF[side["preset_name"]]
        if group == "smoke":
            group = "llama128"
        keys = set(zip([group] * len(d), d.prompt_idx.astype(int), d.task))
        if keys & seen:
            dups.append(w)
            d = d[[(group, int(p), t) not in seen for p, t in zip(d.prompt_idx, d.task)]]
        seen |= keys
        manifests[w] = (side.get("r7b") or {}).get("code_sha256")
        if datas is not None:
            datas[w] = (side.get("r7b") or {}).get("data_sha256") or {}
        cols = ["prompt_idx", "task", "arm", "B", "a_span_nll_tok", "kl_span_tok", "kl_all", "score", "traffic_frac",
                "exact_rows_frac"] + [c for c in ("cr_traffic", "cr_rows", "cr_fallback", "cr_misses", "cr_checks",
                                                  "cr_met_heads", "traffic_measured") if c in d]
        parts.append(d.assign(group=group, block=w, family=d.task.map(FAMILY))[["group", "block", "family"] + cols])
        print(f"{w}: valid ({group}, {len(d[d.arm == 'fp'])} units)", flush=True)
    same_code = len({json.dumps(m, sort_keys=True) for m in manifests.values()}) <= 1
    return excluded, parts, dups, same_code


def read_r7b(pairs, out_stem, root=RESULTS, smoke=False):
    excluded, parts, dups, same_code = _load_valid(pairs, root, smoke)
    out = dict(rules="read_r7b.py docstring (R6's rules for the primary test; R7b's gates and reports)",
               design=list(DESIGN), margin=[RHO, KAPPA], excluded=excluded, duplicate_units_in=dups,
               code_manifests_identical=same_code, groups={})
    md = ["# R7b, the confirmatory read (rules: read_r7b.py; the primary test is R6's)", "",
          f"Design `{DESIGN[0]}`@{DESIGN[1]:g} (the single store) against FP8 KV: mean KL/token < {RHO} × FP8's + {KAPPA} "
          "per claim group × family (one-sided 95% unit-bootstrap upper bound; fp_noise calibrates). Accuracy: "
          f"FP-correct units outside aggregation, lower bound > −{ACC_KEY}. Gates: Holm over the baselines at {ALPHA}.", ""]
    md += [f"- EXCLUDED: {w}: {'; '.join(p)}" for w, p in excluded.items()] or ["- every block valid"]
    if not same_code:
        md.append("- WARNING: the blocks' code manifests differ")
    if dups:
        md.append(f"- units also in an earlier block, read once: {', '.join(dups)}")
    if not parts:
        _write(out, md, out_stem)
        return 1
    rows = pd.concat(parts, ignore_index=True)
    pf = planned_families()
    for g_name, rg in rows.groupby("group"):
        fams = pf.get(g_name, sorted(rg.family.unique()))
        res = dict(families={}, reported={}, accuracy={}, e4={}, gates=gate_tests(rg, g_name))
        for fam in fams:
            g = rg[rg.family == fam]
            if not len(g):
                res["families"][fam] = dict(label="NO_DATA")
                continue
            m, u, n, drop = ni_test(g, *DESIGN)
            mn, un, nn, _ = ni_test(g, *NOISE)
            testable = bool(np.isfinite(un) and un < KAPPA)
            lab = "NOT_TESTABLE" if not testable else ("PASS" if u < KAPPA else "FAIL")
            res["families"][fam] = dict(label=lab, n=n, dropped=drop, mean=m, upper=u, noise_mean=mn, noise_upper=un,
                                        kl_design=float(g[g.arm == DESIGN[0]].kl_span_tok.mean()),
                                        kl_fp8=float(g[g.arm == REF[0]].kl_span_tok.mean()),
                                        kl_noise=float(g[g.arm == NOISE[0]].kl_span_tok.mean()),
                                        reused_units=bool(set(g.task) & LB.REUSED_TASKS))
            for arm, B in REPORTED:
                ma, ua, na, _ = ni_test(g, arm, B)
                res["reported"].setdefault(f"{arm}@{B:g}", {})[fam] = dict(mean=ma, upper=ua, n=na,
                                                                         below_margin=bool(ua < KAPPA))
            res["e4"][fam] = ratio_report(g, DESIGN, E4)
        res["e4"]["all"] = ratio_report(rg, DESIGN, E4)
        labs = [v["label"] for v in res["families"].values()]
        testable = [x for x in labs if x in ("PASS", "FAIL")]
        if "FAIL" in testable:
            verdict = "NOT_SHOWN"
        elif len(testable) * 2 >= len(fams) and testable:
            verdict = "NONINFERIOR"
        else:
            verdict = "INSUFFICIENT"
        res["verdict"], res["testable"] = verdict, f"{len(testable)} of {len(fams)}"
        for arm, B in [DESIGN] + REPORTED + [NOISE]:
            res["accuracy"][f"{arm}@{B:g}"] = acc_test(rg, arm, B)
        res["arms"] = R5.arm_table(rg.rename(columns={"group": "model"}).assign(cell=rg.group))
        out["groups"][g_name] = res
    _markdown(out, md)
    _write(out, md, out_stem)
    return 0


def _f(x, nd=5):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _markdown(out, md):
    md += ["", "## Verdicts", "", "| claim group | verdict | testable families | accuracy (δ = 0.02) | gates (Holm) |",
           "|---|---|---|---|---|"]
    for g, r in out["groups"].items():
        a = r["accuracy"][f"{DESIGN[0]}@{DESIGN[1]:g}"]
        gates = "; ".join(f"{k.split('@')[0]} {v.get('label')}" for k, v in r["gates"].items())
        md.append(f"| {LB.R7B_GROUPS.get(g, g)} | **{r['verdict']}** | {r['testable']} | {a['labels']['0.02']} "
                  f"(n={a['n']}, lower {_f(a['lower'], 4)}) | {gates} |")
    for g, r in out["groups"].items():
        md += ["", f"## {LB.R7B_GROUPS.get(g, g)}", "", f"### Primary test: `{DESIGN[0]}` against {RHO} × FP8 + {KAPPA}", "",
               "| family | label | units (dropped) | KL/tok design | FP8 | fp_noise | mean y | upper U | fp_noise U | E4: KL ratio (95%) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for fam, v in r["families"].items():
            if v["label"] == "NO_DATA":
                md.append(f"| {fam} | NO_DATA | 0 | | | | | | | |")
                continue
            e = r["e4"].get(fam, {})
            er = f"{e['ratio_of_means']:.2f} ({e['lo']:.2f}–{e['hi']:.2f})" if "ratio_of_means" in e else "—"
            md.append(f"| {fam}{' (R6 units)' if v.get('reused_units') else ''} | **{v['label']}** | {v['n']} ({v['dropped']}) | "
                      f"{v['kl_design']:.4f} | {v['kl_fp8']:.4f} | {v['kl_noise']:.4f} | {_f(v['mean'])} | {_f(v['upper'])} | "
                      f"{_f(v['noise_upper'])} | {er} |")
        md += ["", "### Gates: design − baseline, one-sided bootstrap p, Holm at 0.05", "",
               "| baseline | traffic | units | mean d | upper | p | Holm level | label |", "|---|---|---|---|---|---|---|---|"]
        for k, v in r["gates"].items():
            if not v.get("n"):
                md.append(f"| {k} | | 0 | | | | | {v.get('label')} |")
                continue
            md.append(f"| {k} | {v['traffic']:.3f} | {v['n']} | {_f(v['mean'])} | {_f(v['upper'])} | {v['p']:.4f} | "
                      f"{v.get('holm_level', float('nan')):.4f} | **{v['label']}** |")
        md += ["", "### Reported: the primary statistic for other arms (U; ✓ = below the margin)", "",
               "| arm | " + " | ".join(r["families"]) + " |", "|---|" + "---|" * len(r["families"])]
        for arm, fams in r["reported"].items():
            md.append(f"| {arm} | " + " | ".join(
                (f"{_f(fams[f]['upper'])} {'✓' if fams[f]['below_margin'] else ''}" if f in fams else "—")
                for f in r["families"]) + " |")
        md += ["", "### Accuracy against FP8, FP-correct units outside aggregation (one-sided 95% lower bound)", "",
               "| arm | n | units lower / higher | mean | lower | δ = 0.02 | δ = 0.05 |", "|---|---|---|---|---|---|---|"]
        for arm, a in r["accuracy"].items():
            md.append(f"| {arm} | {a['n']} | {a.get('losses', 0)} / {a.get('gains', 0)} | {_f(a['mean'], 4)} | "
                      f"{_f(a['lower'], 4)} | {a['labels']['0.02']} | {a['labels']['0.05']} |")
        md += ["", "### Every arm against FP (90% intervals)", "",
               "| family | arm@B | traffic | KL/tok | dP/tok | ΔAcc FP-correct | ΔAcc FP-failed |", "|---|---|---|---|---|---|---|"]
        for a in sorted(r["arms"], key=lambda a: (a["family"], a["traffic"] if a["traffic"] == a["traffic"] else 9)):
            md.append(f"| {a['family']} | {a['arm']}@{a['B']:g} | {a['traffic']:.3f} | {R5._ci(a['KL_tok'])} | "
                      f"{R5._ci(a['dP_tok'])} | {R5._ci(a['dAcc_fp_correct'], 3)} | {R5._ci(a['dAcc_fp_failed'], 3)} |")


def _write(out, md, out_stem):
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:14]))
    print(f"wrote {out_stem}.md / .json")


# ------------------------------------------------------------------ exploratory blocks
def cover_label(misses, checks, delta=1e-3):
    return "COVER_OK" if misses <= delta * checks + 3.0 * math.sqrt(delta * checks) + 3.0 else "COVER_EXCESS"


def read_explore(pairs, out_stem, root=RESULTS, smoke=False):
    datas = {}
    excluded, parts, dups, same_code = _load_valid(pairs, root, smoke, layout=LB.R7B_EXPLORE_LAYOUT, datas=datas)
    by_group = {}                                        # the frozen calibration outputs must be one version per group
    for w, dsha in datas.items():
        by_group.setdefault(w.split("_")[0], set()).add(json.dumps(dsha, sort_keys=True))
    same_data = all(len(v) == 1 for v in by_group.values())
    out = dict(rules="read_r7b.py docstring, EXPLORATORY", excluded=excluded, code_manifests_identical=same_code,
               data_identical=same_data, data_sha256=datas, groups={})
    md = ["# R7b, the exploratory arms (E1-E3; reported, not tested)", ""]
    md += [f"- EXCLUDED: {w}: {'; '.join(p)}" for w, p in excluded.items()] or ["- every block valid"]
    if not same_code:
        md.append("- WARNING: the blocks' code manifests differ")
    if not same_data:
        md.append("- WARNING: blocks of one group read different calibration outputs (data/tau_*, data/alloc_*)")
    if parts:
        rows = pd.concat(parts, ignore_index=True)
        for g_name, rg in rows.groupby("group"):
            res = dict(e2=ratio_report(rg, ("dnest43ux_v4", 0.125), DESIGN),
                       e3=ratio_report(rg, (LB.WNEST, 0.125), ("dnest41ux_v4", 0.125)), e1={})
            for lv in ("t1", "t2"):
                arm = (f"cnest42ux_v4_{lv}", 0.125)
                c = rg[rg.arm == arm[0]]
                if not len(c):
                    continue
                mis, chk = float(c.cr_misses.sum()), float(c.cr_checks.sum())
                res["e1"][lv] = dict(ratio=ratio_report(rg, arm, DESIGN), traffic=float(c.cr_traffic.mean()),
                                     rows=float(c.cr_rows.mean()), fallback=float(c.cr_fallback.mean()),
                                     misses=mis, checks=chk, cover=cover_label(mis, chk),
                                     kl=float(c.kl_span_tok.mean()))
            res["e3"]["traffic_measured"] = float(rg[rg.arm == LB.WNEST].traffic_measured.mean()) \
                if "traffic_measured" in rg and (rg.arm == LB.WNEST).any() else None
            out["groups"][g_name] = res
            md += ["", f"## {LB.R7B_GROUPS.get(g_name, g_name)}", "",
                   f"- E2 (4+3 / 4+2): {res['e2'].get('ratio_of_means', float('nan')):.3f} "
                   f"({res['e2'].get('lo', float('nan')):.3f}–{res['e2'].get('hi', float('nan')):.3f}), n {res['e2'].get('n')}",
                   f"- E3 (water-filled / uniform 4+1, equal traffic): {res['e3'].get('ratio_of_means', float('nan')):.3f} "
                   f"({res['e3'].get('lo', float('nan')):.3f}–{res['e3'].get('hi', float('nan')):.3f})"]
            for lv, e in res["e1"].items():
                md.append(f"- E1 level {lv}: KL / design {e['ratio'].get('ratio_of_means', float('nan')):.3f}, traffic "
                          f"{e['traffic']:.4f}, rows {e['rows']:.4f}, fallback {e['fallback']:.4f}, misses {e['misses']:.0f} "
                          f"of {e['checks']:.0f} query-head checks ({e['cover']})")
    _write(out, md, out_stem)
    return 0 if parts else 1


# ------------------------------------------------------------------ the gate
def gate_projection(per_unit, group, layout=None):
    """Each of the group's core (and R7B_EXTRA) blocks' projected hours from its pilot's seconds per unit, scaled per task
    by GATE_UNIT_S against the pilot's task; and the blocks over GATE_WALL_FRAC of their wall."""
    pt = LB.PILOT_OF[group][6]
    proj, over = {}, []
    for row in (LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT if layout is None else layout):
        g = LB.R7B_GROUP_OF[row[2]]
        if g != group:
            continue
        sec = sum(row[7] * per_unit * GATE_UNIT_S[g][t] / GATE_UNIT_S[g][pt] for t in row[6].split(","))
        hh, mm, ss = (int(x) for x in row[9].split(":"))
        wall = hh + mm / 60 + ss / 3600
        proj[row[0]] = round(GATE_SLACK * sec / 3600 + GATE_SETUP_H, 2)
        if proj[row[0]] > GATE_WALL_FRAC * wall:
            over.append(f"{row[0]}: projected {proj[row[0]]:.2f} h > {GATE_WALL_FRAC} x {wall:.1f} h")
    return proj, over


def read_gate(tag, job, root=RESULTS, out_dir=FINDINGS):
    import read_stage1h as R1R
    problems = []
    side, d = load_block(tag, job, root)
    per_unit, proj, peaks = float("nan"), {}, []
    row = next((r for r in LB.R7B_PILOTS if r[1] == tag), None)
    group = LB.R7B_GROUP_OF[row[2]] if row else "unknown"
    if side is None:
        problems.append("cannot load the pilot")
    elif row is None or side.get("preset_name") != row[2]:
        problems.append(f"not an R7b pilot ({tag}, {side.get('preset_name')})")
    else:
        validity(tag, side, d, problems, pilot=True)
        if not (d[d.arm == "fp"].kl_all.fillna(0) == 0).all():
            problems.append("FP's KL to itself is not 0")
        degen, n = R1R.noise_degenerate(d.assign(job=str(job)))
        if degen:
            problems.append(f"NOISE_DEGENERATE on {n} units")
        peaks = [float(x) for x in side.get("peak_gib_dev_max") or []]
        if peaks and max(peaks) > PEAK_LIMIT:
            problems.append(f"peak {max(peaks):.1f} GiB > {PEAK_LIMIT}")
        t = []
        for _, g in d.groupby(UKEY):
            t.append(float(g.t_prefill.iloc[0] + g.t_precompute.fillna(0).iloc[0] + (g.t_arm.fillna(0) + g.t_tf.fillna(0)).sum()))
        per_unit = float(np.mean(t)) if t else float("nan")
        if not np.isfinite(per_unit):
            problems.append("no pilot time")
        else:
            proj, over = gate_projection(per_unit, group)
            problems += over
    verdict = "PASS" if not problems else "FAIL"
    pin = os.path.join(FINDINGS, "R7b_code.sha256")
    out = dict(verdict=verdict, problems=problems, group=group, pilot=f"{tag}:{job}",
               gate_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               code_pin_sha256=hashlib.sha256(open(pin, "rb").read()).hexdigest() if os.path.exists(pin) else None,
               seconds_per_unit=per_unit, projected_block_h=proj, peak_gib=max(peaks) if peaks else None)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f".R7b_gate_{group}.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, f"R7b_gate_{group}.json"))
    print(json.dumps(out, indent=1))
    return 0 if verdict == "PASS" else 1


def _pairs(xs):
    return [tuple(x.rsplit(":", 1)) for x in xs]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--r7b", nargs="+", metavar="TAG:JOB")
    g.add_argument("--gate", metavar="TAG:JOB")
    g.add_argument("--explore", nargs="+", metavar="TAG:JOB")
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.gate:
        (t, j), = _pairs([a.gate])
        sys.exit(read_gate(t, j, a.results_root))
    if a.explore:
        sys.exit(read_explore(_pairs(a.explore), a.out_stem or os.path.join(FINDINGS, "R7b_explore"), a.results_root))
    sys.exit(read_r7b(_pairs(a.r7b), a.out_stem or os.path.join(FINDINGS, "R7b_reader"), a.results_root))


if __name__ == "__main__":
    main()
