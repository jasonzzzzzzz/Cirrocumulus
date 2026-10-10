#!/usr/bin/env python3
"""R14 Stage 1h R6: the confirmatory reader and Qwen 128K's gate. The rules below are FROZEN: written 2026-10-10
(plan.md amendment "R6 frozen"), before any R6 output existed. Design: s1h5_lib.py (R6_DESIGN, R6_LAYOUT, presets
h6*); driver: run_s1h5.py; HELMET items: tasks_s1h6.py.

    python read_stage1h_r6.py --r6 TAG:JOB [TAG:JOB ...] [--out-stem findings/R6_reader]
    python read_stage1h_r6.py --gate TAG:JOB          (Qwen 128K's pilot; writes findings/R6_gate_qwen128.json)

CLAIM. For each claim group (Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K, Qwen3-30B-A3B at 128K), the design
  dtail6ux_v4 at 1/8 (the vote's rows exact, the rest from a dithered tier 1 at 6-bit keys and 4-bit values,
  unbiased; traffic 0.398 of FP16 K+V) is non-inferior to FP8 KV (fp8kv, traffic 0.5) on KL per span token.
UNITS. Prompt x task, each in exactly one block; families: retrieval (niah_single, niah_multikey, mk_panel),
  multivalue, tracking (vt), aggregation (cwe, fwe), rag (kilt_nq, kilt_hotpotqa), rerank (msmarco_rerank_psg),
  icl (icl_trec_coarse, icl_banking77). Planned (s1h5_lib.R6_LAYOUT): Llama 128K retrieval 60, multivalue 30,
  tracking 30, aggregation 30, rag 60, icl 60, rerank 30; Qwen 32K retrieval 80, multivalue 40, tracking 40,
  aggregation 30; Qwen 128K rag 60, icl 60, rerank 30.
PRIMARY TEST, per claim group x family. y_u = KL_tok(design) - 1.5 KL_tok(FP8) per unit (both from the unit's own
  process; units with a non-finite value in either arm are dropped and counted). U = the 95th percentile of
  2,000 unit-bootstrap means of y (resampling units with replacement; numpy default_rng(0)): a one-sided 95% upper
  bound. PASS if U < 5e-4, else FAIL.
CALIBRATION (part of the test). The same test on fp_noise (FP rerun) in the same group and family: if fp_noise's U
  >= 5e-4 the family is NOT_TESTABLE at this precision (neither a pass nor a failure of the design).
GROUP. NONINFERIOR if every testable family is PASS and at least half of the group's planned families are testable;
  NOT_SHOWN if any testable family is FAIL; INSUFFICIENT otherwise. Intersection-union: no multiplicity correction.
  Each group is its own claim.
KEY SECONDARY (accuracy), per claim group: the FP-correct units (FP's score >= 1 - 1e-9) of every family but
  aggregation; d_u = score(design) - score(FP8); L = the 5th percentile of 2,000 unit-bootstrap means (one-sided 95%
  lower bound). At delta = 0.02: ACC_PASS if L > -delta and n >= 3 / delta (= 150; with no loss seen, fewer units
  cannot bound the loss rate below delta), ACC_FAIL if L <= -delta, else ACC_UNDERPOWERED. Reported at 0.05 too.
REPORTED, NOT TESTED: per group x family x arm, KL/tok, dP/tok (against FP), accuracy and its change on FP-correct and
  FP-failed units (90% intervals) and traffic; the primary test's statistic for dtail4ux_v4, uniform+v4, uniform@4
  and kivi4_v4; the accuracy secondary for every arm; Theorem R in the probe blocks (presets h6*p): coverage (a
  guarantee: anything below 1 is a finding), median bound / error and the share of head-steps certified within FP8's
  error, v1 and v2, the vote's rows and C/128, C/32, C/8 more, plain and unbiased tail, at the design's 6-bit keys.
VALIDITY (per block; a failing block is EXCLUDED and reported, the rest are read):
  an R6 block (sidecar amend_r6 = 'R6') whose tag, preset, suite and tasks are a row of R6_LAYOUT (pilots excluded);
  its plan is the preset's (s1h5_lib.build_plan); every planned arm on every unit and nothing else; its units are R6's
  planned units for the tag (s1h5_lib.r6_units); the dtail arms evict nothing; the probe, where planned, ran and its
  pass is FP's (|kl_all| <= 1e-6); peak GPU memory <= 76 GiB per device; HELMET blocks carry R6's manifest
  (tasks_s1h6.MANIFEST_SHA256). A unit in two valid blocks is read from the first (sorted by job) and reported.
GATE (Qwen 128K's pilot, R6_PILOT; excluded): validity as above except the units (the pilot's list) and the layout
  row; FP's KL to itself 0; fp_noise not degenerate (R1's NOISE_DEGENERATE); peak <= 76 GiB per device; the projected
  w-hm block (2 x rag 5 + rerank 5 + 2 x icl 5 units at the pilot's seconds per unit of each family, + 15 minutes)
  within 90% of its wall time (R6_LAYOUT). Exits 1 on FAIL, so the node jobs (afterok) never start.
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s1h5_lib as L  # noqa: E402
import tasks_s1h6 as T6  # noqa: E402
import read_stage1h_r5 as R5  # noqa: E402

RESULTS, FINDINGS = R5.RESULTS, R5.FINDINGS
DESIGN = L.R6_DESIGN
REPORTED = [tuple(x) for x in L.R6_REPORTED] + [("uniform+v4", 4.0), ("uniform", 4.0), ("kivi4_v4", 4.0)]
REF, NOISE = ("fp8kv", 8.0), ("fp_noise", 0.0)
RHO, KAPPA = L.R6_MARGIN
N_BOOT, SEED, ALPHA = 2000, 0, 0.05
ACC_DELTAS = (0.02, 0.05)
ACC_KEY = 0.02
ACC_EXCLUDE = ("aggregation",)
PEAK_LIMIT = 76.0
PROBE_KL = 1e-6
GATE_SETUP_H, GATE_WALL_FRAC = 0.25, 0.9
FAMILY = R5.FAMILY
UKEY = ["prompt_idx", "task"]
GROUPS = {"llama128": "Llama-3.1-8B, 128K", "qwen32": "Qwen3-30B-A3B, 32K", "qwen128": "Qwen3-30B-A3B, 128K"}


def planned_families():
    out = {}
    for _n, tag, preset, *_r in L.R6_LAYOUT:
        g = L.R6_GROUP_OF[preset.rstrip("p")]
        for t in _r[3].split(","):
            out.setdefault(g, set()).add(FAMILY[t])
    return {g: sorted(v) for g, v in out.items()}


def upper(y):
    """One-sided 95% unit-bootstrap upper bound of the mean (and the mean, n)."""
    y = np.asarray(y, float)
    bs = y[np.random.default_rng(SEED).integers(0, len(y), (N_BOOT, len(y)))].mean(1)
    return float(y.mean()), float(np.quantile(bs, 1 - ALPHA)), int(len(y))


def lower(y):
    y = np.asarray(y, float)
    bs = y[np.random.default_rng(SEED).integers(0, len(y), (N_BOOT, len(y)))].mean(1)
    return float(y.mean()), float(np.quantile(bs, ALPHA)), int(len(y))


# ------------------------------------------------------------------ loading and validity
def _layout_row(tag, preset, suite, tasks):
    for row in L.R6_LAYOUT:
        if row[1] == tag and row[2] == preset and row[4] == suite and row[6] == tasks:
            return row
    return None


def load_block(tag, job, root=RESULTS):
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(sp) != 1:
        return None, None, None
    side = json.load(open(sp[0]))
    main = pd.read_parquet(os.path.join(d, side["parquet"]))
    hf = (side.get("r5_files") or {}).get("probe_heads")
    heads = pd.read_parquet(os.path.join(d, hf)) if hf else None
    return side, main, heads


def validity(tag, side, d, problems, pilot=False, smoke=False):
    """smoke: test_r14_stage1h_r6's CPU smokes only (the smoke presets; no layout, unit, peak or manifest-cell rule)."""
    preset = side.get("preset_name")
    if side.get("amend_r6") != L.AMEND_R6 or preset not in L.R6_PRESETS or (preset.startswith("h6smoke") and not smoke):
        problems.append(f"not an R6 block (amend_r6 {side.get('amend_r6')!r}, preset {preset!r})")
        return
    tasks = ",".join(side.get("tasks") or [])
    if not (pilot or smoke) and _layout_row(tag, preset, side.get("suite"), tasks) is None:
        problems.append(f"tag / preset / suite / tasks ({tag}, {preset}, {side.get('suite')}, {tasks}) is not a row of "
                        "R6_LAYOUT")
    plan = [tuple(p) for p in side.get("plan", [])]
    want = [tuple(p) for p in L.build_plan(L.PRESETS[preset])]
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
        pu = L.r6_units()
        bad = [(int(p), t) for p, t in units.itertuples(index=False) if int(p) not in pu.get((tag, t), set())]
        if bad:
            problems.append(f"units outside R6's plan for {tag}: {bad[:5]}")
    dt = d[d.arm.str.startswith("dtail")]
    if len(dt) and float(dt.evict_frac.fillna(0).abs().max()) > 0:
        problems.append("a dtail arm evicted rows")
    pr = d[d.arm == "probe"]
    if preset.endswith("p") and (not len(pr) or float(pr.kl_all.abs().max()) > PROBE_KL):
        problems.append(f"the probe's pass is not FP's (max |kl_all| {float(pr.kl_all.abs().max()) if len(pr) else 'none'})")
    peaks = [float(v) for v in side.get("peak_gib_dev_max") or []]
    if int(side.get("n_gpus") or 0) > 0 and (not peaks or max(peaks) > PEAK_LIMIT):
        problems.append(f"peak GPU memory per device {peaks}")
    if side.get("suite") == "r4" and side.get("manifest_sha256") != T6.MANIFEST_SHA256:
        problems.append(f"HELMET manifest {str(side.get('manifest_sha256'))[:12]} is not R6's")


# ------------------------------------------------------------------ the tests
def ni_test(g, arm, B):
    """(mean, U, n, dropped) of y = KL_tok(arm) - RHO KL_tok(FP8) on one group x family's units."""
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
    """Accuracy change against FP8 on the FP-correct units outside aggregation: (mean, L, n) and labels."""
    x = rows[~rows.family.isin(ACC_EXCLUDE)]
    fp = x[x.arm == "fp"].set_index(["block"] + UKEY).score
    a = x[(x.arm == arm) & np.isclose(x.B, B)].set_index(["block"] + UKEY).score
    e = x[x.arm == REF[0]].set_index(["block"] + UKEY).score
    idx = fp.index[fp >= 1 - 1e-9].intersection(a.index).intersection(e.index)
    d = (a[idx] - e[idx]).values
    if not len(d):
        return dict(n=0, mean=float("nan"), lower=float("nan"), labels={str(dl): "ACC_UNDERPOWERED" for dl in ACC_DELTAS})
    m, lo, n = lower(d)
    lab = {}
    for dl in ACC_DELTAS:
        lab[str(dl)] = ("ACC_FAIL" if lo <= -dl else "ACC_PASS" if n >= math.ceil(3 / dl) else "ACC_UNDERPOWERED")
    return dict(n=n, mean=m, lower=lo, losses=int((d < 0).sum()), gains=int((d > 0).sum()), labels=lab)


def read_r6(pairs, out_stem, root=RESULTS, smoke=False):
    excluded, parts, heads, seen, dups = {}, [], [], set(), []
    for tag, job in sorted(pairs, key=lambda p: (p[0], str(p[1]))):
        w = f"{tag}_{job}"
        side, d, h = load_block(tag, job, root)
        if side is None:
            excluded[w] = ["cannot load"]
            print(f"{w}: cannot load", flush=True)
            continue
        pr = []
        validity(tag, side, d, pr, smoke=smoke)
        if pr:
            excluded[w] = pr
            print(f"{w}: EXCLUDED: {'; '.join(pr)}", flush=True)
            continue
        group = L.R6_GROUP_OF.get(side["preset_name"].rstrip("p"), "llama128" if smoke else None)
        keys = set(zip([group] * len(d), d.prompt_idx.astype(int), d.task))
        if keys & seen:
            dups.append(w)
            d = d[[(group, int(p), t) not in seen for p, t in zip(d.prompt_idx, d.task)]]
        seen |= keys
        parts.append(d.assign(group=group, block=w, family=d.task.map(FAMILY))[
            ["group", "block", "family", "prompt_idx", "task", "arm", "B", "a_span_nll_tok", "kl_span_tok", "kl_all",
             "score", "traffic_frac", "exact_rows_frac"]])
        if h is not None and len(h):
            heads.append(h.assign(group=group, block=w))
        print(f"{w}: valid ({group}, {len(d[d.arm == 'fp'])} units)", flush=True)
    out = dict(rules="read_stage1h_r6.py docstring (frozen 2026-10-10; plan.md amendment 'R6 frozen')",
               design=list(DESIGN), margin=[RHO, KAPPA], excluded=excluded, duplicate_units_in=dups, groups={})
    md = ["# R14 Stage 1h — R6, the confirmatory read (rules: read_stage1h_r6.py, frozen 2026-10-10)", "",
          f"Design `{DESIGN[0]}`@{DESIGN[1]:g} against FP8 KV: mean KL/token < {RHO} × FP8's + {KAPPA} per claim group "
          "× family (one-sided 95% unit-bootstrap upper bound; fp_noise calibrates). Accuracy: FP-correct units outside "
          f"aggregation, lower bound > −{ACC_KEY}.", ""]
    md += [f"- EXCLUDED: {w}: {'; '.join(p)}" for w, p in excluded.items()] or ["- every block valid"]
    if dups:
        md.append(f"- units also in an earlier block, read once: {', '.join(dups)}")
    if not parts:
        _write(out, md, out_stem)
        return 1
    rows = pd.concat(parts, ignore_index=True)
    pf = planned_families()
    for g_name, rg in rows.groupby("group"):
        res = dict(families={}, reported={}, accuracy={}, arms=[], theorem_r=[])
        for fam in pf[g_name]:
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
                                        kl_noise=float(g[g.arm == NOISE[0]].kl_span_tok.mean()))
            for arm, B in REPORTED:
                ma, ua, na, _ = ni_test(g, arm, B)
                res["reported"].setdefault(f"{arm}@{B:g}", {})[fam] = dict(mean=ma, upper=ua, n=na,
                                                                         below_margin=bool(ua < KAPPA))
        labs = [v["label"] for v in res["families"].values()]
        testable = [x for x in labs if x in ("PASS", "FAIL")]
        if "FAIL" in testable:
            verdict = "NOT_SHOWN"
        elif len(testable) * 2 >= len(pf[g_name]) and testable:
            verdict = "NONINFERIOR"
        else:
            verdict = "INSUFFICIENT"
        res["verdict"] = verdict
        res["testable"] = f"{len(testable)} of {len(pf[g_name])}"
        for arm, B in [DESIGN] + REPORTED + [NOISE]:
            res["accuracy"][f"{arm}@{B:g}"] = acc_test(rg, arm, B)
        res["arms"] = R5.arm_table(rg.rename(columns={"group": "model"}).assign(cell=rg.group))
        out["groups"][g_name] = res
    if heads:
        H = pd.concat(heads, ignore_index=True)
        for g_name, hg in H.groupby("group"):
            recs = []
            for k in sorted(c for c in hg.columns if c.startswith("rcov_")):
                tg = k[5:]
                recs.append(dict(read=tg, head_steps=int(len(hg)), cover=float(hg[k].mean()), cover_min=float(hg[k].min()),
                                 misses=int(round(float((1 - hg[k]).sum()))), ratio_med=float(hg[f"rratio_{tg}"].median()),
                                 fp8=float(hg[f"rfp8_{tg}"].mean()), rel_med=float(hg[f"rrel_{tg}"].median())))
            out["groups"].setdefault(g_name, {})["theorem_r"] = recs
    _markdown(out, md)
    _write(out, md, out_stem)
    return 0


def _f(x, nd=5):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _markdown(out, md):
    md += ["", "## Verdicts", "", "| claim group | verdict | testable families | accuracy (δ = 0.02) | accuracy (δ = 0.05) |",
           "|---|---|---|---|---|"]
    for g, r in out["groups"].items():
        if "verdict" not in r:
            continue
        a = r["accuracy"][f"{DESIGN[0]}@{DESIGN[1]:g}"]
        md.append(f"| {GROUPS[g]} | **{r['verdict']}** | {r['testable']} | {a['labels']['0.02']} (n={a['n']}, lower "
                  f"{_f(a['lower'], 4)}) | {a['labels']['0.05']} |")
    for g, r in out["groups"].items():
        if "verdict" not in r:
            continue
        md += ["", f"## {GROUPS[g]}", "", f"### Primary test: `{DESIGN[0]}` against {RHO} × FP8 + {KAPPA}", "",
               "| family | label | units (dropped) | KL/tok design | FP8 | fp_noise | mean y | upper U | fp_noise U |",
               "|---|---|---|---|---|---|---|---|---|"]
        for fam, v in r["families"].items():
            if v["label"] == "NO_DATA":
                md.append(f"| {fam} | NO_DATA | 0 | | | | | | |")
                continue
            md.append(f"| {fam} | **{v['label']}** | {v['n']} ({v['dropped']}) | {v['kl_design']:.4f} | {v['kl_fp8']:.4f} | "
                      f"{v['kl_noise']:.4f} | {_f(v['mean'])} | {_f(v['upper'])} | {_f(v['noise_upper'])} |")
        md += ["", "### Reported, not tested: the same statistic for other arms (U; ✓ = below the margin)", "",
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
        if r.get("theorem_r"):
            md += ["", "### Theorem R in the probe blocks (6-bit dithered keys; coverage is a guarantee: it must be 1)", "",
                   "| read (v2 = Theorem R v2) | head-steps | coverage (min) | misses | median bound / error | "
                   "certified within FP8 | rel. error |", "|---|---|---|---|---|---|---|"]
            md += [f"| {t['read']} | {t['head_steps']} | {t['cover']:.5f} ({t['cover_min']:.3f}) | {t['misses']} | "
                   f"{t['ratio_med']:.1f} | {t['fp8']:.3f} | {t['rel_med']:.5f} |" for t in r["theorem_r"]]
        md += ["", "### Every arm against FP (90% intervals)", "",
               "| family | arm@B | traffic | KL/tok | dP/tok | ΔAcc FP-correct | ΔAcc FP-failed |", "|---|---|---|---|---|---|---|"]
        for a in sorted(r["arms"], key=lambda a: (a["family"], a["traffic"])):
            md.append(f"| {a['family']} | {a['arm']}@{a['B']:g} | {a['traffic']:.3f} | {R5._ci(a['KL_tok'])} | "
                      f"{R5._ci(a['dP_tok'])} | {R5._ci(a['dAcc_fp_correct'], 3)} | {R5._ci(a['dAcc_fp_failed'], 3)} |")


def _write(out, md, out_stem):
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"wrote {out_stem}.md / .json")


# ------------------------------------------------------------------ the gate
def read_gate(tag, job, root=RESULTS, out_dir=FINDINGS):
    import read_stage1h as R1R
    problems = []
    side, d, _ = load_block(tag, job, root)
    if side is None:
        problems.append("cannot load the pilot")
    else:
        if tag != L.R6_PILOT[1] or side.get("preset_name") != L.R6_PILOT[2]:
            problems.append(f"not R6's Qwen 128K pilot ({tag}, {side.get('preset_name')})")
        validity(tag, side, d, problems, pilot=True)
        fp = d[d.arm == "fp"]
        if not (fp.kl_all.fillna(0) == 0).all():
            problems.append("FP's KL to itself is not 0")
        degen, n = R1R.noise_degenerate(d.assign(job=str(job)))
        if degen:
            problems.append(f"NOISE_DEGENERATE on {n} units")
    per_fam, proj_h, peaks = {}, float("nan"), []
    if side is not None:
        peaks = [float(x) for x in side.get("peak_gib_dev_max") or []]
        for (p, task), g in d.groupby(UKEY):
            own = g["t_own"].fillna(0) if "t_own" in g else 0.0
            per_fam.setdefault(FAMILY[task], []).append(float(g.t_prefill.iloc[0] + g.t_precompute.fillna(0).iloc[0]
                                                              + (g.t_arm.fillna(0) + g.t_tf.fillna(0) + own).sum()))
        per_fam = {k: float(np.mean(v)) for k, v in per_fam.items()}
        row = next(r for r in L.R6_LAYOUT if r[1] == "h6qwen128_hm")
        need = {"rag": 2 * row[7], "rerank": row[7], "icl": 2 * row[7]}
        if set(need) - set(per_fam):
            problems.append(f"the pilot has no unit of {sorted(set(need) - set(per_fam))}")
        else:
            hh, mm, ss = (int(x) for x in row[9].split(":"))
            wall_h = hh + mm / 60 + ss / 3600
            proj_h = sum(n_ * per_fam[f] for f, n_ in need.items()) / 3600 + GATE_SETUP_H
            if proj_h > GATE_WALL_FRAC * wall_h:
                problems.append(f"projected block {proj_h:.2f} h > {GATE_WALL_FRAC} x {wall_h:.1f} h")
    verdict = "PASS" if not problems else "FAIL"
    out = dict(cell="qwen128", job=str(job), verdict=verdict, problems=problems, peak_gib_dev=peaks,
               seconds_per_unit=per_fam, projected_block_h=round(proj_h, 2) if np.isfinite(proj_h) else None)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "R6_gate_qwen128.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if verdict == "PASS" else 1


def _pairs(xs):
    out = []
    for x in xs:
        t, j = x.rsplit(":", 1)
        out.append((t, j))
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--r6", nargs="+", metavar="TAG:JOB")
    g.add_argument("--gate", metavar="TAG:JOB")
    ap.add_argument("--out-stem", default=os.path.join(FINDINGS, "R6_reader"))
    ap.add_argument("--results-root", default=RESULTS)
    a = ap.parse_args()
    if a.gate:
        (t, j), = _pairs([a.gate])
        sys.exit(read_gate(t, j, a.results_root))
    sys.exit(read_r6(_pairs(a.r6), a.out_stem, a.results_root))


if __name__ == "__main__":
    main()
