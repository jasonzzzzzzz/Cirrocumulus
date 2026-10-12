#!/usr/bin/env python3
"""R7b's calibration stage (plan_r7b.md §5; ../../14h_theory/RESEARCH_PLAN.md §4 "Calibration stage"): the frozen inputs
of the exploratory arms, from calibration prompts only (never test units). Writes data/tau_<group>.json (E1) and
data/alloc_<group>.json (E3); both are hashed into every later block's sidecar.

    python calib_r7b.py stage1 --group llama128 --blocks TAG:JOB [...] [--loose-from-r7a TAG:JOB ...] [--loose 3,8]
    python calib_r7b.py stage2 --group llama128 --blocks TAG:JOB [...] [--strict]
    --blocks auto: the group's finished calibration blocks of that stage in results/ (every planned block, or exit 1).
    The submission chain runs stage 1 at the start of every stage-2 job and stage 2 (--strict: exit 1 unless both levels
    are inside the scan and the exploratory gate passes) at the start of every exploratory job (submit_r7b.slurm R7B_PRE).
    Both are idempotent: they skip the computation when the output already comes from the same blocks, and write
    atomically, so concurrent jobs of one group read and write the same bytes.

STAGE 1 (blocks of the r7bc1<group> presets: per unit, the design and the single-layer arms kcal_{tail,key,val}_l{li}).
  Per unit and layer, KL_l = the arm's KL per answer token (kl_all_tok) and err2_l = layer li's realized output error
  (sum over heads of |o_arm - o_exact|^2, mean over answer steps; the arm's cal_err2).
  E1 (../../14h_theory/theory.md §14): c_l = 2 KL_tail,l / err2_tail,l per unit, pooled by the mean over units (pooling
    across families loses at most 10% on the CPU: findings_klbridge.md §6.3). tau_l(B) = sqrt(2 B / (L H c_l)).
    The scan levels s1..s5 are B = 10, 1, 0.1, 0.01, 0.001. Loose layers (4+4 keys in E1): the layers whose median part-8
    bound / error at 6-bit keys exceeds lib_r7b.LOOSE_RATIO in R7a's probe blocks of the same group (--loose-from-r7a),
    an explicit list (--loose), or else lib_r7b.LOOSE_DEFAULT (the measured lists: Qwen's 14 layers at 32K, none for
    Llama); the source is recorded in data/tau_<group>.json (loose_source).
  E3 (Corollary WF): with the key and value parts of each layer's KL additive (Proposition KL) and each part's MSE
    proportional to its step squared (Corollary M-B), KL_l(r, v) = KL_key,l (3 / (2^r - 1))^2 + KL_val,l (15 / (2^v - 1))^2
    for residual bits r (keys 4 + r) and value bits v, from the single-layer key-only (r = 2) and value-only (v = 4) arms.
    Each unit's surface is divided by its own sum over layers at (2, 4) (every input weighted equally;
    findings_klbridge.md §3), then averaged. The allocation is the per-layer Lagrangian choice over r in {1, 2, 3, 4} and
    v in {3, 4, 5, 6} with the least predicted KL at mean traffic <= the uniform control's (4+1/4: 0.371).
STAGE 2 (blocks of the r7bc2<group> presets: per unit, the design and E1 at s1..s5). The levels t1 and t2 are the B whose
  mean traffic over the calibration units (log-linear in log B between scan levels) is E1's floor + 0.002 and + 0.01.
  The floor is E1's traffic with only the vote's rows read: the design's 0.3984 plus the loose layers' two extra key bits
  (on Qwen3-30B-A3B at 32K about +0.016, so targets on the design's own traffic would sit below the floor and collapse to
  one level). A target outside the scan takes the nearest end and is flagged (bracketed: false; widen the scan before
  the exploratory blocks). They are added to data/tau_<group>.json, with the exploratory blocks' gate (explore_gate:
  stage 2's peak GPU memory, and each exploratory block's projected hours from E1's measured seconds per answer token;
  explore_projection). Submit the exploratory blocks only when it passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
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

H0 = os.path.dirname(os.path.dirname(R7))
RESULTS = os.path.join(H0, "results")
SCAN = {"s1": 10.0, "s2": 1.0, "s3": 0.1, "s4": 0.01, "s5": 0.001}
TARGET_EXTRA = {"t1": 0.002, "t2": 0.01}
RS, VS = (1, 2, 3, 4), (3, 4, 5, 6)
FRAC = 1 / 8


def _blocks(pairs, prefix="r14r7b_"):
    out = []
    for tag, job in pairs:
        d = os.path.join(RESULTS, f"{prefix}{tag}_{job}")
        sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
        if len(sp) != 1:
            raise SystemExit(f"{d}: no sidecar")
        side = json.load(open(sp[0]))
        out.append((side, pd.read_parquet(os.path.join(d, side["parquet"]))))
    return out


def _pairs(xs):
    return [tuple(x.rsplit(":", 1)) for x in xs]


def auto_pairs(group, stage, root=None):
    """The group's finished calibration blocks of a stage (calib1 | calib2) in results/, as TAG:JOB pairs: one per planned
    block (lib_r7b.blocks_of), matched by its sidecar's preset, prompts and offset; exits if one is missing."""
    root = root or RESULTS
    out, missing = [], []
    for b in LB.blocks_of(LB.stage_rows(stage)):
        if b["group"] != group:
            continue
        hit = None
        for d in sorted(glob.glob(os.path.join(root, f"r14r7b_{b['tag']}_*"))):
            sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
            if len(sp) != 1:
                continue
            s = json.load(open(sp[0]))
            if (s.get("preset_name") == b["preset"] and int(s.get("n_prompts") or -1) == b["n"]
                    and int(s.get("prompt_offset") or -1) == b["off"]
                    and str((s.get("r7b") or {}).get("amend", "")).startswith("R7b")
                    and os.path.exists(os.path.join(d, s.get("parquet", "")))):
                hit = os.path.basename(d)[len(f"r14r7b_{b['tag']}_"):]
                break
        if hit is None:
            missing.append(b["name"])
        else:
            out.append((b["tag"], hit))
    if missing or not out:
        raise SystemExit(f"{group}: {stage} blocks not in {root}: {missing or 'none planned'}")
    return out


def _dump(obj, path):
    """Atomic write: concurrent jobs of one group may run the same stage at once."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def traffic(r: int, v: int, f: float = FRAC) -> float:
    return (1 - f) * (LB.BASE_BITS + r + v) / 32.0 + f


def stage1(group, pairs, loose=None, loose_pairs=None):
    p_tau, p_alloc = LB.data_path("tau", group), LB.data_path("alloc", group)
    key = sorted(f"{t}:{j}" for t, j in pairs)
    if os.path.exists(p_tau) and os.path.exists(p_alloc) and loose is None and not loose_pairs:
        old = json.load(open(p_tau))
        if old.get("stage1_blocks") == key:
            print(f"{group}: stage 1 is up to date ({p_tau}, stage {old.get('stage')}, from {key})")
            return old, json.load(open(p_alloc))
    blocks = _blocks(pairs)
    model = LB.PRESETS[blocks[0][0]["preset_name"]]["model"]
    nL = int(LB.PRESETS[blocks[0][0]["preset_name"]]["n_layers"])
    H = LB.SMOKE_N_HEADS if group == "smoke" else LB.N_HEADS[model]
    per_unit = {}
    for side, d in blocks:
        if side.get("r7b", {}).get("preset_kind") != "calib1":
            raise SystemExit(f"{side['preset_name']}: not a stage-1 calibration block")
        for (p, t), g in d.groupby(["prompt_idx", "task"]):
            u = per_unit.setdefault((int(p), t), {"tail": {}, "key": {}, "val": {}})
            for _, r in g.iterrows():
                m = re.fullmatch(LB.KCAL_RE, r.arm)
                if m:
                    u[m.group(1)][int(m.group(2))] = (float(r.kl_all_tok), float(r.cal_err2))
    units = sorted(per_unit)
    if not units:
        raise SystemExit("no calibration units")
    c = {li: [] for li in range(nL)}
    surfaces = []
    for k in units:
        u = per_unit[k]
        if any(len(u[part]) != nL for part in u):
            raise SystemExit(f"unit {k}: {[(part, len(v)) for part, v in u.items()]} layers of {nL}")
        for li in range(nL):
            kl, e2 = u["tail"][li]
            c[li].append(2.0 * max(kl, 0.0) / max(e2, 1e-300))
        sk = {li: max(u["key"][li][0], 0.0) for li in range(nL)}
        sv = {li: max(u["val"][li][0], 0.0) for li in range(nL)}
        z = sum(sk.values()) + sum(sv.values())
        surfaces.append({li: {(r, v): (sk[li] * (3.0 / (2 ** r - 1)) ** 2 + sv[li] * (15.0 / (2 ** v - 1)) ** 2) / max(z, 1e-300)
                              for r in RS for v in VS} for li in range(nL)})
    c_l = {str(li): st.mean(v) for li, v in c.items()}
    loose_source = "--loose (explicit)"
    if loose is None and loose_pairs:
        loose = loose_from_r7a(loose_pairs, nL)
        loose_source = f"R7a's probe blocks {loose_pairs}"
    if loose is None:
        loose = list(LB.LOOSE_DEFAULT.get(model, [])) if group != "smoke" else []
        loose_source = "lib_r7b.LOOSE_DEFAULT (measured: findings_nested.md §7 at 32K; R7a's 128K pilots for Llama)"
        print(f"{group}: loose layers from {loose_source}: {loose}")
    tau = dict(group=group, model=model, n_layers=nL, n_heads=H, c_l=c_l,
               c_spread={str(li): [min(v), max(v)] for li, v in c.items()}, units=[list(k) for k in units],
               levels=dict(SCAN), loose_layers=sorted(int(x) for x in (loose or [])), loose_ratio=LB.LOOSE_RATIO,
               loose_source=loose_source, stage1_blocks=key,
               rule="tau_l = sqrt(2 B / (L H c_l)); c_l = mean over calibration units of 2 KL_tail,l / err2_tail,l",
               stage=1)
    surf = {li: {kv: st.mean(s[li][kv] for s in surfaces) for kv in surfaces[0][li]} for li in range(nL)}
    target = traffic(1, 4)
    best = None
    for e in range(-60, 81):
        lam = 10 ** (e / 10)
        alloc = {li: min(surf[li], key=lambda kv: traffic(*kv) + lam * surf[li][kv]) for li in range(nL)}
        tr = sum(traffic(*alloc[li]) for li in range(nL)) / nL
        kl = sum(surf[li][alloc[li]] for li in range(nL))
        if tr <= target + 1e-12 and (best is None or kl < best[1]):
            best = (tr, kl, lam, alloc)
    uni = sum(surf[li][(1, 4)] for li in range(nL))
    alloc = dict(group=group, model=model, n_layers=nL, alloc={str(li): list(best[3][li]) for li in range(nL)},
                 traffic=best[0], predicted_rel_kl=best[1], uniform41_rel_kl=uni, lam=best[2], target_traffic=target,
                 rule="per-layer argmin traffic + lam KL_l over r in {1..4}, v in {3..6}; KL_l(r, v) from the single-layer "
                      "key-only and value-only KLs scaled by the step ratios; inputs weighted equally",
                 units=[list(k) for k in units])
    os.makedirs(LB.DATA, exist_ok=True)
    _dump(alloc, p_alloc)                                   # alloc first: a reader that sees stage 1's tau finds both
    _dump(tau, p_tau)
    print(f"{group}: {len(units)} calibration units; c_l {min(float(v) for v in c_l.values()):.3g}..{max(float(v) for v in c_l.values()):.3g}; "
          f"loose layers {tau['loose_layers']}; E3 at traffic {best[0]:.4f}: predicted relative KL {best[1]:.3f} against "
          f"uniform 4+1's {uni:.3f}")
    print(f"wrote {LB.data_path('tau', group)}, {LB.data_path('alloc', group)}")
    return tau, alloc


def loose_from_r7a(pairs, nL):
    """The layers whose median part-8 bound / error at 6-bit keys exceeds LOOSE_RATIO in R7a's probe blocks."""
    rat = {}
    for tag, job in pairs:
        d = os.path.join(RESULTS, f"r14r7a_{tag}_{job}")
        for f in glob.glob(os.path.join(d, "r7a_heads_*.parquet")):
            h = pd.read_parquet(f, columns=["layer", "rat_p8_k6"])
            for li, g in h.groupby("layer"):
                rat.setdefault(int(li), []).extend(g.rat_p8_k6.astype(float).tolist())
    if not rat:
        raise SystemExit("no R7a probe records found")
    return [li for li in range(nL) if li in rat and float(np.median(rat[li])) > LB.LOOSE_RATIO]


PEAK_LIMIT, WALL_FRAC, SETUP_H = 76.0, 0.9, 0.25            # read_r7b's gate settings


def explore_projection(group, blocks):
    """The exploratory blocks' gate, from the stage-2 blocks (the same group; per unit, the design and E1 at five levels):
    peak GPU memory, and each exploratory block's projected hours against its wall. E1's cost grows with the answer it
    teacher-forces, and the exploratory blocks hold longer answers than some calibration units (cwe), so E1 is projected
    per answer token: the costliest level's seconds per token x the 90th percentile of the calibration answers' length.
    Per exploratory unit: 2 prefills (fp's, fp_noise's rerun) + fp + 5 store arms (the design, E2, E3's control, E3, FP8, at
    the design's measured time) + 2 E1 arms."""
    peaks, pre, t_by, tok_by, lens = [], [], {}, {}, []
    for side, d in blocks:
        peaks += [float(x) for x in side.get("peak_gib_dev_max") or []]
        for _, g in d.groupby(["prompt_idx", "task"]):
            pre.append(float(np.nan_to_num(g.t_prefill.iloc[0])) + float(np.nan_to_num(g.t_precompute.iloc[0])))
            lens.append(float(g[g.arm == "fp"].tf_len.iloc[0]) if (g.arm == "fp").any() else float("nan"))
            for _, r in g.iterrows():
                t_by.setdefault(r.arm, []).append(float(np.nan_to_num(r.t_arm)) + float(np.nan_to_num(r.t_tf)))
                tok_by.setdefault(r.arm, []).append(float(np.nan_to_num(r.tf_len)))
    if not pre:
        return dict(ok=False, problems=["no stage-2 units"])
    e1_tok = max(sum(t_by[a]) / max(sum(tok_by[a]), 1.0) for a in t_by if a.startswith("cnest"))
    L90 = float(np.nanpercentile(lens, 90)) if np.isfinite(lens).any() else float("nan")
    t_design = float(np.mean(t_by.get(LB.DESIGN[0], [0.0])))
    per_unit = 2 * float(np.mean(pre)) + float(np.mean(t_by.get("fp", [0.0]))) + 5 * t_design + 2 * e1_tok * L90
    proj, problems = {}, []
    for row in LB.R7B_EXPLORE_LAYOUT:
        if LB.R7B_GROUP_OF[row[2]] != group:
            continue
        hh, mm, ss = (int(x) for x in row[9].split(":"))
        wall = hh + mm / 60 + ss / 3600
        n_units = row[7] * len(row[6].split(","))                    # per process (a node job's blocks run at once)
        proj[row[0]] = round(n_units * per_unit / 3600 + SETUP_H, 2)
        if proj[row[0]] > WALL_FRAC * wall:
            problems.append(f"{row[0]}: projected {proj[row[0]]:.2f} h > {WALL_FRAC} x {wall:.1f} h")
    if peaks and max(peaks) > PEAK_LIMIT:
        problems.append(f"peak {max(peaks):.1f} GiB > {PEAK_LIMIT}")
    return dict(ok=not problems, problems=problems, projected_block_h=proj, peak_gib=max(peaks) if peaks else None,
                e1_seconds_per_answer_token=e1_tok, answer_tokens_p90=L90, seconds_per_unit=per_unit)


def ready(tau):
    """Problems that stop the exploratory blocks: a level outside the scan, or the exploratory gate."""
    bad = [f"{n} is outside the scan (widen it and rerun stage 2)" for n, ok in (tau.get("bracketed") or {}).items() if not ok]
    gate = tau.get("explore_gate") or {}
    if not gate.get("ok"):
        bad.append(f"exploratory gate: {gate.get('problems', 'missing')}")
    return bad


def stage2(group, pairs, strict=False):
    tau = LB.load_data("tau", group)
    key = sorted(f"{t}:{j}" for t, j in pairs)
    if tau.get("stage") == 2 and tau.get("stage2_blocks") == key:
        print(f"{group}: stage 2 is up to date (from {key})")
        bad = ready(tau)
        if strict and bad:
            raise SystemExit(f"{group}: not ready for the exploratory blocks: {'; '.join(bad)}")
        return tau
    blocks = _blocks(pairs)
    rows = []
    for side, d in blocks:
        if side.get("r7b", {}).get("preset_kind") != "calib2":
            raise SystemExit(f"{side['preset_name']}: not a stage-2 calibration block")
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    pts = []
    for lv, B in SCAN.items():
        c = d[d.arm == f"cnest42ux_v4_{lv}"]
        if not len(c):
            raise SystemExit(f"no rows for level {lv}")
        pts.append((math.log10(B), float(c.cr_traffic.mean()), float(c.kl_all_tok.mean()), float(c.cr_misses.sum()),
                    float(c.cr_checks.sum())))
    design_tr = LB.traffic_frac(LB.DESIGN[0], LB.DESIGN[1], 131072)
    kv = LB.DESIGN[1]                                       # E1's floor: only the vote's rows read, loose layers at 4+4
    _, res = LB.tau_table(group, "s1", int(tau["n_layers"]))
    floor_tr = sum((1 - kv) * (LB.BASE_BITS + r + 4) / 32.0 + kv for r in res.values()) / len(res)
    pts.sort()                                              # increasing log B: traffic decreases
    levels, targets, bracketed = dict(tau["levels"]), {}, {}
    for name, extra in TARGET_EXTRA.items():
        want = targets[name] = floor_tr + extra
        lb = None
        for (x0, t0, *_), (x1, t1, *_) in zip(pts[:-1], pts[1:]):
            if min(t0, t1) <= want <= max(t0, t1) and t0 != t1:
                lb = x0 + (want - t0) * (x1 - x0) / (t1 - t0)
                break
        bracketed[name] = lb is not None
        if lb is None:                                      # outside the scan: the nearest end, flagged
            lb = pts[0][0] if want > pts[0][1] else pts[-1][0]
        levels[name] = 10 ** lb
    gate = explore_projection(group, blocks)
    tau.update(levels=levels, stage=2, scan=[dict(log10B=x, traffic=t, kl=k, misses=m, checks=c) for x, t, k, m, c in pts],
               design_traffic=design_tr, floor_traffic=floor_tr, target_traffic=targets, bracketed=bracketed,
               explore_gate=gate, stage2_blocks=key)
    _dump(tau, LB.data_path("tau", group))
    print(f"{group}: floor {floor_tr:.4f} (design {design_tr:.4f}); scan {[(round(x, 1), round(t, 4)) for x, t, *_ in pts]}; "
          f"t1 B={levels['t1']:.3g}, t2 B={levels['t2']:.3g}")
    print(f"{group}: exploratory gate {'PASS' if gate['ok'] else 'FAIL'}: peak {gate.get('peak_gib')} GiB, E1 "
          f"{gate.get('e1_seconds_per_answer_token', float('nan')):.2f} s per answer token, projected "
          f"{gate.get('projected_block_h')} h" + (f"; {'; '.join(gate['problems'])}" if gate["problems"] else ""))
    for name, ok in bracketed.items():
        if not ok:
            print(f"WARNING {group}: {name}'s target traffic {targets[name]:.4f} is outside the scan; it takes the nearest "
                  f"end (B={levels[name]:.3g}). Widen the scan before the exploratory blocks.")
    bad = ready(tau)
    if strict and bad:
        raise SystemExit(f"{group}: not ready for the exploratory blocks: {'; '.join(bad)}")
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stage1", "stage2"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--blocks", nargs="+", required=True, metavar="TAG:JOB", help="TAG:JOB pairs, or auto")
    ap.add_argument("--loose", default=None)
    ap.add_argument("--loose-from-r7a", nargs="*", default=None, metavar="TAG:JOB")
    ap.add_argument("--strict", action="store_true", help="stage 2: exit 1 unless the exploratory blocks may run")
    a = ap.parse_args()
    pairs = auto_pairs(a.group, "calib1" if a.stage == "stage1" else "calib2") if a.blocks == ["auto"] else _pairs(a.blocks)
    if a.stage == "stage1":
        loose = [int(x) for x in a.loose.split(",") if x] if a.loose is not None else None
        stage1(a.group, pairs, loose, _pairs(a.loose_from_r7a) if a.loose_from_r7a else None)
    else:
        stage2(a.group, pairs, strict=a.strict)


if __name__ == "__main__":
    main()
