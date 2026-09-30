#!/usr/bin/env python3
"""R14 Stage 1b gate and reader. The rules below are FROZEN: written 2026-09-28,
before any Stage 1b output existed (design: s1b_lib.py).

    python read_stage1b.py --pilot JOB
    python read_stage1b.py --main128 JOB_A JOB_B --main32 JOB_C JOB_D [--out-stem stage1b]

Result directories: h0_measurement/results/r14s1b_<tag>_<job>/, tag pilot128,
main128 or main32 (written by run_s1b.py --mode evaluate).

VALIDITY (any failure -> INVALID, no verdict is printed)
  V1 every (prompt, task) block holds exactly the sidecar plan's (arm, B) rows,
     once; all blocks of a cell share one corpus; (main) FP >= 0.9 on >= 3 of
     the 4 tasks.
  V2 audits: dense = w bits with nothing evicted; routers spend <= B; a hybrid
     evicts exactly its mask source's tokens (same evict_frac in the same
     prompt-task) and spends w x kept; a twin has its base's bits and eviction.
  V3 the teacher-forced replay of fp reproduces fp's own greedy answer on
     >= 98% of answer steps (pooled).
  V4 every twin arm changed its values: mean |tf_sum_nll(twin) - tf_sum_nll(base)| > 0.
LOSSLESS (per cell; the Stage 0/1 rule): delta vs fp (exact keys and values)
  >= -0.02 and a paired prompt-bootstrap 90% lower bound >= -0.05 (resampling
  prompts within each block, 10,000 draws, seed 14). Tasks with FP < 0.9 drop.
BYTES per context token per KV head per layer (d = 128): 16 (bits + key side)
  + (1 - f) 16 (v_bits + v side) + the BF16 tail amortised, as in bytes_model.py.
LENSES V16 = arms without a twin suffix (exact values); V4 = '+v4' arms; V2 =
  '+v2' arms. The fp twins are diagnostics: the value quantizer's own cost.
FAMILIES DENSE = uniform; SIEVE = router_calib; POOL = router_pool_calib; HYBRID
  = hyb_*; router_pool_oracle is a diagnostic. In lens L: D* = the lossless
  DENSE point with the fewest bytes, S* the same over SIEVE and POOL, H* over
  HYBRID, J* = the cheaper of S* and H*.
D1 (per cell, lenses V16 and V4) r1 = bytes(S*) / bytes(H*):
  SIEVE_EARNS if S* exists and (no H* or r1 <= 0.90); HYBRID_SUFFICES if H*
  exists and (no S* or r1 >= 1.00); TIE otherwise; NEITHER if neither exists.
D2 (per cell and lens) rho = bytes(J*) / bytes(D*): WIN <= 0.80 < TIE <= 1.00 <
  LOSS; NO_POINT without a J*, NO_DENSE with a J* but no D*.
  Decision: GO_KERNEL if any cell is WIN in V4 (the kernel
  targets that J*'s family); else SCOPE_EXACT_V if any cell is WIN in V16;
  else STOP_SYSTEMS. V2 is reported, not gating.
D3 (per cell) at the lowest pooled budget Bl: router_pool_calib vs router_calib,
  paired delta with its 90% interval, and deleted needle values (niah tasks):
  POOL_FIXES if delta >= +0.10, its lower bound > 0 and deletions fall by >= 50%;
  POOL_HELPS if the lower bound > 0 otherwise; POOL_NO_EFFECT otherwise.
D4 (pooled over cells; exact-value rows of uniform, router_calib,
  router_pool_calib, router_pool_oracle, which have per-head errors): AUC for a
  failed answer (score < 1) of -tf_c_min_logp against the best per-head
  statistic (median, p99, answer-weighted): SEQUENCE_PROXY if AUC_tf >= 0.85
  and beats the best by >= 0.15; WEAK if AUC_tf >= 0.75; NONE otherwise.
REPORTED, NOT GATED: the half-bit routers; failure signatures (deletion vs
  substitution) per arm; the value quantizer's cost on fp; Stage 1's 128K cell
  means (fp, uniform@3, router_calib@4) against this run's.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_model as BM  # noqa: E402
import s1b_lib as L  # noqa: E402

RESULTS = os.path.join(BM.H0, "results")
PEAK_GIB_MAX = 76.0
WALL_H = {"main128": 10.0, "main32": 4.0}
MAIN_BLOCK_PROMPT_TASKS = 40
V3_MIN = 0.98
D1_EARN, D1_SUFF = 0.90, 1.00
D3_DELTA, D3_DEL_DROP = 0.10, 0.50
D4_STRONG, D4_MARGIN, D4_WEAK = 0.85, 0.15, 0.75


def bk(B) -> str:
    """Display / key spelling of a budget: 3.0 -> '3', 3.5 -> '3.5'."""
    B = float(B)
    return str(int(B)) if B.is_integer() else f"{B:g}"


# ------------------------------------------------------------------ loading
def load_run(tag, job):
    d = os.path.join(RESULTS, f"r14s1b_{tag}_{job}")
    pq = glob.glob(os.path.join(d, "s1b_evaluate_*.parquet"))
    pq = [x for x in pq if not x.endswith("_heads.parquet")]
    js = glob.glob(os.path.join(d, "s1b_evaluate_*.json"))
    hd = glob.glob(os.path.join(d, "s1b_evaluate_*_heads.parquet"))
    if len(pq) != 1 or len(js) != 1 or len(hd) != 1:
        raise SystemExit(f"{d}: expected one evaluate parquet, heads parquet and sidecar")
    rows = pd.read_parquet(pq[0]).assign(job=str(job))
    rows["B"] = rows.B.astype(float)
    heads = pd.read_parquet(hd[0]).assign(job=str(job))
    heads["B"] = heads.B.astype(float)
    return rows, heads, json.load(open(js[0]))


def plan_of(side):
    return sorted((a, L.norm_b(b)) for a, b in side["plan"])


# ----------------------------------------------------------------- validity
def validate(d, sides, problems, main=True):
    for side in sides:
        want = plan_of(side)
        sub = d[d.job == side["_job"]]
        got = sub.groupby(["prompt_idx", "task"]).apply(
            lambda g: sorted(zip(g.arm, g.B.map(L.norm_b))))
        bad = [k for k, v in got.items() if v != want]
        if bad:
            problems.append(f"block {side['_job']}: (prompt, task) with the wrong arms: {bad[:3]}")
    if d.duplicated(["job", "prompt_idx", "task", "arm", "B"]).any():
        problems.append("duplicate rows")
    if d.corpus_sha.nunique() != 1:
        problems.append(f"blocks disagree on corpus: {sorted(d.corpus_sha.unique())}")
    fam = d.arm.map(L.family)
    dn = d[fam == "dense"]
    if not (np.allclose(dn.bits_per_token, dn.B.astype(float)) and (dn.evict_frac == 0).all()):
        problems.append("a dense row is not w bits with nothing evicted")
    rt = d[fam.isin(["sieve", "pool", "diag"])]
    if (rt.bits_per_token > rt.B.astype(float) + 1e-6).any():
        problems.append("a router row overspends its budget")
    key = ["job", "prompt_idx", "task"]
    hyb = {h[0]: h for h in sides[0]["preset"]["hybrids"]}
    for name, (_, src, sB, w, _k) in hyb.items():
        h = d[d.arm == name]
        s = d[(d.arm == src) & (d.B == L.norm_b(sB))]
        m = h.merge(s, on=key, suffixes=("", "_s"))
        if len(m) != len(h) or not np.allclose(m.evict_frac, m.evict_frac_s, atol=1e-9):
            problems.append(f"{name}: eviction differs from its mask source {src}@{sB}")
        elif not np.allclose(m.bits_per_token, w * (1 - m.evict_frac), atol=1e-6):
            problems.append(f"{name}: spend is not {w} x kept")
    tw = d[d.twin != ""]
    for arm in sorted(tw.arm.unique()):
        t = d[d.arm == arm]
        b = d[d.arm == L.base_of(arm)]
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
        if (fps >= BM.RMT.FP_MIN).sum() < min(3, len(fps)):
            problems.append(f"FP below {BM.RMT.FP_MIN} on too many tasks: {fps.round(3).to_dict()}")
    return agree


# ---------------------------------------------------------------- analysis
def add_bytes(d):
    v = d.copy()
    v["tail_tok"] = v.window + v.n_question_tokens + v.gen_len / 2.0
    v["bytes"] = ((BM.D / 8) * (v.bits_per_token + v.key_side)
                  + (1 - v.evict_frac) * (BM.D / 8) * (v.v_bits + v.v_side)
                  + v.tail_tok * BM.TAIL_BYTES / v.ctx_len)
    return v


def lens_of(arm):
    s = L.twin_suffix(arm)
    return {"": "V16", "+v4": "V4", "+v2": "V2"}[s]


def cell_points(d):
    """Lossless test and bytes for every (arm, B) of one cell."""
    fp = d[d.arm == "fp"].groupby("task").score.mean().rename("fp_task")
    d = d.join(fp, on="task")
    d["valid"] = d.fp_task >= BM.RMT.FP_MIN
    v = add_bytes(d[d.valid])
    m = v.groupby(["job", "prompt_idx", "arm", "B"]).score.mean().unstack(["arm", "B"])
    if m.isna().any().any():
        raise SystemExit("missing (arm, B) for some prompts")
    w = BM.boot_weights(m.index)
    ll = BM.lossless_table(m, w)
    byt = v.groupby(["arm", "B"]).bytes.mean()
    ef = v.groupby(["arm", "B"]).evict_frac.mean()
    kb = v.groupby(["arm", "B"]).bits_per_token.mean()
    pts = {}
    for (arm, B), s in ll.items():
        pts[f"{arm}@{bk(B)}"] = dict(arm=arm, B=float(B), lens=lens_of(arm), family=L.family(arm),
                                 bytes=float(byt[(arm, B)]), evict_frac=float(ef[(arm, B)]),
                                 key_bits=float(kb[(arm, B)]), **s)
    return pts, m, w, v, float(m[("fp", 0)].mean()), sorted(v.task.unique())


def best(pts, lens, fams):
    c = [(k, p) for k, p in pts.items() if p["lens"] == lens and p["family"] in fams
         and p["lossless"]]
    return min(c, key=lambda kp: kp[1]["bytes"]) if c else (None, None)


def paired(m, w, a, b):
    diff = (m[a] - m[b]).to_numpy()
    boot = (w @ diff) / w.sum(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    return float(diff.mean()), float(lo), float(hi)


def deletions(v, arm, B):
    fp = v[v.arm == "fp"].set_index(["job", "prompt_idx", "task"]).pred
    x = v[(v.arm == arm) & (v.B == B) & v.task.str.startswith("niah")]
    out = {"deletion": 0, "substitution": 0, "other": 0, "missing": 0}
    for r in x.itertuples():
        for k, n in L.value_errors(r.pred, fp[(r.job, r.prompt_idx, r.task)]).items():
            out[k] += n
    return out


def auc(y, s):
    y = np.asarray(y, int)
    r = pd.Series(np.asarray(s, float)).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def verdict_rho(r):
    if r is None:
        return "NO_POINT"
    return "WIN" if r <= BM.WIN else ("TIE" if r <= BM.TIE else "LOSS")


def analyse_cell(name, d, heads, sides):
    pts, m, w, v, fp_score, tasks = cell_points(d)
    res = dict(cell=name, fp=fp_score, valid_tasks=tasks, n_prompts=int(len(m)), points=pts,
               lenses={}, d3={}, signatures={})
    for lens in ("V16", "V4", "V2"):
        dk, dp = best(pts, lens, ("dense",))
        sk, sp = best(pts, lens, ("sieve", "pool"))
        hk, hp = best(pts, lens, ("hybrid",))
        cands = [x for x in ((sk, sp), (hk, hp)) if x[0]]
        jk, jp = min(cands, key=lambda kp: kp[1]["bytes"]) if cands else (None, None)
        rho = (jp["bytes"] / dp["bytes"]) if (jp and dp) else None
        d2 = "NO_DENSE" if (jp and not dp) else verdict_rho(rho)
        r1 = (sp["bytes"] / hp["bytes"]) if (sp and hp) else None
        if sp and (not hp or r1 <= D1_EARN):
            d1 = "SIEVE_EARNS"
        elif hp and (not sp or r1 >= D1_SUFF):
            d1 = "HYBRID_SUFFICES"
        elif sp and hp:
            d1 = "TIE"
        else:
            d1 = "NEITHER"
        res["lenses"][lens] = dict(D_star=dk, S_star=sk, H_star=hk, J_star=jk,
                                   J_family=(jp["family"] if jp else None),
                                   bytes_D=dp and dp["bytes"], bytes_S=sp and sp["bytes"],
                                   bytes_H=hp and hp["bytes"], bytes_J=jp and jp["bytes"],
                                   rho=rho, d2=d2, r1=r1, d1=d1)
    pool = sorted(float(b) for b in sides[0]["preset"]["pool"])
    cols = {(a_, float(b_)) for a_, b_ in m.columns}
    for B in pool:
        if ("router_calib", B) not in cols or ("router_pool_calib", B) not in cols:
            continue
        dl, lo, hi = paired(m, w, ("router_pool_calib", B), ("router_calib", B))
        ds, dp_ = deletions(v, "router_calib", B), deletions(v, "router_pool_calib", B)
        res["d3"][bk(B)] = dict(delta=dl, lo=lo, hi=hi, del_std=ds["deletion"],
                                del_pool=dp_["deletion"])
    if pool:
        Bl = bk(pool[0])
        x = res["d3"].get(Bl)
        if x:
            drop = (1 - x["del_pool"] / x["del_std"]) if x["del_std"] else 0.0
            x["verdict"] = ("POOL_FIXES" if x["delta"] >= D3_DELTA and x["lo"] > 0
                            and drop >= D3_DEL_DROP else
                            "POOL_HELPS" if x["lo"] > 0 else "POOL_NO_EFFECT")
            res["d3_verdict"] = dict(B=Bl, **x)
    for (arm, B) in sorted({(a, float(b)) for a, b in zip(v.arm, v.B) if a != "fp"}):
        res["signatures"][f"{arm}@{bk(B)}"] = deletions(v, arm, B)
    for t in ("+v4", "+v2"):
        if ("fp" + t, 0.0) in cols:
            res.setdefault("fp_value_cost", {})[t] = dict(zip(("delta", "lo", "hi"),
                                                          paired(m, w, ("fp" + t, 0.0), ("fp", 0.0))))
    # D4 rows: exact-value arms with per-head errors
    arms4 = ["uniform", "router_calib", "router_pool_calib", "router_pool_oracle"]
    hs = heads[heads.arm.isin(arms4)]

    def hstat(g):
        e, a_ = g.err.to_numpy(), g.ans_mass.to_numpy()
        return pd.Series(dict(h_med=np.median(e), h_p99=np.quantile(e, 0.99),
                              h_answ=float(np.nansum(a_ * e) / max(np.nansum(a_), 1e-12))))
    st = hs.groupby(["job", "prompt_idx", "task", "arm", "B"]).apply(hstat).reset_index()
    rows = v[v.arm.isin(arms4)]
    res["_d4"] = rows.merge(st, on=["job", "prompt_idx", "task", "arm", "B"])
    return res


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def read_main(cells, out_stem):
    results, problems = [], []
    for name, (tag, jobs) in cells.items():
        parts = [load_run(tag, j) for j in jobs]
        d = pd.concat([p[0] for p in parts], ignore_index=True)
        heads = pd.concat([p[1] for p in parts], ignore_index=True)
        sides = []
        for (_, _, s), j in zip(parts, jobs):
            s["_job"] = str(j)
            sides.append(s)
        if len({json.dumps(s["plan"]) for s in sides}) != 1:
            problems.append(f"{name}: blocks ran different plans")
        agree = validate(d, sides, problems, main=True)
        if problems:
            continue
        r = analyse_cell(name, d, heads, sides)
        r["fp_replay_agreement"] = agree
        results.append(r)
    if problems:
        raise SystemExit("INVALID Stage 1b data:\n  " + "\n  ".join(problems))
    # D2 decision across cells
    v4win = [r["cell"] for r in results if r["lenses"]["V4"]["d2"] == "WIN"]
    v16win = [r["cell"] for r in results if r["lenses"]["V16"]["d2"] == "WIN"]
    if v4win:
        dec = "GO_KERNEL"
    elif v16win:
        dec = "SCOPE_EXACT_V"
    else:
        dec = "STOP_SYSTEMS"
    # D4 pooled
    x = pd.concat([r.pop("_d4") for r in results], ignore_index=True)
    x = x.dropna(subset=["tf_c_min_logp", "h_med", "h_p99", "h_answ"])
    fail = (x.score < 1).astype(int)
    a_tf = auc(fail, -x.tf_c_min_logp)
    a_h = {s: auc(fail, x[s]) for s in ("h_med", "h_p99", "h_answ")}
    best_h = max(v for v in a_h.values() if v == v)
    d4 = ("SEQUENCE_PROXY" if a_tf >= D4_STRONG and a_tf - best_h >= D4_MARGIN else
          "WEAK" if a_tf >= D4_WEAK else "NONE")
    rep = {}
    s1 = os.path.join(HERE, "stage1.json")
    for r in results:
        if r["cell"].endswith("131072") and os.path.exists(s1):
            u = json.load(open(s1))["unit"]
            for k in ("uniform@3", "router_calib@4"):
                if k in u["points"] and k in r["points"]:
                    rep[k] = (u["points"][k]["score"], r["points"][k]["score"])
            rep["fp"] = (u["fp_score"], r["fp"])
    out = dict(decision=dec, v4_win_cells=v4win, v16_win_cells=v16win,
               d4=dict(verdict=d4, auc_tf=a_tf, auc_head=a_h, n_rows=int(len(x)),
                       fail_rate=float(fail.mean())),
               replication_stage1_128k=rep, cells=results)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    Lh = ["# R14 Stage 1b (read_stage1b.py; rules frozen in its docstring)", "",
          f"**D2 decision: {dec}** · V4 WIN cells: {', '.join(v4win) or 'none'} · "
          f"V16 WIN cells: {', '.join(v16win) or 'none'}",
          f"**D4 (proxy): {d4}** · AUC teacher-forced {fmt(a_tf)} vs per-head "
          + ", ".join(f"{k} {fmt(v)}" for k, v in a_h.items()) + f" ({len(x)} rows)", ""]
    for r in results:
        Lh += [f"## {r['cell']} — FP {r['fp']:.3f}, {r['n_prompts']} prompts, tasks "
               f"{', '.join(r['valid_tasks'])}, fp replay agreement {r['fp_replay_agreement']:.3f}", "",
               "| lens | D* | S* (SIEVE/POOL) | H* (hybrid) | J* | ρ = J*/D* | D2 | r1 = S*/H* | D1 |",
               "|---|---|---|---|---|---:|---|---:|---|"]
        for lens, z in r["lenses"].items():
            Lh.append(f"| {lens} | {z['D_star'] or '—'} ({fmt(z['bytes_D'], 1)}) | "
                      f"{z['S_star'] or '—'} ({fmt(z['bytes_S'], 1)}) | "
                      f"{z['H_star'] or '—'} ({fmt(z['bytes_H'], 1)}) | {z['J_star'] or '—'} | "
                      f"{fmt(z['rho'])} | {z['d2']} | {fmt(z['r1'])} | {z['d1']} |")
        if "d3_verdict" in r:
            z = r["d3_verdict"]
            Lh += ["", f"D3 at B={z['B']}: pooled − standard {z['delta']:+.3f} "
                       f"[{z['lo']:+.3f}, {z['hi']:+.3f}], deletions {z['del_std']} → "
                       f"{z['del_pool']} → **{z['verdict']}**"]
        for B, z in r["d3"].items():
            Lh.append(f"- B={B}: pooled − standard {z['delta']:+.3f} [{z['lo']:+.3f}, "
                      f"{z['hi']:+.3f}], deletions {z['del_std']} → {z['del_pool']}")
        if r.get("fp_value_cost"):
            Lh.append("- value quantizer alone (fp twins): " + ", ".join(
                f"{t} Δ {z['delta']:+.3f} [{z['lo']:+.3f}, {z['hi']:+.3f}]"
                for t, z in r["fp_value_cost"].items()))
        Lh += ["", "| arm@B | lens | score | Δ vs FP | 90% CI | lossless | key bits | "
                   "evicted | bytes | deletions / substitutions |",
               "|---|---|---:|---:|---|---|---:|---:|---:|---|"]
        for k, p in sorted(r["points"].items(), key=lambda kp: (kp[1]["lens"], kp[1]["bytes"])):
            sg = r["signatures"].get(k, {})
            Lh.append(f"| {k} | {p['lens']} | {p['score']:.3f} | {p['delta']:+.3f} | "
                      f"[{p['lo']:+.3f}, {p['hi']:+.3f}] | {'yes' if p['lossless'] else 'no'} | "
                      f"{p['key_bits']:.2f} | {p['evict_frac']:.1%} | {p['bytes']:.1f} | "
                      f"{sg.get('deletion', 0)} / {sg.get('substitution', 0)} |")
        Lh.append("")
    if rep:
        Lh.append("Replication of Stage 1 at 128K (Stage 1 → Stage 1b cell means; "
                  "different prompts): " + ", ".join(f"{k} {a:.3f} → {b:.3f}"
                                                    for k, (a, b) in rep.items()))
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    print("\n".join(Lh[:4]))
    return 0


def gate(job):
    d, heads, side = load_run("pilot128", job)
    side["_job"] = str(job)
    problems = []
    agree = validate(d, [side], problems, main=False)
    pk = float(d.peak_gib.max()) if "peak_gib" in d else float("nan")
    if not pk <= PEAK_GIB_MAX:
        problems.append(f"peak GPU memory {pk} GiB > {PEAK_GIB_MAX}")
    n_main = len(L.build_plan(L.PRESETS["main128"]))
    dec = float(d[d.arm != "fp"].t_arm.median())
    tf = float(d.t_tf.median())
    per = (float(d.t_prefill.median()) + 2 * float(d.t_precompute.max())
           + n_main * (3 * dec + tf))
    proj_h = per * MAIN_BLOCK_PROMPT_TASKS / 3600
    if proj_h > 0.9 * WALL_H["main128"]:
        problems.append(f"projected main128 block {proj_h:.1f} h > 90% of {WALL_H['main128']} h")
    print(f"R14 Stage 1b pilot {job}: {len(d)} rows, plan {len(plan_of(side))} arms, "
          f"fp replay agreement {agree:.3f}, peak {pk:.1f} GiB, median decode {dec:.1f} s, "
          f"median replay {tf:.1f} s, projected main128 block {proj_h:.1f} h")
    print(d.pivot_table(index=["arm", "B"], values=["score", "bits_per_token", "evict_frac",
                                                   "tf_c_min_logp", "t_arm"],
                        aggfunc="mean").round(3).to_string())
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
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1b"))
    a = ap.parse_args()
    if a.pilot and (a.main128 or a.main32):
        ap.error("--pilot or the main blocks, not both")
    if a.pilot:
        sys.exit(gate(a.pilot))
    cells = {}
    if a.main128:
        cells["llama31-8b@131072"] = ("main128", a.main128)
    if a.main32:
        cells["llama31-8b@32768"] = ("main32", a.main32)
    if not cells:
        ap.error("give --pilot JOB or at least one of --main128 / --main32")
    sys.exit(read_main(cells, a.out_stem))


if __name__ == "__main__":
    main()
