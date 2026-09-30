#!/usr/bin/env python3
"""R14 Stage 1 gate and reader (plan.md §5 and amendment A1).

    python read_stage1.py --pilot PILOT_JOB            # mechanics gate; exit 1 = stop the chain
    python read_stage1.py --main JOB_A JOB_B [--out-stem stage1]

--pilot checks the excluded pilot (results/r14s1pilot<JOB>/ and logs/r8_<JOB>.out):
every planned row present once, fp8kv audited at 8 bits with nothing evicted,
every '<arm>+v8' row audited exactly like the base row it follows, FP8 values
marked, peak GPU memory, and a projection of the main job's wall time.

--main reads the two evaluation blocks (results/r14s1job<JOB>/) and applies the
frozen Stage 1 decisions: SIEVE lossless at some B <= 4 re-enters Stage 0 for
Llama @128K with B in {2,3,4}; otherwise KILL_128K. It also measures whether FP8
KV is lossless and whether FP8 values cost any paired arm more than 0.02.
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_model as BM  # noqa: E402

V8 = "+v8"
FP8KV = "fp8kv"
UNIT = "llama31-8b@128K"
PEAK_GIB_MAX = 76.0          # of 79.2 GiB usable on an H100-80GB
MAIN_WALL_H = 10.0           # script_stage1.sh --time for each main block
MAIN_TASKS, MAIN_PROMPTS = 4, 10
V8_DROP_MAX = 0.02           # plan §5: FP8 V may lower an arm's score by at most this


def is_r14(arm: str) -> bool:
    return arm == FP8KV or arm.endswith(V8)


def run_dir(prefix, job):
    d = os.path.join(BM.H0, "results", f"{prefix}{job}")
    pq = [f for f in os.listdir(d) if f.startswith("r8_") and f.endswith(".parquet")]
    js = [f for f in os.listdir(d) if f.startswith("r8_") and f.endswith(".json")]
    if len(pq) != 1 or len(js) != 1:
        raise SystemExit(f"{d}: expected one r8_*.parquet and one r8_*.json")
    return pd.read_parquet(os.path.join(d, pq[0])), json.load(open(os.path.join(d, js[0])))


def expected_plan(side):
    """The plan run_r8 builds from the sidecar's arms, budgets and R14 options."""
    sys.path.insert(0, os.path.join(BM.H0))
    import run_r8 as RR
    from sievelib import router, kv_quant_baselines as QB
    bl = sorted(side["config"].get("bit_list", [1, 2, 3, 4, 5, 6, 8]))
    maxb = max(bl)
    arms, budgets = side["arms"], side["budgets"]
    plan = [("fp", 0)] if "fp" in arms else []
    plan += [(a, B) for a in arms if a != "fp" for B in budgets
             if not ((a == "uniform" or QB.is_arm(a)) and not router.is_width(B, maxb, bl))]
    r = side["r14_stage1"]
    return RR.r14_plan(plan, r["fp8kv"], r["v_fp8_arms"], r["v_fp8_budgets"])


def audit_rows(d, side, problems):
    """Checks shared by the gate and the reader."""
    plan = expected_plan(side)
    got = d.groupby(["prompt_idx", "task"]).apply(lambda g: sorted(zip(g.arm, g.B)))
    want = sorted(plan)
    bad = [k for k, v in got.items() if v != want]
    if bad:
        problems.append(f"(prompt, task) blocks whose arms differ from the plan: {bad[:3]}")
    if d.duplicated(["prompt_idx", "task", "arm", "B"]).any():
        problems.append("duplicate (prompt, task, arm, B) rows")
    f8 = d[d.arm == FP8KV]
    if len(f8) and not ((f8.bits_per_token == 8.0).all() and (f8.evict_frac == 0).all()):
        problems.append("fp8kv rows are not 8 bits with nothing evicted")
    key = ["prompt_idx", "task", "B"]
    for arm in sorted({a for a in d.arm if a.endswith(V8)}):
        base = arm[:-len(V8)]
        m = d[d.arm == arm].merge(d[d.arm == base], on=key, suffixes=("", "_b"))
        if len(m) != (d.arm == arm).sum():
            problems.append(f"{arm}: {len(m)} of {(d.arm == arm).sum()} rows have a base row")
        elif not (np.array_equal(m.bits_per_token, m.bits_per_token_b)
                  and np.array_equal(m.evict_frac, m.evict_frac_b)):
            problems.append(f"{arm}: bits or eviction differ from {base} (keys not reused)")
    if "v_format" not in d:
        problems.append("no v_format column (R14 options did not reach run_r8)")
    else:
        wrong = d[(d.v_format == "fp8_e4m3") != d.arm.map(is_r14)]
        if len(wrong):
            problems.append(f"{len(wrong)} rows with the wrong v_format")
    return plan


def peak_gib(job):
    log = os.path.join(BM.H0, "logs", f"r8_{job}.out")
    if not os.path.exists(log):
        return None
    vals = [float(x) for x in re.findall(r"peak ([0-9.]+) GiB", open(log).read())]
    return max(vals) if vals else None


def gate(job):
    d, side = run_dir("r14s1pilot", job)
    problems = []
    if "r14_stage1" not in side:
        raise SystemExit("GATE FAIL: sidecar has no r14_stage1 record")
    plan = audit_rows(d, side, problems)
    pk = peak_gib(job)
    if pk is None:
        problems.append("no 'peak ... GiB' line in the job log")
    elif pk > PEAK_GIB_MAX:
        problems.append(f"peak GPU memory {pk:.1f} GiB > {PEAK_GIB_MAX}")
    # wall-time projection for one main block (plan A1: 30 decodes per prompt-task)
    first = d[(d.arm != "fp") & ~d.arm.map(is_r14)].groupby(["prompt_idx", "task"]).t_arm.max()
    dec = d[d.arm != "fp"].groupby("arm").t_arm.median()
    per_decode = float(dec[dec < dec.max()].median()) if len(dec) > 1 else float(dec.max())
    per_pair = float(d.t_prefill.median() + d.t_precompute.max()
                     + 30 * max(per_decode, 3.0) + 60.0)      # +60 s: KIVI-32 / KVQuant decodes
    proj_h = per_pair * MAIN_TASKS * MAIN_PROMPTS / 3600
    if proj_h > 0.9 * MAIN_WALL_H:
        problems.append(f"projected main block {proj_h:.1f} h > 90% of {MAIN_WALL_H} h")
    print(f"R14 Stage 1 pilot {job}: {len(d)} rows, plan of {len(plan)} arms per prompt-task")
    print(d.pivot_table(index=["arm", "B"], values=["score", "bits_per_token", "evict_frac",
                                                   "t_arm"], aggfunc="mean").round(3).to_string())
    print(f"prefill {d.t_prefill.median():.0f} s, precompute {d.t_precompute.max():.0f} s, "
          f"median decode {per_decode:.1f} s, peak {pk} GiB, "
          f"projected main block {proj_h:.1f} h (limit {MAIN_WALL_H} h)")
    if problems:
        print("GATE FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASS (mechanics only: the pilot is excluded from every result)")
    return 0


# ---------------------------------------------------------------- main reader
def paired(d, a, b, B_a, B_b, w, index):
    """Point and 90% interval of score(a@B_a) - score(b@B_b), per-prompt task means."""
    v = d[d.valid]
    def col(arm, B):
        s = v[(v.arm == arm) & (v.B == B)].groupby(["job", "prompt_idx"]).score.mean()
        return s.reindex(index).to_numpy()
    diff = col(a, B_a) - col(b, B_b)
    boot = (w @ diff) / w.sum(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    return float(diff.mean()), float(lo), float(hi)


def read_main(jobs, out_stem):
    frames, sides = [], []
    for j in jobs:
        d, side = run_dir("r14s1job", j)
        d["unit"], d["job"] = UNIT, j
        frames.append(d)
        sides.append(side)
    d = pd.concat(frames, ignore_index=True)
    problems = []
    for side, j in zip(sides, jobs):
        audit_rows(d[d.job == j], side, problems)
    corpora = set(d.corpus_sha)
    if len(corpora) != 1:
        problems.append(f"blocks disagree on corpus: {corpora}")
    if problems:
        raise SystemExit("INVALID Stage 1 data:\n  " + "\n  ".join(problems))
    corpus = corpora.pop()
    # --- Stage 0 machinery on the base arms, B in {2, 3, 4}
    base = BM.load({UNIT: jobs}, corpus=corpus, prefix="r14s1job", drop_arms=is_r14)
    res = BM.analyse_unit(base)
    # --- the R14 rows, paired within each block
    full = d.join(d[d.arm == "fp"].groupby("task").score.mean().rename("fp"), on="task")
    full["valid"] = full.fp >= BM.RMT.FP_MIN
    m = BM.prompt_matrix(base.assign(unit=UNIT))
    w = BM.boot_weights(m.index)
    f8 = paired(full, FP8KV, "fp", 8, 0, w, m.index)
    fp8_lossless = f8[0] >= BM.LOSSLESS_POINT - 1e-12 and f8[1] >= BM.LOSSLESS_LOWER
    pairs = {}
    for arm, B in sorted({(a, b) for a, b in zip(full.arm, full.B) if a.endswith(V8)}):
        base_arm = arm[:-len(V8)]
        pt, lo, hi = paired(full, arm, base_arm, B, B, w, m.index)
        pairs[f"{base_arm}@{B}"] = dict(delta=pt, lo=lo, hi=hi, withdrawn=pt < -V8_DROP_MAX)
    # --- decisions (plan §5, A1)
    L16, L8 = res["lenses"]["V16"], res["lenses"]["V8"]
    S, Dk = L16["S_star"], L16["D_star"]
    if S is None:
        decision = "KILL_128K"
        v8_status = "n/a"
        unit_verdict = "NO_POINT"
    else:
        tw = [pairs.get(x) for x in (S, Dk)]
        if any(t is not None and t["withdrawn"] for t in tw):
            v8_status = "WITHDRAWN"
            unit_verdict = L16["verdict"]
        else:
            v8_status = "measured" if all(t is not None for t in tw) else "assumed for " + \
                ", ".join(x for x, t in zip((S, Dk), tw) if t is None)
            unit_verdict = L16["verdict"] if L16["verdict"] == L8["verdict"] else "BYTES_TIE"
        decision = "REENTER_STAGE0"
    out = dict(jobs=jobs, corpus=corpus, unit=res, fp8kv=dict(delta=f8[0], lo=f8[1], hi=f8[2],
               lossless=bool(fp8_lossless)), v8_pairs=pairs, decision=decision,
               unit_verdict=unit_verdict, v8_lens=v8_status)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    L = ["# R14 Stage 1 — Llama-3.1-8B @128K at B = 2, 3, 4 (read_stage1.py)", "",
         f"Blocks {', '.join(jobs)} · corpus {corpus[:8]} · {res['n_prompts']} prompts · "
         f"valid tasks {', '.join(res['valid_tasks'])} · FP {res['fp_score']:.3f}", "",
         f"**Decision: {decision}**" + (f" · unit verdict **{unit_verdict}** (V8 lens: "
                                      f"{v8_status})" if S else ""), "",
         f"- SIEVE S*: {S or 'none lossless at B <= 4'}; dense D*: {Dk}; "
         f"ρ V16 {BM.fmt(L16['rho'])}, V8 {BM.fmt(L8['rho'])}",
         f"- FP8 KV vs FP: Δ {f8[0]:+.3f} [{f8[1]:+.3f}, {f8[2]:+.3f}] → "
         f"{'lossless' if fp8_lossless else 'NOT lossless: FP8 drops out as a comparator'}", "",
         "| FP8-value pair | Δ(+v8 − base) | 90% CI | V8 lens |", "|---|---:|---|---|"]
    for k, p in pairs.items():
        L.append(f"| {k} | {p['delta']:+.3f} | [{p['lo']:+.3f}, {p['hi']:+.3f}] | "
                 f"{'withdrawn' if p['withdrawn'] else 'kept'} |")
    L += ["", "| arm@B | score | Δ vs FP | 90% CI | lossless | bits+side | evicted | "
          "bytes V16 | bytes V8 |", "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for k, p in sorted(res["points"].items(), key=lambda kp: kp[1]["bytes_V8"]):
        L.append(f"| {k} | {p['score']:.3f} | {p['delta']:+.3f} | [{p['lo']:+.3f}, "
                 f"{p['hi']:+.3f}] | {'yes' if p['lossless'] else 'no'} | "
                 f"{p['bits'] + p['side']:.3f} | {p['evict_frac']:.1%} | "
                 f"{p['bytes_V16']:.1f} | {p['bytes_V8']:.1f} |")
    with open(out_stem + ".md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="JOB")
    ap.add_argument("--main", nargs=2, metavar="JOB")
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage1"))
    a = ap.parse_args()
    if bool(a.pilot) == bool(a.main):
        ap.error("give exactly one of --pilot JOB or --main JOB_A JOB_B")
    sys.exit(gate(a.pilot) if a.pilot else read_main(a.main, a.out_stem))


if __name__ == "__main__":
    main()
