#!/usr/bin/env python3
"""Paper tables from R8 accuracy parquets (plan.md readouts 1-4).

    python read_main_table.py <r8_*.parquet> ... [--out tables.md]

Works on the R9 files (jobs 978480-978489) and on the R12-paper files; arms a
file does not have are simply absent from its rows. Only files evaluated on the
same prompts should be pooled -- the caller chooses them.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

D = 128                                   # head_dim of both models in the table
NAMES = {
    "fp": "Full precision (16-bit)",
    "uniform": "TurboQuant-MSE",
    "kivi": "KIVI (g=32)", "kivi_g128": "KIVI (g=128)", "kvquant": "KVQuant-style",
    "evict": "SnapKV", "evict_h2o": "H2O", "adakv": "Ada-KV", "dropkv": "DropKV",
    "obck_ada": "OBCache-K + Ada-KV", "laprox": "LaProx",
    "router_calib": "SIEVE (router)", "interior_cascade": "SIEVE interior only",
    "interior_pool": "SIEVE interior, pooled score (diag.)",
    "router_oracle": "SIEVE oracle router (diag.)",
}
FAMILY = {
    "uniform": "quantization", "kivi": "quantization", "kivi_g128": "quantization",
    "kvquant": "quantization", "evict": "eviction", "evict_h2o": "eviction",
    "adakv": "eviction", "dropkv": "eviction", "obck_ada": "eviction", "laprox": "eviction",
    "router_calib": "SIEVE", "interior_cascade": "SIEVE (ablation)",
    "interior_pool": "diagnostic", "router_oracle": "diagnostic",
}
ORDER = ["uniform", "kivi_g128", "kivi", "kvquant", "evict_h2o", "evict", "dropkv", "adakv",
         "laprox", "obck_ada", "interior_cascade", "router_calib", "interior_pool",
         "router_oracle"]
EVICTORS = {"evict", "evict_h2o", "adakv", "dropkv", "obck_ada", "laprox"}
SIEVE = {"router_calib", "interior_cascade", "interior_pool", "router_oracle"}
MARGIN = 0.05
FP_MIN = 0.9
CELL = ["model", "ctx", "task", "B"]


def side_bits(row) -> float:
    """Side information per key element (plan.md table). Code bits are B."""
    arm, ef = row["arm"], row["evict_frac"]
    if arm == "uniform":
        return 16 / D
    if arm in EVICTORS:
        return (1 - ef) * 16 / D + 1 / D           # norm per kept token + keep bitmap
    if arm in SIEVE:
        return (1 - ef) * 16 / D + 3 / D           # norm per kept token + 3-bit width index
    return float(row.get("side_bits", np.nan))


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        if "side_bits" not in d:
            d["side_bits"] = np.nan
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    fp = d[d.arm == "fp"].groupby(["model", "ctx", "task"]).score.mean().rename("fp")
    d = d.join(fp, on=["model", "ctx", "task"])
    d["valid"] = d.fp >= FP_MIN
    d["side"] = d.apply(side_bits, axis=1)
    return d


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def cell_means(d):
    c = d[(d.arm != "fp") & d.valid].groupby(CELL + ["arm"]).score.mean().unstack("arm")
    return c


def main_table(d):
    v = d[d.valid & (d.arm != "fp")]
    arms = [a for a in ORDER if a in set(v.arm)]
    cm = cell_means(d)
    out = ["| family | method | mean score | Llama-3.1-8B | Qwen3-8B | B=2 | B=3 | "
           "head-output error | evicted | side info (bit/elem) | effective bits (B=2 / B=3) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fpv = d[(d.arm == "fp") & d.valid]
    out.append(f"| — | {NAMES['fp']} | {fmt(fpv.groupby(['model','ctx','task']).score.mean().mean())}"
               " | | | | | — | 0% | — | 16 |")
    for a in arms:
        x = v[v.arm == a]
        by_m = x.groupby("model").score.mean()
        by_b = x.groupby("B").score.mean()
        side = x.groupby("B").side.mean()
        eff = " / ".join(fmt(b + side.get(b, np.nan), 2) for b in (2, 3))
        out.append(
            f"| {FAMILY.get(a, '')} | {NAMES.get(a, a)} | **{fmt(cm[a].mean())}** | "
            f"{fmt(by_m.get('llama31-8b', np.nan))} | {fmt(by_m.get('qwen3-8b', np.nan))} | "
            f"{fmt(by_b.get(2, np.nan))} | {fmt(by_b.get(3, np.nan))} | "
            f"{fmt(x.head_err_mean.mean())} | {x.evict_frac.mean():.0%} | "
            f"{fmt(x.side.mean(), 2)} | {eff} |")
    n = len(cm)
    return "\n".join(out), n


def paired_boot(d, a, b, reps=2000, seed=0):
    """Mean over valid cells of (a - b), with a 90% interval from resampling
    prompt indices within each (model, ctx) -- the unit that shares a haystack."""
    v = d[d.valid & d.arm.isin([a, b])]
    w = v.pivot_table(index=["model", "ctx", "task", "B", "prompt_idx"], columns="arm",
                      values="score").dropna()
    if w.empty:
        return np.nan, np.nan, np.nan
    w["diff"] = w[a] - w[b]
    point = w.groupby(level=[0, 1, 2, 3])["diff"].mean().mean()
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in w.groupby(level=[0, 1])}
    boots = []
    for _ in range(reps):
        parts = []
        for (m, c), g in groups.items():
            prompts = g.index.get_level_values("prompt_idx").unique()
            pick = rng.choice(prompts, len(prompts), replace=True)
            gg = pd.concat([g.xs(p, level="prompt_idx", drop_level=False) for p in pick])
            parts.append(gg.groupby(level=[0, 1, 2, 3])["diff"].mean())
        boots.append(pd.concat(parts).mean())
    lo, hi = np.percentile(boots, [5, 95])
    return point, lo, hi


def head_to_head(d, ref="router_calib"):
    cm = cell_means(d)
    if ref not in cm:
        return "(no SIEVE router arm)"
    out = [f"| {NAMES[ref]} vs | wins | ties | losses | mean Δ | 90% CI (prompt bootstrap) |",
           "|---|---:|---:|---:|---:|---|"]
    for a in [x for x in ORDER if x in cm and x != ref and FAMILY.get(x) != "diagnostic"
              and x != "interior_cascade"]:
        diff = (cm[ref] - cm[a]).dropna()
        w, l = int((diff > MARGIN).sum()), int((diff < -MARGIN).sum())
        p, lo, hi = paired_boot(d, ref, a)
        out.append(f"| {NAMES[a]} | {w} | {len(diff) - w - l} | {l} | {p:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    best_ev = cm[[a for a in cm if a in EVICTORS]].max(axis=1)
    diff = (cm[ref] - best_ev).dropna()
    w, l = int((diff > MARGIN).sum()), int((diff < -MARGIN).sum())
    out.append(f"| best eviction baseline, per cell | {w} | {len(diff) - w - l} | {l} | "
               f"{diff.mean():+.3f} | (per-cell max: no single paired arm) |")
    qa = [a for a in cm if FAMILY.get(a) == "quantization"]
    diff = (cm[ref] - cm[qa].max(axis=1)).dropna()
    w, l = int((diff > MARGIN).sum()), int((diff < -MARGIN).sum())
    out.append(f"| best quantization baseline, per cell | {w} | {len(diff) - w - l} | {l} | "
               f"{diff.mean():+.3f} | (per-cell max) |")
    return "\n".join(out)


def grid(d):
    cm = cell_means(d)
    fp = d[d.arm == "fp"].groupby(["model", "ctx", "task"]).score.mean()
    arms = [a for a in ORDER if a in cm]
    hdr = "| model | ctx | task | B | FP | " + " | ".join(NAMES[a] for a in arms) + " |"
    out = [hdr, "|" + "---|" * 5 + "---:|" * len(arms)]
    allc = d[d.arm != "fp"].groupby(CELL + ["arm"]).score.mean().unstack("arm")
    for key, row in allc.iterrows():
        m, c, t, B = key
        f = fp.get((m, c, t), np.nan)
        best = max(row.get(a, -1) for a in arms if FAMILY.get(a) != "diagnostic")
        cells = []
        for a in arms:
            s = row.get(a, np.nan)
            txt = fmt(s, 2)
            cells.append(f"**{txt}**" if not np.isnan(s) and s >= best - 1e-9
                         and FAMILY.get(a) != "diagnostic" else txt)
        tag = "" if f >= FP_MIN else " †"
        out.append(f"| {m} | {c // 1024}K | {t}{tag} | {B} | {fmt(f, 2)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def failure(d, model="llama31-8b", ctx=131072):
    x = d[(d.model == model) & (d.ctx == ctx)]
    if x.empty:
        return "(no 128K cell)"
    arms = [a for a in ORDER if a in set(x.arm)]
    def kind(r):
        if r.score >= 0.999:
            return "correct"
        return "distractor" if r.distractor else "corrupted/other"
    x = x.assign(kind=x.apply(kind, axis=1))
    out = ["| method | B | score | wrong: distractor | wrong: corrupted/other | evicted | "
           "head-output error | heads routed to interior |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        for B in (2, 3):
            y = x[(x.arm == a) & (x.B == B)]
            if y.empty:
                continue
            k = y.kind.value_counts(normalize=True)
            fi = y.frac_interior.mean() if "frac_interior" in y else np.nan
            out.append(f"| {NAMES[a]} | {B} | {fmt(y.score.mean())} | {k.get('distractor', 0):.0%} | "
                       f"{k.get('corrupted/other', 0):.0%} | {y.evict_frac.mean():.0%} | "
                       f"{fmt(y.head_err_mean.mean())} | "
                       f"{'—' if np.isnan(fi) else f'{fi:.0%}'} |")
    txt = "\n".join(out)
    if "interior_pool" in set(x.arm):
        g = x[x.B == 2].groupby("arm").score.mean()
        gap = g["obck_ada"] - g["interior_cascade"]
        rec = (g["interior_pool"] - g["interior_cascade"]) / gap if gap > 0 else np.nan
        verdict = ("supported" if rec >= 0.5 else "refuted" if rec < 0.2 else "inconclusive")
        txt += (f"\n\nH-pool (plan.md, pre-registered): interior_pool recovers {rec:.0%} of the "
                f"interior_cascade -> OBCache-K+Ada-KV gap at B=2 -> **{verdict}**.")
    return txt


def ceiling_gate(d):
    k = d[(d.task == "niah_multikey") & (d.get("n_keys", 4) == 32)] if "n_keys" in d else d.iloc[:0]
    if k.empty:
        return "(no k32 non-ceiling cell yet)"
    fp = k[k.arm == "fp"].score.mean()
    tq = k[(k.arm == "uniform") & (k.B == 2)].score.mean()
    ok = fp >= 0.95 and 0.5 <= tq <= 0.9
    return (f"FP {fp:.3f}, TurboQuant B=2 {tq:.3f} -> "
            f"{'NON-CEILING (gate passed)' if ok else 'gate FAILED: report as still at ceiling'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = load(a.parquets)
    tab, n = main_table(d)
    txt = [f"# Paper tables ({len(a.parquets)} files, {n} valid model/ctx/task/B cells)\n",
           "## Table 1 - main results\n", tab,
           "\n## Table 2 - SIEVE head-to-head (per valid cell, margin 0.05)\n", head_to_head(d),
           "\n## Failure case - Llama-3.1-8B @ 128K\n", failure(d),
           "\n## Non-ceiling gate\n", ceiling_gate(d),
           "\n## Appendix - full grid (best non-diagnostic method per row in bold; † = FP < 0.9, excluded)\n",
           grid(d)]
    s = "\n".join(txt)
    print(s)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(s + "\n")


if __name__ == "__main__":
    main()
