#!/usr/bin/env python3
"""Appendix end-task tables (LaTeX) from the R8 accuracy parquets.

    cd latex/figures && ../../.venv/bin/python make_endtask_tables.py [parquets...]

Default input: the R9 held-out campaign (jobs 978480-978489). Writes
../tab_endtask_grid.tex, ../tab_endtask_h2h.tex and ../tab_endtask_fail.tex.
Statistics come from h0_measurement/bugs/12_paper_main_table/read_main_table.py,
so the paper and the campaign reader cannot disagree.
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "h0_measurement", "bugs", "12_paper_main_table"))
import read_main_table as RM  # noqa: E402

TEX = {"uniform": "TurboQ.", "kivi": "KIVI-32", "kivi_g128": "KIVI", "kvquant": "KVQuant",
       "evict_h2o": "H2O", "evict": "SnapKV", "dropkv": "DropKV", "adakv": "Ada-KV",
       "laprox": "LaProx", "obck_ada": "OBC+Ada", "interior_cascade": "\\sieve{}-int.",
       "router_calib": "\\sieve{}", "interior_pool": "pool (diag.)",
       "router_oracle": "oracle (diag.)"}
TASK = {"niah_single": "single", "niah_multikey": "multi-key",
        "niah_multivalue": "multi-value", "vt": "var.\\ track."}
MODEL = {"llama31-8b": "Llama-3.1-8B", "qwen3-8b": "Qwen3-8B"}


def grid(d):
    cm = d[d.arm != "fp"].groupby(RM.CELL + ["arm"]).score.mean().unstack("arm")
    fp = d[d.arm == "fp"].groupby(["model", "ctx", "task"]).score.mean()
    arms = [a for a in RM.ORDER if a in cm and RM.FAMILY.get(a) != "diagnostic"]
    out = ["\\begin{tabular}{@{}llllr" + "r" * len(arms) + "@{}}", "\\toprule",
           "model & ctx & task & $B$ & FP & " + " & ".join(TEX[a] for a in arms) + " \\\\",
           "\\midrule"]
    prev = None
    for (m, c, t, B), row in cm.iterrows():
        f = fp.get((m, c, t), np.nan)
        best = max(row[a] for a in arms)
        cells = []
        for a in arms:
            s = f"{row[a]:.2f}"
            cells.append(f"\\textbf{{{s}}}" if row[a] >= best - 1e-9 and f >= RM.FP_MIN else s)
        if prev is not None and (m, c) != prev:
            out.append("\\addlinespace[1pt]")
        prev = (m, c)
        dag = "$^\\ddagger$" if f < RM.FP_MIN else ""
        out.append(f"{MODEL[m]} & {c // 1024}k & {TASK[t]}{dag} & {B} & {f:.2f} & "
                   + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def h2h(d):
    cm = RM.cell_means(d)
    ref = "router_calib"
    out = ["\\begin{tabular}{@{}lrrrrl@{}}", "\\toprule",
           "\\sieve{} (router) vs. & wins & ties & losses & mean $\\Delta$ & 90\\% CI \\\\",
           "\\midrule"]
    for a in [x for x in RM.ORDER if x in cm and x != ref
              and RM.FAMILY.get(x) not in ("diagnostic",) and x != "interior_cascade"]:
        diff = (cm[ref] - cm[a]).dropna()
        w, l = int((diff > RM.MARGIN).sum()), int((diff < -RM.MARGIN).sum())
        p, lo, hi = RM.paired_boot(d, ref, a)
        out.append(f"{RM.NAMES[a]} & {w} & {len(diff) - w - l} & {l} & ${p:+.3f}$ & "
                   f"$[{lo:+.3f}, {hi:+.3f}]$ \\\\")
    ev = [a for a in cm if a in RM.EVICTORS]
    diff = (cm[ref] - cm[ev].max(axis=1)).dropna()
    w, l = int((diff > RM.MARGIN).sum()), int((diff < -RM.MARGIN).sum())
    out.append(f"best eviction method per cell & {w} & {len(diff) - w - l} & {l} & "
               f"${diff.mean():+.3f}$ & --- \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def fail(d):
    x = d[(d.model == "llama31-8b") & (d.ctx == 131072) & (d.arm != "fp")]
    ref = d[d.arm == "fp"].set_index(["model", "ctx", "task", "prompt_idx"]).pred

    def kind(r):
        if r.score >= 0.999:
            return "correct"
        if r.distractor:
            return "distractor"
        a = (re.findall(r"\d+", ref.get((r.model, r.ctx, r.task, r.prompt_idx), "")) or [""])[0]
        got = re.findall(r"\d+", r.pred)
        if not got or not a:
            return "other"
        return ("truncated" if any(len(g) >= 3 and g != a and a.startswith(g[:3]) for g in got)
                else "other")

    k = x[x.task.isin(["niah_single", "niah_multikey"])].copy()
    k["kind"] = k.apply(kind, axis=1)
    out = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
           " & \\multicolumn{2}{c}{task score} & \\multicolumn{3}{c}{single/multi-key answers, $B{=}2,3$}"
           " & evicted & head-output & heads to \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
           "method & $B{=}2$ & $B{=}3$ & truncated & wrong needle & other & ($B{=}2$) & error ($B{=}2$) & interior \\\\",
           "\\midrule"]
    for a in [a for a in RM.ORDER if a in set(x.arm)]:
        y2, y3 = x[(x.arm == a) & (x.B == 2)], x[(x.arm == a) & (x.B == 3)]
        kk = k[k.arm == a].kind.value_counts(normalize=True)
        fi = y2.frac_interior.mean() if "frac_interior" in y2 and y2.frac_interior.notna().any() else np.nan
        out.append(f"{RM.NAMES[a]} & {y2.score.mean():.3f} & {y3.score.mean():.3f} & "
                   f"{kk.get('truncated', 0):.0%} & {kk.get('distractor', 0):.0%} & "
                   f"{kk.get('other', 0):.0%} & {y2.evict_frac.mean():.0%} & "
                   f"{y2.head_err_mean.mean():.3f} & {'---' if np.isnan(fi) else f'{fi:.0%}'} \\\\"
                   .replace("%", "\\%"))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    paths = sys.argv[1:] or [f for j in ("978480", "978483", "978485", "978487", "978489")
                             for f in glob.glob(os.path.join(ROOT, "h0_measurement", "results",
                                                             f"r9job{j}", "r8_*.parquet"))]
    d = RM.load(paths)
    for name, fn in (("grid", grid), ("h2h", h2h), ("fail", fail)):
        with open(os.path.join(HERE, "..", f"tab_endtask_{name}.tex"), "w") as fh:
            fh.write(f"% generated by figures/make_endtask_tables.py from {len(paths)} parquets\n")
            fh.write(fn(d).replace("SIEVE", "\\sieve{}") + "\n")
    print("wrote tab_endtask_{grid,h2h,fail}.tex")


if __name__ == "__main__":
    main()
