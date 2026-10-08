#!/usr/bin/env python3
"""R5.0: one table of where the system's loss comes from, per cell and task, from R1-R4's
finished blocks (no GPU, no model). EXPLORATORY: it describes the runs, it decides nothing.

For each unit (prompt x task) and arm, the change against FP of
  dP      a_span_nll (R1's primary: FP's answer from its first answer-value token to the span end)
  dP/tok  dP divided by that span's token count (the long-answer correction, plan.md R5 theory)
  KL      kl_span (co-primary) and KL/tok
  dAcc    accuracy (lbv2: the forced choice; HELMET re-ranking: NDCG@10)
and the split of the system's loss at r = 1/8 (plan.md R5 theory, part 1):
  budget  qoraclefp_v16 - FP        (the best fixed set of floor(rC) rows, exact keys)
  vote    qreadfp_v16 - qoraclefp_v16   (the question's vote instead of the oracle, exact keys)
  store   qread2t4kq_v4 - qreadfp_v16   (the two-tier store: 4-bit tier 1, exact keys on the set)
The floor system (r = 1/2 at Qwen 32K; qread2t4kqF at R4), Quest (R4) and the dense 4-bit
arms are listed beside it.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/consolidate_r5.py
"""
from __future__ import annotations
import glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "h0_measurement", "results")
OUT = os.path.join(HERE, "findings", "R5_0_split")

CELLS = [   # name, tag, jobs
    ("R1 Llama 128K RULER", "h1cal", ["1032160", "1032161"]),
    ("R2 Qwen 32K RULER", "h2qwen32", ["1032366"]),
    ("R3a Llama 128K harder RULER", "h3llama", ["22594818", "22594819"]),
    ("R3a Qwen 32K harder RULER", "h3qwen", ["22594820", "22594821"]),
    ("R3b Llama 128K cwe/fwe", "h3bllama", ["22560817", "22560818"]),
    ("R3b Qwen 32K cwe/fwe", "h3bqwen", ["22560819", "22560820"]),
    ("R4 Llama 128K LongBench v2", "h4llama128", ["1036792", "1036793"]),
    ("R4 Llama 128K HELMET", "h4llama128", ["1036925", "1036926"]),
]
ARMS = [   # label, arm, B (None = the arm's only B)
    ("FP8 KV", "fp8kv", 8.0), ("noise", "fp_noise", 0.0),
    ("dense KIVI-4", "kivi4_v4", 4.0), ("dense KVQuant-4", "kvquant4_v4", 4.0), ("dense uniform-4", "uniform+v4", 4.0),
    ("oracle 1/8 (exact)", "qoraclefp_v16", 0.125), ("oracle 1/8 (4-bit)", "qoracle4_v4", 0.125),
    ("vote 1/8 (exact)", "qreadfp_v16", 0.125), ("vote 1/8 (4-bit)", "qread4_v4", 0.125),
    ("system 1/8", "qread2t4kq_v4", 0.125), ("system floor r=1/2", "qread2t4kq_v4", 0.5),
    ("system floor (R4)", "qread2t4kqF_v4", 0.125),
    ("Quest exact 1/8", "quest_v16", 0.125), ("Quest 4-bit 1/8", "quest4_v4", 0.125),
]
KEY = ["job", "prompt_idx", "task"]


def load(tag, jobs):
    parts = []
    for j in jobs:
        d = os.path.join(RES, f"r14s1h_{tag}_{j}")
        pq = [p for p in glob.glob(os.path.join(d, "*_evaluate_*.parquet")) if not p.endswith(("_search.parquet",
                                                                                                "_searchlog.parquet"))]
        if len(pq) != 1:
            raise SystemExit(f"{d}: expected one evaluate parquet, found {pq}")
        parts.append(pd.read_parquet(pq[0]).assign(job=j))
    x = pd.concat(parts, ignore_index=True)
    x["B"] = x.B.astype(float).round(4)
    if "fc_correct" in x:                                       # lbv2: the forced choice is the accuracy
        m = x.fc_correct.notna()
        x.loc[m, "score"] = x.loc[m, "fc_correct"].astype(float)
    return x


def span_tokens(fp):
    """Per unit, the token count of FP's span (first answer-value token .. span end)."""
    def n(row):
        vm = str(row.get("tf_vmask") or "")
        if "1" not in vm:
            return np.nan
        return max(int(row["span_end"]) - vm.index("1"), 1)
    return fp.apply(n, axis=1)


def cell_table(x):
    fp = x[x.arm == "fp"].set_index(KEY)
    ntok = span_tokens(fp.reset_index()).set_axis(fp.index)
    rows = []
    for task in sorted(x.task.unique()):
        ft = fp.xs(task, level="task", drop_level=False)
        rec = dict(task=task, n=len(ft), fp_acc=float(ft.score.mean()),
                   span_tok=float(ntok.reindex(ft.index).mean()))
        for label, arm, B in ARMS:
            g = x[(x.arm == arm) & (x.B == B) & (x.task == task)].set_index(KEY)
            if not len(g):
                continue
            idx = g.index.intersection(ft.index)
            dp = (g.a_span_nll.reindex(idx) - ft.a_span_nll.reindex(idx))
            nt = ntok.reindex(idx)
            rec[label] = dict(dP=float(dp.mean()), dP_tok=float((dp / nt).mean()),
                              KL=float(g.kl_span.reindex(idx).mean()),
                              KL_tok=float((g.kl_span.reindex(idx) / nt).mean()),
                              dAcc=float((g.score.reindex(idx) - ft.score.reindex(idx)).mean()), n=int(len(idx)))
        def d(a, b, k="dP"):
            return rec[a][k] - rec[b][k] if a in rec and b in rec else np.nan
        rec["split"] = dict(budget=rec.get("oracle 1/8 (exact)", {}).get("dP", np.nan),
                            vote=d("vote 1/8 (exact)", "oracle 1/8 (exact)"),
                            store=d("system 1/8", "vote 1/8 (exact)"),
                            budget_tok=rec.get("oracle 1/8 (exact)", {}).get("dP_tok", np.nan),
                            vote_tok=d("vote 1/8 (exact)", "oracle 1/8 (exact)", "dP_tok"),
                            store_tok=d("system 1/8", "vote 1/8 (exact)", "dP_tok"))
        rows.append(rec)
    return rows


def _f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{nd}f}"


def main():
    out, md = {}, ["# R5.0 — where the system's loss comes from (R1–R4, exploratory)", "",
                   "Change against FP per unit, averaged. dP = span NLL (nats), /tok = per span token; "
                   "split at r = 1/8: budget = oracle − FP, vote = vote(exact) − oracle, store = system − vote(exact).",
                   ""]
    for name, tag, jobs in CELLS:
        try:
            x = load(tag, jobs)
        except SystemExit as e:
            md += [f"## {name}", "", f"not available: {e}", ""]
            continue
        rows = cell_table(x)
        out[name] = rows
        md += [f"## {name} (jobs {', '.join(jobs)})", "",
               "| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            s = r["split"]
            md.append(f"| {r['task']} | {r['n']} | {r['fp_acc']:.3f} | {r['span_tok']:.0f} | {_f(s['budget'])} | "
                      f"{_f(s['vote'])} | {_f(s['store'])} | {_f(s['budget_tok'], 3)} | {_f(s['vote_tok'], 3)} | "
                      f"{_f(s['store_tok'], 3)} |")
        md += ["", "| task | arm | dP | dP/tok | KL | KL/tok | dAcc |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            for label, _, _ in ARMS:
                if label in r:
                    v = r[label]
                    md.append(f"| {r['task']} | {label} | {_f(v['dP'])} | {_f(v['dP_tok'], 3)} | {_f(v['KL'])} | "
                              f"{_f(v['KL_tok'], 3)} | {_f(v['dAcc'], 3)} |")
        md.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(OUT + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:4]))
    for name, rows in out.items():
        print(f"\n{name}")
        for r in rows:
            s = r["split"]
            print(f"  {r['task']:20s} n={r['n']:3d} FP {r['fp_acc']:.2f} tok {r['span_tok']:5.0f}  budget {_f(s['budget'])}"
                  f"  vote {_f(s['vote'])}  store {_f(s['store'])}   per tok: {_f(s['budget_tok'], 3)} "
                  f"{_f(s['vote_tok'], 3)} {_f(s['store_tok'], 3)}")
    print(f"\nwrote {OUT}.md / .json")


if __name__ == "__main__":
    sys.exit(main())
