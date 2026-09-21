#!/usr/bin/env python3
"""Export waves 3 and 4 to wave3.csv / wave4.csv beside this file.

    .venv/bin/python h0_measurement/bugs/co-design/build_wave_csv.py

WHAT IS IN THEM. Per-head rows at the step every analysis reads -- the level
every number in report.md is computed from, so any of them can be re-derived
with a groupby and no parquet:

  LEAN / FIVE / LEAN+ / FIVE+ cells   the last quantized step (step 4, dense
                                      decode): one row per (prompt, family,
                                      layer, head). What gqa_cascade.py,
                                      errorbars.py and boundary.py read.
  DRIFT cells (R5)                    EVERY measured step (14, to 4,096): what
                                      drift.py reads, since drift is the step axis.

COLUMNS. Kept: provenance (cell, job, model, ctx, prompt, family, layer, head,
kv_head, n_rep, prompt_offset, rot_seed, corpus_sha, schedule, decode_temp ...),
the budget-independent phase statistics (tau, n95, ladder_bits, dead tiers ...),
and every column for the HEADLINE BUDGET B = 3 (errors, gains, corner errors,
interior lag costs, the 44 co-design columns). Dropped: the B = 1, 2, 4 copies
of the same quantities and the per-width noise-fit diagnostics (c<b>_abs,
alpha<b>, resid<b>, spearman_top_b<b>), which no analysis in this folder reads
and which would triple the width. The parquets remain the full record.

`cell` names the run as the sheets do (N16, X1, N7a, ...); `serves` says which
part of the roadmap it was for.
"""
from __future__ import annotations
import glob, json, os, re, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "h0_measurement", "results")

WAVE3 = {  # job -> (cell, kind, serves)
    "job955334": ("X1", "LEAN", "cost probe: qwen3-30b on ONE GPU"),
    "job955335": ("X2", "DRIFT", "R5: N1's second prompt block (prompts 3-5)"),
    "job958168": ("N16", "LEAN", "R4: qwen3-30b @64k, fills the 32k->128k hole"),
    "job958169": ("N17", "DRIFT", "R5: greedy control @8k"),
    "job958170": ("N18", "DRIFT", "R5: length-axis midpoint @32k"),
}
WAVE4 = {
    "job961033": ("N7a", "FIVE+", "R7 blocks + co-design, llama31-8b @128k"),
    "job961034": ("N7b", "FIVE+", "R7 blocks + co-design, llama31-8b @128k"),
    "job961035": ("N5a", "FIVE+", "R7 blocks + co-design, llama31-8b @32k"),
    "job961036": ("N5b", "FIVE+", "R7 blocks + co-design, llama31-8b @32k"),
    "job961037": ("N6a", "FIVE+", "R7 blocks + co-design, qwen3-8b @8k"),
    "job961038": ("N6b", "FIVE+", "R7 blocks + co-design, qwen3-8b @8k"),
    "job961039": ("N11", "LEAN+", "co-design n_rep=1 CONTROL, qwen15-moe @16k"),
    "job961040": ("N14a", "FIVE+", "R7 blocks + co-design, qwen3-30b @8k"),
    "job961041": ("N14b", "FIVE+", "R7 blocks + co-design, qwen3-30b @8k"),
}

# budget-specific column patterns: keep B = 3, drop B = 1, 2, 4
_BUDGETED = re.compile(
    r"^(err_wf|err_uniform|gain_u|evict_frac|mean_bits|lin_ratio|err_evict|gain_e|"
    r"gain_best|in_band|err_practical|gain_practical|gain_best_practical|"
    r"in_band_practical|best_evictor|oracle_evict_advantage|kstar|kstar_frac|"
    r"kstar_over_n95|corner_tokens|corner_bits_used|err_e|err_wf_pp|"
    r"interior_lag_cost|evict_frac_pp|unseen_frac_pp|gain_u_pp|gain_pp|in_band_pp|"
    r"gain_pp_sym|in_band_pp_sym)(\d)")
_NOISE_FIT = re.compile(r"^(c\d+_(abs|rel)|alpha\d+|resid\d+_abs|evict_beats_b\d+|spearman_top_b\d+)$")
_KEEP_NOISE = {"evict_beats_b1", "evict_beats_b2", "c1_rel", "c2_rel", "c3_rel"}  # dead tiers + their costs


def keep(col: str) -> bool:
    if _NOISE_FIT.match(col):
        return col in _KEEP_NOISE
    m = _BUDGETED.match(col)
    if m:
        return m.group(2) == "3"
    # co-design columns carry the budget as a trailing _3 / 3 -- all were run at B = 3
    return True


def load(job: str, cell: str, kind: str, serves: str) -> pd.DataFrame:
    f = glob.glob(os.path.join(RES, job, "h0_*.parquet"))
    if len(f) != 1:
        raise SystemExit(f"{job}: expected one parquet, found {f}")
    f = f[0]
    side = json.load(open(f[:-8] + ".json"))
    d = pd.read_parquet(f)
    if kind == "DRIFT":
        d = d[d.quantized] if "quantized" in d else d
    else:
        d = d[d.quantized]
        d = d[d.step == d.step.max()]
    d = d[[c for c in d.columns if keep(c)]].copy()
    d.insert(0, "serves", serves)
    d.insert(0, "kind", kind)
    d.insert(0, "job", job)
    d.insert(0, "cell", cell)
    d["corner_tag"] = (side.get("corner") or {}).get("tag")
    d["group_alloc"] = bool(side.get("group_alloc", False))
    d["coarse_bits"] = ",".join(str(b) for b in side.get("coarse_bits", [])) or ""
    d["interior_unseen_policy"] = side.get("interior_unseen_policy")
    return d


def build(spec: dict, out: str):
    parts = []
    for job, (cell, kind, serves) in spec.items():
        d = load(job, cell, kind, serves)
        parts.append(d)
        print(f"  {cell:5s} {job}  {kind:6s} {len(d):>8,} rows  {d.shape[1]} cols  "
              f"{d.model.iloc[0]} @{int(d.ctx.iloc[0]):,}")
    df = pd.concat(parts, ignore_index=True, sort=False)
    # stable column order: provenance first, then the rest alphabetically
    first = ["cell", "job", "kind", "serves", "model", "ctx", "n_rep", "prompt",
             "prompt_offset", "family", "step", "layer", "head", "kv_head",
             "rot_seed", "corpus_sha", "corner_tag", "group_alloc", "coarse_bits",
             "schedule", "decode_temp", "interior_unseen_policy"]
    first = [c for c in first if c in df.columns]
    df = df[first + sorted(c for c in df.columns if c not in first)]
    # 7 significant digits: every statistic here is a ratio, an error or a tau,
    # none needs more, and full float64 repr tripled the file. The n_rep = 1
    # identities survive (both sides round the same way); a gain within 1e-7 of
    # the 2x band line could in principle round across it -- none does in these
    # waves (checked by verify(), below).
    df.to_csv(out, index=False, float_format="%.7g")
    mb = os.path.getsize(out) / 2 ** 20
    print(f"-> {out}\n   {len(df):,} rows x {df.shape[1]} columns, {mb:.1f} MB")
    return df


def verify(spec: dict, path: str) -> bool:
    """Round trip: read the CSV back and re-derive, per cell, the statistics the
    reports quote -- straight from the source parquet as well -- and require
    them to agree. Band COUNTS must match exactly (that is the check that 7-digit
    rounding moved no gain across the 2x line); medians to 1e-5 relative."""
    import numpy as np
    df = pd.read_csv(path, low_memory=False)
    ok = True
    bands = ["gain_pp3_accum", "gain_best_practical3", "gain_best3",
             "gain_grp_pp_accum_3", "gain_grp_csv_b4_accum_3"]
    meds = ["tau", "n95", "interior_lag_cost3_accum", "grp_pp_accum_over_head3",
            "grp_or_cost3", "cs_b4_cost3", "csv_b3_accum_cost3",
            "interior_lag_cost3_last_step", "interior_lag_cost3_first"]
    for job, (cell, kind, _) in spec.items():
        src = load(job, cell, kind, "")
        got = df[df.cell == cell]
        if len(got) != len(src):
            print(f"  FAIL {cell}: {len(got)} rows in the CSV, {len(src)} in the parquet"); ok = False
            continue
        for c in bands:
            if c in src.columns:
                a, b = int((src[c] >= 2).sum()), int((got[c] >= 2).sum())
                if a != b:
                    print(f"  FAIL {cell}: band count {c} {a} (parquet) vs {b} (csv)"); ok = False
        for c in meds:
            if c in src.columns and src[c].notna().any():
                a, b = float(src[c].median()), float(got[c].median())
                if not np.isclose(a, b, rtol=1e-5, atol=1e-9):
                    print(f"  FAIL {cell}: median {c} {a} vs {b}"); ok = False
    if "err_wf_grp_pp_accum_3" in df.columns:          # the n_rep = 1 control
        c1 = df[df.n_rep == 1]
        if len(c1):
            d = float((c1.err_wf_grp_pp_accum_3 - c1.err_wf_pp3_accum).abs().max())
            print(f"  n_rep=1 identity in the CSV: max|group - per-head| = {d:.1e}")
            ok &= d == 0.0
    print(f"  {'OK' if ok else 'FAILED'}: {os.path.basename(path)} reproduces the parquets "
          f"(row counts, band counts exactly, medians to 1e-5)")
    return ok


if __name__ == "__main__":
    print("wave 3 (+ the two extras that landed with it):")
    build(WAVE3, os.path.join(HERE, "wave3.csv"))
    print("\nwave 4:")
    build(WAVE4, os.path.join(HERE, "wave4.csv"))
    print("\nround-trip check:")
    good = verify(WAVE3, os.path.join(HERE, "wave3.csv"))
    good &= verify(WAVE4, os.path.join(HERE, "wave4.csv"))
    sys.exit(0 if good else 1)
