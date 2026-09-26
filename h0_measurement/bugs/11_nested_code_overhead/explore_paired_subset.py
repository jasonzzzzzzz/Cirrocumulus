#!/usr/bin/env python3
"""EXPLORATORY (not the frozen protocol): what main job 987153 can still say.

The frozen reader returned invalid_r11 because the two arms' forward passes
diverged numerically (V1). This script does not change that verdict. It reports:

  A. the pairing noise floor: codebook-free quantities, nested/mono, per prompt;
  B. the frozen statistics computed on ALL rows with V1 waived (unpaired, noisy);
  C. the frozen statistics on the EXACTLY paired subset: KV groups
     (prompt, family, layer, kv_head) whose every head matches bit-for-bit on
     all codebook-free columns at step 4 AND on tau at steps 0-4, i.e. the same
     q, K, V and the same accum attention history. Only the codebook differs.
  D. code-level logit-noise ratios c{4,6,8}_abs on exactly matched rows at
     steps 0 and 4 (no allocation involved).

usage: .venv/bin/python h0_measurement/bugs/11_nested_code_overhead/explore_paired_subset.py 987153
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r11reader", HERE / "read_nested_code.py")
R = importlib.util.module_from_spec(spec)
sys.modules["r11reader"] = R
spec.loader.exec_module(R)

JOB = sys.argv[1]
KEY = R.KEY
GKEY = ["prompt", "family", "layer", "kv_head"]
EXTRA_FREE = ["unseen_frac_pp3_accum"]          # accum-evictor state, codebook-free


def load(task: int) -> pd.DataFrame:
    d = R.task_dir("main", JOB, task)
    pq = next(d.glob("h0_*.parquet"))
    df = pd.read_parquet(pq)
    have = set(df.columns)
    want = set(KEY + ["kv_head", "L", "needle_hit"] + R.IDENTITY_COLS + EXTRA_FREE
               + [f"c{w}_abs" for w in (1, 2, 3, 4, 5, 6, 8)])
    for B in R.BUDGETS:
        p = R._pfx(B)
        want |= {f"err_wf_{p}", f"mean_bits_{p}"} | {f"tier_frac_{p}_b{t}" for t in R.PANEL["nested3"]}
    return df[sorted(want & have)].copy()


def exact_same(a: pd.DataFrame, b: pd.DataFrame, cols) -> pd.Series:
    ok = pd.Series(True, index=a.index)
    for c in cols:
        if c in a and c in b:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            ok &= pd.Series((x == y) | (np.isnan(x) & np.isnan(y)), index=a.index)
    return ok


def main() -> None:
    lsha = R.verify_ledger()
    out: dict = {"job": JOB, "note": "EXPLORATORY; frozen verdict is invalid_r11 (V1)",
                 "cells": {}}
    pairs_all, pairs_sub = [], []
    for c in range(4):
        # provenance (V4) through the frozen loader; V2/V3/V5 re-checked below
        m_meta, n_meta = (R.load_task("main", JOB, 2 * c + t, lsha) for t in (0, 1))
        cell = m_meta["cell"]
        name = f"{cell.model}@{cell.ctx}"
        a = load(2 * c).set_index(KEY).sort_index()
        b = load(2 * c + 1).set_index(KEY).sort_index()
        assert a.index.equals(b.index)

        # --- A: noise floor on codebook-free quantities, step 4
        a4, b4 = a.xs(R.STEP, level="step"), b.xs(R.STEP, level="step")
        floor = {}
        for col in ("err_uniform2", "err_uniform3"):
            ea = np.sqrt((a4[col] ** 2).groupby(level="prompt").mean())
            eb = np.sqrt((b4[col] ** 2).groupby(level="prompt").mean())
            r = (eb / ea).to_numpy()
            floor[col] = {"min": float(r.min()), "max": float(r.max()),
                          "gmean": float(np.exp(np.log(r).mean()))}
        rows_diff = float((~exact_same(a4, b4, R.IDENTITY_COLS)).mean())

        # --- C: exactly paired KV groups
        same4 = exact_same(a4, b4, R.IDENTITY_COLS + EXTRA_FREE)
        tau_ok = pd.Series(True, index=a4.index)
        for s in range(R.STEP + 1):
            as_, bs_ = a.xs(s, level="step"), b.xs(s, level="step")
            eq = exact_same(as_, bs_, ["tau"]).reindex(a4.index, fill_value=False)
            tau_ok &= eq
        row_ok = (same4 & tau_ok).to_frame("ok").join(a4[["kv_head"]])
        grp_ok = row_ok.reset_index().groupby(GKEY)["ok"].all()
        keep = row_ok.reset_index().merge(grp_ok.rename("gok").reset_index(), on=GKEY)
        keep = keep.set_index(["prompt", "family", "layer", "head"])["gok"]
        sub_idx = keep[keep].index
        cov_prompt = keep.groupby(level="prompt").mean()
        L = a4.reset_index()["layer"]
        nlay = int(L.max()) + 1
        lay_cov = keep.groupby(level="layer").mean()
        thirds = {f"layers {int(i*nlay/3)}-{int((i+1)*nlay/3)-1}":
                  float(lay_cov[(lay_cov.index >= i * nlay / 3) & (lay_cov.index < (i + 1) * nlay / 3)].mean())
                  for i in range(3)}

        def frame(x4):
            f = x4.reset_index()
            f["step"] = R.STEP
            return f

        pairs_all.append(({"cell": cell, "frame": frame(a4)}, {"cell": cell, "frame": frame(b4)}))
        pairs_sub.append(({"cell": cell, "frame": frame(a4.loc[sub_idx])},
                          {"cell": cell, "frame": frame(b4.loc[sub_idx])}))

        # V2/V3/V5 on the arms (V1 is the known failure)
        for w in R.CHAIN[1:]:
            dfrac = float((a4[f"c{w}_abs"] != b4[f"c{w}_abs"]).mean())
            assert dfrac >= 0.99, (name, w, dfrac)
        for x in (a4, b4):
            for B in R.BUDGETS:
                p = R._pfx(B)
                assert (x[f"mean_bits_{p}"] <= B + 1e-9).all()
                assert np.isfinite(x[f"err_wf_{p}"]).all() and (x[f"err_wf_{p}"] > 0).all()

        # --- D: code-level on exactly matched rows, steps 0 and 4 separately
        code = {}
        for s in (0, R.STEP):
            as_, bs_ = a.xs(s, level="step"), b.xs(s, level="step")
            ok = exact_same(as_, bs_, ["tau", "c1_abs", "c2_abs", "c3_abs", "c5_abs"])
            as_, bs_ = as_[ok], bs_[ok]
            ws = [1, 2, 3, 4, 5, 6, 8]
            gm = {w: math.exp(np.log(as_[f"c{w}_abs"].clip(lower=1e-300)).mean()) for w in ws}
            res = {"rows": int(ok.sum()), "frac_rows": float(ok.mean())}
            for bw in R.CHAIN[1:]:
                ratio = float(np.exp(np.log(bs_[f"c{bw}_abs"] / as_[f"c{bw}_abs"]).mean()))
                gq = math.exp(np.log(bs_[f"c{bw}_abs"].clip(lower=1e-300)).mean())
                r, _ = R.matched_rate(ws, [math.log(gm[w]) for w in ws], math.log(gq))
                res[str(bw)] = {"noise_ratio": ratio, "equiv_width": r, "O_code": bw / r - 1}
            code[f"step{s}"] = res

        out["cells"][name] = {
            "noise_floor_prompt_rms_ratio": floor,
            "step4_rows_differing_codebook_free": rows_diff,
            "paired_subset_frac_rows": float(keep.mean()),
            "paired_subset_frac_by_prompt": {int(k): float(v) for k, v in cov_prompt.items()},
            "paired_subset_layer_coverage": thirds,
            "code_level_exact": code,
        }
        print(f"{name}: step-4 rows differing {rows_diff:.1%}; exactly paired groups cover "
              f"{keep.mean():.1%} of rows; layer coverage {thirds}", flush=True)

    out["all_rows_V1_waived"] = R.analyse(pairs_all)
    out["paired_subset"] = R.analyse(pairs_sub)
    for k in ("all_rows_V1_waived", "paired_subset"):
        out[k].pop("validity", None)
        out[k]["label"] = "EXPLORATORY: not a frozen verdict"
    path = HERE / f"explore_{JOB}.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    for k in ("all_rows_V1_waived", "paired_subset"):
        print(f"\n===== {k} =====")
        print(R.render(out[k]).replace("Validity: valid_r11 (V1-V5 passed)",
                                       "EXPLORATORY: frozen verdict is invalid_r11"))
    print("code-level on exactly matched rows:")
    for n, v in out["cells"].items():
        for s, r in v["code_level_exact"].items():
            print(f"  {n} {s} rows={r['frac_rows']:.1%} " + "  ".join(
                f"{b}:{r[b]['noise_ratio']:.3f}/{100*r[b]['O_code']:+.2f}%" for b in ("4", "6", "8")))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
