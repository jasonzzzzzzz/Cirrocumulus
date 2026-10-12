#!/usr/bin/env python3
"""R7a's driver (plan.md): run_s1h5.main (R6's driver, read-only) on R7a's presets, with three run-time substitutions in
run_s1h5's namespace and nothing edited on disk:
  run_s1h5.L              -> r7_lib (R7a's presets, the seed-suffixed design arms, their plan);
  run_s1h5.run_probe_arm  -> run_probe_r7_arm (the vote, the 4/4 store, then FP's teacher-forced pass with probe_r7);
  run_s1h5.run_arm_h5     -> run_arm_r7 (a seed-suffixed design arm runs run_s1h5.run_dtail_arm with its dither seed).
HELMET blocks read R6's item manifest (tasks_s1h6), as R6's presets do.

Outputs (besides run_s1h5's parquet and sidecar, whose 'r7a' field names R7a's settings):
  r7a_heads_<model>_<ctx>.parquet    per unit, layer, query head, measured step (probe_r7's columns)
  r7a_groups_<model>_<ctx>.parquet   per unit, layer, KV head, step of the first chunk, key width (the certified read)

    python run_r7.py --suite r3 --mode evaluate --preset r7llama128 --ctx 131072 \\
        --tasks niah_single,niah_multikey,niah_multivalue,vt --n-prompts 5 --prompt-offset 9700 --out-dir DIR
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if METH not in sys.path:
    sys.path.append(METH)
import run_s1h5 as RS5  # noqa: E402  (imports the whole chain: run_s1h4 ... run_s1e, run_r8, sievelib)
import r7_lib as L7  # noqa: E402
import probe_r7 as PR  # noqa: E402
import cert_r7 as K7  # noqa: E402

C, S1E, RH = RS5.C, RS5.S1E, RS5.RH
SIDE = dict(head=[], group=[])
ORIG = {}
SMOKE_PROBE = dict(DRAWS={4: 1, 6: 2, 8: 1}, MAX_ROWS=4, LAYERS=(1, 8))   # the CPU smokes (presets *smoke): wiring only


def run_probe_r7_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, nc, L0):
    """run_s1h5.run_probe_arm with R7a's probe: the system's vote at r = 1/8 over the 4/4 store (coin-free), the 4/4 store
    applied (the sketch), FP's answer teacher-forced once more with probe_r7 installed (its outputs are FP's)."""
    t1 = time.time()
    Cn = C.STATE.ctx_len
    votes = RS5.vote_sets(model, past, q_ids, alloc, [0.125], R, Rv, nc, L0, nL)
    PR.P.reset()
    PR.P.keep = {"vote1": votes[0.125]}
    PR.P.rows, PR.P.a0 = PR.probe_rows(len(fp_gen)), L0 + q_ids.shape[1] - 1
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, RS5._store4(alloc), L7.STORE_WIDTH, R, nc, RS5._vfn4(Rv, nc), nL)
    PR.P.mode = "probe"
    PR.install()
    try:
        lg = RS5._fp_pass(model, past, L0, q_ids, fp_gen) if fp_gen else None
    finally:
        PR.uninstall()
        C.STATE.reset_arm()
    t_arm = time.time() - t1
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    unit = dict(RS5.S.unit, ctx_len=Cn)
    hf, gf = PR.head_frame(unit), PR.group_frame(unit)
    SIDE["head"].append(hf)
    SIDE["group"].append(gf)
    extra = dict(summary(hf, gf), tf_only=True, probe_seconds=json.dumps({k: round(v, 2) for k, v in PR.P.timing.items()}))
    PR.P.reset()
    return RS5._row(arm, B, fp_gen, past, tfm, extra, t_arm, 0.0)


def summary(hf, gf) -> dict:
    """One row's summary of R7a's probe (the full records are in the side files)."""
    import numpy as np
    out = {}
    if len(hf):
        out["r7_head_steps"] = int(len(hf))
        out["r7_unread_mass_med"] = float(hf.unread_mass.median())
        out["r7_share2_med"] = float(hf.share2.median())
        for kb in PR.BITS:
            if f"cov_p8_k{kb}" not in hf:
                continue
            nd = hf[f"ndraw_k{kb}"]
            for b in ("v2", "p8"):
                out[f"r7_miss_{b}_k{kb}"] = float(((1.0 - hf[f"cov_{b}_k{kb}"]) * nd).sum())
                out[f"r7_ratio_{b}_k{kb}_med"] = float(hf[f"rat_{b}_k{kb}"].median())
                out[f"r7_fp8_{b}_k{kb}"] = float(hf[f"f8_{b}_k{kb}"].mean())
            ok = hf[f"m_pred_k{kb}"] > 0
            out[f"r7_m_meas_pred_k{kb}_mean"] = float((hf.loc[ok, f"mse_k{kb}"] / hf.loc[ok, f"m_pred_k{kb}"]).mean())
            out[f"r7_m_nn_pred_k{kb}_mean"] = float((hf.loc[ok, f"nn_k{kb}"] / hf.loc[ok, f"m_pred_k{kb}"]).mean())
            out[f"r7_tail_over_fp8_k{kb}_med"] = float((np.sqrt(hf[f"m_pred_k{kb}"]) / hf.err_fp8.clip(lower=1e-300)).median())
    if len(gf):
        for kb, g in gf.groupby("bits"):
            out[f"r7_cr_rows_k{kb}"] = float(g.rows.mean())
            out[f"r7_cr_rows_oracle_k{kb}"] = float(g.rows_or.mean())
            out[f"r7_cr_traffic_k{kb}"] = float(((1 - g.rows) * (kb + 4) / 32 + g.rows).mean())
            out[f"r7_cr_met_k{kb}"] = float(g.met.mean())
            out[f"r7_cr_met_heads_k{kb}"] = float(g.met_heads.sum() / g.r.sum())
    return out


def run_arm_r7(model, past, arm, B, pa, *rest, **kw):
    """A seed-suffixed design arm (r7_lib.SEED_ARM_RE) runs run_s1h5's dtail arm with its own dither seed."""
    if pa.get("seed_index"):
        old = RS5.DITHER_SEED
        RS5.DITHER_SEED = L7.seed_of(pa["seed_index"])
        try:
            res = ORIG["run_arm_h5"](model, past, arm, B, pa, *rest, **kw)
        finally:
            RS5.DITHER_SEED = old
        res[4].update(dither_seed=L7.seed_of(pa["seed_index"]), seed_index=int(pa["seed_index"]))
        return res
    return ORIG["run_arm_h5"](model, past, arm, B, pa, *rest, **kw)


def code_manifest() -> dict:
    """sha256 of this folder's code and of every R6 file the run imports (the reader checks them per block)."""
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "*.py"))) + [os.path.join(METH, f) for f in (
        "run_s1h5.py", "s1h5_lib.py", "rquant_s1h5.py", "cert_s1h5.py", "tasks_s1h6.py", "tasks_s1h4.py", "run_s1h4.py")]
    return {os.path.relpath(f, os.path.dirname(os.path.dirname(HERE))): hashlib.sha256(open(f, "rb").read()).hexdigest()
            for f in files if os.path.exists(f)}


def main():
    argv = sys.argv[1:]
    preset = argv[argv.index("--preset") + 1] if "--preset" in argv else None
    suite = argv[argv.index("--suite") + 1] if "--suite" in argv else "r3"
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else None
    if preset not in L7.R7_PRESETS:
        raise SystemExit(f"--preset must be an R7a preset {L7.R7_PRESETS}, not {preset!r}")
    if any(a.startswith("--dither") for a in argv):
        raise SystemExit("R7a's probe settings are probe_r7's constants (frozen with the run), not --dither-* flags")
    if preset.endswith("smoke"):
        for k_, v_ in SMOKE_PROBE.items():
            setattr(PR, k_, v_)
    ORIG.update(L=RS5.L, run_probe_arm=RS5.run_probe_arm, run_arm_h5=RS5.run_arm_h5)
    RS5.L, RS5.run_probe_arm, RS5.run_arm_h5 = L7, run_probe_r7_arm, run_arm_r7
    SIDE["head"].clear()
    SIDE["group"].clear()
    t6 = suite == "r4"
    if t6:                                            # R7a's HELMET units come from R6's manifest (items 11-40)
        import tasks_s1h6 as T6
        T6.install()
    try:
        RS5.main()
    finally:
        RS5.L, RS5.run_probe_arm, RS5.run_arm_h5 = ORIG["L"], ORIG["run_probe_arm"], ORIG["run_arm_h5"]
        if t6:
            T6.uninstall()
    if not out_dir:
        return
    import glob
    import pandas as pd
    sp = glob.glob(os.path.join(out_dir, "s1h_evaluate_*.json"))
    if len(sp) != 1:
        raise RuntimeError(f"{out_dir}: expected one sidecar, found {sp}")
    side = json.load(open(sp[0]))
    stem = f"{side['model']}_{side['ctx']}"
    files = {}
    for k, name in (("head", "r7a_heads"), ("group", "r7a_groups")):
        parts = [x for x in SIDE[k] if len(x)]
        if parts:
            f = f"{name}_{stem}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(os.path.join(out_dir, f))
            files[name] = f
    side["r7a"] = dict(amend=L7.AMEND_R7A, preset_kind=L7.PRESETS[preset]["r7a"], files=files, probe=PR.settings(),
                       design=L7.DESIGN, seeds=[L7.seed_of(k) for k in range(L7.SEEDS_EXTRA + 1)],
                       seed_stride=L7.SEED_STRIDE, cert="cert_r7 (part 8, v2 copy, certified read, Theorem M)",
                       code_sha256=code_manifest(), driver="run_r7.py (run_s1h5.main with r7_lib and probe_r7)")
    with open(sp[0], "w") as fh:
        json.dump(side, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
