#!/usr/bin/env python3
"""R7b's driver (plan_r7b.md): run_s1h5.main (R6's driver, read-only) on R7b's presets, with two run-time substitutions in
run_s1h5's namespace and nothing edited on disk:
  run_s1h5.L          -> lib_r7b (R7b's presets, arms, plans, traffic model);
  run_s1h5.run_arm_h5 -> run_arm_r7b: R7b's families (dnest, wnest, ddense, cnest, kcal) run by arms_r7b; R6's dtail
                         (E4) at the unit's dither seed; every other arm (fp8kv, kivi4_v4, uniform, uniform+v4,
                         quest_v16, fp_noise) R6's own runner, unchanged.
Every dithered arm of a unit uses the unit's seed (lib_r7b.unit_seed(task, prompt)). HELMET (suite r4) blocks read R7b's
item manifest (tasks_r7b). Outputs: run_s1h5's parquet and sidecar, whose 'r7b' field names R7b's settings, the frozen
data files the arms read (E1's targets, E3's allocation) and the code's sha256.

    python run_r7b.py --suite r3 --mode evaluate --preset r7bllama128 --ctx 131072 \\
        --tasks niah_single,niah_multikey,niah_multivalue,vt --n-prompts 5 --prompt-offset 9800 --out-dir DIR
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)
METH = os.path.join(os.path.dirname(R7), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _p in (R7, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import torch  # noqa: E402
import run_s1h5 as RS5  # noqa: E402  (imports the whole chain: run_s1h4 ... run_s1e, run_r8, sievelib)
import lib_r7b as LB  # noqa: E402
import arms_r7b as AR  # noqa: E402
import tasks_r7b as T7B  # noqa: E402

ORIG = {}
STATE = dict(group=None, preset=None)


def run_arm_r7b(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc, eos, max_new,
                L0, tok, last, sel_q1=None):
    u = RS5.S.unit
    seed = LB.unit_seed(u["task"], u["prompt_idx"])
    AR.UNIT.update(seed=seed, task=u["task"], prompt=int(u["prompt_idx"]), group=STATE["group"])
    fam = pa["family"]
    if fam in LB.NEW_FAMILIES_R7B:
        RS5.RH._fold()
        RS5.RH._ORIG["reset_peaks"]()
        base = (sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
                if torch.cuda.is_available() else float("nan"))
        res = AR.run(fam, model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok)
        pk = RS5.RH._dev_peaks()
        RS5.RH._fold()
        res[4].update(peak_gib_arm=sum(pk) if pk else float("nan"), base_gib_arm=base)
        return res
    if fam == "dtail":                                   # E4: R6's design at the unit's seed (its value coins = dnest's)
        old = RS5.DITHER_SEED
        RS5.DITHER_SEED = seed
        try:
            res = ORIG["run_arm_h5"](model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc,
                                     eos, max_new, L0, tok, last, sel_q1)
        finally:
            RS5.DITHER_SEED = old
        res[4].update(dither_seed=int(seed))
        return res
    return ORIG["run_arm_h5"](model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc, eos,
                              max_new, L0, tok, last, sel_q1)


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def code_manifest() -> dict:
    """sha256 of R7b's code, of R7a's modules it imports, of every R6 file the run imports and of R7b's fixed HELMET data
    (the reader checks it is the same in every block). The calibration stage's outputs (data/tau_*.json, data/alloc_*.json)
    are NOT in it: they appear while core blocks may still be starting, and only the exploratory arms read them. They are
    recorded per block in the sidecar's r7b.data_sha256, which the exploratory reader checks instead."""
    files = sorted(glob.glob(os.path.join(HERE, "*.py"))) + [os.path.join(R7, f) for f in ("cert_r7.py", "tier_r7.py")] \
        + [os.path.join(METH, f) for f in ("run_s1h5.py", "s1h5_lib.py", "rquant_s1h5.py", "tasks_s1h4.py", "tasks_s1h6.py",
                                           "run_s1h4.py")] + [T7B.MANIFEST] + sorted(glob.glob(os.path.join(T7B.ITEMS_DIR, "*")))
    root = os.path.dirname(os.path.dirname(os.path.dirname(R7)))
    return {os.path.relpath(f, root): _sha(f) for f in files if os.path.exists(f)}


def _data_hashes() -> dict:
    """sha256 of the calibration outputs the group's exploratory and stage-2 arms read (data/tau_<group>.json,
    data/alloc_<group>.json), where they exist."""
    out = {}
    for k in ("tau", "alloc"):
        p = LB.data_path(k, STATE["group"])
        if os.path.exists(p):
            out[os.path.basename(p)] = _sha(p)
    return out


def main():
    argv = sys.argv[1:]
    preset = argv[argv.index("--preset") + 1] if "--preset" in argv else None
    suite = argv[argv.index("--suite") + 1] if "--suite" in argv else "r3"
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else None
    if preset not in LB.R7B_PRESETS:
        raise SystemExit(f"--preset must be an R7b preset {LB.R7B_PRESETS}, not {preset!r}")
    if any(a.startswith("--dither") for a in argv):
        raise SystemExit("R7b's dither settings are lib_r7b's (the unit's seed), not --dither-* flags")
    STATE.update(group=LB.R7B_GROUP_OF[preset], preset=preset)
    data0 = _data_hashes()                      # as the arms read them; compared at the end (data_changed)
    ORIG.update(L=RS5.L, run_arm_h5=RS5.run_arm_h5)
    RS5.L, RS5.run_arm_h5 = LB, run_arm_r7b
    hm = suite == "r4"
    if hm:
        T7B.install()
    try:
        RS5.main()
    finally:
        RS5.L, RS5.run_arm_h5 = ORIG["L"], ORIG["run_arm_h5"]
        if hm:
            T7B.uninstall()
    if not out_dir:
        return
    sp = glob.glob(os.path.join(out_dir, "s1h_evaluate_*.json"))
    if len(sp) != 1:
        raise RuntimeError(f"{out_dir}: expected one sidecar, found {sp}")
    side = json.load(open(sp[0]))
    kind = LB.PRESETS[preset]["r7b"]
    data = _data_hashes()
    side["r7b"] = dict(amend=LB.AMEND_R7B, preset_kind=kind, group=STATE["group"], design=list(LB.DESIGN),
                       e4=list(LB.E4_ARM), gates=[list(g) for g in LB.GATES], quest_b=LB.QUEST_B,
                       seeds="lib_r7b.unit_seed(task, prompt) = 1000 x (sha256(salt, task, prompt) mod 1e9) + 7; "
                             "+ 2 layer (keys), + 2 layer + 1 (values)",
                       rotations=[LB.ROT_K, LB.ROT_V], base_bits=LB.BASE_BITS, data_sha256=data0,
                       data_changed=data != data0,
                       manifest_r7b=(T7B.MANIFEST_SHA256 if hm else None), code_sha256=code_manifest(),
                       driver="run_r7b.py (run_s1h5.main with lib_r7b and arms_r7b)")
    with open(sp[0], "w") as fh:
        json.dump(side, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
