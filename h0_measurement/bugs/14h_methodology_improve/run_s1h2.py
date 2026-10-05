#!/usr/bin/env python3
"""R14 Stage 1h R2 driver (design: s1h2_lib.py, plan.md; frozen rules: read_stage1h_r2.py).

run_s1h.py's driver (Stage 1g's arms, Stage 1h's arms, KL, per-arm memory), with
s1h2_lib's presets and one more arm, the single-tier read with a second question
pass (qread4q_v4; s1h2_lib's docstring). run_s1h is imported and installed as it is;
this file only adds the new arm on top. Same arguments as run_s1e.py, e.g.

    python run_s1h2.py --mode evaluate --preset h2qwen32 --ctx 32768 --n-prompts 10 --prompt-offset 9300 --out-dir DIR
    python run_s1h2.py --mode evaluate --preset h2regress --ctx 32768 \
        --prompt-list 8234:niah_multikey,8830:vt --override rot_seed=1 --out-dir DIR

Outputs are s1h_* files (run_s1h's renaming) whose sidecar also carries lib =
s1h2_lib and amend_r2 = R2.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_s1h as RH  # noqa: E402  (sets up every other path)
from sievelib import compress as C  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1d as S1D  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1h2_lib as L  # noqa: E402


# --------------------------------------------- reads with a second question pass
def _readq_store(pa, alloc):
    if int(pa["store"]) == L.STORE4:
        return alloc[("uniform", L.STORE4)]
    return alloc[("qread_store", L.STORE_WIDTH)]


def _select(model, past, q_ids, store, width, r, R, norm_correct, vfn, L0, nL, teacher):
    """The question over the whole store and its vote (as run_s1e.run_read /
    tf_read); returns (past, the selection's evict masks). The view stays enabled."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, store, width, R, norm_correct, vfn, nL)
    S1D.HOOK.sel = S1D._Sel(r, {}, 0)
    S1D._install()
    C.STATE.enabled = True
    try:
        if teacher:
            with torch.no_grad():
                if q_ids.shape[1] > 1:
                    model(q_ids[:, :-1], past_key_values=past, use_cache=True)
        else:
            past = RR._question(model, past, q_ids)
        S1E.select_all_w(past, nL, width)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
    return past, {li: e.clone() for li, e in C.STATE.evict.items()}


def run_readq(model, past, q_ids, store, width, r, R, norm_correct, vfn, eos, max_new, L0, tok, nL):
    """The read with a second question pass: the question again over the selected
    rows only, then the answer over the same rows."""
    try:
        past, sel = _select(model, past, q_ids, store, width, r, R, norm_correct, vfn, L0, nL, teacher=False)
        C.crop_to(past, L0)
        past = RR._question(model, past, q_ids)          # the selection masks the unselected rows here too
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        C.STATE.enabled = False
    return gen, past, sel


def tf_readq(model, past, L0, q_ids, ids, store, width, r, R, norm_correct, vfn, nL):
    """The same read, teacher-forced on `ids`."""
    if not ids:
        return None
    try:
        past, _ = _select(model, past, q_ids, store, width, r, R, norm_correct, vfn, L0, nL, teacher=True)
        C.crop_to(past, L0)
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            x = [int(q_ids[0, -1])] + [int(t) for t in ids[:-1]]
            inp = torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype)
            lg = model(inp, past_key_values=past, use_cache=True).logits[0].float()
    finally:
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(ids)]


def run_readq_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos, max_new,
                  L0, tok):
    """One qreadq arm (run_s1e.run_arm's return tuple), with A2's columns."""
    width = int(pa["store"])
    store = _readq_store(pa, alloc)
    vfn = L1B.v_quantizer(pa["v_bits"], Rv, norm_correct) if pa["v_bits"] < 16 else None
    t1 = time.time()
    gen, past, sel = run_readq(model, past, q_ids, store, width, B, R, norm_correct, vfn, eos, max_new, L0, tok, nL)
    t_arm = time.time() - t1
    au = C.bits_audit()
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    lg = tf_readq(model, past, L0, q_ids, fp_gen, store, width, B, R, norm_correct, vfn, nL)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: tf_readq(model, past, L0, q_ids, ids, store, width, B, R, norm_correct, vfn, nL)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, (pa["family"], pa["twin"], False, pa["base"])))
    stored = dict(stored_bits_per_token=float(width), stored_evict_frac=0.0)
    extra = dict(store_width=width, requestion=True)
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, (arm, B), au, nk_state


def run_arm_h2(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
               eos, max_new, L0, tok, last, sel_q1=None):
    """run_s1h.run_arm_h, or a qreadq arm with the same per-arm memory columns."""
    if pa["family"] != "qreadq":
        return RH.run_arm_h(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv,
                            norm_correct, eos, max_new, L0, tok, last, sel_q1)
    RH._fold()
    RH._ORIG["reset_peaks"]()
    base = (sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
            if torch.cuda.is_available() else float("nan"))
    res = run_readq_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos,
                        max_new, L0, tok)
    pk = RH._dev_peaks()
    RH._fold()
    res[4].update(peak_gib_arm=sum(pk) if pk else float("nan"), base_gib_arm=base)
    return res


# ------------------------------------------------------------------------ main
def _mark_outputs(out_dir, mode):
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(f"s1h_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            side.update(lib="s1h2_lib", amend_r2=L.AMEND_R2,
                        driver="run_s1h2.py (run_s1h.py's driver with s1h2_lib presets and the second-pass reads)")
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode")
    ap.add_argument("--out-dir")
    a, _ = ap.parse_known_args()
    if a.mode and a.mode != "evaluate":
        raise SystemExit("Stage 1h runs evaluation blocks only (no calibration, no reuse)")
    S1F._A2_CHECKED.clear()
    RH.install()
    S1E.L, S1E.run_arm = L, run_arm_h2
    try:
        S1E.main()
    finally:
        RH.uninstall()
    if a.out_dir and a.mode:
        RH._rename_outputs(a.out_dir, a.mode)
        _mark_outputs(a.out_dir, a.mode)


if __name__ == "__main__":
    main()
