#!/usr/bin/env python3
"""R14 Stage 1g driver (design: s1g_lib.py; frozen rules: read_stage1g.py).

run_s1e.main with Stage 1g's presets (s1g_lib) and run_arm_g, which adds the
generalized two-tier read (G1-G3) and the 4-bit-store read (G1), on top of
run_s1f's amended metric (A2: span NLL, own-order replays, self-checks). Same
arguments as run_s1e.py, e.g.

    python run_s1g.py --mode evaluate --preset g128 --ctx 131072 --n-prompts 10 --prompt-offset 9100 --out-dir DIR
    python run_s1g.py --mode evaluate --preset g128 --ctx 131072 \
        --prompt-list 8109:niah_multikey,8901:niah_multikey,8937:niah_multikey --out-dir DIR

THE TWO-TIER READ, GENERALIZED ('qread2t[4][k][8][q]_v{v}'):
  1. tier 1 is the view: the 3-bit reads' store (the dense 4-bit allocation
     with '4'), values at v bits. The question is prefilled over it, and its last
     QREAD_ROWS rows select floor(r C) rows per KV head;
  2. the selected rows come from tier 2: their keys exact (FP8 with '8'), and
     their values too unless 'k' (then tier 1's v-bit values stay). Unselected
     rows keep tier 1;
  3. 'q': the cache is cropped back to the context and the question prefilled
     again over that mixed view, nothing evicted for it;
  4. the answer reads the selected rows only.
The teacher-forced replay repeats the same steps. Stage 1f's names (qread2t_v*,
qread2t8_v*) take this path too, so every two-tier arm shares one implementation.

A2's self-check (run_s1f.a2_columns) runs once per arm NAME and process here, not
once per family: every read path of the stage is checked against its first replay.
Outputs are renamed s1e_* -> s1g_*; the sidecar gains stage = "1g" and amend.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, H0, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import compress as C  # noqa: E402
from sievelib.kv_quant_baselines import fp8_e4m3  # noqa: E402
from sievelib.probe import cache_kv  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1d as S1D  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1g_lib as L  # noqa: E402

_S1E_RUN_ARM, _S1E_LIB, _S1E_TFM_OF = S1E.run_arm, S1E.L, S1E.tfm_of


# ------------------------------------------------------- the two-tier view
def tier1_store(pa, alloc):
    """The first tier's widths per layer: the reads' 3-bit store, or the dense
    4-bit allocation (s1g_lib.build_plan makes sure it is planned)."""
    if int(pa["store"]) == L.STORE4:
        return alloc[("uniform", L.STORE4)]
    return alloc[("qread_store", L.STORE_WIDTH)]


def to_tier2(past, nL, sel_ev, tier2, kv):
    """The selected rows (sel_ev False) from tier 2, in place: their keys exact
    or FP8, and their values too when kv. Unselected rows keep tier 1."""
    Cn = C.STATE.ctx_len
    for li in range(nL):
        K, V = cache_kv(past, li)
        keep = (~sel_ev[li].to(K.device)).unsqueeze(-1)                       # [Hkv, C, 1]
        k2 = K[:, :Cn].float()
        if tier2 == "fp8":
            k2 = fp8_e4m3(k2)
        C.STATE.kdeq[li] = torch.where(keep, k2.to(K.dtype), C.STATE.kdeq[li].to(K.dtype))
        if not kv:
            continue
        v2 = V[:, :Cn].float()
        if tier2 == "fp8":
            v2 = fp8_e4m3(v2)
        if li in C.STATE.vdeq or tier2 == "fp8":
            base = C.STATE.vdeq[li] if li in C.STATE.vdeq else V[:, :Cn]
            C.STATE.vdeq[li] = torch.where(keep, v2.to(V.dtype), base.to(V.dtype))


def _requestion(model, past, L0, q_ids, sel_ev, teacher):
    """G3: the question again over the mixed view, nothing evicted for it; the
    selection is restored for the answer."""
    if q_ids.shape[1] <= 1:
        return past
    C.crop_to(past, L0)
    C.STATE.evict = {li: torch.zeros_like(e) for li, e in sel_ev.items()}
    try:
        if teacher:
            with torch.no_grad():
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
        else:
            past = RR._question(model, past, q_ids)
    finally:
        C.STATE.evict = {li: e.clone() for li, e in sel_ev.items()}
    return past


def run_read2t_g(model, past, q_ids, store, r, R, norm_correct, vfn1, pa, eos, max_new, L0, tok, nL):
    """The generalized two-tier read's decode. Returns (gen, past, the store's
    evict masks, the selection's evict masks)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    q_ev = S1E.apply_store(past, store, L.STORE_WIDTH, R, norm_correct, vfn1, nL)
    S1D.HOOK.sel = S1D._Sel(r, {}, 0)
    S1D._install()
    try:
        past = RR._question(model, past, q_ids)
        S1E.select_all_w(past, nL, pa["tier2_bits"])
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
        to_tier2(past, nL, sel, pa["tier2"], pa["kv"])
        if pa["requestion"]:
            past = _requestion(model, past, L0, q_ids, sel, teacher=False)
        S1D.HOOK.sel.buf.phase = "answer"
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    return gen, past, q_ev, sel


def tf_read2t_g(model, past, L0, q_ids, fp_gen, r, R, norm_correct, vfn1, pa, store, nL):
    """The same steps, teacher-forced on fp_gen."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, store, L.STORE_WIDTH, R, norm_correct, vfn1, nL)
    S1D.HOOK.sel = S1D._Sel(r, {}, 0)
    S1D._install()
    C.STATE.enabled = True
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            S1E.select_all_w(past, nL, pa["tier2_bits"])
            sel = {li: e.clone() for li, e in C.STATE.evict.items()}
            to_tier2(past, nL, sel, pa["tier2"], pa["kv"])
            if pa["requestion"]:
                _requestion(model, past, L0, q_ids, sel, teacher=True)
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            inp = torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype)
            lg = model(inp, past_key_values=past, use_cache=True).logits[0].float()
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(fp_gen)]


def _vfn(pa, Rv, norm_correct):
    v = pa["v_bits"]
    return L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None


def run_read2t_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos,
                   max_new, L0, tok):
    """One generalized two-tier arm (run_s1e.run_arm's return tuple)."""
    t1 = time.time()
    vfn = _vfn(pa, Rv, norm_correct)
    store = tier1_store(pa, alloc)
    gen, past, q_ev, sel = run_read2t_g(model, past, q_ids, store, B, R, norm_correct, vfn, pa, eos, max_new, L0,
                                        tok, nL)
    t_arm = time.time() - t1
    au = C.bits_audit()
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    lg = tf_read2t_g(model, past, L0, q_ids, fp_gen, B, R, norm_correct, vfn, pa, store, nL)
    tfm = S1F.tfm_of_f(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    stored = dict(stored_bits_per_token=float(pa["store"]), stored_evict_frac=0.0)
    extra = dict(store_width=int(pa["store"]), tier2=pa["tier2"], tier2_bits=pa["tier2_bits"],
                 read_v_bits=pa["read_v_bits"], keys_only=not pa["kv"], requestion=bool(pa["requestion"]))
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, (arm, B), au, nk_state


def run_read4_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos,
                  max_new, L0, tok):
    """Single-tier question-time reads over the 4-bit store (run_s1e.run_arm's
    qread branch with the dense 4-bit allocation as the store)."""
    t1 = time.time()
    vfn = _vfn(pa, Rv, norm_correct)
    store = tier1_store(pa, alloc)
    gen, past, q_ev, sel = S1E.run_read(model, past, q_ids, store, L.STORE4, B, {}, R, norm_correct, vfn, eos,
                                        max_new, L0, tok, nL)
    t_arm = time.time() - t1
    au = C.bits_audit()
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    lg = S1E.tf_read(model, past, L0, q_ids, fp_gen, B, {}, L.STORE4, q_ev, nL)
    tfm = S1F.tfm_of_f(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    stored = dict(stored_bits_per_token=float(L.STORE4), stored_evict_frac=0.0)
    extra = dict(store_width=int(L.STORE4), n_protected=0, read_v_bits=pa["v_bits"])
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, (arm, B), au, nk_state


def replay_ids_g(model, past, L0, q_ids, ids, B, pa, protect, alloc, nL, R, Rv, norm_correct):
    """run_s1f.replay_ids for Stage 1g's read paths (two-tier: the generalized
    one; 4-bit-store reads: tf_read at width 4); everything else as Stage 1f."""
    if pa["family"] == "qread2t":
        return tf_read2t_g(model, past, L0, q_ids, ids, B, R, norm_correct, _vfn(pa, Rv, norm_correct), pa,
                           tier1_store(pa, alloc), nL)
    if pa["family"] == "qread" and int(pa["store"]) == L.STORE4:
        q_ev = {li: torch.zeros_like(e) for li, e in C.STATE.evict.items()}
        return S1E.tf_read(model, past, L0, q_ids, ids, B, {}, L.STORE4, q_ev, nL)
    return S1F.replay_ids(model, past, L0, q_ids, ids, B, pa, protect, alloc, nL, R, Rv, norm_correct)


def run_arm_g(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
              eos, max_new, L0, tok, last, sel_q1=None):
    """run_s1e.run_arm plus Stage 1g's read paths (same return tuple), with A2's
    columns in the row's metrics; the self-check runs once per arm name."""
    if pa["family"] == "qread2t":
        res = run_read2t_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct,
                             eos, max_new, L0, tok)
    elif pa["family"] == "qread" and int(pa["store"]) == L.STORE4:
        res = run_read4_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct,
                            eos, max_new, L0, tok)
    else:
        res = _S1E_RUN_ARM(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv,
                           norm_correct, eos, max_new, L0, tok, last, sel_q1)
    gen, past, tfm = res[0], res[1], res[2]
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: replay_ids_g(model, past, L0, q_ids, ids, B, pa, protect, alloc, nL, R, Rv,  # noqa: E731
                                      norm_correct)
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay,
                              (pa["family"], pa["twin"], bool(pa["protect"]), pa["base"])))
    return res


# ------------------------------------------------------------------------ main
def _rename_outputs(out_dir, mode):
    """s1e_* -> s1g_* in out_dir; the sidecar's file fields follow; stage = '1g'."""
    if not os.path.isdir(out_dir):
        return
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("s1e_"):
            os.replace(os.path.join(out_dir, f), os.path.join(out_dir, "s1g_" + f[4:]))
    for f in os.listdir(out_dir):
        if f.startswith(f"s1g_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            for k in ("parquet", "search", "searchlog"):
                if isinstance(side.get(k), str) and side[k].startswith("s1e_"):
                    side[k] = "s1g_" + side[k][4:]
            side.update(stage="1g", driver="run_s1g.py (run_s1e.main with s1g_lib presets and Stage 1g read paths)",
                        amend=L.AMEND, metric_a2="s_set_nll (Stage 1f amendment A2)", k_min=L.K_MIN)
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode")
    ap.add_argument("--out-dir")
    a, _ = ap.parse_known_args()
    if a.mode and a.mode != "evaluate":
        raise SystemExit("Stage 1g runs evaluation blocks only (no calibration, no reuse)")
    S1F._A2_CHECKED.clear()
    S1E.L, S1E.run_arm, S1E.tfm_of = L, run_arm_g, S1F.tfm_of_f
    try:
        S1E.main()
    finally:
        S1E.L, S1E.run_arm, S1E.tfm_of = _S1E_LIB, _S1E_RUN_ARM, _S1E_TFM_OF
    if a.out_dir and a.mode:
        _rename_outputs(a.out_dir, a.mode)


if __name__ == "__main__":
    main()
