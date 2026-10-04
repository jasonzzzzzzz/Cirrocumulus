#!/usr/bin/env python3
"""R14 Stage 1f driver (design: s1f_lib.py; frozen rules: read_stage1f.py).

The Stage 1e driver's main loop (run_s1e.main, unchanged and already run on both
clusters), given Stage 1f's presets (s1f_lib) and one new arm type, the two-tier
read. Same arguments as run_s1e.py, e.g.

    # F4: Qwen recalibration with multivalue (stop rule eos_only, from the preset)
    python run_s1f.py --mode calibrate --preset qwen32f --ctx 32768 --prompt-offset 8700 \
        --task-counts niah_single=10,niah_multikey=30,niah_multivalue=30,vt=10 \
        --routes-1b QWEN_S1B.json --write-routes QWEN_S1F.json --out-dir DIR
    # F2 / F4 evaluation, F3 reuse
    python run_s1f.py --mode evaluate --preset tt128 --ctx 131072 --n-prompts 10 --prompt-offset 8900 --out-dir DIR
    python run_s1f.py --mode reuse --preset reuse32f --ctx 32768 --n-prompts 22 --prompt-offset 8600 \
        --routes-1b S1B.json --routes-1d S1D.json --routes-1e S1E.json --out-dir DIR

TWO-TIER READ ('qread2t_v{v}', 'qread2t8_v{v}'). The question is prefilled over
tier 1, the reads' GPU store (TurboQuant-3 keys; values at v bits), and its last
QREAD_ROWS rows select floor(r C) rows per KV head, as in the question-time
reads. Then the view switches to tier 2 for every context row: the cache's exact
keys and values, or both through FP8 (one scale per KV head). The selection is
kept, so every answer step reads the selected rows exactly. The teacher-forced
replay repeats the same two phases.

Outputs are renamed s1e_* -> s1f_* after the run, and their sidecar gains
stage = "1f". A calibration's routes file gains meta.stage = "1f".

AMENDMENT A2 (s1f_lib's docstring). Every row also carries the all-token span
NLL and the order in which the arm states FP's values. When the arm states all
of them in another order, FP's answer is rewritten into that order and replayed
once more through the arm's own replay path (replay_ids), right after the arm
ran, so its view is still set up; the order-robust s_set_nll takes the smaller
of the two. On the first multi-answer unit of the run, every arm family also
replays FP's own answer a second time, and a2_check records how far that replay
is from the first (it must be ~0). The sidecar gains amend = s1f_lib.AMEND.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
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
import s1b_lib as L1B  # noqa: E402
import s1d_lib as L1D  # noqa: E402
import s1f_lib as L  # noqa: E402

_S1E_RUN_ARM = S1E.run_arm
_S1E_LIB = S1E.L
_S1E_TFM_OF = S1E.tfm_of
_A2_CHECKED: set = set()                   # arm families whose replay was self-checked in this process


# ------------------------------------------------------------- two-tier view
def switch_to_tier2(past, nL, tier2):
    """Every context row of the view becomes tier 2: exact keys and values, or
    both through FP8. The eviction masks (the selection) are left as they are."""
    Cn = C.STATE.ctx_len
    for li in range(nL):
        K, V = cache_kv(past, li)
        Kc = K[:, :Cn].float()
        if tier2 == "fp8":
            C.STATE.kdeq[li] = fp8_e4m3(Kc).to(K.dtype)
            C.STATE.vdeq[li] = fp8_e4m3(V[:, :Cn].float()).to(V.dtype)
        else:
            C.STATE.kdeq[li] = Kc.to(K.dtype)
            C.STATE.vdeq.pop(li, None)


def run_read2t(model, past, q_ids, store, r, R, norm_correct, vfn1, tier2, eos, max_new, L0, tok, nL):
    """The two-tier read's decode. Returns (gen, past, the store's evict masks,
    the selection's evict masks)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    q_ev = S1E.apply_store(past, store, L.STORE_WIDTH, R, norm_correct, vfn1, nL)
    S1D.HOOK.sel = S1D._Sel(r, {}, 0)
    S1D._install()
    try:
        past = RR._question(model, past, q_ids)
        S1E.select_all_w(past, nL, L.TIER2_BITS[tier2])
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
        switch_to_tier2(past, nL, tier2)
        S1D.HOOK.sel.buf.phase = "answer"
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    return gen, past, q_ev, sel


def tf_read2t(model, past, L0, q_ids, fp_gen, r, R, norm_correct, vfn1, tier2, store, nL):
    """The same two phases, teacher-forced on FP's answer."""
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
            S1E.select_all_w(past, nL, L.TIER2_BITS[tier2])
            switch_to_tier2(past, nL, tier2)
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            inp = torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype)
            lg = model(inp, past_key_values=past, use_cache=True).logits[0].float()
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(fp_gen)]


# ------------------------------------------------------------- A2: the metric
def a2_defaults(tfm, am):
    """A2's columns from the replay in FP's order: the all-token span NLL, FP's
    order, and order-robust columns equal to the FP-order ones (a2_columns
    replaces them for a reordered answer). Updates tfm in place and returns it."""
    if not tfm or "a_span_nll" in tfm:
        return tfm
    _, s = L.span_nll(tfm["tf_logp"], am["vmask"], am["span_end"])
    fo = ",".join(map(str, L.answer_order(am["text"], am["found"])))
    a = tfm["a_sum_nll"]
    tfm.update(a_span_nll=s, a_own_nll=a, a_span_own_nll=s, a_set_nll=a, s_set_nll=s, ans_order=fo,
               fp_ans_order=fo, ans_reordered=False, own_replay=False, t_own=0.0, a2_check=float("nan"))
    return tfm


def tfm_of_f(lg, fp_gen, content, am):
    """run_s1e.tfm_of plus A2's columns (main installs it as S1E.tfm_of)."""
    return a2_defaults(_S1E_TFM_OF(lg, fp_gen, content, am), am)


def replay_ids(model, past, L0, q_ids, ids, B, pa, protect, alloc, nL, R, Rv, norm_correct):
    """Logits of the arm's teacher-forced replay of the token sequence `ids`,
    through the path its first replay took (run_s1e.run_arm, run_arm_f). Call it
    right after the arm ran: the view the arm left in C.STATE is reused."""
    fam = pa["family"]
    if fam == "qread2t":
        v = pa["v_bits"]
        vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
        return tf_read2t(model, past, L0, q_ids, ids, B, R, norm_correct, vfn, pa["tier2"],
                         alloc[("qread_store", L.STORE_WIDTH)], nL)
    if fam in ("qread", "qreadfp") and not pa["twin"]:
        q_ev = {li: torch.zeros_like(e) for li, e in C.STATE.evict.items()}   # the reads' store evicts nothing
        return S1E.tf_read(model, past, L0, q_ids, ids, B, protect if pa["protect"] else {}, pa["store"], q_ev,
                           nL)
    return S1D.tf_phased(model, past, L0, q_ids, ids, True)


def _logp(lg, ids):
    t = torch.tensor([int(x) for x in ids], device=lg.device)
    lp = torch.log_softmax(lg.float(), -1)[torch.arange(len(ids), device=lg.device), t]
    return [round(float(x), 5) for x in lp.tolist()]           # as tf_metrics2 stores tf_logp


def a2_columns(tok, fp_gen, am, gen, tfm, replay, check_key=None):
    """A2 for one arm row: the order in which the arm states FP's values; for a
    reordered answer, the replay of FP's answer rewritten into that order; and,
    once per check_key (arm family) and process, on a multi-answer unit, the
    self-check replay of FP's own answer (a2_check)."""
    found = am.get("found") or []
    fo = L.answer_order(am.get("text", ""), found)
    own = L.answer_order(tok.decode(gen), found)
    out = dict(ans_order=",".join(map(str, own)), fp_ans_order=",".join(map(str, fo)),
               ans_reordered=L.is_reordering(fo, own))
    if not tfm:
        return out
    if check_key is not None and len(found) > 1 and check_key not in _A2_CHECKED:
        _A2_CHECKED.add(check_key)
        lp = _logp(replay(list(fp_gen)), fp_gen)
        out["a2_check"] = float(np.max(np.abs(np.asarray(lp) - np.asarray(tfm["tf_logp"], dtype=float))))
    if not out["ans_reordered"]:
        return out
    ids = L.own_order_ids(tok, fp_gen, am, own)
    am2 = L1D.answer_tokens(tok, ids, found) if ids else None
    if not am2 or len(am2["found"]) != len(found):
        return out
    t0 = time.time()
    a, s = L.span_nll(_logp(replay(ids), ids), am2["vmask"], am2["span_end"])
    out.update(own_replay=True, t_own=time.time() - t0, a_own_nll=a, a_span_own_nll=s,
               a_set_nll=float(np.fmin(tfm["a_sum_nll"], a)), s_set_nll=float(np.fmin(tfm["a_span_nll"], s)))
    return out


def run_arm_f(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
              eos, max_new, L0, tok, last, sel_q1=None):
    """run_s1e.run_arm plus the two-tier read (same return tuple), plus A2's
    columns in the row's metrics (tfm, the third element)."""
    if pa["family"] != "qread2t":
        res = _S1E_RUN_ARM(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv,
                           norm_correct, eos, max_new, L0, tok, last, sel_q1)
    else:
        res = run_read2t_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv,
                             norm_correct, eos, max_new, L0, tok)
    gen, past, tfm = res[0], res[1], res[2]
    a2_defaults(tfm, am)
    replay = lambda ids: replay_ids(model, past, L0, q_ids, ids, B, pa, protect, alloc, nL, R, Rv,  # noqa: E731
                                    norm_correct)
    tfm.update(a2_columns(tok, fp_gen, am, gen, tfm, replay, (pa["family"], pa["twin"], bool(pa["protect"]))))
    return res


def run_read2t_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct,
                   eos, max_new, L0, tok):
    """The two-tier read as one arm (run_s1e.run_arm's return tuple)."""
    t1 = time.time()
    v = pa["v_bits"]
    vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
    store = alloc[("qread_store", L.STORE_WIDTH)]
    gen, past, q_ev, sel = run_read2t(model, past, q_ids, store, B, R, norm_correct, vfn, pa["tier2"], eos,
                                      max_new, L0, tok, nL)
    t_arm = time.time() - t1
    au = C.bits_audit()
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    lg = tf_read2t(model, past, L0, q_ids, fp_gen, B, R, norm_correct, vfn, pa["tier2"], store, nL)
    tfm = tfm_of_f(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    stored = dict(stored_bits_per_token=float(L.STORE_WIDTH), stored_evict_frac=0.0)
    extra = dict(store_width=int(L.STORE_WIDTH), tier2=pa["tier2"], tier2_bits=pa["tier2_bits"],
                 read_v_bits=pa["tier2_bits"])
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, (arm, B), au, nk_state


# ------------------------------------------------------------------------ main
def _rename_outputs(out_dir, mode, write_routes):
    """s1e_* -> s1f_* in out_dir; the sidecar's file fields follow; stage = '1f'."""
    if not os.path.isdir(out_dir):
        return
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("s1e_"):
            os.replace(os.path.join(out_dir, f), os.path.join(out_dir, "s1f_" + f[4:]))
    for f in os.listdir(out_dir):
        if f.startswith(f"s1f_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            for k in ("parquet", "search", "searchlog"):
                if isinstance(side.get(k), str) and side[k].startswith("s1e_"):
                    side[k] = "s1f_" + side[k][4:]
            side.update(stage="1f", driver="run_s1f.py (run_s1e.main with s1f_lib presets and two-tier reads)",
                        amend=L.AMEND, metric_a2="s_set_nll: NLL of every token of FP's answer from its first "
                        "answer-value token to the span end; for an answer stating FP's values in another order, the "
                        "smaller of that and the same NLL with FP's answer rewritten into the arm's order")
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)
    if mode == "calibrate" and write_routes and os.path.exists(write_routes):
        j = json.load(open(write_routes))
        src = j["meta"].get("source", "")
        if os.path.basename(src).startswith("s1e_"):
            j["meta"]["source"] = os.path.join(os.path.dirname(src), "s1f_" + os.path.basename(src)[4:])
        j["meta"]["stage"] = "1f"
        with open(write_routes, "w") as fh:
            json.dump(j, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode")
    ap.add_argument("--out-dir")
    ap.add_argument("--write-routes", default="")
    a, _ = ap.parse_known_args()
    S1E.L, S1E.run_arm, S1E.tfm_of = L, run_arm_f, tfm_of_f
    try:
        S1E.main()
    finally:
        S1E.L, S1E.run_arm, S1E.tfm_of = _S1E_LIB, _S1E_RUN_ARM, _S1E_TFM_OF
    if a.out_dir and a.mode:
        _rename_outputs(a.out_dir, a.mode, a.write_routes)


if __name__ == "__main__":
    main()
