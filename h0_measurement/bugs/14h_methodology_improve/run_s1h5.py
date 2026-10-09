#!/usr/bin/env python3
"""R14 Stage 1h R5 driver (draft; design: s1h5_lib.py; hooks: probe_s1h5.py; math: cert_s1h5.py).

run_s1e.main on any earlier suite's units, with that suite's task setup installed unchanged:
  --suite r3    run_s1h3's harness (RULER with R3a2's caps; R1/R2's units at the default
                difficulty, R3a's with --task-cfg, and mk_panel)
  --suite r3b   run_s1h3b's tasks (cwe, fwe, NoLiMa; --task-cfg freq_cw=..,alpha=..)
  --suite r4    run_s1h4's tasks (LongBench v2 and HELMET at 128K, from the R4 manifest)
The R5 arms (probe, inj_top / inj_rnd / inj_top_l{q}, the tail arm qread2t4kqT_v4) run here;
Quest runs through run_s1h4's arm in every suite; every other arm runs through the suite's own
arm runner, exactly as in its run. After the run every row gains per-token NLL / KL columns and
the cost model's traffic_frac and exact_rows_frac (s1h5_lib). Besides the usual parquet and
sidecar (s1h_evaluate_*), the probe and injected-error records are written to
  s1h5_probe_heads_<model>_<ctx>.parquet    per unit, layer, query head, answer step
  s1h5_probe_groups_<model>_<ctx>.parquet   per unit, layer, KV head, answer step (budgets)
  s1h5_trace_<model>_<ctx>.parquet          per unit, layer, KV head, bound, eps (the controller on FP's path)
  s1h5_inject_<model>_<ctx>.parquet         per unit, arm, eps, layer, query head, step

    python run_s1h5.py --suite r3 --mode evaluate --preset h5llama128 --ctx 131072 \\
        --tasks niah_single,niah_multikey,niah_multivalue,vt --n-prompts 10 --prompt-offset 9100 --out-dir DIR
"""
from __future__ import annotations
import json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_s1h2 as RH2  # noqa: E402  (sets up every other path; imports run_s1h)
import run_s1h as RH  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1d as S1D  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import run_s1g as S1G  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L1C  # noqa: E402
from sievelib import compress as C, tasks_ruler as TR  # noqa: E402
import s1h5_lib as L  # noqa: E402
import probe_s1h5 as PB  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import stops_s1h as STOPS  # noqa: E402
import run_s1h3 as R3D  # noqa: E402
import run_s1h3b as R3BD  # noqa: E402
import run_s1h4 as R4D  # noqa: E402
import tasks_s1h as T3B  # noqa: E402

SUITES = ("r3", "r3b", "r4")


class _S:
    run_ref = None
    unit = {}
    side = dict(head=[], group=[], inject=[], trace=[])


S = _S()


# ------------------------------------------------------------- shared steps
def _store4(alloc):
    return alloc[("uniform", L.STORE4)]


def _vfn4(Rv, nc):
    return L1B.v_quantizer(4, Rv, nc)


def vote_sets(model, past, q_ids, alloc, rs, R, Rv, nc, L0, nL):
    """The system's question-time selection (run_s1g.run_read2t_g up to the selection: the
    question over the 4-bit tier, its buffered rows vote) at every r in rs -> {r: {li: keep}}."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, _store4(alloc), L.STORE_WIDTH, R, nc, _vfn4(Rv, nc), nL)
    S1D.HOOK.sel = S1D._Sel(rs[0], {}, 0)
    S1D._install()
    out = {}
    try:
        RR._question(model, past, q_ids)
        for r in rs:
            S1D.HOOK.sel.r = float(r)
            S1E.select_all_w(past, nL, 16)
            out[r] = {li: (~e).cpu() for li, e in C.STATE.evict.items()}
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return out


def _fp_pass(model, past, L0, q_ids, ids):
    """FP's teacher-forced pass of `ids` (compression off), through whatever attention is installed."""
    C.STATE.enabled = False
    return RH._ORIG["tf_phased"](model, past, L0, q_ids, ids, False)


def _row(arm, B, gen, past, tfm, extra, t_arm, t_tf, au=None):
    au = au or {"bits_per_token": 16.0, "evict_frac": 0.0}
    stored = dict(stored_bits_per_token=au["bits_per_token"], stored_evict_frac=au["evict_frac"])
    return gen, past, tfm, stored, extra, None, t_arm, t_tf, (arm, B), au, {}


# --------------------------------------------------------------- the probe
def _probe_summary(hf, gf, tf):
    """One row's summary of the probe (the full records are in the side files)."""
    import numpy as np
    out = {}
    if len(hf):
        for k in ("eps_vote1", "eps_votef", "eps_static1", "eps_step1", "mass_ctx"):
            if k in hf:
                out[f"probe_{k}"] = float(hf[k].mean())
        out["probe_eps_vote1_p95"] = float(hf.eps_vote1.quantile(0.95))
        pos = hf.eps_vote1 > 1e-6
        out["probe_tightness_med"] = float((hf.epsbar_vote1[pos] / hf.eps_vote1[pos]).median()) if pos.any() else np.nan
        out["probe_l1_ok"] = float((hf.err_sys <= hf.l1_bound * (1 + 1e-4) + 1e-6).mean())
        out["probe_l3_ok"] = float((hf.err_tail <= hf.l3_bound * (1 + 1e-4) + 1e-6).mean())
        for k in ("sys", "tail", "tailx", "fp8", "d4", "static1"):
            if f"rel_{k}" in hf:
                out[f"probe_rel_{k}_med"] = float(hf[f"rel_{k}"].median())
        # Mode T's certificates (Lemmas 4 and 5): validity / coverage, and how often the bound is within
        # FP8's own error at the same head and step (certified at FP8's level)
        tags = ["tail", "tailx"] + [f"tailx_p{int(round(1 / f))}" for f in PB.EXTRA_FRAC]
        for k in tags:
            if f"l5_{k}" not in hf:
                continue
            err = hf[f"err_{k}"]
            if f"l4_{k}" in hf:
                out[f"probe_l4_{k}_ok"] = float((err <= hf[f"l4_{k}"] * (1 + 1e-4) + 1e-6).mean())
            out[f"probe_l5_{k}_cover"] = float((err <= hf[f"l5_{k}"] * (1 + 1e-4) + 1e-6).mean())
            out[f"probe_l5_{k}_fp8"] = float((hf[f"l5_{k}"] <= hf.err_fp8).mean())
            out[f"probe_err_{k}_fp8"] = float((err <= hf.err_fp8).mean())
        if "trunc_viol" in hf:
            out["probe_trunc_viol_any"] = float((hf.trunc_viol > 0).mean())
    if len(gf):
        for e in L.TRACE_EPS:
            for k in ("union", "cert", "cert_warm", "cert_hp", "page"):
                out[f"probe_{k}_{e}_frac"] = float((gf[f"{k}_{e}"] / gf.ctx).mean())
            for k in ("cert", "cert_hp", "page"):            # B_cert / B_min (shared-set oracle)
                out[f"probe_{k}_over_union_{e}_med"] = float((gf[f"{k}_{e}"] / gf[f"union_{e}"].clip(lower=1)).median())
    if len(tf):
        for (bound, e), g in tf.groupby(["bound", "eps"]):
            k = f"probe_ctl_{bound}_{e}"
            steps = g.steps.clip(lower=1)
            rows = (g.F_sum / steps / g.ctx).mean()
            out[f"{k}_exact_rows_frac"] = float(rows)
            out[f"{k}_traffic_frac"] = float(L.STORE4 / L.FP16_KV + rows * (16 + 4) / L.FP16_KV)   # 4-bit scan + exact rows
            out[f"{k}_refetch_rate"] = float((g.fetch_ev / steps).mean())
            out[f"{k}_fetched_rows_frac"] = float((g.fetched / g.ctx).mean())
            out[f"{k}_cap_rate"] = float((g.cap_steps / steps).mean())
            out[f"{k}_viol_rate"] = float((g.viol / steps).mean())
            out[f"{k}_eps_true_max"] = float(g.eps_max.max())
    return out


def run_probe_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, nc, L0):
    t1 = time.time()
    Cn = C.STATE.ctx_len
    r1, rf = 0.125, L.floor_r(Cn)
    rs = [r1] + ([rf] if rf > r1 + 1e-9 else [])
    votes = vote_sets(model, past, q_ids, alloc, rs, R, Rv, nc, L0, nL)
    ev = RH.oracle_evict(model, past, L0, q_ids, fp_gen, r1, nL, R, nc) if fp_gen else {}
    PB.P.reset()
    PB.P.keep = {"vote1": votes[r1]}
    if len(rs) > 1:
        PB.P.keep["votef"] = votes[rf]
    if ev:
        PB.P.keep["static1"] = {li: (~e).cpu() for li, e in ev.items()}
    PB.P.k_step = L1C.qread_keep_count(r1, Cn)
    PB.P.trace_eps, PB.P.cap_frac = L.TRACE_EPS, L.CAP_FRAC
    PB.P.rows, PB.P.a0 = PB.probe_rows(len(fp_gen)), L0 + q_ids.shape[1] - 1
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, _store4(alloc), L.STORE_WIDTH, R, nc, _vfn4(Rv, nc), nL)   # tier 1, for comparison
    PB.P.mode = "probe"
    PB.install()
    try:
        lg = _fp_pass(model, past, L0, q_ids, fp_gen) if fp_gen else None
    finally:
        PB.uninstall()
        C.STATE.reset_arm()
    t_arm = time.time() - t1
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    unit = dict(S.unit, ctx_len=Cn)
    hf, gf, tf = PB.head_frame(unit), PB.group_frame(unit), PB.trace_frame(unit)
    S.side["head"].append(hf)
    S.side["group"].append(gf)
    S.side["trace"].append(tf)
    PB.P.reset()
    return _row(arm, B, fp_gen, past, tfm, dict(_probe_summary(hf, gf, tf), tf_only=True), t_arm, 0.0)


# ------------------------------------------------------- injected errors
def run_inject_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, L0, nL):
    t1 = time.time()
    PB.P.reset()
    PB.P.mode, PB.P.eps, PB.P.policy = "inject", float(B), pa["policy"]
    PB.P.layers = None if pa.get("quarter") is None else L.quarter_layers(pa["quarter"], nL)
    PB.P.seed = int(S.unit.get("prompt_idx", 0))
    PB.P.rows, PB.P.a0 = PB.probe_rows(len(fp_gen)), L0 + q_ids.shape[1] - 1
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    PB.install()
    try:
        lg = _fp_pass(model, past, L0, q_ids, fp_gen) if fp_gen else None
    finally:
        PB.uninstall()
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    hf = PB.head_frame(dict(S.unit, arm=arm, eps=float(B), quarter=-1 if pa.get("quarter") is None else pa["quarter"]))
    S.side["inject"].append(hf)
    PB.P.reset()
    extra = dict(tf_only=True)
    if len(hf):
        extra.update(inj_eps_mean=float(hf.eps_real.mean()), inj_eps_max=float(hf.eps_real.max()),
                     inj_rel_mean=float(hf.rel_inj.mean()), inj_kept_frac=float(hf.kept.mean() / C.STATE.ctx_len))
    return _row(arm, B, fp_gen, past, tfm, extra, time.time() - t1, 0.0)


# -------------------------------------------- the tail arm (Lemma 3's design)
def _tail_steps(model, past, L0, q_ids, r, pa, alloc, R, Rv, nc, nL, teacher):
    """The system's steps (run_s1g.run_read2t_g) up to the answer, then nothing evicted: the
    selected rows keep tier 2's exact keys, the others tier 1's 4-bit keys; values 4-bit."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, _store4(alloc), L.STORE_WIDTH, R, nc, _vfn4(Rv, nc), nL)
    S1D.HOOK.sel = S1D._Sel(r, {}, 0)
    S1D._install()
    if teacher:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
    else:
        past = RR._question(model, past, q_ids)
    S1E.select_all_w(past, nL, pa["tier2_bits"])
    sel = {li: e.clone() for li, e in C.STATE.evict.items()}
    S1G.to_tier2(past, nL, sel, pa["tier2"], pa["kv"])
    if pa["requestion"]:
        past = S1G._requestion(model, past, L0, q_ids, sel, teacher)
    for li, e in sel.items():
        C.STATE.evict[li] = torch.zeros_like(e)
        C.STATE.bits[li] = torch.where(~e, int(pa["tier2_bits"]), int(L.STORE4)).long()
    S1D.HOOK.sel.buf.phase = "answer"
    return past, sel


def _tf_tail(model, past, L0, q_ids, ids, r, pa, alloc, R, Rv, nc, nL):
    if not ids:
        return None
    try:
        past, _ = _tail_steps(model, past, L0, q_ids, r, pa, alloc, R, Rv, nc, nL, teacher=True)
        C.STATE.enabled = True
        x = [int(q_ids[0, -1])] + [int(t) for t in ids[:-1]]
        with torch.no_grad():
            lg = model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past,
                       use_cache=True).logits[0].float()
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(ids)]


def run_tail_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, nc, eos, max_new, L0, tok):
    t1 = time.time()
    try:
        past, sel = _tail_steps(model, past, L0, q_ids, B, pa, alloc, R, Rv, nc, nL, teacher=False)
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    t_arm = time.time() - t1
    au = C.bits_audit()
    tt = time.time()
    lg = _tf_tail(model, past, L0, q_ids, fp_gen, B, pa, alloc, R, Rv, nc, nL)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: _tf_tail(model, past, L0, q_ids, ids, B, pa, alloc, R, Rv, nc, nL)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, ("tail", "", False, arm)))
    extra = dict(store_width=int(L.STORE4), tier2=pa["tier2"], tier2_bits=pa["tier2_bits"], keys_only=not pa["kv"],
                 requestion=bool(pa["requestion"]), read_rows_exact=float(B))
    return gen, past, tfm, dict(stored_bits_per_token=float(L.STORE4), stored_evict_frac=0.0), extra, sel, t_arm, \
        t_tf, (arm, B), dict(au, evict_frac=0.0), {}


# ------------------------------------------- R5.3: the tail design's scan (family tail5)
def _tier1(past, kb, vb, R, Rv, nc, nL):
    """Tier 1 at kb-bit keys and vb-bit values (rotated Lloyd-Max, the store's quantizer), nothing evicted."""
    from sievelib.probe import cache_kv
    store = {}
    for li in range(nL):
        K, _ = cache_kv(past, li)
        store[li] = torch.full((K.shape[0], C.STATE.ctx_len), int(kb), dtype=torch.long, device=K.device)
    S1E.apply_store(past, store, int(kb), R, nc, L1B.v_quantizer(int(vb), Rv, nc), nL)


def _tail5_steps(model, past, L0, q_ids, r, pa, R, Rv, nc, nL, teacher, fp_gen):
    """tail{kb}[x][o]_v{vb}: tier 1, the question over it, the selection (the system's vote, or the static
    oracle's rows of FP's answer), tier 2's exact keys (and values with 'x') on the selected rows, the question
    again, then nothing evicted for the answer."""
    ev_or = RH.oracle_evict(model, past, L0, q_ids, fp_gen, r, nL, R, nc) if pa["oracle_sel"] else None
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    _tier1(past, pa["tier1_key_bits"], pa["v_bits"], R, Rv, nc, nL)
    if ev_or is None:
        S1D.HOOK.sel = S1D._Sel(r, {}, 0)
        S1D._install()
    if teacher:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
    else:
        past = RR._question(model, past, q_ids)
    if ev_or is None:
        S1E.select_all_w(past, nL, 16)
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
    else:
        sel = {li: e.to(C.STATE.kdeq[li].device).clone() for li, e in ev_or.items()}
    S1G.to_tier2(past, nL, sel, pa["tier2"], pa["kv"])
    if pa["requestion"]:
        past = S1G._requestion(model, past, L0, q_ids, sel, teacher)
    for li, e in sel.items():
        C.STATE.evict[li] = torch.zeros_like(e)
        C.STATE.bits[li] = torch.where(~e, int(pa["tier2_bits"]), int(pa["tier1_key_bits"])).long()
    if S1D.HOOK.sel is not None:
        S1D.HOOK.sel.buf.phase = "answer"
    return past, sel


def _tf_tail5(model, past, L0, q_ids, ids, r, pa, R, Rv, nc, nL, fp_gen):
    if not ids:
        return None
    try:
        past, _ = _tail5_steps(model, past, L0, q_ids, r, pa, R, Rv, nc, nL, True, fp_gen)
        C.STATE.enabled = True
        x = [int(q_ids[0, -1])] + [int(t) for t in ids[:-1]]
        with torch.no_grad():
            lg = model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past,
                       use_cache=True).logits[0].float()
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(ids)]


def run_tail5_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok):
    t1 = time.time()
    try:
        past, sel = _tail5_steps(model, past, L0, q_ids, B, pa, R, Rv, nc, nL, False, fp_gen)
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    t_arm = time.time() - t1
    au = C.bits_audit()
    tt = time.time()
    lg = _tf_tail5(model, past, L0, q_ids, fp_gen, B, pa, R, Rv, nc, nL, fp_gen)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: _tf_tail5(model, past, L0, q_ids, ids, B, pa, R, Rv, nc, nL, fp_gen)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, ("tail5", "", False, arm)))
    extra = dict(store_width=int(pa["tier1_key_bits"]), tier1_value_bits=int(pa["v_bits"]), tier2=pa["tier2"],
                 tier2_bits=pa["tier2_bits"], exact_values_on_selected=bool(pa["kv"]), oracle_selection=bool(pa["oracle_sel"]),
                 requestion=bool(pa["requestion"]), read_rows_exact=float(B))
    return gen, past, tfm, dict(stored_bits_per_token=float(pa["tier1_key_bits"]), stored_evict_frac=0.0), extra, \
        sel, t_arm, t_tf, (arm, B), dict(au, evict_frac=0.0), {}


# ------------------------------------------------------------------ dispatch
def run_arm_h5(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc, eos, max_new,
               L0, tok, last, sel_q1=None):
    fam = pa["family"]
    if fam == "quest":                                   # Quest in every suite (run_s1h4's arm, unchanged)
        return R4D.run_arm_h4(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc,
                              eos, max_new, L0, tok, last, sel_q1)
    if fam not in ("probe", "inject", "tail", "tail5"):
        return S.run_ref(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, nc, eos,
                         max_new, L0, tok, last, sel_q1)
    RH._fold()
    RH._ORIG["reset_peaks"]()
    base = (sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
            if torch.cuda.is_available() else float("nan"))
    if fam == "probe":
        res = run_probe_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, nc, L0)
    elif fam == "inject":
        res = run_inject_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, L0, nL)
    elif fam == "tail5":
        res = run_tail5_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok)
    else:
        res = run_tail_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, nc, eos, max_new,
                           L0, tok)
    pk = RH._dev_peaks()
    RH._fold()
    res[4].update(peak_gib_arm=sum(pk) if pk else float("nan"), base_gib_arm=base)
    return res


# ---------------------------------------------------------------------- main
def _arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _drop(argv, name):
    if name not in argv:
        return argv
    i = argv.index(name)
    return argv[:i] + argv[i + 2:]


def _span_ntok(vmask, span_end):
    vm = str(vmask or "")
    return max(int(span_end) - vm.index("1"), 1) if "1" in vm and span_end == span_end else float("nan")


def _add_columns(out_dir, mode):
    """Every row (fp included) gains the per-token NLL / KL and the cost model's columns."""
    import numpy as np
    import pandas as pd
    pq = [f for f in os.listdir(out_dir) if f.startswith(f"s1h_{mode}_") and f.endswith(".parquet")]
    if len(pq) != 1:
        raise RuntimeError(f"{out_dir}: expected one evaluate parquet, found {pq}")
    p = os.path.join(out_dir, pq[0])
    d = pd.read_parquet(p)
    n = np.array([_span_ntok(v, e) for v, e in zip(d.get("tf_vmask", [None] * len(d)), d.span_end)], dtype=float)
    T = d.tf_logp.map(lambda x: len(x) if x is not None else 0).astype(float).replace(0, np.nan)
    d["a_span_ntok"] = n
    d["a_span_nll_tok"] = d.a_span_nll / n
    d["kl_span_tok"] = d.kl_span / n
    d["kl_all_tok"] = d.kl_all / T
    d["traffic_frac"] = [L.traffic_frac(a, float(b), int(c)) for a, b, c in zip(d.arm, d.B, d.ctx_len)]
    d["exact_rows_frac"] = [L.exact_rows_frac(a, float(b), int(c)) for a, b, c in zip(d.arm, d.B, d.ctx_len)]
    d.to_parquet(p)


def _write_side(out_dir, mode):
    import pandas as pd
    side_p = [f for f in os.listdir(out_dir) if f.startswith(f"s1h_{mode}_") and f.endswith(".json")]
    if len(side_p) != 1:
        raise RuntimeError(f"{out_dir}: expected one sidecar, found {side_p}")
    sp = os.path.join(out_dir, side_p[0])
    side = json.load(open(sp))
    stem = f"{side['model']}_{side['ctx']}"
    files = {}
    for k, name in (("head", "probe_heads"), ("group", "probe_groups"), ("trace", "trace"), ("inject", "inject")):
        parts = [x for x in S.side[k] if len(x)]
        if parts:
            f = f"s1h5_{name}_{stem}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(os.path.join(out_dir, f))
            files[name] = f
    return sp, side, files


def main():
    argv = sys.argv[1:]
    suite = _arg(argv, "--suite") or "r3"
    argv = _drop(argv, "--suite")
    if suite not in SUITES:
        raise SystemExit(f"--suite must be one of {SUITES}")
    mode, out_dir, preset = _arg(argv, "--mode"), _arg(argv, "--out-dir"), _arg(argv, "--preset")
    if mode != "evaluate":
        raise SystemExit("R5 runs evaluation blocks only")
    if preset not in L.PRESETS or not (L.PRESETS[preset].get("probe") or L.PRESETS[preset].get("tail5")):
        raise SystemExit(f"--preset must be an R5 preset (h5*, h53*), not {preset!r}")
    cfg = None
    if suite == "r3":
        argv, cfg = R3D._pop_task_cfg(argv)
    elif suite == "r3b":
        argv, cfg = R3BD._pop_task_cfg(argv)
        cfg = T3B.task_config_r3b(**(cfg or T3B.DEFAULT_CFG))
    sys.argv = [sys.argv[0]] + argv
    S1F._A2_CHECKED.clear()
    S.side = dict(head=[], group=[], inject=[], trace=[])
    if suite == "r3":
        R3D.install_tasks(cfg)
        S.run_ref = RH2.run_arm_h2
    elif suite == "r3b":
        R3BD.install_tasks(cfg)
        S.run_ref = RH2.run_arm_h2
    else:
        if preset not in L.R4_PRESET_OF:
            raise SystemExit(f"--suite r4 runs at 128K (or the smoke): {sorted(L.R4_PRESET_OF)}")
        R4D.CUR.preset, R4D.CUR.model = L.R4_PRESET_OF[preset], L.PRESETS[preset]["model"]
        R4D.CUR.smoke = preset.endswith("smoke")
        R4D.install_tasks()
        S.run_ref = R4D.run_arm_h4
    STOPS.install(S1E, RR)
    RH.install()
    S1E.L, S1E.run_arm = L, run_arm_h5
    build0 = TR.build

    def build_r5(tok, task, ctx, *, prompt_idx, **kw):
        S.unit = dict(prompt_idx=int(prompt_idx), task=task)
        return build0(tok, task, ctx, prompt_idx=prompt_idx, **kw)
    TR.build = build_r5
    try:
        S1E.main()
    finally:
        TR.build = build0
        RH.uninstall()
        STOPS.uninstall(S1E, RR)
        (R3D.uninstall_tasks() if suite == "r3" else R3BD.uninstall_tasks() if suite == "r3b"
         else R4D.uninstall_tasks())
    if out_dir:
        RH._rename_outputs(out_dir, mode)
        _add_columns(out_dir, mode)
        sp, side, files = _write_side(out_dir, mode)
        side.update(lib="s1h5_lib", amend_r5=L.AMEND_R5, amend_r53=L.AMEND_R53 if preset.startswith("h53") else None,
                    suite=suite, task_cfg_r5=cfg, r5_files=files,
                    probe=dict(grid=list(K.EPS_GRID), page=PB.PAGE, max_rows=PB.MAX_ROWS),
                    driver="run_s1h5.py (run_s1e.main on a suite's units; R5 probe, injected errors, tail arm)")
        if suite == "r4":
            side.update(manifest_sha256=R4D.T.MANIFEST_SHA256, manifest_cell=R4D._cell(side["tasks"][0]))
        with open(sp, "w") as fh:
            json.dump(side, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
