#!/usr/bin/env python3
"""R14 Stage 1h driver (design: plan.md, s1h_lib.py; frozen rules: read_stage1h.py).

run_s1e.main with Stage 1h's presets (s1h_lib) and run_arm_h, which runs Stage 1g's
arms unchanged (run_s1g.run_arm_g) and adds Stage 1h's (fp_noise, fp8kv, kivi/kvquant,
oracle reads). Same arguments as run_s1e.py, e.g.

    python run_s1h.py --mode evaluate --preset h1cal --ctx 131072 --n-prompts 10 --prompt-offset 9100 --out-dir DIR
    python run_s1h.py --mode evaluate --preset h1regress --ctx 131072 \
        --prompt-list 8109:niah_multikey,8901:niah_multikey,8937:niah_multikey --override rot_seed=1 --out-dir DIR

RUN-TIME ADDITIONS (plan.md section 3), installed for the run and removed after it:
  KL to FP      S1D.tf_phased is wrapped: FP's teacher-forced pass (compressed=False)
                stores FP's log-probabilities, keyed by FP's answer. Every arm's metric
                function (S1E.tfm_of and S1F.tfm_of_f, through which all Stage 1e-1g
                arms pass) adds metrics_s1h.kl_columns against it.
  peak memory   each arm's peak and starting GPU memory (peak_gib_arm, base_gib_arm);
                the per-prompt peak_gib keeps its meaning (a running maximum).
  fp_noise      the context is re-prefilled at s1h_lib.noise_chunk (RR.prefill is
                wrapped to record the run's prefill), the first cache is released, and
                FP runs again on the new cache. It must be the last arm.
  fp8kv, kivi*, kvquant*   dense views through compress.apply_bits(keys_fn, values_fn),
                replayed by S1D.tf_phased like every dense arm.
  qoracle*      FP's answer is teacher-forced over the exact store with the row buffer
                of the question-time reads unbounded; the answer's rows score the
                context (s1d_lib.rows_scores) and s1d_lib.select_keep keeps floor(r C)
                per KV head (pooled, as the vote). The arm then reads the store as qread
                does, with that selection. Cached per (FP answer, r) within a prompt.
Every new arm also gets A2's columns (own-order replay, self-check) through its own
replay path. Outputs are renamed s1e_* -> s1h_*; the sidecar gains stage = "1h".
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import OrderedDict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import compress as C, kv_quant_baselines as QB  # noqa: E402
from sievelib.probe import cache_kv  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1d as S1D  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import run_s1g as S1G  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1d_lib as L1D  # noqa: E402
import s1h_lib as L  # noqa: E402
import metrics_s1h as M  # noqa: E402

_ORIG = dict(S1E_L=S1E.L, run_arm=S1E.run_arm, tfm_of=S1E.tfm_of, tfm_of_f=S1F.tfm_of_f, tf_phased=S1D.tf_phased,
             reset_peaks=S1D.reset_peaks, peak_gib_dev=S1E.peak_gib_dev, prefill=RR.prefill)
ORACLE_CHUNK = 16                           # answer rows per rows_scores call
FP_REF_KEEP = 4                             # FP references kept (one per question)


class _State:
    """Per-process state of the Stage 1h additions."""

    def __init__(self):
        self.fp_ref = OrderedDict()         # FP's answer -> FP's teacher-forced log-probs [T, V]
        self.oracle = {}                    # (FP's answer, r) -> evict masks, cleared per question
        self.prefill = None                 # (ids, window, chunk) of the run's last context prefill
        self.rope = {}                      # (n, device) -> RoPE tables (kvquant)
        self.prompt_peak = []               # per-device running maximum, GiB


H = _State()


# ------------------------------------------------------------- KL to FP
def _key(ids):
    return tuple(int(x) for x in ids)


def tf_phased_h(model, past, L0, q_ids, fp_gen, compressed, before_q=None, before_a=None, seg=0):
    """S1D.tf_phased; FP's own pass (compressed=False) also stores FP's log-probs."""
    lg = _ORIG["tf_phased"](model, past, L0, q_ids, fp_gen, compressed, before_q, before_a, seg)
    if not compressed and lg is not None:
        k = _key(fp_gen)
        if k not in H.fp_ref:
            H.oracle.clear()
        H.fp_ref[k] = torch.log_softmax(lg.float(), -1)
        H.fp_ref.move_to_end(k)
        while len(H.fp_ref) > FP_REF_KEEP:
            H.fp_ref.popitem(last=False)
    return lg


def tfm_of_h(lg, fp_gen, content, am):
    """Stage 1f's metric function (A2 columns) plus KL to FP's stored log-probs."""
    out = _ORIG["tfm_of_f"](lg, fp_gen, content, am)
    if lg is not None and out:
        ref = H.fp_ref.get(_key(fp_gen))
        if ref is None:
            raise RuntimeError("KL: no FP reference for this answer (FP's teacher-forced pass did not run first)")
        out.update(M.kl_columns(lg, ref, am["vmask"], am["span_end"]))
    return out


# ------------------------------------------------------------- peak memory
def _dev_peaks():
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]


def _fold():
    cur = _dev_peaks()
    n = max(len(cur), len(H.prompt_peak))
    H.prompt_peak = [max(H.prompt_peak[i] if i < len(H.prompt_peak) else 0.0, cur[i] if i < len(cur) else 0.0)
                     for i in range(n)]


def reset_peaks_h():
    _ORIG["reset_peaks"]()
    H.prompt_peak = []


def peak_gib_dev_h():
    _fold()
    return list(H.prompt_peak)


def prefill_h(model, ids, window, chunk, h2o=False):
    out = _ORIG["prefill"](model, ids, window, chunk, h2o=h2o)
    H.prefill = (ids, int(window), int(chunk))
    H.oracle.clear()
    return out


# ------------------------------------------------------------- dense views
def _width_bits(past, nL, width):
    Cn = C.STATE.ctx_len
    out = {}
    for li in range(nL):
        K, _ = cache_kv(past, li)
        out[li] = torch.full((K.shape[0], Cn), int(width), dtype=torch.long, device=K.device)
    return out


def _rope(model, n):
    dev = next(model.parameters()).device
    k = (int(n), str(dev))
    if k not in H.rope:
        H.rope.clear()
        H.rope[k] = QB.rope_tables(model, int(n), dev)
    return H.rope[k]


def view_fns(model, pa, Rv, norm_correct):
    """(width, keys_fn, values_fn) of a Stage 1h dense view."""
    if pa["family"] == "fp8kv":
        return QB.FP8_BITS, QB.fp8_fn(), QB.fp8_fn()
    lab, b = pa["kq_label"], int(pa["store"])
    rope = _rope(model, C.STATE.ctx_len) if QB.needs_rope([lab]) else None
    vfn = L1B.v_quantizer(pa["v_bits"], Rv, norm_correct) if pa["v_bits"] < 16 else None
    return b, QB.keys_fn(lab, b, rope), vfn


def run_dense_view(model, past, q_ids, width, keys_fn, values_fn, nL, R, norm_correct, eos, max_new, L0, tok):
    """Every context key through keys_fn (and every context value through
    values_fn), nothing evicted; the question prefilled over that view, then the
    answer decoded (run_r8.run_bits with values)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    C.apply_bits(past, _width_bits(past, nL, width), R, norm_correct, keys_fn=keys_fn, values_fn=values_fn)
    try:
        past = RR._question(model, past, q_ids)
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        C.STATE.enabled = False
    return gen, past


# ------------------------------------------------------------- oracle reads
def oracle_evict(model, past, L0, q_ids, fp_gen, r, nL, R, norm_correct):
    """The oracle's selection (module docstring): evict masks [Hkv, C] per layer,
    True = not read. Cached per (FP's answer, r) until the next question."""
    k = (_key(fp_gen), float(r))
    if k in H.oracle:
        return H.oracle[k]
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, None, L.EXACT_WIDTH, R, norm_correct, None, nL)
    sel = S1D._Sel(r, {}, 0)
    sel.buf.rows = 1 << 30                  # keep every row: the answer's rows are filtered below
    S1D.HOOK.sel = sel
    S1D._install()
    C.STATE.enabled = True
    ev = {}
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past, use_cache=True)
            a0 = L0 + q_ids.shape[1] - 1     # absolute position of the answer's first row
            Cn = C.STATE.ctx_len
            for li in range(nL):
                K, _ = cache_kv(past, li)
                q, pos = sel.buf.q[li], sel.buf.pos[li]
                m = pos >= a0
                qa, pa_ = q[m], pos[m]
                if qa.shape[0] != len(x):
                    raise RuntimeError(f"oracle: layer {li} buffered {qa.shape[0]} answer rows, expected {len(x)}")
                score = None
                for s in range(0, qa.shape[0], ORACLE_CHUNK):
                    sc = L1D.rows_scores(qa[s:s + ORACLE_CHUNK], pa_[s:s + ORACLE_CHUNK], C.STATE.kdeq[li], K, Cn,
                                         sel.scaling[li])
                    score = sc if score is None else score + sc
                ev[li] = ~L1D.select_keep(score, r)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    H.oracle[k] = ev
    return ev


def _set_selection(ev_or, width):
    for li, ev in ev_or.items():
        dev = C.STATE.kdeq[li].device
        C.STATE.evict[li] = ev.to(dev).clone()
        C.STATE.bits[li] = torch.where(~C.STATE.evict[li], int(width), 0).long()


def _oracle_store(pa, alloc):
    w = int(pa["store"])
    if w == L.EXACT_WIDTH:
        return None
    if w == L.STORE4:
        return alloc[("uniform", L.STORE4)]
    return alloc[("qread_store", L.STORE_WIDTH)]


def run_oracle(model, past, q_ids, store, width, ev_or, R, norm_correct, vfn, eos, max_new, L0, tok, nL):
    """The read with the oracle's selection: the question over the whole store,
    then the answer reads the selected rows."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, store, width, R, norm_correct, vfn, nL)
    try:
        past = RR._question(model, past, q_ids)
        _set_selection(ev_or, width)
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        C.STATE.enabled = False
    return gen, past, sel


def tf_oracle(model, past, L0, q_ids, ids, store, width, ev_or, R, norm_correct, vfn, nL):
    """The same read, teacher-forced on `ids`."""
    if not ids:
        return None
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, store, width, R, norm_correct, vfn, nL)
    C.STATE.enabled = True
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            _set_selection(ev_or, width)
            x = [int(q_ids[0, -1])] + [int(t) for t in ids[:-1]]
            inp = torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype)
            lg = model(inp, past_key_values=past, use_cache=True).logits[0].float()
    finally:
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(ids)]


# ------------------------------------------------------------- the noise arm
def _release(past):
    """Drop the first cache's tensors before the second prefill (fp_noise is the
    last arm; the driver continues with the cache it returns). Best effort."""
    try:
        if hasattr(past, "layers"):
            for lay in past.layers:
                for name in ("keys", "values"):
                    t = getattr(lay, name, None)
                    if torch.is_tensor(t):
                        setattr(lay, name, t[..., :0, :].clone())
        elif hasattr(past, "key_cache"):
            for i in range(len(past.key_cache)):
                past.key_cache[i] = past.key_cache[i][..., :0, :].clone()
                past.value_cache[i] = past.value_cache[i][..., :0, :].clone()
    except Exception:                                                  # noqa: BLE001
        pass
    C.STATE.reset_arm()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_noise(model, past, q_ids, eos, max_new, L0, tok):
    """FP on a second prefill of the same context at a different chunk size.
    Returns (gen, the new cache, the chunk used)."""
    if H.prefill is None:
        raise RuntimeError("fp_noise: the run's context prefill was not recorded")
    ids, window, chunk = H.prefill
    ch = L.noise_chunk(chunk, ids.shape[1])
    _release(past)
    past2, _ = _ORIG["prefill"](model, ids, window, ch, h2o=False)
    if C.cache_len(past2) != L0:
        raise RuntimeError(f"fp_noise: second prefill holds {C.cache_len(past2)} tokens, expected {L0}")
    gen, past2 = S1E.fp_run(model, past2, q_ids, eos, max_new, L0, tok)
    return gen, past2, ch


# ------------------------------------------------------------- the arm
def run_new_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos, max_new,
                L0, tok):
    """One Stage 1h arm (run_s1e.run_arm's return tuple), with A2's columns."""
    fam, v = pa["family"], pa["v_bits"]
    t1 = time.time()
    stored, extra, sel, nk_state = None, {}, None, {}
    if fam == "fpnoise":
        gen, past, ch = run_noise(model, past, q_ids, eos, max_new, L0, tok)
        t_arm = time.time() - t1
        au = {"bits_per_token": 16.0, "evict_frac": 0.0}
        stored = dict(stored_bits_per_token=16.0, stored_evict_frac=0.0)
        extra.update(noise_chunk=ch, prefill_chunk=H.prefill[2])
        tt = time.time()
        lg = _ORIG["tf_phased"](model, past, L0, q_ids, fp_gen, False)
        replay = lambda ids: _ORIG["tf_phased"](model, past, L0, q_ids, ids, False)  # noqa: E731
    elif fam in ("fp8kv", "kivi", "kvquant"):
        width, kf, vf = view_fns(model, pa, Rv, norm_correct)
        gen, past = run_dense_view(model, past, q_ids, width, kf, vf, nL, R, norm_correct, eos, max_new, L0, tok)
        t_arm = time.time() - t1
        au = C.bits_audit()
        extra.update(kq_label=pa.get("kq_label", "fp8_e4m3"), v_format="fp8_e4m3" if fam == "fp8kv" else
                     ("exact" if v >= 16 else f"turboquant{v}"))
        tt = time.time()
        lg = S1D.tf_phased(model, past, L0, q_ids, fp_gen, True)
        replay = lambda ids: S1D.tf_phased(model, past, L0, q_ids, ids, True)  # noqa: E731
    elif fam in ("qoraclefp", "qoracle"):
        width = int(pa["store"])
        store = _oracle_store(pa, alloc)
        vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
        t0 = time.time()
        ev_or = oracle_evict(model, past, L0, q_ids, fp_gen, B, nL, R, norm_correct)
        t_or = time.time() - t0
        gen, past, sel = run_oracle(model, past, q_ids, store, width, ev_or, R, norm_correct, vfn, eos, max_new, L0,
                                    tok, nL)
        t_arm = time.time() - t1
        au = C.bits_audit()
        nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
        stored = dict(stored_bits_per_token=float(width), stored_evict_frac=0.0)
        extra.update(store_width=width, oracle=True, t_oracle=t_or)
        tt = time.time()
        lg = tf_oracle(model, past, L0, q_ids, fp_gen, store, width, ev_or, R, norm_correct, vfn, nL)
        replay = lambda ids: tf_oracle(model, past, L0, q_ids, ids, store, width, ev_or, R, norm_correct,  # noqa: E731
                                       vfn, nL)
    else:
        raise ValueError(f"not a Stage 1h arm: {arm}")
    tfm = tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, (fam, pa["twin"], False, pa["base"])))
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, (arm, B), au, nk_state


def run_arm_h(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
              eos, max_new, L0, tok, last, sel_q1=None):
    """run_s1g.run_arm_g (Stage 1g's arms, unchanged) or a Stage 1h arm, plus the
    arm's peak GPU memory."""
    _fold()
    _ORIG["reset_peaks"]()
    base = (sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
            if torch.cuda.is_available() else float("nan"))
    if pa["family"] in L.NEW_FAMILIES:
        res = run_new_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos,
                          max_new, L0, tok)
    else:
        res = S1G.run_arm_g(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv,
                            norm_correct, eos, max_new, L0, tok, last, sel_q1)
    pk = _dev_peaks()
    _fold()
    res[4].update(peak_gib_arm=sum(pk) if pk else float("nan"), base_gib_arm=base)
    return res


# ------------------------------------------------------------------------ main
def install():
    S1E.L, S1E.run_arm, S1E.tfm_of = L, run_arm_h, tfm_of_h
    S1F.tfm_of_f = tfm_of_h
    S1D.tf_phased = tf_phased_h
    S1D.reset_peaks = reset_peaks_h
    S1E.peak_gib_dev = peak_gib_dev_h
    RR.prefill = prefill_h


def uninstall():
    S1E.L, S1E.run_arm, S1E.tfm_of = _ORIG["S1E_L"], _ORIG["run_arm"], _ORIG["tfm_of"]
    S1F.tfm_of_f = _ORIG["tfm_of_f"]
    S1D.tf_phased = _ORIG["tf_phased"]
    S1D.reset_peaks = _ORIG["reset_peaks"]
    S1E.peak_gib_dev = _ORIG["peak_gib_dev"]
    RR.prefill = _ORIG["prefill"]


def _rename_outputs(out_dir, mode):
    """s1e_* -> s1h_* in out_dir; the sidecar's file fields follow; stage = '1h'."""
    if not os.path.isdir(out_dir):
        return
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("s1e_"):
            os.replace(os.path.join(out_dir, f), os.path.join(out_dir, "s1h_" + f[4:]))
    for f in os.listdir(out_dir):
        if f.startswith(f"s1h_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            for k in ("parquet", "search", "searchlog"):
                if isinstance(side.get(k), str) and side[k].startswith("s1e_"):
                    side[k] = "s1h_" + side[k][4:]
            side.update(stage="1h", driver="run_s1h.py (run_s1e.main with s1h_lib presets, Stage 1g read paths and "
                        "Stage 1h arms)", amend=L.AMEND, amend_1h=L.AMEND_1H, metric_a2="s_set_nll (Stage 1f A2)",
                        metric_primary=list(L.METRIC_PRIMARY), k_min=L.K_MIN,
                        metric_kl="KL(FP || arm) per token under teacher forcing on FP's answer; kl_span sums the "
                                  "A2 span (FP's first answer-value token to the span end)",
                        noise="fp_noise: context re-prefilled at s1h_lib.noise_chunk(chunk) = chunk // 2",
                        oracle="qoracle*: s1d_lib.select_keep of FP's answer-time rows_scores (teacher-forced, exact "
                               "store), question over the whole store",
                        kv_quant=QB.config_record())
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
    install()
    try:
        S1E.main()
    finally:
        uninstall()
    if a.out_dir and a.mode:
        _rename_outputs(a.out_dir, a.mode)


if __name__ == "__main__":
    main()
