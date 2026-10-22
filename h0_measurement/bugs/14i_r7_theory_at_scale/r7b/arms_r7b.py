"""R7b's arms at run time (lib_r7b's families dnest, wnest, ddense, cnest, kcal), inside run_s1h5's chain. R6's files are
read-only: R6's own steps are reused (run_s1h5._tail5_steps / _tf_tail5: the vote, tier 2 on the read rows, the question
again, decode and the teacher-forced pass), with run_s1h5._tier1 (the vote's store) swapped for the single store's during
an arm, and the tier swap passed in as R6's dtail arms do.

THE SINGLE STORE (dnest4{r}ux_v{v}; store_r7b, a bit-for-bit copy of ../../14h_theory/nested_s1h5):
  vote_store   the question pass's view (run_s1h5._tier1's place): the base keys, quant.quantize_keys at 4 bits in the
               tier's rotation (seed 50), and the tier's dithered values (tier_r7.dither_quantize, rotation 150, seed
               unit + 2 layer + 1: the same call as R6's _dither_swap, so dnest and dtail share their value coins).
               Nothing evicted. R6's vote (run_s1e.select_all_w: SnapKV over the question's last rows) then reads it.
  nest_swap    after the selection: the context keys become the nested reconstruction (base + r-bit dithered residual,
               seed unit + 2 layer); the values stay the store's; the per-coordinate debias is installed over whatever
               attention is installed (the selection hook), as R6's _dither_swap installs its per-row one.
  _pc_attention  the unbiased tail's score correction: ldb = (lmu_lo + lmu_hi) / 2 of the per-coordinate coin moments
               (A2 = (sc/2)^2 sum_j (Rq)_j^2 D_ij^2, A4 = (sc/2)^4 sum_j (Rq)_j^4 D_ij^4; store_r7b.coin_moments_pc and
               tier_r7.debias_log), subtracted from every unread context row's score as an additive mask. The steps
               D_ij come from the stored base index, row maximum and norm (store_r7b.cell_steps), streamed over rows.
DENSE DITHERED (ddense{k}u_v{v}): R6's per-row dithered tier (run_s1h5._dither_swap) on every context row, R6's unbiased
  mask with no row read exactly; no vote, nothing exact.
E1 (cnest42ux_v4_<level>): the single store, its vote and question pass as dnest (teacher-forced); then FP's answer
  teacher-forced with the certified read in every layer: per KV head and answer step, R7a's certified_read (part 8 at the
  store's per-coordinate steps, cert_r7b; the FINE schedule: the vote, then rows by the base sketch's priority) stops at
  the first set whose bound meets tau_l for every query head of the group (the exact read otherwise), and its output
  replaces the attention's. Records the rows read, the fallbacks and the query heads whose error met tau_l.
CALIBRATION (kcal_{tail,key,val}_l{li}): the single store's vote (computed once per unit); FP's question; then FP's answer
  teacher-forced with layer li alone reading the store's tail (keys and values / keys only / values only; the vote's rows
  exact), every other layer exact. Records layer li's realized output error per answer step (sum over heads of
  |o_arm - o_exact|^2): c_l = 2 KL / err2 (../../14h_theory/theory.md §14).
"""
from __future__ import annotations

import contextlib
import functools
import json
import math
import time

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sievelib import compress as C
from sievelib import quant as Q
from sievelib.probe import _sdpa, cache_kv

import run_s1h5 as RS5                     # R6's driver (read-only)
import tier_r7 as T7                       # R7a's tier (pinned; a copy of rquant_s1h5)
import cert_r7b as K7B                     # the bounds at per-coordinate steps
import store_r7b as S7B                    # the single store
import lib_r7b as LB

RH, RR, S1D, S1F = RS5.RH, RS5.RR, RS5.S1D, RS5.S1F
UNIT = dict(seed=None, task=None, prompt=None, group=None)
ROW_CHUNK = 8192                           # rows per debias step
Q_CHUNK = 64                               # queries per debias step (R6's)
CR_CHUNK = 4                               # answer steps per certified-read chunk (R7a's probe)
CR_EXACT_ROWS, CR_HEAVY_ROWS, CR_SHARE = 16, 16, 0.5      # part 8's settings (R7a's probe)
CR_DELTA = K7B.DELTA_T


def _dt(dev) -> torch.dtype:
    """The tier's arithmetic: float32 on the GPU (R6's _dither_swap), float64 on the CPU."""
    return torch.float32 if dev.type == "cuda" else torch.float64


@contextlib.contextmanager
def _swapped(mod, name, fn):
    old = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        yield
    finally:
        setattr(mod, name, old)


# ------------------------------------------------------------------ the per-coordinate debias
PC = dict(on=False, layers={}, orig=None)  # layer -> (idx uint8 [G, C, d], M [G, C], gamma [G, C], read [G, C], Rk, r)


def debias_pc(q: torch.Tensor, idx, M, gamma, Rk, res_bits: int, sc: float, chunk: int = ROW_CHUNK) -> torch.Tensor:
    """ldb [H, q_len, C] float32 from the store's codes: q [H, q_len, d] (head h reads KV head h // (H / G))."""
    H, ql, d = q.shape
    G, Cn = idx.shape[:2]
    rep = H // G
    qr = q.float() @ Rk.float().T
    q2 = (qr * qr).reshape(G, rep * ql, d)
    q4 = q2 * q2
    a = (sc / 2.0) ** 2
    out = torch.empty(H, ql, Cn, dtype=torch.float32, device=q.device)
    for c0 in range(0, Cn, chunk):
        st = S7B.cell_steps(idx[:, c0:c0 + chunk].long(), M[:, c0:c0 + chunk], gamma[:, c0:c0 + chunk], LB.BASE_BITS,
                            res_bits, dtype=torch.float32)
        s2 = st * st
        A2 = a * torch.einsum("gkd,gcd->gkc", q2, s2)
        A4 = (a * a) * torch.einsum("gkd,gcd->gkc", q4, s2 * s2)
        ldb = ((A2 / 6.0 - A4 / 180.0).clamp_min(0.0) + A2 / 6.0) / 2.0
        out[:, :, c0:c0 + chunk] = ldb.view(G, rep, ql, -1).reshape(H, ql, -1)
    return out


def _pc_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    li = C._layer(module)
    st = PC["layers"].get(li)
    if PC["on"] and st is not None and C.STATE.enabled and li in C.STATE.kdeq:
        sc = scaling if scaling is not None else module.head_dim ** -0.5
        idx, M, gamma, read, Rk, rb = st
        H, ql, kl = query.shape[1], query.shape[2], key.shape[2]
        Cn = C.STATE.ctx_len
        rep = H // idx.shape[0]
        read_h = read.repeat_interleave(rep, 0).unsqueeze(1)                     # [H, 1, C]
        bias = torch.zeros(1, H, ql, kl, dtype=query.dtype, device=query.device)
        for j0 in range(0, ql, Q_CHUNK):
            ldb = debias_pc(query[0, :, j0:j0 + Q_CHUNK], idx, M, gamma, Rk, rb, sc)
            bias[0, :, j0:j0 + Q_CHUNK, :Cn] = (-ldb.masked_fill(read_h, 0.0)).to(query.dtype)
            del ldb
        if attention_mask is not None:
            bias = bias + C._additive(attention_mask[:, :, -ql:, :kl], query.dtype)
        attention_mask = bias
    return PC["orig"](module, query, key, value, attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                      **kwargs)


def _pc_off():
    PC.update(on=False, layers={}, orig=None)


# ------------------------------------------------------------------ the single store: vote store and tail
def layer_widths(pa: dict, nL: int) -> dict:
    """{layer: (residual bits, value bits)} of a single-store arm."""
    fam = pa["family"]
    if fam == "wnest":
        return LB.alloc_table(UNIT["group"], nL)
    if fam == "cnest":
        _, res = LB.tau_table(UNIT["group"], pa["level"], nL)
        return {li: (res[li], int(pa["v_bits"])) for li in range(nL)}
    return {li: (int(pa["res_bits"]), int(pa["v_bits"])) for li in range(nL)}


def vote_store(past, kb, vb, R, Rv, nc, nL, *, seed: int, widths: dict):
    """run_s1h5._tier1's place: the single store as the question pass's view (base keys, the tier's dithered values),
    nothing evicted, compression on."""
    Cn = C.STATE.ctx_len
    C.STATE.reset_arm()
    for li in range(nL):
        K, V = cache_kv(past, li)
        dt = _dt(K.device)
        Rk = Q.random_rotation(K.shape[-1], K.device, dt, LB.ROT_K)
        Rvv = Q.random_rotation(V.shape[-1], V.device, dt, LB.ROT_V)
        C.STATE.kdeq[li] = S7B.base_sketch(K[:, :Cn], LB.BASE_BITS, Rk, dtype=dt).to(K.dtype)
        Vh, _ = T7.dither_quantize(V[:, :Cn], int(widths[li][1]), Rvv, seed + 2 * li + 1, dtype=dt)
        C.STATE.vdeq[li] = Vh.to(V.dtype)
        C.STATE.evict[li] = torch.zeros(K.shape[0], Cn, dtype=torch.bool, device=K.device)
        C.STATE.bits[li] = torch.full((K.shape[0], Cn), LB.BASE_BITS, dtype=torch.long, device=K.device)
        del Vh
    C.STATE.enabled = True


def nest_swap(past, sel, nL, seed: int, widths: dict):
    """After the vote: the context keys become the nested tail (seed unit + 2 layer); the per-coordinate debias goes on.
    sel[li] is R6's selection mask (True = not selected); the selected rows are read exactly (to_tier2 follows)."""
    Cn = C.STATE.ctx_len
    layers = {}
    for li in range(nL):
        K, _ = cache_kv(past, li)
        dt = _dt(K.device)
        Rk = Q.random_rotation(K.shape[-1], K.device, dt, LB.ROT_K)
        rb = int(widths[li][0])
        nz = S7B.nested_quantize(K[:, :Cn], LB.BASE_BITS, rb, Rk, seed + 2 * li, dtype=dt)
        C.STATE.kdeq[li] = nz["Xh"].to(K.dtype)
        layers[li] = (nz["idx"].to(torch.uint8), nz["M"], nz["gamma"], (~sel[li]).to(K.device), Rk, rb)   # the tier's dtype
        del nz
    PC.update(on=True, layers=layers, orig=ALL_ATTENTION_FUNCTIONS[C.IMPL])
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = _pc_attention


def _nest_pieces(pa, nL, B):
    seed = UNIT["seed"]
    widths = layer_widths(pa, nL)
    vpa = dict(pa, tier1_key_bits=LB.BASE_BITS, v_bits=4, store=LB.BASE_BITS)   # _tail5_steps' fields (vote_store reads none)
    store = functools.partial(vote_store, seed=seed, widths=widths)
    swap = lambda past_, sel_: nest_swap(past_, sel_, nL, seed, widths)        # noqa: E731
    return seed, widths, vpa, store, swap


def _traffic(widths: dict, read_frac: float) -> float:
    """Mean over layers of (1 - f)(4 + r_l + v_l)/32 + f (s1h5_lib's model at the arm's own read fraction)."""
    return sum((1 - read_frac) * (LB.BASE_BITS + r + v) / 32.0 + read_frac for r, v in widths.values()) / len(widths)


def _tf_nest(model, past, L0, q_ids, ids, B, vpa, R, Rv, nc, nL, store, swap):
    with _swapped(RS5, "_tier1", store):
        try:
            return RS5._tf_tail5(model, past, L0, q_ids, ids, B, vpa, R, Rv, nc, nL, ids, swap)
        finally:
            _pc_off()


def run_nest_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok):
    """dnest / wnest: run_s1h5.run_dtail_arm's steps with the single store (its vote store, its tail)."""
    seed, widths, vpa, store, swap = _nest_pieces(pa, nL, B)
    t1 = time.time()
    with _swapped(RS5, "_tier1", store):
        try:
            past, sel = RS5._tail5_steps(model, past, L0, q_ids, B, vpa, R, Rv, nc, nL, False, fp_gen, swap)
            for li, e in sel.items():
                C.STATE.bits[li] = torch.where(~e, 16, LB.BASE_BITS + int(widths[li][0])).long()
            gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
        finally:
            S1D.HOOK.sel = None
            S1D._uninstall()
            C.STATE.enabled = False
            _pc_off()
    t_arm = time.time() - t1
    au = C.bits_audit()
    read_frac = float(sum(float((~e).double().mean()) for e in sel.values()) / len(sel))
    tt = time.time()
    lg = _tf_nest(model, past, L0, q_ids, fp_gen, B, vpa, R, Rv, nc, nL, store, swap)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: _tf_nest(model, past, L0, q_ids, ids, B, vpa, R, Rv, nc, nL, store, swap)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, (pa["family"], "", False, arm)))
    uniform = len({w for w in widths.values()}) == 1
    extra = dict(store_width=float(LB.BASE_BITS + widths[0][0]) if uniform else float("nan"),
                 tier1_value_bits=int(widths[0][1]) if uniform else -1, tier2="exact", tier2_bits=16,
                 exact_values_on_selected=True, oracle_selection=False, requestion=True, read_rows_exact=float(B),
                 dithered=True, unbiased=True, dither_seed=int(seed), vote_store="single store (base keys, dithered values)",
                 base_bits=LB.BASE_BITS, widths=json.dumps({str(k): list(v) for k, v in widths.items()}),
                 traffic_measured=_traffic(widths, read_frac), read_frac_measured=read_frac)
    return gen, past, tfm, dict(stored_bits_per_token=au["bits_per_token"], stored_evict_frac=0.0), extra, \
        sel, t_arm, t_tf, (arm, B), dict(au, evict_frac=0.0), {}


# ------------------------------------------------------------------ the dense dithered gate
def _r6_unbiased_off(orig0):
    RS5.UNB.update(on=False, layers={}, orig=None)
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = orig0


def _ddense_view(past, L0, nL, kb, vb, seed):
    """Every context row from R6's dithered tier at kb / vb bits (seed unit + 2 layer, + 1), R6's unbiased mask with no
    row read exactly, nothing evicted, compression on."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    Cn = C.STATE.ctx_len
    allrows = {}
    for li in range(nL):
        K, _ = cache_kv(past, li)
        allrows[li] = torch.ones(K.shape[0], Cn, dtype=torch.bool, device=K.device)   # 'evicted' = read from the tier
    old = RS5.DITHER_SEED
    RS5.DITHER_SEED = seed
    try:
        RS5._dither_swap(past, allrows, nL, kb, vb, True)
    finally:
        RS5.DITHER_SEED = old
    for li, e in allrows.items():
        C.STATE.evict[li] = torch.zeros_like(e)
        C.STATE.bits[li] = torch.full(e.shape, int(kb), dtype=torch.long, device=e.device)
    C.STATE.enabled = True


def _tf_ddense(model, past, L0, q_ids, ids, nL, kb, vb, seed):
    if not ids:
        return None
    orig0 = ALL_ATTENTION_FUNCTIONS[C.IMPL]
    try:
        _ddense_view(past, L0, nL, kb, vb, seed)
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            x = [int(q_ids[0, -1])] + [int(t) for t in ids[:-1]]
            lg = model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past,
                       use_cache=True).logits[0].float()
    finally:
        C.STATE.enabled = False
        _r6_unbiased_off(orig0)
    C.crop_to(past, L0)
    return lg[:len(ids)]


def run_ddense_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok):
    seed = UNIT["seed"]
    kb, vb = int(pa["tier1_key_bits"]), int(pa["v_bits"])
    orig0 = ALL_ATTENTION_FUNCTIONS[C.IMPL]
    t1 = time.time()
    try:
        _ddense_view(past, L0, nL, kb, vb, seed)
        past = RR._question(model, past, q_ids)
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        C.STATE.enabled = False
        _r6_unbiased_off(orig0)
    t_arm = time.time() - t1
    au = C.bits_audit()
    tt = time.time()
    lg = _tf_ddense(model, past, L0, q_ids, fp_gen, nL, kb, vb, seed)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: _tf_ddense(model, past, L0, q_ids, ids, nL, kb, vb, seed)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, ("ddense", "", False, arm)))
    extra = dict(store_width=kb, tier1_value_bits=vb, dithered=True, unbiased=True, dither_seed=int(seed),
                 read_rows_exact=0.0, traffic_measured=(kb + vb) / 32.0)
    return gen, past, tfm, dict(stored_bits_per_token=float(kb), stored_evict_frac=0.0), extra, None, t_arm, t_tf, \
        (arm, B), dict(au, evict_frac=0.0), {}


# ------------------------------------------------------------------ E1: the certified read
CR = dict(on=False, taus={}, widths={}, vb=4, rec=[])


def _fixed_rows(qg, Kf, Vf, posq, Cn, sc):
    """lse and softmax-weighted mean of the fixed rows (absolute positions Cn + j), causal (probe_r7._fixed)."""
    sf = torch.einsum("grnd,gfd->grnf", qg, Kf) * sc
    j = torch.arange(Kf.shape[1], device=qg.device) + Cn
    sf = sf.masked_fill(j.view(1, 1, 1, -1) > posq.view(1, 1, -1, 1), K7B.NEG)
    return torch.logsumexp(sf, -1), torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)


def cert_layer(li, query, key, value, sc, Kh, Vh, codes, tau, vb):
    """The certified read's output for every query row of one call: [1, q_len, H, dv] (query's dtype), and its records.
    key / value are the cache's exact rows; Kh / Vh the store's tier (context rows); codes = PC's (idx, M, gamma, read,
    Rk, r): the per-coordinate steps and the vote's rows."""
    q_len, k_len = query.shape[2], key.shape[2]
    base, Cn = k_len - q_len, C.STATE.ctx_len
    H, Hkv = query.shape[1], key.shape[1]
    r = H // Hkv
    dev, dt = query.device, torch.float64
    idx, M, gamma, read, Rkc, rb = codes
    Kc, Vc = key[0, :, :Cn].to(dt), value[0, :, :Cn].to(dt)
    Kf, Vf = key[0, :, Cn:].to(dt), value[0, :, Cn:].to(dt)
    Kh, Vh = Kh.to(dt), Vh.to(dt)
    d, dv = Kc.shape[-1], Vc.shape[-1]
    Rk = Rkc.to(dt)
    step = S7B.cell_steps(idx.long(), M, gamma, LB.BASE_BITS, rb, dtype=dt)            # [G, C, d]
    Rv = Q.random_rotation(dv, dev, dt, LB.ROT_V)
    n2, _ = T7.value_moments(T7.dither_step(Vc, vb, Rv, dtype=dt).view(Hkv, 1, 1, Cn), dv)
    Xb = S7B.base_sketch(Kc, LB.BASE_BITS, Rk, dtype=dt)
    vote = read.to(dev)
    outs, recs = [], []
    for c0 in range(0, q_len, CR_CHUNK):
        ix = torch.arange(c0, min(c0 + CR_CHUNK, q_len), device=dev)
        n = int(ix.numel())
        qg = query[0, :, ix, :].to(dt).reshape(Hkv, r, n, d)
        s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
        lf, of = _fixed_rows(qg, Kf, Vf, ix + base, Cn, sc)
        qr = qg @ Rk.T
        mom = S7B.coin_moments_pc(qr, step, sc)
        ldb = T7.debias_log(mom)
        shd = torch.einsum("grnd,gcd->grnc", qg, Kh) * sc - ldb
        sig, bas = S7B.score_sigma_pc(qr, step, sc), S7B.score_bound_pc(qr, step, sc)
        sl = torch.einsum("grnd,gcd->grnc", qg, Xb) * sc
        sets = K7B.schedule(vote, K7B.sketch_priority(sl, vote), K7B.FINE)
        target = torch.full((Hkv, r, n), float(tau), dtype=dt, device=dev)
        o = K7B.attend(s, Vc, lf, of)
        bk = dict(sigma=sig, nu2=n2, dim=dv, b_as=bas, mom=mom, ldebias=ldb, forms=("hkz",),
                  exact=dict(qr=qr, step=step, scaling=sc, rows=CR_EXACT_ROWS))
        R_ = K7B.certified_read(s, shd, sets, Vc, Vh, lf, of, target, CR_DELTA, bk, o=o, sl=sl, heavy_h=CR_HEAVY_ROWS,
                                share=CR_SHARE)
        J = R_["J"]
        size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(Hkv, n, dtype=dt, device=dev)])
        err_J = (R_["out"] - o).norm(dim=-1)
        recs.append(dict(rows=size.gather(0, J.view(1, Hkv, n)).squeeze(0), fallback=(J == len(sets)).double(),
                         met_heads=(err_J <= target * (1 + 1e-9) + 1e-300).double().sum(1), J=J))
        outs.append(R_["out"].permute(2, 0, 1, 3).reshape(n, H, dv))
        del s, shd, sig, bas, sl, mom, ldb, sets, R_, o
    rows = torch.cat([x["rows"] for x in recs], -1)                                    # [G, q_len]
    rec = dict(layer=li, rows=float(rows.mean()), fallback=float(torch.cat([x["fallback"] for x in recs], -1).mean()),
               met_heads=float(torch.cat([x["met_heads"] for x in recs], -1).sum()), heads=int(r * Hkv * q_len),
               res_bits=int(rb), tau=float(tau))
    return torch.cat(outs, 0).unsqueeze(0).to(query.dtype), rec


def _cert_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    li = C._layer(module)
    if CR["on"] and C.STATE.enabled and li in PC["layers"] and li in C.STATE.kdeq:    # the answer call (CR is on only then)
        sc = scaling if scaling is not None else module.head_dim ** -0.5
        with torch.no_grad():
            out, rec = cert_layer(li, query, key, value, sc, C.STATE.kdeq[li], C.STATE.vdeq[li], PC["layers"][li],
                                  CR["taus"][li], CR["vb"])
        CR["rec"].append(rec)
        return out, None
    return _pc_attention(module, query, key, value, attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                         **kwargs)


def run_cnest_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok):
    """E1 (teacher-forced only): the single store's question pass and vote (as dnest's teacher-forced replay), then FP's
    answer with the certified read in every layer."""
    seed, widths, vpa, store, swap = _nest_pieces(pa, nL, B)
    taus, _ = LB.tau_table(UNIT["group"], pa["level"], nL)
    CR.update(on=False, taus=taus, widths=widths, vb=int(pa["v_bits"]), rec=[])
    t1 = time.time()
    lg = None
    if fp_gen:
        with _swapped(RS5, "_tier1", store):
            try:
                past, sel = RS5._tail5_steps(model, past, L0, q_ids, B, vpa, R, Rv, nc, nL, True, fp_gen, swap)
                ALL_ATTENTION_FUNCTIONS[C.IMPL] = _cert_attention
                CR["on"] = True
                C.STATE.enabled = True
                x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
                with torch.no_grad():
                    lg = model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past,
                               use_cache=True).logits[0].float()[:len(fp_gen)]
            finally:
                CR["on"] = False
                S1D.HOOK.sel = None
                S1D._uninstall()
                C.STATE.enabled = False
                _pc_off()
                C.crop_to(past, L0)
    t_arm = time.time() - t1
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    recs = CR["rec"]
    extra = dict(tf_only=True, dither_seed=int(seed), level=pa["level"])
    if recs:
        by = {}
        for x in recs:
            by.setdefault(x["layer"], []).append(x)
        rows = {li: sum(x["rows"] for x in v) / len(v) for li, v in by.items()}
        tr = sum((1 - rows[li]) * (LB.BASE_BITS + widths[li][0] + widths[li][1]) / 32.0 + rows[li] for li in rows) / len(rows)
        extra.update(cr_rows=sum(rows.values()) / len(rows), cr_traffic=tr,
                     cr_fallback=sum(x["fallback"] for x in recs) / len(recs),
                     cr_met_heads=sum(x["met_heads"] for x in recs) / max(sum(x["heads"] for x in recs), 1),
                     cr_misses=float(sum(x["heads"] - x["met_heads"] for x in recs)),
                     cr_checks=float(sum(x["heads"] for x in recs)),
                     cr_rows_by_layer=json.dumps({str(k): round(v, 6) for k, v in sorted(rows.items())}),
                     cr_loose_layers=json.dumps(sorted(li for li, w in widths.items() if w[0] != 2)))
    CR["rec"] = []
    return RS5._row(arm, B, fp_gen, past, tfm, extra, t_arm, 0.0)


# ------------------------------------------------------------------ calibration: single-layer arms
CAL = dict(on=False, layer=None, err2=[], vote_key=None, vote=None)


def _cal_vote(model, past, L0, q_ids, B, pa, R, Rv, nc, nL):
    """The single store's vote for this unit (computed once, reused by every calibration arm of the unit)."""
    key = (UNIT["task"], UNIT["prompt"], UNIT["seed"])
    if CAL["vote_key"] == key:
        return CAL["vote"]
    widths = {li: (2, 4) for li in range(nL)}
    store = functools.partial(vote_store, seed=UNIT["seed"], widths=widths)
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    try:
        store(past, 4, 4, R, Rv, nc, nL)
        S1D.HOOK.sel = S1D._Sel(B, {}, 0)
        S1D._install()
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
        RS5.S1E.select_all_w(past, nL, 16)
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
        C.crop_to(past, L0)
    CAL.update(vote_key=key, vote=sel)
    return sel


def _cal_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    li = C._layer(module)
    out, w = _pc_attention(module, query, key, value, attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                           **kwargs)
    if CAL["on"] and li == CAL["layer"] and query.shape[2] > 1 and C.STATE.enabled:
        sc = scaling if scaling is not None else module.head_dim ** -0.5
        with torch.no_grad():
            ex = _sdpa(query, key, value, None, sc, True)                              # exact keys and values, causal
            CAL["err2"].append(((out.float() - ex.float()) ** 2).sum((-1, -2))[0].double().cpu())   # [q_len]
    return out, w


def run_kcal_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, nL, R, Rv, nc, eos, max_new, L0, tok):
    li0, part = int(pa["layer"]), pa["part"]
    if li0 >= nL:
        raise ValueError(f"{arm}: layer {li0} of {nL}")
    seed = UNIT["seed"]
    t1 = time.time()
    sel = _cal_vote(model, past, L0, q_ids, B, pa, R, Rv, nc, nL)
    lg = None
    CAL.update(on=False, layer=li0, err2=[])
    if fp_gen:
        orig0 = ALL_ATTENTION_FUNCTIONS[C.IMPL]
        try:
            C.crop_to(past, L0)
            C.STATE.reset_arm()
            with torch.no_grad():                                                         # FP's question
                if q_ids.shape[1] > 1:
                    model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            K, V = cache_kv(past, li0)
            Cn = C.STATE.ctx_len
            dt = _dt(K.device)
            keep = (~sel[li0]).to(K.device)                                              # the vote's rows: exact
            Kc, Vc = K[:, :Cn], V[:, :Cn]
            Rk = Q.random_rotation(K.shape[-1], K.device, dt, LB.ROT_K)
            Rvv = Q.random_rotation(V.shape[-1], V.device, dt, LB.ROT_V)
            if part in ("tail", "key"):
                nz = S7B.nested_quantize(Kc, LB.BASE_BITS, 2, Rk, seed + 2 * li0, dtype=dt)
                kd = torch.where(keep.unsqueeze(-1), Kc.to(dt), nz["Xh"].to(dt)).to(K.dtype)
                PC.update(on=True, layers={li0: (nz["idx"].to(torch.uint8), nz["M"], nz["gamma"], keep, Rk, 2)}, orig=orig0)
                del nz
            else:
                kd = Kc.clone()
                PC.update(on=False, layers={}, orig=orig0)
            C.STATE.kdeq[li0] = kd
            if part in ("tail", "val"):
                Vh, _ = T7.dither_quantize(Vc, 4, Rvv, seed + 2 * li0 + 1, dtype=dt)
                C.STATE.vdeq[li0] = torch.where(keep.unsqueeze(-1), Vc, Vh.to(V.dtype))
                del Vh
            C.STATE.evict[li0] = torch.zeros_like(keep)
            C.STATE.bits[li0] = torch.where(keep, 16, 6 if part != "val" else 16).long()
            C.STATE.enabled = True
            ALL_ATTENTION_FUNCTIONS[C.IMPL] = _cal_attention
            CAL["on"] = True
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            with torch.no_grad():
                lg = model(torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype), past_key_values=past,
                           use_cache=True).logits[0].float()[:len(fp_gen)]
        finally:
            CAL["on"] = False
            C.STATE.enabled = False
            C.STATE.reset_arm()
            _pc_off()
            ALL_ATTENTION_FUNCTIONS[C.IMPL] = orig0
            C.crop_to(past, L0)
    t_arm = time.time() - t1
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    e2 = torch.cat(CAL["err2"]) if CAL["err2"] else torch.zeros(0, dtype=torch.float64)
    extra = dict(tf_only=True, dither_seed=int(seed), cal_layer=li0, cal_part=part,
                 cal_err2=float(e2.mean()) if e2.numel() else float("nan"), cal_steps=int(e2.numel()),
                 cal_read_frac=float((~sel[li0]).double().mean()))
    CAL["err2"] = []
    return RS5._row(arm, B, fp_gen, past, tfm, extra, t_arm, 0.0)


# ------------------------------------------------------------------ dispatch
RUNNERS = dict(dnest=run_nest_arm, wnest=run_nest_arm, ddense=run_ddense_arm, cnest=run_cnest_arm, kcal=run_kcal_arm)


def run(fam, *args):
    return RUNNERS[fam](*args)
