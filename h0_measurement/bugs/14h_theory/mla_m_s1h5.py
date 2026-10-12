#!/usr/bin/env python3
"""Theorem M for MLA (theory.md §10, "MLA"): the tail's exact second moment when one latent draw makes both a row's key
and value errors. Imports cert_s1h5 / rquant_s1h5 from ../14h_methodology_improve and adapters_s1h5 from this folder.
Findings: findings/findings_mla_m.{md,json}.

MLA's tier stores the row x_i = [c_i, k_rope,i] block by block (block 0 = the latent c, dv wide; block 1 = the rope
part), each rotated (R_b) and subtractive-dithered with its own step D_{b,i}: x^_i - x_i = R_b^T eps_b, eps_bj iid
uniform on [-D_b/2, D_b/2]. With q the absorbed query and qr_b = R_b q_b, the score error is
delta_i = -scaling sum_b <qr_b, eps_b>, so Z_i = w~_i / w_i = e^{-delta_i - ldb_i} = e^{-ldb_i} prod_j e^{scaling qr_j eps_j}:
a product over independent coordinates. Head h's value error is e_i = A_h eps_0, A_h = W_h R_0^T (W_h = W_UV of head h),
so e_i and Z_i share eps_0: E[Z_i e_i] != 0 (unlike GQA).

The identity (as for GQA, with e = v^ - v): Z~ (o - o~) = N = sum_T Y_i, Y_i = w_i [(1 - Z_i) u_i - Z_i e_i],
u_i = W_h (c_i - o_lat) (head space). Rows are independent, so E|N|^2 = sum_i (E|Y_i|^2 - |E Y_i|^2) + |sum_i E Y_i|^2,
with, per coordinate j of block 0 (a_j = D_0 / 2, beta_j = scaling qr_j; L(x) = coth x - 1/x):
  E[Z eps_j]   = E Z  * m1_j,   m1_j = a_j L(a_j beta_j)           (the tilted uniform's mean)
  E[Z^2 eps_j] = E Z^2 * m2_j,  m2_j = a_j L(2 a_j beta_j)
  E[Z^2 eps_j eps_k] = E Z^2 (m2_j m2_k + 1{j=k} v2_j),  v2_j = a_j^2 (1 - 2 coth(x)/x + 2/x^2) - m2_j^2, x = 2 a_j beta_j,
so E Y_i = w_i [(1 - E Z) u_i - E Z A m1],
   E|Y_i|^2 = w_i^2 [(1 - 2 E Z + E Z^2)|u_i|^2 - 2 <u_i, A (E Z m1 - E Z^2 m2)> + E Z^2 (sum_j v2_j |A_:j|^2 + |A m2|^2)].
E Z, E Z^2 are the exact sinh products over every coordinate of both blocks. Every term is exact; the MSE prediction
is E|N|^2 / Z^2 (first order, as for GQA).

    python mla_m_s1h5.py moonlight [--draws 200] [--layer 1]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
for _p in (HERE, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import cert_s1h5 as K  # noqa: E402

DT = torch.float64


def _langevin(x):
    """L(x) = coth x - 1/x (odd), stable."""
    ax = x.abs()
    sm = ax < 1e-4
    xs = torch.where(sm, torch.ones_like(x), x)
    return torch.where(sm, x / 3.0, 1.0 / torch.tanh(xs) - 1.0 / xs)


def _tilt2(x):
    """1 - 2 coth(x)/x + 2/x^2: the second moment of a tilted uniform on [-1, 1] (x the tilt), stable."""
    ax = x.abs()
    sm = ax < 1e-3
    xs = torch.where(sm, torch.ones_like(x), x)
    return torch.where(sm, 1.0 / 3.0 + 2.0 * x * x / 45.0, 1.0 - 2.0 / (torch.tanh(xs) * xs) + 2.0 / (xs * xs))


def _log_sinhc(x):
    x = x.abs()
    sm, bg = x < 1e-3, x > 20.0
    xm = torch.where(sm | bg, torch.ones_like(x), x)
    return torch.where(sm, x * x / 6.0, torch.where(bg, x - math.log(2.0) - torch.log(x.clamp_min(1e-300)),
                                                     torch.log(torch.sinh(xm) / xm)))


def second_moment_mla(w, u, qrs, steps, sc, ldb, A, unread, chunk=4):
    """E|N|^2 [G, r, n] for the MLA tail (head space). w [G, r, n, C] true weights (any common scale); u [G, r, n, C, do]
    = W_h (c_i - o_lat); qrs: list over blocks of rotated query blocks [G, r, n, d_b]; steps: list over blocks of [G, C];
    sc: scaling; ldb [G, r, n, C]; A [G, r, do, dv] = W_h R_0^T; unread [G, r, n, C]. Also returns the parts."""
    G, r, n, C = w.shape
    # E Z and E Z^2: exact products over every coordinate of every block
    lz1 = -ldb.clone()
    lz2 = -2.0 * ldb
    for qr, st in zip(qrs, steps):
        a = (st.view(G, 1, 1, C, 1) / 2.0) * (sc * qr.unsqueeze(-2))       # a_j beta_j  [G, r, n, C, d_b]
        lz1 = lz1 + _log_sinhc(a).sum(-1)
        lz2 = lz2 + _log_sinhc(2.0 * a).sum(-1)
    EZ, EZ2 = torch.exp(lz1), torch.exp(lz2)
    wT = w * unread
    colA2 = (A * A).sum(-2)                                                   # [G, r, dv]: |A_:j|^2
    a0 = steps[0].view(G, 1, 1, C, 1) / 2.0
    EY = torch.zeros(G, r, n, u.shape[-1], dtype=DT)
    var_sum = torch.zeros(G, r, n, dtype=DT)
    key_part = torch.zeros(G, r, n, dtype=DT)
    val_part = torch.zeros(G, r, n, dtype=DT)
    cross_part = torch.zeros(G, r, n, dtype=DT)
    for i0 in range(0, n, chunk):
        sl = slice(i0, i0 + chunk)
        beta = sc * qrs[0][:, :, sl].unsqueeze(-2)                           # [G, r, c, 1, dv]
        x1 = a0 * beta
        m1 = a0 * _langevin(x1)                                              # [G, r, c, C, dv]
        m2 = a0 * _langevin(2.0 * x1)
        v2 = (a0 * a0) * _tilt2(2.0 * x1) - m2 * m2
        Am1 = torch.einsum("grod,grncd->grnco", A, m1)                       # [G, r, c, C, do]
        Am2 = torch.einsum("grod,grncd->grnco", A, m2)
        ez, ez2, ww = EZ[:, :, sl], EZ2[:, :, sl], wT[:, :, sl]
        uu = u[:, :, sl]
        u2 = (uu * uu).sum(-1)
        cross = (uu * (ez.unsqueeze(-1) * Am1 - ez2.unsqueeze(-1) * Am2)).sum(-1)
        ze2 = ez2 * (torch.einsum("grncd,grd->grnc", v2, colA2) + (Am2 * Am2).sum(-1))
        k_ = (1.0 - 2.0 * ez + ez2) * u2
        EYi = ww.unsqueeze(-1) * ((1.0 - ez).unsqueeze(-1) * uu - ez.unsqueeze(-1) * Am1)
        EYi2 = ww * ww * (k_ - 2.0 * cross + ze2)
        EY[:, :, sl] = EYi.sum(-2)
        var_sum[:, :, sl] = (EYi2 - (EYi * EYi).sum(-1)).sum(-1)
        key_part[:, :, sl] = (ww * ww * k_).sum(-1)
        val_part[:, :, sl] = (ww * ww * ze2).sum(-1)
        cross_part[:, :, sl] = (ww * ww * (-2.0 * cross)).sum(-1)
    tot = var_sum + (EY * EY).sum(-1)
    return tot, dict(key=key_part, value=val_part, cross=cross_part, mean=(EY * EY).sum(-1))


def setup(key, L=1600, layer=1, nv=32, frac=8):
    """The MLA view (adapters_s1h5.view_of) with test_theory_s1h5.test_mla_models' read set: the vote of a Lloyd-Max
    4-bit sketch (32 voters), the next 16 queries measured, the last 48 positions' own rows fixed."""
    import adapters_s1h5 as A
    from sievelib import quant as Q
    ids = A._ids(key, L)
    if A.MODELS[key][1] is None:
        rec, li = A._kimi_mla(ids)
    else:
        m = A._load(key, k=max(A.MODELS[key][1], layer + 1)).eval()
        with A.capture(m, [layer]) as rec:
            with torch.no_grad():
                m(ids, use_cache=False)
        del m
        li = layer
    pos = torch.arange(L - 48, L)
    v = A.view_of(li, rec[li], pos)
    C = L - 48
    q, Kx = v.q.double(), v.K.double()
    G, r, n_all, dk = q.shape
    dv = v.V.shape[-1]
    sc = v.scaling
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc, Vf = Kx[:, :C, :dv], Kx[:, C:, :dv]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
    Kl = torch.empty_like(Kx[:, :C])
    for i, (a, b_) in enumerate(v.blocks):
        Kl[..., a:b_] = Q.quantize_keys(Kx[:, :C, a:b_].float(), 4, Q.random_rotation(b_ - a, "cpu", torch.float32, i)).double()
    sl = torch.einsum("grnd,gcd->grnc", q, Kl) * sc
    vote = K.topk_keep(torch.softmax(sl[:, :, :nv], -1).sum((1, 2)), C // frac)
    ms = slice(nv, n_all)
    return dict(key=key, layer=li, v=v, q=q[:, :, ms], Kc=Kx[:, :C], Vc=Vc, s=s[:, :, ms], lf=lf[:, :, ms],
                of=of[:, :, ms], W=v.W.double(), sc=sc, G=G, r=r, n=n_all - nv, C=C, dv=dv, blocks=v.blocks,
                km=vote.view(G, 1, 1, C), keep=vote.unsqueeze(1).expand(G, n_all - nv, C))


def validate(S, bits=(4, 6, 8), draws=200):
    """Theorem M (MLA) against Monte Carlo over the coupled dither, in head space; also the GQA formula (no cross term,
    independent errors), to show what the coupling changes."""
    import adapters_s1h5 as A
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    G, r, n, C, dv, sc = S["G"], S["r"], S["n"], S["C"], S["dv"], S["sc"]
    Wd = S["W"]
    head = lambda x_: torch.einsum("grod,grnd->grno", Wd, x_)                 # noqa: E731
    o_lat = A._attend(S["s"], S["Vc"], S["lf"], S["of"])
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    unread = (~S["km"]).expand(G, r, n, C)
    u = torch.einsum("grod,grncd->grnco", Wd, S["Vc"].unsqueeze(1).unsqueeze(1) - o_lat.unsqueeze(-2))
    Rs = [Q.random_rotation(b_ - a, "cpu", DT, 50 + i) for i, (a, b_) in enumerate(S["blocks"])]
    qrs = [S["q"][..., a:b_] @ Rs[i].T for i, (a, b_) in enumerate(S["blocks"])]
    A0 = torch.einsum("grod,jd->groj", Wd, Rs[0])                             # W_h R_0^T: A[o, j] = sum_d W[o, d] R_0[j, d]
    out = dict(model=S["key"], layer=S["layer"], C=C, heads=r, bits={})
    for kb in bits:
        steps = []
        for i, (a, b_) in enumerate(S["blocks"]):
            y = S["Kc"][..., a:b_] @ Rs[i].T
            steps.append(2.0 * y.abs().amax(-1) / (2 ** kb - 1))
        mom = None
        for i, (a, b_) in enumerate(S["blocks"]):
            mb = RQ.coin_moments(qrs[i], steps[i].view(G, 1, 1, C), sc)
            mom = mb if mom is None else {k_: mom[k_] + mb[k_] for k_ in mb}
        ldb = RQ.debias_log(mom)
        pred, parts = second_moment_mla(w, u, qrs, steps, sc, ldb, A0, unread)
        meas = torch.zeros(G, r, n, dtype=DT)
        t0 = time.time()
        for sd in range(draws):
            Kh = torch.empty_like(S["Kc"])
            for i, (a, b_) in enumerate(S["blocks"]):
                Kh[..., a:b_], st = RQ.dither_quantize(S["Kc"][..., a:b_], kb, Rs[i], 1000 * sd + i)
            Vh = Kh[..., :dv]
            sh = torch.einsum("grnd,gcd->grnc", S["q"], Kh) * sc - ldb
            oT = A._mix(S["s"], sh, S["Vc"], Vh, S["keep"], S["lf"], S["of"])
            meas = meas + ((head(oT) - head(o_lat)) ** 2).sum(-1)
        meas = meas / draws
        rr = (meas / pred).flatten()
        no_cross = pred - parts["cross"]
        rr_nc = (meas / no_cross).flatten()
        share = (parts["cross"].abs() / (parts["key"] + parts["value"]).clamp_min(1e-300)).flatten()
        out["bits"][kb] = dict(measured_over_predicted=float(rr.median()),
                               p10_p90=[float(torch.quantile(rr, .1)), float(torch.quantile(rr, .9))],
                               measured_over_predicted_without_cross_term=float(rr_nc.median()),
                               cross_term_share=float(share.median()), seconds=round(time.time() - t0, 1))
        print(f"{S['key']} L{S['layer']} {kb}-bit: measured / predicted MSE {out['bits'][kb]['measured_over_predicted']:.3f} "
              f"(p10-p90 {out['bits'][kb]['p10_p90'][0]:.3f}-{out['bits'][kb]['p10_p90'][1]:.3f}); without the cross term "
              f"{out['bits'][kb]['measured_over_predicted_without_cross_term']:.3f}; |cross| / (key + value) "
              f"{out['bits'][kb]['cross_term_share']:.3f}", flush=True)
    return out


def main():
    argv = sys.argv[1:]
    key = argv[0]
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    S = setup(key, int(g("--L", 1600)), int(g("--layer", 1)))
    res = validate(S, draws=int(g("--draws", 200)))
    path = os.path.join(HERE, "findings", "findings_mla_m.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[f"{key}_L{S['layer']}"] = res
    json.dump(allres, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
