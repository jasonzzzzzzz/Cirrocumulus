#!/usr/bin/env python3
"""The 4-bit study behind theory.md §6b, §7 and §10 and findings/findings_4bit.md (CPU, real weights, oracle_s1h5's setup).

The setup (as ../14h_methodology_improve/oracle_s1h5.py): one layer, context = the first L - 48 rows, the next 32
queries vote (a Lloyd-Max 4-bit sketch, rotation seed 7, coin-free) for C/8 rows per KV head, the last 16 are measured;
tier 1 is dithered (keys at kb bits, values at 4), the tail is unbiased; delta = 0.05 per head-step; the oracle width is
the 95% quantile of the error over 200 draws; bounds are the median over 10 draws.

    python exp4bit_s1h5.py capture llama1b 1[,3,..] [L]  # -> .views/view_llama1b_L<l>_<L>.pt (GPT-OSS sinks kept)
    python exp4bit_s1h5.py diagnose VIEW                 # v2's slack: counterfactuals with true quantities
    python exp4bit_s1h5.py remedies VIEW                 # sketch budget, peeling, both; v3; v3 + budget (4-bit keys)
    python exp4bit_s1h5.py lawcheck VIEW                 # exact-law (truncated) Pinelis radius vs Bennett, true weights
    python exp4bit_s1h5.py evaluate VIEW TAG             # 4/6/8-bit keys: oracle, v2 vs v3, Theorem M
Counterfactual variants use exact quantities and are NOT certificates; everything else is."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
for _p in (METH, ROOT):
    if _p not in sys.path:
        sys.path.append(_p)
import adapters_s1h5 as A  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402
import theory2_s1h5 as K2  # noqa: E402
import theory3_s1h5 as K3  # noqa: E402
from sievelib import quant as Q  # noqa: E402

DT = torch.float64
DELTA = 0.05
CACHE = os.environ.get("TH_CACHE", os.path.join(HERE, ".views"))
E_ = lambda x: torch.exp(x.clamp(max=700.0))  # noqa: E731


# ------------------------------------------------------------------ the setup
def capture(key, layer, L=1600):
    """Cache the view of one layer (or a list of layers, one forward pass) at the last 48 positions of L tokens."""
    os.makedirs(CACHE, exist_ok=True)
    layers = [layer] if isinstance(layer, int) else list(layer)
    ids = A._ids(key, L)
    m = A._load(key, k=max(A.MODELS[key][1] or 0, max(layers) + 1)).eval()
    with A.capture(m, layers) as rec:
        with torch.no_grad():
            m(ids, use_cache=False)
    del m
    pos = torch.arange(L - 48, L)
    paths = []
    for li in layers:
        v = A.view_of(li, rec[li], pos)
        path = os.path.join(CACHE, f"view_{key}_L{li}_{L}.pt")
        torch.save(dict(family=v.family, q=v.q, K=v.K, V=v.V, pos=pos, scaling=v.scaling, sink=v.sink, L=L), path)
        paths.append(path)
    return paths[0] if isinstance(layer, int) else paths


def load_setup(path, nv=32, frac=8):
    d0 = torch.load(path)
    if d0["family"] != "gqa":
        raise SystemExit("GQA views only")
    q, Kx, Vx = d0["q"].to(DT), d0["K"].to(DT), d0["V"].to(DT)
    pos, sc, L = d0["pos"], d0["scaling"], d0["L"]
    G, r, n_all, d = q.shape
    C = L - 48
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc = Vx[:, :C]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    if d0.get("sink") is not None:                        # GPT-OSS learned sinks: a fixed row with value 0
        sk = d0["sink"].to(DT).view(G, r, 1).expand(G, r, n_all)
        lf2 = torch.logaddexp(lf, sk)
        of, lf = of * torch.exp(lf - lf2).unsqueeze(-1), lf2
    Kl = Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)
    sl_all = torch.einsum("grnd,gcd->grnc", q, Kl) * sc
    vote = K.topk_keep(torch.softmax(sl_all[:, :, :nv], -1).sum((1, 2)), C // frac)
    ms = slice(nv, n_all)
    s, lf, of, qm, sl = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms], sl_all[:, :, ms]
    n = n_all - nv
    o = A._attend(s, Vc, lf, of)
    from sievelib.kv_quant_baselines import fp8_e4m3
    e8 = (A._attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kx[:, :C].float()).to(DT)) * sc,
                    fp8_e4m3(Vc.float()).to(DT), lf, of) - o).norm(dim=-1)
    km = vote.view(G, 1, 1, C)
    c = A._attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
    return dict(q=qm, Kc=Kx[:, :C], Vc=Vc, s=s, lf=lf, of=of, o=o, keep=vote.unsqueeze(1).expand(G, n, C), km=km,
                e8=e8, c=c, Rk=Q.random_rotation(d, "cpu", DT, 50), Rv=Q.random_rotation(Vx.shape[-1], "cpu", DT, 150),
                sc=sc, G=G, r=r, n=n, C=C, d=d, dv=Vx.shape[-1], sl=sl)


def draw(S, sd, kb=4, vb=4):
    G, C = S["G"], S["C"]
    Kh, stk = RQ.dither_quantize(S["Kc"], kb, S["Rk"], 1000003 * sd + 1)
    Vh, stv = RQ.dither_quantize(S["Vc"], vb, S["Rv"], 1000003 * sd + 2)
    sh = torch.einsum("grnd,gcd->grnc", S["q"], Kh) * S["sc"]
    st = stk.view(G, 1, 1, C)
    qr = S["q"] @ S["Rk"].T
    mom = RQ.coin_moments(qr, st, S["sc"])
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), S["dv"])
    shd = sh - ldb
    oT = A._mix(S["s"], shd, S["Vc"], Vh, S["keep"], S["lf"], S["of"])
    return dict(Vh=Vh, shd=shd, mom=mom, ldb=ldb, sig=RQ.score_sigma(S["q"].norm(dim=-1, keepdim=True), st, S["sc"]),
                bas=RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, S["sc"]), n2=n2, nm=nm, oT=oT,
                err=(oT - S["o"]).norm(dim=-1), rho=A._dist(S["c"], Vh), oc=(oT - S["c"]).norm(dim=-1), stk=stk,
                stv=stv, qr=qr)


def v2(S, D):
    return K2.tail_bound_rand_v2(S["s"], D["shd"], D["sig"], S["km"], D["rho"], D["nm"], D["n2"], D["oc"], S["lf"],
                                 delta=DELTA, b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"])


def v3(S, D, **kw):
    return K3.tail_bound_rand_v3(S["s"], D["shd"], D["sig"], S["km"], D["rho"], D["nm"], D["n2"],
                                 D["stv"].view(S["G"], 1, 1, S["C"]), S["dv"], D["oc"], S["lf"], DELTA, D["bas"],
                                 D["mom"], D["ldb"], **kw)


def oracle(S, kb, ndraw=200):
    E = torch.stack([draw(S, sd, kb)["err"] for sd in range(ndraw)])
    return E, torch.quantile(E, 1 - DELTA, dim=0)


def summarize(B, E, orc, e8):
    rr = (B / orc).flatten()
    return dict(cover=float((E <= B.unsqueeze(0)).double().mean()), bound_over_oracle=float(rr.median()),
                p10=float(torch.quantile(rr, .1)), p90=float(torch.quantile(rr, .9)),
                within_fp8=float((B <= e8).double().mean()))


# ------------------------------------------------------------------ v2 with switchable slack (diagnosis only)
def _v2_engine(s, shd, sigma, keep, rho, nm, n2, oc, lf, b_as, mom, ldb, cf_w=False, cf_sums=False, cf_rho=None,
               pi=None, eta=0.0, kappas=(None,), vho=None):
    """theory2_s1h5.tail_bound_rand_v2 re-derived with (i) counterfactual switches (true weights in every max,
    true sums for the self-bounds, true distances: NOT certificates) and (ii) the colleague's two remedies, which ARE
    certificates: a truncation budget weighted by a coin-free importance pi (eta = its share), and peeling by a
    true-weight threshold theta = kappa M (heavy rows: the realized term on the truncation event; light rows: v2 with
    weights capped at theta; a union over the kappa grid)."""
    keep_e = keep.expand_as(s)
    unread = ~keep_e
    n_t = unread.sum(-1).double().clamp_min(1.0).unsqueeze(-1)
    sig = sigma.double().expand_as(s)
    pr = torch.full_like(sig, 1.0) * (DELTA / 2.0) / n_t
    if pi is not None and eta > 0:
        pis = pi.double().expand_as(sig).masked_fill(~unread, 0.0)
        pr = (DELTA / 2.0) * ((1.0 - eta) / n_t + eta * pis / pis.sum(-1, keepdim=True).clamp_min(1e-300))
    tz = torch.sqrt(2.0 * torch.log(2.0 / pr)) * sig
    bas = b_as.double().expand_as(s)
    t = torch.minimum(tz, bas)
    exact_cap = bas <= tz
    w, wh, _, wf, tt, unread = K._tail_prep(s, shd, t, keep, lf)
    lfx = torch.as_tensor(lf, dtype=DT).expand(s.shape[:-1])
    m = torch.maximum(torch.maximum(s.double().masked_fill(unread, K.NEG).amax(-1),       # _tail_prep's shift
                                    (shd.double() + tt).masked_fill(~unread, K.NEG).amax(-1)), lfx)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    wtrue = torch.where(unread, torch.exp(s.double() - m), torch.zeros_like(wh))
    sig = sig.masked_fill(~unread, 0.0)
    ex = lambda k_: mom[k_].double().expand_as(w).masked_fill(~unread, 0.0)        # noqa: E731
    lmu_lo, lmu_hi, lmu2_hi, lmu4_hi = ex("lmu_lo"), ex("lmu_hi"), ex("lmu2_hi"), ex("lmu4_hi")
    ld = ldb.double().expand_as(w).masked_fill(~unread, 0.0)
    p = torch.where(exact_cap | ~unread | (sig <= 0), torch.zeros_like(tt),
                    2.0 * torch.exp(-(tt * tt) / (2.0 * sig * sig).clamp_min(1e-300))).clamp(max=0.5)
    lo1 = ((E_(lmu_lo) - torch.sqrt(E_(lmu2_hi) * p)) * E_(-ld)).clamp_min(1e-6)
    hi1 = E_(lmu_hi - ld) / (1.0 - p)
    hi2 = E_(lmu2_hi - 2.0 * ld) / (1.0 - p)
    hi4 = E_(lmu4_hi - 4.0 * ld) / (1.0 - p)
    kap = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    var = (hi2 - lo1 * lo1).clamp_min(0.0)
    rng = torch.maximum(E_(tt - ld) - lo1, hi1 - E_(-tt - ld)).clamp_min(0.0)
    M = wf + w.sum(-1)
    w_up = torch.exp(torch.log(wh) + ld + tt)
    w_lo = torch.exp(torch.log(wh) + ld - tt)
    gam = torch.maximum(E_(tt + ld) - 1.0, 1.0 - E_(-tt + ld)).clamp_min(0.0)
    nmx, n2x = nm.double().expand_as(w), n2.double().expand_as(w)
    rb = (rho.double().expand_as(w) + nmx) if cf_rho is None else cf_rho.double().expand_as(w)
    g_mean, g_var = kap * rb, var * rb * rb + hi2 * n2x
    dk = DELTA / (14.0 * len(kappas))
    best = None
    for kp in kappas:
        if kp is None:
            Lp, Hp, wbar = unread, torch.zeros_like(unread), w_up
        else:
            th = (kp * M).unsqueeze(-1)
            Lp, Hp = unread & (w_lo <= th), unread & (w_up > th)
            wbar = torch.minimum(w_up, th.expand_as(w_up))
        if cf_w:
            wbar = wtrue
        wbar = wbar.masked_fill(~Lp, 0.0)
        lo_1 = torch.where(Lp, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)

        def sb(g, p_):
            if cf_sums:
                return (wtrue ** p_ * g * Lp).sum(-1)
            X = (wh ** p_ * g * Lp).sum(-1)
            mm = hi2 if p_ == 1 else hi4
            return K3.self_bound_rowwise(X, (wbar ** p_ * g * mm * Lp).amax(-1), dk, lo_1 if p_ == 1 else lo_1 ** 2)
        mY = sb(g_mean, 1)
        B2Y = sb(g_var, 2)
        if not cf_sums:
            B2Y = torch.minimum(B2Y, (wbar * torch.sqrt(g_var) * Lp).amax(-1) * sb(torch.sqrt(g_var), 1))
        mD, B2D = sb(kap, 1), sb(var, 2)
        rY = K3.bennett_radius(torch.sqrt(B2Y), (wbar * (rng * rb + E_(tt - ld) * nmx) * Lp).amax(-1).clamp_min(1e-300), dk)
        rD = K3.bennett_radius(torch.sqrt(B2D), (wbar * rng * Lp).amax(-1).clamp_min(1e-300), dk)
        hterm = torch.zeros_like(rY) if kp is None else \
            (wh * (gam * (vho.double().expand_as(w) + nmx) + nmx) * Hp).sum(-1)
        den = M + (wh.sum(-1) - (wh * (1.0 - E_(-tt + ld)).clamp_min(0.0) * Hp).sum(-1) - mD - rD).clamp_min(0.0)
        bnd = (hterm + mY + rY + (mD + rD) * oc.double()) / den
        best = bnd if best is None else torch.minimum(best, bnd)
    return best


def diagnose(path):
    S = load_setup(path)
    E, orc = oracle(S, 4)
    rho_true = A._dist(S["c"], S["Vc"])
    res = {}
    for sd in range(10):
        D = draw(S, sd)
        a = (S["s"], D["shd"], D["sig"], S["km"], D["rho"], D["nm"], D["n2"], D["oc"], S["lf"], D["bas"], D["mom"],
             D["ldb"])
        for nm_, kw in (("v2", {}), ("true weights in every max", dict(cf_w=True)), ("true self-bound sums", dict(cf_sums=True)),
                        ("both", dict(cf_w=True, cf_sums=True)), ("both + true distances", dict(cf_w=True, cf_sums=True, cf_rho=rho_true))):
            res.setdefault(nm_, []).append(_v2_engine(*a, **kw))
    out = {k: float((torch.stack(v).median(0).values / orc).median()) for k, v in res.items()}
    # tail structure (true weights)
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    wt = torch.exp(S["s"] - lZ.unsqueeze(-1)) * (~S["km"]).expand_as(S["s"])
    tot = wt.sum(-1)
    out["tail: unread mass (median)"] = float(tot.median())
    out["tail: heaviest unread row / unread mass (median)"] = float((wt.amax(-1) / tot).median())
    out["tail: effective rows (sum w)^2 / sum w^2 (median)"] = float((tot ** 2 / (wt ** 2).sum(-1)).median())
    out["oracle / FP8 error (median)"] = float((orc / S["e8"]).median())
    out["share within FP8 if a bound were k x oracle, k = 1, 2, 5, 10, 20"] = [
        float((f * orc <= S["e8"]).double().mean()) for f in (1, 2, 5, 10, 20)]
    return out


def remedies(path):
    S = load_setup(path)
    E, orc = oracle(S, 4)
    pi = torch.softmax(S["sl"], -1)                     # the vote's sketch under the current query (coin-free)
    grid = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    variants = {"v2": {}, "budget eta 0.5": dict(pi=pi, eta=0.5), "budget eta 0.9": dict(pi=pi, eta=0.9),
                "peel kappa 0.01": dict(kappas=(0.01,)), "peel kappa 0.001": dict(kappas=(0.001,)),
                "peel grid of 7": dict(kappas=grid), "peel grid + budget eta 0.9": dict(kappas=grid, pi=pi, eta=0.9)}
    res = {}
    for sd in range(10):
        D = draw(S, sd)
        a = (S["s"], D["shd"], D["sig"], S["km"], D["rho"], D["nm"], D["n2"], D["oc"], S["lf"], D["bas"], D["mom"],
             D["ldb"])
        vho = A._dist(D["oT"], D["Vh"])
        for nm_, kw in variants.items():
            res.setdefault(nm_, []).append(_v2_engine(*a, vho=vho, **kw))
        res.setdefault("v3", []).append(v3(S, D))
    out = {}
    for k_, v in res.items():
        B = torch.stack(v).median(0).values
        out[k_] = summarize(B, E, orc, S["e8"])
    return out


def lawcheck(path, n_sel=64):
    """The dominant key sum K = sum_T w_i (1 - Z_i)(v_i - o) with TRUE weights: q95 of |K - E K| over 200 draws against
    Bennett (true B and range at the truncation) and the exact truncated law of Z (rigorous tail integral, P(|delta| >= x
    | E_i) <= 2 e^{-x^2/2 sigma^2} / (1 - p_i)), per-row scales and one common scale."""
    S = load_setup(path)
    G, r, n, C = S["G"], S["r"], S["n"], S["C"]
    unread = ~S["km"].expand(G, r, n, C)
    s = S["s"]
    lZ = torch.logaddexp(torch.logsumexp(s, -1), S["lf"])
    w = torch.exp(s - lZ.unsqueeze(-1)) * unread
    u = A._dist(S["o"], S["Vc"])
    Ks = []
    for sd in range(200):
        D = draw(S, sd)
        a_ = w * (1 - torch.exp(D["shd"] - s))
        Ks.append(torch.einsum("grnc,gcd->grnd", a_, S["Vc"]) - a_.sum(-1, keepdim=True) * S["o"])
        if sd == 0:
            sig, ldb = D["sig"], D["ldb"]
    Kst = torch.stack(Ks)
    q95 = torch.quantile((Kst - Kst.mean(0)).norm(dim=-1), 0.95, dim=0)
    z = K.trunc_z(unread.sum(-1), DELTA / 2.0)
    dk = DELTA / 14
    L = math.log(2 / dk)
    lam = torch.logspace(-1, 7, 60, dtype=DT)
    ln2 = math.log(2.0)

    def log_phi(y):
        ys = y.clamp_min(1e-300)
        small = 2 * torch.log(ys) - ln2 + torch.log1p(ys / 3.0)
        big = y + torch.log1p((-torch.exp(-y) * (1 + y)).clamp(min=-1 + 1e-16))
        return torch.where(y < 1e-3, small, big)
    sel = [(g, h, j) for g in range(G) for h in (0, 2) for j in (0, 5, 10, 15)][:n_sel]
    out = {"bennett": [], "law, per-row scales": [], "law, one common scale": []}
    for (g, h, j) in sel:
        ur = unread[g, h, j]
        wi, ui, sg, lb = w[g, h, j][ur], u[g, h, j][ur], sig[g, h, j][ur], ldb[g, h, j][ur]
        t = z[g, h, j] * sg
        p = (2 * torch.exp(-t * t / (2 * sg * sg))).clamp(max=0.5)
        x = torch.linspace(0, 1, 121, dtype=DT).unsqueeze(1) * t.unsqueeze(0)
        lpx = (ln2 - x * x / (2 * sg * sg)).clamp(max=0.0) - torch.log1p(-p)
        kap = 0.05                                       # generous centring slack (a ceiling check, true weights)
        hp, hm = (torch.exp(x - lb) - 1).abs() + kap, (torch.exp(-x - lb) - 1).abs() + kap

        def law_rad(scale):
            best = float("inf")
            for lm in lam:
                uu = (lm * scale).unsqueeze(0)
                lg = torch.cummax(torch.logaddexp(log_phi(uu * hp), log_phi(uu * hm)) - ln2, 0).values
                d_ = lg[1:] - lg[:-1]
                linc = torch.where(d_ > 0, lg[1:] + torch.log((-torch.expm1(-d_)).clamp_min(1e-300)),
                                   torch.full_like(d_, -float("inf")))
                le = torch.logsumexp(torch.cat([lg[:1], linc + lpx[:-1]], 0), 0)
                best = min(best, (L + float(torch.exp(le.clamp(max=700)).sum())) / float(lm))
            return best
        out["law, per-row scales"].append(law_rad(wi * ui) / float(q95[g, h, j]))
        out["law, one common scale"].append(law_rad(torch.full_like(wi, float((wi * ui).max()))) / float(q95[g, h, j]))
        B2 = float((wi ** 2 * ui ** 2 * torch.expm1(sg * sg)).sum())
        a = float((wi * ui * (torch.exp(t - lb) - 1 + kap)).max())
        out["bennett"].append(float(K3.bennett_radius(torch.tensor([B2]).sqrt(), torch.tensor([a]), dk)) / float(q95[g, h, j]))
    return {k_: float(torch.tensor(v).median()) for k_, v in out.items()}


def evaluate(path, tag):
    S = load_setup(path)
    G, r, n, C = S["G"], S["r"], S["n"], S["C"]
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    u = S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2)
    unread = (~S["km"]).expand(G, r, n, C)
    o_ev = A._attend(S["s"].masked_fill(~S["km"], K.NEG), S["Vc"], S["lf"], S["of"])     # evicting the unread rows
    out = dict(view=tag, C=C, head_steps=G * r * n, bits={},
               unread_mass=float((w * unread).sum(-1).median()),
               unread_second_moment_share=float((((w * unread) ** 2 * (u * u).sum(-1)).sum(-1)
                                                 / (w ** 2 * (u * u).sum(-1)).sum(-1)).median()),
               eviction_over_fp8=float(((o_ev - S["o"]).norm(dim=-1) / S["e8"]).median()))
    for kb in (4, 6, 8):
        E2, Ed, b2s, b3s = [], [], [], []
        for sd in range(200):
            D = draw(S, sd, kb)
            E2.append(D["err"] ** 2)
            oD = A._mix(S["s"], D["shd"], S["Vc"], D["Vh"], torch.zeros_like(S["keep"]), S["lf"], S["of"])
            Ed.append(((oD - S["o"]) ** 2).sum(-1))
            if sd < 10:
                b2s.append(v2(S, D))
                b3s.append(v3(S, D))
            if sd == 0:
                EZ = K3.exact_coin_moments(D["qr"], D["stk"].view(G, 1, 1, C), S["sc"], D["ldb"])
                n2 = D["n2"].expand(G, r, n, C)
                sig_med = float(D["sig"].median())
        E2 = torch.stack(E2)
        E = E2.sqrt()
        orc = torch.quantile(E, 1 - DELTA, dim=0)
        rec = dict(sigma_median=sig_med, oracle_within_fp8=float((orc <= S["e8"]).double().mean()),
                   v2=summarize(torch.stack(b2s).median(0).values, E, orc, S["e8"]),
                   v3=summarize(torch.stack(b3s).median(0).values, E, orc, S["e8"]))
        for nm_, mask, meas in (("tail", unread, E2.mean(0)), ("dense", torch.ones_like(unread), torch.stack(Ed).mean(0))):
            pred = K3.second_moment(w, u, EZ[1], EZ[2], n2, mask)
            rr = (meas / pred).flatten()
            rec[f"M_{nm_}"] = dict(measured_over_predicted=float(rr.median()),
                                   p10_p90=[float(torch.quantile(rr, .1)), float(torch.quantile(rr, .9))],
                                   rms_over_fp8=float((meas.sqrt() / S["e8"]).median()))
            rec[f"_pred_{nm_}"] = pred
        rec["M_tail_over_dense_predicted"] = float((rec.pop("_pred_tail") / rec.pop("_pred_dense")).median())
        rec["M_tail_over_dense_measured"] = float((E2.mean(0) / torch.stack(Ed).mean(0)).median())
        out["bits"][kb] = rec
        print(f"{tag} {kb}-bit: v2 {rec['v2']['bound_over_oracle']:.1f}x (fp8 {rec['v2']['within_fp8']:.3f}) -> "
              f"v3 {rec['v3']['bound_over_oracle']:.1f}x (fp8 {rec['v3']['within_fp8']:.3f}); M tail "
              f"{rec['M_tail']['measured_over_predicted']:.3f}, rms tail/fp8 {rec['M_tail']['rms_over_fp8']:.2f}", flush=True)
    return out


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    cmd, args = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    if cmd == "capture":
        lay = [int(x) for x in args[1].split(",")]
        print(capture(args[0], lay[0] if len(lay) == 1 else lay, int(args[2]) if len(args) > 2 else 1600))
        return
    path = args[0] if os.path.exists(args[0]) else os.path.join(CACHE, args[0])
    res = dict(diagnose=diagnose, remedies=remedies, lawcheck=lawcheck)[cmd](path) if cmd != "evaluate" \
        else evaluate(path, args[1])
    print(json.dumps(res, indent=1))
    out = os.path.join(HERE, "findings", "findings_4bit.json")
    allres = json.load(open(out)) if os.path.exists(out) else {}
    allres.setdefault(cmd, {})[os.path.basename(path) if cmd != "evaluate" else args[1]] = res
    json.dump(allres, open(out, "w"), indent=1)
    print(f"({time.time() - t0:.0f}s; appended to {out})")


if __name__ == "__main__":
    main()
