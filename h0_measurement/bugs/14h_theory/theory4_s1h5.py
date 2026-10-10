#!/usr/bin/env python3
"""Theory part 7 (theory.md §6c-§6e): the sure-sum bound, the mirror split by coin source, and the certified adaptive
read. Pure torch for the bounds and the rule, plus CPU harnesses on real weights. Merged here 2026-10-10 from
../14h_methodology_improve/certread_s1h5.py (written in parallel with part 6; theory.md §6c compares them).
Imports cert_s1h5 / rquant_s1h5 / adapters_s1h5 from ../14h_methodology_improve and theory3_s1h5 from this folder.
Tests: test_theory4_s1h5.py. Findings: findings/findings_part7.{md,json}.

tail_bound_rand_sure (§6c). Theorem R v2 with every unknown weighted sum sum_T w^p g ALSO bounded surely on the
  truncation event (w_i <= w_bar_i there): mode "min" takes the smaller of the self-bound and the sure sum; mode "sure"
  drops the self-bounds (no events). theory3_s1h5.tail_bound_rand_v3(..., sure=True) adds the same to part 6's v3.
tail_bound_rand_split (§6d). The mirror image of part 6's regrouping: Y_i = (w_i - w~_i)(v^_i - c) - w_i e_i, so the
  value sum V = sum w_i e_i has FIXED weights (Hsu-Kakade-Zhang, with w <= w_bar surely) and the key sum
  K = sum (w_i - w~_i)(v^_i - c) has OBSERVED vectors given the value coins (Bennett, sure sums). GQA only.
certified_read (§6e, corollary). A coin-free nested schedule of read sets, each bound at delta / m (union bound), and a
  stop at the first set meeting the target: with probability >= 1 - delta the output meets its target, however the
  coins decide the stop.

    python theory4_s1h5.py certread --model llama1b --layers global --bits 6,8 --draws 2 --delta 0.001 --out x.json
    python theory4_s1h5.py compare  --model llama1b --layers 1 --delta 0.001 --out y.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
for _p in (HERE, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import cert_s1h5 as K  # noqa: E402
import theory3_s1h5 as T3  # noqa: E402

FRACS = (0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)   # S_m = everything is implicit


def tail_bound_rand_sure(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=K.NEG, delta: float = K.DELTA_T,
                         b_as=None, mom=None, ldebias=None, mode="min"):
    """Theorem R, sure sums (GQA): v2 (cert_s1h5.tail_bound_rand_v2) with every unknown weighted sum sum_T w^p g (p = 1, 2)
    ALSO bounded surely on the truncation event that Theorem R already uses: there |delta_i| <= t_i, so
    w_i = w~_i e^{delta_i + ldebias_i} <= w_bar_i and sum w^p g <= sum w_bar^p g. No new event: the radii take any
    variance bound that holds on the truncation event (r is increasing in B), so
      mode "min": each sum is the smaller of its self-bound (v2's events, delta / 14) and the sure sum;
      mode "sure": the sure sums only (no self-bound events: the two Bennett sums at delta / 4).
    The sure sum pays e^{p t_i} on every row; the self-bound pays a ln(1/delta) x (heaviest row) term, so the sure
    sum wins when a few rows carry the tail (6-8-bit keys, where t is small) and the self-bound when many do."""
    P = K._rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    w, wh, wbar, unr, E, tt, ldb = P["w"], P["wh"], P["wbar"], P["unread"], P["E"], P["tt"], P["ldb"]
    kappa, var, rng, hi2, hi4 = P["kappa"], P["var"], P["rng"], P["hi2"], P["hi4"]
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rho.double().expand_as(w) + nm
    g_mean, g_var = kappa * rb, var * rb * rb + hi2 * n2
    a_y = (wbar * (rng * rb + E(tt - ldb) * nm)).amax(-1)
    dk = delta / (14.0 if mode == "min" else 4.0)

    def sb(g, p_):
        sure = (wbar ** p_ * g * unr).sum(-1)
        if mode == "sure":
            return sure
        mm = hi2 if p_ == 1 else hi4
        return torch.minimum(K.self_bound_rowwise((wh ** p_ * g).sum(-1), (wbar ** p_ * g * mm * unr).amax(-1), dk,
                                                  P["lo_1"] if p_ == 1 else P["lo_2"]), sure)
    mY = sb(g_mean, 1)
    B2Y = torch.minimum(sb(g_var, 2), (wbar * torch.sqrt(g_var)).amax(-1) * sb(torch.sqrt(g_var), 1))
    mD = sb(kappa, 1)
    B2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    rY = K.bennett_radius(torch.sqrt(B2Y), a_y, dk)
    rD = K.bennett_radius(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = P["M"] + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    return (num / den).to(s.dtype)



def tail_bound_rand_split(s, s_hat, sigma, keep, rhat, nu2, oc, dim, lse_fix=K.NEG, delta: float = K.DELTA_T,
                          b_as=None, mom=None, ldebias=None):
    """Theorem R, split by coin source (GQA only: a row's key and value dithers are independent draws). Inputs as
    tail_bound_rand with rhat = |v_hat_i - c| (observed) and dim = the value dimension; nu2 = E|e_i|^2 = dim D_i^2 / 12.
    The identity, per unread row i (w_i = e^{s_i} exact and FIXED; w~_i the tier-1 weight; e_i = v_hat_i - v_i):
        Y_i = w_i (v_i - c) - w~_i (v_hat_i - c) = (w_i - w~_i)(v_hat_i - c) - w_i e_i,
    so Z (o - o_T) = K - V - (o_T - c) sum_T (w_i - w~_i) with K = sum_T (w_i - w~_i)(v_hat_i - c), V = sum_T w_i e_i.
      V involves the value coins only, with fixed weights. A subtractive-dithered value error is R^T (uniform on the
        cube of side D_i), so E exp<a, w_i e_i> <= exp(|a|^2 w_i^2 D_i^2 / 24) and V is sub-Gaussian with proxy
        v = sum w_i^2 D_i^2 / 12; Hsu-Kakade-Zhang (2012, Thm 1, A = I_dim): |V|^2 <= v (dim + 2 sqrt(dim x) + 2x)
        with probability >= 1 - e^{-x}. v is bounded surely on the truncation event (w_i <= w_bar_i).
      K, given the value coins (independent of the key coins), is a sum of independent vectors w_i (1 - Z_i) u_i with
        u_i = v_hat_i - c fixed: mean <= sum w_bar kappa rhat, fluctuation by the vector Bennett radius with
        B^2 = sum w_bar^2 var rhat^2 and range max w_bar rng rhat.
      sum_T (w_i - w~_i) as in Theorem R (mean + Bennett), also with the sure sums.
    Events: truncation (key coins) delta / 2; V, K and the weight sum at delta / 6 each. The value part costs
    ~sqrt(1 + 2 sqrt(x / dim) + 2 x / dim) x its standard deviation instead of Bernstein's sqrt(2 ln(2 / delta))."""
    P = K._rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    wh, wbar, unr = P["wh"], P["wbar"], P["unread"]
    kappa, var, rng = P["kappa"], P["var"], P["rng"]
    rh = rhat.double().expand_as(wbar)
    n2 = nu2.double().expand_as(wbar)
    dk = delta / 6.0
    x = math.log(1.0 / dk)
    dim = float(dim)
    rV = torch.sqrt((wbar * wbar * n2 * unr).sum(-1) * (1.0 + 2.0 * math.sqrt(x / dim) + 2.0 * x / dim))
    mK = (wbar * kappa * rh * unr).sum(-1)
    rK = K.bennett_radius(torch.sqrt((wbar * wbar * var * rh * rh * unr).sum(-1)), (wbar * rng * rh * unr).amax(-1), dk)
    mD = (wbar * kappa * unr).sum(-1)
    rD = K.bennett_radius(torch.sqrt((wbar * wbar * var * unr).sum(-1)), (wbar * rng * unr).amax(-1), dk)
    num = mK + rK + rV + (mD + rD) * oc.double()
    den = P["M"] + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    return (num / den).to(s.dtype)

def schedule(vote, pri, fracs=FRACS):
    """Nested coin-free read sets: vote [G, C] bool; pri [G, n, C] (the sketch's priority, coin-free; -inf on the
    vote) -> list of keep masks [G, n, C], S_j = vote ∪ top(f_j C) by pri."""
    G, n, C = pri.shape
    order = pri.argsort(-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(C).expand(G, n, C).contiguous())
    base = vote.unsqueeze(1).expand(G, n, C)
    return [base | (rank < int(round(f * C))) for f in fracs]


def certified_read(s, sh, sigma, sets, Vc, Vh, lf, of, target_fn, delta, b_as=None, mom=None, ldebias=None, nu_max=None,
                   nu2=None, attend=None, mix=None, dist=None, bound="split", Dv=None):
    """Apply the rule on one batch of head-steps. s / sh [G, r, n, C] exact / tier-1 (debiased) scores; sets from
    schedule(); Vc / Vh [G, C, d]; target_fn(oT [G, r, n, d], c) -> [G, r, n]; returns dict with J [G, n] (index into
    sets; len(sets) = read everything), the chosen output [G, r, n, d], and every set's bound [len, G, r, n].
    bound: "split" (tail_bound_rand_split; GQA), "v3s" (theory3_s1h5.tail_bound_rand_v3 with sure sums; needs Dv, the
    value rows' dither steps) or "v2" (cert_s1h5.tail_bound_rand_v2)."""
    m = len(sets) + 1                                    # + the exact read, which needs no bound
    dj = delta / (m - 1)
    G, r, n, C = s.shape
    J = torch.full((G, n), len(sets), dtype=torch.long)
    out = None
    bounds, outs = [], []
    for j, ke in enumerate(sets):
        km = ke.unsqueeze(1)
        c = attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
        oT = mix(s, sh, Vc, Vh, ke, lf, of)
        if bound == "split":
            bd = tail_bound_rand_split(s, sh, sigma, km, dist(c, Vh), nu2, (oT - c).norm(dim=-1), Vc.shape[-1], lf,
                                       delta=dj, b_as=b_as, mom=mom, ldebias=ldebias)
        elif bound == "v3s":
            bd = T3.tail_bound_rand_v3(s, sh, sigma, km, dist(c, Vh), nu_max, nu2, Dv, Vc.shape[-1], (oT - c).norm(dim=-1),
                                       lf, delta=dj, b_as=b_as, mom=mom, ldebias=ldebias, sure=True)
        else:
            bd = K.tail_bound_rand_v2(s, sh, sigma, km, dist(c, Vh), nu_max, nu2, (oT - c).norm(dim=-1), lf, delta=dj,
                                      b_as=b_as, mom=mom, ldebias=ldebias)
        ok = (bd <= target_fn(oT, c)).all(1)             # every query head of the KV head
        J = torch.where((J == len(sets)) & ok, torch.full_like(J, j), J)
        bounds.append(bd)
        outs.append(oT)
    exact = attend(s, Vc, lf, of)
    outs.append(exact)
    O = torch.stack(outs)                                # [m, G, r, n, d]
    idx = J.view(1, G, 1, n, 1).expand(1, G, r, n, O.shape[-1])
    out = O.gather(0, idx).squeeze(0)
    return dict(J=J, out=out, bounds=torch.stack(bounds), outs=O)


def _layers(key, layers, L=1600):
    """Per global GQA layer: the fixed quantities of oracle_s1h5's setup (the vote's 1/8 from a Lloyd-Max sketch, the
    last 16 of 48 tail positions measured, FP8's error, the schedule)."""
    import adapters_s1h5 as A
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    ids = A._ids(key, L)
    m = A._load(key).eval()
    with A.capture(m, list(layers)) as rec:
        with torch.no_grad():
            m(ids, use_cache=False)
    del m
    dt = torch.float64
    pos = torch.arange(L - 48, L)
    for li in layers:
        v = A.view_of(li, rec[li], pos)
        if v.family != "gqa" or getattr(v, "sink", None) is not None:
            print(f"layer {li}: {v.family}{' with sinks' if getattr(v, 'sink', None) is not None else ''}, skipped "
                  "(plain GQA only here)")
            continue
        C, nv = L - 48, 32
        q, Kx, Vx = v.q.to(dt), v.K.to(dt), v.V.to(dt)
        G, r, n_all, d = q.shape
        sc = v.scaling
        s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
        s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
        s, sf = s_all[..., :C], s_all[..., C:]
        Vc, Vf = Vx[:, :C], Vx[:, C:]
        lf = torch.logsumexp(sf, -1)
        of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
        Kl = Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(dt)   # the sketch
        pv = torch.softmax(torch.einsum("grnd,gcd->grnc", q[:, :, :nv], Kl) * sc, -1).sum((1, 2))
        vote = K.topk_keep(pv, C // 8)
        ms = slice(nv, n_all)
        s, lf, of, qm = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms]
        o = A._attend(s, Vc, lf, of)
        sl = torch.einsum("grnd,gcd->grnc", qm, Kl) * sc
        pri = (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote.unsqueeze(1), K.NEG)   # [G, n, C]
        e8 = (A._attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kx[:, :C].float()).to(dt)) * sc,
                        fp8_e4m3(Vc.float()).to(dt), lf, of) - o).norm(dim=-1)
        Rk = Q.random_rotation(d, "cpu", dt, 50)
        yield dict(li=li, C=C, G=G, r=r, n=n_all - nv, sc=sc, s=s, lf=lf, of=of, qm=qm, o=o, Kx=Kx, Vc=Vc, vote=vote,
                   sets=schedule(vote, pri), e8=e8, Rk=Rk, Rv=Q.random_rotation(Vx.shape[-1], "cpu", dt, 150),
                   qr=qm @ Rk.T, dv=Vx.shape[-1])


def _draw(S, kb, sd):
    """One dither draw of tier 1 (kb-bit keys, 4-bit values), unbiased tail: the inputs every bound takes."""
    import rquant_s1h5 as RQ
    G, C, li, sc = S["G"], S["C"], S["li"], S["sc"]
    Kh, stk = RQ.dither_quantize(S["Kx"][:, :C], kb, S["Rk"], 1000003 * sd + 2 * li)
    Vh, stv = RQ.dither_quantize(S["Vc"], 4, S["Rv"], 1000003 * sd + 2 * li + 1)
    st = stk.view(G, 1, 1, C)
    mom = RQ.coin_moments(S["qr"], st, sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), S["dv"])
    return dict(Vh=Vh, sh=torch.einsum("grnd,gcd->grnc", S["qm"], Kh) * sc - ldb, mom=mom, ldb=ldb, n2=n2, nm=nm,
                Dv=stv.view(G, 1, 1, C), sig=RQ.score_sigma(S["qm"].norm(dim=-1, keepdim=True), st, sc),
                bas=RQ.score_bound(S["qr"].abs().sum(-1, keepdim=True), st, sc))


def compare(key="llama1b", layers=(1,), delta=1e-3, bits=(4, 6, 8), draws=3, L=1600):
    """Every certificate on the same draws (the vote's rows read exactly): median bound / error, the share of
    head-steps certified within FP8's error, and coverage."""
    import adapters_s1h5 as A
    out = dict(model=key, delta=delta, draws=draws, results={})
    for S in _layers(key, layers, L):
        km = S["vote"].view(S["G"], 1, 1, S["C"])
        keep = S["vote"].unsqueeze(1).expand(S["G"], S["n"], S["C"])
        c = A._attend(S["s"].masked_fill(~km, K.NEG), S["Vc"], S["lf"], S["of"])
        for kb in bits:
            acc = {}
            for sd in range(draws):
                D = _draw(S, kb, sd)
                oT = A._mix(S["s"], D["sh"], S["Vc"], D["Vh"], keep, S["lf"], S["of"])
                err = (oT - S["o"]).norm(dim=-1)
                rho, oc = A._dist(c, D["Vh"]), (oT - c).norm(dim=-1)
                a8 = (S["s"], D["sh"], D["sig"], km, rho, D["nm"], D["n2"], oc, S["lf"])
                kw = dict(delta=delta, b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"])
                v3a = (S["s"], D["sh"], D["sig"], km, rho, D["nm"], D["n2"], D["Dv"], S["dv"], oc, S["lf"])
                bs = {"v2": K.tail_bound_rand_v2(*a8, **kw),
                      "v3 (part 6)": T3.tail_bound_rand_v3(*v3a, **kw),
                      "v3 + sure sums": T3.tail_bound_rand_v3(*v3a, sure=True, **kw),
                      "sure (v2 + sure sums)": tail_bound_rand_sure(*a8, mode="sure", **kw),
                      "split": tail_bound_rand_split(S["s"], D["sh"], D["sig"], km, rho, D["n2"], oc, S["dv"], S["lf"],
                                                     **kw)}
                for nmb, b in bs.items():
                    a = acc.setdefault(nmb, dict(cover=[], ratio=[], fp8=[]))
                    a["cover"].append(float((err <= b * (1 + 1e-9)).double().mean()))
                    a["ratio"].append(float((b / err).median()))
                    a["fp8"].append(float((b <= S["e8"]).double().mean()))
            res = {k_: {m_: (min(v_) if m_ == "cover" else sum(v_) / len(v_)) for m_, v_ in a.items()} for k_, a in acc.items()}
            out["results"].setdefault(f"L{S['li']}", {})[f"{kb}bit"] = res
            print(f"{key} L{S['li']} {kb}-bit: " + " | ".join(f"{k_} {v_['ratio']:.1f}x, FP8 {v_['fp8']:.2f}, cover "
                                                              f"{v_['cover']:.3f}" for k_, v_ in res.items()), flush=True)
    return out


def run(key="llama1b", layers=(1,), draws=4, delta=0.05, bits=(4, 6, 8), L=1600, eps_rel=(1e-3, 3e-3, 1e-2),
        bounds=("v2", "v3s", "split")):
    """The certified read per layer, key width and draw, at targets eps x |o_T| and at FP8's own error (an oracle
    target), against the best fixed fraction and the oracle stop (the first set whose TRUE error meets the target)."""
    import adapters_s1h5 as A
    out = dict(model=key, layers=list(layers), draws=draws, delta=delta, fracs=list(FRACS), results={})
    for S in _layers(key, layers, L):
        G, r, n, sets, o, e8, li = S["G"], S["r"], S["n"], S["sets"], S["o"], S["e8"], S["li"]
        for kb in bits:
            res = {}
            for sd in range(draws):
                D = _draw(S, kb, sd)
                targets = {f"rel{e:g}": (lambda oT, c, e=e: e * oT.norm(dim=-1)) for e in eps_rel}
                targets["fp8"] = lambda oT, c: e8                               # oracle target: FP8's own error here
                errs = None
                for tn, tf, bnd in ((tn_, tf_, b_) for tn_, tf_ in targets.items() for b_ in bounds):
                    R_ = certified_read(S["s"], D["sh"], D["sig"], sets, S["Vc"], D["Vh"], S["lf"], S["of"], tf, delta,
                                        b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"], nu_max=D["nm"], nu2=D["n2"],
                                        attend=A._attend, mix=A._mix, dist=A._dist, bound=bnd, Dv=D["Dv"])
                    if errs is None:
                        errs = (R_["outs"] - o.unsqueeze(0)).norm(dim=-1)        # [m, G, r, n] true errors per set
                    tgt = torch.stack([tf(R_["outs"][j], None) if tn != "fp8" else e8 for j in range(len(sets) + 1)])
                    err_J = (R_["out"] - o).norm(dim=-1)
                    tJ = tgt.gather(0, R_["J"].view(1, G, 1, n).expand(1, G, r, n)).squeeze(0)
                    meets = (errs <= tgt).all(2)                                  # [m, G, n] every query head meets
                    meets[-1] = True
                    J_or = meets.float().argmax(0)                                # oracle stop: first set that meets
                    size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(G, n, dtype=torch.float64)])
                    readf = lambda Jx: size.gather(0, Jx.view(1, G, n)).squeeze(0)  # noqa: E731
                    j95 = int(torch.nonzero(meets.double().mean((1, 2)) >= 0.95)[0])   # the fixed set meeting 95%
                    a = res.setdefault(f"{tn}/{bnd}", dict(cover=[], read_rule=[], read_oracle=[], read_fixed95=[],
                                                           exact_share=[]))
                    a["cover"].append(float((err_J <= tJ * (1 + 1e-9) + 1e-15).double().mean()))
                    a["read_rule"].append(float(readf(R_["J"]).mean()))
                    a["read_oracle"].append(float(readf(J_or).mean()))
                    a["read_fixed95"].append(float(size[j95].mean()))
                    a["exact_share"].append(float((R_["J"] == len(sets)).double().mean()))
            for tn, a in res.items():
                kb_res = {k_: sum(v_) / len(v_) for k_, v_ in a.items()}
                kb_res["traffic_rule"] = (1 - kb_res["read_rule"]) * (kb + 4) / 32 + kb_res["read_rule"]
                out["results"].setdefault(f"L{li}", {}).setdefault(f"{kb}bit", {})[tn] = kb_res
                print(f"{key} L{li} {kb}bit target {tn}: coverage {kb_res['cover']:.4f}; rows read exactly: rule "
                      f"{kb_res['read_rule']:.3f} (traffic {kb_res['traffic_rule']:.3f}), oracle stop "
                      f"{kb_res['read_oracle']:.3f}, best fixed fraction meeting 95% {kb_res['read_fixed95']:.3f}; "
                      f"head-steps falling back to an exact read {kb_res['exact_share']:.3f}", flush=True)
    return out


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv and not argv[0].startswith("--") else "certread"
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    import adapters_s1h5 as A
    key = g("--model", "llama1b")
    lay = g("--layers", "global")
    if lay == "global":                                  # every global softmax layer among the loaded ones
        m = A._load(key)
        lay = ",".join(str(x) for x in A.global_layers(getattr(m.config, "text_config", m.config)))
        del m
    layers = tuple(int(x) for x in lay.split(","))
    bits = tuple(int(x) for x in g("--bits", "4,6,8").split(","))
    if cmd == "compare":
        res = compare(key, layers, float(g("--delta", 1e-3)), bits, int(g("--draws", 3)), int(g("--L", 1600)))
    else:
        res = run(key, layers, int(g("--draws", 4)), float(g("--delta", 0.05)), bits, int(g("--L", 1600)))
    with open(g("--out", f"{cmd}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
