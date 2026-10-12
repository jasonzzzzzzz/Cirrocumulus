#!/usr/bin/env python3
"""Checks for nested_s1h5.py (the nested coin-free store, theory.md §13), by brute force.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_nested_s1h5.py --fast
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
sys.path.insert(0, HERE)
sys.path.append(METH)
import nested_s1h5 as N  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402
import theory3_s1h5 as K3  # noqa: E402
import theory4_s1h5 as K4  # noqa: E402
import theory5_s1h5 as K5  # noqa: E402
import test_r14_stage1h_r5 as T5  # noqa: E402  (its instance builders; nothing is run on import)
import test_theory3_s1h5 as TT3  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _instance(x, keep, seed, base_bits=4, res_bits=2):
    """test_theory3_s1h5._instance with the nested tier for the keys (values: 4-bit dithered)."""
    from sievelib import quant as Q
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
    d = Kx.shape[-1]
    R, Rv = Q.random_rotation(d, "cpu", DT, 0), Q.random_rotation(d, "cpu", DT, 1)
    nz = N.nested_quantize(Kx, base_bits, res_bits, R, seed)
    Kh, stk = nz["Xh"], nz["step"]
    Vh, stv = RQ.dither_quantize(V, 4, Rv, seed + 7919)
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    sh = torch.einsum("grd,gcd->grc", q, Kh) * sc
    qr = q @ R.T
    mom = N.coin_moments_pc(qr, stk, sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.unsqueeze(1), d)
    o = K.attend(s, V.unsqueeze(1), lf, of)
    c = K.attend(s, V.unsqueeze(1), lf, of, keep)
    oT = K.tail_output(s, sh - ldb, V.unsqueeze(1), Vh.unsqueeze(1), keep, lf, of)
    return dict(s=s, shd=sh - ldb, sig=N.score_sigma_pc(qr, stk, sc), b_as=N.score_bound_pc(qr, stk, sc), mom=mom,
                ldb=ldb, n2=n2, nm=nm, o=o, c=c, oT=oT, V=V, Vh=Vh, stv=stv, stk=stk, qr=qr, d=d, lf=lf, of=of, sc=sc,
                rho=K.center_dist(Vh.unsqueeze(1), c), oc=(oT - c).norm(dim=-1), err=(o - oT).norm(dim=-1), Kh=Kh)


def test_lemma():
    print("\n[nested] Lemma N by simulation: the residual error is uniform on +-D/2 and independent of the data")
    res = N.lemma(n_rows=1024, d=64, draws=200 if FAST else 400)
    ok = all(abs(r["mean"]) < 1e-3 and abs(r["var_over_1_12"] - 1) < 5e-3 and abs(r["m4_over_1_80"] - 1) < 1e-2
             and r["max_abs"] <= 0.5 + 1e-9 and abs(r["corr_with_y"]) < 1e-2 and r["mean_abs_per_coord_max_in_se"] < 6
             and r["base_vs_quantize_keys"] < 1e-4 for r in res.values())
    check("moments of a uniform (mean 0, variance D^2/12, 4th moment D^4/80), never beyond D/2, uncorrelated with y; "
          "the base is quant.quantize_keys", ok)


def test_reduces():
    print("\n[nested] with one step per row, the per-coordinate quantities are rquant_s1h5's and theory3_s1h5's")
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(3)
    x = T5._rand_data(g, kind="outlier", C=128)
    q, Kx, sc = x["q"], x["Kx"], x["sc"]
    G, C, d = Kx.shape
    R = Q.random_rotation(d, "cpu", DT, 0)
    _, stk = RQ.dither_quantize(Kx, 6, R, 5)
    st3 = stk.unsqueeze(-1).expand(G, C, d).contiguous()
    qr = q @ R.T
    m_row, m_pc = RQ.coin_moments(qr, stk.unsqueeze(1), sc), N.coin_moments_pc(qr, st3, sc)
    gap = max(float(((m_row[k] - m_pc[k]).abs() / m_row[k].abs().clamp_min(1e-300)).max()) for k in m_row)
    sg = float(((RQ.score_sigma(q.norm(dim=-1, keepdim=True), stk.unsqueeze(1), sc) - N.score_sigma_pc(qr, st3, sc)).abs()
                / N.score_sigma_pc(qr, st3, sc)).max())
    bs = float(((RQ.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), sc) - N.score_bound_pc(qr, st3, sc)).abs()
                / N.score_bound_pc(qr, st3, sc)).max())
    ldb = RQ.debias_log(m_row)
    e_row = K3.exact_coin_moments(qr, stk.unsqueeze(1), sc, ldb)
    e_pc = N.exact_coin_moments_pc(qr, st3, sc, ldb)
    ex = max(float(((e_row[k] - e_pc[k]).abs() / e_row[k]).max()) for k in (1, 2))
    # part 8's exact-law proxy: one step per row given as [G, C] or as [G, C, d]
    keep = K.topk_keep(torch.softmax(torch.einsum("grd,gcd->grc", q, Kx) * sc, -1).amax(1), 16).unsqueeze(1)
    r_ = TT3._instance(x, keep, 11, bits=6)
    kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"], forms=("hkz",))
    args = (r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"], r_["lf"])
    b2 = K5.tail_bound_rand_split_hkz(*args, exact=dict(qr=r_["qr"], step=r_["stk"], scaling=sc, rows=8, t_min=0.0), **kw)
    st3b = r_["stk"].unsqueeze(-1).expand(*r_["stk"].shape, d).contiguous()
    b3 = K5.tail_bound_rand_split_hkz(*args, exact=dict(qr=r_["qr"], step=st3b, scaling=sc, rows=8, t_min=0.0), **kw)
    xr = float(((b2 - b3).abs() / b2).max())
    check("coin moments, sigma, the a.s. bound, the exact moments and part 8's exact proxy agree to 1e-10",
          max(gap, sg, bs, ex, xr) < 1e-10,
          f"(max relative gaps: moments {gap:.1e}, sigma {sg:.1e}, bound {bs:.1e}, exact {ex:.1e}, part 8 {xr:.1e})")


def test_moments_mc():
    print("\n[nested] E Z, E Z^2 and Theorem M's E|N|^2 with per-coordinate steps, against Monte Carlo")
    g = torch.Generator().manual_seed(9)
    nd = 1500 if FAST else 4000
    worst_mom, worst_rel = 0.0, 0.0
    for trial, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        x = T5._rand_data(g, Hkv=1, r=2, C=96, d=32, kind=kind)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 12).unsqueeze(1)
        Zs, Ns = [], []
        for sd in range(nd):
            z = _instance(x, keep, 100003 * trial + sd, res_bits=2)
            s, lf = z["s"], z["lf"]
            mx = torch.maximum(s.amax(-1), lf).unsqueeze(-1)
            w, wt = torch.exp(s - mx), torch.exp(z["shd"] - mx)
            unr = (~keep).expand_as(s).double()
            Zs.append(wt / w)
            Ns.append(torch.einsum("grc,grcd->grd", (w - wt) * unr, z["V"].unsqueeze(1) - z["o"].unsqueeze(-2))
                      + torch.einsum("grc,gcd->grd", wt * unr, z["V"] - z["Vh"]))
        EZ = N.exact_coin_moments_pc(z["qr"], z["stk"], z["sc"], z["ldb"])     # the steps don't depend on the coins
        Zst = torch.stack(Zs)
        worst_mom = max(worst_mom, float(((Zst.mean(0) - EZ[1]).abs() / (Zst.std(0) / nd ** 0.5)).max()),
                        float((((Zst ** 2).mean(0) - EZ[2]).abs() / ((Zst ** 2).std(0) / nd ** 0.5)).max()))
        u = z["V"].unsqueeze(1) - z["o"].unsqueeze(-2)
        pred = K3.second_moment(w, u, EZ[1], EZ[2], z["n2"].expand_as(w), (~keep).expand_as(s))
        nn = (torch.stack(Ns) ** 2).sum(-1)
        worst_rel = max(worst_rel, float(((nn.mean(0) - pred).abs() / (nn.std(0) / nd ** 0.5)).max()))
    check("E Z and E Z^2 (exact, per coordinate) and E|N|^2 (Theorem M) within 4.5 s.e. of Monte Carlo",
          worst_mom < 4.5 and worst_rel < 4.5, f"(max |MC - exact| / s.e.: moments {worst_mom:.2f}, E|N|^2 {worst_rel:.2f})")


def test_bounds():
    print("\n[nested] every Theorem R form on the nested tier by brute force (random, spiky, 30x outlier, adversarial; "
          "bases 2-4 bits, residuals 2-3 bits)")
    g = torch.Generator().manual_seed(21)
    deltas = (0.5, 0.1, 0.01, K.DELTA_T)
    from sievelib import quant as Q
    forms = ("v1", "v2", "split", "hkz", "hkz+exact", "hkz+exact+r1")
    fails_ = {(f, d_): 0 for f in forms for d_ in deltas}
    n, ratio = 0, {f: [] for f in forms}
    widths = ((2, 2), (4, 2), (4, 3)) if FAST else ((2, 2), (3, 2), (4, 2), (4, 3), (3, 3))
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = T5._rand_data(g, kind=kind, C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32).unsqueeze(1)
        Kl = Q.quantize_keys(x["Kx"].float(), 4, Q.random_rotation(x["Kx"].shape[-1], "cpu", torch.float32, 7)).to(DT)
        sl = torch.einsum("grd,gcd->grc", x["q"], Kl) * x["sc"]
        unr = ~keep.expand_as(s_ex)
        heavy = torch.zeros_like(unr).scatter_(-1, sl.masked_fill(~unr, K.NEG).topk(16, -1).indices, True) & unr
        for bb, rb in widths:
            for seed in range(4 if FAST else 12):
                r_ = _instance(x, keep, 1000 * trial + 10 * (4 * bb + rb) + seed, base_bits=bb, res_bits=rb)
                kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
                args = (r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"], r_["lf"])
                for d_ in deltas:
                    for f in forms:
                        if f == "v1":
                            b = K.tail_bound_rand(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"],
                                                  r_["oc"], r_["lf"], delta=d_, **kw)
                        elif f == "v2":
                            b = K.tail_bound_rand_v2(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"],
                                                     r_["oc"], r_["lf"], delta=d_, **kw)
                        elif f == "split":
                            b = K4.tail_bound_rand_split(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["n2"],
                                                         r_["oc"], r_["d"], r_["lf"], delta=d_, **kw)
                        else:
                            extra = {}
                            if "exact" in f:
                                extra["exact"] = dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=8, t_min=0.0)
                            if "r1" in f:
                                extra.update(heavy=heavy, share=0.5)
                            b = K5.tail_bound_rand_split_hkz(*args, delta=d_, forms=("hkz",), **extra, **kw)
                        fails_[(f, d_)] += int((r_["err"] > b).sum())
                        if d_ == K.DELTA_T and bb + rb >= 6:
                            ratio[f] += (b / r_["err"]).flatten().tolist()
                n += r_["err"].numel()
    med = {f: round(sorted(v)[len(v) // 2], 2) for f, v in ratio.items() if v}
    check("no form fails more than delta x cases at any delta, none at DELTA_T",
          all(f_ <= d_ * n for (f, d_), f_ in fails_.items()) and all(fails_[(f, K.DELTA_T)] == 0 for f in forms),
          f"({n} head cases per form and delta; failures {sum(fails_.values())}; median bound / error at 6-7 bits {med})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_lemma()
    test_reduces()
    test_moments_mc()
    test_bounds()
    print(f"\n{'ALL NESTED TESTS PASSED' if not fails else f'{fails} NESTED TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
