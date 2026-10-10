#!/usr/bin/env python3
"""Checks for theory3_s1h5.py (Theorem R v3, Theorem M and their pieces), by brute force.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory3_s1h5.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory3_s1h5.py      # + Llama-3.2-1B, CPU
"""
import math, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
sys.path.insert(0, HERE)
sys.path.append(METH)
import theory3_s1h5 as K3  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import theory2_s1h5 as K2  # noqa: E402
import test_r14_stage1h_r5 as T5  # noqa: E402  (its instance builders; nothing is run on import)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _instance(x, keep, seed, bits=4, unbiased=True):
    """One dither draw of tier 1 (keys at `bits`, values at 4) on a T5 instance; everything the bounds need."""
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
    d = Kx.shape[-1]
    R, Rv = Q.random_rotation(d, "cpu", DT, 0), Q.random_rotation(d, "cpu", DT, 1)
    Kh, stk = RQ.dither_quantize(Kx, bits, R, seed)
    Vh, stv = RQ.dither_quantize(V, 4, Rv, seed + 7919)
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    sh = torch.einsum("grd,gcd->grc", q, Kh) * sc
    qr = q @ R.T
    sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), stk.unsqueeze(1), sc)
    b_as = RQ.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), sc)
    mom = RQ.coin_moments(qr, stk.unsqueeze(1), sc)
    ldb = RQ.debias_log(mom) if unbiased else torch.zeros_like(mom["lmu_lo"])
    n2, nm = RQ.value_moments(stv.unsqueeze(1), d)
    o = K.attend(s, V.unsqueeze(1), lf, of)
    c = K.attend(s, V.unsqueeze(1), lf, of, keep)
    oT = K.tail_output(s, sh - ldb, V.unsqueeze(1), Vh.unsqueeze(1), keep, lf, of)
    return dict(s=s, shd=sh - ldb, sig=sig, b_as=b_as, mom=mom, ldb=ldb, n2=n2, nm=nm, o=o, c=c, oT=oT, V=V, Vh=Vh,
                stv=stv, stk=stk, qr=qr, d=d, lf=lf, of=of, sc=sc, rho=K.center_dist(Vh.unsqueeze(1), c),
                oc=(oT - c).norm(dim=-1), err=(o - oT).norm(dim=-1))


def test_v3_validity():
    print("\n[theory3] Theorem R v3 by brute force (random, spiky, 30x outlier, adversarial; 2/4/6-bit keys; plain and "
          "unbiased tails)")
    g = torch.Generator().manual_seed(123)
    deltas = (0.3, 0.1, 0.01, 1e-3)
    nf = {d_: 0 for d_ in deltas}
    n, r2, r3, worse = 0, [], [], 0
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = T5._rand_data(g, kind=kind, C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32).unsqueeze(1)
        for seed in range(12 if FAST else 25):
            for bits in (2, 4, 6):
                for unb in (False, True):
                    z = _instance(x, keep, 9000 * trial + seed, bits, unb)
                    a = (z["s"], z["shd"], z["sig"], keep, z["rho"], z["nm"], z["n2"])
                    b2 = K2.tail_bound_rand_v2(*a, z["oc"], z["lf"], b_as=z["b_as"], mom=z["mom"], ldebias=z["ldb"])
                    for d_ in deltas:
                        b3 = K3.tail_bound_rand_v3(*a, z["stv"].unsqueeze(1), z["d"], z["oc"], z["lf"], d_, z["b_as"],
                                                   z["mom"], z["ldb"])
                        nf[d_] += int((z["err"] > b3).sum())
                        if d_ == K.DELTA_T:
                            r3 += (b3 / z["err"]).flatten().tolist()
                            r2 += (b2 / z["err"]).flatten().tolist()
                            worse += int((b3 > 1.05 * b2).sum())
                    n += z["err"].numel()
    med = lambda v: sorted(v)[len(v) // 2]                                    # noqa: E731
    check("v3 never below the error beyond its delta (none at delta = 1e-3), never more than 5% above v2",
          all(f <= d_ * n for d_, f in nf.items()) and nf[K.DELTA_T] == 0 and worse == 0,
          f"({n} head cases; failures {nf}; median bound / error v2 {med(r2):.1f} -> v3 {med(r3):.1f}; "
          f"v3 > 1.05 v2 on {worse})")


def test_identities():
    print("\n[theory3] the identities: Theorem R's regrouped by noise source, and the o-centred one (Theorem M)")
    g = torch.Generator().manual_seed(5)
    worst1 = worst2 = 0.0
    for trial in range(8):
        x = T5._rand_data(g, kind=("random", "spiky", "outlier", "adversarial")[trial % 4], C=128)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 16).unsqueeze(1)
        z = _instance(x, keep, trial)
        s, lf = z["s"], z["lf"]
        mx = torch.maximum(s.amax(-1), lf).unsqueeze(-1)
        w, wt = torch.exp(s - mx), torch.exp(z["shd"] - mx)
        unr = (~keep).expand_as(s).double()
        Mw = (w * (1 - unr)).sum(-1) + torch.exp(lf - mx.squeeze(-1))
        Z, Zt = Mw + (w * unr).sum(-1), Mw + (wt * unr).sum(-1)
        V, Vh = z["V"].unsqueeze(1), z["Vh"].unsqueeze(1)
        D = (w - wt) * unr
        e = V - Vh
        Kc = torch.einsum("grc,grcd->grd", D, V - z["c"].unsqueeze(-2))
        Vt = torch.einsum("grc,grcd->grd", wt * unr, e.expand(*s.shape, e.shape[-1]))
        lhs = Z.unsqueeze(-1) * (z["o"] - z["oT"])
        rhs = Kc - (z["oT"] - z["c"]) * D.sum(-1, keepdim=True) + Vt
        worst1 = max(worst1, float(((lhs - rhs).norm(dim=-1) / lhs.norm(dim=-1).clamp_min(1e-300)).max()))
        N = torch.einsum("grc,grcd->grd", D, V - z["o"].unsqueeze(-2)) + Vt
        worst2 = max(worst2, float(((Zt.unsqueeze(-1) * (z["o"] - z["oT"]) - N).norm(dim=-1)
                                    / N.norm(dim=-1).clamp_min(1e-300)).max()))
    check("Z (o - o~) = K_c - (o~ - c) K_2 + V  and  (M + sum w~)(o - o~) = sum D (v - o) + sum w~ e, to rounding",
          worst1 < 1e-9 and worst2 < 1e-9, f"(max relative residuals {worst1:.1e}, {worst2:.1e})")


def test_second_moment():
    print("\n[theory3] Theorem M: the exact second moment against Monte Carlo (dithered tier, 4-bit keys and values)")
    import rquant_s1h5 as RQ
    g = torch.Generator().manual_seed(9)
    nd = 1500 if FAST else 4000
    worst_rel, worst_mom, viol = 0.0, 0.0, 0
    for trial in range(4):
        x = T5._rand_data(g, Hkv=1, r=2, C=96, d=32, kind=("random", "spiky", "outlier", "adversarial")[trial])
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 12).unsqueeze(1)
        Ns, errs, Zs, Z2s = [], [], [], []
        for sd in range(nd):
            z = _instance(x, keep, 100003 * trial + sd)
            s, lf = z["s"], z["lf"]
            mx = torch.maximum(s.amax(-1), lf).unsqueeze(-1)
            w, wt = torch.exp(s - mx), torch.exp(z["shd"] - mx)
            unr = (~keep).expand_as(s).double()
            Zs.append(wt / w)
            Ns.append(torch.einsum("grc,grcd->grd", (w - wt) * unr, z["V"].unsqueeze(1) - z["o"].unsqueeze(-2))
                      + torch.einsum("grc,gcd->grd", wt * unr, z["V"] - z["Vh"]))
            errs.append(((z["o"] - z["oT"]) ** 2).sum(-1))
        EZ = K3.exact_coin_moments(z["qr"], z["stk"].unsqueeze(1), z["sc"], z["ldb"])
        Zst = torch.stack(Zs)
        worst_mom = max(worst_mom, float(((Zst.mean(0) - EZ[1]).abs() / (Zst.std(0) / nd ** 0.5)).max()))
        u = z["V"].unsqueeze(1) - z["o"].unsqueeze(-2)
        pred = K3.second_moment(w, u, EZ[1], EZ[2], z["n2"].expand_as(w), (~keep).expand_as(s))
        Nst = torch.stack(Ns)
        nn = (Nst ** 2).sum(-1)
        meas, se = nn.mean(0), nn.std(0) / nd ** 0.5
        worst_rel = max(worst_rel, float(((meas - pred).abs() / se).max()))
        Mw = (w * (~keep).logical_not().expand_as(s)).sum(-1) + torch.exp(lf - mx.squeeze(-1))
        e2 = torch.stack(errs)
        viol += int((e2 > nn / Mw.unsqueeze(0) ** 2 * (1 + 1e-9)).sum())             # pointwise |o - o~| <= |N| / M
    check("E Z, E Z^2 from the sinh products match Monte Carlo; E|N|^2 = the formula within 4.5 s.e.; "
          "|o - o~| <= |N| / M on every draw", worst_mom < 4.5 and worst_rel < 4.5 and viol == 0,
          f"(max |MC - exact| / s.e.: moments {worst_mom:.2f}, E|N|^2 {worst_rel:.2f}; pointwise violations {viol})")


def test_chi2():
    print("\n[theory3] the chi^2 value radius: rotated-uniform noise with arbitrary weights and steps")
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for d in (16, 64, 128):
        R = Q.random_rotation(d, "cpu", DT, 2)
        wt = torch.rand(40, generator=g, dtype=DT) ** 3
        st = 0.1 + torch.rand(40, generator=g, dtype=DT)
        eps = (torch.rand(20000, 40, d, generator=g, dtype=DT) - 0.5) * st.view(1, 40, 1)
        Vn = torch.einsum("c,ncd->nd", wt, eps @ R).norm(dim=-1)
        s2 = (wt * wt * st * st / 12.0).sum()
        for L in (1.0, 2.0, 3.0, 5.0):
            freq = float((Vn > K3.chi2_value_radius(s2, d, L)).double().mean())
            worst = max(worst, freq / math.exp(-L))
    check("P(|V| > s sqrt(d + 2 sqrt(dL) + 2L)) <= e^{-L} (d = 16, 64, 128; L = 1..5)", worst <= 1.0,
          f"(worst frequency / e^-L {worst:.3f})")
    # v3's variance self-bound: E|x - e|^4 <= (|x|^2 + E|e|^2)(|x|^2 + nu^2) for the rotated-uniform error, any x.
    # Closed form (odd moments vanish): E|x - e|^4 = r^4 + 2 r^2 n2 + 4 r^2 D^2/12 + d D^4/80 + d (d - 1) D^4/144.
    worst4, worst_mc = 0.0, 0.0
    for d in (2, 16, 64):
        R = Q.random_rotation(d, "cpu", DT, 5)
        D = 0.7
        e = ((torch.rand(40000, d, generator=g, dtype=DT) - 0.5) * D) @ R
        n2, nu2 = d * D * D / 12.0, d * D * D / 4.0
        for scale in (0.0, 0.05, 0.3, 1.0, 3.0, 10.0):
            x = scale * torch.randn(d, generator=g, dtype=DT)
            r2 = float(x @ x)
            exact = r2 * r2 + 2 * r2 * n2 + 4 * r2 * D * D / 12 + d * D ** 4 / 80 + d * (d - 1) * D ** 4 / 144
            worst4 = max(worst4, exact / ((r2 + n2) * (r2 + nu2)))
            mc = ((x - e) ** 2).sum(-1) ** 2
            worst_mc = max(worst_mc, abs(float(mc.mean()) - exact) / (float(mc.std()) / 40000 ** 0.5))
    check("E|x - e|^4 <= (|x|^2 + E|e|^2)(|x|^2 + nu^2) (closed form; d = 2, 16, 64; |x| from 0 to 10 steps), and the "
          "closed form matches Monte Carlo", worst4 <= 1.0 and worst_mc < 4.5,
          f"(worst ratio {worst4:.4f}; worst |MC - closed form| / s.e. {worst_mc:.2f})")


def test_onerow():
    print("\n[theory3] the one-row lemma: a max of fixed quantities needs one row's deviation, not a union")
    g = torch.Generator().manual_seed(4)
    d1, n_trials, viol = 0.05, 4000, 0
    for _ in range(n_trials):
        C = 300
        w = torch.exp(2.0 * torch.randn(C, generator=g, dtype=DT))
        sig = 0.2 + torch.rand(C, generator=g, dtype=DT)
        delta = sig * torch.randn(C, generator=g, dtype=DT)                    # observed w~ = w e^{-delta}
        wt = w * torch.exp(-delta)
        viol += int(float(w.max()) > float((wt * torch.exp(K3.onerow_t(sig, d1))).max()))
    check("P(max_i w_i > max_i w~_i e^{t1}) <= delta1 (t1 = sigma sqrt(2 ln 1/delta1); heavy-tailed weights)",
          viol <= d1 * n_trials, f"({viol} of {n_trials} at delta1 = {d1})")


def test_real():
    print("\n[theory3] Llama-3.2-1B layer 1 (real weights, CPU, 1,600 tokens): v3 vs v2, and Theorem M")
    import adapters_s1h5 as A
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    L = 1600
    ids = A._ids("llama1b", L)
    m = A._load("llama1b").eval()
    with A.capture(m, [1]) as rec:
        with torch.no_grad():
            m(ids, use_cache=False)
    del m
    pos = torch.arange(L - 48, L)
    v = A.view_of(1, rec[1], pos)
    q, Kx, Vx = v.q.double(), v.K.double(), v.V.double()
    G, r, n_all, d = q.shape
    C, nv, sc = L - 48, 32, v.scaling
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc = Vx[:, :C]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    Kl = Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).double()
    vote = K.topk_keep(torch.softmax(torch.einsum("grnd,gcd->grnc", q[:, :, :nv], Kl) * sc, -1).sum((1, 2)), C // 8)
    ms = slice(nv, n_all)
    s, lf, of, qm = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms]
    o = A._attend(s, Vc, lf, of)
    keep, km = vote.unsqueeze(1).expand(G, n_all - nv, C), vote.view(G, 1, 1, C)
    c = A._attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
    Rk, Rv = Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(Vx.shape[-1], "cpu", DT, 150)
    qr = qm @ Rk.T
    lZ = torch.logaddexp(torch.logsumexp(s, -1), lf)
    w = torch.exp(s - lZ.unsqueeze(-1))
    u = Vc.unsqueeze(1).unsqueeze(1) - o.unsqueeze(-2)
    for kb in (4, 6, 8):
        E, b2s, b3s = [], [], []
        for sd in range(60 if FAST else 100):
            Kh, stk = RQ.dither_quantize(Kx[:, :C], kb, Rk, 1000003 * sd + 1)
            Vh, stv = RQ.dither_quantize(Vc, 4, Rv, 1000003 * sd + 2)
            st = stk.view(G, 1, 1, C)
            mom = RQ.coin_moments(qr, st, sc)
            ldb = RQ.debias_log(mom)
            shd = torch.einsum("grnd,gcd->grnc", qm, Kh) * sc - ldb
            oT = A._mix(s, shd, Vc, Vh, keep, lf, of)
            E.append(((oT - o) ** 2).sum(-1))
            if sd < 10:
                sig = RQ.score_sigma(qm.norm(dim=-1, keepdim=True), st, sc)
                bas = RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc)
                n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), Vx.shape[-1])
                rho, oc = A._dist(c, Vh), (oT - c).norm(dim=-1)
                b2s.append(K2.tail_bound_rand_v2(s, shd, sig, km, rho, nm, n2, oc, lf, delta=0.05, b_as=bas, mom=mom,
                                                 ldebias=ldb))
                b3s.append(K3.tail_bound_rand_v3(s, shd, sig, km, rho, nm, n2, stv.view(G, 1, 1, C), Vx.shape[-1], oc,
                                                 lf, 0.05, bas, mom, ldb))
            if sd == 0:
                EZ = K3.exact_coin_moments(qr, st, sc, ldb)
                pred = K3.second_moment(w, u, EZ[1], EZ[2], n2.expand_as(w), (~km).expand_as(w))
        E2 = torch.stack(E)
        err = E2.sqrt()
        B2, B3 = torch.stack(b2s).median(0).values, torch.stack(b3s).median(0).values
        ratio = float((E2.mean(0) / pred).median())
        check(f"{kb}-bit keys: v3 covers every draw and is tighter than v2 at the median; measured MSE / Theorem M's "
              f"first-order prediction within 5%", bool((err <= B3.unsqueeze(0)).all()) and
              float((B3 / B2).median()) < 1.0 and abs(ratio - 1.0) < 0.05,
              f"(median v3 / v2 {float((B3 / B2).median()):.2f}; MSE / prediction {ratio:.3f})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_identities()
    test_chi2()
    test_onerow()
    test_second_moment()
    test_v3_validity()
    if not FAST:
        test_real()
    print(f"\n{'ALL THEORY3 TESTS PASSED' if not fails else f'{fails} THEORY3 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
