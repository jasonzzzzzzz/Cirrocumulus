#!/usr/bin/env python3
"""Checks for theory5_s1h5.py (part 8: the tilted-tail moments and the Hsu-Kakade-Zhang key sum), by quadrature and
brute force.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory5_s1h5.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory5_s1h5.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
sys.path.insert(0, HERE)
sys.path.append(METH)
import theory4_s1h5 as K4  # noqa: E402
import theory5_s1h5 as K5  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import test_r14_stage1h_r5 as T5  # noqa: E402  (its instance builders; nothing is run on import)
import test_theory3_s1h5 as TT3  # noqa: E402  (its _instance; nothing is run on import)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _law(a, n):
    """Score error delta = sum_j a_j U_j (U_j iid uniform on [-1, 1]) for one or two channels: support points and
    probabilities of an n (x n) midpoint grid, exact to O(1/n^2) in every moment."""
    u = (torch.arange(n, dtype=DT) + 0.5) / n * 2.0 - 1.0
    if len(a) == 1:
        return a[0] * u, torch.full((n,), 1.0 / n, dtype=DT)
    x = (a[0] * u.view(-1, 1) + a[1] * u.view(1, -1)).flatten()
    return x, torch.full((n * n,), 1.0 / (n * n), dtype=DT)


def test_proxy():
    print("\n[theory5] the key proxy and the tilted tail, by quadrature (one and two channels: the uniform law is "
          "furthest from Gaussian there)")
    worst, worst_tau, n_case = 0.0, 0.0, 0
    lam_u = torch.cat([-torch.logspace(-2, 3, 120, dtype=DT), torch.logspace(-2, 3, 120, dtype=DT)])
    for a in ([0.05], [0.3], [1.0], [3.0], [0.2, 0.05], [0.6, 0.6], [2.0, 0.5], [4.0, 4.0]):
        x, p = _law(a, 4000 if len(a) == 1 else 1500)
        sig = math.sqrt(sum(v * v for v in a) / 3.0)
        for t in (0.02, 0.1, 0.5, 1.0, 2.0, 4.0, 1e9):
            for ldb in (0.0, 0.3):
                tt = torch.tensor(min(t, 50.0), dtype=DT)
                Z = torch.exp(-x.clamp(-t, t) - ldb)
                X = Z - (p * Z).sum()
                s2 = float(K5.key_proxy(torch.tensor(sig, dtype=DT), tt, torch.tensor(ldb, dtype=DT)))
                lam = lam_u / math.sqrt(s2)                               # the scale where the proxy binds
                psi = torch.logsumexp(lam.view(-1, 1) * X.view(1, -1) + torch.log(p).view(1, -1), -1)
                worst = max(worst, float((psi / (lam * lam * s2 / 2.0)).max()))
                n_case += 1
            if t < 1e8:                                                   # the tilted tail against its bound
                ldb = 0.2
                tail = float((p * torch.exp(-x - ldb) * (x.abs() > t)).sum())
                m = dict(lmu_lo=torch.zeros(1, dtype=DT), lmu_hi=torch.zeros(1, dtype=DT),
                         lmu2_hi=torch.zeros(1, dtype=DT))
                tt = torch.tensor([t], dtype=DT)
                P = dict(wbar=torch.ones(1, dtype=DT), unread=torch.ones(1, dtype=torch.bool), tt=tt,
                         ldb=torch.tensor([ldb], dtype=DT), E=lambda v: torch.exp(v), z=torch.tensor(t / sig, dtype=DT),
                         kappa=None, var=None, rng=None)
                tau = float(K5.moments(P, torch.tensor([sig], dtype=DT), None, m)["tau"])
                if tail > 0:
                    worst_tau = max(worst_tau, tail / tau)
    check("the key proxy s^2 = e^{-2 ldb} sigma^2 (1 + (sqrt 12 / pi)(e^t - 1))^2 bounds the exact log-MGF of the clipped "
          "weight ratio at every lambda (|lambda| s from 0.01 to 1000, both signs; one or two channels; clip levels 0.02 to "
          "none); the tilted-tail tau bounds E[e^{-delta - ldb}; |delta| > t]",
          worst <= 1.0 + 1e-6 and worst_tau <= 1.0 + 1e-6,
          f"({n_case} cases; max psi / (lambda^2 s^2 / 2) {worst:.3f}; max tail / tau {worst_tau:.3f})")


def _conv_law(widths, n=1 << 15):
    """delta = sum_j a_j U_j by repeated convolution on a grid (support points, probabilities)."""
    L = sum(widths)
    x = torch.linspace(-L, L, n, dtype=DT)
    h = float(x[1] - x[0])
    f = torch.zeros(n, dtype=DT)
    f[n // 2] = 1.0
    for a in widths:
        k = max(1, int(round(a / h)))
        box = torch.ones(2 * k + 1, dtype=DT) / (2 * k + 1)
        f = torch.nn.functional.conv1d(f.view(1, 1, -1), box.view(1, 1, -1), padding=k).view(-1)
    return x, f / f.sum()


def test_exact_proxy():
    print("\n[theory5] the exact-law key proxy bounds the exact log-MGF (quadrature for one or two channels, convolution "
          "for eight and 64), and is near the optimum at 6-bit-like noise")
    worst, n_case, near = 0.0, 0, []
    lam_u = torch.cat([-torch.logspace(-3, 3.5, 300, dtype=DT), torch.logspace(-3, 3.5, 300, dtype=DT)])
    g = torch.Generator().manual_seed(9)
    cases = [([0.3], 4.0), ([1.0], 1.5), ([0.2, 0.05], 3.0), ([0.6, 0.6], 2.5)]
    for d_ in (8, 64):
        for s0 in (0.05, 0.1, 0.42):
            w_ = torch.rand(d_, generator=g, dtype=DT) + 0.2
            w_ = w_ * s0 / torch.sqrt((w_ * w_).sum() / 3.0)
            cases.append((w_.tolist(), 5.5))
    for a, z in cases:
        x, p = _law(a, 4000 if len(a) == 1 else 1500) if len(a) <= 2 else _conv_law(a)
        sig = math.sqrt(sum(v * v for v in a) / 3.0)
        for ldb in (0.0, sig * sig / 2.0):
            tt = min(z * sig, sum(a))                                    # the pipeline caps t at the a.s. bound
            Zt = torch.exp(-x.clamp(-tt, tt) - ldb)
            EZ, EZ2 = float((p * Zt).sum()), float((p * Zt * Zt).sum())
            s2 = float(K5.exact_key_proxy(torch.tensor([a], dtype=DT), torch.tensor([tt], dtype=DT),
                                          torch.tensor([ldb], dtype=DT), torch.tensor([EZ * (1 - 1e-9)], dtype=DT),
                                          torch.tensor([EZ2 * (1 + 1e-9)], dtype=DT)))
            lam = lam_u / math.sqrt(s2)
            psi = torch.logsumexp(lam.view(-1, 1) * (Zt - EZ).view(1, -1) + torch.log(p.clamp_min(1e-300)).view(1, -1), -1)
            ratio = psi / (lam * lam * s2 / 2.0)
            worst = max(worst, float(ratio.max()))
            n_case += 1
            if len(a) == 64 and abs(sig - 0.1) < 1e-9:
                opt = float((2 * psi / (lam * lam)).max())
                near.append(s2 / opt)
    check("psi(lambda) <= lambda^2 s^2 / 2 everywhere (|lambda| s from 1e-3 to 3000), and within 1.25x of the "
          "optimum for 64 channels at sigma = 0.1", worst <= 1.0 + 1e-6 and max(near) <= 1.25,
          f"({n_case} cases; max psi / (lambda^2 s^2 / 2) {worst:.3f}; exact / optimum at 64 channels, sigma 0.1: "
          f"{[round(v, 3) for v in near]})")


def test_logconcave():
    print("\n[theory5] the density bound behind the exact proxy: a tilted uniform sum is log-concave, so "
          "sup f^2 Var <= 1 (Bobkov-Chistyakov); checked by convolution")
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for d_ in (1, 2, 8, 64):
        w_ = (torch.rand(d_, generator=g, dtype=DT) + 0.1).tolist()
        x, p = _conv_law(w_)
        h = float(x[1] - x[0])
        for kap in (0.0, 1.0, 5.0, 20.0, 100.0):
            q = torch.softmax(kap * x + torch.log(p.clamp_min(1e-300)), 0)
            m = (q * x).sum()
            worst = max(worst, float((q.max() / h) ** 2 * (q * (x - m) ** 2).sum()))
    check("sup f^2 Var <= 1 (to grid accuracy) for 1-64 channels and tilts 0-100", worst <= 1.0 + 1e-3,
          f"(max {worst:.4f}; the one-channel, strongly tilted law approaches the exponential's 1)")


def test_prep():
    print("\n[theory5] the per-row truncation: heavy=None is cert_s1h5._rand_prep; the heavy split spends delta / 2")
    g = torch.Generator().manual_seed(41)
    x = T5._rand_data(g, kind="random", C=256)
    s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
    keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32).unsqueeze(1)
    r_ = TT3._instance(x, keep, 5, bits=6)
    args = (r_["s"], r_["shd"], r_["sig"], keep, r_["lf"], 1e-3, r_["b_as"], r_["mom"], r_["ldb"])
    P0, P1 = K._rand_prep(*args), K5.prep(*args)
    same = all(torch.equal(P0[k_], P1[k_]) for k_ in ("w", "wh", "wf", "tt", "unread", "ldb", "M", "wbar"))
    unr = P0["unread"]
    hv = torch.zeros_like(unr)
    hv[..., :40] = True
    P2 = K5.prep(*args, heavy=hv, share=0.7)
    spent = (2.0 * torch.exp(-P2["zrow"] ** 2 / 2.0) * unr).sum(-1)
    check("equal maps; the split's failure budget sum_i 2 e^{-z_i^2 / 2} = delta / 2 per head-step",
          same and bool((spent <= 0.5e-3 * (1 + 1e-9)).all()) and bool((spent >= 0.5e-3 * (1 - 1e-6)).all()),
          f"(budget spent {float(spent.min()):.6g} to {float(spent.max()):.6g})")


def test_hkz():
    print("\n[theory5] the Hsu-Kakade-Zhang radius, by simulation (Gaussian and Rademacher coefficients)")
    g = torch.Generator().manual_seed(5)
    bad = []
    for trial in range(6):
        n, d = (3, 40, 300, 5, 64, 1000)[trial], (2, 8, 64, 64, 4, 16)[trial]
        U = torch.randn(n, d, generator=g, dtype=DT) * torch.rand(n, 1, generator=g, dtype=DT)
        pr = torch.rand(n, generator=g, dtype=DT) ** 3
        Mx = (pr.view(-1, 1, 1) * U.unsqueeze(-1) * U.unsqueeze(-2)).sum(0)
        for x in (1.0, 3.0, 6.0):
            r = float(K5.hkz_radius(Mx, x))
            for kind in ("gauss", "rade"):
                N = 200000 if not FAST else 50000
                eta = (torch.randn(N, n, generator=g, dtype=DT) if kind == "gauss"
                       else torch.randint(0, 2, (N, n), generator=g).to(DT) * 2 - 1) * pr.sqrt()
                frac = float(((eta @ U).norm(dim=-1) > r).double().mean())
                bad.append(frac / math.exp(-x))
    check("|sum eta_i u_i| exceeds the radius at x with frequency <= e^{-x} (x = 1, 3, 6; six shapes)",
          max(bad) <= 1.0 + 0.2, f"(max frequency / e^-x {max(bad):.3f})")


def _bf_instances(g, n_trial, n_seed, bits_list=(2, 4, 6, 8)):
    for trial in range(n_trial):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = T5._rand_data(g, kind=kind, C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32).unsqueeze(1)
        for bits in bits_list:
            for seed in range(n_seed):
                r_ = TT3._instance(x, keep, 1000 * trial + 10 * bits + seed, bits=bits)
                r_.update(Kx=x["Kx"], q=x["q"])
                yield bits, keep, r_


def test_reproduces_split():
    print("\n[theory5] forms=('bennett',), tilt=False is exactly theory4_s1h5.tail_bound_rand_split")
    g = torch.Generator().manual_seed(31)
    gap = 0.0
    for bits, keep, r_ in _bf_instances(g, 4, 2):
        kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
        a = K4.tail_bound_rand_split(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["n2"], r_["oc"], r_["d"], r_["lf"],
                                     **kw)
        b = K5.tail_bound_rand_split_hkz(r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"],
                                         r_["d"], r_["lf"], forms=("bennett",), tilt=False, **kw)
        gap = max(gap, float(((a - b) / a).abs().max()))
    check("equal to 1e-10 relative", gap < 1e-10, f"(max relative gap {gap:.1e})")


def test_bounds():
    print("\n[theory5] the new forms by brute force (random, spiky, 30x outlier, adversarial; 2-8-bit keys)")
    g = torch.Generator().manual_seed(21)
    deltas = (0.5, 0.1, 0.01, K.DELTA_T)
    from sievelib import quant as Q
    forms = {"tilt": (("bennett",), True, {}), "hkz": (("hkz",), True, {}), "tilt+hkz": (("bennett", "hkz"), True, {}),
             "hkz_notilt": (("hkz",), False, {}), "hkz+exact": (("hkz",), True, {"exact": 1}),
             "hkz+exact+r1": (("hkz",), True, {"exact": 1, "r1": 1})}
    fails_ = {(f, d): 0 for f in forms for d in deltas}
    n, ratio = 0, {f: [] for f in ("split",) + tuple(forms)}
    for bits, keep, r_ in _bf_instances(g, 8 if FAST else 16, 6 if FAST else 20):
        kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
        args = (r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"], r_["lf"])
        Kx = r_["Kx"]
        Kl = Q.quantize_keys(Kx.float(), 4, Q.random_rotation(Kx.shape[-1], "cpu", torch.float32, 7)).to(DT)
        sl = torch.einsum("grd,gcd->grc", r_["q"], Kl) * r_["sc"]                 # the coin-free sketch
        unr = ~keep.expand_as(r_["s"])
        heavy = torch.zeros_like(unr).scatter_(-1, sl.masked_fill(~unr, K.NEG).topk(16, -1).indices, True) & unr
        for f, (fm, tl, ex_) in forms.items():
            extra = {}
            if "exact" in ex_:
                extra["exact"] = dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=8, t_min=0.0)
            if "r1" in ex_:
                extra.update(heavy=heavy, share=0.5)
            for d_ in deltas:
                b = K5.tail_bound_rand_split_hkz(*args, delta=d_, forms=fm, tilt=tl, **extra, **kw)
                fails_[(f, d_)] += int((r_["err"] > b).sum())
                if bits >= 6 and d_ == K.DELTA_T:
                    ratio[f] += (b / r_["err"]).flatten().tolist()
        if bits >= 6:
            ratio["split"] += (K4.tail_bound_rand_split(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["n2"], r_["oc"],
                                                        r_["d"], r_["lf"], **kw) / r_["err"]).flatten().tolist()
        n += r_["err"].numel()
    med = {f: sorted(v)[len(v) // 2] for f, v in ratio.items()}
    check("no form fails more than delta x cases at any delta, none at DELTA_T; at 6-8 bits hkz+exact is tighter than "
          "the split at the median",
          all(f_ <= d_ * n for (f, d_), f_ in fails_.items()) and all(fails_[(f, K.DELTA_T)] == 0 for f in forms)
          and med["hkz+exact"] < med["split"],
          f"({n} head cases; failures {sum(fails_.values())}; median bound / error {({f: round(v, 2) for f, v in med.items()})})")


def test_certified_read():
    print("\n[theory5] the certified adaptive read with the new bound (exact proxies, remedy 1) meets its target with "
          "probability >= 1 - delta")
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(22)
    tgt_rel = 0.25
    miss, n, early = 0, 0, 0
    for trial in range(4 if FAST else 8):
        x = T5._rand_data(g, kind=("random", "spiky", "outlier", "adversarial")[trial % 4], C=256)
        q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
        G, C, d = Kx.shape
        s = torch.einsum("grd,gcd->grc", q, Kx) * sc
        Kl = Q.quantize_keys(Kx.float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)
        sl = torch.einsum("grd,gcd->grc", q, Kl) * sc
        vote = K.topk_keep(torch.softmax(sl, -1).amax(1), 16)
        pri = (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote, K.NEG).unsqueeze(1)
        sets = K4.schedule(vote, pri)
        R = Q.random_rotation(d, "cpu", DT, 0)
        u = lambda t: t.unsqueeze(2)                                                                   # noqa: E731
        att = lambda s_, V_, lf_, of_: K.attend(s_.squeeze(2), V_.unsqueeze(1), lf_.squeeze(2), of_.squeeze(2)).unsqueeze(2)  # noqa: E731
        mix = lambda s_, sh_, Vr, Vh_, ke, lf_, of_: K.tail_output(                                     # noqa: E731
            s_.squeeze(2), sh_.squeeze(2), Vr.unsqueeze(1), Vh_.unsqueeze(1), ke.squeeze(1).unsqueeze(1), lf_.squeeze(2),
            of_.squeeze(2)).unsqueeze(2)
        o = K.attend(s, V.unsqueeze(1), lf, of)
        for seed in range(6 if FAST else 20):
            for bits in (6, 8):
                Kh, stk = RQ.dither_quantize(Kx, bits, R, 7000 + 100 * trial + 10 * seed + bits)
                Vh, stv = RQ.dither_quantize(V, 4, Q.random_rotation(d, "cpu", DT, 1), 900000 + 100 * trial + seed)
                qr = q @ R.T
                mom = RQ.coin_moments(qr, stk.unsqueeze(1), sc)
                ldb = RQ.debias_log(mom)
                sh = torch.einsum("grd,gcd->grc", q, Kh) * sc - ldb
                sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), stk.unsqueeze(1), sc)
                bas = RQ.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), sc)
                n2, _ = RQ.value_moments(stv.unsqueeze(1), d)
                kw = dict(b_as=u(bas), mom={k_: u(v_) for k_, v_ in mom.items()}, ldebias=u(ldb))

                def bf(km, c, oT, dj):
                    unr = ~km.expand(G, q.shape[1], 1, C)
                    hv = torch.zeros_like(unr).scatter_(-1, u(sl).masked_fill(~unr, K.NEG).topk(8, -1).indices, True) & unr
                    return K5.tail_bound_rand_split_hkz(u(s), u(sh), u(sig), km, Vh, c, u(n2), (oT - c).norm(dim=-1), d,
                                                        u(lf), delta=dj, exact=dict(qr=u(qr), step=stk, scaling=sc,
                                                                                    rows=8, t_min=0.0), heavy=hv, share=0.5, **kw)
                R_ = K5.certified_read(u(s), u(sh), sets, V, Vh, u(lf), u(of), lambda oT, c: tgt_rel * oT.norm(dim=-1),
                                       0.05, bf, att, mix)
                err = (R_["out"].squeeze(2) - o).norm(dim=-1)
                miss += int((err > tgt_rel * R_["out"].squeeze(2).norm(dim=-1) * (1 + 1e-9) + 1e-15).sum())
                early += int((R_["J"] < len(sets)).sum())
                n += err.numel()
    check("certified read (nested coin-free schedule, delta = 0.05, target 25% of the output): the output meets its "
          "target on at least 95% of head cases, and the rule stops before the exact read on some",
          miss <= 0.05 * n and early > 0, f"({n} head cases; misses {miss}; early stops {early})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_proxy()
    test_exact_proxy()
    test_logconcave()
    test_prep()
    test_hkz()
    test_reproduces_split()
    test_bounds()
    test_certified_read()
    print(f"\n{'ALL THEORY5 TESTS PASSED' if not fails else f'{fails} THEORY5 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
