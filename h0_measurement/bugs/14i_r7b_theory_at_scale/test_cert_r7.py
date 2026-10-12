#!/usr/bin/env python3
"""Checks for cert_r7.py and tier_r7.py: the copies against their originals, the GPU forms against the reference forms,
the proxies by quadrature, every bound by brute force, the certified read's stop, Theorem M by Monte Carlo, and parity
with the frozen fixtures of ../14h_theory (fixtures/, when present). Self-contained: its own instance generators and
rotations; ../14h_methodology_improve is imported read-only, and only for the copies' parity (skipped if absent).

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_cert_r7.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_cert_r7.py
    ... test_cert_r7.py --quick      (the worker's preflight: copies, GPU forms, fixtures; under a minute)
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cert_r7 as K  # noqa: E402
import tier_r7 as RQ  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}", flush=True)
    if not cond:
        fails += 1


# ------------------------------------------------------------------ instances (data fixed; only the coins are random)
def rot(d, seed):
    """A fixed rotation (QR of a seeded Gaussian matrix, signs fixed)."""
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def rand_data(g, G=2, r=4, n=1, C=256, d=64, kind="random"):
    """Exact keys, values, queries and fixed rows (test_r14_stage1h_r5._rand_data, with n steps): kinds random, spiky
    (a few rows aligned with the queries), outlier (one huge channel), adversarial (every key along the queries' mean,
    tail values offset)."""
    q = torch.randn(G, r, n, d, generator=g, dtype=DT)
    Kx = torch.randn(G, C, d, generator=g, dtype=DT)
    V = torch.randn(G, C, d, generator=g, dtype=DT)
    qm = q.mean((1, 2))
    if kind == "spiky":
        idx = torch.randint(0, C, (G, 4), generator=g)
        for h in range(G):
            Kx[h, idx[h]] += 2.0 * qm[h]
    if kind == "outlier":
        Kx[..., 0] *= 30.0
        V[..., 1] *= 30.0
    if kind == "adversarial":
        qd = qm / qm.norm(dim=-1, keepdim=True)
        Kx = 0.3 * Kx + (torch.rand(G, C, 1, generator=g, dtype=DT) * 4.0) * qd.unsqueeze(1)
        V = V + 3.0 * qd.unsqueeze(1)
    sf = torch.randn(G, r, n, 3, generator=g, dtype=DT)
    Vf = torch.randn(G, 3, d, generator=g, dtype=DT)
    return dict(q=q, Kx=Kx, V=V, lf=torch.logsumexp(sf, -1), of=torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf),
                sc=d ** -0.5)


def sketch(x, bits=4):
    """A coin-free sketch of the keys (a deterministic uniform quantizer in a fixed rotation): its scores [G, r, n, C]."""
    R = rot(x["Kx"].shape[-1], 7)
    y = x["Kx"] @ R.T
    st = 2.0 * y.abs().amax(-1, keepdim=True) / (2 ** bits - 1)
    yq = (torch.floor(y / st) + 0.5) * st
    return torch.einsum("grnd,gcd->grnc", x["q"], yq @ R) * x["sc"]


def vote_of(x, frac=8):
    """The read set: the sketch's top C/frac rows by summed attention over the group (coin-free), [G, C]."""
    sl = sketch(x)
    return K.topk_keep(torch.softmax(sl, -1).sum((1, 2)), x["Kx"].shape[1] // frac), sl


def instance(x, keep, seed, bits=6, unbiased=True, vbits=4):
    """One dither draw of tier 1 (keys at `bits`, values at vbits) on keep [G, 1, 1, C]: everything the bounds take."""
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lf"], x["of"], x["sc"]
    G, C, d = Kx.shape
    R, Rv = rot(d, 0), rot(d, 1)
    Kh, stk = RQ.dither_quantize(Kx, bits, R, seed)
    Vh, stv = RQ.dither_quantize(V, vbits, Rv, seed + 7919)
    s = torch.einsum("grnd,gcd->grnc", q, Kx) * sc
    sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
    qr = q @ R.T
    st = stk.view(G, 1, 1, C)
    sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), st, sc)
    b_as = RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc)
    mom = RQ.coin_moments(qr, st, sc)
    ldb = RQ.debias_log(mom) if unbiased else torch.zeros_like(mom["lmu_lo"])
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), d)
    o = K.attend(s, V, lf, of)
    c = K.attend(s.masked_fill(~keep, K.NEG), V, lf, of)
    ke = keep[:, 0].expand(G, s.shape[2], C)
    oT = K.mix(s, sh - ldb, V, Vh, ke, lf, of)
    err, zr = K.tail_err(s, sh - ldb, V, Vh, ke, lf, o)
    return dict(s=s, shd=sh - ldb, sig=sig, b_as=b_as, mom=mom, ldb=ldb, n2=n2, nm=nm, o=o, c=c, oT=oT, V=V, Vh=Vh,
                stv=stv, stk=stk, qr=qr, d=d, lf=lf, of=of, sc=sc, rho=K.dist(c, Vh), oc=(oT - c).norm(dim=-1),
                err=err, err_direct=(o - oT).norm(dim=-1), zr=zr, Kx=Kx, q=q)


def _cases(g, n_trial, n_seed, bits_list=(2, 4, 6, 8), C=256):
    for trial in range(n_trial):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = rand_data(g, kind=kind, C=C)
        vote, sl = vote_of(x)
        keep = vote.view(vote.shape[0], 1, 1, -1)
        for bits in bits_list:
            for seed in range(n_seed):
                yield bits, keep, sl, instance(x, keep, 1000 * trial + 10 * bits + seed, bits=bits)


def _p8_kw(r_, bits, sl, keep, exact=True, r1=True):
    kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"], forms=("bennett", "hkz") if bits <= 4 else ("hkz",))
    if exact:
        kw["exact"] = dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=8, t_min=0.0)
    if r1:
        kw.update(heavy=K.heavy_rows(sl, keep, 16), share=0.5)
    return kw


# ------------------------------------------------------------------ the copies
def test_copies():
    print("\n[r7] the copies: tier_r7 = rquant_s1h5 and cert_r7's copied functions = cert_s1h5, bit for bit")
    if not os.path.isdir(METH):
        check("../14h_methodology_improve not present: skipped", True)
        return
    sys.path.append(METH)
    try:
        import rquant_s1h5 as RQ0
        import cert_s1h5 as K0
    except Exception as e:                                   # noqa: BLE001
        check(f"METH modules not importable ({e!r}): skipped", True)
        return
    finally:
        sys.path.remove(METH)
    g = torch.Generator().manual_seed(1)
    same = True
    for dt in (torch.float64, torch.float32):
        X = torch.randn(3, 300, 64, generator=g, dtype=DT) * torch.rand(3, 300, 1, generator=g, dtype=DT) * 5
        R = rot(64, 3)
        for bits in (4, 6, 8):
            a, sa = RQ.dither_quantize(X, bits, R, 1234 + bits, dtype=dt)
            b, sb = RQ0.dither_quantize(X, bits, R, 1234 + bits, dtype=dt)
            same &= torch.equal(a, b) and torch.equal(sa, sb) and torch.equal(RQ.dither_step(X, bits, R, dt), sa)
    q = torch.randn(3, 4, 2, 64, generator=g, dtype=DT)
    st = torch.rand(3, 1, 1, 300, generator=g, dtype=DT)
    m1, m0 = RQ.coin_moments(q, st, 0.125), RQ0.coin_moments(q, st, 0.125)
    same &= all(torch.equal(m1[k], m0[k]) for k in m0) and torch.equal(RQ.debias_log(m1), RQ0.debias_log(m0))
    same &= all(torch.equal(x_, y_) for x_, y_ in zip(RQ.value_moments(st, 64), RQ0.value_moments(st, 64)))
    same &= torch.equal(RQ.score_sigma(q.norm(dim=-1, keepdim=True), st, 0.125),
                        RQ0.score_sigma(q.norm(dim=-1, keepdim=True), st, 0.125))
    same &= torch.equal(RQ.score_bound(q.abs().sum(-1, keepdim=True), st, 0.125),
                        RQ0.score_bound(q.abs().sum(-1, keepdim=True), st, 0.125))
    tier_ok = same
    worst = 0
    for bits, keep, sl, r_ in _cases(g, 4, 2, (4, 6)):
        args = (r_["s"], r_["shd"], r_["sig"], keep, r_["lf"], 1e-3, r_["b_as"], r_["mom"], r_["ldb"])
        P1, P0 = K._rand_prep(*args), K0._rand_prep(*args)
        same &= all(torch.equal(P1[k], P0[k]) for k in P0 if torch.is_tensor(P0[k]))
        a8 = (r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"], r_["oc"], r_["lf"])
        kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
        same &= torch.equal(K.tail_bound_rand_v2(*a8, **kw), K0.tail_bound_rand_v2(*a8, **kw))
        B, a = torch.rand(50, generator=g, dtype=DT), torch.rand(50, generator=g, dtype=DT)
        same &= torch.equal(K.bennett_radius(B, a, 1e-3), K0.bennett_radius(B, a, 1e-3))
        same &= torch.equal(K.trunc_z(torch.tensor([3, 300]), 1e-4), K0.trunc_z(torch.tensor([3, 300]), 1e-4))
        sc_ = torch.randn(4, 300, generator=g, dtype=DT)
        same &= torch.equal(K.topk_keep(sc_, 37), K0.topk_keep(sc_, 37))
        worst += 1
    check("tier: dither_quantize (fp64 and fp32 coins), dither_step, coin_moments, debias_log, value_moments, "
          "score_sigma, score_bound; cert: _rand_prep, tail_bound_rand_v2, bennett_radius, trunc_z, topk_keep",
          same, f"(tier {'equal' if tier_ok else 'DIFFERENT'}; {worst} bound instances)")


# ------------------------------------------------------------------ the GPU forms
def test_tail_err():
    print("\n[r7] tail_err (the difference form) equals |o - o~| computed directly, and its Z~/Z is the design's normaliser "
          "over the exact one")
    g = torch.Generator().manual_seed(12)
    worst_e, worst_z = 0.0, 0.0
    for trial, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        x = rand_data(g, kind=kind, C=300, n=3)
        vote, _ = vote_of(x)
        keep = vote.view(vote.shape[0], 1, 1, -1)
        for bits in (4, 6, 8):
            r_ = instance(x, keep, 100 * trial + bits, bits=bits)
            worst_e = max(worst_e, float(((r_["err"] - r_["err_direct"]).abs() / r_["err_direct"]).max()))
            G, C = vote.shape
            ke = keep.expand_as(r_["s"])
            xs = torch.where(ke, r_["s"], r_["shd"])
            m = torch.maximum(xs.amax(-1), r_["lf"])
            Zt = torch.exp(xs - m.unsqueeze(-1)).sum(-1) + torch.exp(r_["lf"] - m)
            Z = torch.exp(r_["s"] - m.unsqueeze(-1)).sum(-1) + torch.exp(r_["lf"] - m)
            worst_z = max(worst_z, float(((r_["zr"] - Zt / Z).abs() / (Zt / Z)).max()))
    check("relative gaps below 1e-9", worst_e < 1e-9 and worst_z < 1e-12,
          f"(error {worst_e:.1e}; normaliser ratio {worst_z:.1e})")


def test_gram():
    print("\n[r7] the HKZ matrix: gram_exact = the reference weighted_gram; Lemma S's split bounds both norms from above")
    g = torch.Generator().manual_seed(2)
    gap, ok_split, eq_full, infl = 0.0, True, True, []
    for C, H in ((300, 64), (1000, 128), (257, 300)):
        cw = torch.rand(2, 3, 2, C, generator=g, dtype=DT) ** 4 * (torch.rand(2, 3, 2, C, generator=g, dtype=DT) < 0.9)
        Vh = torch.randn(2, C, 16, generator=g, dtype=DT) * 3 + 1.0
        c = torch.randn(2, 3, 2, 16, generator=g, dtype=DT)
        M0 = K.weighted_gram(cw, Vh, c)
        M1 = K.gram_exact(cw, Vh, c, chunk=97)
        gap = max(gap, float(((M1 - M0).abs().amax((-2, -1)) / M0.abs().amax((-2, -1))).max()))
        f0, l0 = (M0 * M0).sum((-2, -1)), torch.linalg.eigvalsh(M0)[..., -1]
        f1, l1, sh = K.gram_norms(cw, Vh, c, rows=H)
        ok_split &= bool((f1 >= f0 * (1 - 1e-12)).all() and (l1 >= l0 * (1 - 1e-12)).all() and (sh >= 0).all())
        if H >= C:
            eq_full &= bool(torch.allclose(f1, f0, rtol=1e-10) and torch.allclose(l1, l0, rtol=1e-10))
        else:
            infl.append(float((l1 / l0).max()))
    check("gram_exact matches the direct form to 1e-11 (relative to the largest entry); the split's |M|_F^2 and |M|_op "
          "are upper bounds; rows >= C is the exact matrix", gap < 1e-11 and ok_split and eq_full,
          f"(max gap {gap:.1e}; split |M|_op inflation up to {max(infl):.3f}x on these heavy-tailed weights)")


def test_series():
    print("\n[r7] Lemma P: the separable series for sum_j log sinhc(a_ij) and Delta = sum_j [g(2a) - 2g(a)] against the "
          "closed form, from near-zero scores to past the series' range (the closed form takes over)")
    g = torch.Generator().manual_seed(3)
    worst1, worst2, n_far = 0.0, 0.0, 0
    for scale in (1e-12, 1e-4, 0.01, 0.1, 0.4, 1.0, 3.0):
        alpha = torch.rand(6, 64, generator=g, dtype=DT) * scale
        alpha[:, 0] *= 4.0                                                # one outlier channel
        step = torch.rand(6, 500, generator=g, dtype=DT) * 2.0
        L1, Dl = K.log_mgf_moments(alpha, step)
        a = alpha.unsqueeze(1) * step.unsqueeze(-1)
        g1, g2 = K._log_sinhc(a), K._log_sinhc(2.0 * a)
        R1, R2 = g1.sum(-1), (g2 - 2.0 * g1).sum(-1)
        worst1 = max(worst1, float(((L1 - R1).abs() / R1.abs().clamp_min(1e-300)).max()))
        worst2 = max(worst2, float(((Dl - R2).abs() / R2.abs().clamp_min(1e-300)).max()))
        n_far += int((2.0 * alpha.amax(-1, keepdim=True) * step > K.SER_X).sum())
    co = K._lsc_coefs(5)
    exact5 = [1 / 6, -1 / 180, 1 / 2835, -1 / 37800, 1 / 467775]
    check("relative error of L1 and Delta below 1e-12 everywhere; the first five coefficients are 1/6, -1/180, 1/2835, "
          "-1/37800, 1/467775", worst1 < 1e-12 and worst2 < 1e-12 and all(abs(x - y) < 1e-18 for x, y in zip(co, exact5)),
          f"(L1 {worst1:.1e}, Delta {worst2:.1e}; {n_far} rows on the closed form)")


def test_theorem_m():
    print("\n[r7] Theorem M per head-step: theorem_m = the reference second_moment, and E|N|^2 by Monte Carlo (the exact "
          "identity N = Z~ (o - o~))")
    g = torch.Generator().manual_seed(4)
    gap, zs, ratios = 0.0, [], []
    n_draw = 200 if FAST else 600
    for trial, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        x = rand_data(g, kind=kind, C=192, n=2)
        vote, _ = vote_of(x)
        keep = vote.view(vote.shape[0], 1, 1, -1)
        G, C, d = x["Kx"].shape
        R, Rv = rot(d, 0), rot(d, 1)
        for bits in (4, 6):
            stk = RQ.dither_step(x["Kx"], bits, R)
            stv = RQ.dither_step(x["V"], 4, Rv)
            qr = x["q"] @ R.T
            mom = RQ.coin_moments(qr, stk.view(G, 1, 1, C), x["sc"])
            ldb = RQ.debias_log(mom)
            s = torch.einsum("grnd,gcd->grnc", x["q"], x["Kx"]) * x["sc"]
            o = K.attend(s, x["V"], x["lf"], x["of"])
            n2 = d * stv * stv / 12.0
            TM = K.theorem_m(s, keep, x["lf"], x["V"], o, qr, stk, x["sc"], ldb, n2, chunk=50)
            # the reference: weights normalized by Z, exact moments by the closed form
            lZ = torch.logaddexp(torch.logsumexp(s, -1), x["lf"])
            w = torch.exp(s - lZ.unsqueeze(-1))
            u = x["V"].unsqueeze(1).unsqueeze(1) - o.unsqueeze(-2)
            EZ = K.exact_coin_moments(qr, stk.view(G, 1, 1, C), x["sc"], ldb)
            ref = K.second_moment(w, u, EZ[1], EZ[2], n2.view(G, 1, 1, C), ~keep.expand_as(s))
            gap = max(gap, float(((TM["pred"] - ref).abs() / ref).max()))
            N2, E2 = [], []
            ke = keep[:, 0].expand(G, s.shape[2], C)
            for sd in range(n_draw):
                Kh, _ = RQ.dither_quantize(x["Kx"], bits, R, 50000 * trial + 1000 * bits + sd)
                Vh, _ = RQ.dither_quantize(x["V"], 4, Rv, 70000 * trial + 1000 * bits + sd)
                sh = torch.einsum("grnd,gcd->grnc", x["q"], Kh) * x["sc"] - ldb
                err, zr = K.tail_err(s, sh, x["V"], Vh, ke, x["lf"], o)
                N2.append((zr * err) ** 2)
                E2.append(err ** 2)
            N2, E2 = torch.stack(N2), torch.stack(E2)
            zs.append(float(((N2.mean(0) - TM["pred"]) / (N2.std(0) / math.sqrt(n_draw))).abs().max()))
            ratios.append(float((E2.mean(0) / TM["pred"]).median()))
    check("theorem_m equals second_moment (closed-form moments) to 1e-10; the Monte Carlo mean of (Z~/Z)^2 |o - o~|^2 is "
          "within 4.5 standard errors of E|N|^2/Z^2 on every head-step", gap < 1e-10 and max(zs) < 4.5,
          f"(max relative gap {gap:.1e}; max |z| {max(zs):.2f}; median E|o - o~|^2 / pred per case "
          f"{[round(v, 3) for v in ratios]})")


# ------------------------------------------------------------------ the proxies (theory5's checks, ported)
def _law(a, n):
    u = (torch.arange(n, dtype=DT) + 0.5) / n * 2.0 - 1.0
    if len(a) == 1:
        return a[0] * u, torch.full((n,), 1.0 / n, dtype=DT)
    x = (a[0] * u.view(-1, 1) + a[1] * u.view(1, -1)).flatten()
    return x, torch.full((n * n,), 1.0 / (n * n), dtype=DT)


def _conv_law(widths, n=1 << 15):
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


def test_proxies():
    print("\n[r7] the key proxies against exact laws (quadrature, convolution): psi(lambda) <= lambda^2 s^2 / 2; the "
          "tilted tail tau bounds E[e^{-delta - ldb}; |delta| > t]")
    worst, worst_tau, worst_ex = 0.0, 0.0, 0.0
    lam_u = torch.cat([-torch.logspace(-2, 3, 80, dtype=DT), torch.logspace(-2, 3, 80, dtype=DT)])
    for a in ([0.05], [0.3], [1.0], [0.2, 0.05], [0.6, 0.6], [2.0, 0.5]):
        x, p = _law(a, 3000 if len(a) == 1 else 1000)
        sig = math.sqrt(sum(v * v for v in a) / 3.0)
        for t in (0.02, 0.1, 0.5, 1.0, 2.0, 1e9):
            for ldb in (0.0, 0.3):
                tt = torch.tensor(min(t, 50.0), dtype=DT)
                Z = torch.exp(-x.clamp(-t, t) - ldb)
                X = Z - (p * Z).sum()
                s2 = float(K.key_proxy(torch.tensor(sig, dtype=DT), tt, torch.tensor(ldb, dtype=DT)))
                lam = lam_u / math.sqrt(s2)
                psi = torch.logsumexp(lam.view(-1, 1) * X.view(1, -1) + torch.log(p).view(1, -1), -1)
                worst = max(worst, float((psi / (lam * lam * s2 / 2.0)).max()))
            if t < 1e8:
                ldb = 0.2
                tail = float((p * torch.exp(-x - ldb) * (x.abs() > t)).sum())
                m = dict(lmu_lo=torch.zeros(1, dtype=DT), lmu_hi=torch.zeros(1, dtype=DT), lmu2_hi=torch.zeros(1, dtype=DT))
                tt = torch.tensor([t], dtype=DT)
                P = dict(wbar=torch.ones(1, dtype=DT), unread=torch.ones(1, dtype=torch.bool), tt=tt,
                         ldb=torch.tensor([ldb], dtype=DT), E=lambda v: torch.exp(v), zrow=torch.tensor([t / sig], dtype=DT),
                         kappa=None, var=None, rng=None)
                tau = float(K.moments(P, torch.tensor([sig], dtype=DT), None, m)["tau"])
                if tail > 0:
                    worst_tau = max(worst_tau, tail / tau)
    g = torch.Generator().manual_seed(9)
    cases = [([0.3], 4.0), ([1.0], 1.5), ([0.6, 0.6], 2.5)]
    for d_ in ((8,) if FAST else (8, 64)):
        for s0 in (0.1, 0.42):
            w_ = torch.rand(d_, generator=g, dtype=DT) + 0.2
            w_ = w_ * s0 / torch.sqrt((w_ * w_).sum() / 3.0)
            cases.append((w_.tolist(), 5.5))
    lam_e = torch.cat([-torch.logspace(-3, 3.5, 200, dtype=DT), torch.logspace(-3, 3.5, 200, dtype=DT)])
    for a, z in cases:
        x, p = _law(a, 3000 if len(a) == 1 else 1000) if len(a) <= 2 else _conv_law(a)
        sig = math.sqrt(sum(v * v for v in a) / 3.0)
        for ldb in (0.0, sig * sig / 2.0):
            tt = min(z * sig, sum(a))
            Zt = torch.exp(-x.clamp(-tt, tt) - ldb)
            EZ, EZ2 = float((p * Zt).sum()), float((p * Zt * Zt).sum())
            s2 = float(K.exact_key_proxy(torch.tensor([a], dtype=DT), torch.tensor([tt], dtype=DT),
                                         torch.tensor([ldb], dtype=DT), torch.tensor([EZ * (1 - 1e-9)], dtype=DT),
                                         torch.tensor([EZ2 * (1 + 1e-9)], dtype=DT)))
            lam = lam_e / math.sqrt(s2)
            psi = torch.logsumexp(lam.view(-1, 1) * (Zt - EZ).view(1, -1) + torch.log(p.clamp_min(1e-300)).view(1, -1), -1)
            worst_ex = max(worst_ex, float((psi / (lam * lam * s2 / 2.0)).max()))
    check("P-LSI and P-exact bound the exact log-MGF everywhere; tau bounds the tilted tail",
          worst <= 1 + 1e-6 and worst_tau <= 1 + 1e-6 and worst_ex <= 1 + 1e-6,
          f"(max psi / (lambda^2 s^2 / 2): P-LSI {worst:.3f}, P-exact {worst_ex:.3f}; max tail / tau {worst_tau:.3f})")


# ------------------------------------------------------------------ brute force
def test_bounds():
    print("\n[r7] every bound by brute force (random, spiky, 30x outlier, adversarial; 2-8-bit keys; the vote's rows read)")
    g = torch.Generator().manual_seed(21)
    deltas = (0.5, 0.1, 0.01, K.DELTA_T)
    forms = {"split": dict(forms=("bennett",), tilt=False), "hkz": {}, "hkz+exact": dict(exact=1),
             "p8 (hkz+exact+r1)": dict(exact=1, r1=1), "p8, split Gram": dict(exact=1, r1=1, rows=40)}
    fails_ = {(f, d): 0 for f in list(forms) + ["v2"] for d in deltas}
    ratio = {f: [] for f in list(forms) + ["v2"]}
    gram_ge = True
    n = 0
    for bits, keep, sl, r_ in _cases(g, 8 if FAST else 16, 5 if FAST else 16):
        for f, o_ in forms.items():
            kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"],
                      forms=o_.get("forms", ("bennett", "hkz") if bits <= 4 else ("hkz",)), tilt=o_.get("tilt", True))
            if o_.get("exact"):
                kw["exact"] = dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=8, t_min=0.0)
            if o_.get("r1"):
                kw.update(heavy=K.heavy_rows(sl, keep, 16), share=0.5)
            for d_ in deltas:
                b = K.bound_p8(r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"],
                               r_["lf"], delta=d_, gram_rows=o_.get("rows"), **kw)
                fails_[(f, d_)] += int((r_["err"] > b).sum())
                if d_ == K.DELTA_T and bits >= 6:
                    ratio[f] += (b / r_["err"]).flatten().tolist()
                if f == "p8, split Gram" and d_ == K.DELTA_T:
                    b0 = K.bound_p8(r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"],
                                    r_["lf"], delta=d_, **kw)
                    gram_ge &= bool((b >= b0 * (1 - 1e-12)).all())
        for d_ in deltas:
            b = K.tail_bound_rand_v2(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"], r_["oc"], r_["lf"],
                                     delta=d_, b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
            fails_[("v2", d_)] += int((r_["err"] > b).sum())
            if d_ == K.DELTA_T and bits >= 6:
                ratio["v2"] += (b / r_["err"]).flatten().tolist()
        n += r_["err"].numel()
    med = {f: sorted(v)[len(v) // 2] for f, v in ratio.items() if v}
    check("no form fails more than delta x cases at any delta and none at 1e-3; the split Gram's bound is never below "
          "the exact one; at 6-8 bits part 8 is tighter than v2 and the split at the median",
          all(v <= d_ * n for (f, d_), v in fails_.items()) and all(fails_[(f, K.DELTA_T)] == 0 for f in ratio)
          and gram_ge and med["p8 (hkz+exact+r1)"] < min(med["split"], med["v2"]),
          f"({n} head cases; failures {sum(fails_.values())}; median bound / error "
          f"{ {f: round(v, 2) for f, v in med.items()} })")


def _naive_read(s, sh, sets, Vc, Vh, lf, of, target, delta, bk, sl):
    """theory5_s1h5.certified_read: every set's bound for every head-step, exact HKZ matrix, no brackets."""
    G, r, n, C = s.shape
    m = len(sets)
    J = torch.full((G, n), m, dtype=torch.long)
    for j, ke in enumerate(sets):
        km = ke.unsqueeze(1)
        c = K.attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
        oT = K.mix(s, sh, Vc, Vh, ke, lf, of)
        kw = dict(bk)
        b = K.bound_p8(s, sh, kw.pop("sigma"), km, Vh, c, kw.pop("nu2"), (oT - c).norm(dim=-1), kw.pop("dim"), lf,
                       delta=delta / m, heavy=K.heavy_rows(sl, km, 16), share=0.5, **kw)
        ok = (b <= target).all(1)
        J = torch.where((J == m) & ok, torch.full_like(J, j), J)
    return J


def test_certified_read():
    print("\n[r7] the certified read: the same stop as the naive rule (every set, every group, exact matrix), with and "
          "without brackets; the output meets its target with probability >= 1 - delta")
    g = torch.Generator().manual_seed(22)
    same, same_nb, miss, n, early, saved = True, True, 0, 0, 0, []
    for trial in range(4 if FAST else 8):
        x = rand_data(g, kind=("random", "spiky", "outlier", "adversarial")[trial % 4], C=256, n=3)
        vote, sl = vote_of(x, frac=16)
        pri = K.sketch_priority(sl, vote)
        sets = K.schedule(vote, pri, K.FRACS)
        for seed in range(3 if FAST else 10):
            for bits in (6, 8):
                r_ = instance(x, vote.view(vote.shape[0], 1, 1, -1), 7000 + 100 * trial + 10 * seed + bits, bits=bits)
                bk = dict(sigma=r_["sig"], nu2=r_["n2"], dim=r_["d"], b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"],
                          forms=("hkz",), exact=dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=8, t_min=0.0))
                tgt = 0.25 * r_["o"].norm(dim=-1) * (0.05 if bits == 8 else 1.0)
                R1 = K.certified_read(r_["s"], r_["shd"], sets, x["V"], r_["Vh"], x["lf"], x["of"], tgt, 0.05, bk,
                                      o=r_["o"], sl=sl)
                R0 = K.certified_read(r_["s"], r_["shd"], sets, x["V"], r_["Vh"], x["lf"], x["of"], tgt, 0.05, bk,
                                      o=r_["o"], sl=sl, bracket=False)
                Jn = _naive_read(r_["s"], r_["shd"], sets, x["V"], r_["Vh"], x["lf"], x["of"], tgt, 0.05, bk, sl)
                same &= torch.equal(R1["J"], Jn)
                same_nb &= torch.equal(R0["J"], Jn)
                err = (R1["out"] - r_["o"]).norm(dim=-1)
                tJ = tgt
                miss += int((err > tJ * (1 + 1e-9) + 1e-15).sum())
                early += int((R1["J"] < len(sets)).sum())
                n += err.numel()
                saved.append(1 - R1["n_gram"] / max(R0["n_gram"], 1))
    check("J identical to the naive rule's (brackets on and off); misses <= delta x cases; some early stops",
          same and same_nb and miss <= 0.05 * n and early > 0,
          f"({n} head cases; misses {miss}; early stops {early}; HKZ matrices saved by the brackets "
          f"{sum(saved) / len(saved):.0%})")


def test_fixtures():
    print("\n[r7] parity with ../14h_theory's part 8 and Theorem M on frozen inputs (fixtures/ref_part8.pt)")
    p = os.path.join(HERE, "fixtures", "ref_part8.pt")
    if not os.path.exists(p):
        check("fixtures/ref_part8.pt missing: skipped (generate it: fixtures/README.md)", True)
        return
    F = torch.load(p, weights_only=False)
    worst, n_cmp = 0.0, 0
    for case in F["cases"]:
        a = case["inputs"]
        for cfg_name, cfg in case["configs"].items():
            kw = dict(b_as=a["b_as"], mom=a["mom"], ldebias=a["ldb"], forms=tuple(cfg["forms"]), tilt=cfg["tilt"])
            if cfg.get("exact"):
                kw["exact"] = dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=cfg["exact_rows"], t_min=cfg["t_min"])
            if cfg.get("r1"):
                kw.update(heavy=a["heavy"], share=0.5)
            b = K.bound_p8(a["s"], a["shd"], a["sig"], a["keep"], a["Vh"], a["c"], a["n2"], a["oc"], a["d"], a["lf"],
                           delta=cfg["delta"], **kw)
            worst = max(worst, float(((b - cfg["bound"]).abs() / cfg["bound"].abs()).max()))
            n_cmp += b.numel()
        if "second_moment" in case:
            sm = case["second_moment"]
            TM = K.theorem_m(a["s"], a["keep"], a["lf"], a["V"], a["o"], a["qr"], a["stk"], a["sc"], a["ldb"],
                             a["n2v"], chunk=64)
            worst = max(worst, float(((TM["pred"] - sm).abs() / sm).max()))
            n_cmp += sm.numel()
        if "exact_proxy" in case:
            ep = case["exact_proxy"]
            v = K.exact_key_proxy(ep["a"], ep["t"], ep["ldb"], ep["ez_lo"], ep["ez2_hi"])
            worst = max(worst, float(((v - ep["out"]).abs() / ep["out"]).max()))
            n_cmp += v.numel()
        if "certread_J" in case:
            cr = case["certread"]
            R_ = K.certified_read(a["s"], a["shd"], cr["sets"], a["V"], a["Vh"], a["lf"], a["of"], cr["target"],
                                  cr["delta"], dict(sigma=a["sig"], nu2=a["n2"], dim=a["d"], b_as=a["b_as"], mom=a["mom"],
                                                    ldebias=a["ldb"], forms=("hkz",),
                                                    exact=dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=16,
                                                               t_min=0.2)),
                                  sl=cr["sl"])
            ok_j = torch.equal(R_["J"], case["certread_J"])
            worst = max(worst, 0.0 if ok_j else float("inf"))
    check(f"bounds, Theorem M, P-exact and the certified read's stop equal the reference to 1e-9 "
          f"(reference sources {F['meta']['sources_sha256_short']})", worst < 1e-9, f"({n_cmp} values; max relative gap {worst:.1e})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    QUICK = "--quick" in sys.argv          # the GPU worker's preflight: the copies, the GPU forms and the reference parity
    test_copies()
    test_tail_err()
    test_gram()
    test_series()
    if not QUICK:
        test_theorem_m()
        test_proxies()
        test_bounds()
        test_certified_read()
    test_fixtures()
    print(f"\n{'ALL R7 CERT TESTS PASSED' if not fails else f'{fails} R7 CERT TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
